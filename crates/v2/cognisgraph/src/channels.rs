//! Channel value types — embeddable inside a [`crate::state::GraphState`]
//! struct to express V1-style channel semantics on top of V2's typed
//! state model.
//!
//! These are plain Rust types you put in your state struct, not a separate
//! abstraction. The state's reducer is responsible for routing updates
//! into them.
//!
//! - [`AnyValue<T>`] — accept any number of writes per step; assert at
//!   least one if `required`.
//! - [`Topic<T>`] — append-only queue with `drain()` consume semantics.
//! - [`BinaryOp<T>`] — fold writes via an associative binary operation.
//! - [`Broadcast<T>`] — multi-consumer queue; each consumer has its own
//!   cursor.
//! - [`Untracked<T>`] — wrapper that is excluded from serialization (and
//!   therefore from checkpoints).

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AnyValue
// ---------------------------------------------------------------------------

/// Accepts any number of writes per superstep; the **last** write wins
/// (mirrors V1 `AnyValue` semantics).
///
/// Useful when more than one node may write the same field in a step and
/// you don't care which wins (e.g. a "done" flag set by either branch).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnyValue<T> {
    inner: Option<T>,
}

impl<T> AnyValue<T> {
    /// Empty channel.
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Build with an initial value.
    pub fn with(value: T) -> Self {
        Self { inner: Some(value) }
    }

    /// Set the channel's value (any-write-wins).
    pub fn set(&mut self, value: T) {
        self.inner = Some(value);
    }

    /// Borrow the current value, if any.
    pub fn get(&self) -> Option<&T> {
        self.inner.as_ref()
    }

    /// Consume the channel and return its value.
    pub fn take(&mut self) -> Option<T> {
        self.inner.take()
    }

    /// True if no value has been written.
    pub fn is_empty(&self) -> bool {
        self.inner.is_none()
    }
}

// ---------------------------------------------------------------------------
// Topic
// ---------------------------------------------------------------------------

/// Append-only queue. Producers `send()`, consumers `drain()` (single
/// consumer) — mirrors V1 `Topic` channel.
///
/// Use [`Broadcast`] when more than one consumer needs to see every event.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Topic<T> {
    queue: Vec<T>,
}

impl<T> Topic<T> {
    /// New empty topic.
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    /// Append `value` to the queue.
    pub fn send(&mut self, value: T) {
        self.queue.push(value);
    }

    /// Append every item from `values`.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        self.queue.extend(values);
    }

    /// Drain the queue, returning everything currently buffered.
    pub fn drain(&mut self) -> Vec<T> {
        std::mem::take(&mut self.queue)
    }

    /// Borrow the current queue contents without consuming them.
    pub fn peek(&self) -> &[T] {
        &self.queue
    }

    /// Number of pending items.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// True if no items are pending.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BinaryOp
// ---------------------------------------------------------------------------

/// Channel that folds writes via an associative binary operation
/// (mirrors V1 `BinaryOp` channel).
///
/// The op is supplied as a `fn(&T, &T) -> T` so the type stays
/// `Serialize`/`Deserialize`. Persisting only the value (not the op) is
/// fine: the op is deterministic and re-supplied on reconstruction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinaryOp<T> {
    value: Option<T>,
    #[serde(skip)]
    op: Option<fn(&T, &T) -> T>,
}

impl<T: PartialEq> PartialEq for BinaryOp<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq> Eq for BinaryOp<T> {}

impl<T: Clone> BinaryOp<T> {
    /// Build an empty channel with operation `op`.
    pub fn new(op: fn(&T, &T) -> T) -> Self {
        Self {
            value: None,
            op: Some(op),
        }
    }

    /// Build pre-seeded with `initial`.
    pub fn with_initial(op: fn(&T, &T) -> T, initial: T) -> Self {
        Self {
            value: Some(initial),
            op: Some(op),
        }
    }

    /// Re-attach the binary op after deserialization (the op itself is
    /// `#[serde(skip)]`; persistence only stores the value).
    pub fn rehydrate(mut self, op: fn(&T, &T) -> T) -> Self {
        self.op = Some(op);
        self
    }

    /// Fold `value` into the channel via the configured op.
    pub fn write(&mut self, value: T) -> cognis2_core::Result<()> {
        let op = self.op.ok_or_else(|| {
            cognis2_core::CognisError::Internal(
                "BinaryOp: write called before rehydrate (no op set)".into(),
            )
        })?;
        self.value = Some(match self.value.as_ref() {
            Some(existing) => op(existing, &value),
            None => value,
        });
        Ok(())
    }

    /// Borrow the current accumulated value.
    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Consume the channel and return the accumulated value.
    pub fn take(&mut self) -> Option<T> {
        self.value.take()
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

/// Multi-consumer broadcast queue (mirrors V1 `Broadcast` channel).
///
/// Producers call `send()`. Each consumer first calls `subscribe()` to
/// obtain a cursor, then calls `read(cursor)` to receive every item
/// produced since its last read. Items are kept in the buffer until **all**
/// active subscribers have consumed them, at which point they're garbage
/// collected on `gc()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Broadcast<T> {
    /// All buffered items + their global sequence numbers.
    items: Vec<(u64, T)>,
    /// Per-subscriber high-water cursor (next sequence number to deliver).
    cursors: HashMap<String, u64>,
    /// Next sequence number to assign.
    next_seq: u64,
}

impl<T> Default for Broadcast<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            cursors: HashMap::new(),
            next_seq: 0,
        }
    }
}

impl<T: Clone> Broadcast<T> {
    /// New empty broadcast.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subscriber. The cursor's name should be unique per
    /// consumer; subsequent calls with the same name are idempotent.
    pub fn subscribe(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.cursors.entry(name).or_insert(self.next_seq);
    }

    /// Drop a subscriber. Once dropped its cursor no longer prevents
    /// garbage collection of older items.
    pub fn unsubscribe(&mut self, name: &str) {
        self.cursors.remove(name);
    }

    /// Append an item to the buffer. All current subscribers will receive it.
    pub fn send(&mut self, value: T) {
        self.items.push((self.next_seq, value));
        self.next_seq += 1;
    }

    /// Read every item the given subscriber has not yet consumed.
    /// Advances the subscriber's cursor.
    pub fn read(&mut self, name: &str) -> Vec<T> {
        let cursor = match self.cursors.get_mut(name) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let out: Vec<T> = self
            .items
            .iter()
            .filter(|(seq, _)| *seq >= *cursor)
            .map(|(_, v)| v.clone())
            .collect();
        *cursor = self.next_seq;
        out
    }

    /// Drop items every subscriber has already consumed. Safe to call
    /// any time; cheap if there's nothing to evict.
    pub fn gc(&mut self) {
        if self.cursors.is_empty() {
            // No subscribers — buffer can be cleared.
            self.items.clear();
            return;
        }
        let min_cursor = self
            .cursors
            .values()
            .copied()
            .min()
            .unwrap_or(self.next_seq);
        self.items.retain(|(seq, _)| *seq >= min_cursor);
    }

    /// Total buffered items not yet GC'd.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Untracked
// ---------------------------------------------------------------------------

/// Transparent wrapper that is **excluded from serialization** — values
/// inside `Untracked<T>` are not persisted to checkpoints (mirrors V1
/// `Untracked` channel).
///
/// Use for in-memory caches, large compute artifacts, or non-`Serialize`
/// types that you still want to live inside `GraphState`.
#[derive(Debug, Clone)]
pub struct Untracked<T> {
    /// The wrapped value. Public so users can read/write without ceremony.
    pub inner: T,
}

impl<T: Default> Default for Untracked<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
        }
    }
}

impl<T> Untracked<T> {
    /// Wrap `value`.
    pub fn new(value: T) -> Self {
        Self { inner: value }
    }

    /// Unwrap.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

// `Untracked<T>` is intentionally not `Serialize` even when `T: Serialize`.
// Users put it inside a `GraphState` struct and either annotate the field
// `#[serde(skip)]` or use a serializer that tolerates skipped fields.
// We provide marker impls below so it can sit inside a `Serialize`-derived
// struct that uses `#[serde(skip)]`.
impl<T> serde::Serialize for Untracked<T> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        // Serializing should be skipped at the field level via
        // `#[serde(skip)]`. Calling this directly produces a unit so
        // tooling that ignores skip directives still gets a stable shape.
        serializer.serialize_unit()
    }
}

impl<'de, T: Default> serde::Deserialize<'de> for Untracked<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        // Consume any payload (most commonly `null` written by our
        // `Serialize` impl) and reconstruct the inner value from
        // `Default`. We use `IgnoredAny` so any shape — null, missing,
        // an arbitrary blob — is accepted.
        serde::de::IgnoredAny::deserialize(deserializer)?;
        Ok(Self::default())
    }
}

// ---------------------------------------------------------------------------
// CustomChannel — pluggable inline channel without subclassing.
// ---------------------------------------------------------------------------

/// Boxed merge function used by [`CustomChannel`]. Receives `(slot, incoming)`
/// and updates `slot` in-place.
pub type CustomMergeFn<T> = Box<dyn Fn(&mut T, T) + Send + Sync>;

/// Custom channel: hold a value of type `T` plus user-supplied write,
/// read, and reset closures. Implements [`Channel`] (with a custom
/// `kind` label) and is therefore registry-compatible alongside the
/// stock channels.
///
/// Use this when none of the built-in channels match your semantics
/// but you don't want a whole new struct + trait impl.
pub struct CustomChannel<T> {
    label: &'static str,
    value: T,
    on_write: CustomMergeFn<T>,
}

impl<T: Default> CustomChannel<T> {
    /// Build with a label, initial value, and a `write(slot, incoming)`
    /// merge function.
    pub fn new<F>(label: &'static str, on_write: F) -> Self
    where
        F: Fn(&mut T, T) + Send + Sync + 'static,
    {
        Self {
            label,
            value: T::default(),
            on_write: Box::new(on_write),
        }
    }
}

impl<T> CustomChannel<T> {
    /// Build with explicit initial value.
    pub fn with_initial<F>(label: &'static str, initial: T, on_write: F) -> Self
    where
        F: Fn(&mut T, T) + Send + Sync + 'static,
    {
        Self {
            label,
            value: initial,
            on_write: Box::new(on_write),
        }
    }

    /// Apply an incoming write through the user-defined merge fn.
    pub fn write(&mut self, value: T) {
        (self.on_write)(&mut self.value, value);
    }

    /// Borrow the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Mutably borrow the current value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Replace the wrapped value, returning the previous one.
    pub fn replace(&mut self, new: T) -> T {
        std::mem::replace(&mut self.value, new)
    }
}

impl<T: Send + Sync> Channel for CustomChannel<T> {
    fn kind(&self) -> &'static str {
        self.label
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for CustomChannel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomChannel")
            .field("label", &self.label)
            .field("value", &self.value)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Type-erased channel reference (rarely needed, exposed for tooling).
// ---------------------------------------------------------------------------

/// Object-safe trait shared by every channel type. Lets diagnostic tools
/// inspect channels uniformly without knowing the concrete type.
pub trait Channel: Send + Sync {
    /// Channel kind label, e.g. `"AnyValue"`, `"Topic"`.
    fn kind(&self) -> &'static str;
}

impl<T: Send + Sync> Channel for AnyValue<T> {
    fn kind(&self) -> &'static str {
        "AnyValue"
    }
}
impl<T: Send + Sync> Channel for Topic<T> {
    fn kind(&self) -> &'static str {
        "Topic"
    }
}
impl<T: Send + Sync> Channel for BinaryOp<T> {
    fn kind(&self) -> &'static str {
        "BinaryOp"
    }
}
impl<T: Send + Sync> Channel for Broadcast<T> {
    fn kind(&self) -> &'static str {
        "Broadcast"
    }
}
impl<T: Send + Sync> Channel for Untracked<T> {
    fn kind(&self) -> &'static str {
        "Untracked"
    }
}

// ---------------------------------------------------------------------------
// `Arc<dyn Channel>` registry — used by tools that want a uniform handle.
// ---------------------------------------------------------------------------

/// Type-erased channel handle. Useful when multiple node implementations
/// share the same channel and you want runtime polymorphism without
/// generic plumbing.
pub type ChannelRef = Arc<dyn Channel>;

#[doc(hidden)]
pub struct _ChannelTag<T>(PhantomData<fn() -> T>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_value_set_and_take() {
        let mut a: AnyValue<i32> = AnyValue::new();
        assert!(a.is_empty());
        a.set(1);
        a.set(2);
        assert_eq!(a.get(), Some(&2));
        assert_eq!(a.take(), Some(2));
        assert!(a.is_empty());
    }

    #[test]
    fn topic_send_drain_round_trip() {
        let mut t: Topic<&'static str> = Topic::new();
        t.send("a");
        t.send("b");
        t.extend(["c", "d"]);
        assert_eq!(t.len(), 4);
        let drained = t.drain();
        assert_eq!(drained, vec!["a", "b", "c", "d"]);
        assert!(t.is_empty());
    }

    #[test]
    fn binary_op_folds_associatively() {
        let mut b: BinaryOp<i32> = BinaryOp::new(|a, b| a + b);
        b.write(1).unwrap();
        b.write(2).unwrap();
        b.write(3).unwrap();
        assert_eq!(b.get(), Some(&6));
    }

    #[test]
    fn binary_op_without_rehydrate_errors() {
        let mut b: BinaryOp<i32> = BinaryOp {
            value: None,
            op: None,
        };
        let err = b.write(1).unwrap_err();
        assert!(matches!(err, cognis2_core::CognisError::Internal(_)));
    }

    #[test]
    fn binary_op_rehydrate_reattaches_op() {
        let b: BinaryOp<i32> = BinaryOp {
            value: Some(5),
            op: None,
        };
        let mut b = b.rehydrate(|a, b| a + b);
        b.write(2).unwrap();
        assert_eq!(b.get(), Some(&7));
    }

    #[test]
    fn broadcast_delivers_to_all_subscribers() {
        let mut b: Broadcast<i32> = Broadcast::new();
        b.subscribe("a");
        b.subscribe("b");
        b.send(1);
        b.send(2);
        assert_eq!(b.read("a"), vec![1, 2]);
        assert_eq!(b.read("b"), vec![1, 2]);
        // Second read returns nothing for a until new sends happen.
        assert!(b.read("a").is_empty());
        b.send(3);
        assert_eq!(b.read("a"), vec![3]);
        assert_eq!(b.read("b"), vec![3]);
    }

    #[test]
    fn broadcast_gc_drops_consumed_items() {
        let mut b: Broadcast<i32> = Broadcast::new();
        b.subscribe("only");
        b.send(1);
        b.send(2);
        let _ = b.read("only");
        b.gc();
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn broadcast_unknown_subscriber_reads_empty() {
        let mut b: Broadcast<i32> = Broadcast::new();
        b.send(1);
        assert!(b.read("ghost").is_empty());
    }

    #[test]
    fn untracked_round_trips_through_serde_to_default() {
        let u = Untracked::new(42i32);
        let json = serde_json::to_string(&u).unwrap();
        // Serialized as unit so it occupies no payload space.
        assert_eq!(json, "null");
        // Deserializing reconstructs Default::default().
        let back: Untracked<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.inner, 0);
    }

    #[test]
    fn channel_kind_strings() {
        let a: AnyValue<i32> = AnyValue::new();
        let t: Topic<i32> = Topic::new();
        let b: BinaryOp<i32> = BinaryOp::new(|a, b| a + b);
        let bc: Broadcast<i32> = Broadcast::new();
        let u: Untracked<i32> = Untracked::default();
        assert_eq!(a.kind(), "AnyValue");
        assert_eq!(t.kind(), "Topic");
        assert_eq!(b.kind(), "BinaryOp");
        assert_eq!(bc.kind(), "Broadcast");
        assert_eq!(u.kind(), "Untracked");
    }

    #[test]
    fn custom_channel_applies_user_merge() {
        // A "running max" channel.
        let mut c: CustomChannel<i32> = CustomChannel::new("Max", |slot, incoming| {
            if incoming > *slot {
                *slot = incoming;
            }
        });
        c.write(3);
        c.write(1);
        c.write(7);
        c.write(5);
        assert_eq!(*c.get(), 7);
        assert_eq!(c.kind(), "Max");
    }

    #[test]
    fn custom_channel_with_initial_seeds_value() {
        let mut c: CustomChannel<Vec<i32>> =
            CustomChannel::with_initial("Concat", vec![1, 2], |slot, incoming| {
                slot.extend(incoming);
            });
        c.write(vec![3, 4]);
        assert_eq!(c.get(), &vec![1, 2, 3, 4]);
    }
}
