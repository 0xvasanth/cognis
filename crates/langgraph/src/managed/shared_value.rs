//! Shared managed values for cross-node state sharing.
//!
//! Provides thread-safe shared state primitives with versioning, history tracking,
//! and change notifications. These values can be shared across graph nodes with
//! controlled access patterns using read/write locks.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Error type for shared value operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SharedValueError {
    /// Attempted to write to a read-only shared value.
    #[error("shared value is read-only")]
    ReadOnly,
    /// The requested version was not found in history.
    #[error("version {0} not found in history")]
    VersionNotFound(u64),
    /// Lock poisoned (should be rare).
    #[error("lock poisoned")]
    LockPoisoned,
}

/// A single history entry recording a value at a specific version.
#[derive(Debug, Clone)]
pub struct HistoryEntry<T> {
    /// The value at this version.
    pub value: T,
    /// The version number.
    pub version: u64,
    /// When this entry was created.
    pub timestamp: Instant,
}

/// Tracks the history of value changes with optional size limits.
#[derive(Debug, Clone)]
pub struct ValueHistory<T> {
    /// The history entries in chronological order.
    pub entries: Vec<HistoryEntry<T>>,
    /// Maximum number of entries to retain.
    max_entries: Option<usize>,
}

impl<T: Clone> ValueHistory<T> {
    /// Create a new empty history with an optional maximum size.
    pub fn new(max_entries: Option<usize>) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Record a new value at the given version.
    pub fn record(&mut self, value: T, version: u64) {
        self.entries.push(HistoryEntry {
            value,
            version,
            timestamp: Instant::now(),
        });
        if let Some(max) = self.max_entries {
            while self.entries.len() > max {
                self.entries.remove(0);
            }
        }
    }

    /// Get the value at a specific version, if it exists in history.
    pub fn get_at_version(&self, version: u64) -> Option<&T> {
        self.entries
            .iter()
            .find(|e| e.version == version)
            .map(|e| &e.value)
    }

    /// Get clones of the values at two versions for comparison.
    pub fn diff_versions(&self, v1: u64, v2: u64) -> Option<(T, T)> {
        let val1 = self.get_at_version(v1)?;
        let val2 = self.get_at_version(v2)?;
        Some((val1.clone(), val2.clone()))
    }

    /// Number of history entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Configuration for a `SharedValue`.
pub struct SharedValueConfig<T> {
    /// The initial value.
    pub initial_value: T,
    /// If true, `set` and `update` will return errors.
    pub read_only: bool,
    /// Maximum number of previous values to retain in history.
    pub max_history: Option<usize>,
    /// Optional callback invoked on value change, receiving `(old, new)`.
    pub on_change: Option<ChangeCallback<T>>,
}

/// Type alias for the on-change callback to reduce type complexity.
type ChangeCallback<T> = Arc<dyn Fn(&T, &T) + Send + Sync>;

impl<T: Default> Default for SharedValueConfig<T> {
    fn default() -> Self {
        Self {
            initial_value: T::default(),
            read_only: false,
            max_history: None,
            on_change: None,
        }
    }
}

/// Inner state protected by a RwLock.
struct SharedValueInner<T> {
    value: T,
    version: u64,
    read_only: bool,
    history: ValueHistory<T>,
    on_change: Option<ChangeCallback<T>>,
}

/// A thread-safe shared value with versioning, history, and change notifications.
///
/// Nodes in the graph can share state through `SharedValue`, which provides
/// read/write access with version tracking. Each mutation increments the version
/// counter, optionally records history, and invokes change callbacks.
pub struct SharedValue<T: Clone + Send + Sync + 'static> {
    inner: Arc<RwLock<SharedValueInner<T>>>,
}

impl<T: Clone + Send + Sync + 'static> Clone for SharedValue<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Clone + Send + Sync + 'static> SharedValue<T> {
    /// Create a new shared value from the given configuration.
    pub fn new(config: SharedValueConfig<T>) -> Self {
        let mut history = ValueHistory::new(config.max_history);
        history.record(config.initial_value.clone(), 0);

        Self {
            inner: Arc::new(RwLock::new(SharedValueInner {
                value: config.initial_value,
                version: 0,
                read_only: config.read_only,
                history,
                on_change: config.on_change,
            })),
        }
    }

    /// Create a shared value with a simple initial value and default settings.
    pub fn from_value(value: T) -> Self {
        Self::new(SharedValueConfig {
            initial_value: value,
            read_only: false,
            max_history: None,
            on_change: None,
        })
    }

    /// Read the current value (clone).
    pub fn get(&self) -> T {
        let inner = self.inner.read().expect("lock poisoned");
        inner.value.clone()
    }

    /// Write a new value, incrementing the version.
    pub fn set(&self, value: T) -> Result<(), SharedValueError> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| SharedValueError::LockPoisoned)?;
        if inner.read_only {
            return Err(SharedValueError::ReadOnly);
        }
        let old = inner.value.clone();
        inner.value = value;
        inner.version += 1;
        let new_clone = inner.value.clone();
        let version = inner.version;
        inner.history.record(new_clone, version);
        if let Some(ref cb) = inner.on_change {
            cb(&old, &inner.value);
        }
        Ok(())
    }

    /// Modify the value in place using a closure, incrementing the version.
    pub fn update(&self, f: impl FnOnce(&mut T)) -> Result<(), SharedValueError> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| SharedValueError::LockPoisoned)?;
        if inner.read_only {
            return Err(SharedValueError::ReadOnly);
        }
        let old = inner.value.clone();
        f(&mut inner.value);
        inner.version += 1;
        let new_clone = inner.value.clone();
        let version = inner.version;
        inner.history.record(new_clone, version);
        if let Some(ref cb) = inner.on_change {
            cb(&old, &inner.value);
        }
        Ok(())
    }

    /// Get the current version number.
    pub fn version(&self) -> u64 {
        let inner = self.inner.read().expect("lock poisoned");
        inner.version
    }

    /// Check whether the value has changed since the given version.
    pub fn has_changed_since(&self, version: u64) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        inner.version > version
    }

    /// Access the value history (requires read lock).
    pub fn history(&self) -> ValueHistory<T> {
        let inner = self.inner.read().expect("lock poisoned");
        inner.history.clone()
    }
}

/// A shared map providing map-specific operations over `HashMap<String, Value>`.
///
/// Built on top of `SharedValue` to provide convenient key-level access
/// to a shared JSON-like map.
#[derive(Clone)]
pub struct SharedMap {
    inner: SharedValue<HashMap<String, Value>>,
}

impl SharedMap {
    /// Create a new empty shared map.
    pub fn new() -> Self {
        Self {
            inner: SharedValue::from_value(HashMap::new()),
        }
    }

    /// Create a shared map with initial data.
    pub fn from_map(map: HashMap<String, Value>) -> Self {
        Self {
            inner: SharedValue::from_value(map),
        }
    }

    /// Get the value for a key.
    pub fn get_key(&self, key: &str) -> Option<Value> {
        let map = self.inner.get();
        map.get(key).cloned()
    }

    /// Set a key-value pair.
    pub fn set_key(&self, key: impl Into<String>, value: Value) -> Result<(), SharedValueError> {
        self.inner.update(|map| {
            map.insert(key.into(), value);
        })
    }

    /// Remove a key, returning the old value if present.
    pub fn remove_key(&self, key: &str) -> Result<Option<Value>, SharedValueError> {
        let mut removed = None;
        self.inner.update(|map| {
            removed = map.remove(key);
        })?;
        Ok(removed)
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<String> {
        let map = self.inner.get();
        map.keys().cloned().collect()
    }

    /// Merge another map into this one (existing keys are overwritten).
    pub fn merge(&self, other: HashMap<String, Value>) -> Result<(), SharedValueError> {
        self.inner.update(|map| {
            map.extend(other);
        })
    }

    /// Get the current version.
    pub fn version(&self) -> u64 {
        self.inner.version()
    }
}

impl Default for SharedMap {
    fn default() -> Self {
        Self::new()
    }
}

/// An atomic counter shared across graph nodes.
///
/// Uses `AtomicU64` for lock-free increment/decrement operations.
#[derive(Debug)]
pub struct SharedCounter {
    value: Arc<AtomicU64>,
}

impl Clone for SharedCounter {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
        }
    }
}

impl SharedCounter {
    /// Create a new counter starting at 0.
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a counter with an initial value.
    pub fn with_value(initial: u64) -> Self {
        Self {
            value: Arc::new(AtomicU64::new(initial)),
        }
    }

    /// Increment the counter by 1, returning the new value.
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement the counter by 1, returning the new value.
    /// Saturates at 0 (will not underflow).
    pub fn decrement(&self) -> u64 {
        loop {
            let current = self.value.load(Ordering::SeqCst);
            if current == 0 {
                return 0;
            }
            match self.value.compare_exchange(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return current - 1,
                Err(_) => continue,
            }
        }
    }

    /// Get the current counter value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::SeqCst)
    }

    /// Reset the counter to 0.
    pub fn reset(&self) {
        self.value.store(0, Ordering::SeqCst);
    }
}

impl Default for SharedCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// An append-only shared collection across graph nodes.
///
/// Items can be pushed from any node and retrieved or drained collectively.
pub struct SharedAccumulator<T: Clone + Send + Sync + 'static> {
    inner: Arc<RwLock<Vec<T>>>,
}

impl<T: Clone + Send + Sync + 'static> Clone for SharedAccumulator<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Clone + Send + Sync + 'static> SharedAccumulator<T> {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Push an item to the accumulator.
    pub fn push(&self, item: T) {
        let mut items = self.inner.write().expect("lock poisoned");
        items.push(item);
    }

    /// Get a clone of all items.
    pub fn get_all(&self) -> Vec<T> {
        let items = self.inner.read().expect("lock poisoned");
        items.clone()
    }

    /// Get the number of accumulated items.
    pub fn len(&self) -> usize {
        let items = self.inner.read().expect("lock poisoned");
        items.len()
    }

    /// Whether the accumulator is empty.
    pub fn is_empty(&self) -> bool {
        let items = self.inner.read().expect("lock poisoned");
        items.is_empty()
    }

    /// Drain all items, returning them and leaving the accumulator empty.
    pub fn drain(&self) -> Vec<T> {
        let mut items = self.inner.write().expect("lock poisoned");
        std::mem::take(&mut *items)
    }
}

impl<T: Clone + Send + Sync + 'static> Default for SharedAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    // --- SharedValue get/set ---
    #[test]
    fn test_shared_value_get_set() {
        let sv = SharedValue::from_value(42i32);
        assert_eq!(sv.get(), 42);
        sv.set(100).unwrap();
        assert_eq!(sv.get(), 100);
    }

    // --- SharedValue versioning ---
    #[test]
    fn test_shared_value_versioning() {
        let sv = SharedValue::from_value(0u64);
        assert_eq!(sv.version(), 0);
        sv.set(1).unwrap();
        assert_eq!(sv.version(), 1);
        sv.set(2).unwrap();
        assert_eq!(sv.version(), 2);
        sv.set(3).unwrap();
        assert_eq!(sv.version(), 3);
    }

    // --- SharedValue update closure ---
    #[test]
    fn test_shared_value_update() {
        let sv = SharedValue::from_value(vec![1, 2, 3]);
        sv.update(|v| v.push(4)).unwrap();
        assert_eq!(sv.get(), vec![1, 2, 3, 4]);
        assert_eq!(sv.version(), 1);
    }

    // --- SharedValue has_changed_since ---
    #[test]
    fn test_shared_value_has_changed_since() {
        let sv = SharedValue::from_value("hello".to_string());
        assert!(!sv.has_changed_since(0));
        sv.set("world".to_string()).unwrap();
        assert!(sv.has_changed_since(0));
        assert!(!sv.has_changed_since(1));
    }

    // --- SharedValue read_only mode ---
    #[test]
    fn test_shared_value_read_only() {
        let sv = SharedValue::new(SharedValueConfig {
            initial_value: 42,
            read_only: true,
            max_history: None,
            on_change: None,
        });
        assert_eq!(sv.get(), 42);
        let err = sv.set(99).unwrap_err();
        assert!(matches!(err, SharedValueError::ReadOnly));
        let err = sv.update(|v| *v = 99).unwrap_err();
        assert!(matches!(err, SharedValueError::ReadOnly));
        // Value should be unchanged.
        assert_eq!(sv.get(), 42);
    }

    // --- ValueHistory tracking ---
    #[test]
    fn test_value_history_tracking() {
        let sv = SharedValue::new(SharedValueConfig {
            initial_value: 0i32,
            read_only: false,
            max_history: Some(3),
            on_change: None,
        });
        // Initial value is version 0 in history.
        sv.set(10).unwrap(); // version 1
        sv.set(20).unwrap(); // version 2
        sv.set(30).unwrap(); // version 3

        let history = sv.history();
        // max_history=3, so oldest entry (version 0) should be evicted.
        assert_eq!(history.len(), 3);
        assert!(history.get_at_version(0).is_none());
        assert_eq!(*history.get_at_version(1).unwrap(), 10);
        assert_eq!(*history.get_at_version(3).unwrap(), 30);
    }

    // --- ValueHistory get_at_version ---
    #[test]
    fn test_value_history_get_at_version() {
        let sv = SharedValue::from_value("a".to_string());
        sv.set("b".to_string()).unwrap();
        sv.set("c".to_string()).unwrap();

        let history = sv.history();
        assert_eq!(history.get_at_version(0).unwrap(), "a");
        assert_eq!(history.get_at_version(1).unwrap(), "b");
        assert_eq!(history.get_at_version(2).unwrap(), "c");
        assert!(history.get_at_version(99).is_none());
    }

    // --- ValueHistory diff_versions ---
    #[test]
    fn test_value_history_diff_versions() {
        let sv = SharedValue::from_value(10i32);
        sv.set(20).unwrap();
        sv.set(30).unwrap();

        let history = sv.history();
        let (v0, v2) = history.diff_versions(0, 2).unwrap();
        assert_eq!(v0, 10);
        assert_eq!(v2, 30);
        assert!(history.diff_versions(0, 99).is_none());
    }

    // --- SharedMap get/set/remove keys ---
    #[test]
    fn test_shared_map_get_set_remove() {
        let map = SharedMap::new();
        assert!(map.get_key("foo").is_none());

        map.set_key("foo", Value::String("bar".into())).unwrap();
        assert_eq!(map.get_key("foo"), Some(Value::String("bar".into())));

        let removed = map.remove_key("foo").unwrap();
        assert_eq!(removed, Some(Value::String("bar".into())));
        assert!(map.get_key("foo").is_none());
    }

    // --- SharedMap merge ---
    #[test]
    fn test_shared_map_merge() {
        let map = SharedMap::new();
        map.set_key("a", Value::from(1)).unwrap();

        let mut other = HashMap::new();
        other.insert("b".to_string(), Value::from(2));
        other.insert("c".to_string(), Value::from(3));
        map.merge(other).unwrap();

        let mut keys = map.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
        assert_eq!(map.get_key("b"), Some(Value::from(2)));
    }

    // --- SharedCounter increment/decrement ---
    #[test]
    fn test_shared_counter_increment_decrement() {
        let counter = SharedCounter::new();
        assert_eq!(counter.get(), 0);
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.increment(), 3);
        assert_eq!(counter.decrement(), 2);
        assert_eq!(counter.get(), 2);
    }

    // --- SharedCounter reset ---
    #[test]
    fn test_shared_counter_reset() {
        let counter = SharedCounter::with_value(50);
        assert_eq!(counter.get(), 50);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    // --- SharedCounter decrement saturates at zero ---
    #[test]
    fn test_shared_counter_decrement_saturates() {
        let counter = SharedCounter::new();
        assert_eq!(counter.decrement(), 0);
        assert_eq!(counter.get(), 0);
    }

    // --- SharedAccumulator push/get_all ---
    #[test]
    fn test_shared_accumulator_push_get_all() {
        let acc = SharedAccumulator::<String>::new();
        assert!(acc.is_empty());
        acc.push("hello".to_string());
        acc.push("world".to_string());
        assert_eq!(acc.len(), 2);
        assert_eq!(
            acc.get_all(),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    // --- SharedAccumulator drain ---
    #[test]
    fn test_shared_accumulator_drain() {
        let acc = SharedAccumulator::new();
        acc.push(1);
        acc.push(2);
        acc.push(3);

        let drained = acc.drain();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);
    }

    // --- Change callback invocation ---
    #[test]
    fn test_shared_value_on_change_callback() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);
        let old_capture = Arc::new(RwLock::new(0i32));
        let new_capture = Arc::new(RwLock::new(0i32));
        let old_c = Arc::clone(&old_capture);
        let new_c = Arc::clone(&new_capture);

        let sv = SharedValue::new(SharedValueConfig {
            initial_value: 10i32,
            read_only: false,
            max_history: None,
            on_change: Some(Arc::new(move |old, new| {
                called_clone.store(true, Ordering::SeqCst);
                *old_c.write().unwrap() = *old;
                *new_c.write().unwrap() = *new;
            })),
        });

        assert!(!called.load(Ordering::SeqCst));
        sv.set(20).unwrap();
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(*old_capture.read().unwrap(), 10);
        assert_eq!(*new_capture.read().unwrap(), 20);
    }

    // --- Thread-safe concurrent access ---
    #[test]
    fn test_shared_value_concurrent_access() {
        let sv = SharedValue::from_value(0i64);
        let mut handles = vec![];

        for _ in 0..10 {
            let sv_clone = sv.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    sv_clone.update(|v| *v += 1).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sv.get(), 1000);
        assert_eq!(sv.version(), 1000);
    }

    #[test]
    fn test_shared_counter_concurrent() {
        let counter = SharedCounter::new();
        let mut handles = vec![];

        for _ in 0..10 {
            let c = counter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    c.increment();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(counter.get(), 1000);
    }
}
