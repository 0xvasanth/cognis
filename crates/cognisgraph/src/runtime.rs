//! Runtime context injection for graph nodes.
//!
//! This module provides the [`Runtime`] struct that bundles run-scoped context,
//! store access, stream writing, and configuration into a single object that
//! can be injected into graph nodes during execution.
//!
//! The design mirrors Python's `langgraph.runtime.Runtime` but takes advantage
//! of Rust's generics and ownership model:
//!
//! - `Runtime<C>` is generic over a user-defined context type `C`.
//! - [`NoContext`] is used as the default when no context is needed.
//! - [`RuntimeBuilder`] provides a fluent API for constructing runtimes.
//! - [`RuntimeProvider`] is a trait for accessing the current runtime.
//! - [`RuntimeScope`] offers RAII-style lifecycle management.

use crate::config::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// StreamWriter
// ---------------------------------------------------------------------------

/// A stream writer function that accepts a `Value` and writes it to the
/// output stream. Used for `stream_mode="custom"`.
pub type StreamWriter = Arc<dyn Fn(Value) + Send + Sync>;

/// Returns a no-op [`StreamWriter`] that discards all values.
pub fn no_op_stream_writer() -> StreamWriter {
    Arc::new(|_| {})
}

// ---------------------------------------------------------------------------
// NoContext
// ---------------------------------------------------------------------------

/// Unit-type placeholder used when a [`Runtime`] does not need user context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NoContext;

// ---------------------------------------------------------------------------
// RuntimeConfig
// ---------------------------------------------------------------------------

/// Configuration metadata associated with a graph run.
///
/// Carries identifiers and user-supplied metadata / tags for the current
/// execution. Serialisable so it can be persisted alongside checkpoints.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Unique identifier for this run.
    #[serde(default)]
    pub run_id: Option<String>,

    /// Identifier for the conversation thread.
    #[serde(default)]
    pub thread_id: Option<String>,

    /// Freeform tags attached to the run.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl RuntimeConfig {
    /// Create a new, empty `RuntimeConfig`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the run id.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set the thread id.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Append a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Append multiple tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Insert a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Runtime<C>
// ---------------------------------------------------------------------------

/// Run-scoped runtime context injected into graph nodes.
///
/// `C` is the user-provided context type (e.g. a struct carrying a `user_id`
/// and a database handle). Use [`NoContext`] when no context is needed.
///
/// # Examples
///
/// ```
/// use cognisgraph::runtime::{Runtime, RuntimeBuilder, NoContext};
///
/// let rt: Runtime<NoContext> = RuntimeBuilder::new().build();
/// assert!(rt.store().is_none());
/// ```
pub struct Runtime<C = NoContext> {
    context: C,
    store: Option<Arc<dyn Store>>,
    stream_writer: StreamWriter,
    config: RuntimeConfig,
    previous: Option<Value>,
}

impl<C> Runtime<C> {
    // -- accessors ---------------------------------------------------------

    /// Returns a reference to the user-provided context.
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Returns a mutable reference to the user-provided context.
    pub fn context_mut(&mut self) -> &mut C {
        &mut self.context
    }

    /// Consumes the runtime and returns the inner context.
    pub fn into_context(self) -> C {
        self.context
    }

    /// Returns a reference to the store, if any.
    pub fn store(&self) -> Option<&dyn Store> {
        self.store.as_deref()
    }

    /// Returns a clone of the `Arc<dyn Store>`, if any.
    pub fn store_arc(&self) -> Option<Arc<dyn Store>> {
        self.store.clone()
    }

    /// Returns the stream writer.
    pub fn stream_writer(&self) -> &StreamWriter {
        &self.stream_writer
    }

    /// Write a value to the custom stream.
    pub fn write_to_stream(&self, value: Value) {
        (self.stream_writer)(value);
    }

    /// Returns a reference to the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns a mutable reference to the runtime configuration.
    pub fn config_mut(&mut self) -> &mut RuntimeConfig {
        &mut self.config
    }

    /// Returns the previous return value, if any.
    pub fn previous(&self) -> Option<&Value> {
        self.previous.as_ref()
    }

    /// Set the previous value.
    pub fn set_previous(&mut self, value: Option<Value>) {
        self.previous = value;
    }

    // -- merge / override --------------------------------------------------

    /// Merge with another runtime. Fields from `other` take priority when set.
    pub fn merge(self, other: Runtime<C>) -> Runtime<C> {
        Runtime {
            context: other.context,
            store: other.store.or(self.store),
            stream_writer: other.stream_writer,
            config: RuntimeConfig {
                run_id: other.config.run_id.or(self.config.run_id),
                thread_id: other.config.thread_id.or(self.config.thread_id),
                tags: if other.config.tags.is_empty() {
                    self.config.tags
                } else {
                    other.config.tags
                },
                metadata: if other.config.metadata.is_empty() {
                    self.config.metadata
                } else {
                    other.config.metadata
                },
            },
            previous: other.previous.or(self.previous),
        }
    }

    /// Replace the context, keeping everything else the same.
    pub fn with_context<C2>(self, context: C2) -> Runtime<C2> {
        Runtime {
            context,
            store: self.store,
            stream_writer: self.stream_writer,
            config: self.config,
            previous: self.previous,
        }
    }

    /// Replace the store.
    pub fn with_store(mut self, store: Option<Arc<dyn Store>>) -> Self {
        self.store = store;
        self
    }

    /// Replace the stream writer.
    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = writer;
        self
    }

    /// Replace the config.
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Replace the previous value.
    pub fn with_previous(mut self, previous: Option<Value>) -> Self {
        self.previous = previous;
        self
    }

    /// Map the context to a new type.
    pub fn map_context<C2, F: FnOnce(C) -> C2>(self, f: F) -> Runtime<C2> {
        Runtime {
            context: f(self.context),
            store: self.store,
            stream_writer: self.stream_writer,
            config: self.config,
            previous: self.previous,
        }
    }
}

impl<C: Clone> Clone for Runtime<C> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            store: self.store.clone(),
            stream_writer: Arc::clone(&self.stream_writer),
            config: self.config.clone(),
            previous: self.previous.clone(),
        }
    }
}

impl<C: fmt::Debug> fmt::Debug for Runtime<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime")
            .field("context", &self.context)
            .field("store", &self.store.as_ref().map(|_| "<Store>"))
            .field("stream_writer", &"<StreamWriter>")
            .field("config", &self.config)
            .field("previous", &self.previous)
            .finish()
    }
}

impl<C: Default> Default for Runtime<C> {
    fn default() -> Self {
        RuntimeBuilder::new().with_context(C::default()).build()
    }
}

// Legacy compat: Runtime<NoContext> can be constructed via Runtime::new()
impl Runtime<NoContext> {
    /// Create a default runtime with [`NoContext`].
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// RuntimeBuilder<C>
// ---------------------------------------------------------------------------

/// Fluent builder for constructing [`Runtime`] instances.
///
/// # Examples
///
/// ```
/// use cognisgraph::runtime::{RuntimeBuilder, RuntimeConfig, NoContext};
///
/// let rt = RuntimeBuilder::new()
///     .with_config(RuntimeConfig::new().with_run_id("r-1"))
///     .build();
/// assert_eq!(rt.config().run_id.as_deref(), Some("r-1"));
/// ```
pub struct RuntimeBuilder<C = NoContext> {
    context: Option<C>,
    store: Option<Arc<dyn Store>>,
    stream_writer: Option<StreamWriter>,
    config: RuntimeConfig,
    previous: Option<Value>,
}

impl RuntimeBuilder<NoContext> {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            context: None,
            store: None,
            stream_writer: None,
            config: RuntimeConfig::default(),
            previous: None,
        }
    }
}

impl<C> RuntimeBuilder<C> {
    /// Set the user context.
    pub fn with_context<C2>(self, context: C2) -> RuntimeBuilder<C2> {
        RuntimeBuilder {
            context: Some(context),
            store: self.store,
            stream_writer: self.stream_writer,
            config: self.config,
            previous: self.previous,
        }
    }

    /// Set the store.
    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// Set the stream writer.
    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = Some(writer);
        self
    }

    /// Set the runtime config.
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the previous value.
    pub fn with_previous(mut self, previous: Value) -> Self {
        self.previous = Some(previous);
        self
    }
}

impl<C: Default> RuntimeBuilder<C> {
    /// Build the [`Runtime`].
    ///
    /// If no context was provided, `C::default()` is used.
    pub fn build(self) -> Runtime<C> {
        Runtime {
            context: self.context.unwrap_or_default(),
            store: self.store,
            stream_writer: self.stream_writer.unwrap_or_else(no_op_stream_writer),
            config: self.config,
            previous: self.previous,
        }
    }
}

impl Default for RuntimeBuilder<NoContext> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RuntimeProvider
// ---------------------------------------------------------------------------

/// Trait for types that can supply a [`Runtime`] reference.
///
/// Implement this on your graph executor or node context so that nodes can
/// access the current runtime via a uniform interface.
pub trait RuntimeProvider<C = NoContext> {
    /// Returns a reference to the current runtime.
    fn runtime(&self) -> &Runtime<C>;
}

// ---------------------------------------------------------------------------
// RuntimeScope
// ---------------------------------------------------------------------------

/// RAII guard that holds a [`Runtime`] for the duration of a scope.
///
/// Useful for bracketing a graph execution: create a `RuntimeScope` at the
/// start, pass `scope.runtime()` into nodes, and the runtime is automatically
/// cleaned up when the scope drops.
///
/// # Examples
///
/// ```
/// use cognisgraph::runtime::{Runtime, RuntimeScope, NoContext};
///
/// let rt = Runtime::<NoContext>::new();
/// let scope = RuntimeScope::new(rt);
/// assert!(scope.runtime().store().is_none());
/// // scope drops here
/// ```
pub struct RuntimeScope<C = NoContext> {
    runtime: Option<Runtime<C>>,
    _on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl<C> RuntimeScope<C> {
    /// Create a new scope that owns the given runtime.
    pub fn new(runtime: Runtime<C>) -> Self {
        Self {
            runtime: Some(runtime),
            _on_drop: None,
        }
    }

    /// Create a new scope with a cleanup callback that runs on drop.
    pub fn with_cleanup<F: FnOnce() + Send + 'static>(runtime: Runtime<C>, on_drop: F) -> Self {
        Self {
            runtime: Some(runtime),
            _on_drop: Some(Box::new(on_drop)),
        }
    }

    /// Access the runtime held by this scope.
    ///
    /// # Panics
    ///
    /// Panics if the runtime has already been taken via [`into_inner`](Self::into_inner).
    pub fn runtime(&self) -> &Runtime<C> {
        self.runtime
            .as_ref()
            .expect("runtime already consumed via into_inner")
    }

    /// Mutably access the runtime held by this scope.
    ///
    /// # Panics
    ///
    /// Panics if the runtime has already been taken via [`into_inner`](Self::into_inner).
    pub fn runtime_mut(&mut self) -> &mut Runtime<C> {
        self.runtime
            .as_mut()
            .expect("runtime already consumed via into_inner")
    }

    /// Consume the scope and return the inner runtime (skipping cleanup).
    pub fn into_inner(mut self) -> Runtime<C> {
        self._on_drop = None;
        self.runtime
            .take()
            .expect("runtime already consumed via into_inner")
    }
}

impl<C> Drop for RuntimeScope<C> {
    fn drop(&mut self) {
        if let Some(f) = self._on_drop.take() {
            f();
        }
    }
}

impl<C> RuntimeProvider<C> for RuntimeScope<C> {
    fn runtime(&self) -> &Runtime<C> {
        RuntimeScope::runtime(self)
    }
}

impl<C: fmt::Debug> fmt::Debug for RuntimeScope<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeScope")
            .field("runtime", &self.runtime)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DEFAULT_RUNTIME
// ---------------------------------------------------------------------------

/// Returns a default runtime with no context, no store, and a no-op stream
/// writer. Equivalent to Python's `DEFAULT_RUNTIME`.
pub fn default_runtime() -> Runtime<NoContext> {
    Runtime::new()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InMemoryStore;
    use serde_json::json;

    // -- NoContext ----------------------------------------------------------

    #[test]
    fn test_no_context_default() {
        let nc = NoContext;
        assert_eq!(nc, NoContext::default());
    }

    #[test]
    fn test_no_context_clone() {
        let nc = NoContext;
        let nc2 = nc;
        assert_eq!(nc, nc2);
    }

    #[test]
    fn test_no_context_debug() {
        assert_eq!(format!("{:?}", NoContext), "NoContext");
    }

    #[test]
    fn test_no_context_serde_roundtrip() {
        let json = serde_json::to_string(&NoContext).unwrap();
        let back: NoContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back, NoContext);
    }

    // -- RuntimeConfig -----------------------------------------------------

    #[test]
    fn test_runtime_config_default() {
        let cfg = RuntimeConfig::new();
        assert!(cfg.run_id.is_none());
        assert!(cfg.thread_id.is_none());
        assert!(cfg.tags.is_empty());
        assert!(cfg.metadata.is_empty());
    }

    #[test]
    fn test_runtime_config_with_run_id() {
        let cfg = RuntimeConfig::new().with_run_id("r-42");
        assert_eq!(cfg.run_id.as_deref(), Some("r-42"));
    }

    #[test]
    fn test_runtime_config_with_thread_id() {
        let cfg = RuntimeConfig::new().with_thread_id("t-1");
        assert_eq!(cfg.thread_id.as_deref(), Some("t-1"));
    }

    #[test]
    fn test_runtime_config_with_tag() {
        let cfg = RuntimeConfig::new().with_tag("fast").with_tag("prod");
        assert_eq!(cfg.tags, vec!["fast", "prod"]);
    }

    #[test]
    fn test_runtime_config_with_tags() {
        let cfg = RuntimeConfig::new().with_tags(vec!["a", "b", "c"]);
        assert_eq!(cfg.tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_runtime_config_with_metadata() {
        let cfg = RuntimeConfig::new()
            .with_metadata("env", json!("prod"))
            .with_metadata("version", json!(2));
        assert_eq!(cfg.metadata.get("env"), Some(&json!("prod")));
        assert_eq!(cfg.metadata.get("version"), Some(&json!(2)));
    }

    #[test]
    fn test_runtime_config_serde_roundtrip() {
        let cfg = RuntimeConfig::new()
            .with_run_id("r-1")
            .with_thread_id("t-1")
            .with_tag("test")
            .with_metadata("k", json!("v"));
        let json_str = serde_json::to_string(&cfg).unwrap();
        let back: RuntimeConfig = serde_json::from_str(&json_str).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn test_runtime_config_serde_empty() {
        let cfg = RuntimeConfig::new();
        let json_str = serde_json::to_string(&cfg).unwrap();
        let back: RuntimeConfig = serde_json::from_str(&json_str).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn test_runtime_config_deserialize_partial() {
        let json_str = r#"{"run_id":"r-1"}"#;
        let cfg: RuntimeConfig = serde_json::from_str(json_str).unwrap();
        assert_eq!(cfg.run_id.as_deref(), Some("r-1"));
        assert!(cfg.thread_id.is_none());
        assert!(cfg.tags.is_empty());
    }

    #[test]
    fn test_runtime_config_debug() {
        let cfg = RuntimeConfig::new().with_run_id("r-1");
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("RuntimeConfig"));
        assert!(dbg.contains("r-1"));
    }

    #[test]
    fn test_runtime_config_equality() {
        let a = RuntimeConfig::new().with_run_id("r-1");
        let b = RuntimeConfig::new().with_run_id("r-1");
        assert_eq!(a, b);
    }

    #[test]
    fn test_runtime_config_inequality() {
        let a = RuntimeConfig::new().with_run_id("r-1");
        let b = RuntimeConfig::new().with_run_id("r-2");
        assert_ne!(a, b);
    }

    // -- Runtime -----------------------------------------------------------

    #[test]
    fn test_runtime_new_default() {
        let rt = Runtime::<NoContext>::new();
        assert_eq!(*rt.context(), NoContext);
        assert!(rt.store().is_none());
        assert!(rt.previous().is_none());
    }

    #[test]
    fn test_runtime_default_impl() {
        let rt = Runtime::<NoContext>::default();
        assert_eq!(*rt.context(), NoContext);
    }

    #[test]
    fn test_runtime_builder_basic() {
        let rt: Runtime<NoContext> = RuntimeBuilder::new().build();
        assert!(rt.store().is_none());
        assert!(rt.previous().is_none());
    }

    #[test]
    fn test_runtime_builder_with_context() {
        #[derive(Debug, Clone, Default, PartialEq)]
        struct Ctx {
            user_id: String,
        }
        let rt = RuntimeBuilder::new()
            .with_context(Ctx {
                user_id: "u-1".into(),
            })
            .build();
        assert_eq!(rt.context().user_id, "u-1");
    }

    #[test]
    fn test_runtime_builder_with_store() {
        let store = Arc::new(InMemoryStore::new());
        store
            .put(&["ns"], "k", json!(1))
            .expect("put should succeed");
        let rt = RuntimeBuilder::new().with_store(store).build();
        assert!(rt.store().is_some());
        let v = rt.store().unwrap().get(&["ns"], "k").unwrap();
        assert_eq!(v, Some(json!(1)));
    }

    #[test]
    fn test_runtime_builder_with_stream_writer() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let c = captured.clone();
        let writer: StreamWriter = Arc::new(move |v| {
            c.lock().unwrap().push(v);
        });
        let rt = RuntimeBuilder::new().with_stream_writer(writer).build();
        rt.write_to_stream(json!("hello"));
        rt.write_to_stream(json!(42));
        let vals = captured.lock().unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], json!("hello"));
        assert_eq!(vals[1], json!(42));
    }

    #[test]
    fn test_runtime_builder_with_config() {
        let cfg = RuntimeConfig::new().with_run_id("r-99");
        let rt = RuntimeBuilder::new().with_config(cfg).build();
        assert_eq!(rt.config().run_id.as_deref(), Some("r-99"));
    }

    #[test]
    fn test_runtime_builder_with_previous() {
        let rt = RuntimeBuilder::new().with_previous(json!("prev")).build();
        assert_eq!(rt.previous(), Some(&json!("prev")));
    }

    #[test]
    fn test_runtime_builder_full() {
        let store = Arc::new(InMemoryStore::new());
        let cfg = RuntimeConfig::new().with_run_id("r-1").with_tag("test");
        let rt = RuntimeBuilder::new()
            .with_context(42u32)
            .with_store(store)
            .with_config(cfg)
            .with_previous(json!({"step": 1}))
            .build();
        assert_eq!(*rt.context(), 42u32);
        assert!(rt.store().is_some());
        assert_eq!(rt.config().run_id.as_deref(), Some("r-1"));
        assert_eq!(rt.previous(), Some(&json!({"step": 1})));
    }

    #[test]
    fn test_runtime_with_context_changes_type() {
        let rt = Runtime::<NoContext>::new();
        let rt2 = rt.with_context(String::from("hello"));
        assert_eq!(rt2.context(), "hello");
    }

    #[test]
    fn test_runtime_with_store() {
        let rt = Runtime::<NoContext>::new();
        let store = Arc::new(InMemoryStore::new());
        let rt2 = rt.with_store(Some(store));
        assert!(rt2.store().is_some());
    }

    #[test]
    fn test_runtime_with_store_none() {
        let store = Arc::new(InMemoryStore::new());
        let rt = RuntimeBuilder::new().with_store(store).build();
        assert!(rt.store().is_some());
        let rt2 = rt.with_store(None);
        assert!(rt2.store().is_none());
    }

    #[test]
    fn test_runtime_with_config() {
        let rt = Runtime::<NoContext>::new();
        let cfg = RuntimeConfig::new().with_thread_id("t-5");
        let rt2 = rt.with_config(cfg);
        assert_eq!(rt2.config().thread_id.as_deref(), Some("t-5"));
    }

    #[test]
    fn test_runtime_with_previous() {
        let rt = Runtime::<NoContext>::new();
        let rt2 = rt.with_previous(Some(json!(99)));
        assert_eq!(rt2.previous(), Some(&json!(99)));
    }

    #[test]
    fn test_runtime_set_previous() {
        let mut rt = Runtime::<NoContext>::new();
        assert!(rt.previous().is_none());
        rt.set_previous(Some(json!("done")));
        assert_eq!(rt.previous(), Some(&json!("done")));
        rt.set_previous(None);
        assert!(rt.previous().is_none());
    }

    #[test]
    fn test_runtime_context_mut() {
        let mut rt = RuntimeBuilder::new().with_context(String::from("a")).build();
        rt.context_mut().push_str("bc");
        assert_eq!(rt.context(), "abc");
    }

    #[test]
    fn test_runtime_config_mut() {
        let mut rt = Runtime::<NoContext>::new();
        rt.config_mut().run_id = Some("modified".into());
        assert_eq!(rt.config().run_id.as_deref(), Some("modified"));
    }

    #[test]
    fn test_runtime_into_context() {
        let rt = RuntimeBuilder::new()
            .with_context(vec![1, 2, 3])
            .build();
        let ctx = rt.into_context();
        assert_eq!(ctx, vec![1, 2, 3]);
    }

    #[test]
    fn test_runtime_store_arc() {
        let store = Arc::new(InMemoryStore::new());
        let rt = RuntimeBuilder::new().with_store(store.clone()).build();
        let arc = rt.store_arc().unwrap();
        // Should be the same store (via Arc)
        arc.put(&["ns"], "k", json!("val")).unwrap();
        assert_eq!(store.get(&["ns"], "k").unwrap(), Some(json!("val")));
    }

    #[test]
    fn test_runtime_store_arc_none() {
        let rt = Runtime::<NoContext>::new();
        assert!(rt.store_arc().is_none());
    }

    #[test]
    fn test_runtime_map_context() {
        let rt = RuntimeBuilder::new().with_context(42i32).build();
        let rt2 = rt.map_context(|n| format!("value={}", n));
        assert_eq!(rt2.context(), "value=42");
    }

    #[test]
    fn test_runtime_merge_other_overrides() {
        let base = RuntimeBuilder::new()
            .with_context(1u32)
            .with_config(RuntimeConfig::new().with_run_id("r-base"))
            .with_previous(json!("base_prev"))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(2u32)
            .with_config(RuntimeConfig::new().with_run_id("r-other"))
            .with_previous(json!("other_prev"))
            .build();
        let merged = base.merge(other);
        assert_eq!(*merged.context(), 2u32);
        assert_eq!(merged.config().run_id.as_deref(), Some("r-other"));
        assert_eq!(merged.previous(), Some(&json!("other_prev")));
    }

    #[test]
    fn test_runtime_merge_fallback_to_base() {
        let store = Arc::new(InMemoryStore::new());
        let base = RuntimeBuilder::new()
            .with_context(1u32)
            .with_store(store)
            .with_config(RuntimeConfig::new().with_run_id("r-base").with_tag("base"))
            .with_previous(json!("base_prev"))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(2u32)
            .build();
        let merged = base.merge(other);
        // other has no store -> falls back to base store
        assert!(merged.store().is_some());
        // other has no run_id -> falls back to base
        assert_eq!(merged.config().run_id.as_deref(), Some("r-base"));
        // other has no previous -> falls back to base
        assert_eq!(merged.previous(), Some(&json!("base_prev")));
    }

    #[test]
    fn test_runtime_merge_tags_from_other() {
        let base = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_tag("a"))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_tag("b"))
            .build();
        let merged = base.merge(other);
        assert_eq!(merged.config().tags, vec!["b"]);
    }

    #[test]
    fn test_runtime_merge_empty_tags_keep_base() {
        let base = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_tag("a"))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(0u32)
            .build();
        let merged = base.merge(other);
        assert_eq!(merged.config().tags, vec!["a"]);
    }

    #[test]
    fn test_runtime_clone() {
        let rt = RuntimeBuilder::new()
            .with_context(String::from("ctx"))
            .with_config(RuntimeConfig::new().with_run_id("r-1"))
            .with_previous(json!(10))
            .build();
        let rt2 = rt.clone();
        assert_eq!(rt2.context(), "ctx");
        assert_eq!(rt2.config().run_id.as_deref(), Some("r-1"));
        assert_eq!(rt2.previous(), Some(&json!(10)));
    }

    #[test]
    fn test_runtime_debug() {
        let rt = Runtime::<NoContext>::new();
        let dbg = format!("{:?}", rt);
        assert!(dbg.contains("Runtime"));
        assert!(dbg.contains("NoContext"));
        assert!(dbg.contains("<StreamWriter>"));
    }

    #[test]
    fn test_runtime_no_op_stream_writer() {
        let rt = Runtime::<NoContext>::new();
        // Should not panic
        rt.write_to_stream(json!({"test": "data"}));
        rt.write_to_stream(Value::Null);
    }

    // -- RuntimeScope ------------------------------------------------------

    #[test]
    fn test_runtime_scope_new() {
        let rt = Runtime::<NoContext>::new();
        let scope = RuntimeScope::new(rt);
        assert_eq!(*scope.runtime().context(), NoContext);
    }

    #[test]
    fn test_runtime_scope_with_cleanup() {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = flag.clone();
        {
            let rt = Runtime::<NoContext>::new();
            let _scope = RuntimeScope::with_cleanup(rt, move || {
                f.store(true, std::sync::atomic::Ordering::SeqCst);
            });
            assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
        }
        // cleanup ran on drop
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_runtime_scope_into_inner_skips_cleanup() {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = flag.clone();
        let rt = Runtime::<NoContext>::new();
        let scope = RuntimeScope::with_cleanup(rt, move || {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        let _rt = scope.into_inner();
        // cleanup should NOT have run
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_runtime_scope_runtime_mut() {
        let rt = Runtime::<NoContext>::new();
        let mut scope = RuntimeScope::new(rt);
        scope.runtime_mut().set_previous(Some(json!("updated")));
        assert_eq!(scope.runtime().previous(), Some(&json!("updated")));
    }

    #[test]
    fn test_runtime_scope_implements_provider() {
        let rt = RuntimeBuilder::new().with_context(99u32).build();
        let scope = RuntimeScope::new(rt);
        fn use_provider<P: RuntimeProvider<u32>>(p: &P) -> u32 {
            *p.runtime().context()
        }
        assert_eq!(use_provider(&scope), 99);
    }

    #[test]
    fn test_runtime_scope_debug() {
        let rt = Runtime::<NoContext>::new();
        let scope = RuntimeScope::new(rt);
        let dbg = format!("{:?}", scope);
        assert!(dbg.contains("RuntimeScope"));
        assert!(dbg.contains("Runtime"));
    }

    // -- RuntimeProvider ---------------------------------------------------

    #[test]
    fn test_runtime_provider_custom_impl() {
        struct MyExecutor {
            rt: Runtime<String>,
        }
        impl RuntimeProvider<String> for MyExecutor {
            fn runtime(&self) -> &Runtime<String> {
                &self.rt
            }
        }
        let exec = MyExecutor {
            rt: RuntimeBuilder::new()
                .with_context(String::from("hello"))
                .build(),
        };
        assert_eq!(exec.runtime().context(), "hello");
    }

    // -- default_runtime ---------------------------------------------------

    #[test]
    fn test_default_runtime_fn() {
        let rt = default_runtime();
        assert_eq!(*rt.context(), NoContext);
        assert!(rt.store().is_none());
        assert!(rt.previous().is_none());
    }

    // -- StreamWriter helpers ----------------------------------------------

    #[test]
    fn test_no_op_stream_writer_fn() {
        let w = no_op_stream_writer();
        w(json!("anything"));
        w(Value::Null);
    }

    #[test]
    fn test_stream_writer_captures_values() {
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let b = buf.clone();
        let writer: StreamWriter = Arc::new(move |v| {
            b.lock().unwrap().push(v);
        });
        writer(json!(1));
        writer(json!(2));
        writer(json!(3));
        let vals = buf.lock().unwrap();
        assert_eq!(vals.len(), 3);
    }

    // -- edge cases --------------------------------------------------------

    #[test]
    fn test_runtime_builder_default_context() {
        // When using Default-able context, builder.build() should work without with_context
        let rt: Runtime<NoContext> = RuntimeBuilder::new().build();
        assert_eq!(*rt.context(), NoContext);
    }

    #[test]
    fn test_runtime_builder_default_impl() {
        let builder = RuntimeBuilder::default();
        let rt = builder.build();
        assert_eq!(*rt.context(), NoContext);
    }

    #[test]
    fn test_runtime_with_stream_writer_replaces() {
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let b = buf.clone();
        let rt = Runtime::<NoContext>::new()
            .with_stream_writer(Arc::new(move |v| {
                b.lock().unwrap().push(v);
            }));
        rt.write_to_stream(json!("x"));
        assert_eq!(buf.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_runtime_merge_metadata_from_other() {
        let base = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_metadata("k1", json!("v1")))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_metadata("k2", json!("v2")))
            .build();
        let merged = base.merge(other);
        // other has metadata, so it wins
        assert_eq!(merged.config().metadata.get("k2"), Some(&json!("v2")));
        assert!(merged.config().metadata.get("k1").is_none());
    }

    #[test]
    fn test_runtime_merge_empty_metadata_keeps_base() {
        let base = RuntimeBuilder::new()
            .with_context(0u32)
            .with_config(RuntimeConfig::new().with_metadata("k1", json!("v1")))
            .build();
        let other = RuntimeBuilder::new()
            .with_context(0u32)
            .build();
        let merged = base.merge(other);
        assert_eq!(merged.config().metadata.get("k1"), Some(&json!("v1")));
    }

    #[test]
    fn test_runtime_config_clone() {
        let cfg = RuntimeConfig::new()
            .with_run_id("r")
            .with_thread_id("t")
            .with_tag("x")
            .with_metadata("m", json!(true));
        let cfg2 = cfg.clone();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_no_context_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(NoContext);
        assert!(set.contains(&NoContext));
    }
}
