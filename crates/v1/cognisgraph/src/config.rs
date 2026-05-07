//! Top-level configuration access for LangGraph.
//!
//! Provides functions to access the current runnable configuration, store,
//! and stream writer from within a graph node at runtime. This is the Rust
//! equivalent of Python's `cognisgraph.config` module.
//!
//! In the Python version, these functions rely on `contextvars` to access
//! the current execution context. In Rust, we use explicit configuration
//! passing via `RunnableConfig` and a `RuntimeConfig` struct.

use crate::errors::{LangGraphError, Result};
use crate::utils::config::RunnableConfig;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for a stream writer function.
///
/// A stream writer accepts a `serde_json::Value` and writes it to the
/// output stream. This is the Rust equivalent of Python's `StreamWriter`.
pub type StreamWriter = Arc<dyn Fn(Value) + Send + Sync>;

/// Runtime configuration that is available during graph execution.
///
/// This struct holds references to the store and stream writer that are
/// active during the current execution. In Python, these are accessed via
/// `get_config()[CONF][CONFIG_KEY_RUNTIME]`.
#[derive(Clone)]
pub struct RuntimeConfig {
    /// The store for persistent state, if any.
    pub store: Option<Arc<dyn Store>>,
    /// The stream writer for custom streaming output.
    pub stream_writer: StreamWriter,
}

impl std::fmt::Debug for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("store", &self.store.as_ref().map(|_| "<store>"))
            .field("stream_writer", &"<fn>")
            .finish()
    }
}

/// Trait for a key-value store that can be accessed from within graph nodes.
///
/// This is a minimal Rust equivalent of Python's `BaseStore`.
pub trait Store: Send + Sync {
    /// Get a value from the store by namespace and key.
    fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Value>>;

    /// Put a value into the store.
    fn put(&self, namespace: &[&str], key: &str, value: Value) -> Result<()>;

    /// Delete a value from the store.
    fn delete(&self, namespace: &[&str], key: &str) -> Result<()>;

    /// List keys in a namespace.
    fn list(&self, namespace: &[&str]) -> Result<Vec<String>>;
}

/// A no-op stream writer that discards all output.
///
/// This is the Rust equivalent of Python's `_no_op_stream_writer`.
pub fn no_op_stream_writer() -> StreamWriter {
    Arc::new(|_: Value| {})
}

impl RuntimeConfig {
    /// Create a new runtime config with no store and a no-op stream writer.
    pub fn new() -> Self {
        Self {
            store: None,
            stream_writer: no_op_stream_writer(),
        }
    }

    /// Create a runtime config with the specified store.
    pub fn with_store(mut self, store: Arc<dyn Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// Create a runtime config with the specified stream writer.
    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = writer;
        self
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the store from a runnable config's configurable section.
///
/// This is the Rust equivalent of Python's `get_store()`. It looks for the
/// runtime config in the configurable section and returns the store if present.
pub fn get_store_from_config(config: &RunnableConfig) -> Result<Option<Arc<dyn Store>>> {
    // In a full implementation, the RuntimeConfig would be embedded in the
    // configurable section. For now, we return None if not found.
    if config
        .configurable
        .contains_key(crate::utils::config::CONFIG_KEY_RUNTIME)
    {
        // The runtime config would be deserialized from the configurable section.
        // Since RuntimeConfig contains non-serializable types (Arc<dyn Store>),
        // in practice it would be passed via a separate mechanism.
        Ok(None)
    } else {
        Ok(None)
    }
}

/// Extract the stream writer from a runnable config.
///
/// Returns a no-op writer if no stream writer is configured.
pub fn get_stream_writer_from_config(_config: &RunnableConfig) -> StreamWriter {
    // In a full implementation, the stream writer would be extracted from
    // the runtime config. For now, return a no-op writer.
    no_op_stream_writer()
}

/// An in-memory store implementation for testing and simple use cases.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    data: std::sync::RwLock<HashMap<String, HashMap<String, Value>>>,
}

impl InMemoryStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn namespace_key(namespace: &[&str]) -> String {
        namespace.join("/")
    }
}

impl Store for InMemoryStore {
    fn get(&self, namespace: &[&str], key: &str) -> Result<Option<Value>> {
        let ns_key = Self::namespace_key(namespace);
        let data = self
            .data
            .read()
            .map_err(|e| LangGraphError::Other(format!("Lock poisoned: {}", e)))?;
        Ok(data.get(&ns_key).and_then(|ns| ns.get(key).cloned()))
    }

    fn put(&self, namespace: &[&str], key: &str, value: Value) -> Result<()> {
        let ns_key = Self::namespace_key(namespace);
        let mut data = self
            .data
            .write()
            .map_err(|e| LangGraphError::Other(format!("Lock poisoned: {}", e)))?;
        data.entry(ns_key)
            .or_default()
            .insert(key.to_string(), value);
        Ok(())
    }

    fn delete(&self, namespace: &[&str], key: &str) -> Result<()> {
        let ns_key = Self::namespace_key(namespace);
        let mut data = self
            .data
            .write()
            .map_err(|e| LangGraphError::Other(format!("Lock poisoned: {}", e)))?;
        if let Some(ns) = data.get_mut(&ns_key) {
            ns.remove(key);
        }
        Ok(())
    }

    fn list(&self, namespace: &[&str]) -> Result<Vec<String>> {
        let ns_key = Self::namespace_key(namespace);
        let data = self
            .data
            .read()
            .map_err(|e| LangGraphError::Other(format!("Lock poisoned: {}", e)))?;
        Ok(data
            .get(&ns_key)
            .map(|ns| ns.keys().cloned().collect())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_no_op_stream_writer() {
        let writer = no_op_stream_writer();
        // Should not panic
        writer(json!({"test": "data"}));
        writer(Value::Null);
    }

    #[test]
    fn test_runtime_config_new() {
        let config = RuntimeConfig::new();
        assert!(config.store.is_none());
        // Stream writer should be callable without panic
        (config.stream_writer)(json!("test"));
    }

    #[test]
    fn test_runtime_config_with_store() {
        let store = Arc::new(InMemoryStore::new());
        let config = RuntimeConfig::new().with_store(store);
        assert!(config.store.is_some());
    }

    #[test]
    fn test_runtime_config_with_stream_writer() {
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();
        let writer: StreamWriter = Arc::new(move |v: Value| {
            received_clone.lock().unwrap().push(v);
        });
        let config = RuntimeConfig::new().with_stream_writer(writer);
        (config.stream_writer)(json!("hello"));
        (config.stream_writer)(json!(42));

        let msgs = received.lock().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], json!("hello"));
        assert_eq!(msgs[1], json!(42));
    }

    #[test]
    fn test_runtime_config_debug() {
        let config = RuntimeConfig::new();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("RuntimeConfig"));
    }

    #[test]
    fn test_in_memory_store_put_get() {
        let store = InMemoryStore::new();
        store
            .put(&["ns", "sub"], "key1", json!({"value": 42}))
            .unwrap();

        let result = store.get(&["ns", "sub"], "key1").unwrap();
        assert_eq!(result, Some(json!({"value": 42})));
    }

    #[test]
    fn test_in_memory_store_get_missing() {
        let store = InMemoryStore::new();
        let result = store.get(&["ns"], "missing").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_in_memory_store_delete() {
        let store = InMemoryStore::new();
        store.put(&["ns"], "key1", json!(1)).unwrap();
        store.delete(&["ns"], "key1").unwrap();
        let result = store.get(&["ns"], "key1").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_in_memory_store_delete_nonexistent() {
        let store = InMemoryStore::new();
        // Should not error
        store.delete(&["ns"], "nonexistent").unwrap();
    }

    #[test]
    fn test_in_memory_store_list() {
        let store = InMemoryStore::new();
        store.put(&["ns"], "a", json!(1)).unwrap();
        store.put(&["ns"], "b", json!(2)).unwrap();
        store.put(&["other"], "c", json!(3)).unwrap();

        let mut keys = store.list(&["ns"]).unwrap();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_in_memory_store_list_empty_namespace() {
        let store = InMemoryStore::new();
        let keys = store.list(&["nonexistent"]).unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_in_memory_store_overwrite() {
        let store = InMemoryStore::new();
        store.put(&["ns"], "key", json!(1)).unwrap();
        store.put(&["ns"], "key", json!(2)).unwrap();
        let result = store.get(&["ns"], "key").unwrap();
        assert_eq!(result, Some(json!(2)));
    }

    #[test]
    fn test_get_store_from_config_empty() {
        let config = RunnableConfig::new();
        let store = get_store_from_config(&config).unwrap();
        assert!(store.is_none());
    }

    #[test]
    fn test_get_stream_writer_from_config() {
        let config = RunnableConfig::new();
        let writer = get_stream_writer_from_config(&config);
        // Should not panic
        writer(json!("test"));
    }

    #[test]
    fn test_in_memory_store_namespace_isolation() {
        let store = InMemoryStore::new();
        store.put(&["a", "b"], "key", json!(1)).unwrap();
        store.put(&["a", "c"], "key", json!(2)).unwrap();

        assert_eq!(store.get(&["a", "b"], "key").unwrap(), Some(json!(1)));
        assert_eq!(store.get(&["a", "c"], "key").unwrap(), Some(json!(2)));
        assert_eq!(store.get(&["a"], "key").unwrap(), None);
    }
}
