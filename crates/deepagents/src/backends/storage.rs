//! Pluggable key-value storage backends for DeepAgents.
//!
//! Provides a [`StorageBackend`] trait and several implementations for
//! persisting agent session state, configuration, and intermediate results.
//!
//! - [`InMemoryBackend`] — fast, ephemeral HashMap-based storage
//! - [`FilesystemBackend`] — durable JSON-file-per-key storage on disk
//! - [`NamespacedBackend`] — key-prefixing wrapper around any backend
//! - [`CachedBackend`] — in-memory caching layer over any backend
//! - [`BackendConfig`] — declarative configuration for backend creation
//! - [`BackendFactory`] — creates backends from [`BackendConfig`]
//! - [`BackendStats`] — runtime statistics snapshot of a backend

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde_json::Value;

// ---------------------------------------------------------------------------
// StorageBackend trait
// ---------------------------------------------------------------------------

/// Trait for pluggable key-value storage backends.
///
/// All keys and values are string-keyed JSON values. Implementations must
/// support basic CRUD operations, key enumeration, and bulk clearing.
pub trait StorageBackend: fmt::Debug {
    /// Returns the human-readable name of this backend.
    fn name(&self) -> &str;

    /// Persist a JSON value under the given key.
    fn save(&mut self, key: &str, value: &Value) -> Result<(), String>;

    /// Load the JSON value stored under the given key, or `None` if absent.
    fn load(&self, key: &str) -> Result<Option<Value>, String>;

    /// Delete the value stored under the given key. Returns `true` if the
    /// key existed and was removed.
    fn delete(&mut self, key: &str) -> Result<bool, String>;

    /// Returns `true` if a value exists for the given key.
    fn exists(&self, key: &str) -> bool;

    /// Returns a list of all keys currently stored.
    fn keys(&self) -> Vec<String>;

    /// Remove all stored key-value pairs.
    fn clear(&mut self) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// InMemoryBackend
// ---------------------------------------------------------------------------

/// Fast, ephemeral in-memory backend backed by a [`HashMap`].
#[derive(Debug)]
pub struct InMemoryBackend {
    data: HashMap<String, Value>,
}

impl InMemoryBackend {
    /// Create a new, empty in-memory backend.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Returns the number of stored entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the backend contains no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for InMemoryBackend {
    fn name(&self) -> &str {
        "memory"
    }

    fn save(&mut self, key: &str, value: &Value) -> Result<(), String> {
        self.data.insert(key.to_string(), value.clone());
        Ok(())
    }

    fn load(&self, key: &str) -> Result<Option<Value>, String> {
        Ok(self.data.get(key).cloned())
    }

    fn delete(&mut self, key: &str) -> Result<bool, String> {
        Ok(self.data.remove(key).is_some())
    }

    fn exists(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    fn clear(&mut self) -> Result<(), String> {
        self.data.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FilesystemStorageBackend
// ---------------------------------------------------------------------------

/// Durable backend that stores each key as a separate JSON file on disk.
///
/// Keys are mapped to files at `<base_path>/<key>.json`. The base directory
/// is created automatically if it does not exist.
#[derive(Debug)]
pub struct FilesystemStorageBackend {
    base_path: PathBuf,
}

impl FilesystemStorageBackend {
    /// Create a new filesystem storage backend rooted at `base_path`.
    ///
    /// The directory is created (including parents) if it does not exist.
    pub fn new(base_path: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&base_path)
            .map_err(|e| format!("failed to create backend directory: {}", e))?;
        Ok(Self { base_path })
    }

    /// Returns the base directory path.
    pub fn base_path(&self) -> &PathBuf {
        &self.base_path
    }

    /// Compute the file path for a given key.
    fn key_path(&self, key: &str) -> PathBuf {
        self.base_path.join(format!("{}.json", key))
    }
}

impl StorageBackend for FilesystemStorageBackend {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn save(&mut self, key: &str, value: &Value) -> Result<(), String> {
        let path = self.key_path(key);
        let content = serde_json::to_string_pretty(value)
            .map_err(|e| format!("failed to serialize value: {}", e))?;
        std::fs::write(&path, content)
            .map_err(|e| format!("failed to write file {}: {}", path.display(), e))?;
        Ok(())
    }

    fn load(&self, key: &str) -> Result<Option<Value>, String> {
        let path = self.key_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read file {}: {}", path.display(), e))?;
        let value: Value = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse JSON from {}: {}", path.display(), e))?;
        Ok(Some(value))
    }

    fn delete(&mut self, key: &str) -> Result<bool, String> {
        let path = self.key_path(key);
        if !path.exists() {
            return Ok(false);
        }
        std::fs::remove_file(&path)
            .map_err(|e| format!("failed to delete file {}: {}", path.display(), e))?;
        Ok(true)
    }

    fn exists(&self, key: &str) -> bool {
        self.key_path(key).exists()
    }

    fn keys(&self) -> Vec<String> {
        let entries = match std::fs::read_dir(&self.base_path) {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };
        entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn clear(&mut self) -> Result<(), String> {
        let keys = self.keys();
        for key in keys {
            self.delete(&key)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NamespacedBackend
// ---------------------------------------------------------------------------

/// Wraps another backend and prefixes all keys with a namespace.
///
/// Keys are transformed to `<namespace>:<key>` before being passed to the
/// inner backend.
#[derive(Debug)]
pub struct NamespacedBackend {
    inner: Box<dyn StorageBackend>,
    namespace: String,
}

impl NamespacedBackend {
    /// Create a new namespaced wrapper around `inner`.
    pub fn new(inner: Box<dyn StorageBackend>, namespace: String) -> Self {
        Self { inner, namespace }
    }

    /// Produce the prefixed key.
    fn prefixed(&self, key: &str) -> String {
        format!("{}:{}", self.namespace, key)
    }

    /// Returns the namespace string.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

impl StorageBackend for NamespacedBackend {
    fn name(&self) -> &str {
        "namespaced"
    }

    fn save(&mut self, key: &str, value: &Value) -> Result<(), String> {
        let prefixed = self.prefixed(key);
        self.inner.save(&prefixed, value)
    }

    fn load(&self, key: &str) -> Result<Option<Value>, String> {
        let prefixed = self.prefixed(key);
        self.inner.load(&prefixed)
    }

    fn delete(&mut self, key: &str) -> Result<bool, String> {
        let prefixed = self.prefixed(key);
        self.inner.delete(&prefixed)
    }

    fn exists(&self, key: &str) -> bool {
        let prefixed = self.prefixed(key);
        self.inner.exists(&prefixed)
    }

    fn keys(&self) -> Vec<String> {
        let prefix = format!("{}:", self.namespace);
        self.inner
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
            .collect()
    }

    fn clear(&mut self) -> Result<(), String> {
        let keys: Vec<String> = self.keys();
        for key in keys {
            self.delete(&key)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CachedBackend
// ---------------------------------------------------------------------------

/// Wraps a backend with an in-memory cache.
///
/// On [`load`](StorageBackend::load), the cache is checked first. On
/// [`save`](StorageBackend::save), the cache is updated alongside the
/// underlying backend.
#[derive(Debug)]
pub struct CachedBackend {
    inner: Box<dyn StorageBackend>,
    cache: HashMap<String, Value>,
    max_cache_size: usize,
    hits: usize,
    misses: usize,
}

impl CachedBackend {
    /// Create a new cached wrapper around `inner` with the given maximum
    /// cache size.
    pub fn new(inner: Box<dyn StorageBackend>, max_cache_size: usize) -> Self {
        Self {
            inner,
            cache: HashMap::new(),
            max_cache_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Returns the number of cache hits since creation.
    pub fn cache_hits(&self) -> usize {
        self.hits
    }

    /// Returns the number of cache misses since creation.
    pub fn cache_misses(&self) -> usize {
        self.misses
    }

    /// Invalidate (remove) a single key from the cache.
    pub fn invalidate(&mut self, key: &str) {
        self.cache.remove(key);
    }

    /// Clear the entire cache without touching the underlying backend.
    pub fn invalidate_all(&mut self) {
        self.cache.clear();
    }

    /// Evict entries if the cache exceeds its maximum size.
    fn maybe_evict(&mut self) {
        while self.cache.len() > self.max_cache_size {
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
    }

    /// Mutable load that properly tracks cache hits/misses without
    /// interior mutability hacks.
    pub fn load_mut(&mut self, key: &str) -> Result<Option<Value>, String> {
        if let Some(cached) = self.cache.get(key) {
            self.hits += 1;
            return Ok(Some(cached.clone()));
        }
        self.misses += 1;
        let result = self.inner.load(key)?;
        if let Some(ref val) = result {
            self.cache.insert(key.to_string(), val.clone());
            self.maybe_evict();
        }
        Ok(result)
    }
}

impl StorageBackend for CachedBackend {
    fn name(&self) -> &str {
        "cached"
    }

    fn save(&mut self, key: &str, value: &Value) -> Result<(), String> {
        self.inner.save(key, value)?;
        self.cache.insert(key.to_string(), value.clone());
        self.maybe_evict();
        Ok(())
    }

    fn load(&self, key: &str) -> Result<Option<Value>, String> {
        if let Some(cached) = self.cache.get(key) {
            return Ok(Some(cached.clone()));
        }
        self.inner.load(key)
    }

    fn delete(&mut self, key: &str) -> Result<bool, String> {
        self.cache.remove(key);
        self.inner.delete(key)
    }

    fn exists(&self, key: &str) -> bool {
        self.cache.contains_key(key) || self.inner.exists(key)
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    fn clear(&mut self) -> Result<(), String> {
        self.cache.clear();
        self.inner.clear()
    }
}

// ---------------------------------------------------------------------------
// BackendConfig
// ---------------------------------------------------------------------------

/// Declarative configuration for constructing a storage backend.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// The type of backend to create (`"memory"`, `"filesystem"`,
    /// `"namespaced"`, `"cached"`).
    pub backend_type: String,
    /// Base path for filesystem backends.
    pub base_path: Option<PathBuf>,
    /// Namespace for namespaced backends.
    pub namespace: Option<String>,
    /// Maximum cache size for cached backends.
    pub cache_size: Option<usize>,
    /// Arbitrary extra configuration values.
    pub extra: HashMap<String, Value>,
}

impl BackendConfig {
    /// Create a new config for the given backend type.
    pub fn new(backend_type: impl Into<String>) -> Self {
        Self {
            backend_type: backend_type.into(),
            base_path: None,
            namespace: None,
            cache_size: None,
            extra: HashMap::new(),
        }
    }

    /// Set the base path (for filesystem backends).
    pub fn with_base_path(mut self, path: PathBuf) -> Self {
        self.base_path = Some(path);
        self
    }

    /// Set the namespace (for namespaced backends).
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Set the cache size (for cached backends).
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }

    /// Set an arbitrary extra configuration value.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Serialize this configuration to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "backend_type": self.backend_type,
            "base_path": self.base_path.as_ref().map(|p| p.display().to_string()),
            "namespace": self.namespace,
            "cache_size": self.cache_size,
            "extra": self.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// BackendFactory
// ---------------------------------------------------------------------------

/// Factory for creating storage backends from configuration.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend from the given configuration.
    ///
    /// Supported backend types:
    /// - `"memory"` — [`InMemoryBackend`]
    /// - `"filesystem"` — [`FilesystemStorageBackend`] (requires `base_path`)
    /// - `"namespaced"` — [`NamespacedBackend`] wrapping an in-memory backend
    ///   (requires `namespace`)
    /// - `"cached"` — [`CachedBackend`] wrapping an in-memory backend
    ///   (uses `cache_size`, defaults to 1000)
    pub fn create(config: &BackendConfig) -> Result<Box<dyn StorageBackend>, String> {
        match config.backend_type.as_str() {
            "memory" => Ok(Box::new(InMemoryBackend::new())),
            "filesystem" => {
                let path = config
                    .base_path
                    .clone()
                    .ok_or_else(|| "filesystem backend requires base_path".to_string())?;
                let backend = FilesystemStorageBackend::new(path)?;
                Ok(Box::new(backend))
            }
            "namespaced" => {
                let namespace = config
                    .namespace
                    .clone()
                    .ok_or_else(|| "namespaced backend requires namespace".to_string())?;
                let inner = Box::new(InMemoryBackend::new());
                Ok(Box::new(NamespacedBackend::new(inner, namespace)))
            }
            "cached" => {
                let max_size = config.cache_size.unwrap_or(1000);
                let inner = Box::new(InMemoryBackend::new());
                Ok(Box::new(CachedBackend::new(inner, max_size)))
            }
            other => Err(format!("unknown backend type: {}", other)),
        }
    }
}

// ---------------------------------------------------------------------------
// BackendStats
// ---------------------------------------------------------------------------

/// A point-in-time statistics snapshot of a storage backend.
#[derive(Debug, Clone)]
pub struct BackendStats {
    /// Number of keys currently stored.
    pub total_keys: usize,
    /// Name of the backend.
    pub backend_name: String,
    /// Arbitrary extra statistics.
    pub extra: HashMap<String, Value>,
}

impl BackendStats {
    /// Collect statistics from the given backend.
    pub fn from_backend(backend: &dyn StorageBackend) -> Self {
        Self {
            total_keys: backend.keys().len(),
            backend_name: backend.name().to_string(),
            extra: HashMap::new(),
        }
    }

    /// Serialize these statistics to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_keys": self.total_keys,
            "backend_name": self.backend_name,
            "extra": self.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- InMemoryBackend CRUD --

    #[test]
    fn test_in_memory_new_is_empty() {
        let backend = InMemoryBackend::new();
        assert!(backend.is_empty());
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_in_memory_save_and_load() {
        let mut backend = InMemoryBackend::new();
        backend
            .save("key1", &serde_json::json!({"data": 42}))
            .unwrap();
        let val = backend.load("key1").unwrap();
        assert_eq!(val, Some(serde_json::json!({"data": 42})));
    }

    #[test]
    fn test_in_memory_load_missing_key() {
        let backend = InMemoryBackend::new();
        let val = backend.load("nonexistent").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn test_in_memory_overwrite() {
        let mut backend = InMemoryBackend::new();
        backend.save("key", &serde_json::json!(1)).unwrap();
        backend.save("key", &serde_json::json!(2)).unwrap();
        assert_eq!(backend.load("key").unwrap(), Some(serde_json::json!(2)));
        assert_eq!(backend.len(), 1);
    }

    #[test]
    fn test_in_memory_delete_existing() {
        let mut backend = InMemoryBackend::new();
        backend.save("key", &serde_json::json!("val")).unwrap();
        let removed = backend.delete("key").unwrap();
        assert!(removed);
        assert!(!backend.exists("key"));
    }

    #[test]
    fn test_in_memory_delete_missing() {
        let mut backend = InMemoryBackend::new();
        let removed = backend.delete("nope").unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_in_memory_exists() {
        let mut backend = InMemoryBackend::new();
        assert!(!backend.exists("key"));
        backend.save("key", &Value::Null).unwrap();
        assert!(backend.exists("key"));
    }

    #[test]
    fn test_in_memory_keys() {
        let mut backend = InMemoryBackend::new();
        backend.save("a", &Value::Null).unwrap();
        backend.save("b", &Value::Null).unwrap();
        backend.save("c", &Value::Null).unwrap();
        let mut keys = backend.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_in_memory_clear() {
        let mut backend = InMemoryBackend::new();
        backend.save("a", &Value::Null).unwrap();
        backend.save("b", &Value::Null).unwrap();
        backend.clear().unwrap();
        assert!(backend.is_empty());
        assert_eq!(backend.keys().len(), 0);
    }

    #[test]
    fn test_in_memory_name() {
        let backend = InMemoryBackend::new();
        assert_eq!(backend.name(), "memory");
    }

    #[test]
    fn test_in_memory_default() {
        let backend = InMemoryBackend::default();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_in_memory_len_after_operations() {
        let mut backend = InMemoryBackend::new();
        backend.save("a", &Value::Null).unwrap();
        backend.save("b", &Value::Null).unwrap();
        assert_eq!(backend.len(), 2);
        backend.delete("a").unwrap();
        assert_eq!(backend.len(), 1);
    }

    // -- FilesystemStorageBackend --

    fn temp_backend_dir() -> PathBuf {
        std::env::temp_dir().join(format!("deepagents_test_{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn test_filesystem_new_creates_dir() {
        let dir = temp_backend_dir();
        let _backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        assert!(dir.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_save_and_load() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        backend
            .save("test_key", &serde_json::json!({"hello": "world"}))
            .unwrap();
        let val = backend.load("test_key").unwrap();
        assert_eq!(val, Some(serde_json::json!({"hello": "world"})));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_load_missing() {
        let dir = temp_backend_dir();
        let backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        let val = backend.load("nonexistent").unwrap();
        assert!(val.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_delete() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        backend.save("del_key", &serde_json::json!(1)).unwrap();
        assert!(backend.delete("del_key").unwrap());
        assert!(!backend.exists("del_key"));
        assert!(!backend.delete("del_key").unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_exists() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        assert!(!backend.exists("mykey"));
        backend.save("mykey", &Value::Null).unwrap();
        assert!(backend.exists("mykey"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_keys() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        backend.save("alpha", &Value::Null).unwrap();
        backend.save("beta", &Value::Null).unwrap();
        let mut keys = backend.keys();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_clear() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        backend.save("x", &Value::Null).unwrap();
        backend.save("y", &Value::Null).unwrap();
        backend.clear().unwrap();
        assert!(backend.keys().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_overwrite() {
        let dir = temp_backend_dir();
        let mut backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        backend.save("key", &serde_json::json!(1)).unwrap();
        backend.save("key", &serde_json::json!(2)).unwrap();
        assert_eq!(backend.load("key").unwrap(), Some(serde_json::json!(2)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_base_path() {
        let dir = temp_backend_dir();
        let backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        assert_eq!(backend.base_path(), &dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_filesystem_name() {
        let dir = temp_backend_dir();
        let backend = FilesystemStorageBackend::new(dir.clone()).unwrap();
        assert_eq!(backend.name(), "filesystem");
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- NamespacedBackend --

    #[test]
    fn test_namespaced_save_and_load() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "ns1".to_string());
        ns.save("key", &serde_json::json!("val")).unwrap();
        assert_eq!(ns.load("key").unwrap(), Some(serde_json::json!("val")));
    }

    #[test]
    fn test_namespaced_key_prefixing() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "myns".to_string());
        ns.save("key", &Value::Null).unwrap();
        // The inner backend should have the prefixed key.
        assert!(ns.inner.exists("myns:key"));
        assert!(!ns.inner.exists("key"));
    }

    #[test]
    fn test_namespaced_keys_strips_prefix() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "pfx".to_string());
        ns.save("a", &Value::Null).unwrap();
        ns.save("b", &Value::Null).unwrap();
        let mut keys = ns.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_namespaced_delete() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "ns".to_string());
        ns.save("k", &Value::Null).unwrap();
        assert!(ns.delete("k").unwrap());
        assert!(!ns.exists("k"));
    }

    #[test]
    fn test_namespaced_clear() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "ns".to_string());
        ns.save("a", &Value::Null).unwrap();
        ns.save("b", &Value::Null).unwrap();
        ns.clear().unwrap();
        assert!(ns.keys().is_empty());
    }

    #[test]
    fn test_namespaced_isolation() {
        let mut inner = InMemoryBackend::new();
        inner
            .save("other:key", &serde_json::json!("other"))
            .unwrap();
        inner.save("myns:key", &serde_json::json!("mine")).unwrap();
        let ns = NamespacedBackend::new(Box::new(inner), "myns".to_string());
        let keys = ns.keys();
        assert_eq!(keys, vec!["key"]);
        assert_eq!(ns.load("key").unwrap(), Some(serde_json::json!("mine")));
    }

    #[test]
    fn test_namespaced_name() {
        let ns = NamespacedBackend::new(Box::new(InMemoryBackend::new()), "ns".to_string());
        assert_eq!(ns.name(), "namespaced");
    }

    #[test]
    fn test_namespaced_namespace_accessor() {
        let ns = NamespacedBackend::new(Box::new(InMemoryBackend::new()), "test_ns".to_string());
        assert_eq!(ns.namespace(), "test_ns");
    }

    // -- CachedBackend --

    #[test]
    fn test_cached_save_and_load() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("key", &serde_json::json!("value")).unwrap();
        let val = cached.load("key").unwrap();
        assert_eq!(val, Some(serde_json::json!("value")));
    }

    #[test]
    fn test_cached_hits_on_second_load() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("k", &serde_json::json!(1)).unwrap();

        // First load_mut after save: should be a cache hit (save populates cache).
        let _ = cached.load_mut("k").unwrap();
        assert_eq!(cached.cache_hits(), 1);
        assert_eq!(cached.cache_misses(), 0);

        // Second load_mut: still a hit.
        let _ = cached.load_mut("k").unwrap();
        assert_eq!(cached.cache_hits(), 2);
    }

    #[test]
    fn test_cached_miss_on_uncached_key() {
        let mut inner = InMemoryBackend::new();
        inner.save("pre", &serde_json::json!("existing")).unwrap();
        let mut cached = CachedBackend::new(Box::new(inner), 100);

        // Loading a key that exists in inner but not in cache.
        let val = cached.load_mut("pre").unwrap();
        assert_eq!(val, Some(serde_json::json!("existing")));
        assert_eq!(cached.cache_misses(), 1);
        assert_eq!(cached.cache_hits(), 0);

        // Now it should be cached.
        let _ = cached.load_mut("pre").unwrap();
        assert_eq!(cached.cache_hits(), 1);
    }

    #[test]
    fn test_cached_invalidate_single() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("k", &serde_json::json!(1)).unwrap();

        // Load to confirm cached.
        let _ = cached.load_mut("k").unwrap();
        assert_eq!(cached.cache_hits(), 1);

        // Invalidate.
        cached.invalidate("k");

        // Next load should be a miss (fetches from inner).
        let val = cached.load_mut("k").unwrap();
        assert_eq!(val, Some(serde_json::json!(1)));
        assert_eq!(cached.cache_misses(), 1);
    }

    #[test]
    fn test_cached_invalidate_all() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("a", &serde_json::json!(1)).unwrap();
        cached.save("b", &serde_json::json!(2)).unwrap();

        cached.invalidate_all();

        let _ = cached.load_mut("a").unwrap();
        let _ = cached.load_mut("b").unwrap();
        assert_eq!(cached.cache_misses(), 2);
    }

    #[test]
    fn test_cached_delete_removes_from_cache() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("k", &serde_json::json!(1)).unwrap();
        cached.delete("k").unwrap();
        let val = cached.load("k").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn test_cached_clear() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("a", &Value::Null).unwrap();
        cached.save("b", &Value::Null).unwrap();
        cached.clear().unwrap();
        assert!(cached.keys().is_empty());
        assert!(cached.load("a").unwrap().is_none());
    }

    #[test]
    fn test_cached_eviction() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 2);
        cached.save("a", &serde_json::json!(1)).unwrap();
        cached.save("b", &serde_json::json!(2)).unwrap();
        cached.save("c", &serde_json::json!(3)).unwrap();
        // Cache should have at most 2 entries, but all 3 exist in inner.
        assert_eq!(cached.keys().len(), 3);
    }

    #[test]
    fn test_cached_exists() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        assert!(!cached.exists("x"));
        cached.save("x", &Value::Null).unwrap();
        assert!(cached.exists("x"));
    }

    #[test]
    fn test_cached_name() {
        let cached = CachedBackend::new(Box::new(InMemoryBackend::new()), 10);
        assert_eq!(cached.name(), "cached");
    }

    #[test]
    fn test_cached_load_returns_from_cache_on_hit() {
        let inner = Box::new(InMemoryBackend::new());
        let mut cached = CachedBackend::new(inner, 100);
        cached.save("x", &serde_json::json!(42)).unwrap();
        // Trait load (immutable) should return cached value.
        let val = cached.load("x").unwrap();
        assert_eq!(val, Some(serde_json::json!(42)));
    }

    #[test]
    fn test_cached_load_falls_through_on_miss() {
        let mut inner = InMemoryBackend::new();
        inner.save("y", &serde_json::json!("inner")).unwrap();
        let cached = CachedBackend::new(Box::new(inner), 100);
        // Not in cache, falls through to inner via trait load.
        let val = cached.load("y").unwrap();
        assert_eq!(val, Some(serde_json::json!("inner")));
    }

    // -- BackendConfig --

    #[test]
    fn test_backend_config_new() {
        let config = BackendConfig::new("memory");
        assert_eq!(config.backend_type, "memory");
        assert!(config.base_path.is_none());
        assert!(config.namespace.is_none());
        assert!(config.cache_size.is_none());
        assert!(config.extra.is_empty());
    }

    #[test]
    fn test_backend_config_builder() {
        let config = BackendConfig::new("filesystem")
            .with_base_path(PathBuf::from("/tmp/test"))
            .with_namespace("ns")
            .with_cache_size(500)
            .with_extra("debug", serde_json::json!(true));
        assert_eq!(config.backend_type, "filesystem");
        assert_eq!(config.base_path, Some(PathBuf::from("/tmp/test")));
        assert_eq!(config.namespace, Some("ns".to_string()));
        assert_eq!(config.cache_size, Some(500));
        assert_eq!(config.extra.get("debug"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_backend_config_to_json() {
        let config = BackendConfig::new("memory").with_cache_size(100);
        let json = config.to_json();
        assert_eq!(json["backend_type"], "memory");
        assert_eq!(json["cache_size"], 100);
        assert!(json["base_path"].is_null());
    }

    #[test]
    fn test_backend_config_to_json_with_path() {
        let config = BackendConfig::new("filesystem").with_base_path(PathBuf::from("/data/store"));
        let json = config.to_json();
        assert_eq!(json["base_path"], "/data/store");
    }

    #[test]
    fn test_backend_config_multiple_extras() {
        let config = BackendConfig::new("memory")
            .with_extra("debug", serde_json::json!(true))
            .with_extra("version", serde_json::json!(2));
        assert_eq!(config.extra.len(), 2);
        assert_eq!(config.extra["debug"], serde_json::json!(true));
        assert_eq!(config.extra["version"], serde_json::json!(2));
    }

    // -- BackendFactory --

    #[test]
    fn test_factory_create_memory() {
        let config = BackendConfig::new("memory");
        let mut backend = BackendFactory::create(&config).unwrap();
        assert_eq!(backend.name(), "memory");
        backend.save("k", &serde_json::json!(1)).unwrap();
        assert_eq!(backend.load("k").unwrap(), Some(serde_json::json!(1)));
    }

    #[test]
    fn test_factory_create_filesystem() {
        let dir = temp_backend_dir();
        let config = BackendConfig::new("filesystem").with_base_path(dir.clone());
        let mut backend = BackendFactory::create(&config).unwrap();
        assert_eq!(backend.name(), "filesystem");
        backend.save("fk", &serde_json::json!("fv")).unwrap();
        assert_eq!(backend.load("fk").unwrap(), Some(serde_json::json!("fv")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_factory_create_namespaced() {
        let config = BackendConfig::new("namespaced").with_namespace("test_ns");
        let mut backend = BackendFactory::create(&config).unwrap();
        assert_eq!(backend.name(), "namespaced");
        backend.save("k", &Value::Null).unwrap();
        assert!(backend.exists("k"));
    }

    #[test]
    fn test_factory_create_cached() {
        let config = BackendConfig::new("cached").with_cache_size(50);
        let mut backend = BackendFactory::create(&config).unwrap();
        assert_eq!(backend.name(), "cached");
        backend.save("k", &serde_json::json!(1)).unwrap();
        assert_eq!(backend.load("k").unwrap(), Some(serde_json::json!(1)));
    }

    #[test]
    fn test_factory_unknown_type() {
        let config = BackendConfig::new("redis");
        let result = BackendFactory::create(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown backend type"));
    }

    #[test]
    fn test_factory_filesystem_missing_path() {
        let config = BackendConfig::new("filesystem");
        let result = BackendFactory::create(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires base_path"));
    }

    #[test]
    fn test_factory_namespaced_missing_namespace() {
        let config = BackendConfig::new("namespaced");
        let result = BackendFactory::create(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires namespace"));
    }

    #[test]
    fn test_factory_cached_default_size() {
        let config = BackendConfig::new("cached");
        let backend = BackendFactory::create(&config).unwrap();
        assert_eq!(backend.name(), "cached");
    }

    // -- BackendStats --

    #[test]
    fn test_stats_from_empty_backend() {
        let backend = InMemoryBackend::new();
        let stats = BackendStats::from_backend(&backend);
        assert_eq!(stats.total_keys, 0);
        assert_eq!(stats.backend_name, "memory");
        assert!(stats.extra.is_empty());
    }

    #[test]
    fn test_stats_from_populated_backend() {
        let mut backend = InMemoryBackend::new();
        backend.save("a", &Value::Null).unwrap();
        backend.save("b", &Value::Null).unwrap();
        backend.save("c", &Value::Null).unwrap();
        let stats = BackendStats::from_backend(&backend);
        assert_eq!(stats.total_keys, 3);
        assert_eq!(stats.backend_name, "memory");
    }

    #[test]
    fn test_stats_to_json() {
        let mut backend = InMemoryBackend::new();
        backend.save("x", &Value::Null).unwrap();
        let stats = BackendStats::from_backend(&backend);
        let json = stats.to_json();
        assert_eq!(json["total_keys"], 1);
        assert_eq!(json["backend_name"], "memory");
    }

    #[test]
    fn test_stats_from_namespaced_backend() {
        let inner = Box::new(InMemoryBackend::new());
        let mut ns = NamespacedBackend::new(inner, "ns".to_string());
        ns.save("a", &Value::Null).unwrap();
        ns.save("b", &Value::Null).unwrap();
        let stats = BackendStats::from_backend(&ns);
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.backend_name, "namespaced");
    }

    // -- Nested backends --

    #[test]
    fn test_nested_cached_namespaced_memory() {
        let memory = Box::new(InMemoryBackend::new());
        let namespaced = Box::new(NamespacedBackend::new(memory, "session1".to_string()));
        let mut cached = CachedBackend::new(namespaced, 100);

        cached
            .save("state", &serde_json::json!({"step": 1}))
            .unwrap();
        let val = cached.load_mut("state").unwrap();
        assert_eq!(val, Some(serde_json::json!({"step": 1})));
        assert_eq!(cached.cache_hits(), 1);
    }

    #[test]
    fn test_nested_multiple_namespaces_isolation() {
        let mut inner = InMemoryBackend::new();
        inner.save("ns_a:key", &serde_json::json!("a_val")).unwrap();
        inner.save("ns_b:key", &serde_json::json!("b_val")).unwrap();

        let ns_a = NamespacedBackend::new(Box::new(inner), "ns_a".to_string());
        assert_eq!(ns_a.load("key").unwrap(), Some(serde_json::json!("a_val")));
        assert_eq!(ns_a.keys(), vec!["key"]);
    }

    #[test]
    fn test_trait_object_usage() {
        let mut backends: Vec<Box<dyn StorageBackend>> = vec![
            Box::new(InMemoryBackend::new()),
            Box::new(NamespacedBackend::new(
                Box::new(InMemoryBackend::new()),
                "ns".to_string(),
            )),
            Box::new(CachedBackend::new(Box::new(InMemoryBackend::new()), 10)),
        ];
        for backend in &mut backends {
            backend.save("test", &serde_json::json!(1)).unwrap();
            assert!(backend.exists("test"));
            assert_eq!(backend.load("test").unwrap(), Some(serde_json::json!(1)));
        }
    }

    #[test]
    fn test_clear_on_empty_backend() {
        let mut backend = InMemoryBackend::new();
        backend.clear().unwrap();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_save_null_value() {
        let mut backend = InMemoryBackend::new();
        backend.save("null_key", &Value::Null).unwrap();
        assert_eq!(backend.load("null_key").unwrap(), Some(Value::Null));
    }

    #[test]
    fn test_save_complex_json() {
        let mut backend = InMemoryBackend::new();
        let complex = serde_json::json!({
            "array": [1, 2, 3],
            "nested": {"a": {"b": true}},
            "string": "hello",
            "number": 3.09,
        });
        backend.save("complex", &complex).unwrap();
        assert_eq!(backend.load("complex").unwrap(), Some(complex));
    }
}
