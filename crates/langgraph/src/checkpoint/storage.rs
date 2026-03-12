//! Storage layer for graph checkpoints.
//!
//! This module provides abstractions over different storage backends for
//! persisting graph state. It includes typed keys, serialized value wrappers,
//! an abstract [`StorageBackend`] trait, and concrete implementations:
//!
//! - [`InMemoryStorage`] — HashMap-backed ephemeral storage
//! - [`LayeredStorage`] — read-through caching over a primary backend
//! - [`StorageMigration`] — migrate data between backends
//! - [`StorageMetrics`] — track storage operation statistics
//! - [`StorageQuota`] — enforce storage limits

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// StorageKey
// ---------------------------------------------------------------------------

/// A typed key for storage operations, consisting of a namespace and key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StorageKey {
    namespace: String,
    key: String,
}

impl StorageKey {
    /// Create a new `StorageKey`.
    pub fn new(namespace: &str, key: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            key: key.to_string(),
        }
    }

    /// Return the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Return the key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Return the full path as `"namespace/key"`.
    pub fn full_path(&self) -> String {
        format!("{}/{}", self.namespace, self.key)
    }
}

impl fmt::Display for StorageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.namespace, self.key)
    }
}

// ---------------------------------------------------------------------------
// StorageValue
// ---------------------------------------------------------------------------

/// A wrapper around serialized data that can hold either JSON or raw bytes.
#[derive(Debug, Clone)]
pub struct StorageValue {
    json: Option<Value>,
    bytes: Vec<u8>,
}

impl StorageValue {
    /// Create a `StorageValue` from a JSON value.
    pub fn from_json(value: Value) -> Self {
        let bytes = serde_json::to_vec(&value).unwrap_or_default();
        Self {
            json: Some(value),
            bytes,
        }
    }

    /// Create a `StorageValue` from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let json = serde_json::from_slice(&data).ok();
        Self { json, bytes: data }
    }

    /// Return a reference to the JSON value, if available.
    pub fn as_json(&self) -> Option<&Value> {
        self.json.as_ref()
    }

    /// Return the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Attempt to parse or return the JSON value, consuming self.
    pub fn to_json(self) -> Option<Value> {
        self.json
    }
}

// ---------------------------------------------------------------------------
// StorageBackend trait
// ---------------------------------------------------------------------------

/// Abstract storage operations for checkpoint persistence.
pub trait StorageBackend {
    /// Retrieve a value by key.
    fn get(&self, key: &StorageKey) -> Option<StorageValue>;

    /// Store a value under the given key.
    fn put(&mut self, key: StorageKey, value: StorageValue);

    /// Delete a value by key. Returns `true` if the key existed.
    fn delete(&mut self, key: &StorageKey) -> bool;

    /// Check whether a key exists.
    fn exists(&self, key: &StorageKey) -> bool;

    /// List all keys in a namespace.
    fn list_keys(&self, namespace: &str) -> Vec<StorageKey>;

    /// Remove all keys in a namespace. Returns the count removed.
    fn clear_namespace(&mut self, namespace: &str) -> usize;
}

// ---------------------------------------------------------------------------
// InMemoryStorage
// ---------------------------------------------------------------------------

/// A HashMap-backed in-memory storage backend.
#[derive(Debug)]
pub struct InMemoryStorage {
    data: HashMap<String, (StorageKey, StorageValue)>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStorage {
    /// Create an empty in-memory storage.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Return the total size in bytes of all stored values.
    pub fn total_size_bytes(&self) -> usize {
        self.data.values().map(|(_, v)| v.size_bytes()).sum()
    }

    /// Return the number of stored keys.
    pub fn key_count(&self) -> usize {
        self.data.len()
    }

    /// Return a sorted list of unique namespaces.
    pub fn namespaces(&self) -> Vec<String> {
        let mut ns: Vec<String> = self
            .data
            .values()
            .map(|(k, _)| k.namespace().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ns.sort();
        ns
    }
}

impl StorageBackend for InMemoryStorage {
    fn get(&self, key: &StorageKey) -> Option<StorageValue> {
        self.data.get(&key.full_path()).map(|(_, v)| v.clone())
    }

    fn put(&mut self, key: StorageKey, value: StorageValue) {
        let path = key.full_path();
        self.data.insert(path, (key, value));
    }

    fn delete(&mut self, key: &StorageKey) -> bool {
        self.data.remove(&key.full_path()).is_some()
    }

    fn exists(&self, key: &StorageKey) -> bool {
        self.data.contains_key(&key.full_path())
    }

    fn list_keys(&self, namespace: &str) -> Vec<StorageKey> {
        let mut keys: Vec<StorageKey> = self
            .data
            .values()
            .filter(|(k, _)| k.namespace() == namespace)
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort_by(|a, b| a.key().cmp(b.key()));
        keys
    }

    fn clear_namespace(&mut self, namespace: &str) -> usize {
        let to_remove: Vec<String> = self
            .data
            .iter()
            .filter(|(_, (k, _))| k.namespace() == namespace)
            .map(|(path, _)| path.clone())
            .collect();
        let count = to_remove.len();
        for path in to_remove {
            self.data.remove(&path);
        }
        count
    }
}

// ---------------------------------------------------------------------------
// LayeredStorage
// ---------------------------------------------------------------------------

/// A storage backend with read-through caching.
///
/// Reads check the cache first, then fall through to the primary backend.
/// Writes go to both the primary and cache.
pub struct LayeredStorage {
    primary: RefCell<Box<dyn StorageBackend>>,
    cache: RefCell<Box<dyn StorageBackend>>,
    gets: Cell<u64>,
    cache_hits: Cell<u64>,
}

impl LayeredStorage {
    /// Create a new layered storage with the given primary and cache backends.
    pub fn new(primary: Box<dyn StorageBackend>, cache: Box<dyn StorageBackend>) -> Self {
        Self {
            primary: RefCell::new(primary),
            cache: RefCell::new(cache),
            gets: Cell::new(0),
            cache_hits: Cell::new(0),
        }
    }

    /// Return the cache hit rate as a value between 0.0 and 1.0.
    ///
    /// Returns 0.0 if no gets have been performed.
    pub fn cache_hit_rate(&self) -> f64 {
        let gets = self.gets.get();
        if gets == 0 {
            0.0
        } else {
            self.cache_hits.get() as f64 / gets as f64
        }
    }

    /// Invalidate all entries in the cache, resetting hit rate statistics.
    pub fn invalidate_cache(&mut self) {
        *self.cache.borrow_mut() = Box::new(InMemoryStorage::new());
        self.gets.set(0);
        self.cache_hits.set(0);
    }
}

impl StorageBackend for LayeredStorage {
    fn get(&self, key: &StorageKey) -> Option<StorageValue> {
        self.gets.set(self.gets.get() + 1);

        // Check cache first.
        if let Some(value) = self.cache.borrow().get(key) {
            self.cache_hits.set(self.cache_hits.get() + 1);
            return Some(value);
        }

        // Fall through to primary, populate cache on hit.
        if let Some(value) = self.primary.borrow().get(key) {
            self.cache.borrow_mut().put(key.clone(), value.clone());
            return Some(value);
        }

        None
    }

    fn put(&mut self, key: StorageKey, value: StorageValue) {
        self.primary.borrow_mut().put(key.clone(), value.clone());
        self.cache.borrow_mut().put(key, value);
    }

    fn delete(&mut self, key: &StorageKey) -> bool {
        self.cache.borrow_mut().delete(key);
        self.primary.borrow_mut().delete(key)
    }

    fn exists(&self, key: &StorageKey) -> bool {
        self.cache.borrow().exists(key) || self.primary.borrow().exists(key)
    }

    fn list_keys(&self, namespace: &str) -> Vec<StorageKey> {
        self.primary.borrow().list_keys(namespace)
    }

    fn clear_namespace(&mut self, namespace: &str) -> usize {
        self.cache.borrow_mut().clear_namespace(namespace);
        self.primary.borrow_mut().clear_namespace(namespace)
    }
}

// ---------------------------------------------------------------------------
// StorageMigration
// ---------------------------------------------------------------------------

/// Migrates data between two storage backends.
pub struct StorageMigration {
    source: Box<dyn StorageBackend>,
    destination: Box<dyn StorageBackend>,
}

impl StorageMigration {
    /// Create a new migration from source to destination.
    pub fn new(source: Box<dyn StorageBackend>, destination: Box<dyn StorageBackend>) -> Self {
        Self {
            source,
            destination,
        }
    }

    /// Migrate all keys in a namespace from source to destination.
    ///
    /// Returns the number of keys migrated.
    pub fn migrate_namespace(&mut self, namespace: &str) -> usize {
        let keys = self.source.list_keys(namespace);
        let count = keys.len();
        for key in keys {
            if let Some(value) = self.source.get(&key) {
                self.destination.put(key, value);
            }
        }
        count
    }

    /// Migrate all keys across all known namespaces.
    ///
    /// Since [`StorageBackend`] doesn't expose a list-all-namespaces method,
    /// this requires the source to be iterable. This implementation collects
    /// namespaces from the source's listed keys.
    ///
    /// **Note:** This method accepts a slice of namespace names to migrate.
    pub fn migrate_all(&mut self, namespaces: &[&str]) -> usize {
        let mut total = 0;
        for ns in namespaces {
            total += self.migrate_namespace(ns);
        }
        total
    }

    /// Verify that all keys in a namespace exist in both source and destination
    /// with matching data.
    pub fn verify(&self, namespace: &str) -> bool {
        let source_keys = self.source.list_keys(namespace);
        let dest_keys = self.destination.list_keys(namespace);

        if source_keys.len() != dest_keys.len() {
            return false;
        }

        for key in &source_keys {
            let source_val = self.source.get(key);
            let dest_val = self.destination.get(key);

            match (source_val, dest_val) {
                (Some(s), Some(d)) => {
                    if s.as_bytes() != d.as_bytes() {
                        return false;
                    }
                }
                (None, None) => {}
                _ => return false,
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// StorageMetrics
// ---------------------------------------------------------------------------

/// Tracks storage operation statistics.
#[derive(Debug, Clone)]
pub struct StorageMetrics {
    gets: u64,
    hits: u64,
    puts: u64,
    deletes: u64,
    bytes_written: usize,
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageMetrics {
    /// Create a new, zeroed metrics tracker.
    pub fn new() -> Self {
        Self {
            gets: 0,
            hits: 0,
            puts: 0,
            deletes: 0,
            bytes_written: 0,
        }
    }

    /// Record a get operation.
    pub fn record_get(&mut self, hit: bool) {
        self.gets += 1;
        if hit {
            self.hits += 1;
        }
    }

    /// Record a put operation.
    pub fn record_put(&mut self, size_bytes: usize) {
        self.puts += 1;
        self.bytes_written += size_bytes;
    }

    /// Record a delete operation.
    pub fn record_delete(&mut self) {
        self.deletes += 1;
    }

    /// Total number of get operations.
    pub fn total_gets(&self) -> u64 {
        self.gets
    }

    /// Total number of put operations.
    pub fn total_puts(&self) -> u64 {
        self.puts
    }

    /// Total number of delete operations.
    pub fn total_deletes(&self) -> u64 {
        self.deletes
    }

    /// Cache hit rate (0.0 to 1.0). Returns 0.0 if no gets recorded.
    pub fn hit_rate(&self) -> f64 {
        if self.gets == 0 {
            0.0
        } else {
            self.hits as f64 / self.gets as f64
        }
    }

    /// Total bytes written via put operations.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    /// Serialize metrics to a JSON object.
    pub fn to_json(&self) -> Value {
        json!({
            "total_gets": self.gets,
            "total_puts": self.puts,
            "total_deletes": self.deletes,
            "hit_rate": self.hit_rate(),
            "bytes_written": self.bytes_written,
        })
    }
}

// ---------------------------------------------------------------------------
// QuotaError
// ---------------------------------------------------------------------------

/// Errors returned when a storage quota is exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaError {
    /// The maximum number of keys has been reached.
    KeyLimitExceeded {
        /// Current number of keys.
        current: usize,
        /// Maximum allowed keys.
        max: usize,
    },
    /// The maximum byte limit would be exceeded.
    ByteLimitExceeded {
        /// Current bytes used.
        current: usize,
        /// Maximum allowed bytes.
        max: usize,
        /// Bytes requested to write.
        requested: usize,
    },
}

impl fmt::Display for QuotaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuotaError::KeyLimitExceeded { current, max } => {
                write!(f, "key limit exceeded: current {current}, max {max}")
            }
            QuotaError::ByteLimitExceeded {
                current,
                max,
                requested,
            } => {
                write!(
                    f,
                    "byte limit exceeded: current {current}, max {max}, requested {requested}"
                )
            }
        }
    }
}

impl std::error::Error for QuotaError {}

// ---------------------------------------------------------------------------
// StorageQuota
// ---------------------------------------------------------------------------

/// Enforces storage limits on keys and bytes.
#[derive(Debug, Clone)]
pub struct StorageQuota {
    max_keys: usize,
    max_bytes: usize,
}

impl StorageQuota {
    /// Create a new quota with the given limits.
    pub fn new(max_keys: usize, max_bytes: usize) -> Self {
        Self {
            max_keys,
            max_bytes,
        }
    }

    /// Check whether a put operation is allowed given current usage.
    pub fn check_put(
        &self,
        current_keys: usize,
        current_bytes: usize,
        new_bytes: usize,
    ) -> Result<(), QuotaError> {
        if current_keys >= self.max_keys {
            return Err(QuotaError::KeyLimitExceeded {
                current: current_keys,
                max: self.max_keys,
            });
        }
        if current_bytes + new_bytes > self.max_bytes {
            return Err(QuotaError::ByteLimitExceeded {
                current: current_bytes,
                max: self.max_bytes,
                requested: new_bytes,
            });
        }
        Ok(())
    }

    /// Serialize the quota configuration to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "max_keys": self.max_keys,
            "max_bytes": self.max_bytes,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------------------------------------------------------------
    // StorageKey tests
    // ---------------------------------------------------------------

    #[test]
    fn test_storage_key_new() {
        let key = StorageKey::new("ns", "k");
        assert_eq!(key.namespace(), "ns");
        assert_eq!(key.key(), "k");
    }

    #[test]
    fn test_storage_key_full_path() {
        let key = StorageKey::new("checkpoints", "abc123");
        assert_eq!(key.full_path(), "checkpoints/abc123");
    }

    #[test]
    fn test_storage_key_display() {
        let key = StorageKey::new("ns", "k");
        assert_eq!(format!("{}", key), "ns/k");
    }

    #[test]
    fn test_storage_key_clone() {
        let key = StorageKey::new("ns", "k");
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_storage_key_equality() {
        let a = StorageKey::new("ns", "k");
        let b = StorageKey::new("ns", "k");
        assert_eq!(a, b);
    }

    #[test]
    fn test_storage_key_inequality() {
        let a = StorageKey::new("ns", "k1");
        let b = StorageKey::new("ns", "k2");
        assert_ne!(a, b);
    }

    #[test]
    fn test_storage_key_inequality_namespace() {
        let a = StorageKey::new("ns1", "k");
        let b = StorageKey::new("ns2", "k");
        assert_ne!(a, b);
    }

    #[test]
    fn test_storage_key_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StorageKey::new("ns", "k1"));
        set.insert(StorageKey::new("ns", "k1"));
        set.insert(StorageKey::new("ns", "k2"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_storage_key_empty_parts() {
        let key = StorageKey::new("", "");
        assert_eq!(key.full_path(), "/");
    }

    // ---------------------------------------------------------------
    // StorageValue tests
    // ---------------------------------------------------------------

    #[test]
    fn test_storage_value_from_json() {
        let val = StorageValue::from_json(json!({"hello": "world"}));
        assert_eq!(val.as_json().unwrap(), &json!({"hello": "world"}));
        assert!(!val.as_bytes().is_empty());
    }

    #[test]
    fn test_storage_value_from_bytes() {
        let data = b"hello world".to_vec();
        let val = StorageValue::from_bytes(data.clone());
        assert_eq!(val.as_bytes(), &data);
    }

    #[test]
    fn test_storage_value_from_bytes_valid_json() {
        let data = serde_json::to_vec(&json!(42)).unwrap();
        let val = StorageValue::from_bytes(data);
        assert_eq!(val.as_json().unwrap(), &json!(42));
    }

    #[test]
    fn test_storage_value_from_bytes_invalid_json() {
        let val = StorageValue::from_bytes(b"not json".to_vec());
        assert!(val.as_json().is_none());
    }

    #[test]
    fn test_storage_value_size_bytes() {
        let val = StorageValue::from_json(json!("test"));
        assert!(val.size_bytes() > 0);
    }

    #[test]
    fn test_storage_value_to_json() {
        let val = StorageValue::from_json(json!({"x": 1}));
        assert_eq!(val.to_json().unwrap(), json!({"x": 1}));
    }

    #[test]
    fn test_storage_value_to_json_from_bytes_non_json() {
        let val = StorageValue::from_bytes(b"raw".to_vec());
        assert!(val.to_json().is_none());
    }

    #[test]
    fn test_storage_value_clone() {
        let val = StorageValue::from_json(json!(123));
        let cloned = val.clone();
        assert_eq!(cloned.as_json().unwrap(), &json!(123));
    }

    // ---------------------------------------------------------------
    // InMemoryStorage tests
    // ---------------------------------------------------------------

    #[test]
    fn test_in_memory_storage_new() {
        let store = InMemoryStorage::new();
        assert_eq!(store.key_count(), 0);
        assert_eq!(store.total_size_bytes(), 0);
    }

    #[test]
    fn test_in_memory_storage_default() {
        let store = InMemoryStorage::default();
        assert_eq!(store.key_count(), 0);
    }

    #[test]
    fn test_in_memory_storage_put_and_get() {
        let mut store = InMemoryStorage::new();
        let key = StorageKey::new("ns", "k1");
        let val = StorageValue::from_json(json!(42));

        store.put(key.clone(), val);
        let retrieved = store.get(&key).unwrap();
        assert_eq!(retrieved.as_json().unwrap(), &json!(42));
    }

    #[test]
    fn test_in_memory_storage_get_missing() {
        let store = InMemoryStorage::new();
        assert!(store.get(&StorageKey::new("ns", "nope")).is_none());
    }

    #[test]
    fn test_in_memory_storage_put_overwrites() {
        let mut store = InMemoryStorage::new();
        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(1)));
        store.put(key.clone(), StorageValue::from_json(json!(2)));

        assert_eq!(store.key_count(), 1);
        assert_eq!(store.get(&key).unwrap().as_json().unwrap(), &json!(2));
    }

    #[test]
    fn test_in_memory_storage_delete_existing() {
        let mut store = InMemoryStorage::new();
        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(1)));

        assert!(store.delete(&key));
        assert_eq!(store.key_count(), 0);
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn test_in_memory_storage_delete_missing() {
        let mut store = InMemoryStorage::new();
        assert!(!store.delete(&StorageKey::new("ns", "nope")));
    }

    #[test]
    fn test_in_memory_storage_exists() {
        let mut store = InMemoryStorage::new();
        let key = StorageKey::new("ns", "k1");
        assert!(!store.exists(&key));

        store.put(key.clone(), StorageValue::from_json(json!(1)));
        assert!(store.exists(&key));
    }

    #[test]
    fn test_in_memory_storage_list_keys() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("alpha", "a1"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("alpha", "a2"),
            StorageValue::from_json(json!(2)),
        );
        store.put(
            StorageKey::new("beta", "b1"),
            StorageValue::from_json(json!(3)),
        );

        let alpha_keys = store.list_keys("alpha");
        assert_eq!(alpha_keys.len(), 2);
        assert!(alpha_keys.iter().all(|k| k.namespace() == "alpha"));

        let beta_keys = store.list_keys("beta");
        assert_eq!(beta_keys.len(), 1);
    }

    #[test]
    fn test_in_memory_storage_list_keys_sorted() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("ns", "c"),
            StorageValue::from_json(json!(3)),
        );
        store.put(
            StorageKey::new("ns", "a"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("ns", "b"),
            StorageValue::from_json(json!(2)),
        );

        let keys = store.list_keys("ns");
        let key_names: Vec<&str> = keys.iter().map(|k| k.key()).collect();
        assert_eq!(key_names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_in_memory_storage_list_keys_empty_namespace() {
        let store = InMemoryStorage::new();
        assert!(store.list_keys("nope").is_empty());
    }

    #[test]
    fn test_in_memory_storage_clear_namespace() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("alpha", "a1"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("alpha", "a2"),
            StorageValue::from_json(json!(2)),
        );
        store.put(
            StorageKey::new("beta", "b1"),
            StorageValue::from_json(json!(3)),
        );

        let removed = store.clear_namespace("alpha");
        assert_eq!(removed, 2);
        assert_eq!(store.key_count(), 1);
        assert!(store.list_keys("alpha").is_empty());
        assert_eq!(store.list_keys("beta").len(), 1);
    }

    #[test]
    fn test_in_memory_storage_clear_empty_namespace() {
        let mut store = InMemoryStorage::new();
        assert_eq!(store.clear_namespace("nope"), 0);
    }

    #[test]
    fn test_in_memory_storage_total_size_bytes() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("ns", "k"),
            StorageValue::from_json(json!({"data": "value"})),
        );
        assert!(store.total_size_bytes() > 0);
    }

    #[test]
    fn test_in_memory_storage_namespaces() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("beta", "b1"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("alpha", "a1"),
            StorageValue::from_json(json!(2)),
        );

        let ns = store.namespaces();
        assert_eq!(ns, vec!["alpha", "beta"]); // sorted
    }

    #[test]
    fn test_in_memory_storage_namespaces_empty() {
        let store = InMemoryStorage::new();
        assert!(store.namespaces().is_empty());
    }

    // ---------------------------------------------------------------
    // LayeredStorage tests
    // ---------------------------------------------------------------

    #[test]
    fn test_layered_storage_put_and_get() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(42)));

        let val = store.get(&key).unwrap();
        assert_eq!(val.as_json().unwrap(), &json!(42));
    }

    #[test]
    fn test_layered_storage_cache_hit() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(1)));

        // First get populates cache, second should hit cache.
        let _ = store.get(&key);
        let _ = store.get(&key);

        // Both gets should be cache hits since put writes to both.
        assert_eq!(store.gets.get(), 2);
        assert_eq!(store.cache_hits.get(), 2);
        assert_eq!(store.cache_hit_rate(), 1.0);
    }

    #[test]
    fn test_layered_storage_cache_miss_then_hit() {
        let mut primary = InMemoryStorage::new();
        let key = StorageKey::new("ns", "k1");
        primary.put(key.clone(), StorageValue::from_json(json!(99)));

        let cache = Box::new(InMemoryStorage::new());
        let store = LayeredStorage::new(Box::new(primary), cache);

        // First get: cache miss, falls through to primary.
        let val = store.get(&key).unwrap();
        assert_eq!(val.as_json().unwrap(), &json!(99));

        // Second get: cache hit.
        let val = store.get(&key).unwrap();
        assert_eq!(val.as_json().unwrap(), &json!(99));

        assert_eq!(store.gets.get(), 2);
        assert_eq!(store.cache_hits.get(), 1);
        assert_eq!(store.cache_hit_rate(), 0.5);
    }

    #[test]
    fn test_layered_storage_cache_hit_rate_no_gets() {
        let store = LayeredStorage::new(
            Box::new(InMemoryStorage::new()),
            Box::new(InMemoryStorage::new()),
        );
        assert_eq!(store.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_layered_storage_delete() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(1)));
        assert!(store.delete(&key));
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn test_layered_storage_exists() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        let key = StorageKey::new("ns", "k1");
        assert!(!store.exists(&key));

        store.put(key.clone(), StorageValue::from_json(json!(1)));
        assert!(store.exists(&key));
    }

    #[test]
    fn test_layered_storage_list_keys() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        store.put(
            StorageKey::new("ns", "a"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("ns", "b"),
            StorageValue::from_json(json!(2)),
        );

        let keys = store.list_keys("ns");
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_layered_storage_clear_namespace() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        store.put(
            StorageKey::new("ns", "a"),
            StorageValue::from_json(json!(1)),
        );
        store.put(
            StorageKey::new("ns", "b"),
            StorageValue::from_json(json!(2)),
        );

        let removed = store.clear_namespace("ns");
        assert_eq!(removed, 2);
        assert!(store.list_keys("ns").is_empty());
    }

    #[test]
    fn test_layered_storage_invalidate_cache() {
        let primary = Box::new(InMemoryStorage::new());
        let cache = Box::new(InMemoryStorage::new());
        let mut store = LayeredStorage::new(primary, cache);

        let key = StorageKey::new("ns", "k1");
        store.put(key.clone(), StorageValue::from_json(json!(1)));

        // Perform a get to bump stats.
        let _ = store.get(&key);
        assert!(store.gets.get() > 0);

        store.invalidate_cache();
        assert_eq!(store.gets.get(), 0);
        assert_eq!(store.cache_hits.get(), 0);
    }

    // ---------------------------------------------------------------
    // StorageMigration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_storage_migration_migrate_namespace() {
        let mut source = InMemoryStorage::new();
        source.put(
            StorageKey::new("ns", "k1"),
            StorageValue::from_json(json!(1)),
        );
        source.put(
            StorageKey::new("ns", "k2"),
            StorageValue::from_json(json!(2)),
        );
        source.put(
            StorageKey::new("other", "k3"),
            StorageValue::from_json(json!(3)),
        );

        let dest = InMemoryStorage::new();
        let mut migration = StorageMigration::new(Box::new(source), Box::new(dest));

        let count = migration.migrate_namespace("ns");
        assert_eq!(count, 2);

        // Verify destination has the keys.
        assert!(migration.destination.exists(&StorageKey::new("ns", "k1")));
        assert!(migration.destination.exists(&StorageKey::new("ns", "k2")));
        assert!(!migration
            .destination
            .exists(&StorageKey::new("other", "k3")));
    }

    #[test]
    fn test_storage_migration_migrate_all() {
        let mut source = InMemoryStorage::new();
        source.put(
            StorageKey::new("ns1", "k1"),
            StorageValue::from_json(json!(1)),
        );
        source.put(
            StorageKey::new("ns2", "k2"),
            StorageValue::from_json(json!(2)),
        );

        let dest = InMemoryStorage::new();
        let mut migration = StorageMigration::new(Box::new(source), Box::new(dest));

        let count = migration.migrate_all(&["ns1", "ns2"]);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_storage_migration_migrate_empty_namespace() {
        let source = InMemoryStorage::new();
        let dest = InMemoryStorage::new();
        let mut migration = StorageMigration::new(Box::new(source), Box::new(dest));

        let count = migration.migrate_namespace("empty");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_storage_migration_verify_success() {
        let mut source = InMemoryStorage::new();
        source.put(
            StorageKey::new("ns", "k1"),
            StorageValue::from_json(json!(1)),
        );
        source.put(
            StorageKey::new("ns", "k2"),
            StorageValue::from_json(json!(2)),
        );

        let dest = InMemoryStorage::new();
        let mut migration = StorageMigration::new(Box::new(source), Box::new(dest));

        migration.migrate_namespace("ns");
        assert!(migration.verify("ns"));
    }

    #[test]
    fn test_storage_migration_verify_failure_missing_key() {
        let mut source = InMemoryStorage::new();
        source.put(
            StorageKey::new("ns", "k1"),
            StorageValue::from_json(json!(1)),
        );

        let dest = InMemoryStorage::new();
        let migration = StorageMigration::new(Box::new(source), Box::new(dest));

        // Not migrated, so verify should fail.
        assert!(!migration.verify("ns"));
    }

    #[test]
    fn test_storage_migration_verify_failure_different_data() {
        let mut source = InMemoryStorage::new();
        source.put(
            StorageKey::new("ns", "k1"),
            StorageValue::from_json(json!(1)),
        );

        let mut dest = InMemoryStorage::new();
        dest.put(
            StorageKey::new("ns", "k1"),
            StorageValue::from_json(json!(999)),
        );

        let migration = StorageMigration::new(Box::new(source), Box::new(dest));
        assert!(!migration.verify("ns"));
    }

    #[test]
    fn test_storage_migration_verify_empty_namespace() {
        let source = InMemoryStorage::new();
        let dest = InMemoryStorage::new();
        let migration = StorageMigration::new(Box::new(source), Box::new(dest));

        assert!(migration.verify("empty"));
    }

    // ---------------------------------------------------------------
    // StorageMetrics tests
    // ---------------------------------------------------------------

    #[test]
    fn test_storage_metrics_new() {
        let m = StorageMetrics::new();
        assert_eq!(m.total_gets(), 0);
        assert_eq!(m.total_puts(), 0);
        assert_eq!(m.total_deletes(), 0);
        assert_eq!(m.bytes_written(), 0);
        assert_eq!(m.hit_rate(), 0.0);
    }

    #[test]
    fn test_storage_metrics_default() {
        let m = StorageMetrics::default();
        assert_eq!(m.total_gets(), 0);
    }

    #[test]
    fn test_storage_metrics_record_get_hit() {
        let mut m = StorageMetrics::new();
        m.record_get(true);
        assert_eq!(m.total_gets(), 1);
        assert_eq!(m.hit_rate(), 1.0);
    }

    #[test]
    fn test_storage_metrics_record_get_miss() {
        let mut m = StorageMetrics::new();
        m.record_get(false);
        assert_eq!(m.total_gets(), 1);
        assert_eq!(m.hit_rate(), 0.0);
    }

    #[test]
    fn test_storage_metrics_record_put() {
        let mut m = StorageMetrics::new();
        m.record_put(100);
        m.record_put(200);
        assert_eq!(m.total_puts(), 2);
        assert_eq!(m.bytes_written(), 300);
    }

    #[test]
    fn test_storage_metrics_record_delete() {
        let mut m = StorageMetrics::new();
        m.record_delete();
        m.record_delete();
        assert_eq!(m.total_deletes(), 2);
    }

    #[test]
    fn test_storage_metrics_hit_rate_mixed() {
        let mut m = StorageMetrics::new();
        m.record_get(true);
        m.record_get(true);
        m.record_get(false);
        m.record_get(false);
        assert!((m.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_storage_metrics_to_json() {
        let mut m = StorageMetrics::new();
        m.record_get(true);
        m.record_put(50);
        m.record_delete();

        let j = m.to_json();
        assert_eq!(j["total_gets"], json!(1));
        assert_eq!(j["total_puts"], json!(1));
        assert_eq!(j["total_deletes"], json!(1));
        assert_eq!(j["bytes_written"], json!(50));
        assert_eq!(j["hit_rate"], json!(1.0));
    }

    // ---------------------------------------------------------------
    // QuotaError tests
    // ---------------------------------------------------------------

    #[test]
    fn test_quota_error_key_limit_display() {
        let err = QuotaError::KeyLimitExceeded {
            current: 100,
            max: 100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("key limit exceeded"));
        assert!(msg.contains("100"));
    }

    #[test]
    fn test_quota_error_byte_limit_display() {
        let err = QuotaError::ByteLimitExceeded {
            current: 900,
            max: 1000,
            requested: 200,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("byte limit exceeded"));
        assert!(msg.contains("900"));
        assert!(msg.contains("1000"));
        assert!(msg.contains("200"));
    }

    #[test]
    fn test_quota_error_is_error() {
        let err = QuotaError::KeyLimitExceeded { current: 1, max: 1 };
        // Verify it implements std::error::Error by calling .to_string().
        let _ = err.to_string();
    }

    #[test]
    fn test_quota_error_equality() {
        let a = QuotaError::KeyLimitExceeded { current: 5, max: 5 };
        let b = QuotaError::KeyLimitExceeded { current: 5, max: 5 };
        assert_eq!(a, b);
    }

    // ---------------------------------------------------------------
    // StorageQuota tests
    // ---------------------------------------------------------------

    #[test]
    fn test_storage_quota_check_put_ok() {
        let quota = StorageQuota::new(100, 10_000);
        assert!(quota.check_put(10, 500, 100).is_ok());
    }

    #[test]
    fn test_storage_quota_check_put_key_limit() {
        let quota = StorageQuota::new(10, 10_000);
        let result = quota.check_put(10, 500, 100);
        assert!(result.is_err());
        match result.unwrap_err() {
            QuotaError::KeyLimitExceeded { current, max } => {
                assert_eq!(current, 10);
                assert_eq!(max, 10);
            }
            _ => panic!("expected KeyLimitExceeded"),
        }
    }

    #[test]
    fn test_storage_quota_check_put_byte_limit() {
        let quota = StorageQuota::new(100, 1000);
        let result = quota.check_put(5, 900, 200);
        assert!(result.is_err());
        match result.unwrap_err() {
            QuotaError::ByteLimitExceeded {
                current,
                max,
                requested,
            } => {
                assert_eq!(current, 900);
                assert_eq!(max, 1000);
                assert_eq!(requested, 200);
            }
            _ => panic!("expected ByteLimitExceeded"),
        }
    }

    #[test]
    fn test_storage_quota_check_put_exact_byte_limit() {
        let quota = StorageQuota::new(100, 1000);
        // Exactly at the limit should be OK.
        assert!(quota.check_put(5, 900, 100).is_ok());
    }

    #[test]
    fn test_storage_quota_check_put_one_over_byte_limit() {
        let quota = StorageQuota::new(100, 1000);
        assert!(quota.check_put(5, 900, 101).is_err());
    }

    #[test]
    fn test_storage_quota_to_json() {
        let quota = StorageQuota::new(50, 5000);
        let j = quota.to_json();
        assert_eq!(j["max_keys"], json!(50));
        assert_eq!(j["max_bytes"], json!(5000));
    }

    // ---------------------------------------------------------------
    // Integration / edge-case tests
    // ---------------------------------------------------------------

    #[test]
    fn test_put_get_delete_cycle() {
        let mut store = InMemoryStorage::new();
        let key = StorageKey::new("test", "cycle");

        store.put(key.clone(), StorageValue::from_json(json!("hello")));
        assert!(store.exists(&key));

        let val = store.get(&key).unwrap();
        assert_eq!(val.as_json().unwrap(), &json!("hello"));

        assert!(store.delete(&key));
        assert!(!store.exists(&key));
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn test_multiple_namespaces() {
        let mut store = InMemoryStorage::new();

        for i in 0..5 {
            let ns = format!("ns{}", i);
            for j in 0..3 {
                let key = StorageKey::new(&ns, &format!("k{}", j));
                store.put(key, StorageValue::from_json(json!(i * 10 + j)));
            }
        }

        assert_eq!(store.key_count(), 15);
        assert_eq!(store.namespaces().len(), 5);

        for i in 0..5 {
            let ns = format!("ns{}", i);
            assert_eq!(store.list_keys(&ns).len(), 3);
        }
    }

    #[test]
    fn test_layered_storage_primary_only_data() {
        // Data only in primary, not in cache.
        let mut primary = InMemoryStorage::new();
        let key = StorageKey::new("ns", "primary_only");
        primary.put(key.clone(), StorageValue::from_json(json!("from_primary")));

        let cache = InMemoryStorage::new();
        let store = LayeredStorage::new(Box::new(primary), Box::new(cache));

        let val = store.get(&key).unwrap();
        assert_eq!(val.as_json().unwrap(), &json!("from_primary"));
    }

    #[test]
    fn test_storage_value_large_data() {
        let large = "x".repeat(100_000);
        let val = StorageValue::from_json(json!(large));
        assert!(val.size_bytes() > 100_000);
    }

    #[test]
    fn test_clear_namespace_returns_zero_for_unknown() {
        let mut store = InMemoryStorage::new();
        store.put(
            StorageKey::new("known", "k"),
            StorageValue::from_json(json!(1)),
        );
        assert_eq!(store.clear_namespace("unknown"), 0);
        assert_eq!(store.key_count(), 1);
    }
}
