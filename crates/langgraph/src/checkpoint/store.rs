//! Key-value store abstraction for LangGraph checkpoint persistence.
//!
//! This module provides a common [`Store`] trait and concrete implementations
//! for different storage backends. The store is designed for associating
//! arbitrary key-value data with graph executions — for example, caching
//! intermediate results, storing user preferences, or persisting shared state
//! across graph nodes.
//!
//! # Provided implementations
//!
//! - [`InMemoryStore`] — HashMap-backed ephemeral store for testing
//! - [`NamespacedStore`] — wraps any `Store` with a fixed namespace prefix
//! - [`BatchStore`] — executes multiple [`StoreOperation`]s atomically

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::errors::Result;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a `SystemTime` as an ISO-8601-ish string for storage.
fn format_timestamp(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let millis = d.subsec_millis();
    format!("{}.{:03}", secs, millis)
}

/// Parse a timestamp string produced by [`format_timestamp`] back to a
/// `SystemTime`.
fn parse_timestamp(s: &str) -> Option<SystemTime> {
    let mut parts = s.splitn(2, '.');
    let secs: u64 = parts.next()?.parse().ok()?;
    let millis: u64 = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    Some(UNIX_EPOCH + Duration::from_secs(secs) + Duration::from_millis(millis))
}

// ---------------------------------------------------------------------------
// StoreKey
// ---------------------------------------------------------------------------

/// A namespaced key used to address values in a [`Store`].
///
/// Keys are logically partitioned by namespace so that independent subsystems
/// can share a single store without collisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreKey {
    /// The namespace this key belongs to.
    pub namespace: String,
    /// The key within the namespace.
    pub key: String,
}

impl StoreKey {
    /// Create a new `StoreKey`.
    pub fn new(namespace: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
        }
    }

    /// Build a `StoreKey` by joining parts with `":"`.
    ///
    /// The first element becomes the namespace and the remaining elements are
    /// joined with `":"` to form the key. Panics if `parts` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use langgraph::checkpoint::store::StoreKey;
    ///
    /// let key = StoreKey::from_parts(&["users", "123", "prefs"]);
    /// assert_eq!(key.namespace, "users");
    /// assert_eq!(key.key, "123:prefs");
    /// ```
    pub fn from_parts(parts: &[&str]) -> Self {
        assert!(!parts.is_empty(), "parts must not be empty");
        if parts.len() == 1 {
            Self {
                namespace: parts[0].to_string(),
                key: String::new(),
            }
        } else {
            Self {
                namespace: parts[0].to_string(),
                key: parts[1..].join(":"),
            }
        }
    }

    /// Return the fully-qualified key as `"namespace:key"`.
    pub fn full_key(&self) -> String {
        if self.key.is_empty() {
            self.namespace.clone()
        } else {
            format!("{}:{}", self.namespace, self.key)
        }
    }

    /// Check whether this key belongs to the given namespace.
    pub fn matches_namespace(&self, ns: &str) -> bool {
        self.namespace == ns
    }
}

// ---------------------------------------------------------------------------
// StoreValue
// ---------------------------------------------------------------------------

/// A timestamped, versioned value stored in a [`Store`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreValue {
    /// The stored data.
    pub data: Value,
    /// Timestamp string of when the value was first created.
    pub created_at: String,
    /// Timestamp string of the most recent update.
    pub updated_at: String,
    /// Monotonically increasing version counter (starts at 1).
    pub version: u64,
    /// Optional time-to-live. If set, the value is considered expired once
    /// `created_at + ttl` has elapsed.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_duration_serde"
    )]
    pub ttl: Option<Duration>,
}

mod optional_duration_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(
        duration: &Option<Duration>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_u64(d.as_secs()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<u64> = Option::deserialize(deserializer)?;
        Ok(opt.map(Duration::from_secs))
    }
}

impl StoreValue {
    /// Create a new `StoreValue` with automatic timestamps and version 1.
    pub fn new(data: Value) -> Self {
        let now = format_timestamp(SystemTime::now());
        Self {
            data,
            created_at: now.clone(),
            updated_at: now,
            version: 1,
            ttl: None,
        }
    }

    /// Check whether this value has expired based on its TTL.
    ///
    /// Returns `false` if no TTL is set.
    pub fn is_expired(&self) -> bool {
        let ttl = match self.ttl {
            Some(ttl) => ttl,
            None => return false,
        };

        let created = match parse_timestamp(&self.created_at) {
            Some(t) => t,
            None => return false,
        };

        match SystemTime::now().duration_since(created) {
            Ok(elapsed) => elapsed >= ttl,
            Err(_) => false,
        }
    }

    /// Update the stored data, bumping the version and `updated_at` timestamp.
    pub fn update(&mut self, data: Value) {
        self.data = data;
        self.version += 1;
        self.updated_at = format_timestamp(SystemTime::now());
    }

    /// Serialize this value to a JSON object.
    pub fn to_json(&self) -> Value {
        json!({
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        })
    }
}

// ---------------------------------------------------------------------------
// StoreOperation
// ---------------------------------------------------------------------------

/// An operation to be executed against a [`Store`].
#[derive(Debug, Clone)]
pub enum StoreOperation {
    /// Retrieve a value by key.
    Get(StoreKey),
    /// Insert or update a value.
    Put(StoreKey, Value),
    /// Delete a value by key.
    Delete(StoreKey),
    /// List all key-value pairs in a namespace.
    List(String),
    /// Remove all key-value pairs in a namespace.
    Clear(String),
}

// ---------------------------------------------------------------------------
// Store trait
// ---------------------------------------------------------------------------

/// A key-value store for persisting arbitrary data alongside checkpoints.
pub trait Store {
    /// Retrieve a value by key.
    fn get(&self, key: &StoreKey) -> Result<Option<StoreValue>>;

    /// Insert or update a value.
    fn put(&mut self, key: StoreKey, value: Value) -> Result<()>;

    /// Delete a value by key. Returns `true` if the key existed.
    fn delete(&mut self, key: &StoreKey) -> Result<bool>;

    /// List all key-value pairs in a namespace.
    fn list(&self, namespace: &str) -> Result<Vec<(StoreKey, StoreValue)>>;

    /// Remove all key-value pairs in a namespace. Returns the number removed.
    fn clear(&mut self, namespace: &str) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// InMemoryStore
// ---------------------------------------------------------------------------

/// A simple in-memory [`Store`] backed by a `HashMap`.
///
/// Suitable for testing and development. Data is lost when the store is
/// dropped.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    data: HashMap<String, (StoreKey, StoreValue)>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStore {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Return the number of entries in the store.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a sorted list of unique namespaces in the store.
    pub fn namespaces(&self) -> Vec<String> {
        let mut ns: Vec<String> = self
            .data
            .values()
            .map(|(k, _)| k.namespace.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ns.sort();
        ns
    }

    /// Remove all expired entries. Returns the number removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self
            .data
            .iter()
            .filter(|(_, (_, v))| v.is_expired())
            .map(|(fk, _)| fk.clone())
            .collect();

        let count = expired_keys.len();
        for fk in expired_keys {
            self.data.remove(&fk);
        }
        count
    }
}

impl Store for InMemoryStore {
    fn get(&self, key: &StoreKey) -> Result<Option<StoreValue>> {
        let full_key = key.full_key();
        match self.data.get(&full_key) {
            Some((_, value)) if value.is_expired() => Ok(None),
            Some((_, value)) => Ok(Some(value.clone())),
            None => Ok(None),
        }
    }

    fn put(&mut self, key: StoreKey, value: Value) -> Result<()> {
        let full_key = key.full_key();
        if let Some((_, existing)) = self.data.get_mut(&full_key) {
            existing.update(value);
        } else {
            let store_value = StoreValue::new(value);
            self.data.insert(full_key, (key, store_value));
        }
        Ok(())
    }

    fn delete(&mut self, key: &StoreKey) -> Result<bool> {
        let full_key = key.full_key();
        Ok(self.data.remove(&full_key).is_some())
    }

    fn list(&self, namespace: &str) -> Result<Vec<(StoreKey, StoreValue)>> {
        let mut results: Vec<(StoreKey, StoreValue)> = self
            .data
            .values()
            .filter(|(k, v)| k.matches_namespace(namespace) && !v.is_expired())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        results.sort_by(|a, b| a.0.key.cmp(&b.0.key));
        Ok(results)
    }

    fn clear(&mut self, namespace: &str) -> Result<usize> {
        let keys_to_remove: Vec<String> = self
            .data
            .iter()
            .filter(|(_, (k, _))| k.matches_namespace(namespace))
            .map(|(fk, _)| fk.clone())
            .collect();
        let count = keys_to_remove.len();
        for fk in keys_to_remove {
            self.data.remove(&fk);
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// NamespacedStore
// ---------------------------------------------------------------------------

/// A [`Store`] wrapper that prepends a fixed namespace prefix to all keys.
///
/// This is useful when multiple subsystems share a single store and each needs
/// its own isolated key space.
pub struct NamespacedStore {
    inner: Box<dyn Store>,
    namespace: String,
}

impl NamespacedStore {
    /// Wrap `inner` so that every key is prefixed with `namespace`.
    pub fn new(inner: Box<dyn Store>, namespace: &str) -> Self {
        Self {
            inner,
            namespace: namespace.to_string(),
        }
    }

    /// Prepend the configured namespace to a key's namespace.
    fn prefixed_key(&self, key: &StoreKey) -> StoreKey {
        StoreKey {
            namespace: format!("{}:{}", self.namespace, key.namespace),
            key: key.key.clone(),
        }
    }

    /// Prepend the configured namespace to a raw namespace string.
    fn prefixed_namespace(&self, ns: &str) -> String {
        format!("{}:{}", self.namespace, ns)
    }
}

impl Store for NamespacedStore {
    fn get(&self, key: &StoreKey) -> Result<Option<StoreValue>> {
        self.inner.get(&self.prefixed_key(key))
    }

    fn put(&mut self, key: StoreKey, value: Value) -> Result<()> {
        let prefixed = self.prefixed_key(&key);
        self.inner.put(prefixed, value)
    }

    fn delete(&mut self, key: &StoreKey) -> Result<bool> {
        self.inner.delete(&self.prefixed_key(key))
    }

    fn list(&self, namespace: &str) -> Result<Vec<(StoreKey, StoreValue)>> {
        self.inner.list(&self.prefixed_namespace(namespace))
    }

    fn clear(&mut self, namespace: &str) -> Result<usize> {
        self.inner.clear(&self.prefixed_namespace(namespace))
    }
}

// ---------------------------------------------------------------------------
// BatchStore
// ---------------------------------------------------------------------------

/// Executes multiple [`StoreOperation`]s against a [`Store`].
///
/// Operations are accumulated via [`add_operation`](Self::add_operation) and
/// then run together via [`execute`](Self::execute).
pub struct BatchStore {
    store: Box<dyn Store>,
    operations: Vec<StoreOperation>,
}

impl BatchStore {
    /// Create a new `BatchStore` wrapping the given store.
    pub fn new(store: Box<dyn Store>) -> Self {
        Self {
            store,
            operations: Vec::new(),
        }
    }

    /// Queue an operation for later execution.
    pub fn add_operation(&mut self, op: StoreOperation) {
        self.operations.push(op);
    }

    /// Return the number of queued operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Execute all queued operations in order and return their results.
    ///
    /// Each operation produces a `Result<Value>`:
    /// - `Get` -- the value's JSON representation, or `null` if missing
    /// - `Put` -- `null` on success
    /// - `Delete` -- `true` / `false` indicating whether the key existed
    /// - `List` -- a JSON array of `{ "key": ..., "value": ... }` objects
    /// - `Clear` -- the number of removed entries
    ///
    /// The operation queue is drained after execution.
    pub fn execute(&mut self) -> Result<Vec<Result<Value>>> {
        let ops: Vec<StoreOperation> = self.operations.drain(..).collect();
        let mut results = Vec::with_capacity(ops.len());

        for op in ops {
            let result = match op {
                StoreOperation::Get(key) => match self.store.get(&key) {
                    Ok(Some(val)) => Ok(val.to_json()),
                    Ok(None) => Ok(Value::Null),
                    Err(e) => Err(e),
                },
                StoreOperation::Put(key, value) => match self.store.put(key, value) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(e),
                },
                StoreOperation::Delete(key) => match self.store.delete(&key) {
                    Ok(existed) => Ok(Value::Bool(existed)),
                    Err(e) => Err(e),
                },
                StoreOperation::List(namespace) => match self.store.list(&namespace) {
                    Ok(entries) => {
                        let arr: Vec<Value> = entries
                            .into_iter()
                            .map(|(k, v)| {
                                json!({
                                    "key": k.full_key(),
                                    "value": v.to_json(),
                                })
                            })
                            .collect();
                        Ok(Value::Array(arr))
                    }
                    Err(e) => Err(e),
                },
                StoreOperation::Clear(namespace) => match self.store.clear(&namespace) {
                    Ok(count) => Ok(json!(count)),
                    Err(e) => Err(e),
                },
            };
            results.push(result);
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// StoreStats
// ---------------------------------------------------------------------------

/// Summary statistics for an [`InMemoryStore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    /// Total number of keys in the store.
    pub total_keys: usize,
    /// Number of distinct namespaces.
    pub namespaces: usize,
    /// Approximate total size of serialized values in bytes.
    pub total_size_bytes: usize,
}

impl StoreStats {
    /// Compute statistics from an [`InMemoryStore`].
    pub fn from_store(store: &InMemoryStore) -> Self {
        let total_keys = store.len();
        let namespaces = store.namespaces().len();
        let total_size_bytes: usize = store
            .data
            .values()
            .map(|(_, v)| v.data.to_string().len())
            .sum();

        Self {
            total_keys,
            namespaces,
            total_size_bytes,
        }
    }

    /// Serialize these statistics to a JSON object.
    pub fn to_json(&self) -> Value {
        json!({
            "total_keys": self.total_keys,
            "namespaces": self.namespaces,
            "total_size_bytes": self.total_size_bytes,
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
    use std::thread;
    use std::time::Duration;

    // ---------------------------------------------------------------
    // StoreKey tests
    // ---------------------------------------------------------------

    #[test]
    fn test_store_key_new() {
        let key = StoreKey::new("ns", "k");
        assert_eq!(key.namespace, "ns");
        assert_eq!(key.key, "k");
    }

    #[test]
    fn test_store_key_from_parts_single() {
        let key = StoreKey::from_parts(&["only"]);
        assert_eq!(key.namespace, "only");
        assert_eq!(key.key, "");
    }

    #[test]
    fn test_store_key_from_parts_multiple() {
        let key = StoreKey::from_parts(&["users", "123", "prefs"]);
        assert_eq!(key.namespace, "users");
        assert_eq!(key.key, "123:prefs");
    }

    #[test]
    fn test_store_key_full_key() {
        let key = StoreKey::new("ns", "k");
        assert_eq!(key.full_key(), "ns:k");
    }

    #[test]
    fn test_store_key_full_key_empty_key() {
        let key = StoreKey::from_parts(&["ns"]);
        assert_eq!(key.full_key(), "ns");
    }

    #[test]
    fn test_store_key_matches_namespace() {
        let key = StoreKey::new("users", "123");
        assert!(key.matches_namespace("users"));
        assert!(!key.matches_namespace("orders"));
    }

    #[test]
    fn test_store_key_equality() {
        let a = StoreKey::new("ns", "k");
        let b = StoreKey::new("ns", "k");
        assert_eq!(a, b);
    }

    #[test]
    fn test_store_key_inequality() {
        let a = StoreKey::new("ns", "k1");
        let b = StoreKey::new("ns", "k2");
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------
    // StoreValue tests
    // ---------------------------------------------------------------

    #[test]
    fn test_store_value_new() {
        let val = StoreValue::new(json!({"hello": "world"}));
        assert_eq!(val.data, json!({"hello": "world"}));
        assert_eq!(val.version, 1);
        assert!(!val.created_at.is_empty());
        assert!(!val.updated_at.is_empty());
        assert!(val.ttl.is_none());
    }

    #[test]
    fn test_store_value_not_expired_without_ttl() {
        let val = StoreValue::new(json!(null));
        assert!(!val.is_expired());
    }

    #[test]
    fn test_store_value_not_expired_with_long_ttl() {
        let mut val = StoreValue::new(json!(null));
        val.ttl = Some(Duration::from_secs(3600));
        assert!(!val.is_expired());
    }

    #[test]
    fn test_store_value_expired() {
        let mut val = StoreValue::new(json!(null));
        // Set created_at far in the past.
        val.created_at = "0.000".to_string();
        val.ttl = Some(Duration::from_secs(1));
        assert!(val.is_expired());
    }

    #[test]
    fn test_store_value_update() {
        let mut val = StoreValue::new(json!(1));
        let original_created = val.created_at.clone();
        thread::sleep(Duration::from_millis(10));

        val.update(json!(2));
        assert_eq!(val.data, json!(2));
        assert_eq!(val.version, 2);
        assert_eq!(val.created_at, original_created);
        // updated_at should differ (or at least not be older).
        assert!(val.updated_at >= original_created);
    }

    #[test]
    fn test_store_value_update_increments_version() {
        let mut val = StoreValue::new(json!(0));
        assert_eq!(val.version, 1);
        val.update(json!(1));
        assert_eq!(val.version, 2);
        val.update(json!(2));
        assert_eq!(val.version, 3);
        val.update(json!(3));
        assert_eq!(val.version, 4);
    }

    #[test]
    fn test_store_value_to_json() {
        let val = StoreValue::new(json!({"x": 1}));
        let j = val.to_json();
        assert_eq!(j["data"], json!({"x": 1}));
        assert_eq!(j["version"], json!(1));
        assert!(j["created_at"].is_string());
        assert!(j["updated_at"].is_string());
    }

    // ---------------------------------------------------------------
    // InMemoryStore CRUD tests
    // ---------------------------------------------------------------

    #[test]
    fn test_in_memory_store_new_is_empty() {
        let store = InMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_in_memory_store_put_and_get() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "k1");

        store.put(key.clone(), json!(42)).unwrap();
        assert_eq!(store.len(), 1);

        let val = store.get(&key).unwrap().unwrap();
        assert_eq!(val.data, json!(42));
        assert_eq!(val.version, 1);
    }

    #[test]
    fn test_in_memory_store_get_missing() {
        let store = InMemoryStore::new();
        let key = StoreKey::new("ns", "nope");
        assert!(store.get(&key).unwrap().is_none());
    }

    #[test]
    fn test_in_memory_store_put_updates_existing() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "k1");

        store.put(key.clone(), json!("v1")).unwrap();
        store.put(key.clone(), json!("v2")).unwrap();

        assert_eq!(store.len(), 1);
        let val = store.get(&key).unwrap().unwrap();
        assert_eq!(val.data, json!("v2"));
        assert_eq!(val.version, 2);
    }

    #[test]
    fn test_in_memory_store_delete_existing() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "k1");
        store.put(key.clone(), json!(1)).unwrap();

        assert!(store.delete(&key).unwrap());
        assert!(store.is_empty());
        assert!(store.get(&key).unwrap().is_none());
    }

    #[test]
    fn test_in_memory_store_delete_missing() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "nope");
        assert!(!store.delete(&key).unwrap());
    }

    #[test]
    fn test_in_memory_store_list_by_namespace() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("alpha", "a1"), json!("a"))
            .unwrap();
        store
            .put(StoreKey::new("alpha", "a2"), json!("b"))
            .unwrap();
        store
            .put(StoreKey::new("beta", "b1"), json!("c"))
            .unwrap();

        let alpha = store.list("alpha").unwrap();
        assert_eq!(alpha.len(), 2);
        assert!(alpha.iter().all(|(k, _)| k.namespace == "alpha"));

        let beta = store.list("beta").unwrap();
        assert_eq!(beta.len(), 1);
    }

    #[test]
    fn test_in_memory_store_list_empty_namespace() {
        let store = InMemoryStore::new();
        let result = store.list("nonexistent").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_in_memory_store_clear_namespace() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("alpha", "a1"), json!(1))
            .unwrap();
        store
            .put(StoreKey::new("alpha", "a2"), json!(2))
            .unwrap();
        store
            .put(StoreKey::new("beta", "b1"), json!(3))
            .unwrap();

        let removed = store.clear("alpha").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.len(), 1);
        assert!(store.list("alpha").unwrap().is_empty());
        assert_eq!(store.list("beta").unwrap().len(), 1);
    }

    #[test]
    fn test_in_memory_store_clear_empty_namespace() {
        let mut store = InMemoryStore::new();
        let removed = store.clear("nothing").unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_in_memory_store_namespaces() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("beta", "b1"), json!(1))
            .unwrap();
        store
            .put(StoreKey::new("alpha", "a1"), json!(2))
            .unwrap();
        store
            .put(StoreKey::new("alpha", "a2"), json!(3))
            .unwrap();

        let ns = store.namespaces();
        assert_eq!(ns, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_in_memory_store_ttl_expiration() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "ephemeral");
        store.put(key.clone(), json!("temp")).unwrap();

        // Manually expire the entry by setting created_at to epoch.
        if let Some((_, val)) = store.data.get_mut(&key.full_key()) {
            val.created_at = "0.000".to_string();
            val.ttl = Some(Duration::from_secs(1));
        }

        // get should return None for expired entries.
        assert!(store.get(&key).unwrap().is_none());
    }

    #[test]
    fn test_in_memory_store_cleanup_expired() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("ns", "keep"), json!("alive"))
            .unwrap();
        store
            .put(StoreKey::new("ns", "expire"), json!("dead"))
            .unwrap();

        // Mark one as expired.
        let expire_key = StoreKey::new("ns", "expire").full_key();
        if let Some((_, val)) = store.data.get_mut(&expire_key) {
            val.created_at = "0.000".to_string();
            val.ttl = Some(Duration::from_secs(1));
        }

        let cleaned = store.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(store.len(), 1);
        assert!(store
            .get(&StoreKey::new("ns", "keep"))
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_in_memory_store_list_excludes_expired() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("ns", "alive"), json!(1))
            .unwrap();
        store
            .put(StoreKey::new("ns", "dead"), json!(2))
            .unwrap();

        let expire_key = StoreKey::new("ns", "dead").full_key();
        if let Some((_, val)) = store.data.get_mut(&expire_key) {
            val.created_at = "0.000".to_string();
            val.ttl = Some(Duration::from_secs(1));
        }

        let items = store.list("ns").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0.key, "alive");
    }

    // ---------------------------------------------------------------
    // NamespacedStore tests
    // ---------------------------------------------------------------

    #[test]
    fn test_namespaced_store_put_and_get() {
        let inner = InMemoryStore::new();
        let mut ns_store = NamespacedStore::new(Box::new(inner), "prefix");

        let key = StoreKey::new("sub", "k1");
        ns_store.put(key.clone(), json!("hello")).unwrap();

        let val = ns_store.get(&key).unwrap().unwrap();
        assert_eq!(val.data, json!("hello"));
    }

    #[test]
    fn test_namespaced_store_isolation() {
        let inner = InMemoryStore::new();
        let mut ns_store = NamespacedStore::new(Box::new(inner), "prefix");

        let key = StoreKey::new("sub", "k1");
        ns_store.put(key.clone(), json!(1)).unwrap();

        let items = ns_store.list("sub").unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_namespaced_store_delete() {
        let inner = InMemoryStore::new();
        let mut ns_store = NamespacedStore::new(Box::new(inner), "pfx");

        let key = StoreKey::new("ns", "k");
        ns_store.put(key.clone(), json!(1)).unwrap();
        assert!(ns_store.delete(&key).unwrap());
        assert!(ns_store.get(&key).unwrap().is_none());
    }

    #[test]
    fn test_namespaced_store_clear() {
        let inner = InMemoryStore::new();
        let mut ns_store = NamespacedStore::new(Box::new(inner), "pfx");

        ns_store
            .put(StoreKey::new("ns", "k1"), json!(1))
            .unwrap();
        ns_store
            .put(StoreKey::new("ns", "k2"), json!(2))
            .unwrap();

        let removed = ns_store.clear("ns").unwrap();
        assert_eq!(removed, 2);
    }

    // ---------------------------------------------------------------
    // BatchStore tests
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_store_operation_count() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        assert_eq!(batch.operation_count(), 0);
        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "k1"),
            json!(1),
        ));
        batch.add_operation(StoreOperation::Get(StoreKey::new("ns", "k1")));
        assert_eq!(batch.operation_count(), 2);
    }

    #[test]
    fn test_batch_store_execute_put_then_get() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "k1"),
            json!(42),
        ));
        batch.add_operation(StoreOperation::Get(StoreKey::new("ns", "k1")));

        let results = batch.execute().unwrap();
        assert_eq!(results.len(), 2);

        // Put returns null.
        assert_eq!(results[0].as_ref().unwrap(), &Value::Null);

        // Get returns the value's JSON.
        let get_result = results[1].as_ref().unwrap();
        assert_eq!(get_result["data"], json!(42));
    }

    #[test]
    fn test_batch_store_execute_delete() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "k1"),
            json!(1),
        ));
        batch.add_operation(StoreOperation::Delete(StoreKey::new("ns", "k1")));
        batch.add_operation(StoreOperation::Delete(StoreKey::new("ns", "k1")));

        let results = batch.execute().unwrap();
        assert_eq!(results[1].as_ref().unwrap(), &Value::Bool(true));
        assert_eq!(results[2].as_ref().unwrap(), &Value::Bool(false));
    }

    #[test]
    fn test_batch_store_execute_list_and_clear() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "a"),
            json!(1),
        ));
        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "b"),
            json!(2),
        ));
        batch.add_operation(StoreOperation::List("ns".to_string()));
        batch.add_operation(StoreOperation::Clear("ns".to_string()));
        batch.add_operation(StoreOperation::List("ns".to_string()));

        let results = batch.execute().unwrap();
        // List after two puts should have 2 entries.
        let list_result = results[2].as_ref().unwrap();
        assert_eq!(list_result.as_array().unwrap().len(), 2);

        // Clear should return 2.
        assert_eq!(results[3].as_ref().unwrap(), &json!(2));

        // List after clear should be empty.
        let list_after = results[4].as_ref().unwrap();
        assert_eq!(list_after.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_batch_store_drains_operations() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        batch.add_operation(StoreOperation::Put(
            StoreKey::new("ns", "k"),
            json!(1),
        ));
        batch.execute().unwrap();
        assert_eq!(batch.operation_count(), 0);
    }

    #[test]
    fn test_batch_store_get_missing_key() {
        let store = InMemoryStore::new();
        let mut batch = BatchStore::new(Box::new(store));

        batch.add_operation(StoreOperation::Get(StoreKey::new("ns", "missing")));
        let results = batch.execute().unwrap();
        assert_eq!(results[0].as_ref().unwrap(), &Value::Null);
    }

    // ---------------------------------------------------------------
    // StoreStats tests
    // ---------------------------------------------------------------

    #[test]
    fn test_store_stats_empty() {
        let store = InMemoryStore::new();
        let stats = StoreStats::from_store(&store);
        assert_eq!(stats.total_keys, 0);
        assert_eq!(stats.namespaces, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[test]
    fn test_store_stats_populated() {
        let mut store = InMemoryStore::new();
        store
            .put(StoreKey::new("alpha", "a1"), json!({"x": 1}))
            .unwrap();
        store
            .put(StoreKey::new("alpha", "a2"), json!({"y": 2}))
            .unwrap();
        store
            .put(StoreKey::new("beta", "b1"), json!("hello"))
            .unwrap();

        let stats = StoreStats::from_store(&store);
        assert_eq!(stats.total_keys, 3);
        assert_eq!(stats.namespaces, 2);
        assert!(stats.total_size_bytes > 0);
    }

    #[test]
    fn test_store_stats_to_json() {
        let store = InMemoryStore::new();
        let stats = StoreStats::from_store(&store);
        let j = stats.to_json();
        assert_eq!(j["total_keys"], json!(0));
        assert_eq!(j["namespaces"], json!(0));
        assert_eq!(j["total_size_bytes"], json!(0));
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn test_version_incrementing_across_updates() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("ns", "versioned");

        for i in 0..10 {
            store.put(key.clone(), json!(i)).unwrap();
        }

        let val = store.get(&key).unwrap().unwrap();
        assert_eq!(val.version, 10);
        assert_eq!(val.data, json!(9));
    }

    #[test]
    fn test_empty_namespace_operations() {
        let mut store = InMemoryStore::new();
        let key = StoreKey::new("", "k");
        store.put(key.clone(), json!("val")).unwrap();

        let val = store.get(&key).unwrap().unwrap();
        assert_eq!(val.data, json!("val"));

        let items = store.list("").unwrap();
        assert_eq!(items.len(), 1);

        store.clear("").unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn test_default_impl() {
        let store = InMemoryStore::default();
        assert!(store.is_empty());
    }
}
