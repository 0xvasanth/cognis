use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::error::Result;

/// Abstract interface for a key-value store.
#[async_trait]
pub trait BaseStore: Send + Sync {
    /// Get values associated with the given keys.
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<Value>>>;

    /// Set values for the given key-value pairs.
    async fn mset(&self, key_value_pairs: Vec<(String, Value)>) -> Result<()>;

    /// Delete the given keys.
    async fn mdelete(&self, keys: &[String]) -> Result<()>;

    /// Get keys matching the given prefix.
    async fn yield_keys(&self, prefix: Option<&str>) -> Result<Vec<String>>;
}

/// In-memory store implementation.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    store: tokio::sync::RwLock<HashMap<String, Value>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            store: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl BaseStore for InMemoryStore {
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<Value>>> {
        let store = self.store.read().await;
        Ok(keys.iter().map(|k| store.get(k).cloned()).collect())
    }

    async fn mset(&self, key_value_pairs: Vec<(String, Value)>) -> Result<()> {
        let mut store = self.store.write().await;
        for (key, value) in key_value_pairs {
            store.insert(key, value);
        }
        Ok(())
    }

    async fn mdelete(&self, keys: &[String]) -> Result<()> {
        let mut store = self.store.write().await;
        for key in keys {
            store.remove(key);
        }
        Ok(())
    }

    async fn yield_keys(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let store = self.store.read().await;
        let keys: Vec<String> = match prefix {
            Some(p) => store.keys().filter(|k| k.starts_with(p)).cloned().collect(),
            None => store.keys().cloned().collect(),
        };
        Ok(keys)
    }
}

/// Abstract interface for a byte-oriented key-value store.
#[async_trait]
pub trait ByteStore: Send + Sync {
    /// Get byte values associated with the given keys.
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>>;

    /// Set byte values for the given key-value pairs.
    async fn mset(&self, kvpairs: Vec<(String, Vec<u8>)>) -> Result<()>;

    /// Delete the given keys.
    async fn mdelete(&self, keys: &[String]) -> Result<()>;

    /// Get keys matching the given prefix.
    async fn yield_keys(&self, prefix: Option<&str>) -> Result<Vec<String>>;
}

/// In-memory byte store implementation.
#[derive(Debug, Default)]
pub struct InMemoryByteStore {
    store: RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryByteStore {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl ByteStore for InMemoryByteStore {
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        let store = self.store.read().await;
        Ok(keys.iter().map(|k| store.get(k).cloned()).collect())
    }

    async fn mset(&self, kvpairs: Vec<(String, Vec<u8>)>) -> Result<()> {
        let mut store = self.store.write().await;
        for (key, value) in kvpairs {
            store.insert(key, value);
        }
        Ok(())
    }

    async fn mdelete(&self, keys: &[String]) -> Result<()> {
        let mut store = self.store.write().await;
        for key in keys {
            store.remove(key);
        }
        Ok(())
    }

    async fn yield_keys(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let store = self.store.read().await;
        let keys: Vec<String> = match prefix {
            Some(p) => store.keys().filter(|k| k.starts_with(p)).cloned().collect(),
            None => store.keys().cloned().collect(),
        };
        Ok(keys)
    }
}
