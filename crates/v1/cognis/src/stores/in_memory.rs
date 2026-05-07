//! Thread-safe in-memory key-value store with optional per-key TTL.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use cognis_core::error::{CognisError, Result};

use super::Store;

/// An entry stored in the in-memory map.
#[derive(Debug, Clone)]
struct Entry {
    value: Vec<u8>,
    expires_at: Option<Instant>,
}

impl Entry {
    fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|t| Instant::now() >= t)
    }
}

/// Thread-safe in-memory key-value store.
///
/// Supports an optional default TTL applied to every `set` call, as well as
/// per-key TTL via [`InMemoryStore::set_with_ttl`].
#[derive(Debug)]
pub struct InMemoryStore {
    data: RwLock<HashMap<String, Entry>>,
    default_ttl: Option<Duration>,
}

impl InMemoryStore {
    /// Create a new empty store with no default TTL.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            default_ttl: None,
        }
    }

    /// Create a new empty store with a default TTL applied to every write.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            default_ttl: Some(ttl),
        }
    }

    /// Store `value` under `key` with an explicit TTL.
    pub fn set_with_ttl(&self, key: &str, value: &[u8], ttl: Duration) -> Result<()> {
        let entry = Entry {
            value: value.to_vec(),
            expires_at: Some(Instant::now() + ttl),
        };
        let mut data = self
            .data
            .write()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        data.insert(key.to_string(), entry);
        Ok(())
    }

    /// Remove all expired entries from the store.
    pub fn evict_expired(&self) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        data.retain(|_, entry| !entry.is_expired());
        Ok(())
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for InMemoryStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let data = self
            .data
            .read()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        match data.get(key) {
            Some(entry) if !entry.is_expired() => Ok(Some(entry.value.clone())),
            _ => Ok(None),
        }
    }

    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let entry = Entry {
            value: value.to_vec(),
            expires_at: self.default_ttl.map(|d| Instant::now() + d),
        };
        let mut data = self
            .data
            .write()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        data.insert(key.to_string(), entry);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<bool> {
        let mut data = self
            .data
            .write()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        Ok(data.remove(key).is_some())
    }

    fn exists(&self, key: &str) -> bool {
        let data = match self.data.read() {
            Ok(d) => d,
            Err(_) => return false,
        };
        match data.get(key) {
            Some(entry) => !entry.is_expired(),
            None => false,
        }
    }

    fn keys(&self) -> Result<Vec<String>> {
        let data = self
            .data
            .read()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        Ok(data
            .iter()
            .filter(|(_, entry)| !entry.is_expired())
            .map(|(k, _)| k.clone())
            .collect())
    }

    fn clear(&self) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|e| CognisError::Other(e.to_string()))?;
        data.clear();
        Ok(())
    }
}
