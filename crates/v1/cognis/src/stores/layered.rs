//! Layered read-through cache store.
//!
//! Composes a fast (cache) store and a slow (persistent) store. Reads check
//! the fast store first; on a cache miss the value is fetched from the slow
//! store and promoted into the fast store.

use cognis_core::error::Result;

use super::Store;

/// A two-tier read-through cache built from any two [`Store`] implementations.
///
/// - **`get`**: tries `fast` first, then falls through to `slow` and populates
///   `fast` on a miss.
/// - **`set`**: writes to both stores.
/// - **`delete`**: removes from both stores.
pub struct LayeredStore {
    fast: Box<dyn Store>,
    slow: Box<dyn Store>,
}

impl LayeredStore {
    /// Create a new layered store.
    ///
    /// `fast` is the cache layer (e.g. in-memory) and `slow` is the
    /// persistent backing store.
    pub fn new(fast: Box<dyn Store>, slow: Box<dyn Store>) -> Self {
        Self { fast, slow }
    }

    /// Remove a key from the fast (cache) layer only.
    pub fn invalidate(&self, key: &str) -> Result<bool> {
        self.fast.delete(key)
    }

    /// Pre-populate the fast store with values from the slow store.
    pub fn warm_up(&self, keys: &[&str]) -> Result<()> {
        for key in keys {
            if let Some(value) = self.slow.get(key)? {
                self.fast.set(key, &value)?;
            }
        }
        Ok(())
    }
}

impl Store for LayeredStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Try fast store first.
        if let Some(value) = self.fast.get(key)? {
            return Ok(Some(value));
        }
        // Fall through to slow store.
        if let Some(value) = self.slow.get(key)? {
            // Promote to fast store.
            self.fast.set(key, &value)?;
            return Ok(Some(value));
        }
        Ok(None)
    }

    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.fast.set(key, value)?;
        self.slow.set(key, value)?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<bool> {
        let a = self.fast.delete(key)?;
        let b = self.slow.delete(key)?;
        Ok(a || b)
    }

    fn exists(&self, key: &str) -> bool {
        self.fast.exists(key) || self.slow.exists(key)
    }

    fn keys(&self) -> Result<Vec<String>> {
        // Union of both stores' keys.
        let mut all = self.fast.keys()?;
        let slow_keys = self.slow.keys()?;
        for k in slow_keys {
            if !all.contains(&k) {
                all.push(k);
            }
        }
        Ok(all)
    }

    fn clear(&self) -> Result<()> {
        self.fast.clear()?;
        self.slow.clear()?;
        Ok(())
    }
}
