//! Key-value store implementations.
//!
//! This module provides several [`Store`] implementations for persisting
//! arbitrary byte data, including an in-memory store with optional TTL,
//! a filesystem-backed store, a namespaced wrapper, and a layered
//! read-through cache.

pub mod file;
pub mod in_memory;
pub mod layered;
pub mod namespaced;

pub use file::FileStore;
pub use in_memory::InMemoryStore;
pub use layered::LayeredStore;
pub use namespaced::NamespacedStore;

use rustchain_core::error::Result;

/// A synchronous key-value store trait for byte data.
///
/// All implementations must be thread-safe (`Send + Sync`).
pub trait Store: Send + Sync {
    /// Retrieve the value associated with `key`, or `None` if absent.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Store `value` under `key`, overwriting any previous value.
    fn set(&self, key: &str, value: &[u8]) -> Result<()>;

    /// Delete the entry for `key`. Returns `true` if the key existed.
    fn delete(&self, key: &str) -> Result<bool>;

    /// Check whether `key` exists in the store.
    fn exists(&self, key: &str) -> bool;

    /// Return all keys currently in the store.
    fn keys(&self) -> Result<Vec<String>>;

    /// Remove all entries from the store.
    fn clear(&self) -> Result<()>;
}
