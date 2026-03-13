//! LLM response caching backends.
//!
//! This module provides a pluggable [`LlmCache`] trait and concrete
//! implementations for caching [`ChatResult`] responses:
//!
//! - [`InMemoryCache`] — bounded LRU cache with optional TTL (always available)
//! - [`SqliteCache`] — persistent SQLite-backed cache (requires the `sqlite` feature)
//!
//! Cache keys are computed from serialized messages and stop sequences using a
//! deterministic hash so that identical inputs always produce the same key.

pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

use cognis_core::messages::Message;
use cognis_core::outputs::ChatResult;

// Re-exports
pub use memory::InMemoryCache;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCache;

/// Pluggable cache backend for LLM responses.
///
/// Implementations must be thread-safe (`Send + Sync`).
#[async_trait]
pub trait LlmCache: Send + Sync {
    /// Look up a cached result by key.
    ///
    /// Returns `None` on a miss or if the entry has expired.
    async fn get(&self, key: &str) -> Option<ChatResult>;

    /// Store a result under the given key.
    async fn put(&self, key: &str, result: &ChatResult);

    /// Remove all entries from the cache.
    async fn clear(&self);
}

/// Compute a deterministic hex cache key from messages and optional stop
/// sequences.
///
/// The key is the hex representation of a 64-bit hash over the JSON-serialized
/// messages concatenated with the stop sequences.
pub fn compute_cache_key(messages: &[Message], stop: Option<&[String]>) -> String {
    let serialized = serde_json::to_string(messages).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    if let Some(stop) = stop {
        stop.hash(&mut hasher);
    }
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::HumanMessage;

    #[test]
    fn test_same_input_produces_same_key() {
        let msgs = vec![Message::Human(HumanMessage::new("hello"))];
        let k1 = compute_cache_key(&msgs, None);
        let k2 = compute_cache_key(&msgs, None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_different_input_produces_different_key() {
        let msgs_a = vec![Message::Human(HumanMessage::new("hello"))];
        let msgs_b = vec![Message::Human(HumanMessage::new("world"))];
        let k1 = compute_cache_key(&msgs_a, None);
        let k2 = compute_cache_key(&msgs_b, None);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_stop_sequences_affect_key() {
        let msgs = vec![Message::Human(HumanMessage::new("hello"))];
        let k1 = compute_cache_key(&msgs, None);
        let k2 = compute_cache_key(&msgs, Some(&["stop".to_string()]));
        assert_ne!(k1, k2);
    }
}
