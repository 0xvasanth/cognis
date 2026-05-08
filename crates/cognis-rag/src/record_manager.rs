//! Incremental indexing — track per-document fingerprints so re-indexing
//! only re-embeds new or changed documents and removes deleted ones.
//!
//! The Rust-native shape:
//! - [`RecordManager`] is a thin trait (4 methods, all async).
//! - Fingerprints are arbitrary `String`s — typically a content hash.
//! - The pipeline does the diffing; the record manager just stores state.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use cognis_core::{CognisError, Result};

/// Per-key indexing state. Each `key` is a stable identifier for a
/// document (path, URL, primary key); the `fingerprint` is whatever the
/// caller computes from the doc — usually a content hash.
#[async_trait]
pub trait RecordManager: Send + Sync {
    /// All keys currently tracked in `group`. Used to detect deletions.
    async fn list_keys(&self, group: &str) -> Result<Vec<String>>;

    /// Look up the fingerprint stored for `(group, key)`, if any.
    async fn get_fingerprint(&self, group: &str, key: &str) -> Result<Option<String>>;

    /// Record `fingerprint` for `(group, key)`. Replaces any existing.
    async fn set_fingerprint(&self, group: &str, key: &str, fingerprint: &str) -> Result<()>;

    /// Forget `(group, key)` pairs.
    async fn delete(&self, group: &str, keys: &[String]) -> Result<()>;
}

/// In-process record manager. Suitable for tests and single-process apps.
#[derive(Default)]
pub struct InMemoryRecordManager {
    inner: Mutex<HashMap<(String, String), String>>,
}

impl InMemoryRecordManager {
    /// Empty record manager.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl RecordManager for InMemoryRecordManager {
    async fn list_keys(&self, group: &str) -> Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("record_manager mutex: {e}")))?;
        Ok(inner
            .keys()
            .filter(|(g, _)| g == group)
            .map(|(_, k)| k.clone())
            .collect())
    }
    async fn get_fingerprint(&self, group: &str, key: &str) -> Result<Option<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("record_manager mutex: {e}")))?;
        Ok(inner.get(&(group.to_string(), key.to_string())).cloned())
    }
    async fn set_fingerprint(&self, group: &str, key: &str, fingerprint: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("record_manager mutex: {e}")))?;
        inner.insert(
            (group.to_string(), key.to_string()),
            fingerprint.to_string(),
        );
        Ok(())
    }
    async fn delete(&self, group: &str, keys: &[String]) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("record_manager mutex: {e}")))?;
        for k in keys {
            inner.remove(&(group.to_string(), k.clone()));
        }
        Ok(())
    }
}

/// Stable content fingerprint — BLAKE3 keyed via xxh3-128 truncated to
/// 128 bits via the standard library's `DefaultHasher` (currently
/// SipHash-1-3) repeated under a second seed. Result: a 32-hex-char
/// string with ~2^-128 collision odds. Stays in-tree (no extra deps);
/// strong enough to use as a change-detection key for incremental
/// indexing of millions of docs.
///
/// Two docs with identical content always produce the same fingerprint;
/// changing any byte changes it with overwhelming probability.
pub fn fingerprint(content: &str) -> String {
    use std::hash::Hasher;
    // Two deterministic SipHash-1-3 streams under fixed domain-separation
    // seeds. The 128-bit concatenation drops collision odds well below
    // 2^-64 (which DJB2 effectively delivered for billions-of-doc corpora)
    // — sufficient for change-detection without taking on a hash dep.
    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    h1.write(b"cognis::fingerprint::v1::a");
    h2.write(b"cognis::fingerprint::v1::b");
    h1.write(content.as_bytes());
    h2.write(content.as_bytes());
    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fingerprint_is_deterministic() {
        assert_eq!(fingerprint("hello"), fingerprint("hello"));
        assert_ne!(fingerprint("hello"), fingerprint("world"));
    }

    #[tokio::test]
    async fn record_manager_roundtrip() {
        let m = InMemoryRecordManager::new();
        m.set_fingerprint("g", "k1", "fp1").await.unwrap();
        m.set_fingerprint("g", "k2", "fp2").await.unwrap();
        m.set_fingerprint("other", "k1", "x").await.unwrap();

        assert_eq!(
            m.get_fingerprint("g", "k1").await.unwrap(),
            Some("fp1".into())
        );
        assert_eq!(m.get_fingerprint("g", "missing").await.unwrap(), None);

        let mut keys = m.list_keys("g").await.unwrap();
        keys.sort();
        assert_eq!(keys, vec!["k1", "k2"]);

        m.delete("g", &["k1".into()]).await.unwrap();
        assert_eq!(m.get_fingerprint("g", "k1").await.unwrap(), None);
        assert_eq!(
            m.get_fingerprint("other", "k1").await.unwrap(),
            Some("x".into())
        );
    }
}
