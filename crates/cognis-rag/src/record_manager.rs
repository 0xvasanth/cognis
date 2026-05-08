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

/// Stable content fingerprint — FNV-1a 128-bit. Result is a 32-hex-char
/// string with ~2^-64 collision odds for non-adversarial inputs (well
/// inside the safe range for change-detection over millions of docs).
///
/// The algorithm is fixed (FNV-1a, the published 128-bit constants), so
/// fingerprints stored to disk stay valid across Rust toolchain
/// upgrades. Stays in-tree — no hash crate dependency.
///
/// Two docs with identical content always produce the same fingerprint;
/// changing any byte changes it.
pub fn fingerprint(content: &str) -> String {
    // FNV-1a 128-bit constants (published, public-domain).
    // See http://www.isthe.com/chongo/tech/comp/fnv/
    const FNV_OFFSET_BASIS: u128 = 0x6c62272e07bb014262b821756295c58d;
    const FNV_PRIME: u128 = 0x0000000001000000000000000000013b;
    let mut h: u128 = FNV_OFFSET_BASIS;
    for b in content.as_bytes() {
        h ^= u128::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    format!("{h:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fingerprint_is_deterministic() {
        assert_eq!(fingerprint("hello"), fingerprint("hello"));
        assert_ne!(fingerprint("hello"), fingerprint("world"));
    }

    #[test]
    fn fingerprint_is_stable_across_releases() {
        // Locked-in reference values. These are the FNV-1a 128-bit
        // outputs of the current implementation; if any change makes
        // this test fail, it means stored fingerprints in production
        // record managers will be invalidated — the algorithm change
        // must go through a versioned format migration, not a silent
        // bump.
        assert_eq!(
            fingerprint(""),
            "6c62272e07bb014262b821756295c58d",
            "empty input must equal the FNV-1a 128 offset basis"
        );
        assert_eq!(fingerprint("hello"), "e3e1efd54283d94f7081314b599d31b3");
        assert_eq!(
            fingerprint("the quick brown fox jumps over the lazy dog"),
            "577ea59947cc87c26ffa73dd35a3f550"
        );
    }

    #[test]
    fn fingerprint_is_32_hex_chars() {
        for s in ["", "a", "longer content with whitespace and digits 12345"] {
            let fp = fingerprint(s);
            assert_eq!(fp.len(), 32, "fp = {fp}");
            assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
        }
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
