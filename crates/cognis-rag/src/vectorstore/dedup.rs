//! Content-deduplicating [`VectorStore`] decorator.
//!
//! [`DedupVectorStore`] wraps any [`VectorStore`] and silently drops
//! documents whose normalised content fingerprint has already been seen.
//! Use it whenever the same fact may arrive multiple times (agent loops,
//! repeated indexing runs, multi-source pipelines) and you want the index
//! to stay clean without extra bookkeeping in application code.
//!
//! The default fingerprint normalises text (lowercase + collapsed
//! whitespace) then applies FNV-1a 128-bit — the same algorithm used by
//! [`crate::record_manager::fingerprint`]. Supply a custom function via
//! [`DedupVectorStore::with_fingerprint`] when your dedup key is not
//! derived from raw text (e.g. a document ID or a composite hash).

use std::collections::HashSet;
use std::collections::HashMap;

use async_trait::async_trait;

use cognis_core::Result;

use super::{Filter, SearchResult, VectorStore};

// ---------------------------------------------------------------------------
// Default fingerprint
// ---------------------------------------------------------------------------

/// Normalise `text` (lowercase + collapse whitespace) then fingerprint with
/// FNV-1a 128-bit. Two pieces of text that differ only in case or spacing
/// produce the same fingerprint and are treated as duplicates.
///
/// The algorithm is the same as [`crate::record_manager::fingerprint`] but
/// applied to the normalised form so that case / whitespace variants
/// collapse. Exposed as a public function so callers can pre-compute
/// fingerprints when seeding an existing store (see
/// [`DedupVectorStore::with_seen`]).
pub fn normalized_fingerprint(text: &str) -> String {
    // Normalise: lowercase + single-space whitespace runs.
    let normalised = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // FNV-1a 128-bit (published constants, public domain).
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013b;
    let mut h: u128 = OFFSET;
    for b in normalised.as_bytes() {
        h ^= u128::from(*b);
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:032x}")
}

// ---------------------------------------------------------------------------
// DedupVectorStore
// ---------------------------------------------------------------------------

/// A [`VectorStore`] decorator that silently skips documents whose
/// normalised content fingerprint is already in the seen-set.
///
/// # Type parameters
///
/// * `S` — the inner [`VectorStore`] implementation.
/// * `F` — the fingerprint function `fn(&str) -> String`. Defaults to
///   [`normalized_fingerprint`] (lowercase + whitespace-collapse + FNV-1a).
///   Provide a custom function via [`DedupVectorStore::with_fingerprint`]
///   when you need a different dedup key (e.g. document ID, composite hash).
///
/// # Persistence across restarts
///
/// The seen-set lives in memory and is cleared on process restart. To
/// survive restarts, query your storage for existing content fingerprints
/// on startup and pass them to [`DedupVectorStore::with_seen`]. The
/// [`normalized_fingerprint`] function is public so you can pre-compute
/// hashes from stored records.
///
/// ```rust,ignore
/// let hashes = db.query_all("SELECT content_hash FROM facts").await?;
/// let store = DedupVectorStore::with_seen(inner, hashes);
/// ```
pub struct DedupVectorStore<S, F = fn(&str) -> String>
where
    S: VectorStore,
    F: Fn(&str) -> String + Send + Sync,
{
    inner: S,
    fingerprint_fn: F,
    seen: HashSet<String>,
}

// --- constructors for the default fingerprint ---

impl<S: VectorStore> DedupVectorStore<S, fn(&str) -> String> {
    /// Wrap `inner` with an empty seen-set and the default
    /// [`normalized_fingerprint`] function.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            fingerprint_fn: normalized_fingerprint,
            seen: HashSet::new(),
        }
    }

    /// Wrap `inner` and pre-populate the seen-set from `seen` fingerprints.
    ///
    /// Use this when re-starting a process and you want to restore the
    /// dedup state from previously persisted fingerprints.
    pub fn with_seen(inner: S, seen: impl IntoIterator<Item = String>) -> Self {
        Self {
            inner,
            fingerprint_fn: normalized_fingerprint,
            seen: seen.into_iter().collect(),
        }
    }
}

// --- constructors with a custom fingerprint function ---

impl<S, F> DedupVectorStore<S, F>
where
    S: VectorStore,
    F: Fn(&str) -> String + Send + Sync,
{
    /// Wrap `inner` with a custom fingerprint function and an empty seen-set.
    ///
    /// The function receives the raw document text and returns a string key.
    /// Documents whose key is already in the seen-set are skipped.
    pub fn with_fingerprint(inner: S, f: F) -> Self {
        Self {
            inner,
            fingerprint_fn: f,
            seen: HashSet::new(),
        }
    }

    /// Wrap `inner` with a custom fingerprint function and a pre-populated
    /// seen-set.
    pub fn with_fingerprint_and_seen(
        inner: S,
        f: F,
        seen: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            inner,
            fingerprint_fn: f,
            seen: seen.into_iter().collect(),
        }
    }

    // --- inspection ---

    /// Whether `text` is already recorded in the seen-set (using the
    /// configured fingerprint function).
    pub fn contains(&self, text: &str) -> bool {
        self.seen.contains(&(self.fingerprint_fn)(text))
    }

    /// Read-only access to the inner store.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Mutable access to the inner store — e.g. to call store-specific
    /// methods not on the [`VectorStore`] trait.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Iterate over all fingerprints currently held in the seen-set.
    /// Useful when persisting state to storage before shutdown.
    pub fn seen_fingerprints(&self) -> impl Iterator<Item = &str> {
        self.seen.iter().map(|s| s.as_str())
    }

    /// Number of unique fingerprints recorded (≥ documents in the inner
    /// store when duplicates were skipped).
    pub fn seen_count(&self) -> usize {
        self.seen.len()
    }
}

// ---------------------------------------------------------------------------
// VectorStore impl
// ---------------------------------------------------------------------------

#[async_trait]
impl<S, F> VectorStore for DedupVectorStore<S, F>
where
    S: VectorStore + Send + Sync,
    F: Fn(&str) -> String + Send + Sync,
{
    /// Filter out texts whose fingerprint is already seen, then delegate
    /// the remainder to the inner store.
    ///
    /// Skipped documents are represented as `"dedup:skipped:{fingerprint}"`
    /// in the returned ID list so that callers whose code expects
    /// `ids.len() == texts.len()` still holds.
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut pass_texts: Vec<String> = Vec::new();
        let mut pass_meta: Vec<HashMap<String, serde_json::Value>> = Vec::new();
        // Tracks final position: None = sent to inner, Some(id) = skipped.
        let mut slots: Vec<Option<String>> = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let fp = (self.fingerprint_fn)(text);
            if self.seen.contains(&fp) {
                slots.push(Some(format!("dedup:skipped:{fp}")));
            } else {
                self.seen.insert(fp);
                pass_texts.push(text.clone());
                if let Some(m) = &metadata {
                    pass_meta.push(m[i].clone());
                }
                slots.push(None);
            }
        }

        let real_meta = if metadata.is_some() && !pass_meta.is_empty() {
            Some(pass_meta)
        } else {
            None
        };

        let mut inner_ids = if !pass_texts.is_empty() {
            self.inner.add_texts(pass_texts, real_meta).await?
        } else {
            Vec::new()
        };

        // Reconstruct output in original order.
        let mut inner_iter = inner_ids.drain(..);
        let ids = slots
            .into_iter()
            .map(|slot| match slot {
                Some(skipped_id) => skipped_id,
                None => inner_iter.next().unwrap_or_default(),
            })
            .collect();
        Ok(ids)
    }

    /// Filter out pre-embedded vectors whose text fingerprint is already
    /// seen, then delegate the remainder to the inner store.
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut pass_vecs: Vec<Vec<f32>> = Vec::new();
        let mut pass_texts: Vec<String> = Vec::new();
        let mut pass_meta: Vec<HashMap<String, serde_json::Value>> = Vec::new();
        let mut slots: Vec<Option<String>> = Vec::with_capacity(texts.len());

        for (i, (text, vec)) in texts.iter().zip(vectors.iter()).enumerate() {
            let fp = (self.fingerprint_fn)(text);
            if self.seen.contains(&fp) {
                slots.push(Some(format!("dedup:skipped:{fp}")));
            } else {
                self.seen.insert(fp);
                pass_texts.push(text.clone());
                pass_vecs.push(vec.clone());
                if let Some(m) = &metadata {
                    pass_meta.push(m[i].clone());
                }
                slots.push(None);
            }
        }

        let real_meta = if metadata.is_some() && !pass_meta.is_empty() {
            Some(pass_meta)
        } else {
            None
        };

        let mut inner_ids = if !pass_texts.is_empty() {
            self.inner.add_vectors(pass_vecs, pass_texts, real_meta).await?
        } else {
            Vec::new()
        };

        let mut inner_iter = inner_ids.drain(..);
        let ids = slots
            .into_iter()
            .map(|slot| match slot {
                Some(skipped_id) => skipped_id,
                None => inner_iter.next().unwrap_or_default(),
            })
            .collect();
        Ok(ids)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.inner.similarity_search(query, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.inner.similarity_search_by_vector(query_vector, k).await
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.inner.similarity_search_with_filter(query, k, filter).await
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        self.inner.delete(ids).await
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;
    use crate::vectorstore::InMemoryVectorStore;
    use std::sync::Arc;

    fn inner() -> InMemoryVectorStore {
        InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)))
    }

    // --- add_texts ---

    #[tokio::test]
    async fn skips_duplicate_on_second_add() {
        let mut store = DedupVectorStore::new(inner());
        store.add_texts(vec!["the workspace uses Go".into()], None).await.unwrap();
        store.add_texts(vec!["the workspace uses Go".into()], None).await.unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn case_and_whitespace_normalisation_deduplicates() {
        let mut store = DedupVectorStore::new(inner());
        store.add_texts(vec!["The workspace uses Go.".into()], None).await.unwrap();
        store.add_texts(vec!["  THE  WORKSPACE   USES  GO.  ".into()], None).await.unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn distinct_content_both_stored() {
        let mut store = DedupVectorStore::new(inner());
        store.add_texts(vec!["Fact A.".into()], None).await.unwrap();
        store.add_texts(vec!["Fact B.".into()], None).await.unwrap();
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn batch_add_with_mixed_duplicates() {
        let mut store = DedupVectorStore::new(inner());
        let ids1 = store
            .add_texts(vec!["unique one".into(), "unique two".into()], None)
            .await
            .unwrap();
        assert_eq!(ids1.len(), 2);
        assert!(!ids1[0].starts_with("dedup:skipped:"));
        assert!(!ids1[1].starts_with("dedup:skipped:"));

        let ids2 = store
            .add_texts(
                vec!["unique one".into(), "unique three".into(), "unique two".into()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(ids2.len(), 3);
        // First and third are duplicates.
        assert!(ids2[0].starts_with("dedup:skipped:"));
        assert!(!ids2[1].starts_with("dedup:skipped:"), "unique three should pass through");
        assert!(ids2[2].starts_with("dedup:skipped:"));
        assert_eq!(store.len(), 3);
    }

    #[tokio::test]
    async fn with_seen_skips_pre_populated_fingerprints() {
        let fp = normalized_fingerprint("already known fact");
        let mut store = DedupVectorStore::with_seen(inner(), [fp]);
        store
            .add_texts(vec!["already known fact".into()], None)
            .await
            .unwrap();
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn read_operations_pass_through() {
        let mut store = DedupVectorStore::new(inner());
        store.add_texts(vec!["searchable fact".into()], None).await.unwrap();
        let results = store.similarity_search("fact", 5).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn delete_passes_through() {
        let mut store = DedupVectorStore::new(inner());
        let ids = store.add_texts(vec!["deletable".into()], None).await.unwrap();
        assert_eq!(store.len(), 1);
        store.delete(ids).await.unwrap();
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn seen_count_tracks_unique_fingerprints() {
        let mut store = DedupVectorStore::new(inner());
        store.add_texts(vec!["a".into(), "b".into()], None).await.unwrap();
        store.add_texts(vec!["a".into()], None).await.unwrap(); // duplicate
        assert_eq!(store.seen_count(), 2);
    }

    #[tokio::test]
    async fn contains_reflects_seen_set() {
        let mut store = DedupVectorStore::new(inner());
        assert!(!store.contains("new fact"));
        store.add_texts(vec!["new fact".into()], None).await.unwrap();
        assert!(store.contains("new fact"));
        // Case variant also matches.
        assert!(store.contains("NEW FACT"));
    }

    // --- custom fingerprint ---

    #[tokio::test]
    async fn custom_fingerprint_uses_provided_function() {
        // Fingerprint only on the first word — any two texts with the same
        // first word are considered duplicates.
        let mut store = DedupVectorStore::with_fingerprint(inner(), |text: &str| {
            text.split_whitespace()
                .next()
                .unwrap_or("")
                .to_lowercase()
                .to_string()
        });
        store.add_texts(vec!["rust is great".into()], None).await.unwrap();
        store.add_texts(vec!["rust is also fast".into()], None).await.unwrap();
        // Both start with "rust" → second is a duplicate.
        assert_eq!(store.len(), 1);
    }

    // --- add_vectors ---

    #[tokio::test]
    async fn add_vectors_deduplicates() {
        let mut store = DedupVectorStore::new(inner());
        let vec = vec![0.1_f32; 8];
        store
            .add_vectors(vec![vec.clone()], vec!["vec fact".into()], None)
            .await
            .unwrap();
        store
            .add_vectors(vec![vec.clone()], vec!["vec fact".into()], None)
            .await
            .unwrap();
        assert_eq!(store.len(), 1);
    }

    // --- normalized_fingerprint properties ---

    #[test]
    fn fingerprint_is_deterministic() {
        assert_eq!(
            normalized_fingerprint("hello world"),
            normalized_fingerprint("hello world")
        );
    }

    #[test]
    fn fingerprint_is_case_insensitive() {
        assert_eq!(
            normalized_fingerprint("Hello World"),
            normalized_fingerprint("hello world")
        );
    }

    #[test]
    fn fingerprint_collapses_whitespace() {
        assert_eq!(
            normalized_fingerprint("hello   world"),
            normalized_fingerprint("hello world")
        );
    }

    #[test]
    fn fingerprint_distinguishes_different_content() {
        assert_ne!(
            normalized_fingerprint("hello world"),
            normalized_fingerprint("goodbye world")
        );
    }
}
