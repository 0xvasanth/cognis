//! Standalone in-memory vector store for similarity search.
//!
//! Provides a lightweight, synchronous vector store with pluggable similarity
//! metrics (cosine, Euclidean, dot product, Manhattan), metadata filtering,
//! configurable search queries, and Maximal Marginal Relevance (MMR) search
//! for result diversity.

use std::collections::HashMap;

use serde_json::Value;

// ---------------------------------------------------------------------------
// VectorEntry
// ---------------------------------------------------------------------------

/// A single entry in the vector store, pairing an embedding with its document
/// text and metadata.
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// The original document text.
    pub document: String,
    /// Arbitrary metadata associated with this entry.
    pub metadata: HashMap<String, Value>,
}

impl VectorEntry {
    /// Create a new vector entry.
    pub fn new(
        id: impl Into<String>,
        embedding: Vec<f64>,
        document: impl Into<String>,
        metadata: HashMap<String, Value>,
    ) -> Self {
        Self {
            id: id.into(),
            embedding,
            document: document.into(),
            metadata,
        }
    }

    /// Return the number of dimensions in the embedding vector.
    pub fn dimensions(&self) -> usize {
        self.embedding.len()
    }

    /// Serialize this entry to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "embedding": self.embedding,
            "document": self.document,
            "metadata": self.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// SimilarityMetric
// ---------------------------------------------------------------------------

/// Similarity metric used for comparing embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity — measures the angle between two vectors.
    /// Higher values indicate greater similarity (range `[-1, 1]`).
    Cosine,
    /// Euclidean (L2) distance — lower values indicate greater similarity.
    /// The returned score is negated so that higher is still better.
    Euclidean,
    /// Dot product — higher values indicate greater similarity.
    DotProduct,
    /// Manhattan (L1) distance — lower values indicate greater similarity.
    /// The returned score is negated so that higher is still better.
    Manhattan,
}

impl SimilarityMetric {
    /// Compute the similarity score between two vectors.
    ///
    /// For distance-based metrics (Euclidean, Manhattan), the returned value
    /// is negated so that higher scores always indicate greater similarity.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        match self {
            SimilarityMetric::Cosine => {
                let mut dot = 0.0f64;
                let mut norm_a = 0.0f64;
                let mut norm_b = 0.0f64;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                if denom == 0.0 {
                    0.0
                } else {
                    dot / denom
                }
            }
            SimilarityMetric::Euclidean => {
                let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                -sum.sqrt()
            }
            SimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
            SimilarityMetric::Manhattan => {
                let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
                -sum
            }
        }
    }

    /// Return the human-readable name of this metric.
    pub fn name(&self) -> &'static str {
        match self {
            SimilarityMetric::Cosine => "cosine",
            SimilarityMetric::Euclidean => "euclidean",
            SimilarityMetric::DotProduct => "dot_product",
            SimilarityMetric::Manhattan => "manhattan",
        }
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result pairing a vector entry with its similarity score
/// and rank within the result set.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched vector entry.
    pub entry: VectorEntry,
    /// The similarity score (higher is better).
    pub score: f64,
    /// The rank within the result set (0-based).
    pub rank: usize,
}

impl SearchResult {
    /// Serialize this search result to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "entry": self.entry.to_json(),
            "score": self.score,
            "rank": self.rank,
        })
    }
}

// ---------------------------------------------------------------------------
// SearchQuery
// ---------------------------------------------------------------------------

/// A query against the vector store with configurable parameters.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The query embedding vector.
    pub vector: Vec<f64>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Optional minimum similarity score threshold.
    pub min_score: Option<f64>,
    /// Optional metadata filter — all key-value pairs must match exactly.
    pub metadata_filter: Option<HashMap<String, Value>>,
}

impl SearchQuery {
    /// Create a new search query builder with the given query vector.
    pub fn builder(vector: Vec<f64>) -> SearchQueryBuilder {
        SearchQueryBuilder {
            vector,
            top_k: 10,
            min_score: None,
            metadata_filter: None,
        }
    }
}

/// Builder for [`SearchQuery`].
#[derive(Debug, Clone)]
pub struct SearchQueryBuilder {
    vector: Vec<f64>,
    top_k: usize,
    min_score: Option<f64>,
    metadata_filter: Option<HashMap<String, Value>>,
}

impl SearchQueryBuilder {
    /// Set the maximum number of results to return.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum similarity score threshold.
    pub fn min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Set the metadata filter. All key-value pairs must match exactly.
    pub fn metadata_filter(mut self, filter: HashMap<String, Value>) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Build the [`SearchQuery`].
    pub fn build(self) -> SearchQuery {
        SearchQuery {
            vector: self.vector,
            top_k: self.top_k,
            min_score: self.min_score,
            metadata_filter: self.metadata_filter,
        }
    }
}

// ---------------------------------------------------------------------------
// InMemoryVectorStore
// ---------------------------------------------------------------------------

/// A synchronous in-memory vector store for similarity search.
///
/// Stores vector entries in a `HashMap` keyed by ID and supports search with
/// configurable similarity metrics, metadata filtering, and score thresholds.
#[derive(Debug)]
pub struct InMemoryVectorStore {
    entries: HashMap<String, VectorEntry>,
    metric: SimilarityMetric,
}

impl InMemoryVectorStore {
    /// Create an empty vector store with the given similarity metric.
    pub fn new(metric: SimilarityMetric) -> Self {
        Self {
            entries: HashMap::new(),
            metric,
        }
    }

    /// Add a single entry to the store.
    pub fn add(&mut self, entry: VectorEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Add a batch of entries to the store.
    pub fn add_batch(&mut self, entries: Vec<VectorEntry>) {
        for entry in entries {
            self.add(entry);
        }
    }

    /// Search the store for the most similar entries to the query.
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchResult> {
        let mut scored: Vec<(String, f64)> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                if let Some(filter) = &query.metadata_filter {
                    filter.iter().all(|(k, v)| entry.metadata.get(k) == Some(v))
                } else {
                    true
                }
            })
            .map(|(id, entry)| {
                let score = self.metric.compute(&query.vector, &entry.embedding);
                (id.clone(), score)
            })
            .collect();

        // Sort by descending score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply min_score filter.
        if let Some(min_score) = query.min_score {
            scored.retain(|(_, score)| *score >= min_score);
        }

        scored.truncate(query.top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (id, score))| SearchResult {
                entry: self.entries[&id].clone(),
                score,
                rank,
            })
            .collect()
    }

    /// Get an entry by its ID.
    pub fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.entries.get(id)
    }

    /// Delete an entry by its ID. Returns `true` if the entry existed.
    pub fn delete(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    /// Return the number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the store.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Return the dimensionality of stored vectors, inferred from the first
    /// entry. Returns `None` if the store is empty.
    pub fn dimensions(&self) -> Option<usize> {
        self.entries.values().next().map(|e| e.embedding.len())
    }

    /// Return a reference to the similarity metric used by this store.
    pub fn metric(&self) -> &SimilarityMetric {
        &self.metric
    }
}

// ---------------------------------------------------------------------------
// VectorStoreConfig
// ---------------------------------------------------------------------------

/// Configuration for creating a vector store.
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    /// The similarity metric to use.
    pub metric: SimilarityMetric,
    /// Optional maximum number of entries.
    pub max_entries: Option<usize>,
    /// Whether to L2-normalize vectors on insert.
    pub normalize_on_insert: bool,
}

impl VectorStoreConfig {
    /// Create a new configuration builder with the given similarity metric.
    pub fn builder(metric: SimilarityMetric) -> VectorStoreConfigBuilder {
        VectorStoreConfigBuilder {
            metric,
            max_entries: None,
            normalize_on_insert: false,
        }
    }
}

/// Builder for [`VectorStoreConfig`].
#[derive(Debug, Clone)]
pub struct VectorStoreConfigBuilder {
    metric: SimilarityMetric,
    max_entries: Option<usize>,
    normalize_on_insert: bool,
}

impl VectorStoreConfigBuilder {
    /// Set the maximum number of entries the store will hold.
    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = Some(max);
        self
    }

    /// Set whether to L2-normalize vectors on insert.
    pub fn normalize_on_insert(mut self, normalize: bool) -> Self {
        self.normalize_on_insert = normalize;
        self
    }

    /// Build the [`VectorStoreConfig`].
    pub fn build(self) -> VectorStoreConfig {
        VectorStoreConfig {
            metric: self.metric,
            max_entries: self.max_entries,
            normalize_on_insert: self.normalize_on_insert,
        }
    }
}

// ---------------------------------------------------------------------------
// VectorStoreStats
// ---------------------------------------------------------------------------

/// Statistics about a vector store's contents.
#[derive(Debug, Clone)]
pub struct VectorStoreStats {
    /// Total number of entries in the store.
    pub total_entries: usize,
    /// Dimensionality of the stored vectors, if any entries exist.
    pub dimensions: Option<usize>,
    /// Name of the similarity metric in use.
    pub metric_name: String,
    /// Average L2 magnitude of stored vectors.
    pub avg_vector_magnitude: f64,
}

impl VectorStoreStats {
    /// Compute statistics from a vector store.
    pub fn from_store(store: &InMemoryVectorStore) -> Self {
        let total_entries = store.len();
        let dimensions = store.dimensions();
        let metric_name = store.metric().name().to_string();

        let avg_vector_magnitude = if total_entries == 0 {
            0.0
        } else {
            let total_mag: f64 = store
                .entries
                .values()
                .map(|e| {
                    let sum_sq: f64 = e.embedding.iter().map(|x| x * x).sum();
                    sum_sq.sqrt()
                })
                .sum();
            total_mag / total_entries as f64
        };

        Self {
            total_entries,
            dimensions,
            metric_name,
            avg_vector_magnitude,
        }
    }

    /// Serialize these statistics to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_entries": self.total_entries,
            "dimensions": self.dimensions,
            "metric_name": self.metric_name,
            "avg_vector_magnitude": self.avg_vector_magnitude,
        })
    }
}

// ---------------------------------------------------------------------------
// MaxMarginalRelevance
// ---------------------------------------------------------------------------

/// Maximal Marginal Relevance (MMR) search for balancing relevance and diversity.
///
/// MMR iteratively selects results that are similar to the query but dissimilar
/// to already-selected results, controlled by a `lambda` parameter:
/// - `lambda = 1.0` — pure relevance (identical to standard similarity search)
/// - `lambda = 0.0` — pure diversity (maximally dissimilar to selected results)
pub struct MaxMarginalRelevance;

impl MaxMarginalRelevance {
    /// Perform MMR search over the given vector store.
    ///
    /// # Arguments
    ///
    /// * `store` — The vector store to search.
    /// * `query` — The query embedding vector.
    /// * `top_k` — Number of results to return.
    /// * `lambda` — Trade-off between relevance and diversity (`0.0` to `1.0`).
    ///
    /// # Returns
    ///
    /// A vector of [`SearchResult`]s ranked by the MMR criterion.
    pub fn search(
        store: &InMemoryVectorStore,
        query: &[f64],
        top_k: usize,
        lambda: f64,
    ) -> Vec<SearchResult> {
        if store.is_empty() || top_k == 0 {
            return vec![];
        }

        // Pre-compute similarity of every candidate to the query.
        let candidates: Vec<(&String, &VectorEntry, f64)> = store
            .entries
            .iter()
            .map(|(id, entry)| {
                let sim = cosine_similarity(query, &entry.embedding);
                (id, entry, sim)
            })
            .collect();

        let mut selected_indices: Vec<usize> = Vec::with_capacity(top_k);
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();

        for _ in 0..top_k.min(candidates.len()) {
            let mut best_idx_in_remaining = 0;
            let mut best_mmr_score = f64::NEG_INFINITY;

            for (ri, &ci) in remaining.iter().enumerate() {
                let relevance = candidates[ci].2;

                // Max similarity to any already-selected candidate.
                let max_sim_to_selected = selected_indices
                    .iter()
                    .map(|&si| {
                        cosine_similarity(&candidates[ci].1.embedding, &candidates[si].1.embedding)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                let diversity_penalty = if selected_indices.is_empty() {
                    0.0
                } else {
                    max_sim_to_selected
                };

                let mmr_score = lambda * relevance - (1.0 - lambda) * diversity_penalty;

                if mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx_in_remaining = ri;
                }
            }

            let chosen = remaining.swap_remove(best_idx_in_remaining);
            selected_indices.push(chosen);
        }

        selected_indices
            .into_iter()
            .enumerate()
            .map(|(rank, ci)| SearchResult {
                entry: candidates[ci].1.clone(),
                score: candidates[ci].2,
                rank,
            })
            .collect()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper functions ----

    fn make_entry(id: &str, embedding: Vec<f64>, document: &str) -> VectorEntry {
        VectorEntry::new(id, embedding, document, HashMap::new())
    }

    fn make_entry_with_metadata(
        id: &str,
        embedding: Vec<f64>,
        document: &str,
        meta: &[(&str, Value)],
    ) -> VectorEntry {
        let metadata: HashMap<String, Value> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        VectorEntry::new(id, embedding, document, metadata)
    }

    // ---- VectorEntry tests ----

    #[test]
    fn test_vector_entry_creation() {
        let entry = make_entry("e1", vec![1.0, 2.0, 3.0], "hello");
        assert_eq!(entry.id, "e1");
        assert_eq!(entry.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.document, "hello");
        assert!(entry.metadata.is_empty());
    }

    #[test]
    fn test_vector_entry_dimensions() {
        let entry = make_entry("e1", vec![0.1; 128], "doc");
        assert_eq!(entry.dimensions(), 128);
    }

    #[test]
    fn test_vector_entry_zero_dimensions() {
        let entry = make_entry("e1", vec![], "doc");
        assert_eq!(entry.dimensions(), 0);
    }

    #[test]
    fn test_vector_entry_to_json() {
        let entry = make_entry("e1", vec![1.0, 2.0], "doc");
        let json = entry.to_json();
        assert_eq!(json["id"], "e1");
        assert_eq!(json["document"], "doc");
        assert_eq!(json["embedding"][0], 1.0);
        assert_eq!(json["embedding"][1], 2.0);
    }

    #[test]
    fn test_vector_entry_with_metadata() {
        let entry = make_entry_with_metadata(
            "e1",
            vec![1.0],
            "doc",
            &[("source", Value::String("test".into()))],
        );
        assert_eq!(
            entry.metadata.get("source"),
            Some(&Value::String("test".into()))
        );
    }

    // ---- SimilarityMetric tests ----

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let score = SimilarityMetric::Cosine.compute(&v, &v);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = SimilarityMetric::Cosine.compute(&a, &b);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let score = SimilarityMetric::Cosine.compute(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let zero = vec![0.0, 0.0];
        let other = vec![1.0, 2.0];
        let score = SimilarityMetric::Cosine.compute(&zero, &other);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let score = SimilarityMetric::Euclidean.compute(&v, &v);
        assert!(score.abs() < 1e-10); // -0.0
    }

    #[test]
    fn test_euclidean_known_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let score = SimilarityMetric::Euclidean.compute(&a, &b);
        assert!((score - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let score = SimilarityMetric::DotProduct.compute(&a, &b);
        assert!((score - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = SimilarityMetric::DotProduct.compute(&a, &b);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let score = SimilarityMetric::Manhattan.compute(&a, &b);
        assert!((score - (-7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_identical() {
        let v = vec![1.0, 2.0];
        let score = SimilarityMetric::Manhattan.compute(&v, &v);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn test_metric_name() {
        assert_eq!(SimilarityMetric::Cosine.name(), "cosine");
        assert_eq!(SimilarityMetric::Euclidean.name(), "euclidean");
        assert_eq!(SimilarityMetric::DotProduct.name(), "dot_product");
        assert_eq!(SimilarityMetric::Manhattan.name(), "manhattan");
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_metric_dimension_mismatch() {
        SimilarityMetric::Cosine.compute(&[1.0, 2.0], &[1.0]);
    }

    // ---- SearchQuery builder tests ----

    #[test]
    fn test_search_query_defaults() {
        let query = SearchQuery::builder(vec![1.0, 0.0]).build();
        assert_eq!(query.vector, vec![1.0, 0.0]);
        assert_eq!(query.top_k, 10);
        assert!(query.min_score.is_none());
        assert!(query.metadata_filter.is_none());
    }

    #[test]
    fn test_search_query_with_top_k() {
        let query = SearchQuery::builder(vec![1.0]).top_k(5).build();
        assert_eq!(query.top_k, 5);
    }

    #[test]
    fn test_search_query_with_min_score() {
        let query = SearchQuery::builder(vec![1.0]).min_score(0.8).build();
        assert_eq!(query.min_score, Some(0.8));
    }

    #[test]
    fn test_search_query_with_metadata_filter() {
        let mut filter = HashMap::new();
        filter.insert("source".to_string(), Value::String("test".into()));
        let query = SearchQuery::builder(vec![1.0])
            .metadata_filter(filter.clone())
            .build();
        assert_eq!(query.metadata_filter, Some(filter));
    }

    #[test]
    fn test_search_query_full_builder() {
        let mut filter = HashMap::new();
        filter.insert("k".to_string(), serde_json::json!(42));
        let query = SearchQuery::builder(vec![0.5, 0.5])
            .top_k(3)
            .min_score(0.5)
            .metadata_filter(filter)
            .build();
        assert_eq!(query.top_k, 3);
        assert_eq!(query.min_score, Some(0.5));
        assert!(query.metadata_filter.is_some());
    }

    // ---- InMemoryVectorStore CRUD tests ----

    #[test]
    fn test_store_new_is_empty() {
        let store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.dimensions().is_none());
    }

    #[test]
    fn test_store_add_and_get() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        let entry = make_entry("e1", vec![1.0, 0.0], "hello");
        store.add(entry);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        let retrieved = store.get("e1").unwrap();
        assert_eq!(retrieved.document, "hello");
    }

    #[test]
    fn test_store_add_batch() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        let entries = vec![
            make_entry("a", vec![1.0, 0.0], "alpha"),
            make_entry("b", vec![0.0, 1.0], "beta"),
            make_entry("c", vec![1.0, 1.0], "gamma"),
        ];
        store.add_batch(entries);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_store_delete() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("e1", vec![1.0], "doc"));
        assert!(store.delete("e1"));
        assert!(store.is_empty());
        assert!(store.get("e1").is_none());
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        assert!(!store.delete("missing"));
    }

    #[test]
    fn test_store_clear() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0], "a"));
        store.add(make_entry("b", vec![2.0], "b"));
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn test_store_dimensions() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        assert_eq!(store.dimensions(), None);
        store.add(make_entry("e1", vec![1.0, 2.0, 3.0], "doc"));
        assert_eq!(store.dimensions(), Some(3));
    }

    #[test]
    fn test_store_overwrite_entry() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("e1", vec![1.0], "old"));
        store.add(make_entry("e1", vec![2.0], "new"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("e1").unwrap().document, "new");
    }

    // ---- InMemoryVectorStore search tests ----

    #[test]
    fn test_search_cosine_returns_most_similar() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0, 0.0], "aligned"));
        store.add(make_entry("b", vec![0.0, 1.0, 0.0], "orthogonal"));
        store.add(make_entry("c", vec![0.9, 0.1, 0.0], "close"));

        let query = SearchQuery::builder(vec![1.0, 0.0, 0.0]).top_k(2).build();
        let results = store.search(&query);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entry.id, "a");
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[1].entry.id, "c");
        assert_eq!(results[1].rank, 1);
    }

    #[test]
    fn test_search_euclidean() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Euclidean);
        store.add(make_entry("near", vec![1.0, 0.0], "near"));
        store.add(make_entry("far", vec![10.0, 10.0], "far"));

        let query = SearchQuery::builder(vec![1.0, 0.0]).top_k(2).build();
        let results = store.search(&query);
        assert_eq!(results[0].entry.id, "near");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_search_dot_product() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::DotProduct);
        store.add(make_entry("big", vec![10.0, 10.0], "big"));
        store.add(make_entry("small", vec![0.1, 0.1], "small"));

        let query = SearchQuery::builder(vec![1.0, 1.0]).top_k(2).build();
        let results = store.search(&query);
        assert_eq!(results[0].entry.id, "big");
    }

    #[test]
    fn test_search_empty_store() {
        let store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        let query = SearchQuery::builder(vec![1.0, 0.0]).top_k(5).build();
        let results = store.search(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_larger_than_store() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("only", vec![1.0, 0.0], "only"));

        let query = SearchQuery::builder(vec![1.0, 0.0]).top_k(100).build();
        let results = store.search(&query);
        assert_eq!(results.len(), 1);
    }

    // ---- Metadata filtering tests ----

    #[test]
    fn test_search_with_metadata_filter() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry_with_metadata(
            "a",
            vec![1.0, 0.0],
            "doc a",
            &[("source", Value::String("wiki".into()))],
        ));
        store.add(make_entry_with_metadata(
            "b",
            vec![0.9, 0.1],
            "doc b",
            &[("source", Value::String("blog".into()))],
        ));
        store.add(make_entry_with_metadata(
            "c",
            vec![0.8, 0.2],
            "doc c",
            &[("source", Value::String("wiki".into()))],
        ));

        let mut filter = HashMap::new();
        filter.insert("source".to_string(), Value::String("wiki".into()));
        let query = SearchQuery::builder(vec![1.0, 0.0])
            .top_k(10)
            .metadata_filter(filter)
            .build();
        let results = store.search(&query);
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|r| { r.entry.metadata.get("source") == Some(&Value::String("wiki".into())) }));
    }

    #[test]
    fn test_search_metadata_filter_no_match() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry_with_metadata(
            "a",
            vec![1.0],
            "doc",
            &[("lang", Value::String("en".into()))],
        ));

        let mut filter = HashMap::new();
        filter.insert("lang".to_string(), Value::String("fr".into()));
        let query = SearchQuery::builder(vec![1.0])
            .metadata_filter(filter)
            .build();
        let results = store.search(&query);
        assert!(results.is_empty());
    }

    // ---- Min score threshold tests ----

    #[test]
    fn test_search_min_score_threshold() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("similar", vec![1.0, 0.0], "similar"));
        store.add(make_entry("different", vec![0.0, 1.0], "different"));

        let query = SearchQuery::builder(vec![1.0, 0.0])
            .top_k(10)
            .min_score(0.9)
            .build();
        let results = store.search(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.id, "similar");
    }

    #[test]
    fn test_search_min_score_filters_all() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![0.0, 1.0], "doc"));

        let query = SearchQuery::builder(vec![1.0, 0.0]).min_score(0.99).build();
        let results = store.search(&query);
        assert!(results.is_empty());
    }

    // ---- SearchResult tests ----

    #[test]
    fn test_search_result_to_json() {
        let result = SearchResult {
            entry: make_entry("e1", vec![1.0], "doc"),
            score: 0.95,
            rank: 0,
        };
        let json = result.to_json();
        assert_eq!(json["score"], 0.95);
        assert_eq!(json["rank"], 0);
        assert_eq!(json["entry"]["id"], "e1");
    }

    // ---- MMR search tests ----

    #[test]
    fn test_mmr_search_empty_store() {
        let store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 5, 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_mmr_search_top_k_zero() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0], "doc"));
        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 0, 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_mmr_search_pure_relevance() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0], "exact match"));
        store.add(make_entry("b", vec![0.9, 0.1], "close"));
        store.add(make_entry("c", vec![0.0, 1.0], "orthogonal"));

        // lambda=1.0 should behave like pure similarity search
        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 3, 1.0);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entry.id, "a");
    }

    #[test]
    fn test_mmr_search_promotes_diversity() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        // Two very similar vectors and one different
        store.add(make_entry("a", vec![1.0, 0.0], "doc a"));
        store.add(make_entry("b", vec![0.99, 0.01], "doc b")); // very similar to a
        store.add(make_entry("c", vec![0.0, 1.0], "doc c")); // very different

        // With low lambda, diversity should be preferred — "c" should appear
        // earlier than with pure relevance.
        let results_diverse = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 3, 0.0);
        let results_relevant = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 3, 1.0);

        let diverse_ids: Vec<&str> = results_diverse
            .iter()
            .map(|r| r.entry.id.as_str())
            .collect();
        let relevant_ids: Vec<&str> = results_relevant
            .iter()
            .map(|r| r.entry.id.as_str())
            .collect();

        // With lambda=0.0, "c" (the diverse pick) should appear in the top 2
        // because it's maximally different from the first selected entry.
        assert!(
            diverse_ids[..2].contains(&"c"),
            "diverse search should include 'c' in top 2, got {:?}",
            diverse_ids
        );

        // With pure relevance (lambda=1.0), "c" should be last since it's
        // orthogonal to the query.
        assert_eq!(
            relevant_ids[2], "c",
            "pure relevance should rank 'c' last, got {:?}",
            relevant_ids
        );
    }

    #[test]
    fn test_mmr_search_single_entry() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("only", vec![1.0, 0.0], "only doc"));
        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 5, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.id, "only");
        assert_eq!(results[0].rank, 0);
    }

    #[test]
    fn test_mmr_search_ranks_are_sequential() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0], "a"));
        store.add(make_entry("b", vec![0.0, 1.0], "b"));
        store.add(make_entry("c", vec![0.5, 0.5], "c"));

        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 3, 0.5);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i);
        }
    }

    // ---- VectorStoreConfig tests ----

    #[test]
    fn test_config_defaults() {
        let config = VectorStoreConfig::builder(SimilarityMetric::Cosine).build();
        assert_eq!(config.metric, SimilarityMetric::Cosine);
        assert!(config.max_entries.is_none());
        assert!(!config.normalize_on_insert);
    }

    #[test]
    fn test_config_builder_all_options() {
        let config = VectorStoreConfig::builder(SimilarityMetric::DotProduct)
            .max_entries(1000)
            .normalize_on_insert(true)
            .build();
        assert_eq!(config.metric, SimilarityMetric::DotProduct);
        assert_eq!(config.max_entries, Some(1000));
        assert!(config.normalize_on_insert);
    }

    // ---- VectorStoreStats tests ----

    #[test]
    fn test_stats_empty_store() {
        let store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        let stats = VectorStoreStats::from_store(&store);
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.dimensions, None);
        assert_eq!(stats.metric_name, "cosine");
        assert_eq!(stats.avg_vector_magnitude, 0.0);
    }

    #[test]
    fn test_stats_with_entries() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Euclidean);
        store.add(make_entry("a", vec![3.0, 4.0], "doc a")); // magnitude = 5.0
        store.add(make_entry("b", vec![0.0, 5.0], "doc b")); // magnitude = 5.0

        let stats = VectorStoreStats::from_store(&store);
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.dimensions, Some(2));
        assert_eq!(stats.metric_name, "euclidean");
        assert!((stats.avg_vector_magnitude - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_to_json() {
        let store = InMemoryVectorStore::new(SimilarityMetric::DotProduct);
        let stats = VectorStoreStats::from_store(&store);
        let json = stats.to_json();
        assert_eq!(json["total_entries"], 0);
        assert_eq!(json["metric_name"], "dot_product");
    }

    // ---- Edge case tests ----

    #[test]
    fn test_zero_vector_search() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0], "doc"));

        // Searching with a zero vector should not panic.
        let query = SearchQuery::builder(vec![0.0, 0.0]).top_k(1).build();
        let results = store.search(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 0.0);
    }

    #[test]
    fn test_zero_vector_entry() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("zero", vec![0.0, 0.0], "zero doc"));

        let query = SearchQuery::builder(vec![1.0, 0.0]).top_k(1).build();
        let results = store.search(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 0.0);
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_search_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("a", vec![1.0, 0.0, 0.0], "doc"));

        let query = SearchQuery::builder(vec![1.0, 0.0]).top_k(1).build();
        store.search(&query);
    }

    #[test]
    fn test_mmr_with_zero_vectors() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry("zero", vec![0.0, 0.0], "zero"));
        store.add(make_entry("nonzero", vec![1.0, 0.0], "nonzero"));

        let results = MaxMarginalRelevance::search(&store, &[1.0, 0.0], 2, 0.5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_combined_metadata_and_min_score() {
        let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
        store.add(make_entry_with_metadata(
            "a",
            vec![1.0, 0.0],
            "doc a",
            &[("cat", Value::String("x".into()))],
        ));
        store.add(make_entry_with_metadata(
            "b",
            vec![0.0, 1.0],
            "doc b",
            &[("cat", Value::String("x".into()))],
        ));

        let mut filter = HashMap::new();
        filter.insert("cat".to_string(), Value::String("x".into()));
        let query = SearchQuery::builder(vec![1.0, 0.0])
            .top_k(10)
            .min_score(0.9)
            .metadata_filter(filter)
            .build();
        let results = store.search(&query);
        // Only "a" should match — it has matching metadata AND high similarity.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.id, "a");
    }
}
