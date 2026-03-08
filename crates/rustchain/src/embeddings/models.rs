//! Embedding model abstractions and utilities.
//!
//! Provides the [`EmbeddingModel`] trait for embedding text, a [`FakeEmbeddingModel`]
//! for deterministic testing, configuration via [`EmbeddingModelConfig`], result types
//! ([`EmbeddingResult`], [`EmbeddingUsage`]), distance utilities ([`EmbeddingDistance`]),
//! normalized embedding wrapper ([`NormalizedEmbedding`]), and a model registry
//! ([`EmbeddingRegistry`]).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use serde_json::Value;

// ---------------------------------------------------------------------------
// EmbeddingModel trait
// ---------------------------------------------------------------------------

/// Trait for embedding models that convert text into dense vectors.
pub trait EmbeddingModel {
    /// Embed a single text string into a vector of f64 values.
    fn embed_text(&self, text: &str) -> Result<Vec<f64>, String>;

    /// Embed a batch of texts. Default implementation calls [`embed_text`] for each.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, String> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }

    /// Return the dimensionality of the embedding vectors this model produces.
    fn dimensions(&self) -> usize;

    /// Return the name of this model.
    fn model_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// FakeEmbeddingModel
// ---------------------------------------------------------------------------

/// A deterministic embedding model for testing.
///
/// Produces repeatable embeddings based on a hash of the input text.
pub struct FakeEmbeddingModel {
    dims: usize,
}

impl FakeEmbeddingModel {
    /// Create a new fake embedding model with the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self { dims: dimensions }
    }
}

impl EmbeddingModel for FakeEmbeddingModel {
    fn embed_text(&self, text: &str) -> Result<Vec<f64>, String> {
        let mut vector = Vec::with_capacity(self.dims);
        for i in 0..self.dims {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Map hash to a value in [-1, 1].
            vector.push(((h % 20000) as f64 / 10000.0) - 1.0);
        }
        Ok(vector)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_name(&self) -> &str {
        "fake-embedding-model"
    }
}

// ---------------------------------------------------------------------------
// EmbeddingModelConfig
// ---------------------------------------------------------------------------

/// Configuration for an embedding model.
///
/// Uses a builder pattern via chained setter methods.
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    /// The model name.
    pub model_name: String,
    /// The dimensionality of the embedding vectors.
    pub dimensions: usize,
    /// Maximum batch size for batch embedding calls.
    pub batch_size: usize,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
    /// Optional timeout in milliseconds for embedding requests.
    pub timeout_ms: Option<u64>,
}

impl EmbeddingModelConfig {
    /// Create a new configuration with the given model name and dimensions.
    ///
    /// Defaults: `batch_size = 100`, `normalize = true`, `timeout_ms = None`.
    pub fn new(model_name: impl Into<String>, dimensions: usize) -> Self {
        Self {
            model_name: model_name.into(),
            dimensions,
            batch_size: 100,
            normalize: true,
            timeout_ms: None,
        }
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to normalize embeddings.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

// ---------------------------------------------------------------------------
// EmbeddingUsage
// ---------------------------------------------------------------------------

/// Token usage information for an embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Total number of tokens consumed.
    pub total_tokens: usize,
}

impl EmbeddingUsage {
    /// Create a new usage record.
    pub fn new(prompt_tokens: usize, total_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            total_tokens,
        }
    }

    /// Serialize this usage to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        })
    }
}

// ---------------------------------------------------------------------------
// EmbeddingResult
// ---------------------------------------------------------------------------

/// The result of an embedding operation, containing the vector and metadata.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector.
    pub vector: Vec<f64>,
    /// The original text that was embedded.
    pub text: String,
    /// The model that produced this embedding.
    pub model: String,
    /// Optional token usage information.
    pub usage: Option<EmbeddingUsage>,
}

impl EmbeddingResult {
    /// Create a new embedding result.
    pub fn new(
        vector: Vec<f64>,
        text: impl Into<String>,
        model: impl Into<String>,
        usage: Option<EmbeddingUsage>,
    ) -> Self {
        Self {
            vector,
            text: text.into(),
            model: model.into(),
            usage,
        }
    }

    /// Return the number of dimensions in the embedding vector.
    pub fn dimension_count(&self) -> usize {
        self.vector.len()
    }

    /// Serialize this result to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut json = serde_json::json!({
            "vector": self.vector,
            "text": self.text,
            "model": self.model,
        });
        if let Some(usage) = &self.usage {
            json["usage"] = usage.to_json();
        }
        json
    }
}

// ---------------------------------------------------------------------------
// EmbeddingDistance
// ---------------------------------------------------------------------------

/// Utility for computing distances and similarities between embedding vectors.
pub struct EmbeddingDistance;

impl EmbeddingDistance {
    /// Compute the cosine similarity between two vectors.
    ///
    /// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector has zero
    /// magnitude.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
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

    /// Compute the Euclidean (L2) distance between two vectors.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }

    /// Compute the dot product of two vectors.
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute the Manhattan (L1) distance between two vectors.
    pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    /// Find the `top_k` most similar candidates to the query vector.
    ///
    /// Returns `(index, cosine_similarity)` pairs sorted by descending similarity.
    pub fn most_similar(query: &[f64], candidates: &[Vec<f64>], top_k: usize) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, v)| (i, Self::cosine_similarity(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

// ---------------------------------------------------------------------------
// NormalizedEmbedding
// ---------------------------------------------------------------------------

/// A wrapper around an embedding vector that guarantees L2 normalization.
///
/// Normalization is performed on construction. For normalized vectors, the dot
/// product is equivalent to cosine similarity.
#[derive(Debug, Clone)]
pub struct NormalizedEmbedding {
    vector: Vec<f64>,
}

impl NormalizedEmbedding {
    /// Create a normalized embedding from a raw vector.
    ///
    /// The input is L2-normalized on creation. A zero-magnitude vector remains
    /// as a zero vector.
    pub fn from_vec(vec: Vec<f64>) -> Self {
        let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let vector = if norm == 0.0 {
            vec
        } else {
            vec.iter().map(|x| x / norm).collect()
        };
        Self { vector }
    }

    /// Return the embedding as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.vector
    }

    /// Return the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Compute the dot product with another normalized embedding.
    ///
    /// For L2-normalized vectors, this is equivalent to cosine similarity.
    pub fn dot_product(&self, other: &NormalizedEmbedding) -> f64 {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "vectors must have the same dimension"
        );
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingRegistry
// ---------------------------------------------------------------------------

/// A registry for managing available embedding models by name.
pub struct EmbeddingRegistry {
    models: HashMap<String, Box<dyn EmbeddingModel>>,
}

impl EmbeddingRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a model under the given name.
    pub fn register(&mut self, name: impl Into<String>, model: Box<dyn EmbeddingModel>) {
        self.models.insert(name.into(), model);
    }

    /// Look up a model by name.
    pub fn get(&self, name: &str) -> Option<&dyn EmbeddingModel> {
        self.models.get(name).map(|b| b.as_ref())
    }

    /// Return the names of all registered models.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Return `true` if no models are registered.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FakeEmbeddingModel tests ----

    #[test]
    fn test_fake_model_dimensions() {
        let model = FakeEmbeddingModel::new(128);
        assert_eq!(model.dimensions(), 128);
    }

    #[test]
    fn test_fake_model_name() {
        let model = FakeEmbeddingModel::new(8);
        assert_eq!(model.model_name(), "fake-embedding-model");
    }

    #[test]
    fn test_fake_model_embed_text_returns_correct_dimensions() {
        let model = FakeEmbeddingModel::new(64);
        let vec = model.embed_text("hello world").unwrap();
        assert_eq!(vec.len(), 64);
    }

    #[test]
    fn test_fake_model_determinism() {
        let model = FakeEmbeddingModel::new(16);
        let v1 = model.embed_text("test input").unwrap();
        let v2 = model.embed_text("test input").unwrap();
        assert_eq!(v1, v2, "same input should produce identical embeddings");
    }

    #[test]
    fn test_fake_model_different_texts_produce_different_embeddings() {
        let model = FakeEmbeddingModel::new(16);
        let v1 = model.embed_text("alpha").unwrap();
        let v2 = model.embed_text("beta").unwrap();
        assert_ne!(
            v1, v2,
            "different inputs should produce different embeddings"
        );
    }

    #[test]
    fn test_fake_model_determinism_across_instances() {
        let m1 = FakeEmbeddingModel::new(8);
        let m2 = FakeEmbeddingModel::new(8);
        let v1 = m1.embed_text("same text").unwrap();
        let v2 = m2.embed_text("same text").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_fake_model_batch_embedding() {
        let model = FakeEmbeddingModel::new(8);
        let texts = vec!["hello", "world", "foo"];
        let results = model.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        for vec in &results {
            assert_eq!(vec.len(), 8);
        }
    }

    #[test]
    fn test_fake_model_batch_consistency_with_single() {
        let model = FakeEmbeddingModel::new(8);
        let batch = model.embed_batch(&["a", "b"]).unwrap();
        let single_a = model.embed_text("a").unwrap();
        let single_b = model.embed_text("b").unwrap();
        assert_eq!(batch[0], single_a);
        assert_eq!(batch[1], single_b);
    }

    #[test]
    fn test_fake_model_empty_batch() {
        let model = FakeEmbeddingModel::new(8);
        let results = model.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fake_model_values_in_range() {
        let model = FakeEmbeddingModel::new(32);
        let vec = model.embed_text("range test").unwrap();
        for &v in &vec {
            assert!(v >= -1.0 && v <= 1.0, "value {} out of [-1, 1] range", v);
        }
    }

    #[test]
    fn test_fake_model_single_dimension() {
        let model = FakeEmbeddingModel::new(1);
        let vec = model.embed_text("one dim").unwrap();
        assert_eq!(vec.len(), 1);
    }

    // ---- EmbeddingModelConfig tests ----

    #[test]
    fn test_config_defaults() {
        let cfg = EmbeddingModelConfig::new("test-model", 768);
        assert_eq!(cfg.model_name, "test-model");
        assert_eq!(cfg.dimensions, 768);
        assert_eq!(cfg.batch_size, 100);
        assert!(cfg.normalize);
        assert!(cfg.timeout_ms.is_none());
    }

    #[test]
    fn test_config_builder() {
        let cfg = EmbeddingModelConfig::new("my-model", 512)
            .with_batch_size(50)
            .with_normalize(false)
            .with_timeout_ms(5000);
        assert_eq!(cfg.model_name, "my-model");
        assert_eq!(cfg.dimensions, 512);
        assert_eq!(cfg.batch_size, 50);
        assert!(!cfg.normalize);
        assert_eq!(cfg.timeout_ms, Some(5000));
    }

    #[test]
    fn test_config_builder_chaining() {
        let cfg = EmbeddingModelConfig::new("m", 256)
            .with_batch_size(10)
            .with_normalize(true)
            .with_timeout_ms(1000);
        assert_eq!(cfg.batch_size, 10);
        assert!(cfg.normalize);
        assert_eq!(cfg.timeout_ms, Some(1000));
    }

    // ---- EmbeddingResult tests ----

    #[test]
    fn test_embedding_result_creation() {
        let result = EmbeddingResult::new(vec![1.0, 2.0, 3.0], "hello", "model-v1", None);
        assert_eq!(result.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.text, "hello");
        assert_eq!(result.model, "model-v1");
        assert!(result.usage.is_none());
    }

    #[test]
    fn test_embedding_result_dimension_count() {
        let result = EmbeddingResult::new(vec![0.1; 768], "text", "model", None);
        assert_eq!(result.dimension_count(), 768);
    }

    #[test]
    fn test_embedding_result_with_usage() {
        let usage = EmbeddingUsage::new(10, 10);
        let result = EmbeddingResult::new(vec![1.0], "hi", "m", Some(usage));
        assert!(result.usage.is_some());
        assert_eq!(result.usage.as_ref().unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_embedding_result_to_json() {
        let result = EmbeddingResult::new(vec![1.0, 2.0], "text", "model", None);
        let json = result.to_json();
        assert_eq!(json["text"], "text");
        assert_eq!(json["model"], "model");
        assert_eq!(json["vector"][0], 1.0);
        assert_eq!(json["vector"][1], 2.0);
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn test_embedding_result_to_json_with_usage() {
        let usage = EmbeddingUsage::new(5, 5);
        let result = EmbeddingResult::new(vec![1.0], "t", "m", Some(usage));
        let json = result.to_json();
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    // ---- EmbeddingUsage tests ----

    #[test]
    fn test_usage_to_json() {
        let usage = EmbeddingUsage::new(42, 100);
        let json = usage.to_json();
        assert_eq!(json["prompt_tokens"], 42);
        assert_eq!(json["total_tokens"], 100);
    }

    // ---- EmbeddingDistance tests ----

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = EmbeddingDistance::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = EmbeddingDistance::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = EmbeddingDistance::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let other = vec![1.0, 2.0, 3.0];
        let sim = EmbeddingDistance::cosine_similarity(&zero, &other);
        assert_eq!(sim, 0.0);
        assert!(!sim.is_nan());
    }

    #[test]
    fn test_euclidean_distance_345() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = EmbeddingDistance::euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = EmbeddingDistance::euclidean_distance(&v, &v);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = EmbeddingDistance::dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dp = EmbeddingDistance::dot_product(&a, &b);
        assert!(dp.abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let d = EmbeddingDistance::manhattan_distance(&a, &b);
        assert!((d - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance_identical() {
        let v = vec![1.0, 2.0];
        let d = EmbeddingDistance::manhattan_distance(&v, &v);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_most_similar_basic() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0], // identical
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![0.9, 0.1, 0.0], // close
        ];
        let results = EmbeddingDistance::most_similar(&query, &candidates, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].0, 0,
            "most similar should be the identical vector"
        );
        assert_eq!(
            results[1].0, 2,
            "second most similar should be the close vector"
        );
    }

    #[test]
    fn test_most_similar_top_k_larger_than_candidates() {
        let query = vec![1.0, 0.0];
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = EmbeddingDistance::most_similar(&query, &candidates, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_most_similar_top_k_zero() {
        let query = vec![1.0, 0.0];
        let candidates = vec![vec![1.0, 0.0]];
        let results = EmbeddingDistance::most_similar(&query, &candidates, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_most_similar_ranking_order() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![0.0, 1.0],   // orthogonal (sim ~ 0)
            vec![0.5, 0.5],   // moderate
            vec![0.99, 0.01], // very close
        ];
        let results = EmbeddingDistance::most_similar(&query, &candidates, 3);
        assert_eq!(results[0].0, 2); // most similar
        assert_eq!(results[1].0, 1); // moderate
        assert_eq!(results[2].0, 0); // least similar
    }

    // ---- NormalizedEmbedding tests ----

    #[test]
    fn test_normalized_embedding_unit_length() {
        let ne = NormalizedEmbedding::from_vec(vec![3.0, 4.0]);
        let norm: f64 = ne.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "should be unit length");
    }

    #[test]
    fn test_normalized_embedding_values() {
        let ne = NormalizedEmbedding::from_vec(vec![3.0, 4.0]);
        let s = ne.as_slice();
        assert!((s[0] - 0.6).abs() < 1e-10);
        assert!((s[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_embedding_dimensions() {
        let ne = NormalizedEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(ne.dimensions(), 3);
    }

    #[test]
    fn test_normalized_embedding_zero_vector() {
        let ne = NormalizedEmbedding::from_vec(vec![0.0, 0.0, 0.0]);
        assert_eq!(ne.as_slice(), &[0.0, 0.0, 0.0]);
        assert_eq!(ne.dimensions(), 3);
    }

    #[test]
    fn test_normalized_embedding_dot_product_equals_cosine() {
        let a = NormalizedEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let b = NormalizedEmbedding::from_vec(vec![4.0, 5.0, 6.0]);
        let dp = a.dot_product(&b);
        // Should equal cosine similarity of the original vectors.
        let cosine = EmbeddingDistance::cosine_similarity(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((dp - cosine).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_embedding_dot_product_identical() {
        let a = NormalizedEmbedding::from_vec(vec![1.0, 0.0]);
        let dp = a.dot_product(&a);
        assert!((dp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_embedding_dot_product_orthogonal() {
        let a = NormalizedEmbedding::from_vec(vec![1.0, 0.0]);
        let b = NormalizedEmbedding::from_vec(vec![0.0, 1.0]);
        let dp = a.dot_product(&b);
        assert!(dp.abs() < 1e-10);
    }

    #[test]
    fn test_normalized_embedding_single_dimension() {
        let ne = NormalizedEmbedding::from_vec(vec![5.0]);
        assert!((ne.as_slice()[0] - 1.0).abs() < 1e-10);
    }

    // ---- EmbeddingRegistry tests ----

    #[test]
    fn test_registry_new_is_empty() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = EmbeddingRegistry::new();
        reg.register("fake", Box::new(FakeEmbeddingModel::new(8)));
        assert_eq!(reg.len(), 1);
        let model = reg.get("fake").unwrap();
        assert_eq!(model.dimensions(), 8);
        assert_eq!(model.model_name(), "fake-embedding-model");
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_model_names() {
        let mut reg = EmbeddingRegistry::new();
        reg.register("model-a", Box::new(FakeEmbeddingModel::new(8)));
        reg.register("model-b", Box::new(FakeEmbeddingModel::new(16)));
        let mut names = reg.model_names();
        names.sort();
        assert_eq!(names, vec!["model-a", "model-b"]);
    }

    #[test]
    fn test_registry_overwrite() {
        let mut reg = EmbeddingRegistry::new();
        reg.register("m", Box::new(FakeEmbeddingModel::new(8)));
        reg.register("m", Box::new(FakeEmbeddingModel::new(16)));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("m").unwrap().dimensions(), 16);
    }

    #[test]
    fn test_registry_multiple_models() {
        let mut reg = EmbeddingRegistry::new();
        reg.register("a", Box::new(FakeEmbeddingModel::new(4)));
        reg.register("b", Box::new(FakeEmbeddingModel::new(8)));
        reg.register("c", Box::new(FakeEmbeddingModel::new(16)));
        assert_eq!(reg.len(), 3);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_embed_through_get() {
        let mut reg = EmbeddingRegistry::new();
        reg.register("test", Box::new(FakeEmbeddingModel::new(4)));
        let model = reg.get("test").unwrap();
        let vec = model.embed_text("hello").unwrap();
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_registry_default() {
        let reg = EmbeddingRegistry::default();
        assert!(reg.is_empty());
    }

    // ---- Edge case tests ----

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_cosine_similarity_dimension_mismatch() {
        EmbeddingDistance::cosine_similarity(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_euclidean_dimension_mismatch() {
        EmbeddingDistance::euclidean_distance(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_dot_product_dimension_mismatch() {
        EmbeddingDistance::dot_product(&[1.0, 2.0, 3.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_manhattan_dimension_mismatch() {
        EmbeddingDistance::manhattan_distance(&[1.0], &[1.0, 2.0]);
    }
}
