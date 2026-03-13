//! Embedding similarity and distance utilities.
//!
//! Builds on top of the low-level distance functions in [`super::distance`] to
//! provide higher-level abstractions for comparing, searching, clustering,
//! normalizing, and projecting embedding vectors.
//!
//! ## Key types
//!
//! - [`SimilarityResult`] — a scored result with rank and optional metadata.
//! - [`EmbeddingSimilarity`] — calculator supporting all distance metrics.
//! - [`PairwiseSimilarityMatrix`] — NxM similarity matrix between two sets of embeddings.
//! - [`KNearestNeighbors`] — top-k search over an embedding collection.
//! - [`ClusterAssignment`] — single-iteration centroid-based clustering.
//! - [`EmbeddingNormalizer`] — L2 normalize, mean-center, unit-variance scaling.
//! - [`DimensionalityReducer`] — random projection for dimensionality reduction.
//! - [`SimilarityThreshold`] — filter results by minimum similarity score.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::distance::{self, DistanceMetric};

// ---------------------------------------------------------------------------
// SimilarityResult
// ---------------------------------------------------------------------------

/// A similarity search result carrying a score, rank, and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// The similarity score (higher = more similar).
    pub score: f32,
    /// The rank of this result (0-based).
    pub rank: usize,
    /// The index into the original embedding collection.
    pub index: usize,
    /// Optional string metadata attached to this result.
    pub metadata: HashMap<String, String>,
}

impl SimilarityResult {
    /// Create a new result with no metadata.
    pub fn new(score: f32, rank: usize, index: usize) -> Self {
        Self {
            score,
            rank,
            index,
            metadata: HashMap::new(),
        }
    }

    /// Create a new result with metadata.
    pub fn with_metadata(
        score: f32,
        rank: usize,
        index: usize,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            score,
            rank,
            index,
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingSimilarity
// ---------------------------------------------------------------------------

/// Calculator that computes similarity between embeddings using a configurable
/// distance metric.
///
/// Wraps [`distance::compute_similarity`] and [`distance::compute_distance`]
/// and returns structured [`SimilarityResult`] values.
pub struct EmbeddingSimilarity {
    metric: DistanceMetric,
}

impl EmbeddingSimilarity {
    /// Create a new calculator with the given metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric }
    }

    /// Return the configured metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Compute the similarity between two vectors (higher = more similar).
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        distance::compute_similarity(self.metric, a, b)
    }

    /// Compute the distance between two vectors (lower = more similar).
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        distance::compute_distance(self.metric, a, b)
    }

    /// Score a query against a list of candidates and return ranked
    /// [`SimilarityResult`] values (descending by score).
    pub fn score_all(&self, query: &[f32], candidates: &[Vec<f32>]) -> Vec<SimilarityResult> {
        let mut results: Vec<SimilarityResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| SimilarityResult::new(self.similarity(query, c), 0, i))
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, r) in results.iter_mut().enumerate() {
            r.rank = rank;
        }
        results
    }
}

// ---------------------------------------------------------------------------
// PairwiseSimilarityMatrix
// ---------------------------------------------------------------------------

/// An NxM similarity matrix between two sets of embeddings.
///
/// Row `i`, column `j` contains the similarity between `embeddings_a[i]` and
/// `embeddings_b[j]`.
pub struct PairwiseSimilarityMatrix {
    /// The similarity values stored row-major.
    pub matrix: Vec<Vec<f32>>,
    /// Number of rows (from set A).
    pub rows: usize,
    /// Number of columns (from set B).
    pub cols: usize,
}

impl PairwiseSimilarityMatrix {
    /// Build a pairwise similarity matrix between two sets of embeddings.
    pub fn compute(
        embeddings_a: &[Vec<f32>],
        embeddings_b: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Self {
        let rows = embeddings_a.len();
        let cols = embeddings_b.len();
        let matrix: Vec<Vec<f32>> = embeddings_a
            .iter()
            .map(|a| {
                embeddings_b
                    .iter()
                    .map(|b| distance::compute_similarity(metric, a, b))
                    .collect()
            })
            .collect();
        Self { matrix, rows, cols }
    }

    /// Build a symmetric pairwise similarity matrix for a single set.
    pub fn compute_symmetric(embeddings: &[Vec<f32>], metric: DistanceMetric) -> Self {
        let n = embeddings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            matrix[i][i] = distance::compute_similarity(metric, &embeddings[i], &embeddings[i]);
            for j in (i + 1)..n {
                let sim = distance::compute_similarity(metric, &embeddings[i], &embeddings[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }
        Self {
            matrix,
            rows: n,
            cols: n,
        }
    }

    /// Get the similarity between row `i` and column `j`.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.matrix[i][j]
    }

    /// Return the index of the most similar item in set B for each item in set A.
    pub fn most_similar_per_row(&self) -> Vec<(usize, f32)> {
        self.matrix
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, &score)| (idx, score))
                    .unwrap_or((0, 0.0))
            })
            .collect()
    }

    /// Return the index of the most similar item in set A for each item in set B.
    pub fn most_similar_per_col(&self) -> Vec<(usize, f32)> {
        (0..self.cols)
            .map(|j| {
                self.matrix
                    .iter()
                    .enumerate()
                    .map(|(i, row)| (i, row[j]))
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, 0.0))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// KNearestNeighbors
// ---------------------------------------------------------------------------

/// Finds the top-k most similar embeddings to a query from a stored collection.
pub struct KNearestNeighbors {
    embeddings: Vec<Vec<f32>>,
    metric: DistanceMetric,
}

impl KNearestNeighbors {
    /// Create a new KNN index over the given embeddings.
    pub fn new(embeddings: Vec<Vec<f32>>, metric: DistanceMetric) -> Self {
        Self { embeddings, metric }
    }

    /// Return the number of stored embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Find the `k` nearest neighbors of `query`.
    ///
    /// Returns [`SimilarityResult`] values sorted by descending similarity.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        let calc = EmbeddingSimilarity::new(self.metric);
        let mut results = calc.score_all(query, &self.embeddings);
        results.truncate(k);
        results
    }

    /// Find the `k` nearest neighbors, attaching metadata from the provided map.
    pub fn search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
        metadata: &[HashMap<String, String>],
    ) -> Vec<SimilarityResult> {
        let mut results = self.search(query, k);
        for r in &mut results {
            if let Some(m) = metadata.get(r.index) {
                r.metadata = m.clone();
            }
        }
        results
    }

    /// Add an embedding to the index.
    pub fn add(&mut self, embedding: Vec<f32>) {
        self.embeddings.push(embedding);
    }

    /// Add multiple embeddings to the index.
    pub fn add_batch(&mut self, embeddings: Vec<Vec<f32>>) {
        self.embeddings.extend(embeddings);
    }
}

// ---------------------------------------------------------------------------
// ClusterAssignment
// ---------------------------------------------------------------------------

/// Simple centroid-based clustering (single k-means iteration).
///
/// Assigns each embedding to the nearest centroid, then recomputes centroids.
pub struct ClusterAssignment {
    /// The cluster label for each embedding (index-aligned).
    pub assignments: Vec<usize>,
    /// The centroids after the assignment step.
    pub centroids: Vec<Vec<f32>>,
    /// Number of clusters.
    pub k: usize,
}

impl ClusterAssignment {
    /// Run a single k-means-style iteration.
    ///
    /// `initial_centroids` provides the starting centroids; each embedding in
    /// `embeddings` is assigned to the nearest centroid, then centroids are
    /// recomputed as the mean of their assigned members.
    pub fn assign(
        embeddings: &[Vec<f32>],
        initial_centroids: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Self {
        let k = initial_centroids.len();
        assert!(k > 0, "must have at least one centroid");
        assert!(
            !embeddings.is_empty(),
            "must have at least one embedding to cluster"
        );

        // Assign each embedding to the nearest centroid.
        let assignments: Vec<usize> = embeddings
            .iter()
            .map(|emb| {
                initial_centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, distance::compute_similarity(metric, emb, c)))
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(ci, _)| ci)
                    .unwrap()
            })
            .collect();

        // Recompute centroids.
        let dim = embeddings[0].len();
        let mut centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &label) in assignments.iter().enumerate() {
            counts[label] += 1;
            for (d, &val) in embeddings[i].iter().enumerate() {
                centroids[label][d] += val;
            }
        }

        for (ci, centroid) in centroids.iter_mut().enumerate() {
            if counts[ci] > 0 {
                let n = counts[ci] as f32;
                for val in centroid.iter_mut() {
                    *val /= n;
                }
            }
        }

        Self {
            assignments,
            centroids,
            k,
        }
    }

    /// Return the indices of embeddings assigned to a given cluster.
    pub fn members(&self, cluster: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the sizes of each cluster.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.k];
        for &c in &self.assignments {
            sizes[c] += 1;
        }
        sizes
    }
}

// ---------------------------------------------------------------------------
// EmbeddingNormalizer
// ---------------------------------------------------------------------------

/// Utilities for normalizing embedding vectors.
///
/// Supports L2 normalization, mean-centering, and unit-variance scaling.
pub struct EmbeddingNormalizer;

impl EmbeddingNormalizer {
    /// L2-normalize a single vector (unit length). Returns zero vector if input
    /// has zero magnitude.
    pub fn l2_normalize(vector: &[f32]) -> Vec<f32> {
        distance::normalize(vector)
    }

    /// L2-normalize each vector in a batch.
    pub fn l2_normalize_batch(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        vectors.iter().map(|v| Self::l2_normalize(v)).collect()
    }

    /// Mean-center a set of vectors (subtract the mean vector from each).
    pub fn mean_center(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if vectors.is_empty() {
            return vec![];
        }
        let mean = distance::mean_vector(vectors);
        vectors
            .iter()
            .map(|v| v.iter().zip(mean.iter()).map(|(a, b)| a - b).collect())
            .collect()
    }

    /// Scale each dimension to unit variance across a set of vectors.
    ///
    /// Dimensions with zero variance are left unchanged.
    pub fn unit_variance(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if vectors.is_empty() {
            return vec![];
        }
        let dim = vectors[0].len();
        let n = vectors.len() as f32;
        let mean = distance::mean_vector(vectors);

        // Compute per-dimension standard deviation.
        let mut variance = vec![0.0f32; dim];
        for v in vectors {
            for (d, &val) in v.iter().enumerate() {
                let diff = val - mean[d];
                variance[d] += diff * diff;
            }
        }
        let stddev: Vec<f32> = variance.iter().map(|v| (v / n).sqrt()).collect();

        vectors
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(d, &val)| {
                        if stddev[d] == 0.0 {
                            val
                        } else {
                            val / stddev[d]
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply mean-centering followed by L2-normalization to each vector.
    pub fn normalize_full(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let centered = Self::mean_center(vectors);
        Self::l2_normalize_batch(&centered)
    }
}

// ---------------------------------------------------------------------------
// DimensionalityReducer
// ---------------------------------------------------------------------------

/// Reduces embedding dimensionality via random projection.
///
/// Uses a seeded deterministic projection matrix (simple hash-based) to map
/// from `input_dim` to `output_dim`. This is based on the Johnson-Lindenstrauss
/// lemma — random projections approximately preserve distances.
pub struct DimensionalityReducer {
    /// The projection matrix, shape `[output_dim][input_dim]`.
    projection: Vec<Vec<f32>>,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Input dimensionality.
    pub input_dim: usize,
}

impl DimensionalityReducer {
    /// Create a new reducer projecting from `input_dim` to `output_dim`.
    ///
    /// Uses `seed` to deterministically generate the random projection matrix
    /// via a simple hash-based PRNG.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        assert!(output_dim > 0, "output_dim must be positive");
        assert!(input_dim > 0, "input_dim must be positive");

        let scale = 1.0 / (output_dim as f32).sqrt();
        let mut projection = Vec::with_capacity(output_dim);
        let mut state = seed;

        for _ in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                // Simple xorshift64 PRNG.
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to {-1, +1} scaled.
                let val = if state % 2 == 0 { scale } else { -scale };
                row.push(val);
            }
            projection.push(row);
        }

        Self {
            projection,
            output_dim,
            input_dim,
        }
    }

    /// Project a single vector to the lower-dimensional space.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.input_dim`.
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(
            vector.len(),
            self.input_dim,
            "vector dimension mismatch: expected {}, got {}",
            self.input_dim,
            vector.len()
        );
        self.projection
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Project a batch of vectors.
    pub fn project_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        vectors.iter().map(|v| self.project(v)).collect()
    }
}

// ---------------------------------------------------------------------------
// SimilarityThreshold
// ---------------------------------------------------------------------------

/// Filters similarity results by a minimum score.
pub struct SimilarityThreshold {
    /// Minimum similarity score to include.
    pub min_score: f32,
}

impl SimilarityThreshold {
    /// Create a new threshold filter.
    pub fn new(min_score: f32) -> Self {
        Self { min_score }
    }

    /// Filter results, keeping only those at or above the threshold.
    pub fn filter(&self, results: Vec<SimilarityResult>) -> Vec<SimilarityResult> {
        results
            .into_iter()
            .filter(|r| r.score >= self.min_score)
            .collect()
    }

    /// Filter and re-rank (re-assign contiguous rank values starting from 0).
    pub fn filter_and_rerank(&self, results: Vec<SimilarityResult>) -> Vec<SimilarityResult> {
        let mut filtered = self.filter(results);
        for (rank, r) in filtered.iter_mut().enumerate() {
            r.rank = rank;
        }
        filtered
    }

    /// Convenience: compute similarities, filter by threshold, return ranked
    /// results.
    pub fn search(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Vec<SimilarityResult> {
        let calc = EmbeddingSimilarity::new(metric);
        let results = calc.score_all(query, candidates);
        self.filter_and_rerank(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- SimilarityResult ---

    #[test]
    fn test_similarity_result_new() {
        let r = SimilarityResult::new(0.95, 0, 3);
        assert!(approx_eq(r.score, 0.95));
        assert_eq!(r.rank, 0);
        assert_eq!(r.index, 3);
        assert!(r.metadata.is_empty());
    }

    #[test]
    fn test_similarity_result_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("label".to_string(), "cat".to_string());
        let r = SimilarityResult::with_metadata(0.8, 1, 5, meta);
        assert_eq!(r.metadata.get("label").unwrap(), "cat");
        assert_eq!(r.index, 5);
    }

    #[test]
    fn test_similarity_result_serialization() {
        let r = SimilarityResult::new(0.5, 2, 7);
        let json = serde_json::to_string(&r).unwrap();
        let decoded: SimilarityResult = serde_json::from_str(&json).unwrap();
        assert!(approx_eq(decoded.score, 0.5));
        assert_eq!(decoded.rank, 2);
        assert_eq!(decoded.index, 7);
    }

    // --- EmbeddingSimilarity ---

    #[test]
    fn test_embedding_similarity_cosine_identical() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let v = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(calc.similarity(&v, &v), 1.0));
    }

    #[test]
    fn test_embedding_similarity_cosine_orthogonal() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(calc.similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_embedding_similarity_euclidean() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Euclidean);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        // similarity = 1 / (1 + 5) = 1/6
        assert!(approx_eq(calc.similarity(&a, &b), 1.0 / 6.0));
    }

    #[test]
    fn test_embedding_similarity_dot_product() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::DotProduct);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(calc.similarity(&a, &b), 32.0));
    }

    #[test]
    fn test_embedding_similarity_manhattan() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Manhattan);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // distance = 7, similarity = 1 / (1 + 7) = 0.125
        assert!(approx_eq(calc.similarity(&a, &b), 0.125));
    }

    #[test]
    fn test_embedding_similarity_chebyshev() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Chebyshev);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // distance = 4, similarity = 1 / (1 + 4) = 0.2
        assert!(approx_eq(calc.similarity(&a, &b), 0.2));
    }

    #[test]
    fn test_embedding_similarity_hamming() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Hamming);
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        // distance = 0.5, similarity = 0.5
        assert!(approx_eq(calc.similarity(&a, &b), 0.5));
    }

    #[test]
    fn test_embedding_similarity_distance() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(calc.distance(&a, &b), 1.0));
    }

    #[test]
    fn test_embedding_similarity_metric_getter() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Manhattan);
        assert_eq!(calc.metric(), DistanceMetric::Manhattan);
    }

    #[test]
    fn test_score_all_ordering() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![1.0, 0.0, 0.0], // identical
            vec![0.7, 0.7, 0.0], // partial
        ];
        let results = calc.score_all(&query, &candidates);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 1); // identical is first
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[1].rank, 1);
        assert_eq!(results[2].rank, 2);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_score_all_empty_candidates() {
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let query = vec![1.0, 0.0];
        let candidates: Vec<Vec<f32>> = vec![];
        let results = calc.score_all(&query, &candidates);
        assert!(results.is_empty());
    }

    // --- PairwiseSimilarityMatrix ---

    #[test]
    fn test_pairwise_matrix_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let m = PairwiseSimilarityMatrix::compute(&a, &b, DistanceMetric::Cosine);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert!(approx_eq(m.get(0, 0), 1.0));
        assert!(approx_eq(m.get(0, 1), 0.0));
        assert!(approx_eq(m.get(1, 0), 0.0));
        assert!(approx_eq(m.get(1, 1), 1.0));
    }

    #[test]
    fn test_pairwise_matrix_asymmetric_shape() {
        let a = vec![vec![1.0, 0.0, 0.0]];
        let b = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let m = PairwiseSimilarityMatrix::compute(&a, &b, DistanceMetric::Cosine);
        assert_eq!(m.rows, 1);
        assert_eq!(m.cols, 3);
        assert!(approx_eq(m.get(0, 0), 1.0));
        assert!(approx_eq(m.get(0, 1), 0.0));
        assert!(approx_eq(m.get(0, 2), 0.0));
    }

    #[test]
    fn test_pairwise_matrix_symmetric() {
        let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let m = PairwiseSimilarityMatrix::compute_symmetric(&vecs, DistanceMetric::Cosine);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        // Symmetric: m[i][j] == m[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(m.get(i, j), m.get(j, i)));
            }
        }
        // Diagonal should be 1.0 for cosine
        assert!(approx_eq(m.get(0, 0), 1.0));
        assert!(approx_eq(m.get(1, 1), 1.0));
    }

    #[test]
    fn test_most_similar_per_row() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let m = PairwiseSimilarityMatrix::compute(&a, &b, DistanceMetric::Cosine);
        let best = m.most_similar_per_row();
        assert_eq!(best[0].0, 1); // [1,0] most similar to b[1]=[1,0]
        assert_eq!(best[1].0, 0); // [0,1] most similar to b[0]=[0,1]
    }

    #[test]
    fn test_most_similar_per_col() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let m = PairwiseSimilarityMatrix::compute(&a, &b, DistanceMetric::Cosine);
        let best = m.most_similar_per_col();
        assert_eq!(best[0].0, 1); // b[0]=[0,1] most similar to a[1]=[0,1]
        assert_eq!(best[1].0, 0); // b[1]=[1,0] most similar to a[0]=[1,0]
    }

    // --- KNearestNeighbors ---

    #[test]
    fn test_knn_search_basic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0],
        ];
        let knn = KNearestNeighbors::new(embeddings, DistanceMetric::Cosine);
        let results = knn.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0); // identical
        assert_eq!(results[1].index, 3); // close
    }

    #[test]
    fn test_knn_search_k_larger_than_collection() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let knn = KNearestNeighbors::new(embeddings, DistanceMetric::Cosine);
        let results = knn.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_knn_len_and_empty() {
        let knn = KNearestNeighbors::new(vec![], DistanceMetric::Cosine);
        assert!(knn.is_empty());
        assert_eq!(knn.len(), 0);

        let knn2 = KNearestNeighbors::new(vec![vec![1.0]], DistanceMetric::Cosine);
        assert!(!knn2.is_empty());
        assert_eq!(knn2.len(), 1);
    }

    #[test]
    fn test_knn_add() {
        let mut knn = KNearestNeighbors::new(vec![vec![1.0, 0.0]], DistanceMetric::Cosine);
        assert_eq!(knn.len(), 1);
        knn.add(vec![0.0, 1.0]);
        assert_eq!(knn.len(), 2);
    }

    #[test]
    fn test_knn_add_batch() {
        let mut knn = KNearestNeighbors::new(vec![], DistanceMetric::Cosine);
        knn.add_batch(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(knn.len(), 3);
    }

    #[test]
    fn test_knn_search_with_metadata() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
        let metadata = vec![
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), "a".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), "b".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("id".to_string(), "c".to_string());
                m
            },
        ];
        let knn = KNearestNeighbors::new(embeddings, DistanceMetric::Cosine);
        let results = knn.search_with_metadata(&[1.0, 0.0], 2, &metadata);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].metadata.get("id").unwrap(), "a");
    }

    #[test]
    fn test_knn_search_ranks_are_sequential() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.7, 0.7],
            vec![0.0, 1.0],
            vec![0.9, 0.1],
        ];
        let knn = KNearestNeighbors::new(embeddings, DistanceMetric::Cosine);
        let results = knn.search(&[1.0, 0.0], 4);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i);
        }
    }

    // --- ClusterAssignment ---

    #[test]
    fn test_cluster_assignment_two_clusters() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);
        assert_eq!(ca.k, 2);
        assert_eq!(ca.assignments[0], 0);
        assert_eq!(ca.assignments[1], 0);
        assert_eq!(ca.assignments[2], 1);
        assert_eq!(ca.assignments[3], 1);
    }

    #[test]
    fn test_cluster_assignment_members() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
        ];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);
        let members_0 = ca.members(0);
        let members_1 = ca.members(1);
        assert!(members_0.contains(&0));
        assert!(members_0.contains(&1));
        assert!(members_1.contains(&2));
    }

    #[test]
    fn test_cluster_sizes() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.0, 1.0],
        ];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);
        let sizes = ca.cluster_sizes();
        assert_eq!(sizes[0], 3);
        assert_eq!(sizes[1], 1);
    }

    #[test]
    fn test_cluster_centroids_recomputed() {
        let embeddings = vec![vec![2.0, 0.0], vec![4.0, 0.0]];
        let centroids = vec![vec![1.0, 0.0]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);
        // Both assigned to cluster 0, centroid should be mean = [3.0, 0.0]
        assert!(approx_eq(ca.centroids[0][0], 3.0));
        assert!(approx_eq(ca.centroids[0][1], 0.0));
    }

    #[test]
    fn test_cluster_single_centroid() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let centroids = vec![vec![0.5, 0.5]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);
        // All should be assigned to cluster 0
        assert!(ca.assignments.iter().all(|&c| c == 0));
        assert_eq!(ca.cluster_sizes(), vec![3]);
    }

    // --- EmbeddingNormalizer ---

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = EmbeddingNormalizer::l2_normalize(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(mag, 1.0));
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = EmbeddingNormalizer::l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_normalize_batch() {
        let vecs = vec![vec![3.0, 4.0], vec![0.0, 5.0]];
        let normed = EmbeddingNormalizer::l2_normalize_batch(&vecs);
        for n in &normed {
            let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(approx_eq(mag, 1.0));
        }
    }

    #[test]
    fn test_mean_center() {
        let vecs = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let centered = EmbeddingNormalizer::mean_center(&vecs);
        // Mean = [3, 5], so centered = [[-1, -1], [1, 1]]
        assert!(approx_eq(centered[0][0], -1.0));
        assert!(approx_eq(centered[0][1], -1.0));
        assert!(approx_eq(centered[1][0], 1.0));
        assert!(approx_eq(centered[1][1], 1.0));
    }

    #[test]
    fn test_mean_center_empty() {
        let centered = EmbeddingNormalizer::mean_center(&[]);
        assert!(centered.is_empty());
    }

    #[test]
    fn test_unit_variance() {
        let vecs = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];
        let scaled = EmbeddingNormalizer::unit_variance(&vecs);
        assert_eq!(scaled.len(), 3);
        // All vectors should be scaled; verify they exist and have correct dim
        for v in &scaled {
            assert_eq!(v.len(), 2);
        }
    }

    #[test]
    fn test_unit_variance_empty() {
        let scaled = EmbeddingNormalizer::unit_variance(&[]);
        assert!(scaled.is_empty());
    }

    #[test]
    fn test_normalize_full() {
        let vecs = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let result = EmbeddingNormalizer::normalize_full(&vecs);
        assert_eq!(result.len(), 2);
        // Each should be unit length
        for v in &result {
            let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(approx_eq(mag, 1.0));
        }
    }

    // --- DimensionalityReducer ---

    #[test]
    fn test_reducer_output_dimensions() {
        let reducer = DimensionalityReducer::new(100, 10, 42);
        assert_eq!(reducer.input_dim, 100);
        assert_eq!(reducer.output_dim, 10);
        let v = vec![1.0; 100];
        let projected = reducer.project(&v);
        assert_eq!(projected.len(), 10);
    }

    #[test]
    fn test_reducer_deterministic() {
        let r1 = DimensionalityReducer::new(50, 5, 123);
        let r2 = DimensionalityReducer::new(50, 5, 123);
        let v = vec![1.0; 50];
        let p1 = r1.project(&v);
        let p2 = r2.project(&v);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_reducer_different_seeds() {
        let r1 = DimensionalityReducer::new(50, 5, 100);
        let r2 = DimensionalityReducer::new(50, 5, 200);
        let v = vec![1.0; 50];
        let p1 = r1.project(&v);
        let p2 = r2.project(&v);
        // With different seeds, projections should differ
        let any_diff = p1.iter().zip(p2.iter()).any(|(a, b)| !approx_eq(*a, *b));
        assert!(any_diff, "different seeds should produce different projections");
    }

    #[test]
    fn test_reducer_batch() {
        let reducer = DimensionalityReducer::new(20, 3, 42);
        let vecs = vec![vec![1.0; 20], vec![2.0; 20], vec![0.5; 20]];
        let projected = reducer.project_batch(&vecs);
        assert_eq!(projected.len(), 3);
        for p in &projected {
            assert_eq!(p.len(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn test_reducer_dimension_mismatch() {
        let reducer = DimensionalityReducer::new(10, 3, 42);
        let v = vec![1.0; 5]; // wrong dim
        reducer.project(&v);
    }

    #[test]
    fn test_reducer_preserves_relative_similarity() {
        // Vectors that are similar in high dim should remain relatively similar
        let reducer = DimensionalityReducer::new(100, 20, 42);
        let a = vec![1.0; 100];
        let mut b = vec![1.0; 100];
        b[0] = 0.9; // slightly different from a
        let c = vec![-1.0; 100]; // very different from a

        let pa = reducer.project(&a);
        let pb = reducer.project(&b);
        let pc = reducer.project(&c);

        let sim_ab = distance::cosine_similarity(&pa, &pb);
        let sim_ac = distance::cosine_similarity(&pa, &pc);
        assert!(
            sim_ab > sim_ac,
            "similar vectors should project closer together"
        );
    }

    // --- SimilarityThreshold ---

    #[test]
    fn test_threshold_filter() {
        let threshold = SimilarityThreshold::new(0.5);
        let results = vec![
            SimilarityResult::new(0.9, 0, 0),
            SimilarityResult::new(0.3, 1, 1),
            SimilarityResult::new(0.7, 2, 2),
            SimilarityResult::new(0.1, 3, 3),
        ];
        let filtered = threshold.filter(results);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.score >= 0.5));
    }

    #[test]
    fn test_threshold_filter_all_pass() {
        let threshold = SimilarityThreshold::new(0.0);
        let results = vec![
            SimilarityResult::new(0.5, 0, 0),
            SimilarityResult::new(0.1, 1, 1),
        ];
        let filtered = threshold.filter(results);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_threshold_filter_none_pass() {
        let threshold = SimilarityThreshold::new(1.0);
        let results = vec![
            SimilarityResult::new(0.5, 0, 0),
            SimilarityResult::new(0.9, 1, 1),
        ];
        let filtered = threshold.filter(results);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_threshold_filter_and_rerank() {
        let threshold = SimilarityThreshold::new(0.5);
        let results = vec![
            SimilarityResult::new(0.9, 0, 0),
            SimilarityResult::new(0.3, 1, 1),
            SimilarityResult::new(0.7, 2, 2),
        ];
        let filtered = threshold.filter_and_rerank(results);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].rank, 0);
        assert_eq!(filtered[1].rank, 1);
    }

    #[test]
    fn test_threshold_search() {
        let threshold = SimilarityThreshold::new(0.5);
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0],   // sim = 1.0
            vec![0.0, 1.0],   // sim = 0.0
            vec![0.7, 0.7],   // sim ~ 0.707
        ];
        let results = threshold.search(&query, &candidates, DistanceMetric::Cosine);
        assert_eq!(results.len(), 2); // only 1.0 and ~0.707 pass
        assert_eq!(results[0].rank, 0);
        assert_eq!(results[1].rank, 1);
    }

    #[test]
    fn test_threshold_exact_boundary() {
        let threshold = SimilarityThreshold::new(0.5);
        let results = vec![SimilarityResult::new(0.5, 0, 0)];
        let filtered = threshold.filter(results);
        assert_eq!(filtered.len(), 1); // exactly at threshold should pass
    }

    // --- Integration tests ---

    #[test]
    fn test_knn_with_threshold() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let knn = KNearestNeighbors::new(embeddings, DistanceMetric::Cosine);
        let results = knn.search(&[1.0, 0.0, 0.0], 4);
        let threshold = SimilarityThreshold::new(0.5);
        let filtered = threshold.filter_and_rerank(results);
        // Only index 0 (1.0) and index 1 (~0.99) should pass
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cluster_then_search_within() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ca = ClusterAssignment::assign(&embeddings, &centroids, DistanceMetric::Cosine);

        // Search within cluster 0
        let cluster_0_members = ca.members(0);
        let cluster_0_embeddings: Vec<Vec<f32>> = cluster_0_members
            .iter()
            .map(|&i| embeddings[i].clone())
            .collect();
        let knn = KNearestNeighbors::new(cluster_0_embeddings, DistanceMetric::Cosine);
        let results = knn.search(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_normalize_then_compare() {
        let vecs = vec![vec![3.0, 4.0], vec![6.0, 8.0]]; // same direction, different magnitude
        let normed = EmbeddingNormalizer::l2_normalize_batch(&vecs);
        let calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
        let sim = calc.similarity(&normed[0], &normed[1]);
        assert!(approx_eq(sim, 1.0)); // same direction after normalization
    }

    #[test]
    fn test_reduce_then_knn() {
        let reducer = DimensionalityReducer::new(50, 5, 42);
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; 50];
                v[i % 50] = 1.0;
                v
            })
            .collect();
        let projected = reducer.project_batch(&embeddings);
        let knn = KNearestNeighbors::new(projected, DistanceMetric::Cosine);
        let query = reducer.project(&{
            let mut v = vec![0.0f32; 50];
            v[0] = 1.0;
            v
        });
        let results = knn.search(&query, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 0); // should find itself first
    }

    #[test]
    fn test_pairwise_then_threshold() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![1.0, 0.0], vec![0.7, 0.7]];
        let m = PairwiseSimilarityMatrix::compute(&a, &b, DistanceMetric::Cosine);
        // Check that we can use threshold logic on matrix rows
        let threshold = SimilarityThreshold::new(0.5);
        let row_0_results: Vec<SimilarityResult> = m.matrix[0]
            .iter()
            .enumerate()
            .map(|(j, &score)| SimilarityResult::new(score, j, j))
            .collect();
        let filtered = threshold.filter(row_0_results);
        assert_eq!(filtered.len(), 2); // both 1.0 and ~0.707 pass
    }
}
