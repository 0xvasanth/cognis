//! Distance and similarity metrics for embedding vectors.
//!
//! Provides various distance functions (cosine, euclidean, dot product, manhattan,
//! chebyshev, hamming, jaccard), batch operations (pairwise distances, nearest
//! neighbors, maximal marginal relevance), vector utilities (normalize, mean,
//! weighted mean), and a high-level `EmbeddingComparator` for ranking, clustering,
//! and deduplication.

/// Supported distance/similarity metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// Cosine similarity / distance.
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot product similarity.
    DotProduct,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Chebyshev (L-infinity) distance.
    Chebyshev,
    /// Hamming distance (treats values as binary: nonzero = 1).
    Hamming,
    /// Jaccard distance (treats values as binary sets).
    Jaccard,
}

// ---------------------------------------------------------------------------
// Core distance / similarity functions
// ---------------------------------------------------------------------------

/// Compute the cosine similarity between two vectors.
///
/// Returns a value in `[-1.0, 1.0]`. Identical direction yields `1.0`,
/// orthogonal vectors yield `0.0`. Returns `0.0` if either vector has zero
/// magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
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

/// Cosine distance: `1.0 - cosine_similarity(a, b)`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Euclidean (L2) distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum.sqrt()
}

/// Dot product of two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Manhattan (L1) distance between two vectors.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev (L-infinity) distance: the maximum absolute difference.
pub fn chebyshev_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Hamming distance: fraction of positions where the binary indicators differ.
///
/// Each element is treated as binary — nonzero maps to 1, zero maps to 0.
pub fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    if a.is_empty() {
        return 0.0;
    }
    let mismatches = a
        .iter()
        .zip(b.iter())
        .filter(|(&x, &y)| (x != 0.0) != (y != 0.0))
        .count();
    mismatches as f32 / a.len() as f32
}

/// Jaccard distance: `1 - |intersection| / |union|` over binary indicators.
///
/// Each element is treated as binary — nonzero maps to 1, zero maps to 0.
pub fn jaccard_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    let mut intersection = 0usize;
    let mut union = 0usize;
    for i in 0..a.len() {
        let a_set = a[i] != 0.0;
        let b_set = b[i] != 0.0;
        if a_set || b_set {
            union += 1;
            if a_set && b_set {
                intersection += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - (intersection as f32 / union as f32)
    }
}

/// Compute the distance between two vectors using the given metric.
pub fn compute_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b), // negate so lower = more similar
        DistanceMetric::Manhattan => manhattan_distance(a, b),
        DistanceMetric::Chebyshev => chebyshev_distance(a, b),
        DistanceMetric::Hamming => hamming_distance(a, b),
        DistanceMetric::Jaccard => jaccard_distance(a, b),
    }
}

/// Compute the similarity between two vectors using the given metric.
///
/// Higher values mean more similar. The exact conversion depends on the metric:
/// - Cosine: cosine similarity directly
/// - Euclidean/Manhattan/Chebyshev: `1 / (1 + distance)`
/// - DotProduct: raw dot product
/// - Hamming/Jaccard: `1 - distance`
pub fn compute_similarity(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(a, b),
        DistanceMetric::Euclidean => 1.0 / (1.0 + euclidean_distance(a, b)),
        DistanceMetric::DotProduct => dot_product(a, b),
        DistanceMetric::Manhattan => 1.0 / (1.0 + manhattan_distance(a, b)),
        DistanceMetric::Chebyshev => 1.0 / (1.0 + chebyshev_distance(a, b)),
        DistanceMetric::Hamming => 1.0 - hamming_distance(a, b),
        DistanceMetric::Jaccard => 1.0 - jaccard_distance(a, b),
    }
}

// ---------------------------------------------------------------------------
// Batch operations
// ---------------------------------------------------------------------------

/// Compute the pairwise distance matrix for a set of vectors.
///
/// Returns an `n x n` matrix where `result[i][j]` is the distance between
/// `vectors[i]` and `vectors[j]`.
pub fn pairwise_distances(vectors: &[Vec<f32>], metric: DistanceMetric) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = compute_distance(metric, &vectors[i], &vectors[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

/// Find the `k` nearest neighbors of `query` among `vectors`.
///
/// Returns a vector of `(index, similarity_score)` pairs sorted by descending
/// similarity.
pub fn nearest_neighbors(
    query: &[f32],
    vectors: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, compute_similarity(metric, query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Select items using Maximal Marginal Relevance (MMR).
///
/// Balances relevance to the query with diversity among selected items.
///
/// # Arguments
/// - `query` — the query embedding
/// - `vectors` — candidate embeddings
/// - `k` — number of items to select
/// - `lambda` — trade-off parameter in `[0, 1]`. `1.0` = pure relevance,
///   `0.0` = pure diversity.
/// - `metric` — distance metric to use
///
/// Returns indices into `vectors`.
pub fn max_marginal_relevance(
    query: &[f32],
    vectors: &[Vec<f32>],
    k: usize,
    lambda: f32,
    metric: DistanceMetric,
) -> Vec<usize> {
    if vectors.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(vectors.len());

    // Pre-compute query similarities.
    let query_sims: Vec<f32> = vectors
        .iter()
        .map(|v| compute_similarity(metric, query, v))
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..vectors.len()).collect();

    // Pick the most relevant item first.
    let first = remaining
        .iter()
        .copied()
        .max_by(|&a, &b| {
            query_sims[a]
                .partial_cmp(&query_sims[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    selected.push(first);
    remaining.retain(|&i| i != first);

    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for &candidate in &remaining {
            let relevance = query_sims[candidate];
            let max_sim_to_selected = selected
                .iter()
                .map(|&s| compute_similarity(metric, &vectors[candidate], &vectors[s]))
                .fold(f32::NEG_INFINITY, f32::max);

            let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;
            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = candidate;
            }
        }

        selected.push(best_idx);
        remaining.retain(|&i| i != best_idx);
    }

    selected
}

// ---------------------------------------------------------------------------
// Vector utilities
// ---------------------------------------------------------------------------

/// L2-normalize a vector. Returns a zero vector if the input has zero magnitude.
pub fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec![0.0; vector.len()]
    } else {
        vector.iter().map(|x| x / norm).collect()
    }
}

/// Compute the element-wise mean of a set of vectors.
///
/// # Panics
/// Panics if `vectors` is empty.
pub fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    assert!(!vectors.is_empty(), "cannot compute mean of empty set");
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dim];
    for v in vectors {
        assert_eq!(v.len(), dim, "all vectors must have the same dimension");
        for (i, &val) in v.iter().enumerate() {
            result[i] += val;
        }
    }
    for val in &mut result {
        *val /= n;
    }
    result
}

/// Compute the weighted mean of a set of vectors.
///
/// # Panics
/// Panics if `vectors` and `weights` have different lengths, or if `vectors` is
/// empty.
pub fn weighted_mean(vectors: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    assert!(
        !vectors.is_empty(),
        "cannot compute weighted mean of empty set"
    );
    assert_eq!(
        vectors.len(),
        weights.len(),
        "vectors and weights must have the same length"
    );
    let dim = vectors[0].len();
    let total_weight: f32 = weights.iter().sum();
    let mut result = vec![0.0f32; dim];
    for (v, &w) in vectors.iter().zip(weights.iter()) {
        assert_eq!(v.len(), dim, "all vectors must have the same dimension");
        for (i, &val) in v.iter().enumerate() {
            result[i] += val * w;
        }
    }
    if total_weight != 0.0 {
        for val in &mut result {
            *val /= total_weight;
        }
    }
    result
}

/// Compute the centroid (mean) of a set of vectors. Alias for [`mean_vector`].
pub fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    mean_vector(vectors)
}

// ---------------------------------------------------------------------------
// EmbeddingComparator
// ---------------------------------------------------------------------------

/// High-level comparator for embedding vectors using a specified distance metric.
///
/// Provides convenience methods for comparing, ranking, clustering, and
/// deduplicating embeddings.
pub struct EmbeddingComparator {
    metric: DistanceMetric,
}

impl EmbeddingComparator {
    /// Create a new comparator with the given metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric }
    }

    /// Compute the similarity between two vectors.
    pub fn compare(&self, a: &[f32], b: &[f32]) -> f32 {
        compute_similarity(self.metric, a, b)
    }

    /// Rank candidate vectors by similarity to the query.
    ///
    /// Returns `(index, similarity_score)` pairs sorted by descending similarity.
    pub fn rank(&self, query: &[f32], candidates: &[Vec<f32>]) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, v)| (i, compute_similarity(self.metric, query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Simple single-linkage threshold clustering.
    ///
    /// Two vectors are in the same cluster if their similarity is at or above
    /// `threshold`. Uses a greedy approach: each vector is assigned to the first
    /// cluster whose representative it is similar enough to.
    pub fn cluster_by_similarity(&self, vectors: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut representatives: Vec<usize> = Vec::new();

        for i in 0..vectors.len() {
            let mut found = false;
            for (ci, &rep) in representatives.iter().enumerate() {
                let sim = compute_similarity(self.metric, &vectors[i], &vectors[rep]);
                if sim >= threshold {
                    clusters[ci].push(i);
                    found = true;
                    break;
                }
            }
            if !found {
                representatives.push(i);
                clusters.push(vec![i]);
            }
        }

        clusters
    }

    /// Deduplicate vectors by similarity threshold.
    ///
    /// Returns the indices of vectors that are considered unique — i.e., no
    /// previously retained vector has similarity >= `threshold` with this one.
    pub fn deduplicate(&self, vectors: &[Vec<f32>], threshold: f32) -> Vec<usize> {
        let mut unique: Vec<usize> = Vec::new();
        for i in 0..vectors.len() {
            let is_dup = unique
                .iter()
                .any(|&u| compute_similarity(self.metric, &vectors[i], &vectors[u]) >= threshold);
            if !is_dup {
                unique.push(i);
            }
        }
        unique
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity 0.0"
        );
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "orthogonal cosine distance should be 1.0"
        );
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-6, "3-4-5 triangle");
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let d = manhattan_distance(&a, &b);
        assert!((d - 7.0).abs() < 1e-6); // |3| + |4| + |0| = 7
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let d = chebyshev_distance(&a, &b);
        assert!((d - 4.0).abs() < 1e-6); // max(|3|, |4|, |0|) = 4
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        let d = hamming_distance(&a, &b);
        assert!((d - 0.5).abs() < 1e-6); // 2 mismatches out of 4
    }

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-6,
            "normalized vector should have unit length"
        );
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_mean_vector() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let m = mean_vector(&vectors);
        assert!((m[0] - 3.0).abs() < 1e-6);
        assert!((m[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_neighbors() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // identical to query
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![0.9, 0.1, 0.0], // close to query
        ];
        let nn = nearest_neighbors(&query, &vectors, 2, DistanceMetric::Cosine);
        assert_eq!(nn.len(), 2);
        assert_eq!(nn[0].0, 0, "first neighbor should be the identical vector");
        assert_eq!(nn[1].0, 2, "second neighbor should be the close vector");
    }

    #[test]
    fn test_max_marginal_relevance_selection() {
        let query = vec![1.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0],   // identical to query
            vec![0.99, 0.01], // very similar to first
            vec![0.0, 1.0],   // orthogonal / diverse
            vec![0.5, 0.5],   // moderate
        ];
        // With lambda=0.0 (pure diversity) and k=2, MMR should pick the most
        // relevant first, then the most diverse second.
        let selected = max_marginal_relevance(&query, &vectors, 2, 0.0, DistanceMetric::Cosine);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 0, "first pick should be most relevant");
        // Second pick should be the most diverse from index 0 => index 2 (orthogonal).
        assert_eq!(
            selected[1], 2,
            "MMR with lambda=0 should pick most diverse vector"
        );
    }

    #[test]
    fn test_pairwise_distance_matrix() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let matrix = pairwise_distances(&vectors, DistanceMetric::Cosine);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        // Diagonal should be zero.
        assert!((matrix[0][0]).abs() < 1e-6);
        assert!((matrix[1][1]).abs() < 1e-6);
        // Symmetric.
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-6);
        // Cosine distance between [1,0] and [0,1] = 1.0.
        assert!((matrix[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_comparator_rank() {
        let comp = EmbeddingComparator::new(DistanceMetric::Cosine);
        let query = vec![1.0, 0.0];
        let candidates = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let ranked = comp.rank(&query, &candidates);
        assert_eq!(
            ranked[0].0, 1,
            "most similar should be the identical vector"
        );
    }

    #[test]
    fn test_cluster_by_similarity() {
        let comp = EmbeddingComparator::new(DistanceMetric::Cosine);
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01], // very similar to 0
            vec![0.0, 1.0],
            vec![0.01, 0.99], // very similar to 2
        ];
        let clusters = comp.cluster_by_similarity(&vectors, 0.95);
        assert_eq!(clusters.len(), 2, "should form 2 clusters");
        assert!(clusters[0].contains(&0));
        assert!(clusters[0].contains(&1));
        assert!(clusters[1].contains(&2));
        assert!(clusters[1].contains(&3));
    }

    #[test]
    fn test_deduplicate() {
        let comp = EmbeddingComparator::new(DistanceMetric::Cosine);
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.999, 0.001], // near-duplicate of 0
            vec![0.0, 1.0],     // unique
            vec![0.0, 0.999],   // near-duplicate of 2
        ];
        let unique = comp.deduplicate(&vectors, 0.99);
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&0));
        assert!(unique.contains(&2));
    }

    #[test]
    #[should_panic(expected = "vectors must have the same dimension")]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        cosine_similarity(&a, &b);
    }

    #[test]
    fn test_zero_vector_handling() {
        let zero = vec![0.0, 0.0, 0.0];
        let other = vec![1.0, 2.0, 3.0];
        // Cosine similarity with a zero vector should return 0.0, not NaN.
        let sim = cosine_similarity(&zero, &other);
        assert!((sim - 0.0).abs() < 1e-6);
        assert!(!sim.is_nan());

        // Normalize of zero vector should return zero vector.
        let n = normalize(&zero);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_weighted_mean() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let weights = vec![3.0, 1.0];
        let wm = weighted_mean(&vectors, &weights);
        assert!((wm[0] - 0.75).abs() < 1e-6);
        assert!((wm[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance() {
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0];
        // intersection = {0}, union = {0,1,2} => jaccard = 1 - 1/3
        let d = jaccard_distance(&a, &b);
        assert!((d - (1.0 - 1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_distance_and_similarity_consistency() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // For cosine: distance = 1 - similarity
        let sim = compute_similarity(DistanceMetric::Cosine, &a, &b);
        let dist = compute_distance(DistanceMetric::Cosine, &a, &b);
        assert!((sim + dist - 1.0).abs() < 1e-6);
    }
}
