/// Internal utilities for vector store implementations.
///
/// Provides cosine similarity computation between matrices of embeddings
/// and the Maximal Marginal Relevance (MMR) algorithm for diverse retrieval.
/// Compute the dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 (Euclidean) norm of a vector.
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
fn cosine_similarity_pair(a: &[f64], b: &[f64]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Row-wise cosine similarity between two matrices of embeddings.
///
/// Given matrix `x` of shape `(n, m)` and matrix `y` of shape `(k, m)`,
/// returns a matrix of shape `(n, k)` where element `(i, j)` is the cosine
/// similarity between the `i`-th row of `x` and the `j`-th row of `y`.
///
/// # Errors
///
/// Returns an error if `x` and `y` have rows of different lengths, or if any
/// row within a matrix has an inconsistent length.
///
/// # Examples
///
/// ```
/// use rustchain_core::vectorstores::utils::cosine_similarity_matrix;
///
/// let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let y = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let result = cosine_similarity_matrix(&x, &y).unwrap();
/// assert!((result[0][0] - 1.0).abs() < 1e-9);
/// assert!((result[0][1] - 0.0).abs() < 1e-9);
/// assert!((result[1][0] - 0.0).abs() < 1e-9);
/// assert!((result[1][1] - 1.0).abs() < 1e-9);
/// ```
pub fn cosine_similarity_matrix(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if x.is_empty() || y.is_empty() {
        return Ok(vec![vec![]]);
    }

    let dim_x = x[0].len();
    let dim_y = y[0].len();

    if dim_x != dim_y {
        return Err(format!(
            "Number of columns in X and Y must be the same. X has {} columns and Y has {} columns.",
            dim_x, dim_y
        ));
    }

    // Validate consistent row lengths.
    for (i, row) in x.iter().enumerate() {
        if row.len() != dim_x {
            return Err(format!(
                "Inconsistent row length in X: row {} has {} columns, expected {}.",
                i,
                row.len(),
                dim_x
            ));
        }
    }
    for (i, row) in y.iter().enumerate() {
        if row.len() != dim_y {
            return Err(format!(
                "Inconsistent row length in Y: row {} has {} columns, expected {}.",
                i,
                row.len(),
                dim_y
            ));
        }
    }

    let result: Vec<Vec<f64>> = x
        .iter()
        .map(|row_x| {
            y.iter()
                .map(|row_y| cosine_similarity_pair(row_x, row_y))
                .collect()
        })
        .collect();

    Ok(result)
}

/// Select embeddings using the Maximal Marginal Relevance (MMR) algorithm.
///
/// MMR iteratively selects embeddings that balance relevance to the query
/// with diversity from already-selected embeddings.
///
/// # Arguments
///
/// * `query_embedding` - The query embedding vector.
/// * `embedding_list` - The candidate embeddings to select from.
/// * `lambda_mult` - Trade-off parameter between relevance and diversity.
///   A value of 1.0 means pure relevance; 0.0 means pure diversity. Defaults to 0.5.
/// * `k` - The number of embeddings to select. Defaults to 4.
///
/// # Returns
///
/// A vector of indices into `embedding_list` representing the selected embeddings.
///
/// # Examples
///
/// ```
/// use rustchain_core::vectorstores::utils::maximal_marginal_relevance;
///
/// let query = vec![1.0, 0.0];
/// let embeddings = vec![
///     vec![1.0, 0.0],
///     vec![0.9, 0.1],
///     vec![0.0, 1.0],
/// ];
/// let selected = maximal_marginal_relevance(&query, &embeddings, 0.5, 2);
/// assert_eq!(selected.len(), 2);
/// assert_eq!(selected[0], 0); // most similar to query
/// ```
pub fn maximal_marginal_relevance(
    query_embedding: &[f64],
    embedding_list: &[Vec<f64>],
    lambda_mult: f64,
    k: usize,
) -> Vec<usize> {
    let effective_k = k.min(embedding_list.len());
    if effective_k == 0 {
        return vec![];
    }

    // Compute similarity of each embedding to the query.
    let similarity_to_query: Vec<f64> = embedding_list
        .iter()
        .map(|emb| cosine_similarity_pair(query_embedding, emb))
        .collect();

    // Start with the most similar embedding.
    let most_similar = similarity_to_query
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap();

    let mut idxs: Vec<usize> = vec![most_similar];
    let mut selected: Vec<Vec<f64>> = vec![embedding_list[most_similar].clone()];

    while idxs.len() < effective_k {
        let mut best_score = f64::NEG_INFINITY;
        let mut idx_to_add = 0;

        for (i, query_score) in similarity_to_query.iter().enumerate() {
            if idxs.contains(&i) {
                continue;
            }

            // Max similarity to any already-selected embedding.
            let redundant_score = selected
                .iter()
                .map(|sel| cosine_similarity_pair(&embedding_list[i], sel))
                .fold(f64::NEG_INFINITY, f64::max);

            let equation_score = lambda_mult * query_score - (1.0 - lambda_mult) * redundant_score;

            if equation_score > best_score {
                best_score = equation_score;
                idx_to_add = i;
            }
        }

        idxs.push(idx_to_add);
        selected.push(embedding_list[idx_to_add].clone());
    }

    idxs
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- cosine_similarity_pair tests ---

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(cosine_similarity_pair(&a, &a), 1.0));
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(cosine_similarity_pair(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(approx_eq(cosine_similarity_pair(&a, &b), -1.0));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let zero = vec![0.0, 0.0];
        assert!(approx_eq(cosine_similarity_pair(&a, &zero), 0.0));
        assert!(approx_eq(cosine_similarity_pair(&zero, &a), 0.0));
    }

    // --- cosine_similarity_matrix tests ---

    #[test]
    fn test_matrix_identity() {
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let y = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = cosine_similarity_matrix(&x, &y).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert!(approx_eq(result[0][0], 1.0));
        assert!(approx_eq(result[0][1], 0.0));
        assert!(approx_eq(result[1][0], 0.0));
        assert!(approx_eq(result[1][1], 1.0));
    }

    #[test]
    fn test_matrix_empty_x() {
        let x: Vec<Vec<f64>> = vec![];
        let y = vec![vec![1.0, 0.0]];
        let result = cosine_similarity_matrix(&x, &y).unwrap();
        assert_eq!(result, vec![Vec::<f64>::new()]);
    }

    #[test]
    fn test_matrix_empty_y() {
        let x = vec![vec![1.0, 0.0]];
        let y: Vec<Vec<f64>> = vec![];
        let result = cosine_similarity_matrix(&x, &y).unwrap();
        assert_eq!(result, vec![Vec::<f64>::new()]);
    }

    #[test]
    fn test_matrix_dimension_mismatch() {
        let x = vec![vec![1.0, 0.0]];
        let y = vec![vec![1.0, 0.0, 0.0]];
        let result = cosine_similarity_matrix(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_single_row() {
        let x = vec![vec![1.0, 1.0]];
        let y = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let result = cosine_similarity_matrix(&x, &y).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        let expected_cos = 1.0 / 2.0_f64.sqrt();
        assert!(approx_eq(result[0][0], expected_cos));
        assert!(approx_eq(result[0][1], expected_cos));
        assert!(approx_eq(result[0][2], 1.0));
    }

    #[test]
    fn test_matrix_non_square() {
        let x = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let y = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = cosine_similarity_matrix(&x, &y).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        assert!(approx_eq(result[0][0], 1.0));
        assert!(approx_eq(result[0][1], 0.0));
        assert!(approx_eq(result[0][2], 0.0));
        assert!(approx_eq(result[1][0], 0.0));
        assert!(approx_eq(result[1][1], 1.0));
        assert!(approx_eq(result[1][2], 0.0));
    }

    // --- maximal_marginal_relevance tests ---

    #[test]
    fn test_mmr_empty_embeddings() {
        let query = vec![1.0, 0.0];
        let embeddings: Vec<Vec<f64>> = vec![];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_k_zero() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![vec![1.0, 0.0]];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_k_greater_than_embeddings() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_mmr_selects_most_similar_first() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![0.0, 1.0], // orthogonal
            vec![1.0, 0.0], // identical
            vec![0.5, 0.5], // partial match
        ];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 1);
        assert_eq!(result, vec![1]); // index 1 is most similar
    }

    #[test]
    fn test_mmr_diversity() {
        // With lambda_mult=0, MMR should maximize diversity (minimize redundancy).
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0],   // identical to query
            vec![0.99, 0.01], // very similar to query and to index 0
            vec![0.0, 1.0],   // orthogonal to query and to index 0
        ];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.0, 2);
        // First selection: most similar = 0
        assert_eq!(result[0], 0);
        // Second selection with lambda=0: maximize diversity from selected
        // index 2 is most different from index 0
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_mmr_pure_relevance() {
        // With lambda_mult=1.0, MMR should behave like pure similarity ranking.
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![0.0, 1.0], // orthogonal
            vec![1.0, 0.0], // identical
            vec![0.9, 0.1], // very similar
        ];
        let result = maximal_marginal_relevance(&query, &embeddings, 1.0, 3);
        assert_eq!(result[0], 1); // most similar
        assert_eq!(result[1], 2); // next most similar
        assert_eq!(result[2], 0); // least similar
    }

    #[test]
    fn test_mmr_returns_unique_indices() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.5],
        ];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 5);
        assert_eq!(result.len(), 5);
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "All indices should be unique");
    }

    #[test]
    fn test_mmr_single_embedding() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![vec![0.5, 0.5]];
        let result = maximal_marginal_relevance(&query, &embeddings, 0.5, 4);
        assert_eq!(result, vec![0]);
    }
}
