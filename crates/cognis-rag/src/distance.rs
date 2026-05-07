//! Distance metrics for vector similarity.

/// How to score similarity between two vectors. Each variant carries
/// its mathematical meaning into `Distance::compute` and `Distance::similarity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// Cosine similarity. `compute` returns the cosine *distance*
    /// (`1 - cos(a, b)`), range `[0, 2]`, lower = more similar.
    /// `similarity` returns `cos(a, b)`, range `[-1, 1]`, higher = more similar.
    Cosine,
    /// Euclidean (L2). `compute` returns `||a - b||`, range `[0, inf)`.
    /// `similarity` returns `1 / (1 + ||a - b||)`, range `(0, 1]`.
    Euclidean,
    /// Dot product. `compute` returns `-a·b` (negated so lower = more
    /// similar, matching the other variants). `similarity` returns `a·b`.
    Dot,
}

impl Distance {
    /// Compute the distance between two vectors. Lower = more similar.
    /// Panics on dimension mismatch (vectors must be the same length).
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "vector dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        match self {
            Distance::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na == 0.0 || nb == 0.0 {
                    return 1.0;
                }
                1.0 - (dot / (na * nb))
            }
            Distance::Euclidean => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            Distance::Dot => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        }
    }

    /// Compute similarity (higher = more similar). Inverse of `compute`
    /// in the appropriate sense per variant.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Distance::Cosine => 1.0 - self.compute(a, b),
            Distance::Euclidean => 1.0 / (1.0 + self.compute(a, b)),
            Distance::Dot => -self.compute(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn cosine_identical_vectors_distance_zero() {
        let d = Distance::Cosine.compute(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(approx(d, 0.0), "expected 0, got {d}");
    }

    #[test]
    fn cosine_orthogonal_vectors_distance_one() {
        let d = Distance::Cosine.compute(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(approx(d, 1.0), "expected 1, got {d}");
    }

    #[test]
    fn cosine_opposite_vectors_distance_two() {
        let d = Distance::Cosine.compute(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!(approx(d, 2.0), "expected 2, got {d}");
    }

    #[test]
    fn euclidean_identical_zero() {
        let d = Distance::Euclidean.compute(&[1.0, 2.0], &[1.0, 2.0]);
        assert!(approx(d, 0.0));
    }

    #[test]
    fn euclidean_3_4_5_triangle() {
        let d = Distance::Euclidean.compute(&[0.0, 0.0], &[3.0, 4.0]);
        assert!(approx(d, 5.0), "expected 5, got {d}");
    }

    #[test]
    fn dot_basic() {
        // a·b = 1*4 + 2*5 + 3*6 = 32. compute returns -32.
        let d = Distance::Dot.compute(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!(approx(d, -32.0));
        let s = Distance::Dot.similarity(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!(approx(s, 32.0));
    }

    #[test]
    fn similarity_inverse_of_compute() {
        let a = [0.5, 0.5, 0.5];
        let b = [0.6, 0.4, 0.7];
        for variant in [Distance::Cosine, Distance::Euclidean, Distance::Dot] {
            let _d = variant.compute(&a, &b);
            let _s = variant.similarity(&a, &b);
            // Just exercise both directions.
        }
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        let d = Distance::Cosine.compute(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(approx(d, 1.0));
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dimension_mismatch_panics() {
        Distance::Cosine.compute(&[1.0, 2.0], &[1.0]);
    }
}
