//! Deduplication transformers for documents.
//!
//! This module provides transformers that remove duplicate documents based on
//! exact content matching, fuzzy similarity, or content hashing.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;

use super::DocumentTransformer;

// ─── ExactDeduplicator ───

/// Removes documents with exactly identical `page_content`.
///
/// The first occurrence of each unique content string is kept.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::dedup::ExactDeduplicator;
///
/// let dedup = ExactDeduplicator::new();
/// let unique = dedup.transform_documents(&docs).await?;
/// ```
pub struct ExactDeduplicator;

impl ExactDeduplicator {
    /// Create a new exact deduplicator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExactDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentTransformer for ExactDeduplicator {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for doc in documents {
            if seen.insert(doc.page_content.clone()) {
                results.push(doc.clone());
            }
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "ExactDeduplicator"
    }
}

// ─── FuzzyDeduplicator ───

/// Removes near-duplicate documents using character n-gram Jaccard similarity.
///
/// Documents whose Jaccard similarity to any previously kept document exceeds
/// the configured threshold are considered duplicates and removed.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::dedup::FuzzyDeduplicator;
///
/// let dedup = FuzzyDeduplicator::new()
///     .with_similarity_threshold(0.85)
///     .with_ngram_size(3);
/// let unique = dedup.transform_documents(&docs).await?;
/// ```
pub struct FuzzyDeduplicator {
    similarity_threshold: f64,
    ngram_size: usize,
}

impl FuzzyDeduplicator {
    /// Create a new fuzzy deduplicator with default threshold 0.9 and n-gram
    /// size 3.
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.9,
            ngram_size: 3,
        }
    }

    /// Set the Jaccard similarity threshold above which documents are
    /// considered duplicates. Must be in `[0.0, 1.0]`.
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set the character n-gram size used for Jaccard similarity computation.
    pub fn with_ngram_size(mut self, size: usize) -> Self {
        self.ngram_size = size;
        self
    }
}

impl Default for FuzzyDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract character n-grams from a string as a set.
fn char_ngrams(text: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        let mut set = HashSet::new();
        set.insert(text.to_string());
        return set;
    }
    chars
        .windows(n)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Compute Jaccard similarity between two sets of n-grams.
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[async_trait]
impl DocumentTransformer for FuzzyDeduplicator {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut kept: Vec<(Document, HashSet<String>)> = Vec::new();

        for doc in documents {
            let ngrams = char_ngrams(&doc.page_content, self.ngram_size);
            let is_duplicate = kept.iter().any(|(_, kept_ngrams)| {
                jaccard_similarity(&ngrams, kept_ngrams) >= self.similarity_threshold
            });
            if !is_duplicate {
                kept.push((doc.clone(), ngrams));
            }
        }

        Ok(kept.into_iter().map(|(doc, _)| doc).collect())
    }

    fn name(&self) -> &str {
        "FuzzyDeduplicator"
    }
}

// ─── ContentHashDeduplicator ───

/// Removes duplicate documents by comparing content hashes.
///
/// Uses a deterministic hash of each document's `page_content`. This is
/// faster than exact string comparison for large documents because only
/// the hash values are compared after the initial pass.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::dedup::ContentHashDeduplicator;
///
/// let dedup = ContentHashDeduplicator::new();
/// let unique = dedup.transform_documents(&docs).await?;
/// ```
pub struct ContentHashDeduplicator;

impl ContentHashDeduplicator {
    /// Create a new content hash deduplicator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ContentHashDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a deterministic hash of a string.
fn compute_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[async_trait]
impl DocumentTransformer for ContentHashDeduplicator {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for doc in documents {
            let hash = compute_hash(&doc.page_content);
            if seen.insert(hash) {
                results.push(doc.clone());
            }
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "ContentHashDeduplicator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    // ─── ExactDeduplicator tests ───

    #[tokio::test]
    async fn test_exact_dedup_removes_duplicates() {
        let dedup = ExactDeduplicator::new();
        let docs = vec![
            make_doc("hello world"),
            make_doc("hello world"),
            make_doc("different text"),
        ];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "hello world");
        assert_eq!(result[1].page_content, "different text");
    }

    #[tokio::test]
    async fn test_exact_dedup_all_unique() {
        let dedup = ExactDeduplicator::new();
        let docs = vec![make_doc("a"), make_doc("b"), make_doc("c")];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_exact_dedup_empty() {
        let dedup = ExactDeduplicator::new();
        let result = dedup.transform_documents(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_exact_dedup_all_same() {
        let dedup = ExactDeduplicator::new();
        let docs = vec![make_doc("same"), make_doc("same"), make_doc("same")];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    // ─── FuzzyDeduplicator tests ───

    #[tokio::test]
    async fn test_fuzzy_dedup_removes_near_duplicates() {
        let dedup = FuzzyDeduplicator::new().with_similarity_threshold(0.8);
        let docs = vec![
            make_doc("the quick brown fox jumps over the lazy dog"),
            make_doc("the quick brown fox jumps over the lazy cat"),
            make_doc("completely different content about rust programming"),
        ];
        let result = dedup.transform_documents(&docs).await.unwrap();
        // The first two are very similar, so only the first should be kept
        // along with the third which is different.
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].page_content,
            "the quick brown fox jumps over the lazy dog"
        );
        assert_eq!(
            result[1].page_content,
            "completely different content about rust programming"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_dedup_keeps_dissimilar() {
        let dedup = FuzzyDeduplicator::new().with_similarity_threshold(0.95);
        let docs = vec![
            make_doc("rust is a systems programming language"),
            make_doc("python is a dynamic programming language"),
        ];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_fuzzy_dedup_exact_duplicates() {
        let dedup = FuzzyDeduplicator::new();
        let docs = vec![make_doc("identical text"), make_doc("identical text")];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    // ─── ContentHashDeduplicator tests ───

    #[tokio::test]
    async fn test_hash_dedup_removes_duplicates() {
        let dedup = ContentHashDeduplicator::new();
        let docs = vec![
            make_doc("hello world"),
            make_doc("hello world"),
            make_doc("different"),
        ];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_hash_dedup_empty() {
        let dedup = ContentHashDeduplicator::new();
        let result = dedup.transform_documents(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_hash_dedup_preserves_order() {
        let dedup = ContentHashDeduplicator::new();
        let docs = vec![
            make_doc("first"),
            make_doc("second"),
            make_doc("first"),
            make_doc("third"),
        ];
        let result = dedup.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].page_content, "first");
        assert_eq!(result[1].page_content, "second");
        assert_eq!(result[2].page_content, "third");
    }

    // ─── Helper function tests ───

    #[test]
    fn test_jaccard_identical() {
        let a = char_ngrams("hello world", 3);
        let b = char_ngrams("hello world", 3);
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_completely_different() {
        let a = char_ngrams("aaaa", 3);
        let b = char_ngrams("zzzz", 3);
        assert!((jaccard_similarity(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_char_ngrams_short_text() {
        let ngrams = char_ngrams("ab", 3);
        assert_eq!(ngrams.len(), 1);
        assert!(ngrams.contains("ab"));
    }
}
