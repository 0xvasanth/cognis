//! Cross-encoder interface for reranking.
//!
//! Cross-encoders score (query, document) pairs for semantic similarity,
//! enabling reranking of search results. This module provides:
//!
//! - [`CrossEncoder`] -- base trait for scoring text pairs
//! - [`CrossEncoderResult`] -- scored pair with index and metadata
//! - [`FakeCrossEncoder`] -- deterministic scorer for testing
//! - [`ThresholdCrossEncoder`] -- filters pairs below a score threshold
//! - [`BatchCrossEncoder`] -- batched scoring with configurable batch size
//! - [`CrossEncoderReranker`] -- reranks documents by cross-encoder score
//! - [`CachedCrossEncoder`] -- LRU-cached scoring wrapper
//! - [`NormalizedCrossEncoder`] -- normalizes scores to \[0, 1\]

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::documents::Document;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Base trait for cross-encoder models that score text pairs for similarity.
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    /// Score a list of (text_a, text_b) pairs.
    ///
    /// Returns one `f64` score per pair, where higher values indicate greater
    /// similarity.
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>>;
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// The result of scoring a single text pair through a cross-encoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrossEncoderResult {
    /// Index of the pair in the original input list.
    pub index: usize,
    /// The similarity score produced by the cross-encoder.
    pub score: f64,
    /// Optional metadata attached to this result.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl CrossEncoderResult {
    /// Create a new result with the given index and score.
    pub fn new(index: usize, score: f64) -> Self {
        Self {
            index,
            score,
            metadata: HashMap::new(),
        }
    }

    /// Attach metadata to this result.
    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }
}

// ---------------------------------------------------------------------------
// FakeCrossEncoder
// ---------------------------------------------------------------------------

/// A deterministic cross-encoder for testing.
///
/// Scores each pair based on the fraction of characters in the second string
/// that also appear in the first string (case-insensitive character overlap).
#[derive(Debug, Clone, Default)]
pub struct FakeCrossEncoder;

impl FakeCrossEncoder {
    pub fn new() -> Self {
        Self
    }

    fn overlap_score(a: &str, b: &str) -> f64 {
        if b.is_empty() {
            return 0.0;
        }
        let a_lower: Vec<char> = a.to_lowercase().chars().collect();
        let matching = b
            .to_lowercase()
            .chars()
            .filter(|c| a_lower.contains(c))
            .count();
        matching as f64 / b.len() as f64
    }
}

#[async_trait]
impl CrossEncoder for FakeCrossEncoder {
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        Ok(pairs
            .iter()
            .map(|(a, b)| Self::overlap_score(a, b))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// ThresholdCrossEncoder
// ---------------------------------------------------------------------------

/// Wraps a cross-encoder and zeroes out scores below a threshold.
///
/// Pairs with a score below `threshold` receive a score of `0.0`.
#[derive(Debug)]
pub struct ThresholdCrossEncoder<E: CrossEncoder> {
    inner: E,
    threshold: f64,
}

impl<E: CrossEncoder> ThresholdCrossEncoder<E> {
    /// Create a new threshold wrapper.
    pub fn new(inner: E, threshold: f64) -> Self {
        Self { inner, threshold }
    }

    /// Return the current threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[async_trait]
impl<E: CrossEncoder> CrossEncoder for ThresholdCrossEncoder<E> {
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let scores = self.inner.score_pairs(pairs).await?;
        Ok(scores
            .into_iter()
            .map(|s| if s >= self.threshold { s } else { 0.0 })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// BatchCrossEncoder
// ---------------------------------------------------------------------------

/// Wraps a cross-encoder and processes pairs in fixed-size batches.
#[derive(Debug)]
pub struct BatchCrossEncoder<E: CrossEncoder> {
    inner: E,
    batch_size: usize,
}

impl<E: CrossEncoder> BatchCrossEncoder<E> {
    /// Create a batch wrapper with the given batch size.
    ///
    /// # Panics
    /// Panics if `batch_size` is zero.
    pub fn new(inner: E, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self { inner, batch_size }
    }

    /// Return the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[async_trait]
impl<E: CrossEncoder> CrossEncoder for BatchCrossEncoder<E> {
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let mut all_scores = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(self.batch_size) {
            let scores = self.inner.score_pairs(chunk).await?;
            all_scores.extend(scores);
        }
        Ok(all_scores)
    }
}

// ---------------------------------------------------------------------------
// CrossEncoderReranker
// ---------------------------------------------------------------------------

/// Reranks documents by scoring each against a query via a cross-encoder.
#[derive(Debug)]
pub struct CrossEncoderReranker<E: CrossEncoder> {
    encoder: E,
    top_k: Option<usize>,
}

impl<E: CrossEncoder> CrossEncoderReranker<E> {
    /// Create a new reranker.
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            top_k: None,
        }
    }

    /// Limit the number of returned documents.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Score and rerank `documents` against `query`.
    ///
    /// Returns [`CrossEncoderResult`]s sorted by descending score. The `index`
    /// field refers to the position in the original `documents` slice.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<CrossEncoderResult>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let pairs: Vec<(String, String)> = documents
            .iter()
            .map(|d| (query.to_string(), d.page_content.clone()))
            .collect();

        let scores = self.encoder.score_pairs(&pairs).await?;

        let mut results: Vec<CrossEncoderResult> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| {
                let mut meta = documents[i].metadata.clone();
                if let Some(id) = &documents[i].id {
                    meta.insert("document_id".to_string(), Value::String(id.clone()));
                }
                CrossEncoderResult {
                    index: i,
                    score,
                    metadata: meta,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = self.top_k {
            results.truncate(k);
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// CachedCrossEncoder
// ---------------------------------------------------------------------------

/// An LRU-cached wrapper around a cross-encoder.
///
/// Caches individual pair scores so repeated queries are not re-scored.
/// The cache key is the pair `(text_a, text_b)`.
#[derive(Debug)]
pub struct CachedCrossEncoder<E: CrossEncoder> {
    inner: E,
    cache: Mutex<LruCache>,
}

/// A simple bounded LRU cache backed by a `Vec`.
#[derive(Debug)]
struct LruCache {
    entries: Vec<((String, String), f64)>,
    capacity: usize,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&mut self, key: &(String, String)) -> Option<f64> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            let entry = self.entries.remove(pos);
            let score = entry.1;
            self.entries.push(entry);
            Some(score)
        } else {
            None
        }
    }

    fn insert(&mut self, key: (String, String), value: f64) {
        // Remove existing entry if present.
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(pos);
        }
        if self.entries.len() >= self.capacity {
            self.entries.remove(0); // evict oldest
        }
        self.entries.push((key, value));
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<E: CrossEncoder> CachedCrossEncoder<E> {
    /// Create a cached wrapper with the given LRU capacity.
    pub fn new(inner: E, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Return the number of entries currently in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

#[async_trait]
impl<E: CrossEncoder> CrossEncoder for CachedCrossEncoder<E> {
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let mut results = vec![0.0_f64; pairs.len()];
        let mut misses: Vec<(usize, (String, String))> = Vec::new();

        {
            let mut cache = self.cache.lock().unwrap();
            for (i, pair) in pairs.iter().enumerate() {
                if let Some(score) = cache.get(pair) {
                    results[i] = score;
                } else {
                    misses.push((i, pair.clone()));
                }
            }
        }

        if !misses.is_empty() {
            let miss_pairs: Vec<(String, String)> = misses.iter().map(|(_, p)| p.clone()).collect();
            let scores = self.inner.score_pairs(&miss_pairs).await?;

            let mut cache = self.cache.lock().unwrap();
            for ((idx, pair), score) in misses.into_iter().zip(scores) {
                cache.insert(pair, score);
                results[idx] = score;
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// NormalizedCrossEncoder
// ---------------------------------------------------------------------------

/// Wraps a cross-encoder and normalizes all scores to the `[0, 1]` range
/// using min-max normalization across the batch.
///
/// If all scores in a batch are equal the output is `0.5` for every pair.
#[derive(Debug)]
pub struct NormalizedCrossEncoder<E: CrossEncoder> {
    inner: E,
}

impl<E: CrossEncoder> NormalizedCrossEncoder<E> {
    /// Create a normalized wrapper.
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<E: CrossEncoder> CrossEncoder for NormalizedCrossEncoder<E> {
    async fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let scores = self.inner.score_pairs(pairs).await?;
        if scores.is_empty() {
            return Ok(scores);
        }

        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range == 0.0 {
            return Ok(vec![0.5; scores.len()]);
        }

        Ok(scores.into_iter().map(|s| (s - min) / range).collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ---- CrossEncoderResult ----

    #[test]
    fn result_new() {
        let r = CrossEncoderResult::new(3, 0.75);
        assert_eq!(r.index, 3);
        assert!((r.score - 0.75).abs() < f64::EPSILON);
        assert!(r.metadata.is_empty());
    }

    #[test]
    fn result_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".into(), Value::String("test".into()));
        let r = CrossEncoderResult::new(0, 0.5).with_metadata(meta.clone());
        assert_eq!(r.metadata, meta);
    }

    #[test]
    fn result_serialization_roundtrip() {
        let r = CrossEncoderResult::new(1, 0.9);
        let json = serde_json::to_string(&r).unwrap();
        let r2: CrossEncoderResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn result_serialization_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("k".into(), Value::Number(42.into()));
        let r = CrossEncoderResult::new(0, 1.0).with_metadata(meta);
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"k\":42"));
    }

    #[test]
    fn result_empty_metadata_not_serialized() {
        let r = CrossEncoderResult::new(0, 0.0);
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("metadata"));
    }

    // ---- FakeCrossEncoder ----

    #[tokio::test]
    async fn fake_identical_strings() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("hello".into(), "hello".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_no_overlap() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("abc".into(), "xyz".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_partial_overlap() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("abcd".into(), "abef".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        // 'a' and 'b' from "abef" match in "abcd" => 2/4 = 0.5
        assert!((scores[0] - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_case_insensitive() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("HELLO".into(), "hello".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_empty_second_string() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("hello".into(), "".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_empty_first_string() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("".into(), "hello".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_both_empty() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("".into(), "".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_multiple_pairs() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![
            ("abc".into(), "abc".into()),
            ("abc".into(), "xyz".into()),
            ("abc".into(), "abx".into()),
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1]).abs() < f64::EPSILON);
        // 'a','b' match out of 3 => 2/3
        assert!((scores[2] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn fake_empty_pairs_list() {
        let enc = FakeCrossEncoder::new();
        let pairs: Vec<(String, String)> = vec![];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!(scores.is_empty());
    }

    #[tokio::test]
    async fn fake_deterministic() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("rust is great".into(), "great rust".into())];
        let s1 = enc.score_pairs(&pairs).await.unwrap();
        let s2 = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(s1, s2);
    }

    // ---- ThresholdCrossEncoder ----

    #[tokio::test]
    async fn threshold_filters_below() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.6);
        let pairs = vec![
            ("abc".into(), "abc".into()), // 1.0 -> keep
            ("abc".into(), "xyz".into()), // 0.0 -> zero
            ("abc".into(), "abx".into()), // 0.666 -> keep
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1]).abs() < f64::EPSILON);
        assert!(scores[2] > 0.0);
    }

    #[tokio::test]
    async fn threshold_exact_boundary() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.5);
        // "abcd" vs "abef" => 0.5, should be kept (>=)
        let pairs = vec![("abcd".into(), "abef".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn threshold_accessor() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.42);
        assert!((enc.threshold() - 0.42).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn threshold_zero_keeps_all() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.0);
        let pairs = vec![("abc".into(), "xyz".into())]; // score 0.0
        let scores = enc.score_pairs(&pairs).await.unwrap();
        // 0.0 >= 0.0 is true, so it stays
        assert!((scores[0]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn threshold_one_filters_imperfect() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 1.0);
        let pairs = vec![
            ("abc".into(), "abc".into()), // 1.0 -> keep
            ("abc".into(), "abx".into()), // 0.666 -> zero
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1]).abs() < f64::EPSILON);
    }

    // ---- BatchCrossEncoder ----

    #[tokio::test]
    async fn batch_produces_same_results() {
        let enc = BatchCrossEncoder::new(FakeCrossEncoder::new(), 2);
        let pairs = vec![
            ("abc".into(), "abc".into()),
            ("abc".into(), "xyz".into()),
            ("abc".into(), "abx".into()),
        ];
        let batch_scores = enc.score_pairs(&pairs).await.unwrap();
        let direct_scores = FakeCrossEncoder::new().score_pairs(&pairs).await.unwrap();
        assert_eq!(batch_scores, direct_scores);
    }

    #[tokio::test]
    async fn batch_size_accessor() {
        let enc = BatchCrossEncoder::new(FakeCrossEncoder::new(), 10);
        assert_eq!(enc.batch_size(), 10);
    }

    #[tokio::test]
    async fn batch_single_item_batches() {
        let enc = BatchCrossEncoder::new(FakeCrossEncoder::new(), 1);
        let pairs = vec![("a".into(), "a".into()), ("b".into(), "b".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn batch_larger_than_input() {
        let enc = BatchCrossEncoder::new(FakeCrossEncoder::new(), 100);
        let pairs = vec![("abc".into(), "abc".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 1);
    }

    #[tokio::test]
    async fn batch_empty_input() {
        let enc = BatchCrossEncoder::new(FakeCrossEncoder::new(), 5);
        let pairs: Vec<(String, String)> = vec![];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn batch_zero_panics() {
        let _ = BatchCrossEncoder::new(FakeCrossEncoder::new(), 0);
    }

    // ---- CrossEncoderReranker ----

    #[tokio::test]
    async fn reranker_basic_order() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let docs = vec![
            Document::new("xyz"), // low overlap with "abc"
            Document::new("abc"), // perfect overlap
            Document::new("abx"), // partial overlap
        ];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 1); // "abc" ranked first
    }

    #[tokio::test]
    async fn reranker_top_k() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(2);
        let docs = vec![
            Document::new("xyz"),
            Document::new("abc"),
            Document::new("abx"),
        ];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn reranker_empty_documents() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let results = reranker.rerank("query", &[]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn reranker_preserves_document_id() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let docs = vec![Document::new("abc").with_id("doc-1")];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert_eq!(
            results[0].metadata.get("document_id"),
            Some(&Value::String("doc-1".into()))
        );
    }

    #[tokio::test]
    async fn reranker_preserves_document_metadata() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let mut meta = HashMap::new();
        meta.insert("source".into(), Value::String("web".into()));
        let docs = vec![Document::new("abc").with_metadata(meta)];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert_eq!(
            results[0].metadata.get("source"),
            Some(&Value::String("web".into()))
        );
    }

    #[tokio::test]
    async fn reranker_top_k_larger_than_docs() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(10);
        let docs = vec![Document::new("abc")];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn reranker_scores_descending() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let docs = vec![
            Document::new("z"),
            Document::new("ab"),
            Document::new("abc"),
        ];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    // ---- CachedCrossEncoder ----

    #[tokio::test]
    async fn cached_returns_correct_scores() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let pairs = vec![("abc".into(), "abc".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn cached_populates_cache() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        assert_eq!(enc.cache_len(), 0);
        let pairs = vec![("abc".into(), "abc".into())];
        enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(enc.cache_len(), 1);
    }

    #[tokio::test]
    async fn cached_serves_from_cache() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let pairs = vec![("abc".into(), "abc".into())];
        let s1 = enc.score_pairs(&pairs).await.unwrap();
        let s2 = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(s1, s2);
        assert_eq!(enc.cache_len(), 1); // still 1, served from cache
    }

    #[tokio::test]
    async fn cached_mixed_hits_and_misses() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let pairs1 = vec![("abc".into(), "abc".into())];
        enc.score_pairs(&pairs1).await.unwrap();

        let pairs2 = vec![
            ("abc".into(), "abc".into()), // hit
            ("xyz".into(), "xyz".into()), // miss
        ];
        let scores = enc.score_pairs(&pairs2).await.unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1] - 1.0).abs() < f64::EPSILON);
        assert_eq!(enc.cache_len(), 2);
    }

    #[tokio::test]
    async fn cached_evicts_lru() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 2);
        // Fill cache with 2 entries
        enc.score_pairs(&[("a".into(), "a".into())]).await.unwrap();
        enc.score_pairs(&[("b".into(), "b".into())]).await.unwrap();
        assert_eq!(enc.cache_len(), 2);

        // Adding a third should evict the oldest ("a","a")
        enc.score_pairs(&[("c".into(), "c".into())]).await.unwrap();
        assert_eq!(enc.cache_len(), 2);
    }

    #[tokio::test]
    async fn cached_empty_input() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let scores = enc.score_pairs(&[]).await.unwrap();
        assert!(scores.is_empty());
        assert_eq!(enc.cache_len(), 0);
    }

    // ---- NormalizedCrossEncoder ----

    #[tokio::test]
    async fn normalized_range() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        let pairs = vec![
            ("abc".into(), "abc".into()), // highest
            ("abc".into(), "xyz".into()), // lowest
            ("abc".into(), "abx".into()), // middle
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON); // max -> 1.0
        assert!((scores[1]).abs() < f64::EPSILON); // min -> 0.0
        assert!(scores[2] > 0.0 && scores[2] < 1.0); // between
    }

    #[tokio::test]
    async fn normalized_all_equal() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        let pairs = vec![("abc".into(), "abc".into()), ("xyz".into(), "xyz".into())];
        // Both are 1.0, so range is 0 -> all become 0.5
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 0.5).abs() < f64::EPSILON);
        assert!((scores[1] - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn normalized_single_pair() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        let pairs = vec![("abc".into(), "abc".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        // single item -> all equal -> 0.5
        assert!((scores[0] - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn normalized_empty() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        let scores = enc.score_pairs(&[]).await.unwrap();
        assert!(scores.is_empty());
    }

    #[tokio::test]
    async fn normalized_two_distinct_values() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        // "abc" vs "abc" => 1.0, "abc" vs "xyz" => 0.0
        let pairs = vec![("abc".into(), "abc".into()), ("abc".into(), "xyz".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1]).abs() < f64::EPSILON);
    }

    // ---- Composition tests ----

    #[tokio::test]
    async fn normalized_threshold_composition() {
        let inner = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.5);
        let enc = NormalizedCrossEncoder::new(inner);
        let pairs = vec![
            ("abc".into(), "abc".into()), // 1.0 (above threshold)
            ("abc".into(), "xyz".into()), // 0.0 (below -> zeroed by threshold)
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        // After threshold: [1.0, 0.0]; after normalize: [1.0, 0.0]
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
        assert!((scores[1]).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn batch_cached_composition() {
        let inner = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let enc = BatchCrossEncoder::new(inner, 2);
        let pairs = vec![
            ("abc".into(), "abc".into()),
            ("abc".into(), "xyz".into()),
            ("abc".into(), "abx".into()),
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 3);
    }

    // ---- Trait object safety ----

    #[tokio::test]
    async fn trait_object_works() {
        let enc: Box<dyn CrossEncoder> = Box::new(FakeCrossEncoder::new());
        let pairs = vec![("abc".into(), "abc".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn arc_trait_object_works() {
        let enc: Arc<dyn CrossEncoder> = Arc::new(FakeCrossEncoder::new());
        let pairs = vec![("abc".into(), "abc".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    // ---- LRU cache internals ----

    #[test]
    fn lru_cache_insert_and_get() {
        let mut cache = LruCache::new(3);
        cache.insert(("a".into(), "b".into()), 0.5);
        assert_eq!(cache.get(&("a".into(), "b".into())), Some(0.5));
    }

    #[test]
    fn lru_cache_miss() {
        let mut cache = LruCache::new(3);
        assert_eq!(cache.get(&("a".into(), "b".into())), None);
    }

    #[test]
    fn lru_cache_eviction_order() {
        let mut cache = LruCache::new(2);
        cache.insert(("a".into(), "1".into()), 1.0);
        cache.insert(("b".into(), "2".into()), 2.0);
        cache.insert(("c".into(), "3".into()), 3.0); // evicts ("a","1")
        assert_eq!(cache.get(&("a".into(), "1".into())), None);
        assert_eq!(cache.get(&("b".into(), "2".into())), Some(2.0));
        assert_eq!(cache.get(&("c".into(), "3".into())), Some(3.0));
    }

    #[test]
    fn lru_cache_access_refreshes() {
        let mut cache = LruCache::new(2);
        cache.insert(("a".into(), "1".into()), 1.0);
        cache.insert(("b".into(), "2".into()), 2.0);
        // Access "a" to make it recently used
        cache.get(&("a".into(), "1".into()));
        // Insert "c", which should evict "b" (now oldest)
        cache.insert(("c".into(), "3".into()), 3.0);
        assert_eq!(cache.get(&("a".into(), "1".into())), Some(1.0));
        assert_eq!(cache.get(&("b".into(), "2".into())), None);
    }

    #[test]
    fn lru_cache_update_existing() {
        let mut cache = LruCache::new(2);
        cache.insert(("a".into(), "1".into()), 1.0);
        cache.insert(("a".into(), "1".into()), 9.0);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&("a".into(), "1".into())), Some(9.0));
    }

    // ---- Additional FakeCrossEncoder tests ----

    #[tokio::test]
    async fn fake_whitespace_overlap() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![("hello world".into(), "hello world".into())];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert!((scores[0] - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn fake_scores_between_zero_and_one() {
        let enc = FakeCrossEncoder::new();
        let pairs = vec![
            ("rust lang".into(), "python lang".into()),
            ("quick brown fox".into(), "lazy dog jumps".into()),
        ];
        let scores = enc.score_pairs(&pairs).await.unwrap();
        for s in &scores {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    // ---- Additional reranker tests ----

    #[tokio::test]
    async fn reranker_single_document() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new());
        let docs = vec![Document::new("hello")];
        let results = reranker.rerank("hello", &docs).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
    }

    #[tokio::test]
    async fn reranker_top_k_zero() {
        let reranker = CrossEncoderReranker::new(FakeCrossEncoder::new()).with_top_k(0);
        let docs = vec![Document::new("abc")];
        let results = reranker.rerank("abc", &docs).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Additional threshold tests ----

    #[tokio::test]
    async fn threshold_empty_input() {
        let enc = ThresholdCrossEncoder::new(FakeCrossEncoder::new(), 0.5);
        let scores = enc.score_pairs(&[]).await.unwrap();
        assert!(scores.is_empty());
    }

    // ---- Additional cached tests ----

    #[tokio::test]
    async fn cached_multiple_pairs_at_once() {
        let enc = CachedCrossEncoder::new(FakeCrossEncoder::new(), 10);
        let pairs = vec![
            ("a".into(), "a".into()),
            ("b".into(), "b".into()),
            ("c".into(), "c".into()),
        ];
        enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(enc.cache_len(), 3);
        // Second call should all be cache hits
        let scores = enc.score_pairs(&pairs).await.unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(enc.cache_len(), 3);
    }

    // ---- Additional normalized tests ----

    #[tokio::test]
    async fn normalized_preserves_order() {
        let enc = NormalizedCrossEncoder::new(FakeCrossEncoder::new());
        let pairs = vec![
            ("abc".into(), "abc".into()),
            ("abc".into(), "abx".into()),
            ("abc".into(), "xyz".into()),
        ];
        let raw = FakeCrossEncoder::new().score_pairs(&pairs).await.unwrap();
        let norm = enc.score_pairs(&pairs).await.unwrap();
        // Order should be preserved
        assert!(raw[0] > raw[1] && norm[0] > norm[1]);
        assert!(raw[1] > raw[2] && norm[1] > norm[2]);
    }
}
