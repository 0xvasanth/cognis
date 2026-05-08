//! Content hashing utilities for deduplication and near-duplicate detection.
//!
//! Provides lightweight, dependency-free hash implementations suitable for
//! content fingerprinting, similarity estimation, and deduplication workflows.
//!
//! # Algorithms
//!
//! - **FNV-1a** — Fast, non-cryptographic hash with good distribution.
//! - **DJB2** — Simple string hash by Daniel J. Bernstein.
//! - **SimHash** — Locality-sensitive hash for near-duplicate detection.
//!
//! # Examples
//!
//! ```rust
//! use cognis_core::utils::hashing::{fnv1a, djb2, ContentFingerprint};
//!
//! let h = fnv1a(b"hello world");
//! assert_ne!(h, 0);
//!
//! let fp = ContentFingerprint::new("hello world");
//! assert!(fp.fnv1a != 0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// FNV-1a
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS_64: u64 = 0xcbf29ce484222325;
const FNV_PRIME_64: u64 = 0x00000100000001B3;

const FNV_OFFSET_BASIS_32: u32 = 0x811c9dc5;
const FNV_PRIME_32: u32 = 0x01000193;

/// Compute a 64-bit FNV-1a hash of the given bytes.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

/// Compute a 32-bit FNV-1a hash of the given bytes.
pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash = FNV_OFFSET_BASIS_32;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME_32);
    }
    hash
}

/// Compute a 64-bit FNV-1a hash of a string.
pub fn fnv1a_str(s: &str) -> u64 {
    fnv1a(s.as_bytes())
}

// ---------------------------------------------------------------------------
// DJB2
// ---------------------------------------------------------------------------

/// Compute a DJB2 hash of the given bytes.
///
/// Uses the classic `hash = hash * 33 + byte` recurrence.
pub fn djb2(data: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &byte in data {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

/// Compute a DJB2 hash of a string.
pub fn djb2_str(s: &str) -> u64 {
    djb2(s.as_bytes())
}

// ---------------------------------------------------------------------------
// SimHash (locality-sensitive hashing)
// ---------------------------------------------------------------------------

/// Compute the SimHash of a text string.
///
/// SimHash produces a fingerprint such that similar documents yield fingerprints
/// with small Hamming distance. The text is split into word-level shingles of
/// the given `ngram_size` (default 3).
pub fn simhash(text: &str, ngram_size: usize) -> u64 {
    let ngram_size = if ngram_size == 0 { 3 } else { ngram_size };
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return 0;
    }

    let mut v = [0i64; 64];

    let shingle_count = if words.len() >= ngram_size {
        words.len() - ngram_size + 1
    } else {
        1
    };

    for i in 0..shingle_count {
        let end = std::cmp::min(i + ngram_size, words.len());
        let shingle: String = words[i..end].join(" ");
        let hash = fnv1a(shingle.as_bytes());

        for (bit, val) in v.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *val += 1;
            } else {
                *val -= 1;
            }
        }
    }

    let mut result: u64 = 0;
    for (bit, val) in v.iter().enumerate() {
        if *val > 0 {
            result |= 1u64 << bit;
        }
    }
    result
}

/// Compute the SimHash of text using the default ngram size (3).
pub fn simhash_default(text: &str) -> u64 {
    simhash(text, 3)
}

/// Compute the Hamming distance between two 64-bit values.
///
/// The Hamming distance is the number of bit positions where the two values
/// differ. For SimHash fingerprints, a smaller distance indicates greater
/// similarity.
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Check whether two SimHash values are "near duplicates" given a threshold.
///
/// `threshold` is the maximum allowed Hamming distance (inclusive).
/// A typical threshold for near-duplicate text is 3–10 depending on content
/// length and desired precision.
pub fn is_near_duplicate(hash_a: u64, hash_b: u64, threshold: u32) -> bool {
    hamming_distance(hash_a, hash_b) <= threshold
}

// ---------------------------------------------------------------------------
// Content Fingerprint
// ---------------------------------------------------------------------------

/// A multi-algorithm fingerprint of a piece of content.
///
/// Combines FNV-1a, DJB2, and SimHash to provide a robust content identifier
/// that supports both exact and near-duplicate matching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ContentFingerprint {
    /// 64-bit FNV-1a hash of the raw bytes.
    pub fnv1a: u64,
    /// 64-bit DJB2 hash of the raw bytes.
    pub djb2: u64,
    /// 64-bit SimHash of the text (0 if input was not valid UTF-8).
    pub simhash: u64,
    /// Length of the original content in bytes.
    pub content_length: usize,
}

impl ContentFingerprint {
    /// Create a fingerprint from a string.
    pub fn new(text: &str) -> Self {
        let bytes = text.as_bytes();
        Self {
            fnv1a: fnv1a(bytes),
            djb2: djb2(bytes),
            simhash: simhash_default(text),
            content_length: bytes.len(),
        }
    }

    /// Create a fingerprint from raw bytes.
    ///
    /// SimHash is computed only if the bytes are valid UTF-8; otherwise it is 0.
    pub fn from_bytes(data: &[u8]) -> Self {
        let sim = match std::str::from_utf8(data) {
            Ok(s) => simhash_default(s),
            Err(_) => 0,
        };
        Self {
            fnv1a: fnv1a(data),
            djb2: djb2(data),
            simhash: sim,
            content_length: data.len(),
        }
    }

    /// Create a fingerprint with a custom SimHash ngram size.
    pub fn with_ngram_size(text: &str, ngram_size: usize) -> Self {
        let bytes = text.as_bytes();
        Self {
            fnv1a: fnv1a(bytes),
            djb2: djb2(bytes),
            simhash: simhash(text, ngram_size),
            content_length: bytes.len(),
        }
    }

    /// Check whether this fingerprint is an exact match with another.
    ///
    /// Both FNV-1a and DJB2 must match and the content lengths must be equal.
    pub fn is_exact_match(&self, other: &Self) -> bool {
        self.fnv1a == other.fnv1a
            && self.djb2 == other.djb2
            && self.content_length == other.content_length
    }

    /// Check whether this fingerprint is a near-duplicate of another.
    pub fn is_near_duplicate(&self, other: &Self, threshold: u32) -> bool {
        is_near_duplicate(self.simhash, other.simhash, threshold)
    }

    /// Return the Hamming distance between the SimHash values.
    pub fn simhash_distance(&self, other: &Self) -> u32 {
        hamming_distance(self.simhash, other.simhash)
    }

    /// Produce a combined 128-bit fingerprint as `(fnv1a, djb2)`.
    pub fn combined_hash(&self) -> (u64, u64) {
        (self.fnv1a, self.djb2)
    }
}

impl std::fmt::Display for ContentFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fingerprint(fnv1a={:#018x}, djb2={:#018x}, simhash={:#018x}, len={})",
            self.fnv1a, self.djb2, self.simhash, self.content_length
        )
    }
}

// ---------------------------------------------------------------------------
// Hash-based Deduplication
// ---------------------------------------------------------------------------

/// A set-based deduplicator that tracks content fingerprints.
///
/// Use this to efficiently detect and skip duplicate content during ingestion
/// or processing pipelines.
#[derive(Debug, Clone)]
pub struct HashDeduplicator {
    /// Stored fingerprints (FNV-1a values used as primary key).
    seen_fnv: HashSet<u64>,
    /// Stored combined hashes for stronger duplicate confirmation.
    seen_combined: HashSet<(u64, u64)>,
    /// SimHash fingerprints for near-duplicate detection.
    simhashes: Vec<u64>,
    /// Hamming distance threshold for near-duplicate detection.
    near_dup_threshold: u32,
}

impl HashDeduplicator {
    /// Create a new deduplicator with the given near-duplicate threshold.
    pub fn new(near_dup_threshold: u32) -> Self {
        Self {
            seen_fnv: HashSet::new(),
            seen_combined: HashSet::new(),
            simhashes: Vec::new(),
            near_dup_threshold,
        }
    }

    /// Create a deduplicator using exact matching only (threshold = 0).
    pub fn exact_only() -> Self {
        Self::new(0)
    }

    /// Check whether the content has been seen before (exact match).
    pub fn is_duplicate(&self, content: &str) -> bool {
        let h = fnv1a_str(content);
        self.seen_fnv.contains(&h)
    }

    /// Check whether the content is a near-duplicate of any previously seen content.
    pub fn is_near_duplicate(&self, content: &str) -> bool {
        if self.near_dup_threshold == 0 {
            return self.is_duplicate(content);
        }
        let sh = simhash_default(content);
        self.simhashes
            .iter()
            .any(|&existing| hamming_distance(existing, sh) <= self.near_dup_threshold)
    }

    /// Insert content into the deduplicator. Returns `true` if the content was
    /// new (not a duplicate), `false` if it was already present.
    pub fn insert(&mut self, content: &str) -> bool {
        let h = fnv1a_str(content);
        if !self.seen_fnv.insert(h) {
            return false;
        }
        let combined = (h, djb2_str(content));
        self.seen_combined.insert(combined);
        self.simhashes.push(simhash_default(content));
        true
    }

    /// Insert and also check for near-duplicates. Returns `true` only if the
    /// content is neither an exact duplicate nor a near-duplicate.
    pub fn insert_if_unique(&mut self, content: &str) -> bool {
        if self.is_duplicate(content) {
            return false;
        }
        if self.near_dup_threshold > 0 && self.is_near_duplicate(content) {
            return false;
        }
        self.insert(content);
        true
    }

    /// Return the number of unique items tracked.
    pub fn len(&self) -> usize {
        self.seen_fnv.len()
    }

    /// Return `true` if no items have been inserted.
    pub fn is_empty(&self) -> bool {
        self.seen_fnv.is_empty()
    }

    /// Clear all tracked fingerprints.
    pub fn clear(&mut self) {
        self.seen_fnv.clear();
        self.seen_combined.clear();
        self.simhashes.clear();
    }

    /// Deduplicate a list of strings, returning only unique entries.
    pub fn deduplicate<'a>(&mut self, items: &[&'a str]) -> Vec<&'a str> {
        items
            .iter()
            .filter(|&&item| self.insert_if_unique(item))
            .copied()
            .collect()
    }
}

impl Default for HashDeduplicator {
    fn default() -> Self {
        Self::new(3)
    }
}

// ---------------------------------------------------------------------------
// HashBuilder — configurable hashing pipeline
// ---------------------------------------------------------------------------

/// Algorithm selection for [`HashBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Fnv1a,
    Fnv1a32,
    Djb2,
    SimHash,
}

/// A builder for configuring and executing a hashing pipeline.
///
/// Allows selecting algorithms, normalizing input, and producing one or more
/// hash values in a single pass.
#[derive(Debug, Clone)]
pub struct HashBuilder {
    algorithms: Vec<HashAlgorithm>,
    normalize_whitespace: bool,
    to_lowercase: bool,
    ngram_size: usize,
}

impl HashBuilder {
    /// Create a new builder with no algorithms selected.
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
            normalize_whitespace: false,
            to_lowercase: false,
            ngram_size: 3,
        }
    }

    /// Add an algorithm to the pipeline.
    pub fn algorithm(mut self, algo: HashAlgorithm) -> Self {
        self.algorithms.push(algo);
        self
    }

    /// Enable all available algorithms.
    pub fn all_algorithms(mut self) -> Self {
        self.algorithms = vec![
            HashAlgorithm::Fnv1a,
            HashAlgorithm::Fnv1a32,
            HashAlgorithm::Djb2,
            HashAlgorithm::SimHash,
        ];
        self
    }

    /// Enable whitespace normalization (collapse runs of whitespace to a single space).
    pub fn normalize_whitespace(mut self, yes: bool) -> Self {
        self.normalize_whitespace = yes;
        self
    }

    /// Enable lowercasing before hashing.
    pub fn to_lowercase(mut self, yes: bool) -> Self {
        self.to_lowercase = yes;
        self
    }

    /// Set the ngram size for SimHash.
    pub fn ngram_size(mut self, size: usize) -> Self {
        self.ngram_size = size;
        self
    }

    /// Normalize the input text according to the builder settings.
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        let mut result: std::borrow::Cow<'a, str> = std::borrow::Cow::Borrowed(text);

        if self.to_lowercase {
            result = std::borrow::Cow::Owned(result.to_lowercase());
        }

        if self.normalize_whitespace {
            let normalized: String = result.split_whitespace().collect::<Vec<_>>().join(" ");
            result = std::borrow::Cow::Owned(normalized);
        }

        result
    }

    /// Hash the given text with all configured algorithms.
    ///
    /// Returns a map of algorithm → hash value. SimHash is stored as a `u64`;
    /// FNV-1a 32-bit is zero-extended to `u64`.
    pub fn hash(&self, text: &str) -> HashResult {
        let normalized = self.normalize(text);
        let bytes = normalized.as_bytes();

        let mut result = HashResult::default();

        for algo in &self.algorithms {
            match algo {
                HashAlgorithm::Fnv1a => result.fnv1a = Some(fnv1a(bytes)),
                HashAlgorithm::Fnv1a32 => result.fnv1a_32 = Some(fnv1a_32(bytes)),
                HashAlgorithm::Djb2 => result.djb2 = Some(djb2(bytes)),
                HashAlgorithm::SimHash => {
                    result.simhash = Some(simhash(&normalized, self.ngram_size));
                }
            }
        }

        result
    }

    /// Hash raw bytes (SimHash is skipped if bytes are not valid UTF-8).
    pub fn hash_bytes(&self, data: &[u8]) -> HashResult {
        let text = std::str::from_utf8(data).ok();

        let mut result = HashResult::default();

        for algo in &self.algorithms {
            match algo {
                HashAlgorithm::Fnv1a => result.fnv1a = Some(fnv1a(data)),
                HashAlgorithm::Fnv1a32 => result.fnv1a_32 = Some(fnv1a_32(data)),
                HashAlgorithm::Djb2 => result.djb2 = Some(djb2(data)),
                HashAlgorithm::SimHash => {
                    if let Some(t) = text {
                        result.simhash = Some(simhash(t, self.ngram_size));
                    }
                }
            }
        }

        result
    }
}

impl Default for HashBuilder {
    fn default() -> Self {
        Self::new()
            .algorithm(HashAlgorithm::Fnv1a)
            .algorithm(HashAlgorithm::Djb2)
    }
}

/// Result of a [`HashBuilder`] hashing pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HashResult {
    pub fnv1a: Option<u64>,
    pub fnv1a_32: Option<u32>,
    pub djb2: Option<u64>,
    pub simhash: Option<u64>,
}

impl HashResult {
    /// Return the first available hash value (preferring FNV-1a).
    pub fn primary_hash(&self) -> Option<u64> {
        self.fnv1a
            .or(self.djb2)
            .or(self.simhash)
            .or(self.fnv1a_32.map(|v| v as u64))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FNV-1a tests ----

    #[test]
    fn test_fnv1a_empty() {
        let h = fnv1a(b"");
        assert_eq!(h, FNV_OFFSET_BASIS_64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a(b"hello");
        let b = fnv1a(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
    }

    #[test]
    fn test_fnv1a_known_value() {
        // FNV-1a of empty should be the offset basis
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
    }

    #[test]
    fn test_fnv1a_str() {
        assert_eq!(fnv1a_str("hello"), fnv1a(b"hello"));
    }

    #[test]
    fn test_fnv1a_32_empty() {
        assert_eq!(fnv1a_32(b""), FNV_OFFSET_BASIS_32);
    }

    #[test]
    fn test_fnv1a_32_deterministic() {
        assert_eq!(fnv1a_32(b"test"), fnv1a_32(b"test"));
    }

    #[test]
    fn test_fnv1a_32_different_from_64() {
        // 32-bit and 64-bit should generally differ (cast to u64 to compare)
        let h32 = fnv1a_32(b"hello") as u64;
        let h64 = fnv1a(b"hello");
        assert_ne!(h32, h64);
    }

    // ---- DJB2 tests ----

    #[test]
    fn test_djb2_empty() {
        assert_eq!(djb2(b""), 5381);
    }

    #[test]
    fn test_djb2_deterministic() {
        assert_eq!(djb2(b"hello"), djb2(b"hello"));
    }

    #[test]
    fn test_djb2_different_inputs() {
        assert_ne!(djb2(b"hello"), djb2(b"world"));
    }

    #[test]
    fn test_djb2_str() {
        assert_eq!(djb2_str("test"), djb2(b"test"));
    }

    #[test]
    fn test_djb2_vs_fnv1a() {
        // Different algorithms should usually produce different values
        assert_ne!(djb2(b"hello"), fnv1a(b"hello"));
    }

    // ---- SimHash tests ----

    #[test]
    fn test_simhash_empty() {
        assert_eq!(simhash("", 3), 0);
    }

    #[test]
    fn test_simhash_deterministic() {
        let a = simhash("the quick brown fox", 3);
        let b = simhash("the quick brown fox", 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_simhash_similar_texts() {
        let a = simhash("the quick brown fox jumps over the lazy dog", 3);
        let b = simhash("the quick brown fox leaps over the lazy dog", 3);
        let dist = hamming_distance(a, b);
        // Similar texts should have small Hamming distance
        assert!(dist < 20, "expected small distance, got {}", dist);
    }

    #[test]
    fn test_simhash_different_texts() {
        let a = simhash("the quick brown fox jumps over the lazy dog", 3);
        let b = simhash(
            "completely unrelated text about quantum physics and mathematics",
            3,
        );
        let dist = hamming_distance(a, b);
        // Very different texts often have larger distance
        assert!(dist > 0, "expected non-zero distance");
    }

    #[test]
    fn test_simhash_single_word() {
        let h = simhash("hello", 3);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_simhash_default() {
        assert_eq!(
            simhash_default("test text here"),
            simhash("test text here", 3)
        );
    }

    #[test]
    fn test_simhash_ngram_size_variation() {
        let a = simhash("the quick brown fox jumps", 2);
        let b = simhash("the quick brown fox jumps", 4);
        // Different ngram sizes may produce different hashes
        // (not guaranteed, but usually true for non-trivial text)
        let _ = (a, b); // just ensure they compile and run
    }

    // ---- Hamming distance tests ----

    #[test]
    fn test_hamming_distance_same() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(u64::MAX, u64::MAX), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        assert_eq!(hamming_distance(0, u64::MAX), 64);
    }

    #[test]
    fn test_hamming_distance_one_bit() {
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0, 2), 1);
    }

    #[test]
    fn test_hamming_distance_symmetric() {
        assert_eq!(hamming_distance(42, 99), hamming_distance(99, 42));
    }

    #[test]
    fn test_is_near_duplicate_true() {
        assert!(is_near_duplicate(0b1111, 0b1110, 1));
    }

    #[test]
    fn test_is_near_duplicate_false() {
        assert!(!is_near_duplicate(0, u64::MAX, 3));
    }

    // ---- ContentFingerprint tests ----

    #[test]
    fn test_fingerprint_new() {
        let fp = ContentFingerprint::new("hello world");
        assert_ne!(fp.fnv1a, 0);
        assert_ne!(fp.djb2, 0);
        assert_eq!(fp.content_length, 11);
    }

    #[test]
    fn test_fingerprint_from_bytes() {
        let fp = ContentFingerprint::from_bytes(b"hello world");
        assert_eq!(fp.content_length, 11);
        assert_eq!(fp.fnv1a, fnv1a(b"hello world"));
    }

    #[test]
    fn test_fingerprint_from_invalid_utf8() {
        let fp = ContentFingerprint::from_bytes(&[0xff, 0xfe, 0xfd]);
        assert_eq!(fp.simhash, 0);
        assert_eq!(fp.content_length, 3);
    }

    #[test]
    fn test_fingerprint_exact_match() {
        let a = ContentFingerprint::new("hello world");
        let b = ContentFingerprint::new("hello world");
        assert!(a.is_exact_match(&b));
    }

    #[test]
    fn test_fingerprint_no_exact_match() {
        let a = ContentFingerprint::new("hello");
        let b = ContentFingerprint::new("world");
        assert!(!a.is_exact_match(&b));
    }

    #[test]
    fn test_fingerprint_near_duplicate() {
        let a = ContentFingerprint::new("the quick brown fox jumps over the lazy dog");
        let b = ContentFingerprint::new("the quick brown fox leaps over the lazy dog");
        // They should be near-duplicates with a generous threshold
        assert!(a.is_near_duplicate(&b, 20));
    }

    #[test]
    fn test_fingerprint_simhash_distance() {
        let a = ContentFingerprint::new("same text");
        let b = ContentFingerprint::new("same text");
        assert_eq!(a.simhash_distance(&b), 0);
    }

    #[test]
    fn test_fingerprint_combined_hash() {
        let fp = ContentFingerprint::new("test");
        let (f, d) = fp.combined_hash();
        assert_eq!(f, fp.fnv1a);
        assert_eq!(d, fp.djb2);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = ContentFingerprint::new("test");
        let s = format!("{}", fp);
        assert!(s.contains("Fingerprint"));
        assert!(s.contains("fnv1a="));
    }

    #[test]
    fn test_fingerprint_with_ngram_size() {
        let fp = ContentFingerprint::with_ngram_size("hello world test", 2);
        assert_ne!(fp.simhash, 0);
    }

    #[test]
    fn test_fingerprint_serde_roundtrip() {
        let fp = ContentFingerprint::new("serialization test");
        let json = serde_json::to_string(&fp).unwrap();
        let fp2: ContentFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(fp, fp2);
    }

    // ---- HashDeduplicator tests ----

    #[test]
    fn test_dedup_empty() {
        let d = HashDeduplicator::new(3);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_dedup_insert_new() {
        let mut d = HashDeduplicator::new(3);
        assert!(d.insert("hello"));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_dedup_insert_duplicate() {
        let mut d = HashDeduplicator::new(3);
        assert!(d.insert("hello"));
        assert!(!d.insert("hello"));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_dedup_is_duplicate() {
        let mut d = HashDeduplicator::new(3);
        d.insert("hello");
        assert!(d.is_duplicate("hello"));
        assert!(!d.is_duplicate("world"));
    }

    #[test]
    fn test_dedup_exact_only() {
        let mut d = HashDeduplicator::exact_only();
        assert!(d.insert_if_unique("hello"));
        assert!(!d.insert_if_unique("hello"));
    }

    #[test]
    fn test_dedup_clear() {
        let mut d = HashDeduplicator::new(3);
        d.insert("hello");
        d.insert("world");
        assert_eq!(d.len(), 2);
        d.clear();
        assert!(d.is_empty());
    }

    #[test]
    fn test_dedup_deduplicate_list() {
        let mut d = HashDeduplicator::exact_only();
        let items = vec!["hello", "world", "hello", "foo", "world"];
        let result = d.deduplicate(&items);
        assert_eq!(result, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_dedup_default() {
        let d = HashDeduplicator::default();
        assert_eq!(d.near_dup_threshold, 3);
    }

    #[test]
    fn test_dedup_insert_if_unique_exact_dup() {
        let mut d = HashDeduplicator::new(5);
        assert!(d.insert_if_unique("hello world"));
        assert!(!d.insert_if_unique("hello world"));
    }

    // ---- HashBuilder tests ----

    #[test]
    fn test_builder_default() {
        let builder = HashBuilder::default();
        let result = builder.hash("test");
        assert!(result.fnv1a.is_some());
        assert!(result.djb2.is_some());
        assert!(result.simhash.is_none());
    }

    #[test]
    fn test_builder_all_algorithms() {
        let builder = HashBuilder::new().all_algorithms();
        let result = builder.hash("test text for hashing");
        assert!(result.fnv1a.is_some());
        assert!(result.fnv1a_32.is_some());
        assert!(result.djb2.is_some());
        assert!(result.simhash.is_some());
    }

    #[test]
    fn test_builder_single_algorithm() {
        let builder = HashBuilder::new().algorithm(HashAlgorithm::Fnv1a);
        let result = builder.hash("test");
        assert!(result.fnv1a.is_some());
        assert!(result.djb2.is_none());
    }

    #[test]
    fn test_builder_normalize_whitespace() {
        let builder = HashBuilder::new()
            .algorithm(HashAlgorithm::Fnv1a)
            .normalize_whitespace(true);
        let a = builder.hash("hello   world");
        let b = builder.hash("hello world");
        assert_eq!(a.fnv1a, b.fnv1a);
    }

    #[test]
    fn test_builder_to_lowercase() {
        let builder = HashBuilder::new()
            .algorithm(HashAlgorithm::Fnv1a)
            .to_lowercase(true);
        let a = builder.hash("HELLO");
        let b = builder.hash("hello");
        assert_eq!(a.fnv1a, b.fnv1a);
    }

    #[test]
    fn test_builder_combined_normalization() {
        let builder = HashBuilder::new()
            .algorithm(HashAlgorithm::Djb2)
            .normalize_whitespace(true)
            .to_lowercase(true);
        let a = builder.hash("  HELLO   WORLD  ");
        let b = builder.hash("hello world");
        assert_eq!(a.djb2, b.djb2);
    }

    #[test]
    fn test_builder_hash_bytes() {
        let builder = HashBuilder::new().algorithm(HashAlgorithm::Fnv1a);
        let result = builder.hash_bytes(b"hello");
        assert!(result.fnv1a.is_some());
    }

    #[test]
    fn test_builder_hash_bytes_invalid_utf8() {
        let builder = HashBuilder::new()
            .algorithm(HashAlgorithm::Fnv1a)
            .algorithm(HashAlgorithm::SimHash);
        let result = builder.hash_bytes(&[0xff, 0xfe]);
        assert!(result.fnv1a.is_some());
        assert!(result.simhash.is_none()); // invalid UTF-8, SimHash skipped
    }

    #[test]
    fn test_builder_ngram_size() {
        let builder = HashBuilder::new()
            .algorithm(HashAlgorithm::SimHash)
            .ngram_size(2);
        let result = builder.hash("the quick brown fox");
        assert!(result.simhash.is_some());
    }

    // ---- HashResult tests ----

    #[test]
    fn test_hash_result_primary_hash_fnv1a() {
        let r = HashResult {
            fnv1a: Some(42),
            djb2: Some(99),
            ..Default::default()
        };
        assert_eq!(r.primary_hash(), Some(42));
    }

    #[test]
    fn test_hash_result_primary_hash_fallback() {
        let r = HashResult {
            djb2: Some(99),
            ..Default::default()
        };
        assert_eq!(r.primary_hash(), Some(99));
    }

    #[test]
    fn test_hash_result_primary_hash_none() {
        let r = HashResult::default();
        assert_eq!(r.primary_hash(), None);
    }

    #[test]
    fn test_hash_result_serde() {
        let r = HashResult {
            fnv1a: Some(123),
            djb2: Some(456),
            ..Default::default()
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: HashResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2);
    }
}
