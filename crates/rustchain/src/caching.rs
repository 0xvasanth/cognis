//! API response caching for LLM calls.
//!
//! This module reduces costs and latency by caching identical (or semantically
//! similar) requests. It provides:
//!
//! - [`CacheKey`] — deterministic key derived from request parameters.
//! - [`CacheEntry`] — a cached response with metadata and TTL support.
//! - [`CacheStore`] — abstract trait for pluggable cache backends.
//! - [`InMemoryCache`] — `HashMap`-backed store with LRU eviction.
//! - [`SemanticCache`] — approximate matching based on embedding similarity.
//! - [`CachePolicy`] — rules that control when caching is applied.
//! - [`CacheStats`] — counters for hit rate, evictions, and estimated savings.
//! - [`CacheWarmer`] — pre-populates a cache with common queries.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_iso() -> String {
    // Simple ISO-ish timestamp from epoch seconds.
    let secs = now_secs();
    format!("{secs}")
}

fn simple_sha256_hex(input: &str) -> String {
    // A basic deterministic hash using the standard library's DefaultHasher.
    // Not cryptographic, but sufficient for cache-key deduplication.
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    let v = h.finish();
    format!("{v:016x}")
}

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// Deterministic key derived from request parameters (model, messages,
/// temperature, tools).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheKey {
    raw: String,
}

impl CacheKey {
    /// Build a key from the constituent request parts.
    pub fn from_parts(
        model: &str,
        messages: &[Value],
        temperature: Option<f64>,
        tools: Option<&[Value]>,
    ) -> Self {
        let mut raw = String::new();
        raw.push_str("model:");
        raw.push_str(model);
        raw.push_str("|messages:");
        for m in messages {
            raw.push_str(&m.to_string());
            raw.push(';');
        }
        if let Some(t) = temperature {
            raw.push_str(&format!("|temp:{t}"));
        }
        if let Some(ts) = tools {
            raw.push_str("|tools:");
            for t in ts {
                raw.push_str(&t.to_string());
                raw.push(';');
            }
        }
        Self { raw }
    }

    /// Return the raw key string.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Return a hex-encoded hash of the key.
    pub fn hash_hex(&self) -> String {
        simple_sha256_hex(&self.raw)
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A cached LLM response together with metadata.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub response: Value,
    pub model: String,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub hit_count: usize,
    pub token_count: usize,
    /// Internal: epoch-seconds when the entry was created.
    created_epoch: u64,
    /// Internal: epoch-seconds when the entry expires (if set).
    expires_epoch: Option<u64>,
    /// Internal: last-access epoch (for LRU eviction).
    last_access: u64,
}

impl CacheEntry {
    /// Create a new entry for the given response and model.
    pub fn new(response: Value, model: impl Into<String>) -> Self {
        let epoch = now_secs();
        Self {
            response,
            model: model.into(),
            created_at: now_iso(),
            expires_at: None,
            hit_count: 0,
            token_count: 0,
            created_epoch: epoch,
            expires_epoch: None,
            last_access: epoch,
        }
    }

    /// Set a TTL in seconds from the creation time.
    pub fn with_ttl_secs(mut self, secs: u64) -> Self {
        let exp = self.created_epoch + secs;
        self.expires_epoch = Some(exp);
        self.expires_at = Some(format!("{exp}"));
        self
    }

    /// Whether the entry has passed its expiration time.
    pub fn is_expired(&self) -> bool {
        match self.expires_epoch {
            Some(exp) => now_secs() >= exp,
            None => false,
        }
    }

    /// Record a cache hit (increments counter and updates last-access).
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_access = now_secs();
    }

    /// Seconds elapsed since creation.
    pub fn age_secs(&self) -> u64 {
        now_secs().saturating_sub(self.created_epoch)
    }

    /// Serialize metadata to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "response": self.response,
            "model": self.model,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "token_count": self.token_count,
        })
    }
}

// ---------------------------------------------------------------------------
// CacheStore trait
// ---------------------------------------------------------------------------

/// Abstract cache storage backend.
pub trait CacheStore {
    /// Retrieve a mutable reference to an entry (for recording hits).
    fn get(&mut self, key: &CacheKey) -> Option<&mut CacheEntry>;
    /// Insert an entry.
    fn put(&mut self, key: CacheKey, entry: CacheEntry);
    /// Remove an entry; returns `true` if it existed.
    fn remove(&mut self, key: &CacheKey) -> bool;
    /// Check membership.
    fn contains(&self, key: &CacheKey) -> bool;
    /// Number of stored entries.
    fn len(&self) -> usize;
    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Remove all entries.
    fn clear(&mut self);
}

// ---------------------------------------------------------------------------
// InMemoryCache
// ---------------------------------------------------------------------------

/// HashMap-backed cache with a maximum capacity and LRU eviction.
pub struct InMemoryCache {
    entries: HashMap<CacheKey, CacheEntry>,
    max_entries: usize,
    default_ttl: Option<u64>,
}

impl InMemoryCache {
    /// Create a new in-memory cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            default_ttl: None,
        }
    }

    /// Set a default TTL (seconds) applied to every inserted entry.
    pub fn with_default_ttl(mut self, secs: u64) -> Self {
        self.default_ttl = Some(secs);
        self
    }

    /// Evict expired entries first, then the least-recently-used entry if
    /// the cache is still at capacity.
    fn evict_if_needed(&mut self) {
        // 1. Remove all expired entries.
        self.entries.retain(|_, e| !e.is_expired());

        // 2. If still at capacity, remove the LRU entry.
        while self.entries.len() >= self.max_entries {
            if let Some(lru_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&lru_key);
            } else {
                break;
            }
        }
    }
}

impl CacheStore for InMemoryCache {
    fn get(&mut self, key: &CacheKey) -> Option<&mut CacheEntry> {
        // Remove if expired, then return reference.
        if let Some(entry) = self.entries.get(key) {
            if entry.is_expired() {
                self.entries.remove(key);
                return None;
            }
        }
        self.entries.get_mut(key)
    }

    fn put(&mut self, key: CacheKey, mut entry: CacheEntry) {
        if let Some(ttl) = self.default_ttl {
            if entry.expires_epoch.is_none() {
                let exp = entry.created_epoch + ttl;
                entry.expires_epoch = Some(exp);
                entry.expires_at = Some(format!("{exp}"));
            }
        }
        // Evict before insertion so we always have room.
        if !self.entries.contains_key(&key) {
            self.evict_if_needed();
        }
        self.entries.insert(key, entry);
    }

    fn remove(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(key).is_some()
    }

    fn contains(&self, key: &CacheKey) -> bool {
        if let Some(e) = self.entries.get(key) {
            !e.is_expired()
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// SemanticCache
// ---------------------------------------------------------------------------

/// Approximate-matching cache that compares embeddings via cosine similarity.
pub struct SemanticCache {
    similarity_threshold: f64,
    entries: Vec<SemanticEntry>,
}

struct SemanticEntry {
    #[allow(dead_code)]
    key: String,
    embedding: Vec<f64>,
    entry: CacheEntry,
}

impl SemanticCache {
    /// Create a new semantic cache.
    ///
    /// `similarity_threshold` should be in `[0.0, 1.0]`; higher means
    /// stricter matching.
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
            entries: Vec::new(),
        }
    }

    /// Insert an entry with its embedding vector.
    pub fn put(&mut self, key: String, embedding: Vec<f64>, entry: CacheEntry) {
        self.entries.push(SemanticEntry {
            key,
            embedding,
            entry,
        });
    }

    /// Find the closest entry whose cosine similarity is at or above the
    /// threshold. Returns `None` if no entry qualifies.
    pub fn get_similar(&self, embedding: &[f64]) -> Option<&CacheEntry> {
        let mut best: Option<(f64, usize)> = None;
        for (i, se) in self.entries.iter().enumerate() {
            let sim = cosine_similarity(&se.embedding, embedding);
            if sim >= self.similarity_threshold && best.is_none_or(|(b, _)| sim > b) {
                best = Some((sim, i));
            }
        }
        best.map(|(_, idx)| &self.entries[idx].entry)
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

// ---------------------------------------------------------------------------
// CachePolicy
// ---------------------------------------------------------------------------

/// Controls whether a given request/response pair should be cached.
pub struct CachePolicy {
    min_tokens: Option<usize>,
    skip_tools: bool,
    skip_streaming: bool,
    max_ttl: Option<u64>,
}

impl CachePolicy {
    pub fn new() -> Self {
        Self {
            min_tokens: None,
            skip_tools: false,
            skip_streaming: false,
            max_ttl: None,
        }
    }

    /// Only cache if the response exceeds this token count.
    pub fn cache_if_tokens_above(mut self, min_tokens: usize) -> Self {
        self.min_tokens = Some(min_tokens);
        self
    }

    /// Do not cache responses where tools were invoked.
    pub fn skip_if_tools_used(mut self) -> Self {
        self.skip_tools = true;
        self
    }

    /// Do not cache streaming requests.
    pub fn skip_if_streaming(mut self) -> Self {
        self.skip_streaming = true;
        self
    }

    /// Set a maximum TTL in seconds.
    pub fn max_ttl_secs(mut self, secs: u64) -> Self {
        self.max_ttl = Some(secs);
        self
    }

    /// Evaluate whether the request + response should be cached.
    pub fn should_cache(&self, request: &Value, response: &Value) -> bool {
        // Check streaming flag in request.
        if self.skip_streaming {
            if let Some(s) = request.get("stream") {
                if s.as_bool() == Some(true) {
                    return false;
                }
            }
        }

        // Check tools in request.
        if self.skip_tools {
            if let Some(tools) = request.get("tools") {
                if tools.is_array() && !tools.as_array().unwrap().is_empty() {
                    return false;
                }
            }
        }

        // Check minimum token count in response.
        if let Some(min) = self.min_tokens {
            let tokens = response
                .get("usage")
                .and_then(|u| u.get("total_tokens"))
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as usize;
            if tokens < min {
                return false;
            }
        }

        true
    }

    /// Serialize policy settings to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "min_tokens": self.min_tokens,
            "skip_tools": self.skip_tools,
            "skip_streaming": self.skip_streaming,
            "max_ttl_secs": self.max_ttl,
        })
    }
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Tracks cache performance metrics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    hits: usize,
    misses: usize,
    evictions: usize,
    insertions: usize,
    saved_tokens: usize,
}

impl CacheStats {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            insertions: 0,
            saved_tokens: 0,
        }
    }

    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    pub fn record_insertion(&mut self) {
        self.insertions += 1;
    }

    /// Record tokens saved by a cache hit.
    pub fn record_saved_tokens(&mut self, tokens: usize) {
        self.saved_tokens += tokens;
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.total_lookups();
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    pub fn total_hits(&self) -> usize {
        self.hits
    }

    pub fn total_misses(&self) -> usize {
        self.misses
    }

    pub fn total_evictions(&self) -> usize {
        self.evictions
    }

    pub fn total_lookups(&self) -> usize {
        self.hits + self.misses
    }

    pub fn estimated_savings_tokens(&self) -> usize {
        self.saved_tokens
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "insertions": self.insertions,
            "hit_rate": self.hit_rate(),
            "total_lookups": self.total_lookups(),
            "estimated_savings_tokens": self.saved_tokens,
        })
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CacheWarmer
// ---------------------------------------------------------------------------

/// Pre-populates a [`CacheStore`] with known queries.
pub struct CacheWarmer {
    queries: Vec<(CacheKey, CacheEntry)>,
}

impl CacheWarmer {
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    /// Add a query to be warmed.
    pub fn add_query(&mut self, key: CacheKey, entry: CacheEntry) {
        self.queries.push((key, entry));
    }

    /// Insert all queued queries into the given store. Returns the number
    /// of entries inserted.
    pub fn warm(&mut self, store: &mut dyn CacheStore) -> usize {
        let count = self.queries.len();
        for (key, entry) in self.queries.drain(..) {
            store.put(key, entry);
        }
        count
    }

    /// Number of queries queued for warming.
    pub fn queries_count(&self) -> usize {
        self.queries.len()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "queries_count": self.queries.len(),
        })
    }
}

impl Default for CacheWarmer {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- CacheKey ----------------------------------------------------------

    #[test]
    fn cache_key_from_parts_basic() {
        let key = CacheKey::from_parts(
            "gpt-4",
            &[json!({"role":"user","content":"hi"})],
            None,
            None,
        );
        assert!(!key.as_str().is_empty());
    }

    #[test]
    fn cache_key_deterministic() {
        let a = CacheKey::from_parts("gpt-4", &[json!("hello")], Some(0.7), None);
        let b = CacheKey::from_parts("gpt-4", &[json!("hello")], Some(0.7), None);
        assert_eq!(a, b);
        assert_eq!(a.hash_hex(), b.hash_hex());
    }

    #[test]
    fn cache_key_different_model() {
        let a = CacheKey::from_parts("gpt-4", &[json!("hi")], None, None);
        let b = CacheKey::from_parts("gpt-3.5", &[json!("hi")], None, None);
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_different_temp() {
        let a = CacheKey::from_parts("m", &[], Some(0.5), None);
        let b = CacheKey::from_parts("m", &[], Some(0.9), None);
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_with_tools() {
        let tools = vec![json!({"name":"calc"})];
        let key = CacheKey::from_parts("m", &[], None, Some(&tools));
        assert!(key.as_str().contains("tools:"));
    }

    #[test]
    fn cache_key_display() {
        let key = CacheKey::from_parts("m", &[], None, None);
        let s = format!("{key}");
        assert_eq!(s, key.as_str());
    }

    #[test]
    fn cache_key_clone() {
        let key = CacheKey::from_parts("m", &[json!("a")], None, None);
        let cloned = key.clone();
        assert_eq!(key, cloned);
    }

    #[test]
    fn cache_key_hash_hex_not_empty() {
        let key = CacheKey::from_parts("m", &[], None, None);
        assert!(!key.hash_hex().is_empty());
    }

    #[test]
    fn cache_key_hash_consistent() {
        let key = CacheKey::from_parts("m", &[json!(1)], Some(0.0), None);
        assert_eq!(key.hash_hex(), key.hash_hex());
    }

    #[test]
    fn cache_key_no_temp() {
        let key = CacheKey::from_parts("m", &[], None, None);
        assert!(!key.as_str().contains("temp:"));
    }

    // ---- CacheEntry --------------------------------------------------------

    #[test]
    fn cache_entry_new() {
        let entry = CacheEntry::new(json!({"text":"hi"}), "gpt-4");
        assert_eq!(entry.model, "gpt-4");
        assert_eq!(entry.hit_count, 0);
        assert_eq!(entry.response, json!({"text":"hi"}));
    }

    #[test]
    fn cache_entry_not_expired_by_default() {
        let entry = CacheEntry::new(json!(null), "m");
        assert!(!entry.is_expired());
    }

    #[test]
    fn cache_entry_with_ttl() {
        let entry = CacheEntry::new(json!(null), "m").with_ttl_secs(3600);
        assert!(entry.expires_at.is_some());
        assert!(!entry.is_expired());
    }

    #[test]
    fn cache_entry_expired_ttl_zero() {
        let mut entry = CacheEntry::new(json!(null), "m");
        // Force expiration in the past.
        entry.expires_epoch = Some(0);
        entry.expires_at = Some("0".into());
        assert!(entry.is_expired());
    }

    #[test]
    fn cache_entry_record_hit() {
        let mut entry = CacheEntry::new(json!(null), "m");
        assert_eq!(entry.hit_count, 0);
        entry.record_hit();
        assert_eq!(entry.hit_count, 1);
        entry.record_hit();
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    fn cache_entry_age_secs() {
        let entry = CacheEntry::new(json!(null), "m");
        // Age should be very small (< 2 seconds).
        assert!(entry.age_secs() < 2);
    }

    #[test]
    fn cache_entry_to_json() {
        let entry = CacheEntry::new(json!("resp"), "gpt-4");
        let j = entry.to_json();
        assert_eq!(j["model"], "gpt-4");
        assert_eq!(j["response"], json!("resp"));
        assert_eq!(j["hit_count"], 0);
    }

    #[test]
    fn cache_entry_to_json_with_ttl() {
        let entry = CacheEntry::new(json!(null), "m").with_ttl_secs(60);
        let j = entry.to_json();
        assert!(j["expires_at"].is_string());
    }

    #[test]
    fn cache_entry_token_count() {
        let mut entry = CacheEntry::new(json!(null), "m");
        entry.token_count = 42;
        assert_eq!(entry.token_count, 42);
    }

    // ---- InMemoryCache (CacheStore) ----------------------------------------

    #[test]
    fn in_memory_put_and_get() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[], None, None);
        cache.put(key.clone(), CacheEntry::new(json!("a"), "m"));
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn in_memory_contains() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[json!(1)], None, None);
        assert!(!cache.contains(&key));
        cache.put(key.clone(), CacheEntry::new(json!(null), "m"));
        assert!(cache.contains(&key));
    }

    #[test]
    fn in_memory_remove() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[], None, None);
        cache.put(key.clone(), CacheEntry::new(json!(null), "m"));
        assert!(cache.remove(&key));
        assert!(!cache.remove(&key));
    }

    #[test]
    fn in_memory_len() {
        let mut cache = InMemoryCache::new(10);
        assert_eq!(cache.len(), 0);
        cache.put(
            CacheKey::from_parts("m", &[json!(1)], None, None),
            CacheEntry::new(json!(null), "m"),
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn in_memory_clear() {
        let mut cache = InMemoryCache::new(10);
        cache.put(
            CacheKey::from_parts("m", &[json!(1)], None, None),
            CacheEntry::new(json!(null), "m"),
        );
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn in_memory_is_empty() {
        let cache = InMemoryCache::new(10);
        assert!(cache.is_empty());
    }

    #[test]
    fn in_memory_evicts_when_full() {
        let mut cache = InMemoryCache::new(2);
        let k1 = CacheKey::from_parts("m", &[json!(1)], None, None);
        let k2 = CacheKey::from_parts("m", &[json!(2)], None, None);
        let k3 = CacheKey::from_parts("m", &[json!(3)], None, None);
        cache.put(k1.clone(), CacheEntry::new(json!(null), "m"));
        cache.put(k2.clone(), CacheEntry::new(json!(null), "m"));
        // This should evict k1 (LRU).
        cache.put(k3.clone(), CacheEntry::new(json!(null), "m"));
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&k3));
    }

    #[test]
    fn in_memory_evicts_expired_first() {
        let mut cache = InMemoryCache::new(2);
        let k1 = CacheKey::from_parts("m", &[json!(1)], None, None);
        let k2 = CacheKey::from_parts("m", &[json!(2)], None, None);
        let k3 = CacheKey::from_parts("m", &[json!(3)], None, None);

        let mut expired = CacheEntry::new(json!(null), "m");
        expired.expires_epoch = Some(0);
        expired.expires_at = Some("0".into());

        cache.put(k1.clone(), expired);
        cache.put(k2.clone(), CacheEntry::new(json!(null), "m"));
        cache.put(k3.clone(), CacheEntry::new(json!(null), "m"));
        // k1 expired -> evicted first; k2 and k3 survive.
        assert!(cache.contains(&k2));
        assert!(cache.contains(&k3));
    }

    #[test]
    fn in_memory_default_ttl() {
        let mut cache = InMemoryCache::new(10).with_default_ttl(3600);
        let key = CacheKey::from_parts("m", &[], None, None);
        cache.put(key.clone(), CacheEntry::new(json!(null), "m"));
        let entry = cache.get(&key).unwrap();
        assert!(entry.expires_at.is_some());
    }

    #[test]
    fn in_memory_get_returns_none_for_expired() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[], None, None);
        let mut entry = CacheEntry::new(json!(null), "m");
        entry.expires_epoch = Some(0);
        entry.expires_at = Some("0".into());
        cache.entries.insert(key.clone(), entry);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn in_memory_contains_false_for_expired() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[], None, None);
        let mut entry = CacheEntry::new(json!(null), "m");
        entry.expires_epoch = Some(0);
        cache.entries.insert(key.clone(), entry);
        assert!(!cache.contains(&key));
    }

    #[test]
    fn in_memory_update_existing_key() {
        let mut cache = InMemoryCache::new(10);
        let key = CacheKey::from_parts("m", &[], None, None);
        cache.put(key.clone(), CacheEntry::new(json!("first"), "m"));
        cache.put(key.clone(), CacheEntry::new(json!("second"), "m"));
        assert_eq!(cache.len(), 1);
        let entry = cache.get(&key).unwrap();
        assert_eq!(entry.response, json!("second"));
    }

    // ---- SemanticCache -----------------------------------------------------

    #[test]
    fn semantic_cache_put_and_get() {
        let mut sc = SemanticCache::new(0.9);
        sc.put(
            "q1".into(),
            vec![1.0, 0.0, 0.0],
            CacheEntry::new(json!("r1"), "m"),
        );
        let result = sc.get_similar(&[1.0, 0.0, 0.0]);
        assert!(result.is_some());
    }

    #[test]
    fn semantic_cache_below_threshold() {
        let sc = SemanticCache::new(0.99);
        // Empty cache => no match.
        assert!(sc.get_similar(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn semantic_cache_similar_vectors() {
        let mut sc = SemanticCache::new(0.95);
        sc.put("q".into(), vec![1.0, 0.0], CacheEntry::new(json!("r"), "m"));
        // Slightly different but still close.
        let result = sc.get_similar(&[0.99, 0.01]);
        assert!(result.is_some());
    }

    #[test]
    fn semantic_cache_orthogonal() {
        let mut sc = SemanticCache::new(0.5);
        sc.put("q".into(), vec![1.0, 0.0], CacheEntry::new(json!("r"), "m"));
        // Orthogonal vector => cosine similarity ~0.
        assert!(sc.get_similar(&[0.0, 1.0]).is_none());
    }

    #[test]
    fn semantic_cache_len() {
        let mut sc = SemanticCache::new(0.9);
        assert_eq!(sc.len(), 0);
        sc.put("q".into(), vec![1.0], CacheEntry::new(json!(null), "m"));
        assert_eq!(sc.len(), 1);
    }

    #[test]
    fn semantic_cache_clear() {
        let mut sc = SemanticCache::new(0.9);
        sc.put("q".into(), vec![1.0], CacheEntry::new(json!(null), "m"));
        sc.clear();
        assert!(sc.is_empty());
    }

    #[test]
    fn semantic_cache_is_empty() {
        let sc = SemanticCache::new(0.9);
        assert!(sc.is_empty());
    }

    #[test]
    fn semantic_cache_best_match() {
        let mut sc = SemanticCache::new(0.8);
        sc.put(
            "far".into(),
            vec![0.5, 0.5],
            CacheEntry::new(json!("far"), "m"),
        );
        sc.put(
            "close".into(),
            vec![1.0, 0.0],
            CacheEntry::new(json!("close"), "m"),
        );
        let result = sc.get_similar(&[0.98, 0.02]).unwrap();
        assert_eq!(result.response, json!("close"));
    }

    // ---- cosine_similarity -------------------------------------------------

    #[test]
    fn cosine_identical() {
        let sim = cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_different_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cosine_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    // ---- CachePolicy -------------------------------------------------------

    #[test]
    fn policy_default_allows_all() {
        let policy = CachePolicy::new();
        assert!(policy.should_cache(&json!({}), &json!({})));
    }

    #[test]
    fn policy_skip_streaming() {
        let policy = CachePolicy::new().skip_if_streaming();
        assert!(!policy.should_cache(&json!({"stream": true}), &json!({})));
        assert!(policy.should_cache(&json!({"stream": false}), &json!({})));
    }

    #[test]
    fn policy_skip_tools() {
        let policy = CachePolicy::new().skip_if_tools_used();
        assert!(!policy.should_cache(&json!({"tools": [{"name":"calc"}]}), &json!({}),));
        assert!(policy.should_cache(&json!({"tools": []}), &json!({})));
    }

    #[test]
    fn policy_min_tokens() {
        let policy = CachePolicy::new().cache_if_tokens_above(100);
        assert!(!policy.should_cache(&json!({}), &json!({"usage": {"total_tokens": 50}}),));
        assert!(policy.should_cache(&json!({}), &json!({"usage": {"total_tokens": 200}}),));
    }

    #[test]
    fn policy_to_json() {
        let policy = CachePolicy::new()
            .cache_if_tokens_above(10)
            .skip_if_streaming()
            .max_ttl_secs(60);
        let j = policy.to_json();
        assert_eq!(j["min_tokens"], 10);
        assert_eq!(j["skip_streaming"], true);
        assert_eq!(j["max_ttl_secs"], 60);
    }

    #[test]
    fn policy_combined() {
        let policy = CachePolicy::new()
            .skip_if_streaming()
            .skip_if_tools_used()
            .cache_if_tokens_above(10);
        // Streaming + tools + low tokens => all fail.
        assert!(!policy.should_cache(
            &json!({"stream": true, "tools": [{"name":"x"}]}),
            &json!({"usage": {"total_tokens": 5}}),
        ));
    }

    #[test]
    fn policy_no_stream_field() {
        let policy = CachePolicy::new().skip_if_streaming();
        // No "stream" field => not streaming => should cache.
        assert!(policy.should_cache(&json!({}), &json!({})));
    }

    #[test]
    fn policy_default_impl() {
        let policy = CachePolicy::default();
        assert!(policy.should_cache(&json!({}), &json!({})));
    }

    // ---- CacheStats --------------------------------------------------------

    #[test]
    fn stats_new() {
        let stats = CacheStats::new();
        assert_eq!(stats.total_hits(), 0);
        assert_eq!(stats.total_misses(), 0);
        assert_eq!(stats.total_evictions(), 0);
        assert_eq!(stats.total_lookups(), 0);
    }

    #[test]
    fn stats_record_hit() {
        let mut stats = CacheStats::new();
        stats.record_hit();
        assert_eq!(stats.total_hits(), 1);
    }

    #[test]
    fn stats_record_miss() {
        let mut stats = CacheStats::new();
        stats.record_miss();
        assert_eq!(stats.total_misses(), 1);
    }

    #[test]
    fn stats_record_eviction() {
        let mut stats = CacheStats::new();
        stats.record_eviction();
        assert_eq!(stats.total_evictions(), 1);
    }

    #[test]
    fn stats_hit_rate() {
        let mut stats = CacheStats::new();
        stats.record_hit();
        stats.record_miss();
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stats_hit_rate_zero() {
        let stats = CacheStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn stats_total_lookups() {
        let mut stats = CacheStats::new();
        stats.record_hit();
        stats.record_hit();
        stats.record_miss();
        assert_eq!(stats.total_lookups(), 3);
    }

    #[test]
    fn stats_estimated_savings() {
        let mut stats = CacheStats::new();
        stats.record_saved_tokens(100);
        stats.record_saved_tokens(200);
        assert_eq!(stats.estimated_savings_tokens(), 300);
    }

    #[test]
    fn stats_to_json() {
        let mut stats = CacheStats::new();
        stats.record_hit();
        stats.record_insertion();
        let j = stats.to_json();
        assert_eq!(j["hits"], 1);
        assert_eq!(j["insertions"], 1);
    }

    #[test]
    fn stats_clone() {
        let mut stats = CacheStats::new();
        stats.record_hit();
        let cloned = stats.clone();
        assert_eq!(cloned.total_hits(), 1);
    }

    #[test]
    fn stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.total_lookups(), 0);
    }

    // ---- CacheWarmer -------------------------------------------------------

    #[test]
    fn warmer_new_empty() {
        let warmer = CacheWarmer::new();
        assert_eq!(warmer.queries_count(), 0);
    }

    #[test]
    fn warmer_add_query() {
        let mut warmer = CacheWarmer::new();
        warmer.add_query(
            CacheKey::from_parts("m", &[], None, None),
            CacheEntry::new(json!(null), "m"),
        );
        assert_eq!(warmer.queries_count(), 1);
    }

    #[test]
    fn warmer_warm() {
        let mut warmer = CacheWarmer::new();
        let key = CacheKey::from_parts("m", &[], None, None);
        warmer.add_query(key.clone(), CacheEntry::new(json!("warm"), "m"));
        let mut store = InMemoryCache::new(10);
        let count = warmer.warm(&mut store);
        assert_eq!(count, 1);
        assert!(store.contains(&key));
    }

    #[test]
    fn warmer_drains_after_warm() {
        let mut warmer = CacheWarmer::new();
        warmer.add_query(
            CacheKey::from_parts("m", &[], None, None),
            CacheEntry::new(json!(null), "m"),
        );
        let mut store = InMemoryCache::new(10);
        warmer.warm(&mut store);
        assert_eq!(warmer.queries_count(), 0);
    }

    #[test]
    fn warmer_to_json() {
        let mut warmer = CacheWarmer::new();
        warmer.add_query(
            CacheKey::from_parts("m", &[], None, None),
            CacheEntry::new(json!(null), "m"),
        );
        let j = warmer.to_json();
        assert_eq!(j["queries_count"], 1);
    }

    #[test]
    fn warmer_default() {
        let warmer = CacheWarmer::default();
        assert_eq!(warmer.queries_count(), 0);
    }

    #[test]
    fn warmer_multiple_queries() {
        let mut warmer = CacheWarmer::new();
        for i in 0..5 {
            warmer.add_query(
                CacheKey::from_parts("m", &[json!(i)], None, None),
                CacheEntry::new(json!(i), "m"),
            );
        }
        let mut store = InMemoryCache::new(10);
        let count = warmer.warm(&mut store);
        assert_eq!(count, 5);
        assert_eq!(store.len(), 5);
    }

    // ---- Integration / misc ------------------------------------------------

    #[test]
    fn full_round_trip() {
        let mut store = InMemoryCache::new(100);
        let mut stats = CacheStats::new();
        let policy = CachePolicy::new();

        let key = CacheKey::from_parts(
            "gpt-4",
            &[json!({"role":"user","content":"hello"})],
            Some(0.7),
            None,
        );
        let request = json!({"model": "gpt-4"});
        let response = json!({"text": "world", "usage": {"total_tokens": 50}});

        // Miss.
        if store.get(&key).is_none() {
            stats.record_miss();
        }

        // Insert if policy allows.
        if policy.should_cache(&request, &response) {
            store.put(key.clone(), CacheEntry::new(response.clone(), "gpt-4"));
            stats.record_insertion();
        }

        // Hit.
        if let Some(entry) = store.get(&key) {
            entry.record_hit();
            stats.record_hit();
        }

        assert_eq!(stats.total_hits(), 1);
        assert_eq!(stats.total_misses(), 1);
    }

    #[test]
    fn cache_key_as_hashmap_key() {
        let mut map = HashMap::new();
        let key = CacheKey::from_parts("m", &[json!("a")], None, None);
        map.insert(key.clone(), 42);
        assert_eq!(map[&key], 42);
    }
}
