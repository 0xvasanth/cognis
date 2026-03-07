//! Memory system for DeepAgents with short-term, long-term, and indexed memory.
//!
//! Mirrors the Python `MemoryMiddleware` for managing agent memory across
//! different timescales and categories.
//!
//! - [`MemoryEntry`] -- a single memory record with metadata
//! - [`MemoryCategory`] -- classification of memory entries
//! - [`ShortTermMemory`] -- bounded, LRU-evicting memory store
//! - [`LongTermMemory`] -- persistent-style memory store
//! - [`MemoryIndex`] -- term-frequency based search index
//! - [`MemoryManager`] -- unified interface combining short-term and long-term memory
//! - [`MemoryStats`] -- statistics about the memory system

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

// ---------------------------------------------------------------------------
// MemoryCategory
// ---------------------------------------------------------------------------

/// Classification of a memory entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// A factual piece of information.
    Fact,
    /// A user or agent preference.
    Preference,
    /// Contextual information relevant to the current task.
    Context,
    /// An instruction or directive.
    Instruction,
    /// A conversation excerpt or summary.
    Conversation,
    /// A custom category with an arbitrary label.
    Custom(String),
}

impl MemoryCategory {
    /// Returns a string representation of the category.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Fact => "fact",
            Self::Preference => "preference",
            Self::Context => "context",
            Self::Instruction => "instruction",
            Self::Conversation => "conversation",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// MemoryEntry
// ---------------------------------------------------------------------------

/// A single memory record with metadata for tracking access and importance.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique key identifying this memory entry.
    pub key: String,
    /// The stored value as arbitrary JSON.
    pub value: Value,
    /// The category of this memory entry.
    pub category: MemoryCategory,
    /// ISO 8601 timestamp when the entry was created.
    pub created_at: String,
    /// ISO 8601 timestamp when the entry was last accessed or updated.
    pub updated_at: String,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Importance score from 0.0 (low) to 1.0 (high).
    pub importance: f64,
}

impl MemoryEntry {
    /// Create a new `MemoryEntryBuilder`.
    pub fn builder(key: impl Into<String>, value: Value) -> MemoryEntryBuilder {
        MemoryEntryBuilder {
            key: key.into(),
            value,
            category: MemoryCategory::Context,
            created_at: None,
            updated_at: None,
            access_count: 0,
            importance: 0.5,
        }
    }

    /// Serialize this entry to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "key": self.key,
            "value": self.value,
            "category": self.category.as_str(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "importance": self.importance,
        })
    }

    /// Record an access: increments `access_count` and refreshes `updated_at`.
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.updated_at = now_timestamp();
    }
}

/// Builder for [`MemoryEntry`].
pub struct MemoryEntryBuilder {
    key: String,
    value: Value,
    category: MemoryCategory,
    created_at: Option<String>,
    updated_at: Option<String>,
    access_count: u64,
    importance: f64,
}

impl MemoryEntryBuilder {
    /// Set the category.
    pub fn category(mut self, category: MemoryCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the creation timestamp.
    pub fn created_at(mut self, ts: impl Into<String>) -> Self {
        self.created_at = Some(ts.into());
        self
    }

    /// Set the updated timestamp.
    pub fn updated_at(mut self, ts: impl Into<String>) -> Self {
        self.updated_at = Some(ts.into());
        self
    }

    /// Set the initial access count.
    pub fn access_count(mut self, count: u64) -> Self {
        self.access_count = count;
        self
    }

    /// Set the importance score (clamped to 0.0..=1.0).
    pub fn importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Build the [`MemoryEntry`].
    pub fn build(self) -> MemoryEntry {
        let ts = now_timestamp();
        MemoryEntry {
            key: self.key,
            value: self.value,
            category: self.category,
            created_at: self.created_at.unwrap_or_else(|| ts.clone()),
            updated_at: self.updated_at.unwrap_or(ts),
            access_count: self.access_count,
            importance: self.importance.clamp(0.0, 1.0),
        }
    }
}

/// Returns a simple timestamp string for internal use.
fn now_timestamp() -> String {
    // In a production system this would use chrono or similar; here we use a
    // monotonically increasing counter approach via the system time.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{}", dur.as_secs(), dur.subsec_millis())
}

// ---------------------------------------------------------------------------
// ShortTermMemory
// ---------------------------------------------------------------------------

/// A bounded memory store that evicts the least-recently-accessed entry when
/// the capacity is reached.
#[derive(Debug)]
pub struct ShortTermMemory {
    entries: HashMap<String, MemoryEntry>,
    capacity: usize,
}

impl ShortTermMemory {
    /// Create a new short-term memory store with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
        }
    }

    /// Store a value. If at capacity, the least-recently-accessed entry is
    /// evicted first.
    pub fn store(&mut self, key: impl Into<String>, value: Value, category: MemoryCategory) {
        let key = key.into();
        if self.entries.contains_key(&key) {
            // Update existing entry in place.
            let entry = self.entries.get_mut(&key).unwrap();
            entry.value = value;
            entry.category = category;
            entry.touch();
            return;
        }

        if self.entries.len() >= self.capacity {
            self.evict_one();
        }

        let entry = MemoryEntry::builder(&key, value)
            .category(category)
            .build();
        self.entries.insert(key, entry);
    }

    /// Look up an entry by key.
    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key)
    }

    /// Look up an entry by key (mutable).
    pub fn get_mut(&mut self, key: &str) -> Option<&mut MemoryEntry> {
        self.entries.get_mut(key)
    }

    /// Remove an entry by key and return whether it existed.
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Search for entries whose key or value string representation contains
    /// the query (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let q = query.to_lowercase();
        self.entries
            .values()
            .filter(|e| {
                e.key.to_lowercase().contains(&q)
                    || e.value.to_string().to_lowercase().contains(&q)
            })
            .collect()
    }

    /// Return all entries matching a given category.
    pub fn by_category(&self, category: &MemoryCategory) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| &e.category == category)
            .collect()
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Evict the entry with the lowest `access_count`. Ties are broken by
    /// `updated_at` (oldest first).
    fn evict_one(&mut self) {
        if let Some(key) = self
            .entries
            .values()
            .min_by(|a, b| {
                a.access_count
                    .cmp(&b.access_count)
                    .then_with(|| a.updated_at.cmp(&b.updated_at))
            })
            .map(|e| e.key.clone())
        {
            self.entries.remove(&key);
        }
    }
}

// ---------------------------------------------------------------------------
// LongTermMemory
// ---------------------------------------------------------------------------

/// Persistent-style memory store (in-memory, but simulates persistence).
#[derive(Debug, Default)]
pub struct LongTermMemory {
    entries: HashMap<String, MemoryEntry>,
}

impl LongTermMemory {
    /// Create a new, empty long-term memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a memory entry, keyed by its `key` field.
    pub fn store(&mut self, entry: MemoryEntry) {
        self.entries.insert(entry.key.clone(), entry);
    }

    /// Look up an entry by key.
    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key)
    }

    /// Remove an entry by key and return whether it existed.
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Search for entries whose key or value string representation contains
    /// the query (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let q = query.to_lowercase();
        self.entries
            .values()
            .filter(|e| {
                e.key.to_lowercase().contains(&q)
                    || e.value.to_string().to_lowercase().contains(&q)
            })
            .collect()
    }

    /// Return all entries matching a given category.
    pub fn by_category(&self, category: &MemoryCategory) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| &e.category == category)
            .collect()
    }

    /// Return all entries with importance at or above `min_importance`.
    pub fn by_importance(&self, min_importance: f64) -> Vec<&MemoryEntry> {
        self.entries
            .values()
            .filter(|e| e.importance >= min_importance)
            .collect()
    }

    /// Number of entries stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize all entries to a JSON value.
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self.entries.values().map(|e| e.to_json()).collect();
        serde_json::json!({
            "entries": entries,
            "count": self.entries.len(),
        })
    }

    /// Merge entries from `other` that do not already exist in this store.
    pub fn merge_from(&mut self, other: &LongTermMemory) {
        for (key, entry) in &other.entries {
            if !self.entries.contains_key(key) {
                self.entries.insert(key.clone(), entry.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryIndex
// ---------------------------------------------------------------------------

/// A simple inverted index for fast text search across memory entry keys.
///
/// Each key is associated with a set of terms extracted from the indexed text.
/// Search queries are matched against these terms and ranked by term frequency.
#[derive(Debug, Default)]
pub struct MemoryIndex {
    /// Maps a term to the set of keys that contain it, with count.
    index: HashMap<String, HashMap<String, usize>>,
    /// Maps a key to the set of terms extracted from its text.
    key_terms: HashMap<String, Vec<String>>,
}

impl MemoryIndex {
    /// Create a new, empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace the indexed text for a given key.
    pub fn add(&mut self, key: impl Into<String>, text: &str) {
        let key = key.into();
        // Remove old terms for this key first.
        self.remove(&key);

        let terms = tokenize(text);
        for term in &terms {
            self.index
                .entry(term.clone())
                .or_default()
                .entry(key.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
        self.key_terms.insert(key, terms);
    }

    /// Remove all indexed terms for a key.
    pub fn remove(&mut self, key: &str) {
        if let Some(terms) = self.key_terms.remove(key) {
            for term in &terms {
                if let Some(keys_map) = self.index.get_mut(term) {
                    keys_map.remove(key);
                    if keys_map.is_empty() {
                        self.index.remove(term);
                    }
                }
            }
        }
    }

    /// Search the index for keys matching the query terms. Returns keys
    /// ranked by total term frequency (descending).
    pub fn search(&self, query: &str) -> Vec<String> {
        let query_terms = tokenize(query);
        let mut scores: HashMap<String, usize> = HashMap::new();

        for term in &query_terms {
            if let Some(keys_map) = self.index.get(term) {
                for (key, count) in keys_map {
                    *scores.entry(key.clone()).or_default() += count;
                }
            }
        }

        let mut ranked: Vec<(String, usize)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.into_iter().map(|(k, _)| k).collect()
    }

    /// Returns the number of distinct terms in the index.
    pub fn term_count(&self) -> usize {
        self.index.len()
    }
}

/// Simple whitespace tokenizer producing lowercase terms.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// MemoryStats
// ---------------------------------------------------------------------------

/// Statistics about the memory system.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of entries in short-term memory.
    pub short_term_count: usize,
    /// Number of entries in long-term memory.
    pub long_term_count: usize,
    /// Total access count across all entries.
    pub total_accesses: u64,
    /// Count of entries per category.
    pub categories: HashMap<String, usize>,
}

impl MemoryStats {
    /// Serialize statistics to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "short_term_count": self.short_term_count,
            "long_term_count": self.long_term_count,
            "total_accesses": self.total_accesses,
            "categories": self.categories,
        })
    }
}

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

/// Unified memory manager combining short-term and long-term memory with
/// automatic promotion of high-importance entries.
#[derive(Debug)]
pub struct MemoryManager {
    short_term: ShortTermMemory,
    long_term: LongTermMemory,
    index: MemoryIndex,
}

impl MemoryManager {
    /// Create a new memory manager with the given short-term capacity.
    pub fn new(short_capacity: usize) -> Self {
        Self {
            short_term: ShortTermMemory::new(short_capacity),
            long_term: LongTermMemory::new(),
            index: MemoryIndex::new(),
        }
    }

    /// Store a memory entry. If `importance >= 0.7`, the entry is also stored
    /// in long-term memory.
    pub fn remember(
        &mut self,
        key: impl Into<String>,
        value: Value,
        category: MemoryCategory,
        importance: f64,
    ) {
        let key = key.into();
        let importance = importance.clamp(0.0, 1.0);

        // Index the text for search.
        let text = format!("{} {}", key, value);
        self.index.add(&key, &text);

        self.short_term.store(&key, value.clone(), category.clone());

        if importance >= 0.7 {
            let entry = MemoryEntry::builder(&key, value)
                .category(category)
                .importance(importance)
                .build();
            self.long_term.store(entry);
        }
    }

    /// Recall a memory by key. Checks short-term first, then long-term.
    pub fn recall(&self, key: &str) -> Option<&MemoryEntry> {
        self.short_term
            .get(key)
            .or_else(|| self.long_term.get(key))
    }

    /// Search both memory stores for entries matching the query.
    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let mut results: HashMap<String, &MemoryEntry> = HashMap::new();

        for entry in self.short_term.search(query) {
            results.insert(entry.key.clone(), entry);
        }
        for entry in self.long_term.search(query) {
            results.entry(entry.key.clone()).or_insert(entry);
        }

        results.into_values().collect()
    }

    /// Promote an entry from short-term to long-term memory.
    ///
    /// Returns an error if the key is not found in short-term memory.
    pub fn promote(&mut self, key: &str) -> Result<(), String> {
        let entry = self
            .short_term
            .entries
            .remove(key)
            .ok_or_else(|| format!("key '{}' not found in short-term memory", key))?;
        self.long_term.store(entry);
        Ok(())
    }

    /// Remove a memory entry from both stores.
    pub fn forget(&mut self, key: &str) {
        self.short_term.remove(key);
        self.long_term.remove(key);
        self.index.remove(key);
    }

    /// Number of entries in short-term memory.
    pub fn short_term_len(&self) -> usize {
        self.short_term.len()
    }

    /// Number of entries in long-term memory.
    pub fn long_term_len(&self) -> usize {
        self.long_term.len()
    }

    /// Compute statistics about the current state of the memory system.
    pub fn stats(&self) -> MemoryStats {
        let mut total_accesses: u64 = 0;
        let mut categories: HashMap<String, usize> = HashMap::new();

        for entry in self.short_term.entries.values() {
            total_accesses += entry.access_count;
            *categories
                .entry(entry.category.as_str().to_string())
                .or_default() += 1;
        }
        for entry in self.long_term.entries.values() {
            total_accesses += entry.access_count;
            *categories
                .entry(entry.category.as_str().to_string())
                .or_default() += 1;
        }

        MemoryStats {
            short_term_count: self.short_term.len(),
            long_term_count: self.long_term.len(),
            total_accesses,
            categories,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MemoryCategory --

    #[test]
    fn test_category_fact_as_str() {
        assert_eq!(MemoryCategory::Fact.as_str(), "fact");
    }

    #[test]
    fn test_category_preference_as_str() {
        assert_eq!(MemoryCategory::Preference.as_str(), "preference");
    }

    #[test]
    fn test_category_context_as_str() {
        assert_eq!(MemoryCategory::Context.as_str(), "context");
    }

    #[test]
    fn test_category_instruction_as_str() {
        assert_eq!(MemoryCategory::Instruction.as_str(), "instruction");
    }

    #[test]
    fn test_category_conversation_as_str() {
        assert_eq!(MemoryCategory::Conversation.as_str(), "conversation");
    }

    #[test]
    fn test_category_custom_as_str() {
        assert_eq!(
            MemoryCategory::Custom("my_cat".into()).as_str(),
            "my_cat"
        );
    }

    #[test]
    fn test_category_display() {
        assert_eq!(MemoryCategory::Fact.to_string(), "fact");
        assert_eq!(
            MemoryCategory::Custom("agent".into()).to_string(),
            "agent"
        );
    }

    // -- MemoryEntry --

    #[test]
    fn test_entry_builder_defaults() {
        let entry = MemoryEntry::builder("k1", serde_json::json!("v1")).build();
        assert_eq!(entry.key, "k1");
        assert_eq!(entry.value, serde_json::json!("v1"));
        assert_eq!(entry.category, MemoryCategory::Context);
        assert_eq!(entry.access_count, 0);
        assert!((entry.importance - 0.5).abs() < f64::EPSILON);
        assert!(!entry.created_at.is_empty());
        assert!(!entry.updated_at.is_empty());
    }

    #[test]
    fn test_entry_builder_with_category() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .category(MemoryCategory::Fact)
            .build();
        assert_eq!(entry.category, MemoryCategory::Fact);
    }

    #[test]
    fn test_entry_builder_with_importance() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .importance(0.9)
            .build();
        assert!((entry.importance - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entry_builder_importance_clamped_high() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .importance(1.5)
            .build();
        assert!((entry.importance - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entry_builder_importance_clamped_low() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .importance(-0.5)
            .build();
        assert!(entry.importance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_entry_builder_with_timestamps() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .created_at("2026-01-01T00:00:00Z")
            .updated_at("2026-01-02T00:00:00Z")
            .build();
        assert_eq!(entry.created_at, "2026-01-01T00:00:00Z");
        assert_eq!(entry.updated_at, "2026-01-02T00:00:00Z");
    }

    #[test]
    fn test_entry_touch() {
        let mut entry = MemoryEntry::builder("k", serde_json::json!(1)).build();
        let original_updated = entry.updated_at.clone();
        assert_eq!(entry.access_count, 0);

        // Small sleep to ensure timestamp difference.
        std::thread::sleep(std::time::Duration::from_millis(2));
        entry.touch();

        assert_eq!(entry.access_count, 1);
        assert!(entry.updated_at >= original_updated);

        entry.touch();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_entry_to_json() {
        let entry = MemoryEntry::builder("my_key", serde_json::json!({"data": 42}))
            .category(MemoryCategory::Fact)
            .importance(0.8)
            .build();
        let json = entry.to_json();
        assert_eq!(json["key"], "my_key");
        assert_eq!(json["value"]["data"], 42);
        assert_eq!(json["category"], "fact");
        assert_eq!(json["access_count"], 0);
        assert!((json["importance"].as_f64().unwrap() - 0.8).abs() < f64::EPSILON);
    }

    // -- ShortTermMemory --

    #[test]
    fn test_short_term_new_empty() {
        let mem = ShortTermMemory::new(10);
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn test_short_term_store_and_get() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("key1", serde_json::json!("value1"), MemoryCategory::Fact);
        assert_eq!(mem.len(), 1);
        let entry = mem.get("key1").unwrap();
        assert_eq!(entry.value, serde_json::json!("value1"));
        assert_eq!(entry.category, MemoryCategory::Fact);
    }

    #[test]
    fn test_short_term_get_nonexistent() {
        let mem = ShortTermMemory::new(10);
        assert!(mem.get("missing").is_none());
    }

    #[test]
    fn test_short_term_get_mut() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!(1), MemoryCategory::Context);
        let entry = mem.get_mut("k").unwrap();
        entry.touch();
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_short_term_remove() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!(1), MemoryCategory::Context);
        assert!(mem.remove("k"));
        assert!(mem.is_empty());
        assert!(!mem.remove("k"));
    }

    #[test]
    fn test_short_term_clear() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("a", serde_json::json!(1), MemoryCategory::Fact);
        mem.store("b", serde_json::json!(2), MemoryCategory::Fact);
        mem.clear();
        assert!(mem.is_empty());
    }

    #[test]
    fn test_short_term_update_existing_key() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!("old"), MemoryCategory::Fact);
        mem.store("k", serde_json::json!("new"), MemoryCategory::Preference);
        assert_eq!(mem.len(), 1);
        let entry = mem.get("k").unwrap();
        assert_eq!(entry.value, serde_json::json!("new"));
        assert_eq!(entry.category, MemoryCategory::Preference);
        assert_eq!(entry.access_count, 1); // touch was called on update
    }

    #[test]
    fn test_short_term_capacity_eviction() {
        let mut mem = ShortTermMemory::new(2);
        mem.store("a", serde_json::json!(1), MemoryCategory::Fact);
        mem.store("b", serde_json::json!(2), MemoryCategory::Fact);

        // Access "a" to increase its access_count so "b" gets evicted.
        mem.get_mut("a").unwrap().touch();

        mem.store("c", serde_json::json!(3), MemoryCategory::Fact);
        assert_eq!(mem.len(), 2);
        // "b" had the lowest access count and should be evicted.
        assert!(mem.get("a").is_some());
        assert!(mem.get("b").is_none());
        assert!(mem.get("c").is_some());
    }

    #[test]
    fn test_short_term_eviction_no_access() {
        let mut mem = ShortTermMemory::new(2);
        mem.store("first", serde_json::json!(1), MemoryCategory::Fact);
        std::thread::sleep(std::time::Duration::from_millis(2));
        mem.store("second", serde_json::json!(2), MemoryCategory::Fact);

        // Both have access_count=0, so "first" (older updated_at) is evicted.
        mem.store("third", serde_json::json!(3), MemoryCategory::Fact);
        assert_eq!(mem.len(), 2);
        assert!(mem.get("first").is_none());
    }

    #[test]
    fn test_short_term_search() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("user_name", serde_json::json!("Alice"), MemoryCategory::Fact);
        mem.store(
            "project",
            serde_json::json!("Rustchain"),
            MemoryCategory::Context,
        );
        mem.store("preference", serde_json::json!("dark mode"), MemoryCategory::Preference);

        let results = mem.search("alice");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "user_name");
    }

    #[test]
    fn test_short_term_search_in_value() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!("hello world"), MemoryCategory::Fact);

        let results = mem.search("world");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_short_term_search_no_results() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!("v"), MemoryCategory::Fact);

        let results = mem.search("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_short_term_by_category() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("a", serde_json::json!(1), MemoryCategory::Fact);
        mem.store("b", serde_json::json!(2), MemoryCategory::Preference);
        mem.store("c", serde_json::json!(3), MemoryCategory::Fact);

        let facts = mem.by_category(&MemoryCategory::Fact);
        assert_eq!(facts.len(), 2);

        let prefs = mem.by_category(&MemoryCategory::Preference);
        assert_eq!(prefs.len(), 1);
    }

    #[test]
    fn test_short_term_by_category_empty() {
        let mem = ShortTermMemory::new(10);
        let results = mem.by_category(&MemoryCategory::Instruction);
        assert!(results.is_empty());
    }

    // -- LongTermMemory --

    #[test]
    fn test_long_term_new_empty() {
        let mem = LongTermMemory::new();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn test_long_term_store_and_get() {
        let mut mem = LongTermMemory::new();
        let entry = MemoryEntry::builder("lt_key", serde_json::json!("lt_val"))
            .category(MemoryCategory::Instruction)
            .importance(0.9)
            .build();
        mem.store(entry);

        let retrieved = mem.get("lt_key").unwrap();
        assert_eq!(retrieved.value, serde_json::json!("lt_val"));
        assert_eq!(retrieved.category, MemoryCategory::Instruction);
    }

    #[test]
    fn test_long_term_remove() {
        let mut mem = LongTermMemory::new();
        let entry = MemoryEntry::builder("k", serde_json::json!(1)).build();
        mem.store(entry);
        assert!(mem.remove("k"));
        assert!(mem.is_empty());
        assert!(!mem.remove("k"));
    }

    #[test]
    fn test_long_term_search() {
        let mut mem = LongTermMemory::new();
        let e1 = MemoryEntry::builder("colors", serde_json::json!("blue"))
            .build();
        let e2 = MemoryEntry::builder("food", serde_json::json!("pizza"))
            .build();
        mem.store(e1);
        mem.store(e2);

        let results = mem.search("blue");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "colors");
    }

    #[test]
    fn test_long_term_by_category() {
        let mut mem = LongTermMemory::new();
        let e1 = MemoryEntry::builder("a", serde_json::json!(1))
            .category(MemoryCategory::Fact)
            .build();
        let e2 = MemoryEntry::builder("b", serde_json::json!(2))
            .category(MemoryCategory::Preference)
            .build();
        mem.store(e1);
        mem.store(e2);

        assert_eq!(mem.by_category(&MemoryCategory::Fact).len(), 1);
        assert_eq!(mem.by_category(&MemoryCategory::Conversation).len(), 0);
    }

    #[test]
    fn test_long_term_by_importance() {
        let mut mem = LongTermMemory::new();
        let low = MemoryEntry::builder("low", serde_json::json!(1))
            .importance(0.3)
            .build();
        let mid = MemoryEntry::builder("mid", serde_json::json!(2))
            .importance(0.6)
            .build();
        let high = MemoryEntry::builder("high", serde_json::json!(3))
            .importance(0.9)
            .build();
        mem.store(low);
        mem.store(mid);
        mem.store(high);

        assert_eq!(mem.by_importance(0.5).len(), 2);
        assert_eq!(mem.by_importance(0.8).len(), 1);
        assert_eq!(mem.by_importance(0.0).len(), 3);
        assert_eq!(mem.by_importance(1.0).len(), 0);
    }

    #[test]
    fn test_long_term_to_json() {
        let mut mem = LongTermMemory::new();
        let entry = MemoryEntry::builder("k", serde_json::json!("v")).build();
        mem.store(entry);

        let json = mem.to_json();
        assert_eq!(json["count"], 1);
        assert!(json["entries"].is_array());
        assert_eq!(json["entries"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_long_term_merge_from() {
        let mut mem_a = LongTermMemory::new();
        let e1 = MemoryEntry::builder("shared", serde_json::json!("from_a")).build();
        let e2 = MemoryEntry::builder("only_a", serde_json::json!("a_val")).build();
        mem_a.store(e1);
        mem_a.store(e2);

        let mut mem_b = LongTermMemory::new();
        let e3 = MemoryEntry::builder("shared", serde_json::json!("from_b")).build();
        let e4 = MemoryEntry::builder("only_b", serde_json::json!("b_val")).build();
        mem_b.store(e3);
        mem_b.store(e4);

        mem_a.merge_from(&mem_b);
        assert_eq!(mem_a.len(), 3);
        // "shared" keeps the value from mem_a
        assert_eq!(mem_a.get("shared").unwrap().value, serde_json::json!("from_a"));
        assert!(mem_a.get("only_b").is_some());
    }

    #[test]
    fn test_long_term_merge_from_empty() {
        let mut mem_a = LongTermMemory::new();
        let e = MemoryEntry::builder("k", serde_json::json!(1)).build();
        mem_a.store(e);

        let mem_b = LongTermMemory::new();
        mem_a.merge_from(&mem_b);
        assert_eq!(mem_a.len(), 1);
    }

    // -- MemoryIndex --

    #[test]
    fn test_index_new_empty() {
        let idx = MemoryIndex::new();
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn test_index_add_and_search() {
        let mut idx = MemoryIndex::new();
        idx.add("doc1", "the quick brown fox");
        idx.add("doc2", "the lazy dog");

        let results = idx.search("fox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "doc1");
    }

    #[test]
    fn test_index_search_multiple_matches() {
        let mut idx = MemoryIndex::new();
        idx.add("a", "hello world");
        idx.add("b", "hello rust");

        let results = idx.search("hello");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_index_search_ranked_by_frequency() {
        let mut idx = MemoryIndex::new();
        idx.add("few", "rust is great");
        idx.add("many", "rust rust rust rocks");

        let results = idx.search("rust");
        assert_eq!(results.len(), 2);
        // "many" should rank first due to higher term frequency.
        assert_eq!(results[0], "many");
    }

    #[test]
    fn test_index_remove() {
        let mut idx = MemoryIndex::new();
        idx.add("k", "some text here");
        assert!(idx.term_count() > 0);

        idx.remove("k");
        let results = idx.search("text");
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_remove_nonexistent() {
        let mut idx = MemoryIndex::new();
        idx.remove("nope"); // should not panic
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn test_index_search_no_results() {
        let mut idx = MemoryIndex::new();
        idx.add("k", "hello world");
        let results = idx.search("missing");
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_term_count() {
        let mut idx = MemoryIndex::new();
        idx.add("k", "one two three");
        assert_eq!(idx.term_count(), 3);
    }

    #[test]
    fn test_index_case_insensitive() {
        let mut idx = MemoryIndex::new();
        idx.add("k", "Hello World");
        let results = idx.search("hello");
        assert_eq!(results.len(), 1);
    }

    // -- MemoryManager --

    #[test]
    fn test_manager_new() {
        let mgr = MemoryManager::new(100);
        assert_eq!(mgr.short_term_len(), 0);
        assert_eq!(mgr.long_term_len(), 0);
    }

    #[test]
    fn test_manager_remember_low_importance() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.3);
        assert_eq!(mgr.short_term_len(), 1);
        assert_eq!(mgr.long_term_len(), 0);
    }

    #[test]
    fn test_manager_remember_high_importance_auto_promote() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.8);
        assert_eq!(mgr.short_term_len(), 1);
        assert_eq!(mgr.long_term_len(), 1);
    }

    #[test]
    fn test_manager_remember_boundary_importance() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.7);
        assert_eq!(mgr.long_term_len(), 1);
    }

    #[test]
    fn test_manager_remember_below_boundary() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.69);
        assert_eq!(mgr.long_term_len(), 0);
    }

    #[test]
    fn test_manager_recall_from_short_term() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.3);
        let entry = mgr.recall("k").unwrap();
        assert_eq!(entry.value, serde_json::json!("v"));
    }

    #[test]
    fn test_manager_recall_from_long_term() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.9);
        // Remove from short-term to test long-term fallback.
        mgr.short_term.remove("k");
        let entry = mgr.recall("k").unwrap();
        assert_eq!(entry.value, serde_json::json!("v"));
    }

    #[test]
    fn test_manager_recall_not_found() {
        let mgr = MemoryManager::new(10);
        assert!(mgr.recall("missing").is_none());
    }

    #[test]
    fn test_manager_search() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember(
            "user_name",
            serde_json::json!("Alice"),
            MemoryCategory::Fact,
            0.5,
        );
        mgr.remember(
            "project",
            serde_json::json!("Rustchain"),
            MemoryCategory::Context,
            0.9,
        );

        let results = mgr.search("alice");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "user_name");
    }

    #[test]
    fn test_manager_promote() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.3);
        assert_eq!(mgr.long_term_len(), 0);

        mgr.promote("k").unwrap();
        assert_eq!(mgr.short_term_len(), 0);
        assert_eq!(mgr.long_term_len(), 1);
        assert!(mgr.recall("k").is_some());
    }

    #[test]
    fn test_manager_promote_not_found() {
        let mut mgr = MemoryManager::new(10);
        let result = mgr.promote("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_forget() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("k", serde_json::json!("v"), MemoryCategory::Fact, 0.9);
        assert_eq!(mgr.short_term_len(), 1);
        assert_eq!(mgr.long_term_len(), 1);

        mgr.forget("k");
        assert_eq!(mgr.short_term_len(), 0);
        assert_eq!(mgr.long_term_len(), 0);
        assert!(mgr.recall("k").is_none());
    }

    #[test]
    fn test_manager_forget_nonexistent() {
        let mut mgr = MemoryManager::new(10);
        mgr.forget("nope"); // should not panic
    }

    #[test]
    fn test_manager_stats() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("a", serde_json::json!(1), MemoryCategory::Fact, 0.3);
        mgr.remember("b", serde_json::json!(2), MemoryCategory::Preference, 0.8);
        mgr.remember("c", serde_json::json!(3), MemoryCategory::Fact, 0.5);

        let stats = mgr.stats();
        assert_eq!(stats.short_term_count, 3);
        assert_eq!(stats.long_term_count, 1); // only "b" has importance >= 0.7
        // "b" is in both short-term and long-term, count categories from both
        assert!(stats.categories.contains_key("fact"));
        assert!(stats.categories.contains_key("preference"));
    }

    #[test]
    fn test_manager_stats_to_json() {
        let mut mgr = MemoryManager::new(10);
        mgr.remember("a", serde_json::json!(1), MemoryCategory::Fact, 0.5);

        let json = mgr.stats().to_json();
        assert_eq!(json["short_term_count"], 1);
        assert_eq!(json["long_term_count"], 0);
        assert!(json["categories"].is_object());
    }

    #[test]
    fn test_manager_stats_empty() {
        let mgr = MemoryManager::new(10);
        let stats = mgr.stats();
        assert_eq!(stats.short_term_count, 0);
        assert_eq!(stats.long_term_count, 0);
        assert_eq!(stats.total_accesses, 0);
        assert!(stats.categories.is_empty());
    }

    // -- Edge cases --

    #[test]
    fn test_short_term_capacity_one() {
        let mut mem = ShortTermMemory::new(1);
        mem.store("a", serde_json::json!(1), MemoryCategory::Fact);
        mem.store("b", serde_json::json!(2), MemoryCategory::Fact);
        assert_eq!(mem.len(), 1);
        assert!(mem.get("a").is_none());
        assert!(mem.get("b").is_some());
    }

    #[test]
    fn test_empty_search_query() {
        let mut mem = ShortTermMemory::new(10);
        mem.store("k", serde_json::json!("v"), MemoryCategory::Fact);
        // Empty query matches everything since "" is contained in all strings.
        let results = mem.search("");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_long_term_to_json_empty() {
        let mem = LongTermMemory::new();
        let json = mem.to_json();
        assert_eq!(json["count"], 0);
        assert!(json["entries"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_entry_builder_access_count() {
        let entry = MemoryEntry::builder("k", serde_json::json!(1))
            .access_count(42)
            .build();
        assert_eq!(entry.access_count, 42);
    }
}
