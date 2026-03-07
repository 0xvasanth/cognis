//! Document store abstraction — a persistent, queryable store for documents.
//!
//! Provides [`DocStore`] trait for CRUD operations on [`Document`]s with
//! text search and metadata filtering, plus an [`InMemoryDocStore`] and
//! an [`IndexedDocStore`] that layers an inverted index for fast text search.

use std::collections::{HashMap, HashSet};

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MetadataCondition
// ---------------------------------------------------------------------------

/// A filter condition applied to document metadata.
#[derive(Debug, Clone)]
pub enum MetadataCondition {
    /// Metadata key equals the given value.
    Equals(String, Value),
    /// Metadata key does not equal the given value.
    NotEquals(String, Value),
    /// Metadata string value contains the given substring.
    Contains(String, String),
    /// Metadata numeric value is greater than the threshold.
    GreaterThan(String, f64),
    /// Metadata numeric value is less than the threshold.
    LessThan(String, f64),
    /// Metadata key exists.
    Exists(String),
    /// Metadata key does not exist.
    NotExists(String),
}

impl MetadataCondition {
    /// Check whether this condition is satisfied by the given metadata map.
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            MetadataCondition::Equals(key, value) => metadata.get(key) == Some(value),
            MetadataCondition::NotEquals(key, value) => metadata.get(key) != Some(value),
            MetadataCondition::Contains(key, substring) => metadata
                .get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.contains(substring.as_str()))
                .unwrap_or(false),
            MetadataCondition::GreaterThan(key, threshold) => metadata
                .get(key)
                .and_then(|v| v.as_f64())
                .map(|n| n > *threshold)
                .unwrap_or(false),
            MetadataCondition::LessThan(key, threshold) => metadata
                .get(key)
                .and_then(|v| v.as_f64())
                .map(|n| n < *threshold)
                .unwrap_or(false),
            MetadataCondition::Exists(key) => metadata.contains_key(key),
            MetadataCondition::NotExists(key) => !metadata.contains_key(key),
        }
    }
}

// ---------------------------------------------------------------------------
// DocStoreQuery
// ---------------------------------------------------------------------------

/// A query against a [`DocStore`].
#[derive(Debug, Clone)]
pub struct DocStoreQuery {
    /// Optional full-text search string.
    pub text: Option<String>,
    /// Metadata filter conditions (all must match).
    pub metadata_filters: Vec<MetadataCondition>,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
    /// Number of results to skip.
    pub offset: usize,
}

impl DocStoreQuery {
    /// Create a new empty query that matches everything.
    pub fn new() -> Self {
        Self {
            text: None,
            metadata_filters: Vec::new(),
            limit: None,
            offset: 0,
        }
    }

    /// Set the text search string.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Add a metadata equality filter.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata_filters
            .push(MetadataCondition::Equals(key.into(), value.into()));
        self
    }

    /// Set the maximum number of results.
    pub fn with_limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the offset for pagination.
    pub fn with_offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }
}

impl Default for DocStoreQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DocStore trait
// ---------------------------------------------------------------------------

/// A queryable document store.
pub trait DocStore: Send + Sync {
    /// Add a document and return its assigned ID.
    fn add(&mut self, doc: Document) -> Result<String>;

    /// Retrieve a document by ID.
    fn get(&self, id: &str) -> Result<Option<Document>>;

    /// Delete a document by ID. Returns `true` if the document existed.
    fn delete(&mut self, id: &str) -> Result<bool>;

    /// Search for documents matching the given query.
    fn search(&self, query: &DocStoreQuery) -> Result<Vec<Document>>;

    /// Update a document in-place. Returns `true` if the document existed.
    fn update(&mut self, id: &str, doc: Document) -> Result<bool>;

    /// Return the number of documents in the store.
    fn count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// InMemoryDocStore
// ---------------------------------------------------------------------------

/// A simple in-memory document store backed by a [`HashMap`].
#[derive(Debug, Default)]
pub struct InMemoryDocStore {
    docs: HashMap<String, Document>,
}

impl InMemoryDocStore {
    /// Create a new, empty in-memory document store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a batch of documents, returning their assigned IDs.
    pub fn add_batch(&mut self, docs: Vec<Document>) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(docs.len());
        for doc in docs {
            let id = self.add(doc)?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Return references to all documents in the store.
    pub fn all_documents(&self) -> Vec<&Document> {
        self.docs.values().collect()
    }

    /// Remove all documents from the store.
    pub fn clear(&mut self) {
        self.docs.clear();
    }

    /// Apply text and metadata filters to the store, returning matching documents.
    fn apply_query(&self, query: &DocStoreQuery) -> Vec<Document> {
        let lower_text = query.text.as_ref().map(|t| t.to_lowercase());

        let mut results: Vec<Document> = self
            .docs
            .values()
            .filter(|doc| {
                // Text filter: case-insensitive contains
                if let Some(ref lt) = lower_text {
                    if !doc.page_content.to_lowercase().contains(lt.as_str()) {
                        return false;
                    }
                }
                // Metadata filters: all must match
                query
                    .metadata_filters
                    .iter()
                    .all(|cond| cond.matches(&doc.metadata))
            })
            .cloned()
            .collect();

        // Stable sort by id for deterministic output
        results.sort_by(|a, b| a.id.cmp(&b.id));

        // Apply offset and limit
        let start = query.offset.min(results.len());
        let results = results.into_iter().skip(start);
        match query.limit {
            Some(n) => results.take(n).collect(),
            None => results.collect(),
        }
    }
}

impl DocStore for InMemoryDocStore {
    fn add(&mut self, mut doc: Document) -> Result<String> {
        let id = doc
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        doc.id = Some(id.clone());
        self.docs.insert(id.clone(), doc);
        Ok(id)
    }

    fn get(&self, id: &str) -> Result<Option<Document>> {
        Ok(self.docs.get(id).cloned())
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        Ok(self.docs.remove(id).is_some())
    }

    fn search(&self, query: &DocStoreQuery) -> Result<Vec<Document>> {
        Ok(self.apply_query(query))
    }

    fn update(&mut self, id: &str, mut doc: Document) -> Result<bool> {
        if self.docs.contains_key(id) {
            doc.id = Some(id.to_string());
            self.docs.insert(id.to_string(), doc);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn count(&self) -> usize {
        self.docs.len()
    }
}

// ---------------------------------------------------------------------------
// DocStoreIndex — inverted index for text search
// ---------------------------------------------------------------------------

/// An inverted index mapping terms to document IDs for fast text search.
#[derive(Debug, Default)]
pub struct DocStoreIndex {
    /// term -> { doc_id -> term_frequency }
    index: HashMap<String, HashMap<String, usize>>,
    /// doc_id -> total token count
    doc_lengths: HashMap<String, usize>,
}

impl DocStoreIndex {
    /// Create a new, empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize content into lowercase terms.
    fn tokenize(content: &str) -> Vec<String> {
        content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Index a document's content under the given ID.
    pub fn index_document(&mut self, id: &str, content: &str) {
        // Remove old entry first
        self.remove(id);

        let tokens = Self::tokenize(content);
        self.doc_lengths.insert(id.to_string(), tokens.len());

        for token in tokens {
            *self
                .index
                .entry(token)
                .or_default()
                .entry(id.to_string())
                .or_insert(0) += 1;
        }
    }

    /// Search the index and return document IDs with relevance scores (TF-based).
    ///
    /// The score for each document is the sum of term frequencies for all
    /// query terms, divided by the document length (normalized TF).
    pub fn search(&self, query: &str) -> Vec<(&str, f64)> {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<&str, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.index.get(token) {
                for (doc_id, &freq) in postings {
                    let doc_len = self.doc_lengths.get(doc_id).copied().unwrap_or(1) as f64;
                    *scores.entry(doc_id.as_str()).or_insert(0.0) += freq as f64 / doc_len;
                }
            }
        }

        let mut results: Vec<(&str, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, id: &str) {
        self.doc_lengths.remove(id);
        self.index.retain(|_, postings| {
            postings.remove(id);
            !postings.is_empty()
        });
    }

    /// Return the number of distinct terms in the index.
    pub fn term_count(&self) -> usize {
        self.index.len()
    }

    /// Return the number of documents in the index.
    pub fn document_count(&self) -> usize {
        self.doc_lengths.len()
    }
}

// ---------------------------------------------------------------------------
// IndexedDocStore — InMemoryDocStore + DocStoreIndex
// ---------------------------------------------------------------------------

/// An in-memory document store augmented with an inverted index for fast
/// text search.
#[derive(Debug, Default)]
pub struct IndexedDocStore {
    store: InMemoryDocStore,
    index: DocStoreIndex,
}

impl IndexedDocStore {
    /// Create a new, empty indexed document store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl DocStore for IndexedDocStore {
    fn add(&mut self, doc: Document) -> Result<String> {
        let content = doc.page_content.clone();
        let id = self.store.add(doc)?;
        self.index.index_document(&id, &content);
        Ok(id)
    }

    fn get(&self, id: &str) -> Result<Option<Document>> {
        self.store.get(id)
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        self.index.remove(id);
        self.store.delete(id)
    }

    fn search(&self, query: &DocStoreQuery) -> Result<Vec<Document>> {
        // If there is a text query, use the index to narrow candidates
        if let Some(ref text) = query.text {
            let scored = self.index.search(text);
            let candidate_ids: Vec<&str> = scored.iter().map(|(id, _)| *id).collect();

            let results: Vec<Document> = candidate_ids
                .into_iter()
                .filter_map(|id| self.store.docs.get(id))
                .filter(|doc| {
                    query
                        .metadata_filters
                        .iter()
                        .all(|cond| cond.matches(&doc.metadata))
                })
                .cloned()
                .collect();

            // Apply offset and limit
            let start = query.offset.min(results.len());
            let results = results.into_iter().skip(start);
            match query.limit {
                Some(n) => Ok(results.take(n).collect()),
                None => Ok(results.collect()),
            }
        } else {
            // No text query — fall back to the underlying store's search
            self.store.search(query)
        }
    }

    fn update(&mut self, id: &str, doc: Document) -> Result<bool> {
        let content = doc.page_content.clone();
        let existed = self.store.update(id, doc)?;
        if existed {
            self.index.index_document(id, &content);
        }
        Ok(existed)
    }

    fn count(&self) -> usize {
        self.store.count()
    }
}

// ---------------------------------------------------------------------------
// DocStoreStats
// ---------------------------------------------------------------------------

/// Summary statistics about a document store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocStoreStats {
    /// Total number of documents.
    pub total_documents: usize,
    /// Total characters across all documents.
    pub total_chars: usize,
    /// Average document length in characters.
    pub avg_doc_length: f64,
    /// Distinct metadata keys across all documents.
    pub metadata_keys: Vec<String>,
}

impl DocStoreStats {
    /// Compute statistics from an [`InMemoryDocStore`].
    pub fn from_store(store: &InMemoryDocStore) -> Self {
        let docs = store.all_documents();
        let total_documents = docs.len();
        let total_chars: usize = docs.iter().map(|d| d.page_content.len()).sum();
        let avg_doc_length = if total_documents > 0 {
            total_chars as f64 / total_documents as f64
        } else {
            0.0
        };

        let mut keys_set: HashSet<String> = HashSet::new();
        for doc in &docs {
            for key in doc.metadata.keys() {
                keys_set.insert(key.clone());
            }
        }
        let mut metadata_keys: Vec<String> = keys_set.into_iter().collect();
        metadata_keys.sort();

        Self {
            total_documents,
            total_chars,
            avg_doc_length,
            metadata_keys,
        }
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_meta(content: &str, meta: Vec<(&str, Value)>) -> Document {
        let metadata: HashMap<String, Value> =
            meta.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Document::new(content).with_metadata(metadata)
    }

    // -----------------------------------------------------------------------
    // MetadataCondition tests
    // -----------------------------------------------------------------------

    #[test]
    fn metadata_equals() {
        let meta: HashMap<String, Value> =
            [("color".into(), json!("red"))].into_iter().collect();
        assert!(MetadataCondition::Equals("color".into(), json!("red")).matches(&meta));
        assert!(!MetadataCondition::Equals("color".into(), json!("blue")).matches(&meta));
    }

    #[test]
    fn metadata_not_equals() {
        let meta: HashMap<String, Value> =
            [("color".into(), json!("red"))].into_iter().collect();
        assert!(MetadataCondition::NotEquals("color".into(), json!("blue")).matches(&meta));
        assert!(!MetadataCondition::NotEquals("color".into(), json!("red")).matches(&meta));
    }

    #[test]
    fn metadata_contains() {
        let meta: HashMap<String, Value> =
            [("tag".into(), json!("hello world"))].into_iter().collect();
        assert!(MetadataCondition::Contains("tag".into(), "world".into()).matches(&meta));
        assert!(!MetadataCondition::Contains("tag".into(), "xyz".into()).matches(&meta));
    }

    #[test]
    fn metadata_greater_than() {
        let meta: HashMap<String, Value> =
            [("score".into(), json!(7.5))].into_iter().collect();
        assert!(MetadataCondition::GreaterThan("score".into(), 5.0).matches(&meta));
        assert!(!MetadataCondition::GreaterThan("score".into(), 10.0).matches(&meta));
    }

    #[test]
    fn metadata_less_than() {
        let meta: HashMap<String, Value> =
            [("score".into(), json!(3.0))].into_iter().collect();
        assert!(MetadataCondition::LessThan("score".into(), 5.0).matches(&meta));
        assert!(!MetadataCondition::LessThan("score".into(), 1.0).matches(&meta));
    }

    #[test]
    fn metadata_exists() {
        let meta: HashMap<String, Value> =
            [("key".into(), json!(null))].into_iter().collect();
        assert!(MetadataCondition::Exists("key".into()).matches(&meta));
        assert!(!MetadataCondition::Exists("missing".into()).matches(&meta));
    }

    #[test]
    fn metadata_not_exists() {
        let meta: HashMap<String, Value> =
            [("key".into(), json!(null))].into_iter().collect();
        assert!(MetadataCondition::NotExists("missing".into()).matches(&meta));
        assert!(!MetadataCondition::NotExists("key".into()).matches(&meta));
    }

    #[test]
    fn metadata_contains_non_string_value() {
        let meta: HashMap<String, Value> =
            [("num".into(), json!(42))].into_iter().collect();
        // Contains on a non-string value should return false
        assert!(!MetadataCondition::Contains("num".into(), "42".into()).matches(&meta));
    }

    #[test]
    fn metadata_greater_than_non_numeric() {
        let meta: HashMap<String, Value> =
            [("name".into(), json!("alice"))].into_iter().collect();
        assert!(!MetadataCondition::GreaterThan("name".into(), 0.0).matches(&meta));
    }

    // -----------------------------------------------------------------------
    // InMemoryDocStore CRUD
    // -----------------------------------------------------------------------

    #[test]
    fn add_and_get() {
        let mut store = InMemoryDocStore::new();
        let id = store.add(make_doc("hello")).unwrap();
        let doc = store.get(&id).unwrap().unwrap();
        assert_eq!(doc.page_content, "hello");
        assert_eq!(doc.id, Some(id));
    }

    #[test]
    fn add_preserves_existing_id() {
        let mut store = InMemoryDocStore::new();
        let doc = make_doc("test").with_id("my-id");
        let id = store.add(doc).unwrap();
        assert_eq!(id, "my-id");
        assert!(store.get("my-id").unwrap().is_some());
    }

    #[test]
    fn get_missing_returns_none() {
        let store = InMemoryDocStore::new();
        assert!(store.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn delete_existing() {
        let mut store = InMemoryDocStore::new();
        let id = store.add(make_doc("delete me")).unwrap();
        assert!(store.delete(&id).unwrap());
        assert!(store.get(&id).unwrap().is_none());
    }

    #[test]
    fn delete_missing() {
        let mut store = InMemoryDocStore::new();
        assert!(!store.delete("nope").unwrap());
    }

    #[test]
    fn update_existing() {
        let mut store = InMemoryDocStore::new();
        let id = store.add(make_doc("original")).unwrap();
        assert!(store.update(&id, make_doc("updated")).unwrap());
        let doc = store.get(&id).unwrap().unwrap();
        assert_eq!(doc.page_content, "updated");
        assert_eq!(doc.id, Some(id));
    }

    #[test]
    fn update_missing() {
        let mut store = InMemoryDocStore::new();
        assert!(!store.update("nope", make_doc("x")).unwrap());
    }

    #[test]
    fn count_tracks_additions_and_deletions() {
        let mut store = InMemoryDocStore::new();
        assert_eq!(store.count(), 0);
        let id = store.add(make_doc("a")).unwrap();
        assert_eq!(store.count(), 1);
        store.add(make_doc("b")).unwrap();
        assert_eq!(store.count(), 2);
        store.delete(&id).unwrap();
        assert_eq!(store.count(), 1);
    }

    // -----------------------------------------------------------------------
    // Batch add
    // -----------------------------------------------------------------------

    #[test]
    fn add_batch() {
        let mut store = InMemoryDocStore::new();
        let docs = vec![make_doc("a"), make_doc("b"), make_doc("c")];
        let ids = store.add_batch(docs).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.count(), 3);
    }

    // -----------------------------------------------------------------------
    // Text search
    // -----------------------------------------------------------------------

    #[test]
    fn text_search_case_insensitive() {
        let mut store = InMemoryDocStore::new();
        store.add(make_doc("Hello World")).unwrap();
        store.add(make_doc("goodbye world")).unwrap();
        store.add(make_doc("nothing here")).unwrap();

        let query = DocStoreQuery::new().with_text("WORLD");
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn text_search_no_matches() {
        let mut store = InMemoryDocStore::new();
        store.add(make_doc("apple")).unwrap();
        let query = DocStoreQuery::new().with_text("banana");
        let results = store.search(&query).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Metadata filtering
    // -----------------------------------------------------------------------

    #[test]
    fn search_with_metadata_filter() {
        let mut store = InMemoryDocStore::new();
        store
            .add(make_doc_with_meta("doc1", vec![("type", json!("article"))]))
            .unwrap();
        store
            .add(make_doc_with_meta("doc2", vec![("type", json!("book"))]))
            .unwrap();

        let query = DocStoreQuery::new().with_metadata("type", json!("article"));
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "doc1");
    }

    #[test]
    fn search_combined_text_and_metadata() {
        let mut store = InMemoryDocStore::new();
        store
            .add(make_doc_with_meta(
                "rust programming",
                vec![("lang", json!("rust"))],
            ))
            .unwrap();
        store
            .add(make_doc_with_meta(
                "rust metal",
                vec![("lang", json!("english"))],
            ))
            .unwrap();
        store
            .add(make_doc_with_meta(
                "python programming",
                vec![("lang", json!("python"))],
            ))
            .unwrap();

        let query = DocStoreQuery::new()
            .with_text("rust")
            .with_metadata("lang", json!("rust"));
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "rust programming");
    }

    // -----------------------------------------------------------------------
    // Limit and offset
    // -----------------------------------------------------------------------

    #[test]
    fn search_with_limit() {
        let mut store = InMemoryDocStore::new();
        for i in 0..10 {
            store.add(make_doc(&format!("doc {i}"))).unwrap();
        }
        let query = DocStoreQuery::new().with_limit(3);
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_with_offset() {
        let mut store = InMemoryDocStore::new();
        for i in 0..5 {
            store
                .add(make_doc(&format!("doc {i}")).with_id(format!("id-{i}")))
                .unwrap();
        }
        let query = DocStoreQuery::new().with_offset(3);
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_with_limit_and_offset() {
        let mut store = InMemoryDocStore::new();
        for i in 0..10 {
            store
                .add(make_doc(&format!("doc {i}")).with_id(format!("id-{i:02}")))
                .unwrap();
        }
        let query = DocStoreQuery::new().with_offset(2).with_limit(3);
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_offset_beyond_results() {
        let mut store = InMemoryDocStore::new();
        store.add(make_doc("a")).unwrap();
        let query = DocStoreQuery::new().with_offset(100);
        let results = store.search(&query).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // all_documents and clear
    // -----------------------------------------------------------------------

    #[test]
    fn all_documents_and_clear() {
        let mut store = InMemoryDocStore::new();
        store.add(make_doc("a")).unwrap();
        store.add(make_doc("b")).unwrap();
        assert_eq!(store.all_documents().len(), 2);
        store.clear();
        assert_eq!(store.count(), 0);
        assert!(store.all_documents().is_empty());
    }

    // -----------------------------------------------------------------------
    // Duplicate IDs
    // -----------------------------------------------------------------------

    #[test]
    fn duplicate_id_overwrites() {
        let mut store = InMemoryDocStore::new();
        store.add(make_doc("first").with_id("same")).unwrap();
        store.add(make_doc("second").with_id("same")).unwrap();
        assert_eq!(store.count(), 1);
        let doc = store.get("same").unwrap().unwrap();
        assert_eq!(doc.page_content, "second");
    }

    // -----------------------------------------------------------------------
    // Empty store
    // -----------------------------------------------------------------------

    #[test]
    fn search_empty_store() {
        let store = InMemoryDocStore::new();
        let query = DocStoreQuery::new().with_text("anything");
        let results = store.search(&query).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // DocStoreIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn index_and_search() {
        let mut idx = DocStoreIndex::new();
        idx.index_document("d1", "the quick brown fox");
        idx.index_document("d2", "the lazy dog");
        idx.index_document("d3", "quick quick fox");

        let results = idx.search("quick fox");
        assert!(!results.is_empty());
        // d3 has "quick" twice so should score higher
        assert_eq!(results[0].0, "d3");
    }

    #[test]
    fn index_remove() {
        let mut idx = DocStoreIndex::new();
        idx.index_document("d1", "hello world");
        assert_eq!(idx.document_count(), 1);
        idx.remove("d1");
        assert_eq!(idx.document_count(), 0);
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn index_search_no_match() {
        let mut idx = DocStoreIndex::new();
        idx.index_document("d1", "hello world");
        let results = idx.search("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn index_empty_query() {
        let idx = DocStoreIndex::new();
        let results = idx.search("");
        assert!(results.is_empty());
    }

    #[test]
    fn index_term_and_document_count() {
        let mut idx = DocStoreIndex::new();
        idx.index_document("d1", "alpha beta");
        idx.index_document("d2", "beta gamma");
        assert_eq!(idx.document_count(), 2);
        assert_eq!(idx.term_count(), 3); // alpha, beta, gamma
    }

    #[test]
    fn index_reindex_same_id() {
        let mut idx = DocStoreIndex::new();
        idx.index_document("d1", "old content");
        idx.index_document("d1", "new content");
        assert_eq!(idx.document_count(), 1);
        let results = idx.search("old");
        assert!(results.is_empty());
        let results = idx.search("new");
        assert_eq!(results.len(), 1);
    }

    // -----------------------------------------------------------------------
    // IndexedDocStore tests
    // -----------------------------------------------------------------------

    #[test]
    fn indexed_store_add_and_search() {
        let mut store = IndexedDocStore::new();
        store.add(make_doc("the quick brown fox")).unwrap();
        store.add(make_doc("the lazy dog")).unwrap();

        let query = DocStoreQuery::new().with_text("fox");
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].page_content.contains("fox"));
    }

    #[test]
    fn indexed_store_delete_removes_from_index() {
        let mut store = IndexedDocStore::new();
        let id = store.add(make_doc("unique searchable content")).unwrap();
        store.delete(&id).unwrap();

        let query = DocStoreQuery::new().with_text("searchable");
        let results = store.search(&query).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn indexed_store_update_reindexes() {
        let mut store = IndexedDocStore::new();
        let id = store.add(make_doc("old text")).unwrap();
        store.update(&id, make_doc("new text")).unwrap();

        let old_query = DocStoreQuery::new().with_text("old");
        assert!(store.search(&old_query).unwrap().is_empty());

        let new_query = DocStoreQuery::new().with_text("new");
        assert_eq!(store.search(&new_query).unwrap().len(), 1);
    }

    #[test]
    fn indexed_store_metadata_filter_with_text() {
        let mut store = IndexedDocStore::new();
        store
            .add(make_doc_with_meta(
                "rust language",
                vec![("type", json!("tech"))],
            ))
            .unwrap();
        store
            .add(make_doc_with_meta(
                "rust metal corrosion",
                vec![("type", json!("science"))],
            ))
            .unwrap();

        let mut query = DocStoreQuery::new().with_text("rust");
        query
            .metadata_filters
            .push(MetadataCondition::Equals("type".into(), json!("tech")));
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "rust language");
    }

    #[test]
    fn indexed_store_no_text_falls_back() {
        let mut store = IndexedDocStore::new();
        store.add(make_doc("alpha")).unwrap();
        store.add(make_doc("beta")).unwrap();

        let query = DocStoreQuery::new();
        let results = store.search(&query).unwrap();
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // DocStoreStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn stats_empty_store() {
        let store = InMemoryDocStore::new();
        let stats = DocStoreStats::from_store(&store);
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_chars, 0);
        assert_eq!(stats.avg_doc_length, 0.0);
        assert!(stats.metadata_keys.is_empty());
    }

    #[test]
    fn stats_populated_store() {
        let mut store = InMemoryDocStore::new();
        store
            .add(make_doc_with_meta("hello", vec![("a", json!(1))]))
            .unwrap();
        store
            .add(make_doc_with_meta("world!!", vec![("b", json!(2))]))
            .unwrap();

        let stats = DocStoreStats::from_store(&store);
        assert_eq!(stats.total_documents, 2);
        assert_eq!(stats.total_chars, 12); // "hello" (5) + "world!!" (7)
        assert!((stats.avg_doc_length - 6.0).abs() < f64::EPSILON);
        assert_eq!(stats.metadata_keys.len(), 2);
        assert!(stats.metadata_keys.contains(&"a".to_string()));
        assert!(stats.metadata_keys.contains(&"b".to_string()));
    }

    #[test]
    fn stats_to_json() {
        let store = InMemoryDocStore::new();
        let stats = DocStoreStats::from_store(&store);
        let json = stats.to_json();
        assert_eq!(json["total_documents"], 0);
    }
}
