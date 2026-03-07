//! Metadata manipulation transformers for documents.
//!
//! This module provides transformers that add, remove, rename, filter, and
//! extract metadata on [`Document`] collections.

use std::collections::HashMap;

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;

use super::DocumentTransformer;

// ─── MetadataAdder ───

/// Adds fixed key-value pairs to every document's metadata.
///
/// Existing keys with the same name are overwritten.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::metadata::MetadataAdder;
///
/// let adder = MetadataAdder::new()
///     .add("source", Value::from("web"))
///     .add("version", Value::from(2));
/// let docs = adder.transform_documents(&docs).await?;
/// ```
pub struct MetadataAdder {
    pairs: HashMap<String, Value>,
}

impl MetadataAdder {
    /// Create a new empty adder.
    pub fn new() -> Self {
        Self {
            pairs: HashMap::new(),
        }
    }

    /// Add a key-value pair to be inserted into every document.
    pub fn add(mut self, key: impl Into<String>, value: Value) -> Self {
        self.pairs.insert(key.into(), value);
        self
    }
}

impl Default for MetadataAdder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentTransformer for MetadataAdder {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut new_doc = doc.clone();
            for (k, v) in &self.pairs {
                new_doc.metadata.insert(k.clone(), v.clone());
            }
            results.push(new_doc);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "MetadataAdder"
    }
}

// ─── MetadataRemover ───

/// Removes specified keys from every document's metadata.
///
/// Keys that do not exist are silently ignored.
pub struct MetadataRemover {
    keys: Vec<String>,
}

impl MetadataRemover {
    /// Create a remover that will strip the given keys.
    pub fn new(keys: Vec<String>) -> Self {
        Self { keys }
    }
}

#[async_trait]
impl DocumentTransformer for MetadataRemover {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut new_doc = doc.clone();
            for key in &self.keys {
                new_doc.metadata.remove(key);
            }
            results.push(new_doc);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "MetadataRemover"
    }
}

// ─── MetadataMapper ───

/// Renames metadata keys according to a mapping (old_key -> new_key).
///
/// If the old key does not exist, that mapping is skipped. If the new key
/// already exists, it is overwritten.
pub struct MetadataMapper {
    mapping: HashMap<String, String>,
}

impl MetadataMapper {
    /// Create a mapper from a mapping of old_key -> new_key.
    pub fn new(mapping: HashMap<String, String>) -> Self {
        Self { mapping }
    }
}

#[async_trait]
impl DocumentTransformer for MetadataMapper {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut new_doc = doc.clone();
            for (old_key, new_key) in &self.mapping {
                if let Some(value) = new_doc.metadata.remove(old_key) {
                    new_doc.metadata.insert(new_key.clone(), value);
                }
            }
            results.push(new_doc);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "MetadataMapper"
    }
}

// ─── MetadataFilter ───

/// A condition for filtering documents by metadata.
#[derive(Debug, Clone)]
pub enum MetadataCondition {
    /// Metadata key equals the given value.
    Equals(String, Value),
    /// Metadata key (string value) contains the given substring.
    Contains(String, String),
    /// Metadata key exists.
    Exists(String),
    /// Metadata key (numeric value) is greater than the given threshold.
    GreaterThan(String, f64),
    /// Metadata key (numeric value) is less than the given threshold.
    LessThan(String, f64),
    /// All conditions must be true.
    And(Vec<MetadataCondition>),
    /// At least one condition must be true.
    Or(Vec<MetadataCondition>),
}

impl MetadataCondition {
    /// Evaluate this condition against a document's metadata.
    pub fn evaluate(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            MetadataCondition::Equals(key, expected) => {
                metadata.get(key).map_or(false, |v| v == expected)
            }
            MetadataCondition::Contains(key, substring) => metadata
                .get(key)
                .and_then(|v| v.as_str())
                .map_or(false, |s| s.contains(substring.as_str())),
            MetadataCondition::Exists(key) => metadata.contains_key(key),
            MetadataCondition::GreaterThan(key, threshold) => metadata
                .get(key)
                .and_then(|v| v.as_f64())
                .map_or(false, |n| n > *threshold),
            MetadataCondition::LessThan(key, threshold) => metadata
                .get(key)
                .and_then(|v| v.as_f64())
                .map_or(false, |n| n < *threshold),
            MetadataCondition::And(conditions) => {
                conditions.iter().all(|c| c.evaluate(metadata))
            }
            MetadataCondition::Or(conditions) => {
                conditions.iter().any(|c| c.evaluate(metadata))
            }
        }
    }
}

/// Filters documents based on metadata conditions.
///
/// Only documents whose metadata satisfies the condition are kept.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::metadata::{MetadataFilter, MetadataCondition};
///
/// let filter = MetadataFilter::new(MetadataCondition::Exists("source".into()));
/// let filtered = filter.transform_documents(&docs).await?;
/// ```
pub struct MetadataFilter {
    condition: MetadataCondition,
}

impl MetadataFilter {
    /// Create a filter with the given condition.
    pub fn new(condition: MetadataCondition) -> Self {
        Self { condition }
    }
}

#[async_trait]
impl DocumentTransformer for MetadataFilter {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|doc| self.condition.evaluate(&doc.metadata))
            .cloned()
            .collect())
    }

    fn name(&self) -> &str {
        "MetadataFilter"
    }
}

// ─── MetadataExtractor ───

/// Extracts metadata from document content using regex patterns.
///
/// Each named pattern maps a metadata key to a regex. If the regex matches
/// the document's `page_content`, the first capture group (or the entire match
/// if there are no groups) is stored as the metadata value.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::metadata::MetadataExtractor;
///
/// let extractor = MetadataExtractor::new()
///     .with_pattern("email", r"[\w.+-]+@[\w-]+\.[\w.-]+");
/// let docs = extractor.transform_documents(&docs).await?;
/// ```
pub struct MetadataExtractor {
    patterns: Vec<(String, Regex)>,
}

impl MetadataExtractor {
    /// Create a new extractor with no patterns.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a named regex pattern. The metadata key will hold the first match.
    ///
    /// # Panics
    ///
    /// Panics if the regex pattern is invalid.
    pub fn with_pattern(mut self, key: impl Into<String>, pattern: &str) -> Self {
        let regex = Regex::new(pattern).expect("invalid regex pattern");
        self.patterns.push((key.into(), regex));
        self
    }
}

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentTransformer for MetadataExtractor {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut new_doc = doc.clone();
            for (key, regex) in &self.patterns {
                if let Some(captures) = regex.captures(&doc.page_content) {
                    // Use first capture group if present, otherwise the whole match.
                    let matched = captures
                        .get(1)
                        .or_else(|| captures.get(0))
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default();
                    new_doc
                        .metadata
                        .insert(key.clone(), Value::from(matched));
                }
            }
            results.push(new_doc);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "MetadataExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_metadata(content: &str, meta: HashMap<String, Value>) -> Document {
        Document::new(content).with_metadata(meta)
    }

    fn simple_meta(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    // ─── MetadataAdder tests ───

    #[tokio::test]
    async fn test_adder_adds_pairs() {
        let adder = MetadataAdder::new()
            .add("source", Value::from("web"))
            .add("version", Value::from(3));
        let docs = vec![make_doc("hello")];
        let result = adder.transform_documents(&docs).await.unwrap();
        assert_eq!(result[0].metadata.get("source").unwrap(), "web");
        assert_eq!(result[0].metadata.get("version").unwrap(), 3);
    }

    #[tokio::test]
    async fn test_adder_overwrites_existing() {
        let adder = MetadataAdder::new().add("key", Value::from("new"));
        let docs = vec![make_doc_with_metadata(
            "text",
            simple_meta(&[("key", Value::from("old"))]),
        )];
        let result = adder.transform_documents(&docs).await.unwrap();
        assert_eq!(result[0].metadata.get("key").unwrap(), "new");
    }

    #[tokio::test]
    async fn test_adder_empty_documents() {
        let adder = MetadataAdder::new().add("k", Value::from("v"));
        let result = adder.transform_documents(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    // ─── MetadataRemover tests ───

    #[tokio::test]
    async fn test_remover_removes_keys() {
        let remover = MetadataRemover::new(vec!["a".into(), "b".into()]);
        let docs = vec![make_doc_with_metadata(
            "text",
            simple_meta(&[
                ("a", Value::from(1)),
                ("b", Value::from(2)),
                ("c", Value::from(3)),
            ]),
        )];
        let result = remover.transform_documents(&docs).await.unwrap();
        assert!(!result[0].metadata.contains_key("a"));
        assert!(!result[0].metadata.contains_key("b"));
        assert!(result[0].metadata.contains_key("c"));
    }

    #[tokio::test]
    async fn test_remover_missing_keys_ignored() {
        let remover = MetadataRemover::new(vec!["nonexistent".into()]);
        let docs = vec![make_doc("text")];
        let result = remover.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    // ─── MetadataMapper tests ───

    #[tokio::test]
    async fn test_mapper_renames_keys() {
        let mut mapping = HashMap::new();
        mapping.insert("old_name".into(), "new_name".into());
        let mapper = MetadataMapper::new(mapping);

        let docs = vec![make_doc_with_metadata(
            "text",
            simple_meta(&[("old_name", Value::from("val"))]),
        )];
        let result = mapper.transform_documents(&docs).await.unwrap();
        assert!(!result[0].metadata.contains_key("old_name"));
        assert_eq!(result[0].metadata.get("new_name").unwrap(), "val");
    }

    #[tokio::test]
    async fn test_mapper_missing_source_key() {
        let mut mapping = HashMap::new();
        mapping.insert("missing".into(), "target".into());
        let mapper = MetadataMapper::new(mapping);

        let docs = vec![make_doc("text")];
        let result = mapper.transform_documents(&docs).await.unwrap();
        assert!(!result[0].metadata.contains_key("target"));
    }

    // ─── MetadataFilter tests ───

    #[tokio::test]
    async fn test_filter_equals() {
        let filter = MetadataFilter::new(MetadataCondition::Equals(
            "status".into(),
            Value::from("active"),
        ));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("status", Value::from("active"))])),
            make_doc_with_metadata("b", simple_meta(&[("status", Value::from("inactive"))])),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "a");
    }

    #[tokio::test]
    async fn test_filter_contains() {
        let filter =
            MetadataFilter::new(MetadataCondition::Contains("path".into(), "docs".into()));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("path", Value::from("/docs/file.md"))])),
            make_doc_with_metadata("b", simple_meta(&[("path", Value::from("/src/main.rs"))])),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "a");
    }

    #[tokio::test]
    async fn test_filter_exists() {
        let filter = MetadataFilter::new(MetadataCondition::Exists("tag".into()));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("tag", Value::from("x"))])),
            make_doc("b"),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "a");
    }

    #[tokio::test]
    async fn test_filter_greater_than() {
        let filter = MetadataFilter::new(MetadataCondition::GreaterThan("score".into(), 0.5));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("score", Value::from(0.8))])),
            make_doc_with_metadata("b", simple_meta(&[("score", Value::from(0.3))])),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "a");
    }

    #[tokio::test]
    async fn test_filter_less_than() {
        let filter = MetadataFilter::new(MetadataCondition::LessThan("score".into(), 0.5));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("score", Value::from(0.8))])),
            make_doc_with_metadata("b", simple_meta(&[("score", Value::from(0.3))])),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "b");
    }

    #[tokio::test]
    async fn test_filter_and_combination() {
        let filter = MetadataFilter::new(MetadataCondition::And(vec![
            MetadataCondition::Exists("source".into()),
            MetadataCondition::GreaterThan("score".into(), 0.5),
        ]));
        let docs = vec![
            make_doc_with_metadata(
                "a",
                simple_meta(&[
                    ("source", Value::from("web")),
                    ("score", Value::from(0.9)),
                ]),
            ),
            make_doc_with_metadata(
                "b",
                simple_meta(&[("score", Value::from(0.9))]),
            ),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "a");
    }

    #[tokio::test]
    async fn test_filter_or_combination() {
        let filter = MetadataFilter::new(MetadataCondition::Or(vec![
            MetadataCondition::Equals("type".into(), Value::from("pdf")),
            MetadataCondition::Equals("type".into(), Value::from("html")),
        ]));
        let docs = vec![
            make_doc_with_metadata("a", simple_meta(&[("type", Value::from("pdf"))])),
            make_doc_with_metadata("b", simple_meta(&[("type", Value::from("txt"))])),
            make_doc_with_metadata("c", simple_meta(&[("type", Value::from("html"))])),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    // ─── MetadataExtractor tests ───

    #[tokio::test]
    async fn test_extractor_extracts_email() {
        let extractor =
            MetadataExtractor::new().with_pattern("email", r"[\w.+-]+@[\w-]+\.[\w.-]+");
        let docs = vec![make_doc("Contact us at hello@example.com for info.")];
        let result = extractor.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("email").and_then(|v| v.as_str()),
            Some("hello@example.com")
        );
    }

    #[tokio::test]
    async fn test_extractor_with_capture_group() {
        let extractor =
            MetadataExtractor::new().with_pattern("year", r"Copyright (\d{4})");
        let docs = vec![make_doc("Copyright 2024 Acme Inc.")];
        let result = extractor.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("year").and_then(|v| v.as_str()),
            Some("2024")
        );
    }

    #[tokio::test]
    async fn test_extractor_no_match() {
        let extractor = MetadataExtractor::new().with_pattern("phone", r"\d{3}-\d{4}");
        let docs = vec![make_doc("No phone number here.")];
        let result = extractor.transform_documents(&docs).await.unwrap();
        assert!(!result[0].metadata.contains_key("phone"));
    }
}
