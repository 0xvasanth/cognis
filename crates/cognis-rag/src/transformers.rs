//! Document-list transformers — `Runnable<Vec<Document>, Vec<Document>>`.
//!
//! Rust-native take: V1 had a separate "DocumentTransformer" trait for
//! pre-store / post-retrieval doc operations. In V2 these are just
//! `Runnable`s — they compose with `.pipe()` and slot into chains
//! anywhere a runnable is expected. No new trait surface.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

/// Reorder retrieved documents so the most-relevant ones sit at the
/// **head and tail** of the list. LLMs attend better to ends than to the
/// middle of long contexts; this is the classic "lost in the middle" fix.
///
/// Assumes the input is already ranked best-first (the standard retriever
/// output). Reshuffles into: `[1, 3, 5, ..., 6, 4, 2]` (best at index 0,
/// next-best at last index).
#[derive(Debug, Default, Clone, Copy)]
pub struct LongContextReorder;

impl LongContextReorder {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    /// Reorder `docs` (assumed best-first ranked) so the best ranks live
    /// at both ends. Pure function — useful for tests and ad-hoc use.
    pub fn reorder(docs: Vec<Document>) -> Vec<Document> {
        let mut head: Vec<Document> = Vec::with_capacity(docs.len());
        let mut tail: Vec<Document> = Vec::with_capacity(docs.len());
        for (i, d) in docs.into_iter().enumerate() {
            if i % 2 == 0 {
                head.push(d);
            } else {
                tail.push(d);
            }
        }
        tail.reverse();
        head.extend(tail);
        head
    }
}

#[async_trait]
impl Runnable<Vec<Document>, Vec<Document>> for LongContextReorder {
    async fn invoke(&self, input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
        Ok(Self::reorder(input))
    }
    fn name(&self) -> &str {
        "LongContextReorder"
    }
}

/// Drop duplicate documents from the list. By default, two docs are
/// considered duplicates if they have the same `content` (whitespace-
/// trimmed). Use [`Dedup::by`] to dedupe on a custom key (e.g. by id,
/// or by a normalized hash of the content).
///
/// First-seen wins; later duplicates are discarded.
pub struct Dedup {
    key_fn: Arc<dyn Fn(&Document) -> String + Send + Sync>,
}

impl Default for Dedup {
    fn default() -> Self {
        Self::new()
    }
}

impl Dedup {
    /// Dedupe by trimmed content.
    pub fn new() -> Self {
        Self {
            key_fn: Arc::new(|d: &Document| d.content.trim().to_string()),
        }
    }

    /// Dedupe by a caller-supplied key function (e.g. `|d| d.id.clone().unwrap_or_default()`).
    pub fn by<F>(key_fn: F) -> Self
    where
        F: Fn(&Document) -> String + Send + Sync + 'static,
    {
        Self {
            key_fn: Arc::new(key_fn),
        }
    }

    /// Pure form — useful for tests and ad-hoc use.
    pub fn dedup(&self, docs: Vec<Document>) -> Vec<Document> {
        let mut seen: HashSet<u64> = HashSet::new();
        let mut out = Vec::with_capacity(docs.len());
        for d in docs {
            let key = (self.key_fn)(&d);
            let mut h = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut h);
            if seen.insert(h.finish()) {
                out.push(d);
            }
        }
        out
    }
}

#[async_trait]
impl Runnable<Vec<Document>, Vec<Document>> for Dedup {
    async fn invoke(&self, input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
        Ok(self.dedup(input))
    }
    fn name(&self) -> &str {
        "Dedup"
    }
}

/// Apply a per-document transform — e.g. tag, summarize, redact.
/// The closure mutates the document in place; documents that fail can
/// be filtered by returning an `Err` (the transform aborts on first
/// error).
pub struct Enrichment {
    f: Arc<dyn Fn(&mut Document) -> Result<()> + Send + Sync>,
    name: &'static str,
}

impl Enrichment {
    /// Wrap a per-doc enrichment closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Document) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            f: Arc::new(f),
            name: "Enrichment",
        }
    }

    /// Override the runnable name (shown in tracing/logs).
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

#[async_trait]
impl Runnable<Vec<Document>, Vec<Document>> for Enrichment {
    async fn invoke(&self, mut input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
        for d in &mut input {
            (self.f)(d)?;
        }
        Ok(input)
    }
    fn name(&self) -> &str {
        self.name
    }
}

/// Set / merge fixed metadata onto every document. Use to tag a batch
/// with provenance (`source = "kb-2026-05"`), pipeline stage, etc.
///
/// Existing keys are overwritten by default; use [`MetadataTransformer::merge_only_missing`]
/// to keep existing values.
#[derive(Debug, Default, Clone)]
pub struct MetadataTransformer {
    fields: HashMap<String, Value>,
    only_missing: bool,
}

impl MetadataTransformer {
    /// Empty transformer — add fields with [`MetadataTransformer::set`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from a map. Keys are merged into every document.
    pub fn from_map(fields: HashMap<String, Value>) -> Self {
        Self {
            fields,
            only_missing: false,
        }
    }

    /// Add a single key / value.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Don't overwrite keys the doc already has.
    pub fn merge_only_missing(mut self) -> Self {
        self.only_missing = true;
        self
    }

    /// Pure form.
    pub fn apply(&self, mut docs: Vec<Document>) -> Vec<Document> {
        for d in &mut docs {
            for (k, v) in &self.fields {
                if self.only_missing && d.metadata.contains_key(k) {
                    continue;
                }
                d.metadata.insert(k.clone(), v.clone());
            }
        }
        docs
    }
}

#[async_trait]
impl Runnable<Vec<Document>, Vec<Document>> for MetadataTransformer {
    async fn invoke(&self, input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
        Ok(self.apply(input))
    }
    fn name(&self) -> &str {
        "MetadataTransformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str) -> Document {
        Document::new(id).with_id(id)
    }

    #[test]
    fn reorder_pattern() {
        // Input ranked best-first: [1, 2, 3, 4, 5]
        // Expected: [1, 3, 5, 4, 2] — best at ends.
        let docs = vec![doc("1"), doc("2"), doc("3"), doc("4"), doc("5")];
        let out = LongContextReorder::reorder(docs);
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert_eq!(ids, vec!["1", "3", "5", "4", "2"]);
    }

    #[test]
    fn empty_passes_through() {
        let out = LongContextReorder::reorder(Vec::new());
        assert!(out.is_empty());
    }

    #[test]
    fn single_doc_passes_through() {
        let out = LongContextReorder::reorder(vec![doc("only")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id.as_deref(), Some("only"));
    }

    #[tokio::test]
    async fn runnable_invoke() {
        let r = LongContextReorder::new();
        let out = r
            .invoke(
                vec![doc("a"), doc("b"), doc("c")],
                RunnableConfig::default(),
            )
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert_eq!(ids, vec!["a", "c", "b"]);
    }

    #[test]
    fn dedup_by_content_keeps_first_seen() {
        let docs = vec![
            Document::new("hello"),
            Document::new("world"),
            Document::new(" hello "), // trimmed dup
            Document::new("rust"),
        ];
        let out = Dedup::new().dedup(docs);
        let contents: Vec<_> = out.iter().map(|d| d.content.clone()).collect();
        assert_eq!(contents, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn dedup_by_id_uses_custom_key() {
        let docs = vec![
            Document::new("a body").with_id("a"),
            Document::new("a body").with_id("b"), // same content, different id
            Document::new("c body").with_id("a"), // dup id of first
        ];
        let out = Dedup::by(|d| d.id.clone().unwrap_or_default()).dedup(docs);
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn enrichment_applies_per_doc() {
        let r = Enrichment::new(|d: &mut Document| {
            d.content = d.content.to_uppercase();
            d.metadata
                .insert("seen".into(), serde_json::Value::Bool(true));
            Ok(())
        });
        let out = r
            .invoke(
                vec![Document::new("hi"), Document::new("ho")],
                RunnableConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(out[0].content, "HI");
        assert_eq!(out[1].content, "HO");
        assert!(out[0].metadata.get("seen").is_some());
    }

    #[test]
    fn metadata_transformer_overwrites_by_default() {
        let docs = vec![
            Document::new("d1").with_metadata("source", serde_json::json!("old")),
            Document::new("d2"),
        ];
        let out = MetadataTransformer::new().set("source", "new").apply(docs);
        assert_eq!(out[0].metadata.get("source").unwrap(), &serde_json::json!("new"));
        assert_eq!(out[1].metadata.get("source").unwrap(), &serde_json::json!("new"));
    }

    #[test]
    fn metadata_transformer_only_missing_preserves_existing() {
        let docs = vec![
            Document::new("d1").with_metadata("source", serde_json::json!("old")),
            Document::new("d2"),
        ];
        let out = MetadataTransformer::new()
            .set("source", "new")
            .merge_only_missing()
            .apply(docs);
        assert_eq!(out[0].metadata.get("source").unwrap(), &serde_json::json!("old"));
        assert_eq!(out[1].metadata.get("source").unwrap(), &serde_json::json!("new"));
    }
}
