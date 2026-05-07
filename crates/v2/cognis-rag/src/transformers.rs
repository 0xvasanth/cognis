//! Document-list transformers — `Runnable<Vec<Document>, Vec<Document>>`.
//!
//! Rust-native take: V1 had a separate "DocumentTransformer" trait for
//! pre-store / post-retrieval doc operations. In V2 these are just
//! `Runnable`s — they compose with `.pipe()` and slot into chains
//! anywhere a runnable is expected. No new trait surface.

use async_trait::async_trait;

use cognis2_core::{Result, Runnable, RunnableConfig};

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
}
