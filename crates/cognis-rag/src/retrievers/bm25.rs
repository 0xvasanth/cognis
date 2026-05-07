//! BM25 retriever — classic Okapi BM25 over an in-memory corpus.
//!
//! Useful as a sparse-baseline retriever to complement embedding search,
//! especially in [`super::EnsembleRetriever`] hybrids.

use async_trait::async_trait;
use std::collections::HashMap;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

const DEFAULT_K1: f32 = 1.5;
const DEFAULT_B: f32 = 0.75;

/// In-memory BM25 retriever. Construct from a corpus via
/// [`BM25Retriever::from_documents`]; the index is built upfront.
pub struct BM25Retriever {
    docs: Vec<Document>,
    /// Term frequencies per document.
    tf: Vec<HashMap<String, u32>>,
    /// Document length (in tokens) per document.
    doc_lens: Vec<u32>,
    /// Inverse document frequency per term.
    idf: HashMap<String, f32>,
    /// Average document length across the corpus.
    avg_doc_len: f32,
    k: usize,
    k1: f32,
    b: f32,
}

impl BM25Retriever {
    /// Build an index from a static corpus.
    pub fn from_documents(docs: Vec<Document>) -> Self {
        let n = docs.len();
        let mut tf: Vec<HashMap<String, u32>> = Vec::with_capacity(n);
        let mut doc_lens: Vec<u32> = Vec::with_capacity(n);
        let mut df: HashMap<String, u32> = HashMap::new();

        for d in &docs {
            let tokens = tokenize(&d.content);
            doc_lens.push(tokens.len() as u32);
            let mut counts: HashMap<String, u32> = HashMap::new();
            for t in &tokens {
                *counts.entry(t.clone()).or_insert(0) += 1;
            }
            for t in counts.keys() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            tf.push(counts);
        }

        let n_f = n.max(1) as f32;
        let avg_doc_len = if n == 0 {
            0.0
        } else {
            doc_lens.iter().map(|&l| l as f32).sum::<f32>() / n_f
        };

        let mut idf: HashMap<String, f32> = HashMap::new();
        for (term, dfreq) in df {
            let v = ((n_f - dfreq as f32 + 0.5) / (dfreq as f32 + 0.5) + 1.0).ln();
            idf.insert(term, v);
        }

        Self {
            docs,
            tf,
            doc_lens,
            idf,
            avg_doc_len,
            k: 4,
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        }
    }

    /// Override the top-k.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Override the BM25 `k1` parameter (default 1.5).
    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Override the BM25 `b` parameter (default 0.75).
    pub fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Score a single document against the query.
    fn score(&self, query_terms: &[String], doc_idx: usize) -> f32 {
        let dl = self.doc_lens[doc_idx] as f32;
        let mut score = 0.0;
        for term in query_terms {
            let f = self.tf[doc_idx].get(term).copied().unwrap_or(0) as f32;
            if f == 0.0 {
                continue;
            }
            let idf = self.idf.get(term).copied().unwrap_or(0.0);
            let denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len.max(1e-6));
            score += idf * (f * (self.k1 + 1.0)) / denom;
        }
        score
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for BM25Retriever {
    async fn invoke(&self, query: String, _: RunnableConfig) -> Result<Vec<Document>> {
        let q = tokenize(&query);
        let mut scored: Vec<(usize, f32)> = (0..self.docs.len())
            .map(|i| (i, self.score(&q, i)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(self.k)
            .map(|(i, _)| self.docs[i].clone())
            .collect())
    }

    fn name(&self) -> &str {
        "BM25Retriever"
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Document> {
        vec![
            Document::new("Rust is a systems programming language").with_id("1"),
            Document::new("Python is a high-level dynamic language").with_id("2"),
            Document::new("Rust has zero-cost abstractions and ownership").with_id("3"),
            Document::new("Cooking with cast iron pans is great").with_id("4"),
        ]
    }

    #[tokio::test]
    async fn ranks_relevant_first() {
        let r = BM25Retriever::from_documents(corpus()).with_k(2);
        let out = r
            .invoke("rust ownership".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(!out.is_empty());
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert!(ids.iter().any(|id| id == "3" || id == "1"));
        assert!(!ids.iter().any(|id| id == "4"));
    }

    #[tokio::test]
    async fn returns_empty_for_no_match() {
        let r = BM25Retriever::from_documents(corpus());
        let out = r
            .invoke("zzz unrelated query xyz".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn respects_k() {
        let r = BM25Retriever::from_documents(corpus()).with_k(1);
        let out = r
            .invoke("language".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(out.len() <= 1);
    }
}
