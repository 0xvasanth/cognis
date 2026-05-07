//! Token-aware splitter using `cognis_core`'s pluggable [`Tokenizer`].

use std::sync::Arc;

use crate::document::Document;

// Re-export the canonical trait from cognis-core so users can write
// `cognis_rag::Tokenizer` or `cognis_core::Tokenizer` interchangeably.
pub use cognis_core::tokenizer::{CharTokenizer, FnTokenizer, Tokenizer};

use super::{child_doc, recursive::RecursiveCharSplitter, TextSplitter};

/// Splits text so each chunk's token count (per the supplied [`Tokenizer`])
/// stays under `max_tokens`. Falls back to a recursive char splitter for
/// the structural cuts; just adds a token-aware re-pack step on top.
pub struct TokenAwareSplitter {
    tokenizer: Arc<dyn Tokenizer>,
    max_tokens: usize,
    overlap_tokens: usize,
    inner: RecursiveCharSplitter,
}

impl TokenAwareSplitter {
    /// Build with a tokenizer + max-token cap.
    pub fn new(tokenizer: Arc<dyn Tokenizer>, max_tokens: usize) -> Self {
        Self {
            tokenizer,
            max_tokens,
            overlap_tokens: 0,
            // Approximate the char budget as 4× tokens — the recursive
            // splitter is just a structural cutter; we re-bound by tokens.
            inner: RecursiveCharSplitter::new()
                .with_chunk_size(max_tokens.saturating_mul(4).max(1)),
        }
    }

    /// Token-overlap between adjacent chunks.
    pub fn with_overlap_tokens(mut self, n: usize) -> Self {
        self.overlap_tokens = n;
        self
    }
}

impl TextSplitter for TokenAwareSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        // First pass: structural cut.
        let intermediate = self.inner.split(doc);
        // Second pass: any chunk over budget gets char-trimmed greedily.
        let mut out: Vec<Document> = Vec::new();
        for d in intermediate {
            if self.tokenizer.count(&d.content) <= self.max_tokens {
                out.push(child_doc(doc, d.content, out.len()));
                continue;
            }
            // Greedy char-walk until we hit max_tokens.
            let mut buf = String::new();
            for ch in d.content.chars() {
                buf.push(ch);
                if self.tokenizer.count(&buf) >= self.max_tokens {
                    out.push(child_doc(doc, std::mem::take(&mut buf), out.len()));
                    if self.overlap_tokens > 0 {
                        let last = out.last().unwrap().content.clone();
                        let tail: String = last
                            .chars()
                            .rev()
                            .take(self.overlap_tokens)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect();
                        buf.push_str(&tail);
                    }
                }
            }
            if !buf.is_empty() {
                out.push(child_doc(doc, buf, out.len()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_caps_chunk_size() {
        let tok: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        let s = TokenAwareSplitter::new(tok, 10);
        let doc = Document::new("a".repeat(50));
        let chunks = s.split(&doc);
        assert!(chunks.iter().all(|c| c.content.chars().count() <= 10));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn fn_tokenizer_works() {
        // Pretend each whitespace-separated word is one token.
        let tok: Arc<dyn Tokenizer> = Arc::new(FnTokenizer(|s: &str| s.split_whitespace().count()));
        assert_eq!(tok.count("hello rust world"), 3);
    }
}
