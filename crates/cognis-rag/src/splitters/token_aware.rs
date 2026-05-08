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

    /// Token-overlap between adjacent chunks. Clamped to `max_tokens - 1`
    /// so the second-pass walker always makes progress; if the caller
    /// passes a larger value, the chunk would just feed itself back in
    /// as overlap and the splitter would loop or emit chunks > max.
    pub fn with_overlap_tokens(mut self, n: usize) -> Self {
        let cap = self.max_tokens.saturating_sub(1);
        self.overlap_tokens = n.min(cap);
        self
    }
}

/// Take a suffix of `s` whose token count is exactly `n_tokens` (or
/// the largest suffix below it). Counts via `tok` so the result is
/// honest about the configured token unit.
fn token_tail(s: &str, n_tokens: usize, tok: &dyn Tokenizer) -> String {
    if n_tokens == 0 {
        return String::new();
    }
    // Walk back char-by-char, growing the tail until the tokenizer
    // reports n_tokens. Char-stepping is bounded by the chunk size so
    // this is O(chunk_chars) — fine for typical RAG sizes.
    let chars: Vec<char> = s.chars().collect();
    let mut tail = String::new();
    for &c in chars.iter().rev() {
        let mut candidate = String::with_capacity(tail.len() + c.len_utf8());
        candidate.push(c);
        candidate.push_str(&tail);
        if tok.count(&candidate) > n_tokens {
            break;
        }
        tail = candidate;
    }
    tail
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
                        let last = &out.last().unwrap().content;
                        buf.push_str(&token_tail(
                            last,
                            self.overlap_tokens,
                            self.tokenizer.as_ref(),
                        ));
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

    #[test]
    fn overlap_clamps_below_max_tokens() {
        let tok: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        let s = TokenAwareSplitter::new(tok, 5).with_overlap_tokens(20);
        // Clamped to max - 1 = 4.
        assert_eq!(s.overlap_tokens, 4);
    }

    #[test]
    fn overlap_uses_token_count_not_char_count() {
        // Word tokenizer: tokens = whitespace-separated word count.
        let tok: Arc<dyn Tokenizer> = Arc::new(FnTokenizer(|s: &str| s.split_whitespace().count()));
        let tail = token_tail("alpha beta gamma delta", 2, tok.as_ref());
        // Last 2 word-tokens: "gamma delta" (length 11 chars), not the
        // last 2 *characters* ("ta"). The tail-walker grows char-by-char,
        // so leading whitespace from word boundaries can be included.
        assert_eq!(tok.count(&tail), 2);
        assert!(tail.ends_with("gamma delta"), "tail = {tail:?}");
    }

    #[test]
    fn overlap_zero_tokens_returns_empty_tail() {
        let tok: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        assert_eq!(token_tail("anything", 0, tok.as_ref()), "");
    }
}
