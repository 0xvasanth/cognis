//! Character-level splitter — split on a single user-supplied separator
//! string with optional chunk overlap.
//!
//! Sister to [`super::recursive::RecursiveCharSplitter`]: where the
//! recursive splitter walks down a list of separators looking for the
//! coarsest fit, this one splits on exactly one separator and packs the
//! resulting fragments into `chunk_size`-bounded chunks.
//!
//! Customization knobs:
//! - `with_chunk_size(n)` — target maximum chunk size (chars).
//! - `with_overlap(n)` — overlap window between adjacent chunks.
//! - `with_separator(s)` — the separator. Default: `"\n\n"`.
//! - `with_keep_separator(bool)` — whether the separator is retained at
//!   the start of subsequent chunks (mirrors V1 behaviour).
//! - `with_length_fn(fn)` — pluggable length measure. Defaults to chars
//!   but accepts any `Fn(&str) -> usize` (e.g. a real tokenizer).

use std::sync::Arc;

use crate::document::Document;

use super::{child_doc, TextSplitter};

/// Function used to measure chunk size. Default counts characters; users
/// can plug in a tokenizer for token-budget splits.
pub type LengthFn = Arc<dyn Fn(&str) -> usize + Send + Sync>;

/// Single-separator character splitter.
pub struct CharacterSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separator: String,
    keep_separator: bool,
    length_fn: LengthFn,
}

impl Default for CharacterSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separator: "\n\n".to_string(),
            keep_separator: false,
            length_fn: Arc::new(|s: &str| s.chars().count()),
        }
    }
}

impl std::fmt::Debug for CharacterSplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharacterSplitter")
            .field("chunk_size", &self.chunk_size)
            .field("chunk_overlap", &self.chunk_overlap)
            .field("separator", &self.separator)
            .field("keep_separator", &self.keep_separator)
            .finish()
    }
}

impl CharacterSplitter {
    /// New splitter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum chunk size measured by `length_fn` (default: chars).
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set the overlap between adjacent chunks.
    pub fn with_overlap(mut self, n: usize) -> Self {
        self.chunk_overlap = n;
        self
    }

    /// Set the separator string used to split text. Default: `"\n\n"`.
    pub fn with_separator(mut self, s: impl Into<String>) -> Self {
        self.separator = s.into();
        self
    }

    /// If `true`, the separator is prepended to subsequent chunks
    /// (preserving boundary context). Default: `false`.
    pub fn with_keep_separator(mut self, keep: bool) -> Self {
        self.keep_separator = keep;
        self
    }

    /// Replace the length measure (default: chars).
    pub fn with_length_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.length_fn = Arc::new(f);
        self
    }

    /// Pack a sequence of fragments into chunk-sized buckets, respecting
    /// `chunk_size` and applying `chunk_overlap` between adjacent chunks.
    fn pack_fragments(&self, fragments: Vec<String>) -> Vec<String> {
        let len = |s: &str| (self.length_fn)(s);
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for frag in fragments.into_iter() {
            if frag.is_empty() {
                continue;
            }
            // If the fragment alone exceeds chunk_size, hard-split by
            // characters so we never produce a chunk wider than the budget.
            if len(&frag) > self.chunk_size {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                let mut buf = String::new();
                for ch in frag.chars() {
                    let cand_len = len(&buf) + len(&ch.to_string());
                    if cand_len > self.chunk_size && !buf.is_empty() {
                        chunks.push(std::mem::take(&mut buf));
                    }
                    buf.push(ch);
                }
                if !buf.is_empty() {
                    chunks.push(buf);
                }
                continue;
            }
            // Try to append to `current` if there's room.
            let separator_cost = if current.is_empty() {
                0
            } else {
                len(&self.separator)
            };
            if len(&current) + separator_cost + len(&frag) <= self.chunk_size {
                if !current.is_empty() {
                    current.push_str(&self.separator);
                }
                current.push_str(&frag);
            } else {
                chunks.push(std::mem::take(&mut current));
                current = frag;
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }

        if self.chunk_overlap > 0 && chunks.len() > 1 {
            for i in 1..chunks.len() {
                let prev = chunks[i - 1].clone();
                let prev_chars: Vec<char> = prev.chars().collect();
                let take = self.chunk_overlap.min(prev_chars.len());
                let tail: String = prev_chars[prev_chars.len() - take..].iter().collect();
                let mut new = tail;
                new.push_str(&chunks[i]);
                chunks[i] = new;
            }
        }
        chunks
    }
}

impl TextSplitter for CharacterSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        if doc.content.is_empty() {
            return Vec::new();
        }
        // Split on the configured separator.
        let raw: Vec<&str> = doc.content.split(self.separator.as_str()).collect();
        let fragments: Vec<String> = if self.keep_separator {
            // Re-attach the separator to the start of each non-first piece.
            raw.iter()
                .enumerate()
                .map(|(i, p)| {
                    if i == 0 {
                        (*p).to_string()
                    } else {
                        format!("{}{}", self.separator, p)
                    }
                })
                .collect()
        } else {
            raw.iter().map(|p| (*p).to_string()).collect()
        };
        let chunks = self.pack_fragments(fragments);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, c)| child_doc(doc, c, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    #[test]
    fn splits_on_default_double_newline() {
        let s = CharacterSplitter::new().with_chunk_size(50).with_overlap(0);
        let d = doc("para one\n\npara two\n\npara three");
        let out = s.split(&d);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("para one"));
        assert!(out[0].content.contains("para three"));
    }

    #[test]
    fn packs_fragments_to_chunk_size() {
        let s = CharacterSplitter::new()
            .with_chunk_size(15)
            .with_overlap(0)
            .with_separator("|");
        let d = doc("aaaaa|bbbbb|ccccc|ddddd");
        let out = s.split(&d);
        // Each pair fits in 15 (5 + 1 + 5 = 11). Three pairs would be
        // 5+1+5+1+5 = 17 > 15. Expect: ["aaaaa|bbbbb", "ccccc|ddddd"].
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, "aaaaa|bbbbb");
        assert_eq!(out[1].content, "ccccc|ddddd");
    }

    #[test]
    fn applies_overlap_between_chunks() {
        let s = CharacterSplitter::new()
            .with_chunk_size(10)
            .with_overlap(3)
            .with_separator("|");
        let d = doc("aaaa|bbbb|cccc");
        let out = s.split(&d);
        assert!(out.len() >= 2);
        // The second chunk should start with the last 3 chars of the first.
        let prev = &out[0].content;
        let prev_tail: String = prev
            .chars()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        assert!(out[1].content.starts_with(&prev_tail));
    }

    #[test]
    fn keep_separator_preserves_delimiter() {
        let s = CharacterSplitter::new()
            .with_chunk_size(20)
            .with_overlap(0)
            .with_separator("\n\n")
            .with_keep_separator(true);
        let d = doc("one\n\ntwo\n\nthree");
        let out = s.split(&d);
        // Joined content should still contain the separator at chunk boundaries.
        let joined = out
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("|");
        assert!(joined.contains("\n\ntwo"));
    }

    #[test]
    fn hard_splits_oversized_fragment() {
        let s = CharacterSplitter::new()
            .with_chunk_size(5)
            .with_overlap(0)
            .with_separator("|");
        let d = doc("abcdefghij");
        let out = s.split(&d);
        // No separator in the input, so single fragment of length 10
        // gets hard-split into 5+5.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content.chars().count(), 5);
        assert_eq!(out[1].content.chars().count(), 5);
    }

    #[test]
    fn custom_length_fn_used() {
        // Pretend each "word" costs 5 units; this lets a 1-word fragment
        // that's 1 char long still be considered "huge".
        let s = CharacterSplitter::new()
            .with_chunk_size(7) // budget 7 units
            .with_overlap(0)
            .with_separator(" ")
            .with_length_fn(|s: &str| s.split_whitespace().count() * 5);
        let d = doc("a b c");
        // Each fragment is 1 word → 5 units. With separator of 1 word
        // (counted: 0 since separator " " has 0 words), two frags fit
        // (10 > 7? yes — so only one per chunk).
        let out = s.split(&d);
        assert!(out.len() >= 2);
    }

    #[test]
    fn empty_doc_returns_no_chunks() {
        let s = CharacterSplitter::new();
        let d = doc("");
        let out = s.split(&d);
        assert!(out.is_empty());
    }

    #[test]
    fn metadata_propagates_to_children() {
        let mut d = doc("aaa|bbb");
        d.metadata.insert(
            "source".into(),
            serde_json::Value::String("file.txt".into()),
        );
        let s = CharacterSplitter::new()
            .with_chunk_size(3)
            .with_overlap(0)
            .with_separator("|");
        let out = s.split(&d);
        assert!(out.iter().all(
            |c| c.metadata.get("source") == Some(&serde_json::Value::String("file.txt".into()))
        ));
        // chunk_index is set on every child.
        assert!(out.iter().enumerate().all(|(i, c)| {
            c.metadata.get("chunk_index").and_then(|v| v.as_u64()) == Some(i as u64)
        }));
    }
}
