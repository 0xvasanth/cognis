//! Sentence splitter — packs sentences into chunks up to a target size.

use crate::document::Document;

use super::{child_doc, TextSplitter};

/// Splits text into chunks at sentence boundaries (`.`, `!`, `?` followed
/// by whitespace or end-of-input). Adjacent sentences are packed into
/// chunks up to `chunk_size` characters.
pub struct SentenceSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 0,
        }
    }
}

impl SentenceSplitter {
    /// Construct.
    pub fn new() -> Self {
        Self::default()
    }
    /// Cap chunk size in characters.
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }
    /// Sentence-overlap (number of trailing sentences to repeat at the
    /// head of the next chunk).
    pub fn with_overlap_sentences(mut self, n: usize) -> Self {
        self.chunk_overlap = n;
        self
    }

    fn split_sentences(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut buf = String::new();
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len() {
            buf.push(chars[i]);
            if matches!(chars[i], '.' | '!' | '?') {
                let next = chars.get(i + 1);
                let is_boundary = matches!(next, Some(c) if c.is_whitespace()) || next.is_none();
                if is_boundary {
                    let s = buf.trim().to_string();
                    if !s.is_empty() {
                        out.push(s);
                    }
                    buf.clear();
                }
            }
        }
        let tail = buf.trim().to_string();
        if !tail.is_empty() {
            out.push(tail);
        }
        out
    }

    fn pack(&self, sentences: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut buf: Vec<String> = Vec::new();
        let mut len = 0usize;
        for s in sentences {
            let sl = s.chars().count();
            if !buf.is_empty() && len + sl + 1 > self.chunk_size {
                out.push(buf.join(" "));
                if self.chunk_overlap > 0 {
                    let keep = buf.len().saturating_sub(self.chunk_overlap);
                    buf = buf.split_off(keep);
                    len = buf.iter().map(|s| s.chars().count() + 1).sum();
                } else {
                    buf.clear();
                    len = 0;
                }
            }
            // After the flush+overlap step, the retained tail may already
            // be at/above chunk_size. Drop oldest sentences until the new
            // one fits — if the sentence alone is bigger than chunk_size,
            // we accept it as its own (oversized) chunk to avoid livelock.
            while !buf.is_empty() && len + sl + 1 > self.chunk_size {
                let dropped = buf.remove(0);
                len = len.saturating_sub(dropped.chars().count() + 1);
            }
            len += sl + 1;
            buf.push(s);
        }
        if !buf.is_empty() {
            out.push(buf.join(" "));
        }
        out
    }
}

impl TextSplitter for SentenceSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        let sentences = Self::split_sentences(&doc.content);
        self.pack(sentences)
            .into_iter()
            .enumerate()
            .map(|(i, c)| child_doc(doc, c, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_terminal_punctuation() {
        let s = SentenceSplitter::new().with_chunk_size(1000);
        let chunks = s.split(&Document::new("Hi there. How are you? I'm fine!"));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Hi there."));
        assert!(chunks[0].content.contains("I'm fine!"));
    }

    #[test]
    fn packs_into_size_bound() {
        let s = SentenceSplitter::new().with_chunk_size(15);
        let text = "One. Two. Three. Four. Five.";
        let chunks = s.split(&Document::new(text));
        assert!(chunks.iter().all(|c| c.content.chars().count() <= 15));
        assert!(chunks.len() >= 2);
    }
}
