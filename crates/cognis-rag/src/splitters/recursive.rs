//! Recursive character splitter — tries successively coarser separators
//! (paragraph → line → sentence → word) until each chunk fits the target size.

use crate::document::Document;

use super::{child_doc, TextSplitter};

/// Splits text by character count, falling back through a list of
/// separators until each chunk is at most `chunk_size` characters.
///
/// Adjacent chunks share a `chunk_overlap` window so context isn't lost
/// at boundaries. Operates on Rust `&str` and counts characters, not tokens.
pub struct RecursiveCharSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
}

impl Default for RecursiveCharSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
                "".to_string(),
            ],
        }
    }
}

impl RecursiveCharSplitter {
    /// Construct with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target maximum chunk size (in chars).
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set the overlap window between adjacent chunks.
    pub fn with_overlap(mut self, n: usize) -> Self {
        self.chunk_overlap = n;
        self
    }

    /// Override the separator list (tried in order, coarsest first).
    pub fn with_separators<I, S>(mut self, seps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.separators = seps.into_iter().map(Into::into).collect();
        self
    }

    fn split_text(&self, text: &str) -> Vec<String> {
        let pieces = self.recurse(text, 0);
        // Merge adjacent pieces up to chunk_size with overlap.
        merge_with_overlap(pieces, self.chunk_size, self.chunk_overlap)
    }

    fn recurse(&self, text: &str, sep_idx: usize) -> Vec<String> {
        if text.chars().count() <= self.chunk_size {
            return if text.is_empty() {
                vec![]
            } else {
                vec![text.to_string()]
            };
        }
        let separator = match self.separators.get(sep_idx) {
            Some(s) if !s.is_empty() => s.clone(),
            _ => return hard_split(text, self.chunk_size),
        };

        let mut pieces = Vec::new();
        for piece in text.split(&separator) {
            if piece.chars().count() <= self.chunk_size {
                if !piece.is_empty() {
                    pieces.push(piece.to_string());
                }
            } else {
                pieces.extend(self.recurse(piece, sep_idx + 1));
            }
        }
        pieces
    }
}

fn merge_with_overlap(pieces: Vec<String>, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for piece in pieces {
        if buf.chars().count() + piece.chars().count() < chunk_size {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(&piece);
        } else {
            if !buf.is_empty() {
                out.push(buf.clone());
            }
            buf = if overlap > 0 {
                let tail: String = out
                    .last()
                    .map(|s| {
                        let n = s.chars().count();
                        let start = n.saturating_sub(overlap);
                        s.chars().skip(start).collect()
                    })
                    .unwrap_or_default();
                if tail.is_empty() {
                    piece
                } else {
                    format!("{tail} {piece}")
                }
            } else {
                piece
            };
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn hard_split(text: &str, chunk_size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(chunk_size.max(1))
        .map(|c| c.iter().collect::<String>())
        .collect()
}

impl TextSplitter for RecursiveCharSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        self.split_text(&doc.content)
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
    fn splits_paragraphs_first() {
        let text = "p1.\n\np2.\n\np3.";
        let s = RecursiveCharSplitter::new()
            .with_chunk_size(4)
            .with_overlap(0);
        let chunks = s.split_text(text);
        assert!(chunks.iter().all(|c| c.chars().count() <= 4));
        assert!(chunks.iter().any(|c| c.contains("p1")));
        assert!(chunks.iter().any(|c| c.contains("p3")));
    }

    #[test]
    fn falls_through_to_chars_for_long_run() {
        let s = RecursiveCharSplitter::new()
            .with_chunk_size(3)
            .with_overlap(0);
        let chunks = s.split_text("abcdefghij");
        assert!(chunks.iter().all(|c| c.chars().count() <= 3));
        assert_eq!(chunks.concat().replace(' ', ""), "abcdefghij");
    }

    #[test]
    fn small_text_returns_one_chunk() {
        let s = RecursiveCharSplitter::new().with_chunk_size(100);
        let chunks = s.split_text("hi");
        assert_eq!(chunks, vec!["hi".to_string()]);
    }

    #[test]
    fn split_doc_propagates_metadata() {
        let doc = Document::new("a b c d e f g h i j k").with_metadata("source", "f.txt");
        let s = RecursiveCharSplitter::new()
            .with_chunk_size(5)
            .with_overlap(0);
        let chunks = s.split(&doc);
        assert!(!chunks.is_empty());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.metadata["source"], "f.txt");
            assert_eq!(c.metadata["chunk_index"], serde_json::json!(i));
        }
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        let s = RecursiveCharSplitter::new().with_chunk_size(10);
        assert!(s.split_text("").is_empty());
    }
}
