use super::code::Language;
use super::{merge_splits, TextSplitter};

/// Recursively splits text trying each separator in order.
/// The most important text splitter -- used for most real-world cases.
pub struct RecursiveCharacterTextSplitter {
    pub separators: Vec<String>,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub keep_separator: bool,
}

impl Default for RecursiveCharacterTextSplitter {
    fn default() -> Self {
        Self {
            separators: vec!["\n\n".into(), "\n".into(), " ".into(), "".into()],
            chunk_size: 4000,
            chunk_overlap: 200,
            keep_separator: false,
        }
    }
}

impl RecursiveCharacterTextSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    pub fn with_separators(mut self, seps: Vec<String>) -> Self {
        self.separators = seps;
        self
    }

    /// Create a splitter configured for a specific programming language.
    pub fn from_language(lang: Language) -> Self {
        Self {
            separators: lang.get_separators(),
            ..Self::default()
        }
    }

    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let mut final_chunks: Vec<String> = Vec::new();

        // Find the appropriate separator
        let mut separator = separators.last().map(|s| s.as_str()).unwrap_or("");
        let mut new_separators: &[String] = &[];

        for (i, sep) in separators.iter().enumerate() {
            if sep.is_empty() || text.contains(sep.as_str()) {
                separator = sep.as_str();
                new_separators = &separators[i + 1..];
                break;
            }
        }

        // For empty separator, just do character-level merge
        if separator.is_empty() {
            let char_strs: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let refs: Vec<&str> = char_strs.iter().map(|s| s.as_str()).collect();
            return merge_splits(&refs, "", self.chunk_size, self.chunk_overlap);
        }

        let splits: Vec<&str> = text.split(separator).collect();

        let mut good_splits: Vec<&str> = Vec::new();

        for s in &splits {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.len() < self.chunk_size {
                good_splits.push(trimmed);
            } else {
                if !good_splits.is_empty() {
                    let merged =
                        merge_splits(&good_splits, separator, self.chunk_size, self.chunk_overlap);
                    final_chunks.extend(merged);
                    good_splits.clear();
                }
                if new_separators.is_empty() {
                    final_chunks.push(trimmed.to_string());
                } else {
                    let sub = self.split_text_recursive(trimmed, new_separators);
                    final_chunks.extend(sub);
                }
            }
        }

        if !good_splits.is_empty() {
            let merged = merge_splits(&good_splits, separator, self.chunk_size, self.chunk_overlap);
            final_chunks.extend(merged);
        }

        final_chunks
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_text_recursive(text, &self.separators)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}
