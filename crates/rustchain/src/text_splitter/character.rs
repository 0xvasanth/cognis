use super::{merge_splits, TextSplitter};

/// Splits text on a single character separator.
pub struct CharacterTextSplitter {
    pub separator: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub keep_separator: bool,
}

impl Default for CharacterTextSplitter {
    fn default() -> Self {
        Self {
            separator: "\n\n".to_string(),
            chunk_size: 4000,
            chunk_overlap: 200,
            keep_separator: false,
        }
    }
}

impl CharacterTextSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
}

impl TextSplitter for CharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let splits: Vec<&str> = if self.separator.is_empty() {
            text.chars().map(|_| "").collect() // fallback
        } else {
            text.split(&self.separator).collect()
        };

        let good_splits: Vec<&str> = splits
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        merge_splits(&good_splits, &self.separator, self.chunk_size, self.chunk_overlap)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}
