use super::TextSplitter;

/// Splits text by approximate token count (chars / chars_per_token).
pub struct TokenTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub chars_per_token: usize,
}

impl Default for TokenTextSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 100,
            chars_per_token: 4,
        }
    }
}

impl TokenTextSplitter {
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

    pub fn with_chars_per_token(mut self, cpt: usize) -> Self {
        self.chars_per_token = cpt;
        self
    }
}

impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let max_chars = self.chunk_size * self.chars_per_token;
        let overlap_chars = self.chunk_overlap * self.chars_per_token;
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + max_chars).min(text.len());
            // Try to break at a space
            let actual_end = if end < text.len() {
                text[start..end]
                    .rfind(' ')
                    .map(|pos| start + pos)
                    .unwrap_or(end)
            } else {
                end
            };
            let chunk = text[start..actual_end].trim();
            if !chunk.is_empty() {
                chunks.push(chunk.to_string());
            }
            let new_start = if actual_end > overlap_chars {
                actual_end - overlap_chars
            } else {
                actual_end
            };
            // Ensure we always advance to prevent infinite loops
            if new_start <= start {
                start = actual_end.max(start + 1);
            } else {
                start = new_start;
            }
        }
        chunks
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}
