use super::TextSplitter;
use rustchain_core::utils::tokens::{estimate_token_count, get_model_context_window};

/// A text splitter that respects token limits rather than character limits.
///
/// Uses heuristic token estimation (or model-aware estimation) to split text
/// into chunks that fit within a specified token budget. Supports configurable
/// overlap between chunks and hierarchical separator-based splitting.
pub struct TokenAwareTextSplitter {
    /// Maximum tokens per chunk.
    pub max_tokens: usize,
    /// Number of overlap tokens between consecutive chunks.
    pub overlap_tokens: usize,
    /// Optional model name for more accurate token estimation.
    pub model_name: Option<String>,
    /// Separators to try in priority order (highest priority first).
    pub separators: Vec<String>,
}

impl Default for TokenAwareTextSplitter {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            overlap_tokens: 50,
            model_name: None,
            separators: vec![
                "\n\n".into(),
                "\n".into(),
                ". ".into(),
                " ".into(),
            ],
        }
    }
}

impl TokenAwareTextSplitter {
    /// Create a new `TokenAwareTextSplitter` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum tokens per chunk.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the overlap tokens between chunks.
    pub fn with_overlap_tokens(mut self, overlap_tokens: usize) -> Self {
        self.overlap_tokens = overlap_tokens;
        self
    }

    /// Set the model name for token estimation context.
    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    /// Set custom separators (highest priority first).
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    /// Create a splitter sized for a specific model's context window.
    ///
    /// Divides the model's context window by `chunks_per_context` to determine
    /// `max_tokens`. Falls back to a default of 2000 tokens if the model is not
    /// recognized.
    pub fn from_model_context(model_name: &str, chunks_per_context: usize) -> Self {
        let context_window = get_model_context_window(model_name).unwrap_or(2000);
        let max_tokens = if chunks_per_context > 0 {
            context_window / chunks_per_context
        } else {
            context_window
        };
        Self {
            max_tokens,
            overlap_tokens: 50,
            model_name: Some(model_name.to_string()),
            separators: vec![
                "\n\n".into(),
                "\n".into(),
                ". ".into(),
                " ".into(),
            ],
        }
    }

    /// Estimate the token count for a piece of text.
    fn estimate_tokens(text: &str, _model: Option<&str>) -> usize {
        estimate_token_count(text)
    }

    /// Split text at the highest-priority separator that produces sub-chunks,
    /// then merge small pieces and add overlap.
    fn split_with_separators(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let text_tokens = Self::estimate_tokens(text, self.model_name.as_deref());
        if text_tokens <= self.max_tokens {
            return vec![text.to_string()];
        }

        // Find the highest-priority separator present in the text.
        let separator = self
            .separators
            .iter()
            .find(|sep| text.contains(sep.as_str()))
            .cloned();

        let pieces: Vec<&str> = match &separator {
            Some(sep) => text.split(sep.as_str()).collect(),
            None => {
                // No separator found; fall back to word-level splitting on whitespace.
                text.split_whitespace().collect()
            }
        };

        // Filter out empty pieces.
        let pieces: Vec<&str> = pieces.iter().copied().filter(|p| !p.is_empty()).collect();

        // Merge small pieces into chunks that fit within max_tokens.
        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut current_tokens: usize = 0;

        for piece in &pieces {
            let piece_tokens = Self::estimate_tokens(piece, self.model_name.as_deref());

            // If a single piece exceeds max_tokens, try to split it further
            // with lower-priority separators.
            if piece_tokens > self.max_tokens {
                // Flush current buffer first.
                if !current.is_empty() {
                    chunks.push(current.trim().to_string());
                    current = String::new();
                    current_tokens = 0;
                }
                let sub_chunks = self.split_subsection(piece);
                chunks.extend(sub_chunks);
                continue;
            }

            let sep_str = separator.as_deref().unwrap_or(" ");
            let would_be = if current.is_empty() {
                piece_tokens
            } else {
                current_tokens
                    + Self::estimate_tokens(sep_str, self.model_name.as_deref())
                    + piece_tokens
            };

            if would_be > self.max_tokens && !current.is_empty() {
                chunks.push(current.trim().to_string());
                current = String::new();
                current_tokens = 0;
            }

            if current.is_empty() {
                current = piece.to_string();
                current_tokens = piece_tokens;
            } else {
                current.push_str(separator.as_deref().unwrap_or(" "));
                current.push_str(piece);
                current_tokens = Self::estimate_tokens(&current, self.model_name.as_deref());
            }
        }

        if !current.is_empty() {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                chunks.push(trimmed);
            }
        }

        // Apply overlap between consecutive chunks.
        if self.overlap_tokens > 0 && chunks.len() > 1 {
            chunks = self.apply_overlap(chunks);
        }

        chunks
    }

    /// Try splitting a subsection using lower-priority separators.
    fn split_subsection(&self, text: &str) -> Vec<String> {
        for sep in &self.separators {
            if text.contains(sep.as_str()) {
                let pieces: Vec<&str> =
                    text.split(sep.as_str()).filter(|p| !p.is_empty()).collect();
                if pieces.len() > 1 {
                    let mut sub_chunks = Vec::new();
                    let mut current = String::new();
                    let mut current_tokens: usize = 0;

                    for piece in &pieces {
                        let piece_tokens =
                            Self::estimate_tokens(piece, self.model_name.as_deref());
                        let would_be = if current.is_empty() {
                            piece_tokens
                        } else {
                            current_tokens
                                + Self::estimate_tokens(sep, self.model_name.as_deref())
                                + piece_tokens
                        };

                        if would_be > self.max_tokens && !current.is_empty() {
                            sub_chunks.push(current.trim().to_string());
                            current = String::new();
                            current_tokens = 0;
                        }

                        if current.is_empty() {
                            current = piece.to_string();
                            current_tokens = piece_tokens;
                        } else {
                            current.push_str(sep);
                            current.push_str(piece);
                            current_tokens =
                                Self::estimate_tokens(&current, self.model_name.as_deref());
                        }
                    }

                    if !current.is_empty() {
                        let trimmed = current.trim().to_string();
                        if !trimmed.is_empty() {
                            sub_chunks.push(trimmed);
                        }
                    }
                    return sub_chunks;
                }
            }
        }
        // Cannot split further; return as-is.
        vec![text.to_string()]
    }

    /// Add overlap from the end of the previous chunk to the start of the next.
    fn apply_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::with_capacity(chunks.len());
        result.push(chunks[0].clone());

        for i in 1..chunks.len() {
            let prev = &chunks[i - 1];
            let overlap_text = self.get_overlap_suffix(prev);
            if overlap_text.is_empty() {
                result.push(chunks[i].clone());
            } else {
                let merged = format!("{} {}", overlap_text.trim(), chunks[i].trim());
                // Only use overlap if the merged chunk still fits.
                let merged_tokens = Self::estimate_tokens(&merged, self.model_name.as_deref());
                if merged_tokens <= self.max_tokens {
                    result.push(merged);
                } else {
                    result.push(chunks[i].clone());
                }
            }
        }

        result
    }

    /// Extract the last `overlap_tokens` worth of text from a string.
    fn get_overlap_suffix(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut suffix_words: Vec<&str> = Vec::new();
        let mut token_count = 0;

        for word in words.iter().rev() {
            let word_tokens = Self::estimate_tokens(word, self.model_name.as_deref());
            if token_count + word_tokens > self.overlap_tokens {
                break;
            }
            token_count += word_tokens;
            suffix_words.push(word);
        }

        suffix_words.reverse();
        suffix_words.join(" ")
    }
}

impl TextSplitter for TokenAwareTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_with_separators(text)
    }

    fn chunk_size(&self) -> usize {
        self.max_tokens
    }

    fn chunk_overlap(&self) -> usize {
        self.overlap_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_returns_single_chunk() {
        let splitter = TokenAwareTextSplitter::new().with_max_tokens(100);
        let result = splitter.split_text("Hello world.");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Hello world.");
    }

    #[test]
    fn test_long_text_splits_into_multiple_chunks() {
        let splitter = TokenAwareTextSplitter::new()
            .with_max_tokens(10)
            .with_overlap_tokens(0);

        // Build text that is well over 10 tokens (~40 chars = ~10 tokens).
        let text = "The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";

        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );

        // Each chunk should be within the token limit (with small tolerance).
        for chunk in &chunks {
            let tokens = estimate_token_count(chunk);
            assert!(
                tokens <= splitter.max_tokens + 2,
                "Chunk has {} tokens, max is {}: {:?}",
                tokens,
                splitter.max_tokens,
                chunk
            );
        }
    }

    #[test]
    fn test_overlap_between_chunks() {
        let splitter = TokenAwareTextSplitter::new()
            .with_max_tokens(15)
            .with_overlap_tokens(5);

        let text = "Alpha beta gamma delta. Epsilon zeta eta theta. \
                     Iota kappa lambda mu. Nu xi omicron pi.";

        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1, "Expected multiple chunks for overlap test");

        // With overlap, later chunks should share some text with the previous chunk.
        let mut found_overlap = false;
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i - 1].split_whitespace().collect();
            let curr_words: Vec<&str> = chunks[i].split_whitespace().collect();
            for word in &prev_words {
                if curr_words.contains(word) && word.len() > 3 {
                    found_overlap = true;
                    break;
                }
            }
            if found_overlap {
                break;
            }
        }
        assert!(found_overlap, "Expected overlap between consecutive chunks");
    }

    #[test]
    fn test_custom_separators() {
        let splitter = TokenAwareTextSplitter::new()
            .with_max_tokens(10)
            .with_overlap_tokens(0)
            .with_separators(vec!["||".into()]);

        let text = "chunk one text here||chunk two text here||chunk three text here";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks with custom separator, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_from_model_context_factory() {
        let splitter = TokenAwareTextSplitter::from_model_context("gpt-4o", 10);
        // gpt-4o has 128_000 context window, divided by 10 = 12_800
        assert_eq!(splitter.max_tokens, 12_800);
        assert_eq!(splitter.model_name.as_deref(), Some("gpt-4o"));

        let splitter_claude = TokenAwareTextSplitter::from_model_context("claude-3-opus", 20);
        // claude-3-opus has 200_000 / 20 = 10_000
        assert_eq!(splitter_claude.max_tokens, 10_000);

        // Unknown model falls back to 2000 / 4 = 500
        let splitter_unknown = TokenAwareTextSplitter::from_model_context("unknown-model", 4);
        assert_eq!(splitter_unknown.max_tokens, 500);
    }

    #[test]
    fn test_empty_text_returns_empty_vec() {
        let splitter = TokenAwareTextSplitter::new();
        let result = splitter.split_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_chunk_size_and_overlap_trait_methods() {
        let splitter = TokenAwareTextSplitter::new()
            .with_max_tokens(256)
            .with_overlap_tokens(32);
        assert_eq!(splitter.chunk_size(), 256);
        assert_eq!(splitter.chunk_overlap(), 32);
    }
}
