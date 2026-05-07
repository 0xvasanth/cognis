//! Pluggable token counting trait.
//!
//! Lives in `cognis-core` so non-RAG code can budget tokens without
//! pulling in `cognis-rag`. Concrete tokenizer crates (tiktoken-rs, HF
//! tokenizers, etc.) are integrated by user code via the trait.

/// Counts tokens in a piece of text.
pub trait Tokenizer: Send + Sync {
    /// Number of tokens this tokenizer would produce for `text`.
    fn count(&self, text: &str) -> usize;
}

/// Trivial char-as-token implementation. Conservative upper bound on
/// real tokenizer counts; useful as a default for budgeting.
#[derive(Debug, Default, Clone, Copy)]
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn count(&self, text: &str) -> usize {
        text.chars().count()
    }
}

/// Closure-backed tokenizer.
pub struct FnTokenizer<F: Fn(&str) -> usize + Send + Sync>(pub F);

impl<F: Fn(&str) -> usize + Send + Sync> Tokenizer for FnTokenizer<F> {
    fn count(&self, text: &str) -> usize {
        (self.0)(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_counts_chars() {
        assert_eq!(CharTokenizer.count("hello"), 5);
    }

    #[test]
    fn fn_tokenizer_invokes_closure() {
        let t = FnTokenizer(|s: &str| s.split_whitespace().count());
        assert_eq!(t.count("hello rust world"), 3);
    }
}
