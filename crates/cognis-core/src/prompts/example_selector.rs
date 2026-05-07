//! Few-shot example selection — pluggable strategies for picking a
//! subset of examples to include in a prompt.
//!
//! [`ExampleSelector<E>`] is the core trait. Two stock implementations
//! ship in `cognis-core`:
//!
//! - [`StaticExampleSelector`] — passes through a fixed list (useful as
//!   a baseline / for tests).
//! - [`LengthBasedExampleSelector`] — picks examples in order until a
//!   token budget is exhausted, using any [`crate::tokenizer::Tokenizer`].
//!
//! Embedding-driven selectors (e.g. semantic similarity, MMR) live in
//! `cognis-rag` because they require an `Embeddings` trait.
//!
//! Custom selectors plug in by implementing `ExampleSelector<E>`. The
//! crate ships traits plus default impls; nothing about the selector
//! contract is closed.

use std::sync::Arc;

use crate::tokenizer::Tokenizer;
use crate::{CognisError, Result};

/// Strategy for choosing which examples to include in a few-shot prompt.
///
/// Implementations receive an `input` value (typed) and the full pool of
/// available examples, and return the examples to use, in render order.
#[allow(clippy::ptr_arg)] // examples are stored as Vec to enable owned slicing
pub trait ExampleSelector<E>: Send + Sync
where
    E: Send + Sync + 'static,
{
    /// Select examples from `examples` to include for `input`. Returns
    /// owned clones (or constructs new examples) so the caller doesn't
    /// have to manage lifetimes against the pool.
    fn select(&self, input: &str, examples: &[E]) -> Result<Vec<E>>;
}

// ---------------------------------------------------------------------------
// StaticExampleSelector
// ---------------------------------------------------------------------------

/// Trivial selector that always returns the entire example list,
/// optionally truncated to `max`. Useful when a fixed shot count is
/// desired without any dynamic logic.
#[derive(Debug, Clone, Default)]
pub struct StaticExampleSelector {
    max: Option<usize>,
}

impl StaticExampleSelector {
    /// Pass-through: every example is returned.
    pub fn all() -> Self {
        Self { max: None }
    }

    /// Cap the returned examples to at most `n`.
    pub fn at_most(n: usize) -> Self {
        Self { max: Some(n) }
    }
}

impl<E> ExampleSelector<E> for StaticExampleSelector
where
    E: Clone + Send + Sync + 'static,
{
    fn select(&self, _input: &str, examples: &[E]) -> Result<Vec<E>> {
        Ok(match self.max {
            Some(n) => examples.iter().take(n).cloned().collect(),
            None => examples.to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// LengthBasedExampleSelector
// ---------------------------------------------------------------------------

/// Function that converts an example into the text used for token
/// counting. Often the same template used to render the example into
/// the prompt; supplying it explicitly lets the selector budget against
/// the actual prompt cost rather than a placeholder.
pub type ExampleRenderFn<E> = Arc<dyn Fn(&E) -> String + Send + Sync>;

/// Greedy: include each example in order until adding the next would
/// exceed the configured `max_tokens` budget. The `input` itself is
/// counted against the budget so callers know how much head-room is
/// left for completion.
#[derive(Clone)]
pub struct LengthBasedExampleSelector<E> {
    max_tokens: usize,
    tokenizer: Arc<dyn Tokenizer>,
    render: ExampleRenderFn<E>,
}

impl<E> std::fmt::Debug for LengthBasedExampleSelector<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LengthBasedExampleSelector")
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl<E> LengthBasedExampleSelector<E>
where
    E: Send + Sync + 'static,
{
    /// Build a length-budgeted selector.
    ///
    /// `render` converts an example to the text that will be substituted
    /// into the prompt — used by the tokenizer to size each example.
    pub fn new<F>(max_tokens: usize, tokenizer: Arc<dyn Tokenizer>, render: F) -> Self
    where
        F: Fn(&E) -> String + Send + Sync + 'static,
    {
        Self {
            max_tokens,
            tokenizer,
            render: Arc::new(render),
        }
    }

    /// Replace the render function (e.g. when the prompt template changes).
    pub fn with_render<F>(mut self, render: F) -> Self
    where
        F: Fn(&E) -> String + Send + Sync + 'static,
    {
        self.render = Arc::new(render);
        self
    }

    /// Replace the tokenizer (e.g. swap `CharTokenizer` for `tiktoken`).
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }
}

impl<E> ExampleSelector<E> for LengthBasedExampleSelector<E>
where
    E: Clone + Send + Sync + 'static,
{
    fn select(&self, input: &str, examples: &[E]) -> Result<Vec<E>> {
        let mut budget = self
            .max_tokens
            .checked_sub(self.tokenizer.count(input))
            .ok_or_else(|| {
                CognisError::Configuration(
                    "LengthBasedExampleSelector: input alone exceeds max_tokens".into(),
                )
            })?;
        let mut out = Vec::new();
        for ex in examples {
            let cost = self.tokenizer.count(&(self.render)(ex));
            if cost > budget {
                break;
            }
            budget -= cost;
            out.push(ex.clone());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::CharTokenizer;

    #[test]
    fn static_selector_returns_all_by_default() {
        let s = StaticExampleSelector::all();
        let pool = vec!["a", "b", "c"];
        let out: Vec<&str> = ExampleSelector::select(&s, "ignored", &pool).unwrap();
        assert_eq!(out, pool);
    }

    #[test]
    fn static_selector_caps_at_most() {
        let s = StaticExampleSelector::at_most(2);
        let pool = vec!["a", "b", "c"];
        let out: Vec<&str> = ExampleSelector::select(&s, "ignored", &pool).unwrap();
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn length_based_stops_at_budget() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        let sel: LengthBasedExampleSelector<String> =
            LengthBasedExampleSelector::new(20, tokenizer, |s: &String| s.clone());
        let pool = vec![
            "five5".to_string(),  // 5 chars
            "five5".to_string(),  // 5 chars (cumulative 10)
            "five5".to_string(),  // 5 chars (cumulative 15) — input is 5 → budget 15
            "ovrflw".to_string(), // 6 chars — would exceed
        ];
        let picked = sel.select("input", &pool).unwrap();
        // Budget is 20 - 5(input) = 15. First three 5-char examples fit
        // exactly; "ovrflw" is rejected.
        assert_eq!(picked.len(), 3);
    }

    #[test]
    fn length_based_rejects_input_alone_too_big() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        let sel: LengthBasedExampleSelector<String> =
            LengthBasedExampleSelector::new(3, tokenizer, |s: &String| s.clone());
        let err = sel.select("longer-than-budget", &[]).unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[test]
    fn length_based_with_custom_renderer() {
        // Render fn doubles the cost of each example.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharTokenizer);
        let sel: LengthBasedExampleSelector<String> =
            LengthBasedExampleSelector::new(10, tokenizer, |s: &String| s.clone() + s);
        let pool = vec![
            "ab".to_string(),  // rendered as "abab" → 4
            "ab".to_string(),  // 4
            "abc".to_string(), // 6 — would push to 14, rejected (budget 10-input=10)
        ];
        let picked = sel.select("", &pool).unwrap();
        assert_eq!(picked.len(), 2);
    }
}
