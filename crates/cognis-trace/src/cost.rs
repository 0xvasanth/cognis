//! USD-cost computation. Per-million-token rates by model id.
//!
//! Defaults are a dated snapshot — clearly marked. Users can override or add
//! models. See spec §8.

use std::collections::HashMap;

use crate::span::{CostDetails, TokenUsage};

/// Per-million-token rates for one model.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelPrice {
    /// USD per 1M input tokens.
    pub input: f64,
    /// USD per 1M output tokens.
    pub output: f64,
    /// USD per 1M cache-read tokens.
    pub cache_read: f64,
    /// USD per 1M cache-write tokens.
    pub cache_write: f64,
}

/// Map of model id to per-million-token rates.
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    inner: HashMap<String, ModelPrice>,
}

impl PriceTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-populated with `default_pricing_2026_05`.
    pub fn with_defaults() -> Self {
        default_pricing_2026_05()
    }

    /// Insert or replace a price entry.
    pub fn insert(&mut self, model_id: impl Into<String>, price: ModelPrice) -> &mut Self {
        self.inner.insert(model_id.into(), price);
        self
    }

    /// Lookup, exact match first, then prefix match (e.g. "gpt-4o" matches
    /// "gpt-4o-2024-08-06"). Returns `None` if no match.
    pub fn get(&self, model_id: &str) -> Option<&ModelPrice> {
        if let Some(p) = self.inner.get(model_id) {
            return Some(p);
        }
        self.inner
            .iter()
            .filter(|(k, _)| model_id.starts_with(k.as_str()))
            .max_by_key(|(k, _)| k.len())
            .map(|(_, v)| v)
    }

    /// Compute structured cost from token usage. Returns `None` when the
    /// model is unknown.
    pub fn compute(&self, model_id: &str, usage: TokenUsage) -> Option<CostDetails> {
        let p = self.get(model_id)?;
        let scale = 1_000_000.0;
        let input = (usage.input as f64) * p.input / scale;
        let output = (usage.output as f64) * p.output / scale;
        let cache_read = (usage.cache_read as f64) * p.cache_read / scale;
        let cache_write = (usage.cache_write as f64) * p.cache_write / scale;
        Some(CostDetails {
            input,
            output,
            cache_read,
            cache_write,
            total: input + output + cache_read + cache_write,
        })
    }
}

/// Snapshot of mainstream provider prices captured 2026-05-06.
/// Stale entries can be overridden with `PriceTable::insert`.
pub fn default_pricing_2026_05() -> PriceTable {
    let mut t = PriceTable::new();

    // OpenAI (USD per 1M tokens, 2026-05 snapshot).
    t.insert("gpt-4o", ModelPrice { input: 2.50, output: 10.00, cache_read: 1.25, cache_write: 0.0 });
    t.insert("gpt-4o-mini", ModelPrice { input: 0.15, output: 0.60, cache_read: 0.075, cache_write: 0.0 });
    t.insert("o1", ModelPrice { input: 15.00, output: 60.00, cache_read: 7.50, cache_write: 0.0 });
    t.insert("o1-mini", ModelPrice { input: 3.00, output: 12.00, cache_read: 1.50, cache_write: 0.0 });

    // Anthropic.
    t.insert("claude-opus-4", ModelPrice { input: 15.00, output: 75.00, cache_read: 1.50, cache_write: 18.75 });
    t.insert("claude-sonnet-4", ModelPrice { input: 3.00, output: 15.00, cache_read: 0.30, cache_write: 3.75 });
    t.insert("claude-haiku-4", ModelPrice { input: 0.80, output: 4.00, cache_read: 0.08, cache_write: 1.00 });

    // Google.
    t.insert("gemini-2.0-flash", ModelPrice { input: 0.10, output: 0.40, cache_read: 0.025, cache_write: 0.0 });
    t.insert("gemini-1.5-pro", ModelPrice { input: 1.25, output: 5.00, cache_read: 0.3125, cache_write: 0.0 });

    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_returns_none() {
        let t = PriceTable::new();
        assert!(t.compute("gpt-4o", TokenUsage::default()).is_none());
    }

    #[test]
    fn exact_match_used_first() {
        let mut t = PriceTable::new();
        t.insert("gpt-4o-2024-08-06", ModelPrice { input: 1.0, output: 2.0, cache_read: 0.0, cache_write: 0.0 });
        t.insert("gpt-4o", ModelPrice { input: 99.0, output: 99.0, cache_read: 0.0, cache_write: 0.0 });
        let p = t.get("gpt-4o-2024-08-06").unwrap();
        assert_eq!(p.input, 1.0);
    }

    #[test]
    fn prefix_match_falls_back() {
        let mut t = PriceTable::new();
        t.insert("gpt-4o", ModelPrice { input: 2.50, output: 10.00, cache_read: 1.25, cache_write: 0.0 });
        let p = t.get("gpt-4o-2024-08-06").unwrap();
        assert_eq!(p.input, 2.50);
    }

    #[test]
    fn longest_prefix_wins() {
        let mut t = PriceTable::new();
        t.insert("gpt", ModelPrice { input: 1.0, output: 1.0, cache_read: 0.0, cache_write: 0.0 });
        t.insert("gpt-4o", ModelPrice { input: 2.50, output: 10.00, cache_read: 0.0, cache_write: 0.0 });
        let p = t.get("gpt-4o-2024-08-06").unwrap();
        assert_eq!(p.input, 2.50);
    }

    #[test]
    fn compute_scales_per_million_tokens() {
        let mut t = PriceTable::new();
        t.insert("gpt-4o", ModelPrice { input: 2.50, output: 10.00, cache_read: 1.25, cache_write: 0.0 });
        let usage = TokenUsage { input: 1_000_000, output: 500_000, cache_read: 0, cache_write: 0 };
        let c = t.compute("gpt-4o", usage).unwrap();
        assert!((c.input - 2.50).abs() < 1e-9);
        assert!((c.output - 5.00).abs() < 1e-9);
        assert!((c.total - 7.50).abs() < 1e-9);
    }

    #[test]
    fn defaults_contains_mainstream_models() {
        let t = default_pricing_2026_05();
        assert!(t.get("gpt-4o").is_some());
        assert!(t.get("claude-sonnet-4").is_some());
        assert!(t.get("gemini-2.0-flash").is_some());
    }
}
