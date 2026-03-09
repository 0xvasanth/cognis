//! Advanced token counting and context window management.
//!
//! Provides trait-based token estimation, context window management with
//! priority-based trimming, token budget allocation across sections, and
//! usage tracking over time.

use serde_json::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TokenCounter trait
// ---------------------------------------------------------------------------

/// Trait for estimating token counts from text or message sequences.
pub trait TokenCounter: Send + Sync {
    /// Estimate the number of tokens in the given text.
    fn count(&self, text: &str) -> usize;

    /// Estimate the total number of tokens across a slice of JSON message values.
    ///
    /// Each value is expected to have a `"content"` field whose text is counted.
    /// A per-message overhead of 3 tokens is added, plus 3 tokens for the
    /// assistant reply priming.
    fn count_messages(&self, messages: &[Value]) -> usize {
        let mut total: usize = 0;
        for msg in messages {
            total += 3; // per-message overhead
            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                total += self.count(content);
            }
        }
        total + 3 // assistant priming
    }

    /// A human-readable name for this counter implementation.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SimpleTokenCounter
// ---------------------------------------------------------------------------

/// Estimates tokens by splitting on whitespace and punctuation, then applying
/// a ~0.75 words-per-token heuristic (i.e. ~1.33 tokens per word).
#[derive(Debug, Clone)]
pub struct SimpleTokenCounter;

impl SimpleTokenCounter {
    /// Create a new `SimpleTokenCounter`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleTokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCounter for SimpleTokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // Split on whitespace to get words, then divide by 0.75 (multiply by ~1.33).
        let word_count = text.split_whitespace().count();
        (word_count as f64 / 0.75).ceil() as usize
    }

    fn name(&self) -> &str {
        "simple"
    }
}

// ---------------------------------------------------------------------------
// CharBasedCounter
// ---------------------------------------------------------------------------

/// Estimates tokens by dividing character count by a configurable
/// characters-per-token ratio.
#[derive(Debug, Clone)]
pub struct CharBasedCounter {
    chars_per_token: f64,
}

impl CharBasedCounter {
    /// Create a new `CharBasedCounter` with the given characters-per-token ratio.
    pub fn new(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }
}

impl Default for CharBasedCounter {
    fn default() -> Self {
        Self::new(4.0)
    }
}

impl TokenCounter for CharBasedCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    fn name(&self) -> &str {
        "char_based"
    }
}

// ---------------------------------------------------------------------------
// ModelTokenCounter
// ---------------------------------------------------------------------------

/// Model-specific token counter that uses known characters-per-token ratios
/// for popular model families.
#[derive(Debug, Clone)]
pub struct ModelTokenCounter {
    model_name: String,
    chars_per_token: f64,
}

impl ModelTokenCounter {
    /// Create a counter tuned for the given model name.
    ///
    /// Known ratios:
    /// - `gpt-4` family: 4.0
    /// - `claude` family: 3.5
    /// - `gemini` family: 4.0
    /// - everything else: 4.0
    pub fn for_model(model_name: &str) -> Self {
        let chars_per_token = if model_name.contains("claude") {
            3.5
        } else if model_name.contains("gpt-4") {
            4.0
        } else if model_name.contains("gemini") {
            4.0
        } else {
            4.0
        };
        Self {
            model_name: model_name.to_string(),
            chars_per_token,
        }
    }

    /// Return the characters-per-token ratio used by this counter.
    pub fn chars_per_token(&self) -> f64 {
        self.chars_per_token
    }
}

impl TokenCounter for ModelTokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Priority level for context items, used to decide trim order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Must never be trimmed.
    Critical,
    /// Trimmed only when lower priorities are exhausted.
    High,
    /// Default priority.
    Normal,
    /// Trimmed before Normal items.
    Low,
    /// Trimmed first.
    Optional,
}

impl Priority {
    /// Numeric weight (higher = more important).
    pub fn weight(&self) -> u32 {
        match self {
            Priority::Critical => 4,
            Priority::High => 3,
            Priority::Normal => 2,
            Priority::Low => 1,
            Priority::Optional => 0,
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight().cmp(&other.weight())
    }
}

// ---------------------------------------------------------------------------
// ContextItem
// ---------------------------------------------------------------------------

/// A single item stored in a [`ContextWindow`].
#[derive(Debug, Clone)]
pub struct ContextItem {
    /// The text content.
    pub text: String,
    /// Priority used for trim decisions.
    pub priority: Priority,
    /// Pre-computed token count for this item.
    pub token_count: usize,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl ContextItem {
    /// Serialize to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "text": self.text,
            "priority": format!("{:?}", self.priority),
            "token_count": self.token_count,
            "label": self.label,
        })
    }
}

// ---------------------------------------------------------------------------
// ContextWindow
// ---------------------------------------------------------------------------

/// Manages a collection of [`ContextItem`]s within a fixed token budget.
pub struct ContextWindow {
    max_tokens: usize,
    counter: Box<dyn TokenCounter>,
    items: Vec<ContextItem>,
}

impl ContextWindow {
    /// Create a new context window with the given token budget and counter.
    pub fn new(max_tokens: usize, counter: Box<dyn TokenCounter>) -> Self {
        Self {
            max_tokens,
            counter,
            items: Vec::new(),
        }
    }

    /// Add text with a given priority. Returns `false` if the item would exceed
    /// the remaining budget (the item is **not** added in that case).
    pub fn add(&mut self, text: &str, priority: Priority) -> bool {
        let token_count = self.counter.count(text);
        if self.used_tokens() + token_count > self.max_tokens {
            return false;
        }
        self.items.push(ContextItem {
            text: text.to_string(),
            priority,
            token_count,
            label: None,
        });
        true
    }

    /// Add text with a priority and a label. Returns `false` if it won't fit.
    pub fn add_labeled(&mut self, text: &str, priority: Priority, label: &str) -> bool {
        let token_count = self.counter.count(text);
        if self.used_tokens() + token_count > self.max_tokens {
            return false;
        }
        self.items.push(ContextItem {
            text: text.to_string(),
            priority,
            token_count,
            label: Some(label.to_string()),
        });
        true
    }

    /// Number of tokens still available.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.used_tokens())
    }

    /// Total tokens used by all items.
    pub fn used_tokens(&self) -> usize {
        self.items.iter().map(|i| i.token_count).sum()
    }

    /// Fraction of the budget currently used (0.0 – 1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.used_tokens() as f64 / self.max_tokens as f64
    }

    /// Return references to all items.
    pub fn content(&self) -> Vec<&ContextItem> {
        self.items.iter().collect()
    }

    /// Remove all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Remove the lowest-priority items until total usage fits within the budget.
    ///
    /// Items with `Priority::Critical` are never removed.
    pub fn trim_to_fit(&mut self) {
        while self.used_tokens() > self.max_tokens {
            // Find the index of the lowest-priority non-critical item.
            let candidate = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.priority != Priority::Critical)
                .min_by_key(|(_, item)| item.priority.weight());

            match candidate {
                Some((idx, _)) => {
                    self.items.remove(idx);
                }
                None => break, // only Critical items remain
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TokenBudget
// ---------------------------------------------------------------------------

/// Splits a total token budget across named sections, tracking allocations
/// and usage.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    total: usize,
    allocations: HashMap<String, usize>,
    used: HashMap<String, usize>,
}

impl TokenBudget {
    /// Create a new budget with the given total token capacity.
    pub fn new(total: usize) -> Self {
        Self {
            total,
            allocations: HashMap::new(),
            used: HashMap::new(),
        }
    }

    /// Allocate `tokens` to a named section, adding to any existing allocation.
    pub fn allocate(&mut self, section: &str, tokens: usize) {
        *self.allocations.entry(section.to_string()).or_insert(0) += tokens;
        self.used.entry(section.to_string()).or_insert(0);
    }

    /// Record that `tokens` have been consumed in the given section.
    pub fn use_tokens(&mut self, section: &str, tokens: usize) {
        *self.used.entry(section.to_string()).or_insert(0) += tokens;
    }

    /// Remaining tokens in a given section.
    pub fn remaining(&self, section: &str) -> usize {
        let allocated = self.allocations.get(section).copied().unwrap_or(0);
        let used = self.used.get(section).copied().unwrap_or(0);
        allocated.saturating_sub(used)
    }

    /// Tokens already used in a given section.
    pub fn used(&self, section: &str) -> usize {
        self.used.get(section).copied().unwrap_or(0)
    }

    /// Total remaining tokens across all sections (total minus all used).
    pub fn total_remaining(&self) -> usize {
        let total_used: usize = self.used.values().sum();
        self.total.saturating_sub(total_used)
    }

    /// List all section names.
    pub fn sections(&self) -> Vec<&str> {
        self.allocations.keys().map(|s| s.as_str()).collect()
    }

    /// Serialize to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        let sections: HashMap<&str, Value> = self
            .allocations
            .keys()
            .map(|k| {
                (
                    k.as_str(),
                    serde_json::json!({
                        "allocated": self.allocations[k],
                        "used": self.used.get(k).copied().unwrap_or(0),
                        "remaining": self.remaining(k),
                    }),
                )
            })
            .collect();
        serde_json::json!({
            "total": self.total,
            "total_remaining": self.total_remaining(),
            "sections": sections,
        })
    }
}

// ---------------------------------------------------------------------------
// TokenUsageTracker
// ---------------------------------------------------------------------------

/// Record of a single usage event.
#[derive(Debug, Clone)]
struct UsageRecord {
    model: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Tracks cumulative token usage across models and requests.
#[derive(Debug, Clone)]
pub struct TokenUsageTracker {
    records: Vec<UsageRecord>,
}

impl TokenUsageTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Record a usage event.
    pub fn record(&mut self, model: &str, prompt_tokens: usize, completion_tokens: usize) {
        self.records.push(UsageRecord {
            model: model.to_string(),
            prompt_tokens,
            completion_tokens,
        });
    }

    /// Total tokens (prompt + completion) across all recorded events.
    pub fn total_tokens(&self) -> usize {
        self.records
            .iter()
            .map(|r| r.prompt_tokens + r.completion_tokens)
            .sum()
    }

    /// Total tokens grouped by model name.
    pub fn by_model(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for r in &self.records {
            *map.entry(r.model.clone()).or_insert(0) += r.prompt_tokens + r.completion_tokens;
        }
        map
    }

    /// Average total tokens per request. Returns 0.0 if no requests recorded.
    pub fn avg_per_request(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.total_tokens() as f64 / self.records.len() as f64
    }

    /// Serialize to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_tokens": self.total_tokens(),
            "request_count": self.records.len(),
            "avg_per_request": self.avg_per_request(),
            "by_model": self.by_model(),
        })
    }
}

impl Default for TokenUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SimpleTokenCounter -------------------------------------------------

    #[test]
    fn simple_counter_empty() {
        let c = SimpleTokenCounter::new();
        assert_eq!(c.count(""), 0);
    }

    #[test]
    fn simple_counter_single_word() {
        let c = SimpleTokenCounter::new();
        // 1 word / 0.75 = 1.33 → ceil = 2
        assert_eq!(c.count("hello"), 2);
    }

    #[test]
    fn simple_counter_multiple_words() {
        let c = SimpleTokenCounter::new();
        // 3 words / 0.75 = 4.0
        assert_eq!(c.count("hello beautiful world"), 4);
    }

    #[test]
    fn simple_counter_name() {
        assert_eq!(SimpleTokenCounter::new().name(), "simple");
    }

    #[test]
    fn simple_counter_messages() {
        let c = SimpleTokenCounter::new();
        let msgs = vec![
            serde_json::json!({"content": "hello"}),
            serde_json::json!({"content": "world"}),
        ];
        // Each message: 3 overhead + 2 tokens = 5, total = 10 + 3 priming = 13
        assert_eq!(c.count_messages(&msgs), 13);
    }

    // -- CharBasedCounter ---------------------------------------------------

    #[test]
    fn char_counter_empty() {
        let c = CharBasedCounter::default();
        assert_eq!(c.count(""), 0);
    }

    #[test]
    fn char_counter_default_ratio() {
        let c = CharBasedCounter::default();
        // "hello" = 5 chars / 4.0 = 1.25 → ceil = 2
        assert_eq!(c.count("hello"), 2);
    }

    #[test]
    fn char_counter_custom_ratio() {
        let c = CharBasedCounter::new(2.0);
        // "hello" = 5 chars / 2.0 = 2.5 → ceil = 3
        assert_eq!(c.count("hello"), 3);
    }

    #[test]
    fn char_counter_name() {
        assert_eq!(CharBasedCounter::default().name(), "char_based");
    }

    #[test]
    fn char_counter_long_text() {
        let c = CharBasedCounter::new(4.0);
        let text = "a".repeat(100);
        assert_eq!(c.count(&text), 25); // 100 / 4.0 = 25.0
    }

    // -- ModelTokenCounter --------------------------------------------------

    #[test]
    fn model_counter_gpt4() {
        let c = ModelTokenCounter::for_model("gpt-4");
        assert!((c.chars_per_token() - 4.0).abs() < f64::EPSILON);
        // "hello" = 5 / 4.0 = 1.25 → 2
        assert_eq!(c.count("hello"), 2);
    }

    #[test]
    fn model_counter_claude() {
        let c = ModelTokenCounter::for_model("claude-3-opus");
        assert!((c.chars_per_token() - 3.5).abs() < f64::EPSILON);
        // "hello" = 5 / 3.5 = 1.43 → 2
        assert_eq!(c.count("hello"), 2);
        // "abcdefg" = 7 / 3.5 = 2.0 → 2
        assert_eq!(c.count("abcdefg"), 2);
    }

    #[test]
    fn model_counter_gemini() {
        let c = ModelTokenCounter::for_model("gemini-pro");
        assert!((c.chars_per_token() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn model_counter_unknown_default() {
        let c = ModelTokenCounter::for_model("llama-3");
        assert!((c.chars_per_token() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn model_counter_name() {
        let c = ModelTokenCounter::for_model("gpt-4");
        assert_eq!(c.name(), "gpt-4");
    }

    #[test]
    fn model_counter_empty() {
        let c = ModelTokenCounter::for_model("gpt-4");
        assert_eq!(c.count(""), 0);
    }

    // -- Priority -----------------------------------------------------------

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Low > Priority::Optional);
    }

    #[test]
    fn priority_weights() {
        assert_eq!(Priority::Critical.weight(), 4);
        assert_eq!(Priority::High.weight(), 3);
        assert_eq!(Priority::Normal.weight(), 2);
        assert_eq!(Priority::Low.weight(), 1);
        assert_eq!(Priority::Optional.weight(), 0);
    }

    // -- ContextItem --------------------------------------------------------

    #[test]
    fn context_item_to_json() {
        let item = ContextItem {
            text: "hello".to_string(),
            priority: Priority::High,
            token_count: 2,
            label: Some("greeting".to_string()),
        };
        let json = item.to_json();
        assert_eq!(json["text"], "hello");
        assert_eq!(json["priority"], "High");
        assert_eq!(json["token_count"], 2);
        assert_eq!(json["label"], "greeting");
    }

    #[test]
    fn context_item_to_json_no_label() {
        let item = ContextItem {
            text: "x".to_string(),
            priority: Priority::Low,
            token_count: 1,
            label: None,
        };
        let json = item.to_json();
        assert!(json["label"].is_null());
    }

    // -- ContextWindow ------------------------------------------------------

    #[test]
    fn context_window_add_and_counts() {
        let mut cw = ContextWindow::new(100, Box::new(CharBasedCounter::default()));
        assert!(cw.add("hello world!", Priority::Normal));
        assert!(cw.used_tokens() > 0);
        assert!(cw.remaining_tokens() < 100);
    }

    #[test]
    fn context_window_add_exceeds_budget() {
        let mut cw = ContextWindow::new(1, Box::new(CharBasedCounter::default()));
        // "hello world" should exceed 1 token budget
        assert!(!cw.add("hello world", Priority::Normal));
        assert_eq!(cw.used_tokens(), 0);
    }

    #[test]
    fn context_window_utilization() {
        let mut cw = ContextWindow::new(10, Box::new(CharBasedCounter::new(1.0)));
        // 5 chars / 1.0 = 5 tokens
        cw.add("hello", Priority::Normal);
        assert!((cw.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn context_window_utilization_empty_budget() {
        let cw = ContextWindow::new(0, Box::new(CharBasedCounter::default()));
        assert!((cw.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn context_window_clear() {
        let mut cw = ContextWindow::new(100, Box::new(CharBasedCounter::default()));
        cw.add("stuff", Priority::Normal);
        cw.clear();
        assert_eq!(cw.used_tokens(), 0);
        assert!(cw.content().is_empty());
    }

    #[test]
    fn context_window_content_returns_items() {
        let mut cw = ContextWindow::new(100, Box::new(CharBasedCounter::default()));
        cw.add("a", Priority::High);
        cw.add("b", Priority::Low);
        assert_eq!(cw.content().len(), 2);
    }

    #[test]
    fn context_window_trim_removes_lowest_priority() {
        // Use chars_per_token=1.0 so token count = char count for simplicity.
        let counter = CharBasedCounter::new(1.0);
        let mut cw = ContextWindow::new(100, Box::new(counter));
        cw.add("aaaa", Priority::Optional);  // 4 tokens
        cw.add("bbbb", Priority::High);      // 4 tokens
        cw.add("cccc", Priority::Normal);    // 4 tokens

        // Shrink budget to force trimming
        cw.max_tokens = 8;
        cw.trim_to_fit();

        // Optional item should have been removed
        assert_eq!(cw.content().len(), 2);
        assert!(cw.content().iter().all(|i| i.priority != Priority::Optional));
    }

    #[test]
    fn context_window_trim_preserves_critical() {
        let counter = CharBasedCounter::new(1.0);
        let mut cw = ContextWindow::new(100, Box::new(counter));
        cw.add("critical", Priority::Critical); // 8 tokens
        cw.add("optional", Priority::Optional); // 8 tokens

        cw.max_tokens = 10;
        cw.trim_to_fit();

        assert_eq!(cw.content().len(), 1);
        assert_eq!(cw.content()[0].priority, Priority::Critical);
    }

    #[test]
    fn context_window_add_labeled() {
        let mut cw = ContextWindow::new(100, Box::new(CharBasedCounter::default()));
        assert!(cw.add_labeled("data", Priority::Normal, "my_label"));
        let items = cw.content();
        assert_eq!(items[0].label.as_deref(), Some("my_label"));
    }

    // -- TokenBudget --------------------------------------------------------

    #[test]
    fn budget_allocate_and_remaining() {
        let mut b = TokenBudget::new(1000);
        b.allocate("system", 200);
        b.allocate("history", 500);
        assert_eq!(b.remaining("system"), 200);
        assert_eq!(b.remaining("history"), 500);
    }

    #[test]
    fn budget_use_tokens() {
        let mut b = TokenBudget::new(1000);
        b.allocate("system", 200);
        b.use_tokens("system", 50);
        assert_eq!(b.remaining("system"), 150);
        assert_eq!(b.used("system"), 50);
    }

    #[test]
    fn budget_total_remaining() {
        let mut b = TokenBudget::new(1000);
        b.allocate("a", 400);
        b.allocate("b", 400);
        b.use_tokens("a", 100);
        b.use_tokens("b", 200);
        assert_eq!(b.total_remaining(), 700);
    }

    #[test]
    fn budget_sections() {
        let mut b = TokenBudget::new(1000);
        b.allocate("system", 200);
        b.allocate("history", 500);
        let mut sections = b.sections();
        sections.sort();
        assert_eq!(sections, vec!["history", "system"]);
    }

    #[test]
    fn budget_unknown_section() {
        let b = TokenBudget::new(1000);
        assert_eq!(b.remaining("nonexistent"), 0);
        assert_eq!(b.used("nonexistent"), 0);
    }

    #[test]
    fn budget_to_json() {
        let mut b = TokenBudget::new(500);
        b.allocate("prompt", 300);
        b.use_tokens("prompt", 100);
        let json = b.to_json();
        assert_eq!(json["total"], 500);
        assert_eq!(json["total_remaining"], 400);
        assert_eq!(json["sections"]["prompt"]["allocated"], 300);
        assert_eq!(json["sections"]["prompt"]["used"], 100);
        assert_eq!(json["sections"]["prompt"]["remaining"], 200);
    }

    #[test]
    fn budget_cumulative_allocation() {
        let mut b = TokenBudget::new(1000);
        b.allocate("x", 100);
        b.allocate("x", 50);
        assert_eq!(b.remaining("x"), 150);
    }

    // -- TokenUsageTracker --------------------------------------------------

    #[test]
    fn tracker_empty() {
        let t = TokenUsageTracker::new();
        assert_eq!(t.total_tokens(), 0);
        assert_eq!(t.avg_per_request(), 0.0);
        assert!(t.by_model().is_empty());
    }

    #[test]
    fn tracker_record_and_total() {
        let mut t = TokenUsageTracker::new();
        t.record("gpt-4", 100, 50);
        t.record("gpt-4", 200, 100);
        assert_eq!(t.total_tokens(), 450);
    }

    #[test]
    fn tracker_by_model() {
        let mut t = TokenUsageTracker::new();
        t.record("gpt-4", 100, 50);
        t.record("claude", 200, 100);
        t.record("gpt-4", 50, 25);
        let by_model = t.by_model();
        assert_eq!(by_model["gpt-4"], 225);
        assert_eq!(by_model["claude"], 300);
    }

    #[test]
    fn tracker_avg_per_request() {
        let mut t = TokenUsageTracker::new();
        t.record("gpt-4", 100, 100); // 200
        t.record("gpt-4", 200, 200); // 400
        // avg = 600 / 2 = 300
        assert!((t.avg_per_request() - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_to_json() {
        let mut t = TokenUsageTracker::new();
        t.record("gpt-4", 100, 50);
        let json = t.to_json();
        assert_eq!(json["total_tokens"], 150);
        assert_eq!(json["request_count"], 1);
        assert_eq!(json["avg_per_request"], 150.0);
    }

    #[test]
    fn tracker_default() {
        let t = TokenUsageTracker::default();
        assert_eq!(t.total_tokens(), 0);
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn count_messages_empty_content() {
        let c = CharBasedCounter::default();
        let msgs = vec![serde_json::json!({"role": "user"})]; // no content field
        // 3 overhead + 0 content + 3 priming = 6
        assert_eq!(c.count_messages(&msgs), 6);
    }

    #[test]
    fn count_messages_empty_slice() {
        let c = CharBasedCounter::default();
        // Just priming = 3
        assert_eq!(c.count_messages(&[]), 3);
    }
}
