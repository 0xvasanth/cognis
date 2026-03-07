//! Context management system for agents.
//!
//! Manages the context window, injects relevant information, and handles
//! context overflow via priority-based eviction.
//!
//! The main types are:
//!
//! - [`ContextPriority`] — importance level for context items
//! - [`ContextItem`] — a single piece of context with priority and metadata
//! - [`ContextBudget`] — token budget with reserved allocation
//! - [`ContextWindow`] — manages items within a token budget
//! - [`ContextInjector`] — trait for injecting items into a window
//! - [`ContextManager`] — orchestrates context assembly from multiple injectors
//! - [`ContextStats`] — utilization statistics for a context window

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during context operations.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    /// A generic context error.
    #[error("context error: {0}")]
    General(String),
}

/// Result alias for context operations.
pub type Result<T> = std::result::Result<T, ContextError>;

// ---------------------------------------------------------------------------
// ContextPriority
// ---------------------------------------------------------------------------

/// Importance level for context items. Higher priority items are retained
/// when evicting to fit within the token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextPriority {
    /// Must always be present (weight 1.0).
    Critical,
    /// Important information (weight 0.8).
    High,
    /// Standard information (weight 0.6).
    Medium,
    /// Nice-to-have information (weight 0.4).
    Low,
    /// Lowest importance, evicted first (weight 0.2).
    Background,
}

impl ContextPriority {
    /// Numeric weight of this priority level.
    pub fn weight(&self) -> f64 {
        match self {
            Self::Critical => 1.0,
            Self::High => 0.8,
            Self::Medium => 0.6,
            Self::Low => 0.4,
            Self::Background => 0.2,
        }
    }

    /// Integer rank for ordering (higher = more important).
    fn rank(&self) -> u8 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Medium => 2,
            Self::Low => 1,
            Self::Background => 0,
        }
    }
}

impl PartialOrd for ContextPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ContextPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ---------------------------------------------------------------------------
// ContextItem
// ---------------------------------------------------------------------------

/// A single piece of context with priority, token estimate, and metadata.
#[derive(Debug, Clone)]
pub struct ContextItem {
    /// Unique identifier.
    pub id: String,
    /// The textual content.
    pub content: String,
    /// Priority level.
    pub priority: ContextPriority,
    /// Estimated token count.
    pub token_estimate: usize,
    /// Name of the source that produced this item.
    pub source: String,
    /// When this item was created.
    pub created_at: Instant,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl ContextItem {
    /// Create a new context item with an auto-generated id and token estimate.
    pub fn new(
        content: impl Into<String>,
        priority: ContextPriority,
        source: impl Into<String>,
    ) -> Self {
        let content = content.into();
        let token_estimate = estimate_tokens(&content);
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            priority,
            token_estimate,
            source: source.into(),
            created_at: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Builder method to attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// How long ago this item was created.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Serialize this item to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "content": self.content,
            "priority": format!("{:?}", self.priority),
            "token_estimate": self.token_estimate,
            "source": self.source,
            "age_ms": self.age().as_millis() as u64,
            "metadata": self.metadata,
        })
    }
}

/// Rough token estimation: ~4 characters per token.
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// ---------------------------------------------------------------------------
// ContextBudget
// ---------------------------------------------------------------------------

/// Token budget configuration for a context window.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Maximum total tokens.
    pub max_tokens: usize,
    /// Tokens reserved for system prompt, framing, etc.
    pub reserved_tokens: usize,
}

impl ContextBudget {
    /// Create a new budget with the given maximum and zero reserved tokens.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            reserved_tokens: 0,
        }
    }

    /// Builder method to set reserved tokens.
    pub fn with_reserved(mut self, tokens: usize) -> Self {
        self.reserved_tokens = tokens;
        self
    }

    /// Tokens available for context items (max minus reserved).
    pub fn available(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_tokens)
    }

    /// Whether `tokens` additional tokens can fit given `current_usage`.
    pub fn can_fit(&self, tokens: usize, current_usage: usize) -> bool {
        current_usage + tokens <= self.available()
    }
}

// ---------------------------------------------------------------------------
// ContextWindow
// ---------------------------------------------------------------------------

/// Manages a collection of [`ContextItem`]s within a [`ContextBudget`].
#[derive(Debug)]
pub struct ContextWindow {
    budget: ContextBudget,
    items: Vec<ContextItem>,
}

impl ContextWindow {
    /// Create a new empty context window with the given budget.
    pub fn new(budget: ContextBudget) -> Self {
        Self {
            budget,
            items: Vec::new(),
        }
    }

    /// Add an item if it fits within the budget. Returns `true` if added.
    pub fn add(&mut self, item: ContextItem) -> bool {
        if self
            .budget
            .can_fit(item.token_estimate, self.current_tokens())
        {
            self.items.push(item);
            true
        } else {
            false
        }
    }

    /// Add an item even if it exceeds the budget, then evict lowest-priority
    /// items until the window is within budget.
    pub fn add_forced(&mut self, item: ContextItem) {
        self.items.push(item);
        self.evict_to_budget();
    }

    /// Remove an item by id. Returns the removed item if found.
    pub fn remove(&mut self, id: &str) -> Option<ContextItem> {
        if let Some(pos) = self.items.iter().position(|i| i.id == id) {
            Some(self.items.remove(pos))
        } else {
            None
        }
    }

    /// All items in insertion order.
    pub fn items(&self) -> &[ContextItem] {
        &self.items
    }

    /// Items sorted by priority from highest to lowest.
    pub fn items_by_priority(&self) -> Vec<&ContextItem> {
        let mut sorted: Vec<&ContextItem> = self.items.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }

    /// Total tokens used by all items.
    pub fn current_tokens(&self) -> usize {
        self.items.iter().map(|i| i.token_estimate).sum()
    }

    /// Tokens remaining within the budget.
    pub fn remaining_tokens(&self) -> usize {
        self.budget
            .available()
            .saturating_sub(self.current_tokens())
    }

    /// Whether the total tokens exceed the available budget.
    pub fn is_over_budget(&self) -> bool {
        self.current_tokens() > self.budget.available()
    }

    /// Evict lowest-priority items until within budget. Returns evicted items.
    pub fn evict_to_budget(&mut self) -> Vec<ContextItem> {
        let mut evicted = Vec::new();
        while self.is_over_budget() && !self.items.is_empty() {
            // Find the index of the lowest-priority item.
            // Among equal priorities, pick the one added first (lowest index).
            let min_idx = self
                .items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.priority.cmp(&b.priority))
                .map(|(i, _)| i)
                .unwrap();
            evicted.push(self.items.remove(min_idx));
        }
        evicted
    }

    /// Remove all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the window contains no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ContextInjector trait
// ---------------------------------------------------------------------------

/// Injects context items into a [`ContextWindow`]. Returns the number of
/// items successfully added.
pub trait ContextInjector: Send + Sync {
    /// Inject items into `window`, returning the count added.
    fn inject(&self, window: &mut ContextWindow) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// SystemPromptInjector
// ---------------------------------------------------------------------------

/// Injects a system prompt as a [`ContextPriority::Critical`] item.
pub struct SystemPromptInjector {
    prompt: String,
}

impl SystemPromptInjector {
    /// Create a new injector with the given prompt text.
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
        }
    }
}

impl ContextInjector for SystemPromptInjector {
    fn inject(&self, window: &mut ContextWindow) -> Result<usize> {
        let item = ContextItem::new(
            self.prompt.clone(),
            ContextPriority::Critical,
            "system_prompt",
        );
        if window.add(item) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryInjector
// ---------------------------------------------------------------------------

/// Injects recent conversation turns as [`ContextPriority::Medium`] items.
pub struct MemoryInjector {
    turns: Vec<(String, String)>,
}

impl MemoryInjector {
    /// Create a new injector with (role, content) turn pairs.
    pub fn new(turns: Vec<(String, String)>) -> Self {
        Self { turns }
    }
}

impl ContextInjector for MemoryInjector {
    fn inject(&self, window: &mut ContextWindow) -> Result<usize> {
        let mut count = 0;
        for (role, content) in &self.turns {
            let text = format!("{}: {}", role, content);
            let item = ContextItem::new(text, ContextPriority::Medium, "memory");
            if window.add(item) {
                count += 1;
            }
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// ToolResultInjector
// ---------------------------------------------------------------------------

/// Injects tool results as [`ContextPriority::High`] items.
pub struct ToolResultInjector {
    results: Vec<(String, String)>,
}

impl ToolResultInjector {
    /// Create a new injector with (tool_name, result) pairs.
    pub fn new(results: Vec<(String, String)>) -> Self {
        Self { results }
    }
}

impl ContextInjector for ToolResultInjector {
    fn inject(&self, window: &mut ContextWindow) -> Result<usize> {
        let mut count = 0;
        for (tool_name, result) in &self.results {
            let text = format!("[{}]: {}", tool_name, result);
            let item = ContextItem::new(text, ContextPriority::High, format!("tool:{}", tool_name));
            if window.add(item) {
                count += 1;
            }
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// ContextStats
// ---------------------------------------------------------------------------

/// Utilization statistics for a context window.
#[derive(Debug, Clone)]
pub struct ContextStats {
    /// Total number of items.
    pub total_items: usize,
    /// Total tokens across all items.
    pub total_tokens: usize,
    /// Number of items grouped by priority name.
    pub items_by_priority: HashMap<String, usize>,
    /// Fraction of available budget used (0.0 to 1.0+).
    pub utilization: f64,
}

impl ContextStats {
    /// Serialize stats to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_items": self.total_items,
            "total_tokens": self.total_tokens,
            "items_by_priority": self.items_by_priority,
            "utilization": self.utilization,
        })
    }
}

// ---------------------------------------------------------------------------
// ContextManager (orchestrator)
// ---------------------------------------------------------------------------

/// Orchestrates context assembly by running injectors in order against a
/// shared [`ContextWindow`].
pub struct ContextManager {
    budget: ContextBudget,
    injectors: Vec<Box<dyn ContextInjector>>,
    window: Option<ContextWindow>,
}

impl ContextManager {
    /// Create a new context manager with the given budget.
    pub fn new(budget: ContextBudget) -> Self {
        Self {
            budget,
            injectors: Vec::new(),
            window: None,
        }
    }

    /// Register an injector. Injectors run in the order they are added.
    pub fn add_injector(&mut self, injector: Box<dyn ContextInjector>) {
        self.injectors.push(injector);
    }

    /// Build a context window by running all injectors in order.
    pub fn build_context(&mut self) -> Result<ContextWindow> {
        let mut window = ContextWindow::new(self.budget.clone());
        for injector in &self.injectors {
            injector.inject(&mut window)?;
        }
        self.window = Some(ContextWindow::new(self.budget.clone()));
        // Clone items into stored window.
        if let Some(ref mut stored) = self.window {
            for item in window.items() {
                stored.add(item.clone());
            }
        }
        Ok(window)
    }

    /// Clear the existing window and rebuild from scratch.
    pub fn rebuild(&mut self) -> Result<ContextWindow> {
        self.window = None;
        self.build_context()
    }

    /// Compute statistics for the most recently built window.
    pub fn stats(&self) -> ContextStats {
        match &self.window {
            Some(window) => {
                let total_items = window.len();
                let total_tokens = window.current_tokens();
                let available = self.budget.available();
                let utilization = if available > 0 {
                    total_tokens as f64 / available as f64
                } else if total_tokens > 0 {
                    f64::INFINITY
                } else {
                    0.0
                };

                let mut items_by_priority: HashMap<String, usize> = HashMap::new();
                for item in window.items() {
                    *items_by_priority
                        .entry(format!("{:?}", item.priority))
                        .or_insert(0) += 1;
                }

                ContextStats {
                    total_items,
                    total_tokens,
                    items_by_priority,
                    utilization,
                }
            }
            None => ContextStats {
                total_items: 0,
                total_tokens: 0,
                items_by_priority: HashMap::new(),
                utilization: 0.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContextPriority tests --

    #[test]
    fn test_priority_weights() {
        assert_eq!(ContextPriority::Critical.weight(), 1.0);
        assert_eq!(ContextPriority::High.weight(), 0.8);
        assert_eq!(ContextPriority::Medium.weight(), 0.6);
        assert_eq!(ContextPriority::Low.weight(), 0.4);
        assert_eq!(ContextPriority::Background.weight(), 0.2);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(ContextPriority::Critical > ContextPriority::High);
        assert!(ContextPriority::High > ContextPriority::Medium);
        assert!(ContextPriority::Medium > ContextPriority::Low);
        assert!(ContextPriority::Low > ContextPriority::Background);
    }

    #[test]
    fn test_priority_eq() {
        assert_eq!(ContextPriority::Critical, ContextPriority::Critical);
        assert_ne!(ContextPriority::Critical, ContextPriority::High);
    }

    #[test]
    fn test_priority_sort() {
        let mut priorities = vec![
            ContextPriority::Low,
            ContextPriority::Critical,
            ContextPriority::Background,
            ContextPriority::High,
            ContextPriority::Medium,
        ];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![
                ContextPriority::Background,
                ContextPriority::Low,
                ContextPriority::Medium,
                ContextPriority::High,
                ContextPriority::Critical,
            ]
        );
    }

    // -- ContextItem tests --

    #[test]
    fn test_item_new() {
        let item = ContextItem::new("Hello, world!", ContextPriority::High, "test");
        assert!(!item.id.is_empty());
        assert_eq!(item.content, "Hello, world!");
        assert_eq!(item.priority, ContextPriority::High);
        assert_eq!(item.source, "test");
        // 13 chars -> ceil(13/4) = 4
        assert_eq!(item.token_estimate, 4);
        assert!(item.metadata.is_empty());
    }

    #[test]
    fn test_item_empty_content() {
        let item = ContextItem::new("", ContextPriority::Low, "src");
        assert_eq!(item.token_estimate, 0);
    }

    #[test]
    fn test_item_with_metadata() {
        let item = ContextItem::new("test", ContextPriority::Medium, "src")
            .with_metadata("key", serde_json::json!("value"))
            .with_metadata("num", serde_json::json!(42));
        assert_eq!(item.metadata.len(), 2);
        assert_eq!(item.metadata["key"], serde_json::json!("value"));
        assert_eq!(item.metadata["num"], serde_json::json!(42));
    }

    #[test]
    fn test_item_age() {
        let item = ContextItem::new("test", ContextPriority::Low, "src");
        // Age should be very small (just created).
        assert!(item.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_item_to_json() {
        let item = ContextItem::new("content", ContextPriority::High, "source")
            .with_metadata("k", serde_json::json!("v"));
        let json = item.to_json();
        assert_eq!(json["content"], "content");
        assert_eq!(json["source"], "source");
        assert_eq!(json["priority"], "High");
        assert!(json["token_estimate"].as_u64().unwrap() > 0);
        assert_eq!(json["metadata"]["k"], "v");
    }

    // -- ContextBudget tests --

    #[test]
    fn test_budget_new() {
        let budget = ContextBudget::new(1000);
        assert_eq!(budget.max_tokens, 1000);
        assert_eq!(budget.reserved_tokens, 0);
        assert_eq!(budget.available(), 1000);
    }

    #[test]
    fn test_budget_with_reserved() {
        let budget = ContextBudget::new(1000).with_reserved(200);
        assert_eq!(budget.available(), 800);
    }

    #[test]
    fn test_budget_can_fit() {
        let budget = ContextBudget::new(100).with_reserved(20);
        // available = 80
        assert!(budget.can_fit(50, 0));
        assert!(budget.can_fit(30, 50));
        assert!(budget.can_fit(80, 0));
        assert!(!budget.can_fit(81, 0));
        assert!(!budget.can_fit(1, 80));
    }

    #[test]
    fn test_budget_zero() {
        let budget = ContextBudget::new(0);
        assert_eq!(budget.available(), 0);
        assert!(!budget.can_fit(1, 0));
        assert!(budget.can_fit(0, 0));
    }

    #[test]
    fn test_budget_reserved_exceeds_max() {
        let budget = ContextBudget::new(10).with_reserved(20);
        assert_eq!(budget.available(), 0);
        assert!(!budget.can_fit(1, 0));
    }

    // -- ContextWindow tests --

    #[test]
    fn test_window_new_empty() {
        let window = ContextWindow::new(ContextBudget::new(100));
        assert!(window.is_empty());
        assert_eq!(window.len(), 0);
        assert_eq!(window.current_tokens(), 0);
        assert_eq!(window.remaining_tokens(), 100);
        assert!(!window.is_over_budget());
    }

    #[test]
    fn test_window_add_success() {
        let mut window = ContextWindow::new(ContextBudget::new(100));
        let item = ContextItem::new("Hello", ContextPriority::Medium, "test");
        assert!(window.add(item));
        assert_eq!(window.len(), 1);
        assert!(window.current_tokens() > 0);
    }

    #[test]
    fn test_window_add_failure_no_space() {
        let mut window = ContextWindow::new(ContextBudget::new(1));
        // "This is a longer string" => many tokens, won't fit in budget of 1
        let item = ContextItem::new("This is a longer string", ContextPriority::High, "test");
        assert!(!window.add(item));
        assert!(window.is_empty());
    }

    #[test]
    fn test_window_add_forced() {
        let mut window = ContextWindow::new(ContextBudget::new(10));
        // Add a low-priority item that fits
        let low = ContextItem::new("low priority background", ContextPriority::Background, "a");
        window.add(low);

        // Force-add a high-priority item that pushes us over budget
        let high = ContextItem::new("high priority item here", ContextPriority::Critical, "b");
        window.add_forced(high);

        // The low-priority item should have been evicted
        assert!(!window.is_over_budget());
        assert!(window
            .items()
            .iter()
            .all(|i| i.priority == ContextPriority::Critical));
    }

    #[test]
    fn test_window_remove() {
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let item = ContextItem::new("test", ContextPriority::Low, "src");
        let id = item.id.clone();
        window.add(item);
        assert_eq!(window.len(), 1);

        let removed = window.remove(&id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, id);
        assert!(window.is_empty());
    }

    #[test]
    fn test_window_remove_not_found() {
        let mut window = ContextWindow::new(ContextBudget::new(100));
        assert!(window.remove("nonexistent").is_none());
    }

    #[test]
    fn test_window_items_by_priority() {
        let mut window = ContextWindow::new(ContextBudget::new(10000));
        window.add(ContextItem::new("low", ContextPriority::Low, "a"));
        window.add(ContextItem::new("crit", ContextPriority::Critical, "b"));
        window.add(ContextItem::new("med", ContextPriority::Medium, "c"));

        let sorted = window.items_by_priority();
        assert_eq!(sorted[0].priority, ContextPriority::Critical);
        assert_eq!(sorted[1].priority, ContextPriority::Medium);
        assert_eq!(sorted[2].priority, ContextPriority::Low);
    }

    #[test]
    fn test_window_tokens_tracking() {
        let mut window = ContextWindow::new(ContextBudget::new(100));
        let item = ContextItem::new("Hello", ContextPriority::Medium, "test");
        let tokens = item.token_estimate;
        window.add(item);
        assert_eq!(window.current_tokens(), tokens);
        assert_eq!(window.remaining_tokens(), 100 - tokens);
    }

    #[test]
    fn test_window_evict_to_budget() {
        let mut window = ContextWindow::new(ContextBudget::new(10000));
        // Add items of various priorities
        window.add(ContextItem::new(
            "background stuff",
            ContextPriority::Background,
            "a",
        ));
        window.add(ContextItem::new(
            "critical stuff",
            ContextPriority::Critical,
            "b",
        ));
        window.add(ContextItem::new(
            "low stuff here",
            ContextPriority::Low,
            "c",
        ));

        // Now shrink the budget by creating a new window with small budget and moving items
        let mut small_window = ContextWindow::new(ContextBudget::new(5));
        for item in window.items().to_vec() {
            small_window.items.push(item);
        }
        assert!(small_window.is_over_budget());

        let evicted = small_window.evict_to_budget();
        assert!(!small_window.is_over_budget());
        // Background and Low should have been evicted first
        assert!(!evicted.is_empty());
        // Critical should remain if it fits
        for remaining in small_window.items() {
            assert!(remaining.priority >= ContextPriority::Low);
        }
    }

    #[test]
    fn test_window_clear() {
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        window.add(ContextItem::new("a", ContextPriority::Low, "src"));
        window.add(ContextItem::new("b", ContextPriority::High, "src"));
        assert_eq!(window.len(), 2);

        window.clear();
        assert!(window.is_empty());
        assert_eq!(window.current_tokens(), 0);
    }

    #[test]
    fn test_window_all_critical_priority() {
        // Budget of 2 tokens; each item ~3 tokens ("aaa bbb ccc" = 12 chars each => 3 tokens)
        let mut window = ContextWindow::new(ContextBudget::new(2));
        // "aaa bbb ccc!" = 12 chars => 3 tokens each
        window.items.push(ContextItem::new(
            "aaa bbb ccc!",
            ContextPriority::Critical,
            "a",
        ));
        window.items.push(ContextItem::new(
            "ddd eee fff!",
            ContextPriority::Critical,
            "b",
        ));
        window.items.push(ContextItem::new(
            "ggg hhh iii!",
            ContextPriority::Critical,
            "c",
        ));

        assert!(window.is_over_budget());
        let evicted = window.evict_to_budget();
        assert!(!window.is_over_budget());
        // Some items should have been evicted
        assert!(!evicted.is_empty());
    }

    #[test]
    fn test_window_single_item() {
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let item = ContextItem::new("only item", ContextPriority::Medium, "src");
        assert!(window.add(item));
        assert_eq!(window.len(), 1);
        assert_eq!(window.items_by_priority().len(), 1);
    }

    // -- Injector tests --

    #[test]
    fn test_system_prompt_injector() {
        let injector = SystemPromptInjector::new("You are a helpful assistant.");
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 1);
        assert_eq!(window.len(), 1);
        assert_eq!(window.items()[0].priority, ContextPriority::Critical);
        assert_eq!(window.items()[0].content, "You are a helpful assistant.");
        assert_eq!(window.items()[0].source, "system_prompt");
    }

    #[test]
    fn test_system_prompt_injector_no_space() {
        let injector = SystemPromptInjector::new("A long system prompt that won't fit");
        let mut window = ContextWindow::new(ContextBudget::new(1));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 0);
        assert!(window.is_empty());
    }

    #[test]
    fn test_memory_injector() {
        let turns = vec![
            ("User".to_string(), "Hello".to_string()),
            ("Assistant".to_string(), "Hi there!".to_string()),
        ];
        let injector = MemoryInjector::new(turns);
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 2);
        assert_eq!(window.len(), 2);
        assert!(window.items()[0].content.contains("User: Hello"));
        assert!(window.items()[1].content.contains("Assistant: Hi there!"));
        assert!(window
            .items()
            .iter()
            .all(|i| i.priority == ContextPriority::Medium));
    }

    #[test]
    fn test_memory_injector_partial_fit() {
        let turns = vec![
            ("User".to_string(), "short".to_string()),
            (
                "Assistant".to_string(),
                "This is a much longer response that should not fit in budget".to_string(),
            ),
        ];
        let injector = MemoryInjector::new(turns);
        // Small budget: first item fits, second doesn't
        let mut window = ContextWindow::new(ContextBudget::new(5));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_tool_result_injector() {
        let results = vec![
            ("search".to_string(), "Found 5 results".to_string()),
            ("calculator".to_string(), "42".to_string()),
        ];
        let injector = ToolResultInjector::new(results);
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 2);
        assert!(window.items()[0]
            .content
            .contains("[search]: Found 5 results"));
        assert!(window.items()[1].content.contains("[calculator]: 42"));
        assert!(window
            .items()
            .iter()
            .all(|i| i.priority == ContextPriority::High));
        assert_eq!(window.items()[0].source, "tool:search");
    }

    #[test]
    fn test_tool_result_injector_empty() {
        let injector = ToolResultInjector::new(vec![]);
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 0);
        assert!(window.is_empty());
    }

    // -- ContextManager tests --

    #[test]
    fn test_manager_build_context() {
        let budget = ContextBudget::new(10000);
        let mut manager = ContextManager::new(budget);
        manager.add_injector(Box::new(SystemPromptInjector::new("System prompt")));
        manager.add_injector(Box::new(MemoryInjector::new(vec![(
            "User".to_string(),
            "Hello".to_string(),
        )])));

        let window = manager.build_context().unwrap();
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_manager_rebuild() {
        let budget = ContextBudget::new(10000);
        let mut manager = ContextManager::new(budget);
        manager.add_injector(Box::new(SystemPromptInjector::new("prompt")));

        let w1 = manager.build_context().unwrap();
        assert_eq!(w1.len(), 1);

        let w2 = manager.rebuild().unwrap();
        assert_eq!(w2.len(), 1);
    }

    #[test]
    fn test_manager_stats_empty() {
        let manager = ContextManager::new(ContextBudget::new(100));
        let stats = manager.stats();
        assert_eq!(stats.total_items, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.utilization, 0.0);
        assert!(stats.items_by_priority.is_empty());
    }

    #[test]
    fn test_manager_stats_after_build() {
        let budget = ContextBudget::new(10000);
        let mut manager = ContextManager::new(budget);
        manager.add_injector(Box::new(SystemPromptInjector::new("System")));
        manager.add_injector(Box::new(ToolResultInjector::new(vec![(
            "tool".to_string(),
            "result".to_string(),
        )])));
        manager.build_context().unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_items, 2);
        assert!(stats.total_tokens > 0);
        assert!(stats.utilization > 0.0);
        assert!(stats.utilization < 1.0);
        assert_eq!(stats.items_by_priority.get("Critical"), Some(&1));
        assert_eq!(stats.items_by_priority.get("High"), Some(&1));
    }

    #[test]
    fn test_stats_to_json() {
        let stats = ContextStats {
            total_items: 3,
            total_tokens: 100,
            items_by_priority: HashMap::from([
                ("Critical".to_string(), 1),
                ("High".to_string(), 2),
            ]),
            utilization: 0.5,
        };
        let json = stats.to_json();
        assert_eq!(json["total_items"], 3);
        assert_eq!(json["total_tokens"], 100);
        assert_eq!(json["utilization"], 0.5);
    }

    #[test]
    fn test_manager_zero_budget() {
        let budget = ContextBudget::new(0);
        let mut manager = ContextManager::new(budget);
        manager.add_injector(Box::new(SystemPromptInjector::new("prompt")));

        let window = manager.build_context().unwrap();
        // Nothing fits in zero budget
        assert!(window.is_empty());
    }

    #[test]
    fn test_manager_multiple_injectors_order() {
        let budget = ContextBudget::new(10000);
        let mut manager = ContextManager::new(budget);
        manager.add_injector(Box::new(SystemPromptInjector::new("System")));
        manager.add_injector(Box::new(MemoryInjector::new(vec![(
            "User".to_string(),
            "Q".to_string(),
        )])));
        manager.add_injector(Box::new(ToolResultInjector::new(vec![(
            "t".to_string(),
            "r".to_string(),
        )])));

        let window = manager.build_context().unwrap();
        assert_eq!(window.len(), 3);
        // Items are in injection order
        assert_eq!(window.items()[0].priority, ContextPriority::Critical);
        assert_eq!(window.items()[1].priority, ContextPriority::Medium);
        assert_eq!(window.items()[2].priority, ContextPriority::High);
    }

    #[test]
    fn test_window_remaining_tokens_with_reserved() {
        let budget = ContextBudget::new(100).with_reserved(30);
        let mut window = ContextWindow::new(budget);
        // available = 70
        assert_eq!(window.remaining_tokens(), 70);
        let item = ContextItem::new("test", ContextPriority::Low, "src");
        let t = item.token_estimate;
        window.add(item);
        assert_eq!(window.remaining_tokens(), 70 - t);
    }

    #[test]
    fn test_evict_preserves_high_priority() {
        // Budget of 4 tokens; each long item is ~5 tokens (20 chars)
        let mut window = ContextWindow::new(ContextBudget::new(4));
        // Directly push items to simulate over-budget
        window.items.push(ContextItem::new(
            "background item text!",
            ContextPriority::Background,
            "a",
        ));
        window.items.push(ContextItem::new(
            "critical item text!!",
            ContextPriority::Critical,
            "b",
        ));

        assert!(window.is_over_budget());
        let evicted = window.evict_to_budget();
        // Background should be evicted before Critical
        assert!(!evicted.is_empty());
        assert!(evicted
            .iter()
            .any(|i| i.priority == ContextPriority::Background));
        // Critical should remain if it fits
        if !window.is_empty() {
            assert_eq!(window.items()[0].priority, ContextPriority::Critical);
        }
    }

    #[test]
    fn test_memory_injector_empty() {
        let injector = MemoryInjector::new(vec![]);
        let mut window = ContextWindow::new(ContextBudget::new(1000));
        let count = injector.inject(&mut window).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_stats_zero_budget_with_items() {
        // Edge case: zero available budget but items forced in
        let budget = ContextBudget::new(0);
        let mut manager = ContextManager::new(budget);
        // No injectors will succeed with zero budget, so stats should show zero
        manager.build_context().unwrap();
        let stats = manager.stats();
        assert_eq!(stats.total_items, 0);
    }
}
