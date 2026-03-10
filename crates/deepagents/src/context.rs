//! Context management for deep agents.
//!
//! This module provides context windowing, compression, filtering, and
//! checkpointing for managing conversation context within token budgets.
//!
//! - [`ContextEntry`] — a single entry in the context window
//! - [`ContextRole`] — the role associated with a context entry
//! - [`ContextWindow`] — manages a bounded context window with token budgeting
//! - [`ContextPolicy`] — controls context management behavior
//! - [`ContextCompressor`] — compresses context by replacing old messages with summaries
//! - [`ContextSnapshot`] — captures a snapshot of context for checkpointing
//! - [`ContextFilter`] — filters context entries by criteria
//! - [`ContextStats`] — statistics about context usage

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// ContextRole
// ---------------------------------------------------------------------------

/// The role associated with a context entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "name")]
pub enum ContextRole {
    /// System-level instructions or configuration.
    System,
    /// User input.
    User,
    /// Assistant response.
    Assistant,
    /// Tool output, carrying the tool name.
    Tool(String),
    /// A summary of previous context entries.
    Summary,
}

impl ContextRole {
    /// Returns a string representation of the role.
    pub fn as_str(&self) -> &str {
        match self {
            ContextRole::System => "system",
            ContextRole::User => "user",
            ContextRole::Assistant => "assistant",
            ContextRole::Tool(_) => "tool",
            ContextRole::Summary => "summary",
        }
    }
}

impl std::fmt::Display for ContextRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextRole::System => write!(f, "system"),
            ContextRole::User => write!(f, "user"),
            ContextRole::Assistant => write!(f, "assistant"),
            ContextRole::Tool(name) => write!(f, "tool:{}", name),
            ContextRole::Summary => write!(f, "summary"),
        }
    }
}

// ---------------------------------------------------------------------------
// ContextEntry
// ---------------------------------------------------------------------------

/// A single entry in the context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The textual content of this entry.
    pub content: String,
    /// The role of the message author.
    pub role: ContextRole,
    /// Estimated token count for this entry.
    pub token_estimate: usize,
    /// ISO-8601 timestamp of when this entry was created.
    pub timestamp: String,
    /// Arbitrary metadata attached to this entry.
    pub metadata: HashMap<String, Value>,
}

impl ContextEntry {
    /// Creates a new context entry with auto-generated id, timestamp, and a
    /// rough token estimate (chars / 4).
    pub fn new(role: ContextRole, content: impl Into<String>) -> Self {
        let content = content.into();
        let token_estimate = estimate_tokens(&content);
        Self {
            id: generate_id(),
            content,
            role,
            token_estimate,
            timestamp: now_iso8601(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a metadata key-value pair, returning `self` for chaining.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serializes the entry to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Returns the number of characters in the content.
    pub fn char_count(&self) -> usize {
        self.content.len()
    }
}

// ---------------------------------------------------------------------------
// ContextWindow
// ---------------------------------------------------------------------------

/// Manages a bounded context window with a maximum token budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    entries: Vec<ContextEntry>,
    max_tokens: usize,
}

impl ContextWindow {
    /// Creates a new, empty context window with the given token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_tokens,
        }
    }

    /// Pushes a new entry into the context window.
    pub fn push(&mut self, entry: ContextEntry) {
        self.entries.push(entry);
    }

    /// Returns a slice of all entries.
    pub fn entries(&self) -> &[ContextEntry] {
        &self.entries
    }

    /// Returns the sum of all token estimates.
    pub fn total_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.token_estimate).sum()
    }

    /// Returns the number of tokens remaining before hitting the budget.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.total_tokens())
    }

    /// Returns the fraction of the budget currently used (0.0 – 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.total_tokens() as f64 / self.max_tokens as f64
    }

    /// Removes the oldest `n` non-system entries.
    pub fn truncate_oldest(&mut self, n: usize) {
        let mut removed = 0;
        self.entries.retain(|e| {
            if removed >= n {
                return true;
            }
            if e.role == ContextRole::System {
                return true;
            }
            removed += 1;
            false
        });
    }

    /// Removes oldest non-system entries until total tokens fit within the budget.
    pub fn truncate_to_fit(&mut self) {
        while self.total_tokens() > self.max_tokens {
            let idx = self
                .entries
                .iter()
                .position(|e| e.role != ContextRole::System);
            match idx {
                Some(i) => {
                    self.entries.remove(i);
                }
                None => break,
            }
        }
    }

    /// Returns references to entries matching the given role.
    pub fn find_by_role(&self, role: &ContextRole) -> Vec<&ContextEntry> {
        self.entries.iter().filter(|e| &e.role == role).collect()
    }

    /// Returns the last `n` entries (or fewer if the window has less).
    pub fn last_n(&self, n: usize) -> &[ContextEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the window contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the maximum token budget.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

// ---------------------------------------------------------------------------
// ContextPolicy
// ---------------------------------------------------------------------------

/// Controls context management behavior — budget limits and compression triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPolicy {
    /// Maximum token budget for the context window.
    pub max_tokens: usize,
    /// Tokens reserved for the model's response.
    pub reserve_for_response: usize,
    /// Whether to preserve system messages during compression/truncation.
    pub keep_system_messages: bool,
    /// Optional hard cap on the number of entries.
    pub max_entries: Option<usize>,
}

impl ContextPolicy {
    /// Creates a new policy with the given maximum token budget.
    /// Defaults: reserve 256 tokens for response, keep system messages.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            reserve_for_response: 256,
            keep_system_messages: true,
            max_entries: None,
        }
    }

    /// Sets the number of tokens to reserve for the response.
    pub fn with_response_reserve(mut self, tokens: usize) -> Self {
        self.reserve_for_response = tokens;
        self
    }

    /// Returns the effective token limit after subtracting the response reserve.
    pub fn effective_limit(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserve_for_response)
    }

    /// Returns `true` if the window exceeds the effective limit or the entry cap.
    pub fn needs_compression(&self, window: &ContextWindow) -> bool {
        if window.total_tokens() > self.effective_limit() {
            return true;
        }
        if let Some(max) = self.max_entries {
            if window.len() > max {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

/// Compresses a context window by replacing the oldest non-system entries with
/// a single summary entry.
#[derive(Debug, Clone)]
pub struct ContextCompressor {
    policy: ContextPolicy,
}

impl ContextCompressor {
    /// Creates a new compressor using the given policy.
    pub fn new(policy: ContextPolicy) -> Self {
        Self { policy }
    }

    /// Compresses the window: removes the oldest non-system entries that exceed
    /// the effective token limit and replaces them with a single [`ContextRole::Summary`]
    /// entry whose content is a concatenation of the removed entries.
    pub fn compress(&self, window: &mut ContextWindow) {
        if !self.policy.needs_compression(window) {
            return;
        }

        // Collect indices of non-system entries.
        let non_system_indices: Vec<usize> = window
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.role != ContextRole::System)
            .map(|(i, _)| i)
            .collect();

        if non_system_indices.is_empty() {
            return;
        }

        // Remove the first half (oldest) of non-system entries.
        let remove_count = non_system_indices.len().div_ceil(2);
        let to_remove: Vec<usize> = non_system_indices.into_iter().take(remove_count).collect();

        // Build summary content from removed entries.
        let mut summary_parts: Vec<String> = Vec::new();
        for &idx in &to_remove {
            let entry = &window.entries[idx];
            summary_parts.push(format!("[{}]: {}", entry.role, entry.content));
        }
        let summary_content = format!(
            "[Summary of {} messages] {}",
            to_remove.len(),
            summary_parts.join(" | ")
        );

        // Remove entries in reverse order to preserve indices.
        for &idx in to_remove.iter().rev() {
            window.entries.remove(idx);
        }

        // Insert summary entry after system messages.
        let insert_pos = window
            .entries
            .iter()
            .position(|e| e.role != ContextRole::System)
            .unwrap_or(window.entries.len());

        window.entries.insert(
            insert_pos,
            ContextEntry::new(ContextRole::Summary, summary_content),
        );
    }

    /// Returns the compression ratio (after / before). Lower is more compressed.
    pub fn compression_ratio(before: usize, after: usize) -> f64 {
        if before == 0 {
            return 1.0;
        }
        after as f64 / before as f64
    }
}

// ---------------------------------------------------------------------------
// ContextSnapshot
// ---------------------------------------------------------------------------

/// A serializable snapshot of a [`ContextWindow`] for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    entries: Vec<ContextEntry>,
    max_tokens: usize,
}

impl ContextSnapshot {
    /// Captures the current state of the window.
    pub fn capture(window: &ContextWindow) -> Self {
        Self {
            entries: window.entries.clone(),
            max_tokens: window.max_tokens,
        }
    }

    /// Restores the window from this snapshot.
    pub fn restore(&self) -> ContextWindow {
        let mut w = ContextWindow::new(self.max_tokens);
        for entry in &self.entries {
            w.push(entry.clone());
        }
        w
    }

    /// Returns the number of entries in the snapshot.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Serializes the snapshot to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Deserializes a snapshot from a JSON [`Value`].
    pub fn from_json(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

// ---------------------------------------------------------------------------
// ContextFilter
// ---------------------------------------------------------------------------

/// Builds composable filters over context entries.
#[derive(Debug, Clone, Default)]
pub struct ContextFilter {
    role: Option<ContextRole>,
    token_min: Option<usize>,
    token_max: Option<usize>,
    metadata_key: Option<String>,
}

impl ContextFilter {
    /// Creates a new filter with no constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filters by role.
    pub fn by_role(mut self, role: ContextRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Filters by token estimate range (inclusive).
    pub fn by_token_range(mut self, min: usize, max: usize) -> Self {
        self.token_min = Some(min);
        self.token_max = Some(max);
        self
    }

    /// Filters by the presence of a metadata key.
    pub fn by_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.metadata_key = Some(key.into());
        self
    }

    /// Applies the filter to a window, returning matching entries.
    pub fn apply<'a>(&self, window: &'a ContextWindow) -> Vec<&'a ContextEntry> {
        window
            .entries()
            .iter()
            .filter(|e| {
                if let Some(ref role) = self.role {
                    if &e.role != role {
                        return false;
                    }
                }
                if let Some(min) = self.token_min {
                    if e.token_estimate < min {
                        return false;
                    }
                }
                if let Some(max) = self.token_max {
                    if e.token_estimate > max {
                        return false;
                    }
                }
                if let Some(ref key) = self.metadata_key {
                    if !e.metadata.contains_key(key) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ContextStats
// ---------------------------------------------------------------------------

/// Statistics about context window usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Total estimated tokens across all entries.
    pub total_tokens: usize,
    /// Maximum token budget of the window.
    pub max_tokens: usize,
    /// Token utilization ratio.
    pub utilization: f64,
    tokens_by_role: HashMap<String, usize>,
    entry_count_by_role: HashMap<String, usize>,
}

impl ContextStats {
    /// Computes statistics from a context window.
    pub fn from_window(window: &ContextWindow) -> Self {
        let mut tokens_by_role: HashMap<String, usize> = HashMap::new();
        let mut entry_count_by_role: HashMap<String, usize> = HashMap::new();

        for entry in window.entries() {
            let key = entry.role.to_string();
            *tokens_by_role.entry(key.clone()).or_default() += entry.token_estimate;
            *entry_count_by_role.entry(key).or_default() += 1;
        }

        Self {
            total_entries: window.len(),
            total_tokens: window.total_tokens(),
            max_tokens: window.max_tokens(),
            utilization: window.utilization(),
            tokens_by_role,
            entry_count_by_role,
        }
    }

    /// Returns tokens grouped by role.
    pub fn tokens_by_role(&self) -> &HashMap<String, usize> {
        &self.tokens_by_role
    }

    /// Returns entry counts grouped by role.
    pub fn entry_count_by_role(&self) -> &HashMap<String, usize> {
        &self.entry_count_by_role
    }

    /// Returns the average token estimate per entry, or 0 if empty.
    pub fn average_entry_size(&self) -> usize {
        if self.total_entries == 0 {
            return 0;
        }
        self.total_tokens / self.total_entries
    }

    /// Serializes the stats to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough token estimation: one token per 4 characters.
fn estimate_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// Generate a simple unique id based on an atomic counter.
fn generate_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ctx-{:016x}", n)
}

/// Returns the current time as an ISO-8601-ish string.
fn now_iso8601() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple but monotonic timestamp representation.
    let days = secs / 86400;
    let rem = secs % 86400;
    let hours = rem / 3600;
    let mins = (rem % 3600) / 60;
    let s = rem % 60;
    format!("epoch+{}d-{:02}:{:02}:{:02}Z", days, hours, mins, s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContextRole tests --

    #[test]
    fn test_role_as_str_system() {
        assert_eq!(ContextRole::System.as_str(), "system");
    }

    #[test]
    fn test_role_as_str_user() {
        assert_eq!(ContextRole::User.as_str(), "user");
    }

    #[test]
    fn test_role_as_str_assistant() {
        assert_eq!(ContextRole::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_role_as_str_tool() {
        assert_eq!(ContextRole::Tool("calc".into()).as_str(), "tool");
    }

    #[test]
    fn test_role_as_str_summary() {
        assert_eq!(ContextRole::Summary.as_str(), "summary");
    }

    #[test]
    fn test_role_display_system() {
        assert_eq!(format!("{}", ContextRole::System), "system");
    }

    #[test]
    fn test_role_display_tool() {
        assert_eq!(format!("{}", ContextRole::Tool("grep".into())), "tool:grep");
    }

    #[test]
    fn test_role_display_summary() {
        assert_eq!(format!("{}", ContextRole::Summary), "summary");
    }

    #[test]
    fn test_role_equality() {
        assert_eq!(ContextRole::User, ContextRole::User);
        assert_ne!(ContextRole::User, ContextRole::System);
    }

    #[test]
    fn test_role_tool_equality() {
        assert_eq!(ContextRole::Tool("a".into()), ContextRole::Tool("a".into()));
        assert_ne!(ContextRole::Tool("a".into()), ContextRole::Tool("b".into()));
    }

    #[test]
    fn test_role_display_user() {
        assert_eq!(format!("{}", ContextRole::User), "user");
    }

    #[test]
    fn test_role_display_assistant() {
        assert_eq!(format!("{}", ContextRole::Assistant), "assistant");
    }

    // -- ContextEntry tests --

    #[test]
    fn test_entry_new() {
        let entry = ContextEntry::new(ContextRole::User, "hello world");
        assert_eq!(entry.role, ContextRole::User);
        assert_eq!(entry.content, "hello world");
        assert!(entry.token_estimate > 0);
        assert!(entry.id.starts_with("ctx-"));
    }

    #[test]
    fn test_entry_with_metadata() {
        let entry = ContextEntry::new(ContextRole::User, "hi")
            .with_metadata("source", Value::String("test".into()));
        assert_eq!(
            entry.metadata.get("source"),
            Some(&Value::String("test".into()))
        );
    }

    #[test]
    fn test_entry_char_count() {
        let entry = ContextEntry::new(ContextRole::User, "abcde");
        assert_eq!(entry.char_count(), 5);
    }

    #[test]
    fn test_entry_to_json() {
        let entry = ContextEntry::new(ContextRole::Assistant, "response");
        let json = entry.to_json();
        assert_eq!(json["content"], "response");
    }

    #[test]
    fn test_entry_token_estimate_empty() {
        let entry = ContextEntry::new(ContextRole::User, "");
        assert_eq!(entry.token_estimate, 0);
    }

    #[test]
    fn test_entry_token_estimate_short() {
        // "abcd" = 4 chars => (4+3)/4 = 1
        let entry = ContextEntry::new(ContextRole::User, "abcd");
        assert_eq!(entry.token_estimate, 1);
    }

    #[test]
    fn test_entry_unique_ids() {
        let a = ContextEntry::new(ContextRole::User, "a");
        let b = ContextEntry::new(ContextRole::User, "b");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn test_entry_metadata_chaining() {
        let entry = ContextEntry::new(ContextRole::User, "hi")
            .with_metadata("a", Value::from(1))
            .with_metadata("b", Value::from(2));
        assert_eq!(entry.metadata.len(), 2);
    }

    #[test]
    fn test_entry_to_json_has_role() {
        let entry = ContextEntry::new(ContextRole::System, "sys");
        let json = entry.to_json();
        assert!(json.get("role").is_some());
    }

    #[test]
    fn test_entry_to_json_has_id() {
        let entry = ContextEntry::new(ContextRole::User, "x");
        let json = entry.to_json();
        let id = json["id"].as_str().unwrap();
        assert!(id.starts_with("ctx-"));
    }

    #[test]
    fn test_entry_char_count_empty() {
        let entry = ContextEntry::new(ContextRole::User, "");
        assert_eq!(entry.char_count(), 0);
    }

    // -- ContextWindow tests --

    #[test]
    fn test_window_new_empty() {
        let w = ContextWindow::new(1000);
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert_eq!(w.total_tokens(), 0);
        assert_eq!(w.remaining_tokens(), 1000);
    }

    #[test]
    fn test_window_push_and_len() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "hello"));
        assert_eq!(w.len(), 1);
        assert!(!w.is_empty());
    }

    #[test]
    fn test_window_total_tokens() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh")); // 8 chars => (8+3)/4 = 2
        w.push(ContextEntry::new(ContextRole::Assistant, "abcdefgh")); // 2 tokens
        assert_eq!(w.total_tokens(), 4);
    }

    #[test]
    fn test_window_remaining_tokens() {
        let mut w = ContextWindow::new(100);
        w.push(ContextEntry::new(ContextRole::User, "abcdefghijklmnop")); // 16 chars => 4 tokens
        assert_eq!(w.remaining_tokens(), 96);
    }

    #[test]
    fn test_window_utilization() {
        let mut w = ContextWindow::new(100);
        let content = "a".repeat(200);
        w.push(ContextEntry::new(ContextRole::User, content)); // ~50 tokens
        let u = w.utilization();
        assert!(u > 0.0 && u <= 1.0);
    }

    #[test]
    fn test_window_utilization_zero_max() {
        let w = ContextWindow::new(0);
        assert_eq!(w.utilization(), 0.0);
    }

    #[test]
    fn test_window_truncate_oldest() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::System, "sys"));
        w.push(ContextEntry::new(ContextRole::User, "u1"));
        w.push(ContextEntry::new(ContextRole::Assistant, "a1"));
        w.push(ContextEntry::new(ContextRole::User, "u2"));

        w.truncate_oldest(2);
        assert_eq!(w.len(), 2);
        assert_eq!(w.entries()[0].role, ContextRole::System);
        assert_eq!(w.entries()[1].content, "u2");
    }

    #[test]
    fn test_window_truncate_oldest_preserves_system() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::System, "sys1"));
        w.push(ContextEntry::new(ContextRole::System, "sys2"));
        w.push(ContextEntry::new(ContextRole::User, "u1"));
        w.truncate_oldest(5);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_window_truncate_to_fit() {
        let mut w = ContextWindow::new(10);
        w.push(ContextEntry::new(ContextRole::System, "sys"));
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(40)));
        w.push(ContextEntry::new(ContextRole::Assistant, "b".repeat(40)));

        w.truncate_to_fit();
        assert!(w.total_tokens() <= 10);
    }

    #[test]
    fn test_window_find_by_role() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "u1"));
        w.push(ContextEntry::new(ContextRole::Assistant, "a1"));
        w.push(ContextEntry::new(ContextRole::User, "u2"));
        let users = w.find_by_role(&ContextRole::User);
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn test_window_find_by_role_empty() {
        let w = ContextWindow::new(1000);
        assert!(w.find_by_role(&ContextRole::User).is_empty());
    }

    #[test]
    fn test_window_last_n() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.push(ContextEntry::new(ContextRole::User, "b"));
        w.push(ContextEntry::new(ContextRole::User, "c"));
        let last = w.last_n(2);
        assert_eq!(last.len(), 2);
        assert_eq!(last[0].content, "b");
        assert_eq!(last[1].content, "c");
    }

    #[test]
    fn test_window_last_n_exceeds_len() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        let last = w.last_n(10);
        assert_eq!(last.len(), 1);
    }

    #[test]
    fn test_window_clear() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.total_tokens(), 0);
    }

    #[test]
    fn test_window_max_tokens() {
        let w = ContextWindow::new(4096);
        assert_eq!(w.max_tokens(), 4096);
    }

    // -- ContextPolicy tests --

    #[test]
    fn test_policy_new() {
        let p = ContextPolicy::new(4096);
        assert_eq!(p.max_tokens, 4096);
        assert_eq!(p.reserve_for_response, 256);
        assert!(p.keep_system_messages);
        assert!(p.max_entries.is_none());
    }

    #[test]
    fn test_policy_with_response_reserve() {
        let p = ContextPolicy::new(4096).with_response_reserve(512);
        assert_eq!(p.reserve_for_response, 512);
    }

    #[test]
    fn test_policy_effective_limit() {
        let p = ContextPolicy::new(4096).with_response_reserve(512);
        assert_eq!(p.effective_limit(), 3584);
    }

    #[test]
    fn test_policy_effective_limit_underflow() {
        let p = ContextPolicy::new(100).with_response_reserve(200);
        assert_eq!(p.effective_limit(), 0);
    }

    #[test]
    fn test_policy_needs_compression_tokens() {
        let p = ContextPolicy::new(10).with_response_reserve(0);
        let mut w = ContextWindow::new(10);
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(100)));
        assert!(p.needs_compression(&w));
    }

    #[test]
    fn test_policy_needs_compression_entries() {
        let mut p = ContextPolicy::new(100000);
        p.max_entries = Some(2);
        let mut w = ContextWindow::new(100000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.push(ContextEntry::new(ContextRole::User, "b"));
        w.push(ContextEntry::new(ContextRole::User, "c"));
        assert!(p.needs_compression(&w));
    }

    #[test]
    fn test_policy_no_compression_needed() {
        let p = ContextPolicy::new(100000);
        let mut w = ContextWindow::new(100000);
        w.push(ContextEntry::new(ContextRole::User, "hello"));
        assert!(!p.needs_compression(&w));
    }

    #[test]
    fn test_policy_default_reserve() {
        let p = ContextPolicy::new(1000);
        assert_eq!(p.effective_limit(), 744);
    }

    // -- ContextCompressor tests --

    #[test]
    fn test_compressor_no_compression_needed() {
        let policy = ContextPolicy::new(100000).with_response_reserve(0);
        let compressor = ContextCompressor::new(policy);
        let mut w = ContextWindow::new(100000);
        w.push(ContextEntry::new(ContextRole::User, "hi"));
        compressor.compress(&mut w);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn test_compressor_compresses_old_entries() {
        let policy = ContextPolicy::new(5).with_response_reserve(0);
        let compressor = ContextCompressor::new(policy);
        let mut w = ContextWindow::new(5);
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(20)));
        w.push(ContextEntry::new(ContextRole::Assistant, "b".repeat(20)));
        w.push(ContextEntry::new(ContextRole::User, "c".repeat(20)));
        w.push(ContextEntry::new(ContextRole::Assistant, "d".repeat(20)));

        let before = w.len();
        compressor.compress(&mut w);
        assert!(w.len() < before);
        assert!(w.entries().iter().any(|e| e.role == ContextRole::Summary));
    }

    #[test]
    fn test_compressor_preserves_system() {
        let policy = ContextPolicy::new(5).with_response_reserve(0);
        let compressor = ContextCompressor::new(policy);
        let mut w = ContextWindow::new(5);
        w.push(ContextEntry::new(ContextRole::System, "instructions"));
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(20)));
        w.push(ContextEntry::new(ContextRole::Assistant, "b".repeat(20)));

        compressor.compress(&mut w);
        assert!(w.entries().iter().any(|e| e.role == ContextRole::System));
    }

    #[test]
    fn test_compressor_compression_ratio() {
        assert_eq!(ContextCompressor::compression_ratio(100, 50), 0.5);
        assert_eq!(ContextCompressor::compression_ratio(100, 100), 1.0);
        assert_eq!(ContextCompressor::compression_ratio(0, 50), 1.0);
    }

    #[test]
    fn test_compressor_summary_content_contains_messages() {
        let policy = ContextPolicy::new(2).with_response_reserve(0);
        let compressor = ContextCompressor::new(policy);
        let mut w = ContextWindow::new(2);
        w.push(ContextEntry::new(ContextRole::User, "important info"));
        w.push(ContextEntry::new(ContextRole::Assistant, "response here"));

        compressor.compress(&mut w);
        let summary = w
            .entries()
            .iter()
            .find(|e| e.role == ContextRole::Summary)
            .unwrap();
        assert!(summary.content.contains("Summary"));
    }

    #[test]
    fn test_compressor_only_system_entries() {
        let policy = ContextPolicy::new(1).with_response_reserve(0);
        let compressor = ContextCompressor::new(policy);
        let mut w = ContextWindow::new(1);
        w.push(ContextEntry::new(ContextRole::System, "a".repeat(100)));
        compressor.compress(&mut w);
        assert_eq!(w.len(), 1);
    }

    // -- ContextSnapshot tests --

    #[test]
    fn test_snapshot_capture_restore() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "hello"));
        w.push(ContextEntry::new(ContextRole::Assistant, "world"));

        let snap = ContextSnapshot::capture(&w);
        assert_eq!(snap.entry_count(), 2);

        let restored = snap.restore();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.max_tokens(), 1000);
        assert_eq!(restored.entries()[0].content, "hello");
    }

    #[test]
    fn test_snapshot_to_json_from_json() {
        let mut w = ContextWindow::new(500);
        w.push(ContextEntry::new(ContextRole::User, "test"));

        let snap = ContextSnapshot::capture(&w);
        let json = snap.to_json();
        let restored_snap = ContextSnapshot::from_json(&json).unwrap();
        assert_eq!(restored_snap.entry_count(), 1);
    }

    #[test]
    fn test_snapshot_from_json_invalid() {
        let result = ContextSnapshot::from_json(&Value::String("bad".into()));
        assert!(result.is_none());
    }

    #[test]
    fn test_snapshot_empty_window() {
        let w = ContextWindow::new(100);
        let snap = ContextSnapshot::capture(&w);
        assert_eq!(snap.entry_count(), 0);
        let restored = snap.restore();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_snapshot_roundtrip_preserves_metadata() {
        let mut w = ContextWindow::new(500);
        w.push(
            ContextEntry::new(ContextRole::User, "test").with_metadata("key", Value::from("val")),
        );

        let snap = ContextSnapshot::capture(&w);
        let json = snap.to_json();
        let snap2 = ContextSnapshot::from_json(&json).unwrap();
        let restored = snap2.restore();
        assert_eq!(
            restored.entries()[0].metadata.get("key"),
            Some(&Value::from("val"))
        );
    }

    #[test]
    fn test_snapshot_preserves_max_tokens() {
        let w = ContextWindow::new(8192);
        let snap = ContextSnapshot::capture(&w);
        let json = snap.to_json();
        let snap2 = ContextSnapshot::from_json(&json).unwrap();
        let restored = snap2.restore();
        assert_eq!(restored.max_tokens(), 8192);
    }

    // -- ContextFilter tests --

    #[test]
    fn test_filter_no_constraints() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.push(ContextEntry::new(ContextRole::Assistant, "b"));
        let results = ContextFilter::new().apply(&w);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_by_role() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.push(ContextEntry::new(ContextRole::Assistant, "b"));
        w.push(ContextEntry::new(ContextRole::User, "c"));

        let results = ContextFilter::new().by_role(ContextRole::User).apply(&w);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_by_token_range() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a")); // ~1 token
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(100))); // ~25 tokens

        let results = ContextFilter::new().by_token_range(10, 50).apply(&w);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_by_metadata_key() {
        let mut w = ContextWindow::new(1000);
        w.push(
            ContextEntry::new(ContextRole::User, "a").with_metadata("source", Value::from("web")),
        );
        w.push(ContextEntry::new(ContextRole::User, "b"));

        let results = ContextFilter::new().by_metadata_key("source").apply(&w);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_combined() {
        let mut w = ContextWindow::new(1000);
        w.push(
            ContextEntry::new(ContextRole::User, "a".repeat(40))
                .with_metadata("tag", Value::from("x")),
        );
        w.push(ContextEntry::new(ContextRole::User, "b"));
        w.push(
            ContextEntry::new(ContextRole::Assistant, "c".repeat(40))
                .with_metadata("tag", Value::from("y")),
        );

        let results = ContextFilter::new()
            .by_role(ContextRole::User)
            .by_metadata_key("tag")
            .apply(&w);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "a".repeat(40));
    }

    #[test]
    fn test_filter_empty_window() {
        let w = ContextWindow::new(1000);
        let results = ContextFilter::new().by_role(ContextRole::User).apply(&w);
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_default() {
        let f = ContextFilter::default();
        let w = ContextWindow::new(1000);
        assert!(f.apply(&w).is_empty());
    }

    #[test]
    fn test_filter_by_role_tool() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::Tool("a".into()), "x"));
        w.push(ContextEntry::new(ContextRole::Tool("b".into()), "y"));
        let results = ContextFilter::new()
            .by_role(ContextRole::Tool("a".into()))
            .apply(&w);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "x");
    }

    // -- ContextStats tests --

    #[test]
    fn test_stats_empty_window() {
        let w = ContextWindow::new(1000);
        let stats = ContextStats::from_window(&w);
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.average_entry_size(), 0);
    }

    #[test]
    fn test_stats_tokens_by_role() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh")); // 2 tokens
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh")); // 2 tokens
        w.push(ContextEntry::new(ContextRole::Assistant, "abcdefgh")); // 2 tokens

        let stats = ContextStats::from_window(&w);
        assert_eq!(*stats.tokens_by_role().get("user").unwrap(), 4);
        assert_eq!(*stats.tokens_by_role().get("assistant").unwrap(), 2);
    }

    #[test]
    fn test_stats_entry_count_by_role() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.push(ContextEntry::new(ContextRole::User, "b"));
        w.push(ContextEntry::new(ContextRole::System, "c"));

        let stats = ContextStats::from_window(&w);
        assert_eq!(*stats.entry_count_by_role().get("user").unwrap(), 2);
        assert_eq!(*stats.entry_count_by_role().get("system").unwrap(), 1);
    }

    #[test]
    fn test_stats_average_entry_size() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh"));
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh"));
        w.push(ContextEntry::new(ContextRole::User, "abcdefgh"));

        let stats = ContextStats::from_window(&w);
        assert_eq!(stats.average_entry_size(), 2);
    }

    #[test]
    fn test_stats_utilization() {
        let mut w = ContextWindow::new(100);
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(200)));
        let stats = ContextStats::from_window(&w);
        assert!(stats.utilization > 0.0);
    }

    #[test]
    fn test_stats_to_json() {
        let w = ContextWindow::new(1000);
        let stats = ContextStats::from_window(&w);
        let json = stats.to_json();
        assert_eq!(json["total_entries"], 0);
    }

    #[test]
    fn test_stats_tool_role() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(
            ContextRole::Tool("calc".into()),
            "result",
        ));
        let stats = ContextStats::from_window(&w);
        assert!(stats.tokens_by_role().contains_key("tool:calc"));
    }

    #[test]
    fn test_stats_max_tokens() {
        let w = ContextWindow::new(2048);
        let stats = ContextStats::from_window(&w);
        assert_eq!(stats.max_tokens, 2048);
    }

    // -- Serialization tests --

    #[test]
    fn test_role_serialize_deserialize() {
        let role = ContextRole::Tool("search".into());
        let json = serde_json::to_string(&role).unwrap();
        let restored: ContextRole = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, role);
    }

    #[test]
    fn test_entry_serialize_deserialize() {
        let entry =
            ContextEntry::new(ContextRole::User, "hello").with_metadata("k", Value::from(42));
        let json = serde_json::to_string(&entry).unwrap();
        let restored: ContextEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.content, "hello");
        assert_eq!(restored.metadata.get("k"), Some(&Value::from(42)));
    }

    #[test]
    fn test_window_remaining_tokens_saturates() {
        let mut w = ContextWindow::new(5);
        w.push(ContextEntry::new(ContextRole::User, "a".repeat(100)));
        assert_eq!(w.remaining_tokens(), 0);
    }

    #[test]
    fn test_truncate_oldest_zero() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        w.truncate_oldest(0);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn test_window_entries_returns_slice() {
        let mut w = ContextWindow::new(1000);
        w.push(ContextEntry::new(ContextRole::User, "a"));
        let entries = w.entries();
        assert_eq!(entries.len(), 1);
    }

    // -- Helper tests --

    #[test]
    fn test_estimate_tokens_boundary() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("ab"), 1);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn test_generate_id_unique() {
        let ids: Vec<String> = (0..100).map(|_| generate_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 100);
    }
}
