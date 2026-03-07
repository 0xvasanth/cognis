//! Conversation memory with windowing, search, and branching capabilities.
//!
//! This module provides a rich conversation memory that stores the full
//! conversation history as turns (human + AI pairs) and supports:
//!
//! - **Windowing** — retrieve a subset of turns by count, token budget, or time
//! - **Branching** — create named branches to explore alternative conversation paths
//! - **Search** — find turns by keyword, metadata, or time range

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rustchain_core::error::{Result, RustChainError};
use serde_json::{json, Value};
use uuid::Uuid;

/// A single conversation turn consisting of a human message and an AI response.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// The human's message.
    pub human: String,
    /// The AI's response.
    pub ai: String,
    /// When this turn was created.
    pub timestamp: Instant,
    /// Arbitrary metadata attached to the turn.
    pub metadata: HashMap<String, Value>,
    /// Unique identifier for this turn.
    pub turn_id: String,
}

impl ConversationTurn {
    /// Create a new conversation turn with an auto-generated turn ID.
    pub fn new(human: impl Into<String>, ai: impl Into<String>) -> Self {
        Self {
            human: human.into(),
            ai: ai.into(),
            timestamp: Instant::now(),
            metadata: HashMap::new(),
            turn_id: Uuid::new_v4().to_string(),
        }
    }

    /// Add metadata to the turn (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize the turn to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "turn_id": self.turn_id,
            "human": self.human,
            "ai": self.ai,
            "metadata": self.metadata,
        })
    }

    /// Check whether the query appears (case-insensitive) in either the human
    /// or AI text.
    pub fn contains(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.human.to_lowercase().contains(&q) || self.ai.to_lowercase().contains(&q)
    }
}

/// Strategy for selecting a window of conversation turns.
#[derive(Debug, Clone)]
pub enum ConversationWindow {
    /// Return all turns.
    All,
    /// Return the last `n` turns.
    Last(usize),
    /// Return as many recent turns as fit within a character budget (rough
    /// token proxy).
    TokenBudget(usize),
    /// Return turns created within the given duration from now.
    TimeBased(Duration),
}

impl ConversationWindow {
    /// Apply the windowing strategy to a slice of turns, returning references
    /// to the selected subset.
    pub fn apply<'a>(&self, turns: &'a [ConversationTurn]) -> Vec<&'a ConversationTurn> {
        match self {
            ConversationWindow::All => turns.iter().collect(),
            ConversationWindow::Last(n) => {
                let start = turns.len().saturating_sub(*n);
                turns[start..].iter().collect()
            }
            ConversationWindow::TokenBudget(budget) => {
                let mut remaining = *budget;
                let mut selected: Vec<&ConversationTurn> = Vec::new();
                for turn in turns.iter().rev() {
                    let cost = turn.human.len() + turn.ai.len();
                    if cost > remaining {
                        break;
                    }
                    remaining -= cost;
                    selected.push(turn);
                }
                selected.reverse();
                selected
            }
            ConversationWindow::TimeBased(duration) => {
                let now = Instant::now();
                turns
                    .iter()
                    .filter(|t| now.duration_since(t.timestamp) <= *duration)
                    .collect()
            }
        }
    }
}

/// A named branch of conversation that diverges from the main history at a
/// specific turn index.
#[derive(Debug, Clone)]
pub struct ConversationBranch {
    name: String,
    branch_point: usize,
    turns: Vec<ConversationTurn>,
}

impl ConversationBranch {
    /// Create a new branch starting from the given turn index.
    pub fn new(name: impl Into<String>, branch_point: usize) -> Self {
        Self {
            name: name.into(),
            branch_point,
            turns: Vec::new(),
        }
    }

    /// The branch name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The turn index in the main history where this branch diverges.
    pub fn branch_point(&self) -> usize {
        self.branch_point
    }

    /// The turns added on this branch (after the branch point).
    pub fn turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    /// Number of turns on this branch.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the branch has no turns of its own.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Append a turn to this branch.
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
    }
}

/// Main conversation memory store with windowing, branching, and search.
#[derive(Debug)]
pub struct ConversationMemory {
    turns: Vec<ConversationTurn>,
    branches: HashMap<String, ConversationBranch>,
    current_branch: Option<String>,
}

impl ConversationMemory {
    /// Create an empty conversation memory.
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            branches: HashMap::new(),
            current_branch: None,
        }
    }

    /// Add a conversation turn.
    pub fn add_turn(&mut self, human: &str, ai: &str) {
        let turn = ConversationTurn::new(human, ai);
        if let Some(branch_name) = &self.current_branch {
            if let Some(branch) = self.branches.get_mut(branch_name) {
                branch.add_turn(turn);
                return;
            }
        }
        self.turns.push(turn);
    }

    /// Add a conversation turn with metadata.
    pub fn add_turn_with_metadata(
        &mut self,
        human: &str,
        ai: &str,
        metadata: HashMap<String, Value>,
    ) {
        let mut turn = ConversationTurn::new(human, ai);
        turn.metadata = metadata;
        if let Some(branch_name) = &self.current_branch {
            if let Some(branch) = self.branches.get_mut(branch_name) {
                branch.add_turn(turn);
                return;
            }
        }
        self.turns.push(turn);
    }

    /// Retrieve turns according to a windowing strategy.
    pub fn get_turns(&self, window: &ConversationWindow) -> Vec<&ConversationTurn> {
        let refs = self.effective_turn_refs();
        match window {
            ConversationWindow::All => refs,
            ConversationWindow::Last(n) => {
                let start = refs.len().saturating_sub(*n);
                refs[start..].to_vec()
            }
            ConversationWindow::TokenBudget(budget) => {
                let mut remaining = *budget;
                let mut selected: Vec<&ConversationTurn> = Vec::new();
                for turn in refs.iter().rev() {
                    let cost = turn.human.len() + turn.ai.len();
                    if cost > remaining {
                        break;
                    }
                    remaining -= cost;
                    selected.push(turn);
                }
                selected.reverse();
                selected
            }
            ConversationWindow::TimeBased(duration) => {
                let now = Instant::now();
                refs.into_iter()
                    .filter(|t| now.duration_since(t.timestamp) <= *duration)
                    .collect()
            }
        }
    }

    /// Search for turns containing the query (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&ConversationTurn> {
        self.effective_turn_refs()
            .into_iter()
            .filter(|t| t.contains(query))
            .collect()
    }

    /// Get a specific turn by index.
    pub fn get_turn(&self, index: usize) -> Option<&ConversationTurn> {
        self.effective_turn_refs().into_iter().nth(index)
    }

    /// Get the last turn.
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.effective_turn_refs().into_iter().last()
    }

    /// Clear the main conversation history (does not affect branches).
    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Number of turns in the current view (main or branch).
    pub fn len(&self) -> usize {
        self.effective_turn_refs().len()
    }

    /// Whether the current view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the windowed turns into a JSON array of message objects.
    pub fn to_messages(&self, window: &ConversationWindow) -> Vec<Value> {
        let turns = self.get_turns(window);
        let mut messages = Vec::new();
        for turn in turns {
            messages.push(json!({ "role": "human", "content": turn.human }));
            messages.push(json!({ "role": "ai", "content": turn.ai }));
        }
        messages
    }

    /// Create a named branch diverging from a specific turn index.
    pub fn create_branch(&mut self, name: &str, from_turn: usize) -> Result<()> {
        if self.branches.contains_key(name) {
            return Err(RustChainError::Other(format!(
                "Branch '{}' already exists",
                name
            )));
        }
        if from_turn > self.turns.len() {
            return Err(RustChainError::Other(format!(
                "Branch point {} exceeds turn count {}",
                from_turn,
                self.turns.len()
            )));
        }
        self.branches
            .insert(name.to_string(), ConversationBranch::new(name, from_turn));
        Ok(())
    }

    /// Get a reference to a branch by name.
    pub fn get_branch(&self, name: &str) -> Option<&ConversationBranch> {
        self.branches.get(name)
    }

    /// Switch the active view to a named branch.
    pub fn switch_branch(&mut self, name: &str) -> Result<()> {
        if !self.branches.contains_key(name) {
            return Err(RustChainError::Other(format!(
                "Branch '{}' does not exist",
                name
            )));
        }
        self.current_branch = Some(name.to_string());
        Ok(())
    }

    /// The currently active branch name, or `None` for the main line.
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    /// List all branch names.
    pub fn branches(&self) -> Vec<&str> {
        self.branches.keys().map(|k| k.as_str()).collect()
    }

    /// Export the entire memory (main + branches) as JSON.
    pub fn export(&self) -> Value {
        let main_turns: Vec<Value> = self.turns.iter().map(|t| t.to_json()).collect();
        let branches: HashMap<&str, Value> = self
            .branches
            .iter()
            .map(|(name, branch)| {
                let branch_val = json!({
                    "branch_point": branch.branch_point(),
                    "turns": branch.turns().iter().map(|t| t.to_json()).collect::<Vec<_>>(),
                });
                (name.as_str(), branch_val)
            })
            .collect();
        json!({
            "turns": main_turns,
            "branches": branches,
            "current_branch": self.current_branch,
        })
    }

    /// Produce a summary of the conversation memory state.
    pub fn summary(&self) -> Value {
        json!({
            "total_turns": self.turns.len(),
            "branch_count": self.branches.len(),
            "current_branch": self.current_branch,
            "branches": self.branches.keys().collect::<Vec<_>>(),
        })
    }

    // -- internal helpers --

    /// Returns references to the effective turns for the current view.
    fn effective_turn_refs(&self) -> Vec<&ConversationTurn> {
        if let Some(branch_name) = &self.current_branch {
            if let Some(branch) = self.branches.get(branch_name) {
                let bp = branch.branch_point();
                let mut refs: Vec<&ConversationTurn> = self.turns.iter().take(bp).collect();
                refs.extend(branch.turns().iter());
                return refs;
            }
        }
        self.turns.iter().collect()
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced search utility over a [`ConversationMemory`].
pub struct ConversationSearch<'a> {
    memory: &'a ConversationMemory,
}

impl<'a> ConversationSearch<'a> {
    /// Create a new search bound to the given memory.
    pub fn new(memory: &'a ConversationMemory) -> Self {
        Self { memory }
    }

    /// Find turns containing the keyword (case-insensitive).
    pub fn by_keyword(&self, keyword: &str) -> Vec<&'a ConversationTurn> {
        self.memory.search(keyword)
    }

    /// Find turns whose timestamp falls within the given time range from now.
    ///
    /// `start` and `end` are durations measured backward from now.
    /// A turn matches if `start <= age <= end`.
    pub fn by_time_range(&self, start: Duration, end: Duration) -> Vec<&'a ConversationTurn> {
        let now = Instant::now();
        self.memory
            .effective_turn_refs()
            .into_iter()
            .filter(|t| {
                let age = now.duration_since(t.timestamp);
                age >= start && age <= end
            })
            .collect()
    }

    /// Find turns that have a specific metadata key-value pair.
    pub fn by_metadata(&self, key: &str, value: &Value) -> Vec<&'a ConversationTurn> {
        self.memory
            .effective_turn_refs()
            .into_iter()
            .filter(|t| t.metadata.get(key) == Some(value))
            .collect()
    }

    /// Return the most recent `n` turns.
    pub fn recent(&self, n: usize) -> Vec<&'a ConversationTurn> {
        let turns = self.memory.effective_turn_refs();
        let start = turns.len().saturating_sub(n);
        turns[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ConversationTurn ──────────────────────────────────────────────

    #[test]
    fn test_turn_new() {
        let turn = ConversationTurn::new("hello", "hi there");
        assert_eq!(turn.human, "hello");
        assert_eq!(turn.ai, "hi there");
        assert!(!turn.turn_id.is_empty());
        assert!(turn.metadata.is_empty());
    }

    #[test]
    fn test_turn_unique_ids() {
        let t1 = ConversationTurn::new("a", "b");
        let t2 = ConversationTurn::new("c", "d");
        assert_ne!(t1.turn_id, t2.turn_id);
    }

    #[test]
    fn test_turn_with_metadata() {
        let turn = ConversationTurn::new("q", "a")
            .with_metadata("topic", json!("science"))
            .with_metadata("score", json!(0.95));
        assert_eq!(turn.metadata.get("topic").unwrap(), &json!("science"));
        assert_eq!(turn.metadata.get("score").unwrap(), &json!(0.95));
    }

    #[test]
    fn test_turn_to_json() {
        let turn = ConversationTurn::new("hi", "hello").with_metadata("key", json!("val"));
        let j = turn.to_json();
        assert_eq!(j["human"], "hi");
        assert_eq!(j["ai"], "hello");
        assert_eq!(j["turn_id"], turn.turn_id);
        assert_eq!(j["metadata"]["key"], "val");
    }

    #[test]
    fn test_turn_contains_case_insensitive() {
        let turn = ConversationTurn::new("What is Rust?", "Rust is a systems language.");
        assert!(turn.contains("rust"));
        assert!(turn.contains("RUST"));
        assert!(turn.contains("systems"));
        assert!(!turn.contains("python"));
    }

    #[test]
    fn test_turn_contains_in_human() {
        let turn = ConversationTurn::new("Tell me about Paris", "Sure!");
        assert!(turn.contains("paris"));
    }

    #[test]
    fn test_turn_contains_in_ai() {
        let turn = ConversationTurn::new("Hello", "Welcome to London!");
        assert!(turn.contains("london"));
    }

    // ── ConversationWindow ────────────────────────────────────────────

    fn make_turns(n: usize) -> Vec<ConversationTurn> {
        (0..n)
            .map(|i| ConversationTurn::new(format!("human {}", i), format!("ai {}", i)))
            .collect()
    }

    #[test]
    fn test_window_all() {
        let turns = make_turns(5);
        let result = ConversationWindow::All.apply(&turns);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_window_last() {
        let turns = make_turns(10);
        let result = ConversationWindow::Last(3).apply(&turns);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].human, "human 7");
        assert_eq!(result[2].human, "human 9");
    }

    #[test]
    fn test_window_last_exceeds_len() {
        let turns = make_turns(2);
        let result = ConversationWindow::Last(10).apply(&turns);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_window_token_budget() {
        // Each turn: "human X" (7 chars) + "ai X" (4 chars) = ~11 chars
        let turns = make_turns(5);
        let result = ConversationWindow::TokenBudget(25).apply(&turns);
        // Should fit about 2 turns (22 chars) but not 3 (33)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_window_token_budget_zero() {
        let turns = make_turns(3);
        let result = ConversationWindow::TokenBudget(0).apply(&turns);
        assert!(result.is_empty());
    }

    #[test]
    fn test_window_time_based() {
        let turns = make_turns(3);
        // All turns just created, so all within 10 seconds
        let result = ConversationWindow::TimeBased(Duration::from_secs(10)).apply(&turns);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_window_empty_turns() {
        let turns: Vec<ConversationTurn> = Vec::new();
        assert!(ConversationWindow::All.apply(&turns).is_empty());
        assert!(ConversationWindow::Last(5).apply(&turns).is_empty());
        assert!(ConversationWindow::TokenBudget(100)
            .apply(&turns)
            .is_empty());
        assert!(ConversationWindow::TimeBased(Duration::from_secs(10))
            .apply(&turns)
            .is_empty());
    }

    // ── ConversationBranch ────────────────────────────────────────────

    #[test]
    fn test_branch_new() {
        let branch = ConversationBranch::new("experiment", 3);
        assert_eq!(branch.name(), "experiment");
        assert_eq!(branch.branch_point(), 3);
        assert!(branch.is_empty());
        assert_eq!(branch.len(), 0);
    }

    #[test]
    fn test_branch_add_turns() {
        let mut branch = ConversationBranch::new("alt", 0);
        branch.add_turn(ConversationTurn::new("h1", "a1"));
        branch.add_turn(ConversationTurn::new("h2", "a2"));
        assert_eq!(branch.len(), 2);
        assert!(!branch.is_empty());
        assert_eq!(branch.turns()[0].human, "h1");
        assert_eq!(branch.turns()[1].human, "h2");
    }

    // ── ConversationMemory ────────────────────────────────────────────

    #[test]
    fn test_memory_new_empty() {
        let mem = ConversationMemory::new();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert!(mem.last_turn().is_none());
        assert!(mem.current_branch().is_none());
    }

    #[test]
    fn test_memory_add_and_retrieve() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("hello", "hi");
        mem.add_turn("how are you", "fine");
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.get_turn(0).unwrap().human, "hello");
        assert_eq!(mem.get_turn(1).unwrap().ai, "fine");
    }

    #[test]
    fn test_memory_last_turn() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("first", "1st");
        mem.add_turn("second", "2nd");
        assert_eq!(mem.last_turn().unwrap().human, "second");
    }

    #[test]
    fn test_memory_clear() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        mem.clear();
        assert!(mem.is_empty());
    }

    #[test]
    fn test_memory_search() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("Tell me about Rust", "Rust is great");
        mem.add_turn("Tell me about Python", "Python is popular");
        mem.add_turn("What about Go?", "Go is fast");

        let results = mem.search("rust");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].human, "Tell me about Rust");

        let results = mem.search("tell");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_memory_search_no_results() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        assert!(mem.search("nonexistent").is_empty());
    }

    #[test]
    fn test_memory_get_turn_out_of_bounds() {
        let mem = ConversationMemory::new();
        assert!(mem.get_turn(0).is_none());
        assert!(mem.get_turn(100).is_none());
    }

    #[test]
    fn test_memory_add_turn_with_metadata() {
        let mut mem = ConversationMemory::new();
        let mut meta = HashMap::new();
        meta.insert("topic".to_string(), json!("science"));
        mem.add_turn_with_metadata("question", "answer", meta);
        let turn = mem.get_turn(0).unwrap();
        assert_eq!(turn.metadata.get("topic").unwrap(), &json!("science"));
    }

    #[test]
    fn test_memory_to_messages() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("hi", "hello");
        mem.add_turn("bye", "goodbye");
        let msgs = mem.to_messages(&ConversationWindow::All);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "human");
        assert_eq!(msgs[0]["content"], "hi");
        assert_eq!(msgs[1]["role"], "ai");
        assert_eq!(msgs[1]["content"], "hello");
    }

    #[test]
    fn test_memory_to_messages_windowed() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        mem.add_turn("c", "d");
        mem.add_turn("e", "f");
        let msgs = mem.to_messages(&ConversationWindow::Last(1));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "e");
    }

    // ── Branching ─────────────────────────────────────────────────────

    #[test]
    fn test_memory_create_branch() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        mem.add_turn("c", "d");
        mem.create_branch("alt", 1).unwrap();
        assert!(mem.get_branch("alt").is_some());
        assert_eq!(mem.get_branch("alt").unwrap().branch_point(), 1);
    }

    #[test]
    fn test_memory_create_duplicate_branch() {
        let mut mem = ConversationMemory::new();
        mem.create_branch("x", 0).unwrap();
        let err = mem.create_branch("x", 0).unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn test_memory_create_branch_out_of_bounds() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        let err = mem.create_branch("bad", 5).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }

    #[test]
    fn test_memory_switch_branch() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        mem.create_branch("alt", 1).unwrap();
        mem.switch_branch("alt").unwrap();
        assert_eq!(mem.current_branch(), Some("alt"));
    }

    #[test]
    fn test_memory_switch_nonexistent_branch() {
        let mut mem = ConversationMemory::new();
        let err = mem.switch_branch("nope").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn test_memory_branch_isolation() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("shared1", "shared_a1");
        mem.add_turn("shared2", "shared_a2");

        mem.create_branch("alt", 1).unwrap();
        mem.switch_branch("alt").unwrap();

        // Add turn on branch — should not appear in main
        mem.add_turn("branch_q", "branch_a");

        // On branch: should see shared1 + branch_q = 2 turns
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.get_turn(0).unwrap().human, "shared1");
        assert_eq!(mem.get_turn(1).unwrap().human, "branch_q");

        // Switch back to main
        mem.current_branch = None;
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.get_turn(1).unwrap().human, "shared2");
    }

    #[test]
    fn test_memory_branches_list() {
        let mut mem = ConversationMemory::new();
        mem.create_branch("a", 0).unwrap();
        mem.create_branch("b", 0).unwrap();
        let mut names = mem.branches();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    // ── Export and Summary ────────────────────────────────────────────

    #[test]
    fn test_memory_export() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("hi", "hello");
        mem.create_branch("alt", 1).unwrap();
        let export = mem.export();
        assert!(export["turns"].is_array());
        assert_eq!(export["turns"].as_array().unwrap().len(), 1);
        assert!(export["branches"]["alt"].is_object());
    }

    #[test]
    fn test_memory_summary() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "b");
        mem.add_turn("c", "d");
        mem.create_branch("x", 0).unwrap();
        let s = mem.summary();
        assert_eq!(s["total_turns"], 2);
        assert_eq!(s["branch_count"], 1);
    }

    // ── ConversationSearch ────────────────────────────────────────────

    #[test]
    fn test_search_by_keyword() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("Rust programming", "Great choice");
        mem.add_turn("Python scripting", "Also good");
        let search = ConversationSearch::new(&mem);
        let results = search.by_keyword("rust");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_metadata() {
        let mut mem = ConversationMemory::new();
        let mut m1 = HashMap::new();
        m1.insert("topic".to_string(), json!("math"));
        mem.add_turn_with_metadata("2+2?", "4", m1);

        let mut m2 = HashMap::new();
        m2.insert("topic".to_string(), json!("history"));
        mem.add_turn_with_metadata("When?", "1776", m2);

        let search = ConversationSearch::new(&mem);
        let results = search.by_metadata("topic", &json!("math"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ai, "4");
    }

    #[test]
    fn test_search_recent() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "1");
        mem.add_turn("b", "2");
        mem.add_turn("c", "3");
        let search = ConversationSearch::new(&mem);
        let results = search.recent(2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].human, "b");
        assert_eq!(results[1].human, "c");
    }

    #[test]
    fn test_search_recent_exceeds() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("a", "1");
        let search = ConversationSearch::new(&mem);
        let results = search.recent(10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_time_range() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("recent", "yes");
        // All turns are just created, so age ≈ 0. Search for 0..10s should match.
        let search = ConversationSearch::new(&mem);
        let results = search.by_time_range(Duration::from_secs(0), Duration::from_secs(10));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_time_range_excludes_old() {
        let mut mem = ConversationMemory::new();
        mem.add_turn("recent", "yes");
        // Search for turns between 60s and 120s old — should exclude recent turns.
        let search = ConversationSearch::new(&mem);
        let results = search.by_time_range(Duration::from_secs(60), Duration::from_secs(120));
        assert!(results.is_empty());
    }

    // ── Default trait ─────────────────────────────────────────────────

    #[test]
    fn test_memory_default() {
        let mem = ConversationMemory::default();
        assert!(mem.is_empty());
    }
}
