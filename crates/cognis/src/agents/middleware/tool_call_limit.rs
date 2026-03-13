//! Tool call limit middleware — enforce limits on tool invocations.
//!
//! Mirrors Python `langchain.agents.middleware.tool_call_limit`.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::{Result, CognisError};

use super::types::{AgentMiddleware, AgentState};

/// What to do when a tool call limit is exceeded.
#[derive(Debug, Clone)]
pub enum LimitExceededBehavior {
    /// Block the exceeded tool calls but continue execution.
    Continue,
    /// Raise an error.
    Error,
    /// End the agent loop immediately.
    End,
}

/// Middleware that enforces limits on the number of tool calls.
pub struct ToolCallLimitMiddleware {
    /// Maximum total tool calls across the entire run.
    pub max_total_calls: Option<usize>,
    /// Per-tool call limits (tool_name -> max_calls).
    pub per_tool_limits: HashMap<String, usize>,
    /// Optional filter: only count/limit calls to this specific tool.
    pub tool_name_filter: Option<String>,
    /// Behavior when limit is exceeded.
    pub on_exceeded: LimitExceededBehavior,
    /// Internal counter for tool calls.
    call_counts: Mutex<HashMap<String, usize>>,
    total_calls: Mutex<usize>,
    /// Stored middleware name (includes tool name if filtered).
    middleware_name: String,
}

impl ToolCallLimitMiddleware {
    pub fn new(max_total_calls: Option<usize>) -> Self {
        Self {
            max_total_calls,
            per_tool_limits: HashMap::new(),
            tool_name_filter: None,
            on_exceeded: LimitExceededBehavior::Error,
            call_counts: Mutex::new(HashMap::new()),
            total_calls: Mutex::new(0),
            middleware_name: "ToolCallLimitMiddleware".to_string(),
        }
    }

    pub fn with_per_tool_limit(mut self, tool_name: impl Into<String>, limit: usize) -> Self {
        self.per_tool_limits.insert(tool_name.into(), limit);
        self
    }

    pub fn with_behavior(mut self, behavior: LimitExceededBehavior) -> Self {
        self.on_exceeded = behavior;
        self
    }

    /// Set a tool name filter. When set, only calls to this specific tool are counted and limited.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.middleware_name = format!("ToolCallLimitMiddleware[{}]", &name);
        self.tool_name_filter = Some(name);
        self
    }

    /// Reset all counters.
    pub fn reset(&self) {
        *self.call_counts.lock().unwrap() = HashMap::new();
        *self.total_calls.lock().unwrap() = 0;
    }

    /// Check if a tool call would exceed limits.
    pub fn would_exceed(&self, tool_name: &str) -> bool {
        let total = *self.total_calls.lock().unwrap();
        if let Some(max) = self.max_total_calls {
            if total >= max {
                return true;
            }
        }
        if let Some(&limit) = self.per_tool_limits.get(tool_name) {
            let counts = self.call_counts.lock().unwrap();
            if counts.get(tool_name).copied().unwrap_or(0) >= limit {
                return true;
            }
        }
        false
    }

    /// Check if a tool call would exceed limits using state-based counts.
    fn would_exceed_from_state(&self, tool_name: &str, state: &AgentState) -> bool {
        // Check total from state
        let total = state
            .extra
            .get("tool_call_count")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| self.total_count() as u64) as usize;
        if let Some(max) = self.max_total_calls {
            if total >= max {
                return true;
            }
        }
        // Check per-tool from state
        if let Some(&limit) = self.per_tool_limits.get(tool_name) {
            let key = format!("tool_call_count:{}", tool_name);
            let count = state
                .extra
                .get(&key)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| {
                    self.call_counts
                        .lock()
                        .unwrap()
                        .get(tool_name)
                        .copied()
                        .unwrap_or(0) as u64
                }) as usize;
            if count >= limit {
                return true;
            }
        }
        false
    }

    /// Record a tool call.
    pub fn record_call(&self, tool_name: &str) {
        *self.total_calls.lock().unwrap() += 1;
        let mut counts = self.call_counts.lock().unwrap();
        *counts.entry(tool_name.to_string()).or_insert(0) += 1;
    }

    /// Get the current total call count.
    pub fn total_count(&self) -> usize {
        *self.total_calls.lock().unwrap()
    }
}

#[async_trait]
impl AgentMiddleware for ToolCallLimitMiddleware {
    fn name(&self) -> &str {
        &self.middleware_name
    }

    async fn after_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        // Check the last message for tool calls
        if let Some(cognis_core::messages::Message::Ai(ai_msg)) = state.messages.last() {
            let mut blocked_tools: Vec<String> = Vec::new();
            let mut updates = HashMap::new();

            for tc in &ai_msg.tool_calls {
                let name = tc.name.as_str();

                // If a tool_name_filter is set, skip tools that don't match.
                if let Some(ref filter) = self.tool_name_filter {
                    if name != filter.as_str() {
                        continue;
                    }
                }

                if self.would_exceed_from_state(name, state) {
                    match &self.on_exceeded {
                        LimitExceededBehavior::Error => {
                            return Err(CognisError::Other(format!(
                                "Tool call limit exceeded for '{}'",
                                name
                            )));
                        }
                        LimitExceededBehavior::End => {
                            updates.insert("jump_to".into(), serde_json::json!("end"));
                            return Ok(Some(updates));
                        }
                        LimitExceededBehavior::Continue => {
                            // Track blocked tools so the tool node can generate
                            // error ToolMessages for them.
                            blocked_tools.push(name.to_string());
                            continue;
                        }
                    }
                }
                // Record the call in internal Mutex counters.
                self.record_call(name);

                // Update state-based counts.
                let new_total = self.total_count();
                updates.insert("tool_call_count".into(), serde_json::json!(new_total));

                let tool_key = format!("tool_call_count:{}", name);
                let new_tool_count = self
                    .call_counts
                    .lock()
                    .unwrap()
                    .get(name)
                    .copied()
                    .unwrap_or(0);
                updates.insert(tool_key, serde_json::json!(new_tool_count));
            }

            // If any tools were blocked in Continue mode, record them in state.
            if !blocked_tools.is_empty() {
                updates.insert(
                    "blocked_tool_calls".into(),
                    serde_json::json!(blocked_tools),
                );
            }

            if !updates.is_empty() {
                return Ok(Some(updates));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_limit_new() {
        let mw = ToolCallLimitMiddleware::new(Some(10));
        assert_eq!(mw.max_total_calls, Some(10));
        assert_eq!(mw.total_count(), 0);
        assert_eq!(mw.name(), "ToolCallLimitMiddleware");
    }

    #[test]
    fn test_tool_call_limit_with_tool_name() {
        let mw = ToolCallLimitMiddleware::new(Some(5)).with_tool_name("search");
        assert_eq!(mw.tool_name_filter, Some("search".to_string()));
        assert_eq!(mw.name(), "ToolCallLimitMiddleware[search]");
    }

    #[test]
    fn test_tool_call_limit_record_and_check() {
        let mw = ToolCallLimitMiddleware::new(Some(2));
        assert!(!mw.would_exceed("test_tool"));
        mw.record_call("test_tool");
        assert!(!mw.would_exceed("test_tool"));
        mw.record_call("test_tool");
        assert!(mw.would_exceed("test_tool")); // total = 2, max = 2
    }

    #[test]
    fn test_per_tool_limit() {
        let mw = ToolCallLimitMiddleware::new(None).with_per_tool_limit("search", 1);
        assert!(!mw.would_exceed("search"));
        mw.record_call("search");
        assert!(mw.would_exceed("search"));
        assert!(!mw.would_exceed("other_tool"));
    }

    #[test]
    fn test_would_exceed_from_state() {
        let mw = ToolCallLimitMiddleware::new(Some(2)).with_per_tool_limit("search", 1);
        let mut state = AgentState::default();
        state.set_extra("tool_call_count", serde_json::json!(1));
        state.set_extra("tool_call_count:search", serde_json::json!(1));
        // Total not exceeded (1 < 2), but per-tool limit exceeded for "search"
        assert!(mw.would_exceed_from_state("search", &state));
        assert!(!mw.would_exceed_from_state("other", &state));
    }

    #[test]
    fn test_reset() {
        let mw = ToolCallLimitMiddleware::new(Some(5));
        mw.record_call("a");
        mw.record_call("b");
        assert_eq!(mw.total_count(), 2);
        mw.reset();
        assert_eq!(mw.total_count(), 0);
    }
}
