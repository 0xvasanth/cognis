//! Model call limit middleware — enforce limits on the number of model invocations.
//!
//! Mirrors Python `langchain.agents.middleware.model_call_limit`.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::{Result, CognisError};

use super::types::{AgentMiddleware, AgentState};

/// What to do when the model call limit is exceeded.
#[derive(Debug, Clone)]
pub enum ExitBehavior {
    /// End the agent loop gracefully.
    End,
    /// Raise an error.
    Error,
}

/// Middleware that enforces limits on the number of model calls.
///
/// Tracks model invocations per thread and per run, preventing runaway loops.
pub struct ModelCallLimitMiddleware {
    /// Maximum model calls per thread (conversation). `None` means unlimited.
    pub thread_limit: Option<usize>,
    /// Maximum model calls per run. `None` means unlimited.
    pub run_limit: Option<usize>,
    /// Behavior when limit is exceeded.
    pub exit_behavior: ExitBehavior,
    /// Per-thread call counters (thread_id -> count).
    thread_counts: Mutex<HashMap<String, usize>>,
    /// Current run call counter.
    run_count: Mutex<usize>,
}

impl ModelCallLimitMiddleware {
    /// Create a new middleware with the given run limit.
    pub fn new(run_limit: Option<usize>) -> Self {
        Self {
            thread_limit: None,
            run_limit,
            exit_behavior: ExitBehavior::Error,
            thread_counts: Mutex::new(HashMap::new()),
            run_count: Mutex::new(0),
        }
    }

    pub fn with_thread_limit(mut self, limit: usize) -> Self {
        self.thread_limit = Some(limit);
        self
    }

    pub fn with_exit_behavior(mut self, behavior: ExitBehavior) -> Self {
        self.exit_behavior = behavior;
        self
    }

    /// Reset all counters.
    pub fn reset(&self) {
        *self.thread_counts.lock().unwrap() = HashMap::new();
        *self.run_count.lock().unwrap() = 0;
    }

    /// Reset just the run counter (called between runs).
    pub fn reset_run(&self) {
        *self.run_count.lock().unwrap() = 0;
    }

    /// Get the current run call count.
    pub fn run_count(&self) -> usize {
        *self.run_count.lock().unwrap()
    }

    /// Get the call count for a specific thread.
    pub fn thread_count(&self, thread_id: &str) -> usize {
        self.thread_counts
            .lock()
            .unwrap()
            .get(thread_id)
            .copied()
            .unwrap_or(0)
    }

    /// Check if the current run would exceed limits (based on internal Mutex counters).
    pub fn would_exceed_run(&self) -> bool {
        if let Some(limit) = self.run_limit {
            let count = *self.run_count.lock().unwrap();
            return count >= limit;
        }
        false
    }

    /// Check if a thread would exceed its limit (based on internal Mutex counters).
    pub fn would_exceed_thread(&self, thread_id: &str) -> bool {
        if let Some(limit) = self.thread_limit {
            let counts = self.thread_counts.lock().unwrap();
            let count = counts.get(thread_id).copied().unwrap_or(0);
            return count >= limit;
        }
        false
    }

    /// Increment counters for a model call.
    fn record_call(&self, thread_id: Option<&str>) {
        *self.run_count.lock().unwrap() += 1;
        if let Some(tid) = thread_id {
            let mut counts = self.thread_counts.lock().unwrap();
            *counts.entry(tid.to_string()).or_insert(0) += 1;
        }
    }
}

#[async_trait]
impl AgentMiddleware for ModelCallLimitMiddleware {
    fn name(&self) -> &str {
        "ModelCallLimitMiddleware"
    }

    async fn before_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        let thread_id = state
            .extra
            .get("thread_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Read current counts from state (state-based counting).
        let run_count = state
            .extra
            .get("model_call_count")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| self.run_count() as u64) as usize;

        // Check run limit
        if let Some(limit) = self.run_limit {
            if run_count >= limit {
                return match &self.exit_behavior {
                    ExitBehavior::Error => Err(CognisError::Other(format!(
                        "Model call run limit exceeded (limit: {:?})",
                        self.run_limit
                    ))),
                    ExitBehavior::End => {
                        let mut updates = HashMap::new();
                        updates.insert("jump_to".into(), serde_json::json!("end"));
                        Ok(Some(updates))
                    }
                };
            }
        }

        // Check thread limit
        if let Some(ref tid) = thread_id {
            let thread_key = format!("model_call_count_thread:{}", tid);
            let thread_count = state
                .extra
                .get(&thread_key)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| self.thread_count(tid) as u64)
                as usize;

            if let Some(limit) = self.thread_limit {
                if thread_count >= limit {
                    return match &self.exit_behavior {
                        ExitBehavior::Error => Err(CognisError::Other(format!(
                            "Model call thread limit exceeded for thread '{}' (limit: {:?})",
                            tid, self.thread_limit
                        ))),
                        ExitBehavior::End => {
                            let mut updates = HashMap::new();
                            updates.insert("jump_to".into(), serde_json::json!("end"));
                            Ok(Some(updates))
                        }
                    };
                }
            }
        }

        // Do NOT increment here — counting happens in after_model.
        Ok(None)
    }

    async fn after_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        let thread_id = state
            .extra
            .get("thread_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Increment the internal Mutex counters (for total_count()/run_count() methods).
        self.record_call(thread_id.as_deref());

        // Build state updates for state-based counting.
        let mut updates = HashMap::new();

        let new_run_count = self.run_count();
        updates.insert("model_call_count".into(), serde_json::json!(new_run_count));

        if let Some(ref tid) = thread_id {
            let thread_key = format!("model_call_count_thread:{}", tid);
            let new_thread_count = self.thread_count(tid);
            updates.insert(thread_key, serde_json::json!(new_thread_count));
        }

        Ok(Some(updates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_call_limit_new() {
        let mw = ModelCallLimitMiddleware::new(Some(10));
        assert_eq!(mw.run_limit, Some(10));
        assert!(mw.thread_limit.is_none());
        assert_eq!(mw.run_count(), 0);
    }

    #[test]
    fn test_model_call_limit_record_and_check() {
        let mw = ModelCallLimitMiddleware::new(Some(3));
        assert!(!mw.would_exceed_run());
        mw.record_call(None);
        mw.record_call(None);
        assert!(!mw.would_exceed_run());
        mw.record_call(None);
        assert!(mw.would_exceed_run());
        assert_eq!(mw.run_count(), 3);
    }

    #[test]
    fn test_model_call_limit_thread() {
        let mw = ModelCallLimitMiddleware::new(None).with_thread_limit(2);
        mw.record_call(Some("thread-1"));
        assert_eq!(mw.thread_count("thread-1"), 1);
        mw.record_call(Some("thread-1"));
        assert!(mw.would_exceed_thread("thread-1"));
        assert!(!mw.would_exceed_thread("thread-2"));
    }

    #[test]
    fn test_model_call_limit_reset() {
        let mw = ModelCallLimitMiddleware::new(Some(10));
        mw.record_call(Some("t1"));
        mw.record_call(Some("t1"));
        assert_eq!(mw.run_count(), 2);
        mw.reset();
        assert_eq!(mw.run_count(), 0);
        assert_eq!(mw.thread_count("t1"), 0);
    }

    #[test]
    fn test_model_call_limit_reset_run() {
        let mw = ModelCallLimitMiddleware::new(Some(10));
        mw.record_call(Some("t1"));
        assert_eq!(mw.run_count(), 1);
        mw.reset_run();
        assert_eq!(mw.run_count(), 0);
        // Thread count should still be present
        assert_eq!(mw.thread_count("t1"), 1);
    }

    #[tokio::test]
    async fn test_model_call_limit_before_model_within_limit() {
        let mw = ModelCallLimitMiddleware::new(Some(5));
        let state = AgentState::default();
        // before_model only checks, does not increment
        let result = mw.before_model(&state).await.unwrap();
        assert!(result.is_none());
        assert_eq!(mw.run_count(), 0); // not incremented yet
    }

    #[tokio::test]
    async fn test_model_call_limit_after_model_increments() {
        let mw = ModelCallLimitMiddleware::new(Some(5));
        let state = AgentState::default();
        let updates = mw.after_model(&state).await.unwrap();
        assert!(updates.is_some());
        let updates = updates.unwrap();
        assert_eq!(updates.get("model_call_count"), Some(&serde_json::json!(1)));
        assert_eq!(mw.run_count(), 1);
    }

    #[tokio::test]
    async fn test_model_call_limit_before_model_exceeds_error() {
        let mw = ModelCallLimitMiddleware::new(Some(1));
        // Simulate state where count already reached limit
        let mut state = AgentState::default();
        state.set_extra("model_call_count", serde_json::json!(1));
        // Should error because state shows count >= limit
        let result = mw.before_model(&state).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_model_call_limit_before_model_exceeds_end() {
        let mw = ModelCallLimitMiddleware::new(Some(1)).with_exit_behavior(ExitBehavior::End);
        let mut state = AgentState::default();
        state.set_extra("model_call_count", serde_json::json!(1));
        let result = mw.before_model(&state).await.unwrap();
        assert!(result.is_some());
        let updates = result.unwrap();
        assert_eq!(updates.get("jump_to"), Some(&serde_json::json!("end")));
    }

    #[test]
    fn test_model_call_limit_name() {
        let mw = ModelCallLimitMiddleware::new(None);
        assert_eq!(mw.name(), "ModelCallLimitMiddleware");
    }
}
