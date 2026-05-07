//! Todo middleware — injects a system message reminding the model to use a
//! todo list. Pairs well with a `todo_*` tool surface.
//!
//! Intentionally minimal: this is a *prompt* middleware. The actual todo
//! state belongs in agent state / a tool, not in middleware.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Message, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Inserts a system message at the head of every request reminding the
/// model to plan/track work. Idempotent — won't insert twice if it's
/// already present.
pub struct TodoMiddleware {
    instruction: String,
}

impl Default for TodoMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl TodoMiddleware {
    /// Default instruction.
    pub fn new() -> Self {
        Self {
            instruction: DEFAULT_TODO_INSTRUCTION.to_string(),
        }
    }

    /// Override the system instruction.
    pub fn with_instruction(mut self, s: impl Into<String>) -> Self {
        self.instruction = s.into();
        self
    }
}

const DEFAULT_TODO_INSTRUCTION: &str =
    "When tackling multi-step work, plan your steps as a checklist before \
     acting. Use the `todo_*` tools (when available) to track progress so \
     the user can audit your reasoning.";

const MARKER: &str = "<!-- cognis2:todo-mw -->";

#[async_trait]
impl Middleware for TodoMiddleware {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let already_present = ctx
            .messages
            .iter()
            .any(|m| matches!(m, Message::System(s) if s.content.contains(MARKER)));
        if !already_present {
            let body = format!("{MARKER}\n{}", self.instruction);
            ctx.messages.insert(0, Message::system(body));
        }
        next.invoke(ctx).await
    }

    fn name(&self) -> &str {
        "TodoMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis2_llm::chat::ChatOptions;
    use cognis2_llm::Client;

    #[tokio::test]
    async fn inserts_marker_once() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(TodoMiddleware::new())
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let _ = pipe
            .invoke(
                vec![
                    Message::human("hi"),
                    Message::system(format!("{MARKER}\nold")),
                ],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        assert_eq!(received.len(), 2);
        // First call: marker added at head.
        let first = &received[0].0;
        assert!(first[0].content().contains(MARKER));
        // Second call: caller already had a marker → middleware doesn't double-insert.
        let second = &received[1].0;
        let count = second
            .iter()
            .filter(|m| m.content().contains(MARKER))
            .count();
        assert_eq!(count, 1);
    }
}
