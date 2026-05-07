//! Planning middleware — inject a "plan first, execute after" instruction.
//!
//! This is the conservative, prompt-only variant. It nudges the model to
//! emit a numbered plan before running tools. For dynamic plan tracking
//! (mark steps in_progress / completed), use the `todo` middleware paired
//! with a todo-list tool.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

const MARKER: &str = "<!-- cognis:planning-mw -->";
const DEFAULT_INSTRUCTION: &str =
    "Before taking any action, write out a short numbered plan of the steps \
     you'll take. Then execute each step. After completing all steps, \
     summarize the result.";

/// Inserts a planning instruction at the head of every request. Idempotent.
pub struct Planning {
    instruction: String,
}

impl Default for Planning {
    fn default() -> Self {
        Self::new()
    }
}

impl Planning {
    /// Default instruction.
    pub fn new() -> Self {
        Self {
            instruction: DEFAULT_INSTRUCTION.to_string(),
        }
    }

    /// Override the instruction.
    pub fn with_instruction(mut self, s: impl Into<String>) -> Self {
        self.instruction = s.into();
        self
    }
}

#[async_trait]
impl Middleware for Planning {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let already = ctx
            .messages
            .iter()
            .any(|m| matches!(m, Message::System(s) if s.content.contains(MARKER)));
        if !already {
            let body = format!("{MARKER}\n{}", self.instruction);
            ctx.messages.insert(0, Message::system(body));
        }
        next.invoke(ctx).await
    }

    fn name(&self) -> &str {
        "Planning"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn injects_planning_instruction() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(Planning::new())
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(
                vec![Message::human("solve x")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        assert!(received[0].0[0].content().contains(MARKER));
    }

    #[tokio::test]
    async fn idempotent_no_double_insert() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(Planning::new())
            .build(Client::new(rec.clone()));
        let already = vec![
            Message::system(format!("{MARKER}\nold")),
            Message::human("hi"),
        ];
        let _ = pipe
            .invoke(already, Vec::new(), ChatOptions::default())
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        let count = received[0]
            .0
            .iter()
            .filter(|m| m.content().contains(MARKER))
            .count();
        assert_eq!(count, 1);
    }
}
