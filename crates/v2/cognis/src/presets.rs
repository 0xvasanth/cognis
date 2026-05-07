//! Named [`AgentBuilder`] presets for common agent shapes.
//!
//! Each function returns a partially-configured `AgentBuilder` — caller
//! adds tools / system prompt overrides as needed and calls `.build()`.
//! The presets are *opinionated defaults*, not magic — read each one's
//! body to see what middleware is wired and decide if it matches your
//! needs.

use std::sync::Arc;

use cognis2_llm::Client;

use crate::agent::AgentBuilder;
use crate::tools::{Approver, Calculator};

/// Researcher: planning + summarization + a calculator. Stateless.
///
/// Suited to "ask a question, get an answer" tasks where the model
/// benefits from explicit planning and may compress long contexts.
pub fn researcher(client: Client) -> AgentBuilder {
    AgentBuilder::new()
        .with_llm(client)
        .with_system_prompt(
            "You are a research assistant. Plan your steps, then execute. \
             Be concise and cite sources when available.",
        )
        .with_tool(Arc::new(Calculator::new()))
        .with_max_iterations(15)
}

/// Coder: high iteration cap, terse system prompt. Stateful so it remembers
/// across edits within a session.
///
/// You'd typically `.with_filesystem(backend)` on top to give the agent a
/// workspace, and `.with_tool(Arc::new(ShellTool::new(...)))` for build/test.
pub fn coder(client: Client) -> AgentBuilder {
    AgentBuilder::new()
        .with_llm(client)
        .with_system_prompt(
            "You are a coding assistant. Make minimal targeted edits. \
             Run tests after every change. Don't introduce abstractions \
             beyond what the task requires.",
        )
        .with_max_iterations(40)
        .stateful()
}

/// Conversational: stateful, no tools by default, low iteration cap.
/// Suited to chat-style UX where the agent doesn't need to take actions.
pub fn conversational(client: Client) -> AgentBuilder {
    AgentBuilder::new()
        .with_llm(client)
        .with_system_prompt("You are a helpful assistant. Be concise.")
        .with_max_iterations(3)
        .stateful()
}

/// Strict: like [`researcher`] but every tool call goes through the
/// supplied [`Approver`]. Use in production where the agent's tool set
/// includes anything mutable.
pub fn strict(client: Client, approver: Arc<dyn Approver>) -> AgentBuilder {
    AgentBuilder::new()
        .with_llm(client)
        .with_system_prompt(
            "You are a careful assistant. Tools require human approval — \
             always explain why you're invoking each one.",
        )
        .with_max_iterations(20)
        .with_max_tool_calls(20)
        .with_approver(approver)
}

/// Builds nothing useful on its own — this is a "no-tools, no-frills"
/// stateless agent. Useful as a base for `.with_tool(...)` chains.
pub fn minimal(client: Client) -> AgentBuilder {
    AgentBuilder::new().with_llm(client).with_max_iterations(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis2_core::{Message, Result, RunnableStream};
    use cognis2_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis2_llm::provider::{LLMProvider, Provider};

    struct FakeProvider;
    #[async_trait]
    impl LLMProvider for FakeProvider {
        fn name(&self) -> &str {
            "fake"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai("ok"),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "fake".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn client() -> Client {
        Client::new(Arc::new(FakeProvider))
    }

    #[test]
    fn presets_build_without_panic() {
        // Each preset should produce a valid builder that completes
        // without errors.
        assert!(researcher(client()).build().is_ok());
        assert!(coder(client()).build().is_ok());
        assert!(conversational(client()).build().is_ok());
        assert!(minimal(client()).build().is_ok());
    }

    #[tokio::test]
    async fn researcher_runs() {
        let mut a = researcher(client()).build().unwrap();
        let r = a.run("test").await.unwrap();
        assert_eq!(r.content, "ok");
    }
}
