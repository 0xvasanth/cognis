//! Wrap any [`Agent`](crate::Agent) as a [`Tool`].
//!
//! Why a tool, not a middleware? An "agent that calls sub-agents" is
//! exactly an agent with extra tools — there's no special control flow.
//! This file is the bridge: take an `Agent`, give it a name + description,
//! and the parent agent can dispatch to it through the standard tool path.
//!
//! The sub-agent receives the parent's task as a single human message, runs
//! its own ReAct loop to completion, and returns the final assistant text.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;
use tokio::sync::Mutex;

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

use crate::agent::Agent;

#[derive(Debug, Deserialize, JsonSchema)]
struct SubAgentInput {
    /// The task / question to delegate to the sub-agent.
    task: String,
}

/// Adapts an [`Agent`] to the [`Tool`] interface.
///
/// Each call clones a *fresh* logical conversation: the sub-agent starts
/// with its own system prompt + the parent-supplied task, runs to
/// completion, returns the final assistant message text. Sub-agent state
/// is held behind a `Mutex` so multiple concurrent tool calls serialize
/// — sub-agents are expected to carry their own state, not be shared
/// reentrant across parent calls.
pub struct SubAgentTool {
    name: String,
    description: String,
    agent: Arc<Mutex<Agent>>,
}

impl SubAgentTool {
    /// Build a sub-agent tool.
    ///
    /// - `name`: how the parent agent addresses this sub-agent.
    /// - `description`: what the sub-agent is good at — the LLM uses this
    ///   to decide when to delegate.
    /// - `agent`: the inner agent. Wrapped in `Mutex` for serialized access.
    pub fn new(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent: Arc::new(Mutex::new(agent)),
        }
    }
}

#[async_trait]
impl Tool for SubAgentTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(SubAgentInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: SubAgentInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("subagent: {e}")))?;
        let mut agent = self.agent.lock().await;
        let resp = agent.run(parsed.task).await?;
        Ok(ToolOutput::Text(resp.content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use serde_json::json;

    use cognis2_core::{Message, Result, RunnableStream};
    use cognis2_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis2_llm::provider::{LLMProvider, Provider};
    use cognis2_llm::Client;

    use crate::agent::default_react_graph;

    /// Tracks calls so we can verify the sub-agent ran.
    struct Constant {
        content: String,
        calls: AtomicUsize,
        seen: StdMutex<Vec<Vec<Message>>>,
    }

    #[async_trait]
    impl LLMProvider for Constant {
        fn name(&self) -> &str {
            "constant"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(messages);
            Ok(ChatResponse {
                message: Message::ai(&self.content),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "constant".into(),
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

    #[tokio::test]
    async fn delegates_task_to_inner_agent() {
        let provider = Arc::new(Constant {
            content: "sub-agent result".into(),
            calls: AtomicUsize::new(0),
            seen: StdMutex::new(Vec::new()),
        });
        let client = Client::new(provider.clone());
        let graph = default_react_graph(client, Vec::new(), 5).unwrap();
        let inner = Agent::wrap(graph);

        let tool = SubAgentTool::new("research", "Researches a topic", inner);
        let mut a = std::collections::HashMap::new();
        a.insert("task".into(), json!("look up rust"));
        let out = tool._run(ToolInput::Structured(a)).await.unwrap();
        assert_eq!(out.as_string(), "sub-agent result");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        // Sub-agent received the task as a human message.
        let received = provider.seen.lock().unwrap();
        assert!(received[0]
            .iter()
            .any(|m| matches!(m, Message::Human(h) if h.content.contains("look up rust"))));
    }
}
