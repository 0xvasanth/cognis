//! Power-user demo: hand-build the graph yourself, hand it to Agent::wrap.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};

struct StaticEcho;
#[async_trait]
impl LLMProvider for StaticEcho {
    fn name(&self) -> &str {
        "echo"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }
    async fn chat_completion(&self, msgs: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let last = msgs
            .last()
            .map(|m| m.content().to_string())
            .unwrap_or_default();
        Ok(ChatResponse {
            message: Message::ai(format!("ECHO: {last}")),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "echo".into(),
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

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::new(Arc::new(StaticEcho));

    // Hand-build a single-node graph that calls the LLM and ends.
    let single_node = node_fn::<AgentState, _, _>("call", move |state, _ctx| {
        let client = client.clone();
        let messages = state.messages.clone();
        async move {
            let resp = client
                .provider()
                .chat_completion(messages, ChatOptions::default())
                .await?;
            Ok(NodeOut {
                update: AgentStateUpdate {
                    messages: vec![resp.message],
                    iterations: 1,
                },
                goto: Goto::end(),
            })
        }
    });
    let graph = Graph::<AgentState>::new()
        .node("call", single_node)
        .start_at("call")
        .compile()?;

    let mut agent = Agent::wrap(graph);
    let resp = agent.run(Message::human("hello custom graph")).await?;
    println!("{}", resp.content);
    Ok(())
}
