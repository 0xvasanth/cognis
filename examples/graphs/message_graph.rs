//! AgentState — the canonical chat state. Append messages, route
//! between an LLM node and an end node.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};

struct Echo;
#[async_trait]
impl LLMProvider for Echo {
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
    let client = Client::new(Arc::new(Echo));
    let llm = node_fn::<AgentState, _, _>("llm", move |state, _| {
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
    let g = Graph::<AgentState>::new()
        .node("llm", llm)
        .start_at("llm")
        .compile()?;
    let mut a = Agent::wrap(g);
    let r = a.run(Message::human("hello")).await?;
    println!("{}", r.content);
    Ok(())
}
