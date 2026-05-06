//! The canonical "5-line agent" demo — works without any network calls.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2::prelude::*;
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
    async fn chat_completion(&self, msgs: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let last = msgs
            .last()
            .map(|m| m.content().to_string())
            .unwrap_or_default();
        Ok(ChatResponse {
            message: Message::ai(format!("(fake reply to: {last})")),
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

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::new(Arc::new(FakeProvider));
    let mut agent = AgentBuilder::new().with_llm(client).build()?;
    let resp = agent.run(Message::human("hello, world")).await?;
    println!("{}", resp.content);
    Ok(())
}
