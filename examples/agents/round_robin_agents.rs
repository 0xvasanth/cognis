//! RoundRobin — distribute each call across an agent pool, one agent
//! per call, cycling. Useful when several interchangeable agents share
//! a role (e.g. "answerer") and you want to load-balance across them.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis::{AgentBuilder, MultiAgentOrchestrator, RoundRobin};
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};

struct Tagged(&'static str);
#[async_trait]
impl LLMProvider for Tagged {
    fn name(&self) -> &str { self.0 }
    fn provider_type(&self) -> Provider { Provider::Ollama }
    async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::ai(format!("from-{}", self.0)),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: self.0.into(),
        })
    }
    async fn chat_completion_stream(&self, _: Vec<Message>, _: ChatOptions) -> Result<RunnableStream<StreamChunk>> { unimplemented!() }
    async fn health_check(&self) -> Result<HealthStatus> { Ok(HealthStatus::Healthy { latency_ms: 0 }) }
}

fn agent(tag: &'static str) -> Result<cognis::Agent> {
    AgentBuilder::new()
        .with_llm(Client::new(Arc::new(Tagged(tag))))
        .stateless()
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let orch = MultiAgentOrchestrator::new(RoundRobin::new())
        .add("alpha", agent("alpha")?)
        .add("beta", agent("beta")?)
        .add("gamma", agent("gamma")?);

    for i in 0..6 {
        let r = orch.run(format!("query #{i}")).await?;
        println!("query #{i} → {}", r.content);
    }
    Ok(())
}
