//! V2 RoutingProvider: dispatch each call to one of N providers based
//! on a predicate. Mirrors V1's RoutingChatModel.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};
use cognis_llm::{ProviderRoute, RoutingProvider};

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

#[tokio::main]
async fn main() -> Result<()> {
    let default_p: Arc<dyn LLMProvider> = Arc::new(Tagged("default"));
    let big_p: Arc<dyn LLMProvider> = Arc::new(Tagged("big-context"));

    let r = RoutingProvider::new("router", default_p)
        .route(ProviderRoute::new(
            "long",
            big_p,
            |msgs, _| msgs.iter().map(|m| m.content().len()).sum::<usize>() > 50,
        ));

    let short = r.chat_completion(vec![Message::human("hi")], Default::default()).await?;
    let long = r.chat_completion(vec![Message::human("a".repeat(80))], Default::default()).await?;
    println!("short → {}", short.message.content());
    println!("long  → {}", long.message.content());
    Ok(())
}
