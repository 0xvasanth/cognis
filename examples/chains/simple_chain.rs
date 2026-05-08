//! Simple Runnable composition (V2's replacement for V1 Chain types).
//!
//! Shows the LCEL pattern: prompt → model → parser, composed via the
//! fluent `.pipe()` method on `RunnableExt`.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::output_parsers::StringParser;
use cognis_core::prompts::PromptTemplate;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};
use cognis_llm::Client;

/// Fake provider that echoes a fixed answer — keeps the demo offline.
struct EchoProvider(&'static str);

#[async_trait]
impl LLMProvider for EchoProvider {
    fn name(&self) -> &str { "echo" }
    fn provider_type(&self) -> Provider { Provider::Ollama }
    async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::ai(self.0),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "echo".into(),
        })
    }
    async fn chat_completion_stream(&self, _: Vec<Message>, _: ChatOptions) -> Result<RunnableStream<StreamChunk>> {
        unimplemented!()
    }
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Simple Composition Example ===\n");

    let prompt: PromptTemplate<serde_json::Value> =
        PromptTemplate::new("Write one short joke about {topic}.");

    let client = if std::env::var("COGNIS_PROVIDER").is_ok() {
        Client::from_env()?
    } else {
        Client::new(Arc::new(EchoProvider("Why don't scientists trust atoms? They make up everything.")))
    };

    let rendered = prompt.invoke(serde_json::json!({"topic": "ice cream"}), Default::default()).await?;
    let reply = client.invoke(vec![Message::human(rendered)]).await?;

    let parser = StringParser::new();
    let final_text = parser.invoke(reply.content().to_string(), Default::default()).await?;

    println!("{final_text}");
    Ok(())
}
