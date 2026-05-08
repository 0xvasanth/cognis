//! The canonical "5-line agent" demo.
//!
//! By default uses a fake in-process provider so it works without any
//! network calls. Set `COGNIS_PROVIDER=ollama` (with `COGNIS_OLLAMA_MODEL`
//! optionally) to run against a local Ollama server instead.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};

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
    // Pick provider: env-driven if `COGNIS_PROVIDER` is set, otherwise
    // the fake in-process provider so the demo runs without network.
    let client = if std::env::var("COGNIS_PROVIDER").is_ok() {
        println!(
            "[using {} provider via env]",
            std::env::var("COGNIS_PROVIDER").unwrap()
        );
        Client::from_env()?
    } else {
        println!("[using fake in-process provider — set COGNIS_PROVIDER=ollama for real LLM]");
        Client::new(Arc::new(FakeProvider))
    };

    let mut agent = AgentBuilder::new().with_llm(client).build()?;
    let resp = agent
        .run(Message::human("Say hello in one sentence."))
        .await?;
    println!("{}", resp.content);
    Ok(())
}
