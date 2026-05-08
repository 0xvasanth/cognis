//! QA over documents — V2 expresses this as a Runnable composition
//! (build context → format prompt → call model). V1's monolithic
//! `QAChain` is replaced by composable building blocks.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};
use cognis_llm::Client;

struct CannedProvider;
#[async_trait]
impl LLMProvider for CannedProvider {
    fn name(&self) -> &str { "canned" }
    fn provider_type(&self) -> Provider { Provider::Ollama }
    async fn chat_completion(&self, msgs: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let q = msgs.last().map(|m| m.content().to_string()).unwrap_or_default();
        Ok(ChatResponse {
            message: Message::ai(format!(
                "Based on the provided documents, the answer to: {q}"
            )),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "canned".into(),
        })
    }
    async fn chat_completion_stream(&self, _: Vec<Message>, _: ChatOptions) -> Result<RunnableStream<StreamChunk>> { unimplemented!() }
    async fn health_check(&self) -> Result<HealthStatus> { Ok(HealthStatus::Healthy { latency_ms: 0 }) }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 QA over Documents ===\n");

    let docs = vec![
        "Rust was first released in 2010 by Mozilla.".to_string(),
        "Rust 1.0 was released on May 15, 2015.".to_string(),
    ];
    let question = "When was Rust 1.0 released?";

    let context = docs
        .iter()
        .enumerate()
        .map(|(i, d)| format!("[{}] {d}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Use only the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    );

    let client = if std::env::var("COGNIS_PROVIDER").is_ok() {
        Client::from_env()?
    } else {
        Client::new(Arc::new(CannedProvider))
    };
    let reply = client.invoke(vec![Message::human(prompt)]).await?;
    println!("Q: {question}");
    println!("A: {}", reply.content());
    Ok(())
}
