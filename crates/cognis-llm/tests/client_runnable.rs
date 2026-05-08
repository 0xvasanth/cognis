//! Verify Client implements Runnable<Vec<Message>, Message> by using a
//! hand-rolled fake provider. No network calls.

use std::sync::Arc;

use async_trait::async_trait;
use cognis_core::{Result, Runnable, RunnableConfig, RunnableStream};
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};
use cognis_llm::{Client, Message};

struct EchoProvider {
    last_messages: std::sync::Mutex<Option<Vec<Message>>>,
    last_opts: std::sync::Mutex<Option<ChatOptions>>,
}

impl EchoProvider {
    fn new() -> Self {
        Self {
            last_messages: std::sync::Mutex::new(None),
            last_opts: std::sync::Mutex::new(None),
        }
    }
}

#[async_trait]
impl LLMProvider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        *self.last_messages.lock().unwrap() = Some(messages.clone());
        *self.last_opts.lock().unwrap() = Some(opts);
        let last = messages
            .last()
            .map(|m| m.content().to_string())
            .unwrap_or_default();
        Ok(ChatResponse {
            message: Message::ai(format!("echo: {last}")),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "echo-1".into(),
        })
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        *self.last_messages.lock().unwrap() = Some(messages.clone());
        *self.last_opts.lock().unwrap() = Some(opts);
        let last = messages
            .last()
            .map(|m| m.content().to_string())
            .unwrap_or_default();
        Ok(RunnableStream::once(Ok(StreamChunk {
            content: format!("stream:{last}"),
            is_done: true,
            ..Default::default()
        })))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

#[tokio::test]
async fn client_invoke_via_runnable_trait() {
    let client = Client::new(Arc::new(EchoProvider::new()));
    // Use UFCS to call Runnable::invoke (avoids ambiguity with Client's own invoke method).
    let cfg = RunnableConfig::default();
    let out: Message = Runnable::invoke(&client, vec![Message::human("hello")], cfg)
        .await
        .unwrap();
    assert_eq!(out.content(), "echo: hello");
}

#[tokio::test]
async fn client_invoke_direct_method() {
    let client = Client::new(Arc::new(EchoProvider::new()));
    let out = client.invoke(vec![Message::human("yo")]).await.unwrap();
    assert_eq!(out.content(), "echo: yo");
}

#[tokio::test]
async fn client_stream_via_runnable() {
    let client = Client::new(Arc::new(EchoProvider::new()));
    let cfg = RunnableConfig::default();
    // Use UFCS to call Runnable::stream (avoids ambiguity with Client's own stream method).
    let s: RunnableStream<Message> = Runnable::stream(&client, vec![Message::human("hi")], cfg)
        .await
        .unwrap();
    let v = s.collect_into_vec().await.unwrap();
    // Default Runnable::stream wraps invoke as a single-item stream.
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].content(), "echo: hi");
}

#[tokio::test]
async fn client_forwards_messages_to_provider() {
    let provider = Arc::new(EchoProvider::new());
    let client = Client::new(Arc::clone(&provider) as Arc<dyn LLMProvider>);
    let _ = client
        .invoke(vec![Message::human("test message")])
        .await
        .unwrap();
    let recorded = provider.last_messages.lock().unwrap().clone().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].content(), "test message");
}
