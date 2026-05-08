//! Integration test for `Client::with_structured_output<T>`.
//!
//! Uses a fake `LLMProvider` that always returns a JSON blob; verifies the
//! StructuredClient deserializes it into a typed value.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use cognis_core::{Result, Runnable, RunnableConfig, RunnableStream};
use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis_llm::provider::{LLMProvider, Provider};
use cognis_llm::Client;
use cognis_llm::Message;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Person {
    name: String,
    age: u32,
}

struct StaticJsonProvider {
    json: String,
}

#[async_trait]
impl LLMProvider for StaticJsonProvider {
    fn name(&self) -> &str {
        "static_json"
    }
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }
    async fn chat_completion(
        &self,
        _messages: Vec<Message>,
        _opts: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::ai(self.json.clone()),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "fake".into(),
        })
    }
    async fn chat_completion_stream(
        &self,
        _messages: Vec<Message>,
        _opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        Ok(RunnableStream::once(Ok(StreamChunk::default())))
    }
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

#[tokio::test]
async fn structured_client_parses_typed_value() {
    let provider = Arc::new(StaticJsonProvider {
        json: r#"{"name": "Ada", "age": 36}"#.to_string(),
    });
    let client = Client::new(provider);
    let typed = client.with_structured_output::<Person>();
    let out = typed
        .invoke(
            vec![Message::human("introduce yourself")],
            RunnableConfig::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        out,
        Person {
            name: "Ada".into(),
            age: 36
        }
    );
}

#[tokio::test]
async fn structured_client_strips_code_fence() {
    let provider = Arc::new(StaticJsonProvider {
        json: "```json\n{\"name\":\"Bob\",\"age\":7}\n```".to_string(),
    });
    let client = Client::new(provider);
    let typed = client.with_structured_output::<Person>();
    let out = typed
        .invoke(vec![Message::human("intro")], RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(out.name, "Bob");
}

#[tokio::test]
async fn structured_client_errors_on_invalid_json() {
    let provider = Arc::new(StaticJsonProvider {
        json: "not json".into(),
    });
    let client = Client::new(provider);
    let typed = client.with_structured_output::<Person>();
    let err = typed
        .invoke(vec![Message::human("intro")], RunnableConfig::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("json parse"));
}
