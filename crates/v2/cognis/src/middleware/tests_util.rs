//! Shared test helpers for middleware tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use cognis2_core::{Message, Result, RunnableStream};
use cognis2_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis2_llm::provider::{LLMProvider, Provider};

use super::{MiddlewareCtx, Next};

/// Build a canned `ChatResponse` with `text` as the assistant content.
pub(crate) fn ok_resp(text: impl Into<String>) -> ChatResponse {
    ChatResponse {
        message: Message::ai(text),
        usage: None,
        finish_reason: "stop".into(),
        model: "test".into(),
    }
}

/// `Next` that always returns the same canned response.
pub(crate) struct FixedNext(pub ChatResponse);

#[async_trait]
impl Next for FixedNext {
    async fn invoke(&self, _ctx: MiddlewareCtx) -> Result<ChatResponse> {
        Ok(self.0.clone())
    }
}

/// `Next` that records every received ctx then returns a canned response.
pub(crate) struct RecordingNext {
    pub seen: Mutex<Vec<MiddlewareCtx>>,
    pub response: ChatResponse,
}

impl RecordingNext {
    pub(crate) fn new(response: ChatResponse) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            response,
        }
    }
}

#[async_trait]
impl Next for RecordingNext {
    async fn invoke(&self, ctx: MiddlewareCtx) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(ctx);
        Ok(self.response.clone())
    }
}

/// Provider whose response per call is determined by the user-supplied closure.
/// `closure(call_index)` returns either the assistant's text or an error.
pub(crate) struct FlakyProvider {
    counter: AtomicUsize,
    inner: Mutex<Box<dyn Fn(usize) -> Result<String> + Send + Sync>>,
}

pub(crate) fn make_flaky_provider<F>(f: F) -> Arc<FlakyProvider>
where
    F: Fn(usize) -> Result<String> + Send + Sync + 'static,
{
    Arc::new(FlakyProvider {
        counter: AtomicUsize::new(0),
        inner: Mutex::new(Box::new(f)),
    })
}

#[async_trait]
impl LLMProvider for FlakyProvider {
    fn name(&self) -> &str {
        "flaky"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }
    async fn chat_completion(
        &self,
        _messages: Vec<Message>,
        _opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let i = self.counter.fetch_add(1, Ordering::SeqCst);
        let f = self.inner.lock().unwrap();
        let text = f(i)?;
        Ok(ChatResponse {
            message: Message::ai(text),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "flaky".into(),
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

/// Provider that captures every received `(messages, opts)` and returns a
/// canned response.
pub(crate) struct RecordingProvider {
    pub received: Mutex<Vec<(Vec<Message>, ChatOptions)>>,
    pub response: Mutex<String>,
}

pub(crate) fn make_recording_provider(
    initial_response: impl Into<String>,
) -> Arc<RecordingProvider> {
    Arc::new(RecordingProvider {
        received: Mutex::new(Vec::new()),
        response: Mutex::new(initial_response.into()),
    })
}

#[async_trait]
impl LLMProvider for RecordingProvider {
    fn name(&self) -> &str {
        "recording"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }
    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.received.lock().unwrap().push((messages, opts));
        Ok(ChatResponse {
            message: Message::ai(self.response.lock().unwrap().clone()),
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "recording".into(),
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
