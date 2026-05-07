//! Graceful-degradation wrapper — drops or downgrades unsupported
//! features rather than erroring.
//!
//! Concretely, when the inner provider doesn't support a feature, the
//! wrapper either silently strips it from the request (e.g. removes
//! tool definitions) or substitutes a fallback path (e.g. emits a
//! single-chunk stream from a non-streaming response).
//!
//! Customization:
//! - [`Capability`] — which features the inner provider lacks. Toggle
//!   per wrapper instance.
//! - [`GracefulDegradationProvider::with_warn`] — opt out of the default
//!   `tracing::warn!` so silent downgrade is truly silent (useful in
//!   batch workloads where the warning floods logs).

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::streaming::Aggregated;
use crate::tools::ToolDefinition;
use crate::Message;

/// Features that may not be supported by every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Tool/function calling.
    Tools,
    /// Streaming chunks.
    Streaming,
    /// Multimodal (image/audio) input.
    Multimodal,
    /// JSON-mode / structured output.
    StructuredOutput,
}

/// Graceful-degradation wrapper.
pub struct GracefulDegradationProvider {
    inner: Arc<dyn LLMProvider>,
    /// Capabilities the *inner* provider does NOT support. The wrapper
    /// silently downgrades requests touching these.
    missing: Vec<Capability>,
    warn: bool,
    name: String,
}

impl GracefulDegradationProvider {
    /// Wrap `inner` with no missing capabilities (no-op until configured).
    pub fn new(inner: Arc<dyn LLMProvider>) -> Self {
        let name = inner.name().to_string();
        Self {
            inner,
            missing: Vec::new(),
            warn: true,
            name,
        }
    }

    /// Mark `cap` as missing on the inner provider.
    pub fn missing(mut self, cap: Capability) -> Self {
        if !self.missing.contains(&cap) {
            self.missing.push(cap);
        }
        self
    }

    /// Mark a list of missing capabilities.
    pub fn missing_many<I: IntoIterator<Item = Capability>>(mut self, caps: I) -> Self {
        for c in caps {
            self = self.missing(c);
        }
        self
    }

    /// Whether to emit `tracing::warn!` on each downgrade. Default `true`.
    pub fn with_warn(mut self, on: bool) -> Self {
        self.warn = on;
        self
    }

    fn lacks(&self, cap: Capability) -> bool {
        self.missing.contains(&cap)
    }

    fn warn_drop(&self, what: &str) {
        if self.warn {
            tracing::warn!(provider = %self.name, "dropping unsupported feature: {what}");
        }
    }
}

#[async_trait]
impl LLMProvider for GracefulDegradationProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn provider_type(&self) -> Provider {
        self.inner.provider_type()
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let messages = if self.lacks(Capability::Multimodal) {
            // Strip parts; keep text content.
            messages
                .into_iter()
                .map(|m| match m {
                    Message::Human(h) if !h.parts.is_empty() => {
                        self.warn_drop("multimodal parts on Human message");
                        Message::human(h.content)
                    }
                    Message::Ai(a) if !a.parts.is_empty() => {
                        self.warn_drop("multimodal parts on Ai message");
                        Message::Ai(crate::AiMessage {
                            content: a.content,
                            tool_calls: a.tool_calls,
                            parts: Vec::new(),
                        })
                    }
                    other => other,
                })
                .collect()
        } else {
            messages
        };
        self.inner.chat_completion(messages, opts).await
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        if self.lacks(Capability::Streaming) {
            self.warn_drop("streaming → emitting single-chunk synthetic stream");
            // Issue a one-shot chat completion and synthesize a single chunk.
            let r = self.chat_completion(messages, opts).await?;
            let chunk = StreamChunk {
                content: r.message.content().to_string(),
                is_delta: false,
                is_done: true,
                finish_reason: Some(r.finish_reason),
                usage: r.usage,
                tool_calls_delta: Vec::new(),
            };
            Ok(RunnableStream::once(Ok(chunk)))
        } else {
            self.inner.chat_completion_stream(messages, opts).await
        }
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        if self.lacks(Capability::Tools) && !tools.is_empty() {
            self.warn_drop("tools → falling back to chat without tools");
            return self.inner.chat_completion(messages, opts).await;
        }
        self.inner
            .chat_completion_with_tools(messages, tools, opts)
            .await
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        self.inner.health_check().await
    }
}

// Surface `Aggregated` here so the type is reachable from a downstream
// caller via this module — used in synthesized streaming downgrade.
#[allow(dead_code)]
fn _aggregated_export_check(_a: Aggregated) {}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis2_core::CognisError;

    struct Inner {
        rejected_tools: std::sync::Mutex<bool>,
    }
    #[async_trait]
    impl LLMProvider for Inner {
        fn name(&self) -> &str {
            "inner"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            // Verify multimodal parts were stripped.
            for m in &messages {
                assert!(m.parts().is_empty(), "parts should have been stripped");
            }
            Ok(ChatResponse {
                message: Message::ai("inner-ok"),
                usage: None,
                finish_reason: "stop".into(),
                model: "inner".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            Err(CognisError::Internal(
                "inner does not support streaming (test)".into(),
            ))
        }
        async fn chat_completion_with_tools(
            &self,
            _: Vec<Message>,
            tools: Vec<ToolDefinition>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            // If we got here with tools, the wrapper failed to drop them.
            *self.rejected_tools.lock().unwrap() = !tools.is_empty();
            Err(CognisError::Configuration(
                "inner does not support tools (test)".into(),
            ))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn drops_multimodal_parts_for_inner() {
        let inner = Arc::new(Inner {
            rejected_tools: Default::default(),
        });
        let p = GracefulDegradationProvider::new(inner)
            .missing(Capability::Multimodal)
            .with_warn(false);
        let msg = Message::human_with_parts(
            "hello",
            vec![cognis2_core::ContentPart::Text {
                text: "ignored".into(),
            }],
        );
        let res = p
            .chat_completion(vec![msg], ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(res.message.content(), "inner-ok");
    }

    #[tokio::test]
    async fn falls_back_to_chat_when_tools_unsupported() {
        let inner = Arc::new(Inner {
            rejected_tools: Default::default(),
        });
        let inner_clone = inner.clone();
        let p = GracefulDegradationProvider::new(inner)
            .missing(Capability::Tools)
            .with_warn(false);
        let res = p
            .chat_completion_with_tools(
                vec![Message::human("x")],
                vec![ToolDefinition {
                    name: "noop".into(),
                    description: "noop".into(),
                    parameters: Some(serde_json::json!({})),
                }],
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(res.message.content(), "inner-ok");
        // The inner's chat_completion_with_tools must NOT have been called.
        assert!(!*inner_clone.rejected_tools.lock().unwrap());
    }

    #[tokio::test]
    async fn synthesizes_single_chunk_stream_when_streaming_unsupported() {
        use futures::StreamExt;
        let inner = Arc::new(Inner {
            rejected_tools: Default::default(),
        });
        let p = GracefulDegradationProvider::new(inner)
            .missing(Capability::Streaming)
            .with_warn(false);
        let mut s = p
            .chat_completion_stream(vec![Message::human("x")], ChatOptions::default())
            .await
            .unwrap();
        let first = s.next().await.unwrap().unwrap();
        assert_eq!(first.content, "inner-ok");
    }

    #[tokio::test]
    async fn no_missing_capabilities_passes_through() {
        let inner = Arc::new(Inner {
            rejected_tools: Default::default(),
        });
        let p = GracefulDegradationProvider::new(inner);
        let res = p.chat_completion(vec![], ChatOptions::default()).await;
        assert!(res.is_ok());
    }
}
