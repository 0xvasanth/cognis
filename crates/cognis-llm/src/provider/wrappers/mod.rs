//! Provider wrappers — composable [`super::LLMProvider`] implementations
//! that wrap an inner provider with additional behavior.
//!
//! Each wrapper itself implements `LLMProvider`, so they compose with
//! each other and with `Client`. Order matters — the outermost wrapper
//! sees requests first and responses last.
//!
//! The wrappers in this module:
//!
//! - [`circuit_breaker::CircuitBreakerProvider`] — opens a circuit after
//!   N consecutive failures, half-opens after a cooldown.
//! - [`load_balancer::LoadBalancerProvider`] — distributes calls across
//!   a fleet of inner providers with a pluggable strategy.
//! - [`routing::RoutingProvider`] — dispatches to one of N providers
//!   based on a user-supplied predicate.
//! - [`graceful::GracefulDegradationProvider`] — drops unsupported
//!   features (tools, streaming) rather than erroring.
//! - [`interceptor::InterceptorProvider`] — chat-shape before/after/error
//!   hooks for request rewriting and response transformation.
//!
//! Customization:
//! - Implement [`super::LLMProvider`] directly for a fully custom wrapper.
//! - Most wrappers expose pluggable strategy traits ([`load_balancer::LoadBalancingStrategy`],
//!   [`circuit_breaker::FailureClassifier`], [`interceptor::ChatInterceptor`])
//!   so the common knobs are swappable without subclassing.

pub mod circuit_breaker;
pub mod graceful;
pub mod interceptor;
pub mod load_balancer;
pub mod routing;

pub use circuit_breaker::{
    CircuitBreakerProvider, CircuitState, CircuitStats, FailureClassifier, RetryableClassifier,
};
pub use graceful::{Capability, GracefulDegradationProvider};
pub use interceptor::{ChatInterceptor, FnChatInterceptor, InterceptorProvider};
pub use load_balancer::{
    LoadBalancerProvider, LoadBalancingStrategy, RandomStrategy, RoundRobinStrategy,
    WeightedRoundRobinStrategy,
};
pub use routing::{ProviderRoute, RoutingProvider, RoutingStrategy};

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared scripted inner provider for wrapper tests: records which
    //! entry point a wrapper dispatched to, so a test can tell a native
    //! streaming forward from the non-streaming fallback.

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use cognis_core::{AiMessage, CognisError, Result, RunnableStream};

    use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta};
    use crate::provider::{LLMProvider, Provider};
    use crate::tools::ToolDefinition;
    use crate::{Message, ToolCall};

    /// Call log shared between a test and its spies.
    pub(crate) type Calls = Arc<Mutex<Vec<String>>>;

    /// Records `"<tag>:<entry point>"` (plus tool count and last message
    /// where relevant) for every call, then answers with a fixed script.
    pub(crate) struct StreamSpy {
        pub(crate) tag: &'static str,
        pub(crate) calls: Calls,
        /// When true, both streaming entry points fail before yielding.
        pub(crate) fail_streams: bool,
    }

    impl StreamSpy {
        pub(crate) fn new(tag: &'static str, calls: Calls) -> Self {
            Self {
                tag,
                calls,
                fail_streams: false,
            }
        }

        pub(crate) fn failing(tag: &'static str, calls: Calls) -> Self {
            Self {
                tag,
                calls,
                fail_streams: true,
            }
        }

        fn log(&self, entry: String) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("{}:{entry}", self.tag));
        }
    }

    fn last(messages: &[Message]) -> String {
        messages
            .last()
            .map(|m| m.content().to_string())
            .unwrap_or_default()
    }

    /// One tool definition, for calls that need a non-empty tool list.
    pub(crate) fn one_tool() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search".into(),
            description: "find".into(),
            parameters: Some(serde_json::json!({"type": "object"})),
        }]
    }

    #[async_trait]
    impl LLMProvider for StreamSpy {
        fn name(&self) -> &str {
            self.tag
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            self.log(format!("chat last={}", last(&messages)));
            Ok(ChatResponse {
                message: Message::ai("plain"),
                usage: None,
                finish_reason: "stop".into(),
                model: self.tag.into(),
            })
        }
        async fn chat_completion_with_tools(
            &self,
            messages: Vec<Message>,
            tools: Vec<ToolDefinition>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            self.log(format!(
                "chat_with_tools tools={} last={}",
                tools.len(),
                last(&messages)
            ));
            Ok(ChatResponse {
                message: Message::Ai(AiMessage {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "search".into(),
                        arguments: serde_json::json!({"q": "rust"}),
                    }],
                    parts: Vec::new(),
                }),
                usage: None,
                finish_reason: "tool_calls".into(),
                model: self.tag.into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            self.log(format!("stream last={}", last(&messages)));
            if self.fail_streams {
                return Err(CognisError::Internal("scripted stream failure".into()));
            }
            let chunks = ["to", "ken"].map(|t| {
                Ok(StreamChunk {
                    content: t.into(),
                    is_delta: true,
                    ..Default::default()
                })
            });
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn chat_completion_stream_with_tools(
            &self,
            messages: Vec<Message>,
            tools: Vec<ToolDefinition>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            self.log(format!(
                "stream_with_tools tools={} last={}",
                tools.len(),
                last(&messages)
            ));
            if self.fail_streams {
                return Err(CognisError::Internal("scripted stream failure".into()));
            }
            let fragments = [
                (Some("c1"), Some("search"), "{\"q\":"),
                (None, None, "\"rust\"}"),
            ];
            let chunks = fragments.map(|(id, name, args)| {
                Ok(StreamChunk {
                    is_delta: true,
                    tool_calls_delta: vec![ToolCallDelta {
                        index: 0,
                        id: id.map(String::from),
                        name: name.map(String::from),
                        arguments_delta: Some(args.into()),
                    }],
                    ..Default::default()
                })
            });
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    /// Drain a stream, asserting every chunk is a native delta.
    pub(crate) async fn collect_deltas(s: RunnableStream<StreamChunk>) -> Vec<StreamChunk> {
        let chunks = s.collect_into_vec().await.unwrap();
        assert!(
            chunks.iter().all(|c| c.is_delta),
            "expected native deltas, got: {chunks:?}"
        );
        chunks
    }
}
