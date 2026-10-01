//! Provider enum + LLMProvider trait. Closed enum, not an open registry —
//! adding a provider means editing the enum.

use std::str::FromStr;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta};
use crate::tools::ToolDefinition;
use crate::Message;

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub mod openrouter;
#[cfg(any(feature = "openai", feature = "azure"))]
mod sse;
pub mod wrappers;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;
#[cfg(feature = "azure")]
pub use azure::AzureProvider;
#[cfg(feature = "google")]
pub use google::GoogleProvider;
#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;
#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;
#[cfg(feature = "openai")]
pub use openrouter::{OpenRouterBuilder, OpenRouterProvider};

/// Closed set of supported providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// OpenAI (chat completions API).
    OpenAI,
    /// Anthropic Claude (Messages API).
    Anthropic,
    /// Google Gemini (generateContent API).
    Google,
    /// Ollama (local models).
    Ollama,
    /// Azure OpenAI (deployment-scoped endpoints).
    Azure,
    /// OpenRouter — OpenAI-compatible wire format, model namespace via
    /// `vendor/model` (e.g. `anthropic/claude-3.5-sonnet`).
    OpenRouter,
}

impl Provider {
    /// Default base URL for the provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1/",
            Provider::Anthropic => "https://api.anthropic.com/v1/",
            Provider::Google => "https://generativelanguage.googleapis.com/v1beta/",
            Provider::Ollama => "http://localhost:11434/api/",
            // Azure is deployment-scoped; users supply the full base URL.
            Provider::Azure => "",
            Provider::OpenRouter => "https://openrouter.ai/api/v1/",
        }
    }

    /// Default model name for the provider.
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenAI => "gpt-4o-mini",
            Provider::Anthropic => "claude-3-5-sonnet-20241022",
            Provider::Google => "gemini-1.5-flash",
            Provider::Ollama => "llama3.2",
            Provider::Azure => "",
            Provider::OpenRouter => "openai/gpt-4o-mini",
        }
    }

    /// Whether this provider requires an API key.
    pub fn requires_auth(&self) -> bool {
        !matches!(self, Provider::Ollama)
    }

    /// Whether this provider's implementation is compiled in.
    pub fn is_implemented(&self) -> bool {
        match self {
            Provider::OpenAI => cfg!(feature = "openai"),
            Provider::Anthropic => cfg!(feature = "anthropic"),
            Provider::Google => cfg!(feature = "google"),
            Provider::Ollama => cfg!(feature = "ollama"),
            Provider::Azure => cfg!(feature = "azure"),
            // OpenRouter rides on the OpenAI provider, so it's available
            // whenever the openai feature is on.
            Provider::OpenRouter => cfg!(feature = "openai"),
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Google => "google",
            Provider::Ollama => "ollama",
            Provider::Azure => "azure",
            Provider::OpenRouter => "openrouter",
        };
        write!(f, "{s}")
    }
}

impl FromStr for Provider {
    type Err = CognisError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "openai" | "gpt" => Ok(Provider::OpenAI),
            "anthropic" | "claude" => Ok(Provider::Anthropic),
            "google" | "gemini" => Ok(Provider::Google),
            "ollama" => Ok(Provider::Ollama),
            "azure" => Ok(Provider::Azure),
            "openrouter" | "open-router" => Ok(Provider::OpenRouter),
            other => Err(CognisError::Configuration(format!(
                "unknown provider `{other}`"
            ))),
        }
    }
}

/// Trait every concrete provider implementation satisfies. The `Client`
/// holds an `Arc<dyn LLMProvider>` and dispatches through it.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Provider name (e.g. "openai").
    fn name(&self) -> &str;

    /// Provider variant.
    fn provider_type(&self) -> Provider;

    /// One-shot chat completion.
    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse>;

    /// Streaming chat completion.
    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>>;

    /// Chat completion with tool definitions. Default falls back to
    /// `chat_completion` (ignores tools); providers that support tool
    /// calling override this.
    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        if !tools.is_empty() {
            tracing::warn!(
                provider = self.name(),
                tool_count = tools.len(),
                "provider does not support tool calling; tools ignored, falling back to chat_completion"
            );
        }
        self.chat_completion(messages, opts).await
    }

    /// Streaming chat completion with tool definitions.
    ///
    /// Default: providers without native streaming tool-calling fall back to
    /// [`LLMProvider::chat_completion_with_tools`] and emit the full result as
    /// a single terminal [`StreamChunk`] (content + fully-formed tool-call
    /// deltas, `is_done = true`). Providers that support it (OpenAI family)
    /// override this to stream real token/tool deltas.
    async fn chat_completion_stream_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        let resp = self
            .chat_completion_with_tools(messages, tools, opts)
            .await?;
        Ok(RunnableStream::once(Ok(terminal_chunk(resp))))
    }

    /// Connectivity probe.
    async fn health_check(&self) -> Result<HealthStatus>;
}

/// Fold a complete [`ChatResponse`] into the single terminal [`StreamChunk`]
/// a non-streaming provider emits: full content, fully-formed tool-call
/// deltas, `is_done = true`.
pub(crate) fn terminal_chunk(resp: ChatResponse) -> StreamChunk {
    let tool_calls_delta = resp
        .message
        .tool_calls()
        .iter()
        .enumerate()
        .map(|(i, tc)| ToolCallDelta {
            index: i as u32,
            id: Some(tc.id.clone()),
            name: Some(tc.name.clone()),
            arguments_delta: Some(tc.arguments.to_string()),
        })
        .collect();
    StreamChunk {
        content: resp.message.content().to_string(),
        is_delta: false,
        is_done: true,
        finish_reason: Some(resp.finish_reason),
        usage: resp.usage,
        tool_calls_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_from_str_aliases() {
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("gpt".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("claude".parse::<Provider>().unwrap(), Provider::Anthropic);
        assert_eq!("gemini".parse::<Provider>().unwrap(), Provider::Google);
        assert_eq!("OLLAMA".parse::<Provider>().unwrap(), Provider::Ollama);
        assert_eq!("azure".parse::<Provider>().unwrap(), Provider::Azure);
        assert!("nope".parse::<Provider>().is_err());
    }

    #[test]
    fn provider_metadata() {
        assert!(Provider::OpenAI.requires_auth());
        assert!(!Provider::Ollama.requires_auth());
        assert_eq!(
            Provider::Ollama.default_base_url(),
            "http://localhost:11434/api/"
        );
    }

    /// Provider implementing ONLY the non-streaming entry points, so
    /// `chat_completion_stream_with_tools` resolves to the trait default.
    struct Fallback {
        message: Message,
    }

    #[async_trait]
    impl LLMProvider for Fallback {
        fn name(&self) -> &str {
            "fallback"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
            unreachable!("with_tools path is used")
        }
        async fn chat_completion_with_tools(
            &self,
            _m: Vec<Message>,
            _t: Vec<ToolDefinition>,
            _o: ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: self.message.clone(),
                usage: Some(crate::chat::Usage {
                    prompt_tokens: 11,
                    completion_tokens: 7,
                    total_tokens: 18,
                }),
                finish_reason: "tool_calls".into(),
                model: "fallback".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unreachable!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn message_with_two_tool_calls() -> Message {
        Message::Ai(cognis_core::AiMessage {
            content: "hi".into(),
            tool_calls: vec![
                crate::ToolCall {
                    id: "c1".into(),
                    name: "search".into(),
                    arguments: serde_json::json!({"q": 1, "nested": {"tags": ["a", "b"]}}),
                },
                crate::ToolCall {
                    id: "c2".into(),
                    name: "lookup".into(),
                    arguments: serde_json::json!({"id": "x\"y"}),
                },
            ],
            parts: Vec::new(),
        })
    }

    async fn fallback_chunks(message: Message) -> Vec<StreamChunk> {
        Fallback { message }
            .chat_completion_stream_with_tools(
                vec![Message::human("x")],
                vec![],
                ChatOptions::default(),
            )
            .await
            .unwrap()
            .collect_into_vec()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn default_stream_with_tools_falls_back_to_one_terminal_chunk() {
        let chunks = fallback_chunks(message_with_two_tool_calls()).await;
        assert_eq!(chunks.len(), 1, "exactly one terminal chunk");
        let chunk = &chunks[0];
        assert_eq!(chunk.content, "hi");
        assert!(chunk.is_done);
        assert!(!chunk.is_delta);
        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
        let usage = chunk.usage.as_ref().expect("usage carried over");
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (11, 7, 18)
        );

        assert_eq!(chunk.tool_calls_delta.len(), 2);
        let first = &chunk.tool_calls_delta[0];
        assert_eq!(first.index, 0);
        assert_eq!(first.id.as_deref(), Some("c1"));
        assert_eq!(first.name.as_deref(), Some("search"));
        assert_eq!(
            first.arguments_delta.as_deref(),
            Some(r#"{"nested":{"tags":["a","b"]},"q":1}"#)
        );
        let second = &chunk.tool_calls_delta[1];
        assert_eq!(second.index, 1);
        assert_eq!(second.id.as_deref(), Some("c2"));
        assert_eq!(second.name.as_deref(), Some("lookup"));
        assert_eq!(second.arguments_delta.as_deref(), Some(r#"{"id":"x\"y"}"#));
    }

    #[tokio::test]
    async fn default_stream_with_tools_chunk_aggregates_back_to_original_message() {
        use crate::streaming::StreamAggregator;

        for original in [
            message_with_two_tool_calls(),
            Message::ai("plain text, no tools"),
            Message::Ai(cognis_core::AiMessage {
                content: String::new(),
                tool_calls: vec![crate::ToolCall {
                    id: "c1".into(),
                    name: "noargs".into(),
                    arguments: serde_json::json!({}),
                }],
                parts: Vec::new(),
            }),
        ] {
            let mut agg = StreamAggregator::new();
            for chunk in fallback_chunks(original.clone()).await {
                agg.push(chunk);
            }
            let out = agg.finalize();
            assert_eq!(out.message, original);
            assert_eq!(out.finish_reason.as_deref(), Some("tool_calls"));
            assert_eq!(out.usage.map(|u| u.total_tokens), Some(18));
        }
    }
}
