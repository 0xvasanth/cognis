//! Provider enum + LLMProvider trait. Closed enum, not an open registry —
//! adding a provider means editing the enum.

use std::str::FromStr;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::tools::ToolDefinition;
use crate::Message;

#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;
#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

/// Closed set of supported providers. Anthropic and Azure are declared but
/// stubbed — full impls land in slice 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// OpenAI (chat completions API).
    OpenAI,
    /// Anthropic Claude. Slice-2 stub.
    Anthropic,
    /// Ollama (local models).
    Ollama,
    /// Azure OpenAI. Slice-2 stub.
    Azure,
}

impl Provider {
    /// Default base URL for the provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1/",
            Provider::Anthropic => "https://api.anthropic.com/v1/",
            Provider::Ollama => "http://localhost:11434/api/",
            Provider::Azure => "", // user must provide
        }
    }

    /// Default model name for the provider.
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenAI => "gpt-4o-mini",
            Provider::Anthropic => "claude-3-5-sonnet-20241022",
            Provider::Ollama => "llama3.2",
            Provider::Azure => "",
        }
    }

    /// Whether this provider requires an API key.
    pub fn requires_auth(&self) -> bool {
        !matches!(self, Provider::Ollama)
    }

    /// Whether this provider is implemented in slice 1.
    pub fn is_implemented(&self) -> bool {
        match self {
            Provider::OpenAI => cfg!(feature = "openai"),
            Provider::Ollama => cfg!(feature = "ollama"),
            Provider::Anthropic | Provider::Azure => false,
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Ollama => "ollama",
            Provider::Azure => "azure",
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
            "ollama" => Ok(Provider::Ollama),
            "azure" => Ok(Provider::Azure),
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
        _tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.chat_completion(messages, opts).await
    }

    /// Connectivity probe.
    async fn health_check(&self) -> Result<HealthStatus>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_from_str_aliases() {
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("gpt".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("claude".parse::<Provider>().unwrap(), Provider::Anthropic);
        assert_eq!("OLLAMA".parse::<Provider>().unwrap(), Provider::Ollama);
        assert!("nope".parse::<Provider>().is_err());
    }

    #[test]
    fn provider_metadata() {
        assert!(Provider::OpenAI.requires_auth());
        assert!(!Provider::Ollama.requires_auth());
        assert_eq!(Provider::Ollama.default_base_url(), "http://localhost:11434/api/");
    }
}
