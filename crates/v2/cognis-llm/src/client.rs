//! User-facing `Client`. Holds an `Arc<dyn LLMProvider>` and dispatches
//! through it. Implements `Runnable<Vec<Message>, Message>` so it composes
//! inside graphs.

use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{CognisError, Result, Runnable, RunnableConfig, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::tools::ToolDefinition;
use crate::Message;

/// Top-level LLM client. Cheap to clone.
#[derive(Clone)]
pub struct Client {
    provider: Arc<dyn LLMProvider>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").field("provider", &self.provider.name()).finish()
    }
}

impl Client {
    /// Wrap any `LLMProvider`.
    pub fn new(provider: Arc<dyn LLMProvider>) -> Self {
        Self { provider }
    }

    /// Fluent builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Build from env vars. Provider-namespaced with fallback:
    /// `COGNIS_OPENAI_API_KEY` overrides `COGNIS_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let provider_str = std::env::var("COGNIS_PROVIDER")
            .map_err(|_| CognisError::Configuration("COGNIS_PROVIDER not set".into()))?;
        let provider = Provider::from_str(&provider_str)?;
        let mut b = Self::builder().provider(provider);

        let key = std::env::var(format!(
            "COGNIS_{}_API_KEY",
            provider.to_string().to_uppercase()
        ))
        .or_else(|_| std::env::var("COGNIS_API_KEY"))
        .ok();
        if let Some(k) = key {
            b = b.api_key(k);
        }

        let url = std::env::var(format!(
            "COGNIS_{}_BASE_URL",
            provider.to_string().to_uppercase()
        ))
        .or_else(|_| std::env::var("COGNIS_BASE_URL"))
        .ok();
        if let Some(u) = url {
            b = b.base_url(u);
        }

        let model = std::env::var(format!(
            "COGNIS_{}_MODEL",
            provider.to_string().to_uppercase()
        ))
        .or_else(|_| std::env::var("COGNIS_MODEL"))
        .ok();
        if let Some(m) = model {
            b = b.model(m);
        }

        b.build()
    }

    /// One-shot chat completion (no tools).
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<Message> {
        Ok(self.provider.chat_completion(messages, ChatOptions::default()).await?.message)
    }

    /// Streaming chat completion.
    pub async fn stream(&self, messages: Vec<Message>) -> Result<RunnableStream<StreamChunk>> {
        self.provider.chat_completion_stream(messages, ChatOptions::default()).await
    }

    /// Chat completion with tool definitions.
    pub async fn invoke_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[Arc<dyn crate::tools::Tool>],
    ) -> Result<Message> {
        let defs: Vec<ToolDefinition> =
            tools.iter().map(|t| ToolDefinition::from_tool(t.as_ref())).collect();
        Ok(self
            .provider
            .chat_completion_with_tools(messages, defs, ChatOptions::default())
            .await?
            .message)
    }

    /// Provider-level full chat completion (with all options).
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.provider.chat_completion(messages, opts).await
    }

    /// Underlying provider.
    pub fn provider(&self) -> &Arc<dyn LLMProvider> {
        &self.provider
    }
}

#[async_trait]
impl Runnable<Vec<Message>, Message> for Client {
    async fn invoke(&self, input: Vec<Message>, _: RunnableConfig) -> Result<Message> {
        Client::invoke(self, input).await
    }
    fn name(&self) -> &str {
        "Client"
    }
}

/// Fluent builder for `Client`.
#[derive(Default)]
pub struct ClientBuilder {
    provider: Option<Provider>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
    organization: Option<String>,
}

impl ClientBuilder {
    /// Provider variant.
    pub fn provider(mut self, p: Provider) -> Self {
        self.provider = Some(p);
        self
    }
    /// API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Base URL override.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    /// Timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// OpenAI organization (only used for OpenAI provider).
    pub fn organization(mut self, o: impl Into<String>) -> Self {
        self.organization = Some(o.into());
        self
    }
    /// Construct the Client.
    pub fn build(self) -> Result<Client> {
        let provider = self
            .provider
            .ok_or_else(|| CognisError::Configuration("Client: provider required".into()))?;
        let arc_provider: Arc<dyn LLMProvider> = match provider {
            #[cfg(feature = "openai")]
            Provider::OpenAI => {
                use crate::provider::openai::OpenAIBuilder;
                let mut b = OpenAIBuilder::default();
                if let Some(k) = self.api_key {
                    b = b.api_key(k);
                }
                if let Some(u) = self.base_url {
                    b = b.base_url(u);
                }
                if let Some(m) = self.model {
                    b = b.model(m);
                }
                if let Some(t) = self.timeout_secs {
                    b = b.timeout_secs(t);
                }
                if let Some(o) = self.organization {
                    b = b.organization(o);
                }
                Arc::new(b.build()?)
            }
            #[cfg(feature = "ollama")]
            Provider::Ollama => {
                use crate::provider::ollama::OllamaBuilder;
                let mut b = OllamaBuilder::default();
                if let Some(u) = self.base_url {
                    b = b.base_url(u);
                }
                if let Some(m) = self.model {
                    b = b.model(m);
                }
                if let Some(t) = self.timeout_secs {
                    b = b.timeout_secs(t);
                }
                Arc::new(b.build()?)
            }
            other => {
                return Err(CognisError::Configuration(format!(
                    "provider `{other}` not implemented in slice 1 (anthropic + azure are slice 2)"
                )))
            }
        };
        Ok(Client { provider: arc_provider })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "openai")]
    #[test]
    fn openai_builder_round_trip() {
        let c = ClientBuilder::default()
            .provider(Provider::OpenAI)
            .api_key("sk-test")
            .model("gpt-4o")
            .build()
            .unwrap();
        assert_eq!(c.provider().name(), "openai");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_builder_round_trip() {
        let c = ClientBuilder::default()
            .provider(Provider::Ollama)
            .model("llama3.2")
            .build()
            .unwrap();
        assert_eq!(c.provider().name(), "ollama");
    }

    #[test]
    fn missing_provider_errors() {
        let err = ClientBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("provider required"));
    }

    #[test]
    fn unimplemented_provider_errors() {
        let err = ClientBuilder::default()
            .provider(Provider::Anthropic)
            .api_key("x")
            .build()
            .unwrap_err();
        assert!(format!("{err}").contains("slice 1"));
    }
}
