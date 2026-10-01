//! OpenRouter provider — OpenAI-compatible wire format with model
//! namespacing (`vendor/model`).
//!
//! Implementation rides on [`super::openai::OpenAIProvider`]; this
//! module is a thin wrapper that:
//!
//! - Defaults `base_url` to `https://openrouter.ai/api/v1/`.
//! - Sets sensible recommended headers (`HTTP-Referer`, `X-Title`) when
//!   the user provides them.
//! - Reports `provider_type() -> Provider::OpenRouter` so observers can
//!   distinguish from a plain OpenAI call against a custom base URL.
//!
//! Customization:
//! - [`OpenRouterBuilder::with_referer`] / [`OpenRouterBuilder::with_title`]
//!   set OpenRouter-specific telemetry headers.
//! - [`OpenRouterBuilder::extra_header`] for any other custom header.

#![cfg(feature = "openai")]

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::openai::{OpenAIBuilder, OpenAIProvider};
use crate::tools::ToolDefinition;
use crate::Message;

use super::{LLMProvider, Provider};

/// OpenRouter provider. Wraps an [`OpenAIProvider`] pointed at
/// `https://openrouter.ai/api/v1/`.
pub struct OpenRouterProvider {
    inner: Arc<OpenAIProvider>,
}

impl std::fmt::Debug for OpenRouterProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterProvider").finish()
    }
}

impl OpenRouterProvider {
    /// Build with API key + default base URL + default model
    /// (`openai/gpt-4o-mini`).
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        OpenRouterBuilder::default().api_key(api_key).build()
    }

    /// Fluent builder.
    pub fn builder() -> OpenRouterBuilder {
        OpenRouterBuilder::default()
    }
}

#[async_trait]
impl LLMProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn provider_type(&self) -> Provider {
        Provider::OpenRouter
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.chat_completion(messages, opts).await
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        self.inner.chat_completion_stream(messages, opts).await
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner
            .chat_completion_with_tools(messages, tools, opts)
            .await
    }

    async fn chat_completion_stream_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        self.inner
            .chat_completion_stream_with_tools(messages, tools, opts)
            .await
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        self.inner.health_check().await
    }
}

/// Fluent builder for [`OpenRouterProvider`].
#[derive(Default)]
pub struct OpenRouterBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
    referer: Option<String>,
    title: Option<String>,
    extra_headers: Vec<(String, String)>,
}

impl OpenRouterBuilder {
    /// API key (required).
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }

    /// Override base URL (default: `https://openrouter.ai/api/v1/`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Default model in OpenRouter `vendor/model` namespacing.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }

    /// `HTTP-Referer` header — OpenRouter uses this for app attribution.
    pub fn with_referer(mut self, r: impl Into<String>) -> Self {
        self.referer = Some(r.into());
        self
    }

    /// `X-Title` header — OpenRouter uses this for app attribution.
    pub fn with_title(mut self, t: impl Into<String>) -> Self {
        self.title = Some(t.into());
        self
    }

    /// Add an arbitrary extra HTTP header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }

    /// Construct.
    pub fn build(self) -> Result<OpenRouterProvider> {
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("OpenRouter: API key required".into()))?;
        let base_url = self
            .base_url
            .unwrap_or_else(|| Provider::OpenRouter.default_base_url().to_string());
        let model = self
            .model
            .unwrap_or_else(|| Provider::OpenRouter.default_model().to_string());

        let mut b = OpenAIBuilder::default()
            .api_key(api_key)
            .base_url(base_url)
            .model(model);
        if let Some(t) = self.timeout_secs {
            b = b.timeout_secs(t);
        }
        if let Some(r) = self.referer {
            b = b.extra_header("HTTP-Referer", r);
        }
        if let Some(t) = self.title {
            b = b.extra_header("X-Title", t);
        }
        for (k, v) in self.extra_headers {
            b = b.extra_header(k, v);
        }

        Ok(OpenRouterProvider {
            inner: Arc::new(b.build()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_api_key() {
        let res = OpenRouterBuilder::default().build();
        assert!(res.is_err());
    }

    #[test]
    fn provider_type_reports_openrouter() {
        let p = OpenRouterProvider::new("sk-test").unwrap();
        assert_eq!(p.provider_type(), Provider::OpenRouter);
        assert_eq!(p.name(), "openrouter");
    }

    #[tokio::test]
    async fn stream_with_tools_delegates_to_inner_native_stream() {
        use crate::provider::sse::tests::{serve_sse_once, tool_call_writes};
        use crate::streaming::StreamAggregator;

        let (base, server) = serve_sse_once(tool_call_writes()).await;
        let p = OpenRouterBuilder::default()
            .api_key("sk-test")
            .base_url(base)
            .build()
            .unwrap();
        let chunks = p
            .chat_completion_stream_with_tools(
                vec![Message::human("weather?")],
                vec![ToolDefinition {
                    name: "weather".into(),
                    description: "look up weather".into(),
                    parameters: Some(serde_json::json!({"type": "object"})),
                }],
                ChatOptions::default(),
            )
            .await
            .unwrap()
            .collect_into_vec()
            .await
            .unwrap();

        // The trait's fallback would have sent `stream: false` and produced
        // one non-delta chunk; real deltas prove the inner override ran.
        let body = server.await.unwrap();
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(body["tools"][0]["function"]["name"], "weather");
        assert_eq!(chunks.len(), 4, "got: {chunks:?}");
        assert!(chunks.iter().all(|c| c.is_delta), "got: {chunks:?}");

        let mut agg = StreamAggregator::new();
        for c in chunks {
            agg.push(c);
        }
        let out = agg.finalize();
        assert_eq!(
            out.message.tool_calls()[0].arguments,
            serde_json::json!({"city": "San Francisco"})
        );
    }

    #[test]
    fn referer_and_title_set_extras() {
        // Surface check: builder should not error when these are set.
        let p = OpenRouterBuilder::default()
            .api_key("sk-test")
            .with_referer("https://example.com")
            .with_title("MyApp")
            .extra_header("X-Custom", "yes")
            .build();
        assert!(p.is_ok());
    }
}
