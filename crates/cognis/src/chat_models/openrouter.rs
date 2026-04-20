//! Convenience builder for OpenRouter — a thin wrapper over `ChatOpenAI`
//! that defaults the base URL and exposes named setters for OpenRouter's
//! attribution headers (`HTTP-Referer` via [`ChatOpenRouterBuilder::app_url`],
//! `X-Title` via [`ChatOpenRouterBuilder::app_name`]).
//!
//! [`ChatOpenRouterBuilder::build`] returns [`ChatOpenAI`]. All HTTP,
//! retry, streaming, and tool-calling behavior is the same as
//! `ChatOpenAI` — this module is purely about discoverability and the
//! default endpoint.
//!
//! [#10](https://github.com/0xvasanth/cognis/issues/10).

use std::collections::HashMap;

use cognis_core::error::Result;

use crate::chat_models::openai::{ChatOpenAI, ChatOpenAIBuilder};

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api";

/// Namespace type — entry point for the OpenRouter builder.
///
/// # Example
///
/// ```no_run
/// use cognis::chat_models::openrouter::ChatOpenRouter;
///
/// let model = ChatOpenRouter::builder()
///     .model("meta-llama/llama-3.3-70b-instruct:free")
///     .api_key("sk-or-...")
///     .app_name("my-assistant")
///     .app_url("https://example.com")
///     .build()
///     .unwrap();
/// ```
pub struct ChatOpenRouter;

impl ChatOpenRouter {
    /// Start building a [`ChatOpenAI`] pre-configured for OpenRouter.
    pub fn builder() -> ChatOpenRouterBuilder {
        ChatOpenRouterBuilder::new()
    }
}

/// Builder that configures [`ChatOpenAI`] for OpenRouter.
///
/// Wraps [`ChatOpenAIBuilder`] and pre-sets the OpenRouter base URL
/// (`https://openrouter.ai/api`). Adds
/// [`app_name`](Self::app_name) / [`app_url`](Self::app_url) setters
/// for OpenRouter's attribution headers (`X-Title` and `HTTP-Referer`).
/// Common model parameters are delegated through to the inner builder.
pub struct ChatOpenRouterBuilder {
    inner: ChatOpenAIBuilder,
}

impl ChatOpenRouterBuilder {
    fn new() -> Self {
        Self {
            inner: ChatOpenAI::builder().base_url(OPENROUTER_BASE_URL),
        }
    }

    /// Set the OpenRouter model identifier, e.g.
    /// `"meta-llama/llama-3.3-70b-instruct:free"`.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    /// Set the OpenRouter API key. Falls back to `OPENAI_API_KEY` env var
    /// if not provided (delegated from [`ChatOpenAIBuilder::api_key`]).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.inner = self.inner.api_key(key);
        self
    }

    /// Set the application name — sent as the `X-Title` HTTP header.
    ///
    /// OpenRouter's public app leaderboard uses this header; set it if
    /// you want your app to appear there.
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.extra_header("X-Title", name);
        self
    }

    /// Set the application URL — sent as the `HTTP-Referer` HTTP header.
    ///
    /// OpenRouter uses this for attribution and its app leaderboard.
    pub fn app_url(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.extra_header("HTTP-Referer", url);
        self
    }

    /// Override the default base URL (`https://openrouter.ai/api`).
    ///
    /// Rarely needed — provided for staging / proxy scenarios.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.base_url(base_url);
        self
    }

    /// Sampling temperature (0.0 - 2.0).
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    /// Maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    /// Maximum retry attempts on transient errors.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.inner = self.inner.max_retries(max_retries);
        self
    }

    /// Default streaming mode. Per-call streaming still uses `stream()`.
    pub fn streaming(mut self, streaming: bool) -> Self {
        self.inner = self.inner.streaming(streaming);
        self
    }

    /// Add an additional custom header beyond `X-Title` / `HTTP-Referer`.
    ///
    /// Delegates to [`ChatOpenAIBuilder::extra_header`], which filters
    /// reserved headers (`Authorization`, `Content-Type`,
    /// `OpenAI-Organization`).
    pub fn extra_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner = self.inner.extra_header(key, value);
        self
    }

    /// Replace all extra headers. See
    /// [`ChatOpenAIBuilder::extra_headers`] for filtering semantics.
    pub fn extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.extra_headers(headers);
        self
    }

    /// Finish building — returns a configured [`ChatOpenAI`].
    pub fn build(self) -> Result<ChatOpenAI> {
        self.inner.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_base_url_to_openrouter() {
        let model = ChatOpenRouter::builder()
            .model("meta-llama/llama-3.3-70b-instruct:free")
            .api_key("sk-or-test")
            .build()
            .unwrap();

        assert_eq!(model.base_url, OPENROUTER_BASE_URL);
    }

    #[test]
    fn app_name_sets_x_title_header() {
        let model = ChatOpenRouter::builder()
            .model("gpt-4o")
            .api_key("sk-or-test")
            .app_name("my-assistant")
            .build()
            .unwrap();

        assert_eq!(
            model.extra_headers.get("X-Title").map(String::as_str),
            Some("my-assistant"),
        );
    }

    #[test]
    fn app_url_sets_http_referer_header() {
        let model = ChatOpenRouter::builder()
            .model("gpt-4o")
            .api_key("sk-or-test")
            .app_url("https://example.com")
            .build()
            .unwrap();

        assert_eq!(
            model.extra_headers.get("HTTP-Referer").map(String::as_str),
            Some("https://example.com"),
        );
    }

    #[test]
    fn app_name_and_app_url_coexist_with_extra_header() {
        let model = ChatOpenRouter::builder()
            .model("gpt-4o")
            .api_key("sk-or-test")
            .app_name("assistant")
            .app_url("https://example.com")
            .extra_header("X-Custom", "value")
            .build()
            .unwrap();

        assert_eq!(model.extra_headers.len(), 3);
        assert_eq!(
            model.extra_headers.get("X-Title").map(String::as_str),
            Some("assistant"),
        );
        assert_eq!(
            model.extra_headers.get("HTTP-Referer").map(String::as_str),
            Some("https://example.com"),
        );
        assert_eq!(
            model.extra_headers.get("X-Custom").map(String::as_str),
            Some("value"),
        );
    }

    #[test]
    fn base_url_override_is_respected() {
        let model = ChatOpenRouter::builder()
            .model("gpt-4o")
            .api_key("sk-or-test")
            .base_url("https://staging.openrouter.example")
            .build()
            .unwrap();

        assert_eq!(model.base_url, "https://staging.openrouter.example");
    }
}
