//! Web search tool with a pluggable provider.
//!
//! Web search is heterogeneous — every provider (Tavily / Brave / Bing /
//! SerpAPI / DuckDuckGo) has a different API shape, key requirement, and
//! result format. The tool ships a [`WebSearchProvider`] trait + one
//! reference implementation ([`TavilyProvider`]) so users can swap
//! providers without forking.
//!
//! Customization:
//! - [`WebSearchTool::new`] takes any `WebSearchProvider`.
//! - Closures matching `Fn(WebSearchInput) -> Future<Output = Result<…>>`
//!   can be used inline via the blanket impl.
//! - [`TavilyProvider`] — reference impl. Free tier available;
//!   sign up at <https://tavily.com>.
//!
//! Feature-gated under `tools-http` because all providers need
//! `reqwest`.

#![cfg(feature = "tools-http")]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use reqwest::header::{HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

const TAVILY_DEFAULT_BASE: &str = "https://api.tavily.com";

/// Tool input — what the LLM sends.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct WebSearchInput {
    /// Search query.
    pub query: String,
    /// Maximum number of results (default: 5).
    #[serde(default)]
    pub max_results: Option<usize>,
}

/// One result from any provider — providers normalize into this shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// Page title.
    pub title: String,
    /// Page URL.
    pub url: String,
    /// Snippet / preview content.
    pub snippet: String,
    /// Provider-supplied relevance score, if available (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Pluggable provider trait. Implementors translate `WebSearchInput`
/// into the provider's native request and normalize results into
/// [`WebSearchResult`] so the tool's output shape stays stable.
#[async_trait]
pub trait WebSearchProvider: Send + Sync {
    /// Run a search.
    async fn search(&self, input: WebSearchInput) -> Result<Vec<WebSearchResult>>;

    /// Friendly name for diagnostics (e.g. `"tavily"`, `"brave"`).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Closure-based provider for inline / testing use.
#[async_trait]
impl<F, Fut> WebSearchProvider for F
where
    F: Fn(WebSearchInput) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Vec<WebSearchResult>>> + Send,
{
    async fn search(&self, input: WebSearchInput) -> Result<Vec<WebSearchResult>> {
        (self)(input).await
    }
}

// ---------------------------------------------------------------------------
// TavilyProvider — reference impl.
// ---------------------------------------------------------------------------

/// Reference implementation of [`WebSearchProvider`] for Tavily.
///
/// Tavily exposes a single POST endpoint at `/search`. The
/// `search_depth` knob trades latency for quality (`"basic"` /
/// `"advanced"`). Set `include_answer: true` to have Tavily synthesize a
/// short answer alongside the raw results — currently dropped from the
/// normalized output but available for users that want to call the
/// underlying client directly.
pub struct TavilyProvider {
    api_key: String,
    base_url: String,
    search_depth: String,
    include_domains: Vec<String>,
    exclude_domains: Vec<String>,
    http: reqwest::Client,
}

impl std::fmt::Debug for TavilyProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TavilyProvider")
            .field("base_url", &self.base_url)
            .field("search_depth", &self.search_depth)
            .finish_non_exhaustive()
    }
}

impl TavilyProvider {
    /// New with API key + defaults (`search_depth = "basic"`).
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::builder().api_key(api_key).build()
    }

    /// Fluent builder.
    pub fn builder() -> TavilyProviderBuilder {
        TavilyProviderBuilder::default()
    }
}

#[async_trait]
impl WebSearchProvider for TavilyProvider {
    async fn search(&self, input: WebSearchInput) -> Result<Vec<WebSearchResult>> {
        #[derive(Serialize)]
        struct Body<'a> {
            api_key: &'a str,
            query: &'a str,
            search_depth: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_results: Option<usize>,
            #[serde(skip_serializing_if = "<[String]>::is_empty")]
            include_domains: &'a [String],
            #[serde(skip_serializing_if = "<[String]>::is_empty")]
            exclude_domains: &'a [String],
        }
        #[derive(Deserialize)]
        struct Resp {
            #[serde(default)]
            results: Vec<Hit>,
        }
        #[derive(Deserialize)]
        struct Hit {
            #[serde(default)]
            title: String,
            #[serde(default)]
            url: String,
            #[serde(default)]
            content: String,
            #[serde(default)]
            score: Option<f32>,
        }
        let url = if self.base_url.ends_with('/') {
            format!("{}search", self.base_url)
        } else {
            format!("{}/search", self.base_url)
        };
        let body = Body {
            api_key: &self.api_key,
            query: &input.query,
            search_depth: &self.search_depth,
            max_results: input.max_results,
            include_domains: &self.include_domains,
            exclude_domains: &self.exclude_domains,
        };
        let resp = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("tavily search: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "tavily search: HTTP {s}: {t}"
            )));
        }
        let parsed: Resp = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("tavily json: {e}")))?;
        Ok(parsed
            .results
            .into_iter()
            .map(|h| WebSearchResult {
                title: h.title,
                url: h.url,
                snippet: h.content,
                score: h.score,
            })
            .collect())
    }

    fn name(&self) -> &str {
        "tavily"
    }
}

/// Fluent builder for [`TavilyProvider`].
#[derive(Default)]
pub struct TavilyProviderBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    search_depth: Option<String>,
    include_domains: Vec<String>,
    exclude_domains: Vec<String>,
    timeout_secs: Option<u64>,
    http: Option<reqwest::Client>,
}

impl TavilyProviderBuilder {
    /// Set the API key (required).
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Override the base URL (default `https://api.tavily.com`).
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// `"basic"` (faster, default) or `"advanced"` (deeper crawl).
    pub fn search_depth(mut self, d: impl Into<String>) -> Self {
        self.search_depth = Some(d.into());
        self
    }
    /// Restrict results to these domains.
    pub fn include_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_domains = domains.into_iter().map(Into::into).collect();
        self
    }
    /// Drop results from these domains.
    pub fn exclude_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_domains = domains.into_iter().map(Into::into).collect();
        self
    }
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Override the HTTP client.
    pub fn http_client(mut self, c: reqwest::Client) -> Self {
        self.http = Some(c);
        self
    }
    /// Build.
    pub fn build(self) -> Result<TavilyProvider> {
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("Tavily: API key required".into()))?;
        let http = match self.http {
            Some(c) => c,
            None => {
                let mut b = reqwest::ClientBuilder::new();
                if let Some(t) = self.timeout_secs {
                    b = b.timeout(Duration::from_secs(t));
                }
                b.build()
                    .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?
            }
        };
        Ok(TavilyProvider {
            api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| TAVILY_DEFAULT_BASE.to_string()),
            search_depth: self.search_depth.unwrap_or_else(|| "basic".to_string()),
            include_domains: self.include_domains,
            exclude_domains: self.exclude_domains,
            http,
        })
    }
}

// `AUTHORIZATION` is imported above for users who want to wrap the
// Tavily client with header-based auth in tests; suppress the
// unused-import lint for the default code path.
#[allow(dead_code)]
fn _silence_authorization_import(_: HeaderValue) {
    let _ = AUTHORIZATION;
}

// ---------------------------------------------------------------------------
// WebSearchTool — wraps a provider as a Tool.
// ---------------------------------------------------------------------------

/// Tool exposing a [`WebSearchProvider`] to the LLM.
pub struct WebSearchTool {
    provider: Arc<dyn WebSearchProvider>,
    name: String,
    description: String,
    default_max_results: usize,
}

impl WebSearchTool {
    /// Wrap a provider with the default tool name `"web_search"`.
    pub fn new<P: WebSearchProvider + 'static>(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
            name: "web_search".into(),
            description: "Search the public web. Returns a list of \
                {title, url, snippet, score} results."
                .into(),
            default_max_results: 5,
        }
    }

    /// Override the registered tool name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the description shown to the LLM.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Override the default `max_results` cap when the LLM doesn't specify.
    pub fn with_default_max_results(mut self, n: usize) -> Self {
        self.default_max_results = n.max(1);
        self
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(WebSearchInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let mut parsed: WebSearchInput =
            serde_json::from_value(input.into_json()).map_err(|e| {
                CognisError::ToolValidationError(format!("web_search: invalid args: {e}"))
            })?;
        if parsed.max_results.is_none() {
            parsed.max_results = Some(self.default_max_results);
        }
        let results = self.provider.search(parsed).await?;
        Ok(ToolOutput::Content(serde_json::json!({
            "results": results,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[test]
    fn tavily_builder_requires_api_key() {
        let err = TavilyProviderBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn tavily_builder_sets_defaults() {
        let p = TavilyProvider::new("sk-test").unwrap();
        assert_eq!(p.search_depth, "basic");
        assert_eq!(p.base_url, TAVILY_DEFAULT_BASE);
    }

    #[test]
    fn tavily_builder_custom_domains() {
        let p = TavilyProviderBuilder::default()
            .api_key("sk-test")
            .include_domains(["docs.rs", "rust-lang.org"])
            .exclude_domains(["spam.example.com"])
            .build()
            .unwrap();
        assert_eq!(p.include_domains.len(), 2);
        assert_eq!(p.exclude_domains.len(), 1);
    }

    #[tokio::test]
    async fn closure_provider_returns_canned_results() {
        // Plug a closure provider; verify the tool surfaces results
        // and applies the default max_results when absent.
        let captured: Arc<Mutex<Option<WebSearchInput>>> = Arc::new(Mutex::new(None));
        let c2 = captured.clone();
        let provider = move |input: WebSearchInput| {
            let c3 = c2.clone();
            async move {
                *c3.lock().unwrap() = Some(input);
                Ok(vec![WebSearchResult {
                    title: "Rust".into(),
                    url: "https://rust-lang.org".into(),
                    snippet: "Rust programming language".into(),
                    score: Some(0.95),
                }])
            }
        };
        let tool = WebSearchTool::new(provider).with_default_max_results(3);
        let mut m = HashMap::new();
        m.insert("query".to_string(), serde_json::json!("rust"));
        let out = tool._run(ToolInput::Structured(m)).await.unwrap();
        match out {
            ToolOutput::Content(v) => {
                assert_eq!(v["results"][0]["title"], "Rust");
                assert_eq!(v["results"][0]["url"], "https://rust-lang.org");
            }
            _ => panic!("expected content"),
        }
        let seen = captured.lock().unwrap().clone().unwrap();
        assert_eq!(seen.query, "rust");
        // Default applied when caller didn't supply.
        assert_eq!(seen.max_results, Some(3));
    }

    #[tokio::test]
    async fn caller_supplied_max_results_overrides_default() {
        let provider = |input: WebSearchInput| async move {
            assert_eq!(input.max_results, Some(10));
            Ok(Vec::new())
        };
        let tool = WebSearchTool::new(provider).with_default_max_results(3);
        let mut m = HashMap::new();
        m.insert("query".to_string(), serde_json::json!("anything"));
        m.insert("max_results".to_string(), serde_json::json!(10));
        let _ = tool._run(ToolInput::Structured(m)).await.unwrap();
    }

    #[tokio::test]
    async fn missing_query_errors_validation() {
        let tool = WebSearchTool::new(|_input: WebSearchInput| async {
            Ok(Vec::<WebSearchResult>::new())
        });
        let res = tool._run(ToolInput::Structured(HashMap::new())).await;
        assert!(matches!(res, Err(CognisError::ToolValidationError(_))));
    }

    #[test]
    fn custom_name_and_description() {
        let t =
            WebSearchTool::new(|_i: WebSearchInput| async { Ok(Vec::<WebSearchResult>::new()) })
                .with_name("search")
                .with_description("custom");
        assert_eq!(t.name(), "search");
        assert_eq!(t.description(), "custom");
    }

    #[test]
    fn schema_serializes_with_query_and_max_results() {
        let t =
            WebSearchTool::new(|_i: WebSearchInput| async { Ok(Vec::<WebSearchResult>::new()) });
        let s = t.args_schema().unwrap();
        let s = s.to_string();
        assert!(s.contains("query"));
        assert!(s.contains("max_results"));
    }
}
