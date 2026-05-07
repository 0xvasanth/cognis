//! Wikipedia search + page-summary tool.
//!
//! Hits the public Wikipedia REST API (no key). The tool exposes one
//! action: `search` (returns top-N matching titles) or `summary`
//! (returns one page's extract by exact title).
//!
//! Customization:
//! - [`WikipediaToolBuilder`] — language code (`"en"`, `"de"`, …),
//!   custom base URL (e.g. for a private Wikipedia mirror), top-k cap,
//!   user-agent override, custom HTTP client.
//! - The tool is feature-gated under `tools-http` because it needs
//!   `reqwest`. Falls back to a clear error when the feature is off.

#![cfg(feature = "tools-http")]

use std::time::Duration;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

const DEFAULT_USER_AGENT: &str = "cognis2/0.1 (+https://github.com/0xvasanth/cognis)";
const DEFAULT_TOP_K: usize = 5;
const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// Action variants the tool understands.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WikipediaAction {
    /// Search for matching pages.
    Search,
    /// Fetch a page summary by exact title.
    Summary,
}

/// Tool input schema.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WikipediaInput {
    /// `search` returns the top-K matching titles; `summary` returns
    /// one page's extract.
    pub action: WikipediaAction,
    /// Search query (for `search`) or exact page title (for `summary`).
    pub query: String,
    /// Override the default top-k cap (`search` only).
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// Wikipedia tool.
pub struct WikipediaTool {
    base_url: String,
    language: String,
    user_agent: String,
    top_k_default: usize,
    http: reqwest::Client,
}

impl std::fmt::Debug for WikipediaTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WikipediaTool")
            .field("language", &self.language)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl WikipediaTool {
    /// Build with default English language and a fresh HTTP client.
    pub fn new() -> Result<Self> {
        WikipediaToolBuilder::default().build()
    }

    /// Fluent builder.
    pub fn builder() -> WikipediaToolBuilder {
        WikipediaToolBuilder::default()
    }

    fn search_url(&self, q: &str, k: usize) -> String {
        format!(
            "{base}/w/api.php?action=query&list=search&srsearch={q}&srlimit={k}&format=json&utf8=1",
            base = self.base_url,
            q = urlencoding_simple(q),
            k = k,
        )
    }

    fn summary_url(&self, title: &str) -> String {
        format!(
            "{base}/api/rest_v1/page/summary/{title}",
            base = self.base_url,
            title = urlencoding_simple(title),
        )
    }

    async fn search(&self, q: &str, k: usize) -> Result<serde_json::Value> {
        #[derive(Deserialize)]
        struct ApiResp {
            query: SearchPayload,
        }
        #[derive(Deserialize)]
        struct SearchPayload {
            search: Vec<SearchHit>,
        }
        #[derive(Deserialize)]
        struct SearchHit {
            title: String,
            #[serde(default)]
            snippet: String,
            #[serde(default)]
            pageid: u64,
        }
        let url = self.search_url(q, k);
        let resp = self
            .http
            .get(&url)
            .header(reqwest::header::USER_AGENT, &self.user_agent)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("wikipedia search: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "wikipedia search: HTTP {s}: {t}"
            )));
        }
        let parsed: ApiResp = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("wikipedia json: {e}")))?;
        let hits: Vec<serde_json::Value> = parsed
            .query
            .search
            .into_iter()
            .map(|h| {
                serde_json::json!({
                    "title": h.title,
                    "snippet": strip_html(&h.snippet),
                    "pageid": h.pageid,
                })
            })
            .collect();
        Ok(serde_json::json!({ "results": hits }))
    }

    async fn summary(&self, title: &str) -> Result<serde_json::Value> {
        let url = self.summary_url(title);
        let resp = self
            .http
            .get(&url)
            .header(reqwest::header::USER_AGENT, &self.user_agent)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("wikipedia summary: {e}")))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(serde_json::json!({"found": false, "title": title}));
        }
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "wikipedia summary: HTTP {s}: {t}"
            )));
        }
        let payload: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("wikipedia summary json: {e}")))?;
        // Pull a stable subset of fields out so the LLM doesn't see the
        // full WMF response shape.
        Ok(serde_json::json!({
            "found": true,
            "title": payload.get("title").cloned().unwrap_or_default(),
            "description": payload.get("description").cloned().unwrap_or_default(),
            "extract": payload.get("extract").cloned().unwrap_or_default(),
            "url": payload.pointer("/content_urls/desktop/page").cloned().unwrap_or_default(),
        }))
    }
}

#[async_trait]
impl Tool for WikipediaTool {
    fn name(&self) -> &str {
        "wikipedia"
    }
    fn description(&self) -> &str {
        "Search Wikipedia or fetch a page summary by exact title. \
         Use action='search' to find pages, action='summary' to read."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(WikipediaInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: WikipediaInput = serde_json::from_value(input.into_json()).map_err(|e| {
            CognisError::ToolValidationError(format!("wikipedia: invalid args: {e}"))
        })?;
        let payload = match parsed.action {
            WikipediaAction::Search => {
                let k = parsed.top_k.unwrap_or(self.top_k_default).max(1);
                self.search(&parsed.query, k).await?
            }
            WikipediaAction::Summary => self.summary(&parsed.query).await?,
        };
        Ok(ToolOutput::Content(payload))
    }
}

/// Strip basic HTML tags (Wikipedia search snippets contain
/// `<span class="...">` decorations). Lightweight; not a full HTML parser.
fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Minimal URL-encoder for query parameters. Handles the characters
/// Wikipedia titles can produce; not a full RFC-3986 implementation.
fn urlencoding_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Fluent builder.
#[derive(Default)]
pub struct WikipediaToolBuilder {
    base_url: Option<String>,
    language: Option<String>,
    user_agent: Option<String>,
    top_k_default: Option<usize>,
    http: Option<reqwest::Client>,
    timeout_secs: Option<u64>,
}

impl WikipediaToolBuilder {
    /// Override base URL (default depends on language).
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Set the language code (default `"en"`). Determines the default
    /// base URL when `base_url` is not set.
    pub fn language(mut self, code: impl Into<String>) -> Self {
        self.language = Some(code.into());
        self
    }
    /// Override the User-Agent header (Wikipedia requires a non-default UA).
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }
    /// Default top-k for search (overridable per-call).
    pub fn top_k_default(mut self, k: usize) -> Self {
        self.top_k_default = Some(k);
        self
    }
    /// Override the HTTP client.
    pub fn http_client(mut self, c: reqwest::Client) -> Self {
        self.http = Some(c);
        self
    }
    /// Override the timeout.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Build.
    pub fn build(self) -> Result<WikipediaTool> {
        let language = self.language.unwrap_or_else(|| "en".to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| format!("https://{language}.wikipedia.org"));
        let http = match self.http {
            Some(c) => c,
            None => reqwest::ClientBuilder::new()
                .timeout(Duration::from_secs(
                    self.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
                ))
                .build()
                .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?,
        };
        Ok(WikipediaTool {
            base_url,
            language,
            user_agent: self
                .user_agent
                .unwrap_or_else(|| DEFAULT_USER_AGENT.to_string()),
            top_k_default: self.top_k_default.unwrap_or(DEFAULT_TOP_K),
            http,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_url_encodes_query() {
        let t = WikipediaTool::new().unwrap();
        let url = t.search_url("rust language", 3);
        assert!(url.contains("srsearch=rust+language"));
        assert!(url.contains("srlimit=3"));
    }

    #[test]
    fn summary_url_encodes_title() {
        let t = WikipediaTool::new().unwrap();
        let url = t.summary_url("Rust (programming language)");
        // Spaces become +, non-ASCII / parens become %-encoded.
        assert!(url.contains("Rust"));
        assert!(url.contains("%28"));
        assert!(url.contains("%29"));
    }

    #[test]
    fn language_code_changes_base_url() {
        let de = WikipediaToolBuilder::default()
            .language("de")
            .build()
            .unwrap();
        assert!(de.base_url.contains("de.wikipedia.org"));
    }

    #[test]
    fn strip_html_removes_tags_only() {
        assert_eq!(
            strip_html(r#"<span class="x">hello</span> world"#),
            "hello world"
        );
        assert_eq!(strip_html("plain"), "plain");
    }

    #[test]
    fn urlencoder_handles_punctuation() {
        assert_eq!(urlencoding_simple("a b"), "a+b");
        assert_eq!(urlencoding_simple("a&b"), "a%26b");
        assert_eq!(urlencoding_simple("a/b"), "a%2Fb");
        assert_eq!(urlencoding_simple("hello"), "hello");
    }

    #[test]
    fn schema_serializes() {
        let t = WikipediaTool::new().unwrap();
        let s = t.args_schema().unwrap();
        assert!(s.to_string().contains("action"));
        assert!(s.to_string().contains("query"));
    }
}
