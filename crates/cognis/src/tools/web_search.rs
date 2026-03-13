//! Web search tools for retrieving information from the internet.
//!
//! Provides a generic [`WebSearchTool`] that can be configured with any search
//! API endpoint, and a ready-to-use [`DuckDuckGoSearchTool`] that queries the
//! DuckDuckGo instant answer API (no API key required).

use async_trait::async_trait;
use cognis_core::error::{CognisError, Result};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// WebSearchTool
// ---------------------------------------------------------------------------

/// A configurable web search tool that queries a search API endpoint.
///
/// # Builder
///
/// ```rust,ignore
/// let tool = WebSearchTool::builder()
///     .api_url("https://my-search-api.example.com/search")
///     .api_key("sk-...")
///     .num_results(3)
///     .build();
/// ```
pub struct WebSearchTool {
    /// The search API endpoint URL.
    api_url: Option<String>,
    /// Optional API key sent as a `Bearer` token.
    api_key: Option<SecretString>,
    /// Maximum number of results to return (default: 5).
    num_results: usize,
    /// Shared HTTP client.
    client: reqwest::Client,
}

/// Builder for [`WebSearchTool`].
pub struct WebSearchToolBuilder {
    api_url: Option<String>,
    api_key: Option<SecretString>,
    num_results: usize,
    client: Option<reqwest::Client>,
}

impl WebSearchToolBuilder {
    /// Set the search API endpoint URL.
    pub fn api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = Some(url.into());
        self
    }

    /// Set the API key (sent as a `Bearer` authorization header).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the number of results to request (default: 5).
    pub fn num_results(mut self, n: usize) -> Self {
        self.num_results = n;
        self
    }

    /// Provide a custom [`reqwest::Client`].
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`WebSearchTool`].
    pub fn build(self) -> WebSearchTool {
        WebSearchTool {
            api_url: self.api_url,
            api_key: self.api_key,
            num_results: self.num_results,
            client: self.client.unwrap_or_default(),
        }
    }
}

impl WebSearchTool {
    /// Create a new builder.
    pub fn builder() -> WebSearchToolBuilder {
        WebSearchToolBuilder {
            api_url: None,
            api_key: None,
            num_results: 5,
            client: None,
        }
    }
}

#[async_trait]
impl BaseTool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information. Input should be a search query string."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let query = extract_query(&input)?;

        let api_url = match &self.api_url {
            Some(url) => url,
            None => {
                return Ok(ToolOutput::Content(Value::String(format!(
                    "WebSearchTool: No api_url configured. \
                     Would search for: \"{query}\". \
                     Please set an api_url to enable real searches."
                ))));
            }
        };

        let mut req = self
            .client
            .get(api_url)
            .query(&[("q", &query), ("num", &self.num_results.to_string())]);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key.expose_secret()));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| CognisError::ToolException(format!("Web search request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(CognisError::ToolException(format!(
                "Web search returned status {}",
                resp.status()
            )));
        }

        let body = resp.text().await.map_err(|e| {
            CognisError::ToolException(format!("Failed to read search response: {e}"))
        })?;

        Ok(ToolOutput::Content(Value::String(body)))
    }
}

// ---------------------------------------------------------------------------
// DuckDuckGoSearchTool
// ---------------------------------------------------------------------------

/// A web search tool that uses the DuckDuckGo instant answer API.
///
/// No API key is required.
///
/// ```rust,ignore
/// let tool = DuckDuckGoSearchTool::new();
/// let result = tool.run_str("Rust programming language").await.unwrap();
/// ```
pub struct DuckDuckGoSearchTool {
    /// Shared HTTP client.
    client: reqwest::Client,
}

impl DuckDuckGoSearchTool {
    /// Create a new DuckDuckGo search tool with a default HTTP client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Create a new tool with a custom [`reqwest::Client`].
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for DuckDuckGoSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Response shape from the DuckDuckGo instant answer API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct DdgResponse {
    #[serde(default, rename = "AbstractText")]
    pub abstract_text: String,
    #[serde(default, rename = "AbstractSource")]
    pub abstract_source: String,
    #[serde(default, rename = "AbstractURL")]
    pub abstract_url: String,
    #[serde(default, rename = "Heading")]
    pub heading: String,
    #[serde(default, rename = "RelatedTopics")]
    pub related_topics: Vec<DdgRelatedTopic>,
}

/// A related topic entry from DuckDuckGo.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct DdgRelatedTopic {
    #[serde(default, rename = "Text")]
    pub text: String,
    #[serde(default, rename = "FirstURL")]
    pub first_url: String,
}

/// Format a DuckDuckGo API response into a readable string.
pub(crate) fn format_ddg_response(resp: &DdgResponse) -> String {
    let mut parts: Vec<String> = Vec::new();

    if !resp.heading.is_empty() {
        parts.push(format!("# {}", resp.heading));
    }

    if !resp.abstract_text.is_empty() {
        parts.push(resp.abstract_text.clone());
        if !resp.abstract_url.is_empty() {
            parts.push(format!("Source: {}", resp.abstract_url));
        }
    }

    if !resp.related_topics.is_empty() {
        parts.push("\nRelated:".to_string());
        for (i, topic) in resp.related_topics.iter().take(5).enumerate() {
            if !topic.text.is_empty() {
                parts.push(format!("{}. {}", i + 1, topic.text));
            }
        }
    }

    if parts.is_empty() {
        "No results found.".to_string()
    } else {
        parts.join("\n")
    }
}

#[async_trait]
impl BaseTool for DuckDuckGoSearchTool {
    fn name(&self) -> &str {
        "duckduckgo_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo. Input should be a search query string."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let query = extract_query(&input)?;

        let resp = self
            .client
            .get("https://api.duckduckgo.com/")
            .query(&[
                ("q", &query),
                ("format", &"json".to_string()),
                ("no_html", &"1".to_string()),
            ])
            .send()
            .await
            .map_err(|e| CognisError::ToolException(format!("DuckDuckGo request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(CognisError::ToolException(format!(
                "DuckDuckGo returned status {}",
                resp.status()
            )));
        }

        let ddg: DdgResponse = resp.json().await.map_err(|e| {
            CognisError::ToolException(format!("Failed to parse DuckDuckGo response: {e}"))
        })?;

        let formatted = format_ddg_response(&ddg);
        Ok(ToolOutput::Content(Value::String(formatted)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a query string from various input formats.
fn extract_query(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(q)) = map.get("query") {
                Ok(q.clone())
            } else {
                Err(CognisError::ToolValidationError(
                    "Missing required field 'query'".into(),
                ))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(q)) = tc.args.get("query") {
                Ok(q.clone())
            } else {
                Err(CognisError::ToolValidationError(
                    "Missing required field 'query'".into(),
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_builder_defaults() {
        let tool = WebSearchTool::builder().build();
        assert_eq!(tool.name(), "web_search");
        assert_eq!(tool.num_results, 5);
        assert!(tool.api_url.is_none());
        assert!(tool.api_key.is_none());
    }

    #[test]
    fn test_web_search_builder_custom() {
        let tool = WebSearchTool::builder()
            .api_url("https://example.com/search")
            .api_key("test-key")
            .num_results(3)
            .build();
        assert_eq!(tool.api_url.as_deref(), Some("https://example.com/search"));
        assert_eq!(tool.num_results, 3);
        assert!(tool.api_key.is_some());
    }

    #[test]
    fn test_web_search_args_schema() {
        let tool = WebSearchTool::builder().build();
        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&Value::String("query".to_string())));
    }

    #[tokio::test]
    async fn test_web_search_no_api_url_returns_placeholder() {
        let tool = WebSearchTool::builder().build();
        let result = tool
            ._run(ToolInput::Text("test query".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("No api_url configured"));
                assert!(s.contains("test query"));
            }
            _ => panic!("Expected Content with String"),
        }
    }

    #[test]
    fn test_ddg_response_parsing() {
        let json_str = r#"{
            "AbstractText": "Rust is a systems programming language.",
            "AbstractSource": "Wikipedia",
            "AbstractURL": "https://en.wikipedia.org/wiki/Rust_(programming_language)",
            "Heading": "Rust (programming language)",
            "RelatedTopics": [
                {"Text": "Rust was designed by Graydon Hoare.", "FirstURL": "https://example.com/1"},
                {"Text": "Rust emphasizes safety.", "FirstURL": "https://example.com/2"}
            ]
        }"#;

        let resp: DdgResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.heading, "Rust (programming language)");
        assert_eq!(
            resp.abstract_text,
            "Rust is a systems programming language."
        );
        assert_eq!(resp.related_topics.len(), 2);

        let formatted = format_ddg_response(&resp);
        assert!(formatted.contains("# Rust (programming language)"));
        assert!(formatted.contains("Rust is a systems programming language."));
        assert!(formatted.contains("1. Rust was designed by Graydon Hoare."));
        assert!(formatted.contains("2. Rust emphasizes safety."));
    }

    #[test]
    fn test_ddg_empty_response() {
        let resp = DdgResponse {
            abstract_text: String::new(),
            abstract_source: String::new(),
            abstract_url: String::new(),
            heading: String::new(),
            related_topics: Vec::new(),
        };
        let formatted = format_ddg_response(&resp);
        assert_eq!(formatted, "No results found.");
    }

    #[test]
    fn test_extract_query_from_text() {
        let input = ToolInput::Text("hello world".to_string());
        assert_eq!(extract_query(&input).unwrap(), "hello world");
    }

    #[test]
    fn test_extract_query_from_structured() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "query".to_string(),
            Value::String("structured query".to_string()),
        );
        let input = ToolInput::Structured(map);
        assert_eq!(extract_query(&input).unwrap(), "structured query");
    }

    #[test]
    fn test_extract_query_missing_field() {
        let map = std::collections::HashMap::new();
        let input = ToolInput::Structured(map);
        assert!(extract_query(&input).is_err());
    }
}
