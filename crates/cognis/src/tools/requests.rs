//! HTTP request tools for making web requests as agent tools.
//!
//! Provides `RequestsTool` which implements `BaseTool` and delegates HTTP calls
//! to a pluggable `HttpClient` trait. This design allows users to bring their
//! own HTTP backend (e.g., `reqwest`) while keeping the tool logic testable
//! via the included `MockHttpClient`.

use async_trait::async_trait;
use cognis_core::error::{Result, CognisError};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// HttpMethod
// ---------------------------------------------------------------------------

/// HTTP methods supported by the request tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl HttpMethod {
    /// Return the method as an uppercase string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
        }
    }
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for HttpMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "GET" => Ok(Self::Get),
            "POST" => Ok(Self::Post),
            "PUT" => Ok(Self::Put),
            "PATCH" => Ok(Self::Patch),
            "DELETE" => Ok(Self::Delete),
            other => Err(format!("Unknown HTTP method: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------

/// Represents an HTTP response returned by an `HttpClient`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body as a string.
    pub body: String,
    /// Response headers.
    pub headers: HashMap<String, String>,
}

impl HttpResponse {
    /// Create a simple response with just a status and body.
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            headers: HashMap::new(),
        }
    }

    /// Builder-style method to add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// HttpClient trait
// ---------------------------------------------------------------------------

/// Abstraction over HTTP clients for testability.
///
/// Users should implement this trait to wrap their preferred HTTP library
/// (e.g., `reqwest`). The `MockHttpClient` is provided for testing.
#[async_trait]
pub trait HttpClient: Send + Sync {
    /// Execute an HTTP request.
    async fn request(
        &self,
        method: HttpMethod,
        url: &str,
        body: Option<&str>,
        headers: &HashMap<String, String>,
    ) -> Result<HttpResponse>;
}

// ---------------------------------------------------------------------------
// RequestConfig
// ---------------------------------------------------------------------------

/// Configuration for HTTP request tools.
#[derive(Debug, Clone)]
pub struct RequestConfig {
    /// Request timeout duration.
    pub timeout: Duration,
    /// Maximum number of characters to return from the response body.
    pub max_response_length: usize,
    /// Optional domain allowlist. `None` means all domains are allowed.
    pub allowed_domains: Option<Vec<String>>,
    /// Default headers to include in every request.
    pub headers: HashMap<String, String>,
    /// Whether to follow HTTP redirects.
    pub follow_redirects: bool,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_response_length: 10_000,
            allowed_domains: None,
            headers: HashMap::new(),
            follow_redirects: true,
        }
    }
}

impl RequestConfig {
    /// Create a new builder for `RequestConfig`.
    pub fn builder() -> RequestConfigBuilder {
        RequestConfigBuilder::default()
    }
}

/// Builder for `RequestConfig`.
#[derive(Debug, Clone, Default)]
pub struct RequestConfigBuilder {
    config: RequestConfig,
}

impl RequestConfigBuilder {
    /// Set the request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the maximum response body length (in characters).
    pub fn max_response_length(mut self, max: usize) -> Self {
        self.config.max_response_length = max;
        self
    }

    /// Set allowed domains. Only URLs whose host matches one of these domains
    /// will be permitted. Pass `None` to allow all domains.
    pub fn allowed_domains(mut self, domains: Option<Vec<String>>) -> Self {
        self.config.allowed_domains = domains;
        self
    }

    /// Add a default header.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.insert(key.into(), value.into());
        self
    }

    /// Set all default headers at once.
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.config.headers = headers;
        self
    }

    /// Set whether to follow redirects.
    pub fn follow_redirects(mut self, follow: bool) -> Self {
        self.config.follow_redirects = follow;
        self
    }

    /// Build the `RequestConfig`.
    pub fn build(self) -> RequestConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// MockHttpClient
// ---------------------------------------------------------------------------

/// A mock HTTP client for testing that returns pre-configured responses.
///
/// Responses are returned in order. If more requests are made than responses
/// provided, the last response is repeated.
pub struct MockHttpClient {
    responses: Mutex<Vec<HttpResponse>>,
    calls: Mutex<Vec<MockCall>>,
}

/// Record of a call made to the `MockHttpClient`.
#[derive(Debug, Clone)]
pub struct MockCall {
    pub method: HttpMethod,
    pub url: String,
    pub body: Option<String>,
    pub headers: HashMap<String, String>,
}

impl MockHttpClient {
    /// Create a new mock client with the given sequence of responses.
    pub fn new(responses: Vec<HttpResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Return all calls that have been made to this client.
    pub fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Return the number of calls made.
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl HttpClient for MockHttpClient {
    async fn request(
        &self,
        method: HttpMethod,
        url: &str,
        body: Option<&str>,
        headers: &HashMap<String, String>,
    ) -> Result<HttpResponse> {
        self.calls.lock().unwrap().push(MockCall {
            method,
            url: url.to_string(),
            body: body.map(|s| s.to_string()),
            headers: headers.clone(),
        });

        let mut responses = self.responses.lock().unwrap();
        if responses.len() > 1 {
            Ok(responses.remove(0))
        } else if let Some(resp) = responses.first() {
            Ok(resp.clone())
        } else {
            Err(CognisError::Other(
                "MockHttpClient: no responses configured".to_string(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// TextExtractor
// ---------------------------------------------------------------------------

/// Simple HTML-to-text helper that strips tags, decodes basic entities,
/// and collapses whitespace.
pub struct TextExtractor;

impl TextExtractor {
    /// Strip HTML tags from the input string.
    pub fn strip_tags(html: &str) -> String {
        let mut result = String::with_capacity(html.len());
        let mut in_tag = false;
        for ch in html.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => {
                    in_tag = false;
                    result.push(' ');
                }
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        result
    }

    /// Decode basic HTML entities.
    pub fn decode_entities(text: &str) -> String {
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
    }

    /// Collapse runs of whitespace into single spaces and trim.
    pub fn collapse_whitespace(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_ws = false;
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    result.push(' ');
                }
                prev_ws = true;
            } else {
                result.push(ch);
                prev_ws = false;
            }
        }
        result.trim().to_string()
    }

    /// Full extraction pipeline: strip tags, decode entities, collapse whitespace.
    pub fn extract(html: &str) -> String {
        let text = Self::strip_tags(html);
        let text = Self::decode_entities(&text);
        Self::collapse_whitespace(&text)
    }
}

// ---------------------------------------------------------------------------
// RequestsTool
// ---------------------------------------------------------------------------

/// Input format for the HTTP request tool (parsed from JSON).
#[derive(Debug, Clone, Deserialize)]
struct RequestInput {
    url: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
}

/// An agent tool that makes HTTP requests via a pluggable `HttpClient`.
pub struct RequestsTool {
    client: Arc<dyn HttpClient>,
    config: RequestConfig,
    default_method: HttpMethod,
}

impl RequestsTool {
    /// Create a new `RequestsTool` with the given client, config, and default HTTP method.
    pub fn new(
        client: Arc<dyn HttpClient>,
        config: RequestConfig,
        default_method: HttpMethod,
    ) -> Self {
        Self {
            client,
            config,
            default_method,
        }
    }

    /// Validate that the URL's host is in the allowed domains list.
    fn validate_domain(&self, url: &str) -> Result<()> {
        if let Some(ref allowed) = self.config.allowed_domains {
            let host = extract_host(url).ok_or_else(|| {
                CognisError::ToolValidationError(format!(
                    "Could not extract host from URL: {url}"
                ))
            })?;
            if !allowed
                .iter()
                .any(|d| host == *d || host.ends_with(&format!(".{d}")))
            {
                return Err(CognisError::ToolValidationError(format!(
                    "Domain '{host}' is not in the allowed domains list"
                )));
            }
        }
        Ok(())
    }

    /// Merge default headers with per-request headers (per-request wins).
    fn merge_headers(&self, extra: &Option<HashMap<String, String>>) -> HashMap<String, String> {
        let mut merged = self.config.headers.clone();
        if let Some(extra) = extra {
            for (k, v) in extra {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    /// Truncate response body to `max_response_length`.
    fn truncate_body(&self, body: &str) -> String {
        if body.len() > self.config.max_response_length {
            let mut truncated: String =
                body.chars().take(self.config.max_response_length).collect();
            truncated.push_str("... [truncated]");
            truncated
        } else {
            body.to_string()
        }
    }

    /// Parse the tool input into a `RequestInput`.
    fn parse_input(&self, input: &ToolInput) -> Result<RequestInput> {
        match input {
            ToolInput::Text(s) => {
                // Try to parse as JSON first; fall back to treating as a URL.
                if let Ok(parsed) = serde_json::from_str::<RequestInput>(s) {
                    Ok(parsed)
                } else {
                    Ok(RequestInput {
                        url: s.trim().to_string(),
                        method: None,
                        body: None,
                        headers: None,
                    })
                }
            }
            ToolInput::Structured(map) => {
                let value = serde_json::to_value(map).map_err(|e| {
                    CognisError::ToolValidationError(format!("Failed to serialize input: {e}"))
                })?;
                serde_json::from_value::<RequestInput>(value).map_err(|e| {
                    CognisError::ToolValidationError(format!("Invalid request input: {e}"))
                })
            }
            ToolInput::ToolCall(tc) => {
                let value = serde_json::to_value(&tc.args).map_err(|e| {
                    CognisError::ToolValidationError(format!("Failed to serialize args: {e}"))
                })?;
                serde_json::from_value::<RequestInput>(value).map_err(|e| {
                    CognisError::ToolValidationError(format!("Invalid request input: {e}"))
                })
            }
        }
    }
}

#[async_trait]
impl BaseTool for RequestsTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Make HTTP requests to URLs. Input should be JSON with 'url' (required), \
         'method' (GET/POST/PUT/PATCH/DELETE), 'body', and 'headers' fields. \
         A plain URL string is also accepted for simple GET requests."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to request"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": "HTTP method (defaults to the tool's default method)"
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH)"
                },
                "headers": {
                    "type": "object",
                    "description": "Additional request headers",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["url"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let req = self.parse_input(&input)?;

        self.validate_domain(&req.url)?;

        let method = if let Some(ref m) = req.method {
            m.parse::<HttpMethod>()
                .map_err(CognisError::ToolValidationError)?
        } else {
            self.default_method
        };

        let headers = self.merge_headers(&req.headers);
        let response = self
            .client
            .request(method, &req.url, req.body.as_deref(), &headers)
            .await?;

        let body = self.truncate_body(&response.body);

        let result = json!({
            "status": response.status,
            "body": body,
            "headers": response.headers,
        });

        Ok(ToolOutput::Content(result))
    }
}

// ---------------------------------------------------------------------------
// Convenience factories
// ---------------------------------------------------------------------------

/// Create an HTTP GET request tool with the given config and client.
pub fn requests_get_tool(config: RequestConfig, client: Arc<dyn HttpClient>) -> RequestsTool {
    RequestsTool::new(client, config, HttpMethod::Get)
}

/// Create an HTTP POST request tool with the given config and client.
pub fn requests_post_tool(config: RequestConfig, client: Arc<dyn HttpClient>) -> RequestsTool {
    RequestsTool::new(client, config, HttpMethod::Post)
}

// ---------------------------------------------------------------------------
// URL helper
// ---------------------------------------------------------------------------

/// Extract the host from a URL string (simple parser, no external deps).
fn extract_host(url: &str) -> Option<String> {
    // Skip scheme
    let without_scheme = if let Some(pos) = url.find("://") {
        &url[pos + 3..]
    } else {
        url
    };
    // Skip userinfo
    let without_user = if let Some(pos) = without_scheme.find('@') {
        &without_scheme[pos + 1..]
    } else {
        without_scheme
    };
    // Take up to port or path
    let host = without_user.split('/').next()?.split(':').next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_lowercase())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_client(responses: Vec<HttpResponse>) -> Arc<MockHttpClient> {
        Arc::new(MockHttpClient::new(responses))
    }

    fn ok_response(body: &str) -> HttpResponse {
        HttpResponse::new(200, body)
    }

    fn default_config() -> RequestConfig {
        RequestConfig::default()
    }

    // --- HttpMethod tests ---

    #[test]
    fn test_http_method_as_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Put.as_str(), "PUT");
        assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
        assert_eq!(HttpMethod::Delete.as_str(), "DELETE");
    }

    #[test]
    fn test_http_method_from_str() {
        assert_eq!("get".parse::<HttpMethod>().unwrap(), HttpMethod::Get);
        assert_eq!("POST".parse::<HttpMethod>().unwrap(), HttpMethod::Post);
        assert_eq!("Put".parse::<HttpMethod>().unwrap(), HttpMethod::Put);
        assert!("INVALID".parse::<HttpMethod>().is_err());
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(format!("{}", HttpMethod::Get), "GET");
        assert_eq!(format!("{}", HttpMethod::Delete), "DELETE");
    }

    #[test]
    fn test_http_method_serde_roundtrip() {
        let method = HttpMethod::Post;
        let json = serde_json::to_string(&method).unwrap();
        let deserialized: HttpMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, method);
    }

    // --- HttpResponse tests ---

    #[test]
    fn test_http_response_new() {
        let resp = HttpResponse::new(404, "not found");
        assert_eq!(resp.status, 404);
        assert_eq!(resp.body, "not found");
        assert!(resp.headers.is_empty());
    }

    #[test]
    fn test_http_response_with_header() {
        let resp = HttpResponse::new(200, "ok").with_header("content-type", "application/json");
        assert_eq!(
            resp.headers.get("content-type").unwrap(),
            "application/json"
        );
    }

    // --- RequestConfig tests ---

    #[test]
    fn test_request_config_defaults() {
        let config = RequestConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_response_length, 10_000);
        assert!(config.allowed_domains.is_none());
        assert!(config.headers.is_empty());
        assert!(config.follow_redirects);
    }

    #[test]
    fn test_request_config_builder() {
        let config = RequestConfig::builder()
            .timeout(Duration::from_secs(60))
            .max_response_length(5000)
            .allowed_domains(Some(vec!["example.com".to_string()]))
            .header("Authorization", "Bearer token123")
            .follow_redirects(false)
            .build();

        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_response_length, 5000);
        assert_eq!(
            config.allowed_domains,
            Some(vec!["example.com".to_string()])
        );
        assert_eq!(
            config.headers.get("Authorization").unwrap(),
            "Bearer token123"
        );
        assert!(!config.follow_redirects);
    }

    #[test]
    fn test_request_config_builder_headers_override() {
        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_string(), "value".to_string());
        let config = RequestConfig::builder()
            .header("A", "1")
            .headers(headers)
            .build();
        assert!(config.headers.get("A").is_none());
        assert_eq!(config.headers.get("X-Custom").unwrap(), "value");
    }

    // --- MockHttpClient tests ---

    #[tokio::test]
    async fn test_mock_client_returns_responses_in_order() {
        let client = MockHttpClient::new(vec![
            HttpResponse::new(200, "first"),
            HttpResponse::new(201, "second"),
        ]);
        let headers = HashMap::new();

        let r1 = client
            .request(HttpMethod::Get, "http://a.com", None, &headers)
            .await
            .unwrap();
        assert_eq!(r1.body, "first");

        let r2 = client
            .request(HttpMethod::Get, "http://b.com", None, &headers)
            .await
            .unwrap();
        assert_eq!(r2.body, "second");
    }

    #[tokio::test]
    async fn test_mock_client_repeats_last_response() {
        let client = MockHttpClient::new(vec![HttpResponse::new(200, "only")]);
        let headers = HashMap::new();

        let r1 = client
            .request(HttpMethod::Get, "http://a.com", None, &headers)
            .await
            .unwrap();
        let r2 = client
            .request(HttpMethod::Get, "http://b.com", None, &headers)
            .await
            .unwrap();
        assert_eq!(r1.body, "only");
        assert_eq!(r2.body, "only");
    }

    #[tokio::test]
    async fn test_mock_client_records_calls() {
        let client = MockHttpClient::new(vec![ok_response("ok")]);
        let mut hdrs = HashMap::new();
        hdrs.insert("X-Test".to_string(), "yes".to_string());

        client
            .request(
                HttpMethod::Post,
                "http://example.com/api",
                Some("body"),
                &hdrs,
            )
            .await
            .unwrap();

        assert_eq!(client.call_count(), 1);
        let calls = client.calls();
        assert_eq!(calls[0].method, HttpMethod::Post);
        assert_eq!(calls[0].url, "http://example.com/api");
        assert_eq!(calls[0].body.as_deref(), Some("body"));
        assert_eq!(calls[0].headers.get("X-Test").unwrap(), "yes");
    }

    #[tokio::test]
    async fn test_mock_client_no_responses_error() {
        let client = MockHttpClient::new(vec![]);
        let headers = HashMap::new();
        let result = client
            .request(HttpMethod::Get, "http://a.com", None, &headers)
            .await;
        assert!(result.is_err());
    }

    // --- TextExtractor tests ---

    #[test]
    fn test_strip_tags() {
        // strip_tags inserts spaces where tags were; collapse_whitespace normalizes
        let raw = TextExtractor::strip_tags("<h1>Hello</h1> <p>World</p>");
        assert!(!raw.contains('<'));
        assert!(raw.contains("Hello"));
        assert!(raw.contains("World"));
    }

    #[test]
    fn test_strip_tags_nested() {
        let raw = TextExtractor::strip_tags("<div><span>inner</span></div>");
        assert!(!raw.contains('<'));
        assert!(raw.contains("inner"));
    }

    #[test]
    fn test_decode_entities() {
        assert_eq!(
            TextExtractor::decode_entities("a &amp; b &lt; c &gt; d &quot;e&quot; f&#39;g&apos;h"),
            "a & b < c > d \"e\" f'g'h"
        );
    }

    #[test]
    fn test_decode_nbsp() {
        assert_eq!(TextExtractor::decode_entities("a&nbsp;b"), "a b");
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(
            TextExtractor::collapse_whitespace("  hello   world  \n\t foo  "),
            "hello world foo"
        );
    }

    #[test]
    fn test_extract_full_pipeline() {
        let html = "<html><body><h1>Title</h1><p>Some &amp; text</p></body></html>";
        assert_eq!(TextExtractor::extract(html), "Title Some & text");
    }

    #[test]
    fn test_extract_plain_text_passthrough() {
        assert_eq!(TextExtractor::extract("plain text"), "plain text");
    }

    // --- extract_host tests ---

    #[test]
    fn test_extract_host_basic() {
        assert_eq!(
            extract_host("https://example.com/path"),
            Some("example.com".to_string())
        );
    }

    #[test]
    fn test_extract_host_with_port() {
        assert_eq!(
            extract_host("http://localhost:8080/api"),
            Some("localhost".to_string())
        );
    }

    #[test]
    fn test_extract_host_with_userinfo() {
        assert_eq!(
            extract_host("http://user:pass@host.com/x"),
            Some("host.com".to_string())
        );
    }

    #[test]
    fn test_extract_host_uppercase() {
        assert_eq!(
            extract_host("https://EXAMPLE.COM"),
            Some("example.com".to_string())
        );
    }

    // --- RequestsTool basic tests ---

    #[tokio::test]
    async fn test_requests_tool_name_and_description() {
        let client = mock_client(vec![ok_response("ok")]);
        let tool = RequestsTool::new(client, default_config(), HttpMethod::Get);
        assert_eq!(tool.name(), "http_request");
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn test_requests_tool_args_schema() {
        let client = mock_client(vec![ok_response("ok")]);
        let tool = RequestsTool::new(client, default_config(), HttpMethod::Get);
        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["properties"]["url"]["type"], "string");
    }

    #[tokio::test]
    async fn test_requests_tool_simple_get_text_input() {
        let client = mock_client(vec![ok_response("hello world")]);
        let tool = RequestsTool::new(client.clone(), default_config(), HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => {
                assert_eq!(v["status"], 200);
                assert_eq!(v["body"], "hello world");
            }
            _ => panic!("Expected Content variant"),
        }

        assert_eq!(client.call_count(), 1);
        assert_eq!(client.calls()[0].method, HttpMethod::Get);
    }

    #[tokio::test]
    async fn test_requests_tool_json_input() {
        let client = mock_client(vec![ok_response("{\"key\": \"value\"}")]);
        let tool = RequestsTool::new(client.clone(), default_config(), HttpMethod::Post);

        let input = ToolInput::Text(
            r#"{"url": "https://api.example.com", "method": "POST", "body": "{\"a\":1}"}"#
                .to_string(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v["status"], 200),
            _ => panic!("Expected Content variant"),
        }

        let calls = client.calls();
        assert_eq!(calls[0].method, HttpMethod::Post);
        assert_eq!(calls[0].body.as_deref(), Some("{\"a\":1}"));
    }

    #[tokio::test]
    async fn test_requests_tool_structured_input() {
        let client = mock_client(vec![ok_response("ok")]);
        let tool = RequestsTool::new(client.clone(), default_config(), HttpMethod::Get);

        let mut map = HashMap::new();
        map.insert(
            "url".to_string(),
            Value::String("https://example.com".to_string()),
        );
        map.insert("method".to_string(), Value::String("DELETE".to_string()));

        let result = tool._run(ToolInput::Structured(map)).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v["status"], 200),
            _ => panic!("Expected Content variant"),
        }
        assert_eq!(client.calls()[0].method, HttpMethod::Delete);
    }

    #[tokio::test]
    async fn test_requests_tool_domain_allowlist_pass() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder()
            .allowed_domains(Some(vec!["example.com".to_string()]))
            .build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://example.com/api".to_string()))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_requests_tool_domain_allowlist_reject() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder()
            .allowed_domains(Some(vec!["example.com".to_string()]))
            .build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://evil.com/steal".to_string()))
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            CognisError::ToolValidationError(msg) => {
                assert!(msg.contains("evil.com"));
                assert!(msg.contains("not in the allowed"));
            }
            e => panic!("Expected ToolValidationError, got: {e:?}"),
        }
    }

    #[tokio::test]
    async fn test_requests_tool_domain_allowlist_subdomain() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder()
            .allowed_domains(Some(vec!["example.com".to_string()]))
            .build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://api.example.com/v1".to_string()))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_requests_tool_no_domain_restriction() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder().allowed_domains(None).build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://anything.com".to_string()))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_requests_tool_truncation() {
        let long_body = "x".repeat(20_000);
        let client = mock_client(vec![ok_response(&long_body)]);
        let config = RequestConfig::builder().max_response_length(100).build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => {
                let body = v["body"].as_str().unwrap();
                assert!(body.len() < 200);
                assert!(body.ends_with("... [truncated]"));
            }
            _ => panic!("Expected Content variant"),
        }
    }

    #[tokio::test]
    async fn test_requests_tool_no_truncation_when_short() {
        let client = mock_client(vec![ok_response("short")]);
        let tool = RequestsTool::new(client, default_config(), HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => {
                let body = v["body"].as_str().unwrap();
                assert_eq!(body, "short");
                assert!(!body.contains("truncated"));
            }
            _ => panic!("Expected Content variant"),
        }
    }

    #[tokio::test]
    async fn test_requests_tool_default_headers_merged() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder().header("X-Default", "yes").build();
        let tool = RequestsTool::new(client.clone(), config, HttpMethod::Get);

        let input = r#"{"url": "https://example.com", "headers": {"X-Extra": "also"}}"#;
        tool._run(ToolInput::Text(input.to_string())).await.unwrap();

        let calls = client.calls();
        assert_eq!(calls[0].headers.get("X-Default").unwrap(), "yes");
        assert_eq!(calls[0].headers.get("X-Extra").unwrap(), "also");
    }

    #[tokio::test]
    async fn test_requests_tool_per_request_headers_override_defaults() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder().header("X-Key", "default").build();
        let tool = RequestsTool::new(client.clone(), config, HttpMethod::Get);

        let input = r#"{"url": "https://example.com", "headers": {"X-Key": "override"}}"#;
        tool._run(ToolInput::Text(input.to_string())).await.unwrap();

        assert_eq!(client.calls()[0].headers.get("X-Key").unwrap(), "override");
    }

    #[tokio::test]
    async fn test_requests_tool_invalid_method() {
        let client = mock_client(vec![ok_response("ok")]);
        let tool = RequestsTool::new(client, default_config(), HttpMethod::Get);

        let input = r#"{"url": "https://example.com", "method": "INVALID"}"#;
        let result = tool._run(ToolInput::Text(input.to_string())).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_requests_tool_response_includes_headers() {
        let resp = HttpResponse::new(200, "ok").with_header("X-Resp", "val");
        let client = mock_client(vec![resp]);
        let tool = RequestsTool::new(client, default_config(), HttpMethod::Get);

        let result = tool
            ._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => {
                assert_eq!(v["headers"]["X-Resp"], "val");
            }
            _ => panic!("Expected Content variant"),
        }
    }

    // --- Convenience factory tests ---

    #[tokio::test]
    async fn test_requests_get_tool_factory() {
        let client = mock_client(vec![ok_response("get response")]);
        let tool = requests_get_tool(default_config(), client.clone());

        tool._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        assert_eq!(client.calls()[0].method, HttpMethod::Get);
    }

    #[tokio::test]
    async fn test_requests_post_tool_factory() {
        let client = mock_client(vec![ok_response("post response")]);
        let tool = requests_post_tool(default_config(), client.clone());

        tool._run(ToolInput::Text("https://example.com".to_string()))
            .await
            .unwrap();
        assert_eq!(client.calls()[0].method, HttpMethod::Post);
    }

    #[tokio::test]
    async fn test_factory_tool_method_can_be_overridden() {
        let client = mock_client(vec![ok_response("ok")]);
        let tool = requests_get_tool(default_config(), client.clone());

        let input = r#"{"url": "https://example.com", "method": "PUT"}"#;
        tool._run(ToolInput::Text(input.to_string())).await.unwrap();
        assert_eq!(client.calls()[0].method, HttpMethod::Put);
    }

    #[tokio::test]
    async fn test_requests_tool_invalid_url_no_host() {
        let client = mock_client(vec![ok_response("ok")]);
        let config = RequestConfig::builder()
            .allowed_domains(Some(vec!["example.com".to_string()]))
            .build();
        let tool = RequestsTool::new(client, config, HttpMethod::Get);

        let result = tool._run(ToolInput::Text("://".to_string())).await;
        assert!(result.is_err());
    }
}
