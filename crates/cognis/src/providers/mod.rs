//! Provider integration framework for connecting to LLM APIs.
//!
//! This module provides the building blocks for interacting with LLM provider
//! HTTP APIs, including configuration management, request construction, response
//! parsing, retry policies with exponential backoff, and a provider registry.
//!
//! ## Key Types
//!
//! - [`ProviderConfig`] — connection settings for a provider (API key, base URL, etc.)
//! - [`ProviderEndpoint`] — describes a specific API endpoint (path, method, content type)
//! - [`RequestBuilder`] — constructs [`ApiRequest`] instances from config and endpoint
//! - [`ApiResponse`] — parsed HTTP response with status classification helpers
//! - [`ProviderRegistry`] — manages named provider configurations
//! - [`RetryPolicy`] — exponential backoff with jitter for transient failures
//! - [`ProviderError`] — error types with retryability classification

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── HttpMethod ───

/// HTTP method for an API request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    /// HTTP GET
    Get,
    /// HTTP POST
    Post,
    /// HTTP PUT
    Put,
    /// HTTP DELETE
    Delete,
    /// HTTP PATCH
    Patch,
}

impl HttpMethod {
    /// Return the method as an uppercase string slice.
    pub fn as_str(&self) -> &str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Delete => "DELETE",
            HttpMethod::Patch => "PATCH",
        }
    }
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─── ProviderConfig ───

/// Connection configuration for an LLM provider.
///
/// Holds credentials, endpoint settings, and retry parameters. Supports a
/// builder pattern for ergonomic construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The provider name (e.g. "openai", "anthropic").
    pub provider_name: String,
    /// API key for authentication.
    pub api_key: Option<String>,
    /// Base URL for the provider API.
    pub base_url: Option<String>,
    /// API version string (e.g. "2024-01-01").
    pub api_version: Option<String>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Additional HTTP headers to include with every request.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Provider-specific extra parameters.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl ProviderConfig {
    /// Create a new `ProviderConfig` with the given provider name and defaults.
    pub fn new(provider_name: impl Into<String>) -> Self {
        Self {
            provider_name: provider_name.into(),
            api_key: None,
            base_url: None,
            api_version: None,
            timeout_ms: 30000,
            max_retries: 3,
            headers: HashMap::new(),
            extra: HashMap::new(),
        }
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API version.
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// Set the request timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum number of retry attempts.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Add an HTTP header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Insert a provider-specific extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Serialize to JSON, masking the API key for safe logging.
    ///
    /// The `api_key` field is replaced with `"sk-****"` if present, or `null`
    /// if absent.
    pub fn to_json(&self) -> Value {
        let mut json = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Some(obj) = json.as_object_mut() {
            if self.api_key.is_some() {
                obj.insert("api_key".to_string(), Value::String("sk-****".to_string()));
            }
        }
        json
    }

    /// Check whether the provider is minimally configured (has an API key or
    /// a base URL set).
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some() || self.base_url.is_some()
    }
}

// ─── ProviderEndpoint ───

/// Describes a specific API endpoint on a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEndpoint {
    /// The URL path (e.g. "/v1/chat/completions").
    pub path: String,
    /// The HTTP method.
    pub method: HttpMethod,
    /// The content type (e.g. "application/json").
    pub content_type: String,
}

impl ProviderEndpoint {
    /// Create a new endpoint with the given path, method, and content type.
    pub fn new(
        path: impl Into<String>,
        method: HttpMethod,
        content_type: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            method,
            content_type: content_type.into(),
        }
    }

    /// Standard chat completions endpoint (`POST /v1/chat/completions`).
    pub fn chat_completions() -> Self {
        Self::new("/v1/chat/completions", HttpMethod::Post, "application/json")
    }

    /// Standard embeddings endpoint (`POST /v1/embeddings`).
    pub fn embeddings() -> Self {
        Self::new("/v1/embeddings", HttpMethod::Post, "application/json")
    }

    /// Standard completions endpoint (`POST /v1/completions`).
    pub fn completions() -> Self {
        Self::new("/v1/completions", HttpMethod::Post, "application/json")
    }
}

// ─── ApiRequest ───

/// A fully constructed API request ready to be sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    /// The full URL to send the request to.
    pub url: String,
    /// The HTTP method.
    pub method: HttpMethod,
    /// HTTP headers.
    pub headers: HashMap<String, String>,
    /// Optional JSON body.
    pub body: Option<Value>,
    /// URL query parameters.
    pub query_params: HashMap<String, String>,
}

impl ApiRequest {
    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ─── RequestBuilder ───

/// Builds an [`ApiRequest`] from a [`ProviderConfig`] and [`ProviderEndpoint`].
pub struct RequestBuilder {
    url: String,
    method: HttpMethod,
    headers: HashMap<String, String>,
    body: Option<Value>,
    query_params: HashMap<String, String>,
}

impl RequestBuilder {
    /// Create a new builder from config and endpoint.
    ///
    /// The URL is constructed by joining the config's `base_url` (or a default)
    /// with the endpoint path. Headers from the config are copied, and the
    /// content type is set from the endpoint.
    pub fn new(config: &ProviderConfig, endpoint: &ProviderEndpoint) -> Self {
        let base = config
            .base_url
            .as_deref()
            .unwrap_or("https://api.example.com");
        let url = format!("{}{}", base.trim_end_matches('/'), endpoint.path);

        let mut headers = config.headers.clone();
        headers.insert("Content-Type".to_string(), endpoint.content_type.clone());

        if let Some(ref key) = config.api_key {
            headers.insert("Authorization".to_string(), format!("Bearer {}", key));
        }

        if let Some(ref version) = config.api_version {
            headers.insert("X-API-Version".to_string(), version.clone());
        }

        Self {
            url,
            method: endpoint.method.clone(),
            headers,
            body: None,
            query_params: HashMap::new(),
        }
    }

    /// Set the JSON body.
    pub fn with_body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Add or override a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add a query parameter.
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    /// Consume the builder and produce the final [`ApiRequest`].
    pub fn build(self) -> ApiRequest {
        ApiRequest {
            url: self.url,
            method: self.method,
            headers: self.headers,
            body: self.body,
            query_params: self.query_params,
        }
    }
}

// ─── ApiResponse ───

/// A parsed API response with status classification helpers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// HTTP status code.
    pub status_code: u16,
    /// Parsed response body.
    pub body: Value,
    /// Response HTTP headers.
    pub headers: HashMap<String, String>,
    /// Request latency in milliseconds.
    pub latency_ms: u64,
}

impl ApiResponse {
    /// Returns `true` if the status code indicates success (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status_code)
    }

    /// Returns `true` if the response indicates rate limiting (429).
    pub fn is_rate_limited(&self) -> bool {
        self.status_code == 429
    }

    /// Returns `true` if the status code indicates a server error (5xx).
    pub fn is_server_error(&self) -> bool {
        (500..600).contains(&self.status_code)
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ─── ProviderRegistry ───

/// A collection of named [`ProviderConfig`] entries.
///
/// Allows registering, looking up, and enumerating provider configurations.
#[derive(Debug, Clone, Default)]
pub struct ProviderRegistry {
    configs: HashMap<String, ProviderConfig>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Register a provider config under the given name.
    pub fn register(&mut self, name: impl Into<String>, config: ProviderConfig) {
        self.configs.insert(name.into(), config);
    }

    /// Look up a provider config by name.
    pub fn get(&self, name: &str) -> Option<&ProviderConfig> {
        self.configs.get(name)
    }

    /// Return the names of all configured (see [`ProviderConfig::is_configured`])
    /// providers.
    pub fn configured_providers(&self) -> Vec<&str> {
        self.configs
            .iter()
            .filter(|(_, c)| c.is_configured())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

// ─── RetryPolicy ───

/// Exponential backoff retry policy with jitter cap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay in milliseconds before the first retry.
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds (cap).
    pub max_delay_ms: u64,
    /// Multiplicative backoff factor applied each attempt.
    pub backoff_factor: f64,
}

impl RetryPolicy {
    /// Create a new retry policy with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_factor: 2.0,
        }
    }

    /// Compute the delay (in ms) before retry attempt `attempt` (0-indexed).
    ///
    /// Uses exponential backoff: `initial_delay_ms * backoff_factor ^ attempt`,
    /// capped at `max_delay_ms`.
    pub fn compute_delay(&self, attempt: u32) -> u64 {
        let delay = self.initial_delay_ms as f64 * self.backoff_factor.powi(attempt as i32);
        let capped = delay.min(self.max_delay_ms as f64);
        capped as u64
    }

    /// Determine whether a request should be retried based on the response and
    /// the current attempt number.
    ///
    /// Retries on rate-limit (429) and server errors (5xx) as long as
    /// `attempt < max_retries`.
    pub fn should_retry(&self, response: &ApiResponse, attempt: u32) -> bool {
        if attempt >= self.max_retries {
            return false;
        }
        response.is_rate_limited() || response.is_server_error()
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ProviderError ───

/// Errors that can occur during provider API interactions.
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// Authentication failed (invalid or missing API key).
    AuthenticationError(String),
    /// Rate limit exceeded, with an optional retry-after duration.
    RateLimitError {
        /// Suggested wait time before retrying, in milliseconds.
        retry_after_ms: Option<u64>,
    },
    /// The provider returned a server error (5xx).
    ServerError(String),
    /// A network-level failure occurred.
    NetworkError(String),
    /// The response could not be parsed or was unexpected.
    InvalidResponse(String),
    /// The provider configuration is invalid or incomplete.
    ConfigError(String),
}

impl ProviderError {
    /// Returns `true` if this error is transient and the request may succeed
    /// on retry.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimitError { .. }
                | ProviderError::ServerError(_)
                | ProviderError::NetworkError(_)
        )
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::AuthenticationError(msg) => write!(f, "authentication error: {}", msg),
            ProviderError::RateLimitError { retry_after_ms } => {
                write!(f, "rate limit exceeded")?;
                if let Some(ms) = retry_after_ms {
                    write!(f, " (retry after {}ms)", ms)?;
                }
                Ok(())
            }
            ProviderError::ServerError(msg) => write!(f, "server error: {}", msg),
            ProviderError::NetworkError(msg) => write!(f, "network error: {}", msg),
            ProviderError::InvalidResponse(msg) => write!(f, "invalid response: {}", msg),
            ProviderError::ConfigError(msg) => write!(f, "config error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── HttpMethod ──

    #[test]
    fn test_http_method_as_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Put.as_str(), "PUT");
        assert_eq!(HttpMethod::Delete.as_str(), "DELETE");
        assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(format!("{}", HttpMethod::Get), "GET");
        assert_eq!(format!("{}", HttpMethod::Post), "POST");
        assert_eq!(format!("{}", HttpMethod::Delete), "DELETE");
    }

    #[test]
    fn test_http_method_equality() {
        assert_eq!(HttpMethod::Get, HttpMethod::Get);
        assert_ne!(HttpMethod::Get, HttpMethod::Post);
    }

    // ── ProviderConfig ──

    #[test]
    fn test_provider_config_defaults() {
        let c = ProviderConfig::new("openai");
        assert_eq!(c.provider_name, "openai");
        assert!(c.api_key.is_none());
        assert!(c.base_url.is_none());
        assert!(c.api_version.is_none());
        assert_eq!(c.timeout_ms, 30000);
        assert_eq!(c.max_retries, 3);
        assert!(c.headers.is_empty());
        assert!(c.extra.is_empty());
    }

    #[test]
    fn test_provider_config_builder() {
        let c = ProviderConfig::new("anthropic")
            .with_api_key("sk-test-key")
            .with_base_url("https://api.anthropic.com")
            .with_api_version("2024-01-01")
            .with_timeout_ms(60000)
            .with_max_retries(5)
            .with_header("X-Custom", "value")
            .with_extra("org_id", json!("org-123"));

        assert_eq!(c.api_key, Some("sk-test-key".to_string()));
        assert_eq!(c.base_url, Some("https://api.anthropic.com".to_string()));
        assert_eq!(c.api_version, Some("2024-01-01".to_string()));
        assert_eq!(c.timeout_ms, 60000);
        assert_eq!(c.max_retries, 5);
        assert_eq!(c.headers["X-Custom"], "value");
        assert_eq!(c.extra["org_id"], json!("org-123"));
    }

    #[test]
    fn test_provider_config_is_configured_with_key() {
        let c = ProviderConfig::new("test").with_api_key("key");
        assert!(c.is_configured());
    }

    #[test]
    fn test_provider_config_is_configured_with_url() {
        let c = ProviderConfig::new("test").with_base_url("http://localhost:8080");
        assert!(c.is_configured());
    }

    #[test]
    fn test_provider_config_not_configured() {
        let c = ProviderConfig::new("test");
        assert!(!c.is_configured());
    }

    #[test]
    fn test_provider_config_to_json_masks_api_key() {
        let c = ProviderConfig::new("openai").with_api_key("sk-super-secret-key-12345");
        let j = c.to_json();
        assert_eq!(j["api_key"], "sk-****");
        assert_eq!(j["provider_name"], "openai");
    }

    #[test]
    fn test_provider_config_to_json_no_key() {
        let c = ProviderConfig::new("test");
        let j = c.to_json();
        assert!(j["api_key"].is_null());
    }

    #[test]
    fn test_provider_config_to_json_preserves_other_fields() {
        let c = ProviderConfig::new("openai")
            .with_api_key("secret")
            .with_base_url("https://api.openai.com")
            .with_timeout_ms(5000);
        let j = c.to_json();
        assert_eq!(j["base_url"], "https://api.openai.com");
        assert_eq!(j["timeout_ms"], 5000);
        assert_eq!(j["api_key"], "sk-****");
    }

    // ── ProviderEndpoint ──

    #[test]
    fn test_endpoint_new() {
        let e = ProviderEndpoint::new("/v1/test", HttpMethod::Get, "text/plain");
        assert_eq!(e.path, "/v1/test");
        assert_eq!(e.method, HttpMethod::Get);
        assert_eq!(e.content_type, "text/plain");
    }

    #[test]
    fn test_endpoint_chat_completions() {
        let e = ProviderEndpoint::chat_completions();
        assert_eq!(e.path, "/v1/chat/completions");
        assert_eq!(e.method, HttpMethod::Post);
        assert_eq!(e.content_type, "application/json");
    }

    #[test]
    fn test_endpoint_embeddings() {
        let e = ProviderEndpoint::embeddings();
        assert_eq!(e.path, "/v1/embeddings");
        assert_eq!(e.method, HttpMethod::Post);
        assert_eq!(e.content_type, "application/json");
    }

    #[test]
    fn test_endpoint_completions() {
        let e = ProviderEndpoint::completions();
        assert_eq!(e.path, "/v1/completions");
        assert_eq!(e.method, HttpMethod::Post);
        assert_eq!(e.content_type, "application/json");
    }

    // ── RequestBuilder ──

    #[test]
    fn test_request_builder_basic() {
        let config = ProviderConfig::new("openai").with_base_url("https://api.openai.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.url, "https://api.openai.com/v1/chat/completions");
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.headers["Content-Type"], "application/json");
        assert!(req.body.is_none());
        assert!(req.query_params.is_empty());
    }

    #[test]
    fn test_request_builder_with_auth() {
        let config = ProviderConfig::new("openai")
            .with_api_key("sk-test")
            .with_base_url("https://api.openai.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.headers["Authorization"], "Bearer sk-test");
    }

    #[test]
    fn test_request_builder_with_api_version() {
        let config = ProviderConfig::new("anthropic")
            .with_api_version("2024-01-01")
            .with_base_url("https://api.anthropic.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.headers["X-API-Version"], "2024-01-01");
    }

    #[test]
    fn test_request_builder_with_body() {
        let config = ProviderConfig::new("openai").with_base_url("https://api.openai.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let body = json!({"model": "gpt-4", "messages": []});
        let req = RequestBuilder::new(&config, &endpoint)
            .with_body(body.clone())
            .build();

        assert_eq!(req.body, Some(body));
    }

    #[test]
    fn test_request_builder_with_header() {
        let config = ProviderConfig::new("test").with_base_url("https://api.test.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint)
            .with_header("X-Request-Id", "abc-123")
            .build();

        assert_eq!(req.headers["X-Request-Id"], "abc-123");
    }

    #[test]
    fn test_request_builder_with_query() {
        let config = ProviderConfig::new("test").with_base_url("https://api.test.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint)
            .with_query("api-version", "2024-01-01")
            .build();

        assert_eq!(req.query_params["api-version"], "2024-01-01");
    }

    #[test]
    fn test_request_builder_default_base_url() {
        let config = ProviderConfig::new("test");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.url, "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_request_builder_trailing_slash_base_url() {
        let config = ProviderConfig::new("test").with_base_url("https://api.test.com/");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.url, "https://api.test.com/v1/chat/completions");
    }

    #[test]
    fn test_request_builder_config_headers_propagated() {
        let config = ProviderConfig::new("test")
            .with_base_url("https://api.test.com")
            .with_header("X-Org", "my-org");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint).build();

        assert_eq!(req.headers["X-Org"], "my-org");
    }

    // ── ApiRequest ──

    #[test]
    fn test_api_request_to_json() {
        let req = ApiRequest {
            url: "https://api.test.com/v1/test".to_string(),
            method: HttpMethod::Post,
            headers: HashMap::new(),
            body: Some(json!({"key": "value"})),
            query_params: HashMap::new(),
        };
        let j = req.to_json();
        assert_eq!(j["url"], "https://api.test.com/v1/test");
        assert_eq!(j["method"], "Post");
        assert_eq!(j["body"]["key"], "value");
    }

    // ── ApiResponse ──

    fn make_response(status_code: u16) -> ApiResponse {
        ApiResponse {
            status_code,
            body: json!({}),
            headers: HashMap::new(),
            latency_ms: 100,
        }
    }

    #[test]
    fn test_api_response_is_success() {
        assert!(make_response(200).is_success());
        assert!(make_response(201).is_success());
        assert!(make_response(299).is_success());
        assert!(!make_response(300).is_success());
        assert!(!make_response(400).is_success());
        assert!(!make_response(500).is_success());
    }

    #[test]
    fn test_api_response_is_rate_limited() {
        assert!(make_response(429).is_rate_limited());
        assert!(!make_response(200).is_rate_limited());
        assert!(!make_response(500).is_rate_limited());
    }

    #[test]
    fn test_api_response_is_server_error() {
        assert!(make_response(500).is_server_error());
        assert!(make_response(502).is_server_error());
        assert!(make_response(503).is_server_error());
        assert!(make_response(599).is_server_error());
        assert!(!make_response(200).is_server_error());
        assert!(!make_response(429).is_server_error());
    }

    #[test]
    fn test_api_response_to_json() {
        let resp = ApiResponse {
            status_code: 200,
            body: json!({"result": "ok"}),
            headers: HashMap::new(),
            latency_ms: 42,
        };
        let j = resp.to_json();
        assert_eq!(j["status_code"], 200);
        assert_eq!(j["latency_ms"], 42);
        assert_eq!(j["body"]["result"], "ok");
    }

    // ── ProviderRegistry ──

    #[test]
    fn test_registry_new_empty() {
        let reg = ProviderRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ProviderRegistry::new();
        reg.register("openai", ProviderConfig::new("openai").with_api_key("key"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        let c = reg.get("openai").unwrap();
        assert_eq!(c.provider_name, "openai");
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = ProviderRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_register_replaces() {
        let mut reg = ProviderRegistry::new();
        reg.register("p", ProviderConfig::new("v1").with_api_key("k1"));
        reg.register("p", ProviderConfig::new("v2").with_api_key("k2"));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("p").unwrap().provider_name, "v2");
    }

    #[test]
    fn test_registry_configured_providers() {
        let mut reg = ProviderRegistry::new();
        reg.register("openai", ProviderConfig::new("openai").with_api_key("key"));
        reg.register(
            "local",
            ProviderConfig::new("local").with_base_url("http://localhost"),
        );
        reg.register("unconfigured", ProviderConfig::new("unconfigured"));

        let configured = reg.configured_providers();
        assert_eq!(configured.len(), 2);
        assert!(configured.contains(&"openai"));
        assert!(configured.contains(&"local"));
        assert!(!configured.contains(&"unconfigured"));
    }

    #[test]
    fn test_registry_multiple_providers() {
        let mut reg = ProviderRegistry::new();
        reg.register("a", ProviderConfig::new("a"));
        reg.register("b", ProviderConfig::new("b"));
        reg.register("c", ProviderConfig::new("c"));
        assert_eq!(reg.len(), 3);
    }

    // ── RetryPolicy ──

    #[test]
    fn test_retry_policy_defaults() {
        let p = RetryPolicy::new();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.initial_delay_ms, 1000);
        assert_eq!(p.max_delay_ms, 60000);
        assert_eq!(p.backoff_factor, 2.0);
    }

    #[test]
    fn test_retry_policy_default_trait() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
    }

    #[test]
    fn test_retry_policy_compute_delay_exponential() {
        let p = RetryPolicy {
            max_retries: 5,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_factor: 2.0,
        };
        assert_eq!(p.compute_delay(0), 1000);
        assert_eq!(p.compute_delay(1), 2000);
        assert_eq!(p.compute_delay(2), 4000);
        assert_eq!(p.compute_delay(3), 8000);
    }

    #[test]
    fn test_retry_policy_compute_delay_capped() {
        let p = RetryPolicy {
            max_retries: 10,
            initial_delay_ms: 1000,
            max_delay_ms: 5000,
            backoff_factor: 2.0,
        };
        assert_eq!(p.compute_delay(0), 1000);
        assert_eq!(p.compute_delay(1), 2000);
        assert_eq!(p.compute_delay(2), 4000);
        assert_eq!(p.compute_delay(3), 5000); // capped
        assert_eq!(p.compute_delay(10), 5000); // still capped
    }

    #[test]
    fn test_retry_policy_should_retry_rate_limited() {
        let p = RetryPolicy::new();
        assert!(p.should_retry(&make_response(429), 0));
        assert!(p.should_retry(&make_response(429), 2));
        assert!(!p.should_retry(&make_response(429), 3)); // max_retries reached
    }

    #[test]
    fn test_retry_policy_should_retry_server_error() {
        let p = RetryPolicy::new();
        assert!(p.should_retry(&make_response(500), 0));
        assert!(p.should_retry(&make_response(502), 1));
        assert!(p.should_retry(&make_response(503), 2));
        assert!(!p.should_retry(&make_response(500), 3));
    }

    #[test]
    fn test_retry_policy_should_not_retry_success() {
        let p = RetryPolicy::new();
        assert!(!p.should_retry(&make_response(200), 0));
    }

    #[test]
    fn test_retry_policy_should_not_retry_client_error() {
        let p = RetryPolicy::new();
        assert!(!p.should_retry(&make_response(400), 0));
        assert!(!p.should_retry(&make_response(401), 0));
        assert!(!p.should_retry(&make_response(404), 0));
    }

    // ── ProviderError ──

    #[test]
    fn test_provider_error_is_retryable() {
        assert!(!ProviderError::AuthenticationError("bad key".into()).is_retryable());
        assert!(ProviderError::RateLimitError {
            retry_after_ms: Some(1000)
        }
        .is_retryable());
        assert!(ProviderError::RateLimitError {
            retry_after_ms: None
        }
        .is_retryable());
        assert!(ProviderError::ServerError("internal".into()).is_retryable());
        assert!(ProviderError::NetworkError("timeout".into()).is_retryable());
        assert!(!ProviderError::InvalidResponse("bad json".into()).is_retryable());
        assert!(!ProviderError::ConfigError("missing key".into()).is_retryable());
    }

    #[test]
    fn test_provider_error_display_auth() {
        let e = ProviderError::AuthenticationError("invalid key".into());
        assert_eq!(e.to_string(), "authentication error: invalid key");
    }

    #[test]
    fn test_provider_error_display_rate_limit_with_retry() {
        let e = ProviderError::RateLimitError {
            retry_after_ms: Some(5000),
        };
        assert_eq!(e.to_string(), "rate limit exceeded (retry after 5000ms)");
    }

    #[test]
    fn test_provider_error_display_rate_limit_no_retry() {
        let e = ProviderError::RateLimitError {
            retry_after_ms: None,
        };
        assert_eq!(e.to_string(), "rate limit exceeded");
    }

    #[test]
    fn test_provider_error_display_server() {
        let e = ProviderError::ServerError("bad gateway".into());
        assert_eq!(e.to_string(), "server error: bad gateway");
    }

    #[test]
    fn test_provider_error_display_network() {
        let e = ProviderError::NetworkError("connection refused".into());
        assert_eq!(e.to_string(), "network error: connection refused");
    }

    #[test]
    fn test_provider_error_display_invalid_response() {
        let e = ProviderError::InvalidResponse("unexpected format".into());
        assert_eq!(e.to_string(), "invalid response: unexpected format");
    }

    #[test]
    fn test_provider_error_display_config() {
        let e = ProviderError::ConfigError("missing api_key".into());
        assert_eq!(e.to_string(), "config error: missing api_key");
    }

    #[test]
    fn test_provider_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ProviderError::ConfigError("test".into()));
        assert!(e.to_string().contains("config error"));
    }

    // ── Edge cases ──

    #[test]
    fn test_provider_config_is_configured_both() {
        let c = ProviderConfig::new("test")
            .with_api_key("key")
            .with_base_url("http://localhost");
        assert!(c.is_configured());
    }

    #[test]
    fn test_request_builder_multiple_queries() {
        let config = ProviderConfig::new("test").with_base_url("https://api.test.com");
        let endpoint = ProviderEndpoint::chat_completions();
        let req = RequestBuilder::new(&config, &endpoint)
            .with_query("a", "1")
            .with_query("b", "2")
            .build();
        assert_eq!(req.query_params.len(), 2);
        assert_eq!(req.query_params["a"], "1");
        assert_eq!(req.query_params["b"], "2");
    }

    #[test]
    fn test_api_response_boundary_status_codes() {
        assert!(!make_response(199).is_success());
        assert!(make_response(200).is_success());
        assert!(make_response(299).is_success());
        assert!(!make_response(300).is_success());
        assert!(!make_response(499).is_server_error());
        assert!(make_response(500).is_server_error());
    }

    #[test]
    fn test_retry_policy_zero_retries() {
        let p = RetryPolicy {
            max_retries: 0,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_factor: 2.0,
        };
        assert!(!p.should_retry(&make_response(500), 0));
    }

    #[test]
    fn test_retry_policy_compute_delay_zero_attempt() {
        let p = RetryPolicy::new();
        assert_eq!(p.compute_delay(0), 1000);
    }
}
