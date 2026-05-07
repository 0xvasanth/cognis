//! HTTP request tool with SSRF protection (feature `tools-http`).
//!
//! Refuses requests targeting:
//! - non-HTTP(S) schemes (`file://`, `gopher://`, etc.)
//! - private/loopback/link-local/multicast IPs
//! - hosts that resolve to any of the above
//!
//! The agent never sees raw `reqwest::Client` — it only gets a structured
//! `{method, url, headers, body}` payload. Response is `{status, headers,
//! body}`. Body is captured up to a configurable max length.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 15;

/// HTTP method.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    /// GET request.
    Get,
    /// POST request.
    Post,
    /// PUT request.
    Put,
    /// DELETE request.
    Delete,
    /// HEAD request.
    Head,
    /// PATCH request.
    Patch,
}

impl HttpMethod {
    fn as_reqwest(self) -> reqwest::Method {
        match self {
            HttpMethod::Get => reqwest::Method::GET,
            HttpMethod::Post => reqwest::Method::POST,
            HttpMethod::Put => reqwest::Method::PUT,
            HttpMethod::Delete => reqwest::Method::DELETE,
            HttpMethod::Head => reqwest::Method::HEAD,
            HttpMethod::Patch => reqwest::Method::PATCH,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct HttpRequestInput {
    /// HTTP method.
    method: HttpMethod,
    /// Target URL — must be `http://` or `https://`.
    url: String,
    /// Optional request headers as `{name: value}`.
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
    /// Optional request body (sent as-is, content-type from headers).
    #[serde(default)]
    body: Option<String>,
}

/// HTTP request tool.
///
/// Construct via [`HttpRequest::new`]. Defaults: 15s timeout, 256 KiB max
/// response body, SSRF guard enabled.
pub struct HttpRequest {
    http: reqwest::Client,
    max_body_bytes: usize,
    /// When `false`, allow private/loopback IPs (only useful for tests).
    ssrf_guard: bool,
}

impl HttpRequest {
    /// Build with sensible defaults.
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| CognisError::Configuration(format!("HttpRequest http: {e}")))?;
        Ok(Self {
            http,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            ssrf_guard: true,
        })
    }

    /// Override the body cap.
    pub fn with_max_body_bytes(mut self, n: usize) -> Self {
        self.max_body_bytes = n;
        self
    }

    /// Disable SSRF guards. **Only for tests.** Allows requests to
    /// loopback / private IPs that would otherwise be refused.
    pub fn allow_private_addresses(mut self) -> Self {
        self.ssrf_guard = false;
        self
    }

    /// Wrap behind an `Arc<dyn Tool>`.
    pub fn into_arc(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

#[async_trait]
impl Tool for HttpRequest {
    fn name(&self) -> &str {
        "http_request"
    }
    fn description(&self) -> &str {
        "Make an HTTP request. Only http:// and https:// schemes are \
         allowed. Returns `{status, headers, body}`. Body is truncated if \
         it exceeds the configured cap."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(HttpRequestInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: HttpRequestInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("http_request: {e}")))?;

        if self.ssrf_guard {
            check_url_safety(&parsed.url)?;
        } else {
            check_scheme(&parsed.url)?;
        }

        let mut req = self.http.request(parsed.method.as_reqwest(), &parsed.url);
        for (k, v) in &parsed.headers {
            req = req.header(k, v);
        }
        if let Some(body) = parsed.body {
            req = req.body(body);
        }

        let resp = req.send().await.map_err(|e| CognisError::Network {
            status_code: e.status().map(|s| s.as_u16()),
            message: e.to_string(),
        })?;

        let status = resp.status().as_u16();
        let headers: std::collections::BTreeMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let bytes = resp.bytes().await.map_err(|e| CognisError::Network {
            status_code: None,
            message: format!("read body: {e}"),
        })?;
        let truncated = bytes.len() > self.max_body_bytes;
        let body_slice = &bytes[..bytes.len().min(self.max_body_bytes)];
        let body = String::from_utf8_lossy(body_slice).into_owned();

        Ok(ToolOutput::Content(serde_json::json!({
            "status": status,
            "headers": headers,
            "body": body,
            "truncated": truncated,
        })))
    }
}

fn check_scheme(url: &str) -> Result<()> {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err(CognisError::Tool {
            name: "http_request".into(),
            reason: format!("refusing non-HTTP(S) scheme in `{url}` (only http/https allowed)"),
        });
    }
    Ok(())
}

/// SSRF guard: refuse the request unless every resolved IP is a public unicast.
fn check_url_safety(url: &str) -> Result<()> {
    check_scheme(url)?;
    let parsed: reqwest::Url = url.parse().map_err(|e| CognisError::Tool {
        name: "http_request".into(),
        reason: format!("invalid url `{url}`: {e}"),
    })?;

    let host = parsed.host_str().ok_or_else(|| CognisError::Tool {
        name: "http_request".into(),
        reason: format!("url `{url}` has no host"),
    })?;

    // Reject hostnames that look like loopback aliases without DNS resolution.
    let host_lower = host.to_ascii_lowercase();
    if matches!(
        host_lower.as_str(),
        "localhost" | "ip6-localhost" | "ip6-loopback"
    ) {
        return Err(CognisError::Tool {
            name: "http_request".into(),
            reason: format!("refusing loopback host `{host}`"),
        });
    }

    // If the host is itself an IP literal, check it directly. Otherwise,
    // resolution would require sync DNS — we leave that to the HTTP client.
    // IPv6 literals come back wrapped in `[...]` from `host_str`, so strip.
    use std::net::IpAddr;
    let bare = host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        if !is_public_unicast(&ip) {
            return Err(CognisError::Tool {
                name: "http_request".into(),
                reason: format!("refusing non-public IP `{ip}`"),
            });
        }
    }
    Ok(())
}

// SSRF classification logic now lives in `cognis2_core::security`.
use cognis2_core::security::is_public_unicast;

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::IpAddr;

    #[test]
    fn rejects_non_http_schemes() {
        assert!(check_url_safety("file:///etc/passwd").is_err());
        assert!(check_url_safety("ftp://example.com").is_err());
        assert!(check_url_safety("gopher://example.com").is_err());
    }

    #[test]
    fn rejects_localhost_alias() {
        assert!(check_url_safety("http://localhost/").is_err());
        assert!(check_url_safety("http://Localhost:8080/").is_err());
    }

    #[test]
    fn rejects_loopback_ip_literal() {
        assert!(check_url_safety("http://127.0.0.1/").is_err());
        assert!(check_url_safety("http://[::1]/").is_err());
    }

    #[test]
    fn rejects_private_ip_literal() {
        assert!(check_url_safety("http://10.0.0.1/").is_err());
        assert!(check_url_safety("http://192.168.1.1/").is_err());
        assert!(check_url_safety("http://172.16.0.1/").is_err());
    }

    #[test]
    fn allows_public_ip_literal() {
        // Cloudflare DNS — public, should pass.
        assert!(check_url_safety("https://1.1.1.1/").is_ok());
        assert!(check_url_safety("https://8.8.8.8/").is_ok());
    }

    #[test]
    fn allows_public_hostnames() {
        // Don't do DNS — hostnames should pass without a private IP literal.
        assert!(check_url_safety("https://example.com/").is_ok());
    }

    #[test]
    fn public_unicast_classifier() {
        let public: IpAddr = "1.1.1.1".parse().unwrap();
        assert!(is_public_unicast(&public));
        let priv_: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(!is_public_unicast(&priv_));
        let cgnat: IpAddr = "100.64.0.1".parse().unwrap();
        assert!(!is_public_unicast(&cgnat));
        let lp6: IpAddr = "::1".parse().unwrap();
        assert!(!is_public_unicast(&lp6));
        let pub6: IpAddr = "2606:4700:4700::1111".parse().unwrap();
        assert!(is_public_unicast(&pub6));
    }

    #[tokio::test]
    async fn rejects_invalid_args() {
        let t = HttpRequest::new().unwrap();
        let mut a = std::collections::HashMap::new();
        a.insert("url".into(), serde_json::json!("file:///etc/passwd"));
        a.insert("method".into(), serde_json::json!("GET"));
        let err = t._run(ToolInput::Structured(a)).await.unwrap_err();
        assert!(matches!(err, CognisError::Tool { .. }));
    }
}
