//! Shared HTTP client for the Langfuse exporter, prompt client, and scorer.

use std::time::Duration;

use base64::Engine;
use reqwest::Client;
use secrecy::ExposeSecret;

use crate::error::TraceError;

use super::config::LangfuseConfig;

/// Thin wrapper around `reqwest::Client` with Langfuse Basic Auth pre-applied.
#[derive(Clone)]
pub struct LangfuseHttp {
    client: Client,
    auth_header: String,
    base_url: String,
}

impl LangfuseHttp {
    /// Construct from config.
    pub fn new(cfg: &LangfuseConfig) -> Result<Self, TraceError> {
        let client = Client::builder()
            .timeout(cfg.timeout)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
        let creds = format!("{}:{}", cfg.public_key, cfg.secret_key.expose_secret());
        let token = base64::engine::general_purpose::STANDARD.encode(creds.as_bytes());
        let auth_header = format!("Basic {token}");
        Ok(Self {
            client,
            auth_header,
            base_url: cfg.host.trim_end_matches('/').to_string(),
        })
    }

    /// Authenticated request builder.
    pub fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        self.client
            .request(method, url)
            .header(reqwest::header::AUTHORIZATION, &self.auth_header)
    }
}
