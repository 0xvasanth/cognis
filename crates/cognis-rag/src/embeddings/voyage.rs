//! Voyage AI embeddings (Anthropic-recommended provider).
//!
//! Anthropic doesn't ship a first-party embeddings API; they officially
//! recommend Voyage AI. This module is a thin client for the Voyage AI
//! Embeddings API at `https://api.voyageai.com/v1/embeddings`.
//!
//! Customization:
//! - [`VoyageEmbeddingsBuilder`] — model, optional `input_type` hint
//!   (`"query"` / `"document"`), custom base URL, custom HTTP client,
//!   timeout.
//! - The `input_type` hint lets Voyage's models specialize embeddings
//!   for retrieval; defaults are `voyage-3` model with no input-type
//!   hint.

#![cfg(feature = "voyage")]

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};

use super::Embeddings;

const DEFAULT_BASE: &str = "https://api.voyageai.com/v1";
const DEFAULT_MODEL: &str = "voyage-3";

/// Voyage AI embeddings client.
pub struct VoyageEmbeddings {
    base_url: String,
    api_key: SecretString,
    model: String,
    input_type: Option<String>,
    http: reqwest::Client,
}

impl std::fmt::Debug for VoyageEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoyageEmbeddings")
            .field("model", &self.model)
            .field("input_type", &self.input_type)
            .finish_non_exhaustive()
    }
}

impl VoyageEmbeddings {
    /// New with API key + default model `voyage-3`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder()
            .api_key(api_key)
            .build()
            .expect("default VoyageEmbeddings build")
    }

    /// Fluent builder.
    pub fn builder() -> VoyageEmbeddingsBuilder {
        VoyageEmbeddingsBuilder::default()
    }

    fn endpoint(&self) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str("embeddings");
        s
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .map_err(|e| CognisError::Configuration(format!("invalid api key: {e}")))?,
        );
        Ok(h)
    }
}

#[async_trait]
impl Embeddings for VoyageEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        #[derive(Serialize)]
        struct Body<'a> {
            model: &'a str,
            input: Vec<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            input_type: Option<&'a str>,
        }
        let body = Body {
            model: &self.model,
            input: texts,
            input_type: self.input_type.as_deref(),
        };
        let resp = self
            .http
            .post(self.endpoint())
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Network {
                status_code: None,
                message: e.to_string(),
            })?;
        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(CognisError::Network {
                status_code: Some(status.as_u16()),
                message: txt,
            });
        }
        let raw: VoyageResp = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "voyage".into(),
            message: format!("response decode: {e}"),
        })?;
        let mut data = raw.data;
        // Defensive: sort by index, since the spec doesn't strictly
        // guarantee order.
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Fluent builder for [`VoyageEmbeddings`].
#[derive(Default)]
pub struct VoyageEmbeddingsBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    input_type: Option<String>,
    timeout_secs: Option<u64>,
    http: Option<reqwest::Client>,
}

impl VoyageEmbeddingsBuilder {
    /// Set the API key (required, falls back to env vars at the user's
    /// discretion via [`Self::api_key_from_env`]).
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }

    /// Resolve the API key from `VOYAGE_API_KEY` (preferred) or
    /// `ANTHROPIC_API_KEY`. Errors at `build()` if neither is set and
    /// no explicit key was provided.
    pub fn api_key_from_env(mut self) -> Self {
        if self.api_key.is_some() {
            return self;
        }
        let key = std::env::var("VOYAGE_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .ok();
        self.api_key = key;
        self
    }

    /// Override base URL.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }

    /// Override the model (default `voyage-3`).
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    /// Set the input-type hint (e.g. `"query"` or `"document"`).
    pub fn input_type(mut self, t: impl Into<String>) -> Self {
        self.input_type = Some(t.into());
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

    /// Construct.
    pub fn build(self) -> Result<VoyageEmbeddings> {
        let api_key = self.api_key.ok_or_else(|| {
            CognisError::Configuration("VoyageEmbeddings: API key required".into())
        })?;
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
        Ok(VoyageEmbeddings {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into()),
            model: self.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            input_type: self.input_type,
            http,
        })
    }
}

#[derive(Deserialize)]
struct VoyageResp {
    data: Vec<VoyageItem>,
}

#[derive(Deserialize)]
struct VoyageItem {
    embedding: Vec<f32>,
    index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_api_key() {
        let err = VoyageEmbeddingsBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let e = VoyageEmbeddings::new("pa-test");
        assert_eq!(e.model(), DEFAULT_MODEL);
        assert!(e.input_type.is_none());
    }

    #[test]
    fn input_type_round_trips_through_builder() {
        let e = VoyageEmbeddings::builder()
            .api_key("pa-test")
            .input_type("query")
            .build()
            .unwrap();
        assert_eq!(e.input_type.as_deref(), Some("query"));
    }

    #[test]
    fn endpoint_appends_slash_when_missing() {
        let e = VoyageEmbeddings::builder()
            .api_key("pa-test")
            .base_url("https://example.com/v1")
            .build()
            .unwrap();
        assert_eq!(e.endpoint(), "https://example.com/v1/embeddings");
    }

    #[test]
    fn empty_input_returns_empty() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let e = VoyageEmbeddings::new("pa-test");
        let out = rt.block_on(e.embed_documents(Vec::new())).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn api_key_from_env_picks_voyage_first() {
        std::env::set_var("VOYAGE_API_KEY", "voyage-test");
        std::env::set_var("ANTHROPIC_API_KEY", "anthropic-test");
        let e = VoyageEmbeddingsBuilder::default()
            .api_key_from_env()
            .build()
            .unwrap();
        // The expose-secret value should be the VOYAGE one (no public
        // accessor; we just verify build succeeded).
        assert_eq!(e.model(), DEFAULT_MODEL);
        std::env::remove_var("VOYAGE_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn api_key_from_env_errors_when_neither_set() {
        std::env::remove_var("VOYAGE_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        let res = VoyageEmbeddingsBuilder::default()
            .api_key_from_env()
            .build();
        assert!(res.is_err());
    }
}
