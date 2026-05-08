//! Web URL loader — fetches a single HTTP(S) URL into one [`Document`].
//!
//! Doesn't strip HTML — that's [`super::HtmlLoader`]'s job. Pipe this loader
//! into a splitter if you want chunked output.

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads the body of a single URL as a [`Document`].
pub struct WebLoader {
    url: String,
    timeout_secs: u64,
}

impl WebLoader {
    /// Construct.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            timeout_secs: 30,
        }
    }

    /// Override the request timeout (default 30s).
    pub fn with_timeout_secs(mut self, n: u64) -> Self {
        self.timeout_secs = n;
        self
    }
}

#[async_trait]
impl DocumentLoader for WebLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| CognisError::Configuration(format!("WebLoader http: {e}")))?;
        let resp = client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| CognisError::Network {
                status_code: e.status().map(|s| s.as_u16()),
                message: e.to_string(),
            })?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        // Surface non-2xx as a Network error rather than emitting a doc
        // whose `content` is the upstream's HTML error page.
        if !status.is_success() {
            return Err(CognisError::Network {
                status_code: Some(status.as_u16()),
                message: format!("WebLoader: {} returned HTTP {}", self.url, status.as_u16()),
            });
        }
        let status = status.as_u16();
        let body = resp.text().await.map_err(|e| CognisError::Network {
            status_code: None,
            message: format!("read body: {e}"),
        })?;
        let doc = Document::new(body)
            .with_metadata("source", self.url.clone())
            .with_metadata("status", serde_json::Value::Number(status.into()))
            .with_metadata("content_type", content_type);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}
