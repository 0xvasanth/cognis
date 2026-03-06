//! Web URL document loader.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::stream;
use reqwest::Client;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

use super::html::extract_text_from_html;

/// Fetches a URL and extracts its text content from HTML.
///
/// Uses `reqwest` to perform a GET request and then applies the same
/// HTML text extraction as [`super::html::HTMLLoader`].
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::web::WebBaseLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = WebBaseLoader::new("https://example.com");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct WebBaseLoader {
    url: String,
    client: Client,
}

impl WebBaseLoader {
    /// Create a new `WebBaseLoader` for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: Client::new(),
        }
    }

    /// Use a custom `reqwest::Client`.
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl BaseLoader for WebBaseLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| RustChainError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(RustChainError::Other(format!(
                "HTTP request returned status {}",
                status
            )));
        }

        let raw_html = response
            .text()
            .await
            .map_err(|e| RustChainError::Other(format!("Failed to read response body: {}", e)))?;

        let content = extract_text_from_html(&raw_html);

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(self.url.clone()));
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/html".to_string()),
        );

        let doc = Document::new(content).with_metadata(metadata);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_loader_construction() {
        let loader = WebBaseLoader::new("https://example.com");
        assert_eq!(loader.url, "https://example.com");
    }

    #[test]
    fn test_web_loader_with_custom_client() {
        let client = Client::builder()
            .user_agent("test-agent")
            .build()
            .unwrap();
        let loader = WebBaseLoader::new("https://example.com").with_client(client);
        assert_eq!(loader.url, "https://example.com");
    }

    #[tokio::test]
    async fn test_web_loader_invalid_url() {
        let loader = WebBaseLoader::new("http://localhost:1/nonexistent");
        let result = loader.load().await;
        assert!(result.is_err());
    }
}
