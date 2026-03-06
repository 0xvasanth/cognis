//! Voyage AI embedding model implementation (Anthropic ecosystem).
//!
//! Provides [`VoyageEmbeddings`], an implementation of the [`Embeddings`] trait
//! for the Voyage AI Embeddings API. Anthropic recommends Voyage AI for embeddings.

use async_trait::async_trait;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::{Result, RustChainError};

/// Builder for constructing a [`VoyageEmbeddings`] instance.
#[derive(Debug)]
pub struct VoyageEmbeddingsBuilder {
    api_key: Option<SecretString>,
    model: Option<String>,
    input_type: Option<String>,
}

impl VoyageEmbeddingsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            api_key: None,
            model: None,
            input_type: None,
        }
    }

    /// Set the API key. Falls back to `VOYAGE_API_KEY` or `ANTHROPIC_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the model name (default: `"voyage-3"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the input type (optional, e.g. `"query"`, `"document"`).
    pub fn input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = Some(input_type.into());
        self
    }

    /// Build the [`VoyageEmbeddings`] instance.
    ///
    /// Returns an error if the API key cannot be resolved from the builder
    /// or environment.
    pub fn build(self) -> Result<VoyageEmbeddings> {
        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("VOYAGE_API_KEY")
                    .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                    .map_err(|_| {
                        RustChainError::Other(
                            "api_key not provided and neither VOYAGE_API_KEY nor ANTHROPIC_API_KEY env var is set".into(),
                        )
                    })?;
                SecretString::from(key)
            }
        };

        Ok(VoyageEmbeddings {
            api_key,
            model: self.model.unwrap_or_else(|| "voyage-3".into()),
            input_type: self.input_type,
            client: Client::new(),
        })
    }
}

impl Default for VoyageEmbeddingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Voyage AI embeddings model (Anthropic ecosystem).
///
/// Implements the Voyage AI Embeddings API for generating text embeddings.
/// Anthropic recommends Voyage AI as the embeddings provider for the Anthropic
/// ecosystem.
///
/// # Example
///
/// ```no_run
/// use rustchain::embeddings::anthropic::VoyageEmbeddings;
///
/// let embeddings = VoyageEmbeddings::builder()
///     .api_key("pa-...")
///     .model("voyage-3")
///     .build()
///     .unwrap();
/// ```
pub struct VoyageEmbeddings {
    /// Secret API key.
    api_key: SecretString,
    /// The model identifier (e.g. "voyage-3").
    pub model: String,
    /// Optional input type for the embedding request ("query" or "document").
    pub input_type: Option<String>,
    /// HTTP client.
    client: Client,
}

impl std::fmt::Debug for VoyageEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoyageEmbeddings")
            .field("model", &self.model)
            .field("input_type", &self.input_type)
            .finish()
    }
}

impl VoyageEmbeddings {
    /// Returns a new builder for `VoyageEmbeddings`.
    pub fn builder() -> VoyageEmbeddingsBuilder {
        VoyageEmbeddingsBuilder::new()
    }

    /// Build the JSON request payload for the Voyage AI Embeddings API.
    fn build_payload(&self, texts: &[String], input_type: Option<&str>) -> Value {
        let mut payload = json!({
            "model": self.model,
            "input": texts,
        });

        // Use the provided input_type override, falling back to the instance default.
        let effective_input_type = input_type.or(self.input_type.as_deref());
        if let Some(it) = effective_input_type {
            payload["input_type"] = json!(it);
        }

        payload
    }

    /// Call the Voyage AI Embeddings API and return raw embedding vectors.
    async fn call_api(
        &self,
        texts: Vec<String>,
        input_type: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let url = "https://api.voyageai.com/v1/embeddings";
        let payload = self.build_payload(&texts, input_type);

        let response = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| RustChainError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(RustChainError::HttpError { status, body });
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| RustChainError::Other(format!("Failed to parse response JSON: {}", e)))?;

        Self::parse_response(&body)
    }

    /// Parse the Voyage AI embeddings response.
    ///
    /// Expected format: `{"data": [{"embedding": [f32, ...], "index": 0}, ...], ...}`
    fn parse_response(body: &Value) -> Result<Vec<Vec<f32>>> {
        let data = body.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            RustChainError::Other("Missing 'data' array in Voyage AI embeddings response".into())
        })?;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    RustChainError::Other("Missing 'embedding' array in response data item".into())
                })?;

            let vec: Vec<f32> = embedding
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        RustChainError::Other("Non-numeric value in embedding array".into())
                    })
                })
                .collect::<Result<Vec<f32>>>()?;

            embeddings.push(vec);
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl Embeddings for VoyageEmbeddings {
    /// Embed a list of documents using the Voyage AI Embeddings API.
    ///
    /// Automatically sets `input_type` to `"document"` for optimal retrieval performance.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(texts, Some("document")).await
    }

    /// Embed a single query text using the Voyage AI Embeddings API.
    ///
    /// Automatically sets `input_type` to `"query"` for optimal retrieval performance.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.call_api(vec![text.to_string()], Some("query")).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| RustChainError::Other("Empty embedding response for query".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "voyage-3");
        assert!(embeddings.input_type.is_none());
    }

    #[test]
    fn test_builder_custom_values() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .model("voyage-3-lite")
            .input_type("query")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "voyage-3-lite");
        assert_eq!(embeddings.input_type, Some("query".to_string()));
    }

    #[test]
    fn test_builder_requires_api_key() {
        // Clear env vars to ensure they are not set
        std::env::remove_var("VOYAGE_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = VoyageEmbeddings::builder().build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("VOYAGE_API_KEY"));
        assert!(err.contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn test_build_payload_for_query() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let texts = vec!["what is machine learning?".to_string()];
        let payload = embeddings.build_payload(&texts, Some("query"));

        assert_eq!(payload["model"], "voyage-3");
        assert_eq!(payload["input"], json!(["what is machine learning?"]));
        assert_eq!(payload["input_type"], "query");
    }

    #[test]
    fn test_build_payload_for_documents() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let payload = embeddings.build_payload(&texts, Some("document"));

        assert_eq!(payload["model"], "voyage-3");
        assert_eq!(payload["input"], json!(["hello", "world"]));
        assert_eq!(payload["input_type"], "document");
    }

    #[test]
    fn test_parse_response() {
        let body = json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "voyage-3",
            "usage": {"total_tokens": 10}
        });

        let result = VoyageEmbeddings::parse_response(&body).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        assert!((result[0][0] - 0.1).abs() < 1e-6);
        assert!((result[1][2] - 0.6).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_embed_documents_empty() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let result = embeddings.embed_documents(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_api_key_from_env_voyage() {
        // Set VOYAGE_API_KEY and verify it's used
        std::env::set_var("VOYAGE_API_KEY", "env-voyage-key");
        std::env::remove_var("ANTHROPIC_API_KEY");

        let embeddings = VoyageEmbeddings::builder().build().unwrap();
        // Just verify it built successfully (key was found from env)
        assert_eq!(embeddings.model, "voyage-3");

        std::env::remove_var("VOYAGE_API_KEY");
    }

    #[test]
    fn test_api_key_from_env_anthropic_fallback() {
        // Only set ANTHROPIC_API_KEY, not VOYAGE_API_KEY
        std::env::remove_var("VOYAGE_API_KEY");
        std::env::set_var("ANTHROPIC_API_KEY", "env-anthropic-key");

        let embeddings = VoyageEmbeddings::builder().build().unwrap();
        assert_eq!(embeddings.model, "voyage-3");

        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn test_custom_model_name() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .model("voyage-code-3")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "voyage-code-3");

        let payload = embeddings.build_payload(&["code snippet".to_string()], Some("document"));
        assert_eq!(payload["model"], "voyage-code-3");
    }

    #[test]
    fn test_debug_does_not_leak_api_key() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("super-secret-key")
            .build()
            .unwrap();

        let debug_str = format!("{:?}", embeddings);
        assert!(!debug_str.contains("super-secret-key"));
        assert!(debug_str.contains("VoyageEmbeddings"));
        assert!(debug_str.contains("voyage-3"));
    }

    #[test]
    fn test_build_payload_without_input_type() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string()];
        let payload = embeddings.build_payload(&texts, None);

        assert_eq!(payload["model"], "voyage-3");
        assert_eq!(payload["input"], json!(["hello"]));
        assert!(payload.get("input_type").is_none());
    }

    #[test]
    fn test_build_payload_with_builder_input_type_default() {
        let embeddings = VoyageEmbeddings::builder()
            .api_key("test-key")
            .input_type("document")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string()];
        // When no override is passed, the builder default is used
        let payload = embeddings.build_payload(&texts, None);
        assert_eq!(payload["input_type"], "document");

        // Override takes precedence
        let payload = embeddings.build_payload(&texts, Some("query"));
        assert_eq!(payload["input_type"], "query");
    }

    #[test]
    fn test_parse_response_missing_data() {
        let body = json!({"error": "something"});
        let result = VoyageEmbeddings::parse_response(&body);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("data"));
    }
}
