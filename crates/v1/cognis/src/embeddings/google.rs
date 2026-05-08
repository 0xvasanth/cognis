//! Google Generative AI embedding model implementation.
//!
//! Provides [`GoogleEmbeddings`], an implementation of the [`Embeddings`] trait
//! for the Google Generative Language API (Gemini).

use async_trait::async_trait;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use cognis_core::embeddings::Embeddings;
use cognis_core::error::{CognisError, Result};

/// Builder for constructing a [`GoogleEmbeddings`] instance.
#[derive(Debug)]
pub struct GoogleEmbeddingsBuilder {
    api_key: Option<SecretString>,
    model: Option<String>,
    task_type: Option<String>,
}

impl GoogleEmbeddingsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            api_key: None,
            model: None,
            task_type: None,
        }
    }

    /// Set the API key. Falls back to `GOOGLE_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the model name (default: `"text-embedding-004"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the task type (optional, e.g. `"RETRIEVAL_DOCUMENT"`, `"RETRIEVAL_QUERY"`).
    pub fn task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Build the [`GoogleEmbeddings`] instance.
    ///
    /// Returns an error if the API key cannot be resolved from the builder
    /// or environment.
    pub fn build(self) -> Result<GoogleEmbeddings> {
        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("GOOGLE_API_KEY").map_err(|_| {
                    CognisError::Other(
                        "api_key not provided and GOOGLE_API_KEY env var not set".into(),
                    )
                })?;
                SecretString::from(key)
            }
        };

        Ok(GoogleEmbeddings {
            api_key,
            model: self.model.unwrap_or_else(|| "text-embedding-004".into()),
            task_type: self.task_type,
            client: Client::new(),
        })
    }
}

impl Default for GoogleEmbeddingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Google Generative AI embeddings model.
///
/// Implements the Google Generative Language API for generating text embeddings
/// using models like `text-embedding-004`.
///
/// # Example
///
/// ```no_run
/// use cognis::embeddings::google::GoogleEmbeddings;
///
/// let embeddings = GoogleEmbeddings::builder()
///     .api_key("your-api-key")
///     .model("text-embedding-004")
///     .build()
///     .unwrap();
/// ```
pub struct GoogleEmbeddings {
    /// Secret API key.
    api_key: SecretString,
    /// The model identifier (e.g. "text-embedding-004").
    pub model: String,
    /// Optional task type for the embedding request.
    pub task_type: Option<String>,
    /// HTTP client.
    client: Client,
}

impl std::fmt::Debug for GoogleEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleEmbeddings")
            .field("model", &self.model)
            .field("task_type", &self.task_type)
            .finish()
    }
}

impl GoogleEmbeddings {
    /// Returns a new builder for `GoogleEmbeddings`.
    pub fn builder() -> GoogleEmbeddingsBuilder {
        GoogleEmbeddingsBuilder::new()
    }

    /// Build the JSON request payload for a single embedContent call.
    fn build_embed_content_payload(&self, text: &str) -> Value {
        let mut payload = json!({
            "model": format!("models/{}", self.model),
            "content": {
                "parts": [{"text": text}]
            }
        });

        if let Some(ref tt) = self.task_type {
            payload["taskType"] = json!(tt);
        }

        payload
    }

    /// Build the JSON request payload for a batchEmbedContents call.
    fn build_batch_payload(&self, texts: &[String]) -> Value {
        let requests: Vec<Value> = texts
            .iter()
            .map(|text| {
                let mut req = json!({
                    "model": format!("models/{}", self.model),
                    "content": {
                        "parts": [{"text": text}]
                    }
                });
                if let Some(ref tt) = self.task_type {
                    req["taskType"] = json!(tt);
                }
                req
            })
            .collect();

        json!({ "requests": requests })
    }

    /// Call the Google embedContent API for a single text.
    async fn call_embed_content(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}",
            self.model,
            self.api_key.expose_secret()
        );
        let payload = self.build_embed_content_payload(text);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| CognisError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(CognisError::HttpError { status, body });
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| CognisError::Other(format!("Failed to parse response JSON: {}", e)))?;

        Self::parse_embedding_values(&body)
    }

    /// Call the Google batchEmbedContents API for multiple texts.
    async fn call_batch_embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model,
            self.api_key.expose_secret()
        );
        let payload = self.build_batch_payload(&texts);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| CognisError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(CognisError::HttpError { status, body });
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| CognisError::Other(format!("Failed to parse response JSON: {}", e)))?;

        let embeddings_arr = body
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CognisError::Other(
                    "Missing 'embeddings' array in Google batchEmbedContents response".into(),
                )
            })?;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(embeddings_arr.len());
        for item in embeddings_arr {
            let values = item
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    CognisError::Other("Missing 'values' array in embedding object".into())
                })?;

            let vec = Self::parse_f32_array(values)?;
            embeddings.push(vec);
        }

        Ok(embeddings)
    }

    /// Parse the `{"embedding": {"values": [...]}}` response from embedContent.
    fn parse_embedding_values(body: &Value) -> Result<Vec<f32>> {
        let values = body
            .get("embedding")
            .and_then(|e| e.get("values"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CognisError::Other(
                    "Missing 'embedding.values' array in Google embedContent response".into(),
                )
            })?;

        Self::parse_f32_array(values)
    }

    /// Parse a JSON array of numbers into `Vec<f32>`.
    fn parse_f32_array(arr: &[Value]) -> Result<Vec<f32>> {
        arr.iter()
            .map(|v| {
                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                    CognisError::Other("Non-numeric value in embedding array".into())
                })
            })
            .collect()
    }
}

#[async_trait]
impl Embeddings for GoogleEmbeddings {
    /// Embed a list of documents using the Google batchEmbedContents API.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_batch_embed(texts).await
    }

    /// Embed a single query text using the Google embedContent API.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.call_embed_content(text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "text-embedding-004");
        assert!(embeddings.task_type.is_none());
    }

    #[test]
    fn test_builder_custom_values() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .model("embedding-001")
            .task_type("RETRIEVAL_DOCUMENT")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "embedding-001");
        assert_eq!(embeddings.task_type, Some("RETRIEVAL_DOCUMENT".to_string()));
    }

    #[test]
    fn test_builder_requires_api_key() {
        // Clear env var to ensure it's not set
        std::env::remove_var("GOOGLE_API_KEY");
        let result = GoogleEmbeddings::builder().build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("GOOGLE_API_KEY"));
    }

    #[test]
    fn test_build_embed_content_payload_without_task_type() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let payload = embeddings.build_embed_content_payload("hello world");

        assert_eq!(payload["model"], "models/text-embedding-004");
        assert_eq!(payload["content"]["parts"][0]["text"], "hello world");
        assert!(payload.get("taskType").is_none());
    }

    #[test]
    fn test_build_embed_content_payload_with_task_type() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .task_type("RETRIEVAL_QUERY")
            .build()
            .unwrap();

        let payload = embeddings.build_embed_content_payload("hello");

        assert_eq!(payload["model"], "models/text-embedding-004");
        assert_eq!(payload["content"]["parts"][0]["text"], "hello");
        assert_eq!(payload["taskType"], "RETRIEVAL_QUERY");
    }

    #[test]
    fn test_build_batch_payload() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .task_type("RETRIEVAL_DOCUMENT")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let payload = embeddings.build_batch_payload(&texts);

        let requests = payload["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["model"], "models/text-embedding-004");
        assert_eq!(requests[0]["content"]["parts"][0]["text"], "hello");
        assert_eq!(requests[0]["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(requests[1]["content"]["parts"][0]["text"], "world");
    }

    #[test]
    fn test_build_batch_payload_without_task_type() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string()];
        let payload = embeddings.build_batch_payload(&texts);

        let requests = payload["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].get("taskType").is_none());
    }

    #[test]
    fn test_parse_embedding_values() {
        let body = json!({
            "embedding": {
                "values": [0.1, 0.2, 0.3]
            }
        });

        let result = GoogleEmbeddings::parse_embedding_values(&body).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.1).abs() < 1e-6);
        assert!((result[1] - 0.2).abs() < 1e-6);
        assert!((result[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_parse_embedding_values_missing() {
        let body = json!({"error": "something"});
        let result = GoogleEmbeddings::parse_embedding_values(&body);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_f32_array() {
        let arr = vec![json!(1.0), json!(2.5), json!(3.7)];
        let result = GoogleEmbeddings::parse_f32_array(&arr).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.5).abs() < 1e-6);
        assert!((result[2] - 3.7).abs() < 1e-6);
    }

    #[test]
    fn test_parse_f32_array_non_numeric() {
        let arr = vec![json!(1.0), json!("not a number")];
        let result = GoogleEmbeddings::parse_f32_array(&arr);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embed_documents_empty() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let result = embeddings.embed_documents(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_debug_does_not_leak_api_key() {
        let embeddings = GoogleEmbeddings::builder()
            .api_key("super-secret-key")
            .build()
            .unwrap();

        let debug_str = format!("{:?}", embeddings);
        assert!(!debug_str.contains("super-secret-key"));
        assert!(debug_str.contains("GoogleEmbeddings"));
        assert!(debug_str.contains("text-embedding-004"));
    }
}
