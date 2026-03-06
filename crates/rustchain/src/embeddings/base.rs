//! Embeddings factory and provider registry.
//!
//! Mirrors Python `langchain.embeddings.base` — provides the `init_embeddings`
//! factory function and model string parsing for embedding providers.

use std::collections::HashMap;

use serde_json::Value;

use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::{Result, RustChainError};

/// Configuration for an embedding provider implementation.
#[derive(Debug, Clone)]
pub struct EmbeddingProviderConfig {
    /// The module/crate name for the provider.
    pub module_name: String,
    /// The class/struct name for the embeddings implementation.
    pub class_name: String,
}

impl EmbeddingProviderConfig {
    /// Create a new embedding provider configuration.
    pub fn new(module_name: impl Into<String>, class_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            class_name: class_name.into(),
        }
    }
}

/// Resolved configuration for an embeddings model.
///
/// Returned by `init_embeddings` with all provider and model information resolved.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// The resolved provider key (e.g. "openai", "cohere").
    pub provider: String,
    /// The model name (e.g. "text-embedding-3-small").
    pub model_name: String,
    /// The provider configuration (module/class info).
    pub provider_config: EmbeddingProviderConfig,
    /// Additional keyword arguments to pass to the model constructor.
    pub kwargs: HashMap<String, Value>,
}

/// Returns the built-in embedding provider registry.
///
/// This is the Rust equivalent of Python's `_BUILTIN_EMBEDDING_PROVIDERS` dict.
pub fn builtin_embedding_providers() -> HashMap<String, EmbeddingProviderConfig> {
    let mut m = HashMap::new();
    m.insert(
        "openai".to_string(),
        EmbeddingProviderConfig::new("rustchain_openai", "OpenAIEmbeddings"),
    );
    m.insert(
        "azure_openai".to_string(),
        EmbeddingProviderConfig::new("rustchain_openai", "AzureOpenAIEmbeddings"),
    );
    m.insert(
        "bedrock".to_string(),
        EmbeddingProviderConfig::new("rustchain_aws", "BedrockEmbeddings"),
    );
    m.insert(
        "cohere".to_string(),
        EmbeddingProviderConfig::new("rustchain_cohere", "CohereEmbeddings"),
    );
    m.insert(
        "google_genai".to_string(),
        EmbeddingProviderConfig::new("rustchain_google_genai", "GoogleGenerativeAIEmbeddings"),
    );
    m.insert(
        "google_vertexai".to_string(),
        EmbeddingProviderConfig::new("rustchain_google_vertexai", "VertexAIEmbeddings"),
    );
    m.insert(
        "huggingface".to_string(),
        EmbeddingProviderConfig::new("rustchain_huggingface", "HuggingFaceEmbeddings"),
    );
    m.insert(
        "mistralai".to_string(),
        EmbeddingProviderConfig::new("rustchain_mistralai", "MistralAIEmbeddings"),
    );
    m.insert(
        "ollama".to_string(),
        EmbeddingProviderConfig::new("rustchain_ollama", "OllamaEmbeddings"),
    );
    m.insert(
        "anthropic".to_string(),
        EmbeddingProviderConfig::new("rustchain_anthropic", "VoyageEmbeddings"),
    );
    m.insert(
        "voyage".to_string(),
        EmbeddingProviderConfig::new("rustchain_anthropic", "VoyageEmbeddings"),
    );
    m
}

/// Parse a model string in "provider:model_name" format for embeddings.
///
/// If the string contains a colon, splits into (provider, model_name).
/// Unlike chat models, embedding model names are less standardized,
/// so inference without a provider prefix is limited.
///
/// # Examples
/// ```
/// use rustchain::embeddings::base::parse_model_string;
///
/// let (provider, model) = parse_model_string("openai:text-embedding-3-small").unwrap();
/// assert_eq!(provider, "openai");
/// assert_eq!(model, "text-embedding-3-small");
/// ```
pub fn parse_model_string(model: &str) -> Result<(String, String)> {
    if let Some((provider, model_name)) = model.split_once(':') {
        if provider.is_empty() {
            return Err(RustChainError::Other(
                "Provider name cannot be empty in 'provider:model' format".to_string(),
            ));
        }
        if model_name.is_empty() {
            return Err(RustChainError::Other(
                "Model name cannot be empty in 'provider:model' format".to_string(),
            ));
        }
        Ok((provider.to_string(), model_name.to_string()))
    } else {
        // Try to infer provider from model name
        if let Some(provider) = attempt_infer_embedding_provider(model) {
            Ok((provider, model.to_string()))
        } else {
            Err(RustChainError::Other(format!(
                "Unable to infer embedding provider for model '{}'. \
                 Use 'provider:model' format (e.g., 'openai:text-embedding-3-small').",
                model
            )))
        }
    }
}

/// Attempt to infer the embedding provider from a model name.
///
/// Returns `Some(provider_key)` if a match is found, `None` otherwise.
///
/// # Known prefix mappings
/// - `text-embedding-` -> "openai"
/// - `embed-` -> "cohere"
/// - `amazon.titan-embed` -> "bedrock"
/// - `models/embedding`, `models/text-embedding` -> "google_genai"
/// - `mistral-embed` -> "mistralai"
fn attempt_infer_embedding_provider(model_name: &str) -> Option<String> {
    let lower = model_name.to_lowercase();

    // OpenAI embeddings
    if lower.starts_with("text-embedding-") {
        return Some("openai".to_string());
    }

    // Cohere embeddings
    if lower.starts_with("embed-") {
        return Some("cohere".to_string());
    }

    // AWS Bedrock embeddings
    if lower.starts_with("amazon.titan-embed") {
        return Some("bedrock".to_string());
    }

    // Google GenAI embeddings
    if lower.starts_with("models/embedding") || lower.starts_with("models/text-embedding") {
        return Some("google_genai".to_string());
    }

    // Mistral embeddings
    if lower.starts_with("mistral-embed") {
        return Some("mistralai".to_string());
    }

    // Voyage AI embeddings (Anthropic ecosystem)
    if lower.starts_with("voyage-") {
        return Some("anthropic".to_string());
    }

    None
}

/// Initialize an embeddings model by provider and model name.
///
/// Returns an `EmbeddingConfig` with all resolved provider and model information.
/// Since provider crates are not dynamically loaded in Rust, this function resolves
/// the configuration that can be used to construct the appropriate model.
///
/// # Arguments
/// - `model` — Model string, either "provider:model_name" or just "model_name" (inferred).
/// - `provider` — Optional explicit provider override (bypasses inference).
/// - `kwargs` — Optional additional keyword arguments for the model constructor.
///
/// # Errors
/// Returns an error if the provider is unknown or cannot be inferred.
pub fn init_embeddings(
    model: &str,
    provider: Option<&str>,
    kwargs: Option<HashMap<String, Value>>,
) -> Result<EmbeddingConfig> {
    let providers = builtin_embedding_providers();
    let kwargs = kwargs.unwrap_or_default();

    let (resolved_provider, model_name) = if let Some(explicit_provider) = provider {
        // Explicit provider given — use it directly
        let model_name = if let Some((_p, m)) = model.split_once(':') {
            m.to_string()
        } else {
            model.to_string()
        };
        (explicit_provider.to_string(), model_name)
    } else {
        parse_model_string(model)?
    };

    if let Some(config) = providers.get(&resolved_provider) {
        Ok(EmbeddingConfig {
            provider: resolved_provider.clone(),
            model_name,
            provider_config: config.clone(),
            kwargs,
        })
    } else {
        Err(RustChainError::Other(format!(
            "Unknown embedding provider '{}'. Available providers: {:?}",
            resolved_provider,
            providers.keys().collect::<Vec<_>>()
        )))
    }
}

/// Create an embeddings provider instance by name.
///
/// Supports the following providers:
/// - `"openai"` — requires the `openai` feature
/// - `"ollama"` — requires the `ollama` feature
///
/// # Arguments
/// - `provider` — Provider name (e.g. "openai", "ollama").
/// - `kwargs` — Optional keyword arguments for the provider constructor.
///   For OpenAI: `api_key`, `model`, `dimensions`, `base_url`.
///   For Ollama: `model`, `base_url`.
///
/// # Errors
/// Returns an error if the provider is unknown or not enabled via feature flag.
pub fn create_embeddings(
    provider: &str,
    kwargs: Option<HashMap<String, Value>>,
) -> Result<Box<dyn Embeddings>> {
    let kwargs = kwargs.unwrap_or_default();

    match provider {
        #[cfg(feature = "openai")]
        "openai" => {
            let mut builder = super::openai::OpenAIEmbeddings::builder();

            if let Some(key) = kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(model) = kwargs.get("model").and_then(|v| v.as_str()) {
                builder = builder.model(model);
            }
            if let Some(dims) = kwargs.get("dimensions").and_then(|v| v.as_u64()) {
                builder = builder.dimensions(dims as usize);
            }
            if let Some(url) = kwargs.get("base_url").and_then(|v| v.as_str()) {
                builder = builder.base_url(url);
            }

            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "ollama")]
        "ollama" => {
            let mut builder = super::ollama::OllamaEmbeddings::builder();

            if let Some(model) = kwargs.get("model").and_then(|v| v.as_str()) {
                builder = builder.model(model);
            }
            if let Some(url) = kwargs.get("base_url").and_then(|v| v.as_str()) {
                builder = builder.base_url(url);
            }

            Ok(Box::new(builder.build()))
        }
        #[cfg(feature = "google")]
        "google" | "google_genai" => {
            let mut builder = super::google::GoogleEmbeddings::builder();

            if let Some(key) = kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(model) = kwargs.get("model").and_then(|v| v.as_str()) {
                builder = builder.model(model);
            }
            if let Some(tt) = kwargs.get("task_type").and_then(|v| v.as_str()) {
                builder = builder.task_type(tt);
            }

            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "anthropic")]
        "anthropic" | "voyage" => {
            let mut builder = super::anthropic::VoyageEmbeddings::builder();

            if let Some(key) = kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(model) = kwargs.get("model").and_then(|v| v.as_str()) {
                builder = builder.model(model);
            }
            if let Some(it) = kwargs.get("input_type").and_then(|v| v.as_str()) {
                builder = builder.input_type(it);
            }

            Ok(Box::new(builder.build()?))
        }
        _ => Err(RustChainError::Other(format!(
            "Unknown or disabled embedding provider '{}'. \
             Make sure the corresponding feature flag is enabled.",
            provider
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string_with_provider() {
        let (provider, model) = parse_model_string("openai:text-embedding-3-small").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "text-embedding-3-small");
    }

    #[test]
    fn test_parse_model_string_cohere() {
        let (provider, model) = parse_model_string("cohere:embed-english-v3.0").unwrap();
        assert_eq!(provider, "cohere");
        assert_eq!(model, "embed-english-v3.0");
    }

    #[test]
    fn test_parse_model_string_infer_openai() {
        let (provider, model) = parse_model_string("text-embedding-3-small").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "text-embedding-3-small");
    }

    #[test]
    fn test_parse_model_string_infer_cohere() {
        let (provider, model) = parse_model_string("embed-english-v3.0").unwrap();
        assert_eq!(provider, "cohere");
        assert_eq!(model, "embed-english-v3.0");
    }

    #[test]
    fn test_parse_model_string_infer_bedrock() {
        let (provider, model) = parse_model_string("amazon.titan-embed-text-v1").unwrap();
        assert_eq!(provider, "bedrock");
        assert_eq!(model, "amazon.titan-embed-text-v1");
    }

    #[test]
    fn test_parse_model_string_infer_mistral() {
        let (provider, model) = parse_model_string("mistral-embed").unwrap();
        assert_eq!(provider, "mistralai");
        assert_eq!(model, "mistral-embed");
    }

    #[test]
    fn test_parse_model_string_infer_google() {
        let (provider, model) = parse_model_string("models/embedding-001").unwrap();
        assert_eq!(provider, "google_genai");
        assert_eq!(model, "models/embedding-001");
    }

    #[test]
    fn test_parse_model_string_unknown() {
        let result = parse_model_string("some-random-embedding");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unable to infer embedding provider"));
    }

    #[test]
    fn test_parse_model_string_empty_provider() {
        let result = parse_model_string(":text-embedding-3-small");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Provider name cannot be empty"));
    }

    #[test]
    fn test_parse_model_string_empty_model() {
        let result = parse_model_string("openai:");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Model name cannot be empty"));
    }

    #[test]
    fn test_builtin_embedding_providers_contains_expected() {
        let providers = builtin_embedding_providers();
        assert!(providers.contains_key("openai"));
        assert!(providers.contains_key("azure_openai"));
        assert!(providers.contains_key("bedrock"));
        assert!(providers.contains_key("cohere"));
        assert!(providers.contains_key("google_genai"));
        assert!(providers.contains_key("google_vertexai"));
        assert!(providers.contains_key("huggingface"));
        assert!(providers.contains_key("mistralai"));
        assert!(providers.contains_key("ollama"));
        assert!(providers.contains_key("anthropic"));
        assert!(providers.contains_key("voyage"));
        assert_eq!(providers.len(), 11);
    }

    #[test]
    fn test_init_embeddings_known_provider() {
        let result = init_embeddings("openai:text-embedding-3-small", None, None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model_name, "text-embedding-3-small");
        assert_eq!(config.provider_config.module_name, "rustchain_openai");
        assert_eq!(config.provider_config.class_name, "OpenAIEmbeddings");
    }

    #[test]
    fn test_init_embeddings_inferred_provider() {
        let result = init_embeddings("text-embedding-3-large", None, None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model_name, "text-embedding-3-large");
    }

    #[test]
    fn test_init_embeddings_explicit_provider() {
        let result = init_embeddings("my-custom-embedding", Some("cohere"), None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "cohere");
        assert_eq!(config.model_name, "my-custom-embedding");
        assert_eq!(config.provider_config.class_name, "CohereEmbeddings");
    }

    #[test]
    fn test_init_embeddings_with_kwargs() {
        let mut kwargs = HashMap::new();
        kwargs.insert("dimensions".to_string(), serde_json::json!(256));
        let result = init_embeddings("openai:text-embedding-3-small", None, Some(kwargs));
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.kwargs["dimensions"], serde_json::json!(256));
    }

    #[test]
    fn test_init_embeddings_unknown_provider() {
        let result = init_embeddings("unknown_provider:some-model", None, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown embedding provider"));
    }

    #[test]
    fn test_init_embeddings_azure_openai() {
        let result = init_embeddings("my-deployment", Some("azure_openai"), None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "azure_openai");
        assert_eq!(config.provider_config.class_name, "AzureOpenAIEmbeddings");
    }

    #[test]
    fn test_embedding_provider_config_fields() {
        let config = EmbeddingProviderConfig::new("my_crate", "MyEmbeddings");
        assert_eq!(config.module_name, "my_crate");
        assert_eq!(config.class_name, "MyEmbeddings");
    }

    #[test]
    fn test_attempt_infer_embedding_provider_none() {
        assert_eq!(attempt_infer_embedding_provider("totally-unknown"), None);
    }

    #[test]
    fn test_attempt_infer_embedding_provider_case_insensitive() {
        assert_eq!(
            attempt_infer_embedding_provider("Text-Embedding-3-Small"),
            Some("openai".to_string())
        );
        assert_eq!(
            attempt_infer_embedding_provider("Embed-English-v3.0"),
            Some("cohere".to_string())
        );
    }
}
