//! Chat model factory and provider registry.
//!
//! Mirrors Python `langchain.chat_models.base` — provides the `init_chat_model`
//! factory function and model/provider parsing utilities.

use std::collections::HashMap;

use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;

/// Configuration for a provider's chat model implementation.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// The module/crate name for the provider.
    pub module_name: String,
    /// The class/struct name for the chat model.
    pub class_name: String,
}

impl ProviderConfig {
    /// Create a new provider configuration.
    pub fn new(module_name: impl Into<String>, class_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            class_name: class_name.into(),
        }
    }
}

/// Resolved configuration for a chat model.
///
/// Returned by `init_chat_model` with all provider and model information resolved.
/// Since provider crates are not dynamically loaded, this struct carries all the
/// information needed to construct the appropriate model elsewhere.
#[derive(Debug, Clone)]
pub struct ChatModelConfig {
    /// The resolved provider key (e.g. "openai", "anthropic").
    pub provider: String,
    /// The model name (e.g. "gpt-4", "claude-3-opus").
    pub model_name: String,
    /// The provider configuration (module/class info).
    pub provider_config: ProviderConfig,
    /// Additional keyword arguments to pass to the model constructor.
    pub kwargs: HashMap<String, Value>,
}

/// Returns the built-in provider registry mapping provider keys to their configs.
///
/// This is the Rust equivalent of Python's `_BUILTIN_CHAT_MODEL_PROVIDERS` dict.
pub fn builtin_providers() -> HashMap<String, ProviderConfig> {
    let mut m = HashMap::new();
    m.insert(
        "openai".to_string(),
        ProviderConfig::new("rustchain_openai", "ChatOpenAI"),
    );
    m.insert(
        "anthropic".to_string(),
        ProviderConfig::new("rustchain_anthropic", "ChatAnthropic"),
    );
    m.insert(
        "anthropic_bedrock".to_string(),
        ProviderConfig::new("rustchain_aws", "ChatAnthropicBedrock"),
    );
    m.insert(
        "azure_ai".to_string(),
        ProviderConfig::new("rustchain_azure_ai", "ChatAzureAI"),
    );
    m.insert(
        "azure_openai".to_string(),
        ProviderConfig::new("rustchain_openai", "AzureChatOpenAI"),
    );
    m.insert(
        "bedrock".to_string(),
        ProviderConfig::new("rustchain_aws", "ChatBedrock"),
    );
    m.insert(
        "bedrock_converse".to_string(),
        ProviderConfig::new("rustchain_aws", "ChatBedrockConverse"),
    );
    m.insert(
        "cohere".to_string(),
        ProviderConfig::new("rustchain_cohere", "ChatCohere"),
    );
    m.insert(
        "deepseek".to_string(),
        ProviderConfig::new("rustchain_deepseek", "ChatDeepSeek"),
    );
    m.insert(
        "fireworks".to_string(),
        ProviderConfig::new("rustchain_fireworks", "ChatFireworks"),
    );
    m.insert(
        "google_anthropic_vertex".to_string(),
        ProviderConfig::new("rustchain_google_vertexai", "ChatAnthropicVertex"),
    );
    m.insert(
        "google_genai".to_string(),
        ProviderConfig::new("rustchain_google_genai", "ChatGoogleGenerativeAI"),
    );
    m.insert(
        "google_vertexai".to_string(),
        ProviderConfig::new("rustchain_google_vertexai", "ChatVertexAI"),
    );
    m.insert(
        "groq".to_string(),
        ProviderConfig::new("rustchain_groq", "ChatGroq"),
    );
    m.insert(
        "huggingface".to_string(),
        ProviderConfig::new("rustchain_huggingface", "ChatHuggingFace"),
    );
    m.insert(
        "ibm".to_string(),
        ProviderConfig::new("rustchain_ibm", "ChatWatsonx"),
    );
    m.insert(
        "mistralai".to_string(),
        ProviderConfig::new("rustchain_mistralai", "ChatMistralAI"),
    );
    m.insert(
        "nvidia".to_string(),
        ProviderConfig::new("rustchain_nvidia", "ChatNVIDIA"),
    );
    m.insert(
        "ollama".to_string(),
        ProviderConfig::new("rustchain_ollama", "ChatOllama"),
    );
    m.insert(
        "openrouter".to_string(),
        ProviderConfig::new("rustchain_openrouter", "ChatOpenRouter"),
    );
    m.insert(
        "perplexity".to_string(),
        ProviderConfig::new("rustchain_perplexity", "ChatPerplexity"),
    );
    m.insert(
        "together".to_string(),
        ProviderConfig::new("rustchain_together", "ChatTogether"),
    );
    m.insert(
        "upstage".to_string(),
        ProviderConfig::new("rustchain_upstage", "ChatUpstage"),
    );
    m.insert(
        "xai".to_string(),
        ProviderConfig::new("rustchain_xai", "ChatXAI"),
    );
    m
}

/// Parse a model string in "provider:model_name" format.
///
/// If the string contains a colon, splits into (provider, model_name).
/// If no colon is present, attempts to infer the provider from the model name.
///
/// # Examples
/// ```
/// use rustchain::chat_models::base::parse_model;
///
/// let (provider, model) = parse_model("openai:gpt-4").unwrap();
/// assert_eq!(provider, "openai");
/// assert_eq!(model, "gpt-4");
/// ```
pub fn parse_model(model: &str) -> Result<(String, String)> {
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
        // Try to infer the provider from the model name
        if let Some(provider) = attempt_infer_provider(model) {
            Ok((provider, model.to_string()))
        } else {
            Err(RustChainError::Other(format!(
                "Unable to infer provider for model '{}'. Use 'provider:model' format or a recognized model name prefix.",
                model
            )))
        }
    }
}

/// Attempt to infer the provider from a model name based on known prefixes.
///
/// Returns `Some(provider_key)` if a match is found, `None` otherwise.
///
/// # Known prefix mappings
/// - `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt` -> "openai"
/// - `claude` -> "anthropic"
/// - `gemini` -> "google_vertexai"
/// - `command` -> "cohere"
/// - `mistral`, `codestral`, `pixtral`, `mixtral` -> "mistralai"
/// - `llama` -> "groq"
/// - `amazon.`, `anthropic.`, `meta.`, `cohere.` (Bedrock ARN-style) -> "bedrock"
/// - `accounts/fireworks` -> "fireworks"
/// - `deepseek` -> "deepseek"
/// - `grok` -> "xai"
/// - `sonar` -> "perplexity"
/// - `solar` -> "upstage"
pub fn attempt_infer_provider(model_name: &str) -> Option<String> {
    let lower = model_name.to_lowercase();

    // OpenAI models
    if lower.starts_with("gpt-")
        || lower.starts_with("o1-")
        || lower.starts_with("o3-")
        || lower.starts_with("o4-")
        || lower == "gpt4"
        || lower == "gpt3"
        || lower.starts_with("chatgpt")
    {
        return Some("openai".to_string());
    }

    // Anthropic models
    if lower.starts_with("claude") {
        return Some("anthropic".to_string());
    }

    // Google models
    if lower.starts_with("gemini") {
        return Some("google_vertexai".to_string());
    }

    // Cohere models
    if lower.starts_with("command") {
        return Some("cohere".to_string());
    }

    // Mistral models (including mixtral which is a Mistral model)
    if lower.starts_with("mistral")
        || lower.starts_with("codestral")
        || lower.starts_with("pixtral")
        || lower.starts_with("mixtral")
    {
        return Some("mistralai".to_string());
    }

    // Groq-hosted models (llama)
    if lower.starts_with("llama") {
        return Some("groq".to_string());
    }

    // AWS Bedrock (ARN-style prefixes)
    if lower.starts_with("amazon.")
        || lower.starts_with("anthropic.")
        || lower.starts_with("meta.")
        || lower.starts_with("cohere.")
    {
        return Some("bedrock".to_string());
    }

    // Fireworks (accounts/fireworks path-style)
    if lower.starts_with("accounts/fireworks") {
        return Some("fireworks".to_string());
    }

    // DeepSeek models
    if lower.starts_with("deepseek") {
        return Some("deepseek".to_string());
    }

    // xAI models
    if lower.starts_with("grok") {
        return Some("xai".to_string());
    }

    // Perplexity models
    if lower.starts_with("sonar") {
        return Some("perplexity".to_string());
    }

    // Upstage models
    if lower.starts_with("solar") {
        return Some("upstage".to_string());
    }

    None
}

/// Initialize a chat model by provider and model name.
///
/// Returns a `ChatModelConfig` with all resolved provider and model information.
/// Since provider crates are not dynamically loaded in Rust, this function resolves
/// the configuration that can be used to construct the appropriate model.
///
/// # Arguments
/// - `model` — Model string, either "provider:model_name" or just "model_name" (inferred).
/// - `model_provider` — Optional explicit provider override (bypasses inference).
/// - `kwargs` — Optional additional keyword arguments for the model constructor.
///
/// # Errors
/// Returns an error if the provider is unknown or cannot be inferred.
pub fn init_chat_model(
    model: &str,
    model_provider: Option<&str>,
    kwargs: Option<HashMap<String, Value>>,
) -> Result<ChatModelConfig> {
    let providers = builtin_providers();
    let kwargs = kwargs.unwrap_or_default();

    let (provider, model_name) = if let Some(explicit_provider) = model_provider {
        // Explicit provider given — use it directly
        let model_name = if let Some((_p, m)) = model.split_once(':') {
            m.to_string()
        } else {
            model.to_string()
        };
        (explicit_provider.to_string(), model_name)
    } else {
        parse_model(model)?
    };

    if let Some(config) = providers.get(&provider) {
        Ok(ChatModelConfig {
            provider: provider.clone(),
            model_name,
            provider_config: config.clone(),
            kwargs,
        })
    } else {
        Err(RustChainError::Other(format!(
            "Unknown provider '{}'. Available providers: {:?}",
            provider,
            providers.keys().collect::<Vec<_>>()
        )))
    }
}

/// Create an actual chat model instance from a model string.
///
/// This is the high-level factory that returns a ready-to-use `Box<dyn BaseChatModel>`.
/// It resolves the provider from the model string and constructs the appropriate
/// implementation behind feature flags.
///
/// # Arguments
/// - `model` — Model string, e.g. `"anthropic:claude-sonnet-4-20250514"` or `"gpt-4o"`.
/// - `kwargs` — Optional parameters like `temperature`, `max_tokens`, `api_key`.
///
/// # Examples
/// ```no_run
/// # use rustchain::chat_models::base::create_chat_model;
/// let model = create_chat_model("anthropic:claude-sonnet-4-20250514", None).unwrap();
/// assert_eq!(model.llm_type(), "anthropic");
/// ```
pub fn create_chat_model(
    model: &str,
    kwargs: Option<HashMap<String, Value>>,
) -> Result<Box<dyn BaseChatModel>> {
    let config = init_chat_model(model, None, kwargs)?;

    match config.provider.as_str() {
        #[cfg(feature = "anthropic")]
        "anthropic" => {
            let mut builder = super::anthropic::ChatAnthropic::builder()
                .model(&config.model_name);
            if let Some(key) = config.kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(temp) = config.kwargs.get("temperature").and_then(|v| v.as_f64()) {
                builder = builder.temperature(temp);
            }
            if let Some(max) = config.kwargs.get("max_tokens").and_then(|v| v.as_u64()) {
                builder = builder.max_tokens(max as u32);
            }
            if let Some(url) = config.kwargs.get("api_url").and_then(|v| v.as_str()) {
                builder = builder.api_url(url);
            }
            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "openai")]
        "openai" => {
            let mut builder = super::openai::ChatOpenAI::builder()
                .model(&config.model_name);
            if let Some(key) = config.kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(temp) = config.kwargs.get("temperature").and_then(|v| v.as_f64()) {
                builder = builder.temperature(temp);
            }
            if let Some(max) = config.kwargs.get("max_tokens").and_then(|v| v.as_u64()) {
                builder = builder.max_tokens(max as u32);
            }
            if let Some(url) = config.kwargs.get("base_url").and_then(|v| v.as_str()) {
                builder = builder.base_url(url);
            }
            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "google")]
        "google" | "google_genai" => {
            let mut builder = super::google::ChatGoogleGenAI::builder()
                .model(&config.model_name);
            if let Some(key) = config.kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(temp) = config.kwargs.get("temperature").and_then(|v| v.as_f64()) {
                builder = builder.temperature(temp);
            }
            if let Some(max) = config.kwargs.get("max_output_tokens").and_then(|v| v.as_u64()) {
                builder = builder.max_output_tokens(max as u32);
            }
            if let Some(url) = config.kwargs.get("base_url").and_then(|v| v.as_str()) {
                builder = builder.base_url(url);
            }
            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "ollama")]
        "ollama" => {
            let mut builder = super::ollama::ChatOllama::builder()
                .model(&config.model_name);
            if let Some(temp) = config.kwargs.get("temperature").and_then(|v| v.as_f64()) {
                builder = builder.temperature(temp);
            }
            if let Some(url) = config.kwargs.get("base_url").and_then(|v| v.as_str()) {
                builder = builder.base_url(url);
            }
            Ok(Box::new(builder.build()?))
        }
        #[cfg(feature = "azure")]
        "azure" | "azure_openai" => {
            let resource_name = config.kwargs.get("resource_name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| RustChainError::Other(
                    "resource_name is required for Azure OpenAI".into()
                ))?;
            let mut builder = super::azure::ChatAzureOpenAI::builder()
                .resource_name(resource_name)
                .deployment_name(&config.model_name);
            if let Some(key) = config.kwargs.get("api_key").and_then(|v| v.as_str()) {
                builder = builder.api_key(key);
            }
            if let Some(temp) = config.kwargs.get("temperature").and_then(|v| v.as_f64()) {
                builder = builder.temperature(temp);
            }
            if let Some(max) = config.kwargs.get("max_tokens").and_then(|v| v.as_u64()) {
                builder = builder.max_tokens(max as u32);
            }
            if let Some(version) = config.kwargs.get("api_version").and_then(|v| v.as_str()) {
                builder = builder.api_version(version);
            }
            Ok(Box::new(builder.build()?))
        }
        other => Err(RustChainError::Other(format!(
            "Provider '{}' is not available. Enable the corresponding feature flag (e.g., --features {}) \
             or use a supported provider. Available with features: anthropic, openai, google, ollama, azure.",
            other, other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_with_provider() {
        let (provider, model) = parse_model("openai:gpt-4").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn test_parse_model_with_provider_anthropic() {
        let (provider, model) = parse_model("anthropic:claude-3-opus").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-opus");
    }

    #[test]
    fn test_parse_model_infer_openai() {
        let (provider, model) = parse_model("gpt-4").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn test_parse_model_infer_anthropic() {
        let (provider, model) = parse_model("claude-3-sonnet").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-sonnet");
    }

    #[test]
    fn test_parse_model_infer_gemini() {
        let (provider, model) = parse_model("gemini-pro").unwrap();
        assert_eq!(provider, "google_vertexai");
        assert_eq!(model, "gemini-pro");
    }

    #[test]
    fn test_parse_model_unknown() {
        let result = parse_model("some-unknown-model");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unable to infer provider"));
    }

    #[test]
    fn test_parse_model_empty_provider() {
        let result = parse_model(":gpt-4");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Provider name cannot be empty"));
    }

    #[test]
    fn test_parse_model_empty_model_name() {
        let result = parse_model("openai:");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Model name cannot be empty"));
    }

    #[test]
    fn test_attempt_infer_provider_openai_gpt() {
        assert_eq!(
            attempt_infer_provider("gpt-4-turbo"),
            Some("openai".to_string())
        );
        assert_eq!(
            attempt_infer_provider("gpt-3.5-turbo"),
            Some("openai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_openai_o_series() {
        assert_eq!(
            attempt_infer_provider("o1-preview"),
            Some("openai".to_string())
        );
        assert_eq!(
            attempt_infer_provider("o3-mini"),
            Some("openai".to_string())
        );
        assert_eq!(
            attempt_infer_provider("o4-mini"),
            Some("openai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_openai_chatgpt() {
        assert_eq!(
            attempt_infer_provider("chatgpt-4o-latest"),
            Some("openai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_anthropic() {
        assert_eq!(
            attempt_infer_provider("claude-3-opus-20240229"),
            Some("anthropic".to_string())
        );
        assert_eq!(
            attempt_infer_provider("claude-instant-1.2"),
            Some("anthropic".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_google() {
        assert_eq!(
            attempt_infer_provider("gemini-1.5-pro"),
            Some("google_vertexai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_cohere() {
        assert_eq!(
            attempt_infer_provider("command-r-plus"),
            Some("cohere".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_mistral() {
        assert_eq!(
            attempt_infer_provider("mistral-large-latest"),
            Some("mistralai".to_string())
        );
        assert_eq!(
            attempt_infer_provider("codestral-latest"),
            Some("mistralai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_mixtral_is_mistralai() {
        // mixtral is a Mistral model, not groq
        assert_eq!(
            attempt_infer_provider("mixtral-8x7b"),
            Some("mistralai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_groq() {
        assert_eq!(
            attempt_infer_provider("llama-3.1-70b"),
            Some("groq".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_bedrock() {
        assert_eq!(
            attempt_infer_provider("amazon.titan-text-express-v1"),
            Some("bedrock".to_string())
        );
        assert_eq!(
            attempt_infer_provider("anthropic.claude-v2"),
            Some("bedrock".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_fireworks_path() {
        assert_eq!(
            attempt_infer_provider("accounts/fireworks/models/llama-v3"),
            Some("fireworks".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_deepseek() {
        assert_eq!(
            attempt_infer_provider("deepseek-chat"),
            Some("deepseek".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_xai() {
        assert_eq!(
            attempt_infer_provider("grok-2"),
            Some("xai".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_perplexity() {
        assert_eq!(
            attempt_infer_provider("sonar-medium-online"),
            Some("perplexity".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_upstage() {
        assert_eq!(
            attempt_infer_provider("solar-pro"),
            Some("upstage".to_string())
        );
    }

    #[test]
    fn test_attempt_infer_provider_unknown() {
        assert_eq!(attempt_infer_provider("some-random-model"), None);
    }

    #[test]
    fn test_attempt_infer_provider_case_insensitive() {
        assert_eq!(
            attempt_infer_provider("GPT-4"),
            Some("openai".to_string())
        );
        assert_eq!(
            attempt_infer_provider("Claude-3-Opus"),
            Some("anthropic".to_string())
        );
    }

    #[test]
    fn test_builtin_providers_contains_expected() {
        let providers = builtin_providers();
        assert!(providers.contains_key("openai"));
        assert!(providers.contains_key("anthropic"));
        assert!(providers.contains_key("anthropic_bedrock"));
        assert!(providers.contains_key("azure_ai"));
        assert!(providers.contains_key("azure_openai"));
        assert!(providers.contains_key("bedrock"));
        assert!(providers.contains_key("bedrock_converse"));
        assert!(providers.contains_key("cohere"));
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("fireworks"));
        assert!(providers.contains_key("google_anthropic_vertex"));
        assert!(providers.contains_key("google_genai"));
        assert!(providers.contains_key("google_vertexai"));
        assert!(providers.contains_key("groq"));
        assert!(providers.contains_key("huggingface"));
        assert!(providers.contains_key("ibm"));
        assert!(providers.contains_key("mistralai"));
        assert!(providers.contains_key("nvidia"));
        assert!(providers.contains_key("ollama"));
        assert!(providers.contains_key("openrouter"));
        assert!(providers.contains_key("perplexity"));
        assert!(providers.contains_key("together"));
        assert!(providers.contains_key("upstage"));
        assert!(providers.contains_key("xai"));
    }

    #[test]
    fn test_init_chat_model_known_provider() {
        let result = init_chat_model("openai:gpt-4", None, None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.provider_config.module_name, "rustchain_openai");
        assert_eq!(config.provider_config.class_name, "ChatOpenAI");
    }

    #[test]
    fn test_init_chat_model_inferred_provider() {
        let result = init_chat_model("gpt-4", None, None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model_name, "gpt-4");
    }

    #[test]
    fn test_init_chat_model_explicit_provider() {
        let result = init_chat_model("my-custom-model", Some("anthropic"), None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model_name, "my-custom-model");
        assert_eq!(config.provider_config.class_name, "ChatAnthropic");
    }

    #[test]
    fn test_init_chat_model_explicit_provider_with_colon() {
        let result = init_chat_model("openai:gpt-4", Some("anthropic"), None);
        assert!(result.is_ok());
        let config = result.unwrap();
        // Explicit provider overrides the one in the model string
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model_name, "gpt-4");
    }

    #[test]
    fn test_init_chat_model_with_kwargs() {
        let mut kwargs = HashMap::new();
        kwargs.insert("temperature".to_string(), serde_json::json!(0.7));
        kwargs.insert("max_tokens".to_string(), serde_json::json!(1024));
        let result = init_chat_model("openai:gpt-4", None, Some(kwargs));
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.kwargs["temperature"], serde_json::json!(0.7));
        assert_eq!(config.kwargs["max_tokens"], serde_json::json!(1024));
    }

    #[test]
    fn test_init_chat_model_unknown_provider() {
        let result = init_chat_model("unknown_provider:some-model", None, None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown provider"));
    }

    #[test]
    fn test_provider_config_fields() {
        let config = ProviderConfig::new("my_crate", "MyChatModel");
        assert_eq!(config.module_name, "my_crate");
        assert_eq!(config.class_name, "MyChatModel");
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn test_create_chat_model_anthropic() {
        let mut kwargs = HashMap::new();
        kwargs.insert("api_key".to_string(), serde_json::json!("test-key"));
        let model = create_chat_model("anthropic:claude-sonnet-4-20250514", Some(kwargs)).unwrap();
        assert_eq!(model.llm_type(), "anthropic");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_create_chat_model_openai() {
        let mut kwargs = HashMap::new();
        kwargs.insert("api_key".to_string(), serde_json::json!("test-key"));
        let model = create_chat_model("openai:gpt-4o", Some(kwargs)).unwrap();
        assert_eq!(model.llm_type(), "openai");
    }

    #[cfg(feature = "google")]
    #[test]
    fn test_create_chat_model_google() {
        let mut kwargs = HashMap::new();
        kwargs.insert("api_key".to_string(), serde_json::json!("test-key"));
        let model = create_chat_model("google_genai:gemini-2.0-flash", Some(kwargs)).unwrap();
        assert_eq!(model.llm_type(), "google_genai");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_create_chat_model_ollama() {
        let model = create_chat_model("ollama:llama3.2", None).unwrap();
        assert_eq!(model.llm_type(), "ollama");
    }

    #[cfg(feature = "azure")]
    #[test]
    fn test_create_chat_model_azure() {
        let mut kwargs = HashMap::new();
        kwargs.insert("api_key".to_string(), serde_json::json!("test-key"));
        kwargs.insert("resource_name".to_string(), serde_json::json!("my-resource"));
        let model = create_chat_model("azure_openai:gpt-4o", Some(kwargs)).unwrap();
        assert_eq!(model.llm_type(), "azure_openai");
    }

    #[test]
    fn test_create_chat_model_unsupported_provider() {
        let result = create_chat_model("nvidia:some-model", None);
        assert!(result.is_err());
    }
}
