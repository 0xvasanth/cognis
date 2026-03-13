//! Chat model factory and provider registry.
//!
//! Provides a [`ChatModelFactory`] for dynamically creating chat model instances
//! by provider name, and a global [`ModelRegistry`] singleton for application-wide
//! provider registration.
//!
//! # Example
//!
//! ```no_run
//! use cognis::chat_models::factory::{ModelConfig, ChatModelFactory, init_chat_model};
//!
//! // Using the factory directly
//! let factory = ChatModelFactory::default();
//! let config = ModelConfig::new("anthropic", "claude-sonnet-4-20250514")
//!     .with_api_key("sk-ant-...");
//! let model = factory.create(&config).unwrap();
//!
//! // Using the convenience function
//! let model = init_chat_model("anthropic", "claude-sonnet-4-20250514").unwrap();
//! ```

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::BaseChatModel;

/// Configuration for creating a chat model instance.
///
/// Contains all the parameters needed to construct a provider-specific chat model,
/// including the provider name, model identifier, credentials, and tuning parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider identifier (e.g., "anthropic", "openai", "ollama").
    pub provider: String,
    /// Model name (e.g., "claude-sonnet-4-20250514", "gpt-4", "llama3.2").
    pub model_name: String,
    /// API key for authenticated providers. Falls back to environment variables if `None`.
    pub api_key: Option<String>,
    /// Base URL override for the provider API.
    pub base_url: Option<String>,
    /// Sampling temperature (0.0 to 2.0 typically).
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Provider-specific extra parameters.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl ModelConfig {
    /// Create a new `ModelConfig` with the given provider and model name.
    pub fn new(provider: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
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

    /// Set the temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Insert an extra provider-specific parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

impl ModelConfig {
    /// Returns the default configuration (all fields empty/None).
    pub fn default_config() -> Self {
        Self {
            provider: String::new(),
            model_name: String::new(),
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            extra: HashMap::new(),
        }
    }
}

/// Type alias for a provider constructor function.
///
/// A constructor takes a [`ModelConfig`] and returns a boxed [`BaseChatModel`].
type ProviderConstructor = Box<dyn Fn(ModelConfig) -> Result<Box<dyn BaseChatModel>> + Send + Sync>;

/// Factory for creating chat model instances by provider name.
///
/// Maintains a registry of provider constructors and creates model instances
/// from a [`ModelConfig`]. Use [`ChatModelFactory::default()`] to get a factory
/// with all compiled-in providers pre-registered.
pub struct ChatModelFactory {
    providers: HashMap<String, ProviderConstructor>,
}

impl std::fmt::Debug for ChatModelFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatModelFactory")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ChatModelFactory {
    /// Create a new empty factory with no providers registered.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider constructor under the given name.
    ///
    /// If a provider with the same name already exists, it is replaced.
    pub fn register(
        &mut self,
        name: &str,
        constructor: Box<dyn Fn(ModelConfig) -> Result<Box<dyn BaseChatModel>> + Send + Sync>,
    ) {
        self.providers.insert(name.to_string(), constructor);
    }

    /// Create a chat model instance from the given configuration.
    ///
    /// Looks up the provider by `config.provider` and invokes its constructor.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not registered or if the constructor fails.
    pub fn create(&self, config: &ModelConfig) -> Result<Box<dyn BaseChatModel>> {
        let constructor = self.providers.get(&config.provider).ok_or_else(|| {
            CognisError::Other(format!(
                "Provider '{}' is not registered. Available providers: {:?}",
                config.provider,
                self.list_providers()
            ))
        })?;
        constructor(config.clone())
    }

    /// Convenience method to create a model from just a provider and model name.
    ///
    /// Constructs a minimal [`ModelConfig`] with only the provider and model name set,
    /// relying on environment variables for credentials.
    pub fn create_from_str(&self, provider: &str, model: &str) -> Result<Box<dyn BaseChatModel>> {
        let config = ModelConfig::new(provider, model);
        self.create(&config)
    }

    /// List all registered provider names.
    pub fn list_providers(&self) -> Vec<String> {
        let mut providers: Vec<String> = self.providers.keys().cloned().collect();
        providers.sort();
        providers
    }
}

impl Default for ChatModelFactory {
    /// Create a factory with all compiled-in providers pre-registered.
    ///
    /// Which providers are available depends on the enabled feature flags:
    /// `anthropic`, `openai`, `google`, `ollama`, `azure`.
    #[allow(unused_mut)]
    fn default() -> Self {
        let mut factory = Self::new();

        #[cfg(feature = "anthropic")]
        factory.register(
            "anthropic",
            Box::new(|config: ModelConfig| {
                let mut builder =
                    super::anthropic::ChatAnthropic::builder().model(&config.model_name);
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key);
                }
                if let Some(ref url) = config.base_url {
                    builder = builder.api_url(url);
                }
                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max) = config.max_tokens {
                    builder = builder.max_tokens(max);
                }
                if let Some(version) = config.extra.get("api_version").and_then(|v| v.as_str()) {
                    builder = builder.api_version(version);
                }
                if let Some(top_p) = config.extra.get("top_p").and_then(|v| v.as_f64()) {
                    builder = builder.top_p(top_p);
                }
                if let Some(top_k) = config.extra.get("top_k").and_then(|v| v.as_u64()) {
                    builder = builder.top_k(top_k as u32);
                }
                Ok(Box::new(builder.build()?) as Box<dyn BaseChatModel>)
            }),
        );

        #[cfg(feature = "openai")]
        factory.register(
            "openai",
            Box::new(|config: ModelConfig| {
                let mut builder = super::openai::ChatOpenAI::builder().model(&config.model_name);
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key);
                }
                if let Some(ref url) = config.base_url {
                    builder = builder.base_url(url);
                }
                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max) = config.max_tokens {
                    builder = builder.max_tokens(max);
                }
                Ok(Box::new(builder.build()?) as Box<dyn BaseChatModel>)
            }),
        );

        #[cfg(feature = "google")]
        factory.register(
            "google",
            Box::new(|config: ModelConfig| {
                let mut builder =
                    super::google::ChatGoogleGenAI::builder().model(&config.model_name);
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key);
                }
                if let Some(ref url) = config.base_url {
                    builder = builder.base_url(url);
                }
                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max) = config.max_tokens {
                    builder = builder.max_output_tokens(max);
                }
                Ok(Box::new(builder.build()?) as Box<dyn BaseChatModel>)
            }),
        );

        #[cfg(feature = "ollama")]
        factory.register(
            "ollama",
            Box::new(|config: ModelConfig| {
                let mut builder = super::ollama::ChatOllama::builder().model(&config.model_name);
                if let Some(ref url) = config.base_url {
                    builder = builder.base_url(url);
                }
                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }
                Ok(Box::new(builder.build()?) as Box<dyn BaseChatModel>)
            }),
        );

        #[cfg(feature = "azure")]
        factory.register(
            "azure",
            Box::new(|config: ModelConfig| {
                let mut builder =
                    super::azure::ChatAzureOpenAI::builder().deployment_name(&config.model_name);
                if let Some(ref key) = config.api_key {
                    builder = builder.api_key(key);
                }
                if let Some(ref url) = config.base_url {
                    builder = builder.azure_endpoint(url);
                }
                if let Some(temp) = config.temperature {
                    builder = builder.temperature(temp);
                }
                if let Some(max) = config.max_tokens {
                    builder = builder.max_tokens(max);
                }
                if let Some(endpoint) = config.extra.get("azure_endpoint").and_then(|v| v.as_str())
                {
                    builder = builder.azure_endpoint(endpoint);
                }
                if let Some(token) = config.extra.get("azure_ad_token").and_then(|v| v.as_str()) {
                    builder = builder.azure_ad_token(token);
                }
                if let Some(version) = config.extra.get("api_version").and_then(|v| v.as_str()) {
                    builder = builder.api_version(version);
                }
                Ok(Box::new(builder.build()?) as Box<dyn BaseChatModel>)
            }),
        );

        factory
    }
}

/// Global model registry providing application-wide provider management.
///
/// Uses interior mutability to allow registration after initialization.
/// Access the global instance via [`ModelRegistry::global()`].
pub struct ModelRegistry {
    factory: Mutex<ChatModelFactory>,
}

impl ModelRegistry {
    /// Get the global `ModelRegistry` singleton.
    ///
    /// The registry is initialized once with all compiled-in providers.
    /// Additional providers can be registered via [`ModelRegistry::register_provider`].
    pub fn global() -> &'static ModelRegistry {
        static INSTANCE: OnceLock<ModelRegistry> = OnceLock::new();
        INSTANCE.get_or_init(|| ModelRegistry {
            factory: Mutex::new(ChatModelFactory::default()),
        })
    }

    /// Register a new provider constructor in the global registry.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn register_provider(
        &self,
        name: &str,
        constructor: Box<dyn Fn(ModelConfig) -> Result<Box<dyn BaseChatModel>> + Send + Sync>,
    ) {
        self.factory
            .lock()
            .expect("ModelRegistry mutex poisoned")
            .register(name, constructor);
    }

    /// Create a chat model instance from the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn create_model(&self, config: &ModelConfig) -> Result<Box<dyn BaseChatModel>> {
        self.factory
            .lock()
            .expect("ModelRegistry mutex poisoned")
            .create(config)
    }

    /// List all registered provider names.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn list_providers(&self) -> Vec<String> {
        self.factory
            .lock()
            .expect("ModelRegistry mutex poisoned")
            .list_providers()
    }
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("providers", &self.list_providers())
            .finish()
    }
}

/// Convenience function to create a chat model from a provider and model name.
///
/// Uses the global [`ModelRegistry`] singleton. Credentials are resolved from
/// environment variables when not provided in the config.
///
/// # Arguments
///
/// * `provider` - Provider identifier (e.g., "anthropic", "openai", "ollama").
/// * `model` - Model name (e.g., "claude-sonnet-4-20250514", "gpt-4").
///
/// # Errors
///
/// Returns an error if the provider is not registered or if model construction fails.
///
/// # Example
///
/// ```no_run
/// use cognis::chat_models::factory::init_chat_model;
///
/// let model = init_chat_model("ollama", "llama3.2").unwrap();
/// ```
pub fn init_chat_model(provider: &str, model: &str) -> Result<Box<dyn BaseChatModel>> {
    let config = ModelConfig::new(provider, model);
    ModelRegistry::global().create_model(&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cognis_core::language_models::chat_model::BaseChatModel;
    use cognis_core::messages::Message;
    use cognis_core::outputs::ChatResult;

    /// A mock chat model for testing the factory without real API keys.
    #[allow(dead_code)]
    struct MockChatModel {
        provider: String,
        model_name: String,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    }

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Ok(ChatResult {
                generations: vec![],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            &self.provider
        }
    }

    /// Helper to create a mock constructor for testing.
    fn mock_constructor(
        provider_name: &str,
    ) -> Box<dyn Fn(ModelConfig) -> Result<Box<dyn BaseChatModel>> + Send + Sync> {
        let name = provider_name.to_string();
        Box::new(move |config: ModelConfig| {
            Ok(Box::new(MockChatModel {
                provider: name.clone(),
                model_name: config.model_name,
                temperature: config.temperature,
                max_tokens: config.max_tokens,
            }) as Box<dyn BaseChatModel>)
        })
    }

    // --- ModelConfig tests ---

    #[test]
    fn test_model_config_new() {
        let config = ModelConfig::new("anthropic", "claude-sonnet-4-20250514");
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model_name, "claude-sonnet-4-20250514");
        assert!(config.api_key.is_none());
        assert!(config.base_url.is_none());
        assert!(config.temperature.is_none());
        assert!(config.max_tokens.is_none());
        assert!(config.extra.is_empty());
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.provider, "");
        assert_eq!(config.model_name, "");
        assert!(config.api_key.is_none());
        assert!(config.extra.is_empty());
    }

    #[test]
    fn test_model_config_builder_methods() {
        let config = ModelConfig::new("openai", "gpt-4")
            .with_api_key("sk-test-key")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(2048)
            .with_extra("top_p", serde_json::json!(0.9));

        assert_eq!(config.provider, "openai");
        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.api_key, Some("sk-test-key".to_string()));
        assert_eq!(config.base_url, Some("https://custom.api.com".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.extra.get("top_p"), Some(&serde_json::json!(0.9)));
    }

    #[test]
    fn test_model_config_serialize_deserialize() {
        let config = ModelConfig::new("anthropic", "claude-sonnet-4-20250514")
            .with_api_key("sk-test")
            .with_temperature(0.5)
            .with_max_tokens(1024);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.provider, "anthropic");
        assert_eq!(deserialized.model_name, "claude-sonnet-4-20250514");
        assert_eq!(deserialized.api_key, Some("sk-test".to_string()));
        assert_eq!(deserialized.temperature, Some(0.5));
        assert_eq!(deserialized.max_tokens, Some(1024));
    }

    #[test]
    fn test_model_config_deserialize_minimal() {
        let json = r#"{"provider":"ollama","model_name":"llama3.2"}"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.model_name, "llama3.2");
        assert!(config.api_key.is_none());
        assert!(config.extra.is_empty());
    }

    // --- ChatModelFactory tests ---

    #[test]
    fn test_factory_new_is_empty() {
        let factory = ChatModelFactory::new();
        assert!(factory.list_providers().is_empty());
    }

    #[test]
    fn test_factory_register_and_list() {
        let mut factory = ChatModelFactory::new();
        factory.register("mock_a", mock_constructor("mock_a"));
        factory.register("mock_b", mock_constructor("mock_b"));

        let providers = factory.list_providers();
        assert_eq!(providers.len(), 2);
        assert!(providers.contains(&"mock_a".to_string()));
        assert!(providers.contains(&"mock_b".to_string()));
    }

    #[test]
    fn test_factory_list_providers_sorted() {
        let mut factory = ChatModelFactory::new();
        factory.register("zeta", mock_constructor("zeta"));
        factory.register("alpha", mock_constructor("alpha"));
        factory.register("mu", mock_constructor("mu"));

        let providers = factory.list_providers();
        assert_eq!(providers, vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn test_factory_create_success() {
        let mut factory = ChatModelFactory::new();
        factory.register("mock", mock_constructor("mock"));

        let config = ModelConfig::new("mock", "test-model");
        let model = factory.create(&config).unwrap();
        assert_eq!(model.llm_type(), "mock");
    }

    #[test]
    fn test_factory_create_unknown_provider() {
        let factory = ChatModelFactory::new();
        let config = ModelConfig::new("nonexistent", "model");
        let result = factory.create(&config);

        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(err.contains("nonexistent"));
        assert!(err.contains("not registered"));
    }

    #[test]
    fn test_factory_create_from_str() {
        let mut factory = ChatModelFactory::new();
        factory.register("mock", mock_constructor("mock"));

        let model = factory.create_from_str("mock", "test-model").unwrap();
        assert_eq!(model.llm_type(), "mock");
    }

    #[test]
    fn test_factory_create_from_str_unknown_provider() {
        let factory = ChatModelFactory::new();
        let result = factory.create_from_str("nonexistent", "model");
        assert!(result.is_err());
    }

    #[test]
    fn test_factory_register_replaces_existing() {
        let mut factory = ChatModelFactory::new();
        factory.register("mock", mock_constructor("version_1"));
        factory.register("mock", mock_constructor("version_2"));

        let config = ModelConfig::new("mock", "test");
        let model = factory.create(&config).unwrap();
        // The second registration should win
        assert_eq!(model.llm_type(), "version_2");
    }

    #[test]
    fn test_factory_constructor_receives_config() {
        let mut factory = ChatModelFactory::new();
        factory.register(
            "check",
            Box::new(|config: ModelConfig| {
                // Verify the config is correctly passed through
                assert_eq!(config.model_name, "the-model");
                assert_eq!(config.api_key, Some("the-key".to_string()));
                assert_eq!(config.temperature, Some(0.42));
                Ok(Box::new(MockChatModel {
                    provider: "check".to_string(),
                    model_name: config.model_name,
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                }) as Box<dyn BaseChatModel>)
            }),
        );

        let config = ModelConfig::new("check", "the-model")
            .with_api_key("the-key")
            .with_temperature(0.42);
        let _model = factory.create(&config).unwrap();
    }

    #[test]
    fn test_factory_constructor_can_fail() {
        let mut factory = ChatModelFactory::new();
        factory.register(
            "failing",
            Box::new(|_config: ModelConfig| Err(CognisError::Other("construction failed".into()))),
        );

        let config = ModelConfig::new("failing", "model");
        let result = factory.create(&config);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(err.contains("construction failed"));
    }

    #[test]
    fn test_factory_debug_format() {
        let mut factory = ChatModelFactory::new();
        factory.register("mock", mock_constructor("mock"));
        let debug = format!("{:?}", factory);
        assert!(debug.contains("ChatModelFactory"));
        assert!(debug.contains("mock"));
    }

    // --- ModelRegistry tests ---

    #[test]
    fn test_registry_global_is_same_instance() {
        let r1 = ModelRegistry::global();
        let r2 = ModelRegistry::global();
        // Both should point to the same static address
        assert!(std::ptr::eq(r1, r2));
    }

    #[test]
    fn test_registry_register_and_create() {
        let registry = ModelRegistry::global();
        registry.register_provider("test_mock", mock_constructor("test_mock"));

        let config = ModelConfig::new("test_mock", "my-model");
        let model = registry.create_model(&config).unwrap();
        assert_eq!(model.llm_type(), "test_mock");
    }

    #[test]
    fn test_registry_list_providers_includes_registered() {
        let registry = ModelRegistry::global();
        registry.register_provider("test_list_check", mock_constructor("test_list_check"));

        let providers = registry.list_providers();
        assert!(providers.contains(&"test_list_check".to_string()));
    }

    #[test]
    fn test_registry_create_unknown_provider() {
        let registry = ModelRegistry::global();
        let config = ModelConfig::new("absolutely_not_a_provider", "model");
        let result = registry.create_model(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_debug_format() {
        let registry = ModelRegistry::global();
        let debug = format!("{:?}", registry);
        assert!(debug.contains("ModelRegistry"));
    }

    // --- ModelConfig extra fields tests ---

    #[test]
    fn test_model_config_multiple_extras() {
        let config = ModelConfig::new("anthropic", "claude")
            .with_extra("top_p", serde_json::json!(0.95))
            .with_extra("top_k", serde_json::json!(40))
            .with_extra("stop_sequences", serde_json::json!(["END"]));

        assert_eq!(config.extra.len(), 3);
        assert_eq!(config.extra["top_p"], serde_json::json!(0.95));
        assert_eq!(config.extra["top_k"], serde_json::json!(40));
        assert_eq!(config.extra["stop_sequences"], serde_json::json!(["END"]));
    }

    #[test]
    fn test_model_config_clone() {
        let config = ModelConfig::new("openai", "gpt-4")
            .with_api_key("key")
            .with_temperature(0.5);
        let cloned = config.clone();
        assert_eq!(cloned.provider, "openai");
        assert_eq!(cloned.model_name, "gpt-4");
        assert_eq!(cloned.api_key, Some("key".to_string()));
        assert_eq!(cloned.temperature, Some(0.5));
    }
}
