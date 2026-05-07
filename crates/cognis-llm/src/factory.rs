//! Provider factory + runtime registry.
//!
//! V1 had `ChatModelFactory` and `ChatModelRegistry` as separate types;
//! v2 collapses them into one [`ProviderRegistry`] that:
//!
//! - Holds named [`ProviderConstructor`] entries — closures that take a
//!   [`ProviderSpec`] and return an `Arc<dyn LLMProvider>`.
//! - Resolves `provider:model-id` strings into a constructor + model
//!   name pair, then invokes the constructor.
//! - Comes pre-loaded with built-in providers when their feature flags
//!   are on (`openai`, `anthropic`, `google`, `ollama`, `azure`,
//!   `openrouter` — the last lives behind `openai`).
//!
//! Customization:
//! - [`ProviderRegistry::register`] to add a custom provider keyed by a
//!   string id.
//! - [`ProviderRegistry::register_alias`] to expose extra names for an
//!   existing provider.
//! - [`ProviderSpec`] is the open struct passed to constructors; users
//!   embed extra fields via the `extras: serde_json::Value` bag.

use std::collections::HashMap;
use std::sync::Arc;

use cognis_core::{CognisError, Result};

use crate::client::Client;
use crate::provider::LLMProvider;

/// Configuration for one provider construction. Constructors interpret
/// the fields they care about and may consult `extras` for non-standard
/// settings.
#[derive(Debug, Clone, Default)]
pub struct ProviderSpec {
    /// Model id (e.g. `"gpt-4o-mini"`, `"claude-3-5-sonnet"`).
    pub model: Option<String>,
    /// API key.
    pub api_key: Option<String>,
    /// Base URL override.
    pub base_url: Option<String>,
    /// HTTP timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// OpenAI organization id.
    pub organization: Option<String>,
    /// Azure-specific: endpoint URL.
    pub azure_endpoint: Option<String>,
    /// Azure-specific: deployment name.
    pub azure_deployment: Option<String>,
    /// Azure-specific: API version.
    pub azure_api_version: Option<String>,
    /// Provider-specific extras for custom registrations.
    pub extras: serde_json::Value,
}

impl ProviderSpec {
    /// Build with just a model id.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Default::default()
        }
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Attach an extras value (provider-specific bag).
    pub fn with_extras(mut self, extras: serde_json::Value) -> Self {
        self.extras = extras;
        self
    }
}

/// Constructor closure: spec → provider. Errors propagate as
/// [`CognisError`] so the registry can surface configuration failures.
pub type ProviderConstructor =
    Arc<dyn Fn(&ProviderSpec) -> Result<Arc<dyn LLMProvider>> + Send + Sync>;

/// Open registry of provider constructors.
///
/// `ProviderRegistry::with_builtins()` returns a registry pre-populated
/// with whatever providers are compiled in via Cargo features.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    constructors: HashMap<String, ProviderConstructor>,
}

impl ProviderRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// New registry pre-loaded with the builtin providers (those with
    /// active feature flags).
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r = r.register_builtins();
        r
    }

    /// Register a constructor under `id`. Replaces any prior entry.
    pub fn register<F>(mut self, id: impl Into<String>, ctor: F) -> Self
    where
        F: Fn(&ProviderSpec) -> Result<Arc<dyn LLMProvider>> + Send + Sync + 'static,
    {
        self.constructors.insert(id.into(), Arc::new(ctor));
        self
    }

    /// Register `alias` to point at the same constructor as `id`. Errors
    /// silently if `id` is not registered.
    pub fn register_alias(mut self, alias: impl Into<String>, id: &str) -> Self {
        if let Some(c) = self.constructors.get(id).cloned() {
            self.constructors.insert(alias.into(), c);
        }
        self
    }

    /// All registered ids (alphabetical).
    pub fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.constructors.keys().cloned().collect();
        v.sort();
        v
    }

    /// Build a provider from a `provider:model-id` string. The model
    /// portion is optional (`"openai"` works; defaults to the provider's
    /// default model). The `spec` overrides any model derived from the
    /// id string.
    pub fn build(&self, ident: &str, mut spec: ProviderSpec) -> Result<Arc<dyn LLMProvider>> {
        let (id, model) = match ident.split_once(':') {
            Some((i, m)) => (i.trim(), Some(m.trim().to_string())),
            None => (ident.trim(), None),
        };
        if spec.model.is_none() {
            spec.model = model;
        }
        let ctor = self.constructors.get(id).ok_or_else(|| {
            CognisError::Configuration(format!(
                "no provider registered as `{id}` (have: {})",
                self.ids().join(", ")
            ))
        })?;
        ctor(&spec)
    }

    /// Convenience: build a [`Client`] directly.
    pub fn build_client(&self, ident: &str, spec: ProviderSpec) -> Result<Client> {
        Ok(Client::new(self.build(ident, spec)?))
    }

    fn register_builtins(self) -> Self {
        // Each builtin uses ClientBuilder under the hood so feature-flag
        // gating, env-driven defaults, and validation stay in one place.
        let mut reg = self;

        #[cfg(feature = "openai")]
        {
            reg = reg.register("openai", |s| {
                build_via_client_builder(crate::Provider::OpenAI, s)
            });
            reg = reg.register("openrouter", |s| {
                let mut b = crate::provider::openrouter::OpenRouterBuilder::default();
                if let Some(k) = &s.api_key {
                    b = b.api_key(k.clone());
                }
                if let Some(u) = &s.base_url {
                    b = b.base_url(u.clone());
                }
                if let Some(m) = &s.model {
                    b = b.model(m.clone());
                }
                if let Some(t) = s.timeout_secs {
                    b = b.timeout_secs(t);
                }
                // OpenRouter-specific extras: pull `referer` / `title`
                // out of the spec.extras bag if present.
                if let Some(r) = s.extras.get("referer").and_then(|v| v.as_str()) {
                    b = b.with_referer(r);
                }
                if let Some(t) = s.extras.get("title").and_then(|v| v.as_str()) {
                    b = b.with_title(t);
                }
                Ok(Arc::new(b.build()?) as Arc<dyn LLMProvider>)
            });
            reg = reg.register_alias("gpt", "openai");
            reg = reg.register_alias("open-router", "openrouter");
        }
        #[cfg(feature = "anthropic")]
        {
            reg = reg.register("anthropic", |s| {
                build_via_client_builder(crate::Provider::Anthropic, s)
            });
            reg = reg.register_alias("claude", "anthropic");
        }
        #[cfg(feature = "google")]
        {
            reg = reg.register("google", |s| {
                build_via_client_builder(crate::Provider::Google, s)
            });
            reg = reg.register_alias("gemini", "google");
        }
        #[cfg(feature = "ollama")]
        {
            reg = reg.register("ollama", |s| {
                build_via_client_builder(crate::Provider::Ollama, s)
            });
        }
        #[cfg(feature = "azure")]
        {
            reg = reg.register("azure", |s| {
                build_via_client_builder(crate::Provider::Azure, s)
            });
        }

        reg
    }
}

/// Helper: drive a builtin provider through `Client::builder()` so the
/// builder's validation logic is single-sourced.
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
fn build_via_client_builder(
    provider: crate::Provider,
    spec: &ProviderSpec,
) -> Result<Arc<dyn LLMProvider>> {
    let mut b = Client::builder().provider(provider);
    if let Some(k) = &spec.api_key {
        b = b.api_key(k.clone());
    }
    if let Some(u) = &spec.base_url {
        b = b.base_url(u.clone());
    }
    if let Some(m) = &spec.model {
        b = b.model(m.clone());
    }
    if let Some(t) = spec.timeout_secs {
        b = b.timeout_secs(t);
    }
    if let Some(o) = &spec.organization {
        b = b.organization(o.clone());
    }
    if let Some(e) = &spec.azure_endpoint {
        b = b.azure_endpoint(e.clone());
    }
    if let Some(d) = &spec.azure_deployment {
        b = b.azure_deployment(d.clone());
    }
    if let Some(v) = &spec.azure_api_version {
        b = b.azure_api_version(v.clone());
    }
    let client = b.build()?;
    Ok(client.provider().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
    use crate::Message;
    use async_trait::async_trait;
    use cognis_core::RunnableStream;

    struct Fake(&'static str);

    #[async_trait]
    impl LLMProvider for Fake {
        fn name(&self) -> &str {
            self.0
        }
        fn provider_type(&self) -> crate::Provider {
            crate::Provider::Ollama
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.0),
                usage: None,
                finish_reason: "stop".into(),
                model: self.0.into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[test]
    fn registers_and_resolves_custom_provider() {
        let reg = ProviderRegistry::new()
            .register("toy", |spec| {
                let label = spec.model.clone().unwrap_or_else(|| "default".into());
                let leaked: &'static str = Box::leak(label.into_boxed_str());
                Ok(Arc::new(Fake(leaked)) as Arc<dyn LLMProvider>)
            })
            .register_alias("plaything", "toy");
        let p = reg.build("toy:weeble", ProviderSpec::default()).unwrap();
        assert_eq!(p.name(), "weeble");
        // alias works too
        let p2 = reg
            .build("plaything:wobble", ProviderSpec::default())
            .unwrap();
        assert_eq!(p2.name(), "wobble");
    }

    #[test]
    fn id_only_string_uses_spec_model() {
        let reg = ProviderRegistry::new().register("toy", |spec| {
            let label = spec.model.clone().unwrap_or_else(|| "default".into());
            let leaked: &'static str = Box::leak(label.into_boxed_str());
            Ok(Arc::new(Fake(leaked)) as Arc<dyn LLMProvider>)
        });
        let p = reg
            .build("toy", ProviderSpec::with_model("custom"))
            .unwrap();
        assert_eq!(p.name(), "custom");
    }

    #[test]
    fn unknown_provider_errors() {
        let reg = ProviderRegistry::new();
        // `Arc<dyn LLMProvider>` doesn't implement Debug, so `unwrap_err`
        // doesn't compile here — match the Result variant directly.
        let err = match reg.build("nope:m", ProviderSpec::default()) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("no provider registered"));
    }

    #[test]
    fn ids_sorted() {
        let reg = ProviderRegistry::new()
            .register("zeta", |_| Ok(Arc::new(Fake("z")) as Arc<dyn LLMProvider>))
            .register("alpha", |_| Ok(Arc::new(Fake("a")) as Arc<dyn LLMProvider>));
        assert_eq!(reg.ids(), vec!["alpha".to_string(), "zeta".into()]);
    }

    #[test]
    fn extras_round_trips_to_constructor() {
        use std::sync::Mutex;
        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_for_ctor = captured.clone();
        let reg = ProviderRegistry::new().register("x", move |spec| {
            *captured_for_ctor.lock().unwrap() = Some(spec.extras.clone());
            Ok(Arc::new(Fake("x")) as Arc<dyn LLMProvider>)
        });
        let _ = reg
            .build(
                "x",
                ProviderSpec::default().with_extras(serde_json::json!({"k": 1})),
            )
            .unwrap();
        let seen = captured.lock().unwrap().clone().unwrap();
        assert_eq!(seen["k"], 1);
    }
}
