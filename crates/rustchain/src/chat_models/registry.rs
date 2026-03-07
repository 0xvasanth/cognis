//! Chat model registry with configuration, capabilities, and model selection.
//!
//! Provides higher-level model management on top of `rustchain_core`'s
//! `BaseChatModel` / `ChatModel` traits. Includes:
//!
//! - [`ModelConfig`] — generation parameters (temperature, top_p, etc.) with
//!   builder pattern and merge support.
//! - [`ModelInfo`] — metadata about a model (provider, context window,
//!   capabilities, cost).
//! - [`ModelRegistry`] — collection of `ModelInfo` entries with lookup and
//!   capability filtering.
//! - [`ModelCapability`] — enum describing what a model supports (streaming,
//!   tool calling, vision, large context, JSON mode).
//! - [`ChatRequest`] / [`ChatResponse`] / [`TokenUsage`] — request/response
//!   types for chat interactions.
//! - [`ModelSelector`] — selects the best model from a registry based on
//!   capability requirements.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::messages::Message;

// ─── ModelConfig ───

/// Generation-level configuration for a chat model invocation.
///
/// Controls sampling parameters, token limits, stop sequences, and
/// provider-specific extras. Supports a builder pattern and merging.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The model name / identifier.
    pub model_name: String,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f64>,
    /// Stop sequences that halt generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Provider-specific extra parameters.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl ModelConfig {
    /// Create a new `ModelConfig` with the given model name.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the nucleus sampling parameter.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Add a stop sequence.
    pub fn with_stop_sequence(mut self, seq: impl Into<String>) -> Self {
        self.stop_sequences.push(seq.into());
        self
    }

    /// Set multiple stop sequences at once (replaces existing).
    pub fn with_stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }

    /// Set the request timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Insert a provider-specific extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Merge another config into this one. Fields that are `Some` in `other`
    /// override the corresponding fields in `self`. Stop sequences and extras
    /// from `other` are appended / inserted (overriding on key collision).
    pub fn merge(&mut self, other: &ModelConfig) {
        if !other.model_name.is_empty() {
            self.model_name = other.model_name.clone();
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if other.top_p.is_some() {
            self.top_p = other.top_p;
        }
        if !other.stop_sequences.is_empty() {
            self.stop_sequences = other.stop_sequences.clone();
        }
        if other.timeout_ms.is_some() {
            self.timeout_ms = other.timeout_ms;
        }
        for (k, v) in &other.extra {
            self.extra.insert(k.clone(), v.clone());
        }
    }
}

// ─── ModelInfo ───

/// Metadata describing a model's identity, capabilities, and cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Provider name (e.g. "anthropic", "openai").
    pub provider: String,
    /// Model identifier (e.g. "claude-sonnet-4-20250514").
    pub model_id: String,
    /// Maximum context window size in tokens.
    pub context_window: usize,
    /// Whether the model supports streaming responses.
    pub supports_streaming: bool,
    /// Whether the model supports tool / function calling.
    pub supports_tools: bool,
    /// Whether the model supports vision / image inputs.
    pub supports_vision: bool,
    /// Cost per input token (USD), if known.
    pub cost_per_input_token: Option<f64>,
    /// Cost per output token (USD), if known.
    pub cost_per_output_token: Option<f64>,
}

impl ModelInfo {
    /// Create a new `ModelInfo` with required fields; capability flags default
    /// to `false` and costs to `None`.
    pub fn new(
        provider: impl Into<String>,
        model_id: impl Into<String>,
        context_window: usize,
    ) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
            context_window,
            supports_streaming: false,
            supports_tools: false,
            supports_vision: false,
            cost_per_input_token: None,
            cost_per_output_token: None,
        }
    }

    /// Set streaming support.
    pub fn with_streaming(mut self, supports: bool) -> Self {
        self.supports_streaming = supports;
        self
    }

    /// Set tool-calling support.
    pub fn with_tools(mut self, supports: bool) -> Self {
        self.supports_tools = supports;
        self
    }

    /// Set vision support.
    pub fn with_vision(mut self, supports: bool) -> Self {
        self.supports_vision = supports;
        self
    }

    /// Set the cost per input token.
    pub fn with_input_cost(mut self, cost: f64) -> Self {
        self.cost_per_input_token = Some(cost);
        self
    }

    /// Set the cost per output token.
    pub fn with_output_cost(mut self, cost: f64) -> Self {
        self.cost_per_output_token = Some(cost);
        self
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ─── ModelCapability ───

/// A capability that a model may or may not support.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelCapability {
    /// The model supports streaming responses.
    Streaming,
    /// The model supports tool / function calling.
    ToolCalling,
    /// The model supports vision / image inputs.
    Vision,
    /// The model has a context window of at least `min_tokens` tokens.
    LargeContext(usize),
    /// The model supports structured JSON output mode.
    Json,
}

impl ModelCapability {
    /// Check whether the given [`ModelInfo`] satisfies this capability.
    pub fn matches(&self, info: &ModelInfo) -> bool {
        match self {
            ModelCapability::Streaming => info.supports_streaming,
            ModelCapability::ToolCalling => info.supports_tools,
            ModelCapability::Vision => info.supports_vision,
            ModelCapability::LargeContext(min) => info.context_window >= *min,
            // JSON mode is assumed available on models that support tool calling.
            ModelCapability::Json => info.supports_tools,
        }
    }
}

// ─── ModelRegistry ───

/// A collection of [`ModelInfo`] entries, indexed by `model_id`.
///
/// Supports registration, lookup, filtering by provider, and filtering by
/// capability.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a model. If a model with the same `model_id` already exists it
    /// is replaced.
    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.model_id.clone(), info);
    }

    /// Look up a model by its identifier.
    pub fn get(&self, model_id: &str) -> Option<&ModelInfo> {
        self.models.get(model_id)
    }

    /// Return all models from a given provider.
    pub fn by_provider(&self, provider: &str) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.provider == provider)
            .collect()
    }

    /// Return all models that satisfy a given capability.
    pub fn with_capability(&self, cap: &ModelCapability) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| cap.matches(m))
            .collect()
    }

    /// Return all registered models.
    pub fn all(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Serialize the registry to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self.models.values().map(|m| m.to_json()).collect();
        Value::Array(entries)
    }
}

// ─── TokenUsage ───

/// Token usage statistics for a chat model response.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Number of tokens in the prompt / input.
    pub prompt_tokens: usize,
    /// Number of tokens in the completion / output.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
}

impl TokenUsage {
    /// Create a new `TokenUsage`.
    pub fn new(prompt_tokens: usize, completion_tokens: usize, total_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ─── ChatRequest ───

/// A request to a chat model, bundling messages, configuration, and metadata.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// The conversation messages.
    pub messages: Vec<Message>,
    /// Generation configuration.
    pub config: ModelConfig,
    /// Arbitrary metadata attached to the request.
    pub metadata: HashMap<String, Value>,
}

impl ChatRequest {
    /// Create a new empty `ChatRequest` with default config.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            config: ModelConfig::default(),
            metadata: HashMap::new(),
        }
    }

    /// Append a message to the request.
    pub fn add_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Set the generation config.
    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Number of messages in the request.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ChatResponse ───

/// A response from a chat model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The generated text content.
    pub content: String,
    /// The model that produced this response.
    pub model: String,
    /// Token usage statistics.
    pub usage: TokenUsage,
    /// The reason generation stopped (e.g. "stop", "length").
    pub finish_reason: Option<String>,
    /// Arbitrary metadata attached to the response.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl ChatResponse {
    /// Create a new `ChatResponse`.
    pub fn new(content: impl Into<String>, model: impl Into<String>, usage: TokenUsage) -> Self {
        Self {
            content: content.into(),
            model: model.into(),
            usage,
            finish_reason: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the finish reason.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ─── ModelSelector ───

/// Selects models from a [`ModelRegistry`] based on capability requirements.
pub struct ModelSelector<'a> {
    registry: &'a ModelRegistry,
}

impl<'a> ModelSelector<'a> {
    /// Create a new selector backed by the given registry.
    pub fn new(registry: &'a ModelRegistry) -> Self {
        Self { registry }
    }

    /// Select the first model that meets all requirements.
    ///
    /// Returns `None` if no model satisfies every capability.
    pub fn select(&self, requirements: &[ModelCapability]) -> Option<&'a ModelInfo> {
        self.registry.models.values().find(|info| {
            requirements.iter().all(|cap| cap.matches(info))
        })
    }

    /// Select the cheapest model (by total cost per token) that meets all
    /// requirements. Cost is computed as `cost_per_input_token + cost_per_output_token`.
    /// Models without cost information are treated as having infinite cost and
    /// are only returned when no costed model qualifies.
    pub fn select_cheapest(&self, requirements: &[ModelCapability]) -> Option<&'a ModelInfo> {
        let mut candidates: Vec<&ModelInfo> = self
            .registry
            .models
            .values()
            .filter(|info| requirements.iter().all(|cap| cap.matches(info)))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by(|a, b| {
            let cost_a = total_cost(a);
            let cost_b = total_cost(b);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Some(candidates[0])
    }
}

/// Helper: compute total cost per token for sorting. Models without cost data
/// get `f64::MAX`.
fn total_cost(info: &ModelInfo) -> f64 {
    match (info.cost_per_input_token, info.cost_per_output_token) {
        (Some(i), Some(o)) => i + o,
        (Some(i), None) => i,
        (None, Some(o)) => o,
        (None, None) => f64::MAX,
    }
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──

    fn sample_info(id: &str, provider: &str) -> ModelInfo {
        ModelInfo::new(provider, id, 8192)
    }

    fn full_info() -> ModelInfo {
        ModelInfo::new("anthropic", "claude-sonnet", 200_000)
            .with_streaming(true)
            .with_tools(true)
            .with_vision(true)
            .with_input_cost(0.003)
            .with_output_cost(0.015)
    }

    // ── ModelConfig ──

    #[test]
    fn test_model_config_new_defaults() {
        let c = ModelConfig::new("gpt-4");
        assert_eq!(c.model_name, "gpt-4");
        assert!(c.temperature.is_none());
        assert!(c.max_tokens.is_none());
        assert!(c.top_p.is_none());
        assert!(c.stop_sequences.is_empty());
        assert!(c.timeout_ms.is_none());
        assert!(c.extra.is_empty());
    }

    #[test]
    fn test_model_config_builder() {
        let c = ModelConfig::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_top_p(0.9)
            .with_stop_sequence("END")
            .with_stop_sequence("STOP")
            .with_timeout_ms(5000)
            .with_extra("foo", serde_json::json!("bar"));

        assert_eq!(c.temperature, Some(0.7));
        assert_eq!(c.max_tokens, Some(1024));
        assert_eq!(c.top_p, Some(0.9));
        assert_eq!(c.stop_sequences, vec!["END", "STOP"]);
        assert_eq!(c.timeout_ms, Some(5000));
        assert_eq!(c.extra["foo"], serde_json::json!("bar"));
    }

    #[test]
    fn test_model_config_with_stop_sequences_replaces() {
        let c = ModelConfig::new("m")
            .with_stop_sequence("a")
            .with_stop_sequences(vec!["x".into(), "y".into()]);
        assert_eq!(c.stop_sequences, vec!["x", "y"]);
    }

    #[test]
    fn test_model_config_to_json() {
        let c = ModelConfig::new("m").with_temperature(0.5);
        let j = c.to_json();
        assert_eq!(j["model_name"], "m");
        assert_eq!(j["temperature"], 0.5);
    }

    #[test]
    fn test_model_config_merge_overrides() {
        let mut base = ModelConfig::new("base")
            .with_temperature(0.5)
            .with_max_tokens(100)
            .with_timeout_ms(1000);

        let override_cfg = ModelConfig::new("override")
            .with_temperature(0.9)
            .with_top_p(0.8);

        base.merge(&override_cfg);
        assert_eq!(base.model_name, "override");
        assert_eq!(base.temperature, Some(0.9));
        assert_eq!(base.max_tokens, Some(100)); // kept from base
        assert_eq!(base.top_p, Some(0.8)); // new from override
        assert_eq!(base.timeout_ms, Some(1000)); // kept from base
    }

    #[test]
    fn test_model_config_merge_empty_model_name_keeps_base() {
        let mut base = ModelConfig::new("base");
        let empty = ModelConfig::default();
        base.merge(&empty);
        assert_eq!(base.model_name, "base");
    }

    #[test]
    fn test_model_config_merge_stop_sequences() {
        let mut base = ModelConfig::new("m")
            .with_stop_sequences(vec!["a".into()]);
        let other = ModelConfig::new("")
            .with_stop_sequences(vec!["x".into(), "y".into()]);
        base.merge(&other);
        assert_eq!(base.stop_sequences, vec!["x", "y"]);
    }

    #[test]
    fn test_model_config_merge_extras() {
        let mut base = ModelConfig::new("m")
            .with_extra("a", serde_json::json!(1));
        let other = ModelConfig::new("")
            .with_extra("b", serde_json::json!(2))
            .with_extra("a", serde_json::json!(99));
        base.merge(&other);
        assert_eq!(base.extra["a"], serde_json::json!(99));
        assert_eq!(base.extra["b"], serde_json::json!(2));
    }

    #[test]
    fn test_model_config_serialize_roundtrip() {
        let c = ModelConfig::new("test")
            .with_temperature(0.7)
            .with_max_tokens(512)
            .with_stop_sequence("END");
        let json = serde_json::to_string(&c).unwrap();
        let c2: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.model_name, "test");
        assert_eq!(c2.temperature, Some(0.7));
        assert_eq!(c2.max_tokens, Some(512));
        assert_eq!(c2.stop_sequences, vec!["END"]);
    }

    // ── ModelInfo ──

    #[test]
    fn test_model_info_new_defaults() {
        let info = ModelInfo::new("openai", "gpt-4", 128_000);
        assert_eq!(info.provider, "openai");
        assert_eq!(info.model_id, "gpt-4");
        assert_eq!(info.context_window, 128_000);
        assert!(!info.supports_streaming);
        assert!(!info.supports_tools);
        assert!(!info.supports_vision);
        assert!(info.cost_per_input_token.is_none());
        assert!(info.cost_per_output_token.is_none());
    }

    #[test]
    fn test_model_info_builder() {
        let info = full_info();
        assert!(info.supports_streaming);
        assert!(info.supports_tools);
        assert!(info.supports_vision);
        assert_eq!(info.cost_per_input_token, Some(0.003));
        assert_eq!(info.cost_per_output_token, Some(0.015));
    }

    #[test]
    fn test_model_info_to_json() {
        let info = ModelInfo::new("p", "m", 4096).with_streaming(true);
        let j = info.to_json();
        assert_eq!(j["provider"], "p");
        assert_eq!(j["model_id"], "m");
        assert_eq!(j["context_window"], 4096);
        assert_eq!(j["supports_streaming"], true);
    }

    // ── ModelCapability ──

    #[test]
    fn test_capability_streaming() {
        let info = ModelInfo::new("p", "m", 4096).with_streaming(true);
        assert!(ModelCapability::Streaming.matches(&info));

        let no = ModelInfo::new("p", "m2", 4096);
        assert!(!ModelCapability::Streaming.matches(&no));
    }

    #[test]
    fn test_capability_tool_calling() {
        let info = ModelInfo::new("p", "m", 4096).with_tools(true);
        assert!(ModelCapability::ToolCalling.matches(&info));
        assert!(!ModelCapability::ToolCalling.matches(&ModelInfo::new("p", "m2", 4096)));
    }

    #[test]
    fn test_capability_vision() {
        let info = ModelInfo::new("p", "m", 4096).with_vision(true);
        assert!(ModelCapability::Vision.matches(&info));
        assert!(!ModelCapability::Vision.matches(&ModelInfo::new("p", "m2", 4096)));
    }

    #[test]
    fn test_capability_large_context() {
        let info = ModelInfo::new("p", "m", 128_000);
        assert!(ModelCapability::LargeContext(100_000).matches(&info));
        assert!(ModelCapability::LargeContext(128_000).matches(&info));
        assert!(!ModelCapability::LargeContext(200_000).matches(&info));
    }

    #[test]
    fn test_capability_json_maps_to_tools() {
        let info = ModelInfo::new("p", "m", 4096).with_tools(true);
        assert!(ModelCapability::Json.matches(&info));
        assert!(!ModelCapability::Json.matches(&ModelInfo::new("p", "m2", 4096)));
    }

    // ── ModelRegistry ──

    #[test]
    fn test_registry_new_is_empty() {
        let reg = ModelRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.all().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ModelRegistry::new();
        reg.register(sample_info("m1", "p1"));
        assert_eq!(reg.len(), 1);
        let info = reg.get("m1").unwrap();
        assert_eq!(info.provider, "p1");
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = ModelRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_register_replaces() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelInfo::new("p1", "m1", 4096));
        reg.register(ModelInfo::new("p2", "m1", 8192));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("m1").unwrap().provider, "p2");
        assert_eq!(reg.get("m1").unwrap().context_window, 8192);
    }

    #[test]
    fn test_registry_by_provider() {
        let mut reg = ModelRegistry::new();
        reg.register(sample_info("a1", "anthropic"));
        reg.register(sample_info("a2", "anthropic"));
        reg.register(sample_info("o1", "openai"));

        let anthropic = reg.by_provider("anthropic");
        assert_eq!(anthropic.len(), 2);
        let openai = reg.by_provider("openai");
        assert_eq!(openai.len(), 1);
        let google = reg.by_provider("google");
        assert!(google.is_empty());
    }

    #[test]
    fn test_registry_with_capability() {
        let mut reg = ModelRegistry::new();
        reg.register(ModelInfo::new("p", "stream", 4096).with_streaming(true));
        reg.register(ModelInfo::new("p", "nostream", 4096));

        let streaming = reg.with_capability(&ModelCapability::Streaming);
        assert_eq!(streaming.len(), 1);
        assert_eq!(streaming[0].model_id, "stream");
    }

    #[test]
    fn test_registry_all() {
        let mut reg = ModelRegistry::new();
        reg.register(sample_info("a", "p"));
        reg.register(sample_info("b", "p"));
        reg.register(sample_info("c", "q"));
        assert_eq!(reg.all().len(), 3);
    }

    #[test]
    fn test_registry_to_json() {
        let mut reg = ModelRegistry::new();
        reg.register(sample_info("m1", "p1"));
        let j = reg.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
    }

    // ── TokenUsage ──

    #[test]
    fn test_token_usage_new() {
        let u = TokenUsage::new(100, 50, 150);
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(u.total_tokens, 150);
    }

    #[test]
    fn test_token_usage_default() {
        let u = TokenUsage::default();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 0);
        assert_eq!(u.total_tokens, 0);
    }

    #[test]
    fn test_token_usage_to_json() {
        let u = TokenUsage::new(10, 20, 30);
        let j = u.to_json();
        assert_eq!(j["prompt_tokens"], 10);
        assert_eq!(j["completion_tokens"], 20);
        assert_eq!(j["total_tokens"], 30);
    }

    #[test]
    fn test_token_usage_equality() {
        let a = TokenUsage::new(1, 2, 3);
        let b = TokenUsage::new(1, 2, 3);
        let c = TokenUsage::new(1, 2, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── ChatRequest ──

    #[test]
    fn test_chat_request_new_empty() {
        let req = ChatRequest::new();
        assert_eq!(req.message_count(), 0);
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn test_chat_request_add_messages() {
        let req = ChatRequest::new()
            .add_message(Message::human("hello"))
            .add_message(Message::ai("hi"))
            .add_message(Message::human("how are you?"));
        assert_eq!(req.message_count(), 3);
    }

    #[test]
    fn test_chat_request_with_config() {
        let cfg = ModelConfig::new("gpt-4").with_temperature(0.5);
        let req = ChatRequest::new().with_config(cfg);
        assert_eq!(req.config.model_name, "gpt-4");
        assert_eq!(req.config.temperature, Some(0.5));
    }

    #[test]
    fn test_chat_request_with_metadata() {
        let req = ChatRequest::new()
            .with_metadata("trace_id", serde_json::json!("abc-123"))
            .with_metadata("user", serde_json::json!("alice"));
        assert_eq!(req.metadata.len(), 2);
        assert_eq!(req.metadata["trace_id"], serde_json::json!("abc-123"));
    }

    #[test]
    fn test_chat_request_default() {
        let req = ChatRequest::default();
        assert_eq!(req.message_count(), 0);
    }

    // ── ChatResponse ──

    #[test]
    fn test_chat_response_new() {
        let usage = TokenUsage::new(10, 20, 30);
        let resp = ChatResponse::new("Hello!", "gpt-4", usage.clone());
        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.usage, usage);
        assert!(resp.finish_reason.is_none());
        assert!(resp.metadata.is_empty());
    }

    #[test]
    fn test_chat_response_with_finish_reason() {
        let resp =
            ChatResponse::new("done", "m", TokenUsage::default()).with_finish_reason("stop");
        assert_eq!(resp.finish_reason, Some("stop".into()));
    }

    #[test]
    fn test_chat_response_with_metadata() {
        let resp = ChatResponse::new("x", "m", TokenUsage::default())
            .with_metadata("latency_ms", serde_json::json!(123));
        assert_eq!(resp.metadata["latency_ms"], serde_json::json!(123));
    }

    #[test]
    fn test_chat_response_to_json() {
        let resp = ChatResponse::new("hi", "m", TokenUsage::new(1, 2, 3))
            .with_finish_reason("stop");
        let j = resp.to_json();
        assert_eq!(j["content"], "hi");
        assert_eq!(j["model"], "m");
        assert_eq!(j["finish_reason"], "stop");
        assert_eq!(j["usage"]["total_tokens"], 3);
    }

    #[test]
    fn test_chat_response_serialize_roundtrip() {
        let resp = ChatResponse::new("text", "model", TokenUsage::new(5, 10, 15))
            .with_finish_reason("length")
            .with_metadata("k", serde_json::json!("v"));
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: ChatResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp2.content, "text");
        assert_eq!(resp2.finish_reason, Some("length".into()));
        assert_eq!(resp2.metadata["k"], serde_json::json!("v"));
    }

    // ── ModelSelector ──

    fn build_registry() -> ModelRegistry {
        let mut reg = ModelRegistry::new();
        reg.register(
            ModelInfo::new("anthropic", "claude-haiku", 200_000)
                .with_streaming(true)
                .with_tools(true)
                .with_input_cost(0.00025)
                .with_output_cost(0.00125),
        );
        reg.register(
            ModelInfo::new("anthropic", "claude-sonnet", 200_000)
                .with_streaming(true)
                .with_tools(true)
                .with_vision(true)
                .with_input_cost(0.003)
                .with_output_cost(0.015),
        );
        reg.register(
            ModelInfo::new("openai", "gpt-4o-mini", 128_000)
                .with_streaming(true)
                .with_tools(true)
                .with_vision(true)
                .with_input_cost(0.00015)
                .with_output_cost(0.0006),
        );
        reg.register(
            ModelInfo::new("openai", "gpt-3.5-turbo", 16_385)
                .with_streaming(true)
                .with_input_cost(0.0005)
                .with_output_cost(0.0015),
        );
        reg
    }

    #[test]
    fn test_selector_select_any() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        // No requirements — should return some model.
        let result = sel.select(&[]);
        assert!(result.is_some());
    }

    #[test]
    fn test_selector_select_with_vision() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select(&[ModelCapability::Vision]);
        assert!(result.is_some());
        assert!(result.unwrap().supports_vision);
    }

    #[test]
    fn test_selector_select_no_match() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        // No model has a 1M context window.
        let result = sel.select(&[ModelCapability::LargeContext(1_000_000)]);
        assert!(result.is_none());
    }

    #[test]
    fn test_selector_select_multiple_requirements() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select(&[
            ModelCapability::Vision,
            ModelCapability::ToolCalling,
            ModelCapability::Streaming,
        ]);
        assert!(result.is_some());
        let info = result.unwrap();
        assert!(info.supports_vision);
        assert!(info.supports_tools);
        assert!(info.supports_streaming);
    }

    #[test]
    fn test_selector_select_cheapest_no_requirements() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select_cheapest(&[]);
        assert!(result.is_some());
        // gpt-4o-mini has the lowest total cost (0.00015 + 0.0006 = 0.00075)
        assert_eq!(result.unwrap().model_id, "gpt-4o-mini");
    }

    #[test]
    fn test_selector_select_cheapest_with_vision() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select_cheapest(&[ModelCapability::Vision]);
        assert!(result.is_some());
        // gpt-4o-mini is cheaper than claude-sonnet and both support vision
        assert_eq!(result.unwrap().model_id, "gpt-4o-mini");
    }

    #[test]
    fn test_selector_select_cheapest_no_match() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select_cheapest(&[ModelCapability::LargeContext(1_000_000)]);
        assert!(result.is_none());
    }

    #[test]
    fn test_selector_select_cheapest_no_cost_fallback() {
        let mut reg = ModelRegistry::new();
        reg.register(
            ModelInfo::new("p", "no-cost", 4096)
                .with_streaming(true),
        );
        let sel = ModelSelector::new(&reg);
        // Even without cost info it should still be returned.
        let result = sel.select_cheapest(&[ModelCapability::Streaming]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().model_id, "no-cost");
    }

    #[test]
    fn test_selector_empty_registry() {
        let reg = ModelRegistry::new();
        let sel = ModelSelector::new(&reg);
        assert!(sel.select(&[]).is_none());
        assert!(sel.select_cheapest(&[]).is_none());
    }

    #[test]
    fn test_selector_select_cheapest_with_tools() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select_cheapest(&[ModelCapability::ToolCalling]);
        assert!(result.is_some());
        // gpt-4o-mini is cheapest with tools
        assert_eq!(result.unwrap().model_id, "gpt-4o-mini");
    }

    #[test]
    fn test_selector_select_cheapest_large_context_and_vision() {
        let reg = build_registry();
        let sel = ModelSelector::new(&reg);
        let result = sel.select_cheapest(&[
            ModelCapability::LargeContext(200_000),
            ModelCapability::Vision,
        ]);
        assert!(result.is_some());
        // Only claude-sonnet has 200k context + vision
        assert_eq!(result.unwrap().model_id, "claude-sonnet");
    }
}
