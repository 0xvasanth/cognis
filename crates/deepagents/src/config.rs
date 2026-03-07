//! Configuration for the Deep Agent.
//!
//! This module provides a layered configuration system:
//!
//! - [`ModelConfig`] — LLM provider and model settings
//! - [`MiddlewareConfig`] — which middleware to enable and their settings
//! - [`BackendConfig`] — state persistence backend selection
//! - [`AgentConfig`] — top-level declarative agent configuration (serializable)
//! - [`AgentConfigBuilder`] — fluent builder for [`AgentConfig`]
//! - [`ConfigLoader`] — load/merge configs from JSON, env vars, etc.
//! - [`DeepAgentConfig`] — runtime config with concrete tool/middleware/backend instances

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::tools::BaseTool;

use crate::backends::{Backend, StateBackend};
use crate::middleware::Middleware;

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Configuration for the LLM model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Provider name (e.g. `"anthropic"`, `"openai"`, `"google"`).
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    #[serde(default = "default_model_name")]
    pub model_name: String,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// API key override (if not set, read from environment).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

fn default_provider() -> String {
    "anthropic".to_string()
}

fn default_model_name() -> String {
    "claude-sonnet-4-20250514".to_string()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model_name: default_model_name(),
            temperature: None,
            max_tokens: None,
            api_key: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MiddlewareConfig
// ---------------------------------------------------------------------------

/// Declarative middleware enablement flags and settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MiddlewareConfig {
    /// Enable the filesystem middleware.
    #[serde(default = "default_true")]
    pub enable_filesystem: bool,
    /// Enable the memory middleware.
    #[serde(default = "default_true")]
    pub enable_memory: bool,
    /// Enable the summarization middleware.
    #[serde(default)]
    pub enable_summarization: bool,
    /// Enable the planning middleware.
    #[serde(default)]
    pub enable_planning: bool,
    /// Enable the logging middleware.
    #[serde(default)]
    pub enable_logging: bool,
    /// Enable the rate limiter middleware.
    #[serde(default)]
    pub enable_rate_limiter: bool,
    /// Root directory for the filesystem middleware.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filesystem_root: Option<PathBuf>,
    /// Maximum context window tokens for summarization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<usize>,
    /// Names of additional custom middleware to load.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub custom_middleware: Vec<String>,
}

fn default_true() -> bool {
    true
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self {
            enable_filesystem: true,
            enable_memory: true,
            enable_summarization: false,
            enable_planning: false,
            enable_logging: false,
            enable_rate_limiter: false,
            filesystem_root: None,
            max_context_tokens: None,
            custom_middleware: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// BackendConfig
// ---------------------------------------------------------------------------

/// Declarative backend selection.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendConfig {
    /// In-memory state backend (default).
    #[default]
    InMemory,
    /// Filesystem-backed persistence.
    Filesystem {
        /// Directory to store session state files.
        path: PathBuf,
    },
    /// Sandboxed execution backend with resource limits.
    Sandbox {
        /// Directories the sandbox is allowed to access.
        allowed_paths: Vec<PathBuf>,
        /// Maximum memory in megabytes.
        max_memory_mb: usize,
    },
}

// ---------------------------------------------------------------------------
// AgentConfig
// ---------------------------------------------------------------------------

/// Top-level declarative agent configuration.
///
/// This is a fully serializable/deserializable description of an agent. It can
/// be loaded from JSON, environment variables, or built with [`AgentConfigBuilder`].
/// It does *not* contain concrete tool or middleware instances — use
/// [`DeepAgentConfig`] for the runtime configuration passed to
/// [`create_deep_agent`](crate::create_deep_agent).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    /// Human-readable agent name.
    pub name: String,
    /// Optional description of the agent's purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional system prompt prepended to the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Model configuration.
    #[serde(default)]
    pub model: ModelConfig,
    /// Middleware configuration.
    #[serde(default)]
    pub middleware: MiddlewareConfig,
    /// Backend configuration.
    #[serde(default)]
    pub backend: BackendConfig,
    /// Maximum number of agent loop iterations.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Names of tools to enable.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Arbitrary key-value tags for categorization / filtering.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
    /// Whether to enable verbose logging.
    #[serde(default)]
    pub verbose: bool,
}

fn default_max_iterations() -> usize {
    50
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "default-agent".to_string(),
            description: None,
            system_prompt: None,
            model: ModelConfig::default(),
            middleware: MiddlewareConfig::default(),
            backend: BackendConfig::default(),
            max_iterations: default_max_iterations(),
            tools: Vec::new(),
            tags: HashMap::new(),
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AgentConfigBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`AgentConfig`].
#[derive(Debug, Default)]
pub struct AgentConfigBuilder {
    config: AgentConfig,
}

impl AgentConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set the agent description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Set the model configuration.
    pub fn model(mut self, model: ModelConfig) -> Self {
        self.config.model = model;
        self
    }

    /// Set the model provider.
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.config.model.provider = provider.into();
        self
    }

    /// Set the model name.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.config.model.model_name = model_name.into();
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.model.temperature = Some(temperature);
        self
    }

    /// Set the max tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.model.max_tokens = Some(max_tokens);
        self
    }

    /// Set the middleware configuration.
    pub fn middleware(mut self, middleware: MiddlewareConfig) -> Self {
        self.config.middleware = middleware;
        self
    }

    /// Enable or disable the filesystem middleware.
    pub fn enable_filesystem(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_filesystem = enabled;
        self
    }

    /// Enable or disable the memory middleware.
    pub fn enable_memory(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_memory = enabled;
        self
    }

    /// Enable or disable the summarization middleware.
    pub fn enable_summarization(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_summarization = enabled;
        self
    }

    /// Enable or disable the planning middleware.
    pub fn enable_planning(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_planning = enabled;
        self
    }

    /// Enable or disable the logging middleware.
    pub fn enable_logging(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_logging = enabled;
        self
    }

    /// Enable or disable the rate limiter middleware.
    pub fn enable_rate_limiter(mut self, enabled: bool) -> Self {
        self.config.middleware.enable_rate_limiter = enabled;
        self
    }

    /// Set the filesystem root for the filesystem middleware.
    pub fn filesystem_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.middleware.filesystem_root = Some(path.into());
        self
    }

    /// Set the maximum context tokens for summarization.
    pub fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.config.middleware.max_context_tokens = Some(tokens);
        self
    }

    /// Set the backend configuration.
    pub fn backend(mut self, backend: BackendConfig) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Add a tool name to the enabled tools list.
    pub fn tool(mut self, tool_name: impl Into<String>) -> Self {
        self.config.tools.push(tool_name.into());
        self
    }

    /// Set all tool names at once.
    pub fn tools(mut self, tools: Vec<String>) -> Self {
        self.config.tools = tools;
        self
    }

    /// Add a tag.
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.tags.insert(key.into(), value.into());
        self
    }

    /// Set all tags at once.
    pub fn tags(mut self, tags: HashMap<String, String>) -> Self {
        self.config.tags = tags;
        self
    }

    /// Set verbose mode.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Build the [`AgentConfig`].
    pub fn build(self) -> AgentConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// ConfigLoader
// ---------------------------------------------------------------------------

/// Errors that can occur during configuration loading.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Environment variable error.
    #[error("env error: {0}")]
    Env(String),
}

/// Load and merge [`AgentConfig`] from various sources.
pub struct ConfigLoader;

impl ConfigLoader {
    /// Deserialize an [`AgentConfig`] from a JSON string.
    pub fn from_json(json: &str) -> Result<AgentConfig, ConfigError> {
        let config: AgentConfig = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Deserialize an [`AgentConfig`] from a [`serde_json::Value`].
    pub fn from_value(value: Value) -> Result<AgentConfig, ConfigError> {
        let config: AgentConfig = serde_json::from_value(value)?;
        Ok(config)
    }

    /// Build an [`AgentConfig`] from `DEEP_AGENT_*` environment variables.
    ///
    /// Supported variables:
    /// - `DEEP_AGENT_NAME` — agent name
    /// - `DEEP_AGENT_DESCRIPTION` — agent description
    /// - `DEEP_AGENT_SYSTEM_PROMPT` — system prompt
    /// - `DEEP_AGENT_PROVIDER` — model provider
    /// - `DEEP_AGENT_MODEL` — model name
    /// - `DEEP_AGENT_TEMPERATURE` — sampling temperature
    /// - `DEEP_AGENT_MAX_TOKENS` — max tokens
    /// - `DEEP_AGENT_API_KEY` — API key
    /// - `DEEP_AGENT_MAX_ITERATIONS` — max iterations
    /// - `DEEP_AGENT_TOOLS` — comma-separated tool names
    /// - `DEEP_AGENT_VERBOSE` — `"true"` or `"1"` for verbose
    /// - `DEEP_AGENT_BACKEND` — `"in_memory"`, `"filesystem"`, or `"sandbox"`
    /// - `DEEP_AGENT_BACKEND_PATH` — path for filesystem backend
    /// - `DEEP_AGENT_FILESYSTEM_ROOT` — root for filesystem middleware
    /// - `DEEP_AGENT_ENABLE_FILESYSTEM` — `"true"`/`"false"`
    /// - `DEEP_AGENT_ENABLE_MEMORY` — `"true"`/`"false"`
    /// - `DEEP_AGENT_ENABLE_SUMMARIZATION` — `"true"`/`"false"`
    /// - `DEEP_AGENT_ENABLE_PLANNING` — `"true"`/`"false"`
    /// - `DEEP_AGENT_ENABLE_LOGGING` — `"true"`/`"false"`
    /// - `DEEP_AGENT_ENABLE_RATE_LIMITER` — `"true"`/`"false"`
    pub fn from_env() -> Result<AgentConfig, ConfigError> {
        let mut config = AgentConfig::default();

        if let Ok(v) = std::env::var("DEEP_AGENT_NAME") {
            config.name = v;
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_DESCRIPTION") {
            config.description = Some(v);
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_SYSTEM_PROMPT") {
            config.system_prompt = Some(v);
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_PROVIDER") {
            config.model.provider = v;
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_MODEL") {
            config.model.model_name = v;
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_TEMPERATURE") {
            config.model.temperature =
                Some(v.parse::<f64>().map_err(|e| {
                    ConfigError::Env(format!("invalid DEEP_AGENT_TEMPERATURE: {e}"))
                })?);
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_MAX_TOKENS") {
            config.model.max_tokens =
                Some(v.parse::<u32>().map_err(|e| {
                    ConfigError::Env(format!("invalid DEEP_AGENT_MAX_TOKENS: {e}"))
                })?);
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_API_KEY") {
            config.model.api_key = Some(v);
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_MAX_ITERATIONS") {
            config.max_iterations = v
                .parse::<usize>()
                .map_err(|e| ConfigError::Env(format!("invalid DEEP_AGENT_MAX_ITERATIONS: {e}")))?;
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_TOOLS") {
            config.tools = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_VERBOSE") {
            config.verbose = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_BACKEND") {
            config.backend = match v.as_str() {
                "filesystem" => {
                    let path = std::env::var("DEEP_AGENT_BACKEND_PATH")
                        .map(PathBuf::from)
                        .map_err(|_| {
                            ConfigError::Env(
                                "DEEP_AGENT_BACKEND_PATH required for filesystem backend"
                                    .to_string(),
                            )
                        })?;
                    BackendConfig::Filesystem { path }
                }
                "sandbox" => BackendConfig::Sandbox {
                    allowed_paths: Vec::new(),
                    max_memory_mb: 512,
                },
                _ => BackendConfig::InMemory,
            };
        }
        if let Ok(v) = std::env::var("DEEP_AGENT_FILESYSTEM_ROOT") {
            config.middleware.filesystem_root = Some(PathBuf::from(v));
        }

        fn parse_bool_env(var: &str) -> Option<bool> {
            std::env::var(var).ok().map(|v| v == "true" || v == "1")
        }

        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_FILESYSTEM") {
            config.middleware.enable_filesystem = v;
        }
        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_MEMORY") {
            config.middleware.enable_memory = v;
        }
        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_SUMMARIZATION") {
            config.middleware.enable_summarization = v;
        }
        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_PLANNING") {
            config.middleware.enable_planning = v;
        }
        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_LOGGING") {
            config.middleware.enable_logging = v;
        }
        if let Some(v) = parse_bool_env("DEEP_AGENT_ENABLE_RATE_LIMITER") {
            config.middleware.enable_rate_limiter = v;
        }

        Ok(config)
    }

    /// Merge two configs, with `overrides` taking precedence over `base`.
    ///
    /// Fields in `overrides` that are non-default replace the corresponding
    /// fields in `base`. For `Option` fields, `Some` in overrides wins. For
    /// collections (`tools`, `tags`, `custom_middleware`), the override
    /// replaces the base if non-empty.
    pub fn merge(base: AgentConfig, overrides: AgentConfig) -> AgentConfig {
        let defaults = AgentConfig::default();

        AgentConfig {
            name: if overrides.name != defaults.name {
                overrides.name
            } else {
                base.name
            },
            description: overrides.description.or(base.description),
            system_prompt: overrides.system_prompt.or(base.system_prompt),
            model: ModelConfig {
                provider: if overrides.model.provider != defaults.model.provider {
                    overrides.model.provider
                } else {
                    base.model.provider
                },
                model_name: if overrides.model.model_name != defaults.model.model_name {
                    overrides.model.model_name
                } else {
                    base.model.model_name
                },
                temperature: overrides.model.temperature.or(base.model.temperature),
                max_tokens: overrides.model.max_tokens.or(base.model.max_tokens),
                api_key: overrides.model.api_key.or(base.model.api_key),
            },
            middleware: MiddlewareConfig {
                enable_filesystem: overrides.middleware.enable_filesystem,
                enable_memory: overrides.middleware.enable_memory,
                enable_summarization: overrides.middleware.enable_summarization
                    || base.middleware.enable_summarization,
                enable_planning: overrides.middleware.enable_planning
                    || base.middleware.enable_planning,
                enable_logging: overrides.middleware.enable_logging
                    || base.middleware.enable_logging,
                enable_rate_limiter: overrides.middleware.enable_rate_limiter
                    || base.middleware.enable_rate_limiter,
                filesystem_root: overrides
                    .middleware
                    .filesystem_root
                    .or(base.middleware.filesystem_root),
                max_context_tokens: overrides
                    .middleware
                    .max_context_tokens
                    .or(base.middleware.max_context_tokens),
                custom_middleware: if !overrides.middleware.custom_middleware.is_empty() {
                    overrides.middleware.custom_middleware
                } else {
                    base.middleware.custom_middleware
                },
            },
            backend: if overrides.backend != defaults.backend {
                overrides.backend
            } else {
                base.backend
            },
            max_iterations: if overrides.max_iterations != defaults.max_iterations {
                overrides.max_iterations
            } else {
                base.max_iterations
            },
            tools: if !overrides.tools.is_empty() {
                overrides.tools
            } else {
                base.tools
            },
            tags: if !overrides.tags.is_empty() {
                let mut merged = base.tags;
                merged.extend(overrides.tags);
                merged
            } else {
                base.tags
            },
            verbose: overrides.verbose || base.verbose,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepAgentConfig (runtime configuration — backwards compatible)
// ---------------------------------------------------------------------------

/// Runtime configuration for building a Deep Agent via [`create_deep_agent`](crate::create_deep_agent).
///
/// This struct holds concrete instances of tools, middleware, and backends.
/// For a serializable/declarative configuration, see [`AgentConfig`].
pub struct DeepAgentConfig {
    /// The model identifier to use (e.g. `"claude-sonnet-4-6"`).
    pub model_name: String,
    /// Maximum number of agent loop iterations before stopping.
    pub max_iterations: u32,
    /// Optional system prompt prepended to the conversation.
    pub system_prompt: Option<String>,
    /// Tools available to the agent.
    pub tools: Vec<Arc<dyn BaseTool>>,
    /// Middleware pipeline applied around model and tool calls.
    pub middleware: Vec<Arc<dyn Middleware>>,
    /// Backend for persisting agent state across sessions.
    pub backend: Box<dyn Backend>,
}

impl Default for DeepAgentConfig {
    fn default() -> Self {
        Self {
            model_name: "claude-sonnet-4-6".to_string(),
            max_iterations: 25,
            system_prompt: None,
            tools: Vec::new(),
            middleware: Vec::new(),
            backend: Box::new(StateBackend::new()),
        }
    }
}

impl DeepAgentConfig {
    /// Create a new config with the given model name.
    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a tool to the agent.
    pub fn with_tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set all tools at once.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a middleware to the pipeline.
    pub fn with_middleware(mut self, mw: Arc<dyn Middleware>) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Set the backend.
    pub fn with_backend(mut self, backend: Box<dyn Backend>) -> Self {
        self.backend = backend;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelConfig tests --

    #[test]
    fn test_model_config_defaults() {
        let mc = ModelConfig::default();
        assert_eq!(mc.provider, "anthropic");
        assert_eq!(mc.model_name, "claude-sonnet-4-20250514");
        assert!(mc.temperature.is_none());
        assert!(mc.max_tokens.is_none());
        assert!(mc.api_key.is_none());
    }

    #[test]
    fn test_model_config_serialize_roundtrip() {
        let mc = ModelConfig {
            provider: "openai".to_string(),
            model_name: "gpt-4".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
            api_key: Some("sk-test".to_string()),
        };
        let json = serde_json::to_string(&mc).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(mc, deserialized);
    }

    #[test]
    fn test_model_config_deserialize_partial() {
        let json = r#"{"provider": "google"}"#;
        let mc: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(mc.provider, "google");
        assert_eq!(mc.model_name, "claude-sonnet-4-20250514"); // default
    }

    // -- MiddlewareConfig tests --

    #[test]
    fn test_middleware_config_defaults() {
        let mc = MiddlewareConfig::default();
        assert!(mc.enable_filesystem);
        assert!(mc.enable_memory);
        assert!(!mc.enable_summarization);
        assert!(!mc.enable_planning);
        assert!(!mc.enable_logging);
        assert!(!mc.enable_rate_limiter);
        assert!(mc.filesystem_root.is_none());
        assert!(mc.max_context_tokens.is_none());
        assert!(mc.custom_middleware.is_empty());
    }

    #[test]
    fn test_middleware_config_serialize_roundtrip() {
        let mc = MiddlewareConfig {
            enable_filesystem: false,
            enable_memory: true,
            enable_summarization: true,
            enable_planning: false,
            enable_logging: true,
            enable_rate_limiter: false,
            filesystem_root: Some(PathBuf::from("/tmp/agent")),
            max_context_tokens: Some(8000),
            custom_middleware: vec!["my_custom".to_string()],
        };
        let json = serde_json::to_string(&mc).unwrap();
        let deserialized: MiddlewareConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(mc, deserialized);
    }

    // -- BackendConfig tests --

    #[test]
    fn test_backend_config_default_is_in_memory() {
        let bc = BackendConfig::default();
        assert_eq!(bc, BackendConfig::InMemory);
    }

    #[test]
    fn test_backend_config_filesystem_roundtrip() {
        let bc = BackendConfig::Filesystem {
            path: PathBuf::from("/data/sessions"),
        };
        let json = serde_json::to_string(&bc).unwrap();
        let deserialized: BackendConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(bc, deserialized);
    }

    #[test]
    fn test_backend_config_sandbox_roundtrip() {
        let bc = BackendConfig::Sandbox {
            allowed_paths: vec![PathBuf::from("/tmp"), PathBuf::from("/home/user")],
            max_memory_mb: 1024,
        };
        let json = serde_json::to_string(&bc).unwrap();
        let deserialized: BackendConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(bc, deserialized);
    }

    // -- AgentConfig tests --

    #[test]
    fn test_agent_config_defaults() {
        let ac = AgentConfig::default();
        assert_eq!(ac.name, "default-agent");
        assert!(ac.description.is_none());
        assert!(ac.system_prompt.is_none());
        assert_eq!(ac.max_iterations, 50);
        assert!(ac.tools.is_empty());
        assert!(ac.tags.is_empty());
        assert!(!ac.verbose);
    }

    #[test]
    fn test_agent_config_full_serialize_roundtrip() {
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "production".to_string());

        let ac = AgentConfig {
            name: "test-agent".to_string(),
            description: Some("A test agent".to_string()),
            system_prompt: Some("You are helpful.".to_string()),
            model: ModelConfig {
                provider: "openai".to_string(),
                model_name: "gpt-4".to_string(),
                temperature: Some(0.5),
                max_tokens: Some(2048),
                api_key: None,
            },
            middleware: MiddlewareConfig {
                enable_filesystem: false,
                enable_memory: true,
                enable_summarization: true,
                enable_planning: false,
                enable_logging: false,
                enable_rate_limiter: false,
                filesystem_root: None,
                max_context_tokens: Some(4000),
                custom_middleware: vec!["audit".to_string()],
            },
            backend: BackendConfig::Filesystem {
                path: PathBuf::from("/data"),
            },
            max_iterations: 100,
            tools: vec!["calculator".to_string(), "search".to_string()],
            tags,
            verbose: true,
        };

        let json = serde_json::to_string_pretty(&ac).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(ac, deserialized);
    }

    #[test]
    fn test_agent_config_deserialize_minimal() {
        let json = r#"{"name": "minimal"}"#;
        let ac: AgentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(ac.name, "minimal");
        assert_eq!(ac.max_iterations, 50);
        assert_eq!(ac.model.provider, "anthropic");
    }

    // -- AgentConfigBuilder tests --

    #[test]
    fn test_builder_defaults() {
        let config = AgentConfigBuilder::new().build();
        assert_eq!(config, AgentConfig::default());
    }

    #[test]
    fn test_builder_full() {
        let config = AgentConfigBuilder::new()
            .name("my-agent")
            .description("Does things")
            .system_prompt("Be concise")
            .provider("openai")
            .model_name("gpt-4")
            .temperature(0.3)
            .max_tokens(1024)
            .enable_filesystem(false)
            .enable_summarization(true)
            .max_iterations(10)
            .tool("calculator")
            .tool("search")
            .tag("team", "platform")
            .verbose(true)
            .backend(BackendConfig::Filesystem {
                path: PathBuf::from("/tmp/state"),
            })
            .build();

        assert_eq!(config.name, "my-agent");
        assert_eq!(config.description, Some("Does things".to_string()));
        assert_eq!(config.system_prompt, Some("Be concise".to_string()));
        assert_eq!(config.model.provider, "openai");
        assert_eq!(config.model.model_name, "gpt-4");
        assert_eq!(config.model.temperature, Some(0.3));
        assert_eq!(config.model.max_tokens, Some(1024));
        assert!(!config.middleware.enable_filesystem);
        assert!(config.middleware.enable_summarization);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.tools, vec!["calculator", "search"]);
        assert_eq!(config.tags.get("team"), Some(&"platform".to_string()));
        assert!(config.verbose);
        assert_eq!(
            config.backend,
            BackendConfig::Filesystem {
                path: PathBuf::from("/tmp/state"),
            }
        );
    }

    #[test]
    fn test_builder_tools_override() {
        let config = AgentConfigBuilder::new()
            .tool("a")
            .tool("b")
            .tools(vec!["x".to_string(), "y".to_string()])
            .build();
        assert_eq!(config.tools, vec!["x", "y"]);
    }

    #[test]
    fn test_builder_filesystem_root() {
        let config = AgentConfigBuilder::new()
            .filesystem_root("/home/user/project")
            .build();
        assert_eq!(
            config.middleware.filesystem_root,
            Some(PathBuf::from("/home/user/project"))
        );
    }

    #[test]
    fn test_builder_max_context_tokens() {
        let config = AgentConfigBuilder::new().max_context_tokens(16000).build();
        assert_eq!(config.middleware.max_context_tokens, Some(16000));
    }

    // -- ConfigLoader tests --

    #[test]
    fn test_config_loader_from_json() {
        let json = r#"{
            "name": "json-agent",
            "max_iterations": 30,
            "model": {
                "provider": "openai",
                "model_name": "gpt-4"
            },
            "tools": ["search"]
        }"#;
        let config = ConfigLoader::from_json(json).unwrap();
        assert_eq!(config.name, "json-agent");
        assert_eq!(config.max_iterations, 30);
        assert_eq!(config.model.provider, "openai");
        assert_eq!(config.tools, vec!["search"]);
    }

    #[test]
    fn test_config_loader_from_json_invalid() {
        let result = ConfigLoader::from_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_loader_from_value() {
        let value = serde_json::json!({
            "name": "value-agent",
            "verbose": true
        });
        let config = ConfigLoader::from_value(value).unwrap();
        assert_eq!(config.name, "value-agent");
        assert!(config.verbose);
    }

    #[test]
    fn test_config_loader_merge_basic() {
        let base = AgentConfigBuilder::new()
            .name("base-agent")
            .description("base description")
            .system_prompt("base prompt")
            .max_iterations(20)
            .tool("tool_a")
            .tag("env", "dev")
            .build();

        let overrides = AgentConfigBuilder::new()
            .name("override-agent")
            .system_prompt("override prompt")
            .tool("tool_b")
            .tag("env", "prod")
            .tag("team", "infra")
            .build();

        let merged = ConfigLoader::merge(base, overrides);
        assert_eq!(merged.name, "override-agent");
        assert_eq!(merged.description, Some("base description".to_string()));
        assert_eq!(merged.system_prompt, Some("override prompt".to_string()));
        assert_eq!(merged.max_iterations, 20); // override is default (50), so base wins
        assert_eq!(merged.tools, vec!["tool_b"]); // override non-empty
        assert_eq!(merged.tags.get("env"), Some(&"prod".to_string()));
        assert_eq!(merged.tags.get("team"), Some(&"infra".to_string()));
    }

    #[test]
    fn test_config_loader_merge_model() {
        let base = AgentConfigBuilder::new()
            .provider("anthropic")
            .model_name("claude-sonnet-4-20250514")
            .temperature(0.5)
            .build();

        let overrides = AgentConfigBuilder::new()
            .provider("openai")
            .max_tokens(2048)
            .build();

        let merged = ConfigLoader::merge(base, overrides);
        assert_eq!(merged.model.provider, "openai");
        assert_eq!(merged.model.temperature, Some(0.5)); // from base
        assert_eq!(merged.model.max_tokens, Some(2048)); // from override
    }

    #[test]
    fn test_config_loader_merge_middleware_flags() {
        let base = AgentConfigBuilder::new()
            .enable_summarization(true)
            .enable_logging(true)
            .build();

        let overrides = AgentConfig::default();

        let merged = ConfigLoader::merge(base, overrides);
        // summarization and logging were true in base, should remain true
        assert!(merged.middleware.enable_summarization);
        assert!(merged.middleware.enable_logging);
    }

    #[test]
    fn test_config_loader_merge_backend() {
        let base = AgentConfigBuilder::new()
            .backend(BackendConfig::Filesystem {
                path: PathBuf::from("/base"),
            })
            .build();

        let overrides = AgentConfig::default(); // InMemory (default)

        let merged = ConfigLoader::merge(base, overrides);
        // Override is default, so base wins
        assert_eq!(
            merged.backend,
            BackendConfig::Filesystem {
                path: PathBuf::from("/base"),
            }
        );
    }

    // -- Env loading test --

    #[test]
    fn test_config_loader_from_env() {
        // Set env vars for this test
        std::env::set_var("DEEP_AGENT_NAME", "env-agent");
        std::env::set_var("DEEP_AGENT_PROVIDER", "google");
        std::env::set_var("DEEP_AGENT_MODEL", "gemini-pro");
        std::env::set_var("DEEP_AGENT_TEMPERATURE", "0.9");
        std::env::set_var("DEEP_AGENT_MAX_TOKENS", "2048");
        std::env::set_var("DEEP_AGENT_MAX_ITERATIONS", "75");
        std::env::set_var("DEEP_AGENT_TOOLS", "search, calculator");
        std::env::set_var("DEEP_AGENT_VERBOSE", "true");
        std::env::set_var("DEEP_AGENT_ENABLE_SUMMARIZATION", "true");
        std::env::set_var("DEEP_AGENT_ENABLE_FILESYSTEM", "false");

        let config = ConfigLoader::from_env().unwrap();
        assert_eq!(config.name, "env-agent");
        assert_eq!(config.model.provider, "google");
        assert_eq!(config.model.model_name, "gemini-pro");
        assert_eq!(config.model.temperature, Some(0.9));
        assert_eq!(config.model.max_tokens, Some(2048));
        assert_eq!(config.max_iterations, 75);
        assert_eq!(config.tools, vec!["search", "calculator"]);
        assert!(config.verbose);
        assert!(config.middleware.enable_summarization);
        assert!(!config.middleware.enable_filesystem);

        // Clean up
        std::env::remove_var("DEEP_AGENT_NAME");
        std::env::remove_var("DEEP_AGENT_PROVIDER");
        std::env::remove_var("DEEP_AGENT_MODEL");
        std::env::remove_var("DEEP_AGENT_TEMPERATURE");
        std::env::remove_var("DEEP_AGENT_MAX_TOKENS");
        std::env::remove_var("DEEP_AGENT_MAX_ITERATIONS");
        std::env::remove_var("DEEP_AGENT_TOOLS");
        std::env::remove_var("DEEP_AGENT_VERBOSE");
        std::env::remove_var("DEEP_AGENT_ENABLE_SUMMARIZATION");
        std::env::remove_var("DEEP_AGENT_ENABLE_FILESYSTEM");
    }

    #[test]
    fn test_config_loader_from_env_invalid_temperature() {
        std::env::set_var("DEEP_AGENT_TEMPERATURE", "not_a_number");
        let result = ConfigLoader::from_env();
        assert!(result.is_err());
        std::env::remove_var("DEEP_AGENT_TEMPERATURE");
    }

    // -- DeepAgentConfig backwards compatibility --

    #[test]
    fn test_deep_agent_config_defaults() {
        let config = DeepAgentConfig::default();
        assert_eq!(config.model_name, "claude-sonnet-4-6");
        assert_eq!(config.max_iterations, 25);
        assert!(config.system_prompt.is_none());
        assert!(config.tools.is_empty());
        assert!(config.middleware.is_empty());
    }

    #[test]
    fn test_deep_agent_config_builder_chain() {
        let config = DeepAgentConfig::default()
            .with_model("gpt-4")
            .with_max_iterations(10)
            .with_system_prompt("Be helpful");
        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.system_prompt, Some("Be helpful".to_string()));
    }

    // -- Misc edge cases --

    #[test]
    fn test_backend_config_in_memory_serialization() {
        let bc = BackendConfig::InMemory;
        let json = serde_json::to_string(&bc).unwrap();
        assert!(json.contains("in_memory"));
        let deserialized: BackendConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, BackendConfig::InMemory);
    }

    #[test]
    fn test_agent_config_empty_json_object() {
        // An empty JSON object should fail because `name` is required
        let result: Result<AgentConfig, _> = serde_json::from_str("{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_enable_rate_limiter() {
        let config = AgentConfigBuilder::new().enable_rate_limiter(true).build();
        assert!(config.middleware.enable_rate_limiter);
    }

    #[test]
    fn test_builder_enable_planning() {
        let config = AgentConfigBuilder::new().enable_planning(true).build();
        assert!(config.middleware.enable_planning);
    }

    #[test]
    fn test_builder_enable_logging() {
        let config = AgentConfigBuilder::new().enable_logging(true).build();
        assert!(config.middleware.enable_logging);
    }

    #[test]
    fn test_builder_enable_memory() {
        let config = AgentConfigBuilder::new().enable_memory(false).build();
        assert!(!config.middleware.enable_memory);
    }
}
