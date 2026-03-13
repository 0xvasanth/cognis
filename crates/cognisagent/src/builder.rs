//! Comprehensive agent builder with fluent API, templates, and directory.
//!
//! This module provides a layered approach to assembling agents:
//!
//! - [`AgentCapability`] -- declares what an agent can do
//! - [`AgentProfile`] -- a validated, immutable agent description
//! - [`ToolSpec`] / [`MiddlewareSpec`] -- declarative tool and middleware descriptions
//! - [`AgentBuilder`] -- fluent builder for assembling all components
//! - [`AgentTemplate`] -- a reusable agent configuration snapshot
//! - [`AgentDirectory`] -- a registry of named templates
//!
//! # Example
//!
//! ```rust
//! use cognisagent::builder::{AgentBuilder, AgentCapability, ToolSpec, MiddlewareSpec};
//!
//! let profile = AgentBuilder::new("my-agent")
//!     .model("claude-sonnet-4-20250514")
//!     .system_prompt("You are helpful.")
//!     .temperature(0.3)
//!     .max_iterations(20)
//!     .capability(AgentCapability::Chat)
//!     .capability(AgentCapability::ToolUse)
//!     .tool(ToolSpec::new("search", "Search the web").required())
//!     .middleware(MiddlewareSpec::new("logging").with_priority(10))
//!     .build()
//!     .unwrap();
//! ```

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

use crate::presets::PresetRegistry;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during agent building.
#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    /// A required field is missing.
    #[error("missing required field: {0}")]
    MissingField(String),
    /// Validation failed with one or more errors.
    #[error("validation failed: {0}")]
    ValidationFailed(String),
    /// The requested preset was not found.
    #[error("preset not found: {0}")]
    PresetNotFound(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, BuilderError>;

// ---------------------------------------------------------------------------
// AgentCapability
// ---------------------------------------------------------------------------

/// Declares a high-level capability an agent possesses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentCapability {
    /// Basic conversational ability.
    Chat,
    /// Ability to invoke tools / function calling.
    ToolUse,
    /// Multi-step planning before execution.
    Planning,
    /// Persistent memory across interactions.
    Memory,
    /// Read/write access to the filesystem.
    FileSystem,
    /// Web search capability.
    WebSearch,
    /// Ability to execute code.
    CodeExecution,
    /// A custom, user-defined capability.
    Custom(String),
}

impl fmt::Display for AgentCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chat => write!(f, "Chat"),
            Self::ToolUse => write!(f, "ToolUse"),
            Self::Planning => write!(f, "Planning"),
            Self::Memory => write!(f, "Memory"),
            Self::FileSystem => write!(f, "FileSystem"),
            Self::WebSearch => write!(f, "WebSearch"),
            Self::CodeExecution => write!(f, "CodeExecution"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentProfile
// ---------------------------------------------------------------------------

/// A validated, immutable description of an agent and its configuration.
#[derive(Debug, Clone)]
pub struct AgentProfile {
    /// Agent name.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Declared capabilities.
    pub capabilities: Vec<AgentCapability>,
    /// Model identifier.
    pub model_name: String,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Maximum agent loop iterations.
    pub max_iterations: usize,
    /// Sampling temperature.
    pub temperature: Option<f64>,
}

impl AgentProfile {
    /// Create a new profile with the given name and model.
    pub fn new(name: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            capabilities: Vec::new(),
            model_name: model_name.into(),
            system_prompt: None,
            max_iterations: 50,
            temperature: None,
        }
    }

    /// Check whether this profile declares the given capability.
    pub fn has_capability(&self, cap: &AgentCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Serialize this profile to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
        })
    }
}

// ---------------------------------------------------------------------------
// ToolSpec
// ---------------------------------------------------------------------------

/// Declarative description of a tool to attach to an agent.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this tool is required (vs. optional).
    pub is_required: bool,
    /// Optional JSON schema for the tool's parameters.
    pub parameters_schema: Option<Value>,
}

impl ToolSpec {
    /// Create a new tool spec with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            is_required: false,
            parameters_schema: None,
        }
    }

    /// Mark this tool as required.
    pub fn required(mut self) -> Self {
        self.is_required = true;
        self
    }

    /// Attach a JSON schema for the tool's parameters.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.parameters_schema = Some(schema);
        self
    }
}

// ---------------------------------------------------------------------------
// MiddlewareSpec
// ---------------------------------------------------------------------------

/// Declarative description of a middleware to attach to an agent.
#[derive(Debug, Clone)]
pub struct MiddlewareSpec {
    /// Middleware name.
    pub name: String,
    /// Execution priority (lower runs first).
    pub priority: i32,
    /// Whether this middleware is enabled.
    pub enabled: bool,
    /// Arbitrary configuration for the middleware.
    pub config: Value,
}

impl MiddlewareSpec {
    /// Create a new middleware spec with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            enabled: true,
            config: Value::Object(serde_json::Map::new()),
        }
    }

    /// Set the execution priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set arbitrary configuration.
    pub fn with_config(mut self, config: Value) -> Self {
        self.config = config;
        self
    }

    /// Enable or disable this middleware.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// AgentBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for assembling an agent from its components.
///
/// Call [`build()`](AgentBuilder::build) to validate and produce an [`AgentProfile`].
#[derive(Debug, Clone)]
pub struct AgentBuilder {
    name: String,
    model_name: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f64>,
    max_iterations: Option<usize>,
    description: Option<String>,
    capabilities: Vec<AgentCapability>,
    tools: Vec<ToolSpec>,
    middlewares: Vec<MiddlewareSpec>,
    memory_enabled: bool,
    planning_enabled: bool,
}

impl AgentBuilder {
    /// Create a new builder with the given agent name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            model_name: None,
            system_prompt: None,
            temperature: None,
            max_iterations: None,
            description: None,
            capabilities: Vec::new(),
            tools: Vec::new(),
            middlewares: Vec::new(),
            memory_enabled: false,
            planning_enabled: false,
        }
    }

    /// Set the model name.
    pub fn model(mut self, model_name: &str) -> Self {
        self.model_name = Some(model_name.to_string());
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the maximum number of agent loop iterations.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set the agent description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a capability.
    pub fn capability(mut self, cap: AgentCapability) -> Self {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
        }
        self
    }

    /// Add a tool specification.
    pub fn tool(mut self, spec: ToolSpec) -> Self {
        self.tools.push(spec);
        self
    }

    /// Add a middleware specification.
    pub fn middleware(mut self, spec: MiddlewareSpec) -> Self {
        self.middlewares.push(spec);
        self
    }

    /// Enable or disable memory.
    pub fn memory_enabled(mut self, enabled: bool) -> Self {
        self.memory_enabled = enabled;
        if enabled && !self.capabilities.contains(&AgentCapability::Memory) {
            self.capabilities.push(AgentCapability::Memory);
        }
        self
    }

    /// Enable or disable planning.
    pub fn planning_enabled(mut self, enabled: bool) -> Self {
        self.planning_enabled = enabled;
        if enabled && !self.capabilities.contains(&AgentCapability::Planning) {
            self.capabilities.push(AgentCapability::Planning);
        }
        self
    }

    /// Initialize a builder from a named preset in the default registry.
    ///
    /// Returns a builder pre-populated with the preset's configuration.
    pub fn from_preset(preset_name: &str) -> Self {
        let registry = PresetRegistry::default();
        let preset = match registry.get(preset_name) {
            Some(p) => p,
            None => {
                // Return an empty builder with the preset name so that build()
                // will fail with a clear error if model is missing.
                return Self::new(preset_name);
            }
        };

        let mut builder = Self::new(&preset.name);
        builder.model_name = Some(preset.config.model.model_name.clone());
        builder.system_prompt = preset.config.system_prompt.clone();
        builder.temperature = preset.config.model.temperature;
        builder.max_iterations = Some(preset.config.max_iterations);
        builder.description = preset.config.description.clone();
        builder.memory_enabled = preset.config.middleware.enable_memory;
        builder.planning_enabled = preset.config.middleware.enable_planning;

        if preset.config.middleware.enable_memory {
            builder.capabilities.push(AgentCapability::Memory);
        }
        if preset.config.middleware.enable_planning {
            builder.capabilities.push(AgentCapability::Planning);
        }
        if preset.config.middleware.enable_filesystem {
            builder.capabilities.push(AgentCapability::FileSystem);
        }

        // Add suggested tools as tool specs.
        for tool_name in &preset.suggested_tools {
            builder.tools.push(ToolSpec::new(
                tool_name,
                format!("Tool from preset: {tool_name}"),
            ));
        }

        builder
    }

    /// Validate the builder state and return warnings.
    ///
    /// Returns `Ok(warnings)` where warnings is a list of non-fatal issues,
    /// or `Err` if validation fails fatally.
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Fatal: model is required.
        if self.model_name.is_none() {
            return Err(BuilderError::MissingField("model_name".to_string()));
        }

        // Fatal: name must not be empty.
        if self.name.is_empty() {
            return Err(BuilderError::MissingField("name".to_string()));
        }

        // Warning: ToolUse capability but no tools.
        if self.capabilities.contains(&AgentCapability::ToolUse) && self.tools.is_empty() {
            warnings.push("ToolUse capability declared but no tools configured".to_string());
        }

        // Warning: tools present but no ToolUse capability.
        if !self.tools.is_empty() && !self.capabilities.contains(&AgentCapability::ToolUse) {
            warnings.push("Tools configured but ToolUse capability not declared".to_string());
        }

        // Warning: Memory capability but memory not enabled.
        if self.capabilities.contains(&AgentCapability::Memory) && !self.memory_enabled {
            warnings.push("Memory capability declared but memory_enabled is false".to_string());
        }

        // Warning: Planning capability but planning not enabled.
        if self.capabilities.contains(&AgentCapability::Planning) && !self.planning_enabled {
            warnings.push("Planning capability declared but planning_enabled is false".to_string());
        }

        // Warning: no capabilities at all.
        if self.capabilities.is_empty() {
            warnings.push("No capabilities declared".to_string());
        }

        // Warning: no system prompt.
        if self.system_prompt.is_none() {
            warnings.push("No system prompt set".to_string());
        }

        // Warning: duplicate tool names.
        let mut seen_tools = std::collections::HashSet::new();
        for tool in &self.tools {
            if !seen_tools.insert(&tool.name) {
                warnings.push(format!("Duplicate tool name: {}", tool.name));
            }
        }

        Ok(warnings)
    }

    /// Validate and build the [`AgentProfile`].
    pub fn build(&self) -> Result<AgentProfile> {
        // Run validation (ignoring warnings).
        let _warnings = self.validate()?;

        let model_name = self
            .model_name
            .clone()
            .ok_or_else(|| BuilderError::MissingField("model_name".to_string()))?;

        Ok(AgentProfile {
            name: self.name.clone(),
            description: self.description.clone(),
            capabilities: self.capabilities.clone(),
            model_name,
            system_prompt: self.system_prompt.clone(),
            max_iterations: self.max_iterations.unwrap_or(50),
            temperature: self.temperature,
        })
    }

    /// Export the builder state as a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "description": self.description,
            "capabilities": self.capabilities.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
            "tools": self.tools.iter().map(|t| serde_json::json!({
                "name": t.name,
                "description": t.description,
                "required": t.is_required,
                "parameters_schema": t.parameters_schema,
            })).collect::<Vec<_>>(),
            "middlewares": self.middlewares.iter().map(|m| serde_json::json!({
                "name": m.name,
                "priority": m.priority,
                "enabled": m.enabled,
                "config": m.config,
            })).collect::<Vec<_>>(),
            "memory_enabled": self.memory_enabled,
            "planning_enabled": self.planning_enabled,
        })
    }
}

// ---------------------------------------------------------------------------
// AgentTemplate
// ---------------------------------------------------------------------------

/// A reusable, frozen agent configuration that can spawn new builders.
#[derive(Debug, Clone)]
pub struct AgentTemplate {
    /// Template name (typically the builder's agent name).
    name: String,
    model_name: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f64>,
    max_iterations: Option<usize>,
    description: Option<String>,
    capabilities: Vec<AgentCapability>,
    tools: Vec<ToolSpec>,
    middlewares: Vec<MiddlewareSpec>,
    memory_enabled: bool,
    planning_enabled: bool,
}

impl AgentTemplate {
    /// Create a template from the current state of a builder.
    pub fn from_builder(builder: &AgentBuilder) -> Self {
        Self {
            name: builder.name.clone(),
            model_name: builder.model_name.clone(),
            system_prompt: builder.system_prompt.clone(),
            temperature: builder.temperature,
            max_iterations: builder.max_iterations,
            description: builder.description.clone(),
            capabilities: builder.capabilities.clone(),
            tools: builder.tools.clone(),
            middlewares: builder.middlewares.clone(),
            memory_enabled: builder.memory_enabled,
            planning_enabled: builder.planning_enabled,
        }
    }

    /// Create a new builder from this template, overriding the agent name.
    pub fn instantiate(&self, name: &str) -> AgentBuilder {
        AgentBuilder {
            name: name.to_string(),
            model_name: self.model_name.clone(),
            system_prompt: self.system_prompt.clone(),
            temperature: self.temperature,
            max_iterations: self.max_iterations,
            description: self.description.clone(),
            capabilities: self.capabilities.clone(),
            tools: self.tools.clone(),
            middlewares: self.middlewares.clone(),
            memory_enabled: self.memory_enabled,
            planning_enabled: self.planning_enabled,
        }
    }

    /// Serialize this template to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "template_name": self.name,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "description": self.description,
            "capabilities": self.capabilities.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
            "tools": self.tools.iter().map(|t| serde_json::json!({
                "name": t.name,
                "description": t.description,
                "required": t.is_required,
                "parameters_schema": t.parameters_schema,
            })).collect::<Vec<_>>(),
            "middlewares": self.middlewares.iter().map(|m| serde_json::json!({
                "name": m.name,
                "priority": m.priority,
                "enabled": m.enabled,
                "config": m.config,
            })).collect::<Vec<_>>(),
            "memory_enabled": self.memory_enabled,
            "planning_enabled": self.planning_enabled,
        })
    }
}

// ---------------------------------------------------------------------------
// AgentDirectory
// ---------------------------------------------------------------------------

/// A registry of named [`AgentTemplate`]s for centralized agent management.
#[derive(Debug, Clone)]
pub struct AgentDirectory {
    templates: HashMap<String, AgentTemplate>,
}

impl AgentDirectory {
    /// Create an empty directory.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register a template under the given name.
    pub fn register(&mut self, name: &str, template: AgentTemplate) {
        self.templates.insert(name.to_string(), template);
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Option<&AgentTemplate> {
        self.templates.get(name)
    }

    /// Create a new builder from a registered template, giving it a new agent name.
    pub fn instantiate(&self, template_name: &str, agent_name: &str) -> Option<AgentBuilder> {
        self.templates
            .get(template_name)
            .map(|t| t.instantiate(agent_name))
    }

    /// List all registered template names.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Return whether the directory is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Remove a template by name. Returns `true` if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        self.templates.remove(name).is_some()
    }
}

impl Default for AgentDirectory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AgentCapability display --

    #[test]
    fn test_capability_display_chat() {
        assert_eq!(AgentCapability::Chat.to_string(), "Chat");
    }

    #[test]
    fn test_capability_display_tool_use() {
        assert_eq!(AgentCapability::ToolUse.to_string(), "ToolUse");
    }

    #[test]
    fn test_capability_display_planning() {
        assert_eq!(AgentCapability::Planning.to_string(), "Planning");
    }

    #[test]
    fn test_capability_display_memory() {
        assert_eq!(AgentCapability::Memory.to_string(), "Memory");
    }

    #[test]
    fn test_capability_display_filesystem() {
        assert_eq!(AgentCapability::FileSystem.to_string(), "FileSystem");
    }

    #[test]
    fn test_capability_display_web_search() {
        assert_eq!(AgentCapability::WebSearch.to_string(), "WebSearch");
    }

    #[test]
    fn test_capability_display_code_execution() {
        assert_eq!(AgentCapability::CodeExecution.to_string(), "CodeExecution");
    }

    #[test]
    fn test_capability_display_custom() {
        let cap = AgentCapability::Custom("MySkill".to_string());
        assert_eq!(cap.to_string(), "Custom(MySkill)");
    }

    // -- AgentProfile --

    #[test]
    fn test_profile_new() {
        let profile = AgentProfile::new("test-agent", "gpt-4");
        assert_eq!(profile.name, "test-agent");
        assert_eq!(profile.model_name, "gpt-4");
        assert!(profile.description.is_none());
        assert!(profile.capabilities.is_empty());
        assert!(profile.system_prompt.is_none());
        assert_eq!(profile.max_iterations, 50);
        assert!(profile.temperature.is_none());
    }

    #[test]
    fn test_profile_has_capability() {
        let mut profile = AgentProfile::new("agent", "model");
        profile.capabilities.push(AgentCapability::Chat);
        profile.capabilities.push(AgentCapability::ToolUse);
        assert!(profile.has_capability(&AgentCapability::Chat));
        assert!(profile.has_capability(&AgentCapability::ToolUse));
        assert!(!profile.has_capability(&AgentCapability::Memory));
    }

    #[test]
    fn test_profile_to_json() {
        let mut profile = AgentProfile::new("agent", "gpt-4");
        profile.capabilities.push(AgentCapability::Chat);
        profile.temperature = Some(0.5);
        let json = profile.to_json();
        assert_eq!(json["name"], "agent");
        assert_eq!(json["model_name"], "gpt-4");
        assert_eq!(json["temperature"], 0.5);
        let caps = json["capabilities"].as_array().unwrap();
        assert_eq!(caps.len(), 1);
        assert_eq!(caps[0], "Chat");
    }

    // -- ToolSpec --

    #[test]
    fn test_tool_spec_new() {
        let tool = ToolSpec::new("search", "Search the web");
        assert_eq!(tool.name, "search");
        assert_eq!(tool.description, "Search the web");
        assert!(!tool.is_required);
        assert!(tool.parameters_schema.is_none());
    }

    #[test]
    fn test_tool_spec_required() {
        let tool = ToolSpec::new("search", "Search").required();
        assert!(tool.is_required);
    }

    #[test]
    fn test_tool_spec_with_schema() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let tool = ToolSpec::new("search", "Search").with_schema(schema.clone());
        assert_eq!(tool.parameters_schema, Some(schema));
    }

    // -- MiddlewareSpec --

    #[test]
    fn test_middleware_spec_new() {
        let mw = MiddlewareSpec::new("logging");
        assert_eq!(mw.name, "logging");
        assert_eq!(mw.priority, 0);
        assert!(mw.enabled);
        assert!(mw.config.is_object());
    }

    #[test]
    fn test_middleware_spec_with_priority() {
        let mw = MiddlewareSpec::new("logging").with_priority(10);
        assert_eq!(mw.priority, 10);
    }

    #[test]
    fn test_middleware_spec_with_config() {
        let config = serde_json::json!({"level": "debug"});
        let mw = MiddlewareSpec::new("logging").with_config(config.clone());
        assert_eq!(mw.config, config);
    }

    #[test]
    fn test_middleware_spec_enabled() {
        let mw = MiddlewareSpec::new("logging").enabled(false);
        assert!(!mw.enabled);
    }

    // -- AgentBuilder full fluent chain --

    #[test]
    fn test_builder_full_chain() {
        let profile = AgentBuilder::new("full-agent")
            .model("claude-sonnet-4-20250514")
            .system_prompt("You are helpful.")
            .temperature(0.3)
            .max_iterations(20)
            .description("A fully configured agent")
            .capability(AgentCapability::Chat)
            .capability(AgentCapability::ToolUse)
            .tool(ToolSpec::new("search", "Search the web"))
            .middleware(MiddlewareSpec::new("logging").with_priority(5))
            .memory_enabled(true)
            .planning_enabled(true)
            .build()
            .unwrap();

        assert_eq!(profile.name, "full-agent");
        assert_eq!(profile.model_name, "claude-sonnet-4-20250514");
        assert_eq!(profile.system_prompt, Some("You are helpful.".to_string()));
        assert_eq!(profile.temperature, Some(0.3));
        assert_eq!(profile.max_iterations, 20);
        assert_eq!(
            profile.description,
            Some("A fully configured agent".to_string())
        );
        assert!(profile.has_capability(&AgentCapability::Chat));
        assert!(profile.has_capability(&AgentCapability::ToolUse));
        assert!(profile.has_capability(&AgentCapability::Memory));
        assert!(profile.has_capability(&AgentCapability::Planning));
    }

    // -- Build validation warnings --

    #[test]
    fn test_builder_warns_tool_use_no_tools() {
        let warnings = AgentBuilder::new("agent")
            .model("gpt-4")
            .capability(AgentCapability::ToolUse)
            .validate()
            .unwrap();
        assert!(warnings
            .iter()
            .any(|w| w.contains("ToolUse") && w.contains("no tools")));
    }

    #[test]
    fn test_builder_warns_tools_no_capability() {
        let warnings = AgentBuilder::new("agent")
            .model("gpt-4")
            .capability(AgentCapability::Chat)
            .tool(ToolSpec::new("search", "Search"))
            .validate()
            .unwrap();
        assert!(warnings
            .iter()
            .any(|w| w.contains("ToolUse capability not declared")));
    }

    #[test]
    fn test_builder_warns_no_capabilities() {
        let warnings = AgentBuilder::new("agent")
            .model("gpt-4")
            .validate()
            .unwrap();
        assert!(warnings.iter().any(|w| w.contains("No capabilities")));
    }

    #[test]
    fn test_builder_warns_no_system_prompt() {
        let warnings = AgentBuilder::new("agent")
            .model("gpt-4")
            .validate()
            .unwrap();
        assert!(warnings.iter().any(|w| w.contains("No system prompt")));
    }

    #[test]
    fn test_builder_warns_duplicate_tools() {
        let warnings = AgentBuilder::new("agent")
            .model("gpt-4")
            .tool(ToolSpec::new("search", "Search 1"))
            .tool(ToolSpec::new("search", "Search 2"))
            .validate()
            .unwrap();
        assert!(warnings
            .iter()
            .any(|w| w.contains("Duplicate tool name: search")));
    }

    // -- Build failure on missing required fields --

    #[test]
    fn test_builder_fails_missing_model() {
        let result = AgentBuilder::new("agent").build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("model_name"));
    }

    #[test]
    fn test_builder_fails_empty_name() {
        let result = AgentBuilder::new("").model("gpt-4").build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("name"));
    }

    // -- from_preset --

    #[test]
    fn test_from_preset_coding_assistant() {
        let profile = AgentBuilder::from_preset("coding-assistant")
            .build()
            .unwrap();
        assert_eq!(profile.name, "coding-assistant");
        assert!(profile.has_capability(&AgentCapability::FileSystem));
        assert!(profile.has_capability(&AgentCapability::Memory));
        assert_eq!(profile.temperature, Some(0.2));
    }

    #[test]
    fn test_from_preset_task_planner() {
        let profile = AgentBuilder::from_preset("task-planner").build().unwrap();
        assert!(profile.has_capability(&AgentCapability::Planning));
        assert!(profile.has_capability(&AgentCapability::Memory));
    }

    #[test]
    fn test_from_preset_unknown_returns_empty_builder() {
        // Unknown preset gives a builder without a model, so build() fails.
        let result = AgentBuilder::from_preset("nonexistent").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_from_preset_can_override() {
        let profile = AgentBuilder::from_preset("coding-assistant")
            .temperature(0.9)
            .system_prompt("Override prompt")
            .build()
            .unwrap();
        assert_eq!(profile.temperature, Some(0.9));
        assert_eq!(profile.system_prompt, Some("Override prompt".to_string()));
    }

    // -- AgentTemplate --

    #[test]
    fn test_template_from_builder() {
        let builder = AgentBuilder::new("tmpl-agent")
            .model("gpt-4")
            .temperature(0.5)
            .capability(AgentCapability::Chat);
        let template = AgentTemplate::from_builder(&builder);
        assert_eq!(template.name, "tmpl-agent");
        assert_eq!(template.model_name, Some("gpt-4".to_string()));
        assert_eq!(template.temperature, Some(0.5));
    }

    #[test]
    fn test_template_instantiate() {
        let builder = AgentBuilder::new("tmpl")
            .model("gpt-4")
            .temperature(0.5)
            .capability(AgentCapability::Chat)
            .tool(ToolSpec::new("search", "Search"));
        let template = AgentTemplate::from_builder(&builder);

        let new_builder = template.instantiate("new-agent");
        let profile = new_builder.build().unwrap();
        assert_eq!(profile.name, "new-agent");
        assert_eq!(profile.model_name, "gpt-4");
        assert_eq!(profile.temperature, Some(0.5));
        assert!(profile.has_capability(&AgentCapability::Chat));
    }

    #[test]
    fn test_template_to_json() {
        let builder = AgentBuilder::new("tmpl")
            .model("gpt-4")
            .capability(AgentCapability::Chat);
        let template = AgentTemplate::from_builder(&builder);
        let json = template.to_json();
        assert_eq!(json["template_name"], "tmpl");
        assert_eq!(json["model_name"], "gpt-4");
    }

    // -- AgentDirectory CRUD --

    #[test]
    fn test_directory_new_is_empty() {
        let dir = AgentDirectory::new();
        assert!(dir.is_empty());
        assert_eq!(dir.len(), 0);
    }

    #[test]
    fn test_directory_register_and_get() {
        let mut dir = AgentDirectory::new();
        let builder = AgentBuilder::new("agent").model("gpt-4");
        let template = AgentTemplate::from_builder(&builder);
        dir.register("my-template", template);
        assert_eq!(dir.len(), 1);
        assert!(dir.get("my-template").is_some());
        assert!(dir.get("other").is_none());
    }

    #[test]
    fn test_directory_instantiate() {
        let mut dir = AgentDirectory::new();
        let builder = AgentBuilder::new("tmpl")
            .model("gpt-4")
            .capability(AgentCapability::Chat);
        dir.register("tmpl", AgentTemplate::from_builder(&builder));

        let new_builder = dir.instantiate("tmpl", "instance-1").unwrap();
        let profile = new_builder.build().unwrap();
        assert_eq!(profile.name, "instance-1");
        assert_eq!(profile.model_name, "gpt-4");
    }

    #[test]
    fn test_directory_instantiate_missing() {
        let dir = AgentDirectory::new();
        assert!(dir.instantiate("missing", "agent").is_none());
    }

    #[test]
    fn test_directory_list() {
        let mut dir = AgentDirectory::new();
        let b1 = AgentBuilder::new("a").model("m");
        let b2 = AgentBuilder::new("b").model("m");
        dir.register("alpha", AgentTemplate::from_builder(&b1));
        dir.register("beta", AgentTemplate::from_builder(&b2));

        let mut names = dir.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_directory_remove() {
        let mut dir = AgentDirectory::new();
        let builder = AgentBuilder::new("a").model("m");
        dir.register("tmpl", AgentTemplate::from_builder(&builder));
        assert_eq!(dir.len(), 1);
        assert!(dir.remove("tmpl"));
        assert_eq!(dir.len(), 0);
        assert!(!dir.remove("tmpl"));
    }

    // -- JSON export/round-trip --

    #[test]
    fn test_builder_to_json() {
        let builder = AgentBuilder::new("json-agent")
            .model("gpt-4")
            .temperature(0.7)
            .capability(AgentCapability::Chat)
            .tool(ToolSpec::new("search", "Search").required());
        let json = builder.to_json();
        assert_eq!(json["name"], "json-agent");
        assert_eq!(json["model_name"], "gpt-4");
        assert_eq!(json["temperature"], 0.7);
        let caps = json["capabilities"].as_array().unwrap();
        assert!(caps.contains(&Value::String("Chat".to_string())));
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "search");
        assert_eq!(tools[0]["required"], true);
    }

    #[test]
    fn test_profile_to_json_roundtrip_fields() {
        let profile = AgentBuilder::new("rt-agent")
            .model("claude-sonnet-4-20250514")
            .system_prompt("Be helpful")
            .temperature(0.4)
            .max_iterations(30)
            .description("Test agent")
            .capability(AgentCapability::Chat)
            .capability(AgentCapability::ToolUse)
            .tool(ToolSpec::new("calc", "Calculator"))
            .build()
            .unwrap();

        let json = profile.to_json();
        assert_eq!(json["name"], "rt-agent");
        assert_eq!(json["model_name"], "claude-sonnet-4-20250514");
        assert_eq!(json["system_prompt"], "Be helpful");
        assert_eq!(json["temperature"], 0.4);
        assert_eq!(json["max_iterations"], 30);
        assert_eq!(json["description"], "Test agent");
        let caps = json["capabilities"].as_array().unwrap();
        assert_eq!(caps.len(), 2);
    }

    // -- Edge cases --

    #[test]
    fn test_builder_default_max_iterations() {
        let profile = AgentBuilder::new("agent").model("gpt-4").build().unwrap();
        assert_eq!(profile.max_iterations, 50);
    }

    #[test]
    fn test_builder_duplicate_capability_ignored() {
        let builder = AgentBuilder::new("agent")
            .model("gpt-4")
            .capability(AgentCapability::Chat)
            .capability(AgentCapability::Chat);
        // Should only have one Chat capability.
        assert_eq!(
            builder
                .capabilities
                .iter()
                .filter(|c| **c == AgentCapability::Chat)
                .count(),
            1
        );
    }

    #[test]
    fn test_memory_enabled_adds_capability() {
        let builder = AgentBuilder::new("agent")
            .model("gpt-4")
            .memory_enabled(true);
        assert!(builder.capabilities.contains(&AgentCapability::Memory));
    }

    #[test]
    fn test_planning_enabled_adds_capability() {
        let builder = AgentBuilder::new("agent")
            .model("gpt-4")
            .planning_enabled(true);
        assert!(builder.capabilities.contains(&AgentCapability::Planning));
    }

    #[test]
    fn test_directory_default_is_empty() {
        let dir = AgentDirectory::default();
        assert!(dir.is_empty());
    }

    #[test]
    fn test_capability_equality() {
        assert_eq!(AgentCapability::Chat, AgentCapability::Chat);
        assert_ne!(AgentCapability::Chat, AgentCapability::ToolUse);
        assert_eq!(
            AgentCapability::Custom("x".to_string()),
            AgentCapability::Custom("x".to_string())
        );
        assert_ne!(
            AgentCapability::Custom("x".to_string()),
            AgentCapability::Custom("y".to_string())
        );
    }
}
