//! Agent factory for creating, managing, and tracking agent instances.
//!
//! This module mirrors the Python `create_deep_agent()` factory pattern,
//! providing a declarative way to specify, create, and manage agents.
//!
//! # Key Types
//!
//! - [`AgentSpec`] -- declarative specification for creating an agent
//! - [`AgentInstance`] -- a created agent with identity and status
//! - [`AgentStatus`] -- lifecycle status of an agent
//! - [`AgentFactory`] -- creates agent instances from specs or templates
//! - [`AgentTemplate`] -- predefined, reusable agent configurations
//! - [`AgentPool`] -- manages multiple concurrent agent instances
//! - [`AgentEvent`] -- lifecycle events emitted by agents
//! - [`AgentEventLog`] -- records and queries agent events
//!
//! # Example
//!
//! ```rust
//! use deepagents::factory::{AgentSpec, AgentFactory, AgentPool};
//!
//! let spec = AgentSpec::builder("my-agent")
//!     .model("claude-sonnet-4-6")
//!     .system_prompt("You are helpful.")
//!     .tool("search")
//!     .build();
//!
//! let factory = AgentFactory::new();
//! let instance = factory.create(spec).unwrap();
//!
//! let mut pool = AgentPool::new();
//! pool.add(instance);
//! assert_eq!(pool.len(), 1);
//! ```

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during agent factory operations.
#[derive(Debug, thiserror::Error)]
pub enum FactoryError {
    /// A required field is missing or invalid.
    #[error("invalid spec: {0}")]
    InvalidSpec(String),
    /// Agent creation failed.
    #[error("creation failed: {0}")]
    CreationFailed(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, FactoryError>;

// ---------------------------------------------------------------------------
// AgentSpec
// ---------------------------------------------------------------------------

/// Declarative specification for creating an agent instance.
///
/// Use the builder pattern via [`AgentSpec::builder`] for ergonomic construction.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    /// Agent name.
    pub name: String,
    /// Model identifier.
    pub model: String,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Tool names to attach.
    pub tools: Vec<String>,
    /// Maximum agent loop iterations.
    pub max_iterations: usize,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Middleware names to enable.
    pub middleware: Vec<String>,
    /// Backend type name.
    pub backend: String,
}

impl AgentSpec {
    /// Create a new spec builder with the given agent name.
    pub fn builder(name: impl Into<String>) -> AgentSpecBuilder {
        AgentSpecBuilder {
            name: name.into(),
            model: "claude-sonnet-4-6".to_string(),
            system_prompt: None,
            tools: Vec::new(),
            max_iterations: 25,
            temperature: None,
            middleware: Vec::new(),
            backend: "memory".to_string(),
        }
    }

    /// Serialize this spec to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "middleware": self.middleware,
            "backend": self.backend,
        })
    }

    /// Validate this spec, returning `Ok(())` if valid.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(FactoryError::InvalidSpec(
                "name must not be empty".to_string(),
            ));
        }
        if self.model.is_empty() {
            return Err(FactoryError::InvalidSpec(
                "model must not be empty".to_string(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(FactoryError::InvalidSpec(
                "max_iterations must be greater than 0".to_string(),
            ));
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(FactoryError::InvalidSpec(
                    "temperature must be between 0.0 and 2.0".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Fluent builder for [`AgentSpec`].
#[derive(Debug, Clone)]
pub struct AgentSpecBuilder {
    name: String,
    model: String,
    system_prompt: Option<String>,
    tools: Vec<String>,
    max_iterations: usize,
    temperature: Option<f64>,
    middleware: Vec<String>,
    backend: String,
}

impl AgentSpecBuilder {
    /// Set the model identifier.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a tool by name.
    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Set all tools at once.
    pub fn tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Add a middleware by name.
    pub fn middleware(mut self, mw: impl Into<String>) -> Self {
        self.middleware.push(mw.into());
        self
    }

    /// Set all middleware at once.
    pub fn middlewares(mut self, mws: Vec<String>) -> Self {
        self.middleware = mws;
        self
    }

    /// Set the backend type.
    pub fn backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = backend.into();
        self
    }

    /// Build the [`AgentSpec`].
    pub fn build(self) -> AgentSpec {
        AgentSpec {
            name: self.name,
            model: self.model,
            system_prompt: self.system_prompt,
            tools: self.tools,
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            middleware: self.middleware,
            backend: self.backend,
        }
    }
}

// ---------------------------------------------------------------------------
// AgentStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of an agent instance.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    /// Agent has been created but not yet started.
    Created,
    /// Agent is actively running.
    Running,
    /// Agent execution is paused.
    Paused,
    /// Agent has completed successfully.
    Completed,
    /// Agent has failed with an error message.
    Failed(String),
}

impl AgentStatus {
    /// Return a string representation of the status variant.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Created => "created",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed(_) => "failed",
        }
    }

    /// Return `true` if this status is terminal (completed or failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_))
    }
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Created => write!(f, "Created"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(err) => write!(f, "Failed({err})"),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentInstance
// ---------------------------------------------------------------------------

/// A created agent instance with identity, specification, and status.
#[derive(Debug, Clone)]
pub struct AgentInstance {
    /// Unique identifier (UUID).
    pub id: String,
    /// The specification used to create this agent.
    pub spec: AgentSpec,
    /// Current lifecycle status.
    pub status: AgentStatus,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl AgentInstance {
    /// Serialize this instance to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "spec": self.spec.to_json(),
            "status": self.status.to_string(),
            "created_at": self.created_at,
            "metadata": self.metadata,
        })
    }

    /// Return `true` if the agent is currently running.
    pub fn is_running(&self) -> bool {
        self.status == AgentStatus::Running
    }
}

// ---------------------------------------------------------------------------
// AgentFactory
// ---------------------------------------------------------------------------

/// Factory for creating agent instances from specifications or templates.
pub struct AgentFactory;

impl AgentFactory {
    /// Create a new factory.
    pub fn new() -> Self {
        Self
    }

    /// Create an agent instance from the given spec.
    pub fn create(&self, spec: AgentSpec) -> Result<AgentInstance> {
        spec.validate()?;
        Ok(AgentInstance {
            id: Uuid::new_v4().to_string(),
            spec,
            status: AgentStatus::Created,
            created_at: "2026-03-08T00:00:00Z".to_string(),
            metadata: HashMap::new(),
        })
    }

    /// Create an agent instance with sensible defaults for the given name.
    pub fn create_with_defaults(&self, name: &str) -> Result<AgentInstance> {
        let spec = AgentSpec::builder(name)
            .model("claude-sonnet-4-6")
            .system_prompt("You are a helpful AI assistant.")
            .max_iterations(25)
            .backend("memory")
            .build();
        self.create(spec)
    }

    /// Create an agent instance from a predefined template.
    pub fn from_template(&self, template: &AgentTemplate) -> Result<AgentInstance> {
        let spec = template.to_spec();
        self.create(spec)
    }
}

impl Default for AgentFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AgentTemplate
// ---------------------------------------------------------------------------

/// A predefined, reusable agent configuration.
#[derive(Debug, Clone)]
pub struct AgentTemplate {
    /// Template name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The agent spec associated with this template.
    spec: Option<AgentSpec>,
    /// Categorization tags.
    pub tags: Vec<String>,
}

impl AgentTemplate {
    /// Create a new template with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            spec: None,
            tags: Vec::new(),
        }
    }

    /// Attach a spec to this template.
    pub fn with_spec(mut self, spec: AgentSpec) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Set categorization tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Convert this template into an [`AgentSpec`].
    ///
    /// If no spec was explicitly set, returns a default spec using the
    /// template name and default model.
    pub fn to_spec(&self) -> AgentSpec {
        match &self.spec {
            Some(spec) => spec.clone(),
            None => AgentSpec::builder(&self.name).build(),
        }
    }

    /// Return a pre-configured coding agent template.
    pub fn coding_agent() -> Self {
        let spec = AgentSpec::builder("coding-agent")
            .model("claude-sonnet-4-6")
            .system_prompt("You are an expert software engineer. Write clean, well-tested code.")
            .tool("read_file")
            .tool("write_file")
            .tool("run_command")
            .tool("search_code")
            .middleware("filesystem")
            .middleware("memory")
            .max_iterations(50)
            .temperature(0.2)
            .build();

        Self::new(
            "coding-agent",
            "An agent specialized in writing and reviewing code",
        )
        .with_spec(spec)
        .with_tags(vec!["coding".to_string(), "development".to_string()])
    }

    /// Return a pre-configured research agent template.
    pub fn research_agent() -> Self {
        let spec = AgentSpec::builder("research-agent")
            .model("claude-sonnet-4-6")
            .system_prompt(
                "You are a thorough research assistant. Gather information, \
                 analyze it, and provide well-sourced summaries.",
            )
            .tool("web_search")
            .tool("read_url")
            .tool("summarize")
            .middleware("memory")
            .middleware("summarization")
            .max_iterations(30)
            .temperature(0.3)
            .build();

        Self::new(
            "research-agent",
            "An agent specialized in research and analysis",
        )
        .with_spec(spec)
        .with_tags(vec!["research".to_string(), "analysis".to_string()])
    }

    /// Return a pre-configured chat agent template.
    pub fn chat_agent() -> Self {
        let spec = AgentSpec::builder("chat-agent")
            .model("claude-sonnet-4-6")
            .system_prompt("You are a friendly, conversational AI assistant.")
            .middleware("memory")
            .max_iterations(10)
            .temperature(0.7)
            .build();

        Self::new("chat-agent", "A general-purpose conversational agent")
            .with_spec(spec)
            .with_tags(vec!["chat".to_string(), "general".to_string()])
    }
}

// ---------------------------------------------------------------------------
// AgentPool
// ---------------------------------------------------------------------------

/// Manages a collection of agent instances.
#[derive(Debug, Default)]
pub struct AgentPool {
    instances: HashMap<String, AgentInstance>,
}

impl AgentPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }

    /// Add an agent instance to the pool.
    pub fn add(&mut self, instance: AgentInstance) {
        self.instances.insert(instance.id.clone(), instance);
    }

    /// Get a reference to an agent instance by ID.
    pub fn get(&self, id: &str) -> Option<&AgentInstance> {
        self.instances.get(id)
    }

    /// Get a mutable reference to an agent instance by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut AgentInstance> {
        self.instances.get_mut(id)
    }

    /// Remove an agent instance by ID, returning it if present.
    pub fn remove(&mut self, id: &str) -> Option<AgentInstance> {
        self.instances.remove(id)
    }

    /// Return all instances matching the given status.
    pub fn by_status(&self, status: &AgentStatus) -> Vec<&AgentInstance> {
        self.instances
            .values()
            .filter(|inst| &inst.status == status)
            .collect()
    }

    /// Return the number of active (non-terminal) agents.
    pub fn active_count(&self) -> usize {
        self.instances
            .values()
            .filter(|inst| !inst.status.is_terminal())
            .count()
    }

    /// Return the total number of agents in the pool.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Return `true` if the pool contains no agents.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Serialize the pool to a JSON value.
    pub fn to_json(&self) -> Value {
        let instances: Vec<Value> = self.instances.values().map(|i| i.to_json()).collect();
        serde_json::json!({
            "count": self.instances.len(),
            "active_count": self.active_count(),
            "instances": instances,
        })
    }
}

// ---------------------------------------------------------------------------
// AgentEvent
// ---------------------------------------------------------------------------

/// Lifecycle events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent was created.
    Created { id: String },
    /// Agent execution started.
    Started { id: String },
    /// Agent execution was paused.
    Paused { id: String },
    /// Agent completed successfully.
    Completed { id: String, result: Value },
    /// Agent failed with an error.
    Failed { id: String, error: String },
}

impl AgentEvent {
    /// Return the agent ID associated with this event.
    pub fn agent_id(&self) -> &str {
        match self {
            Self::Created { id }
            | Self::Started { id }
            | Self::Paused { id }
            | Self::Completed { id, .. }
            | Self::Failed { id, .. } => id,
        }
    }

    /// Return the event type as a string.
    pub fn event_type(&self) -> &str {
        match self {
            Self::Created { .. } => "created",
            Self::Started { .. } => "started",
            Self::Paused { .. } => "paused",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
        }
    }

    /// Serialize this event to a JSON value.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Created { id } => serde_json::json!({
                "type": "created",
                "agent_id": id,
            }),
            Self::Started { id } => serde_json::json!({
                "type": "started",
                "agent_id": id,
            }),
            Self::Paused { id } => serde_json::json!({
                "type": "paused",
                "agent_id": id,
            }),
            Self::Completed { id, result } => serde_json::json!({
                "type": "completed",
                "agent_id": id,
                "result": result,
            }),
            Self::Failed { id, error } => serde_json::json!({
                "type": "failed",
                "agent_id": id,
                "error": error,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentEventLog
// ---------------------------------------------------------------------------

/// Records and queries agent lifecycle events with a configurable capacity.
#[derive(Debug)]
pub struct AgentEventLog {
    events: Vec<AgentEvent>,
    max_events: usize,
}

impl AgentEventLog {
    /// Create a new event log with the given maximum capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            max_events,
        }
    }

    /// Record an event. If the log is at capacity, the oldest event is removed.
    pub fn record(&mut self, event: AgentEvent) {
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Return all events for the given agent ID.
    pub fn events_for(&self, agent_id: &str) -> Vec<&AgentEvent> {
        self.events
            .iter()
            .filter(|e| e.agent_id() == agent_id)
            .collect()
    }

    /// Return the most recent `n` events.
    pub fn recent(&self, n: usize) -> Vec<&AgentEvent> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().collect()
    }

    /// Return the total number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AgentSpec building --

    #[test]
    fn test_spec_builder_defaults() {
        let spec = AgentSpec::builder("test-agent").build();
        assert_eq!(spec.name, "test-agent");
        assert_eq!(spec.model, "claude-sonnet-4-6");
        assert!(spec.system_prompt.is_none());
        assert!(spec.tools.is_empty());
        assert_eq!(spec.max_iterations, 25);
        assert!(spec.temperature.is_none());
        assert!(spec.middleware.is_empty());
        assert_eq!(spec.backend, "memory");
    }

    #[test]
    fn test_spec_builder_full() {
        let spec = AgentSpec::builder("my-agent")
            .model("gpt-4")
            .system_prompt("Be helpful")
            .tool("search")
            .tool("calculator")
            .max_iterations(50)
            .temperature(0.5)
            .middleware("logging")
            .middleware("memory")
            .backend("filesystem")
            .build();

        assert_eq!(spec.name, "my-agent");
        assert_eq!(spec.model, "gpt-4");
        assert_eq!(spec.system_prompt, Some("Be helpful".to_string()));
        assert_eq!(spec.tools, vec!["search", "calculator"]);
        assert_eq!(spec.max_iterations, 50);
        assert_eq!(spec.temperature, Some(0.5));
        assert_eq!(spec.middleware, vec!["logging", "memory"]);
        assert_eq!(spec.backend, "filesystem");
    }

    #[test]
    fn test_spec_builder_tools_override() {
        let spec = AgentSpec::builder("agent")
            .tool("a")
            .tools(vec!["x".to_string(), "y".to_string()])
            .build();
        assert_eq!(spec.tools, vec!["x", "y"]);
    }

    #[test]
    fn test_spec_builder_middlewares_override() {
        let spec = AgentSpec::builder("agent")
            .middleware("a")
            .middlewares(vec!["x".to_string()])
            .build();
        assert_eq!(spec.middleware, vec!["x"]);
    }

    #[test]
    fn test_spec_to_json() {
        let spec = AgentSpec::builder("json-agent")
            .model("gpt-4")
            .temperature(0.7)
            .tool("search")
            .build();
        let json = spec.to_json();
        assert_eq!(json["name"], "json-agent");
        assert_eq!(json["model"], "gpt-4");
        assert_eq!(json["temperature"], 0.7);
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0], "search");
    }

    #[test]
    fn test_spec_to_json_null_optionals() {
        let spec = AgentSpec::builder("agent").build();
        let json = spec.to_json();
        assert!(json["system_prompt"].is_null());
        assert!(json["temperature"].is_null());
    }

    // -- AgentSpec validation --

    #[test]
    fn test_spec_validate_success() {
        let spec = AgentSpec::builder("valid").build();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_spec_validate_empty_name() {
        let spec = AgentSpec::builder("").build();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("name"));
    }

    #[test]
    fn test_spec_validate_empty_model() {
        let spec = AgentSpec::builder("agent").model("").build();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn test_spec_validate_zero_iterations() {
        let spec = AgentSpec::builder("agent").max_iterations(0).build();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("max_iterations"));
    }

    #[test]
    fn test_spec_validate_temperature_too_high() {
        let spec = AgentSpec::builder("agent").temperature(3.0).build();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn test_spec_validate_temperature_negative() {
        let spec = AgentSpec::builder("agent").temperature(-0.1).build();
        let err = spec.validate().unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn test_spec_validate_temperature_valid_boundary() {
        let spec = AgentSpec::builder("agent").temperature(0.0).build();
        assert!(spec.validate().is_ok());
        let spec = AgentSpec::builder("agent").temperature(2.0).build();
        assert!(spec.validate().is_ok());
    }

    // -- AgentStatus --

    #[test]
    fn test_status_as_str() {
        assert_eq!(AgentStatus::Created.as_str(), "created");
        assert_eq!(AgentStatus::Running.as_str(), "running");
        assert_eq!(AgentStatus::Paused.as_str(), "paused");
        assert_eq!(AgentStatus::Completed.as_str(), "completed");
        assert_eq!(AgentStatus::Failed("err".to_string()).as_str(), "failed");
    }

    #[test]
    fn test_status_is_terminal() {
        assert!(!AgentStatus::Created.is_terminal());
        assert!(!AgentStatus::Running.is_terminal());
        assert!(!AgentStatus::Paused.is_terminal());
        assert!(AgentStatus::Completed.is_terminal());
        assert!(AgentStatus::Failed("err".to_string()).is_terminal());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(AgentStatus::Created.to_string(), "Created");
        assert_eq!(AgentStatus::Running.to_string(), "Running");
        assert_eq!(AgentStatus::Paused.to_string(), "Paused");
        assert_eq!(AgentStatus::Completed.to_string(), "Completed");
        assert_eq!(
            AgentStatus::Failed("timeout".to_string()).to_string(),
            "Failed(timeout)"
        );
    }

    #[test]
    fn test_status_equality() {
        assert_eq!(AgentStatus::Created, AgentStatus::Created);
        assert_ne!(AgentStatus::Created, AgentStatus::Running);
        assert_eq!(
            AgentStatus::Failed("a".to_string()),
            AgentStatus::Failed("a".to_string())
        );
        assert_ne!(
            AgentStatus::Failed("a".to_string()),
            AgentStatus::Failed("b".to_string())
        );
    }

    // -- AgentInstance --

    #[test]
    fn test_instance_is_running() {
        let factory = AgentFactory::new();
        let mut instance = factory.create_with_defaults("test").unwrap();
        assert!(!instance.is_running());

        instance.status = AgentStatus::Running;
        assert!(instance.is_running());

        instance.status = AgentStatus::Completed;
        assert!(!instance.is_running());
    }

    #[test]
    fn test_instance_to_json() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("json-test").unwrap();
        let json = instance.to_json();
        assert_eq!(json["spec"]["name"], "json-test");
        assert!(!json["id"].as_str().unwrap().is_empty());
        assert_eq!(json["status"], "Created");
    }

    #[test]
    fn test_instance_has_uuid() {
        let factory = AgentFactory::new();
        let i1 = factory.create_with_defaults("a").unwrap();
        let i2 = factory.create_with_defaults("b").unwrap();
        assert_ne!(i1.id, i2.id);
        // UUID v4 format: 8-4-4-4-12 hex chars
        assert_eq!(i1.id.len(), 36);
    }

    #[test]
    fn test_instance_metadata() {
        let factory = AgentFactory::new();
        let mut instance = factory.create_with_defaults("meta").unwrap();
        instance
            .metadata
            .insert("key".to_string(), serde_json::json!("value"));
        assert_eq!(instance.metadata.get("key").unwrap(), "value");
    }

    // -- AgentFactory --

    #[test]
    fn test_factory_create() {
        let factory = AgentFactory::new();
        let spec = AgentSpec::builder("test-create")
            .model("gpt-4")
            .tool("search")
            .build();
        let instance = factory.create(spec).unwrap();
        assert_eq!(instance.spec.name, "test-create");
        assert_eq!(instance.status, AgentStatus::Created);
    }

    #[test]
    fn test_factory_create_with_defaults() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("defaults-test").unwrap();
        assert_eq!(instance.spec.name, "defaults-test");
        assert_eq!(instance.spec.model, "claude-sonnet-4-6");
        assert_eq!(instance.spec.max_iterations, 25);
        assert!(instance.spec.system_prompt.is_some());
        assert_eq!(instance.spec.backend, "memory");
    }

    #[test]
    fn test_factory_create_invalid_spec() {
        let factory = AgentFactory::new();
        let spec = AgentSpec::builder("").build();
        assert!(factory.create(spec).is_err());
    }

    #[test]
    fn test_factory_from_template() {
        let factory = AgentFactory::new();
        let template = AgentTemplate::coding_agent();
        let instance = factory.from_template(&template).unwrap();
        assert_eq!(instance.spec.name, "coding-agent");
        assert_eq!(instance.spec.temperature, Some(0.2));
        assert!(instance.spec.tools.contains(&"read_file".to_string()));
    }

    #[test]
    fn test_factory_default() {
        let _factory = AgentFactory::default();
    }

    // -- AgentTemplate --

    #[test]
    fn test_template_coding_agent() {
        let tmpl = AgentTemplate::coding_agent();
        assert_eq!(tmpl.name, "coding-agent");
        assert!(!tmpl.description.is_empty());
        assert!(tmpl.tags.contains(&"coding".to_string()));
        let spec = tmpl.to_spec();
        assert_eq!(spec.max_iterations, 50);
        assert_eq!(spec.temperature, Some(0.2));
        assert_eq!(spec.tools.len(), 4);
    }

    #[test]
    fn test_template_research_agent() {
        let tmpl = AgentTemplate::research_agent();
        assert_eq!(tmpl.name, "research-agent");
        assert!(tmpl.tags.contains(&"research".to_string()));
        let spec = tmpl.to_spec();
        assert_eq!(spec.max_iterations, 30);
        assert_eq!(spec.temperature, Some(0.3));
        assert!(spec.tools.contains(&"web_search".to_string()));
    }

    #[test]
    fn test_template_chat_agent() {
        let tmpl = AgentTemplate::chat_agent();
        assert_eq!(tmpl.name, "chat-agent");
        assert!(tmpl.tags.contains(&"chat".to_string()));
        let spec = tmpl.to_spec();
        assert_eq!(spec.max_iterations, 10);
        assert_eq!(spec.temperature, Some(0.7));
        assert!(spec.tools.is_empty());
    }

    #[test]
    fn test_template_new_without_spec() {
        let tmpl = AgentTemplate::new("custom", "A custom template");
        assert_eq!(tmpl.name, "custom");
        assert_eq!(tmpl.description, "A custom template");
        assert!(tmpl.tags.is_empty());
        let spec = tmpl.to_spec();
        assert_eq!(spec.name, "custom");
        assert_eq!(spec.model, "claude-sonnet-4-6");
    }

    #[test]
    fn test_template_with_spec() {
        let spec = AgentSpec::builder("custom")
            .model("gpt-4")
            .temperature(0.9)
            .build();
        let tmpl = AgentTemplate::new("my-tmpl", "desc").with_spec(spec);
        let out = tmpl.to_spec();
        assert_eq!(out.model, "gpt-4");
        assert_eq!(out.temperature, Some(0.9));
    }

    #[test]
    fn test_template_with_tags() {
        let tmpl = AgentTemplate::new("t", "d").with_tags(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(tmpl.tags, vec!["a", "b"]);
    }

    // -- AgentPool --

    #[test]
    fn test_pool_new_is_empty() {
        let pool = AgentPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_pool_add_and_get() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("pool-agent").unwrap();
        let id = instance.id.clone();

        let mut pool = AgentPool::new();
        pool.add(instance);

        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert!(pool.get(&id).is_some());
        assert_eq!(pool.get(&id).unwrap().spec.name, "pool-agent");
    }

    #[test]
    fn test_pool_get_mut() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("mutable").unwrap();
        let id = instance.id.clone();

        let mut pool = AgentPool::new();
        pool.add(instance);

        let inst = pool.get_mut(&id).unwrap();
        inst.status = AgentStatus::Running;
        assert_eq!(pool.get(&id).unwrap().status, AgentStatus::Running);
    }

    #[test]
    fn test_pool_remove() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("removable").unwrap();
        let id = instance.id.clone();

        let mut pool = AgentPool::new();
        pool.add(instance);
        assert_eq!(pool.len(), 1);

        let removed = pool.remove(&id).unwrap();
        assert_eq!(removed.spec.name, "removable");
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_remove_nonexistent() {
        let mut pool = AgentPool::new();
        assert!(pool.remove("nonexistent").is_none());
    }

    #[test]
    fn test_pool_get_nonexistent() {
        let pool = AgentPool::new();
        assert!(pool.get("nonexistent").is_none());
    }

    #[test]
    fn test_pool_by_status() {
        let factory = AgentFactory::new();

        let mut i1 = factory.create_with_defaults("a").unwrap();
        i1.status = AgentStatus::Running;
        let mut i2 = factory.create_with_defaults("b").unwrap();
        i2.status = AgentStatus::Running;
        let i3 = factory.create_with_defaults("c").unwrap(); // Created

        let mut pool = AgentPool::new();
        pool.add(i1);
        pool.add(i2);
        pool.add(i3);

        let running = pool.by_status(&AgentStatus::Running);
        assert_eq!(running.len(), 2);

        let created = pool.by_status(&AgentStatus::Created);
        assert_eq!(created.len(), 1);

        let completed = pool.by_status(&AgentStatus::Completed);
        assert!(completed.is_empty());
    }

    #[test]
    fn test_pool_active_count() {
        let factory = AgentFactory::new();

        let mut i1 = factory.create_with_defaults("a").unwrap();
        i1.status = AgentStatus::Running;
        let mut i2 = factory.create_with_defaults("b").unwrap();
        i2.status = AgentStatus::Completed;
        let mut i3 = factory.create_with_defaults("c").unwrap();
        i3.status = AgentStatus::Failed("err".to_string());
        let i4 = factory.create_with_defaults("d").unwrap(); // Created

        let mut pool = AgentPool::new();
        pool.add(i1);
        pool.add(i2);
        pool.add(i3);
        pool.add(i4);

        assert_eq!(pool.active_count(), 2); // Running + Created
        assert_eq!(pool.len(), 4);
    }

    #[test]
    fn test_pool_to_json() {
        let factory = AgentFactory::new();
        let instance = factory.create_with_defaults("json-pool").unwrap();

        let mut pool = AgentPool::new();
        pool.add(instance);

        let json = pool.to_json();
        assert_eq!(json["count"], 1);
        assert_eq!(json["active_count"], 1);
        assert!(json["instances"].is_array());
        assert_eq!(json["instances"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_pool_empty_to_json() {
        let pool = AgentPool::new();
        let json = pool.to_json();
        assert_eq!(json["count"], 0);
        assert_eq!(json["active_count"], 0);
        assert!(json["instances"].as_array().unwrap().is_empty());
    }

    // -- AgentEvent --

    #[test]
    fn test_event_agent_id() {
        let id = "abc-123".to_string();
        assert_eq!(AgentEvent::Created { id: id.clone() }.agent_id(), "abc-123");
        assert_eq!(AgentEvent::Started { id: id.clone() }.agent_id(), "abc-123");
        assert_eq!(AgentEvent::Paused { id: id.clone() }.agent_id(), "abc-123");
        assert_eq!(
            AgentEvent::Completed {
                id: id.clone(),
                result: Value::Null
            }
            .agent_id(),
            "abc-123"
        );
        assert_eq!(
            AgentEvent::Failed {
                id,
                error: "err".to_string()
            }
            .agent_id(),
            "abc-123"
        );
    }

    #[test]
    fn test_event_event_type() {
        assert_eq!(
            AgentEvent::Created {
                id: "x".to_string()
            }
            .event_type(),
            "created"
        );
        assert_eq!(
            AgentEvent::Started {
                id: "x".to_string()
            }
            .event_type(),
            "started"
        );
        assert_eq!(
            AgentEvent::Paused {
                id: "x".to_string()
            }
            .event_type(),
            "paused"
        );
        assert_eq!(
            AgentEvent::Completed {
                id: "x".to_string(),
                result: Value::Null
            }
            .event_type(),
            "completed"
        );
        assert_eq!(
            AgentEvent::Failed {
                id: "x".to_string(),
                error: "e".to_string()
            }
            .event_type(),
            "failed"
        );
    }

    #[test]
    fn test_event_to_json_created() {
        let json = AgentEvent::Created {
            id: "abc".to_string(),
        }
        .to_json();
        assert_eq!(json["type"], "created");
        assert_eq!(json["agent_id"], "abc");
    }

    #[test]
    fn test_event_to_json_completed() {
        let json = AgentEvent::Completed {
            id: "abc".to_string(),
            result: serde_json::json!({"answer": 42}),
        }
        .to_json();
        assert_eq!(json["type"], "completed");
        assert_eq!(json["result"]["answer"], 42);
    }

    #[test]
    fn test_event_to_json_failed() {
        let json = AgentEvent::Failed {
            id: "abc".to_string(),
            error: "timeout".to_string(),
        }
        .to_json();
        assert_eq!(json["type"], "failed");
        assert_eq!(json["error"], "timeout");
    }

    // -- AgentEventLog --

    #[test]
    fn test_event_log_new_is_empty() {
        let log = AgentEventLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_event_log_record_and_len() {
        let mut log = AgentEventLog::new(100);
        log.record(AgentEvent::Created {
            id: "a".to_string(),
        });
        log.record(AgentEvent::Started {
            id: "a".to_string(),
        });
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_event_log_events_for() {
        let mut log = AgentEventLog::new(100);
        log.record(AgentEvent::Created {
            id: "a".to_string(),
        });
        log.record(AgentEvent::Created {
            id: "b".to_string(),
        });
        log.record(AgentEvent::Started {
            id: "a".to_string(),
        });

        let a_events = log.events_for("a");
        assert_eq!(a_events.len(), 2);
        assert_eq!(a_events[0].event_type(), "created");
        assert_eq!(a_events[1].event_type(), "started");

        let b_events = log.events_for("b");
        assert_eq!(b_events.len(), 1);

        let c_events = log.events_for("c");
        assert!(c_events.is_empty());
    }

    #[test]
    fn test_event_log_recent() {
        let mut log = AgentEventLog::new(100);
        for i in 0..5 {
            log.record(AgentEvent::Created {
                id: format!("agent-{i}"),
            });
        }

        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].agent_id(), "agent-2");
        assert_eq!(recent[1].agent_id(), "agent-3");
        assert_eq!(recent[2].agent_id(), "agent-4");
    }

    #[test]
    fn test_event_log_recent_more_than_available() {
        let mut log = AgentEventLog::new(100);
        log.record(AgentEvent::Created {
            id: "a".to_string(),
        });
        let recent = log.recent(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_event_log_clear() {
        let mut log = AgentEventLog::new(100);
        log.record(AgentEvent::Created {
            id: "a".to_string(),
        });
        log.record(AgentEvent::Started {
            id: "a".to_string(),
        });
        assert_eq!(log.len(), 2);
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_event_log_max_capacity() {
        let mut log = AgentEventLog::new(3);
        for i in 0..5 {
            log.record(AgentEvent::Created {
                id: format!("agent-{i}"),
            });
        }
        assert_eq!(log.len(), 3);
        // Oldest events (agent-0, agent-1) should be evicted
        let all = log.recent(10);
        assert_eq!(all[0].agent_id(), "agent-2");
        assert_eq!(all[1].agent_id(), "agent-3");
        assert_eq!(all[2].agent_id(), "agent-4");
    }

    #[test]
    fn test_event_log_max_capacity_one() {
        let mut log = AgentEventLog::new(1);
        log.record(AgentEvent::Created {
            id: "a".to_string(),
        });
        log.record(AgentEvent::Started {
            id: "b".to_string(),
        });
        assert_eq!(log.len(), 1);
        assert_eq!(log.recent(1)[0].agent_id(), "b");
    }

    // -- Edge cases --

    #[test]
    fn test_factory_error_display() {
        let err = FactoryError::InvalidSpec("bad".to_string());
        assert!(err.to_string().contains("bad"));
        let err = FactoryError::CreationFailed("fail".to_string());
        assert!(err.to_string().contains("fail"));
    }

    #[test]
    fn test_pool_default() {
        let pool = AgentPool::default();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_by_status_with_failed() {
        let factory = AgentFactory::new();
        let mut i1 = factory.create_with_defaults("a").unwrap();
        i1.status = AgentStatus::Failed("err1".to_string());
        let mut i2 = factory.create_with_defaults("b").unwrap();
        i2.status = AgentStatus::Failed("err2".to_string());

        let mut pool = AgentPool::new();
        pool.add(i1);
        pool.add(i2);

        // by_status with Failed matches the exact variant
        let failed = pool.by_status(&AgentStatus::Failed("err1".to_string()));
        assert_eq!(failed.len(), 1);
    }
}
