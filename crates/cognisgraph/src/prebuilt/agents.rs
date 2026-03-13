//! Standalone prebuilt agent primitives for lightweight tool-calling workflows.
//!
//! This module provides self-contained agent building blocks that do **not**
//! require a full [`BaseChatModel`] implementation. Instead, tools are simple
//! closure-based [`ToolDefinition`] values and the think-act loop is driven
//! by explicit tool-call directives in the input.
//!
//! The primary entry point is [`create_react_agent`], which builds a
//! [`ReActAgent`] with a configurable tool set and iteration limit.
//!
//! # Quick Example
//!
//! ```rust
//! use cognisgraph::prebuilt::agents::{create_react_agent, ReactAgentConfig, ToolDefinition};
//! use serde_json::json;
//!
//! let tools = vec![
//!     ToolDefinition::new("calculator", "Evaluate math expressions", |_input| {
//!         Ok(json!({"result": 42}))
//!     }),
//! ];
//! let config = ReactAgentConfig::builder().max_iterations(10).build();
//! let agent = create_react_agent(tools, config);
//! let state = agent.invoke(json!({"input": "What is 6 * 7?"})).unwrap();
//! assert!(state.is_finished);
//! ```

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;

use crate::errors::LangGraphError;

// ---------------------------------------------------------------------------
// ToolChoice
// ---------------------------------------------------------------------------

/// Strategy for selecting which tools the agent may call.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// The agent decides whether to call a tool.
    #[default]
    Auto,
    /// The agent must call at least one tool.
    Required,
    /// The agent must not call any tools.
    None,
    /// The agent must call the specified tool.
    Specific(String),
}

impl ToolChoice {
    /// Return the choice as a human-readable string.
    pub fn as_str(&self) -> &str {
        match self {
            ToolChoice::Auto => "auto",
            ToolChoice::Required => "required",
            ToolChoice::None => "none",
            ToolChoice::Specific(name) => name.as_str(),
        }
    }
}

impl fmt::Display for ToolChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// ToolCall
// ---------------------------------------------------------------------------

/// Represents a pending or resolved tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Unique identifier for this call.
    pub id: String,
    /// Name of the tool to invoke.
    pub tool_name: String,
    /// Arguments to pass to the tool handler.
    pub arguments: Value,
    /// Result of the invocation, if resolved.
    pub result: Option<Value>,
}

impl ToolCall {
    /// Create a new unresolved tool call.
    pub fn new(id: impl Into<String>, tool_name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            tool_name: tool_name.into(),
            arguments,
            result: None,
        }
    }

    /// Return a new `ToolCall` with the given result attached.
    pub fn with_result(mut self, value: Value) -> Self {
        self.result = Some(value);
        self
    }

    /// Whether this call has been resolved (has a result).
    pub fn is_resolved(&self) -> bool {
        self.result.is_some()
    }

    /// Serialize the tool call to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
        })
    }
}

// ---------------------------------------------------------------------------
// ToolDefinition
// ---------------------------------------------------------------------------

/// A named tool with a description and a handler function.
pub struct ToolDefinition {
    /// The tool name (used to match tool calls).
    name: String,
    /// Human-readable description of what the tool does.
    description: String,
    /// The handler that executes the tool logic.
    handler: Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new<F>(name: impl Into<String>, description: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Box::new(handler),
        }
    }

    /// Execute the tool with the given input.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        (self.handler)(input)
    }

    /// The tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tool description.
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AgentState
// ---------------------------------------------------------------------------

/// Mutable state carried through a ReAct agent's think-act loop.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Accumulated conversation messages.
    pub messages: Vec<Value>,
    /// Pending or resolved tool calls for the current iteration.
    pub tool_calls: Vec<ToolCall>,
    /// Current iteration number (0-based).
    pub iteration: usize,
    /// Maximum iterations before the agent is forced to stop.
    pub max_iterations: usize,
    /// Whether the agent has decided to stop.
    pub is_finished: bool,
}

impl AgentState {
    /// Create a new agent state with the given iteration limit.
    pub fn new(max_iterations: usize) -> Self {
        Self {
            messages: Vec::new(),
            tool_calls: Vec::new(),
            iteration: 0,
            max_iterations,
            is_finished: false,
        }
    }

    /// Append a message to the conversation history.
    pub fn add_message(&mut self, msg: Value) {
        self.messages.push(msg);
    }

    /// Register a tool call.
    pub fn add_tool_call(&mut self, call: ToolCall) {
        self.tool_calls.push(call);
    }

    /// Mark the agent as finished (no more iterations).
    pub fn mark_finished(&mut self) {
        self.is_finished = true;
    }

    /// Whether the agent has reached its iteration limit.
    pub fn is_over_limit(&self) -> bool {
        self.iteration >= self.max_iterations
    }

    /// Serialize the state to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "messages": self.messages,
            "tool_calls": self.tool_calls.iter().map(|tc| tc.to_json()).collect::<Vec<_>>(),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "is_finished": self.is_finished,
        })
    }
}

// ---------------------------------------------------------------------------
// ReactAgentConfig
// ---------------------------------------------------------------------------

/// Configuration for a ReAct agent.
#[derive(Debug, Clone)]
pub struct ReactAgentConfig {
    /// Maximum number of think-act iterations (default: 25).
    pub max_iterations: usize,
    /// Tool selection strategy.
    pub tool_choice: ToolChoice,
    /// Optional system message prepended to every invocation.
    pub system_message: Option<String>,
}

impl Default for ReactAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            tool_choice: ToolChoice::Auto,
            system_message: None,
        }
    }
}

impl ReactAgentConfig {
    /// Start building a config with defaults.
    pub fn builder() -> ReactAgentConfigBuilder {
        ReactAgentConfigBuilder::default()
    }
}

/// Builder for [`ReactAgentConfig`].
#[derive(Debug, Clone)]
pub struct ReactAgentConfigBuilder {
    max_iterations: usize,
    tool_choice: ToolChoice,
    system_message: Option<String>,
}

impl Default for ReactAgentConfigBuilder {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            tool_choice: ToolChoice::Auto,
            system_message: None,
        }
    }
}

impl ReactAgentConfigBuilder {
    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the tool choice strategy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Set the system message.
    pub fn system_message(mut self, msg: impl Into<String>) -> Self {
        self.system_message = Some(msg.into());
        self
    }

    /// Build the config.
    pub fn build(self) -> ReactAgentConfig {
        ReactAgentConfig {
            max_iterations: self.max_iterations,
            tool_choice: self.tool_choice,
            system_message: self.system_message,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolNode
// ---------------------------------------------------------------------------

/// A node that executes pending (unresolved) tool calls against a tool registry.
pub struct ToolNode {
    tools: HashMap<String, ToolDefinition>,
}

impl ToolNode {
    /// Create a tool node from a list of tool definitions.
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        let map = tools.into_iter().map(|t| (t.name.clone(), t)).collect();
        Self { tools: map }
    }

    /// Execute all unresolved tool calls in the agent state.
    ///
    /// Each resolved call gets its result attached and a tool-role message is
    /// appended to the state. If a tool is not found or errors, the error
    /// string is stored as the result.
    pub fn execute(&self, state: &mut AgentState) -> Result<(), LangGraphError> {
        let calls = std::mem::take(&mut state.tool_calls);
        let mut resolved = Vec::with_capacity(calls.len());

        for call in calls {
            if call.is_resolved() {
                resolved.push(call);
                continue;
            }

            let result = match self.tools.get(&call.tool_name) {
                Some(tool) => match tool.execute(call.arguments.clone()) {
                    Ok(value) => value,
                    Err(err) => json!({ "error": err }),
                },
                None => json!({ "error": format!("Tool '{}' not found", call.tool_name) }),
            };

            state.add_message(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }));

            resolved.push(call.with_result(result));
        }

        state.tool_calls = resolved;
        Ok(())
    }

    /// List available tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl fmt::Debug for ToolNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolNode")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ReActAgent
// ---------------------------------------------------------------------------

/// A prebuilt ReAct agent that implements the think-act loop over a tool set.
///
/// The agent alternates between an "agent" step (deciding whether to call
/// tools or finish) and a "tools" step (executing pending tool calls). The
/// decision logic uses a simple heuristic: if the input contains tool call
/// directives (JSON objects with `tool_calls`), those are dispatched; otherwise
/// the agent finishes.
pub struct ReActAgent {
    tool_node: ToolNode,
    config: ReactAgentConfig,
    tool_names: Vec<String>,
}

impl ReActAgent {
    /// Create a new ReAct agent from tools and configuration.
    pub fn new(tools: Vec<ToolDefinition>, config: ReactAgentConfig) -> Self {
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        let tool_node = ToolNode::new(tools);
        Self {
            tool_node,
            config,
            tool_names,
        }
    }

    /// Run the agent's think-act loop on the given input.
    ///
    /// The input can be a JSON object. If it contains a `"messages"` array,
    /// those are used as the initial conversation. If it contains
    /// `"tool_calls"`, the agent starts in tool-execution mode.
    ///
    /// Returns the final [`AgentState`] after the agent finishes or hits the
    /// iteration limit.
    pub fn invoke(&self, input: Value) -> crate::errors::Result<AgentState> {
        let mut state = AgentState::new(self.config.max_iterations);

        // Seed system message if configured.
        if let Some(ref sys) = self.config.system_message {
            state.add_message(json!({
                "role": "system",
                "content": sys,
            }));
        }

        // Seed initial messages from input.
        if let Some(messages) = input.get("messages").and_then(|v| v.as_array()) {
            for msg in messages {
                state.add_message(msg.clone());
            }
        } else if input.is_object() && !input.as_object().unwrap().is_empty() {
            // Treat the whole input as a single human message.
            state.add_message(json!({
                "role": "human",
                "content": input,
            }));
        }

        // Seed initial tool calls from input.
        if let Some(calls) = input.get("tool_calls").and_then(|v| v.as_array()) {
            for call_val in calls {
                let tc = ToolCall::new(
                    call_val
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    call_val
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    call_val.get("arguments").cloned().unwrap_or(Value::Null),
                );
                state.add_tool_call(tc);
            }
        }

        // Think-act loop.
        loop {
            if state.is_finished || state.is_over_limit() {
                if state.is_over_limit() && !state.is_finished {
                    state.add_message(json!({
                        "role": "system",
                        "content": "Maximum iterations reached. Stopping.",
                    }));
                    state.mark_finished();
                }
                break;
            }

            // --- Agent step: decide whether there are tool calls to execute ---
            let has_pending = state.tool_calls.iter().any(|tc| !tc.is_resolved());

            if has_pending {
                // --- Tools step: execute pending calls ---
                self.tool_node.execute(&mut state)?;
                state.iteration += 1;
            } else {
                // No pending tool calls — the agent is done.
                state.mark_finished();
            }
        }

        Ok(state)
    }

    /// Return the names of all available tools.
    pub fn available_tools(&self) -> Vec<&str> {
        self.tool_names.iter().map(|s| s.as_str()).collect()
    }

    /// Return a reference to the agent configuration.
    pub fn config(&self) -> &ReactAgentConfig {
        &self.config
    }
}

impl fmt::Debug for ReActAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReActAgent")
            .field("tool_names", &self.tool_names)
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create a prebuilt ReAct agent with the given tools and configuration.
///
/// This is the primary entry point for lightweight tool-calling agents that
/// do not require a full chat model. It constructs a [`ReActAgent`] that
/// implements the ReAct (Reasoning + Acting) pattern with the provided tool
/// definitions and configuration.
///
/// # Arguments
///
/// * `tools` — the tools the agent can invoke.
/// * `config` — agent configuration (iterations, tool choice, system prompt).
///
/// # Returns
///
/// A [`ReActAgent`] ready to process inputs via [`ReActAgent::invoke`].
pub fn create_react_agent(tools: Vec<ToolDefinition>, config: ReactAgentConfig) -> ReActAgent {
    ReActAgent::new(tools, config)
}

/// Create a [`ToolNode`] from a list of tool definitions.
///
/// Useful when you want to embed tool execution into a custom graph without
/// using the full [`ReActAgent`].
pub fn create_tool_node(tools: Vec<ToolDefinition>) -> ToolNode {
    ToolNode::new(tools)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ToolChoice ---------------------------------------------------------

    #[test]
    fn tool_choice_auto_as_str() {
        assert_eq!(ToolChoice::Auto.as_str(), "auto");
    }

    #[test]
    fn tool_choice_required_as_str() {
        assert_eq!(ToolChoice::Required.as_str(), "required");
    }

    #[test]
    fn tool_choice_none_as_str() {
        assert_eq!(ToolChoice::None.as_str(), "none");
    }

    #[test]
    fn tool_choice_specific_as_str() {
        let tc = ToolChoice::Specific("calculator".into());
        assert_eq!(tc.as_str(), "calculator");
    }

    #[test]
    fn tool_choice_default_is_auto() {
        assert_eq!(ToolChoice::default(), ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_display() {
        assert_eq!(format!("{}", ToolChoice::Required), "required");
        assert_eq!(format!("{}", ToolChoice::Specific("foo".into())), "foo");
    }

    #[test]
    fn tool_choice_equality() {
        assert_eq!(ToolChoice::Auto, ToolChoice::Auto);
        assert_ne!(ToolChoice::Auto, ToolChoice::Required);
        assert_eq!(
            ToolChoice::Specific("a".into()),
            ToolChoice::Specific("a".into())
        );
        assert_ne!(
            ToolChoice::Specific("a".into()),
            ToolChoice::Specific("b".into())
        );
    }

    #[test]
    fn tool_choice_clone() {
        let tc = ToolChoice::Specific("x".into());
        let tc2 = tc.clone();
        assert_eq!(tc, tc2);
    }

    // -- ToolCall -----------------------------------------------------------

    #[test]
    fn tool_call_new_is_unresolved() {
        let tc = ToolCall::new("id1", "calc", json!({"expr": "1+1"}));
        assert!(!tc.is_resolved());
        assert_eq!(tc.id, "id1");
        assert_eq!(tc.tool_name, "calc");
    }

    #[test]
    fn tool_call_with_result_is_resolved() {
        let tc = ToolCall::new("id1", "calc", json!({})).with_result(json!(42));
        assert!(tc.is_resolved());
        assert_eq!(tc.result, Some(json!(42)));
    }

    #[test]
    fn tool_call_to_json_unresolved() {
        let tc = ToolCall::new("id1", "calc", json!({"x": 1}));
        let j = tc.to_json();
        assert_eq!(j["id"], "id1");
        assert_eq!(j["tool_name"], "calc");
        assert!(j["result"].is_null());
    }

    #[test]
    fn tool_call_to_json_resolved() {
        let tc = ToolCall::new("id2", "search", json!({})).with_result(json!("found"));
        let j = tc.to_json();
        assert_eq!(j["result"], "found");
    }

    #[test]
    fn tool_call_clone() {
        let tc = ToolCall::new("id1", "calc", json!(1));
        let tc2 = tc.clone();
        assert_eq!(tc2.id, "id1");
        assert_eq!(tc2.tool_name, "calc");
    }

    #[test]
    fn tool_call_arguments_preserved() {
        let args = json!({"a": 1, "b": [2, 3]});
        let tc = ToolCall::new("id", "tool", args.clone());
        assert_eq!(tc.arguments, args);
    }

    // -- ToolDefinition -----------------------------------------------------

    #[test]
    fn tool_definition_creation() {
        let tool = ToolDefinition::new("echo", "Echoes input", |input| Ok(input));
        assert_eq!(tool.name(), "echo");
        assert_eq!(tool.description(), "Echoes input");
    }

    #[test]
    fn tool_definition_execute_success() {
        let tool = ToolDefinition::new("double", "Doubles a number", |input| {
            let n = input.as_i64().unwrap_or(0);
            Ok(json!(n * 2))
        });
        let result = tool.execute(json!(5)).unwrap();
        assert_eq!(result, json!(10));
    }

    #[test]
    fn tool_definition_execute_error() {
        let tool = ToolDefinition::new("fail", "Always fails", |_| {
            Err("something went wrong".into())
        });
        let result = tool.execute(json!(null));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "something went wrong");
    }

    #[test]
    fn tool_definition_debug() {
        let tool = ToolDefinition::new("t", "d", |v| Ok(v));
        let dbg = format!("{:?}", tool);
        assert!(dbg.contains("ToolDefinition"));
        assert!(dbg.contains("\"t\""));
    }

    #[test]
    fn tool_definition_complex_handler() {
        let tool = ToolDefinition::new("concat", "Concatenates strings", |input| {
            let arr = input.as_array().ok_or("expected array")?;
            let result: String = arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(json!(result))
        });
        let result = tool.execute(json!(["hello", "world"])).unwrap();
        assert_eq!(result, json!("hello, world"));
    }

    #[test]
    fn tool_definition_null_input() {
        let tool = ToolDefinition::new("null_ok", "Handles null", |input| {
            if input.is_null() {
                Ok(json!("received null"))
            } else {
                Ok(input)
            }
        });
        let result = tool.execute(Value::Null).unwrap();
        assert_eq!(result, json!("received null"));
    }

    // -- AgentState ---------------------------------------------------------

    #[test]
    fn agent_state_new_defaults() {
        let state = AgentState::new(10);
        assert!(state.messages.is_empty());
        assert!(state.tool_calls.is_empty());
        assert_eq!(state.iteration, 0);
        assert_eq!(state.max_iterations, 10);
        assert!(!state.is_finished);
    }

    #[test]
    fn agent_state_add_message() {
        let mut state = AgentState::new(5);
        state.add_message(json!({"role": "human", "content": "hi"}));
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0]["role"], "human");
    }

    #[test]
    fn agent_state_add_multiple_messages() {
        let mut state = AgentState::new(5);
        state.add_message(json!("msg1"));
        state.add_message(json!("msg2"));
        state.add_message(json!("msg3"));
        assert_eq!(state.messages.len(), 3);
    }

    #[test]
    fn agent_state_add_tool_call() {
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("id1", "calc", json!({})));
        assert_eq!(state.tool_calls.len(), 1);
        assert_eq!(state.tool_calls[0].tool_name, "calc");
    }

    #[test]
    fn agent_state_mark_finished() {
        let mut state = AgentState::new(5);
        assert!(!state.is_finished);
        state.mark_finished();
        assert!(state.is_finished);
    }

    #[test]
    fn agent_state_is_over_limit_false() {
        let state = AgentState::new(3);
        assert!(!state.is_over_limit());
    }

    #[test]
    fn agent_state_is_over_limit_at_boundary() {
        let mut state = AgentState::new(3);
        state.iteration = 3;
        assert!(state.is_over_limit());
    }

    #[test]
    fn agent_state_is_over_limit_beyond() {
        let mut state = AgentState::new(3);
        state.iteration = 5;
        assert!(state.is_over_limit());
    }

    #[test]
    fn agent_state_to_json() {
        let mut state = AgentState::new(2);
        state.add_message(json!("hello"));
        state.add_tool_call(ToolCall::new("tc1", "tool", json!({})));
        let j = state.to_json();
        assert_eq!(j["max_iterations"], 2);
        assert_eq!(j["messages"].as_array().unwrap().len(), 1);
        assert_eq!(j["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(j["iteration"], 0);
        assert_eq!(j["is_finished"], false);
    }

    #[test]
    fn agent_state_clone() {
        let mut state = AgentState::new(5);
        state.add_message(json!("m"));
        state.iteration = 2;
        let cloned = state.clone();
        assert_eq!(cloned.iteration, 2);
        assert_eq!(cloned.messages.len(), 1);
    }

    // -- ReactAgentConfig ---------------------------------------------------

    #[test]
    fn config_default_values() {
        let config = ReactAgentConfig::default();
        assert_eq!(config.max_iterations, 25);
        assert_eq!(config.tool_choice, ToolChoice::Auto);
        assert!(config.system_message.is_none());
    }

    #[test]
    fn config_builder_max_iterations() {
        let config = ReactAgentConfig::builder().max_iterations(10).build();
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn config_builder_tool_choice() {
        let config = ReactAgentConfig::builder()
            .tool_choice(ToolChoice::Required)
            .build();
        assert_eq!(config.tool_choice, ToolChoice::Required);
    }

    #[test]
    fn config_builder_system_message() {
        let config = ReactAgentConfig::builder()
            .system_message("You are helpful.")
            .build();
        assert_eq!(config.system_message, Some("You are helpful.".into()));
    }

    #[test]
    fn config_builder_all_options() {
        let config = ReactAgentConfig::builder()
            .max_iterations(5)
            .tool_choice(ToolChoice::Specific("calc".into()))
            .system_message("Be concise.")
            .build();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.tool_choice, ToolChoice::Specific("calc".into()));
        assert_eq!(config.system_message, Some("Be concise.".into()));
    }

    #[test]
    fn config_builder_default_matches_config_default() {
        let from_builder = ReactAgentConfig::builder().build();
        let from_default = ReactAgentConfig::default();
        assert_eq!(from_builder.max_iterations, from_default.max_iterations);
        assert_eq!(from_builder.tool_choice, from_default.tool_choice);
        assert_eq!(from_builder.system_message, from_default.system_message);
    }

    #[test]
    fn config_builder_chained() {
        let config = ReactAgentConfig::builder()
            .max_iterations(3)
            .max_iterations(7)
            .build();
        assert_eq!(config.max_iterations, 7);
    }

    // -- ToolNode -----------------------------------------------------------

    #[test]
    fn tool_node_execute_resolves_calls() {
        let tools = vec![ToolDefinition::new("echo", "echoes", |v| Ok(v))];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "echo", json!("hello")));
        node.execute(&mut state).unwrap();
        assert!(state.tool_calls[0].is_resolved());
        assert_eq!(state.tool_calls[0].result, Some(json!("hello")));
    }

    #[test]
    fn tool_node_skips_resolved_calls() {
        let tools = vec![ToolDefinition::new("echo", "echoes", |v| Ok(v))];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "echo", json!("x")).with_result(json!("already")));
        let msg_count_before = state.messages.len();
        node.execute(&mut state).unwrap();
        assert_eq!(state.messages.len(), msg_count_before);
        assert_eq!(state.tool_calls[0].result, Some(json!("already")));
    }

    #[test]
    fn tool_node_unknown_tool_produces_error() {
        let node = create_tool_node(vec![]);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "nonexistent", json!({})));
        node.execute(&mut state).unwrap();
        assert!(state.tool_calls[0].is_resolved());
        let result = state.tool_calls[0].result.as_ref().unwrap();
        assert!(result["error"].as_str().unwrap().contains("not found"));
    }

    #[test]
    fn tool_node_handler_error_produces_error_result() {
        let tools = vec![ToolDefinition::new("bad", "fails", |_| Err("boom".into()))];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "bad", json!({})));
        node.execute(&mut state).unwrap();
        assert!(state.tool_calls[0].is_resolved());
        let result = state.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result["error"], "boom");
    }

    #[test]
    fn tool_node_multiple_calls() {
        let tools = vec![
            ToolDefinition::new("a", "tool a", |_| Ok(json!("result_a"))),
            ToolDefinition::new("b", "tool b", |_| Ok(json!("result_b"))),
        ];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "a", json!({})));
        state.add_tool_call(ToolCall::new("c2", "b", json!({})));
        node.execute(&mut state).unwrap();
        assert_eq!(state.tool_calls.len(), 2);
        assert!(state.tool_calls.iter().all(|tc| tc.is_resolved()));
        assert_eq!(state.messages.len(), 2);
    }

    #[test]
    fn tool_node_adds_tool_messages() {
        let tools = vec![ToolDefinition::new("t", "desc", |_| Ok(json!(99)))];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "t", json!({})));
        node.execute(&mut state).unwrap();
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0]["role"], "tool");
        assert_eq!(state.messages[0]["tool_call_id"], "c1");
    }

    #[test]
    fn tool_node_tool_names() {
        let tools = vec![
            ToolDefinition::new("alpha", "a", |v| Ok(v)),
            ToolDefinition::new("beta", "b", |v| Ok(v)),
        ];
        let node = create_tool_node(tools);
        let mut names = node.tool_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn tool_node_empty() {
        let node = create_tool_node(vec![]);
        assert!(node.tool_names().is_empty());
    }

    #[test]
    fn tool_node_debug() {
        let tools = vec![ToolDefinition::new("x", "d", |v| Ok(v))];
        let node = create_tool_node(tools);
        let dbg = format!("{:?}", node);
        assert!(dbg.contains("ToolNode"));
    }

    #[test]
    fn tool_node_mixed_resolved_unresolved() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!("ok")))];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "t", json!({})).with_result(json!("pre")));
        state.add_tool_call(ToolCall::new("c2", "t", json!({})));
        node.execute(&mut state).unwrap();
        assert_eq!(state.tool_calls[0].result, Some(json!("pre")));
        assert_eq!(state.tool_calls[1].result, Some(json!("ok")));
        // Only one new message for the unresolved call.
        assert_eq!(state.messages.len(), 1);
    }

    // -- create_react_agent / ReActAgent ------------------------------------

    #[test]
    fn create_react_agent_returns_agent() {
        let agent = create_react_agent(vec![], ReactAgentConfig::default());
        assert!(agent.available_tools().is_empty());
        assert_eq!(agent.config().max_iterations, 25);
    }

    #[test]
    fn react_agent_available_tools() {
        let tools = vec![
            ToolDefinition::new("calc", "calculator", |v| Ok(v)),
            ToolDefinition::new("search", "web search", |v| Ok(v)),
        ];
        let agent = create_react_agent(tools, ReactAgentConfig::default());
        let mut names = agent.available_tools();
        names.sort();
        assert_eq!(names, vec!["calc", "search"]);
    }

    #[test]
    fn react_agent_invoke_no_tools_finishes_immediately() {
        let agent = create_react_agent(vec![], ReactAgentConfig::default());
        let state = agent
            .invoke(json!({"messages": [{"role": "human", "content": "hi"}]}))
            .unwrap();
        assert!(state.is_finished);
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.iteration, 0);
    }

    #[test]
    fn react_agent_invoke_with_tool_calls() {
        let tools = vec![ToolDefinition::new("echo", "echoes", |v| Ok(v))];
        let agent = create_react_agent(tools, ReactAgentConfig::default());
        let input = json!({
            "messages": [{"role": "human", "content": "echo this"}],
            "tool_calls": [{"id": "tc1", "tool_name": "echo", "arguments": "ping"}]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        assert!(state.tool_calls[0].is_resolved());
        assert_eq!(state.tool_calls[0].result, Some(json!("ping")));
    }

    #[test]
    fn react_agent_invoke_with_system_message() {
        let config = ReactAgentConfig::builder()
            .system_message("You are a bot.")
            .build();
        let agent = create_react_agent(vec![], config);
        let state = agent.invoke(json!({})).unwrap();
        assert!(state.is_finished);
        assert_eq!(state.messages[0]["role"], "system");
        assert_eq!(state.messages[0]["content"], "You are a bot.");
    }

    #[test]
    fn react_agent_max_iteration_limit() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!("ok")))];
        let config = ReactAgentConfig::builder().max_iterations(1).build();
        let agent = create_react_agent(tools, config);
        let input = json!({
            "tool_calls": [{"id": "tc1", "tool_name": "t", "arguments": null}]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        assert_eq!(state.iteration, 1);
    }

    #[test]
    fn react_agent_zero_max_iterations() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!("ok")))];
        let config = ReactAgentConfig::builder().max_iterations(0).build();
        let agent = create_react_agent(tools, config);
        let input = json!({
            "tool_calls": [{"id": "tc1", "tool_name": "t", "arguments": null}]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        assert_eq!(state.iteration, 0);
        assert!(!state.tool_calls[0].is_resolved());
    }

    #[test]
    fn react_agent_invoke_plain_object_input() {
        let agent = create_react_agent(vec![], ReactAgentConfig::default());
        let state = agent.invoke(json!({"query": "hello"})).unwrap();
        assert!(state.is_finished);
        assert_eq!(state.messages[0]["role"], "human");
    }

    #[test]
    fn react_agent_invoke_empty_input() {
        let agent = create_react_agent(vec![], ReactAgentConfig::default());
        let state = agent.invoke(json!({})).unwrap();
        assert!(state.is_finished);
        assert!(state.messages.is_empty());
    }

    #[test]
    fn react_agent_all_tools_fail() {
        let tools = vec![
            ToolDefinition::new("bad1", "fails", |_| Err("err1".into())),
            ToolDefinition::new("bad2", "fails", |_| Err("err2".into())),
        ];
        let agent = create_react_agent(tools, ReactAgentConfig::default());
        let input = json!({
            "tool_calls": [
                {"id": "c1", "tool_name": "bad1", "arguments": null},
                {"id": "c2", "tool_name": "bad2", "arguments": null}
            ]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        assert!(state.tool_calls.iter().all(|tc| tc.is_resolved()));
        assert_eq!(
            state.tool_calls[0].result.as_ref().unwrap()["error"],
            "err1"
        );
        assert_eq!(
            state.tool_calls[1].result.as_ref().unwrap()["error"],
            "err2"
        );
    }

    #[test]
    fn react_agent_config_accessor() {
        let config = ReactAgentConfig::builder()
            .max_iterations(7)
            .tool_choice(ToolChoice::None)
            .build();
        let agent = create_react_agent(vec![], config);
        assert_eq!(agent.config().max_iterations, 7);
        assert_eq!(agent.config().tool_choice, ToolChoice::None);
    }

    #[test]
    fn react_agent_debug() {
        let agent = create_react_agent(vec![], ReactAgentConfig::default());
        let dbg = format!("{:?}", agent);
        assert!(dbg.contains("ReActAgent"));
    }

    #[test]
    fn react_agent_tool_call_with_unknown_tool() {
        let tools = vec![ToolDefinition::new("known", "d", |_| Ok(json!("ok")))];
        let agent = create_react_agent(tools, ReactAgentConfig::default());
        let input = json!({
            "tool_calls": [{"id": "c1", "tool_name": "unknown", "arguments": null}]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        let result = state.tool_calls[0].result.as_ref().unwrap();
        assert!(result["error"].as_str().unwrap().contains("not found"));
    }

    #[test]
    fn react_agent_system_message_plus_tool_calls() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!("res")))];
        let config = ReactAgentConfig::builder()
            .system_message("sys prompt")
            .build();
        let agent = create_react_agent(tools, config);
        let input = json!({
            "messages": [{"role": "human", "content": "do it"}],
            "tool_calls": [{"id": "c1", "tool_name": "t", "arguments": null}]
        });
        let state = agent.invoke(input).unwrap();
        assert!(state.is_finished);
        // System message should be first.
        assert_eq!(state.messages[0]["role"], "system");
        assert_eq!(state.messages[0]["content"], "sys prompt");
        // Human message next.
        assert_eq!(state.messages[1]["role"], "human");
        // Tool result message should follow.
        assert_eq!(state.messages[2]["role"], "tool");
    }

    #[test]
    fn react_agent_iteration_count_after_tool_execution() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!("ok")))];
        let config = ReactAgentConfig::builder().max_iterations(10).build();
        let agent = create_react_agent(tools, config);
        let input = json!({
            "tool_calls": [{"id": "c1", "tool_name": "t", "arguments": null}]
        });
        let state = agent.invoke(input).unwrap();
        // One iteration for the tool call, then finishes.
        assert_eq!(state.iteration, 1);
        assert!(state.is_finished);
    }

    // -- create_tool_node ---------------------------------------------------

    #[test]
    fn create_tool_node_returns_functional_node() {
        let tools = vec![ToolDefinition::new("t", "d", |_| Ok(json!(1)))];
        let node = create_tool_node(tools);
        assert_eq!(node.tool_names().len(), 1);
    }

    #[test]
    fn create_tool_node_empty_vec() {
        let node = create_tool_node(vec![]);
        assert!(node.tool_names().is_empty());
    }

    #[test]
    fn tool_node_execute_with_complex_result() {
        let tools = vec![ToolDefinition::new("obj", "returns object", |_| {
            Ok(json!({"key": "value", "nested": {"a": 1}}))
        })];
        let node = create_tool_node(tools);
        let mut state = AgentState::new(5);
        state.add_tool_call(ToolCall::new("c1", "obj", json!({})));
        node.execute(&mut state).unwrap();
        let result = state.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result["key"], "value");
        assert_eq!(result["nested"]["a"], 1);
    }
}
