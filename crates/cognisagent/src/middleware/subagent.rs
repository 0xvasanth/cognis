//! Sub-agent middleware — delegates tasks to isolated sub-agents.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use cognis_core::error::Result as CoreResult;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};

use crate::agent::DeepAgentError;
use crate::middleware::Middleware;

/// Declarative specification for a sub-agent.
///
/// Matches Python's `SubAgent` TypedDict from DeepAgents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpec {
    /// Unique identifier for the sub-agent.
    pub name: String,
    /// What this sub-agent does — used by the main agent to decide delegation.
    pub description: String,
    /// System prompt / instructions for the sub-agent.
    pub system_prompt: String,
    /// Model identifier (e.g., `"openai:gpt-4o"`). Inherits main agent's model if None.
    #[serde(default)]
    pub model: Option<String>,
    /// Tool names available to this sub-agent.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Skill source paths for SkillsMiddleware.
    #[serde(default)]
    pub skills: Vec<String>,
    /// Tool names that require human approval before execution.
    #[serde(default)]
    pub interrupt_on: Vec<String>,
    /// Maximum iterations before the sub-agent stops.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    25
}

/// A pre-compiled sub-agent with a ready-to-invoke function.
///
/// Matches Python's `CompiledSubAgent` TypedDict from DeepAgents.
/// An async function that invokes a compiled sub-agent.
pub type SubAgentInvokeFn = dyn Fn(
        Value,
    )
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, DeepAgentError>> + Send>>
    + Send
    + Sync;

/// The invoke function accepts JSON state (must contain `"messages"`)
/// and returns the final state.
pub struct CompiledSubAgentSpec {
    /// Unique identifier.
    pub name: String,
    /// What this sub-agent does.
    pub description: String,
    /// The compiled invoke function.
    pub invoke: Box<SubAgentInvokeFn>,
}

impl std::fmt::Debug for CompiledSubAgentSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledSubAgentSpec")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("invoke", &"<async fn>")
            .finish()
    }
}

/// Status of a sub-agent execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubAgentStatus {
    /// The sub-agent is currently running.
    Running,
    /// The sub-agent completed successfully.
    Completed,
    /// The sub-agent failed.
    Failed,
}

/// Tracks the state of an individual sub-agent invocation.
#[derive(Debug, Clone)]
pub struct SubAgentHandle {
    /// Description of the task delegated to this sub-agent.
    pub task: String,
    /// Current status of the sub-agent.
    pub status: SubAgentStatus,
    /// The result produced by the sub-agent (if completed).
    pub result: Option<String>,
}

/// Middleware that allows the main agent to delegate tasks to isolated sub-agents.
///
/// Provides a `SubAgentTool` that the main agent can invoke with a task description.
/// The tool spins up a lightweight sub-agent (a single model call), runs it to
/// completion, and returns the sub-agent's output.
pub struct SubAgentMiddleware {
    /// The chat model used by sub-agents.
    model: Arc<dyn BaseChatModel>,
    /// Maximum number of iterations a sub-agent may run.
    pub max_iterations: u32,
    /// Registry of active and completed sub-agents.
    active_subagents: Arc<Mutex<HashMap<String, SubAgentHandle>>>,
}

impl SubAgentMiddleware {
    /// Create a new `SubAgentMiddleware` with the given model and iteration limit.
    pub fn new(model: Arc<dyn BaseChatModel>, max_iterations: u32) -> Self {
        Self {
            model,
            max_iterations,
            active_subagents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the set of tools this middleware provides.
    pub fn tools(&self) -> Vec<Arc<dyn BaseTool>> {
        vec![Arc::new(SubAgentTool {
            model: self.model.clone(),
            max_iterations: self.max_iterations,
            active_subagents: self.active_subagents.clone(),
        })]
    }

    /// Get a snapshot of all sub-agent handles.
    pub async fn subagents(&self) -> HashMap<String, SubAgentHandle> {
        self.active_subagents.lock().await.clone()
    }
}

#[async_trait]
impl Middleware for SubAgentMiddleware {
    fn name(&self) -> &str {
        "subagent"
    }
}

// ---------------------------------------------------------------------------
// SubAgentTool
// ---------------------------------------------------------------------------

/// A tool that delegates a task to an isolated sub-agent.
///
/// Input schema:
/// ```json
/// { "task": "description of subtask", "context": "optional context" }
/// ```
///
/// The tool creates a new chat model invocation with the task, runs it, and
/// returns the sub-agent's response text.
pub struct SubAgentTool {
    model: Arc<dyn BaseChatModel>,
    max_iterations: u32,
    active_subagents: Arc<Mutex<HashMap<String, SubAgentHandle>>>,
}

#[async_trait]
impl BaseTool for SubAgentTool {
    fn name(&self) -> &str {
        "delegate_to_subagent"
    }

    fn description(&self) -> &str {
        "Delegate a task to an isolated sub-agent that will execute it and return the result"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the subtask to delegate"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to provide to the sub-agent"
                }
            },
            "required": ["task"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let task = extract_string_arg(&input, "task")?;
        let context = extract_string_arg(&input, "context").ok();

        let agent_id = uuid::Uuid::new_v4().to_string();

        // Register the sub-agent as running.
        {
            let mut agents = self.active_subagents.lock().await;
            agents.insert(
                agent_id.clone(),
                SubAgentHandle {
                    task: task.clone(),
                    status: SubAgentStatus::Running,
                    result: None,
                },
            );
        }

        // Build messages for the sub-agent.
        let mut messages: Vec<Message> = Vec::new();

        let system_content = format!(
            "You are a sub-agent. Complete the following task and return the result.\n\
             Maximum iterations: {}",
            self.max_iterations
        );
        messages.push(Message::System(cognis_core::messages::SystemMessage::new(
            &system_content,
        )));

        let mut user_content = task.clone();
        if let Some(ctx) = context {
            user_content = format!("{user_content}\n\nContext:\n{ctx}");
        }
        messages.push(Message::Human(HumanMessage::new(&user_content)));

        // Invoke the model.
        let result = self.model._generate(&messages, None).await;

        match result {
            Ok(chat_result) => {
                let response_text = chat_result
                    .generations
                    .first()
                    .map(|g| g.message.content().text())
                    .unwrap_or_default();

                // Mark completed.
                {
                    let mut agents = self.active_subagents.lock().await;
                    if let Some(handle) = agents.get_mut(&agent_id) {
                        handle.status = SubAgentStatus::Completed;
                        handle.result = Some(response_text.clone());
                    }
                }

                Ok(ToolOutput::Content(Value::String(response_text)))
            }
            Err(e) => {
                // Mark failed.
                {
                    let mut agents = self.active_subagents.lock().await;
                    if let Some(handle) = agents.get_mut(&agent_id) {
                        handle.status = SubAgentStatus::Failed;
                        handle.result = Some(format!("Error: {e}"));
                    }
                }

                Err(cognis_core::error::CognisError::ToolException(format!(
                    "Sub-agent failed: {e}"
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_string_arg(input: &ToolInput, key: &str) -> CoreResult<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => map
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                cognis_core::error::CognisError::ToolException(format!(
                    "Missing required argument: {key}"
                ))
            }),
        ToolInput::ToolCall(tc) => tc
            .args
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                cognis_core::error::CognisError::ToolException(format!(
                    "Missing required argument: {key}"
                ))
            }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::FakeMessagesListChatModel;
    use cognis_core::messages::AIMessage;

    #[tokio::test]
    async fn test_subagent_tool_runs_and_returns_result() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Sub-agent completed the task successfully"),
        )]));

        let tool = SubAgentTool {
            model,
            max_iterations: 5,
            active_subagents: Arc::new(Mutex::new(HashMap::new())),
        };

        let input = ToolInput::Structured({
            let mut m = HashMap::new();
            m.insert("task".into(), json!("Summarize the document"));
            m
        });

        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "Sub-agent completed the task successfully");
            }
            other => panic!("Expected ToolOutput::Content(String), got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_subagent_tool_with_context() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Done with context"),
        )]));

        let tool = SubAgentTool {
            model,
            max_iterations: 3,
            active_subagents: Arc::new(Mutex::new(HashMap::new())),
        };

        let input = ToolInput::Structured({
            let mut m = HashMap::new();
            m.insert("task".into(), json!("Analyze data"));
            m.insert(
                "context".into(),
                json!("The data is about weather patterns"),
            );
            m
        });

        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert_eq!(s, "Done with context");
            }
            other => panic!("Expected ToolOutput::Content(String), got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_subagent_middleware_provides_tool() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("result"),
        )]));

        let mw = SubAgentMiddleware::new(model, 10);
        let tools = mw.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "delegate_to_subagent");
    }

    #[tokio::test]
    async fn test_subagent_middleware_name() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("x"),
        )]));

        let mw = SubAgentMiddleware::new(model, 5);
        assert_eq!(mw.name(), "subagent");
    }

    #[tokio::test]
    async fn test_subagent_tracks_handle_status() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("done"),
        )]));

        let active = Arc::new(Mutex::new(HashMap::new()));
        let tool = SubAgentTool {
            model,
            max_iterations: 5,
            active_subagents: active.clone(),
        };

        let input = ToolInput::Structured({
            let mut m = HashMap::new();
            m.insert("task".into(), json!("Do something"));
            m
        });

        let _ = tool._run(input).await.unwrap();

        let agents = active.lock().await;
        assert_eq!(agents.len(), 1);
        let handle = agents.values().next().unwrap();
        assert_eq!(handle.status, SubAgentStatus::Completed);
        assert_eq!(handle.result.as_deref(), Some("done"));
        assert_eq!(handle.task, "Do something");
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_subagent_spec_serialization_roundtrip() {
        let spec = SubAgentSpec {
            name: "researcher".into(),
            description: "Searches for information".into(),
            system_prompt: "You are a research assistant.".into(),
            model: Some("openai:gpt-4o".into()),
            tools: vec!["web_search".into()],
            skills: vec![],
            interrupt_on: vec![],
            max_iterations: 10,
        };
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["name"], "researcher");
        assert_eq!(json["max_iterations"], 10);
        let deserialized: SubAgentSpec = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.name, "researcher");
    }

    #[test]
    fn test_subagent_spec_defaults() {
        let json = json!({
            "name": "helper",
            "description": "Helps",
            "system_prompt": "Be helpful."
        });
        let spec: SubAgentSpec = serde_json::from_value(json).unwrap();
        assert_eq!(spec.model, None);
        assert!(spec.tools.is_empty());
        assert_eq!(spec.max_iterations, 25);
    }

    #[tokio::test]
    async fn test_compiled_subagent_invoke() {
        let compiled = CompiledSubAgentSpec {
            name: "echo".into(),
            description: "Echoes input".into(),
            invoke: Box::new(|state| Box::pin(async move { Ok(state) })),
        };
        let input = json!({"messages": [{"type": "human", "content": "hi"}]});
        let result = (compiled.invoke)(input.clone()).await.unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_compiled_subagent_debug() {
        let compiled = CompiledSubAgentSpec {
            name: "test".into(),
            description: "test agent".into(),
            invoke: Box::new(|s| Box::pin(async move { Ok(s) })),
        };
        let debug = format!("{:?}", compiled);
        assert!(debug.contains("test"));
        assert!(debug.contains("<async fn>"));
    }
}
