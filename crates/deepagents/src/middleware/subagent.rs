//! Sub-agent middleware — delegates tasks to isolated sub-agents.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use rustchain_core::error::Result as CoreResult;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};

use crate::middleware::Middleware;

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
        messages.push(Message::System(
            rustchain_core::messages::SystemMessage::new(&system_content),
        ));

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

                Err(rustchain_core::error::RustChainError::ToolException(
                    format!("Sub-agent failed: {e}"),
                ))
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
                rustchain_core::error::RustChainError::ToolException(format!(
                    "Missing required argument: {key}"
                ))
            }),
        ToolInput::ToolCall(tc) => tc
            .args
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                rustchain_core::error::RustChainError::ToolException(format!(
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
    use rustchain_core::language_models::fake::FakeMessagesListChatModel;
    use rustchain_core::messages::AIMessage;

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
