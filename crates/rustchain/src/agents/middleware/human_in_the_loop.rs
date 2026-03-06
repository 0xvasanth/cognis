//! Human-in-the-loop (HITL) middleware types and middleware.
//!
//! Provides types for requesting human review of agent actions and a middleware
//! that intercepts tool calls for human approval.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::messages::{Message, MessageType};

use super::types::{AgentMiddleware, AgentState};

/// An action that the agent wants to take, presented for human review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// The name of the tool to call.
    pub name: String,
    /// The arguments for the tool call.
    pub args: Value,
}

/// A request for human review of an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    /// The proposed action.
    pub action: Action,
    /// A human-readable description of the action.
    pub description: Option<String>,
}

/// The full HITL request, grouping one or more action requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HITLRequest {
    /// The action requests that need human review.
    pub action_requests: Vec<ActionRequest>,
    /// A message to display to the human reviewer.
    pub message: Option<String>,
}

/// The human's decision about an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum Decision {
    /// Approve the action to proceed.
    Approve,
    /// Edit the action before proceeding, providing modified arguments.
    Edit {
        /// The edited action payload (typically modified tool call args).
        edited_action: Value,
    },
    /// Reject the action entirely.
    Reject {
        /// Optional rejection reason.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

/// The human's response to a HITL request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HITLResponse {
    /// The decision for each action request (parallel to action_requests).
    pub decisions: Vec<Decision>,
    /// Optional feedback message from the human.
    pub feedback: Option<String>,
}

/// What to interrupt on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptOn {
    /// Interrupt before any tool call.
    ToolCall,
    /// Interrupt before a specific tool call.
    SpecificTool(String),
    /// Interrupt before model response is used.
    ModelResponse,
}

/// Handler function for HITL interrupts.
///
/// Given a request for human review, returns the human's response.
/// In production, this might block waiting for user input, call an API, etc.
pub type InterruptHandler = Arc<dyn Fn(HITLRequest) -> Result<HITLResponse> + Send + Sync>;

/// Middleware that requests human review before certain actions proceed.
///
/// When the model produces tool calls that match the `interrupt_on` configuration,
/// this middleware creates an `HITLRequest` and calls the `interrupt_handler` (if set).
/// Based on the human's `Decision`:
/// - `Approve` — tool call proceeds unchanged
/// - `Edit` — tool call args are replaced with the edited values
/// - `Reject` — an error state is signaled
pub struct HumanInTheLoopMiddleware {
    /// Set of events that trigger a human review request.
    pub interrupt_on: HashMap<InterruptOn, bool>,
    /// Optional message template for the HITL request.
    pub message_template: Option<String>,
    /// Optional handler called when an interrupt is triggered.
    pub interrupt_handler: Option<InterruptHandler>,
}

impl HumanInTheLoopMiddleware {
    /// Create a new HITL middleware that interrupts on all tool calls.
    pub fn on_tool_calls() -> Self {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert(InterruptOn::ToolCall, true);
        Self {
            interrupt_on,
            message_template: None,
            interrupt_handler: None,
        }
    }

    /// Create a new HITL middleware with custom interrupt configuration.
    pub fn new(interrupt_on: HashMap<InterruptOn, bool>) -> Self {
        Self {
            interrupt_on,
            message_template: None,
            interrupt_handler: None,
        }
    }

    pub fn with_message_template(mut self, template: impl Into<String>) -> Self {
        self.message_template = Some(template.into());
        self
    }

    /// Set the interrupt handler that will be called when human review is needed.
    pub fn with_interrupt_handler(mut self, handler: InterruptHandler) -> Self {
        self.interrupt_handler = Some(handler);
        self
    }

    /// Add an interrupt trigger.
    pub fn on(mut self, trigger: InterruptOn) -> Self {
        self.interrupt_on.insert(trigger, true);
        self
    }

    /// Check if we should interrupt for a given event.
    pub fn should_interrupt(&self, event: &InterruptOn) -> bool {
        self.interrupt_on.get(event).copied().unwrap_or(false)
    }

    /// Check if we should interrupt for a specific tool name.
    fn should_interrupt_tool(&self, tool_name: &str) -> bool {
        self.should_interrupt(&InterruptOn::ToolCall)
            || self.should_interrupt(&InterruptOn::SpecificTool(tool_name.to_string()))
    }

    /// Create a HITL request for a tool call.
    pub fn create_tool_request(&self, tool_name: &str, tool_input: &Value) -> HITLRequest {
        HITLRequest {
            action_requests: vec![ActionRequest {
                action: Action {
                    name: tool_name.to_string(),
                    args: tool_input.clone(),
                },
                description: Some(format!("Call tool '{}'", tool_name)),
            }],
            message: self.message_template.clone().or_else(|| {
                Some(format!(
                    "The agent wants to call tool '{}'. Approve?",
                    tool_name
                ))
            }),
        }
    }
}

#[async_trait]
impl AgentMiddleware for HumanInTheLoopMiddleware {
    fn name(&self) -> &str {
        "HumanInTheLoopMiddleware"
    }

    async fn after_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        // Find the last AI message
        let last_ai = state
            .messages
            .iter()
            .rev()
            .find(|m| m.message_type() == MessageType::Ai);

        let last_ai = match last_ai {
            Some(msg) => msg,
            None => return Ok(None),
        };

        // Extract tool calls from the AI message
        let tool_calls = match last_ai {
            Message::Ai(ai_msg) => &ai_msg.tool_calls,
            _ => return Ok(None),
        };

        if tool_calls.is_empty() {
            return Ok(None);
        }

        // Build action requests for tool calls that match interrupt_on config
        let mut action_requests = Vec::new();
        let mut needs_review = false;

        for tc in tool_calls {
            if self.should_interrupt_tool(&tc.name) {
                needs_review = true;
                action_requests.push(ActionRequest {
                    action: Action {
                        name: tc.name.clone(),
                        args: serde_json::to_value(&tc.args).unwrap_or(Value::Null),
                    },
                    description: Some(format!("Call tool '{}'", tc.name)),
                });
            }
        }

        if !needs_review {
            return Ok(None);
        }

        // If we have an interrupt handler, call it and process the response
        if let Some(ref handler) = self.interrupt_handler {
            let request = HITLRequest {
                action_requests,
                message: self.message_template.clone().or_else(|| {
                    Some("The agent wants to execute tool calls. Please review.".to_string())
                }),
            };

            let response = handler(request)?;

            let mut updates = HashMap::new();
            let mut any_rejected = false;
            let mut rejection_message: Option<String> = None;

            // Process each decision
            for (i, decision) in response.decisions.iter().enumerate() {
                match decision {
                    Decision::Approve => {
                        // Pass through — no changes needed
                    }
                    Decision::Edit { edited_action } => {
                        // Store edited tool call args in state for the executor to pick up
                        updates.insert(format!("hitl_edit_{}", i), edited_action.clone());
                    }
                    Decision::Reject { message } => {
                        any_rejected = true;
                        if rejection_message.is_none() {
                            rejection_message = message.clone();
                        }
                    }
                }
            }

            if any_rejected {
                updates.insert("hitl_rejected".into(), serde_json::json!(true));
                if let Some(msg) = rejection_message {
                    updates.insert("hitl_rejection_message".into(), serde_json::json!(msg));
                }
            }

            updates.insert("hitl_pending".into(), serde_json::json!(false));
            updates.insert("hitl_resolved".into(), serde_json::json!(true));
            return Ok(Some(updates));
        }

        // No handler set — signal that review is needed via state flag
        let mut updates = HashMap::new();
        updates.insert("hitl_pending".into(), serde_json::json!(true));
        updates.insert(
            "hitl_request".into(),
            serde_json::to_value(&HITLRequest {
                action_requests,
                message: self.message_template.clone().or_else(|| {
                    Some("The agent wants to execute tool calls. Please review.".to_string())
                }),
            })?,
        );
        Ok(Some(updates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{tool_types::ToolCall, AIMessage};

    #[test]
    fn test_decision_serde() {
        let approve_json = serde_json::to_string(&Decision::Approve).unwrap();
        assert!(approve_json.contains("approve"));

        let edit = Decision::Edit {
            edited_action: serde_json::json!({"query": "new"}),
        };
        let edit_json = serde_json::to_string(&edit).unwrap();
        assert!(edit_json.contains("edit"));

        let reject = Decision::Reject {
            message: Some("too risky".into()),
        };
        let reject_json = serde_json::to_string(&reject).unwrap();
        let parsed: Decision = serde_json::from_str(&reject_json).unwrap();
        match parsed {
            Decision::Reject { message } => assert_eq!(message, Some("too risky".into())),
            _ => panic!("Expected Reject"),
        }
    }

    #[test]
    fn test_action_request_serde() {
        let req = ActionRequest {
            action: Action {
                name: "search".into(),
                args: serde_json::json!({"query": "test"}),
            },
            description: Some("Call search".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ActionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.action.name, "search");
    }

    #[test]
    fn test_hitl_request_creation() {
        let mw = HumanInTheLoopMiddleware::on_tool_calls();
        let req = mw.create_tool_request("search", &serde_json::json!({"q": "test"}));
        assert_eq!(req.action_requests.len(), 1);
        assert_eq!(req.action_requests[0].action.name, "search");
        assert!(req.message.is_some());
        assert!(req.message.unwrap().contains("search"));
    }

    #[test]
    fn test_hitl_should_interrupt() {
        let mw = HumanInTheLoopMiddleware::on_tool_calls()
            .on(InterruptOn::SpecificTool("dangerous".into()));
        assert!(mw.should_interrupt(&InterruptOn::ToolCall));
        assert!(mw.should_interrupt(&InterruptOn::SpecificTool("dangerous".into())));
        assert!(!mw.should_interrupt(&InterruptOn::ModelResponse));
    }

    #[test]
    fn test_hitl_middleware_name() {
        let mw = HumanInTheLoopMiddleware::on_tool_calls();
        assert_eq!(mw.name(), "HumanInTheLoopMiddleware");
    }

    #[test]
    fn test_hitl_response_serde() {
        let resp = HITLResponse {
            decisions: vec![
                Decision::Approve,
                Decision::Edit {
                    edited_action: serde_json::json!({"query": "modified"}),
                },
            ],
            feedback: Some("Looks good with edit".into()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HITLResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.decisions.len(), 2);
    }

    #[test]
    fn test_hitl_with_message_template() {
        let mw =
            HumanInTheLoopMiddleware::on_tool_calls().with_message_template("Please review: {}");
        assert_eq!(mw.message_template, Some("Please review: {}".into()));
    }

    #[test]
    fn test_hitl_with_interrupt_handler() {
        let handler: InterruptHandler = Arc::new(|_req| {
            Ok(HITLResponse {
                decisions: vec![Decision::Approve],
                feedback: None,
            })
        });
        let mw = HumanInTheLoopMiddleware::on_tool_calls().with_interrupt_handler(handler);
        assert!(mw.interrupt_handler.is_some());
    }

    #[test]
    fn test_interrupt_on_serde() {
        let trigger = InterruptOn::SpecificTool("search".into());
        let json = serde_json::to_string(&trigger).unwrap();
        let parsed: InterruptOn = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, InterruptOn::SpecificTool("search".into()));
    }

    #[tokio::test]
    async fn test_after_model_with_tool_calls_and_handler() {
        let handler: InterruptHandler = Arc::new(|req| {
            assert_eq!(req.action_requests.len(), 1);
            assert_eq!(req.action_requests[0].action.name, "search");
            Ok(HITLResponse {
                decisions: vec![Decision::Approve],
                feedback: None,
            })
        });
        let mw = HumanInTheLoopMiddleware::on_tool_calls().with_interrupt_handler(handler);

        let mut ai_msg = AIMessage::new("Let me search");
        ai_msg.tool_calls = vec![ToolCall {
            name: "search".into(),
            args: {
                let mut m = HashMap::new();
                m.insert("query".into(), serde_json::json!("test"));
                m
            },
            id: Some("tc-1".into()),
        }];

        let state = AgentState::new(vec![Message::Ai(ai_msg)]);
        let result = mw.after_model(&state).await.unwrap();
        assert!(result.is_some());
        let updates = result.unwrap();
        assert_eq!(updates.get("hitl_resolved"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_after_model_with_reject() {
        let handler: InterruptHandler = Arc::new(|_req| {
            Ok(HITLResponse {
                decisions: vec![Decision::Reject {
                    message: Some("Nope".into()),
                }],
                feedback: None,
            })
        });
        let mw = HumanInTheLoopMiddleware::on_tool_calls().with_interrupt_handler(handler);

        let mut ai_msg = AIMessage::new("");
        ai_msg.tool_calls = vec![ToolCall {
            name: "dangerous_tool".into(),
            args: HashMap::new(),
            id: Some("tc-2".into()),
        }];

        let state = AgentState::new(vec![Message::Ai(ai_msg)]);
        let result = mw.after_model(&state).await.unwrap();
        assert!(result.is_some());
        let updates = result.unwrap();
        assert_eq!(updates.get("hitl_rejected"), Some(&serde_json::json!(true)));
        assert_eq!(
            updates.get("hitl_rejection_message"),
            Some(&serde_json::json!("Nope"))
        );
    }

    #[tokio::test]
    async fn test_after_model_no_handler_sets_pending() {
        let mw = HumanInTheLoopMiddleware::on_tool_calls();

        let mut ai_msg = AIMessage::new("");
        ai_msg.tool_calls = vec![ToolCall {
            name: "search".into(),
            args: HashMap::new(),
            id: Some("tc-3".into()),
        }];

        let state = AgentState::new(vec![Message::Ai(ai_msg)]);
        let result = mw.after_model(&state).await.unwrap();
        assert!(result.is_some());
        let updates = result.unwrap();
        assert_eq!(updates.get("hitl_pending"), Some(&serde_json::json!(true)));
        assert!(updates.contains_key("hitl_request"));
    }

    #[tokio::test]
    async fn test_after_model_no_tool_calls() {
        let mw = HumanInTheLoopMiddleware::on_tool_calls();
        let state = AgentState::new(vec![Message::ai("No tools needed")]);
        let result = mw.after_model(&state).await.unwrap();
        assert!(result.is_none());
    }
}
