use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::messages::Message;

/// Represents a request to execute an action by an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentAction {
    pub tool: String,
    pub tool_input: Value,
    pub log: String,
}

impl AgentAction {
    pub fn new(tool: impl Into<String>, tool_input: Value, log: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            tool_input,
            log: log.into(),
        }
    }
}

/// Final return value of an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentFinish {
    pub return_values: HashMap<String, Value>,
    pub log: String,
}

impl AgentFinish {
    pub fn new(return_values: HashMap<String, Value>, log: impl Into<String>) -> Self {
        Self {
            return_values,
            log: log.into(),
        }
    }
}

/// Result of running an AgentAction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentStep {
    pub action: AgentAction,
    pub observation: String,
}

impl AgentStep {
    pub fn new(action: AgentAction, observation: impl Into<String>) -> Self {
        Self {
            action,
            observation: observation.into(),
        }
    }
}

/// An agent action that includes a message log for more detailed tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentActionMessageLog {
    pub tool: String,
    pub tool_input: Value,
    pub log: String,
    pub message_log: Vec<Message>,
}

impl AgentActionMessageLog {
    pub fn new(
        tool: impl Into<String>,
        tool_input: Value,
        log: impl Into<String>,
        message_log: Vec<Message>,
    ) -> Self {
        Self {
            tool: tool.into(),
            tool_input,
            log: log.into(),
            message_log,
        }
    }
}
