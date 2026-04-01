//! LangGraph error types.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error codes for LangGraph errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    GraphRecursionLimit,
    InvalidConcurrentGraphUpdate,
    InvalidGraphNodeReturnValue,
    MultipleSubgraphs,
    InvalidChatHistory,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GraphRecursionLimit => "GRAPH_RECURSION_LIMIT",
            Self::InvalidConcurrentGraphUpdate => "INVALID_CONCURRENT_GRAPH_UPDATE",
            Self::InvalidGraphNodeReturnValue => "INVALID_GRAPH_NODE_RETURN_VALUE",
            Self::MultipleSubgraphs => "MULTIPLE_SUBGRAPHS",
            Self::InvalidChatHistory => "INVALID_CHAT_HISTORY",
        }
    }
}

/// LangGraph error type.
#[derive(Debug, Error)]
pub enum LangGraphError {
    #[error("Graph recursion limit reached: {0}")]
    GraphRecursionError(String),

    #[error("Invalid update: {0}")]
    InvalidUpdateError(String),

    #[error("Empty channel: channel has not been updated yet")]
    EmptyChannelError,

    #[error("Graph interrupted")]
    GraphInterrupt(Vec<crate::types::Interrupt>),

    #[error("Empty input: {0}")]
    EmptyInputError(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Parent command")]
    ParentCommand(crate::types::Command),

    #[error("Graph bubble up")]
    GraphBubbleUp,

    #[error("Node interrupt: {0}")]
    NodeInterrupt(String),

    #[error("{0}")]
    Other(String),
}

impl From<cognis_core::error::CognisError> for LangGraphError {
    fn from(err: cognis_core::error::CognisError) -> Self {
        LangGraphError::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LangGraphError>;

/// Create an error message with a troubleshooting link.
pub fn create_error_message(message: &str, error_code: ErrorCode) -> String {
    format!(
        "{}\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/cognisgraph/errors/{}",
        message,
        error_code.as_str()
    )
}
