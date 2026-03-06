//! Agent module providing middleware, tool-calling agents, and structured output support.
//!
//! Mirrors Python `langchain.agents`.

pub mod executor;
pub mod middleware;
pub mod structured_output;
pub mod tool_calling;

pub use middleware::types::{
    AgentMiddleware, AgentState, JumpTo, ModelCallResult, ModelRequest, ModelResponse,
};
pub use structured_output::{
    AutoStrategy, ErrorHandling, OutputToolBinding, ProviderStrategy, ProviderStrategyBinding,
    ResponseFormat, SchemaKind, SchemaSpec, StructuredOutputError,
    StructuredOutputValidationError, MultipleStructuredOutputsError, ToolStrategy,
};
pub use executor::{
    AgentAction, AgentExecutor, AgentExecutorBuilder, AgentResult, AgentStep,
    EarlyStoppingMethod,
};
pub use tool_calling::{
    AgentOutput, format_to_tool_messages, parse_ai_message_to_agent_output,
};
