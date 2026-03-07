//! Agent module providing middleware, tool-calling agents, and structured output support.
//!
//! Mirrors Python `langchain.agents`.

pub mod executor;
pub mod middleware;
pub mod output_parser;
pub mod structured_output;
pub mod tool_calling;

pub use executor::{
    AgentAction, AgentExecutor, AgentExecutorBuilder, AgentResult, AgentStep, EarlyStoppingMethod,
};
pub use middleware::types::{
    AgentMiddleware, AgentState, JumpTo, ModelCallResult, ModelRequest, ModelResponse,
};
pub use output_parser::{
    AgentOutputParser, JsonOutputParser, ReActOutputParser, ToolCallOutputParser, XmlOutputParser,
};
pub use structured_output::{
    AutoStrategy, ErrorHandling, MultipleStructuredOutputsError, OutputToolBinding,
    ProviderStrategy, ProviderStrategyBinding, ResponseFormat, SchemaKind, SchemaSpec,
    StructuredOutputError, StructuredOutputValidationError, ToolStrategy,
};
pub use tool_calling::{format_to_tool_messages, parse_ai_message_to_agent_output, AgentOutput};
