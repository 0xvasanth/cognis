//! Agent module providing middleware, tool-calling agents, structured output support,
//! and output parsers for converting raw LLM text into structured agent actions.
//!
//! Output parsers for common formats:
//! - [`ReActOutputParser`] — parses `Thought: / Action: / Action Input:` and `Final Answer:` blocks
//! - [`JsonOutputParser`] — parses `{"action": "tool", "action_input": {...}}` JSON
//! - [`XmlOutputParser`] — parses `<tool>name</tool><tool_input>input</tool_input>` XML
//! - [`ToolCallOutputParser`] — converts structured tool-call objects from chat models
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
