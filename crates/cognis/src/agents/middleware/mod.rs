//! Agent middleware system.
//!
//! Provides middleware hooks for intercepting and modifying agent behavior
//! at various points in the execution loop.

pub mod context_editing;
pub mod execution;
pub mod file_search;
pub mod human_in_the_loop;
pub mod model_call_limit;
pub mod model_fallback;
pub mod model_retry;
pub mod pii;
pub mod redaction;
pub mod retry;
pub mod shell_tool;
pub mod summarization;
pub mod todo;
pub mod tool_call_limit;
pub mod tool_emulator;
pub mod tool_retry;
pub mod tool_selection;
pub mod types;

pub use types::*;

// Re-export key types from sub-modules for convenience.
pub use context_editing::{ClearToolUsesEdit, ContextEdit, ContextEditingMiddleware};
pub use execution::{
    CodexSandboxExecutionPolicy, DockerExecutionPolicy, ExecutionPolicy, ExecutionPolicyConfig,
    ExecutionPolicyKind, HostExecutionPolicy,
};
pub use file_search::FilesystemFileSearchMiddleware;
pub use human_in_the_loop::{Decision, HITLRequest, HITLResponse, HumanInTheLoopMiddleware};
pub use model_call_limit::{ExitBehavior, ModelCallLimitMiddleware};
pub use model_fallback::ModelFallbackMiddleware;
pub use model_retry::ModelRetryMiddleware;
pub use pii::PIIMiddleware;
pub use redaction::{PIIMatch, RedactionStrategy};
pub use shell_tool::{CommandExecutionResult, ShellToolMiddleware};
pub use summarization::SummarizationMiddleware;
pub use todo::{Todo, TodoListMiddleware, TodoStatus};
pub use tool_call_limit::ToolCallLimitMiddleware;
pub use tool_emulator::LLMToolEmulator;
pub use tool_retry::{OnToolFailure, ToolRetryMiddleware};
pub use tool_selection::LLMToolSelectorMiddleware;
