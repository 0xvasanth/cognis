//! Built-in tools shipped with `cognis2`.
//!
//! All built-ins implement [`cognis2_llm::Tool`]. They're meant to be
//! drop-in for common agent needs:
//!
//! - [`Calculator`] — pure-Rust math expression evaluator.
//! - [`HttpRequest`] — HTTP GET/POST with SSRF guards (feature `tools-http`).
//! - filesystem tools (read/write/edit/ls/glob/grep) wired to a [`Backend`].
//! - [`ShellTool`] — sandboxed shell with command allowlist.
//! - [`SubAgentTool`] — wraps an [`Agent`](crate::Agent) as a callable tool.
//! - [`ApprovalGatedTool`] — wraps any tool with a human-approval check.

pub mod approval;
pub mod cached;
pub mod calculator;
pub mod filesystem;
#[cfg(feature = "tools-http")]
pub mod http;
pub mod retriever;
pub mod shell;
pub mod subagent;

pub use approval::{AllowList, ApprovalGatedTool, Approver, AutoApprove, Decision, RejectAll};
pub use cached::CachedTool;
pub use calculator::Calculator;
pub use filesystem::{
    register_filesystem_tools, FileEditTool, FileExistsTool, FileGlobTool, FileGrepTool,
    FileListTool, FileReadTool, FileWriteTool,
};
#[cfg(feature = "tools-http")]
pub use http::{HttpMethod, HttpRequest};
pub use retriever::RetrieverTool;
pub use shell::ShellTool;
pub use subagent::SubAgentTool;
