//! Built-in tools shipped with `cognis`.
//!
//! All built-ins implement [`cognis_llm::Tool`]. They're meant to be
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
pub mod human;
pub mod json_query;
#[cfg(feature = "tools-http")]
pub mod openapi;
pub mod retriever;
pub mod shell;
pub mod subagent;
#[cfg(feature = "tools-http")]
pub mod web_search;
#[cfg(feature = "tools-http")]
pub mod wikipedia;

pub use approval::{AllowList, ApprovalGatedTool, Approver, AutoApprove, Decision, RejectAll};
pub use cached::CachedTool;
pub use calculator::Calculator;
pub use filesystem::{
    register_filesystem_tools, FileEditTool, FileExistsTool, FileGlobTool, FileGrepTool,
    FileListTool, FileReadTool, FileWriteTool,
};
// Re-export `HumanResponder` from the tool module under a distinct
// alias so it doesn't collide with the homonymous middleware trait.
#[cfg(feature = "tools-http")]
pub use http::{HttpMethod, HttpRequest};
pub use human::{HumanResponder as ToolHumanResponder, HumanTool, StaticResponder};
pub use json_query::{DotPathEngine, JsonQueryTool, QueryEngine};
#[cfg(feature = "tools-http")]
pub use openapi::{AuthScheme, BearerAuth, HeaderAuth, OpenApiToolset};
pub use retriever::RetrieverTool;
pub use shell::ShellTool;
pub use subagent::SubAgentTool;
#[cfg(feature = "tools-http")]
pub use web_search::{
    TavilyProvider, TavilyProviderBuilder, WebSearchInput, WebSearchProvider, WebSearchResult,
    WebSearchTool,
};
#[cfg(feature = "tools-http")]
pub use wikipedia::{WikipediaAction, WikipediaTool, WikipediaToolBuilder};
