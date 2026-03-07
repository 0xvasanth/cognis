//! Concrete tool implementations for use with the agent executor.
//!
//! This module provides ready-to-use tools that can be plugged into agents:
//!
//! - [`calculator`] -- Arithmetic expression evaluation.
//! - [`shell`] -- Shell command execution.
//! - [`web_search`] -- Web search via DuckDuckGo (requires provider feature).
//! - [`wikipedia`] -- Wikipedia article lookup (requires provider feature).
//! - [`json_query`] -- JSON path querying.
//! - [`openapi`] -- OpenAPI spec-driven HTTP tool generation.
//! - [`cached`] -- Cache wrapper for any tool with TTL and stats.
//! - [`validation`] -- Tool call validation and auto-correction.
//! - [`python_repl`] -- Mock Python REPL with code sanitization.
//! - [`retriever`] -- Retriever-backed document lookup tool.
//! - [`human`] -- Human input tool for interactive prompts, approval wrapper, and toolkit system.
//! - [`requests`] -- HTTP request tools (GET/POST) with domain allowlisting and mock client.
//! - [`file_management`] -- File management tools (read, write, copy, move, list, delete) with path safety and toolkit grouping.
pub mod cached;
pub mod calculator;
pub mod file_management;
pub mod human;
pub mod json_query;
pub mod openapi;
pub mod python_repl;
pub mod requests;
pub mod retriever;
pub mod shell;
pub mod validation;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub mod web_search;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub mod wikipedia;

pub use cached::{CacheEntry, CacheStats, CachedTool};
pub use calculator::CalculatorTool;
pub use json_query::JsonQueryTool;
pub use openapi::{
    generate_tools, DryRunExecutor, HttpExecutor, OpenAPISpec, OpenAPITool, OpenAPIToolkit,
    OperationInfo, ParameterInfo,
};

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub use openapi::ReqwestExecutor;
pub use python_repl::{
    CodeSanitizer, MockPythonREPL, PythonREPLConfig, PythonREPLConfigBuilder, PythonREPLResult,
    PythonREPLTool, SanitizationError,
};
pub use requests::{
    requests_get_tool, requests_post_tool, HttpClient, HttpMethod, HttpResponse, MockHttpClient,
    RequestConfig, RequestConfigBuilder, RequestsTool, TextExtractor,
};
pub use retriever::{
    create_retriever_tool, DocumentFormatter, MultiRetrieverTool, RetrieverTool,
    RetrieverToolBuilder, RoutingStrategy,
};
pub use shell::ShellTool;
pub use validation::{
    StrictnessMode, ToolCallCorrector, ToolCallValidator, ValidatedToolExecutor, ValidationError,
    ValidationResult, ValidationSchemaBuilder,
};

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub use web_search::{DuckDuckGoSearchTool, WebSearchTool};

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub use wikipedia::WikipediaTool;
