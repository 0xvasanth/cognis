/// Concrete tool implementations for use with the agent executor.
pub mod cached;
pub mod calculator;
pub mod json_query;
pub mod openapi;
pub mod shell;

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
pub use shell::ShellTool;

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
