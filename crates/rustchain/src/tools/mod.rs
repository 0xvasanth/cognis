/// Concrete tool implementations for use with the agent executor.
pub mod cached;
pub mod calculator;
pub mod json_query;
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

pub use cached::{CachedTool, CacheEntry, CacheStats};
pub use calculator::CalculatorTool;
pub use json_query::JsonQueryTool;
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
