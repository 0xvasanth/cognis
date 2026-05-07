//! Typed parsers that convert LLM string output into structured values.
//!
//! Each parser implements:
//! - [`OutputParser<T>`] — for direct parsing.
//! - `Runnable<String, T>` — for chain composition (`prompt | model | parser`).
//!
//! Parsers also expose a `format_instructions()` snippet that can be embedded
//! in a prompt to tell the LLM how to format its output.

pub mod boolean;
pub mod json;
pub mod list;
pub mod string;
pub mod structured;
pub mod xml;

pub use boolean::BooleanParser;
pub use json::JsonParser;
pub use list::{CommaListParser, NumberedListParser};
pub use string::StringParser;
pub use structured::{
    JsonExtraction, JsonExtractor, StructuredOutputConfig, StructuredOutputParser,
};
pub use xml::XmlParser;

use crate::Result;

/// Parses LLM string output into a typed value `T`.
pub trait OutputParser<T>: Send + Sync
where
    T: Send + 'static,
{
    /// Attempt to parse the LLM output.
    fn parse(&self, text: &str) -> Result<T>;

    /// Optional human-readable instructions to embed in a prompt to tell
    /// the LLM how to format its output.
    fn format_instructions(&self) -> Option<String> {
        None
    }
}
