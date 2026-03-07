//! Output parsers with LLM-based error correction and structured extraction.
//!
//! This module provides output parsers that can automatically fix malformed
//! LLM output by sending it back to a language model for correction:
//!
//! - [`OutputFixingParser`] -- wraps an inner parser and uses an LLM to fix
//!   malformed output on parse failure.
//! - [`RetryOutputParser`] -- retries parsing up to N times, feeding errors
//!   back to the LLM for correction.
//!
//! And structured output parsers for extracting typed data from LLM text:
//!
//! - [`StructuredOutputParser`] -- validates parsed JSON against a JSON schema.
//! - [`JsonOutputParser`] -- extracts JSON from text (handles code fences).
//! - [`MarkdownListParser`] -- parses markdown lists into a JSON array.
//! - [`KeyValueParser`] -- parses `key: value` lines into a JSON object.
//! - [`RegexParser`] -- extracts named regex groups into a JSON object.
//! - [`CommaSeparatedListParser`] -- parses CSV values into a JSON array.
//! - [`BooleanParser`] -- parses yes/no/true/false/1/0 into a boolean.
//! - [`EnumParser`] -- validates output is one of allowed values.
//! - [`CombiningParser`] -- tries multiple parsers, returns first success.

mod fixing;
mod retry;
pub mod structured;

pub use fixing::OutputFixingParser;
pub use retry::RetryOutputParser;
pub use structured::{
    BooleanParser, CombiningParser, CommaSeparatedListParser, EnumParser, JsonOutputParser,
    KeyValueParser, MarkdownListParser, RegexParser, StructuredOutputParser,
};
