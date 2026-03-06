//! Output parsers with LLM-based error correction.
//!
//! This module provides output parsers that can automatically fix malformed
//! LLM output by sending it back to a language model for correction:
//!
//! - [`OutputFixingParser`] -- wraps an inner parser and uses an LLM to fix
//!   malformed output on parse failure.
//! - [`RetryOutputParser`] -- retries parsing up to N times, feeding errors
//!   back to the LLM for correction.
//! - [`StructuredOutputParser`] -- parses JSON and validates against a JSON
//!   schema with detailed field-level error messages.

mod fixing;
mod retry;
mod structured;

pub use fixing::OutputFixingParser;
pub use retry::RetryOutputParser;
pub use structured::StructuredOutputParser;
