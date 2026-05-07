//! Boolean parser — yes/no, true/false, 1/0.

use async_trait::async_trait;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Parses common yes/no representations into `bool`. Case-insensitive,
/// whitespace tolerant. Anything else errors with
/// `CognisError::Serialization`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BooleanParser;

impl BooleanParser {
    /// Construct a `BooleanParser`.
    pub fn new() -> Self {
        Self
    }
}

impl OutputParser<bool> for BooleanParser {
    fn parse(&self, text: &str) -> Result<bool> {
        let t = text.trim().to_lowercase();
        match t.as_str() {
            "yes" | "true" | "y" | "1" => Ok(true),
            "no" | "false" | "n" | "0" => Ok(false),
            other => Err(CognisError::Serialization(format!(
                "BooleanParser: cannot parse `{other}` as bool"
            ))),
        }
    }
    fn format_instructions(&self) -> Option<String> {
        Some("Reply with exactly one word: `yes` or `no`.".into())
    }
}

#[async_trait]
impl Runnable<String, bool> for BooleanParser {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<bool> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "BooleanParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_truthy() {
        let p = BooleanParser::new();
        for s in ["yes", "YES", "true", "TRUE", "y", "1", " yes "] {
            assert!(p.parse(s).unwrap(), "expected true for {s}");
        }
    }

    #[test]
    fn parses_falsy() {
        let p = BooleanParser::new();
        for s in ["no", "NO", "false", "n", "0"] {
            assert!(!p.parse(s).unwrap(), "expected false for {s}");
        }
    }

    #[test]
    fn invalid_errors() {
        let p = BooleanParser::new();
        assert!(p.parse("maybe").is_err());
    }
}
