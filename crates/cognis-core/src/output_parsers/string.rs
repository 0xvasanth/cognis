//! Identity parser — passes the LLM output through unchanged.

use async_trait::async_trait;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Identity parser: returns the input string unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct StringParser;

impl StringParser {
    /// Construct a `StringParser`.
    pub fn new() -> Self {
        Self
    }
}

impl OutputParser<String> for StringParser {
    fn parse(&self, text: &str) -> Result<String> {
        Ok(text.to_string())
    }
}

#[async_trait]
impl Runnable<String, String> for StringParser {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<String> {
        Ok(input)
    }
    fn name(&self) -> &str {
        "StringParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn passes_through() {
        let p = StringParser::new();
        let out = p
            .invoke("hello".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "hello");
        assert_eq!(p.parse("x").unwrap(), "x");
    }
}
