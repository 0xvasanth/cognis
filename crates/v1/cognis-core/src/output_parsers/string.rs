use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Trivial parser that returns the text output unchanged.
pub struct StrOutputParser;

impl OutputParser for StrOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        Ok(Value::String(text.to_string()))
    }

    fn parser_type(&self) -> &str {
        "str_output_parser"
    }
}

#[async_trait]
impl Runnable for StrOutputParser {
    fn name(&self) -> &str {
        "StrOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
