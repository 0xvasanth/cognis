use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::outputs::Generation;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Parses JSON from LLM output, stripping markdown fences if present.
pub struct JsonOutputParser {
    /// Optional JSON schema for format instructions.
    pub schema: Option<Value>,
}

impl JsonOutputParser {
    pub fn new() -> Self {
        Self { schema: None }
    }

    pub fn with_schema(schema: Value) -> Self {
        Self {
            schema: Some(schema),
        }
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip markdown code fences from JSON output.
fn parse_json_markdown(text: &str) -> Result<Value> {
    let trimmed = text.trim();

    // Try to extract from ```json ... ``` blocks
    let json_str = if trimmed.starts_with("```") {
        let after_fence = if let Some(rest) = trimmed.strip_prefix("```json") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```JSON") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```") {
            rest
        } else {
            trimmed
        };

        after_fence
            .trim()
            .strip_suffix("```")
            .unwrap_or(after_fence)
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str(json_str).map_err(|e| CognisError::OutputParserError {
        message: format!("Failed to parse JSON: {}", e),
        observation: Some(json_str.to_string()),
        llm_output: Some(text.to_string()),
    })
}

impl OutputParser for JsonOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        parse_json_markdown(text)
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        if result.is_empty() {
            return Err(CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }
        match parse_json_markdown(&result[0].text) {
            Ok(v) => Ok(v),
            Err(_) if partial => Ok(Value::Null),
            Err(e) => Err(e),
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        match &self.schema {
            Some(schema) => Some(format!(
                "Return a JSON object that conforms to the following schema:\n```json\n{}\n```\n\
                 Return ONLY the JSON object, with no additional text or markdown formatting.",
                serde_json::to_string_pretty(schema).unwrap_or_default()
            )),
            None => Some(
                "Return a JSON object. Return ONLY the JSON, with no additional text or markdown formatting."
                    .into(),
            ),
        }
    }

    fn parser_type(&self) -> &str {
        "json_output_parser"
    }
}

#[async_trait]
impl Runnable for JsonOutputParser {
    fn name(&self) -> &str {
        "JsonOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
