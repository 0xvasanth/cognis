//! Output parsers for OpenAI-style tool calls.
//!
//! Mirrors Python `langchain_core.output_parsers.openai_tools`.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{Result, CognisError};
use crate::outputs::Generation;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Parse a single raw tool call dict into a structured form.
///
/// Expects `{"function": {"name": "...", "arguments": "..."}, "id": "..."}`.
pub fn parse_tool_call(raw: &Value, return_id: bool) -> Result<Value> {
    let function = raw
        .get("function")
        .ok_or_else(|| CognisError::OutputParserError {
            message: "Tool call missing 'function' key".into(),
            observation: Some(raw.to_string()),
            llm_output: None,
        })?;

    let name = function
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let args_str = function
        .get("arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");

    let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

    let mut result = json!({
        "type": name,
        "args": args,
    });

    if return_id {
        if let Some(id) = raw.get("id") {
            result["id"] = id.clone();
        }
    }

    Ok(result)
}

/// Parse a list of raw tool calls.
pub fn parse_tool_calls(raw_calls: &[Value], return_id: bool) -> Result<Vec<Value>> {
    raw_calls
        .iter()
        .map(|raw| parse_tool_call(raw, return_id))
        .collect()
}

/// Parses tool calls from OpenAI-style chat completions.
///
/// Expects the input to be a ChatGeneration or AIMessage containing
/// `tool_calls` in `additional_kwargs`.
pub struct OpenAIToolsOutputParser {
    /// If true, only return the first tool call.
    pub first_tool_only: bool,
    /// If true, include the tool call ID in the output.
    pub return_id: bool,
}

impl OpenAIToolsOutputParser {
    pub fn new() -> Self {
        Self {
            first_tool_only: false,
            return_id: false,
        }
    }

    pub fn first_only(mut self) -> Self {
        self.first_tool_only = true;
        self
    }

    pub fn with_id(mut self) -> Self {
        self.return_id = true;
        self
    }
}

impl Default for OpenAIToolsOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for OpenAIToolsOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // Try to parse as JSON containing tool_calls
        let parsed: Value =
            serde_json::from_str(text).map_err(|e| CognisError::OutputParserError {
                message: format!("Failed to parse tool calls JSON: {}", e),
                observation: Some(text.to_string()),
                llm_output: None,
            })?;

        self.extract_tool_calls(&parsed)
    }

    fn parse_result(&self, result: &[Generation], _partial: bool) -> Result<Value> {
        if result.is_empty() {
            return Err(CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }

        let gen = &result[0];

        // Check for tool_calls in generation_info
        if let Some(info) = &gen.generation_info {
            if let Some(tool_calls) = info.get("tool_calls") {
                return self.extract_from_array(tool_calls);
            }
        }

        // Try parsing the text as JSON
        self.parse(&gen.text)
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "openai_tools"
    }
}

impl OpenAIToolsOutputParser {
    /// Extract tool calls from a JSON value, searching in various locations.
    pub fn extract_tool_calls(&self, value: &Value) -> Result<Value> {
        // Look for tool_calls in various locations
        let tool_calls = value
            .get("tool_calls")
            .or_else(|| {
                value
                    .get("additional_kwargs")
                    .and_then(|ak| ak.get("tool_calls"))
            })
            .or_else(|| value.get("message").and_then(|m| m.get("tool_calls")));

        match tool_calls {
            Some(calls) => self.extract_from_array(calls),
            None => {
                // If the value itself is an array of tool calls
                if value.is_array() {
                    self.extract_from_array(value)
                } else {
                    Err(CognisError::OutputParserError {
                        message: "No tool_calls found in output".into(),
                        observation: Some(value.to_string()),
                        llm_output: None,
                    })
                }
            }
        }
    }

    /// Extract and parse tool calls from a JSON array value.
    pub fn extract_from_array(&self, calls: &Value) -> Result<Value> {
        let arr = calls
            .as_array()
            .ok_or_else(|| CognisError::OutputParserError {
                message: "tool_calls is not an array".into(),
                observation: Some(calls.to_string()),
                llm_output: None,
            })?;

        let parsed = parse_tool_calls(arr, self.return_id)?;

        if self.first_tool_only {
            Ok(parsed.into_iter().next().unwrap_or(Value::Null))
        } else {
            Ok(Value::Array(parsed))
        }
    }
}

#[async_trait]
impl Runnable for OpenAIToolsOutputParser {
    fn name(&self) -> &str {
        "OpenAIToolsOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        match &input {
            Value::String(s) => self.parse(s),
            other => self.extract_tool_calls(other),
        }
    }
}

/// Parses tool calls and returns a specific key from the arguments.
pub struct JsonOutputKeyToolsParser {
    pub key_name: String,
    pub return_id: bool,
    pub first_tool_only: bool,
}

impl JsonOutputKeyToolsParser {
    pub fn new(key_name: impl Into<String>) -> Self {
        Self {
            key_name: key_name.into(),
            return_id: false,
            first_tool_only: false,
        }
    }
}

impl OutputParser for JsonOutputKeyToolsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let parsed: Value =
            serde_json::from_str(text).map_err(|e| CognisError::OutputParserError {
                message: format!("Failed to parse: {}", e),
                observation: Some(text.to_string()),
                llm_output: None,
            })?;

        let calls = parsed
            .as_array()
            .ok_or_else(|| CognisError::OutputParserError {
                message: "Expected array of tool calls".into(),
                observation: None,
                llm_output: None,
            })?;

        let extracted: Vec<Value> = calls
            .iter()
            .filter_map(|call| {
                call.get("args")
                    .and_then(|args| args.get(&self.key_name))
                    .cloned()
            })
            .collect();

        if self.first_tool_only {
            Ok(extracted.into_iter().next().unwrap_or(Value::Null))
        } else {
            Ok(Value::Array(extracted))
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "json_output_key_tools"
    }
}

#[async_trait]
impl Runnable for JsonOutputKeyToolsParser {
    fn name(&self) -> &str {
        "JsonOutputKeyToolsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
