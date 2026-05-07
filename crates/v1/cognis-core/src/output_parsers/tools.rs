use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{CognisError, Result};
use crate::outputs::ChatGeneration;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Extracts tool calls from an AIMessage in ChatGeneration results.
pub struct ToolCallOutputParser {
    /// If true, returns only the first tool call instead of an array.
    pub first_tool_only: bool,
    /// If true, includes the tool call ID in the output.
    pub return_id: bool,
}

impl ToolCallOutputParser {
    pub fn new() -> Self {
        Self {
            first_tool_only: false,
            return_id: true,
        }
    }

    pub fn first_only(mut self) -> Self {
        self.first_tool_only = true;
        self
    }

    pub fn without_id(mut self) -> Self {
        self.return_id = false;
        self
    }

    /// Parse tool calls from a ChatGeneration.
    pub fn parse_chat_generation(&self, generation: &ChatGeneration) -> Result<Value> {
        let tool_calls = match &generation.message {
            crate::messages::Message::Ai(ai_msg) => &ai_msg.tool_calls,
            _ => {
                return if self.first_tool_only {
                    Err(CognisError::OutputParserError {
                        message: "Expected AIMessage in ChatGeneration for tool call parsing"
                            .into(),
                        observation: None,
                        llm_output: Some(generation.text.clone()),
                    })
                } else {
                    Ok(json!([]))
                }
            }
        };
        if tool_calls.is_empty() {
            return if self.first_tool_only {
                Err(CognisError::OutputParserError {
                    message: "No tool calls found in AIMessage".into(),
                    observation: None,
                    llm_output: Some(generation.text.clone()),
                })
            } else {
                Ok(json!([]))
            };
        }

        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                let mut obj = json!({
                    "type": tc.name,
                    "args": tc.args,
                });
                if self.return_id {
                    if let Some(id) = &tc.id {
                        obj.as_object_mut().unwrap().insert("id".into(), json!(id));
                    }
                }
                obj
            })
            .collect();

        if self.first_tool_only {
            Ok(calls.into_iter().next().unwrap())
        } else {
            Ok(Value::Array(calls))
        }
    }
}

impl Default for ToolCallOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for ToolCallOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // Tool calls aren't in text — this parser works from ChatGeneration
        Err(CognisError::OutputParserError {
            message: "ToolCallOutputParser requires ChatGeneration, not raw text. \
                      Use parse_chat_generation() instead."
                .into(),
            observation: None,
            llm_output: Some(text.to_string()),
        })
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "tool_call_output_parser"
    }
}

#[async_trait]
impl Runnable for ToolCallOutputParser {
    fn name(&self) -> &str {
        "ToolCallOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        // Try to deserialize as a ChatGeneration or AIMessage
        if let Ok(gen) = serde_json::from_value::<ChatGeneration>(input.clone()) {
            return self.parse_chat_generation(&gen);
        }

        // Try as AIMessage directly
        if let Ok(ai_msg) = serde_json::from_value::<crate::messages::AIMessage>(input) {
            let gen = ChatGeneration::new(ai_msg);
            return self.parse_chat_generation(&gen);
        }

        Err(CognisError::TypeMismatch {
            expected: "ChatGeneration or AIMessage".into(),
            got: "unrecognized input".into(),
        })
    }
}
