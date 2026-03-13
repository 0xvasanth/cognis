use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Parses comma-separated values from LLM output.
pub struct CommaSeparatedListOutputParser;

impl OutputParser for CommaSeparatedListOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let items: Vec<Value> = text
            .split(',')
            .map(|s| Value::String(s.trim().to_string()))
            .filter(|v| v.as_str() != Some(""))
            .collect();
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a list of comma separated values, \
             eg: `foo, bar, baz`"
                .into(),
        )
    }

    fn parser_type(&self) -> &str {
        "comma_separated_list"
    }
}

#[async_trait]
impl Runnable for CommaSeparatedListOutputParser {
    fn name(&self) -> &str {
        "CommaSeparatedListOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}

/// Parses numbered list items from LLM output (e.g., "1. foo\n2. bar").
pub struct NumberedListOutputParser;

impl OutputParser for NumberedListOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let items: Vec<Value> = text
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                // Match patterns like "1. item" or "1) item"
                let rest = trimmed
                    .strip_prefix(|c: char| c.is_ascii_digit())
                    .and_then(|s| {
                        // Skip remaining digits
                        let s = s.trim_start_matches(|c: char| c.is_ascii_digit());
                        s.strip_prefix(". ").or_else(|| s.strip_prefix(") "))
                    });
                rest.map(|s| Value::String(s.trim().to_string()))
            })
            .collect();
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a numbered list, eg:\n\
             1. foo\n\
             2. bar\n\
             3. baz"
                .into(),
        )
    }

    fn parser_type(&self) -> &str {
        "numbered_list"
    }
}

#[async_trait]
impl Runnable for NumberedListOutputParser {
    fn name(&self) -> &str {
        "NumberedListOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}

/// Parses markdown list items (lines starting with `- ` or `* `).
pub struct MarkdownListOutputParser;

impl OutputParser for MarkdownListOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let items: Vec<Value> = text
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("- ")
                    .or_else(|| trimmed.strip_prefix("* "))
                    .map(|s| Value::String(s.trim().to_string()))
            })
            .collect();
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a markdown list, eg:\n\
             - foo\n\
             - bar\n\
             - baz"
                .into(),
        )
    }

    fn parser_type(&self) -> &str {
        "markdown_list"
    }
}

#[async_trait]
impl Runnable for MarkdownListOutputParser {
    fn name(&self) -> &str {
        "MarkdownListOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
