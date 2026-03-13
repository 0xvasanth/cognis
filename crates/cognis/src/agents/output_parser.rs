//! Agent output parsers for converting raw LLM text into structured agent actions.
//!
//! Provides parsers for common output formats:
//! - **ReAct** — `Thought: ... / Action: ... / Action Input: ...` and `Final Answer: ...`
//! - **JSON** — `{"action": "tool", "action_input": {...}}`
//! - **XML** — `<tool>name</tool><tool_input>input</tool_input>`
//! - **ToolCall** — Structured tool-call objects from chat models

use std::collections::HashMap;

use regex::Regex;
use serde_json::Value;

use cognis_core::agents::{AgentAction, AgentFinish};
use cognis_core::error::{Result, CognisError};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The result of parsing a single LLM output: either an action to take or
/// a final answer.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentOutput {
    /// The agent wants to invoke a tool.
    Action(AgentAction),
    /// The agent has produced a final answer.
    Finish(AgentFinish),
}

/// Trait for parsing raw LLM text into an [`AgentOutput`].
pub trait AgentOutputParser: Send + Sync {
    /// Parse the given text into an agent action or finish signal.
    fn parse(&self, text: &str) -> Result<AgentOutput>;
}

// ---------------------------------------------------------------------------
// ReAct output parser
// ---------------------------------------------------------------------------

/// Parses the ReAct format produced by many LLMs:
///
/// ```text
/// Thought: I need to search for ...
/// Action: search
/// Action Input: {"query": "rust"}
/// ```
///
/// Or a final answer:
///
/// ```text
/// Final Answer: The answer is 42
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReActOutputParser;

impl ReActOutputParser {
    pub fn new() -> Self {
        Self
    }
}

impl AgentOutputParser for ReActOutputParser {
    fn parse(&self, text: &str) -> Result<AgentOutput> {
        // Check for Final Answer first
        if let Some(idx) = text.find("Final Answer:") {
            let answer = text[idx + "Final Answer:".len()..].trim().to_string();
            let mut return_values = HashMap::new();
            return_values.insert("output".to_string(), Value::String(answer));
            return Ok(AgentOutput::Finish(AgentFinish::new(
                return_values,
                text.to_string(),
            )));
        }

        // Look for Action: and Action Input:
        let action_re = Regex::new(r"(?mi)^\s*Action\s*:\s*(.+?)$").unwrap();
        let action_input_re = Regex::new(r"(?msi)^\s*Action\s+Input\s*:\s*(.+)").unwrap();

        let tool = action_re
            .captures(text)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .ok_or_else(|| CognisError::OutputParserError {
                message: "Could not find `Action:` in LLM output".to_string(),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;

        let raw_input = action_input_re
            .captures(text)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();

        // Try to parse as JSON; fall back to a plain string value.
        let tool_input =
            serde_json::from_str::<Value>(&raw_input).unwrap_or(Value::String(raw_input));

        Ok(AgentOutput::Action(AgentAction::new(
            tool,
            tool_input,
            text.to_string(),
        )))
    }
}

// ---------------------------------------------------------------------------
// JSON output parser
// ---------------------------------------------------------------------------

/// Parses a JSON object (possibly wrapped in a Markdown code block) with the
/// shape:
///
/// ```json
/// {"action": "tool_name", "action_input": {"key": "value"}}
/// ```
///
/// The special action `"Final Answer"` is treated as a finish signal.
#[derive(Debug, Clone, Default)]
pub struct JsonOutputParser;

impl JsonOutputParser {
    pub fn new() -> Self {
        Self
    }

    /// Attempts to extract a JSON object from text, stripping optional
    /// Markdown code fences.
    fn extract_json(text: &str) -> Option<&str> {
        let trimmed = text.trim();

        // Try to extract from ```json ... ``` or ``` ... ``` blocks
        let code_block_re = Regex::new(r"(?s)```(?:json)?\s*\n?(.*?)\n?\s*```").unwrap();
        if let Some(caps) = code_block_re.captures(trimmed) {
            return caps.get(1).map(|m| m.as_str().trim());
        }

        // Otherwise look for the first { ... } block
        let start = trimmed.find('{')?;
        let mut depth = 0i32;
        let bytes = trimmed.as_bytes();
        let mut in_string = false;
        let mut escape_next = false;
        for (i, &b) in bytes.iter().enumerate().skip(start) {
            if escape_next {
                escape_next = false;
                continue;
            }
            match b {
                b'\\' if in_string => {
                    escape_next = true;
                }
                b'"' => {
                    in_string = !in_string;
                }
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&trimmed[start..=i]);
                    }
                }
                _ => {}
            }
        }
        None
    }
}

impl AgentOutputParser for JsonOutputParser {
    fn parse(&self, text: &str) -> Result<AgentOutput> {
        let json_str =
            Self::extract_json(text).ok_or_else(|| CognisError::OutputParserError {
                message: "Could not find a JSON object in the LLM output".to_string(),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;

        let parsed: Value =
            serde_json::from_str(json_str).map_err(|e| CognisError::OutputParserError {
                message: format!("Failed to parse JSON: {e}"),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;

        let action = parsed
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CognisError::OutputParserError {
                message: "JSON object missing `action` field".to_string(),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;

        let action_input = parsed.get("action_input").cloned().unwrap_or(Value::Null);

        // "Final Answer" is the conventional finish signal.
        if action.eq_ignore_ascii_case("final answer")
            || action.eq_ignore_ascii_case("final_answer")
        {
            let output = match &action_input {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            let mut return_values = HashMap::new();
            return_values.insert("output".to_string(), Value::String(output));
            return Ok(AgentOutput::Finish(AgentFinish::new(
                return_values,
                text.to_string(),
            )));
        }

        Ok(AgentOutput::Action(AgentAction::new(
            action.to_string(),
            action_input,
            text.to_string(),
        )))
    }
}

// ---------------------------------------------------------------------------
// XML output parser
// ---------------------------------------------------------------------------

/// Parses XML-style agent output:
///
/// ```xml
/// <tool>search</tool>
/// <tool_input>{"query": "rust"}</tool_input>
/// ```
///
/// A `<tool>` value of `"final_answer"` or `"Final Answer"` signals
/// termination.
#[derive(Debug, Clone, Default)]
pub struct XmlOutputParser;

impl XmlOutputParser {
    pub fn new() -> Self {
        Self
    }

    fn extract_tag(text: &str, tag: &str) -> Option<String> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = text.find(&open)? + open.len();
        let end = text.find(&close)?;
        if start <= end {
            Some(text[start..end].trim().to_string())
        } else {
            None
        }
    }
}

impl AgentOutputParser for XmlOutputParser {
    fn parse(&self, text: &str) -> Result<AgentOutput> {
        let tool =
            Self::extract_tag(text, "tool").ok_or_else(|| CognisError::OutputParserError {
                message: "Could not find <tool>...</tool> in the LLM output".to_string(),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;

        let raw_input = Self::extract_tag(text, "tool_input").unwrap_or_default();

        let tool_input =
            serde_json::from_str::<Value>(&raw_input).unwrap_or(Value::String(raw_input));

        if tool.eq_ignore_ascii_case("final_answer") || tool.eq_ignore_ascii_case("final answer") {
            let output = match &tool_input {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            let mut return_values = HashMap::new();
            return_values.insert("output".to_string(), Value::String(output));
            return Ok(AgentOutput::Finish(AgentFinish::new(
                return_values,
                text.to_string(),
            )));
        }

        Ok(AgentOutput::Action(AgentAction::new(
            tool,
            tool_input,
            text.to_string(),
        )))
    }
}

// ---------------------------------------------------------------------------
// Tool-call output parser
// ---------------------------------------------------------------------------

/// Parses structured tool-call output from chat models.
///
/// Expects a JSON value representing an AI message with a `tool_calls` array.
/// Each tool call should have `name`, `args`, and optionally `id`.
///
/// If no tool calls are present the message `content` is treated as the final
/// answer.
#[derive(Debug, Clone, Default)]
pub struct ToolCallOutputParser;

impl ToolCallOutputParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a structured AI message (as [`serde_json::Value`]) into an
    /// [`AgentOutput`].
    pub fn parse_ai_message(&self, ai_message: &Value) -> Result<AgentOutput> {
        let tool_calls = ai_message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            let content = ai_message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut return_values = HashMap::new();
            let log = content.clone();
            return_values.insert("output".to_string(), Value::String(content));
            return Ok(AgentOutput::Finish(AgentFinish::new(return_values, log)));
        }

        // Take the first tool call for the single-action output.
        let tc = &tool_calls[0];
        let tool_name = tc
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let tool_input = tc
            .get("args")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let log = format!(
            "Calling tool `{}` with args: {}",
            tool_name,
            serde_json::to_string(&tool_input).unwrap_or_default()
        );

        Ok(AgentOutput::Action(AgentAction::new(
            tool_name, tool_input, log,
        )))
    }
}

impl AgentOutputParser for ToolCallOutputParser {
    /// Parses text as a JSON AI-message object and delegates to
    /// [`Self::parse_ai_message`].
    fn parse(&self, text: &str) -> Result<AgentOutput> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| CognisError::OutputParserError {
                message: format!("ToolCallOutputParser expects a JSON AI message: {e}"),
                observation: None,
                llm_output: Some(text.to_string()),
            })?;
        self.parse_ai_message(&value)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // ReAct parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn react_parse_action_with_json_input() {
        let parser = ReActOutputParser::new();
        let text = "Thought: I need to search\nAction: search\nAction Input: {\"query\": \"rust\"}";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, json!({"query": "rust"}));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn react_parse_action_with_plain_string_input() {
        let parser = ReActOutputParser::new();
        let text = "Thought: Let me look it up\nAction: search\nAction Input: rust language";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, Value::String("rust language".into()));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn react_parse_final_answer() {
        let parser = ReActOutputParser::new();
        let text = "Thought: I now know the answer\nFinal Answer: 42";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Finish(f) => {
                assert_eq!(
                    f.return_values.get("output"),
                    Some(&Value::String("42".into()))
                );
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn react_parse_final_answer_multiline() {
        let parser = ReActOutputParser::new();
        let text = "Thought: I know.\nFinal Answer: Line one\nLine two";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Finish(f) => {
                let output = f.return_values.get("output").unwrap().as_str().unwrap();
                assert!(output.contains("Line one"));
                assert!(output.contains("Line two"));
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn react_parse_error_on_missing_action() {
        let parser = ReActOutputParser::new();
        let text = "Thought: hmm, I'm confused";
        assert!(parser.parse(text).is_err());
    }

    // -----------------------------------------------------------------------
    // JSON parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn json_parse_action() {
        let parser = JsonOutputParser::new();
        let text = r#"{"action": "search", "action_input": {"query": "test"}}"#;
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, json!({"query": "test"}));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn json_parse_final_answer() {
        let parser = JsonOutputParser::new();
        let text = r#"{"action": "Final Answer", "action_input": "The answer is 42"}"#;
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Finish(f) => {
                assert_eq!(
                    f.return_values.get("output"),
                    Some(&Value::String("The answer is 42".into()))
                );
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn json_parse_code_block() {
        let parser = JsonOutputParser::new();
        let text =
            "Here is my response:\n```json\n{\"action\": \"calc\", \"action_input\": \"2+2\"}\n```";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "calc");
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn json_parse_embedded_json() {
        let parser = JsonOutputParser::new();
        let text = "I think we should call a tool.\n{\"action\": \"lookup\", \"action_input\": {\"id\": 5}}\nDone.";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "lookup");
                assert_eq!(a.tool_input, json!({"id": 5}));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn json_parse_error_no_json() {
        let parser = JsonOutputParser::new();
        let text = "I don't know what to do.";
        assert!(parser.parse(text).is_err());
    }

    #[test]
    fn json_parse_error_missing_action_field() {
        let parser = JsonOutputParser::new();
        let text = r#"{"tool": "search"}"#;
        assert!(parser.parse(text).is_err());
    }

    #[test]
    fn json_parse_final_answer_case_insensitive() {
        let parser = JsonOutputParser::new();
        let text = r#"{"action": "final_answer", "action_input": "done"}"#;
        let result = parser.parse(text).unwrap();
        assert!(matches!(result, AgentOutput::Finish(_)));
    }

    // -----------------------------------------------------------------------
    // XML parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn xml_parse_action() {
        let parser = XmlOutputParser::new();
        let text = "<tool>search</tool>\n<tool_input>{\"query\": \"rust\"}</tool_input>";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, json!({"query": "rust"}));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn xml_parse_final_answer() {
        let parser = XmlOutputParser::new();
        let text = "<tool>final_answer</tool>\n<tool_input>The result is 42</tool_input>";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Finish(f) => {
                assert_eq!(
                    f.return_values.get("output"),
                    Some(&Value::String("The result is 42".into()))
                );
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn xml_parse_string_input() {
        let parser = XmlOutputParser::new();
        let text = "<tool>greet</tool><tool_input>hello world</tool_input>";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "greet");
                assert_eq!(a.tool_input, Value::String("hello world".into()));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn xml_parse_error_missing_tool_tag() {
        let parser = XmlOutputParser::new();
        let text = "<tool_input>some input</tool_input>";
        assert!(parser.parse(text).is_err());
    }

    // -----------------------------------------------------------------------
    // ToolCall parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn toolcall_parse_with_tool_calls() {
        let parser = ToolCallOutputParser::new();
        let msg = json!({
            "content": "",
            "tool_calls": [
                {"name": "search", "args": {"q": "test"}, "id": "call_1"}
            ]
        });
        let result = parser.parse_ai_message(&msg).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, json!({"q": "test"}));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn toolcall_parse_no_tool_calls() {
        let parser = ToolCallOutputParser::new();
        let msg = json!({
            "content": "The answer is 42",
            "tool_calls": []
        });
        let result = parser.parse_ai_message(&msg).unwrap();
        match result {
            AgentOutput::Finish(f) => {
                assert_eq!(
                    f.return_values.get("output"),
                    Some(&Value::String("The answer is 42".into()))
                );
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn toolcall_parse_from_text() {
        let parser = ToolCallOutputParser::new();
        let text =
            r#"{"content": "hi", "tool_calls": [{"name": "calc", "args": {"x": 1}, "id": "c1"}]}"#;
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "calc");
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn toolcall_parse_invalid_json() {
        let parser = ToolCallOutputParser::new();
        assert!(parser.parse("not json").is_err());
    }

    #[test]
    fn toolcall_parse_missing_tool_calls_field() {
        let parser = ToolCallOutputParser::new();
        let msg = json!({"content": "hello"});
        let result = parser.parse_ai_message(&msg).unwrap();
        // No tool_calls field means finish
        assert!(matches!(result, AgentOutput::Finish(_)));
    }

    // -----------------------------------------------------------------------
    // Edge-case / whitespace tests
    // -----------------------------------------------------------------------

    #[test]
    fn react_parse_with_extra_whitespace() {
        let parser = ReActOutputParser::new();
        let text =
            "  Thought:  thinking hard  \n  Action:   search  \n  Action Input:   query text  ";
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "search");
                assert_eq!(a.tool_input, Value::String("query text".into()));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn json_parse_nested_objects() {
        let parser = JsonOutputParser::new();
        let text = r#"{"action": "api_call", "action_input": {"url": "http://example.com", "body": {"key": "val"}}}"#;
        let result = parser.parse(text).unwrap();
        match result {
            AgentOutput::Action(a) => {
                assert_eq!(a.tool, "api_call");
                assert_eq!(
                    a.tool_input,
                    json!({"url": "http://example.com", "body": {"key": "val"}})
                );
            }
            _ => panic!("Expected Action"),
        }
    }
}
