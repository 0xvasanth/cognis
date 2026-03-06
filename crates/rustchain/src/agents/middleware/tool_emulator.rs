//! Tool emulator middleware — emulate tool execution using an LLM.
//!
//! Useful for testing agent workflows without running actual tools.
//! The LLM generates plausible tool results instead of executing the real tool.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::Message;
use rustchain_core::tools::base::BaseTool;

use super::types::AgentMiddleware;

/// Middleware that emulates tool calls using an LLM instead of executing the real tool.
///
/// This is primarily useful for testing agent workflows in environments where
/// actual tool execution is not desirable or possible.
pub struct LLMToolEmulator {
    /// The LLM model to use for emulating tool responses.
    pub model: Arc<dyn BaseChatModel>,
    /// If true, emulate all tools. Otherwise, only emulate tools in `tools_to_emulate`.
    pub emulate_all: bool,
    /// Set of tool names to emulate (ignored if `emulate_all` is true).
    pub tools_to_emulate: HashSet<String>,
    /// Optional system prompt to guide the emulation LLM.
    pub system_prompt: Option<String>,
}

impl LLMToolEmulator {
    /// Create an emulator that emulates all tools.
    pub fn all(model: Arc<dyn BaseChatModel>) -> Self {
        Self {
            model,
            emulate_all: true,
            tools_to_emulate: HashSet::new(),
            system_prompt: Some(
                "You are emulating a tool call. Given the tool name and input, \
                 generate a realistic response that the tool would produce. \
                 Respond only with the tool's output, no explanation."
                    .into(),
            ),
        }
    }

    /// Create an emulator for specific tools.
    pub fn for_tools(model: Arc<dyn BaseChatModel>, tools: Vec<String>) -> Self {
        Self {
            model,
            emulate_all: false,
            tools_to_emulate: tools.into_iter().collect(),
            system_prompt: Some(
                "You are emulating a tool call. Given the tool name and input, \
                 generate a realistic response that the tool would produce. \
                 Respond only with the tool's output, no explanation."
                    .into(),
            ),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Check if a tool should be emulated.
    pub fn should_emulate(&self, tool_name: &str) -> bool {
        if self.emulate_all {
            return true;
        }
        self.tools_to_emulate.contains(tool_name)
    }

    /// Generate an emulated tool response using the LLM.
    ///
    /// Builds a prompt with the tool name, description, and input, then calls the
    /// LLM to generate a realistic emulated response. The LLM response text is
    /// parsed as JSON if possible, otherwise wrapped in a result object.
    pub async fn emulate_tool_call(
        &self,
        tool_name: &str,
        tool_description: &str,
        input: &Value,
    ) -> Result<Value> {
        let input_str =
            serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string());

        // Build messages for the emulation LLM
        let system_prompt = self.system_prompt.as_deref().unwrap_or(
            "You are emulating a tool call. Given the tool name, description, and input, \
             generate a realistic response that the tool would produce. \
             Respond only with the tool's output as valid JSON, no explanation.",
        );

        let messages = vec![
            Message::system(system_prompt),
            Message::human(format!(
                "Tool Name: {}\nTool Description: {}\nInput:\n{}\n\nGenerate a realistic JSON output for this tool call:",
                tool_name, tool_description, input_str
            )),
        ];

        match self.model.invoke_messages(&messages, None).await {
            Ok(ai_msg) => {
                let response_text = ai_msg.base.content.text();
                // Try to parse the LLM response as JSON
                match serde_json::from_str::<Value>(&response_text) {
                    Ok(json_val) => Ok(json_val),
                    Err(_) => {
                        // Wrap the raw text in a result object
                        Ok(serde_json::json!({
                            "_emulated": true,
                            "tool": tool_name,
                            "result": response_text
                        }))
                    }
                }
            }
            Err(_) => {
                // Fallback: return a placeholder if the LLM call fails
                Ok(serde_json::json!({
                    "_emulated": true,
                    "tool": tool_name,
                    "input": input,
                    "result": format!("[Emulated result for tool '{}']", tool_name)
                }))
            }
        }
    }
}

#[async_trait]
impl AgentMiddleware for LLMToolEmulator {
    fn name(&self) -> &str {
        "LLMToolEmulator"
    }

    async fn wrap_tool_call(
        &self,
        tool: &dyn BaseTool,
        input: &Value,
        handler: &(dyn for<'a, 'b> Fn(&'a dyn BaseTool, &'b Value) -> Result<Value> + Send + Sync),
    ) -> Result<Value> {
        if self.should_emulate(tool.name()) {
            self.emulate_tool_call(tool.name(), tool.description(), input).await
        } else {
            handler(tool, input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::chat_model::BaseChatModel;
    use rustchain_core::messages::Message;
    use rustchain_core::outputs::ChatResult;

    /// Mock chat model for testing.
    struct MockChatModel;

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        fn llm_type(&self) -> &str {
            "mock"
        }

        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Ok(ChatResult { generations: vec![], llm_output: None })
        }
    }

    fn mock_model() -> Arc<dyn BaseChatModel> {
        Arc::new(MockChatModel)
    }

    #[test]
    fn test_emulator_all() {
        let emulator = LLMToolEmulator::all(mock_model());
        assert!(emulator.emulate_all);
        assert!(emulator.should_emulate("any_tool"));
        assert!(emulator.should_emulate("another_tool"));
    }

    #[test]
    fn test_emulator_specific_tools() {
        let emulator = LLMToolEmulator::for_tools(
            mock_model(),
            vec!["search".into(), "calculator".into()],
        );
        assert!(!emulator.emulate_all);
        assert!(emulator.should_emulate("search"));
        assert!(emulator.should_emulate("calculator"));
        assert!(!emulator.should_emulate("filesystem"));
    }

    #[tokio::test]
    async fn test_emulate_tool_call() {
        let emulator = LLMToolEmulator::all(mock_model());
        let input = serde_json::json!({"query": "test"});
        let result = emulator
            .emulate_tool_call("search", "Search the web", &input)
            .await
            .unwrap();
        // MockChatModel returns empty generations, so fallback kicks in
        assert_eq!(result["_emulated"], true);
        assert_eq!(result["tool"], "search");
    }

    #[test]
    fn test_emulator_name() {
        let emulator = LLMToolEmulator::all(mock_model());
        assert_eq!(emulator.name(), "LLMToolEmulator");
    }

    #[test]
    fn test_emulator_with_system_prompt() {
        let emulator = LLMToolEmulator::all(mock_model())
            .with_system_prompt("Custom prompt");
        assert_eq!(emulator.system_prompt.as_deref(), Some("Custom prompt"));
    }

    #[tokio::test]
    async fn test_emulate_tool_call_result_structure() {
        let emulator = LLMToolEmulator::all(mock_model());
        let input = serde_json::json!({"x": 1, "y": 2});
        let result = emulator
            .emulate_tool_call("add", "Add two numbers", &input)
            .await
            .unwrap();
        assert!(result.get("_emulated").is_some());
        assert!(result.get("result").is_some());
    }
}
