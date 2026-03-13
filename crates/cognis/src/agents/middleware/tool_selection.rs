//! LLM-based tool selection middleware — uses an LLM to filter available tools.
//!
//! Mirrors Python `langchain.agents.middleware.tool_selection`.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::error::Result;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognis_core::tools::base::BaseTool;

use super::types::{AgentMiddleware, AsyncModelHandler, ModelCallResult, ModelRequest};

/// Middleware that uses an LLM to select relevant tools before the main model call.
///
/// When a large number of tools are available, this middleware narrows down
/// the tool set to the most relevant ones based on the current conversation context.
pub struct LLMToolSelectorMiddleware {
    /// The LLM model used for tool selection.
    pub selector_model: Arc<dyn BaseChatModel>,
    /// Maximum number of tools to select.
    pub max_tools: usize,
    /// Tools that should always be included regardless of LLM selection.
    pub always_include: HashSet<String>,
    /// System prompt for the selection LLM.
    pub system_prompt: String,
}

impl LLMToolSelectorMiddleware {
    pub fn new(selector_model: Arc<dyn BaseChatModel>, max_tools: usize) -> Self {
        Self {
            selector_model,
            max_tools,
            always_include: HashSet::new(),
            system_prompt: "You are a tool selector. Given the conversation context and a list \
                           of available tools, select the most relevant tools for the current task. \
                           Respond with a JSON array of tool names."
                .into(),
        }
    }

    pub fn with_always_include(mut self, tool_name: impl Into<String>) -> Self {
        self.always_include.insert(tool_name.into());
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Select tools based on the current context.
    ///
    /// In production, this would call the selector LLM. For now, it applies
    /// a simple heuristic: include always_include tools, then fill up to max_tools.
    pub fn select_tools(
        &self,
        available_tools: &[Arc<dyn BaseTool>],
        _messages: &[Message],
    ) -> Vec<Arc<dyn BaseTool>> {
        let mut selected = Vec::new();

        // Always include specified tools first
        for tool in available_tools {
            if self.always_include.contains(tool.name()) {
                selected.push(Arc::clone(tool));
            }
        }

        // Fill remaining slots
        for tool in available_tools {
            if selected.len() >= self.max_tools {
                break;
            }
            if !self.always_include.contains(tool.name()) {
                selected.push(Arc::clone(tool));
            }
        }

        selected
    }
}

#[async_trait]
impl AgentMiddleware for LLMToolSelectorMiddleware {
    fn name(&self) -> &str {
        "LLMToolSelectorMiddleware"
    }

    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        // Select relevant tools based on conversation context
        let selected_tools = self.select_tools(&request.tools, &request.messages);

        // Construct a new request with only the selected tools
        let filtered_request = ModelRequest {
            model: request.model.clone(),
            messages: request.messages.clone(),
            system_message: request.system_message.clone(),
            tool_choice: request.tool_choice.clone(),
            tools: selected_tools,
            response_format: request.response_format.clone(),
            state: request.state.clone(),
            model_settings: request.model_settings.clone(),
        };

        let response = handler(&filtered_request).await?;
        Ok(ModelCallResult::Response(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cognis_core::outputs::ChatResult;
    use cognis_core::tools::types::{ToolInput, ToolOutput};

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
            Ok(ChatResult {
                generations: vec![],
                llm_output: None,
            })
        }
    }

    /// Mock tool for testing.
    struct MockTool {
        tool_name: String,
    }

    #[async_trait]
    impl BaseTool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "A mock tool"
        }

        async fn _run(&self, _input: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Content(serde_json::json!("mock result")))
        }
    }

    fn mock_model() -> Arc<dyn BaseChatModel> {
        Arc::new(MockChatModel)
    }

    fn mock_tools(names: &[&str]) -> Vec<Arc<dyn BaseTool>> {
        names
            .iter()
            .map(|&name| {
                Arc::new(MockTool {
                    tool_name: name.to_string(),
                }) as Arc<dyn BaseTool>
            })
            .collect()
    }

    #[test]
    fn test_tool_selector_new() {
        let selector = LLMToolSelectorMiddleware::new(mock_model(), 5);
        assert_eq!(selector.max_tools, 5);
        assert_eq!(selector.name(), "LLMToolSelectorMiddleware");
    }

    #[test]
    fn test_tool_selector_always_include() {
        let selector =
            LLMToolSelectorMiddleware::new(mock_model(), 2).with_always_include("search");

        let tools = mock_tools(&["calculator", "search", "filesystem"]);
        let messages = vec![Message::human("test")];
        let selected = selector.select_tools(&tools, &messages);

        // "search" should be included, plus one more up to max_tools
        assert!(selected.len() <= 2);
        assert!(selected.iter().any(|t| t.name() == "search"));
    }

    #[test]
    fn test_tool_selector_max_tools_limit() {
        let selector = LLMToolSelectorMiddleware::new(mock_model(), 2);
        let tools = mock_tools(&["a", "b", "c", "d", "e"]);
        let messages = vec![Message::human("test")];
        let selected = selector.select_tools(&tools, &messages);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_tool_selector_fewer_than_max() {
        let selector = LLMToolSelectorMiddleware::new(mock_model(), 10);
        let tools = mock_tools(&["a", "b"]);
        let messages = vec![Message::human("test")];
        let selected = selector.select_tools(&tools, &messages);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_tool_selector_custom_system_prompt() {
        let selector =
            LLMToolSelectorMiddleware::new(mock_model(), 5).with_system_prompt("Custom prompt");
        assert_eq!(selector.system_prompt, "Custom prompt");
    }
}
