//! Agent executor — the core loop that runs model -> tool calls -> tool results -> model
//! until the model stops calling tools or the iteration limit is reached.

use std::collections::HashMap;
use std::sync::Arc;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{Message, ToolMessage};
use rustchain_core::tools::base::BaseTool;

use super::middleware::types::AgentMiddleware;

/// The result of running an agent to completion.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The full message history including the initial messages and all
    /// intermediate model/tool messages.
    pub messages: Vec<Message>,
    /// The final text output from the model.
    pub output: String,
}

/// Builder for constructing an [`AgentExecutor`].
pub struct AgentExecutorBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    tools: Vec<Arc<dyn BaseTool>>,
    middleware: Vec<Arc<dyn AgentMiddleware>>,
    max_iterations: u32,
}

impl AgentExecutorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            middleware: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Set the chat model (required).
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Add a single tool.
    pub fn tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools at once.
    pub fn tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add a middleware.
    pub fn middleware(mut self, mw: Arc<dyn AgentMiddleware>) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Set the maximum number of iterations (default: 10).
    pub fn max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    /// Build the [`AgentExecutor`].
    ///
    /// # Panics
    /// Panics if no model was provided.
    pub fn build(self) -> AgentExecutor {
        let model = self
            .model
            .expect("AgentExecutor requires a model — call .model() on the builder");
        let tools: HashMap<String, Arc<dyn BaseTool>> = self
            .tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();
        AgentExecutor {
            model,
            tools,
            middleware: self.middleware,
            max_iterations: self.max_iterations,
        }
    }
}

impl Default for AgentExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Executes an agent loop: model generates, tools execute, repeat until done.
pub struct AgentExecutor {
    /// The chat model used for generation.
    pub model: Arc<dyn BaseChatModel>,
    /// Available tools keyed by name.
    pub tools: HashMap<String, Arc<dyn BaseTool>>,
    /// Middleware pipeline (currently stored but not invoked in the base loop).
    pub middleware: Vec<Arc<dyn AgentMiddleware>>,
    /// Maximum iterations before returning an error.
    pub max_iterations: u32,
}

impl AgentExecutor {
    /// Create a new builder.
    pub fn builder() -> AgentExecutorBuilder {
        AgentExecutorBuilder::new()
    }

    /// Run the agent loop to completion.
    ///
    /// 1. Starts with `initial_messages`.
    /// 2. Calls the model, appends the AI message.
    /// 3. If the AI message contains tool calls, executes each tool and appends
    ///    tool result messages, then loops back to step 2.
    /// 4. If the AI message has no tool calls, returns the final text.
    /// 5. If `max_iterations` is exceeded, returns
    ///    [`RustChainError::RecursionLimitExceeded`].
    pub async fn run(&self, initial_messages: &[Message]) -> Result<AgentResult> {
        let mut messages: Vec<Message> = initial_messages.to_vec();

        for _ in 0..self.max_iterations {
            // Call the model
            let chat_result = self.model._generate(&messages, None).await?;
            let generation = chat_result
                .generations
                .into_iter()
                .next()
                .ok_or_else(|| RustChainError::Other("No generations returned".into()))?;

            // Extract the AI message
            let ai_msg = match &generation.message {
                Message::Ai(ai) => ai.clone(),
                _ => {
                    return Err(RustChainError::Other(
                        "Expected AIMessage from model generation".into(),
                    ))
                }
            };

            // Push the AI message into the conversation
            messages.push(generation.message.clone());

            // If no tool calls, we are done
            if ai_msg.tool_calls.is_empty() {
                let output = ai_msg.base.content.text();
                return Ok(AgentResult { messages, output });
            }

            // Execute each tool call
            for tool_call in &ai_msg.tool_calls {
                let tool_call_id = tool_call
                    .id
                    .clone()
                    .unwrap_or_default();

                let result_text = match self.tools.get(&tool_call.name) {
                    Some(tool) => {
                        let args_value =
                            serde_json::to_value(&tool_call.args).unwrap_or_default();
                        match tool.run_json(&args_value).await {
                            Ok(value) => match value {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            },
                            Err(e) => format!("Error: {e}"),
                        }
                    }
                    None => {
                        format!("Error: tool '{}' not found", tool_call.name)
                    }
                };

                messages.push(Message::Tool(ToolMessage::new(
                    &result_text,
                    &tool_call_id,
                )));
            }
        }

        Err(RustChainError::RecursionLimitExceeded(format!(
            "Agent exceeded maximum iterations ({})",
            self.max_iterations
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rustchain_core::messages::{AIMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult};

    /// A mock model that always returns a simple text response (no tool calls).
    struct NoToolModel;

    #[async_trait]
    impl BaseChatModel for NoToolModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let ai = AIMessage::new("Hello from the model");
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai)],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "mock-no-tool"
        }
    }

    #[test]
    fn test_executor_builder() {
        let model: Arc<dyn BaseChatModel> = Arc::new(NoToolModel);
        let executor = AgentExecutor::builder()
            .model(model)
            .max_iterations(5)
            .build();

        assert_eq!(executor.max_iterations, 5);
        assert!(executor.tools.is_empty());
        assert!(executor.middleware.is_empty());
    }

    #[tokio::test]
    async fn test_executor_no_tool_calls() {
        let model: Arc<dyn BaseChatModel> = Arc::new(NoToolModel);
        let executor = AgentExecutor::builder().model(model).build();

        let result = executor
            .run(&[Message::human("Hi")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "Hello from the model");
        // human + ai response
        assert_eq!(result.messages.len(), 2);
    }
}
