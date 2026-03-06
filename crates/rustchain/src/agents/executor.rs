//! Agent executor — the core loop that runs model -> tool calls -> tool results -> model
//! until the model stops calling tools or the iteration limit is reached.

use std::collections::HashMap;
use std::sync::Arc;

use rustchain_core::callbacks::base::CallbackHandler;
use rustchain_core::callbacks::manager::CallbackManager;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{Message, ToolMessage};
use rustchain_core::tools::base::BaseTool;
use uuid::Uuid;

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
    callbacks: Vec<Arc<dyn CallbackHandler>>,
}

impl AgentExecutorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            middleware: Vec::new(),
            max_iterations: 10,
            callbacks: Vec::new(),
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

    /// Add a single callback handler.
    pub fn callback(mut self, handler: Arc<dyn CallbackHandler>) -> Self {
        self.callbacks.push(handler);
        self
    }

    /// Add multiple callback handlers at once.
    pub fn callbacks(mut self, handlers: Vec<Arc<dyn CallbackHandler>>) -> Self {
        self.callbacks.extend(handlers);
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
            callbacks: self.callbacks,
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
    /// Callback handlers for observability events.
    pub callbacks: Vec<Arc<dyn CallbackHandler>>,
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
    ///
    /// Callback events are fired throughout the loop for observability.
    pub async fn run(&self, initial_messages: &[Message]) -> Result<AgentResult> {
        let cb = CallbackManager::new(self.callbacks.clone(), None);
        let chain_run_id = Uuid::new_v4();

        let serialized_chain = serde_json::json!({"name": "AgentExecutor"});
        let inputs = serde_json::json!({
            "messages": initial_messages.iter().map(|m| m.content().text()).collect::<Vec<_>>()
        });

        // Fire chain_start; ignore errors from non-raise_error handlers
        let _ = cb.on_chain_start(&serialized_chain, &inputs, chain_run_id).await;

        let mut messages: Vec<Message> = initial_messages.to_vec();

        for _ in 0..self.max_iterations {
            let llm_run_id = Uuid::new_v4();
            let serialized_llm = serde_json::json!({"name": self.model.llm_type()});
            let prompts: Vec<String> = messages.iter().map(|m| m.content().text()).collect();

            // Fire llm_start
            let _ = cb.on_llm_start(&serialized_llm, &prompts, llm_run_id).await;

            // Call the model
            let chat_result = match self.model._generate(&messages, None).await {
                Ok(r) => {
                    // Fire llm_end — build an LLMResult from the ChatResult
                    let llm_result = rustchain_core::outputs::LLMResult {
                        generations: vec![r.generations.iter().map(|g| {
                            rustchain_core::outputs::Generation::new(&g.text)
                        }).collect()],
                        llm_output: r.llm_output.clone(),
                        run: None,
                    };
                    let _ = cb.on_llm_end(&llm_result, llm_run_id).await;
                    r
                }
                Err(e) => {
                    let _ = cb.on_llm_error(&e.to_string(), llm_run_id).await;
                    let _ = cb.on_chain_error(&e.to_string(), chain_run_id).await;
                    return Err(e);
                }
            };

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
                let outputs = serde_json::json!({"output": output});
                let _ = cb.on_chain_end(&outputs, chain_run_id).await;
                return Ok(AgentResult { messages, output });
            }

            // Execute each tool call
            for tool_call in &ai_msg.tool_calls {
                let tool_call_id = tool_call
                    .id
                    .clone()
                    .unwrap_or_default();

                let tool_run_id = Uuid::new_v4();
                let serialized_tool = serde_json::json!({"name": &tool_call.name});
                let tool_input = serde_json::to_string(&tool_call.args).unwrap_or_default();

                // Fire tool_start
                let _ = cb.on_tool_start(&serialized_tool, &tool_input, tool_run_id).await;

                let result_text = match self.tools.get(&tool_call.name) {
                    Some(tool) => {
                        let args_value =
                            serde_json::to_value(&tool_call.args).unwrap_or_default();
                        match tool.run_json(&args_value).await {
                            Ok(value) => {
                                let text = match value {
                                    serde_json::Value::String(s) => s,
                                    other => other.to_string(),
                                };
                                // Fire tool_end
                                let _ = cb.on_tool_end(&text, tool_run_id).await;
                                text
                            }
                            Err(e) => {
                                let err_text = format!("Error: {e}");
                                let _ = cb.on_tool_error(&err_text, tool_run_id).await;
                                err_text
                            }
                        }
                    }
                    None => {
                        let err_text = format!("Error: tool '{}' not found", tool_call.name);
                        let _ = cb.on_tool_error(&err_text, tool_run_id).await;
                        err_text
                    }
                };

                messages.push(Message::Tool(ToolMessage::new(
                    &result_text,
                    &tool_call_id,
                )));
            }
        }

        let err = RustChainError::RecursionLimitExceeded(format!(
            "Agent exceeded maximum iterations ({})",
            self.max_iterations
        ));
        let _ = cb.on_chain_error(&err.to_string(), chain_run_id).await;
        Err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rustchain_core::callbacks::base::CallbackHandler;
    use rustchain_core::messages::tool_types::ToolCall;
    use rustchain_core::messages::{AIMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult, LLMResult};
    use serde_json::Value;
    use std::sync::Mutex;

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

    /// A mock model that returns a tool call on the first invocation,
    /// then returns a plain text response on the second.
    struct ToolCallingModel {
        call_count: Mutex<u32>,
    }

    impl ToolCallingModel {
        fn new() -> Self {
            Self {
                call_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for ToolCallingModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            if *count == 1 {
                // First call: return a tool call
                let mut ai = AIMessage::new("I'll use the calculator");
                ai.tool_calls.push(ToolCall {
                    name: "calculator".to_string(),
                    args: HashMap::from([("expr".to_string(), Value::String("2+2".to_string()))]),
                    id: Some("call_1".to_string()),
                });
                Ok(ChatResult {
                    generations: vec![ChatGeneration::new(ai)],
                    llm_output: None,
                })
            } else {
                // Second call: final answer
                let ai = AIMessage::new("The answer is 4");
                Ok(ChatResult {
                    generations: vec![ChatGeneration::new(ai)],
                    llm_output: None,
                })
            }
        }

        fn llm_type(&self) -> &str {
            "mock-tool-calling"
        }
    }

    /// A mock model that always fails.
    struct FailingModel;

    #[async_trait]
    impl BaseChatModel for FailingModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Err(RustChainError::Other("model error".into()))
        }

        fn llm_type(&self) -> &str {
            "mock-failing"
        }
    }

    /// A simple mock tool.
    struct CalculatorTool;

    #[async_trait]
    impl BaseTool for CalculatorTool {
        fn name(&self) -> &str {
            "calculator"
        }

        fn description(&self) -> &str {
            "A calculator"
        }

        async fn _run(
            &self,
            _input: rustchain_core::tools::types::ToolInput,
        ) -> Result<rustchain_core::tools::types::ToolOutput> {
            Ok(rustchain_core::tools::types::ToolOutput::Content(Value::String("4".to_string())))
        }
    }

    /// A callback handler that records events in order.
    struct RecordingCallbackHandler {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingCallbackHandler {
        fn new(events: Arc<Mutex<Vec<String>>>) -> Self {
            Self { events }
        }
    }

    #[async_trait]
    impl CallbackHandler for RecordingCallbackHandler {
        async fn on_chain_start(
            &self,
            _serialized: &Value,
            _inputs: &Value,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_start".to_string());
            Ok(())
        }

        async fn on_chain_end(
            &self,
            _outputs: &Value,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_end".to_string());
            Ok(())
        }

        async fn on_chain_error(
            &self,
            _error: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_error".to_string());
            Ok(())
        }

        async fn on_llm_start(
            &self,
            _serialized: &Value,
            _prompts: &[String],
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_start".to_string());
            Ok(())
        }

        async fn on_llm_end(
            &self,
            _response: &LLMResult,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_end".to_string());
            Ok(())
        }

        async fn on_llm_error(
            &self,
            _error: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_error".to_string());
            Ok(())
        }

        async fn on_tool_start(
            &self,
            _serialized: &Value,
            _input_str: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("tool_start".to_string());
            Ok(())
        }

        async fn on_tool_end(
            &self,
            _output: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("tool_end".to_string());
            Ok(())
        }

        async fn on_tool_error(
            &self,
            _error: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> rustchain_core::error::Result<()> {
            self.events.lock().unwrap().push("tool_error".to_string());
            Ok(())
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

    #[tokio::test]
    async fn test_executor_callbacks_no_tools() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<dyn CallbackHandler> =
            Arc::new(RecordingCallbackHandler::new(events.clone()));

        let model: Arc<dyn BaseChatModel> = Arc::new(NoToolModel);
        let executor = AgentExecutor::builder()
            .model(model)
            .callback(handler)
            .build();

        let result = executor
            .run(&[Message::human("Hi")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "Hello from the model");

        let recorded = events.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["chain_start", "llm_start", "llm_end", "chain_end"]
        );
    }

    #[tokio::test]
    async fn test_executor_callbacks_with_tool_calls() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<dyn CallbackHandler> =
            Arc::new(RecordingCallbackHandler::new(events.clone()));

        let model: Arc<dyn BaseChatModel> = Arc::new(ToolCallingModel::new());
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .callback(handler)
            .build();

        let result = executor
            .run(&[Message::human("What is 2+2?")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "The answer is 4");

        let recorded = events.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![
                "chain_start",
                "llm_start",
                "llm_end",
                "tool_start",
                "tool_end",
                "llm_start",
                "llm_end",
                "chain_end",
            ]
        );
    }

    #[tokio::test]
    async fn test_executor_callbacks_on_llm_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<dyn CallbackHandler> =
            Arc::new(RecordingCallbackHandler::new(events.clone()));

        let model: Arc<dyn BaseChatModel> = Arc::new(FailingModel);
        let executor = AgentExecutor::builder()
            .model(model)
            .callback(handler)
            .build();

        let err = executor.run(&[Message::human("Hi")]).await;
        assert!(err.is_err());

        let recorded = events.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec!["chain_start", "llm_start", "llm_error", "chain_error"]
        );
    }

    #[tokio::test]
    async fn test_executor_callbacks_on_recursion_limit() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<dyn CallbackHandler> =
            Arc::new(RecordingCallbackHandler::new(events.clone()));

        // A model that always returns tool calls (never finishes)
        struct AlwaysToolCallModel;

        #[async_trait]
        impl BaseChatModel for AlwaysToolCallModel {
            async fn _generate(
                &self,
                _messages: &[Message],
                _stop: Option<&[String]>,
            ) -> Result<ChatResult> {
                let mut ai = AIMessage::new("calling tool");
                ai.tool_calls.push(ToolCall {
                    name: "calculator".to_string(),
                    args: HashMap::new(),
                    id: Some("call_x".to_string()),
                });
                Ok(ChatResult {
                    generations: vec![ChatGeneration::new(ai)],
                    llm_output: None,
                })
            }

            fn llm_type(&self) -> &str {
                "mock-always-tool"
            }
        }

        let model: Arc<dyn BaseChatModel> = Arc::new(AlwaysToolCallModel);
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .callback(handler)
            .max_iterations(2)
            .build();

        let err = executor.run(&[Message::human("loop")]).await;
        assert!(err.is_err());

        let recorded = events.lock().unwrap().clone();
        // 2 iterations: each has llm_start, llm_end, tool_start, tool_end
        // Then chain_error at the end
        assert_eq!(
            recorded,
            vec![
                "chain_start",
                "llm_start",
                "llm_end",
                "tool_start",
                "tool_end",
                "llm_start",
                "llm_end",
                "tool_start",
                "tool_end",
                "chain_error",
            ]
        );
    }
}
