//! Agent executor — the core loop that runs model -> tool calls -> tool results -> model
//! until the model stops calling tools or the iteration limit is reached.
//!
//! Supports:
//! - Step limits (`max_iterations`)
//! - Time budgets (`max_execution_time_secs`)
//! - Intermediate step tracking (`return_intermediate_steps`)
//! - Early stopping strategies (`EarlyStoppingMethod`)
//! - Parsing error recovery (`handle_parsing_errors`)
//! - `Runnable` trait integration

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::callbacks::base::CallbackHandler;
use rustchain_core::callbacks::manager::CallbackManager;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{Message, ToolMessage};
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;
use rustchain_core::tools::base::BaseTool;
use uuid::Uuid;

use super::middleware::types::AgentMiddleware;

/// What to do when the agent hits the iteration or time limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarlyStoppingMethod {
    /// Return whatever we have accumulated so far (the last AI message content).
    Force,
    /// Ask the LLM to generate a final answer given the conversation so far.
    GenerateResponse,
}

/// A single intermediate step in the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// The action the agent took (tool name, input, and log).
    pub action: AgentAction,
    /// The observation returned by the tool.
    pub observation: String,
}

/// An action the agent wants to take — a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    /// The name of the tool to call.
    pub tool_name: String,
    /// The input to pass to the tool (JSON).
    pub tool_input: Value,
    /// A human-readable log describing the action.
    pub log: String,
}

/// The result of running an agent to completion.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The full message history including the initial messages and all
    /// intermediate model/tool messages.
    pub messages: Vec<Message>,
    /// The final text output from the model.
    pub output: String,
    /// Optional intermediate steps (populated when `return_intermediate_steps` is true).
    pub intermediate_steps: Vec<AgentStep>,
}

/// Builder for constructing an [`AgentExecutor`].
pub struct AgentExecutorBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    tools: Vec<Arc<dyn BaseTool>>,
    middleware: Vec<Arc<dyn AgentMiddleware>>,
    max_iterations: usize,
    max_execution_time_secs: Option<u64>,
    return_intermediate_steps: bool,
    early_stopping_method: Option<EarlyStoppingMethod>,
    handle_parsing_errors: bool,
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
            max_execution_time_secs: None,
            return_intermediate_steps: false,
            early_stopping_method: None,
            handle_parsing_errors: false,
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
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the maximum execution time in seconds.
    pub fn max_execution_time_secs(mut self, secs: u64) -> Self {
        self.max_execution_time_secs = Some(secs);
        self
    }

    /// Whether to include intermediate steps in the result (default: false).
    pub fn return_intermediate_steps(mut self, yes: bool) -> Self {
        self.return_intermediate_steps = yes;
        self
    }

    /// Set the early stopping method.
    ///
    /// When set, reaching the iteration/time limit returns an `Ok` result
    /// using the specified strategy instead of an error.
    /// By default (when not set), exceeding the limit returns
    /// `Err(RecursionLimitExceeded)`.
    pub fn early_stopping_method(mut self, method: EarlyStoppingMethod) -> Self {
        self.early_stopping_method = Some(method);
        self
    }

    /// Whether to handle parsing errors by retrying (default: false).
    pub fn handle_parsing_errors(mut self, yes: bool) -> Self {
        self.handle_parsing_errors = yes;
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
            max_execution_time_secs: self.max_execution_time_secs,
            return_intermediate_steps: self.return_intermediate_steps,
            early_stopping_method: self.early_stopping_method,
            handle_parsing_errors: self.handle_parsing_errors,
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
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
    /// Optional time budget in seconds.
    pub max_execution_time_secs: Option<u64>,
    /// Whether to include intermediate steps in the result.
    pub return_intermediate_steps: bool,
    /// What to do when the iteration/time limit is reached.
    /// `None` means return `Err(RecursionLimitExceeded)`.
    pub early_stopping_method: Option<EarlyStoppingMethod>,
    /// Whether to handle parsing errors by feeding the error back to the model.
    pub handle_parsing_errors: bool,
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
    /// 5. If `max_iterations` is exceeded or time runs out, applies the
    ///    `early_stopping_method`.
    ///
    /// Callback events are fired throughout the loop for observability.
    pub async fn run(&self, initial_messages: &[Message]) -> Result<AgentResult> {
        let cb = CallbackManager::new(self.callbacks.clone(), None);
        let chain_run_id = Uuid::new_v4();

        let serialized_chain = serde_json::json!({"name": "AgentExecutor"});
        let inputs = serde_json::json!({
            "messages": initial_messages.iter().map(|m| m.content().text()).collect::<Vec<_>>()
        });

        let _ = cb.on_chain_start(&serialized_chain, &inputs, chain_run_id).await;

        let mut messages: Vec<Message> = initial_messages.to_vec();
        let mut intermediate_steps: Vec<AgentStep> = Vec::new();
        let start_time = std::time::Instant::now();

        for _iteration in 0..self.max_iterations {
            // Check time budget
            if let Some(max_secs) = self.max_execution_time_secs {
                if start_time.elapsed().as_secs() >= max_secs {
                    let reason = "Agent exceeded time limit";
                    match self.early_stopping_method {
                        Some(_) => {
                            return self.handle_early_stop(
                                &cb,
                                chain_run_id,
                                &mut messages,
                                intermediate_steps,
                                reason,
                            ).await;
                        }
                        None => {
                            let err = RustChainError::RecursionLimitExceeded(
                                reason.to_string(),
                            );
                            let _ = cb.on_chain_error(&err.to_string(), chain_run_id).await;
                            return Err(err);
                        }
                    }
                }
            }

            let llm_run_id = Uuid::new_v4();
            let serialized_llm = serde_json::json!({"name": self.model.llm_type()});
            let prompts: Vec<String> = messages.iter().map(|m| m.content().text()).collect();
            let _ = cb.on_llm_start(&serialized_llm, &prompts, llm_run_id).await;

            // Call the model
            let chat_result = match self.model._generate(&messages, None).await {
                Ok(r) => {
                    let llm_result = rustchain_core::outputs::LLMResult {
                        generations: vec![r
                            .generations
                            .iter()
                            .map(|g| rustchain_core::outputs::Generation::new(&g.text))
                            .collect()],
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
                    if self.handle_parsing_errors {
                        // Feed parsing error back into conversation and retry
                        messages.push(Message::human(
                            "Error: expected an AI message from the model. Please try again.",
                        ));
                        continue;
                    }
                    return Err(RustChainError::Other(
                        "Expected AIMessage from model generation".into(),
                    ));
                }
            };

            // Push the AI message into the conversation
            messages.push(generation.message.clone());

            // If no tool calls, we are done
            if ai_msg.tool_calls.is_empty() {
                let output = ai_msg.base.content.text();
                let outputs = serde_json::json!({"output": output});
                let _ = cb.on_chain_end(&outputs, chain_run_id).await;
                return Ok(AgentResult {
                    messages,
                    output,
                    intermediate_steps: if self.return_intermediate_steps {
                        intermediate_steps
                    } else {
                        Vec::new()
                    },
                });
            }

            // Execute each tool call
            for tool_call in &ai_msg.tool_calls {
                let tool_call_id = tool_call.id.clone().unwrap_or_default();

                let tool_run_id = Uuid::new_v4();
                let serialized_tool = serde_json::json!({"name": &tool_call.name});
                let tool_input_str =
                    serde_json::to_string(&tool_call.args).unwrap_or_default();
                let _ = cb
                    .on_tool_start(&serialized_tool, &tool_input_str, tool_run_id)
                    .await;

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
                                let _ = cb.on_tool_end(&text, tool_run_id).await;
                                text
                            }
                            Err(e) => {
                                let err_text = format!("Error: {e}");
                                let _ = cb.on_tool_error(&err_text, tool_run_id).await;
                                if self.handle_parsing_errors {
                                    err_text
                                } else {
                                    err_text
                                }
                            }
                        }
                    }
                    None => {
                        let err_text =
                            format!("Error: tool '{}' not found", tool_call.name);
                        let _ = cb.on_tool_error(&err_text, tool_run_id).await;
                        err_text
                    }
                };

                // Record intermediate step
                intermediate_steps.push(AgentStep {
                    action: AgentAction {
                        tool_name: tool_call.name.clone(),
                        tool_input: serde_json::to_value(&tool_call.args)
                            .unwrap_or_default(),
                        log: format!(
                            "Calling tool `{}` with args: {}",
                            tool_call.name, tool_input_str
                        ),
                    },
                    observation: result_text.clone(),
                });

                messages.push(Message::Tool(ToolMessage::new(
                    &result_text,
                    &tool_call_id,
                )));
            }
        }

        // Iteration limit reached
        let reason = format!(
            "Agent exceeded maximum iterations ({})",
            self.max_iterations
        );
        match self.early_stopping_method {
            Some(_) => {
                self.handle_early_stop(
                    &cb,
                    chain_run_id,
                    &mut messages,
                    intermediate_steps,
                    &reason,
                )
                .await
            }
            None => {
                let err = RustChainError::RecursionLimitExceeded(reason);
                let _ = cb.on_chain_error(&err.to_string(), chain_run_id).await;
                Err(err)
            }
        }
    }

    /// Handle early stopping when the iteration or time limit is reached.
    async fn handle_early_stop(
        &self,
        cb: &CallbackManager,
        chain_run_id: Uuid,
        messages: &mut Vec<Message>,
        intermediate_steps: Vec<AgentStep>,
        reason: &str,
    ) -> Result<AgentResult> {
        // early_stopping_method is guaranteed to be Some when this is called
        let method = self.early_stopping_method.unwrap_or(EarlyStoppingMethod::Force);
        match method {
            EarlyStoppingMethod::Force => {
                // Return whatever we have — use the last AI message content or
                // the reason string if no AI message exists.
                let output = messages
                    .iter()
                    .rev()
                    .find_map(|m| match m {
                        Message::Ai(ai) => Some(ai.base.content.text()),
                        _ => None,
                    })
                    .unwrap_or_else(|| format!("Agent stopped early: {reason}"));
                let outputs = serde_json::json!({"output": &output});
                let _ = cb.on_chain_end(&outputs, chain_run_id).await;
                Ok(AgentResult {
                    messages: messages.clone(),
                    output,
                    intermediate_steps: if self.return_intermediate_steps {
                        intermediate_steps
                    } else {
                        Vec::new()
                    },
                })
            }
            EarlyStoppingMethod::GenerateResponse => {
                // Ask the model to summarize what it knows so far
                messages.push(Message::human(
                    "Based on everything above, please provide your final answer now.",
                ));
                match self.model._generate(messages, None).await {
                    Ok(result) => {
                        let generation = result.generations.into_iter().next().ok_or_else(
                            || RustChainError::Other("No generations returned".into()),
                        )?;
                        let output = generation.message.content().text();
                        messages.push(generation.message);
                        let outputs = serde_json::json!({"output": &output});
                        let _ = cb.on_chain_end(&outputs, chain_run_id).await;
                        Ok(AgentResult {
                            messages: messages.clone(),
                            output,
                            intermediate_steps: if self.return_intermediate_steps {
                                intermediate_steps
                            } else {
                                Vec::new()
                            },
                        })
                    }
                    Err(e) => {
                        let _ = cb.on_chain_error(&e.to_string(), chain_run_id).await;
                        Err(e)
                    }
                }
            }
        }
    }
}

/// Implements the `Runnable` trait so `AgentExecutor` can be composed in LCEL chains.
///
/// Input: a JSON value containing either:
/// - `{ "messages": [...] }` — an array of `Message` objects
/// - A JSON string — treated as a single human message
/// - A JSON array — treated as `Vec<Message>`
///
/// Output: a JSON object `{ "output": "...", "intermediate_steps": [...] }`.
#[async_trait]
impl Runnable for AgentExecutor {
    fn name(&self) -> &str {
        "AgentExecutor"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let messages = parse_agent_input(input)?;
        let result = self.run(&messages).await?;
        let mut output = serde_json::json!({ "output": result.output });
        if self.return_intermediate_steps {
            output["intermediate_steps"] =
                serde_json::to_value(&result.intermediate_steps)
                    .unwrap_or(Value::Array(vec![]));
        }
        Ok(output)
    }
}

/// Parse a `Value` input into a `Vec<Message>` for agent invocation.
fn parse_agent_input(input: Value) -> Result<Vec<Message>> {
    match input {
        Value::String(s) => Ok(vec![Message::human(&s)]),
        Value::Array(_) => serde_json::from_value(input)
            .map_err(|e| RustChainError::Other(format!("Failed to deserialize messages: {e}"))),
        Value::Object(ref map) => {
            if let Some(msgs) = map.get("messages") {
                serde_json::from_value(msgs.clone()).map_err(|e| {
                    RustChainError::Other(format!("Failed to deserialize messages: {e}"))
                })
            } else if let Some(text) = map.get("input").and_then(|v| v.as_str()) {
                Ok(vec![Message::human(text)])
            } else {
                Err(RustChainError::TypeMismatch {
                    expected: "String, Array, or Object with 'messages'/'input'".into(),
                    got: "Object without recognized keys".into(),
                })
            }
        }
        _ => Err(RustChainError::TypeMismatch {
            expected: "String or Array of Messages".into(),
            got: format!("{}", input),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rustchain_core::callbacks::base::CallbackHandler;
    use rustchain_core::language_models::fake::{
        FakeListChatModel, FakeMessagesListChatModel, GenericFakeChatModel,
    };
    use rustchain_core::messages::tool_types::ToolCall;
    use rustchain_core::messages::{AIMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult, LLMResult};
    use serde_json::Value;
    use std::sync::Mutex;

    /// A simple mock tool that returns a fixed result.
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
            Ok(rustchain_core::tools::types::ToolOutput::Content(
                Value::String("4".to_string()),
            ))
        }
    }

    /// A mock tool that echoes its input.
    struct EchoTool;

    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the input"
        }

        async fn _run(
            &self,
            input: rustchain_core::tools::types::ToolInput,
        ) -> Result<rustchain_core::tools::types::ToolOutput> {
            let text = match input {
                rustchain_core::tools::types::ToolInput::Text(s) => s,
                rustchain_core::tools::types::ToolInput::Structured(map) => {
                    serde_json::to_string(&map).unwrap_or_default()
                }
                other => format!("{:?}", other),
            };
            Ok(rustchain_core::tools::types::ToolOutput::Content(
                Value::String(text),
            ))
        }
    }

    /// A mock model that always calls a tool (never finishes on its own).
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

    /// A mock model that returns a tool call on first invocation, then a text response.
    struct ToolThenAnswerModel {
        call_count: Mutex<u32>,
    }

    impl ToolThenAnswerModel {
        fn new() -> Self {
            Self {
                call_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for ToolThenAnswerModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            if *count == 1 {
                let mut ai = AIMessage::new("I'll use the calculator");
                ai.tool_calls.push(ToolCall {
                    name: "calculator".to_string(),
                    args: HashMap::from([(
                        "expr".to_string(),
                        Value::String("2+2".to_string()),
                    )]),
                    id: Some("call_1".to_string()),
                });
                Ok(ChatResult {
                    generations: vec![ChatGeneration::new(ai)],
                    llm_output: None,
                })
            } else {
                let ai = AIMessage::new("The answer is 4");
                Ok(ChatResult {
                    generations: vec![ChatGeneration::new(ai)],
                    llm_output: None,
                })
            }
        }

        fn llm_type(&self) -> &str {
            "mock-tool-then-answer"
        }
    }

    /// A model that calls two tools, then answers.
    struct MultiToolModel {
        call_count: Mutex<u32>,
    }

    impl MultiToolModel {
        fn new() -> Self {
            Self {
                call_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for MultiToolModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            match *count {
                1 => {
                    let mut ai = AIMessage::new("First, calculate");
                    ai.tool_calls.push(ToolCall {
                        name: "calculator".to_string(),
                        args: HashMap::from([(
                            "expr".to_string(),
                            Value::String("2+2".to_string()),
                        )]),
                        id: Some("call_1".to_string()),
                    });
                    Ok(ChatResult {
                        generations: vec![ChatGeneration::new(ai)],
                        llm_output: None,
                    })
                }
                2 => {
                    let mut ai = AIMessage::new("Now echo");
                    ai.tool_calls.push(ToolCall {
                        name: "echo".to_string(),
                        args: HashMap::from([(
                            "text".to_string(),
                            Value::String("hello".to_string()),
                        )]),
                        id: Some("call_2".to_string()),
                    });
                    Ok(ChatResult {
                        generations: vec![ChatGeneration::new(ai)],
                        llm_output: None,
                    })
                }
                _ => {
                    let ai = AIMessage::new("Done: 4 and hello");
                    Ok(ChatResult {
                        generations: vec![ChatGeneration::new(ai)],
                        llm_output: None,
                    })
                }
            }
        }

        fn llm_type(&self) -> &str {
            "mock-multi-tool"
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

    // ── Test 1: Simple Q&A without tools (immediate answer) ──

    #[tokio::test]
    async fn test_simple_qa_no_tools() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Hello there!".into()]));
        let executor = AgentExecutor::builder().model(model).build();

        let result = executor
            .run(&[Message::human("Hi")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "Hello there!");
        assert_eq!(result.messages.len(), 2); // human + ai
        assert!(result.intermediate_steps.is_empty());
    }

    // ── Test 2: Single tool call and response ──

    #[tokio::test]
    async fn test_single_tool_call() {
        let model: Arc<dyn BaseChatModel> = Arc::new(ToolThenAnswerModel::new());
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .build();

        let result = executor
            .run(&[Message::human("What is 2+2?")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "The answer is 4");
        // human + ai(tool_call) + tool_result + ai(final)
        assert_eq!(result.messages.len(), 4);
    }

    // ── Test 3: Multi-step tool usage ──

    #[tokio::test]
    async fn test_multi_step_tool_usage() {
        let model: Arc<dyn BaseChatModel> = Arc::new(MultiToolModel::new());
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let echo: Arc<dyn BaseTool> = Arc::new(EchoTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(calc)
            .tool(echo)
            .return_intermediate_steps(true)
            .build();

        let result = executor
            .run(&[Message::human("Do two things")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "Done: 4 and hello");
        assert_eq!(result.intermediate_steps.len(), 2);
        assert_eq!(result.intermediate_steps[0].action.tool_name, "calculator");
        assert_eq!(result.intermediate_steps[0].observation, "4");
        assert_eq!(result.intermediate_steps[1].action.tool_name, "echo");
    }

    // ── Test 4: Max iterations limit reached (Force) ──

    #[tokio::test]
    async fn test_max_iterations_force() {
        let model: Arc<dyn BaseChatModel> = Arc::new(AlwaysToolCallModel);
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .max_iterations(2)
            .early_stopping_method(EarlyStoppingMethod::Force)
            .build();

        let result = executor
            .run(&[Message::human("loop")])
            .await
            .expect("Force stopping should return Ok");

        // The last AI message content is "calling tool"
        assert_eq!(result.output, "calling tool");
    }

    // ── Test 5: Return intermediate steps ──

    #[tokio::test]
    async fn test_return_intermediate_steps() {
        let model: Arc<dyn BaseChatModel> = Arc::new(ToolThenAnswerModel::new());
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .return_intermediate_steps(true)
            .build();

        let result = executor
            .run(&[Message::human("Compute")])
            .await
            .expect("should succeed");

        assert_eq!(result.intermediate_steps.len(), 1);
        assert_eq!(result.intermediate_steps[0].action.tool_name, "calculator");
        assert_eq!(result.intermediate_steps[0].observation, "4");
    }

    // ── Test 6: Intermediate steps not returned when disabled ──

    #[tokio::test]
    async fn test_no_intermediate_steps_when_disabled() {
        let model: Arc<dyn BaseChatModel> = Arc::new(ToolThenAnswerModel::new());
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .return_intermediate_steps(false)
            .build();

        let result = executor
            .run(&[Message::human("Compute")])
            .await
            .expect("should succeed");

        assert!(result.intermediate_steps.is_empty());
    }

    // ── Test 7: Early stopping GenerateResponse method ──

    #[tokio::test]
    async fn test_early_stopping_generate_response() {
        // Use FakeMessagesListChatModel: first N calls return tool calls,
        // then the generate-response call returns a final answer.
        let responses = vec![
            // Iteration 1: tool call
            {
                let mut ai = AIMessage::new("calling tool");
                ai.tool_calls.push(ToolCall {
                    name: "calculator".to_string(),
                    args: HashMap::new(),
                    id: Some("c1".to_string()),
                });
                Message::Ai(ai)
            },
            // Iteration 2: tool call again (hits limit)
            {
                let mut ai = AIMessage::new("calling tool again");
                ai.tool_calls.push(ToolCall {
                    name: "calculator".to_string(),
                    args: HashMap::new(),
                    id: Some("c2".to_string()),
                });
                Message::Ai(ai)
            },
            // GenerateResponse call: final answer
            Message::Ai(AIMessage::new("Final summary answer")),
        ];
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeMessagesListChatModel::new(responses));
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .max_iterations(2)
            .early_stopping_method(EarlyStoppingMethod::GenerateResponse)
            .build();

        let result = executor
            .run(&[Message::human("loop")])
            .await
            .expect("GenerateResponse should succeed");

        assert_eq!(result.output, "Final summary answer");
    }

    // ── Test 8: Handle parsing errors ──

    #[tokio::test]
    async fn test_handle_parsing_errors() {
        // A model where the first call returns a non-AI message variant,
        // and the second returns a proper AI message. We use
        // GenericFakeChatModel which always returns AIMessage, so let's
        // instead use a custom model.
        struct BadThenGoodModel {
            count: Mutex<u32>,
        }

        #[async_trait]
        impl BaseChatModel for BadThenGoodModel {
            async fn _generate(
                &self,
                _messages: &[Message],
                _stop: Option<&[String]>,
            ) -> Result<ChatResult> {
                let mut c = self.count.lock().unwrap();
                *c += 1;
                if *c == 1 {
                    // Return a Human message instead of AI — a "parsing" error
                    Ok(ChatResult {
                        generations: vec![ChatGeneration {
                            message: Message::human("oops"),
                            text: "oops".to_string(),
                            generation_info: None,
                        }],
                        llm_output: None,
                    })
                } else {
                    Ok(ChatResult {
                        generations: vec![ChatGeneration::new(AIMessage::new("recovered"))],
                        llm_output: None,
                    })
                }
            }

            fn llm_type(&self) -> &str {
                "bad-then-good"
            }
        }

        let model: Arc<dyn BaseChatModel> = Arc::new(BadThenGoodModel {
            count: Mutex::new(0),
        });
        let executor = AgentExecutor::builder()
            .model(model)
            .handle_parsing_errors(true)
            .build();

        let result = executor
            .run(&[Message::human("Hi")])
            .await
            .expect("should recover from parsing error");

        assert_eq!(result.output, "recovered");
    }

    // ── Test 9: Builder pattern ──

    #[test]
    fn test_builder_pattern() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["ok".into()]));
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);

        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .max_iterations(5)
            .max_execution_time_secs(30)
            .return_intermediate_steps(true)
            .early_stopping_method(EarlyStoppingMethod::GenerateResponse)
            .handle_parsing_errors(true)
            .build();

        assert_eq!(executor.max_iterations, 5);
        assert_eq!(executor.max_execution_time_secs, Some(30));
        assert!(executor.return_intermediate_steps);
        assert_eq!(
            executor.early_stopping_method,
            Some(EarlyStoppingMethod::GenerateResponse)
        );
        assert!(executor.handle_parsing_errors);
        assert_eq!(executor.tools.len(), 1);
        assert!(executor.tools.contains_key("calculator"));
    }

    // ── Test 10: Empty tools list ──

    #[tokio::test]
    async fn test_empty_tools_list() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["No tools needed".into()]));
        let executor = AgentExecutor::builder().model(model).build();

        assert!(executor.tools.is_empty());

        let result = executor
            .run(&[Message::human("Question")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "No tools needed");
    }

    // ── Test 11: Runnable trait implementation ──

    #[tokio::test]
    async fn test_runnable_trait_invoke_with_string() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["runnable result".into()]));
        let executor = AgentExecutor::builder().model(model).build();

        let result = executor
            .invoke(Value::String("Hello".into()), None)
            .await
            .expect("should succeed");

        assert_eq!(result["output"], "runnable result");
    }

    // ── Test 12: Runnable trait with object input ──

    #[tokio::test]
    async fn test_runnable_trait_invoke_with_object() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["from input".into()]));
        let executor = AgentExecutor::builder().model(model).build();

        let input = serde_json::json!({ "input": "What is 2+2?" });
        let result = executor
            .invoke(input, None)
            .await
            .expect("should succeed");

        assert_eq!(result["output"], "from input");
    }

    // ── Test 13: Runnable name ──

    #[test]
    fn test_runnable_name() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["ok".into()]));
        let executor = AgentExecutor::builder().model(model).build();
        assert_eq!(executor.name(), "AgentExecutor");
    }

    // ── Test 14: Runnable with intermediate steps in output ──

    #[tokio::test]
    async fn test_runnable_with_intermediate_steps() {
        let model: Arc<dyn BaseChatModel> = Arc::new(ToolThenAnswerModel::new());
        let tool: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let executor = AgentExecutor::builder()
            .model(model)
            .tool(tool)
            .return_intermediate_steps(true)
            .build();

        let result = executor
            .invoke(Value::String("Compute".into()), None)
            .await
            .expect("should succeed");

        assert_eq!(result["output"], "The answer is 4");
        let steps = result["intermediate_steps"].as_array().unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0]["action"]["tool_name"], "calculator");
        assert_eq!(steps[0]["observation"], "4");
    }

    // ── Test 15: Callbacks with tool calls ──

    #[tokio::test]
    async fn test_callbacks_with_tool_calls() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<dyn CallbackHandler> =
            Arc::new(RecordingCallbackHandler::new(events.clone()));

        let model: Arc<dyn BaseChatModel> = Arc::new(ToolThenAnswerModel::new());
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

    // ── Test 16: GenericFakeChatModel with agent ──

    #[tokio::test]
    async fn test_generic_fake_chat_model_agent() {
        let model: Arc<dyn BaseChatModel> = Arc::new(GenericFakeChatModel::from_messages(
            vec![AIMessage::new("The capital of France is Paris")],
        ));
        let executor = AgentExecutor::builder().model(model).build();

        let result = executor
            .run(&[Message::human("What is the capital of France?")])
            .await
            .expect("should succeed");

        assert_eq!(result.output, "The capital of France is Paris");
    }

    // ── Test 17: Missing tool returns error message in observation ──

    #[tokio::test]
    async fn test_missing_tool_error_in_observation() {
        // Model calls a tool that doesn't exist
        let responses = vec![
            {
                let mut ai = AIMessage::new("calling unknown_tool");
                ai.tool_calls.push(ToolCall {
                    name: "nonexistent".to_string(),
                    args: HashMap::new(),
                    id: Some("c1".to_string()),
                });
                Message::Ai(ai)
            },
            Message::Ai(AIMessage::new("I see the tool was not found")),
        ];
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeMessagesListChatModel::new(responses));
        let executor = AgentExecutor::builder()
            .model(model)
            .return_intermediate_steps(true)
            .build();

        let result = executor
            .run(&[Message::human("Use nonexistent tool")])
            .await
            .expect("should succeed even if tool not found");

        assert_eq!(result.output, "I see the tool was not found");
        assert_eq!(result.intermediate_steps.len(), 1);
        assert!(result.intermediate_steps[0]
            .observation
            .contains("tool 'nonexistent' not found"));
    }
}
