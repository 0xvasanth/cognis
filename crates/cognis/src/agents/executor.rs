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
//!
//! Also provides a planner-based execution model:
//! - [`AgentPlanner`] trait for pluggable planning logic
//! - [`ToolExecutor`] for managing and invoking tools by name
//! - [`ExecutorConfig`] with builder pattern for iteration/time limits
//! - [`PlannerAgentExecutor`] — the main planning loop
//! - [`AgentFinish`], [`AgentDecision`], [`PlannerAgentStep`] enums for control flow

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::callbacks::base::CallbackHandler;
use cognis_core::callbacks::manager::CallbackManager;
use cognis_core::callbacks::{ToolEndEvent, ToolErrorEvent, ToolErrorKind, ToolStartEvent};
use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{Message, ToolMessage};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;
use cognis_core::tools::base::BaseTool;
use uuid::Uuid;

#[allow(deprecated)]
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
    /// Optional message log for tracing.
    #[serde(default)]
    pub message_log: Vec<String>,
}

impl AgentAction {
    /// Create a new `AgentAction`.
    pub fn new(tool: impl Into<String>, tool_input: Value, log: impl Into<String>) -> Self {
        Self {
            tool_name: tool.into(),
            tool_input,
            log: log.into(),
            message_log: Vec::new(),
        }
    }

    /// Create a new `AgentAction` with a message log.
    pub fn with_message_log(mut self, message_log: Vec<String>) -> Self {
        self.message_log = message_log;
        self
    }

    /// Serialize to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "tool": self.tool_name,
            "tool_input": self.tool_input,
            "log": self.log,
            "message_log": self.message_log,
        })
    }
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
    #[allow(deprecated)]
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
    #[allow(deprecated)]
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
    #[allow(deprecated)]
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

        let _ = cb
            .on_chain_start(&serialized_chain, &inputs, chain_run_id)
            .await;

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
                            return self
                                .handle_early_stop(
                                    &cb,
                                    chain_run_id,
                                    &mut messages,
                                    intermediate_steps,
                                    reason,
                                )
                                .await;
                        }
                        None => {
                            let err = CognisError::RecursionLimitExceeded(reason.to_string());
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
                    let llm_result = cognis_core::outputs::LLMResult {
                        generations: vec![r
                            .generations
                            .iter()
                            .map(|g| cognis_core::outputs::Generation::new(&g.text))
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
                .ok_or_else(|| CognisError::Other("No generations returned".into()))?;

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
                    return Err(CognisError::Other(
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
                let args_value = serde_json::to_value(&tool_call.args).unwrap_or_default();
                let tool_input_str = serde_json::to_string(&args_value).unwrap_or_default();
                let tool_call_id_opt = tool_call.id.clone();

                let _ = cb
                    .on_tool_start(ToolStartEvent {
                        tool: tool_call.name.clone(),
                        serialized: serialized_tool.clone(),
                        input_str: tool_input_str.clone(),
                        inputs: args_value.clone(),
                        tool_call_id: tool_call_id_opt.clone(),
                        run_id: tool_run_id,
                        parent_run_id: Some(chain_run_id),
                        tags: vec![],
                        metadata: Default::default(),
                    })
                    .await;

                let result_text = match self.tools.get(&tool_call.name) {
                    Some(tool) => {
                        match tool.run_json(&args_value).await {
                            Ok(value) => {
                                let text = match value {
                                    serde_json::Value::String(s) => s,
                                    other => other.to_string(),
                                };
                                let _ = cb
                                    .on_tool_end(ToolEndEvent {
                                        tool: tool_call.name.clone(),
                                        output_str: text.clone(),
                                        // TEMPORARY: Task 6 will replace this with the real Value.
                                        output_value: serde_json::Value::String(text.clone()),
                                        artifact: None,
                                        tool_call_id: tool_call_id_opt.clone(),
                                        run_id: tool_run_id,
                                        parent_run_id: Some(chain_run_id),
                                    })
                                    .await;
                                text
                            }
                            Err(e) => {
                                let err_text = format!("Error: {e}");
                                let _ = cb
                                    .on_tool_error(ToolErrorEvent {
                                        tool: tool_call.name.clone(),
                                        error: err_text.clone(),
                                        error_kind: ToolErrorKind::Execution,
                                        tool_call_id: tool_call_id_opt.clone(),
                                        run_id: tool_run_id,
                                        parent_run_id: Some(chain_run_id),
                                    })
                                    .await;
                                err_text
                            }
                        }
                    }
                    None => {
                        let err_text = format!("Error: tool '{}' not found", tool_call.name);
                        let _ = cb
                            .on_tool_error(ToolErrorEvent {
                                tool: tool_call.name.clone(),
                                error: err_text.clone(),
                                error_kind: ToolErrorKind::NotFound,
                                tool_call_id: tool_call_id_opt.clone(),
                                run_id: tool_run_id,
                                parent_run_id: Some(chain_run_id),
                            })
                            .await;
                        err_text
                    }
                };

                // Record intermediate step
                intermediate_steps.push(AgentStep {
                    action: AgentAction::new(
                        tool_call.name.clone(),
                        serde_json::to_value(&tool_call.args).unwrap_or_default(),
                        format!(
                            "Calling tool `{}` with args: {}",
                            tool_call.name, tool_input_str
                        ),
                    ),
                    observation: result_text.clone(),
                });

                messages.push(Message::Tool(ToolMessage::new(&result_text, &tool_call_id)));
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
                let err = CognisError::RecursionLimitExceeded(reason);
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
        let method = self
            .early_stopping_method
            .unwrap_or(EarlyStoppingMethod::Force);
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
                        let generation =
                            result.generations.into_iter().next().ok_or_else(|| {
                                CognisError::Other("No generations returned".into())
                            })?;
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

    /// Execute the agent and stream all intermediate events as they occur.
    ///
    /// Returns a `Stream` of [`StreamEvent`] values. Each event represents a
    /// step in the agent loop: LLM invocations, tool calls, agent decisions,
    /// and the final result. This is the Rust equivalent of Python LangChain's
    /// `Runnable.astream_events(version="v2")`.
    ///
    /// The stream completes when the agent finishes or errors out.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = executor.astream_events(messages).await?;
    /// while let Some(event) = stream.next().await {
    ///     let ev = event?;
    ///     // Match on ev.event (EventType enum) to handle each phase.
    /// }
    /// ```
    pub async fn astream_events(
        &self,
        initial_messages: Vec<cognis_core::messages::Message>,
    ) -> cognis_core::error::Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = cognis_core::error::Result<
                            cognis_core::tracers::event_stream::StreamEvent,
                        >,
                    > + Send,
            >,
        >,
    > {
        use cognis_core::callbacks::CallbackHandler;
        use cognis_core::tracers::event_stream::EventStreamCallbackHandler;
        use futures::StreamExt;
        use std::sync::Arc;
        use tokio_stream::wrappers::ReceiverStream;

        // Create the event stream handler and take its receiver.
        let handler = Arc::new(EventStreamCallbackHandler::with_defaults());
        let rx = handler
            .take_receiver()
            .expect("fresh EventStreamCallbackHandler must have a receiver");

        // Build a new executor that includes the event handler in its callbacks.
        // Clone the inner fields; AgentExecutor itself isn't Clone.
        let mut callbacks = self.callbacks.clone();
        callbacks.push(handler as Arc<dyn CallbackHandler>);

        #[allow(deprecated)]
        let spawned_executor = AgentExecutor {
            model: self.model.clone(),
            tools: self.tools.clone(),
            middleware: self.middleware.clone(),
            max_iterations: self.max_iterations,
            max_execution_time_secs: self.max_execution_time_secs,
            return_intermediate_steps: self.return_intermediate_steps,
            early_stopping_method: self.early_stopping_method,
            handle_parsing_errors: self.handle_parsing_errors,
            callbacks,
        };

        // Spawn the run in the background. Errors surface as OnChainError events.
        tokio::spawn(async move {
            let _ = spawned_executor.run(&initial_messages).await;
        });

        // Wrap the mpsc receiver in a Stream; each item is Ok(StreamEvent).
        let stream = ReceiverStream::new(rx).map(Ok);
        Ok(Box::pin(stream))
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
                serde_json::to_value(&result.intermediate_steps).unwrap_or(Value::Array(vec![]));
        }
        Ok(output)
    }
}

/// Parse a `Value` input into a `Vec<Message>` for agent invocation.
fn parse_agent_input(input: Value) -> Result<Vec<Message>> {
    match input {
        Value::String(s) => Ok(vec![Message::human(&s)]),
        Value::Array(_) => serde_json::from_value(input)
            .map_err(|e| CognisError::Other(format!("Failed to deserialize messages: {e}"))),
        Value::Object(ref map) => {
            if let Some(msgs) = map.get("messages") {
                serde_json::from_value(msgs.clone())
                    .map_err(|e| CognisError::Other(format!("Failed to deserialize messages: {e}")))
            } else if let Some(text) = map.get("input").and_then(|v| v.as_str()) {
                Ok(vec![Message::human(text)])
            } else {
                Err(CognisError::TypeMismatch {
                    expected: "String, Array, or Object with 'messages'/'input'".into(),
                    got: "Object without recognized keys".into(),
                })
            }
        }
        _ => Err(CognisError::TypeMismatch {
            expected: "String or Array of Messages".into(),
            got: format!("{}", input),
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Planner-based agent execution model
// ═══════════════════════════════════════════════════════════════════════════

/// A completed agent result — the agent has decided to stop and return values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinish {
    /// Key-value pairs the agent returns (typically includes "output").
    pub return_values: HashMap<String, Value>,
    /// Human-readable log of the finish decision.
    pub log: String,
}

impl AgentFinish {
    /// Create a new `AgentFinish`.
    pub fn new(return_values: HashMap<String, Value>, log: impl Into<String>) -> Self {
        Self {
            return_values,
            log: log.into(),
        }
    }

    /// Serialize to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "return_values": self.return_values,
            "log": self.log,
        })
    }
}

/// An intermediate step variant: either an action or a finish.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlannerAgentStep {
    /// The agent wants to invoke a tool.
    Action(AgentAction),
    /// The agent has finished and produced a final result.
    Finish(AgentFinish),
}

impl PlannerAgentStep {
    /// Returns `true` if this step is an action.
    pub fn is_action(&self) -> bool {
        matches!(self, Self::Action(_))
    }

    /// Returns `true` if this step is a finish.
    pub fn is_finish(&self) -> bool {
        matches!(self, Self::Finish(_))
    }
}

/// What the agent planner decides to do next.
#[derive(Debug, Clone)]
pub enum AgentDecision {
    /// Continue with one or more tool invocations.
    Continue(Vec<AgentAction>),
    /// The agent is done.
    Finish(AgentFinish),
    /// An error occurred during planning.
    Error(String),
}

/// Trait for the "brain" that decides what the agent should do next.
///
/// Implementations receive the current intermediate steps and input,
/// then return an [`AgentDecision`].
pub trait AgentPlanner: Send + Sync {
    /// Decide the next action(s) given intermediate steps so far and the original input.
    fn plan(&self, intermediate_steps: &[(AgentAction, String)], input: &Value) -> AgentDecision;
}

/// Manages available tools and executes them by name.
pub struct ToolExecutor {
    tools: HashMap<String, Box<dyn Fn(Value) -> Result<String> + Send + Sync>>,
}

impl ToolExecutor {
    /// Create an empty `ToolExecutor`.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool handler under the given name.
    pub fn add_tool(
        &mut self,
        name: String,
        handler: Box<dyn Fn(Value) -> Result<String> + Send + Sync>,
    ) {
        self.tools.insert(name, handler);
    }

    /// Execute a tool by name with the given JSON input.
    pub fn execute(&self, name: &str, input: Value) -> Result<String> {
        match self.tools.get(name) {
            Some(handler) => handler(input),
            None => Err(CognisError::Other(format!("Tool '{}' not found", name))),
        }
    }

    /// Check whether a tool with the given name is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Return the names of all registered tools.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the planner-based agent executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of planning iterations (default: 15).
    pub max_iterations: usize,
    /// Optional time budget in milliseconds.
    pub max_execution_time_ms: Option<u64>,
    /// Whether to include intermediate steps in the result (default: false).
    pub return_intermediate_steps: bool,
    /// What to do when the iteration/time limit is reached.
    pub early_stopping_method: EarlyStoppingMethod,
}

impl ExecutorConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self {
            max_iterations: 15,
            max_execution_time_ms: None,
            return_intermediate_steps: false,
            early_stopping_method: EarlyStoppingMethod::Force,
        }
    }

    /// Set max iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the time budget in milliseconds.
    pub fn max_execution_time_ms(mut self, ms: u64) -> Self {
        self.max_execution_time_ms = Some(ms);
        self
    }

    /// Set whether to return intermediate steps.
    pub fn return_intermediate_steps(mut self, yes: bool) -> Self {
        self.return_intermediate_steps = yes;
        self
    }

    /// Set the early stopping method.
    pub fn early_stopping_method(mut self, method: EarlyStoppingMethod) -> Self {
        self.early_stopping_method = method;
        self
    }
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of running the planner-based agent executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorResult {
    /// The final output value.
    pub output: Value,
    /// Intermediate steps taken (action, observation) pairs.
    pub intermediate_steps: Vec<(AgentAction, String)>,
    /// Number of planning iterations used.
    pub iterations: usize,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

impl ExecutorResult {
    /// Serialize to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "output": self.output,
            "intermediate_steps": self.intermediate_steps.iter().map(|(a, obs)| {
                serde_json::json!({
                    "action": {
                        "tool": a.tool_name,
                        "tool_input": a.tool_input,
                        "log": a.log,
                    },
                    "observation": obs,
                })
            }).collect::<Vec<_>>(),
            "iterations": self.iterations,
            "duration_ms": self.duration_ms,
        })
    }
}

/// The planner-based agent executor loop.
///
/// Repeatedly calls the planner to decide what to do, executes tool actions
/// via the `ToolExecutor`, and accumulates intermediate steps until the
/// planner returns `Finish` or the iteration limit is reached.
pub struct PlannerAgentExecutor {
    planner: Box<dyn AgentPlanner>,
    tools: ToolExecutor,
    config: ExecutorConfig,
}

impl PlannerAgentExecutor {
    /// Create a new planner-based executor.
    pub fn new(
        planner: Box<dyn AgentPlanner>,
        tools: ToolExecutor,
        config: ExecutorConfig,
    ) -> Self {
        Self {
            planner,
            tools,
            config,
        }
    }

    /// Run the planning loop to completion.
    pub fn execute(&self, input: Value) -> Result<ExecutorResult> {
        let start = std::time::Instant::now();
        let mut intermediate_steps: Vec<(AgentAction, String)> = Vec::new();
        let mut iterations = 0usize;

        loop {
            // Check iteration limit
            if iterations >= self.config.max_iterations {
                return match self.config.early_stopping_method {
                    EarlyStoppingMethod::Force => Ok(ExecutorResult {
                        output: Value::String(format!(
                            "Agent stopped: reached max iterations ({})",
                            self.config.max_iterations,
                        )),
                        intermediate_steps: if self.config.return_intermediate_steps {
                            intermediate_steps
                        } else {
                            Vec::new()
                        },
                        iterations,
                        duration_ms: start.elapsed().as_millis() as u64,
                    }),
                    EarlyStoppingMethod::GenerateResponse => {
                        // Ask the planner one last time with a signal to wrap up
                        let decision = self.planner.plan(&intermediate_steps, &input);
                        match decision {
                            AgentDecision::Finish(finish) => Ok(ExecutorResult {
                                output: Value::Object(
                                    finish
                                        .return_values
                                        .into_iter()
                                        .collect::<serde_json::Map<String, Value>>(),
                                ),
                                intermediate_steps: if self.config.return_intermediate_steps {
                                    intermediate_steps
                                } else {
                                    Vec::new()
                                },
                                iterations,
                                duration_ms: start.elapsed().as_millis() as u64,
                            }),
                            _ => Ok(ExecutorResult {
                                output: Value::String(format!(
                                    "Agent stopped: reached max iterations ({})",
                                    self.config.max_iterations,
                                )),
                                intermediate_steps: if self.config.return_intermediate_steps {
                                    intermediate_steps
                                } else {
                                    Vec::new()
                                },
                                iterations,
                                duration_ms: start.elapsed().as_millis() as u64,
                            }),
                        }
                    }
                };
            }

            // Check time budget
            if let Some(max_ms) = self.config.max_execution_time_ms {
                if start.elapsed().as_millis() as u64 >= max_ms {
                    return Ok(ExecutorResult {
                        output: Value::String("Agent stopped: time limit exceeded".into()),
                        intermediate_steps: if self.config.return_intermediate_steps {
                            intermediate_steps
                        } else {
                            Vec::new()
                        },
                        iterations,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
            }

            iterations += 1;

            let decision = self.planner.plan(&intermediate_steps, &input);

            match decision {
                AgentDecision::Continue(actions) => {
                    for action in actions {
                        let observation = match self
                            .tools
                            .execute(&action.tool_name, action.tool_input.clone())
                        {
                            Ok(result) => result,
                            Err(e) => format!("Error: {}", e),
                        };
                        intermediate_steps.push((action, observation));
                    }
                }
                AgentDecision::Finish(finish) => {
                    return Ok(ExecutorResult {
                        output: Value::Object(
                            finish
                                .return_values
                                .into_iter()
                                .collect::<serde_json::Map<String, Value>>(),
                        ),
                        intermediate_steps: if self.config.return_intermediate_steps {
                            intermediate_steps
                        } else {
                            Vec::new()
                        },
                        iterations,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                AgentDecision::Error(msg) => {
                    return Err(CognisError::Other(msg));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use cognis_core::callbacks::base::CallbackHandler;
    use cognis_core::language_models::fake::{
        FakeListChatModel, FakeMessagesListChatModel, GenericFakeChatModel,
    };
    use cognis_core::messages::tool_types::ToolCall;
    use cognis_core::messages::{AIMessage, Message};
    use cognis_core::outputs::{ChatGeneration, ChatResult, LLMResult};
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
            _input: cognis_core::tools::types::ToolInput,
        ) -> Result<cognis_core::tools::types::ToolOutput> {
            Ok(cognis_core::tools::types::ToolOutput::Content(
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
            input: cognis_core::tools::types::ToolInput,
        ) -> Result<cognis_core::tools::types::ToolOutput> {
            let text = match input {
                cognis_core::tools::types::ToolInput::Text(s) => s,
                cognis_core::tools::types::ToolInput::Structured(map) => {
                    serde_json::to_string(&map).unwrap_or_default()
                }
                other => format!("{:?}", other),
            };
            Ok(cognis_core::tools::types::ToolOutput::Content(
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
                    args: HashMap::from([("expr".to_string(), Value::String("2+2".to_string()))]),
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
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_start".to_string());
            Ok(())
        }

        async fn on_chain_end(
            &self,
            _outputs: &Value,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_end".to_string());
            Ok(())
        }

        async fn on_chain_error(
            &self,
            _error: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("chain_error".to_string());
            Ok(())
        }

        async fn on_llm_start(
            &self,
            _serialized: &Value,
            _prompts: &[String],
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_start".to_string());
            Ok(())
        }

        async fn on_llm_end(
            &self,
            _response: &LLMResult,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_end".to_string());
            Ok(())
        }

        async fn on_llm_error(
            &self,
            _error: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_error".to_string());
            Ok(())
        }

        async fn on_tool_start(&self, _event: ToolStartEvent) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("tool_start".to_string());
            Ok(())
        }

        async fn on_tool_end(&self, _event: ToolEndEvent) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("tool_end".to_string());
            Ok(())
        }

        async fn on_tool_error(&self, _event: ToolErrorEvent) -> cognis_core::error::Result<()> {
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
        let executor = AgentExecutor::builder().model(model).tool(tool).build();

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
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(responses));
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
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["ok".into()]));
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
        let result = executor.invoke(input, None).await.expect("should succeed");

        assert_eq!(result["output"], "from input");
    }

    // ── Test 13: Runnable name ──

    #[test]
    fn test_runnable_name() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["ok".into()]));
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
        let model: Arc<dyn BaseChatModel> =
            Arc::new(GenericFakeChatModel::from_messages(vec![AIMessage::new(
                "The capital of France is Paris",
            )]));
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
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(responses));
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

    // ═══════════════════════════════════════════════════════════════════════
    // Planner-based executor tests
    // ═══════════════════════════════════════════════════════════════════════

    /// A planner that finishes immediately with the input value.
    struct ImmediateFinishPlanner;

    impl AgentPlanner for ImmediateFinishPlanner {
        fn plan(
            &self,
            _intermediate_steps: &[(AgentAction, String)],
            input: &Value,
        ) -> AgentDecision {
            let mut rv = HashMap::new();
            rv.insert("output".to_string(), input.clone());
            AgentDecision::Finish(AgentFinish::new(rv, "Finished immediately"))
        }
    }

    /// A planner that takes one action then finishes.
    struct OneActionPlanner;

    impl AgentPlanner for OneActionPlanner {
        fn plan(
            &self,
            intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            if intermediate_steps.is_empty() {
                AgentDecision::Continue(vec![AgentAction::new(
                    "calculator",
                    serde_json::json!({"expr": "2+2"}),
                    "Computing 2+2",
                )])
            } else {
                let obs = &intermediate_steps[0].1;
                let mut rv = HashMap::new();
                rv.insert("output".to_string(), Value::String(obs.clone()));
                AgentDecision::Finish(AgentFinish::new(rv, "Got the answer"))
            }
        }
    }

    /// A planner that takes two actions then finishes.
    struct TwoActionPlanner;

    impl AgentPlanner for TwoActionPlanner {
        fn plan(
            &self,
            intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            match intermediate_steps.len() {
                0 => AgentDecision::Continue(vec![AgentAction::new(
                    "calculator",
                    serde_json::json!({"expr": "2+2"}),
                    "Step 1",
                )]),
                1 => AgentDecision::Continue(vec![AgentAction::new(
                    "echo",
                    serde_json::json!({"text": "hello"}),
                    "Step 2",
                )]),
                _ => {
                    let mut rv = HashMap::new();
                    rv.insert(
                        "output".to_string(),
                        Value::String(format!(
                            "{} and {}",
                            intermediate_steps[0].1, intermediate_steps[1].1
                        )),
                    );
                    AgentDecision::Finish(AgentFinish::new(rv, "All done"))
                }
            }
        }
    }

    /// A planner that never finishes (always continues).
    struct NeverFinishPlanner;

    impl AgentPlanner for NeverFinishPlanner {
        fn plan(
            &self,
            _intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            AgentDecision::Continue(vec![AgentAction::new(
                "calculator",
                serde_json::json!({}),
                "Looping forever",
            )])
        }
    }

    /// A planner that returns an error.
    struct ErrorPlanner;

    impl AgentPlanner for ErrorPlanner {
        fn plan(
            &self,
            _intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            AgentDecision::Error("Planning failed".to_string())
        }
    }

    /// A planner that calls a missing tool.
    struct MissingToolPlanner;

    impl AgentPlanner for MissingToolPlanner {
        fn plan(
            &self,
            intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            if intermediate_steps.is_empty() {
                AgentDecision::Continue(vec![AgentAction::new(
                    "nonexistent_tool",
                    serde_json::json!({}),
                    "Calling missing tool",
                )])
            } else {
                let mut rv = HashMap::new();
                rv.insert(
                    "output".to_string(),
                    Value::String(intermediate_steps[0].1.clone()),
                );
                AgentDecision::Finish(AgentFinish::new(rv, "Done"))
            }
        }
    }

    /// A planner that continues with multiple actions at once.
    struct MultiActionPlanner;

    impl AgentPlanner for MultiActionPlanner {
        fn plan(
            &self,
            intermediate_steps: &[(AgentAction, String)],
            _input: &Value,
        ) -> AgentDecision {
            if intermediate_steps.is_empty() {
                AgentDecision::Continue(vec![
                    AgentAction::new("calculator", serde_json::json!({"a": 1}), "First"),
                    AgentAction::new("echo", serde_json::json!({"b": 2}), "Second"),
                ])
            } else {
                let mut rv = HashMap::new();
                rv.insert(
                    "output".to_string(),
                    Value::String("multi-done".to_string()),
                );
                AgentDecision::Finish(AgentFinish::new(rv, "Done"))
            }
        }
    }

    fn make_calculator_tool() -> ToolExecutor {
        let mut tools = ToolExecutor::new();
        tools.add_tool(
            "calculator".to_string(),
            Box::new(|_input: Value| Ok("4".to_string())),
        );
        tools
    }

    fn make_two_tools() -> ToolExecutor {
        let mut tools = ToolExecutor::new();
        tools.add_tool(
            "calculator".to_string(),
            Box::new(|_input: Value| Ok("4".to_string())),
        );
        tools.add_tool(
            "echo".to_string(),
            Box::new(|input: Value| {
                Ok(input
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("echoed")
                    .to_string())
            }),
        );
        tools
    }

    // ── Planner Test 1: AgentAction creation ──

    #[test]
    fn test_agent_action_new() {
        let action = AgentAction::new("tool_a", serde_json::json!({"key": "val"}), "my log");
        assert_eq!(action.tool_name, "tool_a");
        assert_eq!(action.tool_input, serde_json::json!({"key": "val"}));
        assert_eq!(action.log, "my log");
        assert!(action.message_log.is_empty());
    }

    // ── Planner Test 2: AgentAction with message log ──

    #[test]
    fn test_agent_action_with_message_log() {
        let action = AgentAction::new("tool_a", serde_json::json!(null), "log")
            .with_message_log(vec!["msg1".to_string(), "msg2".to_string()]);
        assert_eq!(action.message_log.len(), 2);
        assert_eq!(action.message_log[0], "msg1");
    }

    // ── Planner Test 3: AgentAction to_json ──

    #[test]
    fn test_agent_action_to_json() {
        let action = AgentAction::new("tool_b", serde_json::json!(42), "do thing");
        let json = action.to_json();
        assert_eq!(json["tool"], "tool_b");
        assert_eq!(json["tool_input"], 42);
        assert_eq!(json["log"], "do thing");
        assert!(json["message_log"].as_array().unwrap().is_empty());
    }

    // ── Planner Test 4: AgentFinish creation ──

    #[test]
    fn test_agent_finish_new() {
        let mut rv = HashMap::new();
        rv.insert("output".to_string(), Value::String("done".into()));
        let finish = AgentFinish::new(rv.clone(), "finished");
        assert_eq!(finish.return_values["output"], "done");
        assert_eq!(finish.log, "finished");
    }

    // ── Planner Test 5: AgentFinish to_json ──

    #[test]
    fn test_agent_finish_to_json() {
        let mut rv = HashMap::new();
        rv.insert("result".to_string(), Value::Number(42.into()));
        let finish = AgentFinish::new(rv, "completed");
        let json = finish.to_json();
        assert_eq!(json["return_values"]["result"], 42);
        assert_eq!(json["log"], "completed");
    }

    // ── Planner Test 6: AgentFinish empty return values ──

    #[test]
    fn test_agent_finish_empty_return_values() {
        let finish = AgentFinish::new(HashMap::new(), "empty");
        assert!(finish.return_values.is_empty());
        let json = finish.to_json();
        assert!(json["return_values"].as_object().unwrap().is_empty());
    }

    // ── Planner Test 7: PlannerAgentStep is_action ──

    #[test]
    fn test_planner_agent_step_is_action() {
        let step = PlannerAgentStep::Action(AgentAction::new("t", serde_json::json!(null), "l"));
        assert!(step.is_action());
        assert!(!step.is_finish());
    }

    // ── Planner Test 8: PlannerAgentStep is_finish ──

    #[test]
    fn test_planner_agent_step_is_finish() {
        let step = PlannerAgentStep::Finish(AgentFinish::new(HashMap::new(), "done"));
        assert!(step.is_finish());
        assert!(!step.is_action());
    }

    // ── Planner Test 9: ToolExecutor add and execute ──

    #[test]
    fn test_tool_executor_add_and_execute() {
        let mut te = ToolExecutor::new();
        te.add_tool(
            "adder".to_string(),
            Box::new(|input: Value| {
                let a = input["a"].as_i64().unwrap_or(0);
                let b = input["b"].as_i64().unwrap_or(0);
                Ok(format!("{}", a + b))
            }),
        );
        let result = te
            .execute("adder", serde_json::json!({"a": 3, "b": 7}))
            .unwrap();
        assert_eq!(result, "10");
    }

    // ── Planner Test 10: ToolExecutor missing tool ──

    #[test]
    fn test_tool_executor_missing_tool() {
        let te = ToolExecutor::new();
        let result = te.execute("missing", serde_json::json!({}));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Tool 'missing' not found"));
    }

    // ── Planner Test 11: ToolExecutor has_tool ──

    #[test]
    fn test_tool_executor_has_tool() {
        let mut te = ToolExecutor::new();
        assert!(!te.has_tool("calc"));
        te.add_tool("calc".to_string(), Box::new(|_| Ok("0".to_string())));
        assert!(te.has_tool("calc"));
        assert!(!te.has_tool("other"));
    }

    // ── Planner Test 12: ToolExecutor tool_names ──

    #[test]
    fn test_tool_executor_tool_names() {
        let mut te = ToolExecutor::new();
        te.add_tool("alpha".to_string(), Box::new(|_| Ok("a".to_string())));
        te.add_tool("beta".to_string(), Box::new(|_| Ok("b".to_string())));
        let mut names = te.tool_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    // ── Planner Test 13: ToolExecutor empty ──

    #[test]
    fn test_tool_executor_empty() {
        let te = ToolExecutor::new();
        assert!(te.tool_names().is_empty());
        assert!(!te.has_tool("anything"));
    }

    // ── Planner Test 14: ToolExecutor default ──

    #[test]
    fn test_tool_executor_default() {
        let te = ToolExecutor::default();
        assert!(te.tool_names().is_empty());
    }

    // ── Planner Test 15: ExecutorConfig defaults ──

    #[test]
    fn test_executor_config_defaults() {
        let config = ExecutorConfig::new();
        assert_eq!(config.max_iterations, 15);
        assert_eq!(config.max_execution_time_ms, None);
        assert!(!config.return_intermediate_steps);
        assert_eq!(config.early_stopping_method, EarlyStoppingMethod::Force);
    }

    // ── Planner Test 16: ExecutorConfig builder ──

    #[test]
    fn test_executor_config_builder() {
        let config = ExecutorConfig::new()
            .max_iterations(5)
            .max_execution_time_ms(1000)
            .return_intermediate_steps(true)
            .early_stopping_method(EarlyStoppingMethod::GenerateResponse);
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.max_execution_time_ms, Some(1000));
        assert!(config.return_intermediate_steps);
        assert_eq!(
            config.early_stopping_method,
            EarlyStoppingMethod::GenerateResponse
        );
    }

    // ── Planner Test 17: ExecutorConfig default trait ──

    #[test]
    fn test_executor_config_default_trait() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_iterations, 15);
    }

    // ── Planner Test 18: EarlyStoppingMethod variants ──

    #[test]
    fn test_early_stopping_method_variants() {
        let force = EarlyStoppingMethod::Force;
        let gen = EarlyStoppingMethod::GenerateResponse;
        assert_ne!(force, gen);
        assert_eq!(force, EarlyStoppingMethod::Force);
        assert_eq!(gen, EarlyStoppingMethod::GenerateResponse);
    }

    // ── Planner Test 19: Immediate finish planner ──

    #[test]
    fn test_planner_immediate_finish() {
        let planner = Box::new(ImmediateFinishPlanner);
        let tools = ToolExecutor::new();
        let config = ExecutorConfig::new();
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("hello"))
            .expect("should succeed");
        assert_eq!(result.output["output"], "hello");
        assert_eq!(result.iterations, 1);
        assert!(result.intermediate_steps.is_empty());
    }

    // ── Planner Test 20: One action then finish ──

    #[test]
    fn test_planner_one_action_then_finish() {
        let planner = Box::new(OneActionPlanner);
        let tools = make_calculator_tool();
        let config = ExecutorConfig::new().return_intermediate_steps(true);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("compute"))
            .expect("should succeed");
        assert_eq!(result.output["output"], "4");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.intermediate_steps.len(), 1);
        assert_eq!(result.intermediate_steps[0].0.tool_name, "calculator");
        assert_eq!(result.intermediate_steps[0].1, "4");
    }

    // ── Planner Test 21: Multi-step execution ──

    #[test]
    fn test_planner_multi_step_execution() {
        let planner = Box::new(TwoActionPlanner);
        let tools = make_two_tools();
        let config = ExecutorConfig::new().return_intermediate_steps(true);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("do things"))
            .expect("should succeed");
        assert_eq!(result.output["output"], "4 and hello");
        assert_eq!(result.iterations, 3);
        assert_eq!(result.intermediate_steps.len(), 2);
        assert_eq!(result.intermediate_steps[0].0.tool_name, "calculator");
        assert_eq!(result.intermediate_steps[1].0.tool_name, "echo");
    }

    // ── Planner Test 22: Max iterations limit ──

    #[test]
    fn test_planner_max_iterations_limit() {
        let planner = Box::new(NeverFinishPlanner);
        let tools = make_calculator_tool();
        let config = ExecutorConfig::new().max_iterations(3);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("loop"))
            .expect("should stop with Force");
        assert!(result.output.as_str().unwrap().contains("max iterations"));
        assert_eq!(result.iterations, 3);
    }

    // ── Planner Test 23: Error from planner ──

    #[test]
    fn test_planner_error() {
        let planner = Box::new(ErrorPlanner);
        let tools = ToolExecutor::new();
        let config = ExecutorConfig::new();
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor.execute(serde_json::json!("go"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Planning failed"));
    }

    // ── Planner Test 24: Missing tool handled gracefully ──

    #[test]
    fn test_planner_missing_tool_graceful() {
        let planner = Box::new(MissingToolPlanner);
        let tools = ToolExecutor::new(); // no tools registered
        let config = ExecutorConfig::new().return_intermediate_steps(true);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("try missing"))
            .expect("should succeed");
        assert!(result.output["output"].as_str().unwrap().contains("Error"));
        assert_eq!(result.intermediate_steps.len(), 1);
        assert!(result.intermediate_steps[0].1.contains("not found"));
    }

    // ── Planner Test 25: Intermediate steps not returned when disabled ──

    #[test]
    fn test_planner_no_intermediate_steps_when_disabled() {
        let planner = Box::new(OneActionPlanner);
        let tools = make_calculator_tool();
        let config = ExecutorConfig::new().return_intermediate_steps(false);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("compute"))
            .expect("should succeed");
        assert!(result.intermediate_steps.is_empty());
    }

    // ── Planner Test 26: ExecutorResult to_json ──

    #[test]
    fn test_executor_result_to_json() {
        let result = ExecutorResult {
            output: Value::String("answer".into()),
            intermediate_steps: vec![(
                AgentAction::new("tool_x", serde_json::json!(1), "step log"),
                "observation_x".to_string(),
            )],
            iterations: 2,
            duration_ms: 150,
        };
        let json = result.to_json();
        assert_eq!(json["output"], "answer");
        assert_eq!(json["iterations"], 2);
        assert_eq!(json["duration_ms"], 150);
        let steps = json["intermediate_steps"].as_array().unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0]["action"]["tool"], "tool_x");
        assert_eq!(steps[0]["observation"], "observation_x");
    }

    // ── Planner Test 27: ExecutorResult empty steps to_json ──

    #[test]
    fn test_executor_result_empty_to_json() {
        let result = ExecutorResult {
            output: Value::Null,
            intermediate_steps: vec![],
            iterations: 0,
            duration_ms: 0,
        };
        let json = result.to_json();
        assert_eq!(json["output"], Value::Null);
        assert!(json["intermediate_steps"].as_array().unwrap().is_empty());
    }

    // ── Planner Test 28: Multi-action in single step ──

    #[test]
    fn test_planner_multi_action_single_step() {
        let planner = Box::new(MultiActionPlanner);
        let tools = make_two_tools();
        let config = ExecutorConfig::new().return_intermediate_steps(true);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("multi"))
            .expect("should succeed");
        assert_eq!(result.output["output"], "multi-done");
        // Two actions in first iteration, then finish in second
        assert_eq!(result.iterations, 2);
        assert_eq!(result.intermediate_steps.len(), 2);
    }

    // ── Planner Test 29: Duration is tracked ──

    #[test]
    fn test_planner_duration_tracked() {
        let planner = Box::new(ImmediateFinishPlanner);
        let tools = ToolExecutor::new();
        let config = ExecutorConfig::new();
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("quick"))
            .expect("should succeed");
        // Duration should be >= 0 (just check it's a reasonable number)
        assert!(result.duration_ms < 1000);
    }

    // ── Planner Test 30: AgentDecision Continue variant ──

    #[test]
    fn test_agent_decision_continue() {
        let decision = AgentDecision::Continue(vec![AgentAction::new(
            "tool",
            serde_json::json!(null),
            "log",
        )]);
        match decision {
            AgentDecision::Continue(actions) => assert_eq!(actions.len(), 1),
            _ => panic!("Expected Continue"),
        }
    }

    // ── Planner Test 31: AgentDecision Finish variant ──

    #[test]
    fn test_agent_decision_finish() {
        let decision = AgentDecision::Finish(AgentFinish::new(HashMap::new(), "done"));
        match decision {
            AgentDecision::Finish(f) => assert_eq!(f.log, "done"),
            _ => panic!("Expected Finish"),
        }
    }

    // ── Planner Test 32: AgentDecision Error variant ──

    #[test]
    fn test_agent_decision_error() {
        let decision = AgentDecision::Error("oops".to_string());
        match decision {
            AgentDecision::Error(msg) => assert_eq!(msg, "oops"),
            _ => panic!("Expected Error"),
        }
    }

    // ── Planner Test 33: ToolExecutor overwrite existing tool ──

    #[test]
    fn test_tool_executor_overwrite_tool() {
        let mut te = ToolExecutor::new();
        te.add_tool("calc".to_string(), Box::new(|_| Ok("old".to_string())));
        te.add_tool("calc".to_string(), Box::new(|_| Ok("new".to_string())));
        let result = te.execute("calc", serde_json::json!(null)).unwrap();
        assert_eq!(result, "new");
    }

    // ── Planner Test 34: ToolExecutor tool returns error ──

    #[test]
    fn test_tool_executor_tool_returns_error() {
        let mut te = ToolExecutor::new();
        te.add_tool(
            "fail_tool".to_string(),
            Box::new(|_| Err(CognisError::Other("tool broke".into()))),
        );
        let result = te.execute("fail_tool", serde_json::json!(null));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("tool broke"));
    }

    // ── Planner Test 35: AgentAction serialization roundtrip ──

    #[test]
    fn test_agent_action_serialization() {
        let action = AgentAction::new("my_tool", serde_json::json!({"x": 1}), "log msg")
            .with_message_log(vec!["m1".to_string()]);
        let json_str = serde_json::to_string(&action).unwrap();
        let deserialized: AgentAction = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.tool_name, "my_tool");
        assert_eq!(deserialized.tool_input, serde_json::json!({"x": 1}));
        assert_eq!(deserialized.log, "log msg");
        assert_eq!(deserialized.message_log, vec!["m1"]);
    }

    // ── Planner Test 36: AgentFinish serialization roundtrip ──

    #[test]
    fn test_agent_finish_serialization() {
        let mut rv = HashMap::new();
        rv.insert("key".to_string(), Value::Bool(true));
        let finish = AgentFinish::new(rv, "final");
        let json_str = serde_json::to_string(&finish).unwrap();
        let deserialized: AgentFinish = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.return_values["key"], true);
        assert_eq!(deserialized.log, "final");
    }

    // ── Planner Test 37: PlannerAgentStep serialization ──

    #[test]
    fn test_planner_agent_step_serialization() {
        let step = PlannerAgentStep::Action(AgentAction::new("t", serde_json::json!(1), "l"));
        let json_str = serde_json::to_string(&step).unwrap();
        let deserialized: PlannerAgentStep = serde_json::from_str(&json_str).unwrap();
        assert!(deserialized.is_action());
    }

    // ── Planner Test 38: Finish PlannerAgentStep serialization ──

    #[test]
    fn test_planner_agent_step_finish_serialization() {
        let step = PlannerAgentStep::Finish(AgentFinish::new(HashMap::new(), "end"));
        let json_str = serde_json::to_string(&step).unwrap();
        let deserialized: PlannerAgentStep = serde_json::from_str(&json_str).unwrap();
        assert!(deserialized.is_finish());
    }

    // ── Planner Test 39: Max iterations with GenerateResponse ──

    #[test]
    fn test_planner_max_iterations_generate_response() {
        // NeverFinishPlanner always continues, so GenerateResponse will also
        // not produce a finish, and we expect the fallback message.
        let planner = Box::new(NeverFinishPlanner);
        let tools = make_calculator_tool();
        let config = ExecutorConfig::new()
            .max_iterations(2)
            .early_stopping_method(EarlyStoppingMethod::GenerateResponse);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("loop"))
            .expect("should stop");
        assert!(result.output.as_str().unwrap().contains("max iterations"));
    }

    // ── Planner Test 40: Max iterations 1 ──

    #[test]
    fn test_planner_max_iterations_one() {
        let planner = Box::new(NeverFinishPlanner);
        let tools = make_calculator_tool();
        let config = ExecutorConfig::new().max_iterations(1);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("one"))
            .expect("should stop at 1 iteration");
        assert_eq!(result.iterations, 1);
    }

    // ── Planner Test 41: Zero max iterations ──

    #[test]
    fn test_planner_max_iterations_zero() {
        let planner = Box::new(ImmediateFinishPlanner);
        let tools = ToolExecutor::new();
        let config = ExecutorConfig::new().max_iterations(0);
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let result = executor
            .execute(serde_json::json!("zero"))
            .expect("should stop immediately");
        assert_eq!(result.iterations, 0);
        assert!(result.output.as_str().unwrap().contains("max iterations"));
    }

    // ── Planner Test 42: ToolExecutor with many tools ──

    #[test]
    fn test_tool_executor_many_tools() {
        let mut te = ToolExecutor::new();
        for i in 0..10 {
            let name = format!("tool_{}", i);
            let expected = format!("result_{}", i);
            te.add_tool(name, Box::new(move |_| Ok(expected.clone())));
        }
        assert_eq!(te.tool_names().len(), 10);
        assert!(te.has_tool("tool_5"));
        assert_eq!(
            te.execute("tool_7", serde_json::json!(null)).unwrap(),
            "result_7"
        );
    }

    // ── Planner Test 43: AgentAction to_json with message_log ──

    #[test]
    fn test_agent_action_to_json_with_message_log() {
        let action = AgentAction::new("t", serde_json::json!(null), "l")
            .with_message_log(vec!["a".into(), "b".into()]);
        let json = action.to_json();
        let msgs = json["message_log"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], "a");
        assert_eq!(msgs[1], "b");
    }

    // ── Planner Test 44: ExecutorResult with multiple steps ──

    #[test]
    fn test_executor_result_multiple_steps_to_json() {
        let result = ExecutorResult {
            output: Value::String("final".into()),
            intermediate_steps: vec![
                (
                    AgentAction::new("t1", serde_json::json!(1), "s1"),
                    "obs1".to_string(),
                ),
                (
                    AgentAction::new("t2", serde_json::json!(2), "s2"),
                    "obs2".to_string(),
                ),
                (
                    AgentAction::new("t3", serde_json::json!(3), "s3"),
                    "obs3".to_string(),
                ),
            ],
            iterations: 4,
            duration_ms: 500,
        };
        let json = result.to_json();
        assert_eq!(json["intermediate_steps"].as_array().unwrap().len(), 3);
        assert_eq!(json["iterations"], 4);
    }

    // ── Planner Test 45: Tool error propagated as observation ──

    #[test]
    fn test_planner_tool_error_in_observation() {
        /// Planner that calls a tool that errors, then finishes with the observation.
        struct FailingToolPlanner;
        impl AgentPlanner for FailingToolPlanner {
            fn plan(
                &self,
                intermediate_steps: &[(AgentAction, String)],
                _input: &Value,
            ) -> AgentDecision {
                if intermediate_steps.is_empty() {
                    AgentDecision::Continue(vec![AgentAction::new(
                        "fail_tool",
                        serde_json::json!({}),
                        "calling fail",
                    )])
                } else {
                    let mut rv = HashMap::new();
                    rv.insert(
                        "error".to_string(),
                        Value::String(intermediate_steps[0].1.clone()),
                    );
                    AgentDecision::Finish(AgentFinish::new(rv, "done"))
                }
            }
        }

        let mut tools = ToolExecutor::new();
        tools.add_tool(
            "fail_tool".to_string(),
            Box::new(|_| Err(CognisError::Other("kaboom".into()))),
        );
        let config = ExecutorConfig::new().return_intermediate_steps(true);
        let executor = PlannerAgentExecutor::new(Box::new(FailingToolPlanner), tools, config);

        let result = executor
            .execute(serde_json::json!("go"))
            .expect("should succeed");
        assert!(result.output["error"].as_str().unwrap().contains("kaboom"));
        assert!(result.intermediate_steps[0].1.contains("Error"));
    }

    // ── Planner Test 46: Planner with complex JSON input ──

    #[test]
    fn test_planner_complex_json_input() {
        let planner = Box::new(ImmediateFinishPlanner);
        let tools = ToolExecutor::new();
        let config = ExecutorConfig::new();
        let executor = PlannerAgentExecutor::new(planner, tools, config);

        let input = serde_json::json!({
            "query": "What is the weather?",
            "context": {"location": "SF", "units": "celsius"},
            "history": ["msg1", "msg2"]
        });
        let result = executor.execute(input.clone()).expect("should succeed");
        assert_eq!(result.output["output"], input);
    }

    // ── Planner Test 47: AgentFinish with multiple return values ──

    #[test]
    fn test_agent_finish_multiple_return_values() {
        let mut rv = HashMap::new();
        rv.insert("output".to_string(), Value::String("answer".into()));
        rv.insert("confidence".to_string(), serde_json::json!(0.95));
        rv.insert("sources".to_string(), serde_json::json!(["doc1", "doc2"]));
        let finish = AgentFinish::new(rv, "multi-value finish");
        assert_eq!(finish.return_values.len(), 3);
        let json = finish.to_json();
        assert_eq!(json["return_values"]["confidence"], 0.95);
        assert_eq!(json["return_values"]["sources"][0], "doc1");
    }
}
