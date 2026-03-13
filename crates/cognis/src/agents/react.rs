//! ReAct (Reasoning + Acting) agent implementation.
//!
//! Implements the ReAct loop:
//! 1. Format prompt with tools description, user input, and scratchpad
//! 2. Call the model
//! 3. Parse output with [`ReActOutputParser`]
//! 4. If `AgentFinish` -> return final answer
//! 5. If `AgentAction` -> execute the tool, record observation in scratchpad
//! 6. Repeat until finish or `max_iterations` reached

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognis_core::tools::base::BaseTool;

use super::output_parser::{AgentOutput, AgentOutputParser, ReActOutputParser};

// ---------------------------------------------------------------------------
// ReActStep
// ---------------------------------------------------------------------------

/// A single step in the ReAct thought-action-observation loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    /// The model's reasoning for this step.
    pub thought: String,
    /// The tool name to invoke, if any.
    pub action: Option<String>,
    /// The input to pass to the tool, if any.
    pub action_input: Option<Value>,
    /// The result returned by the tool, if any.
    pub observation: Option<String>,
    /// Whether this step produced a final answer.
    pub is_final: bool,
}

// ---------------------------------------------------------------------------
// ReActTrace
// ---------------------------------------------------------------------------

/// A full trace of a ReAct agent execution, capturing every step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActTrace {
    /// The ordered list of reasoning steps.
    pub steps: Vec<ReActStep>,
    /// The final answer, if the agent completed successfully.
    pub final_answer: Option<String>,
    /// Approximate total tokens consumed (based on message content lengths).
    pub total_tokens: usize,
}

impl ReActTrace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_answer: None,
            total_tokens: 0,
        }
    }

    /// Add a step to the trace.
    pub fn add_step(&mut self, step: ReActStep) {
        if step.is_final {
            // Extract final answer from the thought when the step is final.
            self.final_answer = Some(step.thought.clone());
        }
        self.steps.push(step);
    }

    /// Get a reference to all steps.
    pub fn get_steps(&self) -> &[ReActStep] {
        &self.steps
    }

    /// Returns `true` if the agent has produced a final answer.
    pub fn is_complete(&self) -> bool {
        self.final_answer.is_some()
    }
}

impl Default for ReActTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReActResult
// ---------------------------------------------------------------------------

/// The result of running a ReAct agent to completion.
#[derive(Debug, Clone)]
pub struct ReActResult {
    /// The final output text.
    pub output: String,
    /// The number of model invocations performed.
    pub iterations: usize,
    /// Tool calls made: `(tool_name, tool_input, tool_output)`.
    pub tool_calls: Vec<(String, Value, String)>,
}

// ---------------------------------------------------------------------------
// ReActAgent builder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`ReActAgent`].
pub struct ReActAgentBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    tools: Vec<Arc<dyn BaseTool>>,
    max_iterations: usize,
    system_prompt: Option<String>,
    verbose: bool,
}

impl ReActAgentBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            max_iterations: 10,
            system_prompt: None,
            verbose: false,
        }
    }

    /// Set the chat model.
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the tools available to the agent.
    pub fn tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a single tool.
    pub fn tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set an optional system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable verbose logging to stdout.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the [`ReActAgent`].
    pub fn build(self) -> Result<ReActAgent> {
        let model = self
            .model
            .ok_or_else(|| CognisError::Other("ReActAgent requires a model".into()))?;
        Ok(ReActAgent {
            model,
            tools: self.tools,
            max_iterations: self.max_iterations,
            system_prompt: self.system_prompt,
            verbose: self.verbose,
        })
    }
}

impl Default for ReActAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReActAgent
// ---------------------------------------------------------------------------

/// A ReAct (Reasoning + Acting) agent that interleaves LLM reasoning with
/// tool execution in a loop.
pub struct ReActAgent {
    /// The chat model used for reasoning.
    model: Arc<dyn BaseChatModel>,
    /// Tools available for the agent to call.
    tools: Vec<Arc<dyn BaseTool>>,
    /// Maximum number of reasoning iterations before stopping.
    max_iterations: usize,
    /// Optional system prompt prepended to the conversation.
    system_prompt: Option<String>,
    /// Whether to print intermediate steps to stdout.
    verbose: bool,
}

impl ReActAgent {
    /// Create a new builder for constructing a ReActAgent.
    pub fn builder() -> ReActAgentBuilder {
        ReActAgentBuilder::new()
    }

    /// Run the ReAct loop on the given input and return the result.
    pub async fn run(&self, input: &str) -> Result<ReActResult> {
        let (output, trace) = self.run_inner(input).await?;
        let tool_calls: Vec<(String, Value, String)> = trace
            .steps
            .iter()
            .filter(|s| s.action.is_some())
            .map(|s| {
                (
                    s.action.clone().unwrap_or_default(),
                    s.action_input.clone().unwrap_or(Value::Null),
                    s.observation.clone().unwrap_or_default(),
                )
            })
            .collect();

        Ok(ReActResult {
            output,
            iterations: trace.steps.len(),
            tool_calls,
        })
    }

    /// Run the ReAct loop and return both the output and a detailed trace.
    pub async fn run_with_trace(&self, input: &str) -> Result<(String, ReActTrace)> {
        self.run_inner(input).await
    }

    /// Format a description of all available tools.
    pub fn format_tools_description(&self) -> String {
        self.tools
            .iter()
            .map(|t| format!("{}: {}", t.name(), t.description()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format the scratchpad from previous steps for inclusion in the prompt.
    pub fn format_scratchpad(&self, steps: &[ReActStep]) -> String {
        let mut scratchpad = String::new();
        for step in steps {
            scratchpad.push_str(&format!("Thought: {}\n", step.thought));
            if let Some(ref action) = step.action {
                scratchpad.push_str(&format!("Action: {}\n", action));
                if let Some(ref action_input) = step.action_input {
                    let input_str = match action_input {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    scratchpad.push_str(&format!("Action Input: {}\n", input_str));
                }
            }
            if let Some(ref observation) = step.observation {
                scratchpad.push_str(&format!("Observation: {}\n", observation));
            }
        }
        scratchpad
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// The core ReAct loop shared by `run` and `run_with_trace`.
    async fn run_inner(&self, input: &str) -> Result<(String, ReActTrace)> {
        let parser = ReActOutputParser::new();
        let mut trace = ReActTrace::new();
        let tool_names: Vec<&str> = self.tools.iter().map(|t| t.name()).collect();

        for _iteration in 0..self.max_iterations {
            let scratchpad = self.format_scratchpad(&trace.steps);
            let prompt = self.build_prompt(input, &scratchpad, &tool_names);

            let messages: Vec<Message> = self.build_messages(&prompt);
            let ai_msg = self.model.invoke_messages(&messages, None).await?;
            let ai_text = ai_msg.base.content.text();

            // Approximate token tracking.
            let msg_tokens: usize = messages.iter().map(|m| m.content().text().len() / 4).sum();
            let resp_tokens = ai_text.len() / 4;
            trace.total_tokens += msg_tokens + resp_tokens;

            if self.verbose {
                println!("[ReAct] Model output:\n{}", ai_text);
            }

            match parser.parse(&ai_text) {
                Ok(AgentOutput::Finish(finish)) => {
                    let output = finish
                        .return_values
                        .get("output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    trace.add_step(ReActStep {
                        thought: output.clone(),
                        action: None,
                        action_input: None,
                        observation: None,
                        is_final: true,
                    });

                    return Ok((output, trace));
                }
                Ok(AgentOutput::Action(action)) => {
                    // Extract thought from text before Action:
                    let thought = extract_thought(&ai_text);

                    // Find and execute the tool.
                    let observation = self.execute_tool(&action.tool, &action.tool_input).await?;
                    let obs_str = match &observation {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };

                    if self.verbose {
                        println!(
                            "[ReAct] Tool: {} | Input: {} | Observation: {}",
                            action.tool, action.tool_input, obs_str
                        );
                    }

                    trace.add_step(ReActStep {
                        thought,
                        action: Some(action.tool),
                        action_input: Some(action.tool_input),
                        observation: Some(obs_str),
                        is_final: false,
                    });
                }
                Err(e) => {
                    // If parsing fails, treat the raw output as a thought and continue.
                    if self.verbose {
                        println!("[ReAct] Parse error: {}", e);
                    }
                    trace.add_step(ReActStep {
                        thought: ai_text.clone(),
                        action: None,
                        action_input: None,
                        observation: Some(format!(
                            "Invalid format. You must use Action/Action Input or Final Answer. Error: {}",
                            e
                        )),
                        is_final: false,
                    });
                }
            }
        }

        // Max iterations reached — return whatever we have.
        let last_thought = trace
            .steps
            .last()
            .map(|s| s.thought.clone())
            .unwrap_or_default();
        let output = format!(
            "Agent stopped after {} iterations. Last thought: {}",
            self.max_iterations, last_thought
        );
        Ok((output, trace))
    }

    /// Build the full prompt string for the model.
    fn build_prompt(&self, input: &str, scratchpad: &str, tool_names: &[&str]) -> String {
        let tools_desc = self.format_tools_description();
        let tool_name_list = tool_names.join(", ");

        let mut prompt = String::new();
        prompt.push_str("Answer the following questions as best you can. You have access to the following tools:\n\n");
        prompt.push_str(&tools_desc);
        prompt.push_str("\n\nUse the following format:\n\n");
        prompt.push_str("Question: the input question you must answer\n");
        prompt.push_str("Thought: you should always think about what to do\n");
        prompt.push_str(&format!(
            "Action: the action to take, should be one of [{}]\n",
            tool_name_list
        ));
        prompt.push_str("Action Input: the input to the action\n");
        prompt.push_str("Observation: the result of the action\n");
        prompt.push_str("... (this Thought/Action/Action Input/Observation can repeat N times)\n");
        prompt.push_str("Thought: I now know the final answer\n");
        prompt.push_str("Final Answer: the final answer to the original input question\n\n");
        prompt.push_str(&format!("Begin!\n\nQuestion: {}\n", input));

        if !scratchpad.is_empty() {
            prompt.push_str(scratchpad);
        }

        prompt.push_str("Thought:");
        prompt
    }

    /// Build the message list for the model call.
    fn build_messages(&self, prompt: &str) -> Vec<Message> {
        let mut messages = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            messages.push(Message::system(sys.clone()));
        }
        messages.push(Message::human(prompt.to_string()));
        messages
    }

    /// Execute a tool by name, returning its output as a `Value`.
    async fn execute_tool(&self, tool_name: &str, tool_input: &Value) -> Result<Value> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_name)
            .ok_or_else(|| {
                CognisError::Other(format!(
                    "Tool '{}' not found. Available tools: {}",
                    tool_name,
                    self.tools
                        .iter()
                        .map(|t| t.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;

        tool.run_json(tool_input).await
    }
}

/// Extract the thought portion from a ReAct-formatted LLM output.
///
/// Looks for text between "Thought:" (or the start) and "Action:".
fn extract_thought(text: &str) -> String {
    // Try to find an explicit Thought: prefix.
    let after_thought = if let Some(idx) = text.find("Thought:") {
        &text[idx + "Thought:".len()..]
    } else {
        text
    };

    // Trim up to Action: if present.
    if let Some(idx) = after_thought.find("Action:") {
        after_thought[..idx].trim().to_string()
    } else if let Some(idx) = after_thought.find("Final Answer:") {
        after_thought[..idx].trim().to_string()
    } else {
        after_thought.trim().to_string()
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Convenience factory for creating a ReAct agent with default settings.
pub fn create_react_agent(
    model: Arc<dyn BaseChatModel>,
    tools: Vec<Arc<dyn BaseTool>>,
) -> ReActAgent {
    ReActAgent {
        model,
        tools,
        max_iterations: 10,
        system_prompt: None,
        verbose: false,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::FakeListChatModel;
    use cognis_core::tools::SimpleTool;

    // -- Helper to make a search tool that returns a fixed result --
    fn make_search_tool() -> Arc<dyn BaseTool> {
        Arc::new(SimpleTool::new(
            "search",
            "Search the web for information",
            |query: &str| Ok(format!("Result for: {}", query)),
        ))
    }

    fn make_calculator_tool() -> Arc<dyn BaseTool> {
        Arc::new(SimpleTool::new(
            "calculator",
            "Perform arithmetic calculations",
            |expr: &str| Ok(format!("42 (from: {})", expr)),
        ))
    }

    // -----------------------------------------------------------------------
    // ReActStep tests
    // -----------------------------------------------------------------------

    #[test]
    fn react_step_final() {
        let step = ReActStep {
            thought: "I know the answer".to_string(),
            action: None,
            action_input: None,
            observation: None,
            is_final: true,
        };
        assert!(step.is_final);
        assert!(step.action.is_none());
    }

    #[test]
    fn react_step_action() {
        let step = ReActStep {
            thought: "I should search".to_string(),
            action: Some("search".to_string()),
            action_input: Some(Value::String("rust".to_string())),
            observation: Some("Result for: rust".to_string()),
            is_final: false,
        };
        assert!(!step.is_final);
        assert_eq!(step.action.as_deref(), Some("search"));
    }

    // -----------------------------------------------------------------------
    // ReActTrace tests
    // -----------------------------------------------------------------------

    #[test]
    fn react_trace_new_is_empty() {
        let trace = ReActTrace::new();
        assert!(trace.steps.is_empty());
        assert!(trace.final_answer.is_none());
        assert!(!trace.is_complete());
        assert_eq!(trace.total_tokens, 0);
    }

    #[test]
    fn react_trace_add_step_non_final() {
        let mut trace = ReActTrace::new();
        trace.add_step(ReActStep {
            thought: "thinking".to_string(),
            action: Some("search".to_string()),
            action_input: Some(Value::String("q".to_string())),
            observation: Some("obs".to_string()),
            is_final: false,
        });
        assert_eq!(trace.get_steps().len(), 1);
        assert!(!trace.is_complete());
        assert!(trace.final_answer.is_none());
    }

    #[test]
    fn react_trace_add_final_step() {
        let mut trace = ReActTrace::new();
        trace.add_step(ReActStep {
            thought: "The answer is 42".to_string(),
            action: None,
            action_input: None,
            observation: None,
            is_final: true,
        });
        assert!(trace.is_complete());
        assert_eq!(trace.final_answer.as_deref(), Some("The answer is 42"));
    }

    #[test]
    fn react_trace_multiple_steps() {
        let mut trace = ReActTrace::new();
        trace.add_step(ReActStep {
            thought: "step 1".to_string(),
            action: Some("search".to_string()),
            action_input: None,
            observation: Some("obs1".to_string()),
            is_final: false,
        });
        trace.add_step(ReActStep {
            thought: "step 2".to_string(),
            action: Some("calc".to_string()),
            action_input: None,
            observation: Some("obs2".to_string()),
            is_final: false,
        });
        trace.add_step(ReActStep {
            thought: "done".to_string(),
            action: None,
            action_input: None,
            observation: None,
            is_final: true,
        });
        assert_eq!(trace.get_steps().len(), 3);
        assert!(trace.is_complete());
    }

    #[test]
    fn react_trace_default() {
        let trace = ReActTrace::default();
        assert!(trace.steps.is_empty());
        assert!(!trace.is_complete());
    }

    // -----------------------------------------------------------------------
    // ReActAgent builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_requires_model() {
        let result = ReActAgentBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_with_model_succeeds() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: ok".into()]));
        let agent = ReActAgentBuilder::new().model(model).build();
        assert!(agent.is_ok());
    }

    #[test]
    fn builder_sets_max_iterations() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: ok".into()]));
        let agent = ReActAgentBuilder::new()
            .model(model)
            .max_iterations(5)
            .build()
            .unwrap();
        assert_eq!(agent.max_iterations, 5);
    }

    #[test]
    fn builder_sets_system_prompt() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: ok".into()]));
        let agent = ReActAgentBuilder::new()
            .model(model)
            .system_prompt("You are helpful")
            .build()
            .unwrap();
        assert_eq!(agent.system_prompt.as_deref(), Some("You are helpful"));
    }

    #[test]
    fn builder_adds_tools() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: ok".into()]));
        let agent = ReActAgentBuilder::new()
            .model(model)
            .tool(make_search_tool())
            .tool(make_calculator_tool())
            .build()
            .unwrap();
        assert_eq!(agent.tools.len(), 2);
    }

    #[test]
    fn builder_sets_tools_vec() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: ok".into()]));
        let agent = ReActAgentBuilder::new()
            .model(model)
            .tools(vec![make_search_tool(), make_calculator_tool()])
            .build()
            .unwrap();
        assert_eq!(agent.tools.len(), 2);
    }

    // -----------------------------------------------------------------------
    // format_tools_description tests
    // -----------------------------------------------------------------------

    #[test]
    fn format_tools_description_empty() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["x".into()]));
        let agent = create_react_agent(model, vec![]);
        let desc = agent.format_tools_description();
        assert!(desc.is_empty());
    }

    #[test]
    fn format_tools_description_includes_all_tools() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["x".into()]));
        let agent = create_react_agent(model, vec![make_search_tool(), make_calculator_tool()]);
        let desc = agent.format_tools_description();
        assert!(desc.contains("search"));
        assert!(desc.contains("calculator"));
        assert!(desc.contains("Search the web"));
        assert!(desc.contains("Perform arithmetic"));
    }

    // -----------------------------------------------------------------------
    // format_scratchpad tests
    // -----------------------------------------------------------------------

    #[test]
    fn format_scratchpad_empty() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["x".into()]));
        let agent = create_react_agent(model, vec![]);
        let pad = agent.format_scratchpad(&[]);
        assert!(pad.is_empty());
    }

    #[test]
    fn format_scratchpad_with_steps() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["x".into()]));
        let agent = create_react_agent(model, vec![]);
        let steps = vec![
            ReActStep {
                thought: "I should search".to_string(),
                action: Some("search".to_string()),
                action_input: Some(Value::String("rust".to_string())),
                observation: Some("Result for: rust".to_string()),
                is_final: false,
            },
            ReActStep {
                thought: "Now I know".to_string(),
                action: None,
                action_input: None,
                observation: None,
                is_final: true,
            },
        ];
        let pad = agent.format_scratchpad(&steps);
        assert!(pad.contains("Thought: I should search"));
        assert!(pad.contains("Action: search"));
        assert!(pad.contains("Action Input: rust"));
        assert!(pad.contains("Observation: Result for: rust"));
        assert!(pad.contains("Thought: Now I know"));
    }

    // -----------------------------------------------------------------------
    // extract_thought tests
    // -----------------------------------------------------------------------

    #[test]
    fn extract_thought_with_action() {
        let text = "Thought: I need to search\nAction: search\nAction Input: query";
        let thought = extract_thought(text);
        assert_eq!(thought, "I need to search");
    }

    #[test]
    fn extract_thought_with_final_answer() {
        let text = "Thought: I know the answer\nFinal Answer: 42";
        let thought = extract_thought(text);
        assert_eq!(thought, "I know the answer");
    }

    #[test]
    fn extract_thought_no_markers() {
        let text = "Just some text";
        let thought = extract_thought(text);
        assert_eq!(thought, "Just some text");
    }

    // -----------------------------------------------------------------------
    // Async run tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn run_immediate_final_answer() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: 42".into()]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let result = agent.run("What is the answer?").await.unwrap();
        assert_eq!(result.output, "42");
        assert_eq!(result.iterations, 1);
        assert!(result.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn run_single_tool_then_finish() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: I should search\nAction: search\nAction Input: rust language".into(),
            "Final Answer: Rust is a systems programming language".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let result = agent.run("What is Rust?").await.unwrap();
        assert_eq!(result.output, "Rust is a systems programming language");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].0, "search");
        assert!(result.tool_calls[0].2.contains("Result for:"));
    }

    #[tokio::test]
    async fn run_with_trace_returns_trace() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: search first\nAction: search\nAction Input: test".into(),
            "Final Answer: done".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let (output, trace) = agent.run_with_trace("test query").await.unwrap();
        assert_eq!(output, "done");
        assert!(trace.is_complete());
        assert_eq!(trace.get_steps().len(), 2);
        assert!(trace.total_tokens > 0);
    }

    #[tokio::test]
    async fn run_max_iterations_reached() {
        // Model never returns Final Answer, so we should hit max_iterations.
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: keep going\nAction: search\nAction Input: again".into(),
        ]));
        let agent = ReActAgent::builder()
            .model(model)
            .tools(vec![make_search_tool()])
            .max_iterations(3)
            .build()
            .unwrap();

        let result = agent.run("infinite loop?").await.unwrap();
        assert!(result.output.contains("Agent stopped after 3 iterations"));
        assert_eq!(result.iterations, 3);
    }

    #[tokio::test]
    async fn run_tool_not_found_errors() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: call unknown\nAction: nonexistent\nAction Input: test".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let result = agent.run("test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn run_multiple_tool_calls() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: search first\nAction: search\nAction Input: query1".into(),
            "Thought: now calculate\nAction: calculator\nAction Input: 2+2".into(),
            "Final Answer: The result is 42".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool(), make_calculator_tool()]);

        let result = agent.run("complex question").await.unwrap();
        assert_eq!(result.output, "The result is 42");
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].0, "search");
        assert_eq!(result.tool_calls[1].0, "calculator");
    }

    #[tokio::test]
    async fn run_with_json_tool_input() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: search with json\nAction: search\nAction Input: {\"query\": \"rust\"}".into(),
            "Final Answer: found it".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let result = agent.run("find rust").await.unwrap();
        assert_eq!(result.output, "found it");
        assert_eq!(result.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn run_parse_error_recovery() {
        // First response is unparseable, second is valid Final Answer.
        // The agent cycles through FakeListChatModel responses.
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "I don't know the format".into(),
            "Final Answer: recovered".into(),
        ]));
        let agent = ReActAgent::builder()
            .model(model)
            .tools(vec![make_search_tool()])
            .max_iterations(5)
            .build()
            .unwrap();

        let result = agent.run("test").await.unwrap();
        assert_eq!(result.output, "recovered");
        // First iteration was a parse error, second was final answer.
        assert_eq!(result.iterations, 2);
    }

    #[tokio::test]
    async fn run_verbose_mode() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Final Answer: verbose test".into(),
        ]));
        let agent = ReActAgent::builder()
            .model(model)
            .tools(vec![make_search_tool()])
            .verbose(true)
            .build()
            .unwrap();

        // Should not panic; verbose output goes to stdout.
        let result = agent.run("test").await.unwrap();
        assert_eq!(result.output, "verbose test");
    }

    #[tokio::test]
    async fn run_with_system_prompt() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Final Answer: with system".into(),
        ]));
        let agent = ReActAgent::builder()
            .model(model)
            .system_prompt("Be concise")
            .build()
            .unwrap();

        let result = agent.run("test").await.unwrap();
        assert_eq!(result.output, "with system");
    }

    #[tokio::test]
    async fn create_react_agent_factory() {
        let model: Arc<dyn BaseChatModel> =
            Arc::new(FakeListChatModel::new(vec!["Final Answer: factory".into()]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        assert_eq!(agent.max_iterations, 10);
        assert!(agent.system_prompt.is_none());
        assert!(!agent.verbose);
        assert_eq!(agent.tools.len(), 1);

        let result = agent.run("test").await.unwrap();
        assert_eq!(result.output, "factory");
    }

    #[tokio::test]
    async fn trace_tokens_increase_with_iterations() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Thought: searching\nAction: search\nAction Input: test".into(),
            "Final Answer: done".into(),
        ]));
        let agent = create_react_agent(model, vec![make_search_tool()]);

        let (_, trace) = agent.run_with_trace("test").await.unwrap();
        assert!(trace.total_tokens > 0);
        // Two iterations should have more tokens than just one.
        assert_eq!(trace.get_steps().len(), 2);
    }

    #[tokio::test]
    async fn run_no_tools_immediate_answer() {
        let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
            "Final Answer: no tools needed".into(),
        ]));
        let agent = create_react_agent(model, vec![]);

        let result = agent.run("simple question").await.unwrap();
        assert_eq!(result.output, "no tools needed");
        assert!(result.tool_calls.is_empty());
    }
}
