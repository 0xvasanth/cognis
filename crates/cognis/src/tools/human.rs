//! Human input tool, approval wrapper, and toolkit system.
//!
//! Provides tools for human-in-the-loop workflows:
//! - [`HumanInputTool`] — asks a human for input via a pluggable [`InputProvider`]
//! - [`HumanApprovalTool`] — wraps another tool and requires human approval before execution
//! - [`Toolkit`] — groups related tools together with metadata
//! - [`ToolkitBuilder`] — builder pattern for constructing a [`Toolkit`]

use async_trait::async_trait;
use cognis_core::error::{Result, CognisError};
use cognis_core::tools::base::{BaseTool, ToolSchema};
use cognis_core::tools::types::{ToolInput, ToolOutput};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// InputProvider trait
// ---------------------------------------------------------------------------

/// Trait for providing human input. Implement this to customize how input is
/// collected (stdin, GUI dialog, web form, etc.).
#[async_trait]
pub trait InputProvider: Send + Sync {
    /// Prompt the human and return their response.
    async fn get_input(&self, prompt: &str) -> Result<String>;
}

// ---------------------------------------------------------------------------
// StdinInputProvider
// ---------------------------------------------------------------------------

/// Reads input from standard input.
pub struct StdinInputProvider;

#[async_trait]
impl InputProvider for StdinInputProvider {
    async fn get_input(&self, prompt: &str) -> Result<String> {
        println!("{}", prompt);
        let mut buf = String::new();
        std::io::stdin()
            .read_line(&mut buf)
            .map_err(CognisError::IoError)?;
        Ok(buf.trim().to_string())
    }
}

// ---------------------------------------------------------------------------
// MockInputProvider
// ---------------------------------------------------------------------------

/// Returns pre-configured responses in order. Thread-safe via [`AtomicUsize`].
/// Useful for testing without requiring actual stdin.
pub struct MockInputProvider {
    responses: Vec<String>,
    counter: AtomicUsize,
}

impl MockInputProvider {
    /// Create a new mock provider that returns `responses` in order.
    /// When all responses are exhausted, wraps around to the beginning.
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            counter: AtomicUsize::new(0),
        }
    }

    /// Returns the number of times input has been requested so far.
    pub fn call_count(&self) -> usize {
        self.counter.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl InputProvider for MockInputProvider {
    async fn get_input(&self, _prompt: &str) -> Result<String> {
        if self.responses.is_empty() {
            return Err(CognisError::ToolException(
                "MockInputProvider has no responses configured".into(),
            ));
        }
        let idx = self.counter.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        Ok(self.responses[idx].clone())
    }
}

// ---------------------------------------------------------------------------
// HumanInputTool
// ---------------------------------------------------------------------------

/// A tool that asks a human for input. The tool input text is used as the
/// prompt shown to the human.
pub struct HumanInputTool {
    provider: Arc<dyn InputProvider>,
}

impl HumanInputTool {
    /// Create a new human input tool with the given input provider.
    pub fn new(provider: Arc<dyn InputProvider>) -> Self {
        Self { provider }
    }

    /// Create a human input tool that reads from stdin.
    pub fn stdin() -> Self {
        Self {
            provider: Arc::new(StdinInputProvider),
        }
    }
}

#[async_trait]
impl BaseTool for HumanInputTool {
    fn name(&self) -> &str {
        "human_input"
    }

    fn description(&self) -> &str {
        "Ask a human for input when you need clarification or approval"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or prompt to show the human"
                }
            },
            "required": ["prompt"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let prompt = extract_prompt(&input)?;
        let response = self.provider.get_input(&prompt).await?;
        Ok(ToolOutput::Content(Value::String(response)))
    }
}

/// Extract prompt string from various input formats.
fn extract_prompt(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(p)) = map.get("prompt") {
                Ok(p.clone())
            } else {
                // Fall back to serializing the whole map as the prompt
                Ok(serde_json::to_string(map).unwrap_or_default())
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(p)) = tc.args.get("prompt") {
                Ok(p.clone())
            } else {
                Ok(serde_json::to_string(&tc.args).unwrap_or_default())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HumanApprovalTool
// ---------------------------------------------------------------------------

/// Wraps an inner tool and requires human approval before executing it.
/// If the human approves (response starts with "y" or "yes", case-insensitive),
/// the inner tool is executed. Otherwise, returns "Action denied by human".
pub struct HumanApprovalTool {
    inner: Arc<dyn BaseTool>,
    provider: Arc<dyn InputProvider>,
    approval_prompt: String,
}

impl HumanApprovalTool {
    /// Create a new approval wrapper with a default prompt.
    pub fn new(inner: Arc<dyn BaseTool>, provider: Arc<dyn InputProvider>) -> Self {
        let tool_name = inner.name().to_string();
        Self {
            inner,
            provider,
            approval_prompt: format!("Do you approve running tool '{}'? (yes/no)", tool_name),
        }
    }

    /// Create a new approval wrapper with a custom approval prompt.
    pub fn with_prompt(
        inner: Arc<dyn BaseTool>,
        provider: Arc<dyn InputProvider>,
        approval_prompt: String,
    ) -> Self {
        Self {
            inner,
            provider,
            approval_prompt,
        }
    }

    /// Returns the approval prompt.
    pub fn approval_prompt(&self) -> &str {
        &self.approval_prompt
    }
}

/// Check if a response indicates approval.
fn is_approved(response: &str) -> bool {
    let trimmed = response.trim().to_lowercase();
    trimmed == "y" || trimmed == "yes" || trimmed.starts_with("yes")
}

#[async_trait]
impl BaseTool for HumanApprovalTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn args_schema(&self) -> Option<Value> {
        self.inner.args_schema()
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        // Build the approval message including the input details
        let input_desc = match &input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => serde_json::to_string_pretty(map).unwrap_or_default(),
            ToolInput::ToolCall(tc) => serde_json::to_string_pretty(&tc.args).unwrap_or_default(),
        };

        let full_prompt = format!("{}\nInput: {}", self.approval_prompt, input_desc);
        let response = self.provider.get_input(&full_prompt).await?;

        if is_approved(&response) {
            self.inner._run(input).await
        } else {
            Ok(ToolOutput::Content(Value::String(
                "Action denied by human".to_string(),
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Toolkit
// ---------------------------------------------------------------------------

/// A named collection of related tools.
pub struct Toolkit {
    /// Name of the toolkit.
    pub name: String,
    /// Description of what this toolkit provides.
    pub description: String,
    /// The tools in this toolkit.
    tools: Vec<Arc<dyn BaseTool>>,
}

impl Toolkit {
    /// Create a new toolkit.
    pub fn new(name: String, description: String, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        Self {
            name,
            description,
            tools,
        }
    }

    /// Get a reference to all tools in this toolkit.
    pub fn get_tools(&self) -> &[Arc<dyn BaseTool>] {
        &self.tools
    }

    /// Find a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn BaseTool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Get the schemas for all tools in this toolkit.
    pub fn get_schemas(&self) -> Vec<ToolSchema> {
        self.tools
            .iter()
            .map(|t| ToolSchema {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.args_schema(),
                extras: None,
            })
            .collect()
    }

    /// Returns the number of tools in the toolkit.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if the toolkit contains no tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns an iterator over the tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
}

// ---------------------------------------------------------------------------
// ToolkitBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`Toolkit`].
pub struct ToolkitBuilder {
    name: String,
    description: String,
    tools: Vec<Arc<dyn BaseTool>>,
}

impl ToolkitBuilder {
    /// Start building a new toolkit with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            tools: Vec::new(),
        }
    }

    /// Set the toolkit description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add a tool to the toolkit.
    pub fn tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to the toolkit.
    pub fn tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Build the toolkit.
    pub fn build(self) -> Toolkit {
        Toolkit {
            name: self.name,
            description: self.description,
            tools: self.tools,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::calculator::CalculatorTool;

    fn mock_provider(responses: Vec<&str>) -> Arc<MockInputProvider> {
        Arc::new(MockInputProvider::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    // -- MockInputProvider tests --

    #[tokio::test]
    async fn test_mock_provider_returns_responses_in_order() {
        let provider = mock_provider(vec!["first", "second", "third"]);
        assert_eq!(provider.get_input("a").await.unwrap(), "first");
        assert_eq!(provider.get_input("b").await.unwrap(), "second");
        assert_eq!(provider.get_input("c").await.unwrap(), "third");
    }

    #[tokio::test]
    async fn test_mock_provider_wraps_around() {
        let provider = mock_provider(vec!["a", "b"]);
        assert_eq!(provider.get_input("").await.unwrap(), "a");
        assert_eq!(provider.get_input("").await.unwrap(), "b");
        assert_eq!(provider.get_input("").await.unwrap(), "a");
    }

    #[tokio::test]
    async fn test_mock_provider_call_count() {
        let provider = mock_provider(vec!["x"]);
        assert_eq!(provider.call_count(), 0);
        let _ = provider.get_input("").await;
        assert_eq!(provider.call_count(), 1);
        let _ = provider.get_input("").await;
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_provider_empty_returns_error() {
        let provider = Arc::new(MockInputProvider::new(vec![]));
        let result = provider.get_input("prompt").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_provider_single_response() {
        let provider = mock_provider(vec!["only"]);
        assert_eq!(provider.get_input("").await.unwrap(), "only");
        assert_eq!(provider.get_input("").await.unwrap(), "only");
    }

    // -- HumanInputTool tests --

    #[tokio::test]
    async fn test_human_input_tool_name() {
        let tool = HumanInputTool::new(mock_provider(vec!["ok"]));
        assert_eq!(tool.name(), "human_input");
    }

    #[tokio::test]
    async fn test_human_input_tool_description() {
        let tool = HumanInputTool::new(mock_provider(vec!["ok"]));
        assert!(tool.description().contains("human"));
    }

    #[tokio::test]
    async fn test_human_input_tool_has_schema() {
        let tool = HumanInputTool::new(mock_provider(vec!["ok"]));
        let schema = tool.args_schema();
        assert!(schema.is_some());
        let schema = schema.unwrap();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["prompt"].is_object());
    }

    #[tokio::test]
    async fn test_human_input_tool_text_input() {
        let provider = mock_provider(vec!["hello human"]);
        let tool = HumanInputTool::new(provider);
        let result = tool.run_str("What is your name?").await.unwrap();
        assert_eq!(result, Value::String("hello human".into()));
    }

    #[tokio::test]
    async fn test_human_input_tool_structured_input() {
        let provider = mock_provider(vec!["structured response"]);
        let tool = HumanInputTool::new(provider);
        let input = json!({"prompt": "Enter value"});
        let result = tool.run_json(&input).await.unwrap();
        assert_eq!(result, Value::String("structured response".into()));
    }

    #[tokio::test]
    async fn test_human_input_tool_multiple_calls() {
        let provider = mock_provider(vec!["first", "second"]);
        let tool = HumanInputTool::new(provider);
        let r1 = tool.run_str("q1").await.unwrap();
        let r2 = tool.run_str("q2").await.unwrap();
        assert_eq!(r1, Value::String("first".into()));
        assert_eq!(r2, Value::String("second".into()));
    }

    // -- HumanApprovalTool tests --

    #[tokio::test]
    async fn test_approval_tool_approved_yes() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("2 + 3").await.unwrap();
        assert_eq!(result, Value::String("5".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_approved_y() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["y"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("10 * 5").await.unwrap();
        assert_eq!(result, Value::String("50".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_approved_yes_please() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes please"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("1 + 1").await.unwrap();
        assert_eq!(result, Value::String("2".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_approved_case_insensitive() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["YES"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("3 + 3").await.unwrap();
        assert_eq!(result, Value::String("6".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_denied() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["no"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("2 + 3").await.unwrap();
        assert_eq!(result, Value::String("Action denied by human".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_denied_random_text() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["nope"]);
        let tool = HumanApprovalTool::new(inner, provider);
        let result = tool.run_str("2 + 3").await.unwrap();
        assert_eq!(result, Value::String("Action denied by human".into()));
    }

    #[tokio::test]
    async fn test_approval_tool_delegates_name() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes"]);
        let tool = HumanApprovalTool::new(inner, provider);
        assert_eq!(tool.name(), "calculator");
    }

    #[tokio::test]
    async fn test_approval_tool_delegates_description() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes"]);
        let tool = HumanApprovalTool::new(inner, provider);
        assert!(tool.description().contains("math"));
    }

    #[tokio::test]
    async fn test_approval_tool_custom_prompt() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes"]);
        let tool =
            HumanApprovalTool::with_prompt(inner, provider, "Custom approval prompt?".into());
        assert_eq!(tool.approval_prompt(), "Custom approval prompt?");
    }

    #[tokio::test]
    async fn test_approval_tool_default_prompt_contains_tool_name() {
        let inner: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["yes"]);
        let tool = HumanApprovalTool::new(inner, provider);
        assert!(tool.approval_prompt().contains("calculator"));
    }

    // -- Toolkit tests --

    #[test]
    fn test_toolkit_new() {
        let toolkit = Toolkit::new("test".into(), "A test toolkit".into(), vec![]);
        assert_eq!(toolkit.name, "test");
        assert_eq!(toolkit.description, "A test toolkit");
        assert!(toolkit.is_empty());
    }

    #[test]
    fn test_toolkit_get_tools() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let toolkit = Toolkit::new("math".into(), "Math tools".into(), vec![calc]);
        assert_eq!(toolkit.get_tools().len(), 1);
        assert_eq!(toolkit.get_tools()[0].name(), "calculator");
    }

    #[test]
    fn test_toolkit_get_tool_by_name() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let toolkit = Toolkit::new("math".into(), "desc".into(), vec![calc]);
        let found = toolkit.get_tool("calculator");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name(), "calculator");
    }

    #[test]
    fn test_toolkit_get_tool_not_found() {
        let toolkit = Toolkit::new("empty".into(), "desc".into(), vec![]);
        assert!(toolkit.get_tool("nonexistent").is_none());
    }

    #[test]
    fn test_toolkit_get_schemas() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["ok"]);
        let human: Arc<dyn BaseTool> = Arc::new(HumanInputTool::new(provider));
        let toolkit = Toolkit::new("mixed".into(), "desc".into(), vec![calc, human]);
        let schemas = toolkit.get_schemas();
        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].name, "calculator");
        assert_eq!(schemas[1].name, "human_input");
    }

    #[test]
    fn test_toolkit_len() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let toolkit = Toolkit::new("t".into(), "d".into(), vec![calc]);
        assert_eq!(toolkit.len(), 1);
        assert!(!toolkit.is_empty());
    }

    #[test]
    fn test_toolkit_tool_names() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let toolkit = Toolkit::new("t".into(), "d".into(), vec![calc]);
        assert_eq!(toolkit.tool_names(), vec!["calculator"]);
    }

    // -- ToolkitBuilder tests --

    #[test]
    fn test_toolkit_builder_basic() {
        let toolkit = ToolkitBuilder::new("my_toolkit")
            .description("My tools")
            .build();
        assert_eq!(toolkit.name, "my_toolkit");
        assert_eq!(toolkit.description, "My tools");
        assert!(toolkit.is_empty());
    }

    #[test]
    fn test_toolkit_builder_with_tools() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let toolkit = ToolkitBuilder::new("math")
            .description("Math toolkit")
            .tool(calc)
            .build();
        assert_eq!(toolkit.len(), 1);
    }

    #[test]
    fn test_toolkit_builder_with_multiple_tools() {
        let calc: Arc<dyn BaseTool> = Arc::new(CalculatorTool);
        let provider = mock_provider(vec!["ok"]);
        let human: Arc<dyn BaseTool> = Arc::new(HumanInputTool::new(provider));
        let toolkit = ToolkitBuilder::new("all")
            .description("All tools")
            .tools(vec![calc, human])
            .build();
        assert_eq!(toolkit.len(), 2);
        assert!(toolkit.get_tool("calculator").is_some());
        assert!(toolkit.get_tool("human_input").is_some());
    }

    // -- is_approved helper tests --

    #[test]
    fn test_is_approved_yes_variants() {
        assert!(is_approved("yes"));
        assert!(is_approved("Yes"));
        assert!(is_approved("YES"));
        assert!(is_approved("y"));
        assert!(is_approved("Y"));
        assert!(is_approved("yes please"));
        assert!(is_approved("  yes  "));
    }

    #[test]
    fn test_is_approved_no_variants() {
        assert!(!is_approved("no"));
        assert!(!is_approved("No"));
        assert!(!is_approved("n"));
        assert!(!is_approved("nope"));
        assert!(!is_approved(""));
        assert!(!is_approved("maybe"));
    }
}
