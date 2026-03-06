//! Integration tests for `AgentExecutor` with mock model and tools.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use rustchain::agents::AgentExecutor;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{AIMessage, Message, ToolCall};
use rustchain_core::outputs::{ChatGeneration, ChatResult};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};

// ---------------------------------------------------------------------------
// MockToolModel — first call returns a tool call, second call returns text
// ---------------------------------------------------------------------------

struct MockToolModel {
    call_count: AtomicU32,
}

impl MockToolModel {
    fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl BaseChatModel for MockToolModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        let ai = if n == 0 {
            // First call: return a tool call
            let mut args = HashMap::new();
            args.insert("a".to_string(), json!(2));
            args.insert("b".to_string(), json!(3));
            AIMessage::new("").with_tool_calls(vec![ToolCall {
                name: "add".to_string(),
                args,
                id: Some("call_1".to_string()),
            }])
        } else {
            // Second call: final text answer
            AIMessage::new("The answer is 5.")
        };
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "mock-tool-model"
    }
}

// ---------------------------------------------------------------------------
// AlwaysToolModel — always returns tool calls (for max-iterations test)
// ---------------------------------------------------------------------------

struct AlwaysToolModel {
    call_count: AtomicU32,
}

impl AlwaysToolModel {
    fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl BaseChatModel for AlwaysToolModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut args = HashMap::new();
        args.insert("a".to_string(), json!(1));
        args.insert("b".to_string(), json!(n));
        let ai = AIMessage::new("").with_tool_calls(vec![ToolCall {
            name: "add".to_string(),
            args,
            id: Some(format!("call_{}", n)),
        }]);
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "always-tool-model"
    }
}

// ---------------------------------------------------------------------------
// AddTool — adds two numbers
// ---------------------------------------------------------------------------

struct AddTool;

#[async_trait]
impl BaseTool for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers a and b"
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let map = match input {
            ToolInput::Structured(m) => m,
            _ => {
                return Err(RustChainError::ToolValidationError(
                    "expected structured input".into(),
                ))
            }
        };
        let a = map.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = map.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sum = a + b;
        Ok(ToolOutput::Content(Value::String(sum.to_string())))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_agent_tool_calling_loop() {
    let model: Arc<dyn BaseChatModel> = Arc::new(MockToolModel::new());
    let tool: Arc<dyn BaseTool> = Arc::new(AddTool);

    let executor = AgentExecutor::builder().model(model).tool(tool).build();

    let result = executor
        .run(&[Message::human("What is 2 + 3?")])
        .await
        .expect("agent should complete successfully");

    assert_eq!(result.output, "The answer is 5.");
    // Messages: human, ai+tool_call, tool_result, ai+final
    assert_eq!(result.messages.len(), 4);

    // Verify message types
    assert!(matches!(result.messages[0], Message::Human(_)));
    assert!(matches!(result.messages[1], Message::Ai(_)));
    assert!(matches!(result.messages[2], Message::Tool(_)));
    assert!(matches!(result.messages[3], Message::Ai(_)));
}

#[tokio::test]
async fn test_agent_max_iterations() {
    let model: Arc<dyn BaseChatModel> = Arc::new(AlwaysToolModel::new());
    let tool: Arc<dyn BaseTool> = Arc::new(AddTool);

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool)
        .max_iterations(3)
        .build();

    let err = executor
        .run(&[Message::human("loop forever")])
        .await
        .expect_err("should exceed max iterations");

    match err {
        RustChainError::RecursionLimitExceeded(msg) => {
            assert!(msg.contains("3"), "error should mention the limit: {msg}");
        }
        other => panic!("expected RecursionLimitExceeded, got: {other:?}"),
    }
}
