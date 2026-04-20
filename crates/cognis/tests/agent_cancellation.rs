//! Cooperative-cancellation integration tests for the agent executors.
//!
//! Verifies that `AgentExecutor::run_with_cancel` honours a
//! `CancellationToken` both before the loop starts and during in-flight
//! model and tool calls.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis::agents::{
    AgentExecutor, Plan, PlanAndExecuteAgent, PlanStep, Planner, ReActAgent, StepExecutor,
};
use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::fake::FakeListChatModel;
use cognis_core::messages::{AIMessage, Message, ToolCall};
use cognis_core::outputs::{ChatGeneration, ChatResult};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::CancellationToken;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fake model that always returns a tool call to an `add` tool, so the agent
/// loop keeps iterating indefinitely until it is cancelled or hits its
/// iteration cap. Each `_generate` sleeps for `delay_ms` to give concurrent
/// cancellation a reliable window in which to fire.
struct LoopingToolModel {
    call_count: AtomicU32,
    delay_ms: u64,
}

impl LoopingToolModel {
    fn new(delay_ms: u64) -> Self {
        Self {
            call_count: AtomicU32::new(0),
            delay_ms,
        }
    }
}

#[async_trait]
impl BaseChatModel for LoopingToolModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut args = HashMap::new();
        args.insert("a".to_string(), json!(1));
        args.insert("b".to_string(), json!(n));
        let ai = AIMessage::new("").with_tool_calls(vec![ToolCall {
            name: "add".to_string(),
            args,
            id: Some(format!("call_{n}")),
        }]);
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "looping-tool-model"
    }
}

/// Trivial add tool used by `LoopingToolModel`.
struct AddTool;

#[async_trait]
impl BaseTool for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers"
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let (a, b) = match input {
            ToolInput::Structured(map) => {
                let a = map.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                let b = map.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                (a, b)
            }
            _ => (0, 0),
        };
        Ok(ToolOutput::Content(json!(a + b)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pre_cancelled_token_aborts_immediately() {
    // A pre-cancelled token must abort before the first model call completes.
    let model = Arc::new(LoopingToolModel::new(0));
    let executor = AgentExecutor::builder()
        .model(model)
        .tools(vec![Arc::new(AddTool) as Arc<dyn BaseTool>])
        .max_iterations(5)
        .build();

    let cancel = CancellationToken::cancelled_now();
    let msgs = vec![Message::human("hello")];
    let result = executor.run_with_cancel(&msgs, cancel).await;
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_during_run_breaks_loop() {
    // A slow model + generous iteration budget gives a concurrent cancel
    // task a reliable window to fire. The run must resolve quickly with
    // either Ok (if the model happened to complete first) or Cancelled —
    // the critical assertion is that it does not hang.
    let slow_model = Arc::new(LoopingToolModel::new(10));
    let executor = AgentExecutor::builder()
        .model(slow_model)
        .tools(vec![Arc::new(AddTool) as Arc<dyn BaseTool>])
        .max_iterations(100)
        .build();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_clone.cancel();
    });

    let msgs = vec![Message::human("hello")];
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        executor.run_with_cancel(&msgs, cancel),
    )
    .await
    .expect("run_with_cancel should not hang past the cancellation signal");

    // The model always produces tool calls, so we will definitely hit
    // enough iterations for cancel to fire before the max-iterations cap.
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!(
            "expected Cancelled after concurrent cancel, got {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn run_with_default_token_behaves_as_before() {
    // Backward-compat: the historical `run()` path is preserved. A
    // `FakeListChatModel` returning a final text terminates the loop on the
    // first iteration and must not be disturbed by the wrapper delegating
    // to `run_with_cancel`.
    let model = Arc::new(FakeListChatModel::new(vec!["done".to_string()]));
    let executor = AgentExecutor::builder()
        .model(model)
        .max_iterations(5)
        .build();
    let msgs = vec![Message::human("hi")];
    let result = executor.run(&msgs).await;
    let result = result.expect("run should succeed for final-text models");
    assert_eq!(result.output, "done");
}

// ---------------------------------------------------------------------------
// ReActAgent cancellation
// ---------------------------------------------------------------------------

/// A fake chat model that always parses as an `Action: noop / Action Input: {}`
/// so the ReAct loop keeps iterating until cancelled or max-iterations runs
/// out. A per-call delay gives concurrent cancellation a reliable window.
struct LoopingReActModel {
    delay_ms: u64,
}

#[async_trait]
impl BaseChatModel for LoopingReActModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
        // A response that parses as an `Action` step so the ReAct loop
        // keeps running.
        let body = "Thought: still thinking\nAction: noop\nAction Input: {}";
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(body))],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "looping-react-model"
    }
}

struct NoopTool;

#[async_trait]
impl BaseTool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "A tool that does nothing."
    }
    async fn _run(&self, _input: ToolInput) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!("ok")))
    }
}

#[tokio::test]
async fn react_run_with_cancel_honors_token() {
    let model: Arc<dyn BaseChatModel> = Arc::new(LoopingReActModel { delay_ms: 5 });
    let agent = ReActAgent::builder()
        .model(model)
        .tools(vec![Arc::new(NoopTool) as Arc<dyn BaseTool>])
        .max_iterations(100)
        .build()
        .expect("build ReActAgent");

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_clone.cancel();
    });

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run_with_cancel("hello", cancel),
    )
    .await
    .expect("ReAct should respond to cancel quickly");
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn react_pre_cancelled_aborts_immediately() {
    let model: Arc<dyn BaseChatModel> = Arc::new(LoopingReActModel { delay_ms: 0 });
    let agent = ReActAgent::builder()
        .model(model)
        .tools(vec![Arc::new(NoopTool) as Arc<dyn BaseTool>])
        .max_iterations(5)
        .build()
        .expect("build ReActAgent");

    let cancel = CancellationToken::cancelled_now();
    let result = agent.run_with_cancel("hi", cancel).await;
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// PlanAndExecute cancellation
// ---------------------------------------------------------------------------

/// A planner producing many pending steps so the loop has material to run.
struct MultiStepPlanner;

impl Planner for MultiStepPlanner {
    fn create_plan(&self, goal: &str) -> Result<Plan> {
        let steps: Vec<PlanStep> = (0..20)
            .map(|i| PlanStep::new(i, format!("step {i}")))
            .collect();
        Ok(Plan::new(goal, steps))
    }
}

/// A step executor that sleeps briefly so a concurrent cancel can fire.
struct SlowStepExecutor {
    delay_ms: u64,
}

#[async_trait]
impl StepExecutor for SlowStepExecutor {
    async fn execute_step(&self, step: &PlanStep, _context: &Value) -> Result<String> {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
        Ok(format!("completed {}", step.description))
    }
}

#[tokio::test]
async fn plan_and_execute_run_with_cancel_honors_token() {
    let agent = PlanAndExecuteAgent::builder()
        .planner(MultiStepPlanner)
        .executor(SlowStepExecutor { delay_ms: 10 })
        .build()
        .expect("build PlanAndExecuteAgent");

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(25)).await;
        cancel_clone.cancel();
    });

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run_with_cancel("do things", cancel),
    )
    .await
    .expect("plan-and-execute should respond to cancel quickly");
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn plan_and_execute_pre_cancelled_aborts_before_planning() {
    let agent = PlanAndExecuteAgent::builder()
        .planner(MultiStepPlanner)
        .executor(SlowStepExecutor { delay_ms: 0 })
        .build()
        .expect("build PlanAndExecuteAgent");

    let cancel = CancellationToken::cancelled_now();
    let result = agent.run_with_cancel("do things", cancel).await;
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// AgentExecutor: between-iteration check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_between_iterations_emits_cancelled_error() {
    // Fires cancel after the first model call so the loop-boundary check
    // sees the token on the next iteration. A small per-call delay on the
    // model guarantees the spawned cancel task can observe the call count
    // before the loop exhausts its iteration budget.
    let model = Arc::new(LoopingToolModel::new(5));
    let executor = AgentExecutor::builder()
        .model(model.clone())
        .tools(vec![Arc::new(AddTool) as Arc<dyn BaseTool>])
        .max_iterations(100)
        .build();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let spy = model.clone();
    tokio::spawn(async move {
        // Poll until we see the first call, then cancel.
        loop {
            if spy.call_count.load(Ordering::SeqCst) >= 1 {
                cancel_clone.cancel();
                break;
            }
            tokio::task::yield_now().await;
        }
    });

    let msgs = vec![Message::human("hello")];
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        executor.run_with_cancel(&msgs, cancel),
    )
    .await
    .expect("agent should respond to cancel quickly");
    match result {
        Err(CognisError::Cancelled(_)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}
