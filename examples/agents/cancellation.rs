//! Demonstrates client-side Stop with `AgentExecutor::run_with_cancel`.
//!
//! Spawns the agent on a `tokio::task`, sleeps briefly, then fires
//! `CancellationToken::cancel()`. The agent aborts cleanly — both the
//! in-flight model call and the agent loop return promptly — with
//! `CognisError::Cancelled(...)` surfaced to the caller.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example cancellation -p cognis-examples --features cognis/all-providers
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cognis::agents::AgentExecutor;
use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{AIMessage, Message, ToolCall};
use cognis_core::outputs::{ChatGeneration, ChatResult};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::CancellationToken;
use serde_json::{json, Value};

/// A fake chat model that keeps calling the `noop` tool forever. Each call
/// sleeps briefly so the cancel task has a window to fire.
struct LoopingToolModel;

#[async_trait]
impl BaseChatModel for LoopingToolModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut args = HashMap::new();
        args.insert("n".to_string(), json!(1));
        let ai = AIMessage::new("").with_tool_calls(vec![ToolCall {
            name: "noop".into(),
            args,
            id: Some("call_1".into()),
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

struct NoopTool;

#[async_trait]
impl BaseTool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "Does nothing"
    }
    async fn _run(&self, _input: ToolInput) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(Value::String("ok".into())))
    }
}

#[tokio::main]
async fn main() {
    let executor = AgentExecutor::builder()
        .model(Arc::new(LoopingToolModel))
        .tools(vec![Arc::new(NoopTool) as Arc<dyn BaseTool>])
        .max_iterations(1000)
        .return_intermediate_steps(true)
        .build();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    let handle = tokio::spawn(async move {
        let msgs = vec![Message::human("run many steps")];
        executor.run_with_cancel(&msgs, cancel).await
    });

    println!("running for 50ms then cancelling...");
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancel_clone.cancel();
    println!("cancel signal sent");

    match handle.await.expect("task join") {
        Err(CognisError::Cancelled(reason)) => {
            println!("agent aborted cleanly: {reason}");
        }
        Ok(r) => {
            println!(
                "agent finished normally (completed before cancel): iterations = {}",
                r.intermediate_steps.len()
            );
        }
        Err(e) => {
            println!("unexpected error: {e}");
        }
    }
}
