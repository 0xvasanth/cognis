//! Integration tests for `AgentExecutor::parallel_tool_calls`.
//!
//! Covers (1) opt-in parallel dispatch beats serial latency, (2) tool-message
//! ordering matches `tool_calls` order regardless of completion order, and
//! (3) default serial dispatch is preserved when the knob is left off.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis::agents::AgentExecutor;
use cognis_core::error::Result;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::tool_types::ToolCall;
use cognis_core::messages::{AIMessage, Message};
use cognis_core::outputs::{ChatGeneration, ChatResult};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};

/// A chat model that, on the first call, emits N `tool_calls` naming each of
/// `tool_names` (with no args), and on the second call returns a final text.
struct BatchToolModel {
    tool_names: Vec<String>,
    call_count: AtomicU32,
}

impl BatchToolModel {
    fn new(tool_names: Vec<&str>) -> Self {
        Self {
            tool_names: tool_names.into_iter().map(String::from).collect(),
            call_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl BaseChatModel for BatchToolModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        let ai = if n == 0 {
            let calls: Vec<ToolCall> = self
                .tool_names
                .iter()
                .enumerate()
                .map(|(i, name)| ToolCall {
                    name: name.clone(),
                    args: HashMap::new(),
                    id: Some(format!("call_{i}")),
                })
                .collect();
            AIMessage::new("").with_tool_calls(calls)
        } else {
            AIMessage::new("done")
        };
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "batch-tool-model"
    }
}

/// A tool that sleeps for `delay` before returning its name.
struct SleepyTool {
    name: &'static str,
    delay: Duration,
}

#[async_trait]
impl BaseTool for SleepyTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Sleeps then returns its name."
    }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({ "type": "object", "properties": {} }))
    }
    async fn _run(&self, _input: ToolInput) -> Result<ToolOutput> {
        tokio::time::sleep(self.delay).await;
        Ok(ToolOutput::Content(json!({ "from": self.name })))
    }
}

fn tool(name: &'static str, delay_ms: u64) -> Arc<dyn BaseTool> {
    Arc::new(SleepyTool {
        name,
        delay: Duration::from_millis(delay_ms),
    })
}

#[tokio::test]
async fn parallel_dispatch_runs_concurrently() {
    let model = Arc::new(BatchToolModel::new(vec!["a", "b", "c"]));
    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool("a", 120))
        .tool(tool("b", 120))
        .tool(tool("c", 120))
        .parallel_tool_calls(true)
        .max_iterations(5)
        .build();

    let start = Instant::now();
    let result = executor.run(&[Message::human("go")]).await.unwrap();
    let elapsed = start.elapsed();

    // Serial would be >= 360ms (3 × 120ms). Parallel should land well under
    // that — allow a generous 250ms ceiling to absorb scheduler jitter.
    assert!(
        elapsed < Duration::from_millis(250),
        "parallel dispatch took {elapsed:?}, expected < 250ms",
    );
    assert_eq!(result.output, "done");
}

#[tokio::test]
async fn serial_default_is_preserved() {
    let model = Arc::new(BatchToolModel::new(vec!["a", "b", "c"]));
    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool("a", 80))
        .tool(tool("b", 80))
        .tool(tool("c", 80))
        .max_iterations(5)
        .build();

    let start = Instant::now();
    let _ = executor.run(&[Message::human("go")]).await.unwrap();
    let elapsed = start.elapsed();

    // Serial: three sequential 80ms sleeps → >= ~240ms. Use 200ms as a safe
    // lower bound to avoid flaking if timers underrun slightly.
    assert!(
        elapsed >= Duration::from_millis(200),
        "serial dispatch took {elapsed:?}, expected >= 200ms",
    );
}

#[tokio::test]
async fn parallel_dispatch_preserves_tool_call_order() {
    // Reverse delays so "a" finishes last — if we appended by completion
    // order, "a" would appear after "b"/"c". We must see the original order.
    let model = Arc::new(BatchToolModel::new(vec!["a", "b", "c"]));
    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool("a", 150))
        .tool(tool("b", 50))
        .tool(tool("c", 10))
        .parallel_tool_calls(true)
        .return_intermediate_steps(true)
        .build();

    let result = executor.run(&[Message::human("go")]).await.unwrap();

    let observed: Vec<&str> = result
        .intermediate_steps
        .iter()
        .map(|s| s.action.tool_name.as_str())
        .collect();
    assert_eq!(observed, vec!["a", "b", "c"]);

    // ToolMessage sequence in messages should match too.
    let tool_msgs: Vec<&str> = result
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some(t.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_msgs, vec!["call_0", "call_1", "call_2"]);
}

#[tokio::test]
async fn parallel_flag_is_a_noop_for_single_tool_call() {
    // Single tool call: batch_size == 1, parallel branch is skipped per
    // the `batch_size > 1` guard. Just verify it still runs to completion.
    let model = Arc::new(BatchToolModel::new(vec!["a"]));
    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool("a", 30))
        .parallel_tool_calls(true)
        .build();

    let result = executor.run(&[Message::human("go")]).await.unwrap();
    assert_eq!(result.output, "done");
}
