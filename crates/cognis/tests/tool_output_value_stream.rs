//! Integration: AgentExecutor::astream_events emits typed tool output.
//!
//! End-to-end verification that a tool returning `ToolOutput::Content(json!(...))`
//! has its structured value preserved all the way through the stream:
//!
//! 1. Executor runs the tool and fires `on_tool_end` with `output_value`
//!    set to the raw `serde_json::Value`.
//! 2. `EventStreamCallbackHandler::on_tool_end` forwards that value into
//!    `EventData.output` verbatim (no stringification, no re-wrapping).
//! 3. `AgentExecutor::astream_events` yields the corresponding
//!    `StreamEvent` with `EventType::OnToolEnd` and the structured payload
//!    intact in `ev.data.output`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use cognis::agents::AgentExecutor;
use cognis_core::language_models::fake::FakeMessagesListChatModel;
use cognis_core::messages::tool_types::ToolCall;
use cognis_core::messages::{AIMessage, HumanMessage, Message};
use cognis_core::tools::{BaseTool, ToolInput, ToolOutput};
use cognis_core::tracers::EventType;

/// Tool that returns a structured JSON object as its content.
struct JsonTool;

#[async_trait]
impl BaseTool for JsonTool {
    fn name(&self) -> &str {
        "json_tool"
    }

    fn description(&self) -> &str {
        "returns structured JSON"
    }

    async fn _run(&self, _input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        Ok(ToolOutput::Content(
            json!({"answer": 42, "confidence": 0.9}),
        ))
    }
}

#[tokio::test]
async fn astream_events_preserves_tool_output_shape() {
    // Model script: first turn requests a call to `json_tool`; second turn
    // finalizes with a plain AI response.
    let tool_call = ToolCall {
        name: "json_tool".to_string(),
        args: HashMap::new(),
        id: Some("call_1".to_string()),
    };
    let first = Message::Ai(AIMessage::new("calling tool").with_tool_calls(vec![tool_call]));
    let second = Message::Ai(AIMessage::new("done"));

    let model = Arc::new(FakeMessagesListChatModel::new(vec![first, second]));
    let tool: Arc<dyn BaseTool> = Arc::new(JsonTool);

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(tool)
        .max_iterations(3)
        .build();

    let messages = vec![Message::Human(HumanMessage::new("go"))];
    let mut stream = executor.astream_events(messages).await.unwrap();

    let mut saw_typed_output = false;
    while let Some(ev) = stream.next().await {
        let ev = ev.unwrap();
        if ev.event == EventType::OnToolEnd {
            let out = ev.data.output.expect("output should be set on OnToolEnd");
            assert_eq!(
                out,
                json!({"answer": 42, "confidence": 0.9}),
                "OnToolEnd.data.output must be the raw structured value (no stringification)"
            );
            saw_typed_output = true;
        }
    }
    assert!(
        saw_typed_output,
        "expected at least one OnToolEnd event carrying the typed tool output"
    );
}
