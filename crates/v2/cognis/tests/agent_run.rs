//! End-to-end: AgentBuilder → Agent::run with a fake provider + 1 tool.
//! Exercises the standard ReAct flow without network calls.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2::prelude::*;
use cognis2_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use cognis2_llm::provider::{LLMProvider, Provider};

/// Simulate an LLM that on the FIRST call requests a tool, on subsequent
/// calls returns a plain message.
struct Sequencer {
    idx: std::sync::atomic::AtomicUsize,
    responses: Vec<Message>,
}

impl Sequencer {
    fn new(responses: Vec<Message>) -> Self {
        Self {
            idx: 0.into(),
            responses,
        }
    }
}

#[async_trait]
impl LLMProvider for Sequencer {
    fn name(&self) -> &str {
        "sequencer"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }
    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let _ = (messages, opts);
        use std::sync::atomic::Ordering;
        let n = self.idx.fetch_add(1, Ordering::SeqCst);
        let message = self.responses.get(n).cloned().unwrap_or(Message::ai("end"));
        Ok(ChatResponse {
            message,
            usage: Some(Usage::default()),
            finish_reason: "stop".into(),
            model: "seq".into(),
        })
    }
    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<cognis2_core::RunnableStream<StreamChunk>> {
        let _ = (messages, opts);
        unimplemented!()
    }
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

struct EchoTool;
#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echoes input"
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}))
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(input.into_json()))
    }
}

fn ai_with_tool_call(name: &str, args: serde_json::Value) -> Message {
    Message::Ai(cognis2_core::AiMessage {
        content: String::new(),
        tool_calls: vec![ToolCall {
            id: format!("call_{name}"),
            name: name.to_string(),
            arguments: args,
        }],
    })
}

#[tokio::test]
async fn five_line_agent_runs_react_loop() {
    // Provider scripted to: 1st call -> tool call to "echo", 2nd call -> "done"
    let provider = Arc::new(Sequencer::new(vec![
        ai_with_tool_call("echo", serde_json::json!({"x": 42})),
        Message::ai("done"),
    ]));
    let client = Client::new(provider);

    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(EchoTool) as Arc<dyn Tool>)
        .build()
        .unwrap();

    let resp = agent
        .run(Message::human("call echo then say done"))
        .await
        .unwrap();
    assert_eq!(resp.content, "done");
    assert_eq!(resp.state.iterations, 2);
    // Messages added during this run: ai(tool_call) + tool + ai(done) = 3
    assert_eq!(resp.messages.len(), 3);
}

#[tokio::test]
async fn stateful_mode_carries_history() {
    let provider = Arc::new(Sequencer::new(vec![
        Message::ai("hello"),
        Message::ai("again"),
    ]));
    let client = Client::new(provider);

    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .stateful()
        .build()
        .unwrap();

    let r1 = agent.run(Message::human("hi")).await.unwrap();
    assert_eq!(r1.content, "hello");

    let r2 = agent.run(Message::human("hi again")).await.unwrap();
    assert_eq!(r2.content, "again");
    // Memory should have grown: system + 2 user + 2 ai = 5 messages in seed
    let mem = agent.memory().unwrap();
    assert!(mem.seed().len() >= 4);
}

#[tokio::test]
async fn missing_llm_errors() {
    let err = AgentBuilder::new().build().unwrap_err();
    assert!(format!("{err}").contains("with_llm"));
}

#[tokio::test]
async fn max_iterations_capped() {
    // Provider always returns a tool call → infinite loop without max_iterations.
    let provider = Arc::new(Sequencer::new(vec![
        ai_with_tool_call("echo", serde_json::json!({})),
        ai_with_tool_call("echo", serde_json::json!({})),
        ai_with_tool_call("echo", serde_json::json!({})),
        ai_with_tool_call("echo", serde_json::json!({})),
        ai_with_tool_call("echo", serde_json::json!({})),
    ]));
    let client = Client::new(provider);

    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(EchoTool) as Arc<dyn Tool>)
        .with_max_iterations(2)
        .build()
        .unwrap();

    let resp = agent.run(Message::human("loop forever")).await.unwrap();
    assert!(resp.content.contains("max_iterations=2"));
}
