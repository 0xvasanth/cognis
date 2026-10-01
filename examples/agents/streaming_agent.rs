//! What you'll learn:
//!   Three ways to consume a model's output as it is produced: agent token
//!   streaming through a tool call (`with_streaming`), typed list elements
//!   (`stream_elements`), and progressively-filled objects (`stream_partial`).
//!
//! Why this matters:
//!   UIs and pipelines can act on the first token, the first finished list
//!   item, or the first populated field without waiting for the full reply.
//!
//! Scenario:
//!   A scripted offline provider replays canned stream chunks, one list per
//!   model call, so each section shows exactly what arrives and when.
//!   Section 1 runs the full agent loop: the model streams some text and a
//!   tool call whose arguments are split across chunks, the tool runs, and
//!   a second model call streams the answer. Sections 2 and 3 are single
//!   model turns with no tool loop; they add no format instructions, so the
//!   prompt must ask the model for a JSON array / object.
//!
//! Run with:
//!   cargo run -p cognis-examples --example agents_streaming_agent
//!
//! Sample output:
//!   == 1. token streaming through a tool call ==
//!   Let me| check.|
//!   [tool start] weather {"city":"Paris"}
//!   [tool end] weather -> 18C and sunny in Paris
//!   It is| 18C| and sunny| in Paris.|
//!   == 2. stream_elements::<Step> ==
//!   step 1: gather
//!   step 2: write
//!   == 3. stream_partial::<Report> ==
//!   ReportPartial { title: Some("Q3 "), score: None }
//!   ReportPartial { title: Some("Q3 summary"), score: None }
//!   ReportPartial { title: Some("Q3 summary"), score: Some(92) }

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cognis::cognis_llm::chat::ToolCallDelta;
use cognis::cognis_llm::HealthStatus;
use cognis::prelude::*;
use futures::StreamExt;
use serde::Deserialize;

/// Replays one scripted list of chunks per model call, in order.
struct ScriptedProvider {
    turns: Mutex<VecDeque<Vec<StreamChunk>>>,
}

impl ScriptedProvider {
    fn next_turn(&self) -> RunnableStream<StreamChunk> {
        let chunks = self
            .turns
            .lock()
            .ok()
            .and_then(|mut turns| turns.pop_front())
            .unwrap_or_default();
        RunnableStream::new(futures::stream::iter(chunks.into_iter().map(Ok)))
    }
}

#[async_trait]
impl LLMProvider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }

    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }

    async fn chat_completion(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        unreachable!("this demo only streams")
    }

    /// The plain streaming path: what `stream_elements` / `stream_partial` use.
    async fn chat_completion_stream(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        Ok(self.next_turn())
    }

    /// The tool-calling streaming path: what a streaming agent's think step uses.
    async fn chat_completion_stream_with_tools(
        &self,
        _m: Vec<Message>,
        _t: Vec<ToolDefinition>,
        _o: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        Ok(self.next_turn())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

/// A text delta.
fn text(content: &str) -> StreamChunk {
    StreamChunk {
        content: content.into(),
        is_delta: true,
        ..Default::default()
    }
}

/// A fragment of a tool call. Providers send `id` and `name` once, on the
/// first fragment, and the JSON arguments in pieces.
fn tool_call(id: Option<&str>, name: Option<&str>, arguments: &str) -> StreamChunk {
    StreamChunk {
        is_delta: true,
        tool_calls_delta: vec![ToolCallDelta {
            index: 0,
            id: id.map(String::from),
            name: name.map(String::from),
            arguments_delta: Some(arguments.into()),
        }],
        ..Default::default()
    }
}

/// The terminal chunk of a model turn.
fn finish(reason: &str) -> StreamChunk {
    StreamChunk {
        is_done: true,
        finish_reason: Some(reason.into()),
        ..Default::default()
    }
}

/// A client whose model calls replay `turns`, one entry per call.
fn scripted_client(turns: Vec<Vec<StreamChunk>>) -> Client {
    Client::new(Arc::new(ScriptedProvider {
        turns: Mutex::new(turns.into()),
    }))
}

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "weather"
    }

    fn description(&self) -> &str {
        "Current weather for a city."
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args = input.into_json();
        let city = args["city"].as_str().unwrap_or("somewhere");
        Ok(ToolOutput::Text(format!("18C and sunny in {city}")))
    }
}

#[derive(Deserialize)]
struct Step {
    id: u32,
    title: String,
}

// `Partial` here is both the derive and the trait, re-exported by the
// umbrella crate. The `crate` attribute points the derive at the core crate
// through `cognis`, which is what you need when `cognis` is your only
// dependency; drop it if you depend on `cognis-core` directly.
#[derive(Partial)]
#[partial(crate = "cognis::cognis_core")]
#[allow(dead_code)]
struct Report {
    title: String,
    score: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. token streaming through a tool call ==");
    let client = scripted_client(vec![
        // First model call: some text, then a tool call whose JSON
        // arguments arrive in three fragments.
        vec![
            text("Let me"),
            text(" check."),
            tool_call(Some("call_1"), Some("weather"), "{\"ci"),
            tool_call(None, None, "ty\":\"Pa"),
            tool_call(None, None, "ris\"}"),
            finish("tool_calls"),
        ],
        // Second model call, after the tool result: the final answer.
        vec![
            text("It is"),
            text(" 18C"),
            text(" and sunny"),
            text(" in Paris."),
            finish("stop"),
        ],
    ]);
    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(WeatherTool))
        .with_streaming(true)
        .build()?;
    let mut events = agent.stream(Message::human("Weather in Paris?")).await?;
    while let Some(ev) = events.next().await {
        match ev {
            Event::OnLlmToken { token, .. } => print!("{token}|"),
            // Tokens of a model turn all arrive before its node ends.
            Event::OnNodeEnd { node, .. } if node == "think" => println!(),
            Event::OnToolStart { tool, args, .. } => println!("[tool start] {tool} {args}"),
            Event::OnToolEnd { tool, result, .. } => {
                println!(
                    "[tool end] {tool} -> {}",
                    result.as_str().unwrap_or_default()
                )
            }
            _ => {}
        }
    }

    println!("== 2. stream_elements::<Step> ==");
    // The elements are split across chunks; each is yielded once it closes.
    let client = scripted_client(vec![vec![
        text("[{\"id\":1,\"title\":\"gat"),
        text("her\"},{\"id\":2,\"ti"),
        text("tle\":\"write\"}]"),
    ]]);
    let mut agent = AgentBuilder::new().with_llm(client).build()?;
    let mut steps = agent
        .stream_elements::<Step>(Message::human("Reply with a JSON array of steps."))
        .await?;
    while let Some(step) = steps.next().await {
        let step = step?;
        println!("step {}: {}", step.id, step.title);
    }

    println!("== 3. stream_partial::<Report> ==");
    // The title streams as a growing prefix. The score is withheld while it
    // could still grow (`9` might become `92`) and appears once `}` ends it.
    let client = scripted_client(vec![vec![
        text("{\"title\":\"Q3 "),
        text("summary\","),
        text("\"score\":9"),
        text("2}"),
    ]]);
    let mut agent = AgentBuilder::new().with_llm(client).build()?;
    let mut snapshots = agent
        .stream_partial::<Report>(Message::human("Reply with a JSON object."))
        .await?;
    while let Some(snapshot) = snapshots.next().await {
        println!("{:?}", snapshot?);
    }

    Ok(())
}
