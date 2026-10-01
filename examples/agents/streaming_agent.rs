//! What you'll learn:
//!   Three ways to consume a model's output as it is produced: agent token
//!   streaming (`with_streaming`), typed list elements (`stream_elements`),
//!   and progressively-filled objects (`stream_partial`).
//!
//! Why this matters:
//!   UIs and pipelines can act on the first token, the first finished list
//!   item, or the first populated field without waiting for the full reply.
//!
//! Scenario:
//!   A scripted offline provider streams canned text fragments, so each
//!   section shows exactly what arrives and when. `stream_elements` and
//!   `stream_partial` add no format instructions; the prompt must ask the
//!   model for a JSON array / object.
//!
//! Run with:
//!   cargo run -p cognis-examples --example agents_streaming_agent
//!
//! Sample output:
//!   == 1. token streaming ==
//!   Paris| is| the| capital.|
//!   == 2. stream_elements::<Step> ==
//!   step 1: gather
//!   step 2: write
//!   == 3. stream_partial::<Report> ==
//!   ReportPartial { title: Some("Q3 "), score: None }
//!   ReportPartial { title: Some("Q3 summary"), score: None }
//!   ReportPartial { title: Some("Q3 summary"), score: Some(9) }
//!   ReportPartial { title: Some("Q3 summary"), score: Some(92) }

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::RunnableStream;
use cognis_llm::HealthStatus;
use cognis_macros::Partial;
use futures::StreamExt;
use serde::Deserialize;

/// Streams the given text fragments as delta chunks, one per fragment.
struct ScriptedProvider {
    fragments: Vec<&'static str>,
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

    async fn chat_completion_stream(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        let chunks: Vec<Result<StreamChunk>> = self
            .fragments
            .iter()
            .map(|f| {
                Ok(StreamChunk {
                    content: (*f).into(),
                    is_delta: true,
                    is_done: false,
                    finish_reason: None,
                    usage: None,
                    tool_calls_delta: vec![],
                })
            })
            .collect();
        Ok(RunnableStream::new(futures::stream::iter(chunks)))
    }

    async fn chat_completion_stream_with_tools(
        &self,
        messages: Vec<Message>,
        _t: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        self.chat_completion_stream(messages, opts).await
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::Healthy { latency_ms: 0 })
    }
}

fn agent_streaming(fragments: &[&'static str], streaming: bool) -> Result<Agent> {
    let client = Client::new(Arc::new(ScriptedProvider {
        fragments: fragments.to_vec(),
    }));
    AgentBuilder::new()
        .with_llm(client)
        .with_streaming(streaming)
        .build()
}

#[derive(Deserialize)]
struct Step {
    id: u32,
    title: String,
}

#[derive(Partial)]
#[allow(dead_code)]
struct Report {
    title: String,
    score: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. token streaming ==");
    let mut agent = agent_streaming(&["Paris", " is", " the", " capital."], true)?;
    let mut events = agent.stream(Message::human("Capital of France?")).await?;
    while let Some(ev) = events.next().await {
        if let Event::OnLlmToken { token, .. } = ev {
            print!("{token}|");
        }
    }
    println!();

    println!("== 2. stream_elements::<Step> ==");
    // The second element is split across chunks; it is yielded only once it closes.
    let mut agent = agent_streaming(
        &[
            "[{\"id\":1,\"title\":\"gat",
            "her\"},{\"id\":2,\"ti",
            "tle\":\"write\"}]",
        ],
        false,
    )?;
    let mut steps = agent
        .stream_elements::<Step>(Message::human("Reply with a JSON array of steps."))
        .await?;
    while let Some(step) = steps.next().await {
        let step = step?;
        println!("step {}: {}", step.id, step.title);
    }

    println!("== 3. stream_partial::<Report> ==");
    let mut agent = agent_streaming(
        &["{\"title\":\"Q3 ", "summary\",", "\"score\":9", "2}"],
        false,
    )?;
    let mut snapshots = agent
        .stream_partial::<Report>(Message::human("Reply with a JSON object."))
        .await?;
    while let Some(snapshot) = snapshots.next().await {
        println!("{:?}", snapshot?);
    }

    Ok(())
}
