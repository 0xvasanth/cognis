//! What you'll learn:
//!   How the streaming APIs behave against a real local model: token
//!   streaming through a tool call, typed list elements, and
//!   progressively-filled objects.
//!
//! Why this matters:
//!   A scripted provider proves the wiring; a live server proves the wire
//!   format. Each section prints when every item arrived, so you can see
//!   whether output is incremental or delivered in one piece.
//!
//! Scenario:
//!   Section 1 uses Ollama's native API, which has no streaming tool-call
//!   path yet, so each model call arrives as one chunk. Section 2 points the
//!   OpenAI provider at Ollama's OpenAI-compatible `/v1` endpoint, which
//!   streams answer tokens one by one (Ollama still sends each tool call as
//!   a single event with complete arguments). Sections 3 and 4 are single
//!   model turns over Ollama's native stream; the prompt asks for the JSON
//!   shape because no format instruction is injected.
//!
//! Run with (needs a tool-capable model, e.g. `qwen2.5:3b` or `llama3.1`):
//!   ollama pull qwen2.5:3b
//!   COGNIS_OLLAMA_MODEL=qwen2.5:3b \
//!     cargo run -p cognis-examples --example agents_streaming_ollama
//!
//! `OLLAMA_HOST` overrides the server address (default
//! `http://localhost:11434`).

use std::sync::Arc;
use std::time::Instant;

use cognis::prelude::*;
use futures::StreamExt;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Step {
    id: u32,
    title: String,
}

#[derive(Partial)]
#[partial(crate = "cognis::cognis_core")]
#[allow(dead_code)]
struct Report {
    title: String,
    summary: String,
    score: u32,
}

/// Streams one agent run and prints each token and tool event with its
/// arrival time. Returns how many token events arrived.
async fn run_tool_agent(client: Client) -> Result<usize> {
    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(Calculator::new()))
        .with_system_prompt(
            "You are a math assistant. Use the `calculator` tool for every \
             calculation, then state the final answer in one sentence.",
        )
        .with_max_iterations(4)
        .with_streaming(true)
        .build()?;

    let started = Instant::now();
    let mut tokens = 0usize;
    let mut events = agent.stream(Message::human("What is 23 * 17 + 4?")).await?;
    while let Some(ev) = events.next().await {
        let ms = started.elapsed().as_millis();
        match ev {
            Event::OnLlmToken { token, .. } => {
                tokens += 1;
                println!("  [{ms:>6} ms] token {token:?}");
            }
            Event::OnToolStart { tool, args, .. } => {
                println!("  [{ms:>6} ms] tool start: {tool} {args}");
            }
            Event::OnToolEnd { tool, result, .. } => {
                println!("  [{ms:>6} ms] tool end:   {tool} -> {result}");
            }
            Event::OnError { error, .. } => println!("  [{ms:>6} ms] ERROR {error}"),
            _ => {}
        }
    }
    println!("  token events: {tokens}");
    Ok(tokens)
}

#[tokio::main]
async fn main() -> Result<()> {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let host = host.trim_end_matches('/').to_string();
    let model = std::env::var("COGNIS_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:3b".into());
    println!("server: {host}  model: {model}\n");

    let native = || {
        Client::builder()
            .provider(Provider::Ollama)
            .base_url(format!("{host}/api/"))
            .model(model.clone())
            .build()
    };

    println!("== 1. agent + tool, native Ollama API (fallback: one chunk per model call) ==");
    run_tool_agent(native()?).await?;

    println!("\n== 2. agent + tool, OpenAI-compatible /v1 endpoint (native token streaming) ==");
    let openai_compat = Client::builder()
        .provider(Provider::OpenAI)
        .base_url(format!("{host}/v1/"))
        .api_key("ollama")
        .model(model.clone())
        .build()?;
    run_tool_agent(openai_compat).await?;

    println!("\n== 3. stream_elements::<Step> ==");
    let mut agent = AgentBuilder::new()
        .with_llm(native()?)
        .with_system_prompt(
            "Reply with ONLY a JSON array, no prose and no code fence. Each \
             element is an object with an integer `id` and a string `title`.",
        )
        .build()?;
    let started = Instant::now();
    let mut steps = agent
        .stream_elements::<Step>(Message::human("List 5 steps to brew pour-over coffee."))
        .await?;
    let mut count = 0usize;
    while let Some(step) = steps.next().await {
        let ms = started.elapsed().as_millis();
        match step {
            Ok(step) => {
                count += 1;
                println!("  [{ms:>6} ms] step {}: {}", step.id, step.title);
            }
            Err(e) => println!("  [{ms:>6} ms] ERROR {e}"),
        }
    }
    println!("  elements: {count}");

    println!("\n== 4. stream_partial::<Report> ==");
    let mut agent = AgentBuilder::new()
        .with_llm(native()?)
        .with_system_prompt(
            "Reply with ONLY one JSON object, no prose and no code fence, with \
             exactly these keys in this order: `title` (string), `summary` \
             (string, two sentences), `score` (integer 0-100).",
        )
        .build()?;
    let started = Instant::now();
    let mut snapshots = agent
        .stream_partial::<Report>(Message::human(
            "Write a short review of the Rust borrow checker.",
        ))
        .await?;
    let mut count = 0usize;
    let mut last = None;
    while let Some(snapshot) = snapshots.next().await {
        let ms = started.elapsed().as_millis();
        match snapshot {
            Ok(report) => {
                count += 1;
                let summary_len = report.summary.as_deref().map_or(0, str::len);
                println!(
                    "  [{ms:>6} ms] title={:?} summary_len={summary_len} score={:?}",
                    report.title, report.score
                );
                last = Some(report);
            }
            Err(e) => println!("  [{ms:>6} ms] ERROR {e}"),
        }
    }
    println!("  snapshots: {count}");
    if let Some(report) = last {
        println!("  final: {report:?}");
    }
    Ok(())
}
