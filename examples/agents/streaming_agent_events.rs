//! What you'll learn:
//!   How to consume an agent run as a stream of structured `Event`s —
//!   `OnNodeStart`, `OnNodeEnd`, `OnEnd` — instead of waiting for the
//!   final `AgentResponse`.
//!
//! Why this matters:
//!   Event streaming is what powers production tracing, progress UIs,
//!   and per-step instrumentation. The same event types flow through
//!   any graph-backed runnable, so an observer you write here works
//!   equally for a custom graph in `cognisgraph`.
//!
//! Scenario:
//!   The user asks "What is the capital of France?". Instead of
//!   awaiting the final reply, we tail the agent's `Event` stream and
//!   print every node start/end — what a tracing UI consumes in real
//!   code.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example agents_streaming_events
//!
//! Sample output (against ollama / llama3.1):
//!   [start] step=0 node=think
//!   [end]   step=0 node=think
//!   [done]

use cognis::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("Answer in one sentence.")
        .build()?;

    let mut s = agent
        .stream(Message::human("What is the capital of France?"))
        .await?;
    while let Some(ev) = s.next().await {
        match ev {
            Event::OnNodeStart { node, step, .. } => println!("[start] step={step} node={node}"),
            Event::OnNodeEnd { node, step, .. } => println!("[end]   step={step} node={node}"),
            Event::OnEnd { .. } => println!("[done]"),
            _ => {}
        }
    }
    Ok(())
}
