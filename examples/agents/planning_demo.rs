//! What you'll learn:
//!   How to wrap a `Client` in a `MiddlewarePipeline` so every call to
//!   the LLM is preceded by a "plan-then-act" system fragment, then
//!   feed the resulting client back into an `AgentBuilder`.
//!
//! Why this matters:
//!   `Planning` is one of several drop-in middlewares — alongside
//!   `RateLimit`, `PromptCaching`, `PiiRedactor` — that decorate a
//!   client without changing the agent code that consumes it. This is
//!   how you bolt cross-cutting policy onto an LLM call site.
//!
//! Scenario:
//!   The user asks "Plan how to make tea." The `Planning` middleware
//!   silently injects a "think first, then act" system fragment so the
//!   model produces a plan before writing the recipe.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example agents_planning
//!
//! Sample output (against ollama / llama3.1):
//!   pipelined client name: ollama
//!   Here's a step-by-step plan for making tea:
//!
//!   **Tools Needed:**
//!
//!   * Tea leaves (black, green, or herbal)
//!   * Teapot
//!   * Tea infuser (optional)
//!   ...
//!   * Adjust the amount of sugar or milk based on personal preference.
//!
//!   Enjoy your perfect cup of tea!

use cognis::prelude::*;
use cognis::{MiddlewarePipeline, Planning};

#[tokio::main]
async fn main() -> Result<()> {
    let raw = Client::from_env()?;
    let pipe = MiddlewarePipeline::new().push(Planning::new()).build(raw);
    println!("pipelined client name: {}", pipe.client().provider().name());

    let mut agent = AgentBuilder::new()
        .with_llm(pipe.client().clone())
        .build()?;
    let r = agent.run(Message::human("Plan how to make tea.")).await?;
    println!("{}", r.content);
    Ok(())
}
