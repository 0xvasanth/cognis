//! What you'll learn:
//!   How to bypass `AgentBuilder` and assemble the underlying graph
//!   yourself, then hand it to `Agent::wrap` so it still drives like
//!   a normal agent.
//!
//! Why this matters:
//!   Most users never need this — but when you want a custom routing
//!   topology (multi-step planners, branching workflows, custom
//!   message-shaping), the graph is yours to build. `Agent::wrap` keeps
//!   the public `run`/`stream` surface identical regardless of what's
//!   inside.
//!
//! Scenario:
//!   Hand-built single-node graph that calls the LLM and ends. The shape
//!   you fall back to when `AgentBuilder`'s opinionated topology isn't
//!   right for your workflow.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example 03_custom_graph
//!
//! Sample output (against ollama / llama3.1):
//!   Hello!
//!
//!   A customized graph, I assume you'd like to create a specific type of graph with tailored features. Can you please provide more context or details about the kind of graph you're interested in? This will help me better understand and assist you.
//!
//!   Here are some examples of types of graphs:
//!
//!   1. Line Graph: for showing trends over time
//!   2. Bar Chart: for comparing categorical data
//!   ...
//!   5. Heatmap: for visualizing matrix data
//!
//!   Or perhaps you have something else in mind?

use cognis::prelude::*;
use cognis_llm::chat::ChatOptions;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;

    // Hand-build a single-node graph that calls the LLM and ends.
    let single_node = node_fn::<AgentState, _, _>("call", move |state, _ctx| {
        let client = client.clone();
        let messages = state.messages.clone();
        async move {
            let resp = client
                .provider()
                .chat_completion(messages, ChatOptions::default())
                .await?;
            Ok(NodeOut {
                update: AgentStateUpdate {
                    messages: vec![resp.message],
                    iterations: 1,
                },
                goto: Goto::end(),
            })
        }
    });
    let graph = Graph::<AgentState>::new()
        .node("call", single_node)
        .start_at("call")
        .compile()?;

    let mut agent = Agent::wrap(graph);
    let resp = agent.run(Message::human("hello custom graph")).await?;
    println!("{}", resp.content);
    Ok(())
}
