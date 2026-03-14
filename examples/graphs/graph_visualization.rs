//! Graph Visualization Example
//!
//! Demonstrates building StateGraphs and exporting as Mermaid diagrams and ASCII art.
//!
//! Run with: cargo run -p cognis-examples --example graph_visualization

#[path = "../shared.rs"]
mod shared;
use std::collections::HashMap;
use std::sync::Arc;

use cognisgraph::graph::branch::RouterResult;
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};
use serde_json::{json, Value};

fn noop_action() -> AsyncNodeAction {
    Arc::new(|_state: Value| Box::pin(async move { Ok(json!({})) }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Linear graph
    let linear_graph = StateGraph::new()
        .add_node("fetch_data", noop_action())
        .add_node("process", noop_action())
        .add_node("store_results", noop_action())
        .set_entry_point("fetch_data")
        .add_edge("fetch_data", "process")
        .add_edge("process", "store_results")
        .set_finish_point("store_results")
        .compile()?;

    println!("Linear graph (Mermaid):\n{}", linear_graph.draw_mermaid());
    println!("Linear graph (ASCII):\n{}", linear_graph.draw_ascii());

    // Conditional branching (ReAct-style agent)
    let mut path_map = HashMap::new();
    path_map.insert("continue".to_string(), "tools".to_string());
    path_map.insert("end".to_string(), "__end__".to_string());

    let router = Arc::new(|_state: &Value| RouterResult::Single("continue".to_string()));

    let agent_graph = StateGraph::new()
        .add_node("agent", noop_action())
        .add_node("tools", noop_action())
        .set_entry_point("agent")
        .add_conditional_edges("agent", router, Some(path_map))
        .add_edge("tools", "agent")
        .compile()?;

    println!("Agent graph (Mermaid):\n{}", agent_graph.draw_mermaid());
    println!("Agent graph (ASCII):\n{}", agent_graph.draw_ascii());

    // Multi-branch router
    let mut router_map = HashMap::new();
    router_map.insert("technical".to_string(), "tech_support".to_string());
    router_map.insert("billing".to_string(), "billing_agent".to_string());
    router_map.insert("general".to_string(), "general_agent".to_string());

    let classify_router = Arc::new(|_state: &Value| RouterResult::Single("technical".to_string()));

    let router_graph = StateGraph::new()
        .add_node("classifier", noop_action())
        .add_node("tech_support", noop_action())
        .add_node("billing_agent", noop_action())
        .add_node("general_agent", noop_action())
        .add_node("respond", noop_action())
        .set_entry_point("classifier")
        .add_conditional_edges("classifier", classify_router, Some(router_map))
        .add_edge("tech_support", "respond")
        .add_edge("billing_agent", "respond")
        .add_edge("general_agent", "respond")
        .set_finish_point("respond")
        .compile()?;

    println!("Router graph (Mermaid):\n{}", router_graph.draw_mermaid());

    // Compact vs default ASCII rendering
    let compact_opts = cognisgraph::graph::ascii::AsciiRenderOptions {
        compact: true,
        ..Default::default()
    };
    println!("Default:\n{}", linear_graph.draw_ascii());
    println!(
        "Compact:\n{}",
        linear_graph.draw_ascii_with_options(&compact_opts)
    );

    // Mermaid live URL
    println!("Mermaid live URL: {}", agent_graph.draw_mermaid_url());

    // LLM demo
    let model = shared::get_chat_model(vec![
        "Graph visualization helps developers understand data flow and debug pipelines.".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Why is graph visualization useful for understanding agent workflows?",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
