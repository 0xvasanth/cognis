//! Graph Visualization Example
//!
//! Demonstrates building a StateGraph and exporting it for visualization:
//! - Building a StateGraph with nodes and edges (including conditional edges)
//! - Exporting as a Mermaid diagram (for GitHub, docs, mermaid.live)
//! - Exporting as ASCII art (for terminal display)
//! - Showing both vertical and compact rendering options
//!
//! No API keys required.
//!
//! Run with: cargo run -p cognis-examples --example graph_visualization

use std::collections::HashMap;
use std::sync::Arc;

use cognisgraph::graph::branch::RouterResult;
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};
use serde_json::{json, Value};

/// A no-op async node action for demonstration purposes.
fn noop_action() -> AsyncNodeAction {
    Arc::new(|_state: Value| Box::pin(async move { Ok(json!({})) }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Visualization Example ===\n");

    // -------------------------------------------------------------------------
    // Part 1: Simple linear graph
    // -------------------------------------------------------------------------
    println!("--- Part 1: Simple Linear Graph ---\n");

    let linear_graph = StateGraph::new()
        .add_node("fetch_data", noop_action())
        .add_node("process", noop_action())
        .add_node("store_results", noop_action())
        .set_entry_point("fetch_data")
        .add_edge("fetch_data", "process")
        .add_edge("process", "store_results")
        .set_finish_point("store_results")
        .compile()?;

    // Export as Mermaid diagram.
    let mermaid = linear_graph.draw_mermaid();
    println!("Mermaid diagram:");
    println!("{}", mermaid);
    println!();

    // Export as ASCII art.
    let ascii = linear_graph.draw_ascii();
    println!("ASCII art:");
    println!("{}", ascii);
    println!();

    // -------------------------------------------------------------------------
    // Part 2: Graph with conditional branching (ReAct-style agent)
    // -------------------------------------------------------------------------
    println!("--- Part 2: Conditional Branching (Agent Pattern) ---\n");

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

    let agent_mermaid = agent_graph.draw_mermaid();
    println!("Mermaid diagram (agent with tool loop):");
    println!("{}", agent_mermaid);
    println!();

    let agent_ascii = agent_graph.draw_ascii();
    println!("ASCII art:");
    println!("{}", agent_ascii);
    println!();

    // -------------------------------------------------------------------------
    // Part 3: Multi-branch graph (router pattern)
    // -------------------------------------------------------------------------
    println!("--- Part 3: Multi-Branch Router ---\n");

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

    let router_mermaid = router_graph.draw_mermaid();
    println!("Mermaid diagram (router):");
    println!("{}", router_mermaid);
    println!();

    let router_ascii = router_graph.draw_ascii();
    println!("ASCII art:");
    println!("{}", router_ascii);
    println!();

    // -------------------------------------------------------------------------
    // Part 4: Compact ASCII rendering
    // -------------------------------------------------------------------------
    println!("--- Part 4: Compact ASCII Rendering ---\n");

    let compact_opts = cognisgraph::graph::ascii::AsciiRenderOptions {
        compact: true,
        ..Default::default()
    };

    println!("Default rendering:");
    let default_ascii = linear_graph.draw_ascii();
    println!("{}", default_ascii);

    println!("Compact rendering:");
    let compact_ascii = linear_graph.draw_ascii_with_options(&compact_opts);
    println!("{}", compact_ascii);

    let default_lines = default_ascii.lines().count();
    let compact_lines = compact_ascii.lines().count();
    println!(
        "Line count: default={}, compact={}\n",
        default_lines, compact_lines
    );

    // -------------------------------------------------------------------------
    // Part 5: Mermaid live URL
    // -------------------------------------------------------------------------
    println!("--- Part 5: Mermaid Live URL ---\n");

    let url = agent_graph.draw_mermaid_url();
    println!("Open this URL to view the agent graph interactively:");
    println!("{}", url);

    println!("\nDone!");
    Ok(())
}
