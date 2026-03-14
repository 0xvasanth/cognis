//! Message Graph Example
//!
//! Demonstrates building a message-based conversation graph where messages
//! flow through nodes that can classify, route, and respond to user input.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example message_graph`

#[path = "../shared.rs"]
mod shared;

use cognisgraph::graph::{
    GraphMessage, MessageGraph, MessageGraphBuilder, MessageNode, MessageRole, MessageState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Message Graph Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Build a graph with conditional routing
    // -----------------------------------------------------------------------
    // A classifier node inspects the user's message and labels it,
    // then a conditional edge routes to the appropriate handler.

    let mut graph = MessageGraph::new();

    graph.add_node(MessageNode::new("classifier", |state: &MessageState| {
        let query = state.last_human().map(|m| m.content.as_str()).unwrap_or("");
        let label = if query.contains("weather") {
            "weather"
        } else if query.contains("math") || query.contains('+') || query.contains("calculate") {
            "math"
        } else {
            "general"
        };
        vec![GraphMessage::new(
            MessageRole::Ai,
            format!("category:{}", label),
        )]
    }));

    graph.add_node(MessageNode::new("weather_handler", |_: &MessageState| {
        vec![GraphMessage::new(
            MessageRole::Ai,
            "It is sunny and 25C today.",
        )]
    }));

    graph.add_node(MessageNode::new("math_handler", |_: &MessageState| {
        vec![GraphMessage::new(MessageRole::Ai, "The answer is 42.")]
    }));

    graph.add_node(MessageNode::new("general_handler", |_: &MessageState| {
        vec![GraphMessage::new(
            MessageRole::Ai,
            "I can help with general questions.",
        )]
    }));

    graph.set_entry_point("classifier");
    graph.add_conditional_edge("classifier", |state: &MessageState| {
        let last = state.last_ai().map(|m| m.content.as_str()).unwrap_or("");
        if last.contains("category:weather") {
            "weather_handler".to_string()
        } else if last.contains("category:math") {
            "math_handler".to_string()
        } else {
            "general_handler".to_string()
        }
    });
    graph.set_finish_point("weather_handler");

    println!(
        "Graph: {} nodes, {} edges\n",
        graph.node_count(),
        graph.edge_count()
    );

    // Run several queries through the routing graph
    for query in [
        "What is the weather like?",
        "Please calculate 2+2",
        "Tell me a joke",
    ] {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, query));
        let result = graph.execute(state).unwrap();
        let response = result.last_ai().unwrap();
        println!("  Q: {}", query);
        println!("  A: {}\n", response.content);
    }

    // -----------------------------------------------------------------------
    // 2. Build the same pipeline with the fluent builder API
    // -----------------------------------------------------------------------
    println!("--- Builder API ---\n");

    let graph = MessageGraphBuilder::new()
        .node("start", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Starting pipeline...")]
        })
        .node("transform", |state: &MessageState| {
            let count = state.len();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Transformed. Messages so far: {}", count),
            )]
        })
        .node("finalize", |state: &MessageState| {
            let total = state.len();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Done. Total messages: {}", total + 1),
            )]
        })
        .edge("start", "transform")
        .edge("transform", "finalize")
        .entry("start")
        .finish("finalize")
        .build()
        .expect("Graph should be valid");

    let mut state = MessageState::new();
    state.add(GraphMessage::new(MessageRole::Human, "Go!"));
    let result = graph.execute(state).unwrap();
    for msg in result.messages() {
        println!("  [{}] {}", msg.role, msg.content);
    }

    // -----------------------------------------------------------------------
    // 3. LLM-powered node
    // -----------------------------------------------------------------------
    println!("\n--- LLM Demo ---\n");

    let model = shared::get_chat_model(vec![
        "Message graphs let you compose conversational workflows with routing logic.".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Explain how message graphs help build conversational AI workflows.",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  LLM: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
