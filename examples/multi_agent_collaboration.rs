//! Multi-Agent Collaboration Example
//!
//! Demonstrates creating multiple agents using deepagents with SubAgentMiddleware.
//! A researcher agent and a writer agent collaborate through a coordinated pipeline
//! built on top of LangGraph's StateGraph.
//!
//! No API keys required -- uses FakeMessagesListChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example multi_agent_collaboration

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use deepagents::agent::create_deep_agent;
use deepagents::config::DeepAgentConfig;
use deepagents::middleware::subagent::SubAgentMiddleware;
use langgraph::graph::state::{AsyncNodeAction, StateGraph};
use rustchain_core::language_models::FakeMessagesListChatModel;
use rustchain_core::messages::tool_types::ToolCall;
use rustchain_core::messages::{AIMessage, Message};
use rustchain_core::tools::BaseTool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Agent Collaboration Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Create the researcher agent
    // -------------------------------------------------------------------------
    // The researcher receives a topic and produces "research notes".
    // It delegates to a sub-agent for fact-checking via SubAgentMiddleware.

    println!("--- Step 1: Setting up researcher agent ---\n");

    // The sub-agent model (used by SubAgentMiddleware) returns fact-checked content.
    let subagent_model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
        AIMessage::new(
            "Fact-check result: Rust was first released in 2010 by Mozilla Research. \
             It reached version 1.0 in May 2015. The borrow checker is a key innovation.",
        ),
    )]));

    let subagent_mw = SubAgentMiddleware::new(subagent_model.clone(), 3);
    let subagent_tools = subagent_mw.tools();

    println!(
        "  SubAgentMiddleware provides {} tool(s):",
        subagent_tools.len()
    );
    for tool in &subagent_tools {
        println!("    - {} : {}", tool.name(), tool.description());
    }

    // The researcher model first calls the sub-agent tool, then returns findings.
    let mut researcher_tool_call = AIMessage::new("Let me research this topic.");
    researcher_tool_call.tool_calls.push(ToolCall {
        name: "delegate_to_subagent".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert(
                "task".to_string(),
                json!("Research the history of Rust programming language"),
            );
            m.insert(
                "context".to_string(),
                json!("Focus on key milestones and innovations"),
            );
            m
        },
        id: Some("call_research_001".to_string()),
    });

    let researcher_final = AIMessage::new(
        "Research findings: Rust was created by Mozilla Research, first released in 2010, \
         and reached 1.0 in May 2015. Its key innovation is the borrow checker which enables \
         memory safety without garbage collection. It has won 'most loved language' in Stack \
         Overflow surveys multiple years running.",
    );

    let researcher_model = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(researcher_tool_call),
        Message::Ai(researcher_final),
    ]));

    let researcher_config = DeepAgentConfig::default()
        .with_system_prompt("You are a thorough researcher. Use the delegate_to_subagent tool to fact-check important claims.")
        .with_tools(subagent_tools);

    let researcher_graph = create_deep_agent(researcher_model, researcher_config)?;
    println!(
        "  Researcher agent compiled: nodes = {:?}\n",
        researcher_graph.node_names()
    );

    // -------------------------------------------------------------------------
    // Step 2: Create the writer agent
    // -------------------------------------------------------------------------
    // The writer takes research notes and produces a polished article.

    println!("--- Step 2: Setting up writer agent ---\n");

    let writer_model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
        AIMessage::new(
            "# The Rust Programming Language: A Brief History\n\n\
             Rust emerged from Mozilla Research in 2010 as a bold experiment in systems \
             programming. The language reached its 1.0 milestone in May 2015, proving that \
             memory safety and performance need not be mutually exclusive.\n\n\
             At the heart of Rust's innovation lies the borrow checker -- a compile-time \
             system that enforces strict ownership rules. This eliminates entire classes of \
             bugs (use-after-free, data races) without the overhead of garbage collection.\n\n\
             The developer community has embraced Rust enthusiastically, voting it the 'most \
             loved programming language' in Stack Overflow surveys for multiple consecutive years.",
        ),
    )]));

    let writer_config = DeepAgentConfig::default().with_system_prompt(
        "You are a skilled technical writer. Transform research notes into polished articles.",
    );

    let writer_graph = create_deep_agent(writer_model, writer_config)?;
    println!(
        "  Writer agent compiled: nodes = {:?}\n",
        writer_graph.node_names()
    );

    // -------------------------------------------------------------------------
    // Step 3: Build a coordination graph connecting both agents
    // -------------------------------------------------------------------------
    // The coordination graph runs the researcher, extracts findings, then passes
    // them to the writer for polishing.

    println!("--- Step 3: Building coordination graph ---\n");

    let researcher_graph = Arc::new(researcher_graph);
    let writer_graph = Arc::new(writer_graph);

    let research_node: AsyncNodeAction = {
        let graph = researcher_graph.clone();
        Arc::new(move |state: Value| {
            let graph = graph.clone();
            Box::pin(async move {
                let topic = state
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Rust programming");

                println!("  [research] Researching topic: \"{topic}\"");

                let input = json!({
                    "messages": [
                        {"type": "human", "content": format!("Research this topic: {topic}")}
                    ]
                });

                let result = graph.invoke(input).await?;

                // Extract the last AI message as research notes.
                let messages = result["messages"].as_array().cloned().unwrap_or_default();
                let research_notes = messages
                    .iter()
                    .rev()
                    .find_map(|m| {
                        let msg: Message = serde_json::from_value(m.clone()).ok()?;
                        if let Message::Ai(ai) = &msg {
                            if ai.tool_calls.is_empty() {
                                return Some(ai.base.content.text());
                            }
                        }
                        None
                    })
                    .unwrap_or_else(|| "No research found.".to_string());

                println!(
                    "  [research] Produced {} chars of notes",
                    research_notes.len()
                );

                Ok(json!({
                    "research_notes": research_notes,
                    "research_complete": true,
                }))
            })
        })
    };

    let write_node: AsyncNodeAction = {
        let graph = writer_graph.clone();
        Arc::new(move |state: Value| {
            let graph = graph.clone();
            Box::pin(async move {
                let notes = state
                    .get("research_notes")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No notes provided.");

                println!(
                    "  [write] Writing article from {} chars of research notes",
                    notes.len()
                );

                let input = json!({
                    "messages": [
                        {"type": "human", "content": format!("Write a polished article based on these research notes:\n\n{notes}")}
                    ]
                });

                let result = graph.invoke(input).await?;

                let messages = result["messages"].as_array().cloned().unwrap_or_default();
                let article = messages
                    .iter()
                    .rev()
                    .find_map(|m| {
                        let msg: Message = serde_json::from_value(m.clone()).ok()?;
                        if let Message::Ai(_) = &msg {
                            Some(msg.content().text())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "Failed to write article.".to_string());

                println!("  [write] Produced {} chars of article", article.len());

                Ok(json!({
                    "article": article,
                    "write_complete": true,
                }))
            })
        })
    };

    let coordination_graph = StateGraph::new()
        .add_node("research", research_node)
        .add_node("write", write_node)
        .add_edge("__start__", "research")
        .add_edge("research", "write")
        .add_edge("write", "__end__")
        .compile()?;

    println!(
        "  Coordination graph nodes: {:?}\n",
        coordination_graph.node_names()
    );

    // -------------------------------------------------------------------------
    // Step 4: Run the full pipeline
    // -------------------------------------------------------------------------
    println!("--- Step 4: Running the collaboration pipeline ---\n");

    let result = coordination_graph
        .invoke(json!({ "topic": "The history and innovations of Rust programming language" }))
        .await?;

    // -------------------------------------------------------------------------
    // Step 5: Display the results
    // -------------------------------------------------------------------------
    println!("\n--- Final Results ---\n");

    if let Some(notes) = result.get("research_notes").and_then(|v| v.as_str()) {
        println!("Research Notes ({} chars):", notes.len());
        println!("  {}\n", &notes[..notes.len().min(120)]);
    }

    if let Some(article) = result.get("article").and_then(|v| v.as_str()) {
        println!("Final Article:\n");
        println!("{article}");
    }

    // Check sub-agent execution history.
    let subagents = subagent_mw.subagents().await;
    println!("\n--- Sub-Agent History ---");
    println!("  Total sub-agents spawned: {}", subagents.len());
    for (id, handle) in &subagents {
        println!(
            "  [{:.8}] task=\"{}\" status={:?}",
            id, handle.task, handle.status
        );
    }

    println!("\nDone!");
    Ok(())
}
