//! Node Scheduler Example
//!
//! Demonstrates priority-based scheduling with dependencies and resource
//! constraints using the LangGraph Pregel scheduler.
//!
//! Run with: `cargo run -p cognis-examples --example node_scheduler`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashSet;
use std::time::Duration;

use cognisgraph::pregel::scheduler::{
    NodePriority, ResourceAwareScheduler, ResourcePool, ScheduledNode, SchedulingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Node Scheduler Example ===\n");

    // --- Build a pipeline with priorities and dependencies ---
    // Scenario: fetch -> parse -> validate -> store
    let mut scheduler = ResourceAwareScheduler::new(
        SchedulingStrategy::Priority,
        ResourcePool::new(2, 512), // 2 concurrent slots, 512 MB
    );

    scheduler.add_node(
        ScheduledNode::new("fetch_data", NodePriority::Critical)
            .with_cost(64)
            .with_deadline(Duration::from_secs(5)),
    );
    scheduler.add_node(
        ScheduledNode::new("parse_data", NodePriority::High)
            .with_cost(128)
            .with_dependency("fetch_data"),
    );
    scheduler.add_node(
        ScheduledNode::new("validate_data", NodePriority::Normal)
            .with_cost(256)
            .with_dependency("parse_data"),
    );
    scheduler.add_node(
        ScheduledNode::new("store_results", NodePriority::Low)
            .with_cost(32)
            .with_dependency("validate_data"),
    );

    println!("Pipeline: fetch_data -> parse_data -> validate_data -> store_results");
    println!("Resource pool: 2 slots, 512 MB\n");

    // --- Execute in dependency-aware batches ---
    let mut completed = HashSet::new();
    let mut batch_num = 1;

    loop {
        let batch = scheduler.schedule_next(&completed);
        if batch.is_empty() {
            break;
        }

        println!("Batch {}:", batch_num);
        for node in &batch {
            println!(
                "  {} (priority: {:?}, cost: {} MB)",
                node.name, node.priority, node.estimated_cost
            );
            completed.insert(node.name.clone());
        }

        // Release resources after batch completes
        for node in &batch {
            scheduler.release_resources(node.estimated_cost);
        }
        batch_num += 1;
    }

    // --- Print scheduling metrics ---
    let metrics = scheduler.metrics();
    println!("\nScheduler metrics:");
    println!(
        "  Nodes scheduled: {}, Total cost: {} MB, Batches: {}",
        metrics.nodes_scheduled, metrics.total_cost, metrics.batches_created
    );

    // --- LLM demo: ask about scheduling strategies ---
    println!("\n--- LLM Demo ---");
    let model = shared::get_chat_model(vec![
        "For optimal scheduling, prioritize critical calls first, batch independent \
         tasks, and use resource limits to prevent API overload."
            .into(),
    ]);

    let messages = vec![cognis_core::messages::Message::human(
        "What is the best strategy for scheduling multiple LLM calls in a pipeline?",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
