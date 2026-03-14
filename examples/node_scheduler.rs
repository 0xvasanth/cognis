//! Node Scheduler Example
//!
//! Demonstrates the LangGraph Pregel scheduler for managing node execution
//! order, priorities, resource allocation, and batch scheduling. The scheduler
//! supports multiple strategies and dependency-aware execution.
//!
//! Features shown:
//! - Creating ScheduledNodes with different priorities
//! - FIFO scheduling strategy
//! - Priority-based scheduling
//! - ShortestJobFirst scheduling
//! - RoundRobin scheduling
//! - Dependency-aware scheduling (nodes with dependencies)
//! - Batch scheduling with next_batch
//! - ResourcePool acquire/release
//! - ResourceAwareScheduler combining scheduling + resources
//! - SchedulerMetrics
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example node_scheduler`

mod shared;
use std::collections::HashSet;
use std::time::Duration;

use cognisgraph::pregel::scheduler::{
    NodePriority, ResourceAwareScheduler, ResourcePool, ScheduledNode, Scheduler, SchedulerMetrics,
    SchedulingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Node Scheduler Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating ScheduledNodes with Different Priorities
    // -----------------------------------------------------------------------
    println!("--- 1. Creating ScheduledNodes ---");
    println!(
        "Nodes carry a name, priority, estimated cost, dependencies, and optional deadline.\n"
    );

    let critical = ScheduledNode::new("critical_alert", NodePriority::Critical).with_cost(50);
    let high = ScheduledNode::new("process_request", NodePriority::High)
        .with_cost(200)
        .with_deadline(Duration::from_secs(10));
    let normal = ScheduledNode::new("compute_embeddings", NodePriority::Normal).with_cost(500);
    let low = ScheduledNode::new("update_cache", NodePriority::Low).with_cost(100);
    let background = ScheduledNode::new("cleanup_logs", NodePriority::Background).with_cost(30);

    println!("Created nodes:");
    for node in [&critical, &high, &normal, &low, &background] {
        println!(
            "  - {:25} priority={:?}, cost={}, deadline={:?}",
            node.name, node.priority, node.estimated_cost, node.deadline
        );
    }

    println!("\nPriority weights:");
    println!(
        "  Critical={}, High={}, Normal={}, Low={}, Background={}",
        NodePriority::Critical.weight(),
        NodePriority::High.weight(),
        NodePriority::Normal.weight(),
        NodePriority::Low.weight(),
        NodePriority::Background.weight(),
    );

    // -----------------------------------------------------------------------
    // 2. FIFO Scheduling Strategy
    // -----------------------------------------------------------------------
    println!("\n--- 2. FIFO Scheduling ---");
    println!("Nodes execute in the order they were added, regardless of priority.\n");

    let mut fifo = Scheduler::new(SchedulingStrategy::FIFO);
    fifo.add_node(ScheduledNode::new("first_added", NodePriority::Low));
    fifo.add_node(ScheduledNode::new("second_added", NodePriority::Critical));
    fifo.add_node(ScheduledNode::new("third_added", NodePriority::Normal));

    let completed = HashSet::new();
    println!("Execution order (FIFO):");
    while let Some(node) = fifo.next(&completed) {
        println!("  -> {} (priority: {:?})", node.name, node.priority);
    }

    // -----------------------------------------------------------------------
    // 3. Priority-Based Scheduling
    // -----------------------------------------------------------------------
    println!("\n--- 3. Priority-Based Scheduling ---");
    println!("Higher priority nodes execute first.\n");

    let mut priority_sched = Scheduler::new(SchedulingStrategy::Priority);
    priority_sched.add_node(ScheduledNode::new(
        "background_task",
        NodePriority::Background,
    ));
    priority_sched.add_node(ScheduledNode::new("normal_task", NodePriority::Normal));
    priority_sched.add_node(ScheduledNode::new("critical_task", NodePriority::Critical));
    priority_sched.add_node(ScheduledNode::new("high_task", NodePriority::High));
    priority_sched.add_node(ScheduledNode::new("low_task", NodePriority::Low));

    let completed = HashSet::new();
    println!("Execution order (Priority):");
    while let Some(node) = priority_sched.next(&completed) {
        println!(
            "  -> {:25} (priority: {:?}, weight: {})",
            node.name,
            node.priority,
            node.priority.weight()
        );
    }

    // -----------------------------------------------------------------------
    // 4. Shortest Job First Scheduling
    // -----------------------------------------------------------------------
    println!("\n--- 4. ShortestJobFirst Scheduling ---");
    println!("Nodes with the lowest estimated cost execute first.\n");

    let mut sjf = Scheduler::new(SchedulingStrategy::ShortestJobFirst);
    sjf.add_node(ScheduledNode::new("heavy_compute", NodePriority::Normal).with_cost(1000));
    sjf.add_node(ScheduledNode::new("quick_lookup", NodePriority::Normal).with_cost(10));
    sjf.add_node(ScheduledNode::new("medium_transform", NodePriority::Normal).with_cost(250));
    sjf.add_node(ScheduledNode::new("tiny_ping", NodePriority::Normal).with_cost(1));

    let completed = HashSet::new();
    println!("Execution order (ShortestJobFirst):");
    while let Some(node) = sjf.next(&completed) {
        println!("  -> {:25} (cost: {} MB)", node.name, node.estimated_cost);
    }

    // -----------------------------------------------------------------------
    // 5. Round Robin Scheduling
    // -----------------------------------------------------------------------
    println!("\n--- 5. RoundRobin Scheduling ---");
    println!("Nodes are cycled through in order.\n");

    let mut rr = Scheduler::new(SchedulingStrategy::RoundRobin);
    rr.add_node(ScheduledNode::new("agent_alpha", NodePriority::Normal));
    rr.add_node(ScheduledNode::new("agent_beta", NodePriority::Normal));
    rr.add_node(ScheduledNode::new("agent_gamma", NodePriority::Normal));

    let completed = HashSet::new();
    println!("Execution order (RoundRobin):");
    while let Some(node) = rr.next(&completed) {
        println!("  -> {}", node.name);
    }

    // -----------------------------------------------------------------------
    // 6. Dependency-Aware Scheduling
    // -----------------------------------------------------------------------
    println!("\n--- 6. Dependency-Aware Scheduling ---");
    println!("Nodes wait until their dependencies are completed.\n");

    let mut dep_sched = Scheduler::new(SchedulingStrategy::FIFO);
    dep_sched.add_node(ScheduledNode::new("fetch_data", NodePriority::High));
    dep_sched.add_node(
        ScheduledNode::new("parse_data", NodePriority::Normal).with_dependency("fetch_data"),
    );
    dep_sched.add_node(
        ScheduledNode::new("validate_data", NodePriority::Normal).with_dependency("parse_data"),
    );
    dep_sched.add_node(
        ScheduledNode::new("store_results", NodePriority::Low)
            .with_dependency("validate_data")
            .with_dependency("parse_data"),
    );

    let mut completed = HashSet::new();
    println!("Pipeline: fetch_data -> parse_data -> validate_data -> store_results");
    println!("Simulating step-by-step execution:\n");

    let mut step = 1;
    loop {
        match dep_sched.next(&completed) {
            Some(node) => {
                println!(
                    "  Step {}: Execute '{}' (deps satisfied: {:?})",
                    step, node.name, node.dependencies
                );
                completed.insert(node.name.clone());
                step += 1;
            }
            None => {
                if dep_sched.is_empty() {
                    println!("\n  All nodes completed.");
                } else {
                    println!(
                        "\n  {} nodes remain but dependencies are unsatisfied.",
                        dep_sched.remaining()
                    );
                }
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Batch Scheduling with next_batch
    // -----------------------------------------------------------------------
    println!("\n--- 7. Batch Scheduling ---");
    println!("Schedule up to N nodes in parallel batches.\n");

    let mut batch_sched = Scheduler::new(SchedulingStrategy::Priority);
    batch_sched.add_node(ScheduledNode::new("task_a", NodePriority::Critical));
    batch_sched.add_node(ScheduledNode::new("task_b", NodePriority::High));
    batch_sched.add_node(ScheduledNode::new("task_c", NodePriority::Normal));
    batch_sched.add_node(ScheduledNode::new("task_d", NodePriority::Normal));
    batch_sched.add_node(ScheduledNode::new("task_e", NodePriority::Low));

    let completed = HashSet::new();
    let mut batch_num = 1;

    loop {
        let batch = batch_sched.next_batch(&completed, 2);
        if batch.is_empty() {
            break;
        }
        let names: Vec<&str> = batch.iter().map(|n| n.name.as_str()).collect();
        println!(
            "  Batch {}: {:?} ({} nodes in parallel)",
            batch_num,
            names,
            batch.len()
        );
        batch_num += 1;
    }
    println!("  Remaining: {}", batch_sched.remaining());

    // Show batch with dependencies
    println!("\nBatch scheduling with dependencies:");
    let mut dep_batch = Scheduler::new(SchedulingStrategy::FIFO);
    dep_batch.add_node(ScheduledNode::new("download", NodePriority::Normal));
    dep_batch.add_node(ScheduledNode::new("preprocess", NodePriority::Normal));
    dep_batch
        .add_node(ScheduledNode::new("analyze", NodePriority::Normal).with_dependency("download"));

    let completed = HashSet::new();
    let batch = dep_batch.next_batch(&completed, 3);
    let names: Vec<&str> = batch.iter().map(|n| n.name.as_str()).collect();
    println!(
        "  Ready batch (max 3): {:?} ('analyze' blocked by 'download')",
        names
    );

    // -----------------------------------------------------------------------
    // 8. ResourcePool Acquire/Release
    // -----------------------------------------------------------------------
    println!("\n--- 8. ResourcePool ---");
    println!("Track concurrent execution slots and memory allocation.\n");

    let mut pool = ResourcePool::new(3, 1024);
    println!("Pool created: max_concurrent=3, max_memory=1024 MB");
    println!(
        "  Available: {} slots, {} MB",
        pool.available_slots(),
        pool.available_memory()
    );
    println!("  Utilization: {:.1}%\n", pool.utilization() * 100.0);

    // Acquire resources
    println!("Acquiring resources:");
    let acquired = pool.acquire(256);
    println!(
        "  acquire(256 MB): {} -> slots={}, memory={} MB, util={:.1}%",
        acquired,
        pool.available_slots(),
        pool.available_memory(),
        pool.utilization() * 100.0
    );

    let acquired = pool.acquire(512);
    println!(
        "  acquire(512 MB): {} -> slots={}, memory={} MB, util={:.1}%",
        acquired,
        pool.available_slots(),
        pool.available_memory(),
        pool.utilization() * 100.0
    );

    let acquired = pool.acquire(300);
    println!(
        "  acquire(300 MB): {} -> slots={}, memory={} MB",
        acquired,
        pool.available_slots(),
        pool.available_memory()
    );

    // Try to exceed memory
    let acquired = pool.acquire(500);
    println!(
        "\n  acquire(500 MB) - exceeds remaining memory: {}",
        acquired
    );

    // Release and re-acquire
    println!("\nReleasing resources:");
    pool.release(256);
    println!(
        "  release(256 MB): slots={}, memory={} MB",
        pool.available_slots(),
        pool.available_memory()
    );

    pool.release(512);
    println!(
        "  release(512 MB): slots={}, memory={} MB, util={:.1}%",
        pool.available_slots(),
        pool.available_memory(),
        pool.utilization() * 100.0
    );

    // -----------------------------------------------------------------------
    // 9. ResourceAwareScheduler
    // -----------------------------------------------------------------------
    println!("\n--- 9. ResourceAwareScheduler ---");
    println!("Combines scheduling strategy with resource constraints.\n");

    let pool = ResourcePool::new(2, 512);
    let mut ras = ResourceAwareScheduler::new(SchedulingStrategy::Priority, pool);

    ras.add_node(ScheduledNode::new("llm_call", NodePriority::Critical).with_cost(256));
    ras.add_node(ScheduledNode::new("embedding", NodePriority::High).with_cost(128));
    ras.add_node(ScheduledNode::new("indexing", NodePriority::Normal).with_cost(384));
    ras.add_node(ScheduledNode::new("logging", NodePriority::Low).with_cost(32));

    let completed = HashSet::new();
    println!("Nodes: llm_call(256MB), embedding(128MB), indexing(384MB), logging(32MB)");
    println!("Pool:  2 slots, 512 MB max\n");

    // First batch
    let batch = ras.schedule_next(&completed);
    println!("Batch 1:");
    for node in &batch {
        println!(
            "  -> {} (priority: {:?}, cost: {} MB)",
            node.name, node.priority, node.estimated_cost
        );
    }
    println!("  Remaining: {}", ras.remaining());

    // Release resources from first batch
    for node in &batch {
        ras.release_resources(node.estimated_cost);
    }
    println!("\n  (released resources from batch 1)");

    // Second batch
    let batch2 = ras.schedule_next(&completed);
    println!("\nBatch 2:");
    for node in &batch2 {
        println!(
            "  -> {} (priority: {:?}, cost: {} MB)",
            node.name, node.priority, node.estimated_cost
        );
    }
    println!("  Remaining: {}", ras.remaining());

    // Release and get remaining
    for node in &batch2 {
        ras.release_resources(node.estimated_cost);
    }

    if !ras.is_empty() {
        let batch3 = ras.schedule_next(&completed);
        println!("\nBatch 3:");
        for node in &batch3 {
            println!(
                "  -> {} (priority: {:?}, cost: {} MB)",
                node.name, node.priority, node.estimated_cost
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. SchedulerMetrics
    // -----------------------------------------------------------------------
    println!("\n--- 10. SchedulerMetrics ---");
    println!("Track scheduling statistics.\n");

    let mut metrics = SchedulerMetrics::new();
    println!("Initial metrics: {:?}", metrics);

    // Simulate scheduling events
    metrics.record_scheduling(3, 750);
    metrics.record_batch();
    metrics.record_scheduling(2, 200);
    metrics.record_batch();
    metrics.record_scheduling(1, 50);
    metrics.record_batch();

    println!("\nAfter simulated scheduling:");
    println!("  Nodes scheduled: {}", metrics.nodes_scheduled);
    println!("  Total cost:      {} MB", metrics.total_cost);
    println!("  Batches created:  {}", metrics.batches_created);
    println!(
        "  Average wait:     {} ms",
        metrics.average_wait_time.as_millis()
    );

    println!("\nMetrics as JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&metrics.to_json()).unwrap()
    );

    // Show metrics from ResourceAwareScheduler
    println!("\nMetrics from ResourceAwareScheduler:");
    let ras_metrics = ras.metrics();
    println!(
        "  {}",
        serde_json::to_string_pretty(&ras_metrics.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 11. Real LLM Demo — LLM call in scheduled node context
    // -----------------------------------------------------------------------
    println!("\n--- 11. Real LLM Demo ---");
    println!("Simulate an LLM call as a scheduled node task.\n");

    let model = shared::get_chat_model(vec![
        "For optimal scheduling, prioritize critical LLM calls first, batch similar requests, and use resource-aware scheduling to avoid overloading the model API.".into(),
    ]);
    let messages = vec![
        cognis_core::messages::Message::human(
            "What is the best strategy for scheduling multiple LLM calls in a pipeline?"
        ),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  LLM Response: {}", gen.message.content().text());
    }

    println!("\n=== Node Scheduler Example Complete ===");
    Ok(())
}
