//! Graph Metrics Example
//!
//! Simulates a multi-node graph pipeline, collects execution metrics across
//! runs, detects bottlenecks, and asks an LLM to suggest optimizations.
//!
//! Run with: `cargo run -p cognis-examples --example graph_metrics`

#[path = "../shared.rs"]
mod shared;

use std::time::Duration;

use cognis_core::messages::Message;
use cognisgraph::graph::metrics::{
    GraphMetrics, InMemoryMetricsCollector, MetricsAggregator, MetricsCollector, MetricsExporter,
    MetricsReport,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Metrics Example ===\n");

    // -- Collect metrics across two simulated graph runs -----------------------
    let mut collector = InMemoryMetricsCollector::new();

    // Run 1: happy path (fetch -> process -> output)
    collector.on_run_start("run-001");
    collector.on_node_executed("fetch", Duration::from_millis(120), false);
    collector.on_edge_traversed("fetch", "process");
    collector.on_node_executed("process", Duration::from_millis(250), false);
    collector.on_edge_traversed("process", "output");
    collector.on_node_executed("output", Duration::from_millis(30), false);
    collector.on_run_finish("run-001");

    // Run 2: process errors, triggers retry loop
    collector.on_run_start("run-002");
    collector.on_node_executed("fetch", Duration::from_millis(95), false);
    collector.on_edge_traversed("fetch", "process");
    collector.on_node_executed("process", Duration::from_millis(200), true);
    collector.on_edge_traversed("process", "retry");
    collector.on_node_executed("retry", Duration::from_millis(50), false);
    collector.on_edge_traversed("retry", "process");
    collector.on_node_executed("process", Duration::from_millis(70), false);
    collector.on_edge_traversed("process", "output");
    collector.on_node_executed("output", Duration::from_millis(25), false);
    collector.on_run_finish("run-002");

    // -- Aggregate and report --------------------------------------------------
    let mut aggregator = MetricsAggregator::new();
    aggregator.merge_all(collector.history());

    let report = MetricsReport::from_aggregator(&aggregator);

    println!(
        "Runs: {}  |  Nodes executed: {}  |  Errors: {}",
        report.total_runs, report.total_nodes_executed, report.total_errors
    );
    println!(
        "Total duration: {:?}  |  Avg run: {:?}",
        report.total_duration, report.avg_run_duration
    );
    println!("Error rate: {:.1}%", report.error_rate * 100.0);

    if let Some(ref node) = report.bottleneck_node {
        println!(
            "Bottleneck: {} ({:?})",
            node,
            report.bottleneck_duration.unwrap_or(Duration::ZERO)
        );
    }

    println!("\nNode summaries:");
    for ns in &report.node_summaries {
        println!(
            "  {:<10} execs={}, avg={:?}, p95={:?}, err={:.0}%",
            ns.name,
            ns.execution_count,
            ns.avg_duration,
            ns.p95,
            ns.error_rate * 100.0
        );
    }

    // -- Export to JSON and ask LLM for optimization advice --------------------
    let report_json = MetricsExporter::report_to_json(&report)?;

    let model = shared::get_chat_model(vec![
        "The 'process' node is the clear bottleneck — it has the highest total duration and a \
         retry loop (process->retry->process). Recommendations: (1) optimize the process step, \
         (2) add input validation before processing to reduce errors, (3) cache results to \
         avoid redundant work on retries."
            .to_string(),
    ]);

    let prompt = format!(
        "Analyze this graph execution metrics report and suggest optimizations:\n\n{}",
        report_json
    );
    let response = model
        .invoke_messages(&[Message::human(prompt)], None)
        .await?;

    println!("\nLLM Analysis:\n{}", response.base.content.text());

    println!("\n=== Done ===");
    Ok(())
}
