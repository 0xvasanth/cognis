//! Graph Metrics Example
//!
//! Demonstrates the graph execution metrics system from `cognisgraph::graph::metrics`:
//! - Recording per-node execution timings with `NodeMetrics`
//! - Tracking edge traversals with `EdgeMetrics`
//! - Collecting metrics via `InMemoryMetricsCollector`
//! - Aggregating across multiple runs with `MetricsAggregator`
//! - Generating a `MetricsReport` with bottleneck detection and percentiles
//! - Exporting metrics to JSON with `MetricsExporter`
//! - Using the chat model to analyze the metrics report
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_metrics`

#[path = "../shared.rs"]
mod shared;

use std::time::Duration;

use cognisgraph::graph::metrics::{
    EdgeMetrics, GraphMetrics, InMemoryMetricsCollector, MetricsAggregator, MetricsCollector,
    MetricsExporter, MetricsReport, NodeMetrics,
};

use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Metrics Example ===\n");

    // -----------------------------------------------------------------------
    // 1. NodeMetrics — per-node execution statistics
    // -----------------------------------------------------------------------
    println!("--- 1. NodeMetrics ---");
    println!("Track execution count, duration, min/max, errors, and percentiles.\n");

    let mut fetch_node = NodeMetrics::new("fetch_data");
    fetch_node.record(Duration::from_millis(120), false);
    fetch_node.record(Duration::from_millis(95), false);
    fetch_node.record(Duration::from_millis(200), false);
    fetch_node.record(Duration::from_millis(150), true); // simulate an error
    fetch_node.record(Duration::from_millis(110), false);

    println!("  Node: {}", fetch_node.node_name);
    println!("  Executions: {}", fetch_node.execution_count);
    println!("  Total duration: {:?}", fetch_node.total_duration);
    println!("  Avg duration: {:?}", fetch_node.avg_duration());
    println!("  Min duration: {:?}", fetch_node.min_duration);
    println!("  Max duration: {:?}", fetch_node.max_duration);
    println!("  p50: {:?}", fetch_node.p50());
    println!("  p95: {:?}", fetch_node.p95());
    println!("  p99: {:?}", fetch_node.p99());
    println!(
        "  Error rate: {:.1}% ({} errors)\n",
        fetch_node.error_rate() * 100.0,
        fetch_node.error_count
    );

    // -----------------------------------------------------------------------
    // 2. EdgeMetrics — edge traversal tracking
    // -----------------------------------------------------------------------
    println!("--- 2. EdgeMetrics ---");
    println!("Count how many times each edge in the graph is traversed.\n");

    let mut edge_fetch_to_process = EdgeMetrics::new("fetch_data", "process");
    for _ in 0..5 {
        edge_fetch_to_process.record_traversal();
    }
    let mut edge_process_to_output = EdgeMetrics::new("process", "output");
    for _ in 0..3 {
        edge_process_to_output.record_traversal();
    }
    let mut edge_process_to_retry = EdgeMetrics::new("process", "retry");
    edge_process_to_retry.record_traversal();

    println!(
        "  {} -> {}: {} traversals",
        edge_fetch_to_process.from, edge_fetch_to_process.to, edge_fetch_to_process.traversal_count
    );
    println!(
        "  {} -> {}: {} traversals",
        edge_process_to_output.from,
        edge_process_to_output.to,
        edge_process_to_output.traversal_count
    );
    println!(
        "  {} -> {}: {} traversals (cold path)\n",
        edge_process_to_retry.from, edge_process_to_retry.to, edge_process_to_retry.traversal_count
    );

    // -----------------------------------------------------------------------
    // 3. InMemoryMetricsCollector — collecting metrics during graph runs
    // -----------------------------------------------------------------------
    println!("--- 3. InMemoryMetricsCollector ---");
    println!("Pluggable sink that records node executions and edge traversals.\n");

    let mut collector = InMemoryMetricsCollector::new();

    // Simulate run 1
    collector.on_run_start("run-001");
    collector.on_node_executed("fetch_data", Duration::from_millis(120), false);
    collector.on_edge_traversed("fetch_data", "process");
    collector.on_node_executed("process", Duration::from_millis(80), false);
    collector.on_edge_traversed("process", "output");
    collector.on_node_executed("output", Duration::from_millis(30), false);
    collector.on_run_finish("run-001");

    // Simulate run 2
    collector.on_run_start("run-002");
    collector.on_node_executed("fetch_data", Duration::from_millis(95), false);
    collector.on_edge_traversed("fetch_data", "process");
    collector.on_node_executed("process", Duration::from_millis(200), true); // error
    collector.on_edge_traversed("process", "retry");
    collector.on_node_executed("retry", Duration::from_millis(50), false);
    collector.on_edge_traversed("retry", "process");
    collector.on_node_executed("process", Duration::from_millis(70), false);
    collector.on_edge_traversed("process", "output");
    collector.on_node_executed("output", Duration::from_millis(25), false);
    collector.on_run_finish("run-002");

    println!("  Completed runs: {}", collector.completed_runs());
    let snapshot = collector.snapshot();
    println!("  Current snapshot run_id: {:?}", snapshot.run_id);
    println!(
        "  Nodes tracked in last run: {}\n",
        snapshot.node_metrics.len()
    );

    // -----------------------------------------------------------------------
    // 4. GraphMetrics — aggregated single-run metrics
    // -----------------------------------------------------------------------
    println!("--- 4. GraphMetrics ---");
    println!("Aggregate stats for a single graph execution.\n");

    let mut gm = GraphMetrics::new("demo-run");
    gm.start();

    // Simulate a pipeline: fetch -> transform -> validate -> output
    let nodes_data = vec![
        ("fetch", 130, false),
        ("transform", 250, false),
        ("validate", 40, false),
        ("transform", 180, false), // re-executed
        ("validate", 35, true),    // validation error
        ("output", 20, false),
    ];
    for (name, ms, is_err) in &nodes_data {
        gm.record_node(name, Duration::from_millis(*ms), *is_err);
    }
    gm.record_edge("fetch", "transform");
    gm.record_edge("transform", "validate");
    gm.record_edge("validate", "transform"); // retry edge
    gm.record_edge("transform", "validate");
    gm.record_edge("validate", "output");
    gm.finish();

    println!("  Total nodes executed: {}", gm.total_nodes_executed);
    println!("  Total errors: {}", gm.total_errors);
    println!("  Total duration: {:?}", gm.total_duration);

    if let Some(bottleneck) = gm.bottleneck() {
        println!(
            "  Bottleneck: {} ({:?} total)",
            bottleneck.node_name, bottleneck.total_duration
        );
    }
    if let Some(hot) = gm.hot_edge() {
        println!(
            "  Hot edge: {} -> {} ({} traversals)",
            hot.from, hot.to, hot.traversal_count
        );
    }
    if let Some(cold) = gm.cold_edge() {
        println!(
            "  Cold edge: {} -> {} ({} traversals)",
            cold.from, cold.to, cold.traversal_count
        );
    }

    println!("\n  Nodes by duration (descending):");
    for nm in gm.nodes_by_duration() {
        println!(
            "    {} — {:?} total, {} executions, {:.0}% error rate",
            nm.node_name,
            nm.total_duration,
            nm.execution_count,
            nm.error_rate() * 100.0
        );
    }

    println!("\n  Edges by traversal count:");
    for em in gm.edges_by_traversal() {
        println!(
            "    {} -> {} — {} traversals",
            em.from, em.to, em.traversal_count
        );
    }

    // -----------------------------------------------------------------------
    // 5. ExecutionProfile — timeline of node executions
    // -----------------------------------------------------------------------
    println!("\n--- 5. ExecutionProfile ---");
    println!("Timeline of node executions within a single graph run.\n");

    let profile = &gm.profile;
    println!("  Execution order: {:?}", profile.execution_order());
    println!("  Total profile duration: {:?}", profile.total_duration());
    println!("  Entries:");
    for entry in &profile.entries {
        println!(
            "    [+{:?}] {} — {:?}{}",
            entry.offset,
            entry.node_name,
            entry.duration,
            if entry.is_error { " (ERROR)" } else { "" }
        );
    }

    // -----------------------------------------------------------------------
    // 6. MetricsAggregator — combine metrics from multiple runs
    // -----------------------------------------------------------------------
    println!("\n--- 6. MetricsAggregator ---");
    println!("Merge metrics from multiple graph runs into a single summary.\n");

    let mut aggregator = MetricsAggregator::new();
    aggregator.merge_all(collector.history());
    aggregator.merge(&gm);

    println!("  Total runs aggregated: {}", aggregator.total_runs);
    println!(
        "  Total duration across runs: {:?}",
        aggregator.total_duration
    );
    println!(
        "  Total nodes executed: {}",
        aggregator.total_nodes_executed
    );
    println!("  Total errors: {}", aggregator.total_errors);
    println!("  Avg run duration: {:?}", aggregator.avg_run_duration());
    println!("  Error rate: {:.2}%", aggregator.error_rate() * 100.0);

    println!("\n  Top 3 nodes by duration:");
    for nm in aggregator.top_nodes_by_duration(3) {
        println!(
            "    {} — {:?} total, p50={:?}, p95={:?}",
            nm.node_name,
            nm.total_duration,
            nm.p50(),
            nm.p95()
        );
    }

    println!("\n  Top 3 edges by traversal count:");
    for em in aggregator.top_edges_by_traversal(3) {
        println!(
            "    {} -> {} — {} traversals",
            em.from, em.to, em.traversal_count
        );
    }

    // -----------------------------------------------------------------------
    // 7. MetricsReport — bottleneck detection and summary
    // -----------------------------------------------------------------------
    println!("\n--- 7. MetricsReport ---");
    println!("Generate a summary report with bottleneck detection.\n");

    let report = MetricsReport::from_aggregator(&aggregator);

    println!("  Total runs: {}", report.total_runs);
    println!("  Total nodes executed: {}", report.total_nodes_executed);
    println!("  Total errors: {}", report.total_errors);
    println!("  Total duration: {:?}", report.total_duration);
    println!("  Avg run duration: {:?}", report.avg_run_duration);
    println!("  Error rate: {:.2}%", report.error_rate * 100.0);

    if let Some(ref bn) = report.bottleneck_node {
        println!(
            "  Bottleneck node: {} ({:?})",
            bn,
            report.bottleneck_duration.unwrap_or(Duration::ZERO)
        );
    }
    println!("  Hot paths: {:?}", report.hot_path);
    println!("  Cold paths: {:?}", report.cold_path);

    println!("\n  Node summaries:");
    for ns in &report.node_summaries {
        println!(
            "    {} — {} execs, avg={:?}, p50={:?}, p95={:?}, p99={:?}, err={:.0}%",
            ns.name,
            ns.execution_count,
            ns.avg_duration,
            ns.p50,
            ns.p95,
            ns.p99,
            ns.error_rate * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // 8. MetricsExporter — JSON export
    // -----------------------------------------------------------------------
    println!("\n--- 8. MetricsExporter ---");
    println!("Export metrics data structures to JSON.\n");

    let report_json = MetricsExporter::report_to_json(&report)?;
    // Print a truncated preview
    let preview: String = report_json.chars().take(500).collect();
    println!("  Report JSON (first 500 chars):");
    println!("  {}", preview);
    if report_json.len() > 500 {
        println!("  ... ({} total chars)", report_json.len());
    }

    let node_json = MetricsExporter::node_to_json(
        aggregator
            .top_nodes_by_duration(1)
            .first()
            .expect("at least one node"),
    )?;
    println!("\n  Top node as JSON:");
    println!("  {}", node_json);

    let gm_json = MetricsExporter::to_json(&gm)?;
    println!("\n  Full GraphMetrics JSON length: {} chars", gm_json.len());

    // -----------------------------------------------------------------------
    // 9. LLM analysis of the metrics report
    // -----------------------------------------------------------------------
    println!("\n--- 9. LLM Analysis of Metrics ---");
    println!("Ask the chat model to analyze the metrics report.\n");

    let model = shared::get_chat_model(vec![
        "Based on the metrics report, the main bottleneck is the 'transform' node \
         which accounts for the highest total execution time across all runs. \
         The error rate of ~15% in the 'validate' node suggests input quality issues. \
         The hot path from transform->validate indicates a retry loop. \
         Recommendations: 1) Optimize the transform step, 2) Add input validation \
         before the transform, 3) Consider caching transformed results."
            .to_string(),
    ]);

    let prompt = format!(
        "Analyze this graph execution metrics report and provide optimization recommendations:\n\n{}",
        report_json.chars().take(1500).collect::<String>()
    );

    let messages = vec![Message::human(prompt)];
    let response = model.invoke_messages(&messages, None).await?;
    println!("  LLM Analysis:");
    println!("  {}", response.base.content.text());

    println!("\n=== Graph Metrics Example Complete ===");
    Ok(())
}
