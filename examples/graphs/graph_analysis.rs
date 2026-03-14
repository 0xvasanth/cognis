//! Graph Analysis Example
//!
//! Demonstrates graph analysis utilities: topology, path finding, metrics,
//! cycle detection, subgraph extraction, and visualization.
//!
//! Run with: `cargo run -p cognis-examples --example graph_analysis`

#[path = "../shared.rs"]
mod shared;
use cognisgraph::graph::analysis::{
    CycleDetector, GraphMetrics, GraphTopology, GraphVisualizer, PathFinder, SubgraphExtractor,
};

fn main() {
    // Build a data processing pipeline graph
    let mut topo = GraphTopology::new();
    topo.add_edge("ingest", "validate");
    topo.add_edge("validate", "transform");
    topo.add_edge("transform", "enrich");
    topo.add_edge("enrich", "store");
    topo.add_edge("transform", "filter");
    topo.add_edge("filter", "store");

    println!("Nodes: {:?}", topo.node_names());
    println!("Edges: {:?}", topo.edge_list());
    println!("Is DAG: {}", topo.is_dag());

    if let Ok(order) = topo.topological_sort() {
        println!("Topological order: {:?}", order);
    }

    // Path finding
    let finder = PathFinder::new(&topo);
    if let Some(path) = finder.shortest_path("ingest", "store") {
        println!("Shortest path (ingest->store): {:?}", path);
    }
    let all = finder.all_paths("ingest", "store");
    println!("All paths (ingest->store): {}", all.len());
    for (i, path) in all.iter().enumerate() {
        println!("  path {}: {:?}", i + 1, path);
    }

    // Graph metrics
    let metrics = GraphMetrics::from_topology(&topo);
    println!(
        "Metrics: nodes={}, edges={}, density={:.4}, max_depth={}",
        metrics.node_count, metrics.edge_count, metrics.density, metrics.max_depth
    );

    // Cycle detection
    println!("Cycles in pipeline: {}", CycleDetector::detect(&topo).len());

    let mut cyclic = GraphTopology::new();
    cyclic.add_edge("draft", "review");
    cyclic.add_edge("review", "approve");
    cyclic.add_edge("approve", "publish");
    cyclic.add_edge("publish", "review");
    let cycles = CycleDetector::detect(&cyclic);
    println!("Cycles in feedback graph: {}", cycles.len());
    for (i, cycle) in cycles.iter().enumerate() {
        println!("  cycle {}: {:?}", i + 1, cycle);
    }

    // Subgraph extraction
    let downstream = SubgraphExtractor::downstream(&topo, "transform");
    println!("Downstream from 'transform': {:?}", downstream.node_names());

    // Visualization
    println!("Mermaid:\n{}", GraphVisualizer::to_mermaid(&topo));
    println!("DOT:\n{}", GraphVisualizer::to_dot(&topo));

    // LLM demo
    let model = shared::get_chat_model(vec![
        "This DAG has 6 nodes representing a data pipeline that branches at 'transform' into parallel paths before converging at 'store'.".into(),
    ]);
    let mermaid = GraphVisualizer::to_mermaid(&topo);
    let messages = vec![cognis_core::messages::Message::human(&format!(
        "Analyze this pipeline graph in 2-3 sentences:\n{}",
        mermaid
    ))];
    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(async { model.invoke_messages(&messages, None).await }) {
        Ok(response) => println!("LLM: {}", response.base.content.text()),
        Err(e) => println!("LLM error: {}", e),
    }
}
