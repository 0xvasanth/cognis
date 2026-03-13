//! Graph Analysis Example
//!
//! Demonstrates the graph analysis utilities provided by the `cognisgraph::graph::analysis`
//! module for inspecting, measuring, and visualizing directed graph structures.
//!
//! Features shown:
//! - GraphTopology construction with nodes and edges
//! - In-degree, out-degree, sources, and sinks
//! - DAG detection and topological sort
//! - PathFinder: shortest path, all paths, and reachability
//! - GraphMetrics computation (density, depth, connectivity)
//! - CycleDetector finding all elementary cycles
//! - SubgraphExtractor downstream extraction
//! - GraphVisualizer Mermaid and DOT output
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_analysis`

mod shared;
use cognisgraph::graph::analysis::{
    CycleDetector, GraphMetrics, GraphTopology, GraphVisualizer, PathFinder, SubgraphExtractor,
};

fn main() {
    println!("=== Graph Analysis Example ===\n");

    // -----------------------------------------------------------------------
    // 1. GraphTopology Construction
    // -----------------------------------------------------------------------
    println!("--- 1. GraphTopology Construction ---");
    println!("Build a directed graph representing a data processing pipeline.\n");

    let mut topo = GraphTopology::new();

    // Add nodes explicitly (edges also add nodes implicitly)
    topo.add_node("ingest");

    // Add directed edges to form a pipeline with branching
    //
    //   ingest -> validate -> transform -> enrich -> store
    //                            \-> filter -> store
    topo.add_edge("ingest", "validate");
    topo.add_edge("validate", "transform");
    topo.add_edge("transform", "enrich");
    topo.add_edge("enrich", "store");
    topo.add_edge("transform", "filter");
    topo.add_edge("filter", "store");

    println!("Nodes ({}): {:?}", topo.node_count(), topo.node_names());
    println!("Edges ({}): {:?}", topo.edge_count(), topo.edge_list());

    // -----------------------------------------------------------------------
    // 2. In-Degree, Out-Degree, Sources, and Sinks
    // -----------------------------------------------------------------------
    println!("\n--- 2. Degree Analysis ---");
    println!("Examine the connectivity of each node.\n");

    for node in topo.node_names() {
        println!(
            "  {:<12} in-degree={}, out-degree={}",
            node,
            topo.in_degree(node),
            topo.out_degree(node),
        );
    }

    let sources = topo.sources();
    let sinks = topo.sinks();
    println!("\nSource nodes (no incoming edges): {:?}", sources);
    println!("Sink nodes (no outgoing edges):   {:?}", sinks);

    // -----------------------------------------------------------------------
    // 3. DAG Detection and Topological Sort
    // -----------------------------------------------------------------------
    println!("\n--- 3. DAG Detection and Topological Sort ---");
    println!("Check if the graph is a DAG and compute execution order.\n");

    println!("Is DAG: {}", topo.is_dag());

    match topo.topological_sort() {
        Ok(order) => println!("Topological order: {:?}", order),
        Err(e) => println!("Topological sort failed: {}", e),
    }

    // -----------------------------------------------------------------------
    // 4. PathFinder: Shortest Path, All Paths, Reachability
    // -----------------------------------------------------------------------
    println!("\n--- 4. PathFinder ---");
    println!("Find paths and check reachability between nodes.\n");

    let finder = PathFinder::new(&topo);

    // Shortest path
    match finder.shortest_path("ingest", "store") {
        Some(path) => println!("Shortest path (ingest -> store): {:?}", path),
        None => println!("No path from ingest to store"),
    }

    // All paths
    let all = finder.all_paths("ingest", "store");
    println!("All paths (ingest -> store): {} found", all.len());
    for (i, path) in all.iter().enumerate() {
        println!("  path {}: {:?}", i + 1, path);
    }

    // Reachability
    println!(
        "\nIs 'store' reachable from 'ingest'?  {}",
        finder.is_reachable("ingest", "store")
    );
    println!(
        "Is 'ingest' reachable from 'store'?  {}",
        finder.is_reachable("store", "ingest")
    );
    println!(
        "Is 'filter' reachable from 'validate'? {}",
        finder.is_reachable("validate", "filter")
    );

    // -----------------------------------------------------------------------
    // 5. GraphMetrics
    // -----------------------------------------------------------------------
    println!("\n--- 5. GraphMetrics ---");
    println!("Compute structural metrics for the graph.\n");

    let metrics = GraphMetrics::from_topology(&topo);
    println!("Node count:     {}", metrics.node_count);
    println!("Edge count:     {}", metrics.edge_count);
    println!("Density:        {:.4}", metrics.density);
    println!("Avg in-degree:  {:.4}", metrics.avg_in_degree);
    println!("Avg out-degree: {:.4}", metrics.avg_out_degree);
    println!("Max depth:      {}", metrics.max_depth);
    println!("Is connected:   {}", metrics.is_connected);
    println!("\nMetrics JSON: {}", metrics.to_json());

    // -----------------------------------------------------------------------
    // 6. CycleDetector
    // -----------------------------------------------------------------------
    println!("\n--- 6. CycleDetector ---");
    println!("Detect cycles in graphs.\n");

    // The pipeline graph is a DAG, so no cycles expected
    let cycles = CycleDetector::detect(&topo);
    println!("Cycles in pipeline graph: {} (expected 0)", cycles.len());

    // Build a graph with a cycle: review -> approve -> publish -> review
    let mut cyclic = GraphTopology::new();
    cyclic.add_edge("draft", "review");
    cyclic.add_edge("review", "approve");
    cyclic.add_edge("approve", "publish");
    cyclic.add_edge("publish", "review"); // feedback loop

    let cycles = CycleDetector::detect(&cyclic);
    println!("\nCycles in feedback graph: {}", cycles.len());
    for (i, cycle) in cycles.iter().enumerate() {
        println!("  cycle {}: {:?}", i + 1, cycle);
    }
    println!("Is DAG: {}", cyclic.is_dag());

    // -----------------------------------------------------------------------
    // 7. SubgraphExtractor — Downstream Extraction
    // -----------------------------------------------------------------------
    println!("\n--- 7. SubgraphExtractor ---");
    println!("Extract the downstream subgraph reachable from a given node.\n");

    let downstream = SubgraphExtractor::downstream(&topo, "transform");
    println!(
        "Downstream from 'transform' — nodes: {:?}",
        downstream.node_names()
    );
    println!(
        "Downstream from 'transform' — edges: {:?}",
        downstream.edge_list()
    );

    // Extract a specific subset of nodes
    let subset = SubgraphExtractor::extract(&topo, &["validate", "transform", "enrich"]);
    println!(
        "\nSubgraph (validate, transform, enrich) — nodes: {:?}",
        subset.node_names()
    );
    println!(
        "Subgraph (validate, transform, enrich) — edges: {:?}",
        subset.edge_list()
    );

    // -----------------------------------------------------------------------
    // 8. GraphVisualizer — Mermaid and DOT Output
    // -----------------------------------------------------------------------
    println!("\n--- 8. GraphVisualizer ---");
    println!("Generate visual representations of the graph.\n");

    println!("Mermaid diagram:");
    println!("{}", GraphVisualizer::to_mermaid(&topo));

    println!("\nDOT (Graphviz) output:");
    println!("{}", GraphVisualizer::to_dot(&topo));

    println!("\nASCII representation:");
    println!("{}", GraphVisualizer::to_ascii(&topo));

    println!("\n=== Graph Analysis Example Complete ===");
}
