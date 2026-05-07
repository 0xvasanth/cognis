//! Mermaid diagram export for graph visualization.
//!
//! This module provides functions to export a [`CompiledStateGraph`] as a
//! [Mermaid](https://mermaid.js.org/) flowchart diagram, useful for
//! visualizing the graph topology.

use base64::Engine;
use std::collections::{BTreeSet, HashSet};

use super::state::{CompiledStateGraph, Edge};
use crate::constants::{END, START};

/// Generate a Mermaid flowchart diagram string from a compiled state graph.
///
/// The output is valid Mermaid syntax that can be rendered by any Mermaid-compatible
/// tool (e.g., GitHub Markdown, mermaid.live, documentation sites).
///
/// # Node shapes
/// - `__start__` and `__end__` use stadium-shaped nodes: `([name])`
/// - Regular nodes use rectangular nodes: `[name]`
/// - Conditional branch decision points use diamond nodes: `{name}`
///
/// # Example output
/// ```text
/// graph TD
///     __start__([__start__])
///     node1[node1]
///     __end__([__end__])
///     __start__ --> node1
///     node1 --> __end__
/// ```
pub fn to_mermaid(graph: &CompiledStateGraph) -> String {
    let mut lines = Vec::new();
    lines.push("graph TD".to_string());

    // Collect all node names (sorted for deterministic output).
    let mut all_nodes = BTreeSet::new();
    let mut has_start = false;
    let mut has_end = false;

    for edge in &graph.edges {
        match edge {
            Edge::Direct { from, to } => {
                if from == START {
                    has_start = true;
                } else {
                    all_nodes.insert(from.clone());
                }
                if to == END {
                    has_end = true;
                } else {
                    all_nodes.insert(to.clone());
                }
            }
            Edge::Conditional { from, branch } => {
                if from == START {
                    has_start = true;
                } else {
                    all_nodes.insert(from.clone());
                }
                // Add known targets from the path map.
                if let Some(ends) = &branch.ends {
                    for target in ends.values() {
                        if target == END {
                            has_end = true;
                        } else {
                            all_nodes.insert(target.clone());
                        }
                    }
                }
            }
        }
    }

    // Emit node declarations.
    if has_start {
        lines.push(format!("    {}([{}])", sanitize_id(START), START));
    }
    for node in &all_nodes {
        lines.push(format!("    {}[{}]", sanitize_id(node), node));
    }
    if has_end {
        lines.push(format!("    {}([{}])", sanitize_id(END), END));
    }

    // Track conditional edge counter for unique decision node IDs.
    let mut cond_counter = 0u32;

    // Emit edges.
    for edge in &graph.edges {
        match edge {
            Edge::Direct { from, to } => {
                lines.push(format!("    {} --> {}", sanitize_id(from), sanitize_id(to)));
            }
            Edge::Conditional { from, branch } => {
                if let Some(ends) = &branch.ends {
                    // Create a diamond decision node for the condition.
                    let decision_id = format!("condition_{}", cond_counter);
                    cond_counter += 1;
                    lines.push(format!("    {}{{{{{}}}}}", decision_id, decision_id));
                    lines.push(format!("    {} --> {}", sanitize_id(from), decision_id));
                    // Sort branches for deterministic output.
                    let mut branches: Vec<_> = ends.iter().collect();
                    branches.sort_by_key(|(k, _)| (*k).clone());
                    for (label, target) in branches {
                        lines.push(format!(
                            "    {} -->|\"{}\"| {}",
                            decision_id,
                            label,
                            sanitize_id(target)
                        ));
                    }
                } else {
                    // No path map — we cannot statically determine targets,
                    // so show a generic conditional marker.
                    let decision_id = format!("condition_{}", cond_counter);
                    cond_counter += 1;
                    lines.push(format!("    {}{{{{{}}}}}", decision_id, decision_id));
                    lines.push(format!("    {} --> {}", sanitize_id(from), decision_id));
                    lines.push(format!("    {} -->|\"*\"| ...", decision_id));
                }
            }
        }
    }

    // Emit styles for interrupt nodes.
    let mut interrupt_nodes: Vec<&String> = graph
        .interrupt_before
        .iter()
        .chain(graph.interrupt_after.iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    interrupt_nodes.sort();
    for name in interrupt_nodes {
        lines.push(format!(
            "    style {} fill:#f9f,stroke:#333",
            sanitize_id(name)
        ));
    }

    lines.join("\n")
}

/// Generate a [mermaid.live](https://mermaid.live) URL for the given graph.
///
/// The diagram definition is base64-encoded and embedded in the URL so that
/// opening it renders the graph immediately.
pub fn to_mermaid_url(graph: &CompiledStateGraph) -> String {
    let diagram = to_mermaid(graph);
    let json_payload = serde_json::json!({
        "code": diagram,
        "mermaid": { "theme": "default" }
    });
    let json_str = serde_json::to_string(&json_payload).unwrap_or_default();
    let encoded = base64::engine::general_purpose::STANDARD.encode(json_str.as_bytes());
    format!("https://mermaid.live/edit#base64:{}", encoded)
}

/// Sanitize a node name for use as a Mermaid node ID.
///
/// Mermaid IDs must not contain certain characters. We replace any
/// non-alphanumeric, non-underscore character with `_`.
fn sanitize_id(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::branch::{RouterFn, RouterResult};
    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Helper: a no-op async node action.
    fn noop_action() -> AsyncNodeAction {
        Arc::new(|_state: Value| Box::pin(async move { Ok(json!({})) }))
    }

    #[test]
    fn test_simple_linear_graph() {
        let graph = StateGraph::new()
            .add_node("a", noop_action())
            .add_node("b", noop_action())
            .add_node("c", noop_action())
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.starts_with("graph TD"));
        assert!(mermaid.contains("__start__([__start__])"));
        assert!(mermaid.contains("a[a]"));
        assert!(mermaid.contains("b[b]"));
        assert!(mermaid.contains("c[c]"));
        assert!(mermaid.contains("__end__([__end__])"));
        assert!(mermaid.contains("__start__ --> a"));
        assert!(mermaid.contains("a --> b"));
        assert!(mermaid.contains("b --> c"));
        assert!(mermaid.contains("c --> __end__"));
    }

    #[test]
    fn test_conditional_edges_with_path_map() {
        let mut path_map = HashMap::new();
        path_map.insert("left".to_string(), "node_left".to_string());
        path_map.insert("right".to_string(), "node_right".to_string());

        let router: RouterFn = Arc::new(|_state: &Value| RouterResult::Single("left".to_string()));

        let graph = StateGraph::new()
            .add_node("start_node", noop_action())
            .add_node("node_left", noop_action())
            .add_node("node_right", noop_action())
            .set_entry_point("start_node")
            .add_conditional_edges("start_node", router, Some(path_map))
            .set_finish_point("node_left")
            .set_finish_point("node_right")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.contains("condition_0"));
        assert!(mermaid.contains("start_node --> condition_0"));
        assert!(mermaid.contains("|\"left\"|"));
        assert!(mermaid.contains("|\"right\"|"));
        assert!(mermaid.contains("node_left"));
        assert!(mermaid.contains("node_right"));
    }

    #[test]
    fn test_start_and_end_nodes_have_rounded_shape() {
        let graph = StateGraph::new()
            .add_node("only", noop_action())
            .set_entry_point("only")
            .set_finish_point("only")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        // START and END should use stadium shape ([...])
        assert!(mermaid.contains("__start__([__start__])"));
        assert!(mermaid.contains("__end__([__end__])"));
        // Regular node should use rectangular shape [...]
        assert!(mermaid.contains("only[only]"));
    }

    #[test]
    fn test_multiple_branches_from_same_node() {
        let graph = StateGraph::new()
            .add_node("root", noop_action())
            .add_node("b1", noop_action())
            .add_node("b2", noop_action())
            .add_node("b3", noop_action())
            .set_entry_point("root")
            .add_edge("root", "b1")
            .add_edge("root", "b2")
            .add_edge("root", "b3")
            .set_finish_point("b1")
            .set_finish_point("b2")
            .set_finish_point("b3")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.contains("root --> b1"));
        assert!(mermaid.contains("root --> b2"));
        assert!(mermaid.contains("root --> b3"));
    }

    #[test]
    fn test_valid_mermaid_syntax() {
        let graph = StateGraph::new()
            .add_node("process", noop_action())
            .add_node("check", noop_action())
            .set_entry_point("process")
            .add_edge("process", "check")
            .set_finish_point("check")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        let lines: Vec<&str> = mermaid.lines().collect();

        // First line must be the graph declaration.
        assert_eq!(lines[0], "graph TD");

        // All subsequent lines should be indented.
        for line in &lines[1..] {
            assert!(
                line.starts_with("    "),
                "Line is not properly indented: '{}'",
                line
            );
        }

        // Should contain arrow syntax.
        assert!(mermaid.contains(" --> "));
    }

    #[test]
    fn test_mermaid_url_generation() {
        let graph = StateGraph::new()
            .add_node("x", noop_action())
            .set_entry_point("x")
            .set_finish_point("x")
            .compile()
            .unwrap();

        let url = graph.draw_mermaid_url();
        assert!(url.starts_with("https://mermaid.live/edit#base64:"));

        // Decode and verify.
        let encoded = url
            .strip_prefix("https://mermaid.live/edit#base64:")
            .unwrap();
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let decoded_str = String::from_utf8(decoded_bytes).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&decoded_str).unwrap();
        let code = parsed["code"].as_str().unwrap();
        assert!(code.starts_with("graph TD"));
        assert!(code.contains("__start__"));
    }

    #[test]
    fn test_self_loop_edge() {
        let graph = StateGraph::new()
            .add_node("looper", noop_action())
            .set_entry_point("looper")
            .add_edge("looper", "looper")
            .set_finish_point("looper")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.contains("looper --> looper"));
    }

    #[test]
    fn test_conditional_edge_without_path_map() {
        let router: RouterFn =
            Arc::new(|_state: &Value| RouterResult::Single("target".to_string()));

        let graph = StateGraph::new()
            .add_node("src", noop_action())
            .add_node("target", noop_action())
            .set_entry_point("src")
            .add_conditional_edges("src", router, None)
            .set_finish_point("target")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        // Should have a condition decision node even without a path map.
        assert!(mermaid.contains("condition_0"));
        assert!(mermaid.contains("src --> condition_0"));
    }
}
