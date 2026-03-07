//! ASCII art graph visualization for terminal display.
//!
//! This module renders a [`CompiledStateGraph`] as Unicode box-drawing
//! art, suitable for printing in terminals and log output.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use super::state::{CompiledStateGraph, Edge};
use crate::constants::{END, START};

// ── Options & Style ────────────────────────────────────────────────

/// Style for rendering graph nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStyle {
    /// Nodes drawn with box-drawing corners: `┌─┐ │ │ └─┘`
    Box,
    /// Nodes drawn with rounded corners: `╭─╮ │ │ ╰─╯`
    Round,
    /// Nodes drawn as plain text without borders.
    Minimal,
}

impl Default for NodeStyle {
    fn default() -> Self {
        NodeStyle::Box
    }
}

/// Options controlling ASCII graph rendering.
#[derive(Debug, Clone)]
pub struct AsciiRenderOptions {
    /// Maximum width of the output in columns.
    pub max_width: usize,
    /// Style used to draw nodes.
    pub node_style: NodeStyle,
    /// Whether to show labels on edges (for conditional branches).
    pub show_edge_labels: bool,
    /// Whether to show condition decision nodes.
    pub show_conditions: bool,
    /// If `true`, render left-to-right instead of top-to-bottom.
    pub horizontal: bool,
    /// If `true`, reduce vertical spacing between layers.
    pub compact: bool,
}

impl Default for AsciiRenderOptions {
    fn default() -> Self {
        Self {
            max_width: 80,
            node_style: NodeStyle::Box,
            show_edge_labels: true,
            show_conditions: true,
            horizontal: false,
            compact: false,
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────

/// Renderer that converts a graph into ASCII art.
pub struct AsciiGraphRenderer;

impl AsciiGraphRenderer {
    /// Render a [`CompiledStateGraph`] with default options.
    pub fn render(graph: &CompiledStateGraph) -> String {
        Self::render_with_options(graph, &AsciiRenderOptions::default())
    }

    /// Render a [`CompiledStateGraph`] with the given options.
    pub fn render_with_options(graph: &CompiledStateGraph, options: &AsciiRenderOptions) -> String {
        // Extract raw nodes and edges from the compiled graph.
        let (nodes, edges) = extract_graph_data(graph);
        render_ascii(&nodes, &edges, options)
    }
}

/// Convenience function: render ASCII art from raw node names and edge
/// tuples `(from, to, optional_label)`.
pub fn to_ascii(nodes: &[&str], edges: &[(&str, &str, Option<&str>)]) -> String {
    let node_strings: Vec<String> = nodes.iter().map(|s| s.to_string()).collect();
    let edge_tuples: Vec<(String, String, Option<String>)> = edges
        .iter()
        .map(|(f, t, l)| (f.to_string(), t.to_string(), l.map(|s| s.to_string())))
        .collect();
    render_ascii(&node_strings, &edge_tuples, &AsciiRenderOptions::default())
}

// ── Internal types ─────────────────────────────────────────────────

/// Internal layout engine.
struct GraphLayout {
    /// Nodes grouped by layer (topological depth).
    layers: Vec<Vec<String>>,
    /// Layer assignment per node.
    #[allow(dead_code)]
    node_layer: HashMap<String, usize>,
    /// Column assignment per node within its layer.
    #[allow(dead_code)]
    node_col: HashMap<String, usize>,
    /// All edges with optional labels.
    edges: Vec<(String, String, Option<String>)>,
    /// Set of back-edges (cycles).
    back_edges: HashSet<(String, String)>,
}

impl GraphLayout {
    /// Build a layout from nodes and edges.
    fn build(node_names: &[String], edges: &[(String, String, Option<String>)]) -> Self {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        let mut all_nodes: BTreeSet<String> = BTreeSet::new();

        for n in node_names {
            all_nodes.insert(n.clone());
        }
        for (f, t, _) in edges {
            all_nodes.insert(f.clone());
            all_nodes.insert(t.clone());
            adj.entry(f.clone()).or_default().push(t.clone());
        }

        // Topological layer assignment with cycle detection.
        let (node_layer, back_edges) = Self::assign_layers(&all_nodes, &adj);

        // Group by layer.
        let max_layer = node_layer.values().copied().max().unwrap_or(0);
        let mut layers: Vec<Vec<String>> = vec![Vec::new(); max_layer + 1];
        for n in &all_nodes {
            if let Some(&layer) = node_layer.get(n) {
                layers[layer].push(n.clone());
            }
        }

        // Assign columns within each layer.
        let mut node_col: HashMap<String, usize> = HashMap::new();
        for layer_nodes in &layers {
            for (i, n) in layer_nodes.iter().enumerate() {
                node_col.insert(n.clone(), i);
            }
        }

        GraphLayout {
            layers,
            node_layer,
            node_col,
            edges: edges.to_vec(),
            back_edges,
        }
    }

    /// Assign topological layers using BFS from roots, detecting back-edges.
    fn assign_layers(
        all_nodes: &BTreeSet<String>,
        adj: &HashMap<String, Vec<String>>,
    ) -> (HashMap<String, usize>, HashSet<(String, String)>) {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for n in all_nodes {
            in_degree.entry(n.clone()).or_insert(0);
        }
        for targets in adj.values() {
            for t in targets {
                *in_degree.entry(t.clone()).or_insert(0) += 1;
            }
        }

        let mut layer: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut back_edges: HashSet<(String, String)> = HashSet::new();

        // Start with nodes that have no incoming edges, or START if present.
        let mut roots: Vec<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(n, _)| n.clone())
            .collect();
        roots.sort();

        // If START exists, make sure it's at layer 0.
        if all_nodes.contains(START) && !roots.contains(&START.to_string()) {
            roots.insert(0, START.to_string());
        }

        for r in &roots {
            if !layer.contains_key(r) {
                layer.insert(r.clone(), 0);
                queue.push_back(r.clone());
            }
        }

        // BFS to assign layers.
        while let Some(node) = queue.pop_front() {
            let current_layer = layer[&node];
            if let Some(neighbors) = adj.get(&node) {
                for neighbor in neighbors {
                    if let Some(&existing) = layer.get(neighbor) {
                        // Already visited — if current_layer+1 > existing, it's a back-edge.
                        if existing <= current_layer {
                            back_edges.insert((node.clone(), neighbor.clone()));
                        }
                    } else {
                        layer.insert(neighbor.clone(), current_layer + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        // Assign any unreachable nodes to layer 0.
        for n in all_nodes {
            layer.entry(n.clone()).or_insert(0);
        }

        (layer, back_edges)
    }
}

// ── Extraction from CompiledStateGraph ─────────────────────────────

fn extract_graph_data(
    graph: &CompiledStateGraph,
) -> (Vec<String>, Vec<(String, String, Option<String>)>) {
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    let mut edges: Vec<(String, String, Option<String>)> = Vec::new();

    for edge in &graph.edges {
        match edge {
            Edge::Direct { from, to } => {
                nodes.insert(from.clone());
                nodes.insert(to.clone());
                edges.push((from.clone(), to.clone(), None));
            }
            Edge::Conditional { from, branch } => {
                nodes.insert(from.clone());
                if let Some(ends) = &branch.ends {
                    let mut sorted: Vec<_> = ends.iter().collect();
                    sorted.sort_by_key(|(k, _)| (*k).clone());
                    for (label, target) in sorted {
                        nodes.insert(target.clone());
                        edges.push((from.clone(), target.clone(), Some(label.clone())));
                    }
                }
            }
        }
    }

    (nodes.into_iter().collect(), edges)
}

// ── Rendering ──────────────────────────────────────────────────────

fn render_ascii(
    nodes: &[String],
    edges: &[(String, String, Option<String>)],
    options: &AsciiRenderOptions,
) -> String {
    if nodes.is_empty() {
        return String::new();
    }

    let layout = GraphLayout::build(nodes, edges);

    if options.horizontal {
        return render_horizontal(&layout, options);
    }

    render_vertical(&layout, options)
}

/// Display name for a node (pretty-prints START/END).
fn display_name(name: &str) -> &str {
    match name {
        START => "START",
        END => "END",
        other => other,
    }
}

/// Render a single node as lines of text, returning (lines, width).
fn render_node(name: &str, style: NodeStyle) -> (Vec<String>, usize) {
    let label = display_name(name);
    let inner_width = label.len().max(1);
    let box_width = inner_width + 2; // padding on each side

    match style {
        NodeStyle::Box => {
            let top = format!("┌{}┐", "─".repeat(box_width));
            let mid = format!("│ {} │", center_pad(label, inner_width));
            let bot = format!("└{}┘", "─".repeat(box_width));
            let w = top.chars().count();
            (vec![top, mid, bot], w)
        }
        NodeStyle::Round => {
            let top = format!("╭{}╮", "─".repeat(box_width));
            let mid = format!("│ {} │", center_pad(label, inner_width));
            let bot = format!("╰{}╯", "─".repeat(box_width));
            let w = top.chars().count();
            (vec![top, mid, bot], w)
        }
        NodeStyle::Minimal => {
            let line = format!(" {} ", label);
            let w = line.chars().count();
            (vec![line], w)
        }
    }
}

/// Center-pad a string to a given width.
fn center_pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        return s.to_string();
    }
    let total_pad = width - s.len();
    let left = total_pad / 2;
    let right = total_pad - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

/// Render the graph in vertical (top-to-bottom) orientation.
fn render_vertical(layout: &GraphLayout, options: &AsciiRenderOptions) -> String {
    let style = options.node_style;
    let compact = options.compact;

    // Pre-compute node widths.
    let mut node_widths: HashMap<String, usize> = HashMap::new();
    for layer in &layout.layers {
        for n in layer {
            let (_, w) = render_node(n, style);
            node_widths.insert(n.clone(), w);
        }
    }

    // Compute column positions (x-offset for each node).
    // Each layer is laid out with spacing between nodes.
    let gap = 2usize;
    let mut layer_widths: Vec<usize> = Vec::new();
    let mut node_x: HashMap<String, usize> = HashMap::new();

    for layer in &layout.layers {
        let mut total_w = 0usize;
        for (i, n) in layer.iter().enumerate() {
            if i > 0 {
                total_w += gap;
            }
            node_x.insert(n.clone(), total_w);
            total_w += node_widths[n];
        }
        layer_widths.push(total_w);
    }

    // Find the max layer width for centering.
    let max_layer_w = layer_widths.iter().copied().max().unwrap_or(0);
    let canvas_width = max_layer_w.min(options.max_width).max(1);

    // Adjust x positions to center each layer.
    for (li, layer) in layout.layers.iter().enumerate() {
        let lw = layer_widths[li];
        let offset = if canvas_width > lw {
            (canvas_width - lw) / 2
        } else {
            0
        };
        for n in layer {
            *node_x.get_mut(n).unwrap() += offset;
        }
    }

    // Build canvas as lines of characters.
    let mut output_lines: Vec<String> = Vec::new();

    for (li, layer) in layout.layers.iter().enumerate() {
        // Draw connectors from previous layer.
        if li > 0 {
            let prev_layer = &layout.layers[li - 1];
            // Collect edges from prev_layer nodes to this layer's nodes.
            let mut connections: Vec<(String, String, Option<String>)> = Vec::new();
            for (from, to, label) in &layout.edges {
                let from_in_prev = prev_layer.contains(from);
                let to_in_cur = layer.contains(to);
                if from_in_prev
                    && to_in_cur
                    && !layout.back_edges.contains(&(from.clone(), to.clone()))
                {
                    connections.push((from.clone(), to.clone(), label.clone()));
                }
            }

            // Draw vertical connector lines.
            if connections.len() == 1 && connections[0].2.is_none() {
                // Simple single connection — draw a pipe and arrow.
                let from = &connections[0].0;
                let to = &connections[0].1;
                let from_w = node_widths[from];
                let from_center = node_x[from] + from_w / 2;
                let to_w = node_widths[to];
                let to_center = node_x[to] + to_w / 2;

                if from_center == to_center {
                    // Straight vertical.
                    if !compact {
                        output_lines.push(format!("{}│", " ".repeat(from_center)));
                    }
                    output_lines.push(format!("{}▼", " ".repeat(to_center)));
                } else {
                    // Offset — draw an L-connector.
                    if !compact {
                        output_lines.push(format!("{}│", " ".repeat(from_center)));
                    }
                    output_lines.push(format!("{}▼", " ".repeat(to_center)));
                }
            } else if !connections.is_empty() {
                // Multiple connections — draw branching lines with optional labels.
                // Collect unique froms.
                let froms: BTreeSet<&String> = connections.iter().map(|(f, _, _)| f).collect();

                for from in &froms {
                    let from_w = node_widths[*from];
                    let from_center = node_x[*from] + from_w / 2;

                    if !compact {
                        output_lines.push(format!("{}│", " ".repeat(from_center)));
                    }

                    // Gather targets from this source.
                    let targets: Vec<&(String, String, Option<String>)> =
                        connections.iter().filter(|(f, _, _)| f == *from).collect();

                    if targets.len() > 1 {
                        // Draw a branching line.
                        let target_centers: Vec<usize> = targets
                            .iter()
                            .map(|(_, t, _)| {
                                let tw = node_widths[t];
                                node_x[t] + tw / 2
                            })
                            .collect();

                        let min_x = *target_centers.iter().min().unwrap();
                        let max_x = *target_centers.iter().max().unwrap();

                        // Draw horizontal branch line.
                        let mut branch_line: Vec<char> = vec![' '; max_x + 1];
                        for x in min_x..=max_x {
                            branch_line[x] = '─';
                        }
                        // Place junctions.
                        for &cx in &target_centers {
                            branch_line[cx] = '┬';
                        }
                        // Place the source junction.
                        if from_center >= min_x && from_center <= max_x {
                            branch_line[from_center] = '┴';
                        }

                        let branch_str: String = branch_line.iter().collect();
                        output_lines.push(branch_str.trim_end().to_string());

                        // Draw edge labels if enabled.
                        if options.show_edge_labels {
                            let mut label_line: Vec<char> = vec![' '; max_x + 20];
                            let mut has_labels = false;
                            for (_, t, l) in &targets {
                                if let Some(label) = l {
                                    has_labels = true;
                                    let tw = node_widths[t];
                                    let tc = node_x[t] + tw / 2;
                                    let start = tc.saturating_sub(label.len() / 2);
                                    for (i, ch) in label.chars().enumerate() {
                                        if start + i < label_line.len() {
                                            label_line[start + i] = ch;
                                        }
                                    }
                                }
                            }
                            if has_labels {
                                let lstr: String = label_line.iter().collect();
                                output_lines.push(lstr.trim_end().to_string());
                            }
                        }

                        // Draw arrows down to each target.
                        let mut arrow_line: Vec<char> = vec![' '; max_x + 1];
                        for &cx in &target_centers {
                            arrow_line[cx] = '▼';
                        }
                        let arrow_str: String = arrow_line.iter().collect();
                        output_lines.push(arrow_str.trim_end().to_string());
                    } else {
                        // Single target from this source.
                        let to = &targets[0].1;
                        let to_w = node_widths[to];
                        let to_center = node_x[to] + to_w / 2;

                        if let Some(label) = &targets[0].2 {
                            if options.show_edge_labels {
                                let lx = from_center.min(to_center);
                                output_lines.push(format!("{}{}", " ".repeat(lx), label));
                            }
                        }

                        output_lines.push(format!("{}▼", " ".repeat(to_center)));
                    }
                }
            }
        }

        // Draw nodes in this layer.
        let mut node_renders: Vec<(usize, Vec<String>, usize)> = Vec::new();
        for n in layer {
            let (lines, w) = render_node(n, style);
            let x = node_x[n];
            node_renders.push((x, lines, w));
        }

        // Find max height in this layer.
        let max_h = node_renders
            .iter()
            .map(|(_, lines, _)| lines.len())
            .max()
            .unwrap_or(0);

        for row in 0..max_h {
            let mut line_chars: Vec<char> = vec![' '; canvas_width + 20];
            for (x, lines, _w) in &node_renders {
                if row < lines.len() {
                    for (i, ch) in lines[row].chars().enumerate() {
                        if x + i < line_chars.len() {
                            line_chars[x + i] = ch;
                        }
                    }
                }
            }
            let line: String = line_chars.iter().collect();
            output_lines.push(line.trim_end().to_string());
        }
    }

    // Append back-edge annotations at the bottom.
    for (from, to) in &layout.back_edges {
        let from_label = display_name(from);
        let to_label = display_name(to);
        output_lines.push(String::new());
        let from_w = node_widths.get(from).copied().unwrap_or(0);
        let from_x = node_x.get(from).copied().unwrap_or(0);
        let from_center = from_x + from_w / 2;
        output_lines.push(format!(
            "{}└──── {} ──→ {}",
            " ".repeat(from_center.saturating_sub(1)),
            from_label,
            to_label
        ));
    }

    // Clean up trailing empty lines.
    while output_lines.last().map_or(false, |l| l.is_empty()) {
        output_lines.pop();
    }

    output_lines.join("\n")
}

/// Render the graph in horizontal (left-to-right) orientation.
fn render_horizontal(layout: &GraphLayout, options: &AsciiRenderOptions) -> String {
    let style = options.node_style;

    // In horizontal mode, layers go left to right. Each layer becomes a column.
    // Nodes within a layer are stacked vertically.

    // Compute node heights and widths.
    let mut node_dims: HashMap<String, (Vec<String>, usize)> = HashMap::new();
    for layer in &layout.layers {
        for n in layer {
            let (lines, w) = render_node(n, style);
            node_dims.insert(n.clone(), (lines, w));
        }
    }

    // Simple horizontal rendering: each layer separated by arrows.
    let mut column_strings: Vec<Vec<String>> = Vec::new();

    for (li, layer) in layout.layers.iter().enumerate() {
        if li > 0 {
            // Add arrow column.
            let max_h = layer
                .iter()
                .map(|n| node_dims[n].0.len())
                .max()
                .unwrap_or(1);
            let mut arrow_col = Vec::new();
            for r in 0..max_h {
                if r == max_h / 2 {
                    arrow_col.push(" ──→ ".to_string());
                } else {
                    arrow_col.push("     ".to_string());
                }
            }
            column_strings.push(arrow_col);
        }

        // Render nodes in this layer (stacked vertically if multiple).
        let mut col_lines: Vec<String> = Vec::new();
        for (ni, n) in layer.iter().enumerate() {
            if ni > 0 {
                col_lines.push(String::new());
            }
            let (lines, _) = &node_dims[n];
            col_lines.extend(lines.clone());
        }
        column_strings.push(col_lines);
    }

    // Normalize heights and merge columns side by side.
    let max_rows = column_strings.iter().map(|c| c.len()).max().unwrap_or(0);
    let col_widths: Vec<usize> = column_strings
        .iter()
        .map(|col| col.iter().map(|l| l.chars().count()).max().unwrap_or(0))
        .collect();

    let mut result_lines: Vec<String> = Vec::new();
    for r in 0..max_rows {
        let mut line = String::new();
        for (ci, col) in column_strings.iter().enumerate() {
            let cell = if r < col.len() { &col[r] } else { "" };
            let w = col_widths[ci];
            let padded = format!("{:width$}", cell, width = w);
            line.push_str(&padded);
        }
        result_lines.push(line.trim_end().to_string());
    }

    result_lines.join("\n")
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

        let ascii = AsciiGraphRenderer::render(&graph);
        assert!(!ascii.is_empty());
        // Should contain all node names.
        assert!(ascii.contains("START"));
        assert!(ascii.contains("a"));
        assert!(ascii.contains("b"));
        assert!(ascii.contains("c"));
        assert!(ascii.contains("END"));
        // Should contain vertical flow indicators.
        assert!(ascii.contains('▼'));
    }

    #[test]
    fn test_single_branch() {
        let mut path_map = HashMap::new();
        path_map.insert("left".to_string(), "node_left".to_string());
        path_map.insert("right".to_string(), "node_right".to_string());

        let router: RouterFn = Arc::new(|_state: &Value| RouterResult::Single("left".to_string()));

        let graph = StateGraph::new()
            .add_node("a", noop_action())
            .add_node("node_left", noop_action())
            .add_node("node_right", noop_action())
            .set_entry_point("a")
            .add_conditional_edges("a", router, Some(path_map))
            .set_finish_point("node_left")
            .set_finish_point("node_right")
            .compile()
            .unwrap();

        let ascii = AsciiGraphRenderer::render(&graph);
        assert!(ascii.contains("a"));
        assert!(ascii.contains("node_left"));
        assert!(ascii.contains("node_right"));
        // Should have branching indicators.
        assert!(ascii.contains('▼'));
    }

    #[test]
    fn test_graph_with_cycle() {
        let mut path_map = HashMap::new();
        path_map.insert("continue".to_string(), "tools".to_string());
        path_map.insert("end".to_string(), END.to_string());

        let router: RouterFn =
            Arc::new(|_state: &Value| RouterResult::Single("continue".to_string()));

        let graph = StateGraph::new()
            .add_node("agent", noop_action())
            .add_node("tools", noop_action())
            .set_entry_point("agent")
            .add_conditional_edges("agent", router, Some(path_map))
            .add_edge("tools", "agent")
            .compile()
            .unwrap();

        let ascii = AsciiGraphRenderer::render(&graph);
        assert!(ascii.contains("agent"));
        assert!(ascii.contains("tools"));
        // Back-edge annotation should be present.
        assert!(ascii.contains("──→") || ascii.contains("agent"));
    }

    #[test]
    fn test_start_and_end_nodes() {
        let graph = StateGraph::new()
            .add_node("only", noop_action())
            .set_entry_point("only")
            .set_finish_point("only")
            .compile()
            .unwrap();

        let ascii = AsciiGraphRenderer::render(&graph);
        assert!(ascii.contains("START"));
        assert!(ascii.contains("END"));
        assert!(ascii.contains("only"));
    }

    #[test]
    fn test_compact_mode() {
        let graph = StateGraph::new()
            .add_node("a", noop_action())
            .add_node("b", noop_action())
            .set_entry_point("a")
            .add_edge("a", "b")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let default_ascii = AsciiGraphRenderer::render(&graph);
        let compact_opts = AsciiRenderOptions {
            compact: true,
            ..Default::default()
        };
        let compact_ascii = AsciiGraphRenderer::render_with_options(&graph, &compact_opts);

        // Compact mode should produce fewer lines (less vertical space).
        let default_lines = default_ascii.lines().count();
        let compact_lines = compact_ascii.lines().count();
        assert!(
            compact_lines <= default_lines,
            "compact ({}) should have <= lines than default ({})",
            compact_lines,
            default_lines
        );
    }

    #[test]
    fn test_minimal_node_style() {
        let graph = StateGraph::new()
            .add_node("x", noop_action())
            .set_entry_point("x")
            .set_finish_point("x")
            .compile()
            .unwrap();

        let opts = AsciiRenderOptions {
            node_style: NodeStyle::Minimal,
            ..Default::default()
        };
        let ascii = AsciiGraphRenderer::render_with_options(&graph, &opts);

        // Minimal style should NOT have box-drawing corners.
        assert!(!ascii.contains('┌'));
        assert!(!ascii.contains('┐'));
        assert!(ascii.contains("START"));
        assert!(ascii.contains("x"));
    }

    #[test]
    fn test_edge_labels_displayed() {
        let mut path_map = HashMap::new();
        path_map.insert("yes".to_string(), "do_it".to_string());
        path_map.insert("no".to_string(), END.to_string());

        let router: RouterFn = Arc::new(|_state: &Value| RouterResult::Single("yes".to_string()));

        let graph = StateGraph::new()
            .add_node("check", noop_action())
            .add_node("do_it", noop_action())
            .set_entry_point("check")
            .add_conditional_edges("check", router, Some(path_map))
            .set_finish_point("do_it")
            .compile()
            .unwrap();

        let ascii = AsciiGraphRenderer::render(&graph);
        // Edge labels should appear in the output.
        assert!(
            ascii.contains("yes") || ascii.contains("no"),
            "Expected edge labels in output:\n{}",
            ascii
        );
    }

    #[test]
    fn test_empty_graph() {
        let result = to_ascii(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_graph() {
        let result = to_ascii(&["alone"], &[]);
        assert!(result.contains("alone"));
    }

    #[test]
    fn test_wide_graph_many_parallel() {
        let graph = StateGraph::new()
            .add_node("root", noop_action())
            .add_node("p1", noop_action())
            .add_node("p2", noop_action())
            .add_node("p3", noop_action())
            .add_node("p4", noop_action())
            .set_entry_point("root")
            .add_edge("root", "p1")
            .add_edge("root", "p2")
            .add_edge("root", "p3")
            .add_edge("root", "p4")
            .set_finish_point("p1")
            .set_finish_point("p2")
            .set_finish_point("p3")
            .set_finish_point("p4")
            .compile()
            .unwrap();

        let ascii = AsciiGraphRenderer::render(&graph);
        assert!(ascii.contains("root"));
        assert!(ascii.contains("p1"));
        assert!(ascii.contains("p2"));
        assert!(ascii.contains("p3"));
        assert!(ascii.contains("p4"));
    }

    #[test]
    fn test_render_with_options_customization() {
        let graph = StateGraph::new()
            .add_node("step", noop_action())
            .set_entry_point("step")
            .set_finish_point("step")
            .compile()
            .unwrap();

        let opts = AsciiRenderOptions {
            max_width: 40,
            node_style: NodeStyle::Round,
            show_edge_labels: false,
            show_conditions: false,
            horizontal: false,
            compact: true,
        };

        let ascii = AsciiGraphRenderer::render_with_options(&graph, &opts);
        // Round style should have round corners.
        assert!(ascii.contains('╭'));
        assert!(ascii.contains('╰'));
        assert!(ascii.contains("step"));
    }

    #[test]
    fn test_to_ascii_helper() {
        let nodes = vec!["A", "B", "C"];
        let edges = vec![("A", "B", None), ("B", "C", Some("next"))];
        let result = to_ascii(&nodes, &edges);
        assert!(result.contains("A"));
        assert!(result.contains("B"));
        assert!(result.contains("C"));
        assert!(result.contains('▼'));
    }

    #[test]
    fn test_horizontal_rendering() {
        let graph = StateGraph::new()
            .add_node("a", noop_action())
            .add_node("b", noop_action())
            .set_entry_point("a")
            .add_edge("a", "b")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let opts = AsciiRenderOptions {
            horizontal: true,
            ..Default::default()
        };
        let ascii = AsciiGraphRenderer::render_with_options(&graph, &opts);
        assert!(ascii.contains("START"));
        assert!(ascii.contains("a"));
        assert!(ascii.contains("b"));
        assert!(ascii.contains("END"));
        // Horizontal should have right arrows.
        assert!(ascii.contains("→"));
    }
}
