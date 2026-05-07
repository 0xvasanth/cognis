//! Render compiled graphs as Mermaid or ASCII diagrams for docs / debugging.

pub mod ascii;
pub mod mermaid;

use std::collections::BTreeSet;

use crate::compiled::CompiledGraph;
use crate::state::GraphState;

/// A static-route edge `(from, to)`. Dynamic routing via `Goto::Send` /
/// `Goto::Multiple` only surfaces what the node *can* go to if the graph
/// builder declared it via [`crate::builder::Graph::edge`].
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Edge {
    /// Source node name.
    pub from: String,
    /// Destination node name (or `__END__`).
    pub to: String,
}

/// Extract the static edges declared on the graph (via `edge(from, to)`).
/// Used by both renderers.
pub(crate) fn extract_edges<S: GraphState>(g: &CompiledGraph<S>) -> Vec<Edge> {
    let mut set: BTreeSet<Edge> = BTreeSet::new();
    for (from, to) in &g.graph.edges {
        set.insert(Edge {
            from: from.clone(),
            to: to.clone(),
        });
    }
    set.into_iter().collect()
}
