//! ASCII renderer — single-pass, line-oriented.

use crate::compiled::CompiledGraph;
use crate::state::GraphState;

use super::extract_edges;

impl<S: GraphState> CompiledGraph<S> {
    /// Render the graph as an ASCII summary suitable for terminal output.
    ///
    /// Layout:
    /// ```text
    /// Graph (start: <name>)
    ///   nodes:
    ///     - a
    ///     - b
    ///     - c
    ///   edges:
    ///     a -> b
    ///     b -> c
    /// ```
    pub fn to_ascii(&self) -> String {
        let mut out = String::new();
        match &self.graph.start {
            Some(s) => out.push_str(&format!("Graph (start: {s})\n")),
            None => out.push_str("Graph (start: <unset>)\n"),
        }

        let mut names: Vec<&String> = self.graph.nodes.keys().collect();
        names.sort();
        out.push_str("  nodes:\n");
        for n in names {
            out.push_str(&format!("    - {n}\n"));
        }

        let edges = extract_edges(self);
        if !edges.is_empty() {
            out.push_str("  edges:\n");
            for e in edges {
                out.push_str(&format!("    {} -> {}\n", e.from, e.to));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Graph;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone)]
    struct S;
    #[derive(Default)]
    struct SU;
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, _: Self::Update) {}
    }

    #[test]
    fn ascii_lists_nodes_and_edges() {
        let g = Graph::<S>::new()
            .node(
                "a",
                node_fn::<S, _, _>("a", |_s, _c| async move {
                    Ok(NodeOut {
                        update: SU,
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("a")
            .compile()
            .unwrap();

        let s = g.to_ascii();
        assert!(s.contains("start: a"));
        assert!(s.contains("- a"));
    }
}
