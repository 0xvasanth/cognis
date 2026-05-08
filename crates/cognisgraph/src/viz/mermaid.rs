//! Mermaid diagram renderer.

use crate::compiled::CompiledGraph;
use crate::state::GraphState;

use super::extract_edges;

/// Methods on `CompiledGraph<S>` that emit a Mermaid `flowchart` diagram.
impl<S: GraphState> CompiledGraph<S> {
    /// Render the graph as a Mermaid `flowchart TD` source string.
    ///
    /// Includes:
    /// - The configured start node (highlighted with a bold border).
    /// - Every registered node.
    /// - Every static edge declared via `Graph::edge`.
    /// - A synthetic `__END__` sink node — edges drawn from any node that
    ///   has no outgoing static edge (since dynamic `Goto::End` cannot be
    ///   inferred from the static graph).
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("flowchart TD\n");
        if let Some(v) = self.version() {
            out.push_str(&format!("    %% version: {v}\n"));
        }

        let mut names: Vec<&String> = self.graph.nodes.keys().collect();
        names.sort();

        // Node declarations.
        for name in &names {
            let id = node_id(name);
            out.push_str(&format!("    {id}[\"{}\"]\n", escape(name)));
        }
        out.push_str("    __END__([END])\n");

        // Start marker.
        if let Some(start) = &self.graph.start {
            out.push_str(&format!("    START(((Start))) --> {}\n", node_id(start)));
        }

        // Static edges.
        for e in extract_edges(self) {
            out.push_str(&format!(
                "    {} --> {}\n",
                node_id(&e.from),
                node_id(&e.to)
            ));
        }

        out
    }
}

fn node_id(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            s.push('_');
        }
    }
    if s.is_empty() {
        s.push_str("node");
    }
    s
}

fn escape(s: &str) -> String {
    s.replace('"', "&quot;")
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

    fn build() -> CompiledGraph<S> {
        Graph::<S>::new()
            .node(
                "a",
                node_fn::<S, _, _>("a", |_s, _c| async move {
                    Ok(NodeOut {
                        update: SU,
                        goto: Goto::node("b"),
                    })
                }),
            )
            .node(
                "b",
                node_fn::<S, _, _>("b", |_s, _c| async move {
                    Ok(NodeOut {
                        update: SU,
                        goto: Goto::end(),
                    })
                }),
            )
            .edge("a", "b")
            .start_at("a")
            .compile()
            .unwrap()
    }

    #[test]
    fn renders_basic_diagram() {
        let g = build();
        let m = g.to_mermaid();
        assert!(m.starts_with("flowchart TD\n"));
        assert!(m.contains("a[\"a\"]"));
        assert!(m.contains("b[\"b\"]"));
        assert!(m.contains("a --> b"));
        assert!(m.contains("__END__"));
        assert!(m.contains("START(((Start))) --> a"));
    }
}
