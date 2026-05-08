//! GraphViz DOT renderer.
//!
//! The output is valid `dot` source — pipe through `dot -Tsvg -o graph.svg`
//! or paste into <https://dreampuf.github.io/GraphvizOnline/>.

use crate::compiled::CompiledGraph;
use crate::state::GraphState;

use super::extract_edges;

impl<S: GraphState> CompiledGraph<S> {
    /// Render the graph as a GraphViz `digraph` source string.
    ///
    /// Layout:
    /// - Every registered node is a rectangle.
    /// - The configured start node is highlighted with a bold border.
    /// - A synthetic `__END__` node is included as a stadium-shaped sink.
    /// - Static edges declared via `Graph::edge` are drawn as arrows.
    ///
    /// Dynamic routing (via `Goto::Send` / `Goto::Multiple` returned from a
    /// node) is **not** captured — only what the builder declared statically.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph G {\n");
        out.push_str("    rankdir=TB;\n");
        out.push_str("    node [shape=box, style=rounded];\n");

        let mut names: Vec<&String> = self.graph.nodes.keys().collect();
        names.sort();

        // Node declarations.
        for name in &names {
            let id = node_id(name);
            let label = escape(name);
            if Some(*name) == self.graph.start.as_ref() {
                out.push_str(&format!("    {id} [label=\"{label}\", penwidth=2.0];\n"));
            } else {
                out.push_str(&format!("    {id} [label=\"{label}\"];\n"));
            }
        }
        out.push_str(
            "    __END__ [label=\"END\", shape=oval, style=filled, fillcolor=\"#eeeeee\"];\n",
        );

        // Static edges.
        for e in extract_edges(self) {
            out.push_str(&format!(
                "    {} -> {};\n",
                node_id(&e.from),
                node_id(&e.to)
            ));
        }

        out.push_str("}\n");
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
    s.replace('"', "\\\"")
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
    fn renders_basic_digraph() {
        let g = build();
        let d = g.to_dot();
        assert!(d.starts_with("digraph G {\n"));
        assert!(
            d.contains("a [label=\"a\", penwidth=2.0];"),
            "start node bold:\n{d}"
        );
        assert!(d.contains("b [label=\"b\"];"));
        assert!(d.contains("a -> b;"));
        assert!(d.contains("__END__"));
        assert!(d.trim_end().ends_with('}'));
    }

    #[test]
    fn escapes_special_chars_in_node_labels() {
        let g = Graph::<S>::new()
            .node(
                "node-with-hyphen",
                node_fn::<S, _, _>("node-with-hyphen", |_, _| async move {
                    Ok(NodeOut {
                        update: SU,
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("node-with-hyphen")
            .compile()
            .unwrap();
        let d = g.to_dot();
        // ID is sanitized, label is preserved.
        assert!(d.contains("node_with_hyphen [label=\"node-with-hyphen\""));
    }
}
