//! Static graph analysis — cycle detection, longest path, reachability.
//!
//! Operates on the **static** edges declared via [`crate::builder::Graph::edge`].
//! Dynamic routing (a node returning `Goto::Send` to a peer that wasn't
//! declared statically) is invisible here — by design, since this is a
//! diagnostics tool for the graph topology you wrote, not its runtime
//! behavior.
//!
//! ```ignore
//! let analysis = compiled.analyze();
//! if analysis.has_cycle {
//!     eprintln!("cycle detected");
//! }
//! println!("longest path: {} hops", analysis.depth);
//! ```
//!
//! `depth` is the longest *acyclic* path from the start node to any
//! reachable node, measured in edge hops. For graphs with cycles, depth
//! is `0` (it's only meaningful for DAGs).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::compiled::CompiledGraph;
use crate::state::GraphState;
use crate::viz::extract_edges;

/// Output of [`CompiledGraph::analyze`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphAnalysis {
    /// Number of registered nodes (excludes the synthetic `__END__`).
    pub node_count: usize,
    /// Number of static edges declared on the graph.
    pub edge_count: usize,
    /// `true` if the static edge graph contains a cycle reachable from
    /// the start node.
    pub has_cycle: bool,
    /// Longest path from the start node to any reachable node, measured
    /// in edges. `0` if the graph contains a cycle (depth is undefined
    /// for non-DAGs) or has no start node.
    pub depth: usize,
    /// Nodes registered on the graph but not reachable from the start
    /// via static edges. Sorted alphabetically.
    pub unreachable: Vec<String>,
}

impl<S: GraphState> CompiledGraph<S> {
    /// Compute static-graph diagnostics. See [`GraphAnalysis`].
    pub fn analyze(&self) -> GraphAnalysis {
        let nodes: Vec<&String> = self.graph.nodes.keys().collect();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &nodes {
            adj.insert(node.as_str(), Vec::new());
        }
        let edges = extract_edges(self);
        for e in &edges {
            adj.entry(e.from.as_str()).or_default().push(e.to.as_str());
        }

        let start = self.graph.start.as_deref();

        // Reachability via BFS from start. Skips the synthetic __END__
        // sink — we care about whether registered nodes are reached.
        let mut reachable: HashSet<&str> = HashSet::new();
        if let Some(s) = start {
            let mut q: VecDeque<&str> = VecDeque::from([s]);
            reachable.insert(s);
            while let Some(n) = q.pop_front() {
                if let Some(nexts) = adj.get(n) {
                    for &m in nexts {
                        if m == "__END__" {
                            continue;
                        }
                        if reachable.insert(m) {
                            q.push_back(m);
                        }
                    }
                }
            }
        }
        let mut unreachable: Vec<String> = nodes
            .iter()
            .filter(|n| !reachable.contains(n.as_str()))
            .map(|n| (*n).clone())
            .collect();
        unreachable.sort();

        // Cycle detection over the reachable subgraph via DFS three-color.
        let has_cycle = match start {
            Some(s) => detect_cycle(s, &adj, &reachable),
            None => false,
        };

        // Longest path: only meaningful for DAGs reachable from start.
        let depth = if has_cycle || start.is_none() {
            0
        } else {
            longest_path(start.unwrap(), &adj, &reachable)
        };

        GraphAnalysis {
            node_count: self.graph.nodes.len(),
            edge_count: edges.len(),
            has_cycle,
            depth,
            unreachable,
        }
    }
}

fn detect_cycle<'a>(
    start: &'a str,
    adj: &HashMap<&'a str, Vec<&'a str>>,
    reachable: &HashSet<&'a str>,
) -> bool {
    enum Color {
        White,
        Gray,
        Black,
    }
    let mut color: HashMap<&str, Color> = HashMap::new();
    for &n in reachable {
        color.insert(n, Color::White);
    }
    fn visit<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        reachable: &HashSet<&'a str>,
        color: &mut HashMap<&'a str, Color>,
    ) -> bool {
        color.insert(node, Color::Gray);
        if let Some(nexts) = adj.get(node) {
            for &m in nexts {
                if m == "__END__" || !reachable.contains(m) {
                    continue;
                }
                match color.get(m) {
                    Some(Color::Gray) => return true, // back-edge → cycle
                    Some(Color::White) => {
                        if visit(m, adj, reachable, color) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        color.insert(node, Color::Black);
        false
    }
    visit(start, adj, reachable, &mut color)
}

fn longest_path<'a>(
    start: &'a str,
    adj: &HashMap<&'a str, Vec<&'a str>>,
    reachable: &HashSet<&'a str>,
) -> usize {
    // Memoized DFS over the DAG. Caller must ensure no cycles.
    fn dfs<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        reachable: &HashSet<&'a str>,
        memo: &mut HashMap<&'a str, usize>,
    ) -> usize {
        if let Some(&d) = memo.get(node) {
            return d;
        }
        let mut best = 0;
        if let Some(nexts) = adj.get(node) {
            for &m in nexts {
                if m == "__END__" || !reachable.contains(m) {
                    continue;
                }
                let d = 1 + dfs(m, adj, reachable, memo);
                if d > best {
                    best = d;
                }
            }
        }
        memo.insert(node, best);
        best
    }
    let mut memo: HashMap<&str, usize> = HashMap::new();
    dfs(start, adj, reachable, &mut memo)
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

    fn nf(name: &'static str) -> impl crate::Node<S> + 'static {
        node_fn::<S, _, _>(name, |_, _| async move {
            Ok(NodeOut {
                update: SU,
                goto: Goto::end(),
            })
        })
    }

    #[test]
    fn linear_three_node_chain_depth_2() {
        // a → b → c
        let g = Graph::<S>::new()
            .node("a", nf("a"))
            .node("b", nf("b"))
            .node("c", nf("c"))
            .edge("a", "b")
            .edge("b", "c")
            .start_at("a")
            .compile()
            .unwrap();
        let r = g.analyze();
        assert_eq!(r.node_count, 3);
        assert_eq!(r.edge_count, 2);
        assert!(!r.has_cycle);
        assert_eq!(r.depth, 2);
        assert!(r.unreachable.is_empty());
    }

    #[test]
    fn diamond_depth_is_2() {
        // a → b → d
        // a → c → d
        let g = Graph::<S>::new()
            .node("a", nf("a"))
            .node("b", nf("b"))
            .node("c", nf("c"))
            .node("d", nf("d"))
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .start_at("a")
            .compile()
            .unwrap();
        let r = g.analyze();
        assert!(!r.has_cycle);
        assert_eq!(r.depth, 2);
    }

    #[test]
    fn self_loop_is_a_cycle() {
        // a → a (cycle)
        let g = Graph::<S>::new()
            .node("a", nf("a"))
            .edge("a", "a")
            .start_at("a")
            .compile()
            .unwrap();
        let r = g.analyze();
        assert!(r.has_cycle);
        assert_eq!(r.depth, 0, "depth undefined for graphs with cycles");
    }

    #[test]
    fn detects_back_edge_cycle() {
        // a → b → c → a
        let g = Graph::<S>::new()
            .node("a", nf("a"))
            .node("b", nf("b"))
            .node("c", nf("c"))
            .edge("a", "b")
            .edge("b", "c")
            .edge("c", "a")
            .start_at("a")
            .compile()
            .unwrap();
        let r = g.analyze();
        assert!(r.has_cycle);
    }

    #[test]
    fn unreachable_nodes_listed() {
        // a → b ;  c is registered but no edge to/from start.
        let g = Graph::<S>::new()
            .node("a", nf("a"))
            .node("b", nf("b"))
            .node("c", nf("c"))
            .edge("a", "b")
            .start_at("a")
            .compile()
            .unwrap();
        let r = g.analyze();
        assert_eq!(r.unreachable, vec!["c".to_string()]);
        assert!(!r.has_cycle);
        assert_eq!(r.depth, 1);
    }
}
