//! `Goto` — where the engine routes after a node finishes.

/// Routing instruction returned by a node, telling the engine which node(s)
/// to schedule next. Branching/looping is data: a node decides at runtime
/// where to send execution.
#[derive(Debug, Clone, PartialEq)]
pub enum Goto {
    /// Jump to a single named node.
    Node(String),

    /// Fan out to multiple named nodes in parallel. All named nodes run in
    /// the next superstep; their updates merge atomically.
    Multiple(Vec<String>),

    /// Dispatch to named nodes with per-target payloads. Each `(name, payload)`
    /// pair spawns one task; the receiving node reads the payload via
    /// `NodeCtx::payload()`. Useful for map-reduce: fan one node into many
    /// workers, each with different input.
    Send(Vec<(String, serde_json::Value)>),

    /// Stop **this branch** without terminating the whole graph. Useful
    /// when a node has nothing more to dispatch but other concurrent
    /// branches are still running. Distinct from [`Goto::End`], which
    /// terminates the entire graph.
    Halt,

    /// Terminate the graph.
    End,
}

impl Goto {
    /// Convenience constructor: `Goto::node("think")` over `Goto::Node("think".into())`.
    pub fn node(name: impl Into<String>) -> Self {
        Goto::Node(name.into())
    }

    /// Convenience constructor: `Goto::end()` over `Goto::End`.
    pub fn end() -> Self {
        Goto::End
    }

    /// Convenience: `Goto::halt()` over `Goto::Halt`.
    pub fn halt() -> Self {
        Goto::Halt
    }

    /// All node-name targets contained in this Goto. Empty for `End` / `Halt`.
    pub fn targets(&self) -> Vec<&str> {
        match self {
            Goto::Node(n) => vec![n.as_str()],
            Goto::Multiple(ns) => ns.iter().map(|s| s.as_str()).collect(),
            Goto::Send(targets) => targets.iter().map(|(n, _)| n.as_str()).collect(),
            Goto::End | Goto::Halt => vec![],
        }
    }

    /// True if this Goto terminates the graph.
    pub fn is_end(&self) -> bool {
        matches!(self, Goto::End)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_work() {
        assert_eq!(Goto::node("think"), Goto::Node("think".into()));
        assert!(Goto::end().is_end());
    }

    #[test]
    fn targets_correct() {
        assert_eq!(Goto::node("a").targets(), vec!["a"]);
        assert_eq!(
            Goto::Multiple(vec!["a".into(), "b".into()]).targets(),
            vec!["a", "b"]
        );
        assert!(Goto::end().targets().is_empty());
    }
}
