//! Conditional branching for graph edges.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::errors::LangGraphError;
use crate::types::Send as GraphSend;

/// A routing function that inspects state and returns one or more destination nodes.
pub type RouterFn = Arc<dyn Fn(&Value) -> RouterResult + std::marker::Send + Sync>;

/// Result from a router function.
#[derive(Debug, Clone)]
pub enum RouterResult {
    /// Route to a single node.
    Single(String),
    /// Route to multiple nodes.
    Multiple(Vec<String>),
    /// Send with custom state to specific nodes.
    Sends(Vec<GraphSend>),
}

/// Specification for a conditional branch.
#[derive(Clone)]
pub struct Branch {
    /// The routing function.
    pub path: RouterFn,
    /// Optional mapping of route values to node names.
    pub ends: Option<HashMap<String, String>>,
    /// Optional default node name used when the router returns a key not
    /// present in `ends`. When `None` and a path map is provided, unmapped
    /// keys pass through as literal node names (existing behaviour).
    pub default: Option<String>,
}

impl fmt::Debug for Branch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Branch")
            .field("ends", &self.ends)
            .field("default", &self.default)
            .field("path", &"<RouterFn>")
            .finish()
    }
}

impl Branch {
    /// Create a new branch with the given routing function.
    pub fn new(path: RouterFn) -> Self {
        Self {
            path,
            ends: None,
            default: None,
        }
    }

    /// Set a mapping from route values to actual node names.
    pub fn with_path_map(mut self, map: HashMap<String, String>) -> Self {
        self.ends = Some(map);
        self
    }

    /// Set a default target node used when the router returns a key that is
    /// not present in the path map.
    pub fn with_default(mut self, default: String) -> Self {
        self.default = Some(default);
        self
    }

    /// Resolve the routing result asynchronously.
    /// Falls back to sync resolution if no async router is set.
    pub async fn resolve_async(&self, state: &Value) -> Result<Vec<String>, LangGraphError> {
        self.resolve(state)
    }

    /// Resolve a single route key through the path map and default.
    fn resolve_key(&self, key: String) -> Result<String, LangGraphError> {
        if let Some(ends) = &self.ends {
            if let Some(mapped) = ends.get(&key) {
                Ok(mapped.clone())
            } else if let Some(default) = &self.default {
                Ok(default.clone())
            } else {
                // Passthrough: use the key as-is (preserves existing behaviour).
                Ok(key)
            }
        } else {
            Ok(key)
        }
    }

    /// Resolve the routing result to actual destination node names.
    pub fn resolve(&self, state: &Value) -> Result<Vec<String>, LangGraphError> {
        let result = (self.path)(state);
        match result {
            RouterResult::Single(node) => Ok(vec![self.resolve_key(node)?]),
            RouterResult::Multiple(nodes) => {
                nodes.into_iter().map(|n| self.resolve_key(n)).collect()
            }
            RouterResult::Sends(sends) => Ok(sends.into_iter().map(|s| s.node).collect()),
        }
    }

    /// Resolve the routing result, preserving [`GraphSend`] instructions.
    ///
    /// Returns the raw [`RouterResult`] with path-map resolution applied to
    /// `Single` and `Multiple` variants. `Sends` are returned as-is (with
    /// node names resolved through the path map when present).
    pub fn resolve_raw(&self, state: &Value) -> Result<RouterResult, LangGraphError> {
        let result = (self.path)(state);
        match result {
            RouterResult::Single(node) => {
                let mapped = self.resolve_key(node)?;
                Ok(RouterResult::Single(mapped))
            }
            RouterResult::Multiple(nodes) => {
                let mapped: Result<Vec<String>, LangGraphError> =
                    nodes.into_iter().map(|n| self.resolve_key(n)).collect();
                Ok(RouterResult::Multiple(mapped?))
            }
            RouterResult::Sends(sends) => {
                let mapped = if let Some(ends) = &self.ends {
                    sends
                        .into_iter()
                        .map(|mut s| {
                            if let Some(mapped_name) = ends.get(&s.node) {
                                s.node = mapped_name.clone();
                            } else if let Some(default) = &self.default {
                                s.node = default.clone();
                            }
                            s
                        })
                        .collect()
                } else {
                    sends
                };
                Ok(RouterResult::Sends(mapped))
            }
        }
    }
}

/// An async routing function.
pub type AsyncRouterFn =
    Arc<dyn Fn(&Value) -> Pin<Box<dyn Future<Output = RouterResult> + Send + '_>> + Send + Sync>;

/// A branch with an async routing function.
#[derive(Clone)]
pub struct AsyncBranch {
    /// The async routing function.
    pub path: AsyncRouterFn,
    /// Optional mapping of route values to node names.
    pub ends: Option<HashMap<String, String>>,
}

impl fmt::Debug for AsyncBranch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncBranch")
            .field("ends", &self.ends)
            .field("path", &"<AsyncRouterFn>")
            .finish()
    }
}

impl AsyncBranch {
    pub fn new(path: AsyncRouterFn) -> Self {
        Self { path, ends: None }
    }

    pub fn with_path_map(mut self, map: HashMap<String, String>) -> Self {
        self.ends = Some(map);
        self
    }

    pub async fn resolve(&self, state: &Value) -> Result<Vec<String>, LangGraphError> {
        let result = (self.path)(state).await;
        match result {
            RouterResult::Single(node) => {
                if let Some(ends) = &self.ends {
                    Ok(vec![ends.get(&node).cloned().unwrap_or(node)])
                } else {
                    Ok(vec![node])
                }
            }
            RouterResult::Multiple(nodes) => {
                if let Some(ends) = &self.ends {
                    Ok(nodes
                        .into_iter()
                        .map(|n| ends.get(&n).cloned().unwrap_or(n))
                        .collect())
                } else {
                    Ok(nodes)
                }
            }
            RouterResult::Sends(sends) => Ok(sends.into_iter().map(|s| s.node).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Send as GraphSend;
    use serde_json::json;

    #[test]
    fn test_branch_single_route() {
        let branch = Branch::new(Arc::new(|_state: &Value| {
            RouterResult::Single("node_b".to_string())
        }));

        let result = branch.resolve(&json!({})).unwrap();
        assert_eq!(result, vec!["node_b".to_string()]);
    }

    #[test]
    fn test_branch_with_path_map() {
        let mut map = HashMap::new();
        map.insert("left".to_string(), "node_left".to_string());
        map.insert("right".to_string(), "node_right".to_string());

        let branch = Branch::new(Arc::new(|_state: &Value| {
            RouterResult::Single("left".to_string())
        }))
        .with_path_map(map);

        let result = branch.resolve(&json!({})).unwrap();
        assert_eq!(result, vec!["node_left".to_string()]);
    }

    #[test]
    fn test_branch_multiple_routes() {
        let branch = Branch::new(Arc::new(|_state: &Value| {
            RouterResult::Multiple(vec!["a".to_string(), "b".to_string()])
        }));

        let result = branch.resolve(&json!({})).unwrap();
        assert_eq!(result, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_branch_sends() {
        let branch = Branch::new(Arc::new(|_state: &Value| {
            RouterResult::Sends(vec![
                GraphSend {
                    node: "worker".to_string(),
                    arg: json!({"task": 1}),
                },
                GraphSend {
                    node: "worker".to_string(),
                    arg: json!({"task": 2}),
                },
            ])
        }));

        let result = branch.resolve(&json!({})).unwrap();
        assert_eq!(result, vec!["worker".to_string(), "worker".to_string()]);
    }

    #[tokio::test]
    async fn test_async_branch_single_route() {
        let branch = AsyncBranch::new(Arc::new(|_state: &Value| {
            Box::pin(async { RouterResult::Single("node_b".to_string()) })
        }));
        let result = branch.resolve(&json!({})).await.unwrap();
        assert_eq!(result, vec!["node_b".to_string()]);
    }

    #[tokio::test]
    async fn test_async_branch_with_path_map() {
        let mut map = HashMap::new();
        map.insert("go".to_string(), "target".to_string());
        let branch = AsyncBranch::new(Arc::new(|_state: &Value| {
            Box::pin(async { RouterResult::Single("go".to_string()) })
        }))
        .with_path_map(map);
        let result = branch.resolve(&json!({})).await.unwrap();
        assert_eq!(result, vec!["target".to_string()]);
    }

    #[test]
    fn test_branch_unmapped_key_passes_through() {
        let mut map = HashMap::new();
        map.insert("known".to_string(), "mapped_node".to_string());

        let branch = Branch::new(Arc::new(|_state: &Value| {
            RouterResult::Single("unknown".to_string())
        }))
        .with_path_map(map);

        let result = branch.resolve(&json!({})).unwrap();
        assert_eq!(result, vec!["unknown".to_string()]);
    }
}
