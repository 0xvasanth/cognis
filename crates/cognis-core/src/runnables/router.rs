use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::RunnableConfig;

/// Dispatches to a named runnable based on a `key` field in the input.
///
/// Expects input of the form `{"key": "<name>", "input": <value>}`.
pub struct RouterRunnable {
    runnables: HashMap<String, Arc<dyn Runnable>>,
}

impl RouterRunnable {
    pub fn new(runnables: HashMap<String, Arc<dyn Runnable>>) -> Self {
        Self { runnables }
    }
}

#[async_trait]
impl Runnable for RouterRunnable {
    fn name(&self) -> &str {
        "RouterRunnable"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let key = input
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CognisError::InvalidKey("Input must have a string 'key' field".into())
            })?
            .to_string();

        let inner_input = input.get("input").cloned().unwrap_or(Value::Null);

        let runnable = self.runnables.get(&key).ok_or_else(|| {
            CognisError::InvalidKey(format!(
                "No runnable found for key '{}'. Available: {:?}",
                key,
                self.runnables.keys().collect::<Vec<_>>()
            ))
        })?;

        runnable.invoke(inner_input, config).await
    }
}

// ---------------------------------------------------------------------------
// RunnableRouter — closure-based dynamic routing
// ---------------------------------------------------------------------------

/// A routing function that examines input and returns a route name.
type RoutingFn = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// Dynamic router that dispatches to named runnables based on a user-provided
/// routing function.
///
/// The routing function examines the input `Value` and returns a route name
/// (a `String` key). The router then looks up and invokes the corresponding
/// runnable. If the route is not found and a default runnable is configured,
/// the default is invoked instead. If no default exists, an error is returned.
///
/// # Builder pattern
///
/// ```ignore
/// use cognis_core::runnables::RunnableRouter;
///
/// let router = RunnableRouter::new(|input| {
///         input.get("type").and_then(|v| v.as_str()).unwrap_or("default").to_string()
///     })
///     .route("math", Arc::new(math_runnable))
///     .route("code", Arc::new(code_runnable))
///     .default(Arc::new(fallback_runnable))
///     .build();
/// ```
pub struct RunnableRouter {
    /// Named routes.
    routes: HashMap<String, Arc<dyn Runnable>>,
    /// Function that examines input and returns a route name.
    routing_fn: RoutingFn,
    /// Optional default runnable when no route matches.
    default: Option<Arc<dyn Runnable>>,
}

impl RunnableRouter {
    /// Create a new `RunnableRouter` with the given routing function.
    ///
    /// The routing function receives the full input `Value` and must return
    /// the name of the route to invoke.
    pub fn new(routing_fn: impl Fn(&Value) -> String + Send + Sync + 'static) -> Self {
        Self {
            routes: HashMap::new(),
            routing_fn: Arc::new(routing_fn),
            default: None,
        }
    }

    /// Add a named route.
    pub fn route(mut self, name: impl Into<String>, runnable: Arc<dyn Runnable>) -> Self {
        self.routes.insert(name.into(), runnable);
        self
    }

    /// Set the default fallback runnable, invoked when no route matches.
    pub fn default(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.default = Some(runnable);
        self
    }

    /// Finalize and return the router. This is a no-op but completes the
    /// builder pattern for readability.
    pub fn build(self) -> Self {
        self
    }

    /// Returns the list of registered route names.
    pub fn route_names(&self) -> Vec<&String> {
        self.routes.keys().collect()
    }
}

#[async_trait]
impl Runnable for RunnableRouter {
    fn name(&self) -> &str {
        "RunnableRouter"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let route_key = (self.routing_fn)(&input);

        if let Some(runnable) = self.routes.get(&route_key) {
            return runnable.invoke(input, config).await;
        }

        if let Some(ref default) = self.default {
            return default.invoke(input, config).await;
        }

        Err(CognisError::InvalidKey(format!(
            "No route found for key '{}' and no default configured. Available routes: {:?}",
            route_key,
            self.routes.keys().collect::<Vec<_>>()
        )))
    }
}

// ---------------------------------------------------------------------------
// RunnableFnBranch — closure-based conditional branching
// ---------------------------------------------------------------------------

/// A condition function that examines input and returns whether the branch
/// should be taken.
type ConditionFn = Arc<dyn Fn(&Value) -> bool + Send + Sync>;

/// Conditional branching runnable that evaluates closure-based conditions in
/// order, executing the first matching branch's runnable.
///
/// This is a simpler alternative to [`RunnableRouter`] when you need
/// if/elif/else style logic rather than named routing. Conditions are
/// evaluated sequentially; the first one returning `true` wins.
///
/// If no condition matches, the default runnable is invoked.
///
/// # Example
///
/// ```ignore
/// use cognis_core::runnables::RunnableFnBranch;
///
/// let branch = RunnableFnBranch::new(
///     vec![
///         (Arc::new(|v: &Value| v.get("lang").and_then(|l| l.as_str()) == Some("rust")),
///          Arc::new(rust_handler) as Arc<dyn Runnable>),
///     ],
///     Arc::new(default_handler),
/// );
/// ```
pub struct RunnableFnBranch {
    /// Ordered list of (condition, runnable) pairs.
    branches: Vec<(ConditionFn, Arc<dyn Runnable>)>,
    /// Default runnable invoked when no condition matches.
    default: Arc<dyn Runnable>,
}

impl RunnableFnBranch {
    /// Create a new `RunnableFnBranch` from condition/runnable pairs and a default.
    pub fn new(
        branches: Vec<(ConditionFn, Arc<dyn Runnable>)>,
        default: Arc<dyn Runnable>,
    ) -> Self {
        Self { branches, default }
    }

    /// Returns the number of condition branches (excluding the default).
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
}

#[async_trait]
impl Runnable for RunnableFnBranch {
    fn name(&self) -> &str {
        "RunnableFnBranch"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        for (condition, runnable) in &self.branches {
            if condition(&input) {
                return runnable.invoke(input, config).await;
            }
        }
        self.default.invoke(input, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::RunnableLambda;
    use serde_json::json;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn label_runnable(label: &'static str) -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new(label, move |input: Value| async move {
            Ok(json!({ "from": label, "input": input }))
        }))
    }

    fn config_reader_runnable() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::with_config(
            "config_reader",
            |input: Value, config: Option<RunnableConfig>| async move {
                let tag = config
                    .and_then(|c| c.metadata.get("tag").cloned())
                    .unwrap_or_else(|| json!("none"));
                Ok(json!({ "input": input, "tag": tag }))
            },
        ))
    }

    // ── RunnableRouter tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_basic_routing_to_correct_branch() {
        let router = RunnableRouter::new(|input| {
            input
                .get("route")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string()
        })
        .route("alpha", label_runnable("alpha_handler"))
        .route("beta", label_runnable("beta_handler"))
        .build();

        let result = router
            .invoke(json!({"route": "alpha", "data": 1}), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "alpha_handler");

        let result = router
            .invoke(json!({"route": "beta", "data": 2}), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "beta_handler");
    }

    #[tokio::test]
    async fn test_router_default_fallback_when_no_route_matches() {
        let router = RunnableRouter::new(|_| "nonexistent".to_string())
            .route("a", label_runnable("a_handler"))
            .default(label_runnable("fallback"))
            .build();

        let result = router.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "fallback");
    }

    #[tokio::test]
    async fn test_router_error_when_no_route_and_no_default() {
        let router = RunnableRouter::new(|_| "missing".to_string())
            .route("a", label_runnable("a_handler"))
            .build();

        let err = router.invoke(json!("test"), None).await.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("missing"),
            "Error should mention the missing route key: {msg}"
        );
        assert!(
            msg.contains("No route found"),
            "Error should indicate no route found: {msg}"
        );
    }

    #[tokio::test]
    async fn test_router_multiple_routes() {
        let router = RunnableRouter::new(|input| {
            input
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string()
        })
        .route("math", label_runnable("math_solver"))
        .route("code", label_runnable("code_gen"))
        .route("chat", label_runnable("chatbot"))
        .route("search", label_runnable("searcher"))
        .build();

        for (cat, expected) in [
            ("math", "math_solver"),
            ("code", "code_gen"),
            ("chat", "chatbot"),
            ("search", "searcher"),
        ] {
            let result = router.invoke(json!({"category": cat}), None).await.unwrap();
            assert_eq!(
                result["from"], expected,
                "Route '{cat}' should go to '{expected}'"
            );
        }
    }

    #[tokio::test]
    async fn test_router_routing_fn_receives_full_input() {
        // The routing function should see the entire input, not a subset.
        let router = RunnableRouter::new(|input| {
            // Verify we can access nested fields
            let nested = input
                .get("meta")
                .and_then(|m| m.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            nested.to_string()
        })
        .route("query", label_runnable("query_handler"))
        .build();

        let input = json!({"meta": {"type": "query"}, "payload": "hello"});
        let result = router.invoke(input, None).await.unwrap();
        assert_eq!(result["from"], "query_handler");
    }

    #[tokio::test]
    async fn test_router_name_formatting() {
        let router = RunnableRouter::new(|_| "x".to_string()).build();
        assert_eq!(router.name(), "RunnableRouter");
    }

    #[tokio::test]
    async fn test_router_config_passing_through() {
        let router = RunnableRouter::new(|_| "reader".to_string())
            .route("reader", config_reader_runnable())
            .build();

        let mut config = RunnableConfig::default();
        config.metadata.insert("tag".to_string(), json!("routed"));

        let result = router
            .invoke(json!({"data": 42}), Some(&config))
            .await
            .unwrap();
        assert_eq!(result["tag"], "routed");
    }

    #[tokio::test]
    async fn test_router_routing_based_on_json_field_value() {
        let router =
            RunnableRouter::new(
                |input| match input.get("priority").and_then(|v| v.as_i64()) {
                    Some(p) if p >= 10 => "high".to_string(),
                    Some(_) => "low".to_string(),
                    None => "unknown".to_string(),
                },
            )
            .route("high", label_runnable("high_priority"))
            .route("low", label_runnable("low_priority"))
            .default(label_runnable("unknown_priority"))
            .build();

        let result = router.invoke(json!({"priority": 15}), None).await.unwrap();
        assert_eq!(result["from"], "high_priority");

        let result = router.invoke(json!({"priority": 3}), None).await.unwrap();
        assert_eq!(result["from"], "low_priority");

        let result = router.invoke(json!({}), None).await.unwrap();
        assert_eq!(result["from"], "unknown_priority");
    }

    #[tokio::test]
    async fn test_router_complex_routing_logic_nested_fields() {
        let router = RunnableRouter::new(|input| {
            let domain = input
                .get("request")
                .and_then(|r| r.get("domain"))
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let action = input
                .get("request")
                .and_then(|r| r.get("action"))
                .and_then(|a| a.as_str())
                .unwrap_or("");
            format!("{}.{}", domain, action)
        })
        .route("user.create", label_runnable("create_user"))
        .route("user.delete", label_runnable("delete_user"))
        .route("order.create", label_runnable("create_order"))
        .default(label_runnable("unknown_action"))
        .build();

        let result = router
            .invoke(
                json!({"request": {"domain": "user", "action": "create"}}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(result["from"], "create_user");

        let result = router
            .invoke(
                json!({"request": {"domain": "order", "action": "create"}}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(result["from"], "create_order");

        let result = router
            .invoke(
                json!({"request": {"domain": "billing", "action": "refund"}}),
                None,
            )
            .await
            .unwrap();
        assert_eq!(result["from"], "unknown_action");
    }

    // ── RunnableFnBranch tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_fn_branch_first_match_wins() {
        let branch = RunnableFnBranch::new(
            vec![
                (
                    Arc::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(1)),
                    label_runnable("first"),
                ),
                (
                    Arc::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()).is_some()),
                    label_runnable("second"),
                ),
            ],
            label_runnable("default"),
        );

        // x=1 matches both conditions, but first wins
        let result = branch.invoke(json!({"x": 1}), None).await.unwrap();
        assert_eq!(result["from"], "first");

        // x=2 matches only second
        let result = branch.invoke(json!({"x": 2}), None).await.unwrap();
        assert_eq!(result["from"], "second");
    }

    #[tokio::test]
    async fn test_fn_branch_falls_through_to_default() {
        let branch = RunnableFnBranch::new(
            vec![(
                Arc::new(|v: &Value| v.get("match").and_then(|m| m.as_bool()) == Some(true)),
                label_runnable("matched"),
            )],
            label_runnable("default"),
        );

        let result = branch.invoke(json!({"match": false}), None).await.unwrap();
        assert_eq!(result["from"], "default");

        let result = branch.invoke(json!({}), None).await.unwrap();
        assert_eq!(result["from"], "default");
    }

    #[tokio::test]
    async fn test_fn_branch_single_condition() {
        let branch = RunnableFnBranch::new(
            vec![(Arc::new(|_: &Value| true), label_runnable("always"))],
            label_runnable("never"),
        );

        let result = branch.invoke(json!("anything"), None).await.unwrap();
        assert_eq!(result["from"], "always");
    }

    #[tokio::test]
    async fn test_fn_branch_name() {
        let branch = RunnableFnBranch::new(vec![], label_runnable("default"));
        assert_eq!(branch.name(), "RunnableFnBranch");
    }
}
