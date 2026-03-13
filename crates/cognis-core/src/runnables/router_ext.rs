use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::RunnableConfig;

// ---------------------------------------------------------------------------
// ContentRouter — routes based on content inspection predicates
// ---------------------------------------------------------------------------

/// A named route entry with a predicate and target runnable.
struct ContentRoute {
    name: String,
    predicate: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    target: Arc<dyn Runnable>,
}

/// Routes input to a target runnable based on content inspection.
///
/// Predicates are evaluated in insertion order; the first matching predicate
/// determines the target runnable that will be invoked.
pub struct ContentRouter {
    routes: Vec<ContentRoute>,
}

impl ContentRouter {
    /// Create a new empty `ContentRouter`.
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a named route with a predicate and target runnable.
    pub fn add_route(
        &mut self,
        name: String,
        predicate: Box<dyn Fn(&Value) -> bool + Send + Sync>,
        target: Arc<dyn Runnable>,
    ) {
        self.routes.push(ContentRoute {
            name,
            predicate,
            target,
        });
    }

    /// Evaluate predicates in order and return the first match.
    pub fn route(&self, input: &Value) -> Option<(&str, Arc<dyn Runnable>)> {
        for entry in &self.routes {
            if (entry.predicate)(input) {
                return Some((&entry.name, Arc::clone(&entry.target)));
            }
        }
        None
    }
}

impl Default for ContentRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for ContentRouter {
    fn name(&self) -> &str {
        "ContentRouter"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        match self.route(&input) {
            Some((_name, target)) => target.invoke(input, config).await,
            None => Err(CognisError::InvalidKey(
                "No content route matched the input".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// RegexRouter — routes based on regex pattern matching on string input
// ---------------------------------------------------------------------------

/// A compiled regex route entry.
struct RegexRoute {
    pattern: Regex,
    target: Arc<dyn Runnable>,
}

/// Routes string input to a target based on regex pattern matching.
///
/// Patterns are evaluated in insertion order; the first match wins.
/// The input must be a JSON string value for pattern matching to work.
pub struct RegexRouter {
    routes: Vec<RegexRoute>,
}

impl RegexRouter {
    /// Create a new empty `RegexRouter`.
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a regex pattern and its target runnable.
    ///
    /// # Panics
    /// Panics if the pattern is not a valid regex.
    pub fn add_pattern(&mut self, pattern: &str, target: Arc<dyn Runnable>) {
        let regex = Regex::new(pattern).expect("Invalid regex pattern");
        self.routes.push(RegexRoute {
            pattern: regex,
            target,
        });
    }
}

impl Default for RegexRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for RegexRouter {
    fn name(&self) -> &str {
        "RegexRouter"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let text = input.as_str().ok_or_else(|| CognisError::TypeMismatch {
            expected: "string".into(),
            got: format!("{}", input),
        })?;

        for route in &self.routes {
            if route.pattern.is_match(text) {
                return route.target.invoke(input, config).await;
            }
        }

        Err(CognisError::InvalidKey(format!(
            "No regex pattern matched the input: {:?}",
            text
        )))
    }
}

// ---------------------------------------------------------------------------
// KeyRouter — routes based on a specific JSON key's value
// ---------------------------------------------------------------------------

/// Routes input based on the string value of a specific JSON key.
///
/// Looks up the value at `key` in the input object and dispatches to the
/// matching route. An optional default runnable handles unmatched values.
pub struct KeyRouter {
    key: String,
    routes: HashMap<String, Arc<dyn Runnable>>,
    default: Option<Arc<dyn Runnable>>,
}

impl KeyRouter {
    /// Create a new `KeyRouter` that inspects the given JSON key.
    pub fn new(key: String) -> Self {
        Self {
            key,
            routes: HashMap::new(),
            default: None,
        }
    }

    /// Add a route for when the key has the given string value.
    pub fn add_route(&mut self, value: String, target: Arc<dyn Runnable>) {
        self.routes.insert(value, target);
    }

    /// Set a default runnable invoked when the key value does not match any route.
    pub fn with_default(&mut self, target: Arc<dyn Runnable>) {
        self.default = Some(target);
    }
}

#[async_trait]
impl Runnable for KeyRouter {
    fn name(&self) -> &str {
        "KeyRouter"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let key_value = input
            .get(&self.key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CognisError::InvalidKey(format!("Input must have a string '{}' field", self.key))
            })?
            .to_string();

        if let Some(target) = self.routes.get(&key_value) {
            return target.invoke(input, config).await;
        }

        if let Some(ref default) = self.default {
            return default.invoke(input, config).await;
        }

        Err(CognisError::InvalidKey(format!(
            "No route for key '{}' value '{}' and no default configured. Available: {:?}",
            self.key,
            key_value,
            self.routes.keys().collect::<Vec<_>>()
        )))
    }
}

// ---------------------------------------------------------------------------
// FallbackRouter — tries multiple runnables in order, returns first success
// ---------------------------------------------------------------------------

/// Tries multiple runnables in order and returns the result of the first
/// that succeeds. If all runnables fail, the last error is returned.
pub struct FallbackRouter {
    runnables: Vec<Arc<dyn Runnable>>,
}

impl FallbackRouter {
    /// Create a new `FallbackRouter` with the given list of runnables to try.
    pub fn new(runnables: Vec<Arc<dyn Runnable>>) -> Self {
        Self { runnables }
    }
}

#[async_trait]
impl Runnable for FallbackRouter {
    fn name(&self) -> &str {
        "FallbackRouter"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        if self.runnables.is_empty() {
            return Err(CognisError::Other(
                "FallbackRouter has no runnables to try".into(),
            ));
        }

        let mut last_error = None;
        for runnable in &self.runnables {
            match runnable.invoke(input.clone(), config).await {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap())
    }
}

// ---------------------------------------------------------------------------
// ConditionalBranch — evaluates conditions and branches
// ---------------------------------------------------------------------------

/// A condition/action pair for conditional branching.
struct ConditionalArm {
    condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    then: Arc<dyn Runnable>,
}

/// Evaluates conditions in order and executes the first matching branch.
///
/// Similar to an if/elif/else chain. An optional `otherwise` runnable acts
/// as the else clause.
pub struct ConditionalBranch {
    arms: Vec<ConditionalArm>,
    otherwise: Option<Arc<dyn Runnable>>,
}

impl ConditionalBranch {
    /// Create a new empty `ConditionalBranch`.
    pub fn new() -> Self {
        Self {
            arms: Vec::new(),
            otherwise: None,
        }
    }

    /// Add a condition/action pair.
    pub fn when(
        mut self,
        condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
        then: Arc<dyn Runnable>,
    ) -> Self {
        self.arms.push(ConditionalArm { condition, then });
        self
    }

    /// Set the fallback runnable for when no condition matches.
    pub fn otherwise(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.otherwise = Some(runnable);
        self
    }
}

impl Default for ConditionalBranch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for ConditionalBranch {
    fn name(&self) -> &str {
        "ConditionalBranch"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        for arm in &self.arms {
            if (arm.condition)(&input) {
                return arm.then.invoke(input, config).await;
            }
        }

        if let Some(ref otherwise) = self.otherwise {
            return otherwise.invoke(input, config).await;
        }

        Err(CognisError::InvalidKey(
            "No condition matched and no otherwise branch configured".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// RoutingTable — a named collection of routes for inspection
// ---------------------------------------------------------------------------

/// A named, inspectable collection of routes.
///
/// Unlike the router types above, `RoutingTable` does not implement `Runnable`
/// — it is a data structure for managing and inspecting routes.
pub struct RoutingTable {
    routes: Vec<(String, Arc<dyn Runnable>)>,
}

impl RoutingTable {
    /// Create a new empty routing table.
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a named route to the table.
    pub fn add(&mut self, name: impl Into<String>, target: Arc<dyn Runnable>) {
        let name = name.into();
        // Replace if exists
        if let Some(entry) = self.routes.iter_mut().find(|(n, _)| *n == name) {
            entry.1 = target;
        } else {
            self.routes.push((name, target));
        }
    }

    /// Get a route by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Runnable>> {
        self.routes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, target)| Arc::clone(target))
    }

    /// Return the names of all routes in insertion order.
    pub fn names(&self) -> Vec<&str> {
        self.routes.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Return the number of routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Return whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Serialize the routing table to a JSON value listing route names
    /// and their associated runnable names.
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self
            .routes
            .iter()
            .map(|(name, target)| {
                json!({
                    "name": name,
                    "runnable": target.name(),
                })
            })
            .collect();
        json!({ "routes": entries })
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    fn failing_runnable(msg: &'static str) -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new(msg, move |_input: Value| async move {
            Err(CognisError::Other(msg.into()))
        }))
    }

    // ── ContentRouter tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_content_router_first_predicate_match() {
        let mut router = ContentRouter::new();
        router.add_route(
            "numbers".into(),
            Box::new(|v: &Value| v.is_number()),
            label_runnable("number_handler"),
        );
        router.add_route(
            "strings".into(),
            Box::new(|v: &Value| v.is_string()),
            label_runnable("string_handler"),
        );

        let result = router.invoke(json!(42), None).await.unwrap();
        assert_eq!(result["from"], "number_handler");
    }

    #[tokio::test]
    async fn test_content_router_second_predicate_match() {
        let mut router = ContentRouter::new();
        router.add_route(
            "numbers".into(),
            Box::new(|v: &Value| v.is_number()),
            label_runnable("number_handler"),
        );
        router.add_route(
            "strings".into(),
            Box::new(|v: &Value| v.is_string()),
            label_runnable("string_handler"),
        );

        let result = router.invoke(json!("hello"), None).await.unwrap();
        assert_eq!(result["from"], "string_handler");
    }

    #[tokio::test]
    async fn test_content_router_no_match_error() {
        let mut router = ContentRouter::new();
        router.add_route(
            "numbers".into(),
            Box::new(|v: &Value| v.is_number()),
            label_runnable("number_handler"),
        );

        let err = router.invoke(json!("text"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("No content route matched"));
    }

    #[tokio::test]
    async fn test_content_router_empty_no_routes() {
        let router = ContentRouter::new();
        let err = router.invoke(json!(1), None).await.unwrap_err();
        assert!(format!("{}", err).contains("No content route matched"));
    }

    #[tokio::test]
    async fn test_content_router_ordering_matters() {
        let mut router = ContentRouter::new();
        // Both predicates match everything
        router.add_route(
            "first".into(),
            Box::new(|_: &Value| true),
            label_runnable("first_handler"),
        );
        router.add_route(
            "second".into(),
            Box::new(|_: &Value| true),
            label_runnable("second_handler"),
        );

        let result = router.invoke(json!("anything"), None).await.unwrap();
        assert_eq!(result["from"], "first_handler");
    }

    #[tokio::test]
    async fn test_content_router_route_method() {
        let mut router = ContentRouter::new();
        router.add_route(
            "arrays".into(),
            Box::new(|v: &Value| v.is_array()),
            label_runnable("array_handler"),
        );

        let (name, _target) = router.route(&json!([1, 2, 3])).unwrap();
        assert_eq!(name, "arrays");

        assert!(router.route(&json!("not array")).is_none());
    }

    #[tokio::test]
    async fn test_content_router_complex_predicate() {
        let mut router = ContentRouter::new();
        router.add_route(
            "has_name".into(),
            Box::new(|v: &Value| v.get("name").and_then(|n| n.as_str()).is_some()),
            label_runnable("name_handler"),
        );
        router.add_route(
            "has_id".into(),
            Box::new(|v: &Value| v.get("id").and_then(|n| n.as_i64()).is_some()),
            label_runnable("id_handler"),
        );

        let result = router.invoke(json!({"name": "Alice"}), None).await.unwrap();
        assert_eq!(result["from"], "name_handler");

        let result = router.invoke(json!({"id": 42}), None).await.unwrap();
        assert_eq!(result["from"], "id_handler");
    }

    #[tokio::test]
    async fn test_content_router_name() {
        let router = ContentRouter::new();
        assert_eq!(router.name(), "ContentRouter");
    }

    // ── RegexRouter tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_regex_router_basic_match() {
        let mut router = RegexRouter::new();
        router.add_pattern(r"^hello", label_runnable("greeting_handler"));
        router.add_pattern(r"\d+", label_runnable("number_handler"));

        let result = router.invoke(json!("hello world"), None).await.unwrap();
        assert_eq!(result["from"], "greeting_handler");
    }

    #[tokio::test]
    async fn test_regex_router_second_pattern_match() {
        let mut router = RegexRouter::new();
        router.add_pattern(r"^hello", label_runnable("greeting_handler"));
        router.add_pattern(r"\d+", label_runnable("number_handler"));

        let result = router.invoke(json!("value is 42"), None).await.unwrap();
        assert_eq!(result["from"], "number_handler");
    }

    #[tokio::test]
    async fn test_regex_router_no_match() {
        let mut router = RegexRouter::new();
        router.add_pattern(r"^hello", label_runnable("greeting_handler"));

        let err = router.invoke(json!("goodbye"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("No regex pattern matched"));
    }

    #[tokio::test]
    async fn test_regex_router_non_string_input() {
        let mut router = RegexRouter::new();
        router.add_pattern(r".*", label_runnable("catch_all"));

        let err = router.invoke(json!(42), None).await.unwrap_err();
        assert!(format!("{}", err).contains("expected"));
    }

    #[tokio::test]
    async fn test_regex_router_empty_no_patterns() {
        let router = RegexRouter::new();
        let err = router.invoke(json!("anything"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("No regex pattern matched"));
    }

    #[tokio::test]
    async fn test_regex_router_complex_pattern() {
        let mut router = RegexRouter::new();
        router.add_pattern(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            label_runnable("email_handler"),
        );
        router.add_pattern(r"^https?://", label_runnable("url_handler"));

        let result = router
            .invoke(json!("user@example.com"), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "email_handler");

        let result = router
            .invoke(json!("https://example.com"), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "url_handler");
    }

    #[tokio::test]
    async fn test_regex_router_name() {
        let router = RegexRouter::new();
        assert_eq!(router.name(), "RegexRouter");
    }

    // ── KeyRouter tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_key_router_basic_routing() {
        let mut router = KeyRouter::new("type".into());
        router.add_route("query".into(), label_runnable("query_handler"));
        router.add_route("command".into(), label_runnable("command_handler"));

        let result = router
            .invoke(json!({"type": "query", "data": "search"}), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "query_handler");

        let result = router
            .invoke(json!({"type": "command", "data": "run"}), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "command_handler");
    }

    #[tokio::test]
    async fn test_key_router_with_default() {
        let mut router = KeyRouter::new("action".into());
        router.add_route("save".into(), label_runnable("save_handler"));
        router.with_default(label_runnable("default_handler"));

        let result = router
            .invoke(json!({"action": "unknown"}), None)
            .await
            .unwrap();
        assert_eq!(result["from"], "default_handler");
    }

    #[tokio::test]
    async fn test_key_router_missing_key_error() {
        let router = KeyRouter::new("action".into());

        let err = router
            .invoke(json!({"other": "field"}), None)
            .await
            .unwrap_err();
        assert!(format!("{}", err).contains("action"));
    }

    #[tokio::test]
    async fn test_key_router_no_match_no_default_error() {
        let mut router = KeyRouter::new("status".into());
        router.add_route("active".into(), label_runnable("active_handler"));

        let err = router
            .invoke(json!({"status": "inactive"}), None)
            .await
            .unwrap_err();
        assert!(format!("{}", err).contains("inactive"));
    }

    #[tokio::test]
    async fn test_key_router_non_string_key_value_error() {
        let mut router = KeyRouter::new("count".into());
        router.add_route("1".into(), label_runnable("one_handler"));

        let err = router.invoke(json!({"count": 1}), None).await.unwrap_err();
        assert!(format!("{}", err).contains("count"));
    }

    #[tokio::test]
    async fn test_key_router_name() {
        let router = KeyRouter::new("key".into());
        assert_eq!(router.name(), "KeyRouter");
    }

    // ── FallbackRouter tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_fallback_router_first_succeeds() {
        let router = FallbackRouter::new(vec![label_runnable("first"), label_runnable("second")]);

        let result = router.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "first");
    }

    #[tokio::test]
    async fn test_fallback_router_first_fails_second_succeeds() {
        let router =
            FallbackRouter::new(vec![failing_runnable("error1"), label_runnable("second")]);

        let result = router.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "second");
    }

    #[tokio::test]
    async fn test_fallback_router_all_fail() {
        let router =
            FallbackRouter::new(vec![failing_runnable("error1"), failing_runnable("error2")]);

        let err = router.invoke(json!("test"), None).await.unwrap_err();
        // Last error should be returned
        assert!(format!("{}", err).contains("error2"));
    }

    #[tokio::test]
    async fn test_fallback_router_empty() {
        let router = FallbackRouter::new(vec![]);

        let err = router.invoke(json!("test"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("no runnables"));
    }

    #[tokio::test]
    async fn test_fallback_router_single_runnable_succeeds() {
        let router = FallbackRouter::new(vec![label_runnable("only")]);

        let result = router.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "only");
    }

    #[tokio::test]
    async fn test_fallback_router_single_runnable_fails() {
        let router = FallbackRouter::new(vec![failing_runnable("boom")]);

        let err = router.invoke(json!("test"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("boom"));
    }

    #[tokio::test]
    async fn test_fallback_router_name() {
        let router = FallbackRouter::new(vec![]);
        assert_eq!(router.name(), "FallbackRouter");
    }

    #[tokio::test]
    async fn test_fallback_router_third_succeeds() {
        let router = FallbackRouter::new(vec![
            failing_runnable("error1"),
            failing_runnable("error2"),
            label_runnable("third"),
        ]);

        let result = router.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "third");
    }

    // ── ConditionalBranch tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_conditional_branch_first_when_matches() {
        let branch = ConditionalBranch::new()
            .when(
                Box::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(1)),
                label_runnable("one"),
            )
            .when(
                Box::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(2)),
                label_runnable("two"),
            )
            .otherwise(label_runnable("other"));

        let result = branch.invoke(json!({"x": 1}), None).await.unwrap();
        assert_eq!(result["from"], "one");
    }

    #[tokio::test]
    async fn test_conditional_branch_second_when_matches() {
        let branch = ConditionalBranch::new()
            .when(
                Box::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(1)),
                label_runnable("one"),
            )
            .when(
                Box::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(2)),
                label_runnable("two"),
            )
            .otherwise(label_runnable("other"));

        let result = branch.invoke(json!({"x": 2}), None).await.unwrap();
        assert_eq!(result["from"], "two");
    }

    #[tokio::test]
    async fn test_conditional_branch_otherwise() {
        let branch = ConditionalBranch::new()
            .when(
                Box::new(|v: &Value| v.get("x").and_then(|x| x.as_i64()) == Some(1)),
                label_runnable("one"),
            )
            .otherwise(label_runnable("fallback"));

        let result = branch.invoke(json!({"x": 99}), None).await.unwrap();
        assert_eq!(result["from"], "fallback");
    }

    #[tokio::test]
    async fn test_conditional_branch_no_match_no_otherwise() {
        let branch =
            ConditionalBranch::new().when(Box::new(|_: &Value| false), label_runnable("never"));

        let err = branch.invoke(json!("test"), None).await.unwrap_err();
        assert!(format!("{}", err).contains("No condition matched"));
    }

    #[tokio::test]
    async fn test_conditional_branch_empty_branches_with_otherwise() {
        let branch = ConditionalBranch::new().otherwise(label_runnable("only"));

        let result = branch.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "only");
    }

    #[tokio::test]
    async fn test_conditional_branch_name() {
        let branch = ConditionalBranch::new();
        assert_eq!(branch.name(), "ConditionalBranch");
    }

    #[tokio::test]
    async fn test_conditional_branch_first_match_wins() {
        // Both conditions are true, first should win
        let branch = ConditionalBranch::new()
            .when(Box::new(|_: &Value| true), label_runnable("first"))
            .when(Box::new(|_: &Value| true), label_runnable("second"));

        let result = branch.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "first");
    }

    // ── RoutingTable tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_routing_table_add_and_get() {
        let mut table = RoutingTable::new();
        table.add("alpha", label_runnable("alpha_handler"));
        table.add("beta", label_runnable("beta_handler"));

        assert!(table.get("alpha").is_some());
        assert!(table.get("beta").is_some());
        assert!(table.get("gamma").is_none());
    }

    #[tokio::test]
    async fn test_routing_table_names() {
        let mut table = RoutingTable::new();
        table.add("a", label_runnable("a_handler"));
        table.add("b", label_runnable("b_handler"));
        table.add("c", label_runnable("c_handler"));

        let names = table.names();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn test_routing_table_len_and_is_empty() {
        let mut table = RoutingTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);

        table.add("x", label_runnable("x_handler"));
        assert!(!table.is_empty());
        assert_eq!(table.len(), 1);

        table.add("y", label_runnable("y_handler"));
        assert_eq!(table.len(), 2);
    }

    #[tokio::test]
    async fn test_routing_table_replace_existing() {
        let mut table = RoutingTable::new();
        table.add("route", label_runnable("old_handler"));
        table.add("route", label_runnable("new_handler"));

        assert_eq!(table.len(), 1);
        let target = table.get("route").unwrap();
        assert_eq!(target.name(), "new_handler");
    }

    #[tokio::test]
    async fn test_routing_table_to_json() {
        let mut table = RoutingTable::new();
        table.add("first", label_runnable("handler_a"));
        table.add("second", label_runnable("handler_b"));

        let json_val = table.to_json();
        let routes = json_val["routes"].as_array().unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0]["name"], "first");
        assert_eq!(routes[0]["runnable"], "handler_a");
        assert_eq!(routes[1]["name"], "second");
        assert_eq!(routes[1]["runnable"], "handler_b");
    }

    #[tokio::test]
    async fn test_routing_table_empty_to_json() {
        let table = RoutingTable::new();
        let json_val = table.to_json();
        let routes = json_val["routes"].as_array().unwrap();
        assert!(routes.is_empty());
    }

    #[tokio::test]
    async fn test_routing_table_get_returns_usable_runnable() {
        let mut table = RoutingTable::new();
        table.add("test", label_runnable("test_handler"));

        let target = table.get("test").unwrap();
        let result = target.invoke(json!("input"), None).await.unwrap();
        assert_eq!(result["from"], "test_handler");
    }
}
