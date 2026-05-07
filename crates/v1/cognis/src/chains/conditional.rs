use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::Result;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// Condition trait
// ---------------------------------------------------------------------------

/// A condition that evaluates an input `Value` to a boolean.
pub trait Condition: Send + Sync {
    /// Evaluate the condition against the given input.
    fn evaluate(&self, input: &Value) -> Result<bool>;
}

// ---------------------------------------------------------------------------
// ClosureCondition
// ---------------------------------------------------------------------------

/// A [`Condition`] backed by a boxed closure.
pub struct ClosureCondition {
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&Value) -> Result<bool> + Send + Sync>,
}

impl ClosureCondition {
    /// Create a new `ClosureCondition` from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Value) -> Result<bool> + Send + Sync + 'static,
    {
        Self { func: Box::new(f) }
    }
}

impl Condition for ClosureCondition {
    fn evaluate(&self, input: &Value) -> Result<bool> {
        (self.func)(input)
    }
}

// ---------------------------------------------------------------------------
// KeyExistsCondition
// ---------------------------------------------------------------------------

/// A [`Condition`] that checks whether a key exists in the input object.
pub struct KeyExistsCondition {
    key: String,
}

impl KeyExistsCondition {
    /// Create a new `KeyExistsCondition`.
    pub fn new(key: impl Into<String>) -> Self {
        Self { key: key.into() }
    }
}

impl Condition for KeyExistsCondition {
    fn evaluate(&self, input: &Value) -> Result<bool> {
        match input {
            Value::Object(map) => Ok(map.contains_key(&self.key)),
            _ => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// KeyEqualsCondition
// ---------------------------------------------------------------------------

/// A [`Condition`] that checks whether `input[key] == expected`.
pub struct KeyEqualsCondition {
    key: String,
    expected: Value,
}

impl KeyEqualsCondition {
    /// Create a new `KeyEqualsCondition`.
    pub fn new(key: impl Into<String>, expected: Value) -> Self {
        Self {
            key: key.into(),
            expected,
        }
    }
}

impl Condition for KeyEqualsCondition {
    fn evaluate(&self, input: &Value) -> Result<bool> {
        match input.get(&self.key) {
            Some(val) => Ok(val == &self.expected),
            None => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// KeyContainsCondition
// ---------------------------------------------------------------------------

/// A [`Condition`] that checks whether `input[key]` (as a string) contains a substring.
pub struct KeyContainsCondition {
    key: String,
    substring: String,
}

impl KeyContainsCondition {
    /// Create a new `KeyContainsCondition`.
    pub fn new(key: impl Into<String>, substring: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            substring: substring.into(),
        }
    }
}

impl Condition for KeyContainsCondition {
    fn evaluate(&self, input: &Value) -> Result<bool> {
        match input.get(&self.key) {
            Some(Value::String(s)) => Ok(s.contains(&self.substring)),
            _ => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// ConditionalChain
// ---------------------------------------------------------------------------

/// A chain that routes execution based on a single boolean condition.
///
/// - If the condition evaluates to `true`, runs the `if_true` branch.
/// - If the condition evaluates to `false` and an `if_false` branch exists, runs it.
/// - Otherwise, passes through the input unchanged.
pub struct ConditionalChain {
    condition: Box<dyn Condition>,
    if_true: Arc<dyn Runnable>,
    if_false: Option<Arc<dyn Runnable>>,
}

/// Builder for [`ConditionalChain`].
pub struct ConditionalChainBuilder {
    condition: Box<dyn Condition>,
    if_true: Option<Arc<dyn Runnable>>,
    if_false: Option<Arc<dyn Runnable>>,
}

impl ConditionalChain {
    /// Start building a `ConditionalChain` with the given condition.
    pub fn builder(condition: impl Condition + 'static) -> ConditionalChainBuilder {
        ConditionalChainBuilder {
            condition: Box::new(condition),
            if_true: None,
            if_false: None,
        }
    }
}

impl ConditionalChainBuilder {
    /// Set the runnable to execute when the condition is true.
    pub fn then(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.if_true = Some(runnable);
        self
    }

    /// Set the runnable to execute when the condition is false.
    pub fn otherwise(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.if_false = Some(runnable);
        self
    }

    /// Build the `ConditionalChain`.
    ///
    /// # Panics
    /// Panics if `then` was not called.
    pub fn build(self) -> ConditionalChain {
        ConditionalChain {
            condition: self.condition,
            if_true: self
                .if_true
                .expect("ConditionalChain requires a `then` branch"),
            if_false: self.if_false,
        }
    }
}

#[async_trait]
impl Runnable for ConditionalChain {
    fn name(&self) -> &str {
        "ConditionalChain"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        if self.condition.evaluate(&input)? {
            self.if_true.invoke(input, config).await
        } else if let Some(ref if_false) = self.if_false {
            if_false.invoke(input, config).await
        } else {
            Ok(input)
        }
    }
}

// ---------------------------------------------------------------------------
// BranchChain
// ---------------------------------------------------------------------------

/// A chain that evaluates multiple conditions in order and runs the first matching branch.
///
/// - Evaluates conditions in insertion order.
/// - Runs the runnable associated with the first condition that returns `true`.
/// - If no condition matches and a default is set, runs the default.
/// - If no condition matches and no default, passes through the input.
pub struct BranchChain {
    branches: Vec<(Box<dyn Condition>, Arc<dyn Runnable>)>,
    default: Option<Arc<dyn Runnable>>,
}

/// Builder for [`BranchChain`].
pub struct BranchChainBuilder {
    branches: Vec<(Box<dyn Condition>, Arc<dyn Runnable>)>,
    default: Option<Arc<dyn Runnable>>,
}

impl BranchChain {
    /// Start building a `BranchChain`.
    pub fn builder() -> BranchChainBuilder {
        BranchChainBuilder {
            branches: Vec::new(),
            default: None,
        }
    }
}

impl BranchChainBuilder {
    /// Add a branch with a condition and corresponding runnable.
    pub fn branch(
        mut self,
        condition: impl Condition + 'static,
        runnable: Arc<dyn Runnable>,
    ) -> Self {
        self.branches.push((Box::new(condition), runnable));
        self
    }

    /// Set the default runnable to execute when no condition matches.
    pub fn default(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.default = Some(runnable);
        self
    }

    /// Build the `BranchChain`.
    pub fn build(self) -> BranchChain {
        BranchChain {
            branches: self.branches,
            default: self.default,
        }
    }
}

#[async_trait]
impl Runnable for BranchChain {
    fn name(&self) -> &str {
        "BranchChain"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        for (condition, runnable) in &self.branches {
            if condition.evaluate(&input)? {
                return runnable.invoke(input, config).await;
            }
        }
        if let Some(ref default) = self.default {
            default.invoke(input, config).await
        } else {
            Ok(input)
        }
    }
}

// ---------------------------------------------------------------------------
// SwitchChain
// ---------------------------------------------------------------------------

/// A chain that routes execution based on the string value of `input[key]`.
///
/// - Looks up `input[key]` as a string.
/// - If a matching case exists, runs the associated runnable.
/// - If no match and a default is set, runs the default.
/// - If no match and no default, passes through the input.
pub struct SwitchChain {
    key: String,
    cases: HashMap<String, Arc<dyn Runnable>>,
    default: Option<Arc<dyn Runnable>>,
}

/// Builder for [`SwitchChain`].
pub struct SwitchChainBuilder {
    key: String,
    cases: HashMap<String, Arc<dyn Runnable>>,
    default: Option<Arc<dyn Runnable>>,
}

impl SwitchChain {
    /// Start building a `SwitchChain` that routes on the given key.
    pub fn builder(key: impl Into<String>) -> SwitchChainBuilder {
        SwitchChainBuilder {
            key: key.into(),
            cases: HashMap::new(),
            default: None,
        }
    }
}

impl SwitchChainBuilder {
    /// Add a case mapping a string value to a runnable.
    pub fn case(mut self, value: impl Into<String>, runnable: Arc<dyn Runnable>) -> Self {
        self.cases.insert(value.into(), runnable);
        self
    }

    /// Set the default runnable for when no case matches.
    pub fn default(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.default = Some(runnable);
        self
    }

    /// Build the `SwitchChain`.
    pub fn build(self) -> SwitchChain {
        SwitchChain {
            key: self.key,
            cases: self.cases,
            default: self.default,
        }
    }
}

#[async_trait]
impl Runnable for SwitchChain {
    fn name(&self) -> &str {
        "SwitchChain"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let case_value = input
            .get(&self.key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(ref case_str) = case_value {
            if let Some(runnable) = self.cases.get(case_str) {
                return runnable.invoke(input, config).await;
            }
        }

        if let Some(ref default) = self.default {
            default.invoke(input, config).await
        } else {
            Ok(input)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::error::CognisError;
    use cognis_core::runnables::RunnableLambda;
    use serde_json::json;

    // -- Helpers --

    fn upper_lambda() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("upper", |v: Value| async move {
            let s = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
            Ok(json!({ "text": s.to_uppercase() }))
        }))
    }

    fn lower_lambda() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("lower", |v: Value| async move {
            let s = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
            Ok(json!({ "text": s.to_lowercase() }))
        }))
    }

    fn add_tag_lambda(tag: &'static str) -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("add_tag", move |v: Value| async move {
            let mut obj = v.as_object().cloned().unwrap_or_default();
            obj.insert("tag".to_string(), json!(tag));
            Ok(Value::Object(obj))
        }))
    }

    fn double_lambda() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("double", |v: Value| async move {
            let n = v.get("value").and_then(|x| x.as_i64()).unwrap_or(0);
            Ok(json!({ "value": n * 2 }))
        }))
    }

    fn negate_lambda() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("negate", |v: Value| async move {
            let n = v.get("value").and_then(|x| x.as_i64()).unwrap_or(0);
            Ok(json!({ "value": -n }))
        }))
    }

    // =======================================================================
    // Condition tests
    // =======================================================================

    #[test]
    fn test_closure_condition_true() {
        let cond = ClosureCondition::new(|v: &Value| {
            Ok(v.get("flag").and_then(|f| f.as_bool()).unwrap_or(false))
        });
        assert!(cond.evaluate(&json!({ "flag": true })).unwrap());
    }

    #[test]
    fn test_closure_condition_false() {
        let cond = ClosureCondition::new(|v: &Value| {
            Ok(v.get("flag").and_then(|f| f.as_bool()).unwrap_or(false))
        });
        assert!(!cond.evaluate(&json!({ "flag": false })).unwrap());
    }

    #[test]
    fn test_closure_condition_error() {
        let cond = ClosureCondition::new(|_v: &Value| Err(CognisError::Other("boom".into())));
        assert!(cond.evaluate(&json!({})).is_err());
    }

    #[test]
    fn test_key_exists_present() {
        let cond = KeyExistsCondition::new("name");
        assert!(cond.evaluate(&json!({ "name": "Alice" })).unwrap());
    }

    #[test]
    fn test_key_exists_absent() {
        let cond = KeyExistsCondition::new("name");
        assert!(!cond.evaluate(&json!({ "age": 30 })).unwrap());
    }

    #[test]
    fn test_key_exists_non_object() {
        let cond = KeyExistsCondition::new("x");
        assert!(!cond.evaluate(&json!("hello")).unwrap());
    }

    #[test]
    fn test_key_equals_match() {
        let cond = KeyEqualsCondition::new("status", json!("active"));
        assert!(cond.evaluate(&json!({ "status": "active" })).unwrap());
    }

    #[test]
    fn test_key_equals_no_match() {
        let cond = KeyEqualsCondition::new("status", json!("active"));
        assert!(!cond.evaluate(&json!({ "status": "inactive" })).unwrap());
    }

    #[test]
    fn test_key_equals_missing_key() {
        let cond = KeyEqualsCondition::new("status", json!("active"));
        assert!(!cond.evaluate(&json!({ "other": "value" })).unwrap());
    }

    #[test]
    fn test_key_equals_numeric() {
        let cond = KeyEqualsCondition::new("count", json!(42));
        assert!(cond.evaluate(&json!({ "count": 42 })).unwrap());
        assert!(!cond.evaluate(&json!({ "count": 99 })).unwrap());
    }

    #[test]
    fn test_key_contains_match() {
        let cond = KeyContainsCondition::new("message", "hello");
        assert!(cond
            .evaluate(&json!({ "message": "say hello world" }))
            .unwrap());
    }

    #[test]
    fn test_key_contains_no_match() {
        let cond = KeyContainsCondition::new("message", "goodbye");
        assert!(!cond
            .evaluate(&json!({ "message": "say hello world" }))
            .unwrap());
    }

    #[test]
    fn test_key_contains_non_string_value() {
        let cond = KeyContainsCondition::new("count", "5");
        assert!(!cond.evaluate(&json!({ "count": 5 })).unwrap());
    }

    #[test]
    fn test_key_contains_missing_key() {
        let cond = KeyContainsCondition::new("message", "hi");
        assert!(!cond.evaluate(&json!({ "other": "hi" })).unwrap());
    }

    // =======================================================================
    // ConditionalChain tests
    // =======================================================================

    #[tokio::test]
    async fn test_conditional_true_branch() {
        let chain = ConditionalChain::builder(KeyEqualsCondition::new("mode", json!("upper")))
            .then(upper_lambda())
            .build();

        let result = chain
            .invoke(json!({ "mode": "upper", "text": "hello" }), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "HELLO");
    }

    #[tokio::test]
    async fn test_conditional_false_with_otherwise() {
        let chain = ConditionalChain::builder(KeyEqualsCondition::new("mode", json!("upper")))
            .then(upper_lambda())
            .otherwise(lower_lambda())
            .build();

        let result = chain
            .invoke(json!({ "mode": "lower", "text": "HELLO" }), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "hello");
    }

    #[tokio::test]
    async fn test_conditional_false_passthrough() {
        let chain = ConditionalChain::builder(KeyEqualsCondition::new("mode", json!("upper")))
            .then(upper_lambda())
            .build();

        let input = json!({ "mode": "other", "text": "hello" });
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_conditional_chain_name() {
        let chain = ConditionalChain::builder(KeyExistsCondition::new("x"))
            .then(upper_lambda())
            .build();
        assert_eq!(chain.name(), "ConditionalChain");
    }

    #[tokio::test]
    async fn test_conditional_with_closure_condition() {
        let chain = ConditionalChain::builder(ClosureCondition::new(|v: &Value| {
            Ok(v.get("value").and_then(|x| x.as_i64()).unwrap_or(0) > 10)
        }))
        .then(double_lambda())
        .otherwise(negate_lambda())
        .build();

        let result = chain.invoke(json!({ "value": 20 }), None).await.unwrap();
        assert_eq!(result["value"], 40);

        let result = chain.invoke(json!({ "value": 5 }), None).await.unwrap();
        assert_eq!(result["value"], -5);
    }

    // =======================================================================
    // BranchChain tests
    // =======================================================================

    #[tokio::test]
    async fn test_branch_first_match() {
        let chain = BranchChain::builder()
            .branch(
                KeyEqualsCondition::new("type", json!("a")),
                add_tag_lambda("matched_a"),
            )
            .branch(
                KeyEqualsCondition::new("type", json!("b")),
                add_tag_lambda("matched_b"),
            )
            .build();

        let result = chain.invoke(json!({ "type": "a" }), None).await.unwrap();
        assert_eq!(result["tag"], "matched_a");
    }

    #[tokio::test]
    async fn test_branch_second_match() {
        let chain = BranchChain::builder()
            .branch(
                KeyEqualsCondition::new("type", json!("a")),
                add_tag_lambda("matched_a"),
            )
            .branch(
                KeyEqualsCondition::new("type", json!("b")),
                add_tag_lambda("matched_b"),
            )
            .build();

        let result = chain.invoke(json!({ "type": "b" }), None).await.unwrap();
        assert_eq!(result["tag"], "matched_b");
    }

    #[tokio::test]
    async fn test_branch_no_match_with_default() {
        let chain = BranchChain::builder()
            .branch(
                KeyEqualsCondition::new("type", json!("a")),
                add_tag_lambda("matched_a"),
            )
            .default(add_tag_lambda("default"))
            .build();

        let result = chain.invoke(json!({ "type": "z" }), None).await.unwrap();
        assert_eq!(result["tag"], "default");
    }

    #[tokio::test]
    async fn test_branch_no_match_passthrough() {
        let chain = BranchChain::builder()
            .branch(
                KeyEqualsCondition::new("type", json!("a")),
                add_tag_lambda("matched_a"),
            )
            .build();

        let input = json!({ "type": "z" });
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_branch_chain_name() {
        let chain = BranchChain::builder()
            .branch(KeyExistsCondition::new("x"), upper_lambda())
            .build();
        assert_eq!(chain.name(), "BranchChain");
    }

    #[tokio::test]
    async fn test_branch_first_wins_when_multiple_match() {
        let chain = BranchChain::builder()
            .branch(KeyExistsCondition::new("x"), add_tag_lambda("first"))
            .branch(KeyExistsCondition::new("x"), add_tag_lambda("second"))
            .build();

        let result = chain.invoke(json!({ "x": 1 }), None).await.unwrap();
        assert_eq!(result["tag"], "first");
    }

    #[tokio::test]
    async fn test_branch_empty_branches_passthrough() {
        let chain = BranchChain::builder().build();

        let input = json!({ "data": 42 });
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    // =======================================================================
    // SwitchChain tests
    // =======================================================================

    #[tokio::test]
    async fn test_switch_matching_case() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .case("python", add_tag_lambda("python_branch"))
            .build();

        let result = chain.invoke(json!({ "lang": "rust" }), None).await.unwrap();
        assert_eq!(result["tag"], "rust_branch");
    }

    #[tokio::test]
    async fn test_switch_second_case() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .case("python", add_tag_lambda("python_branch"))
            .build();

        let result = chain
            .invoke(json!({ "lang": "python" }), None)
            .await
            .unwrap();
        assert_eq!(result["tag"], "python_branch");
    }

    #[tokio::test]
    async fn test_switch_no_match_with_default() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .default(add_tag_lambda("unknown"))
            .build();

        let result = chain.invoke(json!({ "lang": "go" }), None).await.unwrap();
        assert_eq!(result["tag"], "unknown");
    }

    #[tokio::test]
    async fn test_switch_no_match_passthrough() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .build();

        let input = json!({ "lang": "go" });
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_switch_missing_key_passthrough() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .build();

        let input = json!({ "other": "value" });
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_switch_missing_key_with_default() {
        let chain = SwitchChain::builder("lang")
            .case("rust", add_tag_lambda("rust_branch"))
            .default(add_tag_lambda("fallback"))
            .build();

        let result = chain
            .invoke(json!({ "other": "value" }), None)
            .await
            .unwrap();
        assert_eq!(result["tag"], "fallback");
    }

    #[tokio::test]
    async fn test_switch_chain_name() {
        let chain = SwitchChain::builder("x").case("a", upper_lambda()).build();
        assert_eq!(chain.name(), "SwitchChain");
    }

    #[tokio::test]
    async fn test_switch_non_string_key_value() {
        // When the key exists but is not a string, treat it as no match
        let chain = SwitchChain::builder("code")
            .case("42", add_tag_lambda("matched"))
            .default(add_tag_lambda("default"))
            .build();

        let result = chain.invoke(json!({ "code": 42 }), None).await.unwrap();
        assert_eq!(result["tag"], "default");
    }

    // =======================================================================
    // Integration / cross-cutting tests
    // =======================================================================

    #[tokio::test]
    async fn test_conditional_with_key_contains() {
        let chain = ConditionalChain::builder(KeyContainsCondition::new("text", "urgent"))
            .then(upper_lambda())
            .otherwise(lower_lambda())
            .build();

        let result = chain
            .invoke(json!({ "text": "This is urgent!" }), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "THIS IS URGENT!");

        let result = chain
            .invoke(json!({ "text": "Normal Message" }), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "normal message");
    }

    #[tokio::test]
    async fn test_branch_with_mixed_conditions() {
        let chain = BranchChain::builder()
            .branch(
                KeyContainsCondition::new("text", "error"),
                add_tag_lambda("error_handler"),
            )
            .branch(
                KeyEqualsCondition::new("priority", json!("high")),
                add_tag_lambda("high_priority"),
            )
            .branch(
                KeyExistsCondition::new("debug"),
                add_tag_lambda("debug_mode"),
            )
            .default(add_tag_lambda("normal"))
            .build();

        let r = chain
            .invoke(json!({ "text": "an error occurred" }), None)
            .await
            .unwrap();
        assert_eq!(r["tag"], "error_handler");

        let r = chain
            .invoke(json!({ "text": "ok", "priority": "high" }), None)
            .await
            .unwrap();
        assert_eq!(r["tag"], "high_priority");

        let r = chain
            .invoke(json!({ "text": "ok", "debug": true }), None)
            .await
            .unwrap();
        assert_eq!(r["tag"], "debug_mode");

        let r = chain.invoke(json!({ "text": "ok" }), None).await.unwrap();
        assert_eq!(r["tag"], "normal");
    }

    #[tokio::test]
    async fn test_conditional_error_propagation() {
        let chain = ConditionalChain::builder(ClosureCondition::new(|_v: &Value| {
            Err(CognisError::Other("condition failed".into()))
        }))
        .then(upper_lambda())
        .build();

        let result = chain.invoke(json!({ "text": "hello" }), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("condition failed"));
    }

    #[tokio::test]
    async fn test_branch_error_propagation() {
        let chain = BranchChain::builder()
            .branch(
                ClosureCondition::new(|_v: &Value| Err(CognisError::Other("branch error".into()))),
                upper_lambda(),
            )
            .build();

        let result = chain.invoke(json!({ "text": "hello" }), None).await;
        assert!(result.is_err());
    }
}
