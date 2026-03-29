use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;

/// Chains two runnables in sequence, piping the first's output as the second's input.
///
/// This is the core building block for the LCEL pipe operator (`|`).
/// Unlike `RunnableSequence` which takes a flat `Vec` of steps, `RunnablePipe`
/// is a binary combinator that composes exactly two runnables, making it
/// natural for chaining: `a.pipe(b).pipe(c)`.
pub struct RunnablePipe {
    first: Arc<dyn Runnable>,
    second: Arc<dyn Runnable>,
}

impl RunnablePipe {
    /// Create a new pipe that chains `first` into `second`.
    pub fn new(first: Arc<dyn Runnable>, second: Arc<dyn Runnable>) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl Runnable for RunnablePipe {
    fn name(&self) -> &str {
        // We store a leaked string for the combined name so we can return &str.
        // This is acceptable because pipes are typically long-lived.
        // Instead, we'll use a simpler approach and return a static fallback,
        // with the formatted name available via `pipe_name()`.
        "RunnablePipe"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let intermediate = self.first.invoke(input, config).await?;
        self.second.invoke(intermediate, config).await
    }

    async fn batch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<Value>> {
        let intermediates = self.first.batch(inputs, config).await?;
        self.second.batch(intermediates, config).await
    }
}

impl RunnablePipe {
    /// Returns the formatted pipe name as `"first | second"`.
    pub fn pipe_name(&self) -> String {
        format!("{} | {}", self.first.name(), self.second.name())
    }
}

/// Creates an `Arc<dyn Runnable>` pipe from two `Arc<dyn Runnable>` values.
///
/// This is a free function alternative to the `RunnableExt::pipe` method,
/// useful when working with trait objects directly.
pub fn pipe(first: Arc<dyn Runnable>, second: Arc<dyn Runnable>) -> Arc<dyn Runnable> {
    Arc::new(RunnablePipe::new(first, second))
}

/// Ergonomic builder for chaining multiple runnables into a pipe sequence.
///
/// `PipeBuilder` collects steps and produces a single `Arc<dyn Runnable>`
/// that executes them in order. For efficiency, it flattens the steps into
/// a `RunnableSequence` rather than nesting `RunnablePipe`s.
///
/// # Example
/// ```ignore
/// use cognis_core::runnables::pipe::PipeBuilder;
///
/// let chain = PipeBuilder::new(step1)
///     .pipe(step2)
///     .pipe(step3)
///     .build();
/// ```
pub struct PipeBuilder {
    steps: Vec<Arc<dyn Runnable>>,
}

impl PipeBuilder {
    /// Start building a pipe chain with the first runnable.
    pub fn new(first: Arc<dyn Runnable>) -> Self {
        Self { steps: vec![first] }
    }

    /// Append a runnable to the chain.
    pub fn pipe(mut self, next: Arc<dyn Runnable>) -> Self {
        self.steps.push(next);
        self
    }

    /// Build the chain into a single `Arc<dyn Runnable>`.
    ///
    /// If only one step was provided, returns it directly.
    /// If two steps, returns a `RunnablePipe`.
    /// If three or more, returns a `RunnableSequence` for efficiency
    /// (avoids deeply nested pipes).
    pub fn build(self) -> Result<Arc<dyn Runnable>> {
        match self.steps.len() {
            0 => unreachable!("PipeBuilder always has at least one step"),
            1 => Ok(self.steps.into_iter().next().unwrap()),
            2 => {
                let mut iter = self.steps.into_iter();
                let first = iter.next().unwrap();
                let second = iter.next().unwrap();
                Ok(Arc::new(RunnablePipe::new(first, second)))
            }
            _ => {
                let seq = super::sequence::RunnableSequence::new(self.steps)?;
                Ok(Arc::new(seq))
            }
        }
    }

    /// Returns the formatted pipe name as `"step1 | step2 | step3"`.
    pub fn name(&self) -> String {
        self.steps
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// A newtype wrapper around `Arc<dyn Runnable>` that supports the `|` operator.
///
/// This enables LCEL-style composition:
/// ```ignore
/// let chain = prompt | model | parser;
/// ```
///
/// Internally collects steps and builds a single flat `RunnableSequence`
/// so that `a | b | c` produces `Seq([a, b, c])` rather than nested
/// `Seq([Seq([a, b]), c])`. This avoids consuming one recursion limit
/// level per `|` operator.
#[derive(Clone)]
pub struct RunnableRef {
    /// The composed runnable (built on first access or at `|`).
    pub(crate) inner: Arc<dyn Runnable>,
    /// Accumulated steps for flattening. Empty once finalized into `inner`.
    steps: Vec<Arc<dyn Runnable>>,
}

impl RunnableRef {
    /// Wrap any `Runnable` for use with the `|` operator.
    pub fn new(r: Arc<dyn Runnable>) -> Self {
        Self {
            inner: r.clone(),
            steps: vec![r],
        }
    }

    /// Unwrap into the inner `Arc<dyn Runnable>`.
    ///
    /// If multiple steps have been composed with `|`, this builds and
    /// returns a flat `RunnableSequence`.
    pub fn into_inner(self) -> Arc<dyn Runnable> {
        if self.steps.len() <= 1 {
            self.inner
        } else {
            self.build_sequence()
        }
    }

    /// Access the composed runnable.
    ///
    /// Builds a flat sequence from accumulated `|` steps if needed.
    pub fn runnable(&self) -> Arc<dyn Runnable> {
        if self.steps.len() <= 1 {
            self.inner.clone()
        } else {
            self.build_sequence()
        }
    }

    fn build_sequence(&self) -> Arc<dyn Runnable> {
        match crate::runnables::RunnableSequence::new(self.steps.clone()) {
            Ok(seq) => Arc::new(seq),
            Err(_) => self.inner.clone(),
        }
    }
}

impl std::ops::BitOr for RunnableRef {
    type Output = RunnableRef;

    fn bitor(self, rhs: RunnableRef) -> RunnableRef {
        // Collect all steps into a flat list for a single RunnableSequence.
        let mut steps = self.steps;
        steps.extend(rhs.steps);
        let inner = match crate::runnables::RunnableSequence::new(steps.clone()) {
            Ok(seq) => Arc::new(seq) as Arc<dyn Runnable>,
            Err(_) => unreachable!("pipe operator always provides at least 2 steps"),
        };
        RunnableRef { inner, steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{RunnableLambda, RunnableParallel};
    use serde_json::json;
    use std::collections::HashMap;

    fn add_one() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 1))
        }))
    }

    fn double() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("double", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        }))
    }

    fn to_string_runnable() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("to_string", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(format!("result:{}", n)))
        }))
    }

    fn parse_int() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("parse_int", |v: Value| async move {
            let s = v.as_str().unwrap();
            let n: i64 = s.parse().map_err(|e: std::num::ParseIntError| {
                crate::error::CognisError::Other(e.to_string())
            })?;
            Ok(json!(n))
        }))
    }

    fn failing_runnable() -> Arc<dyn Runnable> {
        Arc::new(RunnableLambda::new("fail", |_v: Value| async move {
            Err(crate::error::CognisError::Other(
                "intentional failure".into(),
            ))
        }))
    }

    // Test 1: Basic pipe chaining two lambdas
    #[tokio::test]
    async fn test_pipe_basic_two_lambdas() {
        let piped = RunnablePipe::new(add_one(), double());
        // (5 + 1) * 2 = 12
        let result = piped.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(12));
    }

    // Test 2: Triple chain a.pipe(b).pipe(c)
    #[tokio::test]
    async fn test_pipe_triple_chain() {
        let first = RunnablePipe::new(add_one(), double());
        let chained: Arc<dyn Runnable> = Arc::new(first);
        let piped = RunnablePipe::new(chained, add_one());
        // ((5 + 1) * 2) + 1 = 13
        let result = piped.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(13));
    }

    // Test 3: Pipe name formatting
    #[tokio::test]
    async fn test_pipe_name_formatting() {
        let piped = RunnablePipe::new(add_one(), double());
        assert_eq!(piped.pipe_name(), "add_one | double");
        assert_eq!(piped.name(), "RunnablePipe");
    }

    // Test 4: PipeBuilder ergonomic API
    #[tokio::test]
    async fn test_pipe_builder_multi_step() {
        let chain = PipeBuilder::new(add_one())
            .pipe(double())
            .pipe(add_one())
            .build()
            .unwrap();
        // ((3 + 1) * 2) + 1 = 9
        let result = chain.invoke(json!(3), None).await.unwrap();
        assert_eq!(result, json!(9));
    }

    // Test 5: PipeBuilder name
    #[tokio::test]
    async fn test_pipe_builder_name() {
        let builder = PipeBuilder::new(add_one())
            .pipe(double())
            .pipe(to_string_runnable());
        assert_eq!(builder.name(), "add_one | double | to_string");
    }

    // Test 6: RunnableParallel with multiple branches
    #[tokio::test]
    async fn test_parallel_multiple_branches() {
        let mut steps = HashMap::new();
        steps.insert("added".to_string(), add_one());
        steps.insert("doubled".to_string(), double());
        let parallel = RunnableParallel::new(steps);

        let result = parallel.invoke(json!(5), None).await.unwrap();
        assert_eq!(result["added"], json!(6));
        assert_eq!(result["doubled"], json!(10));
    }

    // Test 7: RunnableParallel with single branch
    #[tokio::test]
    async fn test_parallel_single_branch() {
        let mut steps = HashMap::new();
        steps.insert("only".to_string(), add_one());
        let parallel = RunnableParallel::new(steps);

        let result = parallel.invoke(json!(10), None).await.unwrap();
        assert_eq!(result["only"], json!(11));
    }

    // Test 8: Pipe + Parallel composition
    #[tokio::test]
    async fn test_pipe_with_parallel() {
        let mut steps = HashMap::new();
        steps.insert("added".to_string(), add_one());
        steps.insert("doubled".to_string(), double());
        let parallel: Arc<dyn Runnable> = Arc::new(RunnableParallel::new(steps));

        // Extract the "added" field
        let extract = Arc::new(RunnableLambda::new(
            "extract_added",
            |v: Value| async move { Ok(v["added"].clone()) },
        ));

        let chain = PipeBuilder::new(parallel).pipe(extract).build().unwrap();

        let result = chain.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(6));
    }

    // Test 9: Error propagation through pipe
    #[tokio::test]
    async fn test_pipe_error_propagation() {
        let piped = RunnablePipe::new(failing_runnable(), double());
        let result = piped.invoke(json!(5), None).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("intentional failure"));
    }

    // Test 10: Error propagation in second step
    #[tokio::test]
    async fn test_pipe_error_in_second_step() {
        let piped = RunnablePipe::new(add_one(), failing_runnable());
        let result = piped.invoke(json!(5), None).await;
        assert!(result.is_err());
    }

    // Test 11: Config passing through pipe
    #[tokio::test]
    async fn test_pipe_config_passing() {
        let config_reader = Arc::new(RunnableLambda::with_config(
            "config_reader",
            |v: Value, config: Option<RunnableConfig>| async move {
                let run_name = config
                    .and_then(|c| c.run_name)
                    .unwrap_or_else(|| "none".to_string());
                Ok(json!({
                    "input": v,
                    "run_name": run_name,
                }))
            },
        ));

        let piped = RunnablePipe::new(add_one(), config_reader);
        let mut cfg = RunnableConfig::default();
        cfg.run_name = Some("test_run".to_string());

        let result = piped.invoke(json!(5), Some(&cfg)).await.unwrap();
        assert_eq!(result["input"], json!(6));
        assert_eq!(result["run_name"], json!("test_run"));
    }

    // Test 12: Pipe with different value types (string -> number -> string)
    #[tokio::test]
    async fn test_pipe_type_transformation() {
        let chain = PipeBuilder::new(parse_int())
            .pipe(double())
            .pipe(to_string_runnable())
            .build()
            .unwrap();

        let result = chain.invoke(json!("7"), None).await.unwrap();
        assert_eq!(result, json!("result:14"));
    }

    // Test 13: PipeBuilder single step returns the step directly
    #[tokio::test]
    async fn test_pipe_builder_single_step() {
        let chain = PipeBuilder::new(add_one()).build().unwrap();
        let result = chain.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(6));
    }

    // Test 14: Batch through pipe
    #[tokio::test]
    async fn test_pipe_batch() {
        let piped = RunnablePipe::new(add_one(), double());
        let inputs = vec![json!(1), json!(2), json!(3)];
        let results = piped.batch(inputs, None).await.unwrap();
        assert_eq!(results, vec![json!(4), json!(6), json!(8)]);
    }
}

#[cfg(test)]
mod pipe_operator_tests {
    use super::*;
    use crate::runnables::lambda::RunnableLambda;
    use serde_json::{json, Value};

    #[tokio::test]
    async fn test_pipe_operator_two_steps() {
        let add_one = RunnableRef::new(Arc::new(RunnableLambda::new(
            "add_one",
            |v: Value| async move {
                let n = v.as_i64().unwrap_or(0);
                Ok(json!(n + 1))
            },
        )));
        let double = RunnableRef::new(Arc::new(RunnableLambda::new(
            "double",
            |v: Value| async move {
                let n = v.as_i64().unwrap_or(0);
                Ok(json!(n * 2))
            },
        )));

        let chain = add_one | double;
        let result = chain.runnable().invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(12)); // (5 + 1) * 2
    }

    #[tokio::test]
    async fn test_pipe_operator_three_steps() {
        let a = RunnableRef::new(Arc::new(RunnableLambda::new("a", |v: Value| async move {
            let n = v.as_i64().unwrap_or(0);
            Ok(json!(n + 1))
        })));
        let b = RunnableRef::new(Arc::new(RunnableLambda::new("b", |v: Value| async move {
            let n = v.as_i64().unwrap_or(0);
            Ok(json!(n * 2))
        })));
        let c = RunnableRef::new(Arc::new(RunnableLambda::new("c", |v: Value| async move {
            let n = v.as_i64().unwrap_or(0);
            Ok(json!(n - 3))
        })));

        let chain = a | b | c;
        let result = chain.runnable().invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(9)); // ((5 + 1) * 2) - 3
    }
}
