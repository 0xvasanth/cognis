//! Lightweight evaluation harness for any `Runnable<I, O>`.
//!
//! Three pieces:
//! - [`EvalCase<I, O>`] — one input plus an expected reference.
//! - [`Evaluator<O>`] — produces a score in `[0.0, 1.0]` for an actual `O`
//!   given a reference `O`.
//! - [`EvalRunner`] — runs every case through a runnable, scores each
//!   actual output, returns an [`EvalReport`].
//!
//! Built-in evaluators:
//! - [`ExactMatch`] — `1.0` iff `actual == expected`.
//! - [`Contains`] — `1.0` iff `actual` (string) contains `expected`.
//! - [`LlmJudge`] — asks a `Client` to score; useful when "exact match"
//!   isn't appropriate but you still want a numeric signal.

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use cognis2_core::{Result, Runnable, RunnableConfig};

pub mod evaluators;
pub use evaluators::{Contains, ExactMatch, LlmJudge};

/// One eval input + reference output.
#[derive(Debug, Clone)]
pub struct EvalCase<I, O> {
    /// Optional human-readable name for reports.
    pub name: Option<String>,
    /// Input passed to the runnable under test.
    pub input: I,
    /// Reference output the evaluator scores against.
    pub expected: O,
}

impl<I, O> EvalCase<I, O> {
    /// Build an unnamed case.
    pub fn new(input: I, expected: O) -> Self {
        Self {
            name: None,
            input,
            expected,
        }
    }

    /// Builder: name the case.
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }
}

/// Scores `actual` against `expected`. `0.0` = wrong, `1.0` = perfect.
#[async_trait]
pub trait Evaluator<O>: Send + Sync {
    /// Score a single result.
    async fn score(&self, actual: &O, expected: &O) -> Result<f32>;
}

/// One row of an [`EvalReport`].
#[derive(Debug, Clone)]
pub struct EvalRow<O> {
    /// Name (if any) of the case.
    pub name: Option<String>,
    /// Score in `[0.0, 1.0]`.
    pub score: f32,
    /// The actual output produced.
    pub actual: O,
}

/// Aggregated evaluation report.
#[derive(Debug, Clone)]
pub struct EvalReport<O> {
    /// Per-case rows in original case order.
    pub rows: Vec<EvalRow<O>>,
}

impl<O> EvalReport<O> {
    /// Mean score across cases. Returns `0.0` for an empty report.
    pub fn mean(&self) -> f32 {
        if self.rows.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.rows.iter().map(|r| r.score).sum();
        sum / self.rows.len() as f32
    }

    /// Number of cases scoring `>= threshold`.
    pub fn passing(&self, threshold: f32) -> usize {
        self.rows.iter().filter(|r| r.score >= threshold).count()
    }

    /// Total case count.
    pub fn total(&self) -> usize {
        self.rows.len()
    }
}

/// Drives a [`Runnable`] over a set of cases and scores each output.
pub struct EvalRunner<I, O> {
    runnable: Arc<dyn Runnable<I, O>>,
    evaluator: Arc<dyn Evaluator<O>>,
    cases: Vec<EvalCase<I, O>>,
    concurrency: usize,
}

impl<I, O> EvalRunner<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Sync + Clone + 'static,
{
    /// Build a runner.
    pub fn new(
        runnable: Arc<dyn Runnable<I, O>>,
        evaluator: Arc<dyn Evaluator<O>>,
        cases: Vec<EvalCase<I, O>>,
    ) -> Self {
        Self {
            runnable,
            evaluator,
            cases,
            concurrency: 4,
        }
    }

    /// Set max concurrent case evaluations.
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Run every case and produce a report. Concurrency-bounded.
    pub async fn run(&self) -> Result<EvalReport<O>> {
        // Two passes: invoke under concurrency cap, then score.
        let invoke_futs = self.cases.iter().map(|c| {
            let r = self.runnable.clone();
            let i = c.input.clone();
            async move { r.invoke(i, RunnableConfig::default()).await }
        });
        // join_all without manual concurrency tuning — eval cases are
        // expected to be smallish; if you need a tighter bound, lower
        // `concurrency` and we'll honor it via buffered streaming.
        let actuals: Vec<O> = if self.concurrency >= self.cases.len() {
            join_all(invoke_futs)
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?
        } else {
            use futures::stream::{self, StreamExt};
            stream::iter(invoke_futs)
                .buffered(self.concurrency)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?
        };

        let mut rows = Vec::with_capacity(actuals.len());
        for (case, actual) in self.cases.iter().zip(actuals) {
            let score = self.evaluator.score(&actual, &case.expected).await?;
            rows.push(EvalRow {
                name: case.name.clone(),
                score,
                actual,
            });
        }
        Ok(EvalReport { rows })
    }
}
