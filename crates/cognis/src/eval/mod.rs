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

use cognis_core::{Result, Runnable, RunnableConfig};

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

    /// Pass rate at `threshold` in `[0.0, 1.0]`. `0.0` for empty.
    pub fn pass_rate(&self, threshold: f32) -> f32 {
        if self.rows.is_empty() {
            return 0.0;
        }
        self.passing(threshold) as f32 / self.rows.len() as f32
    }

    /// Lowest scoring case (or `None` for empty).
    pub fn worst(&self) -> Option<&EvalRow<O>> {
        self.rows.iter().min_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Highest scoring case (or `None` for empty).
    pub fn best(&self) -> Option<&EvalRow<O>> {
        self.rows.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Total case count.
    pub fn total(&self) -> usize {
        self.rows.len()
    }

    /// Render a plain-text summary suitable for stdout / logs:
    ///
    /// ```text
    /// case        score   status
    /// hello       1.00    PASS
    /// world       0.50    FAIL
    /// (unnamed)   1.00    PASS
    /// ───
    /// total: 3   pass: 2 (66.7%)   mean: 0.83
    /// ```
    ///
    /// `pass_threshold` is the floor for the PASS / FAIL column. Pass
    /// `0.5` for binary 0/1 evaluators, `0.8` for tighter cutoffs, etc.
    pub fn summary(&self, pass_threshold: f32) -> String {
        let mut out = String::new();
        if self.rows.is_empty() {
            out.push_str("(empty report)\n");
            return out;
        }
        let name_width = self
            .rows
            .iter()
            .map(|r| r.name.as_deref().unwrap_or("(unnamed)").len())
            .max()
            .unwrap_or(8)
            .max(4);
        out.push_str(&format!(
            "{:<width$}  {:>6}  {}\n",
            "case",
            "score",
            "status",
            width = name_width
        ));
        for r in &self.rows {
            let status = if r.score >= pass_threshold {
                "PASS"
            } else {
                "FAIL"
            };
            let name = r.name.as_deref().unwrap_or("(unnamed)");
            out.push_str(&format!(
                "{:<width$}  {:>6.2}  {}\n",
                name,
                r.score,
                status,
                width = name_width
            ));
        }
        let pass = self.passing(pass_threshold);
        let total = self.total();
        let pass_pct = self.pass_rate(pass_threshold) * 100.0;
        out.push_str("───\n");
        out.push_str(&format!(
            "total: {total}   pass: {pass} ({pass_pct:.1}%)   mean: {:.2}\n",
            self.mean()
        ));
        out
    }

    /// Markdown table summary. Same data as [`EvalReport::summary`] but
    /// renders as a GitHub-flavored markdown table for PR comments / docs.
    pub fn markdown(&self, pass_threshold: f32) -> String {
        let mut out = String::new();
        if self.rows.is_empty() {
            out.push_str("_(empty report)_\n");
            return out;
        }
        out.push_str("| case | score | status |\n");
        out.push_str("|------|------:|--------|\n");
        for r in &self.rows {
            let status = if r.score >= pass_threshold {
                "PASS"
            } else {
                "FAIL"
            };
            let name = r.name.as_deref().unwrap_or("(unnamed)");
            out.push_str(&format!("| {name} | {:.2} | {status} |\n", r.score));
        }
        out.push_str(&format!(
            "\n**total**: {} · **pass**: {} ({:.1}%) · **mean**: {:.2}\n",
            self.total(),
            self.passing(pass_threshold),
            self.pass_rate(pass_threshold) * 100.0,
            self.mean()
        ));
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    fn report_with(rows: Vec<(Option<&str>, f32)>) -> EvalReport<String> {
        EvalReport {
            rows: rows
                .into_iter()
                .map(|(n, s)| EvalRow {
                    name: n.map(str::to_string),
                    score: s,
                    actual: String::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn pass_rate_and_passing() {
        let r = report_with(vec![(Some("a"), 1.0), (Some("b"), 0.4), (Some("c"), 0.9)]);
        assert_eq!(r.passing(0.5), 2);
        assert!((r.pass_rate(0.5) - 0.6666).abs() < 0.01);
    }

    #[test]
    fn best_and_worst() {
        let r = report_with(vec![(Some("a"), 0.3), (Some("b"), 0.9), (Some("c"), 0.5)]);
        assert_eq!(r.best().unwrap().name.as_deref(), Some("b"));
        assert_eq!(r.worst().unwrap().name.as_deref(), Some("a"));
    }

    #[test]
    fn empty_report_extremes() {
        let r: EvalReport<String> = EvalReport { rows: vec![] };
        assert!(r.best().is_none());
        assert!(r.worst().is_none());
        assert_eq!(r.pass_rate(0.5), 0.0);
        assert!(r.summary(0.5).contains("empty"));
    }

    #[test]
    fn summary_renders_pass_fail_and_aggregates() {
        let r = report_with(vec![(Some("hello"), 1.0), (Some("world"), 0.4)]);
        let s = r.summary(0.5);
        assert!(s.contains("hello"));
        assert!(s.contains("PASS"));
        assert!(s.contains("FAIL"));
        assert!(s.contains("total: 2"));
        assert!(s.contains("pass: 1"));
    }

    #[test]
    fn markdown_renders_table() {
        let r = report_with(vec![(Some("hello"), 1.0)]);
        let md = r.markdown(0.5);
        assert!(md.contains("| case |"));
        assert!(md.contains("| hello | 1.00 | PASS |"));
        assert!(md.contains("**mean**"));
    }
}
