//! Built-in [`Evaluator`] implementations.

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatOptions;
use cognis_llm::Client;

use super::Evaluator;

/// Returns `1.0` iff `actual == expected`, else `0.0`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExactMatch;

#[async_trait]
impl<O> Evaluator<O> for ExactMatch
where
    O: PartialEq + Send + Sync + 'static,
{
    async fn score(&self, actual: &O, expected: &O) -> Result<f32> {
        Ok(if actual == expected { 1.0 } else { 0.0 })
    }
}

/// String-substring match: returns `1.0` iff `actual` contains `expected`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Contains;

#[async_trait]
impl Evaluator<String> for Contains {
    async fn score(&self, actual: &String, expected: &String) -> Result<f32> {
        Ok(if actual.contains(expected.as_str()) {
            1.0
        } else {
            0.0
        })
    }
}

/// LLM-as-a-judge evaluator. Asks the model to rate `actual` against
/// `expected` on a 0-10 scale; the score is normalized to `[0.0, 1.0]`.
pub struct LlmJudge {
    client: Client,
    prompt: String,
}

const DEFAULT_LLM_JUDGE_PROMPT: &str =
    "You are an evaluator. Rate how well ACTUAL satisfies EXPECTED on a \
     scale 0-10. Reply with ONLY a single integer.\n\n\
     EXPECTED:\n{expected}\n\nACTUAL:\n{actual}";

impl LlmJudge {
    /// Construct an LLM judge backed by `client`.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            prompt: DEFAULT_LLM_JUDGE_PROMPT.to_string(),
        }
    }

    /// Override the judge prompt. Use `{expected}` / `{actual}` placeholders.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }
}

#[async_trait]
impl Evaluator<String> for LlmJudge {
    async fn score(&self, actual: &String, expected: &String) -> Result<f32> {
        let prompt = self
            .prompt
            .replace("{expected}", expected)
            .replace("{actual}", actual);
        let resp = self
            .client
            .chat(vec![Message::human(prompt)], ChatOptions::default())
            .await?;
        let text = resp.message.content().trim().to_string();
        // Extract first int.
        let mut digits = String::new();
        for c in text.chars() {
            if c.is_ascii_digit() {
                digits.push(c);
            } else if !digits.is_empty() {
                break;
            }
        }
        let n: f32 = digits.parse().unwrap_or(0.0);
        Ok((n / 10.0).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis_core::{Result, Runnable, RunnableConfig};

    struct StaticOut(String);
    #[async_trait]
    impl Runnable<String, String> for StaticOut {
        async fn invoke(&self, _: String, _: RunnableConfig) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn exact_match_evaluator_scores() {
        let r: Arc<dyn Runnable<String, String>> = Arc::new(StaticOut("hello".into()));
        let runner = EvalRunner::new(
            r,
            Arc::new(ExactMatch) as Arc<dyn Evaluator<String>>,
            vec![
                EvalCase::new("a".into(), "hello".to_string()).with_name("match"),
                EvalCase::new("b".into(), "world".to_string()).with_name("miss"),
            ],
        );
        let report = runner.run().await.unwrap();
        assert_eq!(report.total(), 2);
        assert_eq!(report.passing(0.5), 1);
        assert!((report.mean() - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn contains_evaluator_partial_pass() {
        let r: Arc<dyn Runnable<String, String>> =
            Arc::new(StaticOut("the rust programming language".into()));
        let runner = EvalRunner::new(
            r,
            Arc::new(Contains) as Arc<dyn Evaluator<String>>,
            vec![
                EvalCase::new("q1".into(), "rust".into()),
                EvalCase::new("q2".into(), "python".into()),
            ],
        );
        let report = runner.run().await.unwrap();
        assert_eq!(report.passing(1.0), 1);
    }
}
