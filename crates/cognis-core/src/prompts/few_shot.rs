//! Few-shot string prompt: prefix + per-example template + suffix.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::Serialize;

use crate::prompts::template::{render, scan_variables};
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Few-shot string prompt with a static list of examples.
///
/// Renders as:
/// `<prefix><sep><ex1><sep><ex2><sep>...<sep><suffix>`
///
/// The prefix and suffix are rendered against the call input. Each example
/// is rendered against itself (separate `Serialize` type).
#[derive(Debug, Clone)]
pub struct FewShotTemplate<I = serde_json::Value, E = serde_json::Value> {
    prefix: String,
    example_template: String,
    examples: Vec<E>,
    suffix: String,
    separator: String,
    _input: PhantomData<fn() -> I>,
}

impl<I, E> FewShotTemplate<I, E>
where
    I: Serialize + Send + Sync + 'static,
    E: Serialize + Send + Sync + Clone + 'static,
{
    /// Build a few-shot template.
    pub fn new(
        prefix: impl Into<String>,
        example_template: impl Into<String>,
        examples: Vec<E>,
        suffix: impl Into<String>,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            example_template: example_template.into(),
            examples,
            suffix: suffix.into(),
            separator: "\n\n".into(),
            _input: PhantomData,
        }
    }

    /// Override the separator used between rendered examples (default `\n\n`).
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Render the full prompt against the input.
    pub fn render(&self, input: &I) -> Result<String> {
        let input_ctx =
            serde_json::to_value(input).map_err(|e| CognisError::Serialization(e.to_string()))?;
        let mut rendered_examples = Vec::with_capacity(self.examples.len());
        for ex in &self.examples {
            let ex_ctx =
                serde_json::to_value(ex).map_err(|e| CognisError::Serialization(e.to_string()))?;
            rendered_examples.push(render(&self.example_template, &ex_ctx)?);
        }
        let prefix = render(&self.prefix, &input_ctx)?;
        let suffix = render(&self.suffix, &input_ctx)?;
        let body = rendered_examples.join(&self.separator);
        Ok(format!(
            "{prefix}{sep}{body}{sep}{suffix}",
            sep = self.separator
        ))
    }

    /// Variables referenced by the prefix and suffix.
    pub fn input_variables(&self) -> Vec<String> {
        let mut out = scan_variables(&self.prefix);
        for v in scan_variables(&self.suffix) {
            if !out.contains(&v) {
                out.push(v);
            }
        }
        out
    }
}

#[async_trait]
impl<I, E> Runnable<I, String> for FewShotTemplate<I, E>
where
    I: Serialize + Send + Sync + 'static,
    E: Serialize + Send + Sync + Clone + 'static,
{
    async fn invoke(&self, input: I, _: RunnableConfig) -> Result<String> {
        self.render(&input)
    }
    fn name(&self) -> &str {
        "FewShotTemplate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn renders_prefix_examples_suffix() {
        let examples = vec![json!({"q": "2+2", "a": "4"}), json!({"q": "3+3", "a": "6"})];
        let p: FewShotTemplate<Value, Value> = FewShotTemplate::new(
            "Math problems for {topic}:",
            "Q: {q}\nA: {a}",
            examples,
            "Q: {question}\nA:",
        );
        let out = p
            .render(&json!({"topic": "addition", "question": "5+5"}))
            .unwrap();
        assert!(out.starts_with("Math problems for addition:"));
        assert!(out.contains("Q: 2+2\nA: 4"));
        assert!(out.contains("Q: 3+3\nA: 6"));
        assert!(out.ends_with("Q: 5+5\nA:"));
    }

    #[test]
    fn separator_override() {
        let p: FewShotTemplate<Value, Value> =
            FewShotTemplate::new("P", "{x}", vec![json!({"x": "a"}), json!({"x": "b"})], "S")
                .with_separator(" | ");
        let out = p.render(&json!({})).unwrap();
        assert_eq!(out, "P | a | b | S");
    }
}
