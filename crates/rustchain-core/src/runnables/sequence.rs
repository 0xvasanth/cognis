use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::{ensure_config, RunnableConfig};
use super::RunnableStream;

/// Chains multiple runnables in sequence, piping output to input.
pub struct RunnableSequence {
    name: Option<String>,
    steps: Vec<Arc<dyn Runnable>>,
}

impl RunnableSequence {
    /// Create a new sequence from a list of runnables.
    ///
    /// # Errors
    /// Returns an error if steps is empty.
    pub fn new(steps: Vec<Arc<dyn Runnable>>) -> Result<Self> {
        if steps.is_empty() {
            return Err(RustChainError::Other(
                "RunnableSequence requires at least one step".into(),
            ));
        }
        Ok(Self { name: None, steps })
    }

    /// Set a custom name for this sequence.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

#[async_trait]
impl Runnable for RunnableSequence {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("RunnableSequence")
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let mut cfg = ensure_config(config);
        if cfg.recursion_limit == 0 {
            return Err(RustChainError::RecursionLimitExceeded(
                "Recursion limit reached in RunnableSequence".into(),
            ));
        }
        cfg.recursion_limit = cfg.recursion_limit.saturating_sub(1);

        let mut current = input;
        for step in &self.steps {
            current = step.invoke(current, Some(&cfg)).await?;
        }
        Ok(current)
    }

    async fn batch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<Value>> {
        let mut cfg = ensure_config(config);
        if cfg.recursion_limit == 0 {
            return Err(RustChainError::RecursionLimitExceeded(
                "Recursion limit reached in RunnableSequence batch".into(),
            ));
        }
        cfg.recursion_limit = cfg.recursion_limit.saturating_sub(1);

        let mut current_batch = inputs;
        for step in &self.steps {
            current_batch = step.batch(current_batch, Some(&cfg)).await?;
        }
        Ok(current_batch)
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let mut cfg = ensure_config(config);
        if cfg.recursion_limit == 0 {
            return Err(RustChainError::RecursionLimitExceeded(
                "Recursion limit reached in RunnableSequence stream".into(),
            ));
        }
        cfg.recursion_limit = cfg.recursion_limit.saturating_sub(1);

        if self.steps.len() == 1 {
            return self.steps[0].stream(input, Some(&cfg)).await;
        }

        // Invoke all steps except the last, then stream the last step.
        let mut current = input;
        for step in &self.steps[..self.steps.len() - 1] {
            current = step.invoke(current, Some(&cfg)).await?;
        }

        let last = &self.steps[self.steps.len() - 1];
        last.stream(current, Some(&cfg)).await
    }
}
