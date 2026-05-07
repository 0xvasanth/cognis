//! Multi-stage doc-list transformer chain.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

type Stage = Arc<dyn Runnable<Vec<Document>, Vec<Document>>>;

/// Chain N doc-list transformers back-to-back. Each stage's output feeds
/// the next.
///
/// Already expressible via repeated `.pipe()` — this type just gives the
/// pattern a name and a builder.
pub struct CompressorPipeline {
    stages: Vec<Stage>,
}

impl Default for CompressorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorPipeline {
    /// Empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage.
    pub fn stage(mut self, s: Stage) -> Self {
        self.stages.push(s);
        self
    }
}

#[async_trait]
impl Runnable<Vec<Document>, Vec<Document>> for CompressorPipeline {
    async fn invoke(
        &self,
        mut input: Vec<Document>,
        config: RunnableConfig,
    ) -> Result<Vec<Document>> {
        for s in &self.stages {
            input = s.invoke(input, config.clone()).await?;
        }
        Ok(input)
    }
    fn name(&self) -> &str {
        "CompressorPipeline"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DropOdd;
    #[async_trait]
    impl Runnable<Vec<Document>, Vec<Document>> for DropOdd {
        async fn invoke(&self, input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(input
                .into_iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, d)| d)
                .collect())
        }
    }

    struct Take2;
    #[async_trait]
    impl Runnable<Vec<Document>, Vec<Document>> for Take2 {
        async fn invoke(&self, input: Vec<Document>, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(input.into_iter().take(2).collect())
        }
    }

    #[tokio::test]
    async fn stages_run_in_order() {
        let p = CompressorPipeline::new()
            .stage(Arc::new(DropOdd))
            .stage(Arc::new(Take2));
        let docs: Vec<Document> = (0..6).map(|i| Document::new(i.to_string())).collect();
        let out = p.invoke(docs, RunnableConfig::default()).await.unwrap();
        // DropOdd → indices 0,2,4 (3 items) → Take2 → 2 items.
        assert_eq!(out.len(), 2);
    }
}
