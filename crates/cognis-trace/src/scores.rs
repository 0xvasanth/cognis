//! `ScoreSink` trait — submit evaluation scores to an external store.

use async_trait::async_trait;

use crate::error::TraceError;
use crate::span::ScoreRecord;

/// Submit evaluation scores out-of-band.
#[async_trait]
pub trait ScoreSink: Send + Sync {
    /// Push one score.
    async fn submit(&self, score: ScoreRecord) -> Result<(), TraceError>;

    /// Push many scores. Default loops `submit`.
    async fn submit_many(&self, scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        for s in scores {
            self.submit(s).await?;
        }
        Ok(())
    }
}
