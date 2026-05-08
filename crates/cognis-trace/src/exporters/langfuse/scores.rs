//! Out-of-band Langfuse scorer. POSTs to `/api/public/scores`.

use async_trait::async_trait;

use crate::error::TraceError;
use crate::scores::ScoreSink;
use crate::span::{ScoreRecord, ScoreValue};

use super::client::LangfuseHttp;
use super::config::LangfuseConfig;
use super::wire::{envelope_id, ScoreBody};

/// Out-of-band scorer (separate from the trace ingestion pipeline).
pub struct LangfuseScorer {
    http: LangfuseHttp,
}

impl LangfuseScorer {
    /// Construct from config.
    pub fn new(cfg: LangfuseConfig) -> Result<Self, TraceError> {
        Ok(Self {
            http: LangfuseHttp::new(&cfg)?,
        })
    }

    fn body(s: &ScoreRecord) -> ScoreBody {
        let value = match &s.value {
            ScoreValue::Numeric(n) => serde_json::json!(*n),
            ScoreValue::Categorical(c) => serde_json::json!(c),
            ScoreValue::Boolean(b) => serde_json::json!(if *b { 1 } else { 0 }),
        };
        ScoreBody {
            id: envelope_id(),
            trace_id: s.trace_id.map(|u| u.to_string()),
            observation_id: Some(s.run_id.to_string()),
            session_id: s.session_id.clone(),
            name: s.name.clone(),
            value,
            comment: s.comment.clone(),
        }
    }
}

#[async_trait]
impl ScoreSink for LangfuseScorer {
    async fn submit(&self, score: ScoreRecord) -> Result<(), TraceError> {
        let body = Self::body(&score);
        let resp = self
            .http
            .request(reqwest::Method::POST, "/api/public/scores")
            .json(&body)
            .send()
            .await
            .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
        if resp.status().is_success() {
            return Ok(());
        }
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        Err(TraceError::BackendStatus {
            backend: "langfuse",
            status,
            body: body.chars().take(512).collect(),
        })
    }
}
