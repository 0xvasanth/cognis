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

    /// Build a Langfuse score body, picking exactly one valid linkage.
    ///
    /// Langfuse rejects scores that mix linkage modes. Valid combos:
    /// - `traceId` + `observationId` — score on a span within a trace.
    /// - `traceId` alone — score on the trace itself.
    /// - `sessionId` alone — score on the session.
    ///
    /// Priority: prefer trace/observation linkage when a trace_id is
    /// available, otherwise fall back to session-only linkage. Returns
    /// `None` if neither is set (we can't link the score to anything).
    fn body(s: &ScoreRecord) -> Option<ScoreBody> {
        let value = match &s.value {
            ScoreValue::Numeric(n) => serde_json::json!(*n),
            ScoreValue::Categorical(c) => serde_json::json!(c),
            ScoreValue::Boolean(b) => serde_json::json!(if *b { 1 } else { 0 }),
        };
        let (trace_id, observation_id, session_id) = match (s.trace_id, &s.session_id) {
            (Some(t), _) => (
                Some(t.to_string()),
                Some(s.run_id.to_string()),
                None, // never set session_id alongside trace/observation
            ),
            (None, Some(sess)) => (None, None, Some(sess.clone())),
            (None, None) => return None,
        };
        Some(ScoreBody {
            id: envelope_id(),
            trace_id,
            observation_id,
            session_id,
            name: s.name.clone(),
            value,
            comment: s.comment.clone(),
        })
    }
}

#[async_trait]
impl ScoreSink for LangfuseScorer {
    async fn submit(&self, score: ScoreRecord) -> Result<(), TraceError> {
        let body = match Self::body(&score) {
            Some(b) => b,
            None => {
                tracing::warn!(
                    name = %score.name,
                    "langfuse score has no trace_id or session_id; skipping"
                );
                return Ok(());
            }
        };
        let resp = self
            .http
            .request(reqwest::Method::POST, "/api/public/scores")
            .json(&body)
            .send()
            .await
            .map_err(|e| TraceError::Network {
                backend: "langfuse",
                source: e,
            })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn rec(trace_id: Option<Uuid>, session_id: Option<&str>) -> ScoreRecord {
        ScoreRecord {
            run_id: Uuid::nil(),
            trace_id,
            session_id: session_id.map(str::to_string),
            name: "novelty".into(),
            value: ScoreValue::Numeric(1.0),
            comment: None,
        }
    }

    #[test]
    fn body_with_trace_carries_observation_and_no_session() {
        let r = rec(Some(Uuid::nil()), Some("sess-1"));
        let b = LangfuseScorer::body(&r).unwrap();
        assert!(b.trace_id.is_some());
        assert!(b.observation_id.is_some());
        assert!(
            b.session_id.is_none(),
            "session must not co-exist with trace linkage"
        );
    }

    #[test]
    fn body_with_session_only_omits_trace_and_observation() {
        let r = rec(None, Some("sess-2"));
        let b = LangfuseScorer::body(&r).unwrap();
        assert!(b.trace_id.is_none());
        assert!(b.observation_id.is_none());
        assert_eq!(b.session_id.as_deref(), Some("sess-2"));
    }

    #[test]
    fn body_returns_none_when_unlinked() {
        let r = rec(None, None);
        assert!(LangfuseScorer::body(&r).is_none());
    }
}
