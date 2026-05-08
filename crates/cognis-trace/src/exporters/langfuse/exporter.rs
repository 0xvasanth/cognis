//! Langfuse trace exporter — POST /api/public/ingestion.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::TraceError;
use crate::exporter::TraceExporter;
use crate::meta;
use crate::span::{ObservationLevel, ScoreRecord, ScoreValue, Span, SpanKind};

use super::client::LangfuseHttp;
use super::config::LangfuseConfig;
use super::wire::{
    envelope_id, format_iso8601, GenerationBody, IngestionEvent, IngestionRequest,
    IngestionResponse, ScoreBody, SpanBody, TraceBody,
};

/// Langfuse exporter.
pub struct LangfuseExporter {
    http: LangfuseHttp,
    cfg: LangfuseConfig,
}

impl LangfuseExporter {
    /// Construct from config.
    pub fn new(cfg: LangfuseConfig) -> Result<Self, TraceError> {
        Ok(Self {
            http: LangfuseHttp::new(&cfg)?,
            cfg,
        })
    }

    /// Read public key + secret key from environment.
    pub fn from_env() -> Result<Self, TraceError> {
        Self::new(LangfuseConfig::from_env()?)
    }

    fn level_str(level: ObservationLevel) -> &'static str {
        match level {
            ObservationLevel::Default => "DEFAULT",
            ObservationLevel::Debug => "DEBUG",
            ObservationLevel::Warning => "WARNING",
            ObservationLevel::Error => "ERROR",
        }
    }

    fn span_to_body(&self, span: &Span) -> SpanBody {
        SpanBody {
            id: span.run_id.to_string(),
            trace_id: span.trace_id.to_string(),
            parent_observation_id: span.parent_run_id.map(|u| u.to_string()),
            name: Some(span.name.clone()),
            start_time: Some(format_iso8601(span.started_at)),
            end_time: span.ended_at.map(format_iso8601),
            level: Some(Self::level_str(span.level).to_string()),
            status_message: span.status_message.clone(),
            metadata: if span.metadata.is_empty() {
                None
            } else {
                Some(serde_json::to_value(&span.metadata).ok().unwrap_or_default())
            },
            input: span.input.clone(),
            output: span.output.clone(),
            environment: self.cfg.environment.clone(),
        }
    }

    fn span_to_event(&self, span: &Span) -> IngestionEvent {
        let now = format_iso8601(std::time::SystemTime::now());
        match span.kind {
            SpanKind::Generation => {
                let span_body = self.span_to_body(span);
                let g = span.generation.as_ref();
                let usage_details = g.map(|g| {
                    let mut m = HashMap::new();
                    m.insert("input".into(), g.usage.input as u64);
                    m.insert("output".into(), g.usage.output as u64);
                    m.insert("cache_read_input".into(), g.usage.cache_read as u64);
                    m.insert("cache_creation_input".into(), g.usage.cache_write as u64);
                    m.insert("total".into(), g.usage.total() as u64);
                    m
                });
                let cost_details = g.and_then(|g| g.cost).map(|c| {
                    let mut m = HashMap::new();
                    m.insert("input".into(), c.input);
                    m.insert("output".into(), c.output);
                    m.insert("cache_read_input".into(), c.cache_read);
                    m.insert("cache_creation_input".into(), c.cache_write);
                    m.insert("total".into(), c.total);
                    m
                });
                IngestionEvent::GenerationCreate {
                    id: envelope_id(),
                    timestamp: now,
                    body: GenerationBody {
                        span: span_body,
                        completion_start_time: g.and_then(|g| g.completion_start_time).map(format_iso8601),
                        model: g.map(|g| g.model.clone()),
                        model_parameters: g
                            .map(|g| g.model_parameters.clone())
                            .filter(|m| !m.is_empty()),
                        usage_details,
                        cost_details,
                        prompt_name: g.and_then(|g| g.prompt_name.clone()),
                        prompt_version: g.and_then(|g| g.prompt_version),
                    },
                }
            }
            _ => IngestionEvent::SpanCreate {
                id: envelope_id(),
                timestamp: now,
                body: self.span_to_body(span),
            },
        }
    }

    fn maybe_trace_event(&self, span: &Span) -> Option<IngestionEvent> {
        if span.parent_run_id.is_some() {
            return None; // not a trace root
        }
        // Pull trace-level fields from `metadata` (well-known keys) and tags.
        let session_id = span.session_id.clone();
        let user_id = span.user_id.clone();
        let tags = if span.tags.is_empty() { None } else { Some(span.tags.clone()) };
        Some(IngestionEvent::TraceCreate {
            id: envelope_id(),
            timestamp: format_iso8601(std::time::SystemTime::now()),
            body: TraceBody {
                id: span.trace_id.to_string(),
                timestamp: format_iso8601(span.started_at),
                name: Some(span.name.clone()),
                user_id,
                input: span.input.clone(),
                output: span.output.clone(),
                session_id,
                release: self.cfg.release.clone(),
                version: meta::read_string(
                    &serde_json::Value::Object(
                        span.metadata
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    ),
                    crate::meta::keys::VERSION,
                ),
                metadata: if span.metadata.is_empty() {
                    None
                } else {
                    Some(serde_json::to_value(&span.metadata).ok().unwrap_or_default())
                },
                tags,
                environment: self.cfg.environment.clone(),
                public: None,
            },
        })
    }

    fn score_to_event(&self, s: &ScoreRecord) -> IngestionEvent {
        let value = match &s.value {
            ScoreValue::Numeric(n) => serde_json::json!(*n),
            ScoreValue::Categorical(c) => serde_json::json!(c),
            ScoreValue::Boolean(b) => serde_json::json!(if *b { 1 } else { 0 }),
        };
        IngestionEvent::ScoreCreate {
            id: envelope_id(),
            timestamp: format_iso8601(std::time::SystemTime::now()),
            body: ScoreBody {
                id: envelope_id(),
                trace_id: s.trace_id.map(|u| u.to_string()),
                observation_id: Some(s.run_id.to_string()),
                session_id: s.session_id.clone(),
                name: s.name.clone(),
                value,
                comment: s.comment.clone(),
            },
        }
    }

    async fn post_batch(&self, batch: Vec<IngestionEvent>) -> Result<(), TraceError> {
        let req = IngestionRequest { batch };
        let resp = self
            .http
            .request(reqwest::Method::POST, "/api/public/ingestion")
            .json(&req)
            .send()
            .await
            .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
        let status = resp.status();
        if status == 207 {
            // Successful batch envelope; per-event errors logged.
            let body: IngestionResponse = resp
                .json()
                .await
                .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
            if !body.errors.is_empty() {
                for err in &body.errors {
                    tracing::warn!(
                        envelope_id = err.id.as_str(),
                        status = err.status,
                        message = err.message.as_deref().unwrap_or(""),
                        "langfuse ingestion per-event error"
                    );
                }
            }
            return Ok(());
        }
        let code = status.as_u16();
        let body = resp.text().await.unwrap_or_default();
        Err(TraceError::BackendStatus {
            backend: "langfuse",
            status: code,
            body: body.chars().take(512).collect(),
        })
    }
}

#[async_trait]
impl TraceExporter for LangfuseExporter {
    async fn export_spans(&self, spans: Vec<Span>) -> Result<(), TraceError> {
        if spans.is_empty() {
            return Ok(());
        }
        let mut batch = Vec::with_capacity(spans.len() * 2);
        for s in &spans {
            if let Some(t) = self.maybe_trace_event(s) {
                batch.push(t);
            }
            batch.push(self.span_to_event(s));
        }
        self.post_batch(batch).await
    }

    async fn export_scores(&self, scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        if scores.is_empty() {
            return Ok(());
        }
        let batch = scores.iter().map(|s| self.score_to_event(s)).collect();
        self.post_batch(batch).await
    }

    fn name(&self) -> &str {
        "langfuse"
    }
}
