//! `TracingHandler` — `CallbackHandler` impl that translates lifecycle
//! events into `Span`s and fans them out to per-exporter batchers.

use std::sync::Arc;

use cognis_core::callbacks::CallbackHandler;
use dashmap::DashMap;
use uuid::Uuid;

use crate::batch::{Batcher, BatcherConfig};
use crate::cost::PriceTable;
use crate::exporter::TraceExporter;
use crate::span::{ScoreRecord, Span, SpanBuilder};

/// Bridges `CallbackHandler` events to one or more `TraceExporter`s.
pub struct TracingHandler {
    exporters: Vec<Arc<dyn TraceExporter>>,
    inflight: DashMap<Uuid, SpanBuilder>,
    span_batchers: Vec<Batcher<Span>>,
    score_batchers: Vec<Batcher<ScoreRecord>>,
    pricing: Arc<PriceTable>,
}

impl TracingHandler {
    /// Start a new builder.
    pub fn builder() -> TracingHandlerBuilder {
        TracingHandlerBuilder::default()
    }

    /// Submit an out-of-band evaluation score for an existing run_id.
    pub fn record_score(&self, score: ScoreRecord) {
        for b in &self.score_batchers {
            b.send(score.clone());
        }
    }

    /// Stats per exporter (sent, dropped, failed).
    pub fn stats(&self, exporter_name: &str) -> Option<(usize, usize, usize)> {
        for (i, e) in self.exporters.iter().enumerate() {
            if e.name() == exporter_name {
                return self.span_batchers.get(i).map(|b| b.stats().snapshot());
            }
        }
        None
    }

    /// Graceful shutdown: drain batchers, then call each exporter's
    /// `shutdown()`. Must be awaited.
    pub async fn shutdown(self) {
        let Self {
            exporters,
            span_batchers,
            score_batchers,
            ..
        } = self;
        for b in span_batchers {
            b.shutdown().await;
        }
        for b in score_batchers {
            b.shutdown().await;
        }
        for e in exporters {
            if let Err(err) = e.shutdown().await {
                tracing::warn!(exporter = e.name(), error = %err, "exporter shutdown failed");
            }
        }
    }
}

/// Builder for `TracingHandler`.
#[derive(Default)]
pub struct TracingHandlerBuilder {
    exporters: Vec<Arc<dyn TraceExporter>>,
    pricing: Option<PriceTable>,
    batcher_cfg: BatcherConfig,
}

impl TracingHandlerBuilder {
    /// Append an exporter.
    pub fn with_exporter<E: TraceExporter + 'static>(mut self, e: E) -> Self {
        self.exporters.push(Arc::new(e));
        self
    }

    /// Use the dated default pricing snapshot.
    pub fn with_default_pricing(mut self) -> Self {
        self.pricing = Some(PriceTable::with_defaults());
        self
    }

    /// Provide a fully custom price table.
    pub fn with_pricing(mut self, p: PriceTable) -> Self {
        self.pricing = Some(p);
        self
    }

    /// Override or insert one model's price.
    pub fn override_price(mut self, model: impl Into<String>, p: crate::cost::ModelPrice) -> Self {
        let mut t = self.pricing.unwrap_or_default();
        t.insert(model, p);
        self.pricing = Some(t);
        self
    }

    /// Override the per-exporter batcher config.
    pub fn with_batcher_config(mut self, cfg: BatcherConfig) -> Self {
        self.batcher_cfg = cfg;
        self
    }

    /// Finalize. Spawns one `Batcher<Span>` and one `Batcher<ScoreRecord>`
    /// per exporter.
    pub fn build(self) -> TracingHandler {
        let cfg = self.batcher_cfg;
        let pricing = Arc::new(self.pricing.unwrap_or_default());

        let mut span_batchers = Vec::with_capacity(self.exporters.len());
        let mut score_batchers = Vec::with_capacity(self.exporters.len());
        for e in &self.exporters {
            let e_for_spans = e.clone();
            span_batchers.push(Batcher::spawn(cfg, move |batch: Vec<Span>| {
                let e = e_for_spans.clone();
                async move { e.export_spans(batch).await }
            }));
            let e_for_scores = e.clone();
            score_batchers.push(Batcher::spawn(cfg, move |batch: Vec<ScoreRecord>| {
                let e = e_for_scores.clone();
                async move { e.export_scores(batch).await }
            }));
        }

        TracingHandler {
            exporters: self.exporters,
            inflight: DashMap::new(),
            span_batchers,
            score_batchers,
            pricing,
        }
    }
}

// CallbackHandler impl is in Tasks 12 and 13 — for now only `name` is overridden.
impl CallbackHandler for TracingHandler {
    fn name(&self) -> &str {
        "cognis_trace::TracingHandler"
    }
    // All other methods left as defaults (no-op) for now — Tasks 12 and 13 fill them.
}
