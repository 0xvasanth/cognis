//! `TracingHandler` — `CallbackHandler` impl that translates lifecycle
//! events into `Span`s and fans them out to per-exporter batchers.

use std::sync::Arc;

use cognis_core::callbacks::CallbackHandler;
use dashmap::DashMap;
use uuid::Uuid;

use crate::batch::{Batcher, BatcherConfig};
use crate::cost::PriceTable;
use crate::exporter::TraceExporter;
use crate::span::{ScoreRecord, Span, SpanBuilder, SpanKind};

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

impl TracingHandler {
    fn start_span(
        &self,
        kind: SpanKind,
        name: &str,
        input: Option<serde_json::Value>,
        run_id: Uuid,
    ) {
        let parent = crate::parent::peek();
        let trace_id = parent.unwrap_or(run_id);
        let b = SpanBuilder::open(
            run_id,
            parent,
            trace_id,
            kind,
            name.to_string(),
            input,
            std::time::SystemTime::now(),
        );
        self.inflight.insert(run_id, b);
        crate::parent::push(run_id);
    }

    fn finish_ok(&self, run_id: Uuid, output: Option<serde_json::Value>) {
        if let Some((_, b)) = self.inflight.remove(&run_id) {
            let span = b.finish_ok(output, std::time::SystemTime::now());
            self.dispatch(span);
        }
        crate::parent::pop(run_id);
    }

    fn finish_error(&self, run_id: Uuid, message: &str) {
        if let Some((_, b)) = self.inflight.remove(&run_id) {
            let span = b.finish_error(message, std::time::SystemTime::now());
            self.dispatch(span);
        }
        crate::parent::pop(run_id);
    }

    fn dispatch(&self, span: Span) {
        for b in &self.span_batchers {
            b.send(span.clone());
        }
    }

    /// Parse a provider's `on_llm_end` payload into a `Generation` per the
    /// schema documented in spec §4.3. Missing fields default cleanly.
    fn parse_generation(&self, model_hint: &str, payload: &serde_json::Value) -> crate::span::Generation {
        use crate::span::{Generation, TokenUsage};

        let obj = payload.as_object();
        let model = obj
            .and_then(|o| o.get("model"))
            .and_then(|v| v.as_str())
            .unwrap_or(model_hint)
            .to_string();
        let provider = obj
            .and_then(|o| o.get("provider"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let finish_reason = obj
            .and_then(|o| o.get("finish_reason"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let model_parameters = obj
            .and_then(|o| o.get("model_parameters"))
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let usage = obj
            .and_then(|o| o.get("usage"))
            .and_then(|v| v.as_object())
            .map(|u| TokenUsage {
                input: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                output: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                cache_read: u.get("cache_read_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                cache_write: u.get("cache_creation_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            })
            .unwrap_or_default();
        let prompt_name = obj
            .and_then(|o| o.get("prompt_name"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let prompt_version = obj
            .and_then(|o| o.get("prompt_version"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let cost = self.pricing.compute(&model, usage);
        Generation {
            model,
            provider,
            model_parameters,
            usage,
            cost,
            completion_start_time: None,
            finish_reason,
            prompt_name,
            prompt_version,
        }
    }
}

impl CallbackHandler for TracingHandler {
    fn name(&self) -> &str {
        "cognis_trace::TracingHandler"
    }

    fn on_chain_start(&self, runnable: &str, input: &serde_json::Value, run_id: Uuid) {
        self.start_span(SpanKind::Chain, runnable, Some(input.clone()), run_id);
    }

    fn on_chain_end(&self, _runnable: &str, output: &serde_json::Value, run_id: Uuid) {
        self.finish_ok(run_id, Some(output.clone()));
    }

    fn on_chain_error(&self, _runnable: &str, error: &str, run_id: Uuid) {
        self.finish_error(run_id, error);
    }

    fn on_tool_start(&self, tool: &str, args: &serde_json::Value, run_id: Uuid) {
        self.start_span(SpanKind::Tool, tool, Some(args.clone()), run_id);
    }

    fn on_tool_end(&self, _tool: &str, result: &serde_json::Value, run_id: Uuid) {
        self.finish_ok(run_id, Some(result.clone()));
    }

    fn on_tool_error(&self, _tool: &str, error: &str, run_id: Uuid) {
        self.finish_error(run_id, error);
    }

    fn on_node_start(&self, node: &str, _step: u64, run_id: Uuid) {
        self.start_span(SpanKind::Span, node, None, run_id);
    }

    fn on_node_end(&self, _node: &str, _step: u64, output: &serde_json::Value, run_id: Uuid) {
        self.finish_ok(run_id, Some(output.clone()));
    }

    fn on_checkpoint(&self, _step: u64, _run_id: Uuid) {
        // No-op for trace tree shape; checkpoints are visible via the
        // graph engine's own observers.
    }

    fn on_custom(&self, kind: &str, payload: &serde_json::Value, run_id: Uuid) {
        // Emit a discrete EVENT span: open and immediately close.
        let trace_id = crate::parent::peek().unwrap_or(run_id);
        let now = std::time::SystemTime::now();
        let mut b = SpanBuilder::open(
            run_id,
            crate::parent::peek(),
            trace_id,
            SpanKind::Event,
            kind,
            Some(payload.clone()),
            now,
        );
        b.span.metadata.insert("kind".into(), serde_json::Value::String(kind.into()));
        let span = b.finish_ok(None, now);
        self.dispatch(span);
    }

    fn on_llm_start(&self, model: &str, prompt: &serde_json::Value, run_id: Uuid) {
        self.start_span(SpanKind::Generation, model, Some(prompt.clone()), run_id);
    }

    fn on_llm_token(&self, _token: &str, _run_id: Uuid) {
        // Streaming tokens are not buffered into the trace by default.
    }

    fn on_llm_end(&self, model: &str, output: &serde_json::Value, run_id: Uuid) {
        let generation = self.parse_generation(model, output);
        if let Some((_, b)) = self.inflight.remove(&run_id) {
            let b = b.with_generation(generation);
            // Use the payload's `content` field as `output` if present, else the whole payload.
            let out = output
                .as_object()
                .and_then(|o| o.get("content").cloned())
                .or_else(|| Some(output.clone()));
            let span = b.finish_ok(out, std::time::SystemTime::now());
            self.dispatch(span);
        }
        crate::parent::pop(run_id);
    }

    fn on_llm_error(&self, _model: &str, error: &str, run_id: Uuid) {
        self.finish_error(run_id, error);
    }
}
