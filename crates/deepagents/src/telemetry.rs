//! Telemetry and observability system for tracking agent execution metrics,
//! token usage, and performance.
//!
//! Provides [`TelemetryCollector`] for recording [`Metric`]s, [`SpanContext`]s,
//! and [`TokenUsage`] data. Collected telemetry can be exported via the
//! [`TelemetryExporter`] trait, with built-in [`JsonExporter`] and
//! [`PrettyPrintExporter`] implementations.
//!
//! # Example
//!
//! ```rust,ignore
//! use deepagents::telemetry::{TelemetryCollector, TokenUsage, MetricType, MetricValue, Metric};
//!
//! let mut collector = TelemetryCollector::new();
//!
//! // Record token usage
//! let usage = TokenUsage::new(100, 50, "claude-sonnet-4");
//! collector.record_token_usage(usage);
//!
//! // Record a metric
//! let metric = Metric::new("request_count", MetricType::Counter, MetricValue::Count(1));
//! collector.record_metric(metric);
//!
//! // Start and finish a span
//! let mut span = collector.start_span("model_call");
//! span.set_attribute("model", serde_json::json!("claude-sonnet-4"));
//! span.finish();
//! collector.finish_span(span);
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// MetricType
// ---------------------------------------------------------------------------

/// The kind of metric being recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricType {
    /// A monotonically increasing counter.
    Counter,
    /// A value that can go up and down.
    Gauge,
    /// A distribution of values.
    Histogram,
    /// A duration measurement.
    Timer,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter => write!(f, "counter"),
            Self::Gauge => write!(f, "gauge"),
            Self::Histogram => write!(f, "histogram"),
            Self::Timer => write!(f, "timer"),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricValue
// ---------------------------------------------------------------------------

/// A concrete value attached to a metric.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// An integer count.
    Count(u64),
    /// A floating-point measurement.
    Float(f64),
    /// A duration measurement.
    Duration(Duration),
    /// A distribution of floating-point samples.
    Distribution(Vec<f64>),
}

impl MetricValue {
    /// Try to extract the value as a `u64` count.
    pub fn as_count(&self) -> Option<u64> {
        match self {
            Self::Count(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract the value as an `f64`.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract the value as a `Duration`.
    pub fn as_duration(&self) -> Option<Duration> {
        match self {
            Self::Duration(v) => Some(*v),
            _ => None,
        }
    }

    /// Serialize this value to a JSON representation.
    fn to_json(&self) -> Value {
        match self {
            Self::Count(v) => serde_json::json!(v),
            Self::Float(v) => serde_json::json!(v),
            Self::Duration(v) => serde_json::json!(v.as_secs_f64()),
            Self::Distribution(v) => serde_json::json!(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

/// A single recorded metric with a name, type, value, labels, and timestamp.
#[derive(Debug, Clone)]
pub struct Metric {
    /// The metric name.
    pub name: String,
    /// The kind of metric.
    pub metric_type: MetricType,
    /// The measured value.
    pub value: MetricValue,
    /// Arbitrary key-value labels for grouping and filtering.
    pub labels: HashMap<String, String>,
    /// When the metric was recorded.
    pub timestamp: Instant,
}

impl Metric {
    /// Create a new metric with the given name, type, and value.
    pub fn new(name: impl Into<String>, metric_type: MetricType, value: MetricValue) -> Self {
        Self {
            name: name.into(),
            metric_type,
            value,
            labels: HashMap::new(),
            timestamp: Instant::now(),
        }
    }

    /// Add a label to this metric (builder pattern).
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Serialize this metric to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "type": self.metric_type.to_string(),
            "value": self.value.to_json(),
            "labels": self.labels,
        })
    }
}

// ---------------------------------------------------------------------------
// TokenUsage
// ---------------------------------------------------------------------------

/// Tracks token consumption for a single model call.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u64,
    /// Number of tokens in the completion.
    pub completion_tokens: u64,
    /// Total tokens (prompt + completion).
    pub total_tokens: u64,
    /// The model used.
    pub model: String,
    /// Optional estimated cost in USD.
    pub cost_estimate: Option<f64>,
}

impl TokenUsage {
    /// Create a new token usage record. `total_tokens` is computed automatically.
    pub fn new(prompt_tokens: u64, completion_tokens: u64, model: impl Into<String>) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            model: model.into(),
            cost_estimate: None,
        }
    }

    /// Estimate the cost in USD based on model name.
    ///
    /// Uses rough per-token pricing. Unknown models get a default rate.
    pub fn estimate_cost(&mut self) {
        let (prompt_rate, completion_rate) = match self.model.as_str() {
            m if m.contains("gpt-4") => (0.00003, 0.00006),
            m if m.contains("gpt-3.5") => (0.0000005, 0.0000015),
            m if m.contains("claude-3-opus") || m.contains("claude-opus") => (0.000015, 0.000075),
            m if m.contains("claude-3-sonnet") || m.contains("claude-sonnet") => {
                (0.000003, 0.000015)
            }
            m if m.contains("claude-3-haiku") || m.contains("claude-haiku") => {
                (0.00000025, 0.00000125)
            }
            _ => (0.000001, 0.000002),
        };
        self.cost_estimate =
            Some(self.prompt_tokens as f64 * prompt_rate + self.completion_tokens as f64 * completion_rate);
    }

    /// Convert this usage record to a [`Metric`].
    pub fn to_metric(&self) -> Metric {
        Metric::new(
            "token_usage",
            MetricType::Counter,
            MetricValue::Count(self.total_tokens),
        )
        .with_label("model", &self.model)
        .with_label("prompt_tokens", self.prompt_tokens.to_string())
        .with_label("completion_tokens", self.completion_tokens.to_string())
    }

    /// Serialize to a JSON value.
    fn to_json(&self) -> Value {
        serde_json::json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_estimate": self.cost_estimate,
        })
    }
}

// ---------------------------------------------------------------------------
// SpanStatus
// ---------------------------------------------------------------------------

/// Status of a completed span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// The operation completed successfully.
    Ok,
    /// The operation failed with an error message.
    Error(String),
    /// The operation was cancelled.
    Cancelled,
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Error(msg) => write!(f, "error: {}", msg),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// SpanContext
// ---------------------------------------------------------------------------

/// Represents a traced span of execution with timing, attributes, and status.
#[derive(Debug, Clone)]
pub struct SpanContext {
    /// Unique identifier for this span.
    pub span_id: String,
    /// Parent span id, if this is a child span.
    pub parent_id: Option<String>,
    /// The operation being traced.
    pub operation: String,
    /// When the span started.
    pub start_time: Instant,
    /// When the span ended (set by [`finish`](Self::finish)).
    pub end_time: Option<Instant>,
    /// Arbitrary attributes attached to this span.
    pub attributes: HashMap<String, Value>,
    /// The status of the span.
    pub status: SpanStatus,
}

impl SpanContext {
    /// Create a new root span for the given operation.
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_id: None,
            operation: operation.into(),
            start_time: Instant::now(),
            end_time: None,
            attributes: HashMap::new(),
            status: SpanStatus::Ok,
        }
    }

    /// Create a child span linked to a parent.
    pub fn child(parent_id: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            span_id: uuid::Uuid::new_v4().to_string(),
            parent_id: Some(parent_id.into()),
            operation: operation.into(),
            start_time: Instant::now(),
            end_time: None,
            attributes: HashMap::new(),
            status: SpanStatus::Ok,
        }
    }

    /// Mark this span as finished, recording the end time.
    pub fn finish(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Return the duration of this span. If not yet finished, returns
    /// the elapsed time since the start.
    pub fn duration(&self) -> Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Set an attribute on this span.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: Value) {
        self.attributes.insert(key.into(), value);
    }

    /// Set the status of this span.
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// Serialize to a JSON value (excluding `Instant` fields, which use durations instead).
    fn to_json(&self) -> Value {
        serde_json::json!({
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "duration_secs": self.duration().as_secs_f64(),
            "finished": self.end_time.is_some(),
            "attributes": self.attributes,
            "status": self.status.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// TelemetryCollector
// ---------------------------------------------------------------------------

/// Collects metrics, spans, and token usage records for later export.
#[derive(Debug)]
pub struct TelemetryCollector {
    metrics: Vec<Metric>,
    spans: Vec<SpanContext>,
    token_usages: Vec<TokenUsage>,
}

impl TelemetryCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            spans: Vec::new(),
            token_usages: Vec::new(),
        }
    }

    /// Record a metric.
    pub fn record_metric(&mut self, metric: Metric) {
        self.metrics.push(metric);
    }

    /// Record a token usage entry.
    pub fn record_token_usage(&mut self, usage: TokenUsage) {
        self.token_usages.push(usage);
    }

    /// Start a new span and return it. The caller should call
    /// [`SpanContext::finish`] and then [`finish_span`](Self::finish_span)
    /// to record it.
    pub fn start_span(&self, operation: impl Into<String>) -> SpanContext {
        SpanContext::new(operation)
    }

    /// Record a finished span.
    pub fn finish_span(&mut self, span: SpanContext) {
        self.spans.push(span);
    }

    /// Return a reference to all recorded metrics.
    pub fn get_metrics(&self) -> &[Metric] {
        &self.metrics
    }

    /// Return a reference to all recorded spans.
    pub fn get_spans(&self) -> &[SpanContext] {
        &self.spans
    }

    /// Compute an aggregate summary of all token usage records.
    ///
    /// Returns `(total_prompt, total_completion, total_tokens, total_cost)`.
    pub fn get_token_usage_summary(&self) -> (u64, u64, u64, f64) {
        let mut total_prompt = 0u64;
        let mut total_completion = 0u64;
        let mut total_tokens = 0u64;
        let mut total_cost = 0.0f64;

        for usage in &self.token_usages {
            total_prompt += usage.prompt_tokens;
            total_completion += usage.completion_tokens;
            total_tokens += usage.total_tokens;
            total_cost += usage.cost_estimate.unwrap_or(0.0);
        }

        (total_prompt, total_completion, total_tokens, total_cost)
    }

    /// Clear all collected data.
    pub fn reset(&mut self) {
        self.metrics.clear();
        self.spans.clear();
        self.token_usages.clear();
    }

    /// Export all collected telemetry as a JSON value.
    pub fn to_json(&self) -> Value {
        let (tp, tc, tt, cost) = self.get_token_usage_summary();
        serde_json::json!({
            "metrics": self.metrics.iter().map(|m| m.to_json()).collect::<Vec<_>>(),
            "spans": self.spans.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "token_usage": {
                "records": self.token_usages.iter().map(|u| u.to_json()).collect::<Vec<_>>(),
                "summary": {
                    "total_prompt_tokens": tp,
                    "total_completion_tokens": tc,
                    "total_tokens": tt,
                    "total_cost_estimate": cost,
                }
            },
        })
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TelemetryExporter
// ---------------------------------------------------------------------------

/// Trait for exporting collected telemetry data.
pub trait TelemetryExporter {
    /// Export the collector's data as a string.
    fn export(&self, collector: &TelemetryCollector) -> Result<String, Box<dyn std::error::Error>>;
}

// ---------------------------------------------------------------------------
// JsonExporter
// ---------------------------------------------------------------------------

/// Exports telemetry data as a JSON string.
#[derive(Debug, Clone, Default)]
pub struct JsonExporter;

impl JsonExporter {
    /// Create a new JSON exporter.
    pub fn new() -> Self {
        Self
    }
}

impl TelemetryExporter for JsonExporter {
    fn export(&self, collector: &TelemetryCollector) -> Result<String, Box<dyn std::error::Error>> {
        let json = collector.to_json();
        Ok(serde_json::to_string_pretty(&json)?)
    }
}

// ---------------------------------------------------------------------------
// PrettyPrintExporter
// ---------------------------------------------------------------------------

/// Exports telemetry data as a human-readable text summary.
#[derive(Debug, Clone, Default)]
pub struct PrettyPrintExporter;

impl PrettyPrintExporter {
    /// Create a new pretty-print exporter.
    pub fn new() -> Self {
        Self
    }
}

impl TelemetryExporter for PrettyPrintExporter {
    fn export(&self, collector: &TelemetryCollector) -> Result<String, Box<dyn std::error::Error>> {
        let mut out = String::new();

        out.push_str("=== Telemetry Report ===\n\n");

        // Metrics
        out.push_str(&format!("Metrics ({})\n", collector.get_metrics().len()));
        out.push_str(&"-".repeat(40));
        out.push('\n');
        for m in collector.get_metrics() {
            let labels_str = if m.labels.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> = m.labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
                format!(" [{}]", pairs.join(", "))
            };
            out.push_str(&format!(
                "  {} ({}): {}{}\n",
                m.name,
                m.metric_type,
                match &m.value {
                    MetricValue::Count(v) => format!("{}", v),
                    MetricValue::Float(v) => format!("{:.4}", v),
                    MetricValue::Duration(v) => format!("{:.3}ms", v.as_secs_f64() * 1000.0),
                    MetricValue::Distribution(v) => format!("{} samples", v.len()),
                },
                labels_str,
            ));
        }

        out.push('\n');

        // Spans
        out.push_str(&format!("Spans ({})\n", collector.get_spans().len()));
        out.push_str(&"-".repeat(40));
        out.push('\n');
        for s in collector.get_spans() {
            let parent = match &s.parent_id {
                Some(id) => format!(" (parent: {})", &id[..8.min(id.len())]),
                None => String::new(),
            };
            out.push_str(&format!(
                "  {} [{}] {:.3}ms {}{}\n",
                s.operation,
                s.status,
                s.duration().as_secs_f64() * 1000.0,
                &s.span_id[..8.min(s.span_id.len())],
                parent,
            ));
        }

        out.push('\n');

        // Token usage
        let (tp, tc, tt, cost) = collector.get_token_usage_summary();
        out.push_str("Token Usage Summary\n");
        out.push_str(&"-".repeat(40));
        out.push('\n');
        out.push_str(&format!("  Prompt tokens:     {}\n", tp));
        out.push_str(&format!("  Completion tokens: {}\n", tc));
        out.push_str(&format!("  Total tokens:      {}\n", tt));
        out.push_str(&format!("  Estimated cost:    ${:.6}\n", cost));

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MetricValue helpers --

    #[test]
    fn test_metric_value_as_count() {
        let v = MetricValue::Count(42);
        assert_eq!(v.as_count(), Some(42));
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_duration(), None);
    }

    #[test]
    fn test_metric_value_as_float() {
        let v = MetricValue::Float(3.14);
        assert_eq!(v.as_float(), Some(3.14));
        assert_eq!(v.as_count(), None);
        assert_eq!(v.as_duration(), None);
    }

    #[test]
    fn test_metric_value_as_duration() {
        let d = Duration::from_millis(500);
        let v = MetricValue::Duration(d);
        assert_eq!(v.as_duration(), Some(d));
        assert_eq!(v.as_count(), None);
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn test_metric_value_distribution_returns_none_for_helpers() {
        let v = MetricValue::Distribution(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.as_count(), None);
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_duration(), None);
    }

    #[test]
    fn test_metric_value_to_json_count() {
        let v = MetricValue::Count(10);
        assert_eq!(v.to_json(), serde_json::json!(10));
    }

    #[test]
    fn test_metric_value_to_json_float() {
        let v = MetricValue::Float(2.5);
        assert_eq!(v.to_json(), serde_json::json!(2.5));
    }

    #[test]
    fn test_metric_value_to_json_duration() {
        let v = MetricValue::Duration(Duration::from_secs(1));
        assert_eq!(v.to_json(), serde_json::json!(1.0));
    }

    #[test]
    fn test_metric_value_to_json_distribution() {
        let v = MetricValue::Distribution(vec![1.0, 2.0]);
        assert_eq!(v.to_json(), serde_json::json!([1.0, 2.0]));
    }

    // -- Metric creation and labels --

    #[test]
    fn test_metric_new() {
        let m = Metric::new("requests", MetricType::Counter, MetricValue::Count(1));
        assert_eq!(m.name, "requests");
        assert_eq!(m.metric_type, MetricType::Counter);
        assert!(m.labels.is_empty());
    }

    #[test]
    fn test_metric_with_labels() {
        let m = Metric::new("latency", MetricType::Timer, MetricValue::Duration(Duration::from_millis(100)))
            .with_label("endpoint", "/api/chat")
            .with_label("method", "POST");
        assert_eq!(m.labels.len(), 2);
        assert_eq!(m.labels["endpoint"], "/api/chat");
        assert_eq!(m.labels["method"], "POST");
    }

    #[test]
    fn test_metric_to_json() {
        let m = Metric::new("errors", MetricType::Gauge, MetricValue::Count(5))
            .with_label("service", "agent");
        let json = m.to_json();
        assert_eq!(json["name"], "errors");
        assert_eq!(json["type"], "gauge");
        assert_eq!(json["value"], 5);
        assert_eq!(json["labels"]["service"], "agent");
    }

    // -- MetricType display --

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Timer.to_string(), "timer");
    }

    // -- TokenUsage --

    #[test]
    fn test_token_usage_auto_total() {
        let usage = TokenUsage::new(100, 50, "gpt-4");
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert!(usage.cost_estimate.is_none());
    }

    #[test]
    fn test_token_usage_estimate_cost_gpt4() {
        let mut usage = TokenUsage::new(1000, 500, "gpt-4");
        usage.estimate_cost();
        assert!(usage.cost_estimate.is_some());
        let cost = usage.cost_estimate.unwrap();
        // 1000 * 0.00003 + 500 * 0.00006 = 0.03 + 0.03 = 0.06
        assert!((cost - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_token_usage_estimate_cost_claude_sonnet() {
        let mut usage = TokenUsage::new(1000, 500, "claude-sonnet-4");
        usage.estimate_cost();
        let cost = usage.cost_estimate.unwrap();
        // 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
        assert!((cost - 0.0105).abs() < 1e-10);
    }

    #[test]
    fn test_token_usage_estimate_cost_unknown_model() {
        let mut usage = TokenUsage::new(1000, 500, "some-unknown-model");
        usage.estimate_cost();
        let cost = usage.cost_estimate.unwrap();
        // 1000 * 0.000001 + 500 * 0.000002 = 0.001 + 0.001 = 0.002
        assert!((cost - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_token_usage_to_metric() {
        let usage = TokenUsage::new(200, 100, "gpt-3.5-turbo");
        let metric = usage.to_metric();
        assert_eq!(metric.name, "token_usage");
        assert_eq!(metric.metric_type, MetricType::Counter);
        assert_eq!(metric.value.as_count(), Some(300));
        assert_eq!(metric.labels["model"], "gpt-3.5-turbo");
        assert_eq!(metric.labels["prompt_tokens"], "200");
        assert_eq!(metric.labels["completion_tokens"], "100");
    }

    // -- SpanContext --

    #[test]
    fn test_span_context_new() {
        let span = SpanContext::new("model_call");
        assert_eq!(span.operation, "model_call");
        assert!(span.parent_id.is_none());
        assert!(span.end_time.is_none());
        assert_eq!(span.status, SpanStatus::Ok);
        assert!(span.attributes.is_empty());
        assert!(!span.span_id.is_empty());
    }

    #[test]
    fn test_span_context_child() {
        let parent = SpanContext::new("parent_op");
        let child = SpanContext::child(&parent.span_id, "child_op");
        assert_eq!(child.parent_id.as_deref(), Some(parent.span_id.as_str()));
        assert_eq!(child.operation, "child_op");
        assert_ne!(child.span_id, parent.span_id);
    }

    #[test]
    fn test_span_context_finish() {
        let mut span = SpanContext::new("op");
        assert!(span.end_time.is_none());
        span.finish();
        assert!(span.end_time.is_some());
    }

    #[test]
    fn test_span_context_duration_after_finish() {
        let mut span = SpanContext::new("op");
        std::thread::sleep(Duration::from_millis(10));
        span.finish();
        let dur = span.duration();
        assert!(dur >= Duration::from_millis(10));
    }

    #[test]
    fn test_span_context_duration_while_running() {
        let span = SpanContext::new("op");
        // Duration should be non-negative even if not finished
        let dur = span.duration();
        assert!(dur < Duration::from_secs(10));
    }

    #[test]
    fn test_span_context_set_attribute() {
        let mut span = SpanContext::new("op");
        span.set_attribute("model", serde_json::json!("gpt-4"));
        span.set_attribute("temperature", serde_json::json!(0.7));
        assert_eq!(span.attributes.len(), 2);
        assert_eq!(span.attributes["model"], serde_json::json!("gpt-4"));
    }

    #[test]
    fn test_span_context_set_status() {
        let mut span = SpanContext::new("op");
        assert_eq!(span.status, SpanStatus::Ok);
        span.set_status(SpanStatus::Error("timeout".into()));
        assert_eq!(span.status, SpanStatus::Error("timeout".into()));
    }

    #[test]
    fn test_span_context_set_status_cancelled() {
        let mut span = SpanContext::new("op");
        span.set_status(SpanStatus::Cancelled);
        assert_eq!(span.status, SpanStatus::Cancelled);
    }

    // -- SpanStatus display --

    #[test]
    fn test_span_status_display() {
        assert_eq!(SpanStatus::Ok.to_string(), "ok");
        assert_eq!(SpanStatus::Error("fail".into()).to_string(), "error: fail");
        assert_eq!(SpanStatus::Cancelled.to_string(), "cancelled");
    }

    // -- TelemetryCollector --

    #[test]
    fn test_collector_new_empty() {
        let c = TelemetryCollector::new();
        assert!(c.get_metrics().is_empty());
        assert!(c.get_spans().is_empty());
        assert_eq!(c.get_token_usage_summary(), (0, 0, 0, 0.0));
    }

    #[test]
    fn test_collector_record_metric() {
        let mut c = TelemetryCollector::new();
        c.record_metric(Metric::new("test", MetricType::Counter, MetricValue::Count(1)));
        assert_eq!(c.get_metrics().len(), 1);
        assert_eq!(c.get_metrics()[0].name, "test");
    }

    #[test]
    fn test_collector_record_token_usage() {
        let mut c = TelemetryCollector::new();
        c.record_token_usage(TokenUsage::new(100, 50, "gpt-4"));
        let (tp, tc, tt, _) = c.get_token_usage_summary();
        assert_eq!(tp, 100);
        assert_eq!(tc, 50);
        assert_eq!(tt, 150);
    }

    #[test]
    fn test_collector_token_usage_summary_aggregation() {
        let mut c = TelemetryCollector::new();

        let mut u1 = TokenUsage::new(100, 50, "gpt-4");
        u1.estimate_cost();
        c.record_token_usage(u1);

        let mut u2 = TokenUsage::new(200, 100, "gpt-4");
        u2.estimate_cost();
        c.record_token_usage(u2);

        let (tp, tc, tt, cost) = c.get_token_usage_summary();
        assert_eq!(tp, 300);
        assert_eq!(tc, 150);
        assert_eq!(tt, 450);
        // (100*0.00003 + 50*0.00006) + (200*0.00003 + 100*0.00006) = 0.006 + 0.012 = 0.018
        assert!(cost > 0.0);
    }

    #[test]
    fn test_collector_start_and_finish_span() {
        let mut c = TelemetryCollector::new();
        let mut span = c.start_span("test_op");
        span.set_attribute("key", serde_json::json!("value"));
        span.finish();
        c.finish_span(span);
        assert_eq!(c.get_spans().len(), 1);
        assert_eq!(c.get_spans()[0].operation, "test_op");
        assert!(c.get_spans()[0].end_time.is_some());
    }

    #[test]
    fn test_collector_multiple_spans() {
        let mut c = TelemetryCollector::new();
        for i in 0..5 {
            let mut span = c.start_span(format!("op_{}", i));
            span.finish();
            c.finish_span(span);
        }
        assert_eq!(c.get_spans().len(), 5);
    }

    #[test]
    fn test_collector_reset() {
        let mut c = TelemetryCollector::new();
        c.record_metric(Metric::new("m", MetricType::Counter, MetricValue::Count(1)));
        c.record_token_usage(TokenUsage::new(10, 5, "model"));
        let mut span = c.start_span("op");
        span.finish();
        c.finish_span(span);

        assert!(!c.get_metrics().is_empty());
        assert!(!c.get_spans().is_empty());

        c.reset();

        assert!(c.get_metrics().is_empty());
        assert!(c.get_spans().is_empty());
        assert_eq!(c.get_token_usage_summary(), (0, 0, 0, 0.0));
    }

    #[test]
    fn test_collector_to_json() {
        let mut c = TelemetryCollector::new();
        c.record_metric(Metric::new("req", MetricType::Counter, MetricValue::Count(5)));
        c.record_token_usage(TokenUsage::new(50, 25, "test-model"));

        let json = c.to_json();
        assert!(json["metrics"].is_array());
        assert_eq!(json["metrics"].as_array().unwrap().len(), 1);
        assert!(json["spans"].is_array());
        assert!(json["token_usage"]["summary"]["total_tokens"].is_number());
        assert_eq!(json["token_usage"]["summary"]["total_tokens"], 75);
    }

    #[test]
    fn test_collector_default() {
        let c = TelemetryCollector::default();
        assert!(c.get_metrics().is_empty());
    }

    // -- JsonExporter --

    #[test]
    fn test_json_exporter() {
        let mut c = TelemetryCollector::new();
        c.record_metric(Metric::new("x", MetricType::Gauge, MetricValue::Float(1.5)));

        let exporter = JsonExporter::new();
        let result = exporter.export(&c).unwrap();
        assert!(result.contains("\"x\""));
        assert!(result.contains("gauge"));
        // Should be valid JSON
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["metrics"].is_array());
    }

    #[test]
    fn test_json_exporter_empty() {
        let c = TelemetryCollector::new();
        let exporter = JsonExporter::new();
        let result = exporter.export(&c).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["metrics"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["spans"].as_array().unwrap().len(), 0);
    }

    // -- PrettyPrintExporter --

    #[test]
    fn test_pretty_print_exporter() {
        let mut c = TelemetryCollector::new();
        c.record_metric(
            Metric::new("requests", MetricType::Counter, MetricValue::Count(10))
                .with_label("service", "agent"),
        );
        let mut usage = TokenUsage::new(100, 50, "gpt-4");
        usage.estimate_cost();
        c.record_token_usage(usage);

        let mut span = c.start_span("model_invoke");
        span.finish();
        c.finish_span(span);

        let exporter = PrettyPrintExporter::new();
        let result = exporter.export(&c).unwrap();

        assert!(result.contains("Telemetry Report"));
        assert!(result.contains("Metrics (1)"));
        assert!(result.contains("requests"));
        assert!(result.contains("Spans (1)"));
        assert!(result.contains("model_invoke"));
        assert!(result.contains("Token Usage Summary"));
        assert!(result.contains("100"));
        assert!(result.contains("50"));
    }

    #[test]
    fn test_pretty_print_exporter_empty() {
        let c = TelemetryCollector::new();
        let exporter = PrettyPrintExporter::new();
        let result = exporter.export(&c).unwrap();
        assert!(result.contains("Metrics (0)"));
        assert!(result.contains("Spans (0)"));
    }

    // -- SpanContext JSON --

    #[test]
    fn test_span_context_to_json() {
        let mut span = SpanContext::new("test_op");
        span.set_attribute("model", serde_json::json!("gpt-4"));
        span.finish();
        let json = span.to_json();
        assert_eq!(json["operation"], "test_op");
        assert_eq!(json["finished"], true);
        assert!(json["span_id"].is_string());
        assert!(json["parent_id"].is_null());
        assert_eq!(json["status"], "ok");
    }

    #[test]
    fn test_span_context_child_to_json() {
        let parent = SpanContext::new("parent");
        let child = SpanContext::child(&parent.span_id, "child");
        let json = child.to_json();
        assert_eq!(json["parent_id"], parent.span_id);
    }
}
