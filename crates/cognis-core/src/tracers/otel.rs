//! OpenTelemetry-compatible callback trace handler.
//!
//! Produces structured trace data compatible with the OpenTelemetry data model
//! without requiring any external OpenTelemetry SDK dependency.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::documents::Document;
use crate::error::Result;
use crate::outputs::LLMResult;

/// Status of a trace span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "code", content = "message")]
pub enum SpanStatus {
    /// The span completed successfully.
    Ok,
    /// The span ended with an error.
    Error(String),
}

/// A timestamped event within a span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// ISO-8601 timestamp of the event.
    pub timestamp: String,
    /// Arbitrary key-value attributes for the event.
    pub attributes: HashMap<String, Value>,
}

/// A single trace span, modeled after the OpenTelemetry Span data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Trace-wide identifier grouping related spans.
    pub trace_id: String,
    /// Unique identifier for this span.
    pub span_id: String,
    /// Parent span identifier, if this span is nested.
    pub parent_span_id: Option<String>,
    /// Operation name describing what this span covers.
    pub operation_name: String,
    /// ISO-8601 start time.
    pub start_time: String,
    /// ISO-8601 end time (set when the span is closed).
    pub end_time: Option<String>,
    /// Span outcome status.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes.
    pub attributes: HashMap<String, Value>,
    /// Timestamped events recorded during the span's lifetime.
    pub events: Vec<SpanEvent>,
}

/// Returns the current time as an ISO-8601 string (RFC 3339 subset).
fn now_iso8601() -> String {
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    // Calculate date/time components from unix timestamp
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Convert days since epoch to year/month/day (simplified civil calendar)
    let (year, month, day) = civil_from_days(days as i64);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

/// Converts days since Unix epoch to (year, month, day).
/// Algorithm from Howard Hinnant's date algorithms.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Callback handler that collects OpenTelemetry-compatible trace spans.
///
/// Spans are created on `*_start` callbacks and completed on `*_end` or
/// `*_error` callbacks. Completed spans are moved from `active_spans` to
/// `spans` for later retrieval via [`get_spans`](Self::get_spans) or
/// [`to_json`](Self::to_json).
pub struct OtelTraceCallbackHandler {
    /// Completed spans, in order of completion.
    spans: Arc<RwLock<Vec<TraceSpan>>>,
    /// Currently in-flight spans, keyed by run_id string.
    active_spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
    /// Trace identifier shared by all spans produced by this handler.
    pub trace_id: String,
}

impl OtelTraceCallbackHandler {
    /// Creates a new handler with a freshly generated trace ID.
    pub fn new() -> Self {
        Self {
            spans: Arc::new(RwLock::new(Vec::new())),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            trace_id: Uuid::new_v4().to_string(),
        }
    }

    /// Returns a snapshot of all completed spans.
    pub fn get_spans(&self) -> Vec<TraceSpan> {
        self.spans.read().unwrap().clone()
    }

    /// Exports all completed spans as a JSON value in an OTel-compatible
    /// resource-spans envelope.
    pub fn to_json(&self) -> Value {
        let spans = self.spans.read().unwrap();
        serde_json::json!({
            "resourceSpans": [{
                "resource": {
                    "attributes": []
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "cognis-core"
                    },
                    "spans": serde_json::to_value(&*spans).unwrap_or_default()
                }]
            }]
        })
    }

    /// Clears all completed spans.
    pub fn clear(&self) {
        self.spans.write().unwrap().clear();
    }

    /// Start a new span and register it as active.
    fn start_span(
        &self,
        run_id: Uuid,
        operation_name: &str,
        parent_run_id: Option<Uuid>,
        attributes: HashMap<String, Value>,
    ) {
        let span = TraceSpan {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: parent_run_id.map(|id| id.to_string()),
            operation_name: operation_name.to_string(),
            start_time: now_iso8601(),
            end_time: None,
            status: SpanStatus::Ok,
            attributes,
            events: Vec::new(),
        };
        self.active_spans
            .write()
            .unwrap()
            .insert(run_id.to_string(), span);
    }

    /// Finalize an active span (set end_time, status) and move it to
    /// completed spans.
    fn end_span(&self, run_id: Uuid, status: SpanStatus, extra_attrs: HashMap<String, Value>) {
        let key = run_id.to_string();
        let mut active = self.active_spans.write().unwrap();
        if let Some(mut span) = active.remove(&key) {
            span.end_time = Some(now_iso8601());
            span.status = status;
            span.attributes.extend(extra_attrs);
            self.spans.write().unwrap().push(span);
        }
    }
}

impl Default for OtelTraceCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for OtelTraceCallbackHandler {
    async fn on_llm_start(
        &self,
        serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let model_name = serialized
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mut attrs = HashMap::new();
        attrs.insert("model_name".into(), Value::String(model_name.to_string()));
        attrs.insert("prompts_count".into(), serde_json::json!(prompts.len()));
        self.start_span(run_id, "llm", parent_run_id, attrs);
        Ok(())
    }

    async fn on_llm_end(
        &self,
        response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut attrs = HashMap::new();
        let generation_count: usize = response.generations.iter().map(|g| g.len()).sum();
        attrs.insert(
            "generation_count".into(),
            serde_json::json!(generation_count),
        );
        if let Some(ref llm_output) = response.llm_output {
            if let Some(token_usage) = llm_output.get("token_usage") {
                attrs.insert("token_usage".into(), token_usage.clone());
            }
        }
        self.end_span(run_id, SpanStatus::Ok, attrs);
        Ok(())
    }

    async fn on_llm_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.end_span(run_id, SpanStatus::Error(error.to_string()), HashMap::new());
        Ok(())
    }

    async fn on_tool_start(
        &self,
        serialized: &Value,
        input_str: &str,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let tool_name = serialized
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mut attrs = HashMap::new();
        attrs.insert("tool_name".into(), Value::String(tool_name.to_string()));
        attrs.insert("input".into(), Value::String(input_str.to_string()));
        self.start_span(run_id, "tool", parent_run_id, attrs);
        Ok(())
    }

    async fn on_tool_end(
        &self,
        output: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut attrs = HashMap::new();
        attrs.insert("output_length".into(), serde_json::json!(output.len()));
        self.end_span(run_id, SpanStatus::Ok, attrs);
        Ok(())
    }

    async fn on_tool_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.end_span(run_id, SpanStatus::Error(error.to_string()), HashMap::new());
        Ok(())
    }

    async fn on_chain_start(
        &self,
        serialized: &Value,
        _inputs: &Value,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let chain_type = serialized
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mut attrs = HashMap::new();
        attrs.insert("chain_type".into(), Value::String(chain_type.to_string()));
        self.start_span(run_id, "chain", parent_run_id, attrs);
        Ok(())
    }

    async fn on_chain_end(
        &self,
        _outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.end_span(run_id, SpanStatus::Ok, HashMap::new());
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.end_span(run_id, SpanStatus::Error(error.to_string()), HashMap::new());
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        _documents: &[Document],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_llm_result(generations_count: usize) -> LLMResult {
        use crate::outputs::Generation;
        let gens: Vec<Generation> = (0..generations_count)
            .map(|i| Generation {
                text: format!("gen_{}", i),
                generation_info: None,
            })
            .collect();
        LLMResult {
            generations: vec![gens],
            llm_output: None,
            run: None,
        }
    }

    #[tokio::test]
    async fn test_llm_start_end_creates_complete_span() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();
        let serialized = serde_json::json!({"name": "gpt-4"});

        handler
            .on_llm_start(&serialized, &["hello".into()], run_id, None)
            .await
            .unwrap();
        handler
            .on_llm_end(&make_llm_result(1), run_id, None)
            .await
            .unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.operation_name, "llm");
        assert!(span.end_time.is_some());
        assert_eq!(span.status, SpanStatus::Ok);
        assert_eq!(
            span.attributes.get("model_name").unwrap(),
            &Value::String("gpt-4".into())
        );
        assert_eq!(
            span.attributes.get("generation_count").unwrap(),
            &serde_json::json!(1)
        );
    }

    #[tokio::test]
    async fn test_llm_error_creates_error_span() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();
        let serialized = serde_json::json!({"name": "gpt-4"});

        handler
            .on_llm_start(&serialized, &["prompt".into()], run_id, None)
            .await
            .unwrap();
        handler
            .on_llm_error("rate limit exceeded", run_id, None)
            .await
            .unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            spans[0].status,
            SpanStatus::Error("rate limit exceeded".into())
        );
        assert!(spans[0].end_time.is_some());
    }

    #[tokio::test]
    async fn test_tool_start_end_creates_span_with_attributes() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();
        let serialized = serde_json::json!({"name": "calculator"});

        handler
            .on_tool_start(&serialized, "2+2", run_id, None)
            .await
            .unwrap();
        handler.on_tool_end("4", run_id, None).await.unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.operation_name, "tool");
        assert_eq!(
            span.attributes.get("tool_name").unwrap(),
            &Value::String("calculator".into())
        );
        assert_eq!(
            span.attributes.get("input").unwrap(),
            &Value::String("2+2".into())
        );
        assert_eq!(
            span.attributes.get("output_length").unwrap(),
            &serde_json::json!(1)
        );
    }

    #[tokio::test]
    async fn test_chain_start_end_creates_span() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();
        let serialized = serde_json::json!({"name": "RetrievalQA"});
        let inputs = serde_json::json!({"question": "what is rust?"});

        handler
            .on_chain_start(&serialized, &inputs, run_id, None)
            .await
            .unwrap();
        handler
            .on_chain_end(&serde_json::json!({"answer": "a language"}), run_id, None)
            .await
            .unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.operation_name, "chain");
        assert_eq!(
            span.attributes.get("chain_type").unwrap(),
            &Value::String("RetrievalQA".into())
        );
        assert!(span.end_time.is_some());
    }

    #[tokio::test]
    async fn test_multiple_operations_create_multiple_spans() {
        let handler = OtelTraceCallbackHandler::new();

        // LLM span
        let llm_id = Uuid::new_v4();
        handler
            .on_llm_start(
                &serde_json::json!({"name": "gpt-4"}),
                &["p".into()],
                llm_id,
                None,
            )
            .await
            .unwrap();
        handler
            .on_llm_end(&make_llm_result(1), llm_id, None)
            .await
            .unwrap();

        // Tool span
        let tool_id = Uuid::new_v4();
        handler
            .on_tool_start(
                &serde_json::json!({"name": "search"}),
                "query",
                tool_id,
                None,
            )
            .await
            .unwrap();
        handler.on_tool_end("result", tool_id, None).await.unwrap();

        // Chain span
        let chain_id = Uuid::new_v4();
        handler
            .on_chain_start(
                &serde_json::json!({"name": "QA"}),
                &serde_json::json!({}),
                chain_id,
                None,
            )
            .await
            .unwrap();
        handler
            .on_chain_end(&serde_json::json!({}), chain_id, None)
            .await
            .unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].operation_name, "llm");
        assert_eq!(spans[1].operation_name, "tool");
        assert_eq!(spans[2].operation_name, "chain");
    }

    #[tokio::test]
    async fn test_to_json_produces_valid_json() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();

        handler
            .on_llm_start(
                &serde_json::json!({"name": "claude"}),
                &["hi".into()],
                run_id,
                None,
            )
            .await
            .unwrap();
        handler
            .on_llm_end(&make_llm_result(2), run_id, None)
            .await
            .unwrap();

        let json = handler.to_json();
        assert!(json.get("resourceSpans").is_some());
        let scope_spans = &json["resourceSpans"][0]["scopeSpans"][0];
        assert_eq!(scope_spans["scope"]["name"], "cognis-core");
        let spans_array = scope_spans["spans"].as_array().unwrap();
        assert_eq!(spans_array.len(), 1);
        assert_eq!(spans_array[0]["operation_name"], "llm");
    }

    #[tokio::test]
    async fn test_span_has_valid_trace_id_and_span_id() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();

        handler
            .on_chain_start(
                &serde_json::json!({"name": "test"}),
                &serde_json::json!({}),
                run_id,
                None,
            )
            .await
            .unwrap();
        handler
            .on_chain_end(&serde_json::json!({}), run_id, None)
            .await
            .unwrap();

        let spans = handler.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];

        // trace_id must be a valid UUID
        assert!(Uuid::parse_str(&span.trace_id).is_ok());
        // span_id must be a valid UUID
        assert!(Uuid::parse_str(&span.span_id).is_ok());
        // trace_id must match the handler's trace_id
        assert_eq!(span.trace_id, handler.trace_id);
        // parent_span_id should be None for a root span
        assert!(span.parent_span_id.is_none());
    }

    #[tokio::test]
    async fn test_clear_removes_completed_spans() {
        let handler = OtelTraceCallbackHandler::new();
        let run_id = Uuid::new_v4();

        handler
            .on_llm_start(
                &serde_json::json!({"name": "m"}),
                &["p".into()],
                run_id,
                None,
            )
            .await
            .unwrap();
        handler
            .on_llm_end(&make_llm_result(1), run_id, None)
            .await
            .unwrap();
        assert_eq!(handler.get_spans().len(), 1);

        handler.clear();
        assert!(handler.get_spans().is_empty());
    }
}
