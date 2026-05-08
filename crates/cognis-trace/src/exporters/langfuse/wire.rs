//! Wire-format types matching Langfuse's `/api/public/ingestion` schema
//! (verified against the Langfuse OpenAPI spec, 2026-05-06).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Top-level ingestion request.
#[derive(Debug, Serialize)]
pub struct IngestionRequest {
    /// Discriminated event list.
    pub batch: Vec<IngestionEvent>,
}

/// Discriminated by `type`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum IngestionEvent {
    /// Trace root.
    TraceCreate {
        /// Envelope id (deduplication key).
        id: String,
        /// Envelope timestamp.
        timestamp: String,
        /// Trace body.
        body: TraceBody,
    },
    /// Generic span.
    SpanCreate {
        /// Envelope id.
        id: String,
        /// Envelope timestamp.
        timestamp: String,
        /// Span body.
        body: SpanBody,
    },
    /// LLM generation.
    GenerationCreate {
        /// Envelope id.
        id: String,
        /// Envelope timestamp.
        timestamp: String,
        /// Generation body.
        body: GenerationBody,
    },
    /// Out-of-band score.
    ScoreCreate {
        /// Envelope id.
        id: String,
        /// Envelope timestamp.
        timestamp: String,
        /// Score body.
        body: ScoreBody,
    },
}

/// Langfuse `TraceBody`.
#[derive(Debug, Serialize)]
pub struct TraceBody {
    /// Trace id (= our `trace_id`).
    pub id: String,
    /// Wall-clock ISO-8601.
    pub timestamp: String,
    /// Optional trace name.
    pub name: Option<String>,
    /// User id.
    #[serde(rename = "userId")]
    pub user_id: Option<String>,
    /// Trace input.
    pub input: Option<serde_json::Value>,
    /// Trace output.
    pub output: Option<serde_json::Value>,
    /// Session id.
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
    /// Release tag.
    pub release: Option<String>,
    /// Version tag.
    pub version: Option<String>,
    /// Free-form metadata.
    pub metadata: Option<serde_json::Value>,
    /// Tags.
    pub tags: Option<Vec<String>>,
    /// Environment tag.
    pub environment: Option<String>,
    /// Public visibility.
    pub public: Option<bool>,
}

/// Common observation fields used by Span and Generation.
#[derive(Debug, Serialize)]
pub struct SpanBody {
    /// Observation id.
    pub id: String,
    /// Owning trace id.
    #[serde(rename = "traceId")]
    pub trace_id: String,
    /// Parent observation id.
    #[serde(rename = "parentObservationId", skip_serializing_if = "Option::is_none")]
    pub parent_observation_id: Option<String>,
    /// Friendly name.
    pub name: Option<String>,
    /// ISO-8601 start.
    #[serde(rename = "startTime")]
    pub start_time: Option<String>,
    /// ISO-8601 end.
    #[serde(rename = "endTime", skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    /// Severity.
    pub level: Option<String>,
    /// Status message (when level != DEFAULT).
    #[serde(rename = "statusMessage", skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    /// Free-form metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    /// Output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    /// Environment tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
}

/// Generation-specific body. Wraps `SpanBody` (Langfuse's body inherits).
#[derive(Debug, Serialize)]
pub struct GenerationBody {
    /// Common fields.
    #[serde(flatten)]
    pub span: SpanBody,
    /// TTFT.
    #[serde(rename = "completionStartTime", skip_serializing_if = "Option::is_none")]
    pub completion_start_time: Option<String>,
    /// Concrete model id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Provider request parameters.
    #[serde(rename = "modelParameters", skip_serializing_if = "Option::is_none")]
    pub model_parameters: Option<HashMap<String, serde_json::Value>>,
    /// Per-category usage (Langfuse keys: `input`, `output`,
    /// `cache_read_input`, `cache_creation_input`, `total`).
    #[serde(rename = "usageDetails", skip_serializing_if = "Option::is_none")]
    pub usage_details: Option<HashMap<String, u64>>,
    /// Per-category cost.
    #[serde(rename = "costDetails", skip_serializing_if = "Option::is_none")]
    pub cost_details: Option<HashMap<String, f64>>,
    /// Prompt name.
    #[serde(rename = "promptName", skip_serializing_if = "Option::is_none")]
    pub prompt_name: Option<String>,
    /// Prompt version.
    #[serde(rename = "promptVersion", skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<u32>,
}

/// Score body.
#[derive(Debug, Serialize)]
pub struct ScoreBody {
    /// Score id.
    pub id: String,
    /// Owning trace id.
    #[serde(rename = "traceId", skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Owning observation id.
    #[serde(rename = "observationId", skip_serializing_if = "Option::is_none")]
    pub observation_id: Option<String>,
    /// Owning session id.
    #[serde(rename = "sessionId", skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Score name.
    pub name: String,
    /// Score value (numeric / categorical / boolean).
    pub value: serde_json::Value,
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
}

/// Response shape: 207 with successes + per-event errors.
#[derive(Debug, Deserialize)]
pub struct IngestionResponse {
    /// Successful events.
    #[serde(default)]
    pub successes: Vec<IngestionSuccess>,
    /// Failed events.
    #[serde(default)]
    pub errors: Vec<IngestionError>,
}

/// One successful envelope.
#[derive(Debug, Deserialize)]
pub struct IngestionSuccess {
    /// Envelope id.
    pub id: String,
    /// Per-event status.
    pub status: u16,
}

/// One failed envelope.
#[derive(Debug, Deserialize)]
pub struct IngestionError {
    /// Envelope id.
    pub id: String,
    /// Per-event status.
    pub status: u16,
    /// Error message.
    pub message: Option<String>,
}

/// Format a `SystemTime` as RFC 3339 UTC for Langfuse.
pub fn format_iso8601(t: std::time::SystemTime) -> String {
    let dt = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = dt.as_secs() as i64;
    let nanos = dt.subsec_nanos();
    // Use a small dependency-free RFC3339 formatter.
    // Year/month/day from secs:
    let (year, month, day, hour, minute, second) = secs_to_ymd_hms(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hour, minute, second, nanos / 1_000_000
    )
}

fn secs_to_ymd_hms(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    // Simple algorithm — accurate to seconds for 1970..2100.
    let mut s = secs;
    let day_secs = 86_400;
    let mut days = (s / day_secs) as i64;
    s -= days * day_secs;
    if s < 0 {
        days -= 1;
        s += day_secs;
    }
    let hour = (s / 3600) as u32;
    let minute = ((s / 60) % 60) as u32;
    let second = (s % 60) as u32;

    let mut year = 1970i32;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let year_days = if leap { 366 } else { 365 };
        if days >= year_days as i64 {
            days -= year_days as i64;
            year += 1;
        } else {
            break;
        }
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_lens = if leap {
        [31u32, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31u32, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u32;
    for &m in &month_lens {
        if (days as u32) >= m {
            days -= m as i64;
            month += 1;
        } else {
            break;
        }
    }
    let day = (days as u32) + 1;
    (year, month, day, hour, minute, second)
}

/// Generate a fresh envelope id.
pub fn envelope_id() -> String {
    Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, Duration};

    #[test]
    fn format_iso8601_unix_epoch() {
        let t = SystemTime::UNIX_EPOCH;
        assert_eq!(format_iso8601(t), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn format_iso8601_2026_05_06() {
        // 2026-05-06T00:00:00 UTC
        let secs: u64 = 1_778_025_600;
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        assert_eq!(format_iso8601(t), "2026-05-06T00:00:00.000Z");
    }

    #[test]
    fn trace_create_serializes_with_kebab_type() {
        let ev = IngestionEvent::TraceCreate {
            id: "e1".into(),
            timestamp: "2026-05-06T00:00:00.000Z".into(),
            body: TraceBody {
                id: "t1".into(),
                timestamp: "2026-05-06T00:00:00.000Z".into(),
                name: Some("hello".into()),
                user_id: None,
                input: None,
                output: None,
                session_id: None,
                release: None,
                version: None,
                metadata: None,
                tags: None,
                environment: None,
                public: None,
            },
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains("\"type\":\"trace-create\""));
    }
}
