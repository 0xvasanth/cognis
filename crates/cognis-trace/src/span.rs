//! Core span / generation / score types. Aligned with Langfuse's
//! observation model so the Langfuse exporter is a 1:1 mapping. See
//! the spec, §4.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of an observation. Aligned with Langfuse's `ObservationType`
/// enum so values cross-walk directly. See spec §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SpanKind {
    /// Generic span (chains, custom blocks).
    Span,
    /// LLM generation. Has a `Some(generation)` payload.
    Generation,
    /// Discrete event (no duration semantics).
    Event,
    /// Agent-loop span.
    Agent,
    /// Tool execution.
    Tool,
    /// Logical chain block.
    Chain,
    /// Retriever step.
    Retriever,
    /// Embedding step.
    Embedding,
    /// Guardrail / safety check.
    Guardrail,
}

/// Observation severity, matching Langfuse's `ObservationLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ObservationLevel {
    /// Normal completion.
    Default,
    /// Diagnostic.
    Debug,
    /// Recoverable issue.
    Warning,
    /// Failure.
    Error,
}

impl Default for ObservationLevel {
    fn default() -> Self {
        Self::Default
    }
}

/// One node in the trace tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// This observation's id.
    pub run_id: Uuid,
    /// Parent observation id, when nested.
    pub parent_run_id: Option<Uuid>,
    /// Root run_id of the trace this span belongs to.
    pub trace_id: Uuid,
    /// Observation type.
    pub kind: SpanKind,
    /// Friendly name (e.g. "openai.gpt-4o", "search_tool", "agent_node").
    pub name: String,
    /// Wall-clock start.
    pub started_at: SystemTime,
    /// Wall-clock end (None until span closes).
    pub ended_at: Option<SystemTime>,
    /// Severity.
    pub level: ObservationLevel,
    /// Set when level == Warning or Error.
    pub status_message: Option<String>,
    /// Input payload (Value to be backend-agnostic).
    pub input: Option<serde_json::Value>,
    /// Output payload.
    pub output: Option<serde_json::Value>,
    /// Set only on the trace root.
    pub session_id: Option<String>,
    /// Set only on the trace root.
    pub user_id: Option<String>,
    /// Set only on the trace root.
    pub tags: Vec<String>,
    /// Free-form metadata (excluding well-known fields above).
    pub metadata: HashMap<String, serde_json::Value>,
    /// Some iff `kind == Generation`.
    pub generation: Option<Generation>,
}

/// LLM-specific data carried on a `Generation` span.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Generation {
    /// Concrete model id (e.g. "gpt-4o-2024-08-06").
    pub model: String,
    /// Provider tag (e.g. "openai", "anthropic").
    pub provider: String,
    /// Provider-supplied request parameters (temperature, max_tokens, …).
    #[serde(default)]
    pub model_parameters: HashMap<String, serde_json::Value>,
    /// Token usage broken out by category.
    pub usage: TokenUsage,
    /// USD cost, structured. None when the model is not in the price table.
    pub cost: Option<CostDetails>,
    /// Time-to-first-token, when the provider reports it.
    pub completion_start_time: Option<SystemTime>,
    /// Provider's stop reason ("stop", "length", "tool_calls", …).
    pub finish_reason: Option<String>,
    /// Set when the prompt came from a `PromptStore`.
    pub prompt_name: Option<String>,
    /// Set when the prompt came from a `PromptStore`.
    pub prompt_version: Option<u32>,
}

/// Token counts broken out by category.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens (uncached portion).
    pub input: u32,
    /// Output tokens.
    pub output: u32,
    /// Cached input tokens that were read.
    pub cache_read: u32,
    /// Cached input tokens written this call.
    pub cache_write: u32,
}

impl TokenUsage {
    /// Sum of all token categories.
    pub fn total(&self) -> u32 {
        self.input + self.output + self.cache_read + self.cache_write
    }
}

/// USD cost, broken out by token category.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CostDetails {
    /// Input-token cost.
    pub input: f64,
    /// Output-token cost.
    pub output: f64,
    /// Cache-read cost.
    pub cache_read: f64,
    /// Cache-write cost.
    pub cache_write: f64,
    /// Convenience: input + output + cache_read + cache_write.
    pub total: f64,
}

/// Numeric, categorical, or boolean evaluation score.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScoreValue {
    /// Continuous numeric score.
    Numeric(f64),
    /// Categorical label (1–500 chars per Langfuse contract).
    Categorical(String),
    /// Boolean — serialized as 1.0 / 0.0 by Langfuse.
    Boolean(bool),
}

/// Evaluation score attached to a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRecord {
    /// The observation this score applies to.
    pub run_id: Uuid,
    /// Optional explicit trace pointer (for cross-trace scores).
    pub trace_id: Option<Uuid>,
    /// Optional session pointer (session-level scores).
    pub session_id: Option<String>,
    /// Score name (e.g. "novelty", "factuality").
    pub name: String,
    /// The score value.
    pub value: ScoreValue,
    /// Optional human comment.
    pub comment: Option<String>,
}

/// In-flight span being assembled between `on_*_start` and `on_*_end`.
/// Pure data; no behavior — `TracingHandler` drives it.
#[derive(Debug, Clone)]
pub struct SpanBuilder {
    /// The span being assembled. `ended_at`, `output`, `level`, `status_message`,
    /// and (for Generation) `generation` are filled at close time.
    pub span: Span,
}

impl SpanBuilder {
    /// Open a new span at `now`.
    pub fn open(
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
        trace_id: Uuid,
        kind: SpanKind,
        name: impl Into<String>,
        input: Option<serde_json::Value>,
        now: SystemTime,
    ) -> Self {
        Self {
            span: Span {
                run_id,
                parent_run_id,
                trace_id,
                kind,
                name: name.into(),
                started_at: now,
                ended_at: None,
                level: ObservationLevel::Default,
                status_message: None,
                input,
                output: None,
                session_id: None,
                user_id: None,
                tags: Vec::new(),
                metadata: HashMap::new(),
                generation: None,
            },
        }
    }

    /// Mark the span ended successfully and stamp the output.
    pub fn finish_ok(mut self, output: Option<serde_json::Value>, now: SystemTime) -> Span {
        self.span.ended_at = Some(now);
        self.span.output = output;
        self.span
    }

    /// Mark the span ended with an error.
    pub fn finish_error(mut self, message: impl Into<String>, now: SystemTime) -> Span {
        self.span.ended_at = Some(now);
        self.span.level = ObservationLevel::Error;
        self.span.status_message = Some(message.into());
        self.span
    }

    /// Attach a populated `Generation` payload (used on `on_llm_end`).
    pub fn with_generation(mut self, gen: Generation) -> Self {
        self.span.kind = SpanKind::Generation;
        self.span.generation = Some(gen);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_kind_serializes_uppercase() {
        let s = serde_json::to_string(&SpanKind::Generation).unwrap();
        assert_eq!(s, "\"GENERATION\"");
    }

    #[test]
    fn observation_level_default_is_default() {
        assert_eq!(ObservationLevel::default(), ObservationLevel::Default);
    }

    #[test]
    fn token_usage_total_sums_categories() {
        let u = TokenUsage {
            input: 10,
            output: 20,
            cache_read: 5,
            cache_write: 3,
        };
        assert_eq!(u.total(), 38);
    }

    #[test]
    fn score_value_numeric_serializes_as_number() {
        let v = ScoreValue::Numeric(0.9);
        assert_eq!(serde_json::to_string(&v).unwrap(), "0.9");
    }

    #[test]
    fn score_value_categorical_serializes_as_string() {
        let v = ScoreValue::Categorical("good".into());
        assert_eq!(serde_json::to_string(&v).unwrap(), "\"good\"");
    }

    #[test]
    fn span_builder_opens_with_default_level() {
        let id = Uuid::new_v4();
        let now = SystemTime::now();
        let b = SpanBuilder::open(id, None, id, SpanKind::Chain, "x", None, now);
        assert_eq!(b.span.level, ObservationLevel::Default);
        assert!(b.span.ended_at.is_none());
    }

    #[test]
    fn span_builder_finish_ok_sets_end_and_output() {
        let id = Uuid::new_v4();
        let now = SystemTime::now();
        let b = SpanBuilder::open(id, None, id, SpanKind::Chain, "x", None, now);
        let span = b.finish_ok(Some(serde_json::json!({"k": "v"})), now);
        assert!(span.ended_at.is_some());
        assert_eq!(span.level, ObservationLevel::Default);
        assert_eq!(span.output, Some(serde_json::json!({"k": "v"})));
    }

    #[test]
    fn span_builder_finish_error_sets_level_and_message() {
        let id = Uuid::new_v4();
        let now = SystemTime::now();
        let b = SpanBuilder::open(id, None, id, SpanKind::Tool, "t", None, now);
        let span = b.finish_error("boom", now);
        assert_eq!(span.level, ObservationLevel::Error);
        assert_eq!(span.status_message.as_deref(), Some("boom"));
    }

    #[test]
    fn span_builder_with_generation_flips_kind() {
        let id = Uuid::new_v4();
        let b = SpanBuilder::open(id, None, id, SpanKind::Span, "g", None, SystemTime::now())
            .with_generation(Generation {
                model: "gpt-4o".into(),
                provider: "openai".into(),
                ..Default::default()
            });
        assert_eq!(b.span.kind, SpanKind::Generation);
        assert!(b.span.generation.is_some());
    }
}
