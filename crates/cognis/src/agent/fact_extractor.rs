//! LLM-driven structured extraction primitives.
//!
//! Two public types:
//!
//! * [`LlmExtractor<O>`] — the generic building block. Give it a system
//!   prompt and any `DeserializeOwned + JsonSchema` output type and it
//!   calls the model, auto-generates format instructions from the schema,
//!   and parses the response. `Runnable<String, O>`.
//!
//! * [`FactExtractor`] — a ready-to-use specialisation of `LlmExtractor`
//!   for the common "distil agent output into atomic memory facts" case.
//!   Takes [`FactExtractionInput`] and returns `Vec<`[`Fact`]`>`. Parse
//!   failures are logged and swallowed (returns `Ok(vec![])`) so the
//!   memory write path never stalls on a badly-formatted model response.
//!   `Runnable<FactExtractionInput, Vec<Fact>>`.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cognis_core::output_parsers::{OutputParser, StructuredOutputConfig, StructuredOutputParser};
use cognis_core::{Message, Result, Runnable, RunnableConfig};
use cognis_llm::chat::ChatOptions;
use cognis_llm::Client;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LlmExtractor<O>
// ---------------------------------------------------------------------------

/// Generic structured extractor: renders text input into an LLM prompt,
/// calls the model, and parses the response as `O`.
///
/// The JSON Schema for `O` is derived automatically via [`JsonSchema`] and
/// appended to the system prompt as format instructions. Models that
/// respond with leading prose still parse correctly because
/// [`StructuredOutputParser`] scans for the first balanced JSON
/// object/array by default.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::agent::{LlmExtractor, LlmExtractorBuilder};
/// use cognis_llm::{Client, ClientBuilder, Provider};
/// use serde::Deserialize;
/// use schemars::JsonSchema;
/// use std::sync::Arc;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct Sentiment { label: String, score: f32 }
///
/// let client = Arc::new(ClientBuilder::new().provider(Provider::Anthropic).build()?);
/// let extractor = LlmExtractor::<Sentiment>::builder(client)
///     .system_prompt("Classify the sentiment of the following text.")
///     .build();
///
/// let result = extractor.invoke("I love this!".into(), Default::default()).await?;
/// ```
pub struct LlmExtractor<O> {
    client: Arc<Client>,
    system_prompt: String,
    parser: StructuredOutputParser<O>,
}

impl<O> Clone for LlmExtractor<O> {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            system_prompt: self.system_prompt.clone(),
            parser: self.parser.clone(),
        }
    }
}

impl<O> LlmExtractor<O> {
    /// Start building a new extractor backed by `client`.
    pub fn builder(client: Arc<Client>) -> LlmExtractorBuilder<O> {
        LlmExtractorBuilder {
            client,
            system_prompt: None,
            parser_config: None,
            _out: PhantomData,
        }
    }
}

/// Builder for [`LlmExtractor`].
pub struct LlmExtractorBuilder<O> {
    client: Arc<Client>,
    system_prompt: Option<String>,
    parser_config: Option<StructuredOutputConfig>,
    _out: PhantomData<fn() -> O>,
}

impl<O> LlmExtractorBuilder<O> {
    /// Override the system-level instruction. Format instructions derived
    /// from the output schema are always appended, so you only need to
    /// describe the *task* here.
    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = Some(p.into());
        self
    }

    /// Override the JSON extraction config (e.g. switch to strict mode or
    /// provide a custom extractor for tag-delimited responses).
    pub fn parser_config(mut self, c: StructuredOutputConfig) -> Self {
        self.parser_config = Some(c);
        self
    }

    /// Build the extractor.
    pub fn build(self) -> LlmExtractor<O> {
        let system_prompt = self.system_prompt.unwrap_or_else(|| {
            "You are a precise information extraction assistant. \
             Extract only what is explicitly present in the provided text."
                .to_string()
        });
        let parser = match self.parser_config {
            Some(cfg) => StructuredOutputParser::with_config(cfg),
            None => StructuredOutputParser::new(),
        };
        LlmExtractor {
            client: self.client,
            system_prompt,
            parser,
        }
    }
}

#[async_trait]
impl<O> Runnable<String, O> for LlmExtractor<O>
where
    O: serde::de::DeserializeOwned + JsonSchema + Send + 'static,
{
    /// Send `text` to the model as the human turn. The system prompt
    /// already includes schema-derived format instructions.
    async fn invoke(&self, text: String, _: RunnableConfig) -> Result<O> {
        let instructions = OutputParser::format_instructions(&self.parser).unwrap_or_default();
        let system = format!("{}\n\n{instructions}", self.system_prompt);
        let messages = vec![Message::system(system), Message::human(text)];
        let resp = self.client.chat(messages, ChatOptions::default()).await?;
        OutputParser::parse(&self.parser, resp.message.content())
    }

    fn name(&self) -> &str {
        "LlmExtractor"
    }
}

// ---------------------------------------------------------------------------
// FactKind + Fact + FactExtractionInput
// ---------------------------------------------------------------------------

/// Semantic category of an extracted fact.
///
/// Choosing the right kind helps downstream systems filter or prioritise
/// facts before writing to a memory store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    /// A standing decision that must always be followed.
    Rule,
    /// A softer guideline; follow unless there is a reason not to.
    Preference,
    /// Situational information that informs decisions without being prescriptive.
    Context,
    /// A past choice with its rationale — informative, not prescriptive for the future.
    Decision,
    /// An ongoing state worth being aware of.
    Observation,
}

/// One atomic, self-contained fact extracted from agent output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Fact {
    /// Self-contained statement that is interpretable without the surrounding
    /// text it was extracted from.
    pub content: String,
    /// Semantic category — used for filtering and routing in memory systems.
    pub kind: FactKind,
    /// Relative importance 0.0–1.0. Higher values surface the fact first
    /// during retrieval ranking.
    pub importance: f32,
}

/// Input to [`FactExtractor::invoke`].
#[derive(Debug, Clone)]
pub struct FactExtractionInput {
    /// The raw text to distil — typically the full output of an agent turn.
    pub text: String,
    /// Optional hints that frame the extraction context, e.g.
    /// `"project: billing-v2"` or `"user: alice"`. Rendered as a bulleted
    /// section above the text so the model can filter for relevance.
    pub context_hints: Vec<String>,
    /// Upper bound on the number of facts returned. Defaults to 7.
    pub max_facts: usize,
}

impl FactExtractionInput {
    /// Minimal constructor — no hints, default `max_facts = 7`.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            context_hints: vec![],
            max_facts: 7,
        }
    }

    /// Append a context hint (builder-style, chainable).
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.context_hints.push(hint.into());
        self
    }

    /// Override the maximum number of returned facts (builder-style).
    pub fn with_max_facts(mut self, n: usize) -> Self {
        self.max_facts = n;
        self
    }
}

impl Default for FactExtractionInput {
    fn default() -> Self {
        Self::new("")
    }
}

// ---------------------------------------------------------------------------
// FactExtractor
// ---------------------------------------------------------------------------

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are extracting reusable memory facts from an AI agent's completed work.

Extract only atomic, self-contained facts — each must be understandable
without the surrounding text it came from.

Classify each fact as exactly one of:
  rule        — a standing decision that must always be followed
  preference  — a softer guideline; follow unless there is a reason not to
  context     — situational information that informs decisions
  decision    — a past choice with its rationale (informative, not prescriptive)
  observation — an ongoing state worth being aware of

Rate importance 0.0–1.0:
  1.0 = critical invariant that affects every future action
  0.5 = moderately useful background
  0.0 = borderline; included only for completeness

If the output contains no useful facts, return an empty JSON array: []";

/// Distils raw agent output into a ranked list of atomic [`Fact`]s.
///
/// Built on [`LlmExtractor<Vec<Fact>>`]. Any parse or model failure is
/// logged at WARN and swallowed — the caller always receives `Ok(vec![])`
/// rather than an error. This is intentional: the memory write path must
/// never block because of a badly-formatted model response.
///
/// # Customisation
///
/// Override the system prompt or parser config via [`FactExtractor::builder`].
/// The default prompt is tuned for general-purpose agents; for domain-specific
/// extraction (e.g. medical, legal, financial), replacing the prompt via
/// `FactExtractorBuilder::system_prompt` is often sufficient.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::agent::{FactExtractor, FactExtractionInput};
/// use cognis_llm::{Client, ClientBuilder, Provider};
/// use std::sync::Arc;
///
/// let client = Arc::new(ClientBuilder::new().provider(Provider::Anthropic).build()?);
/// let extractor = FactExtractor::new(client);
///
/// let facts = extractor.invoke(
///     FactExtractionInput::new("Chose monolithic architecture for the API service.")
///         .with_hint("project: billing-v2")
///         .with_max_facts(5),
///     Default::default(),
/// ).await?;
/// ```
pub struct FactExtractor {
    inner: LlmExtractor<Vec<Fact>>,
}

impl FactExtractor {
    /// Create with the default extraction prompt.
    pub fn new(client: Arc<Client>) -> Self {
        Self::builder(client).build()
    }

    /// Start building a customised [`FactExtractor`].
    pub fn builder(client: Arc<Client>) -> FactExtractorBuilder {
        FactExtractorBuilder {
            client,
            system_prompt: None,
            parser_config: None,
        }
    }

    fn render_user_message(input: &FactExtractionInput) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !input.context_hints.is_empty() {
            let hints = input
                .context_hints
                .iter()
                .map(|h| format!("- {h}"))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("Context:\n{hints}"));
        }

        parts.push(format!("Agent output:\n---\n{}\n---", input.text));
        parts.push(format!(
            "Extract up to {} atomic facts. Return a JSON array only — no prose.",
            input.max_facts
        ));

        parts.join("\n\n")
    }
}

/// Builder for [`FactExtractor`].
pub struct FactExtractorBuilder {
    client: Arc<Client>,
    system_prompt: Option<String>,
    parser_config: Option<StructuredOutputConfig>,
}

impl FactExtractorBuilder {
    /// Override the extraction system prompt. The JSON schema hint is
    /// still appended automatically; you only need to describe the task.
    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = Some(p.into());
        self
    }

    /// Override the JSON parser config.
    pub fn parser_config(mut self, c: StructuredOutputConfig) -> Self {
        self.parser_config = Some(c);
        self
    }

    /// Build the extractor.
    pub fn build(self) -> FactExtractor {
        let mut builder = LlmExtractor::<Vec<Fact>>::builder(self.client).system_prompt(
            self.system_prompt
                .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string()),
        );
        if let Some(cfg) = self.parser_config {
            builder = builder.parser_config(cfg);
        }
        FactExtractor {
            inner: builder.build(),
        }
    }
}

#[async_trait]
impl Runnable<FactExtractionInput, Vec<Fact>> for FactExtractor {
    async fn invoke(
        &self,
        input: FactExtractionInput,
        config: RunnableConfig,
    ) -> Result<Vec<Fact>> {
        let max = input.max_facts;
        let user_msg = Self::render_user_message(&input);
        let facts = match self.inner.invoke(user_msg, config).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "fact_extractor: extraction failed, returning empty vec");
                vec![]
            }
        };
        Ok(facts.into_iter().take(max).collect())
    }

    fn name(&self) -> &str {
        "FactExtractor"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::RunnableStream;
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
    use cognis_llm::provider::LLMProvider;
    use cognis_llm::{Client, Provider};

    /// Test provider that returns a fixed response string.
    struct CannedProvider {
        response: String,
    }

    #[async_trait]
    impl LLMProvider for CannedProvider {
        fn name(&self) -> &str {
            "canned"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.response.clone()),
                usage: None,
                finish_reason: "stop".into(),
                model: "canned".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn canned_client(response: impl Into<String>) -> Arc<Client> {
        Arc::new(Client::new(Arc::new(CannedProvider {
            response: response.into(),
        })))
    }

    // --- LlmExtractor ---

    #[tokio::test]
    async fn llm_extractor_parses_structured_output() {
        #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
        struct Sentiment {
            label: String,
            score: f32,
        }

        let json = r#"{"label":"positive","score":0.95}"#;
        let extractor = LlmExtractor::<Sentiment>::builder(canned_client(json))
            .system_prompt("Classify sentiment.")
            .build();

        let result = extractor
            .invoke("I love this product!".into(), Default::default())
            .await
            .unwrap();

        assert_eq!(result.label, "positive");
        assert!((result.score - 0.95).abs() < 1e-4);
    }

    #[tokio::test]
    async fn llm_extractor_extracts_json_from_prose() {
        #[derive(Deserialize, JsonSchema)]
        struct Answer {
            value: i32,
        }

        // Model wraps JSON in prose — parser should still find it.
        let response = r#"Sure! Here is the answer: {"value": 42} Hope that helps."#;
        let extractor = LlmExtractor::<Answer>::builder(canned_client(response)).build();
        let out = extractor
            .invoke("What is 6×7?".into(), Default::default())
            .await
            .unwrap();
        assert_eq!(out.value, 42);
    }

    #[tokio::test]
    async fn llm_extractor_propagates_parse_error() {
        #[derive(Deserialize, JsonSchema)]
        struct Answer {
            _value: i32,
        }

        // Not JSON at all.
        let extractor = LlmExtractor::<Answer>::builder(canned_client("not json")).build();
        let result = extractor.invoke("input".into(), Default::default()).await;
        assert!(result.is_err());
    }

    // --- FactExtractor ---

    #[tokio::test]
    async fn fact_extractor_parses_facts() {
        let json = r#"[
            {"content":"Use monolithic architecture for the API","kind":"rule","importance":0.9},
            {"content":"Prefer Rust for performance-critical paths","kind":"preference","importance":0.7}
        ]"#;

        let extractor = FactExtractor::new(canned_client(json));
        let facts = extractor
            .invoke(
                FactExtractionInput::new("Chose monolithic architecture."),
                Default::default(),
            )
            .await
            .unwrap();

        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].kind, FactKind::Rule);
        assert_eq!(facts[1].kind, FactKind::Preference);
    }

    #[tokio::test]
    async fn fact_extractor_returns_empty_vec_on_parse_failure() {
        let extractor = FactExtractor::new(canned_client("this is not json at all"));
        let facts = extractor
            .invoke(FactExtractionInput::new("some text"), Default::default())
            .await
            .unwrap();
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn fact_extractor_respects_max_facts() {
        // Model returns 5 facts but caller requested max 2.
        let json = r#"[
            {"content":"A","kind":"rule","importance":1.0},
            {"content":"B","kind":"context","importance":0.8},
            {"content":"C","kind":"decision","importance":0.6},
            {"content":"D","kind":"observation","importance":0.4},
            {"content":"E","kind":"preference","importance":0.2}
        ]"#;
        let extractor = FactExtractor::new(canned_client(json));
        let facts = extractor
            .invoke(
                FactExtractionInput::new("lots of info").with_max_facts(2),
                Default::default(),
            )
            .await
            .unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[tokio::test]
    async fn fact_extractor_returns_empty_on_model_empty_array() {
        let extractor = FactExtractor::new(canned_client("[]"));
        let facts = extractor
            .invoke(
                FactExtractionInput::new("no useful info"),
                Default::default(),
            )
            .await
            .unwrap();
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn render_user_message_includes_context_hints() {
        let input = FactExtractionInput::new("output text")
            .with_hint("project: billing-v2")
            .with_hint("user: alice");
        let rendered = FactExtractor::render_user_message(&input);
        assert!(rendered.contains("project: billing-v2"));
        assert!(rendered.contains("user: alice"));
        assert!(rendered.contains("output text"));
    }

    #[tokio::test]
    async fn render_user_message_omits_context_section_when_no_hints() {
        let input = FactExtractionInput::new("plain text");
        let rendered = FactExtractor::render_user_message(&input);
        assert!(!rendered.contains("Context:"));
        assert!(rendered.contains("plain text"));
    }

    #[tokio::test]
    async fn fact_extraction_input_builder_chain() {
        let input = FactExtractionInput::new("text")
            .with_hint("hint1")
            .with_hint("hint2")
            .with_max_facts(3);
        assert_eq!(input.context_hints.len(), 2);
        assert_eq!(input.max_facts, 3);
    }
}
