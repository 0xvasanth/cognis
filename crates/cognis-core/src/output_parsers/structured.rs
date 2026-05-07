//! Schema-driven structured output parser — accepts any
//! `T: DeserializeOwned + JsonSchema`, builds an LLM prompt fragment from
//! the schema, and parses the model's text response into `T`.
//!
//! Differs from [`super::JsonParser`] in two ways:
//! 1. `format_instructions()` returns a *schema-aware* prompt fragment
//!    (the JSON Schema rendered as a hint), not a generic "reply with
//!    JSON" string.
//! 2. Tolerates JSON embedded in surrounding prose by extracting the
//!    largest balanced object/array; useful for chatty models.
//!
//! Customization: the parser is built from a [`StructuredOutputConfig`]
//! that lets callers swap the prompt template and the JSON-extraction
//! strategy without subclassing the parser type.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Strategy for locating a JSON value inside a string that may also
/// contain prose. The default is [`JsonExtraction::FirstBalanced`] which
/// picks the first `{...}` or `[...]` substring whose braces balance.
///
/// Implement [`JsonExtractor`] for fully custom logic (e.g. parsing a
/// specific markdown section, or stripping `<json>...</json>` tags).
#[derive(Clone)]
pub enum JsonExtraction {
    /// First balanced `{...}` or `[...]` block. Tolerates leading prose,
    /// markdown fences, and trailing apologies.
    FirstBalanced,
    /// Treat the entire trimmed input as JSON. Strict; useful when you've
    /// instructed the model in the prompt to emit only JSON.
    Strict,
    /// User-supplied extractor. The closure receives the raw model text
    /// and returns a `&str` slice that should be JSON.
    Custom(Arc<dyn JsonExtractor>),
}

impl std::fmt::Debug for JsonExtraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstBalanced => f.write_str("FirstBalanced"),
            Self::Strict => f.write_str("Strict"),
            Self::Custom(_) => f.write_str("Custom(<extractor>)"),
        }
    }
}

/// Object-safe extractor trait — implement to plug a custom JSON-locator.
pub trait JsonExtractor: Send + Sync {
    /// Locate a JSON substring inside `text`.
    /// Returns the slice that should be passed to `serde_json::from_str`.
    /// Return `None` to fall through to a `Serialization` error.
    fn extract<'a>(&self, text: &'a str) -> Option<&'a str>;
}

/// Default extractor: first balanced `{...}` / `[...]` block.
fn extract_first_balanced(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    // Find the first `{` or `[`.
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            x if x == open => depth += 1,
            x if x == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Function passed to [`StructuredOutputConfig::with_format_template`].
/// Receives the rendered JSON Schema as a string and returns the prompt
/// fragment to embed.
pub type FormatTemplate = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Configuration knobs exposed to callers without forcing parser-subclass
/// gymnastics. Defaults are tuned for general-purpose chat models.
#[derive(Clone)]
pub struct StructuredOutputConfig {
    extraction: JsonExtraction,
    format_template: Option<FormatTemplate>,
}

impl Default for StructuredOutputConfig {
    fn default() -> Self {
        Self {
            extraction: JsonExtraction::FirstBalanced,
            format_template: None,
        }
    }
}

impl std::fmt::Debug for StructuredOutputConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredOutputConfig")
            .field("extraction", &self.extraction)
            .field("format_template", &self.format_template.is_some())
            .finish()
    }
}

impl StructuredOutputConfig {
    /// Construct with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the JSON extraction strategy.
    pub fn with_extraction(mut self, e: JsonExtraction) -> Self {
        self.extraction = e;
        self
    }

    /// Provide a custom prompt-fragment template. The closure receives
    /// the JSON Schema rendered as a pretty-printed string and returns
    /// the prompt text to embed.
    pub fn with_format_template<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.format_template = Some(Arc::new(f));
        self
    }
}

/// Schema-driven output parser.
///
/// Build a parser via:
/// ```ignore
/// use serde::Deserialize;
/// use schemars::JsonSchema;
/// use cognis_core::output_parsers::StructuredOutputParser;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct Answer { topic: String, summary: String }
///
/// let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
/// let instructions = p.format_instructions().unwrap();
/// // ... feed instructions into your prompt, then parse model output:
/// let parsed: Answer = p.parse(model_output).unwrap();
/// ```
pub struct StructuredOutputParser<T> {
    config: StructuredOutputConfig,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for StructuredOutputParser<T> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            _t: PhantomData,
        }
    }
}

impl<T> Default for StructuredOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StructuredOutputParser<T> {
    /// New parser with default config.
    pub fn new() -> Self {
        Self {
            config: StructuredOutputConfig::default(),
            _t: PhantomData,
        }
    }

    /// New parser with caller-supplied config.
    pub fn with_config(config: StructuredOutputConfig) -> Self {
        Self {
            config,
            _t: PhantomData,
        }
    }

    /// Borrow the active config (read-only).
    pub fn config(&self) -> &StructuredOutputConfig {
        &self.config
    }
}

impl<T> StructuredOutputParser<T>
where
    T: JsonSchema,
{
    /// Render the JSON Schema for `T` as a pretty string. Used by
    /// `format_instructions` and exposed for advanced callers that want
    /// to embed the schema themselves.
    pub fn schema_string(&self) -> String {
        let schema = schemars::schema_for!(T);
        serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".to_string())
    }
}

impl<T> OutputParser<T> for StructuredOutputParser<T>
where
    T: DeserializeOwned + JsonSchema + Send + 'static,
{
    fn parse(&self, text: &str) -> Result<T> {
        let trimmed = text.trim();
        let candidate: &str = match &self.config.extraction {
            JsonExtraction::Strict => trimmed,
            JsonExtraction::FirstBalanced => extract_first_balanced(trimmed).ok_or_else(|| {
                CognisError::Serialization(
                    "structured parser: no balanced JSON object/array found in output".into(),
                )
            })?,
            JsonExtraction::Custom(extractor) => extractor.extract(trimmed).ok_or_else(|| {
                CognisError::Serialization(
                    "structured parser: custom extractor returned None".into(),
                )
            })?,
        };
        serde_json::from_str(candidate)
            .map_err(|e| CognisError::Serialization(format!("structured parser: deserialize: {e}")))
    }

    fn format_instructions(&self) -> Option<String> {
        let schema = self.schema_string();
        if let Some(tmpl) = &self.config.format_template {
            return Some(tmpl(&schema));
        }
        Some(format!(
            "Reply with a single JSON value that conforms to this schema. \
             Do not include any prose, markdown fences, or commentary outside \
             the JSON.\n\nSchema:\n{schema}"
        ))
    }
}

#[async_trait]
impl<T> Runnable<String, T> for StructuredOutputParser<T>
where
    T: DeserializeOwned + JsonSchema + Send + 'static,
{
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<T> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "StructuredOutputParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct Answer {
        topic: String,
        steps: Vec<String>,
    }

    #[test]
    fn parses_clean_json() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
        let out = p.parse(r#"{"topic":"rust","steps":["a","b"]}"#).unwrap();
        assert_eq!(out.topic, "rust");
        assert_eq!(out.steps, vec!["a".to_string(), "b".into()]);
    }

    #[test]
    fn extracts_balanced_json_from_prose() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
        let text = r#"Sure! Here is the answer:
{"topic":"rust","steps":["x"]}
Hope that helps!"#;
        let out = p.parse(text).unwrap();
        assert_eq!(out.topic, "rust");
    }

    #[test]
    fn handles_nested_braces() {
        #[derive(Deserialize, JsonSchema)]
        struct Wrap {
            outer: serde_json::Value,
        }
        let p: StructuredOutputParser<Wrap> = StructuredOutputParser::new();
        let text = r#"prelude {"outer":{"a":{"b":1}},"extra":"ignored"} suffix"#;
        let out = p.parse(text).unwrap();
        assert_eq!(out.outer["a"]["b"], 1);
    }

    #[test]
    fn strict_mode_rejects_prose() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::with_config(
            StructuredOutputConfig::new().with_extraction(JsonExtraction::Strict),
        );
        let err = p
            .parse(r#"prelude {"topic":"x","steps":[]} suffix"#)
            .unwrap_err();
        assert!(matches!(err, CognisError::Serialization(_)));
    }

    #[test]
    fn custom_extractor_used() {
        struct TagExtractor;
        impl JsonExtractor for TagExtractor {
            fn extract<'a>(&self, text: &'a str) -> Option<&'a str> {
                let start = text.find("<json>")? + "<json>".len();
                let end = text.find("</json>")?;
                Some(&text[start..end])
            }
        }
        let cfg = StructuredOutputConfig::new()
            .with_extraction(JsonExtraction::Custom(Arc::new(TagExtractor)));
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::with_config(cfg);
        let out = p
            .parse(r#"see <json>{"topic":"x","steps":[]}</json> done"#)
            .unwrap();
        assert_eq!(out.topic, "x");
    }

    #[test]
    fn format_instructions_includes_schema() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
        let s = OutputParser::format_instructions(&p).unwrap();
        assert!(s.contains("\"topic\""));
        assert!(s.contains("\"steps\""));
    }

    #[test]
    fn custom_format_template_is_used() {
        let cfg = StructuredOutputConfig::new()
            .with_format_template(|schema| format!("<custom>{schema}</custom>"));
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::with_config(cfg);
        let s = OutputParser::format_instructions(&p).unwrap();
        assert!(s.starts_with("<custom>"));
        assert!(s.ends_with("</custom>"));
    }

    #[test]
    fn invalid_json_returns_serialization_error() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
        let err = p.parse("plain text, no JSON here").unwrap_err();
        assert!(matches!(err, CognisError::Serialization(_)));
    }

    #[test]
    fn ignores_braces_inside_strings() {
        let p: StructuredOutputParser<Answer> = StructuredOutputParser::new();
        // Embedded `{` inside a string must not break the brace counter.
        let out = p.parse(r#"{"topic":"a {nested} b","steps":[]}"#).unwrap();
        assert_eq!(out.topic, "a {nested} b");
    }
}
