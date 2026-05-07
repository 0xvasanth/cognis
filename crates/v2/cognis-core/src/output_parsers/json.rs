//! Typed JSON parser — deserializes LLM output into any `DeserializeOwned`.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Parses JSON output into a typed value `T`.
///
/// Tolerates code-fenced output (` ```json ... ``` ` or plain ` ``` ... ``` `)
/// by stripping the fences before parsing. Tolerates leading/trailing
/// whitespace.
#[derive(Debug, Clone, Copy)]
pub struct JsonParser<T> {
    _t: PhantomData<fn() -> T>,
}

impl<T> Default for JsonParser<T> {
    fn default() -> Self {
        Self { _t: PhantomData }
    }
}

impl<T> JsonParser<T> {
    /// Construct a `JsonParser<T>`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> OutputParser<T> for JsonParser<T>
where
    T: DeserializeOwned + Send + 'static,
{
    fn parse(&self, text: &str) -> Result<T> {
        let cleaned = strip_code_fence(text.trim());
        serde_json::from_str(cleaned)
            .map_err(|e| CognisError::Serialization(format!("json parse: {e}")))
    }

    fn format_instructions(&self) -> Option<String> {
        Some(
            "Reply with a single JSON object. Do not include any text before \
             or after the JSON. Do not wrap the JSON in markdown code fences."
                .to_string(),
        )
    }
}

#[async_trait]
impl<T> Runnable<String, T> for JsonParser<T>
where
    T: DeserializeOwned + Send + 'static,
{
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<T> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "JsonParser"
    }
}

fn strip_code_fence(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    let s = s.trim_start_matches('\n').trim_start();
    s.strip_suffix("```").unwrap_or(s).trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Foo {
        bar: String,
        baz: u32,
    }

    #[tokio::test]
    async fn parses_plain_json() {
        let p: JsonParser<Foo> = JsonParser::new();
        let out = p
            .invoke(r#"{"bar":"hi","baz":7}"#.into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(
            out,
            Foo {
                bar: "hi".into(),
                baz: 7
            }
        );
    }

    #[test]
    fn strips_json_code_fence() {
        let p: JsonParser<Foo> = JsonParser::new();
        let out = p.parse("```json\n{\"bar\":\"hi\",\"baz\":7}\n```").unwrap();
        assert_eq!(out.baz, 7);
    }

    #[test]
    fn strips_plain_code_fence() {
        let p: JsonParser<Foo> = JsonParser::new();
        let out = p.parse("```\n{\"bar\":\"hi\",\"baz\":1}\n```").unwrap();
        assert_eq!(out.baz, 1);
    }

    #[test]
    fn invalid_json_errors_with_serialization_kind() {
        let p: JsonParser<Foo> = JsonParser::new();
        let err = p.parse("not json").unwrap_err();
        assert!(matches!(err, CognisError::Serialization(_)));
    }

    #[test]
    fn format_instructions_set() {
        let p: JsonParser<Foo> = JsonParser::new();
        assert!(OutputParser::format_instructions(&p).is_some());
    }
}
