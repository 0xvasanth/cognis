//! Self-correcting output parsers.
//!
//! [`OutputFixingParser`] wraps an inner [`OutputParser`] and, on parse
//! failure, asks an LLM-like `Runnable<String, String>` to repair the
//! malformed output, then re-parses once. [`RetryParser`] loops the
//! repair-then-parse cycle up to `max_retries` times.
//!
//! Both parsers also implement `Runnable<String, T>` so they slot into
//! a chain (`prompt | model | fixing_parser`).

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Single-shot fix: try the inner parser, and on failure ask `fixer`
/// to rewrite the output, then parse the rewrite. If the second parse
/// also fails, return the second error.
pub struct OutputFixingParser<T, P> {
    inner: P,
    fixer: Arc<dyn Runnable<String, String>>,
    _marker: PhantomData<fn() -> T>,
}

impl<T, P> OutputFixingParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T>,
{
    /// Wrap `inner` with a single repair attempt via `fixer`.
    pub fn new(inner: P, fixer: Arc<dyn Runnable<String, String>>) -> Self {
        Self {
            inner,
            fixer,
            _marker: PhantomData,
        }
    }

    /// Async parse: try once, repair via `fixer` on failure, try once more.
    pub async fn parse_with_fix(&self, text: &str) -> Result<T> {
        match self.inner.parse(text) {
            Ok(v) => Ok(v),
            Err(parse_err) => {
                let format_hint = self
                    .inner
                    .format_instructions()
                    .unwrap_or_else(|| "Return only the requested format.".to_string());
                let prompt = format!(
                    "The previous output failed to parse with error:\n{parse_err}\n\n\
                     Previous output:\n{text}\n\n\
                     Format requirements:\n{format_hint}\n\n\
                     Return a corrected version. Output ONLY the corrected content — no \
                     explanations, no markdown fences."
                );
                let fixed = self
                    .fixer
                    .invoke(prompt, RunnableConfig::default())
                    .await?;
                self.inner.parse(&fixed)
            }
        }
    }
}

impl<T, P> OutputParser<T> for OutputFixingParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T>,
{
    fn parse(&self, text: &str) -> Result<T> {
        // Sync path: cannot call the async fixer, so just defer.
        self.inner.parse(text)
    }
    fn format_instructions(&self) -> Option<String> {
        self.inner.format_instructions()
    }
}

#[async_trait]
impl<T, P> Runnable<String, T> for OutputFixingParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T> + Send + Sync,
{
    async fn invoke(&self, input: String, _config: RunnableConfig) -> Result<T> {
        self.parse_with_fix(&input).await
    }
    fn name(&self) -> &str {
        "OutputFixingParser"
    }
}

/// Looped fix: try the inner parser, ask `fixer` to rewrite on failure,
/// repeat up to `max_retries` extra times. Returns the last parse error
/// if every attempt fails.
pub struct RetryParser<T, P> {
    inner: P,
    fixer: Arc<dyn Runnable<String, String>>,
    max_retries: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T, P> RetryParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T>,
{
    /// Default to 3 repair attempts.
    pub fn new(inner: P, fixer: Arc<dyn Runnable<String, String>>) -> Self {
        Self::with_retries(inner, fixer, 3)
    }

    /// Customize the maximum number of repair attempts. `0` means no retries
    /// — equivalent to calling `inner.parse(...)` directly.
    pub fn with_retries(
        inner: P,
        fixer: Arc<dyn Runnable<String, String>>,
        max_retries: usize,
    ) -> Self {
        Self {
            inner,
            fixer,
            max_retries,
            _marker: PhantomData,
        }
    }

    /// Async parse with up to `max_retries` repair-then-parse cycles.
    pub async fn parse_with_retries(&self, text: &str) -> Result<T> {
        let mut current = text.to_string();
        let mut last_err: Option<CognisError> = None;
        for _ in 0..=self.max_retries {
            match self.inner.parse(&current) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last_err = Some(e);
                    if self.max_retries == 0 {
                        break;
                    }
                    let format_hint = self
                        .inner
                        .format_instructions()
                        .unwrap_or_else(|| "Return only the requested format.".to_string());
                    let prompt = format!(
                        "Previous output failed to parse: {}\n\n\
                         Previous output:\n{current}\n\n\
                         Format requirements:\n{format_hint}\n\n\
                         Return a corrected version. Output ONLY the corrected content.",
                        last_err.as_ref().unwrap()
                    );
                    current = self
                        .fixer
                        .invoke(prompt, RunnableConfig::default())
                        .await?;
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| CognisError::Internal("RetryParser exhausted retries".into())))
    }
}

impl<T, P> OutputParser<T> for RetryParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T>,
{
    fn parse(&self, text: &str) -> Result<T> {
        self.inner.parse(text)
    }
    fn format_instructions(&self) -> Option<String> {
        self.inner.format_instructions()
    }
}

#[async_trait]
impl<T, P> Runnable<String, T> for RetryParser<T, P>
where
    T: Send + 'static,
    P: OutputParser<T> + Send + Sync,
{
    async fn invoke(&self, input: String, _config: RunnableConfig) -> Result<T> {
        self.parse_with_retries(&input).await
    }
    fn name(&self) -> &str {
        "RetryParser"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::compose::lambda;
    use crate::output_parsers::JsonParser;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    fn fixer_returns(value: &'static str) -> Arc<dyn Runnable<String, String>> {
        let v = value.to_string();
        Arc::new(lambda(move |_: String| {
            let v = v.clone();
            async move { Ok::<_, CognisError>(v) }
        }))
    }

    #[tokio::test]
    async fn fixing_parser_repairs_invalid_json() {
        let parser = OutputFixingParser::new(
            JsonParser::<Person>::new(),
            fixer_returns(r#"{"name":"Ada","age":36}"#),
        );
        let bad = r#"{name: Ada, age: 36"#; // malformed
        let p = parser.parse_with_fix(bad).await.unwrap();
        assert_eq!(p, Person { name: "Ada".into(), age: 36 });
    }

    #[tokio::test]
    async fn fixing_parser_passes_through_valid() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let fixer: Arc<dyn Runnable<String, String>> = Arc::new(lambda(move |_: String| {
            let c = calls2.clone();
            async move {
                c.fetch_add(1, Ordering::Relaxed);
                Ok::<_, CognisError>(String::from(r#"{"name":"X","age":0}"#))
            }
        }));
        let parser = OutputFixingParser::new(JsonParser::<Person>::new(), fixer);
        let good = r#"{"name":"Bob","age":42}"#;
        let p = parser.parse_with_fix(good).await.unwrap();
        assert_eq!(p, Person { name: "Bob".into(), age: 42 });
        assert_eq!(calls.load(Ordering::Relaxed), 0, "fixer must not be called for valid input");
    }

    #[tokio::test]
    async fn retry_parser_loops_until_valid() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let a = attempts.clone();
        let fixer: Arc<dyn Runnable<String, String>> = Arc::new(lambda(move |_: String| {
            let a = a.clone();
            async move {
                let n = a.fetch_add(1, Ordering::Relaxed);
                Ok::<_, CognisError>(if n < 2 {
                    "still invalid".into()
                } else {
                    r#"{"name":"Eve","age":29}"#.into()
                })
            }
        }));
        let parser =
            RetryParser::with_retries(JsonParser::<Person>::new(), fixer, 5);
        let p = parser.parse_with_retries("garbage").await.unwrap();
        assert_eq!(p, Person { name: "Eve".into(), age: 29 });
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn retry_parser_returns_last_error_after_exhaustion() {
        let fixer = fixer_returns("still bad");
        let parser =
            RetryParser::with_retries(JsonParser::<Person>::new(), fixer, 2);
        let err = parser.parse_with_retries("garbage").await.unwrap_err();
        // Should surface a parse error, not "exhausted retries".
        assert!(
            !err.to_string().contains("exhausted"),
            "expected a real parse error, got: {err}"
        );
    }
}
