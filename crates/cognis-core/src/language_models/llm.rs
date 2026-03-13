use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::outputs::LLMResult;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;
use crate::runnables::RunnableStream;

use super::base::BaseLanguageModel;

/// Trait for text-completion language models (string in, string out).
///
/// Implementors must provide `_generate`. Optionally override `_stream`
/// for streaming support.
#[async_trait]
pub trait BaseLLM: BaseLanguageModel {
    /// Core generation method. Implementors must override this.
    async fn _generate(&self, prompts: &[String], stop: Option<&[String]>) -> Result<LLMResult>;

    /// Optional streaming support. Default returns `NotImplemented`.
    async fn _stream(&self, _prompt: &str, _stop: Option<&[String]>) -> Result<RunnableStream> {
        Err(crate::error::CognisError::NotImplemented(
            "Streaming not supported for this LLM".into(),
        ))
    }

    /// The LLM type identifier for logging/caching.
    fn llm_type(&self) -> &str;
}

/// Helper to extract a prompt string from a `Value` input.
pub fn extract_prompt(input: &serde_json::Value) -> String {
    match input {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => map
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        other => other.to_string(),
    }
}

/// Runnable wrapper for `BaseLLM`, bridging text-completion models into LCEL chains.
///
/// Accepts `Value::String` (or extracts prompt from object). Returns `Value::String`.
pub struct LLMRunnable {
    llm: Arc<dyn BaseLLM>,
    name: String,
}

impl LLMRunnable {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        let name = format!("LLMRunnable({})", llm.llm_type());
        Self { llm, name }
    }
}

#[async_trait]
impl Runnable for LLMRunnable {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let prompt = extract_prompt(&input);
        let result = self.llm._generate(&[prompt], None).await?;
        let text = result
            .generations
            .first()
            .and_then(|gens| gens.first())
            .map(|g| g.text.clone())
            .unwrap_or_default();
        Ok(Value::String(text))
    }

    async fn stream(
        &self,
        input: Value,
        _config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let prompt = extract_prompt(&input);
        self.llm._stream(&prompt, None).await
    }
}

/// Marker trait combining `BaseLLM` + `BaseLanguageModel` for `LLMRunnable`.
/// This lets us use `dyn BaseLLM` since `BaseLLM: BaseLanguageModel`.
fn _assert_llm_runnable_is_send() {
    fn _assert<T: Send + Sync>() {}
    _assert::<LLMRunnable>();
}
