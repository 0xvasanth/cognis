use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// The simplest chain: prompt template + chat model -> response.
///
/// The prompt template uses `{variable}` placeholders that are filled
/// from the input JSON object before being sent to the model as a human message.
pub struct LLMChain {
    model: Arc<dyn BaseChatModel>,
    prompt_template: String,
    output_key: String,
}

/// Builder for [`LLMChain`].
pub struct LLMChainBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    prompt_template: Option<String>,
    output_key: String,
}

impl LLMChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            prompt_template: None,
            output_key: "text".to_string(),
        }
    }

    /// Set the chat model (required).
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the prompt template string with `{variable}` placeholders (required).
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = Some(prompt.into());
        self
    }

    /// Set the output key for the result map. Default: `"text"`.
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Build the [`LLMChain`]. Panics if model or prompt is not set.
    pub fn build(self) -> LLMChain {
        LLMChain {
            model: self.model.expect("model is required for LLMChain"),
            prompt_template: self
                .prompt_template
                .expect("prompt is required for LLMChain"),
            output_key: self.output_key,
        }
    }
}

impl Default for LLMChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMChain {
    /// Create a new builder.
    pub fn builder() -> LLMChainBuilder {
        LLMChainBuilder::new()
    }

    /// Format the prompt template by replacing `{variable}` placeholders with
    /// values from the input JSON object.
    fn format_prompt(&self, input: &Value) -> Result<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let obj = input.as_object().ok_or_else(|| CognisError::TypeMismatch {
            expected: "JSON object".into(),
            got: format!("{}", input),
        })?;

        let mut missing: Vec<String> = Vec::new();
        let result = re.replace_all(&self.prompt_template, |caps: &regex::Captures| {
            let key = &caps[1];
            match obj.get(key) {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => {
                    missing.push(key.to_string());
                    String::new()
                }
            }
        });

        if !missing.is_empty() {
            return Err(CognisError::InvalidKey(format!(
                "Missing input variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(result.into_owned())
    }
}

#[async_trait]
impl Runnable for LLMChain {
    fn name(&self) -> &str {
        "LLMChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let formatted = self.format_prompt(&input)?;
        let messages = vec![Message::Human(HumanMessage::new(&formatted))];
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        let text = ai_msg.base.content.text();
        Ok(json!({ &self.output_key: text }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::FakeListChatModel;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    #[tokio::test]
    async fn test_llm_chain_basic() {
        let chain = LLMChain::builder()
            .model(fake_model(vec!["The answer is 4"]))
            .prompt("What is {question}?")
            .build();

        let result = chain
            .invoke(json!({"question": "2+2"}), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "The answer is 4");
    }

    #[tokio::test]
    async fn test_llm_chain_multiple_variables() {
        let chain = LLMChain::builder()
            .model(fake_model(vec!["Paris is the capital of France"]))
            .prompt("What is the {attribute} of {country}?")
            .build();

        let result = chain
            .invoke(json!({"attribute": "capital", "country": "France"}), None)
            .await
            .unwrap();
        assert_eq!(result["text"], "Paris is the capital of France");
    }

    #[tokio::test]
    async fn test_llm_chain_missing_variable() {
        let chain = LLMChain::builder()
            .model(fake_model(vec!["response"]))
            .prompt("Tell me about {topic} in {language}")
            .build();

        let result = chain.invoke(json!({"topic": "rust"}), None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("language"),
            "Error should mention missing key: {err}"
        );
    }

    #[tokio::test]
    async fn test_llm_chain_as_runnable() {
        let chain = LLMChain::builder()
            .model(fake_model(vec!["42"]))
            .prompt("Answer: {q}")
            .output_key("answer")
            .build();

        let runnable: &dyn Runnable = &chain;
        let result = runnable
            .invoke(json!({"q": "meaning of life"}), None)
            .await
            .unwrap();
        assert_eq!(result["answer"], "42");
        assert_eq!(runnable.name(), "LLMChain");
    }
}
