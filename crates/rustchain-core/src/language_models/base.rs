use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::messages::Message;
use crate::outputs::{ChatResult, LLMResult};

/// Core interface for all language models (both LLMs and chat models).
///
/// This is the base trait shared by `BaseLLM` and `BaseChatModel`.
#[async_trait]
pub trait BaseLanguageModel: Send + Sync {
    /// Generate completions for the given prompts.
    async fn generate(&self, prompts: &[String]) -> Result<LLMResult>;

    /// Generate chat completions for the given messages.
    async fn generate_chat(&self, messages: &[Vec<Message>]) -> Result<ChatResult>;

    /// Predict the next token/completion for a single prompt.
    async fn predict(&self, text: &str) -> Result<String> {
        let result = self.generate(&[text.to_string()]).await?;
        Ok(result
            .generations
            .first()
            .and_then(|gens| gens.first())
            .map(|g| g.text.clone())
            .unwrap_or_default())
    }

    /// Get the number of tokens in the given text.
    fn get_num_tokens(&self, text: &str) -> usize {
        // Default rough estimate: ~4 chars per token
        text.len() / 4
    }

    /// Get identifying parameters of this model.
    fn identifying_params(&self) -> HashMap<String, Value> {
        HashMap::new()
    }

    /// Get the type of language model.
    fn model_type(&self) -> &str;
}
