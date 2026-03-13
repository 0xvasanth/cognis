use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::Result;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// Run multiple runnables in sequence, passing output of one as input to the next.
///
/// State accumulates across steps: the output of each chain is merged into the
/// running state object. At the end, the full accumulated state is returned
/// (optionally filtered by `output_keys`).
pub struct SequentialChain {
    chains: Vec<Arc<dyn Runnable>>,
    #[allow(dead_code)]
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

/// Builder for [`SequentialChain`].
pub struct SequentialChainBuilder {
    chains: Vec<Arc<dyn Runnable>>,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

impl SequentialChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            chains: Vec::new(),
            input_keys: Vec::new(),
            output_keys: Vec::new(),
        }
    }

    /// Add a single chain to the sequence.
    pub fn chain(mut self, chain: Arc<dyn Runnable>) -> Self {
        self.chains.push(chain);
        self
    }

    /// Add multiple chains to the sequence.
    pub fn chains(mut self, chains: Vec<Arc<dyn Runnable>>) -> Self {
        self.chains.extend(chains);
        self
    }

    /// Set the expected input keys (optional, for documentation/validation).
    pub fn input_keys(mut self, keys: Vec<String>) -> Self {
        self.input_keys = keys;
        self
    }

    /// Set the output keys to filter the final result (optional).
    pub fn output_keys(mut self, keys: Vec<String>) -> Self {
        self.output_keys = keys;
        self
    }

    /// Build the [`SequentialChain`].
    pub fn build(self) -> SequentialChain {
        SequentialChain {
            chains: self.chains,
            input_keys: self.input_keys,
            output_keys: self.output_keys,
        }
    }
}

impl Default for SequentialChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialChain {
    /// Create a new builder.
    pub fn builder() -> SequentialChainBuilder {
        SequentialChainBuilder::new()
    }
}

/// Merge the entries from `source` object into `target` object.
/// Non-object values are left unchanged.
fn merge_objects(target: &mut Value, source: Value) {
    if let (Some(target_map), Some(source_map)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_map {
            target_map.insert(k.clone(), v.clone());
        }
    }
}

#[async_trait]
impl Runnable for SequentialChain {
    fn name(&self) -> &str {
        "SequentialChain"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        if self.chains.is_empty() {
            return Ok(input);
        }

        let mut state = input;

        for chain in &self.chains {
            let result = chain.invoke(state.clone(), config).await?;
            merge_objects(&mut state, result);
        }

        // Filter by output_keys if specified
        if !self.output_keys.is_empty() {
            if let Some(map) = state.as_object() {
                let filtered: serde_json::Map<String, Value> = self
                    .output_keys
                    .iter()
                    .filter_map(|k| map.get(k).map(|v| (k.clone(), v.clone())))
                    .collect();
                return Ok(Value::Object(filtered));
            }
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chains::llm::LLMChain;
    use cognis_core::language_models::fake::FakeListChatModel;
    use serde_json::json;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    use cognis_core::language_models::chat_model::BaseChatModel;

    #[tokio::test]
    async fn test_sequential_two_chains() {
        // Chain 1: takes "topic" -> produces "text" (a sentence)
        let chain1 = Arc::new(
            LLMChain::builder()
                .model(fake_model(vec!["Rust is a systems programming language"]))
                .prompt("Tell me about {topic}")
                .build(),
        ) as Arc<dyn Runnable>;

        // Chain 2: takes "text" -> produces "summary"
        let chain2 = Arc::new(
            LLMChain::builder()
                .model(fake_model(vec!["Rust: systems lang"]))
                .prompt("Summarize: {text}")
                .output_key("summary")
                .build(),
        ) as Arc<dyn Runnable>;

        let seq = SequentialChain::builder()
            .chain(chain1)
            .chain(chain2)
            .build();

        let result = seq.invoke(json!({"topic": "Rust"}), None).await.unwrap();
        assert_eq!(result["summary"], "Rust: systems lang");
    }

    #[tokio::test]
    async fn test_sequential_state_accumulates() {
        let chain1 = Arc::new(
            LLMChain::builder()
                .model(fake_model(vec!["first output"]))
                .prompt("{input}")
                .output_key("step1")
                .build(),
        ) as Arc<dyn Runnable>;

        let chain2 = Arc::new(
            LLMChain::builder()
                .model(fake_model(vec!["second output"]))
                .prompt("{step1}")
                .output_key("step2")
                .build(),
        ) as Arc<dyn Runnable>;

        let seq = SequentialChain::builder()
            .chain(chain1)
            .chain(chain2)
            .build();

        let result = seq.invoke(json!({"input": "hello"}), None).await.unwrap();
        // All keys should be present
        assert_eq!(result["input"], "hello");
        assert_eq!(result["step1"], "first output");
        assert_eq!(result["step2"], "second output");
    }

    #[tokio::test]
    async fn test_sequential_as_runnable() {
        let chain1 = Arc::new(
            LLMChain::builder()
                .model(fake_model(vec!["response"]))
                .prompt("{q}")
                .build(),
        ) as Arc<dyn Runnable>;

        let seq = SequentialChain::builder().chain(chain1).build();

        let runnable: &dyn Runnable = &seq;
        assert_eq!(runnable.name(), "SequentialChain");
        let result = runnable.invoke(json!({"q": "test"}), None).await.unwrap();
        assert_eq!(result["text"], "response");
    }
}
