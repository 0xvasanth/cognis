//! Few-shot prompt templates with static and dynamic example selection.
//!
//! Provides [`FewShotPromptTemplate`] for constructing prompts that include
//! examples (few-shot learning). Examples can be supplied statically or
//! selected dynamically via an [`ExampleSelector`] implementation such as
//! [`SemanticSimilaritySelector`] or [`LengthBasedSelector`].

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::embeddings::Embeddings;
use cognis_core::error::{CognisError, Result};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// A single example represented as a map of field names to values.
pub type Example = HashMap<String, String>;

/// Trait for dynamically selecting examples based on the input.
pub trait ExampleSelector: Send + Sync {
    /// Select the most relevant examples for the given input.
    fn select_examples(&self, input: &str) -> Vec<Example>;

    /// Add an example to the selector's pool.
    fn add_example(&mut self, example: Example);
}

/// Selects examples by semantic similarity using an embeddings model.
///
/// Stores examples alongside their pre-computed embedding vectors and returns
/// the top-k most similar examples to the query at format time.
pub struct SemanticSimilaritySelector {
    embeddings: Box<dyn Embeddings>,
    examples: Vec<Example>,
    example_embeddings: Vec<Vec<f32>>,
    /// The key in each example whose value is embedded for comparison.
    input_key: String,
    k: usize,
    /// Cached runtime for blocking on async embed calls within the sync trait.
    runtime: tokio::runtime::Runtime,
}

impl SemanticSimilaritySelector {
    /// Create a new builder for `SemanticSimilaritySelector`.
    pub fn builder(embeddings: Box<dyn Embeddings>) -> SemanticSimilaritySelectorBuilder {
        SemanticSimilaritySelectorBuilder {
            embeddings,
            examples: Vec::new(),
            input_key: "input".to_string(),
            k: 3,
        }
    }
}

/// Builder for [`SemanticSimilaritySelector`].
pub struct SemanticSimilaritySelectorBuilder {
    embeddings: Box<dyn Embeddings>,
    examples: Vec<Example>,
    input_key: String,
    k: usize,
}

impl SemanticSimilaritySelectorBuilder {
    /// Set the number of examples to return.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the key in each example whose value is used for embedding comparison.
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Provide initial examples.
    pub fn examples(mut self, examples: Vec<Example>) -> Self {
        self.examples = examples;
        self
    }

    /// Build the selector, computing embeddings for all initial examples.
    pub fn build(self) -> SemanticSimilaritySelector {
        let runtime = tokio::runtime::Runtime::new()
            .expect("failed to create tokio runtime for SemanticSimilaritySelector");

        let texts: Vec<String> = self
            .examples
            .iter()
            .map(|ex| ex.get(&self.input_key).cloned().unwrap_or_default())
            .collect();

        let example_embeddings = if texts.is_empty() {
            Vec::new()
        } else {
            runtime
                .block_on(self.embeddings.embed_documents(texts))
                .unwrap_or_default()
        };

        SemanticSimilaritySelector {
            embeddings: self.embeddings,
            examples: self.examples,
            example_embeddings,
            input_key: self.input_key,
            k: self.k,
            runtime,
        }
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl ExampleSelector for SemanticSimilaritySelector {
    fn select_examples(&self, input: &str) -> Vec<Example> {
        let query_embedding = self
            .runtime
            .block_on(self.embeddings.embed_query(input))
            .unwrap_or_default();

        let mut scored: Vec<(usize, f32)> = self
            .example_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(self.k)
            .map(|(i, _)| self.examples[i].clone())
            .collect()
    }

    fn add_example(&mut self, example: Example) {
        let text = example.get(&self.input_key).cloned().unwrap_or_default();
        let embedding = self
            .runtime
            .block_on(self.embeddings.embed_query(&text))
            .unwrap_or_default();
        self.examples.push(example);
        self.example_embeddings.push(embedding);
    }
}

/// Selects examples that fit within a character-length budget.
///
/// Examples are added in order until the cumulative formatted length would
/// exceed the configured maximum.
pub struct LengthBasedSelector {
    examples: Vec<Example>,
    max_length: usize,
    /// Template used to format each example for length measurement.
    example_template: String,
}

impl LengthBasedSelector {
    /// Create a new builder for `LengthBasedSelector`.
    pub fn builder() -> LengthBasedSelectorBuilder {
        LengthBasedSelectorBuilder {
            examples: Vec::new(),
            max_length: 1000,
            example_template: String::new(),
        }
    }

    /// Format a single example using the example template.
    fn format_example(&self, example: &Example) -> String {
        if self.example_template.is_empty() {
            // Fallback: concatenate key=value pairs
            example
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            format_template_simple(&self.example_template, example)
        }
    }
}

/// Builder for [`LengthBasedSelector`].
pub struct LengthBasedSelectorBuilder {
    examples: Vec<Example>,
    max_length: usize,
    example_template: String,
}

impl LengthBasedSelectorBuilder {
    /// Set the maximum character length budget.
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Provide initial examples.
    pub fn examples(mut self, examples: Vec<Example>) -> Self {
        self.examples = examples;
        self
    }

    /// Set the template used to format examples for length measurement.
    pub fn example_template(mut self, template: impl Into<String>) -> Self {
        self.example_template = template.into();
        self
    }

    /// Build the selector.
    pub fn build(self) -> LengthBasedSelector {
        LengthBasedSelector {
            examples: self.examples,
            max_length: self.max_length,
            example_template: self.example_template,
        }
    }
}

impl ExampleSelector for LengthBasedSelector {
    fn select_examples(&self, _input: &str) -> Vec<Example> {
        let mut selected = Vec::new();
        let mut total_length = 0;

        for example in &self.examples {
            let formatted = self.format_example(example);
            let new_length = total_length + formatted.len();
            if new_length > self.max_length {
                break;
            }
            total_length = new_length;
            selected.push(example.clone());
        }

        selected
    }

    fn add_example(&mut self, example: Example) {
        self.examples.push(example);
    }
}

/// A few-shot prompt template that formats examples into the prompt.
///
/// Supports both static examples and dynamic selection via an
/// [`ExampleSelector`]. The final prompt is assembled as:
///
/// ```text
/// {prefix}
/// {example_separator}
/// {formatted_example_1}
/// {example_separator}
/// {formatted_example_2}
/// {example_separator}
/// {suffix}  (with input variables substituted)
/// ```
pub struct FewShotPromptTemplate {
    /// Text before the examples.
    pub prefix: String,
    /// Text after the examples (usually contains `{input}` or similar placeholders).
    pub suffix: String,
    /// Template for formatting each individual example.
    pub example_template: String,
    /// Separator placed between formatted examples.
    pub example_separator: String,
    /// Dynamic example selector (takes precedence over static examples).
    pub selector: Option<Box<dyn ExampleSelector>>,
    /// Static examples (used when no selector is set).
    pub examples: Option<Vec<Example>>,
    /// The key used to look up the input text for the selector.
    pub input_key: String,
}

impl FewShotPromptTemplate {
    /// Create a new builder for `FewShotPromptTemplate`.
    pub fn builder() -> FewShotPromptTemplateBuilder {
        FewShotPromptTemplateBuilder {
            prefix: String::new(),
            suffix: String::new(),
            example_template: String::new(),
            example_separator: "\n\n".to_string(),
            selector: None,
            examples: None,
            input_key: "input".to_string(),
        }
    }

    /// Format the few-shot prompt with the given input variables.
    pub fn format(&self, input_variables: &HashMap<String, String>) -> Result<String> {
        let examples = self.get_examples(input_variables)?;
        let formatted_examples: Vec<String> = examples
            .iter()
            .map(|ex| format_template_simple(&self.example_template, ex))
            .collect();

        let mut parts = Vec::new();

        if !self.prefix.is_empty() {
            parts.push(self.prefix.clone());
        }

        if !formatted_examples.is_empty() {
            parts.push(formatted_examples.join(&self.example_separator));
        }

        if !self.suffix.is_empty() {
            let rendered_suffix = format_template_simple(&self.suffix, input_variables);
            parts.push(rendered_suffix);
        }

        Ok(parts.join(&self.example_separator))
    }

    /// Get the examples to use, either from the selector or the static list.
    fn get_examples(&self, input_variables: &HashMap<String, String>) -> Result<Vec<Example>> {
        if let Some(ref selector) = self.selector {
            let input_text = input_variables
                .get(&self.input_key)
                .cloned()
                .unwrap_or_default();
            Ok(selector.select_examples(&input_text))
        } else if let Some(ref examples) = self.examples {
            Ok(examples.clone())
        } else {
            Ok(Vec::new())
        }
    }
}

#[async_trait]
impl Runnable for FewShotPromptTemplate {
    fn name(&self) -> &str {
        "FewShotPromptTemplate"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, String> = match input {
            Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| {
                    let s = match v {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    (k, s)
                })
                .collect(),
            _ => {
                return Err(CognisError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let text = self.format(&kwargs)?;
        Ok(Value::String(text))
    }
}

/// Builder for [`FewShotPromptTemplate`].
pub struct FewShotPromptTemplateBuilder {
    prefix: String,
    suffix: String,
    example_template: String,
    example_separator: String,
    selector: Option<Box<dyn ExampleSelector>>,
    examples: Option<Vec<Example>>,
    input_key: String,
}

impl FewShotPromptTemplateBuilder {
    /// Set the prefix text that appears before examples.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the suffix text that appears after examples.
    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Set the template for formatting each example.
    pub fn example_template(mut self, template: impl Into<String>) -> Self {
        self.example_template = template.into();
        self
    }

    /// Set the separator placed between formatted examples.
    pub fn example_separator(mut self, separator: impl Into<String>) -> Self {
        self.example_separator = separator.into();
        self
    }

    /// Set a dynamic example selector.
    pub fn selector(mut self, selector: Box<dyn ExampleSelector>) -> Self {
        self.selector = Some(selector);
        self
    }

    /// Set static examples.
    pub fn examples(mut self, examples: Vec<Example>) -> Self {
        self.examples = Some(examples);
        self
    }

    /// Set the input key used to look up the query text for the selector.
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Build the `FewShotPromptTemplate`.
    pub fn build(self) -> FewShotPromptTemplate {
        FewShotPromptTemplate {
            prefix: self.prefix,
            suffix: self.suffix,
            example_template: self.example_template,
            example_separator: self.example_separator,
            selector: self.selector,
            examples: self.examples,
            input_key: self.input_key,
        }
    }
}

/// Simple template string formatting: replaces `{key}` with values from the map.
/// Unknown placeholders are left as-is.
fn format_template_simple(template: &str, variables: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                result.push('{');
                continue;
            }
            let mut name = String::new();
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
                name.push(inner);
            }
            if let Some(value) = variables.get(&name) {
                result.push_str(value);
            } else {
                // Leave placeholder as-is if not found
                result.push('{');
                result.push_str(&name);
                result.push('}');
            }
        } else if ch == '}' {
            if chars.peek() == Some(&'}') {
                chars.next();
                result.push('}');
            } else {
                result.push('}');
            }
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_example(input: &str, output: &str) -> Example {
        let mut ex = HashMap::new();
        ex.insert("input".to_string(), input.to_string());
        ex.insert("output".to_string(), output.to_string());
        ex
    }

    // ---------- Static examples formatting ----------

    #[test]
    fn test_static_examples_formatting() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Translate English to French:")
            .suffix("Input: {input}\nOutput:")
            .example_template("Input: {input}\nOutput: {output}")
            .examples(vec![
                make_example("hello", "bonjour"),
                make_example("goodbye", "au revoir"),
            ])
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "thank you".to_string());

        let result = template.format(&vars).unwrap();
        assert!(result.contains("Translate English to French:"));
        assert!(result.contains("Input: hello\nOutput: bonjour"));
        assert!(result.contains("Input: goodbye\nOutput: au revoir"));
        assert!(result.contains("Input: thank you\nOutput:"));
    }

    // ---------- Dynamic example selection with SemanticSimilaritySelector ----------

    #[test]
    fn test_semantic_similarity_selector() {
        let embeddings = Box::new(DeterministicFakeEmbedding::new(64));

        let examples = vec![
            make_example("cat", "chat"),
            make_example("dog", "chien"),
            make_example("house", "maison"),
            make_example("car", "voiture"),
            make_example("tree", "arbre"),
        ];

        let selector = SemanticSimilaritySelector::builder(embeddings)
            .k(2)
            .examples(examples)
            .build();

        let selected = selector.select_examples("cat");
        assert_eq!(selected.len(), 2);
        // The first result for "cat" input should be the "cat" example itself
        // since it has identical embedding.
        assert_eq!(selected[0].get("input").unwrap(), "cat");
    }

    #[test]
    fn test_semantic_selector_with_few_shot_template() {
        let embeddings = Box::new(DeterministicFakeEmbedding::new(64));

        let examples = vec![
            make_example("hello", "bonjour"),
            make_example("goodbye", "au revoir"),
            make_example("thank you", "merci"),
            make_example("please", "s'il vous plait"),
        ];

        let selector = SemanticSimilaritySelector::builder(embeddings)
            .k(2)
            .examples(examples)
            .build();

        let template = FewShotPromptTemplate::builder()
            .prefix("Translate English to French:")
            .suffix("Input: {input}\nOutput:")
            .example_template("Input: {input}\nOutput: {output}")
            .selector(Box::new(selector))
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "hello".to_string());

        let result = template.format(&vars).unwrap();
        assert!(result.contains("Translate English to French:"));
        // Should contain the "hello" example since it matches exactly
        assert!(result.contains("Input: hello\nOutput: bonjour"));
        assert!(result.contains("Input: hello\nOutput:"));
    }

    // ---------- LengthBasedSelector respects budget ----------

    #[test]
    fn test_length_based_selector_respects_budget() {
        let examples = vec![
            make_example("a", "1"),       // "Input: a\nOutput: 1" = 20 chars
            make_example("bb", "22"),     // "Input: bb\nOutput: 22" = 22 chars
            make_example("ccc", "333"),   // "Input: ccc\nOutput: 333" = 24 chars
            make_example("dddd", "4444"), // won't fit
        ];

        let selector = LengthBasedSelector::builder()
            .examples(examples)
            .example_template("Input: {input}\nOutput: {output}")
            .max_length(66)
            .build();

        let selected = selector.select_examples("anything");
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].get("input").unwrap(), "a");
        assert_eq!(selected[1].get("input").unwrap(), "bb");
        assert_eq!(selected[2].get("input").unwrap(), "ccc");
    }

    #[test]
    fn test_length_based_selector_empty_when_budget_zero() {
        let examples = vec![make_example("a", "1")];

        let selector = LengthBasedSelector::builder()
            .examples(examples)
            .example_template("Input: {input}\nOutput: {output}")
            .max_length(0)
            .build();

        let selected = selector.select_examples("anything");
        assert!(selected.is_empty());
    }

    // ---------- Custom example template ----------

    #[test]
    fn test_custom_example_template() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Examples:")
            .suffix("Q: {input}\nA:")
            .example_template("Q: {input}\nA: {output}")
            .examples(vec![
                make_example("What is 1+1?", "2"),
                make_example("What is 2+2?", "4"),
            ])
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "What is 3+3?".to_string());

        let result = template.format(&vars).unwrap();
        assert!(result.contains("Q: What is 1+1?\nA: 2"));
        assert!(result.contains("Q: What is 2+2?\nA: 4"));
        assert!(result.contains("Q: What is 3+3?\nA:"));
    }

    // ---------- Custom separator ----------

    #[test]
    fn test_custom_separator() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Start")
            .suffix("End")
            .example_template("{input} -> {output}")
            .example_separator("\n---\n")
            .examples(vec![make_example("a", "1"), make_example("b", "2")])
            .build();

        let vars = HashMap::new();
        let result = template.format(&vars).unwrap();
        assert_eq!(result, "Start\n---\na -> 1\n---\nb -> 2\n---\nEnd");
    }

    // ---------- Prefix and suffix formatting ----------

    #[test]
    fn test_prefix_and_suffix_formatting() {
        let template = FewShotPromptTemplate::builder()
            .prefix("You are a helpful assistant.")
            .suffix("User: {input}\nAssistant:")
            .example_template("User: {input}\nAssistant: {output}")
            .examples(vec![make_example("Hi", "Hello!")])
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "How are you?".to_string());

        let result = template.format(&vars).unwrap();
        assert!(result.starts_with("You are a helpful assistant."));
        assert!(result.ends_with("User: How are you?\nAssistant:"));
    }

    // ---------- Empty examples ----------

    #[test]
    fn test_empty_examples() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Prefix")
            .suffix("Suffix: {input}")
            .example_template("{input} -> {output}")
            .examples(vec![])
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "test".to_string());

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "Prefix\n\nSuffix: test");
    }

    #[test]
    fn test_no_examples_no_selector() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Prefix")
            .suffix("Suffix")
            .example_template("{input}")
            .build();

        let vars = HashMap::new();
        let result = template.format(&vars).unwrap();
        assert_eq!(result, "Prefix\n\nSuffix");
    }

    // ---------- Runnable trait implementation ----------

    #[tokio::test]
    async fn test_runnable_trait_implementation() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Examples:")
            .suffix("Input: {input}\nOutput:")
            .example_template("Input: {input}\nOutput: {output}")
            .examples(vec![make_example("hi", "hello")])
            .build();

        let result = template
            .invoke(serde_json::json!({"input": "hey"}), None)
            .await
            .unwrap();

        match result {
            Value::String(s) => {
                assert!(s.contains("Examples:"));
                assert!(s.contains("Input: hi\nOutput: hello"));
                assert!(s.contains("Input: hey\nOutput:"));
            }
            _ => panic!("Expected string output from Runnable::invoke"),
        }
    }

    #[tokio::test]
    async fn test_runnable_invoke_type_error() {
        let template = FewShotPromptTemplate::builder()
            .prefix("P")
            .suffix("S")
            .example_template("{input}")
            .build();

        let result = template
            .invoke(serde_json::json!("not an object"), None)
            .await;
        assert!(result.is_err());
    }

    // ---------- Add example to selector ----------

    #[test]
    fn test_add_example_to_selector() {
        let embeddings = Box::new(DeterministicFakeEmbedding::new(32));

        let mut selector = SemanticSimilaritySelector::builder(embeddings)
            .k(5)
            .examples(vec![make_example("cat", "chat")])
            .build();

        assert_eq!(selector.examples.len(), 1);

        selector.add_example(make_example("dog", "chien"));
        assert_eq!(selector.examples.len(), 2);
        assert_eq!(selector.example_embeddings.len(), 2);

        let selected = selector.select_examples("dog");
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_add_example_to_length_selector() {
        let mut selector = LengthBasedSelector::builder()
            .max_length(1000)
            .example_template("{input} -> {output}")
            .build();

        selector.add_example(make_example("a", "1"));
        selector.add_example(make_example("b", "2"));

        let selected = selector.select_examples("anything");
        assert_eq!(selected.len(), 2);
    }

    // ---------- Few-shot with input variables in suffix ----------

    #[test]
    fn test_suffix_with_multiple_input_variables() {
        let template = FewShotPromptTemplate::builder()
            .prefix("Translate {source_lang} to {target_lang}:")
            .suffix("Input ({source_lang}): {input}\nOutput ({target_lang}):")
            .example_template("Input: {input}\nOutput: {output}")
            .examples(vec![make_example("hello", "bonjour")])
            .build();

        let mut vars = HashMap::new();
        vars.insert("input".to_string(), "goodbye".to_string());
        vars.insert("source_lang".to_string(), "English".to_string());
        vars.insert("target_lang".to_string(), "French".to_string());

        let result = template.format(&vars).unwrap();
        // Prefix doesn't get variable substitution (it's a plain string),
        // but suffix does.
        assert!(result.contains("Input (English): goodbye"));
        assert!(result.contains("Output (French):"));
    }

    // ---------- Builder default separator ----------

    #[test]
    fn test_default_separator_is_double_newline() {
        let template = FewShotPromptTemplate::builder()
            .prefix("P")
            .suffix("S")
            .example_template("{input}")
            .examples(vec![make_example("a", "1")])
            .build();

        assert_eq!(template.example_separator, "\n\n");
    }
}
