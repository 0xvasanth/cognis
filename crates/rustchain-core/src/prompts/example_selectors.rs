//! Concrete example selector implementations.
//!
//! Mirrors Python `langchain_core.example_selectors`.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::prompts::base::PromptTemplate;
use crate::prompts::example_selector::BaseExampleSelector;
use crate::prompts::string_formatter::format_template;

/// Compute length of text by splitting on whitespace and newlines.
fn get_length_based(text: &str) -> usize {
    text.split(|c: char| c.is_whitespace()).filter(|s| !s.is_empty()).count()
}

/// Select examples based on the total prompt length constraint.
///
/// Greedily adds examples until `max_length` would be exceeded.
pub struct LengthBasedExampleSelector {
    examples: Mutex<Vec<HashMap<String, Value>>>,
    example_prompt: PromptTemplate,
    max_length: usize,
    get_text_length: fn(&str) -> usize,
    example_text_lengths: Mutex<Vec<usize>>,
}

impl LengthBasedExampleSelector {
    pub fn new(
        examples: Vec<HashMap<String, Value>>,
        example_prompt: PromptTemplate,
        max_length: usize,
    ) -> Self {
        let lengths: Vec<usize> = examples
            .iter()
            .map(|ex| {
                let formatted = format_template(
                    &example_prompt.template,
                    example_prompt.template_format,
                    ex,
                )
                .unwrap_or_default();
                get_length_based(&formatted)
            })
            .collect();

        Self {
            examples: Mutex::new(examples),
            example_prompt,
            max_length,
            get_text_length: get_length_based,
            example_text_lengths: Mutex::new(lengths),
        }
    }

    /// Set a custom text length function.
    pub fn with_length_fn(mut self, f: fn(&str) -> usize) -> Self {
        self.get_text_length = f;
        // Recompute lengths with new function.
        {
            let examples = self.examples.lock().unwrap();
            let lengths: Vec<usize> = examples
                .iter()
                .map(|ex| {
                    let formatted = format_template(
                        &self.example_prompt.template,
                        self.example_prompt.template_format,
                        ex,
                    )
                    .unwrap_or_default();
                    f(&formatted)
                })
                .collect();
            *self.example_text_lengths.lock().unwrap() = lengths;
        }
        self
    }
}

#[async_trait]
impl BaseExampleSelector for LengthBasedExampleSelector {
    async fn select_examples(
        &self,
        input: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        let input_text: String = input
            .values()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        let mut remaining = self.max_length.saturating_sub((self.get_text_length)(&input_text));
        let examples = self.examples.lock().unwrap();
        let lengths = self.example_text_lengths.lock().unwrap();

        let mut selected = Vec::new();
        for (i, ex) in examples.iter().enumerate() {
            if i >= lengths.len() {
                break;
            }
            if lengths[i] > remaining {
                break;
            }
            remaining -= lengths[i];
            selected.push(ex.clone());
        }
        Ok(selected)
    }

    async fn add_example(&self, example: HashMap<String, Value>) -> Result<()> {
        let formatted = format_template(
            &self.example_prompt.template,
            self.example_prompt.template_format,
            &example,
        )
        .unwrap_or_default();
        let len = (self.get_text_length)(&formatted);
        self.examples.lock().unwrap().push(example);
        self.example_text_lengths.lock().unwrap().push(len);
        Ok(())
    }
}
