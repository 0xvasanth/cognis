use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{CognisError, Result};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::string_formatter::{format_template, get_template_variables, TemplateFormat};

/// A prompt template for image content blocks.
///
/// Produces `{"url": "...", "detail": "..."}` output suitable for
/// multimodal LLM messages.
pub struct ImagePromptTemplate {
    /// URL template string (may contain f-string variables).
    pub url: String,
    /// Detail level: "auto", "low", or "high".
    pub detail: String,
    pub input_variables: Vec<String>,
    pub template_format: TemplateFormat,
}

impl ImagePromptTemplate {
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        let input_variables = get_template_variables(&url, TemplateFormat::FString);
        Self {
            url,
            detail: "auto".into(),
            input_variables,
            template_format: TemplateFormat::FString,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Format the image template with the given variables.
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<Value> {
        let formatted_url = format_template(&self.url, self.template_format, kwargs)?;
        Ok(serde_json::json!({
            "url": formatted_url,
            "detail": self.detail,
        }))
    }
}

#[async_trait]
impl Runnable for ImagePromptTemplate {
    fn name(&self) -> &str {
        "ImagePromptTemplate"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(CognisError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        self.format(&kwargs)
    }
}
