use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::prompt_values::{PromptValue, StringPromptValue};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::string_formatter::{format_template, get_template_variables, TemplateFormat};

/// A partial variable value — either a static value or a dynamic callable.
pub enum PartialValue {
    Static(Value),
    Dynamic(Box<dyn Fn() -> Value + Send + Sync>),
}

impl PartialValue {
    pub fn resolve(&self) -> Value {
        match self {
            Self::Static(v) => v.clone(),
            Self::Dynamic(f) => f(),
        }
    }
}

/// A simple string prompt template with f-string variable substitution.
///
/// Implements `Runnable` so it can be composed in LCEL chains.
pub struct PromptTemplate {
    pub template: String,
    pub template_format: TemplateFormat,
    pub input_variables: Vec<String>,
    pub partial_variables: HashMap<String, PartialValue>,
}

impl PromptTemplate {
    /// Create a template from a format string. Input variables are auto-extracted.
    pub fn from_template(template: impl Into<String>) -> Self {
        let template = template.into();
        let input_variables = get_template_variables(&template, TemplateFormat::FString);
        Self {
            template,
            template_format: TemplateFormat::FString,
            input_variables,
            partial_variables: HashMap::new(),
        }
    }

    /// Create a template with explicit input variables and optional partial variables.
    pub fn new(
        template: impl Into<String>,
        input_variables: Vec<String>,
        partial_variables: HashMap<String, PartialValue>,
    ) -> Self {
        Self {
            template: template.into(),
            template_format: TemplateFormat::FString,
            input_variables,
            partial_variables,
        }
    }

    /// Create a new template with some variables pre-filled.
    pub fn partial(mut self, kwargs: HashMap<String, PartialValue>) -> Self {
        for k in kwargs.keys() {
            self.input_variables.retain(|v| v != k);
        }
        self.partial_variables.extend(kwargs);
        self
    }

    fn merge_variables(&self, kwargs: &HashMap<String, Value>) -> HashMap<String, Value> {
        let mut merged: HashMap<String, Value> = self
            .partial_variables
            .iter()
            .map(|(k, v)| (k.clone(), v.resolve()))
            .collect();
        merged.extend(kwargs.iter().map(|(k, v)| (k.clone(), v.clone())));
        merged
    }

    /// Format the template with the given variables, returning a string.
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<String> {
        let merged = self.merge_variables(kwargs);
        format_template(&self.template, self.template_format, &merged)
    }

    /// Format the template and wrap as a `StringPromptValue`.
    pub fn format_prompt(&self, kwargs: &HashMap<String, Value>) -> Result<Box<dyn PromptValue>> {
        let text = self.format(kwargs)?;
        Ok(Box::new(StringPromptValue::new(text)))
    }
}

#[async_trait]
impl Runnable for PromptTemplate {
    fn name(&self) -> &str {
        "PromptTemplate"
    }

    /// Input: JSON object with template variables. Output: formatted string as Value::String.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            other if self.input_variables.len() == 1 => {
                let mut m = HashMap::new();
                m.insert(self.input_variables[0].clone(), other);
                m
            }
            _ => {
                return Err(crate::error::CognisError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let text = self.format(&kwargs)?;
        Ok(Value::String(text))
    }
}
