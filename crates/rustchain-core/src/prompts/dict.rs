use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::string_formatter::{format_template, get_template_variables, TemplateFormat};

/// A prompt template that preserves dict structure.
///
/// The template is a `Value::Object` whose string values may contain
/// f-string variables. `format()` recursively substitutes variables
/// while preserving the dict structure.
pub struct DictPromptTemplate {
    pub template: Value,
    pub input_variables: Vec<String>,
    pub template_format: TemplateFormat,
}

impl DictPromptTemplate {
    pub fn new(template: Value) -> Self {
        let input_variables = collect_variables(&template, TemplateFormat::FString);
        Self {
            template,
            input_variables,
            template_format: TemplateFormat::FString,
        }
    }

    /// Format the dict template with the given variables.
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<Value> {
        format_value(&self.template, self.template_format, kwargs)
    }
}

/// Recursively collect f-string variables from nested Value structures.
fn collect_variables(value: &Value, format: TemplateFormat) -> Vec<String> {
    let mut vars = Vec::new();
    match value {
        Value::String(s) => {
            for v in get_template_variables(s, format) {
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                for var in collect_variables(v, format) {
                    if !vars.contains(&var) {
                        vars.push(var);
                    }
                }
            }
        }
        Value::Array(arr) => {
            for v in arr {
                for var in collect_variables(v, format) {
                    if !vars.contains(&var) {
                        vars.push(var);
                    }
                }
            }
        }
        _ => {}
    }
    vars
}

/// Recursively format f-string variables in a Value structure.
fn format_value(
    value: &Value,
    format: TemplateFormat,
    kwargs: &HashMap<String, Value>,
) -> Result<Value> {
    match value {
        Value::String(s) => {
            let formatted = format_template(s, format, kwargs)?;
            Ok(Value::String(formatted))
        }
        Value::Object(map) => {
            let mut result = serde_json::Map::new();
            for (k, v) in map {
                result.insert(k.clone(), format_value(v, format, kwargs)?);
            }
            Ok(Value::Object(result))
        }
        Value::Array(arr) => {
            let result: Result<Vec<Value>> = arr
                .iter()
                .map(|v| format_value(v, format, kwargs))
                .collect();
            Ok(Value::Array(result?))
        }
        other => Ok(other.clone()),
    }
}

#[async_trait]
impl Runnable for DictPromptTemplate {
    fn name(&self) -> &str {
        "DictPromptTemplate"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(RustChainError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        self.format(&kwargs)
    }
}
