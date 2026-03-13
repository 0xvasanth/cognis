//! A higher-level prompt template with partial variable support and `Runnable` integration.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::{CognisError, Result};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// A prompt template that supports partial (pre-filled) variables and
/// implements the `Runnable` trait for LCEL chain composition.
pub struct PromptTemplate {
    /// The raw template string with `{variable}` placeholders.
    pub template: String,
    /// All variable names extracted from the template.
    pub variables: Vec<String>,
    /// Variables that have been pre-filled and do not need to be supplied at
    /// format time.
    pub partial_variables: HashMap<String, String>,
}

impl PromptTemplate {
    /// Create a new template, auto-extracting `{variable}` placeholders.
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let variables = extract_variables(&template);
        Self {
            template,
            variables,
            partial_variables: HashMap::new(),
        }
    }

    /// Pre-fill a variable so it does not need to be supplied at format time.
    pub fn with_partial(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.partial_variables.insert(key.into(), value.into());
        self
    }

    /// Return the list of variables that still need to be supplied (all
    /// variables minus those already set as partials).
    pub fn input_variables(&self) -> Vec<&str> {
        self.variables
            .iter()
            .filter(|v| !self.partial_variables.contains_key(v.as_str()))
            .map(String::as_str)
            .collect()
    }

    /// Render the template with the given variables, falling back to partials
    /// for any keys not present in `variables`.
    pub fn format(&self, variables: &HashMap<String, String>) -> Result<String> {
        let mut merged = self.partial_variables.clone();
        merged.extend(variables.iter().map(|(k, v)| (k.clone(), v.clone())));
        format_template_str(&self.template, &merged)
    }
}

#[async_trait]
impl Runnable for PromptTemplate {
    fn name(&self) -> &str {
        "PromptTemplate"
    }

    /// Input: a JSON object whose string values are substituted into the
    /// template. Output: the rendered string as `Value::String`.
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

/// Extract `{variable}` placeholders from a template string.
fn extract_variables(template: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                continue;
            }
            let mut name = String::new();
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
                name.push(inner);
            }
            if !name.is_empty() && !vars.contains(&name) {
                vars.push(name);
            }
        } else if ch == '}' && chars.peek() == Some(&'}') {
            chars.next();
        }
    }
    vars
}

/// Render a template string by replacing `{var}` placeholders.
fn format_template_str(template: &str, variables: &HashMap<String, String>) -> Result<String> {
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
            let value = variables.get(&name).ok_or_else(|| {
                CognisError::Other(format!(
                    "Missing variable '{}'. Available: {:?}",
                    name,
                    variables.keys().collect::<Vec<_>>()
                ))
            })?;
            result.push_str(value);
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
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_extract_variables() {
        let t = PromptTemplate::new("Hello {name}, you are {age} years old.");
        assert_eq!(t.variables, vec!["name", "age"]);
    }

    #[test]
    fn test_format_with_all_variables() {
        let t = PromptTemplate::new("Hello {name}!");
        let mut vars = HashMap::new();
        vars.insert("name".into(), "World".into());
        assert_eq!(t.format(&vars).unwrap(), "Hello World!");
    }

    #[test]
    fn test_partial_variables() {
        let t =
            PromptTemplate::new("Hello {name}, welcome to {place}!").with_partial("place", "Rust");

        assert_eq!(t.input_variables(), vec!["name"]);

        let mut vars = HashMap::new();
        vars.insert("name".into(), "Alice".into());
        assert_eq!(t.format(&vars).unwrap(), "Hello Alice, welcome to Rust!");
    }

    #[test]
    fn test_missing_variable_error() {
        let t = PromptTemplate::new("Hello {name}!");
        let vars = HashMap::new();
        let err = t.format(&vars).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Missing variable 'name'"));
    }

    #[tokio::test]
    async fn test_runnable_invoke() {
        let t = PromptTemplate::new("Hello {name}!");
        let result = t
            .invoke(serde_json::json!({"name": "World"}), None)
            .await
            .unwrap();
        assert_eq!(result, Value::String("Hello World!".into()));
    }

    #[tokio::test]
    async fn test_runnable_invoke_non_string_values() {
        let t = PromptTemplate::new("Count: {n}");
        let result = t.invoke(serde_json::json!({"n": 42}), None).await.unwrap();
        assert_eq!(result, Value::String("Count: 42".into()));
    }
}
