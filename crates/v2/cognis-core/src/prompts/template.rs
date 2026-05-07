//! String template + the rendering engine shared by all prompt types.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// A typed string prompt template.
///
/// Inputs are any `Serialize` type — fields are read via serde reflection,
/// so plain structs, `HashMap<String, Value>`, and `serde_json::Value` all
/// work transparently.
///
/// Placeholders are `{name}`. Literal braces are `{{` and `}}`. Dotted
/// paths (`{user.name}`) descend into nested objects.
#[derive(Debug, Clone)]
pub struct PromptTemplate<I = Value> {
    template: String,
    _input: PhantomData<fn() -> I>,
}

impl<I> PromptTemplate<I>
where
    I: Serialize + Send + Sync + 'static,
{
    /// Build a template from a string. Doesn't validate placeholder
    /// names — invalid placeholders surface at render time.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            _input: PhantomData,
        }
    }

    /// Render the template against the given input.
    pub fn render(&self, input: &I) -> Result<String> {
        let value =
            serde_json::to_value(input).map_err(|e| CognisError::Serialization(e.to_string()))?;
        render(&self.template, &value)
    }

    /// The raw template string.
    pub fn template_str(&self) -> &str {
        &self.template
    }

    /// All `{name}` placeholders in the template, in first-occurrence order.
    pub fn input_variables(&self) -> Vec<String> {
        scan_variables(&self.template)
    }
}

#[async_trait]
impl<I> Runnable<I, String> for PromptTemplate<I>
where
    I: Serialize + Send + Sync + 'static,
{
    async fn invoke(&self, input: I, _: RunnableConfig) -> Result<String> {
        self.render(&input)
    }

    fn name(&self) -> &str {
        "PromptTemplate"
    }
}

/// Render `{var}` placeholders against a `serde_json::Value` context.
///
/// `{{` and `}}` are literal `{` and `}`. Dotted keys descend into nested
/// objects. Returns `CognisError::Configuration` for missing variables or
/// unclosed braces.
pub(crate) fn render(template: &str, ctx: &Value) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for nc in chars.by_ref() {
                    if nc == '}' {
                        closed = true;
                        break;
                    }
                    name.push(nc);
                }
                if !closed {
                    return Err(CognisError::Configuration(format!(
                        "unclosed `{{` in template: {template}"
                    )));
                }
                let key = name.trim();
                let resolved = lookup(ctx, key).ok_or_else(|| {
                    CognisError::Configuration(format!("missing template variable `{key}`"))
                })?;
                out.push_str(&value_to_string(&resolved));
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Find all `{name}` placeholders in first-occurrence order.
pub(crate) fn scan_variables(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
            }
            '{' => {
                let mut name = String::new();
                for nc in chars.by_ref() {
                    if nc == '}' {
                        break;
                    }
                    name.push(nc);
                }
                let trimmed = name.trim().to_string();
                if !trimmed.is_empty() && !out.contains(&trimmed) {
                    out.push(trimmed);
                }
            }
            _ => {}
        }
    }
    out
}

fn lookup(ctx: &Value, key: &str) -> Option<Value> {
    let mut cur = ctx.clone();
    for segment in key.split('.') {
        cur = match cur {
            Value::Object(mut m) => m.remove(segment)?,
            _ => return None,
        };
    }
    Some(cur)
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn renders_simple() {
        let p = PromptTemplate::<Value>::new("hello {name}");
        let out = p
            .invoke(json!({"name": "world"}), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn renders_typed_struct() {
        #[derive(Serialize)]
        struct Ctx {
            name: String,
        }
        let p: PromptTemplate<Ctx> = PromptTemplate::new("hi {name}");
        let out = p
            .render(&Ctx {
                name: "rust".into(),
            })
            .unwrap();
        assert_eq!(out, "hi rust");
    }

    #[test]
    fn dotted_paths() {
        let p: PromptTemplate<Value> = PromptTemplate::new("{user.name} aged {user.age}");
        let out = p
            .render(&json!({"user": {"name": "Ada", "age": 36}}))
            .unwrap();
        assert_eq!(out, "Ada aged 36");
    }

    #[test]
    fn literal_braces() {
        let p: PromptTemplate<Value> = PromptTemplate::new("{{not a var}} {x}");
        let out = p.render(&json!({"x": 1})).unwrap();
        assert_eq!(out, "{not a var} 1");
    }

    #[test]
    fn missing_variable_errors() {
        let p: PromptTemplate<Value> = PromptTemplate::new("hi {name}");
        let err = p.render(&json!({})).unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[test]
    fn unclosed_brace_errors() {
        let p: PromptTemplate<Value> = PromptTemplate::new("hi {name");
        let err = p.render(&json!({"name": "x"})).unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[test]
    fn input_variables_returns_unique_in_order() {
        let p: PromptTemplate<Value> = PromptTemplate::new("{a} {b} {a} {c}");
        assert_eq!(p.input_variables(), vec!["a", "b", "c"]);
    }
}
