use std::collections::HashMap;

use serde_json::Value;

use crate::error::{Result, CognisError};

/// Template format types supported by prompt templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemplateFormat {
    /// Python f-string style: `{variable_name}`
    #[default]
    FString,
}

/// Extract variable names from a template string.
pub fn get_template_variables(template: &str, format: TemplateFormat) -> Vec<String> {
    match format {
        TemplateFormat::FString => extract_fstring_variables(template),
    }
}

/// Format a template string with the given variables.
pub fn format_template(
    template: &str,
    format: TemplateFormat,
    kwargs: &HashMap<String, Value>,
) -> Result<String> {
    match format {
        TemplateFormat::FString => format_fstring(template, kwargs),
    }
}

fn extract_fstring_variables(template: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if chars.peek() == Some(&'{') {
                // Escaped brace
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
            if !name.is_empty()
                && !name.contains('.')
                && !name.contains('[')
                && !name.chars().all(|c| c.is_ascii_digit())
                && !vars.contains(&name)
            {
                vars.push(name);
            }
        } else if ch == '}' && chars.peek() == Some(&'}') {
            chars.next();
        }
    }
    vars
}

fn format_fstring(template: &str, kwargs: &HashMap<String, Value>) -> Result<String> {
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
            let value = kwargs.get(&name).ok_or_else(|| {
                CognisError::Other(format!(
                    "Missing template variable '{}'. Available: {:?}",
                    name,
                    kwargs.keys().collect::<Vec<_>>()
                ))
            })?;
            match value {
                Value::String(s) => result.push_str(s),
                Value::Null => result.push_str(""),
                other => result.push_str(&other.to_string()),
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
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_variables() {
        let vars = extract_fstring_variables("Hello {name}, you are {age} years old.");
        assert_eq!(vars, vec!["name", "age"]);
    }

    #[test]
    fn test_extract_escaped_braces() {
        let vars = extract_fstring_variables("Use {{braces}} and {var}.");
        assert_eq!(vars, vec!["var"]);
    }

    #[test]
    fn test_extract_no_duplicates() {
        let vars = extract_fstring_variables("{x} + {x} = {y}");
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_format_basic() {
        let mut kwargs = HashMap::new();
        kwargs.insert("name".into(), Value::String("Alice".into()));
        kwargs.insert("age".into(), serde_json::json!(30));
        let result = format_fstring("Hello {name}, you are {age}.", &kwargs).unwrap();
        assert_eq!(result, "Hello Alice, you are 30.");
    }

    #[test]
    fn test_format_escaped_braces() {
        let kwargs = HashMap::new();
        let result = format_fstring("Use {{braces}}", &kwargs).unwrap();
        assert_eq!(result, "Use {braces}");
    }

    #[test]
    fn test_format_missing_variable() {
        let kwargs = HashMap::new();
        let result = format_fstring("Hello {name}", &kwargs);
        assert!(result.is_err());
    }
}
