//! Prompt templating engine with variable substitution, conditionals, loops,
//! and template composition for building LLM prompts.
//!
//! Supports Handlebars-like syntax:
//! - `{{variable}}` — variable substitution
//! - `{{#if var}}...{{/if}}` — conditional blocks
//! - `{{#each list}}...{{/each}}` — loop blocks
//! - Template composition via [`TemplateComposer`]
//! - Reusable fragments via [`PartialTemplate`]
//! - Pre-render validation via [`TemplateValidator`]

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ---------------------------------------------------------------------------
// TemplateError
// ---------------------------------------------------------------------------

/// Errors that can occur during template parsing and rendering.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    /// A required variable was not provided.
    MissingVariable { name: String },
    /// The template contains invalid syntax at the given byte position.
    InvalidSyntax { position: usize, message: String },
    /// A general rendering error.
    RenderError(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::MissingVariable { name } => {
                write!(f, "missing required variable: {name}")
            }
            TemplateError::InvalidSyntax { position, message } => {
                write!(f, "invalid syntax at position {position}: {message}")
            }
            TemplateError::RenderError(msg) => write!(f, "render error: {msg}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ---------------------------------------------------------------------------
// TemplateValue
// ---------------------------------------------------------------------------

/// A value that can be substituted into a template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<TemplateValue>),
    None,
}

impl TemplateValue {
    /// Returns the value as a `&str` if it is a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            TemplateValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Converts any variant to a string representation.
    pub fn as_string_lossy(&self) -> String {
        match self {
            TemplateValue::String(s) => s.clone(),
            TemplateValue::Integer(i) => i.to_string(),
            TemplateValue::Float(f) => f.to_string(),
            TemplateValue::Bool(b) => b.to_string(),
            TemplateValue::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.as_string_lossy()).collect();
                format!("[{}]", parts.join(", "))
            }
            TemplateValue::None => String::new(),
        }
    }

    /// Returns `true` if this is the `None` variant.
    pub fn is_none(&self) -> bool {
        matches!(self, TemplateValue::None)
    }

    /// Converts the value to a `serde_json::Value`.
    pub fn to_json(&self) -> JsonValue {
        match self {
            TemplateValue::String(s) => JsonValue::String(s.clone()),
            TemplateValue::Integer(i) => serde_json::json!(*i),
            TemplateValue::Float(f) => serde_json::json!(*f),
            TemplateValue::Bool(b) => JsonValue::Bool(*b),
            TemplateValue::List(items) => {
                JsonValue::Array(items.iter().map(|v| v.to_json()).collect())
            }
            TemplateValue::None => JsonValue::Null,
        }
    }

    /// Returns `true` if the value is "truthy" for conditional evaluation.
    fn is_truthy(&self) -> bool {
        match self {
            TemplateValue::String(s) => !s.is_empty(),
            TemplateValue::Integer(i) => *i != 0,
            TemplateValue::Float(f) => *f != 0.0,
            TemplateValue::Bool(b) => *b,
            TemplateValue::List(l) => !l.is_empty(),
            TemplateValue::None => false,
        }
    }
}

impl From<String> for TemplateValue {
    fn from(s: String) -> Self {
        TemplateValue::String(s)
    }
}

impl From<&str> for TemplateValue {
    fn from(s: &str) -> Self {
        TemplateValue::String(s.to_string())
    }
}

impl From<i64> for TemplateValue {
    fn from(i: i64) -> Self {
        TemplateValue::Integer(i)
    }
}

impl From<f64> for TemplateValue {
    fn from(f: f64) -> Self {
        TemplateValue::Float(f)
    }
}

impl From<bool> for TemplateValue {
    fn from(b: bool) -> Self {
        TemplateValue::Bool(b)
    }
}

impl From<Vec<String>> for TemplateValue {
    fn from(v: Vec<String>) -> Self {
        TemplateValue::List(v.into_iter().map(TemplateValue::String).collect())
    }
}

// ---------------------------------------------------------------------------
// TemplateVariable
// ---------------------------------------------------------------------------

/// A named variable with a value and required/optional status.
#[derive(Debug, Clone)]
pub struct TemplateVariable {
    /// The variable name.
    pub name: String,
    /// The current value.
    pub value: TemplateValue,
    /// Whether the variable must be provided before rendering.
    pub required: bool,
}

impl TemplateVariable {
    /// Creates a new variable with the given name and value, marked as required.
    pub fn new(name: impl Into<String>, value: impl Into<TemplateValue>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            required: true,
        }
    }

    /// Creates a required variable with no value set yet.
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: TemplateValue::None,
            required: true,
        }
    }

    /// Creates an optional variable with a default value.
    pub fn optional(name: impl Into<String>, default: impl Into<TemplateValue>) -> Self {
        Self {
            name: name.into(),
            value: default.into(),
            required: false,
        }
    }

    /// Converts the variable to a JSON value.
    pub fn to_json(&self) -> JsonValue {
        serde_json::json!({
            "name": self.name,
            "value": self.value.to_json(),
            "required": self.required,
        })
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// A prompt template that supports `{{variable}}` substitution, `{{#if}}`
/// conditionals, and `{{#each}}` loops.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: HashMap<String, TemplateValue>,
}

impl PromptTemplate {
    /// Creates a new template from a raw template string.
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
            variables: HashMap::new(),
        }
    }

    /// Sets a single variable.
    pub fn with_variable(
        mut self,
        name: impl Into<String>,
        value: impl Into<TemplateValue>,
    ) -> Self {
        self.variables.insert(name.into(), value.into());
        self
    }

    /// Sets multiple variables at once.
    pub fn with_variables(mut self, vars: HashMap<String, TemplateValue>) -> Self {
        self.variables.extend(vars);
        self
    }

    /// Renders the template by substituting all variables, evaluating
    /// conditionals, and expanding loops.
    pub fn render(&self) -> Result<String, TemplateError> {
        let mut output = self.template.clone();

        // 1. Process {{#each list}}...{{/each}} blocks
        output = process_each_blocks(&output, &self.variables)?;

        // 2. Process {{#if var}}...{{else}}...{{/if}} blocks
        output = process_if_blocks(&output, &self.variables)?;

        // 3. Substitute plain {{variable}} placeholders
        output = substitute_variables(&output, &self.variables)?;

        Ok(output)
    }

    /// Returns the list of variable names found in the template string.
    pub fn variables(&self) -> Vec<String> {
        extract_variable_names(&self.template)
    }

    /// Returns variables present in the template but not yet set.
    pub fn missing_variables(&self) -> Vec<String> {
        let needed = self.variables();
        needed
            .into_iter()
            .filter(|name| !self.variables.contains_key(name))
            .collect()
    }

    /// Returns the raw template string.
    pub fn template_str(&self) -> &str {
        &self.template
    }

    /// Converts the template to a JSON representation.
    pub fn to_json(&self) -> JsonValue {
        let vars: serde_json::Map<String, JsonValue> = self
            .variables
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();
        serde_json::json!({
            "template": self.template,
            "variables": vars,
        })
    }
}

// ---------------------------------------------------------------------------
// ConditionalBlock
// ---------------------------------------------------------------------------

/// Represents a `{{#if var}}...{{else}}...{{/if}}` conditional block.
#[derive(Debug, Clone)]
pub struct ConditionalBlock {
    condition_var: String,
    true_block: String,
    false_block: Option<String>,
}

impl ConditionalBlock {
    /// Creates a new conditional block.
    pub fn new(
        condition_var: impl Into<String>,
        true_block: impl Into<String>,
        false_block: Option<String>,
    ) -> Self {
        Self {
            condition_var: condition_var.into(),
            true_block: true_block.into(),
            false_block,
        }
    }

    /// Evaluates the block given a set of variables.
    pub fn evaluate(&self, vars: &HashMap<String, TemplateValue>) -> String {
        let is_true = vars
            .get(&self.condition_var)
            .map(|v| v.is_truthy())
            .unwrap_or(false);

        if is_true {
            self.true_block.clone()
        } else {
            self.false_block.clone().unwrap_or_default()
        }
    }
}

// ---------------------------------------------------------------------------
// LoopBlock
// ---------------------------------------------------------------------------

/// Represents a `{{#each items}}...{{/each}}` loop block.
#[derive(Debug, Clone)]
pub struct LoopBlock {
    list_var: String,
    body_template: String,
}

impl LoopBlock {
    /// Creates a new loop block.
    pub fn new(list_var: impl Into<String>, body_template: impl Into<String>) -> Self {
        Self {
            list_var: list_var.into(),
            body_template: body_template.into(),
        }
    }

    /// Evaluates the loop by iterating over the list variable and rendering
    /// the body template for each item. Inside the body, `{{this}}` refers to
    /// the current item and `{{@index}}` to the zero-based index.
    pub fn evaluate(
        &self,
        vars: &HashMap<String, TemplateValue>,
    ) -> Result<String, TemplateError> {
        let list = vars.get(&self.list_var).ok_or_else(|| {
            TemplateError::MissingVariable {
                name: self.list_var.clone(),
            }
        })?;

        let items = match list {
            TemplateValue::List(items) => items,
            _ => {
                return Err(TemplateError::RenderError(format!(
                    "variable '{}' is not a list",
                    self.list_var
                )));
            }
        };

        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let mut rendered = self.body_template.replace("{{this}}", &item.as_string_lossy());
            rendered = rendered.replace("{{@index}}", &idx.to_string());
            parts.push(rendered);
        }

        Ok(parts.join(""))
    }
}

// ---------------------------------------------------------------------------
// TemplateComposer
// ---------------------------------------------------------------------------

/// Composes multiple named [`PromptTemplate`]s and renders them together.
#[derive(Debug, Clone)]
pub struct TemplateComposer {
    templates: HashMap<String, PromptTemplate>,
}

impl TemplateComposer {
    /// Creates a new empty composer.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Registers a named template.
    pub fn register(&mut self, name: impl Into<String>, template: PromptTemplate) {
        self.templates.insert(name.into(), template);
    }

    /// Retrieves a template by name.
    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    /// Renders the named templates in order and joins them with a separator.
    pub fn compose(&self, names: &[&str], separator: &str) -> Result<String, TemplateError> {
        let mut rendered = Vec::with_capacity(names.len());
        for name in names {
            let tmpl = self.templates.get(*name).ok_or_else(|| {
                TemplateError::RenderError(format!("template '{}' not found", name))
            })?;
            rendered.push(tmpl.render()?);
        }
        Ok(rendered.join(separator))
    }

    /// Returns the number of registered templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Converts the composer to a JSON representation.
    pub fn to_json(&self) -> JsonValue {
        let map: serde_json::Map<String, JsonValue> = self
            .templates
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();
        serde_json::json!({ "templates": map })
    }
}

impl Default for TemplateComposer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PartialTemplate
// ---------------------------------------------------------------------------

/// A reusable template fragment that can be rendered with a set of variables.
#[derive(Debug, Clone)]
pub struct PartialTemplate {
    name: String,
    content: String,
}

impl PartialTemplate {
    /// Creates a new partial template.
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }

    /// Returns the partial's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Renders the partial with the given variables.
    pub fn render_with(
        &self,
        vars: &HashMap<String, TemplateValue>,
    ) -> Result<String, TemplateError> {
        let tmpl = PromptTemplate {
            template: self.content.clone(),
            variables: vars.clone(),
        };
        tmpl.render()
    }
}

// ---------------------------------------------------------------------------
// TemplateValidator
// ---------------------------------------------------------------------------

/// Validates template syntax before rendering.
#[derive(Debug, Clone)]
pub struct TemplateValidator {
    _private: (),
}

impl TemplateValidator {
    /// Creates a new validator.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Validates the template string, returning a list of errors if any.
    pub fn validate(&self, template: &str) -> Result<(), Vec<TemplateError>> {
        let mut errors = Vec::new();

        // Check balanced {{ and }}
        let mut pos = 0;
        let bytes = template.as_bytes();
        while pos < bytes.len() {
            if pos + 1 < bytes.len() && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
                // Find matching }}
                if let Some(end) = template[pos + 2..].find("}}") {
                    let inner = &template[pos + 2..pos + 2 + end];
                    let trimmed = inner.trim();

                    // Check for empty tags
                    if trimmed.is_empty() {
                        errors.push(TemplateError::InvalidSyntax {
                            position: pos,
                            message: "empty template tag".to_string(),
                        });
                    }

                    pos = pos + 2 + end + 2;
                } else {
                    errors.push(TemplateError::InvalidSyntax {
                        position: pos,
                        message: "unclosed '{{' tag".to_string(),
                    });
                    break;
                }
            } else {
                pos += 1;
            }
        }

        // Check balanced #if / /if
        let if_opens = count_occurrences(template, "{{#if ");
        let if_closes = count_occurrences(template, "{{/if}}");
        if if_opens != if_closes {
            errors.push(TemplateError::InvalidSyntax {
                position: 0,
                message: format!(
                    "mismatched #if/#/if blocks: {} opens, {} closes",
                    if_opens, if_closes
                ),
            });
        }

        // Check balanced #each / /each
        let each_opens = count_occurrences(template, "{{#each ");
        let each_closes = count_occurrences(template, "{{/each}}");
        if each_opens != each_closes {
            errors.push(TemplateError::InvalidSyntax {
                position: 0,
                message: format!(
                    "mismatched #each/#/each blocks: {} opens, {} closes",
                    each_opens, each_closes
                ),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Extracts all variable names from the template.
    pub fn extract_variables(&self, template: &str) -> Vec<String> {
        extract_variable_names(template)
    }
}

impl Default for TemplateValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Counts non-overlapping occurrences of `needle` in `haystack`.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// Extracts plain variable names from `{{name}}` tags (not block tags).
fn extract_variable_names(template: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut pos = 0;
    let bytes = template.as_bytes();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            if let Some(end) = template[pos + 2..].find("}}") {
                let inner = template[pos + 2..pos + 2 + end].trim();

                // Skip block tags (#if, /if, #each, /each, else) and special vars
                if !inner.starts_with('#')
                    && !inner.starts_with('/')
                    && inner != "else"
                    && inner != "this"
                    && !inner.starts_with('@')
                    && !inner.is_empty()
                {
                    let name = inner.to_string();
                    if seen.insert(name.clone()) {
                        names.push(name);
                    }
                }

                pos = pos + 2 + end + 2;
            } else {
                break;
            }
        } else {
            pos += 1;
        }
    }

    names
}

/// Substitutes `{{name}}` placeholders with variable values.
fn substitute_variables(
    template: &str,
    vars: &HashMap<String, TemplateValue>,
) -> Result<String, TemplateError> {
    let mut result = String::with_capacity(template.len());
    let mut pos = 0;
    let bytes = template.as_bytes();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            if let Some(end) = template[pos + 2..].find("}}") {
                let name = template[pos + 2..pos + 2 + end].trim();

                if let Some(val) = vars.get(name) {
                    result.push_str(&val.as_string_lossy());
                } else {
                    return Err(TemplateError::MissingVariable {
                        name: name.to_string(),
                    });
                }

                pos = pos + 2 + end + 2;
            } else {
                result.push('{');
                pos += 1;
            }
        } else {
            result.push(template.as_bytes()[pos] as char);
            pos += 1;
        }
    }

    Ok(result)
}

/// Processes `{{#if var}}...{{else}}...{{/if}}` blocks (non-nested).
fn process_if_blocks(
    template: &str,
    vars: &HashMap<String, TemplateValue>,
) -> Result<String, TemplateError> {
    let mut output = template.to_string();

    loop {
        // Find innermost #if block to handle nesting
        let if_start = match find_last_occurrence(&output, "{{#if ") {
            Some(pos) => pos,
            None => break,
        };

        // Find the matching /if after this #if
        let after_tag = match output[if_start + 6..].find("}}") {
            Some(pos) => if_start + 6 + pos + 2,
            None => {
                return Err(TemplateError::InvalidSyntax {
                    position: if_start,
                    message: "unclosed #if tag".to_string(),
                });
            }
        };

        let var_name = output[if_start + 6..after_tag - 2].trim().to_string();

        let end_tag = "{{/if}}";
        let end_pos = match output[after_tag..].find(end_tag) {
            Some(pos) => after_tag + pos,
            None => {
                return Err(TemplateError::InvalidSyntax {
                    position: if_start,
                    message: "missing {{/if}}".to_string(),
                });
            }
        };

        let body = &output[after_tag..end_pos];

        // Split on {{else}} if present
        let (true_block, false_block) = if let Some(else_pos) = body.find("{{else}}") {
            (&body[..else_pos], &body[else_pos + 8..])
        } else {
            (body, "")
        };

        let cond = ConditionalBlock::new(&var_name, true_block, Some(false_block.to_string()));
        let replacement = cond.evaluate(vars);

        output = format!(
            "{}{}{}",
            &output[..if_start],
            replacement,
            &output[end_pos + end_tag.len()..]
        );
    }

    Ok(output)
}

/// Finds the last (rightmost) occurrence of a substring to process innermost blocks first.
fn find_last_occurrence(haystack: &str, needle: &str) -> Option<usize> {
    haystack.rfind(needle)
}

/// Processes `{{#each list}}...{{/each}}` blocks (non-nested).
fn process_each_blocks(
    template: &str,
    vars: &HashMap<String, TemplateValue>,
) -> Result<String, TemplateError> {
    let mut output = template.to_string();

    loop {
        let each_start = match output.find("{{#each ") {
            Some(pos) => pos,
            None => break,
        };

        let after_tag = match output[each_start + 8..].find("}}") {
            Some(pos) => each_start + 8 + pos + 2,
            None => {
                return Err(TemplateError::InvalidSyntax {
                    position: each_start,
                    message: "unclosed #each tag".to_string(),
                });
            }
        };

        let var_name = output[each_start + 8..after_tag - 2].trim().to_string();

        let end_tag = "{{/each}}";
        let end_pos = match output[after_tag..].find(end_tag) {
            Some(pos) => after_tag + pos,
            None => {
                return Err(TemplateError::InvalidSyntax {
                    position: each_start,
                    message: "missing {{/each}}".to_string(),
                });
            }
        };

        let body = &output[after_tag..end_pos];
        let loop_block = LoopBlock::new(&var_name, body);
        let replacement = loop_block.evaluate(vars)?;

        output = format!(
            "{}{}{}",
            &output[..each_start],
            replacement,
            &output[end_pos + end_tag.len()..]
        );
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TemplateValue tests --

    #[test]
    fn test_template_value_from_string() {
        let v: TemplateValue = "hello".into();
        assert_eq!(v, TemplateValue::String("hello".to_string()));
    }

    #[test]
    fn test_template_value_from_owned_string() {
        let v: TemplateValue = String::from("world").into();
        assert_eq!(v, TemplateValue::String("world".to_string()));
    }

    #[test]
    fn test_template_value_from_i64() {
        let v: TemplateValue = 42i64.into();
        assert_eq!(v, TemplateValue::Integer(42));
    }

    #[test]
    fn test_template_value_from_f64() {
        let v: TemplateValue = 3.14f64.into();
        assert_eq!(v, TemplateValue::Float(3.14));
    }

    #[test]
    fn test_template_value_from_bool() {
        let v: TemplateValue = true.into();
        assert_eq!(v, TemplateValue::Bool(true));
    }

    #[test]
    fn test_template_value_from_vec_string() {
        let v: TemplateValue = vec!["a".to_string(), "b".to_string()].into();
        assert_eq!(
            v,
            TemplateValue::List(vec![
                TemplateValue::String("a".to_string()),
                TemplateValue::String("b".to_string()),
            ])
        );
    }

    #[test]
    fn test_template_value_as_str() {
        let v = TemplateValue::String("test".to_string());
        assert_eq!(v.as_str(), Some("test"));
        assert_eq!(TemplateValue::Integer(1).as_str(), None);
    }

    #[test]
    fn test_template_value_as_string_lossy() {
        assert_eq!(TemplateValue::String("hi".into()).as_string_lossy(), "hi");
        assert_eq!(TemplateValue::Integer(99).as_string_lossy(), "99");
        assert_eq!(TemplateValue::Float(1.5).as_string_lossy(), "1.5");
        assert_eq!(TemplateValue::Bool(false).as_string_lossy(), "false");
        assert_eq!(TemplateValue::None.as_string_lossy(), "");
    }

    #[test]
    fn test_template_value_list_as_string_lossy() {
        let v = TemplateValue::List(vec![
            TemplateValue::String("x".into()),
            TemplateValue::Integer(1),
        ]);
        assert_eq!(v.as_string_lossy(), "[x, 1]");
    }

    #[test]
    fn test_template_value_is_none() {
        assert!(TemplateValue::None.is_none());
        assert!(!TemplateValue::Bool(false).is_none());
    }

    #[test]
    fn test_template_value_to_json() {
        assert_eq!(
            TemplateValue::String("a".into()).to_json(),
            JsonValue::String("a".into())
        );
        assert_eq!(TemplateValue::Integer(5).to_json(), serde_json::json!(5));
        assert_eq!(TemplateValue::Bool(true).to_json(), JsonValue::Bool(true));
        assert_eq!(TemplateValue::None.to_json(), JsonValue::Null);
    }

    #[test]
    fn test_template_value_truthy() {
        assert!(TemplateValue::Bool(true).is_truthy());
        assert!(!TemplateValue::Bool(false).is_truthy());
        assert!(TemplateValue::String("yes".into()).is_truthy());
        assert!(!TemplateValue::String("".into()).is_truthy());
        assert!(TemplateValue::Integer(1).is_truthy());
        assert!(!TemplateValue::Integer(0).is_truthy());
        assert!(!TemplateValue::None.is_truthy());
    }

    // -- TemplateVariable tests --

    #[test]
    fn test_template_variable_new() {
        let v = TemplateVariable::new("name", "Alice");
        assert_eq!(v.name, "name");
        assert_eq!(v.value, TemplateValue::String("Alice".into()));
        assert!(v.required);
    }

    #[test]
    fn test_template_variable_required() {
        let v = TemplateVariable::required("age");
        assert_eq!(v.name, "age");
        assert!(v.value.is_none());
        assert!(v.required);
    }

    #[test]
    fn test_template_variable_optional() {
        let v = TemplateVariable::optional("color", "blue");
        assert_eq!(v.name, "color");
        assert_eq!(v.value, TemplateValue::String("blue".into()));
        assert!(!v.required);
    }

    #[test]
    fn test_template_variable_to_json() {
        let v = TemplateVariable::new("x", 42i64);
        let j = v.to_json();
        assert_eq!(j["name"], "x");
        assert_eq!(j["value"], 42);
        assert_eq!(j["required"], true);
    }

    // -- PromptTemplate basic tests --

    #[test]
    fn test_simple_substitution() {
        let result = PromptTemplate::new("Hello, {{name}}!")
            .with_variable("name", "World")
            .render()
            .unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_multiple_variables() {
        let result = PromptTemplate::new("{{greeting}}, {{name}}!")
            .with_variable("greeting", "Hi")
            .with_variable("name", "Bob")
            .render()
            .unwrap();
        assert_eq!(result, "Hi, Bob!");
    }

    #[test]
    fn test_integer_variable() {
        let result = PromptTemplate::new("Count: {{n}}")
            .with_variable("n", 42i64)
            .render()
            .unwrap();
        assert_eq!(result, "Count: 42");
    }

    #[test]
    fn test_missing_variable_error() {
        let err = PromptTemplate::new("Hello, {{name}}!").render().unwrap_err();
        assert!(matches!(err, TemplateError::MissingVariable { name } if name == "name"));
    }

    #[test]
    fn test_no_variables_plain_text() {
        let result = PromptTemplate::new("No variables here.")
            .render()
            .unwrap();
        assert_eq!(result, "No variables here.");
    }

    #[test]
    fn test_variables_extraction() {
        let tmpl = PromptTemplate::new("{{a}} and {{b}} and {{a}} again");
        let vars = tmpl.variables();
        assert_eq!(vars, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_missing_variables_detection() {
        let tmpl = PromptTemplate::new("{{a}} {{b}} {{c}}")
            .with_variable("b", "set");
        let missing = tmpl.missing_variables();
        assert!(missing.contains(&"a".to_string()));
        assert!(missing.contains(&"c".to_string()));
        assert!(!missing.contains(&"b".to_string()));
    }

    #[test]
    fn test_with_variables_hashmap() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), TemplateValue::String("1".into()));
        vars.insert("y".to_string(), TemplateValue::String("2".into()));
        let result = PromptTemplate::new("{{x}}+{{y}}")
            .with_variables(vars)
            .render()
            .unwrap();
        assert_eq!(result, "1+2");
    }

    #[test]
    fn test_template_to_json() {
        let tmpl = PromptTemplate::new("{{x}}")
            .with_variable("x", "val");
        let j = tmpl.to_json();
        assert_eq!(j["template"], "{{x}}");
        assert_eq!(j["variables"]["x"], "val");
    }

    #[test]
    fn test_template_str() {
        let tmpl = PromptTemplate::new("Hello");
        assert_eq!(tmpl.template_str(), "Hello");
    }

    #[test]
    fn test_repeated_variable() {
        let result = PromptTemplate::new("{{x}} and {{x}}")
            .with_variable("x", "a")
            .render()
            .unwrap();
        assert_eq!(result, "a and a");
    }

    #[test]
    fn test_variable_with_spaces_in_tag() {
        let result = PromptTemplate::new("{{ name }}")
            .with_variable("name", "trimmed")
            .render()
            .unwrap();
        assert_eq!(result, "trimmed");
    }

    // -- ConditionalBlock tests --

    #[test]
    fn test_conditional_true() {
        let cond = ConditionalBlock::new("flag", "yes", Some("no".to_string()));
        let mut vars = HashMap::new();
        vars.insert("flag".to_string(), TemplateValue::Bool(true));
        assert_eq!(cond.evaluate(&vars), "yes");
    }

    #[test]
    fn test_conditional_false() {
        let cond = ConditionalBlock::new("flag", "yes", Some("no".to_string()));
        let mut vars = HashMap::new();
        vars.insert("flag".to_string(), TemplateValue::Bool(false));
        assert_eq!(cond.evaluate(&vars), "no");
    }

    #[test]
    fn test_conditional_missing_var() {
        let cond = ConditionalBlock::new("missing", "yes", Some("no".to_string()));
        let vars = HashMap::new();
        assert_eq!(cond.evaluate(&vars), "no");
    }

    #[test]
    fn test_conditional_no_else() {
        let cond = ConditionalBlock::new("flag", "shown", None);
        let mut vars = HashMap::new();
        vars.insert("flag".to_string(), TemplateValue::Bool(false));
        assert_eq!(cond.evaluate(&vars), "");
    }

    #[test]
    fn test_if_block_in_template() {
        let result = PromptTemplate::new("{{#if show}}visible{{/if}}")
            .with_variable("show", true)
            .render()
            .unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_if_else_block_in_template() {
        let result = PromptTemplate::new("{{#if show}}yes{{else}}no{{/if}}")
            .with_variable("show", false)
            .render()
            .unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_if_with_substitution() {
        let result =
            PromptTemplate::new("{{#if greet}}Hello, {{name}}!{{else}}Goodbye.{{/if}}")
                .with_variable("greet", true)
                .with_variable("name", "Alice")
                .render()
                .unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    // -- LoopBlock tests --

    #[test]
    fn test_loop_block_basic() {
        let mut vars = HashMap::new();
        vars.insert(
            "items".to_string(),
            TemplateValue::List(vec![
                TemplateValue::String("a".into()),
                TemplateValue::String("b".into()),
            ]),
        );
        let lb = LoopBlock::new("items", "{{this}},");
        assert_eq!(lb.evaluate(&vars).unwrap(), "a,b,");
    }

    #[test]
    fn test_loop_block_with_index() {
        let mut vars = HashMap::new();
        vars.insert(
            "items".to_string(),
            TemplateValue::List(vec![
                TemplateValue::String("x".into()),
                TemplateValue::String("y".into()),
            ]),
        );
        let lb = LoopBlock::new("items", "{{@index}}:{{this}} ");
        assert_eq!(lb.evaluate(&vars).unwrap(), "0:x 1:y ");
    }

    #[test]
    fn test_loop_block_missing_var() {
        let vars = HashMap::new();
        let lb = LoopBlock::new("items", "{{this}}");
        let err = lb.evaluate(&vars).unwrap_err();
        assert!(matches!(err, TemplateError::MissingVariable { .. }));
    }

    #[test]
    fn test_loop_block_not_a_list() {
        let mut vars = HashMap::new();
        vars.insert("items".to_string(), TemplateValue::String("oops".into()));
        let lb = LoopBlock::new("items", "{{this}}");
        let err = lb.evaluate(&vars).unwrap_err();
        assert!(matches!(err, TemplateError::RenderError(_)));
    }

    #[test]
    fn test_each_block_in_template() {
        let result = PromptTemplate::new("Items: {{#each items}}{{this}} {{/each}}")
            .with_variable(
                "items",
                TemplateValue::List(vec![
                    TemplateValue::String("a".into()),
                    TemplateValue::String("b".into()),
                    TemplateValue::String("c".into()),
                ]),
            )
            .render()
            .unwrap();
        assert_eq!(result, "Items: a b c ");
    }

    #[test]
    fn test_each_empty_list() {
        let result = PromptTemplate::new("{{#each items}}{{this}}{{/each}}")
            .with_variable("items", TemplateValue::List(vec![]))
            .render()
            .unwrap();
        assert_eq!(result, "");
    }

    // -- TemplateComposer tests --

    #[test]
    fn test_composer_register_and_get() {
        let mut composer = TemplateComposer::new();
        composer.register("sys", PromptTemplate::new("System prompt"));
        assert!(composer.get("sys").is_some());
        assert!(composer.get("missing").is_none());
    }

    #[test]
    fn test_composer_template_count() {
        let mut composer = TemplateComposer::new();
        assert_eq!(composer.template_count(), 0);
        composer.register("a", PromptTemplate::new("A"));
        composer.register("b", PromptTemplate::new("B"));
        assert_eq!(composer.template_count(), 2);
    }

    #[test]
    fn test_composer_compose() {
        let mut composer = TemplateComposer::new();
        composer.register(
            "greeting",
            PromptTemplate::new("Hello, {{name}}!").with_variable("name", "User"),
        );
        composer.register("instruction", PromptTemplate::new("Please help."));

        let result = composer
            .compose(&["greeting", "instruction"], "\n")
            .unwrap();
        assert_eq!(result, "Hello, User!\nPlease help.");
    }

    #[test]
    fn test_composer_compose_missing_template() {
        let composer = TemplateComposer::new();
        let err = composer.compose(&["nope"], "\n").unwrap_err();
        assert!(matches!(err, TemplateError::RenderError(_)));
    }

    #[test]
    fn test_composer_to_json() {
        let mut composer = TemplateComposer::new();
        composer.register("t", PromptTemplate::new("test"));
        let j = composer.to_json();
        assert!(j["templates"]["t"].is_object());
    }

    // -- PartialTemplate tests --

    #[test]
    fn test_partial_template_render() {
        let partial = PartialTemplate::new("header", "Welcome, {{user}}!");
        let mut vars = HashMap::new();
        vars.insert("user".to_string(), TemplateValue::String("Admin".into()));
        assert_eq!(partial.render_with(&vars).unwrap(), "Welcome, Admin!");
    }

    #[test]
    fn test_partial_template_name() {
        let partial = PartialTemplate::new("footer", "End.");
        assert_eq!(partial.name(), "footer");
    }

    #[test]
    fn test_partial_template_missing_var() {
        let partial = PartialTemplate::new("p", "{{missing}}");
        let vars = HashMap::new();
        let err = partial.render_with(&vars).unwrap_err();
        assert!(matches!(err, TemplateError::MissingVariable { .. }));
    }

    // -- TemplateValidator tests --

    #[test]
    fn test_validator_valid_template() {
        let v = TemplateValidator::new();
        assert!(v.validate("Hello, {{name}}!").is_ok());
    }

    #[test]
    fn test_validator_unclosed_tag() {
        let v = TemplateValidator::new();
        let errors = v.validate("Hello, {{name").unwrap_err();
        assert!(!errors.is_empty());
        assert!(matches!(errors[0], TemplateError::InvalidSyntax { .. }));
    }

    #[test]
    fn test_validator_empty_tag() {
        let v = TemplateValidator::new();
        let errors = v.validate("Hello, {{}}").unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_mismatched_if() {
        let v = TemplateValidator::new();
        let errors = v.validate("{{#if x}}yes").unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_mismatched_each() {
        let v = TemplateValidator::new();
        let errors = v.validate("{{#each x}}body").unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_balanced_blocks() {
        let v = TemplateValidator::new();
        assert!(v
            .validate("{{#if x}}a{{/if}}{{#each y}}b{{/each}}")
            .is_ok());
    }

    #[test]
    fn test_validator_extract_variables() {
        let v = TemplateValidator::new();
        let vars = v.extract_variables("{{a}} {{#if b}}{{c}}{{/if}} {{#each d}}{{this}}{{/each}}");
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"c".to_string()));
        // b is inside #if tag, not a plain variable — extracted by extract_variable_names
        // d is inside #each tag — not extracted
        assert!(!vars.contains(&"this".to_string()));
    }

    // -- TemplateError tests --

    #[test]
    fn test_error_display_missing_variable() {
        let e = TemplateError::MissingVariable {
            name: "x".to_string(),
        };
        assert_eq!(e.to_string(), "missing required variable: x");
    }

    #[test]
    fn test_error_display_invalid_syntax() {
        let e = TemplateError::InvalidSyntax {
            position: 5,
            message: "bad".to_string(),
        };
        assert_eq!(e.to_string(), "invalid syntax at position 5: bad");
    }

    #[test]
    fn test_error_display_render_error() {
        let e = TemplateError::RenderError("fail".to_string());
        assert_eq!(e.to_string(), "render error: fail");
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(TemplateError::RenderError("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // -- Edge cases and integration tests --

    #[test]
    fn test_if_and_each_combined() {
        let result = PromptTemplate::new(
            "{{#if show}}List: {{#each items}}{{this}} {{/each}}{{else}}Hidden{{/if}}",
        )
        .with_variable("show", true)
        .with_variable(
            "items",
            TemplateValue::List(vec![
                TemplateValue::String("a".into()),
                TemplateValue::String("b".into()),
            ]),
        )
        .render()
        .unwrap();
        assert_eq!(result, "List: a b ");
    }

    #[test]
    fn test_if_false_hides_each() {
        let result = PromptTemplate::new(
            "{{#if show}}{{#each items}}{{this}}{{/each}}{{else}}none{{/if}}",
        )
        .with_variable("show", false)
        .with_variable("items", TemplateValue::List(vec![]))
        .render()
        .unwrap();
        assert_eq!(result, "none");
    }

    #[test]
    fn test_bool_variable_substitution() {
        let result = PromptTemplate::new("Value: {{flag}}")
            .with_variable("flag", true)
            .render()
            .unwrap();
        assert_eq!(result, "Value: true");
    }

    #[test]
    fn test_float_variable_substitution() {
        let result = PromptTemplate::new("Pi: {{pi}}")
            .with_variable("pi", 3.14f64)
            .render()
            .unwrap();
        assert_eq!(result, "Pi: 3.14");
    }

    #[test]
    fn test_empty_template() {
        let result = PromptTemplate::new("").render().unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_template_with_no_tags() {
        let result = PromptTemplate::new("Just plain text.").render().unwrap();
        assert_eq!(result, "Just plain text.");
    }

    #[test]
    fn test_multiline_template() {
        let result = PromptTemplate::new("Line 1: {{a}}\nLine 2: {{b}}")
            .with_variable("a", "X")
            .with_variable("b", "Y")
            .render()
            .unwrap();
        assert_eq!(result, "Line 1: X\nLine 2: Y");
    }

    #[test]
    fn test_composer_compose_with_separator() {
        let mut composer = TemplateComposer::new();
        composer.register("a", PromptTemplate::new("A"));
        composer.register("b", PromptTemplate::new("B"));
        composer.register("c", PromptTemplate::new("C"));
        let result = composer.compose(&["a", "b", "c"], " | ").unwrap();
        assert_eq!(result, "A | B | C");
    }

    #[test]
    fn test_composer_default() {
        let composer = TemplateComposer::default();
        assert_eq!(composer.template_count(), 0);
    }

    #[test]
    fn test_validator_default() {
        let v = TemplateValidator::default();
        assert!(v.validate("{{ok}}").is_ok());
    }

    #[test]
    fn test_loop_integer_items() {
        let result = PromptTemplate::new("{{#each nums}}{{this}},{{/each}}")
            .with_variable(
                "nums",
                TemplateValue::List(vec![
                    TemplateValue::Integer(1),
                    TemplateValue::Integer(2),
                    TemplateValue::Integer(3),
                ]),
            )
            .render()
            .unwrap();
        assert_eq!(result, "1,2,3,");
    }

    #[test]
    fn test_none_variable_substitution() {
        let result = PromptTemplate::new("Value: {{x}}")
            .with_variable("x", TemplateValue::None)
            .render()
            .unwrap();
        assert_eq!(result, "Value: ");
    }

    #[test]
    fn test_variables_deduplication() {
        let tmpl = PromptTemplate::new("{{x}} {{x}} {{y}}");
        let vars = tmpl.variables();
        assert_eq!(vars.len(), 2);
    }
}
