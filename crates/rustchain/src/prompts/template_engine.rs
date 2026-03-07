//! Advanced prompt template engine with conditionals, loops, and partial support.
//!
//! Provides a mini template language that goes beyond simple `{variable}` substitution,
//! supporting `{%if%}...{%else%}...{%endif%}` conditionals, `{%for item in list%}...{%endfor%}`
//! loops, and `{>partial_name}` partial includes.

use std::collections::HashMap;

use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// TemplateVariable
// ---------------------------------------------------------------------------

/// A dynamically-typed value that can be used inside a template context.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateVariable {
    /// A string value.
    String(String),
    /// A numeric value.
    Number(f64),
    /// A boolean value.
    Boolean(bool),
    /// A list of template variables.
    List(Vec<TemplateVariable>),
    /// A map of named template variables.
    Map(HashMap<String, TemplateVariable>),
    /// A null / absent value.
    Null,
}

impl TemplateVariable {
    /// Render this variable to a string representation.
    pub fn as_string(&self) -> String {
        match self {
            TemplateVariable::String(s) => s.clone(),
            TemplateVariable::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            TemplateVariable::Boolean(b) => format!("{}", b),
            TemplateVariable::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.as_string()).collect();
                format!("[{}]", parts.join(", "))
            }
            TemplateVariable::Map(map) => {
                let parts: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.as_string()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            TemplateVariable::Null => "null".to_string(),
        }
    }

    /// Returns `true` if this value is considered "truthy" for conditionals.
    ///
    /// Falsy values: `Null`, `Boolean(false)`, empty `String`, empty `List`,
    /// empty `Map`, and `Number(0.0)`.
    pub fn is_truthy(&self) -> bool {
        match self {
            TemplateVariable::Null => false,
            TemplateVariable::Boolean(b) => *b,
            TemplateVariable::String(s) => !s.is_empty(),
            TemplateVariable::Number(n) => *n != 0.0,
            TemplateVariable::List(l) => !l.is_empty(),
            TemplateVariable::Map(m) => !m.is_empty(),
        }
    }

    /// Convert a `serde_json::Value` into a `TemplateVariable`.
    pub fn from_json(value: &Value) -> Self {
        match value {
            Value::Null => TemplateVariable::Null,
            Value::Bool(b) => TemplateVariable::Boolean(*b),
            Value::Number(n) => TemplateVariable::Number(n.as_f64().unwrap_or(0.0)),
            Value::String(s) => TemplateVariable::String(s.clone()),
            Value::Array(arr) => {
                TemplateVariable::List(arr.iter().map(TemplateVariable::from_json).collect())
            }
            Value::Object(obj) => {
                let map = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), TemplateVariable::from_json(v)))
                    .collect();
                TemplateVariable::Map(map)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TemplateContext
// ---------------------------------------------------------------------------

/// A variable store used for template rendering.
#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    variables: HashMap<String, TemplateVariable>,
}

impl TemplateContext {
    /// Create a new, empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable in the context.
    pub fn set(&mut self, key: impl Into<String>, value: TemplateVariable) {
        self.variables.insert(key.into(), value);
    }

    /// Get a variable from the context.
    pub fn get(&self, key: &str) -> Option<&TemplateVariable> {
        self.variables.get(key)
    }

    /// Convenience setter for string values.
    pub fn set_string(&mut self, key: impl Into<String>, val: impl Into<String>) {
        self.set(key, TemplateVariable::String(val.into()));
    }

    /// Convenience setter for numeric values.
    pub fn set_number(&mut self, key: impl Into<String>, val: f64) {
        self.set(key, TemplateVariable::Number(val));
    }

    /// Convenience setter for boolean values.
    pub fn set_bool(&mut self, key: impl Into<String>, val: bool) {
        self.set(key, TemplateVariable::Boolean(val));
    }

    /// Convenience setter for list values.
    pub fn set_list(&mut self, key: impl Into<String>, vec: Vec<TemplateVariable>) {
        self.set(key, TemplateVariable::List(vec));
    }

    /// Build a context from a JSON object. Non-object values are ignored.
    pub fn from_json(value: &Value) -> Self {
        let mut ctx = Self::new();
        if let Value::Object(obj) = value {
            for (k, v) in obj {
                ctx.set(k.clone(), TemplateVariable::from_json(v));
            }
        }
        ctx
    }

    /// Merge another context into this one. Existing keys are overwritten.
    pub fn merge(&mut self, other: &TemplateContext) {
        for (k, v) in &other.variables {
            self.variables.insert(k.clone(), v.clone());
        }
    }

    /// Return all keys in the context.
    pub fn keys(&self) -> Vec<&str> {
        self.variables.keys().map(String::as_str).collect()
    }

    /// Return the number of variables in the context.
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Return `true` if the context has no variables.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }
}

// ---------------------------------------------------------------------------
// TemplateBlock (AST)
// ---------------------------------------------------------------------------

/// A parsed template AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateBlock {
    /// Literal text.
    Text(String),
    /// Variable substitution: `{variable}`.
    Variable(String),
    /// Conditional block: `{%if cond%}...{%else%}...{%endif%}`.
    Conditional {
        /// The variable name used as the condition.
        condition: String,
        /// Blocks rendered when the condition is truthy.
        if_blocks: Vec<TemplateBlock>,
        /// Blocks rendered when the condition is falsy.
        else_blocks: Vec<TemplateBlock>,
    },
    /// Loop block: `{%for item in items%}...{%endfor%}`.
    Loop {
        /// The loop variable name (e.g., `item`).
        variable: String,
        /// The iterable variable name (e.g., `items`).
        iterable: String,
        /// The body blocks rendered for each iteration.
        body: Vec<TemplateBlock>,
    },
    /// Partial include: `{>partial_name}`.
    Partial(String),
}

// ---------------------------------------------------------------------------
// TemplateEngine
// ---------------------------------------------------------------------------

/// An advanced template engine that parses and renders templates with
/// conditionals, loops, and partial includes.
#[derive(Debug, Clone, Default)]
pub struct TemplateEngine {
    partials: HashMap<String, String>,
}

impl TemplateEngine {
    /// Create a new template engine with no registered partials.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named partial template.
    pub fn register_partial(&mut self, name: &str, template: &str) {
        self.partials.insert(name.to_string(), template.to_string());
    }

    /// Parse a template string into a list of AST blocks.
    pub fn parse(&self, template: &str) -> Result<Vec<TemplateBlock>> {
        let mut blocks = Vec::new();
        let mut pos = 0;
        let bytes = template.as_bytes();
        let len = bytes.len();

        while pos < len {
            // Check for block tags: {%...%}
            if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'%' {
                let tag_start = pos;
                let close = find_tag_close(template, pos + 2, "%}")
                    .ok_or_else(|| RustChainError::Other("Unclosed {%...%} tag".into()))?;
                let tag_content = template[pos + 2..close].trim();
                pos = close + 2;

                if let Some(cond) = tag_content
                    .strip_prefix("if ")
                    .or_else(|| tag_content.strip_prefix("if\t"))
                {
                    let condition = cond.trim().to_string();
                    let (if_blocks, else_blocks, new_pos) =
                        self.parse_conditional(template, pos)?;
                    blocks.push(TemplateBlock::Conditional {
                        condition,
                        if_blocks,
                        else_blocks,
                    });
                    pos = new_pos;
                } else if let Some(for_expr) = tag_content
                    .strip_prefix("for ")
                    .or_else(|| tag_content.strip_prefix("for\t"))
                {
                    let (variable, iterable) = parse_for_expr(for_expr.trim())?;
                    let (body, new_pos) = self.parse_loop(template, pos)?;
                    blocks.push(TemplateBlock::Loop {
                        variable,
                        iterable,
                        body,
                    });
                    pos = new_pos;
                } else {
                    return Err(RustChainError::Other(format!(
                        "Unknown block tag at position {}: '{}'",
                        tag_start, tag_content
                    )));
                }
            }
            // Check for partial: {>name}
            else if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'>' {
                let close = template[pos + 2..]
                    .find('}')
                    .ok_or_else(|| RustChainError::Other("Unclosed {>...} partial tag".into()))?;
                let name = template[pos + 2..pos + 2 + close].trim().to_string();
                blocks.push(TemplateBlock::Partial(name));
                pos = pos + 2 + close + 1;
            }
            // Check for variable: {name} (but not {{ escaped)
            else if bytes[pos] == b'{' {
                if pos + 1 < len && bytes[pos + 1] == b'{' {
                    // Escaped brace
                    blocks.push(TemplateBlock::Text("{".to_string()));
                    pos += 2;
                } else {
                    let close = template[pos + 1..]
                        .find('}')
                        .ok_or_else(|| RustChainError::Other("Unclosed {variable} tag".into()))?;
                    let name = template[pos + 1..pos + 1 + close].trim().to_string();
                    if !name.is_empty() {
                        blocks.push(TemplateBlock::Variable(name));
                    }
                    pos = pos + 1 + close + 1;
                }
            }
            // Escaped closing brace
            else if bytes[pos] == b'}' && pos + 1 < len && bytes[pos + 1] == b'}' {
                blocks.push(TemplateBlock::Text("}".to_string()));
                pos += 2;
            }
            // Plain text
            else {
                let mut text = String::new();
                while pos < len
                    && (bytes[pos] != b'{')
                    && !(bytes[pos] == b'}' && pos + 1 < len && bytes[pos + 1] == b'}')
                {
                    text.push(bytes[pos] as char);
                    pos += 1;
                }
                if !text.is_empty() {
                    blocks.push(TemplateBlock::Text(text));
                }
            }
        }

        Ok(blocks)
    }

    /// Parse the body of a conditional block, returning (if_blocks, else_blocks, new_pos).
    fn parse_conditional(
        &self,
        template: &str,
        start: usize,
    ) -> Result<(Vec<TemplateBlock>, Vec<TemplateBlock>, usize)> {
        let mut if_blocks = Vec::new();
        let mut else_blocks = Vec::new();
        let mut in_else = false;
        let mut pos = start;
        let bytes = template.as_bytes();
        let len = bytes.len();

        while pos < len {
            if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'%' {
                let close = find_tag_close(template, pos + 2, "%}").ok_or_else(|| {
                    RustChainError::Other("Unclosed {%...%} tag in conditional".into())
                })?;
                let tag_content = template[pos + 2..close].trim();

                if tag_content == "endif" {
                    return Ok((if_blocks, else_blocks, close + 2));
                } else if tag_content == "else" {
                    in_else = true;
                    pos = close + 2;
                    continue;
                } else if tag_content.starts_with("if ") {
                    // Nested conditional
                    let cond = tag_content.strip_prefix("if ").unwrap().trim().to_string();
                    pos = close + 2;
                    let (nested_if, nested_else, new_pos) =
                        self.parse_conditional(template, pos)?;
                    let block = TemplateBlock::Conditional {
                        condition: cond,
                        if_blocks: nested_if,
                        else_blocks: nested_else,
                    };
                    if in_else {
                        else_blocks.push(block);
                    } else {
                        if_blocks.push(block);
                    }
                    pos = new_pos;
                    continue;
                } else if tag_content.starts_with("for ") {
                    let for_expr = tag_content.strip_prefix("for ").unwrap().trim();
                    let (variable, iterable) = parse_for_expr(for_expr)?;
                    pos = close + 2;
                    let (body, new_pos) = self.parse_loop(template, pos)?;
                    let block = TemplateBlock::Loop {
                        variable,
                        iterable,
                        body,
                    };
                    if in_else {
                        else_blocks.push(block);
                    } else {
                        if_blocks.push(block);
                    }
                    pos = new_pos;
                    continue;
                }
            }

            // Parse a single block element
            let (block, new_pos) = self.parse_single_block(template, pos)?;
            if in_else {
                else_blocks.push(block);
            } else {
                if_blocks.push(block);
            }
            pos = new_pos;
        }

        Err(RustChainError::Other(
            "Unterminated conditional block: missing {%endif%}".into(),
        ))
    }

    /// Parse the body of a loop block, returning (body, new_pos).
    fn parse_loop(&self, template: &str, start: usize) -> Result<(Vec<TemplateBlock>, usize)> {
        let mut body = Vec::new();
        let mut pos = start;
        let bytes = template.as_bytes();
        let len = bytes.len();

        while pos < len {
            if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'%' {
                let close = find_tag_close(template, pos + 2, "%}")
                    .ok_or_else(|| RustChainError::Other("Unclosed {%...%} tag in loop".into()))?;
                let tag_content = template[pos + 2..close].trim();

                if tag_content == "endfor" {
                    return Ok((body, close + 2));
                } else if tag_content.starts_with("if ") {
                    let cond = tag_content.strip_prefix("if ").unwrap().trim().to_string();
                    pos = close + 2;
                    let (nested_if, nested_else, new_pos) =
                        self.parse_conditional(template, pos)?;
                    body.push(TemplateBlock::Conditional {
                        condition: cond,
                        if_blocks: nested_if,
                        else_blocks: nested_else,
                    });
                    pos = new_pos;
                    continue;
                } else if tag_content.starts_with("for ") {
                    let for_expr = tag_content.strip_prefix("for ").unwrap().trim();
                    let (variable, iterable) = parse_for_expr(for_expr)?;
                    pos = close + 2;
                    let (nested_body, new_pos) = self.parse_loop(template, pos)?;
                    body.push(TemplateBlock::Loop {
                        variable,
                        iterable,
                        body: nested_body,
                    });
                    pos = new_pos;
                    continue;
                }
            }

            let (block, new_pos) = self.parse_single_block(template, pos)?;
            body.push(block);
            pos = new_pos;
        }

        Err(RustChainError::Other(
            "Unterminated loop block: missing {%endfor%}".into(),
        ))
    }

    /// Parse a single non-block-level template element (text, variable, or partial).
    fn parse_single_block(&self, template: &str, pos: usize) -> Result<(TemplateBlock, usize)> {
        let bytes = template.as_bytes();
        let len = bytes.len();

        if pos >= len {
            return Ok((TemplateBlock::Text(String::new()), pos));
        }

        // Partial
        if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'>' {
            let close = template[pos + 2..]
                .find('}')
                .ok_or_else(|| RustChainError::Other("Unclosed {>...} partial tag".into()))?;
            let name = template[pos + 2..pos + 2 + close].trim().to_string();
            return Ok((TemplateBlock::Partial(name), pos + 2 + close + 1));
        }

        // Variable or escaped brace
        if bytes[pos] == b'{' {
            if pos + 1 < len && bytes[pos + 1] == b'{' {
                return Ok((TemplateBlock::Text("{".to_string()), pos + 2));
            }
            let close = template[pos + 1..]
                .find('}')
                .ok_or_else(|| RustChainError::Other("Unclosed {variable} tag".into()))?;
            let name = template[pos + 1..pos + 1 + close].trim().to_string();
            return Ok((TemplateBlock::Variable(name), pos + 1 + close + 1));
        }

        // Escaped closing brace
        if bytes[pos] == b'}' && pos + 1 < len && bytes[pos + 1] == b'}' {
            return Ok((TemplateBlock::Text("}".to_string()), pos + 2));
        }

        // Plain text - consume until next special char
        let mut text = String::new();
        let mut p = pos;
        while p < len
            && (bytes[p] != b'{')
            && !(bytes[p] == b'}' && p + 1 < len && bytes[p + 1] == b'}')
        {
            text.push(bytes[p] as char);
            p += 1;
        }
        Ok((TemplateBlock::Text(text), p))
    }

    /// Parse and render a template with the given context.
    pub fn render(&self, template: &str, context: &TemplateContext) -> Result<String> {
        let blocks = self.parse(template)?;
        self.render_blocks(&blocks, context)
    }

    /// Render a list of pre-parsed template blocks with the given context.
    pub fn render_blocks(
        &self,
        blocks: &[TemplateBlock],
        context: &TemplateContext,
    ) -> Result<String> {
        let mut output = String::new();
        for block in blocks {
            match block {
                TemplateBlock::Text(text) => output.push_str(text),
                TemplateBlock::Variable(name) => {
                    let value = context.get(name).ok_or_else(|| {
                        RustChainError::Other(format!("Missing template variable '{}'", name))
                    })?;
                    output.push_str(&value.as_string());
                }
                TemplateBlock::Conditional {
                    condition,
                    if_blocks,
                    else_blocks,
                } => {
                    let is_truthy = context
                        .get(condition)
                        .map(|v| v.is_truthy())
                        .unwrap_or(false);
                    if is_truthy {
                        output.push_str(&self.render_blocks(if_blocks, context)?);
                    } else {
                        output.push_str(&self.render_blocks(else_blocks, context)?);
                    }
                }
                TemplateBlock::Loop {
                    variable,
                    iterable,
                    body,
                } => {
                    let list = context.get(iterable).ok_or_else(|| {
                        RustChainError::Other(format!("Missing iterable variable '{}'", iterable))
                    })?;
                    if let TemplateVariable::List(items) = list {
                        for item in items {
                            let mut loop_ctx = context.clone();
                            loop_ctx.set(variable.clone(), item.clone());
                            output.push_str(&self.render_blocks(body, &loop_ctx)?);
                        }
                    } else {
                        return Err(RustChainError::Other(format!(
                            "Variable '{}' is not a list",
                            iterable
                        )));
                    }
                }
                TemplateBlock::Partial(name) => {
                    let partial_template = self.partials.get(name).ok_or_else(|| {
                        RustChainError::Other(format!("Unknown partial '{}'", name))
                    })?;
                    let rendered = self.render(partial_template, context)?;
                    output.push_str(&rendered);
                }
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate (engine-backed)
// ---------------------------------------------------------------------------

/// A prompt template that wraps a template string with a [`TemplateEngine`]
/// for advanced rendering with conditionals, loops, and partials.
#[derive(Debug, Clone)]
pub struct AdvancedPromptTemplate {
    template: String,
    engine: TemplateEngine,
}

impl AdvancedPromptTemplate {
    /// Create a new prompt template with a default engine.
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
            engine: TemplateEngine::new(),
        }
    }

    /// Replace the engine with a custom one (builder pattern).
    pub fn with_engine(mut self, engine: TemplateEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Format (render) the template with the given context.
    pub fn format(&self, context: &TemplateContext) -> Result<String> {
        self.engine.render(&self.template, context)
    }

    /// Convenience method: format the template using a JSON object as context.
    pub fn format_json(&self, vars: &Value) -> Result<String> {
        let context = TemplateContext::from_json(vars);
        self.format(&context)
    }

    /// Extract all variable names referenced in the template.
    ///
    /// This performs a best-effort extraction from the template string,
    /// returning variable names from `{var}` substitutions, `{%if var%}`
    /// conditions, and `{%for x in var%}` iterables.
    pub fn variables(&self) -> Vec<String> {
        let blocks = self.engine.parse(&self.template).unwrap_or_default();
        let mut vars = Vec::new();
        collect_block_variables(&blocks, &mut vars);
        vars
    }

    /// Validate that all required variables are present in the context.
    ///
    /// Returns a list of missing variable names, or an empty list if all
    /// variables are present.
    pub fn validate(&self, context: &TemplateContext) -> Result<Vec<String>> {
        let required = self.variables();
        let missing: Vec<String> = required
            .into_iter()
            .filter(|v| context.get(v).is_none())
            .collect();
        Ok(missing)
    }
}

/// Recursively collect variable names from template blocks.
///
/// `loop_vars` tracks locally-scoped loop iteration variables that should
/// not be reported as top-level template variables.
fn collect_block_variables(blocks: &[TemplateBlock], vars: &mut Vec<String>) {
    collect_block_variables_inner(blocks, vars, &[]);
}

fn collect_block_variables_inner(
    blocks: &[TemplateBlock],
    vars: &mut Vec<String>,
    loop_vars: &[String],
) {
    for block in blocks {
        match block {
            TemplateBlock::Variable(name) => {
                if !vars.contains(name) && !loop_vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            TemplateBlock::Conditional {
                condition,
                if_blocks,
                else_blocks,
            } => {
                if !vars.contains(condition) && !loop_vars.contains(condition) {
                    vars.push(condition.clone());
                }
                collect_block_variables_inner(if_blocks, vars, loop_vars);
                collect_block_variables_inner(else_blocks, vars, loop_vars);
            }
            TemplateBlock::Loop {
                variable,
                iterable,
                body,
            } => {
                if !vars.contains(iterable) && !loop_vars.contains(iterable) {
                    vars.push(iterable.clone());
                }
                let mut inner_loop_vars = loop_vars.to_vec();
                inner_loop_vars.push(variable.clone());
                collect_block_variables_inner(body, vars, &inner_loop_vars);
            }
            TemplateBlock::Text(_) | TemplateBlock::Partial(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Find the position of a closing tag sequence (e.g., `%}`) starting from `start`.
fn find_tag_close(template: &str, start: usize, close_seq: &str) -> Option<usize> {
    template[start..].find(close_seq).map(|i| start + i)
}

/// Parse a `for` expression like `item in items` into `(variable, iterable)`.
fn parse_for_expr(expr: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = expr.splitn(3, ' ').collect();
    if parts.len() < 3 || parts[1] != "in" {
        return Err(RustChainError::Other(format!(
            "Invalid for expression: '{}'. Expected 'variable in iterable'.",
            expr
        )));
    }
    Ok((parts[0].to_string(), parts[2].trim().to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- TemplateVariable ----

    #[test]
    fn test_variable_string_as_string() {
        let v = TemplateVariable::String("hello".into());
        assert_eq!(v.as_string(), "hello");
    }

    #[test]
    fn test_variable_number_as_string_integer() {
        let v = TemplateVariable::Number(42.0);
        assert_eq!(v.as_string(), "42");
    }

    #[test]
    fn test_variable_number_as_string_float() {
        let v = TemplateVariable::Number(3.14);
        assert_eq!(v.as_string(), "3.14");
    }

    #[test]
    fn test_variable_boolean_as_string() {
        assert_eq!(TemplateVariable::Boolean(true).as_string(), "true");
        assert_eq!(TemplateVariable::Boolean(false).as_string(), "false");
    }

    #[test]
    fn test_variable_null_as_string() {
        assert_eq!(TemplateVariable::Null.as_string(), "null");
    }

    #[test]
    fn test_variable_list_as_string() {
        let v = TemplateVariable::List(vec![
            TemplateVariable::Number(1.0),
            TemplateVariable::Number(2.0),
        ]);
        assert_eq!(v.as_string(), "[1, 2]");
    }

    #[test]
    fn test_variable_is_truthy() {
        assert!(TemplateVariable::Boolean(true).is_truthy());
        assert!(!TemplateVariable::Boolean(false).is_truthy());
        assert!(TemplateVariable::String("hi".into()).is_truthy());
        assert!(!TemplateVariable::String("".into()).is_truthy());
        assert!(TemplateVariable::Number(1.0).is_truthy());
        assert!(!TemplateVariable::Number(0.0).is_truthy());
        assert!(!TemplateVariable::Null.is_truthy());
        assert!(TemplateVariable::List(vec![TemplateVariable::Null]).is_truthy());
        assert!(!TemplateVariable::List(vec![]).is_truthy());
        assert!(!TemplateVariable::Map(HashMap::new()).is_truthy());
    }

    #[test]
    fn test_variable_from_json_string() {
        let v = TemplateVariable::from_json(&json!("hello"));
        assert_eq!(v, TemplateVariable::String("hello".into()));
    }

    #[test]
    fn test_variable_from_json_number() {
        let v = TemplateVariable::from_json(&json!(42));
        assert_eq!(v, TemplateVariable::Number(42.0));
    }

    #[test]
    fn test_variable_from_json_bool() {
        let v = TemplateVariable::from_json(&json!(true));
        assert_eq!(v, TemplateVariable::Boolean(true));
    }

    #[test]
    fn test_variable_from_json_null() {
        let v = TemplateVariable::from_json(&json!(null));
        assert_eq!(v, TemplateVariable::Null);
    }

    #[test]
    fn test_variable_from_json_array() {
        let v = TemplateVariable::from_json(&json!([1, "two"]));
        match v {
            TemplateVariable::List(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], TemplateVariable::Number(1.0));
                assert_eq!(items[1], TemplateVariable::String("two".into()));
            }
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_variable_from_json_object() {
        let v = TemplateVariable::from_json(&json!({"key": "value"}));
        match v {
            TemplateVariable::Map(map) => {
                assert_eq!(
                    map.get("key"),
                    Some(&TemplateVariable::String("value".into()))
                );
            }
            _ => panic!("Expected Map"),
        }
    }

    // ---- TemplateContext ----

    #[test]
    fn test_context_set_get() {
        let mut ctx = TemplateContext::new();
        ctx.set("name", TemplateVariable::String("Alice".into()));
        assert_eq!(
            ctx.get("name"),
            Some(&TemplateVariable::String("Alice".into()))
        );
        assert_eq!(ctx.get("missing"), None);
    }

    #[test]
    fn test_context_convenience_setters() {
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "Bob");
        ctx.set_number("age", 30.0);
        ctx.set_bool("active", true);
        ctx.set_list("items", vec![TemplateVariable::String("a".into())]);

        assert_eq!(
            ctx.get("name"),
            Some(&TemplateVariable::String("Bob".into()))
        );
        assert_eq!(ctx.get("age"), Some(&TemplateVariable::Number(30.0)));
        assert_eq!(ctx.get("active"), Some(&TemplateVariable::Boolean(true)));
        assert_eq!(ctx.len(), 4);
        assert!(!ctx.is_empty());
    }

    #[test]
    fn test_context_from_json() {
        let ctx = TemplateContext::from_json(&json!({"name": "Alice", "age": 30}));
        assert_eq!(ctx.len(), 2);
        assert_eq!(
            ctx.get("name"),
            Some(&TemplateVariable::String("Alice".into()))
        );
        assert_eq!(ctx.get("age"), Some(&TemplateVariable::Number(30.0)));
    }

    #[test]
    fn test_context_merge() {
        let mut ctx1 = TemplateContext::new();
        ctx1.set_string("a", "1");
        ctx1.set_string("b", "2");

        let mut ctx2 = TemplateContext::new();
        ctx2.set_string("b", "overwritten");
        ctx2.set_string("c", "3");

        ctx1.merge(&ctx2);
        assert_eq!(ctx1.len(), 3);
        assert_eq!(
            ctx1.get("b"),
            Some(&TemplateVariable::String("overwritten".into()))
        );
        assert_eq!(ctx1.get("c"), Some(&TemplateVariable::String("3".into())));
    }

    #[test]
    fn test_context_keys() {
        let mut ctx = TemplateContext::new();
        ctx.set_string("x", "1");
        ctx.set_string("y", "2");
        let mut keys = ctx.keys();
        keys.sort();
        assert_eq!(keys, vec!["x", "y"]);
    }

    #[test]
    fn test_context_empty() {
        let ctx = TemplateContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);
    }

    // ---- Simple variable substitution ----

    #[test]
    fn test_render_simple_variable() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "World");
        let result = engine.render("Hello {name}!", &ctx).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_render_multiple_variables() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_string("first", "Jane");
        ctx.set_string("last", "Doe");
        let result = engine.render("{first} {last}", &ctx).unwrap();
        assert_eq!(result, "Jane Doe");
    }

    #[test]
    fn test_render_number_variable() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_number("count", 5.0);
        let result = engine.render("Count: {count}", &ctx).unwrap();
        assert_eq!(result, "Count: 5");
    }

    #[test]
    fn test_render_missing_variable_error() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let err = engine.render("Hello {name}!", &ctx).unwrap_err();
        assert!(format!("{}", err).contains("Missing template variable 'name'"));
    }

    // ---- Conditionals ----

    #[test]
    fn test_conditional_truthy() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("show", true);
        let result = engine
            .render("Start{%if show%} visible{%endif%} end", &ctx)
            .unwrap();
        assert_eq!(result, "Start visible end");
    }

    #[test]
    fn test_conditional_falsy() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("show", false);
        let result = engine
            .render("Start{%if show%} visible{%endif%} end", &ctx)
            .unwrap();
        assert_eq!(result, "Start end");
    }

    #[test]
    fn test_conditional_with_else() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("premium", true);
        let result = engine
            .render(
                "{%if premium%}Welcome, VIP!{%else%}Please subscribe.{%endif%}",
                &ctx,
            )
            .unwrap();
        assert_eq!(result, "Welcome, VIP!");

        ctx.set_bool("premium", false);
        let result = engine
            .render(
                "{%if premium%}Welcome, VIP!{%else%}Please subscribe.{%endif%}",
                &ctx,
            )
            .unwrap();
        assert_eq!(result, "Please subscribe.");
    }

    #[test]
    fn test_conditional_missing_var_is_falsy() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let result = engine
            .render("{%if missing%}yes{%else%}no{%endif%}", &ctx)
            .unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_conditional_with_variable_inside() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("greet", true);
        ctx.set_string("name", "Alice");
        let result = engine
            .render("{%if greet%}Hello {name}!{%endif%}", &ctx)
            .unwrap();
        assert_eq!(result, "Hello Alice!");
    }

    // ---- Loops ----

    #[test]
    fn test_loop_over_list() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_list(
            "items",
            vec![
                TemplateVariable::String("a".into()),
                TemplateVariable::String("b".into()),
                TemplateVariable::String("c".into()),
            ],
        );
        let result = engine
            .render("{%for item in items%}{item} {%endfor%}", &ctx)
            .unwrap();
        assert_eq!(result, "a b c ");
    }

    #[test]
    fn test_loop_empty_list() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_list("items", vec![]);
        let result = engine
            .render("Before{%for item in items%}{item}{%endfor%}After", &ctx)
            .unwrap();
        assert_eq!(result, "BeforeAfter");
    }

    #[test]
    fn test_loop_non_list_error() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_string("items", "not a list");
        let err = engine
            .render("{%for item in items%}{item}{%endfor%}", &ctx)
            .unwrap_err();
        assert!(format!("{}", err).contains("not a list"));
    }

    // ---- Nested structures ----

    #[test]
    fn test_nested_conditional_in_loop() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_list(
            "users",
            vec![
                TemplateVariable::String("admin".into()),
                TemplateVariable::String("guest".into()),
            ],
        );
        ctx.set_bool("show_role", true);
        let result = engine
            .render(
                "{%for user in users%}{user}{%if show_role%}(active){%endif%} {%endfor%}",
                &ctx,
            )
            .unwrap();
        assert_eq!(result, "admin(active) guest(active) ");
    }

    #[test]
    fn test_nested_loops() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_list(
            "rows",
            vec![
                TemplateVariable::String("A".into()),
                TemplateVariable::String("B".into()),
            ],
        );
        // Note: inner loop uses same context variable name shadowing
        let result = engine
            .render("{%for row in rows%}[{row}]{%endfor%}", &ctx)
            .unwrap();
        assert_eq!(result, "[A][B]");
    }

    #[test]
    fn test_nested_conditionals() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("a", true);
        ctx.set_bool("b", true);
        let result = engine
            .render("{%if a%}A{%if b%}B{%endif%}{%endif%}", &ctx)
            .unwrap();
        assert_eq!(result, "AB");

        ctx.set_bool("b", false);
        let result = engine
            .render("{%if a%}A{%if b%}B{%endif%}{%endif%}", &ctx)
            .unwrap();
        assert_eq!(result, "A");
    }

    // ---- Partials ----

    #[test]
    fn test_partial_include() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("greeting", "Hello {name}!");
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "World");
        let result = engine.render("Say: {>greeting}", &ctx).unwrap();
        assert_eq!(result, "Say: Hello World!");
    }

    #[test]
    fn test_partial_unknown_error() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let err = engine.render("{>unknown}", &ctx).unwrap_err();
        assert!(format!("{}", err).contains("Unknown partial 'unknown'"));
    }

    #[test]
    fn test_nested_partials() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("inner", "({name})");
        engine.register_partial("outer", "Start {>inner} End");
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "deep");
        let result = engine.render("{>outer}", &ctx).unwrap();
        assert_eq!(result, "Start (deep) End");
    }

    // ---- AdvancedPromptTemplate ----

    #[test]
    fn test_prompt_template_format() {
        let tmpl = AdvancedPromptTemplate::new("Hello {name}!");
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "World");
        assert_eq!(tmpl.format(&ctx).unwrap(), "Hello World!");
    }

    #[test]
    fn test_prompt_template_format_json() {
        let tmpl = AdvancedPromptTemplate::new("Hello {name}, age {age}!");
        let result = tmpl
            .format_json(&json!({"name": "Alice", "age": 30}))
            .unwrap();
        assert_eq!(result, "Hello Alice, age 30!");
    }

    #[test]
    fn test_prompt_template_with_engine() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("sig", "-- {author}");
        let tmpl = AdvancedPromptTemplate::new("Body\n{>sig}").with_engine(engine);
        let mut ctx = TemplateContext::new();
        ctx.set_string("author", "Bob");
        assert_eq!(tmpl.format(&ctx).unwrap(), "Body\n-- Bob");
    }

    #[test]
    fn test_prompt_template_variables() {
        let tmpl = AdvancedPromptTemplate::new(
            "Hello {name}! {%if premium%}VIP{%endif%} {%for x in items%}{x}{%endfor%}",
        );
        let mut vars = tmpl.variables();
        vars.sort();
        assert_eq!(vars, vec!["items", "name", "premium"]);
    }

    #[test]
    fn test_prompt_template_validate_all_present() {
        let tmpl = AdvancedPromptTemplate::new("Hello {name}!");
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "World");
        let missing = tmpl.validate(&ctx).unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn test_prompt_template_validate_missing() {
        let tmpl = AdvancedPromptTemplate::new("Hello {name}, {role}!");
        let mut ctx = TemplateContext::new();
        ctx.set_string("name", "Alice");
        let missing = tmpl.validate(&ctx).unwrap();
        assert_eq!(missing, vec!["role"]);
    }

    // ---- Edge cases ----

    #[test]
    fn test_empty_template() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let result = engine.render("", &ctx).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_no_variables_template() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let result = engine.render("Just plain text.", &ctx).unwrap();
        assert_eq!(result, "Just plain text.");
    }

    #[test]
    fn test_escaped_braces() {
        let engine = TemplateEngine::new();
        let ctx = TemplateContext::new();
        let result = engine.render("Use {{braces}} here.", &ctx).unwrap();
        assert_eq!(result, "Use {braces} here.");
    }

    #[test]
    fn test_parse_returns_ast() {
        let engine = TemplateEngine::new();
        let blocks = engine.parse("Hello {name}!").unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], TemplateBlock::Text("Hello ".into()));
        assert_eq!(blocks[1], TemplateBlock::Variable("name".into()));
        assert_eq!(blocks[2], TemplateBlock::Text("!".into()));
    }

    #[test]
    fn test_conditional_loop_combined() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set_bool("has_items", true);
        ctx.set_list(
            "items",
            vec![
                TemplateVariable::String("x".into()),
                TemplateVariable::String("y".into()),
            ],
        );
        let result = engine
            .render(
                "{%if has_items%}Items: {%for i in items%}{i}, {%endfor%}{%else%}No items.{%endif%}",
                &ctx,
            )
            .unwrap();
        assert_eq!(result, "Items: x, y, ");
    }

    #[test]
    fn test_context_from_json_non_object() {
        // Non-object JSON should produce an empty context
        let ctx = TemplateContext::from_json(&json!("just a string"));
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_prompt_template_variables_deduplication() {
        let tmpl = AdvancedPromptTemplate::new("{name} and {name} again");
        let vars = tmpl.variables();
        assert_eq!(vars, vec!["name"]);
    }
}
