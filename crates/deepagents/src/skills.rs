//! Skill system for loading and managing reusable agent skills.
//!
//! Skills are parameterised prompt templates that can be registered, searched,
//! rendered with arguments, and executed with history tracking. This mirrors
//! the Python `deepagents` `SkillsMiddleware` functionality.
//!
//! ## Core types
//!
//! - [`SkillDefinition`] -- metadata describing a skill (name, description, tags, etc.)
//! - [`SkillParameter`] -- a typed parameter accepted by a skill template
//! - [`SkillParamType`] -- the type of a skill parameter (text, number, boolean, list, json)
//! - [`SkillTemplate`] -- a skill with parameters and renderable instructions
//! - [`SkillRegistry`] -- manages a collection of skill templates
//! - [`SkillExecution`] -- records a single skill invocation
//! - [`SkillHistory`] -- tracks execution history with capacity limits

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::agent::DeepAgentError;

/// Result type for skill operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// SkillParamType
// ---------------------------------------------------------------------------

/// The type of a skill parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum SkillParamType {
    /// Free-form text.
    Text,
    /// Numeric value (integer or float).
    Number,
    /// Boolean flag.
    Boolean,
    /// A JSON array.
    List,
    /// Arbitrary JSON value.
    Json,
}

impl SkillParamType {
    /// Returns a human-readable name for this type.
    pub fn type_name(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::List => "list",
            Self::Json => "json",
        }
    }

    /// Validates whether the given JSON value matches this parameter type.
    pub fn validate(&self, value: &Value) -> bool {
        match self {
            Self::Text => value.is_string(),
            Self::Number => value.is_number(),
            Self::Boolean => value.is_boolean(),
            Self::List => value.is_array(),
            Self::Json => true, // any JSON value is valid
        }
    }
}

// ---------------------------------------------------------------------------
// SkillDefinition
// ---------------------------------------------------------------------------

/// Metadata describing a skill.
#[derive(Debug, Clone)]
pub struct SkillDefinition {
    /// Unique skill name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Semver version string.
    pub version: String,
    /// The skill prompt/template containing `{{param}}` placeholders.
    pub instructions: String,
    /// Tags for categorisation and search.
    pub tags: Vec<String>,
    /// Optional author.
    pub author: Option<String>,
}

impl SkillDefinition {
    /// Create a new definition with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            version: String::from("0.1.0"),
            instructions: String::new(),
            tags: Vec::new(),
            author: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the instructions template.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Serialize this definition to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "instructions": self.instructions,
            "tags": self.tags,
            "author": self.author,
        })
    }

    /// Returns `true` if the skill has the given tag (case-insensitive).
    pub fn matches_tag(&self, tag: &str) -> bool {
        let lower = tag.to_lowercase();
        self.tags.iter().any(|t| t.to_lowercase() == lower)
    }
}

// ---------------------------------------------------------------------------
// SkillParameter
// ---------------------------------------------------------------------------

/// Describes a parameter accepted by a [`SkillTemplate`].
#[derive(Debug, Clone)]
pub struct SkillParameter {
    /// Parameter name (used as placeholder key in instructions).
    pub name: String,
    /// The expected value type.
    pub param_type: SkillParamType,
    /// Whether this parameter must be provided.
    pub required: bool,
    /// Default value used when the parameter is not supplied.
    pub default_value: Option<Value>,
    /// Human-readable description.
    pub description: String,
}

impl SkillParameter {
    /// Create a new required parameter with the given name and type.
    pub fn new(name: impl Into<String>, param_type: SkillParamType) -> Self {
        Self {
            name: name.into(),
            param_type,
            required: true,
            default_value: None,
            description: String::new(),
        }
    }

    /// Mark the parameter as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set whether the parameter is required.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

// ---------------------------------------------------------------------------
// SkillTemplate
// ---------------------------------------------------------------------------

/// A skill with typed parameters and renderable instructions.
#[derive(Debug, Clone)]
pub struct SkillTemplate {
    /// The skill definition (metadata + instructions template).
    pub definition: SkillDefinition,
    /// Parameters accepted by this skill.
    pub parameters: Vec<SkillParameter>,
}

impl SkillTemplate {
    /// Create a new template from a definition.
    pub fn new(definition: SkillDefinition) -> Self {
        Self {
            definition,
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the template.
    pub fn add_parameter(mut self, param: SkillParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Render the instructions template by replacing `{{param_name}}` placeholders
    /// with the supplied argument values.
    ///
    /// Missing required parameters cause an error. Optional parameters without a
    /// supplied value use their default (or empty string).
    pub fn render(&self, args: &HashMap<String, Value>) -> Result<String> {
        self.validate_args(args)?;

        let mut output = self.definition.instructions.clone();
        for param in &self.parameters {
            let placeholder = format!("{{{{{}}}}}", param.name);
            let value_str = if let Some(val) = args.get(&param.name) {
                value_to_string(val)
            } else if let Some(ref def) = param.default_value {
                value_to_string(def)
            } else {
                String::new()
            };
            output = output.replace(&placeholder, &value_str);
        }
        Ok(output)
    }

    /// Validate the provided arguments against the parameter definitions.
    ///
    /// Checks that all required parameters are present and that supplied values
    /// match the expected types.
    pub fn validate_args(&self, args: &HashMap<String, Value>) -> Result<()> {
        for param in &self.parameters {
            if let Some(val) = args.get(&param.name) {
                if !param.param_type.validate(val) {
                    return Err(DeepAgentError::Other(format!(
                        "parameter '{}' expected type '{}', got {}",
                        param.name,
                        param.param_type.type_name(),
                        val,
                    )));
                }
            } else if param.required && param.default_value.is_none() {
                return Err(DeepAgentError::Other(format!(
                    "missing required parameter '{}'",
                    param.name,
                )));
            }
        }
        Ok(())
    }

    /// Returns references to all required parameters.
    pub fn required_params(&self) -> Vec<&SkillParameter> {
        self.parameters.iter().filter(|p| p.required).collect()
    }
}

/// Helper to convert a JSON value to a display string.
fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// SkillRegistry
// ---------------------------------------------------------------------------

/// Manages a collection of skill templates.
pub struct SkillRegistry {
    skills: HashMap<String, SkillTemplate>,
}

impl SkillRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Register a skill template.
    ///
    /// Returns an error if a skill with the same name already exists.
    pub fn register(&mut self, skill: SkillTemplate) -> Result<()> {
        let name = skill.definition.name.clone();
        if self.skills.contains_key(&name) {
            return Err(DeepAgentError::Other(format!(
                "skill '{}' is already registered",
                name,
            )));
        }
        self.skills.insert(name, skill);
        Ok(())
    }

    /// Get a reference to a skill template by name.
    pub fn get(&self, name: &str) -> Option<&SkillTemplate> {
        self.skills.get(name)
    }

    /// Remove a skill from the registry.
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        if self.skills.remove(name).is_none() {
            return Err(DeepAgentError::Other(
                format!("skill '{}' not found", name,),
            ));
        }
        Ok(())
    }

    /// Search for skills whose name, description, or tags contain the query
    /// string (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&SkillTemplate> {
        let lower = query.to_lowercase();
        self.skills
            .values()
            .filter(|s| {
                s.definition.name.to_lowercase().contains(&lower)
                    || s.definition.description.to_lowercase().contains(&lower)
                    || s.definition
                        .tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&lower))
            })
            .collect()
    }

    /// List all skills that have the given tag (case-insensitive).
    pub fn list_by_tag(&self, tag: &str) -> Vec<&SkillTemplate> {
        self.skills
            .values()
            .filter(|s| s.definition.matches_tag(tag))
            .collect()
    }

    /// Return references to all registered skills.
    pub fn all(&self) -> Vec<&SkillTemplate> {
        self.skills.values().collect()
    }

    /// Return the number of registered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Returns `true` if the registry contains no skills.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Serialize the registry to a JSON value (array of skill definitions).
    pub fn to_json(&self) -> Value {
        let arr: Vec<Value> = self
            .skills
            .values()
            .map(|s| s.definition.to_json())
            .collect();
        Value::Array(arr)
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SkillExecution
// ---------------------------------------------------------------------------

/// Records a single skill invocation.
#[derive(Debug, Clone)]
pub struct SkillExecution {
    /// Name of the skill that was executed.
    pub skill_name: String,
    /// Arguments supplied to the skill.
    pub args: HashMap<String, Value>,
    /// The rendered output (instructions with placeholders filled in).
    pub rendered_output: String,
    /// ISO-8601 timestamp of execution.
    pub executed_at: String,
    /// Whether the execution was successful.
    pub success: bool,
    /// Error message if the execution failed.
    pub error: Option<String>,
}

impl SkillExecution {
    /// Serialize this execution record to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "skill_name": self.skill_name,
            "args": self.args,
            "rendered_output": self.rendered_output,
            "executed_at": self.executed_at,
            "success": self.success,
            "error": self.error,
        })
    }
}

// ---------------------------------------------------------------------------
// SkillHistory
// ---------------------------------------------------------------------------

/// Tracks skill execution history with a configurable capacity limit.
pub struct SkillHistory {
    entries: Vec<SkillExecution>,
    max_entries: usize,
}

impl SkillHistory {
    /// Create a new history with the given maximum capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Record a skill execution. If the history exceeds `max_entries`, the
    /// oldest entry is evicted.
    pub fn record(&mut self, execution: SkillExecution) {
        self.entries.push(execution);
        while self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    /// Return the most recent `n` executions (newest last).
    pub fn recent(&self, n: usize) -> Vec<&SkillExecution> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].iter().collect()
    }

    /// Return all executions for the given skill name.
    pub fn by_skill(&self, name: &str) -> Vec<&SkillExecution> {
        self.entries
            .iter()
            .filter(|e| e.skill_name == name)
            .collect()
    }

    /// Calculate the success rate for a given skill (0.0 to 1.0).
    ///
    /// Returns 0.0 if there are no executions for the skill.
    pub fn success_rate(&self, name: &str) -> f64 {
        let execs: Vec<_> = self.by_skill(name);
        if execs.is_empty() {
            return 0.0;
        }
        let successes = execs.iter().filter(|e| e.success).count();
        successes as f64 / execs.len() as f64
    }

    /// Return the total number of recorded executions.
    pub fn total_executions(&self) -> usize {
        self.entries.len()
    }

    /// Clear all recorded executions.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SkillParamType --

    #[test]
    fn test_param_type_text_name() {
        assert_eq!(SkillParamType::Text.type_name(), "text");
    }

    #[test]
    fn test_param_type_number_name() {
        assert_eq!(SkillParamType::Number.type_name(), "number");
    }

    #[test]
    fn test_param_type_boolean_name() {
        assert_eq!(SkillParamType::Boolean.type_name(), "boolean");
    }

    #[test]
    fn test_param_type_list_name() {
        assert_eq!(SkillParamType::List.type_name(), "list");
    }

    #[test]
    fn test_param_type_json_name() {
        assert_eq!(SkillParamType::Json.type_name(), "json");
    }

    #[test]
    fn test_param_type_text_validates_string() {
        assert!(SkillParamType::Text.validate(&json!("hello")));
        assert!(!SkillParamType::Text.validate(&json!(42)));
    }

    #[test]
    fn test_param_type_number_validates_number() {
        assert!(SkillParamType::Number.validate(&json!(42)));
        assert!(SkillParamType::Number.validate(&json!(3.14)));
        assert!(!SkillParamType::Number.validate(&json!("42")));
    }

    #[test]
    fn test_param_type_boolean_validates_bool() {
        assert!(SkillParamType::Boolean.validate(&json!(true)));
        assert!(SkillParamType::Boolean.validate(&json!(false)));
        assert!(!SkillParamType::Boolean.validate(&json!(1)));
    }

    #[test]
    fn test_param_type_list_validates_array() {
        assert!(SkillParamType::List.validate(&json!([1, 2, 3])));
        assert!(SkillParamType::List.validate(&json!([])));
        assert!(!SkillParamType::List.validate(&json!("not a list")));
    }

    #[test]
    fn test_param_type_json_validates_anything() {
        assert!(SkillParamType::Json.validate(&json!(null)));
        assert!(SkillParamType::Json.validate(&json!("text")));
        assert!(SkillParamType::Json.validate(&json!(42)));
        assert!(SkillParamType::Json.validate(&json!({"key": "val"})));
    }

    // -- SkillDefinition --

    #[test]
    fn test_definition_new_defaults() {
        let d = SkillDefinition::new("test");
        assert_eq!(d.name, "test");
        assert_eq!(d.description, "");
        assert_eq!(d.version, "0.1.0");
        assert_eq!(d.instructions, "");
        assert!(d.tags.is_empty());
        assert!(d.author.is_none());
    }

    #[test]
    fn test_definition_builder_chain() {
        let d = SkillDefinition::new("summarize")
            .with_description("Summarizes text")
            .with_version("1.0.0")
            .with_instructions("Summarize: {{text}}")
            .with_tag("nlp")
            .with_tag("text")
            .with_author("Alice");
        assert_eq!(d.name, "summarize");
        assert_eq!(d.description, "Summarizes text");
        assert_eq!(d.version, "1.0.0");
        assert_eq!(d.instructions, "Summarize: {{text}}");
        assert_eq!(d.tags, vec!["nlp", "text"]);
        assert_eq!(d.author.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_definition_to_json() {
        let d = SkillDefinition::new("test")
            .with_description("A test skill")
            .with_tag("demo");
        let j = d.to_json();
        assert_eq!(j["name"], "test");
        assert_eq!(j["description"], "A test skill");
        assert_eq!(j["tags"].as_array().unwrap().len(), 1);
        assert_eq!(j["tags"][0], "demo");
    }

    #[test]
    fn test_definition_to_json_includes_author() {
        let d = SkillDefinition::new("s").with_author("Bob");
        let j = d.to_json();
        assert_eq!(j["author"], "Bob");
    }

    #[test]
    fn test_definition_to_json_null_author() {
        let d = SkillDefinition::new("s");
        let j = d.to_json();
        assert!(j["author"].is_null());
    }

    #[test]
    fn test_definition_matches_tag_case_insensitive() {
        let d = SkillDefinition::new("s").with_tag("NLP");
        assert!(d.matches_tag("nlp"));
        assert!(d.matches_tag("NLP"));
        assert!(d.matches_tag("Nlp"));
        assert!(!d.matches_tag("other"));
    }

    #[test]
    fn test_definition_matches_tag_no_tags() {
        let d = SkillDefinition::new("s");
        assert!(!d.matches_tag("anything"));
    }

    // -- SkillParameter --

    #[test]
    fn test_parameter_new_defaults() {
        let p = SkillParameter::new("input", SkillParamType::Text);
        assert_eq!(p.name, "input");
        assert_eq!(p.param_type, SkillParamType::Text);
        assert!(p.required);
        assert!(p.default_value.is_none());
        assert_eq!(p.description, "");
    }

    #[test]
    fn test_parameter_builder_chain() {
        let p = SkillParameter::new("count", SkillParamType::Number)
            .optional()
            .with_default(json!(10))
            .with_description("Number of items");
        assert!(!p.required);
        assert_eq!(p.default_value, Some(json!(10)));
        assert_eq!(p.description, "Number of items");
    }

    #[test]
    fn test_parameter_with_required() {
        let p = SkillParameter::new("x", SkillParamType::Text).with_required(false);
        assert!(!p.required);
        let p2 = p.with_required(true);
        assert!(p2.required);
    }

    // -- SkillTemplate --

    #[test]
    fn test_template_new() {
        let def = SkillDefinition::new("t");
        let tmpl = SkillTemplate::new(def);
        assert_eq!(tmpl.definition.name, "t");
        assert!(tmpl.parameters.is_empty());
    }

    #[test]
    fn test_template_add_parameter() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("t"))
            .add_parameter(SkillParameter::new("a", SkillParamType::Text))
            .add_parameter(SkillParameter::new("b", SkillParamType::Number));
        assert_eq!(tmpl.parameters.len(), 2);
    }

    #[test]
    fn test_template_required_params() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("t"))
            .add_parameter(SkillParameter::new("req1", SkillParamType::Text))
            .add_parameter(SkillParameter::new("opt1", SkillParamType::Text).optional())
            .add_parameter(SkillParameter::new("req2", SkillParamType::Number));
        let required = tmpl.required_params();
        assert_eq!(required.len(), 2);
        let names: Vec<&str> = required.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"req1"));
        assert!(names.contains(&"req2"));
    }

    #[test]
    fn test_template_render_simple() {
        let tmpl =
            SkillTemplate::new(SkillDefinition::new("greet").with_instructions("Hello, {{name}}!"))
                .add_parameter(SkillParameter::new("name", SkillParamType::Text));

        let mut args = HashMap::new();
        args.insert("name".into(), json!("World"));
        let result = tmpl.render(&args).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_template_render_multiple_placeholders() {
        let tmpl = SkillTemplate::new(
            SkillDefinition::new("msg")
                .with_instructions("From {{sender}} to {{receiver}}: {{body}}"),
        )
        .add_parameter(SkillParameter::new("sender", SkillParamType::Text))
        .add_parameter(SkillParameter::new("receiver", SkillParamType::Text))
        .add_parameter(SkillParameter::new("body", SkillParamType::Text));

        let mut args = HashMap::new();
        args.insert("sender".into(), json!("Alice"));
        args.insert("receiver".into(), json!("Bob"));
        args.insert("body".into(), json!("Hi!"));
        let result = tmpl.render(&args).unwrap();
        assert_eq!(result, "From Alice to Bob: Hi!");
    }

    #[test]
    fn test_template_render_with_default() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("s").with_instructions("Count: {{n}}"))
            .add_parameter(
                SkillParameter::new("n", SkillParamType::Number)
                    .optional()
                    .with_default(json!(5)),
            );

        let args = HashMap::new();
        let result = tmpl.render(&args).unwrap();
        assert_eq!(result, "Count: 5");
    }

    #[test]
    fn test_template_render_number_arg() {
        let tmpl =
            SkillTemplate::new(SkillDefinition::new("s").with_instructions("Items: {{count}}"))
                .add_parameter(SkillParameter::new("count", SkillParamType::Number));

        let mut args = HashMap::new();
        args.insert("count".into(), json!(42));
        let result = tmpl.render(&args).unwrap();
        assert_eq!(result, "Items: 42");
    }

    #[test]
    fn test_template_validate_args_missing_required() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("s"))
            .add_parameter(SkillParameter::new("required_field", SkillParamType::Text));

        let args = HashMap::new();
        let err = tmpl.validate_args(&args);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("required_field"));
    }

    #[test]
    fn test_template_validate_args_wrong_type() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("s"))
            .add_parameter(SkillParameter::new("num", SkillParamType::Number));

        let mut args = HashMap::new();
        args.insert("num".into(), json!("not a number"));
        let err = tmpl.validate_args(&args);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("num"));
        assert!(msg.contains("number"));
    }

    #[test]
    fn test_template_validate_args_optional_missing_ok() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("s"))
            .add_parameter(SkillParameter::new("opt", SkillParamType::Text).optional());

        let args = HashMap::new();
        assert!(tmpl.validate_args(&args).is_ok());
    }

    #[test]
    fn test_template_validate_args_required_with_default_ok() {
        let tmpl = SkillTemplate::new(SkillDefinition::new("s"))
            .add_parameter(SkillParameter::new("x", SkillParamType::Number).with_default(json!(0)));

        let args = HashMap::new();
        assert!(tmpl.validate_args(&args).is_ok());
    }

    // -- SkillRegistry --

    #[test]
    fn test_registry_new_is_empty() {
        let reg = SkillRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let reg = SkillRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = SkillRegistry::new();
        let tmpl = SkillTemplate::new(SkillDefinition::new("greet"));
        reg.register(tmpl).unwrap();
        assert_eq!(reg.len(), 1);
        let got = reg.get("greet").unwrap();
        assert_eq!(got.definition.name, "greet");
    }

    #[test]
    fn test_registry_duplicate_registration_fails() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("dup")))
            .unwrap();
        let err = reg.register(SkillTemplate::new(SkillDefinition::new("dup")));
        assert!(err.is_err());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = SkillRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("rem")))
            .unwrap();
        reg.unregister("rem").unwrap();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_unregister_missing_fails() {
        let mut reg = SkillRegistry::new();
        assert!(reg.unregister("ghost").is_err());
    }

    #[test]
    fn test_registry_search_by_name() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("summarize_text")))
            .unwrap();
        reg.register(SkillTemplate::new(SkillDefinition::new("translate")))
            .unwrap();
        let results = reg.search("summar");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].definition.name, "summarize_text");
    }

    #[test]
    fn test_registry_search_by_description() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s1").with_description("Analyzes sentiment"),
        ))
        .unwrap();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s2").with_description("Generates code"),
        ))
        .unwrap();
        let results = reg.search("sentiment");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].definition.name, "s1");
    }

    #[test]
    fn test_registry_search_by_tag() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s1").with_tag("nlp"),
        ))
        .unwrap();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s2").with_tag("code"),
        ))
        .unwrap();
        let results = reg.search("nlp");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_registry_search_case_insensitive() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("MySkill")))
            .unwrap();
        let results = reg.search("myskill");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_registry_search_no_match() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("abc")))
            .unwrap();
        let results = reg.search("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_search_empty_registry() {
        let reg = SkillRegistry::new();
        let results = reg.search("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_list_by_tag() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s1").with_tag("nlp"),
        ))
        .unwrap();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s2").with_tag("nlp").with_tag("code"),
        ))
        .unwrap();
        reg.register(SkillTemplate::new(
            SkillDefinition::new("s3").with_tag("code"),
        ))
        .unwrap();
        let nlp = reg.list_by_tag("nlp");
        assert_eq!(nlp.len(), 2);
        let code = reg.list_by_tag("code");
        assert_eq!(code.len(), 2);
        let other = reg.list_by_tag("other");
        assert!(other.is_empty());
    }

    #[test]
    fn test_registry_all() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("a")))
            .unwrap();
        reg.register(SkillTemplate::new(SkillDefinition::new("b")))
            .unwrap();
        assert_eq!(reg.all().len(), 2);
    }

    #[test]
    fn test_registry_to_json() {
        let mut reg = SkillRegistry::new();
        reg.register(SkillTemplate::new(SkillDefinition::new("x")))
            .unwrap();
        let j = reg.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
        assert_eq!(j[0]["name"], "x");
    }

    // -- SkillExecution --

    #[test]
    fn test_execution_to_json() {
        let exec = SkillExecution {
            skill_name: "greet".into(),
            args: {
                let mut m = HashMap::new();
                m.insert("name".into(), json!("World"));
                m
            },
            rendered_output: "Hello, World!".into(),
            executed_at: "2026-03-08T00:00:00Z".into(),
            success: true,
            error: None,
        };
        let j = exec.to_json();
        assert_eq!(j["skill_name"], "greet");
        assert_eq!(j["success"], true);
        assert!(j["error"].is_null());
        assert_eq!(j["rendered_output"], "Hello, World!");
    }

    #[test]
    fn test_execution_to_json_with_error() {
        let exec = SkillExecution {
            skill_name: "fail".into(),
            args: HashMap::new(),
            rendered_output: String::new(),
            executed_at: "2026-03-08T00:00:00Z".into(),
            success: false,
            error: Some("something broke".into()),
        };
        let j = exec.to_json();
        assert_eq!(j["success"], false);
        assert_eq!(j["error"], "something broke");
    }

    // -- SkillHistory --

    #[test]
    fn test_history_new_empty() {
        let h = SkillHistory::new(100);
        assert_eq!(h.total_executions(), 0);
    }

    fn make_exec(name: &str, success: bool) -> SkillExecution {
        SkillExecution {
            skill_name: name.into(),
            args: HashMap::new(),
            rendered_output: String::new(),
            executed_at: "2026-03-08T00:00:00Z".into(),
            success,
            error: if success { None } else { Some("err".into()) },
        }
    }

    #[test]
    fn test_history_record_and_total() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("a", true));
        h.record(make_exec("b", false));
        assert_eq!(h.total_executions(), 2);
    }

    #[test]
    fn test_history_recent() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("a", true));
        h.record(make_exec("b", true));
        h.record(make_exec("c", true));
        let recent = h.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].skill_name, "b");
        assert_eq!(recent[1].skill_name, "c");
    }

    #[test]
    fn test_history_recent_more_than_available() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("a", true));
        let recent = h.recent(5);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_history_by_skill() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("a", true));
        h.record(make_exec("b", false));
        h.record(make_exec("a", false));
        let a_execs = h.by_skill("a");
        assert_eq!(a_execs.len(), 2);
        let b_execs = h.by_skill("b");
        assert_eq!(b_execs.len(), 1);
    }

    #[test]
    fn test_history_by_skill_no_match() {
        let h = SkillHistory::new(10);
        assert!(h.by_skill("none").is_empty());
    }

    #[test]
    fn test_history_success_rate_all_success() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("s", true));
        h.record(make_exec("s", true));
        assert!((h.success_rate("s") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_success_rate_mixed() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("s", true));
        h.record(make_exec("s", false));
        h.record(make_exec("s", true));
        h.record(make_exec("s", false));
        assert!((h.success_rate("s") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_success_rate_no_executions() {
        let h = SkillHistory::new(10);
        assert!((h.success_rate("none") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_success_rate_all_failures() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("f", false));
        h.record(make_exec("f", false));
        assert!((h.success_rate("f") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_max_entries_eviction() {
        let mut h = SkillHistory::new(3);
        h.record(make_exec("a", true));
        h.record(make_exec("b", true));
        h.record(make_exec("c", true));
        h.record(make_exec("d", true));
        assert_eq!(h.total_executions(), 3);
        // "a" should have been evicted
        let names: Vec<&str> = h.recent(3).iter().map(|e| e.skill_name.as_str()).collect();
        assert!(!names.contains(&"a"));
        assert!(names.contains(&"d"));
    }

    #[test]
    fn test_history_clear() {
        let mut h = SkillHistory::new(10);
        h.record(make_exec("a", true));
        h.record(make_exec("b", true));
        h.clear();
        assert_eq!(h.total_executions(), 0);
    }

    #[test]
    fn test_history_max_entries_one() {
        let mut h = SkillHistory::new(1);
        h.record(make_exec("first", true));
        h.record(make_exec("second", true));
        assert_eq!(h.total_executions(), 1);
        assert_eq!(h.recent(1)[0].skill_name, "second");
    }
}
