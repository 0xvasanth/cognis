//! Chat prompt templates for composing multi-message prompts.
//!
//! Provides [`ChatPromptTemplate`] for building prompts that consist of
//! multiple messages (system, human, AI) with `{variable}` placeholders,
//! and [`MessagePromptTemplate`] for individual message templates.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::messages::Message;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

/// Extract `{variable}` placeholders from a template string.
///
/// Recognises `{var_name}` patterns and skips escaped braces (`{{` / `}}`).
/// Returns a deduplicated list of variable names in the order they first appear.
pub fn extract_template_variables(template: &str) -> Vec<String> {
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

/// Format a template string by substituting `{var}` placeholders with values.
fn format_template(template: &str, variables: &HashMap<String, String>) -> Result<String> {
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
                RustChainError::Other(format!(
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

/// Convert a `serde_json::Value` to a string suitable for template substitution.
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A template for a single message in a chat prompt.
///
/// Each variant holds a template string with `{variable}` placeholders,
/// except `Placeholder` which holds a variable name that expands to a list
/// of messages at format time.
#[derive(Debug, Clone)]
pub enum MessagePromptTemplate {
    /// System message template.
    System(String),
    /// Human message template.
    Human(String),
    /// AI message template.
    Ai(String),
    /// Expands to a list of messages from the variables map.
    /// The contained string is the variable name (without braces).
    Placeholder(String),
}

impl MessagePromptTemplate {
    /// Format this template into one or more messages.
    ///
    /// For `System`, `Human`, and `Ai` variants, substitutes `{variable}`
    /// placeholders from `vars` and returns a single-element `Vec`.
    ///
    /// For `Placeholder`, looks up the variable name in `vars` and expects
    /// either a JSON array of message objects or a pre-existing message list.
    pub fn format(&self, vars: &HashMap<String, Value>) -> Result<Vec<Message>> {
        match self {
            MessagePromptTemplate::System(tmpl) => {
                let string_vars = to_string_map(vars);
                let text = format_template(tmpl, &string_vars)?;
                Ok(vec![Message::system(text)])
            }
            MessagePromptTemplate::Human(tmpl) => {
                let string_vars = to_string_map(vars);
                let text = format_template(tmpl, &string_vars)?;
                Ok(vec![Message::human(text)])
            }
            MessagePromptTemplate::Ai(tmpl) => {
                let string_vars = to_string_map(vars);
                let text = format_template(tmpl, &string_vars)?;
                Ok(vec![Message::ai(text)])
            }
            MessagePromptTemplate::Placeholder(var_name) => {
                let value = vars.get(var_name).ok_or_else(|| {
                    RustChainError::Other(format!("Missing placeholder variable '{}'", var_name))
                })?;
                parse_messages_from_value(value)
            }
        }
    }

    /// Return the variable names referenced by this template.
    pub fn variables(&self) -> Vec<String> {
        match self {
            MessagePromptTemplate::System(tmpl)
            | MessagePromptTemplate::Human(tmpl)
            | MessagePromptTemplate::Ai(tmpl) => extract_template_variables(tmpl),
            MessagePromptTemplate::Placeholder(name) => vec![name.clone()],
        }
    }
}

/// Helper to create a `MessagePromptTemplate::Placeholder`.
///
/// This mirrors Python LangChain's `MessagesPlaceholder` which inserts a
/// dynamic list of messages into the prompt at a given position.
///
/// # Example
///
/// ```
/// use rustchain::prompts::chat::messages_placeholder;
///
/// let ph = messages_placeholder("history");
/// ```
pub fn messages_placeholder(variable_name: impl Into<String>) -> MessagePromptTemplate {
    MessagePromptTemplate::Placeholder(variable_name.into())
}

/// A chat prompt template that composes multiple message templates.
///
/// Supports `{variable}` placeholders within individual messages and
/// `Placeholder` entries that expand to lists of messages at format time.
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    /// The ordered list of message templates.
    pub messages: Vec<MessagePromptTemplate>,
    /// All input variable names detected across all templates.
    input_variables: Vec<String>,
    /// Pre-filled variables that do not need to be supplied at format time.
    partial_variables: HashMap<String, Value>,
}

impl ChatPromptTemplate {
    /// Create a new `ChatPromptTemplate` from a list of message templates.
    ///
    /// Input variables are automatically detected from all templates.
    pub fn new(messages: Vec<MessagePromptTemplate>) -> Self {
        let input_variables = collect_variables(&messages);
        Self {
            messages,
            input_variables,
            partial_variables: HashMap::new(),
        }
    }

    /// Convenience constructor from `(role, template)` tuples.
    ///
    /// Supported roles: `"system"`, `"human"` / `"user"`, `"ai"` / `"assistant"`,
    /// `"placeholder"`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustchain::prompts::chat::ChatPromptTemplate;
    ///
    /// let prompt = ChatPromptTemplate::from_messages(vec![
    ///     ("system", "You are a helpful {role}."),
    ///     ("human", "{input}"),
    /// ]);
    /// ```
    pub fn from_messages(tuples: Vec<(&str, &str)>) -> Self {
        let messages: Vec<MessagePromptTemplate> = tuples
            .into_iter()
            .map(|(role, tmpl)| match role.to_lowercase().as_str() {
                "system" => MessagePromptTemplate::System(tmpl.to_string()),
                "human" | "user" => MessagePromptTemplate::Human(tmpl.to_string()),
                "ai" | "assistant" => MessagePromptTemplate::Ai(tmpl.to_string()),
                "placeholder" => {
                    // Strip braces if present: "{history}" -> "history"
                    let name = tmpl.trim_matches(|c| c == '{' || c == '}');
                    MessagePromptTemplate::Placeholder(name.to_string())
                }
                other => {
                    // Default to human for unknown roles
                    eprintln!("Warning: unknown role '{}', treating as human", other);
                    MessagePromptTemplate::Human(tmpl.to_string())
                }
            })
            .collect();

        Self::new(messages)
    }

    /// Format the chat prompt into a list of messages.
    ///
    /// All input variables (minus partials) must be present in `vars`.
    pub fn format_messages(&self, vars: &HashMap<String, Value>) -> Result<Vec<Message>> {
        let merged = self.merge_vars(vars);
        let mut result = Vec::new();
        for template in &self.messages {
            let msgs = template.format(&merged)?;
            result.extend(msgs);
        }
        Ok(result)
    }

    /// Format the chat prompt into a single string representation.
    ///
    /// Each message is rendered as `role: content` on its own line.
    pub fn format_prompt(&self, vars: &HashMap<String, Value>) -> Result<String> {
        let messages = self.format_messages(vars)?;
        let mut parts = Vec::new();
        for msg in &messages {
            let role = msg.message_type().as_str();
            let content = msg.content().text();
            parts.push(format!("{}: {}", role, content));
        }
        Ok(parts.join("\n"))
    }

    /// Pre-fill some variables, returning a new template that requires fewer
    /// inputs at format time.
    pub fn partial(mut self, vars: HashMap<String, Value>) -> Self {
        for (k, v) in vars {
            self.partial_variables.insert(k.clone(), v);
            self.input_variables.retain(|iv| iv != &k);
        }
        self
    }

    /// Return the list of input variables that still need to be supplied.
    pub fn input_variables(&self) -> &[String] {
        &self.input_variables
    }

    /// Merge user-supplied variables with partials.
    fn merge_vars(&self, vars: &HashMap<String, Value>) -> HashMap<String, Value> {
        let mut merged: HashMap<String, Value> = self
            .partial_variables
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        merged.extend(vars.iter().map(|(k, v)| (k.clone(), v.clone())));
        merged
    }
}

#[async_trait]
impl Runnable for ChatPromptTemplate {
    fn name(&self) -> &str {
        "ChatPromptTemplate"
    }

    /// Invoke the chat prompt template.
    ///
    /// Input: a JSON object whose values are substituted into the templates.
    /// Output: a JSON object with a `"messages"` key containing the formatted
    /// message array.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let vars: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(RustChainError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let messages = self.format_messages(&vars)?;
        let serialized = serde_json::to_value(&messages)?;
        Ok(serde_json::json!({ "messages": serialized }))
    }
}

/// Collect all unique variable names from a list of message templates.
fn collect_variables(messages: &[MessagePromptTemplate]) -> Vec<String> {
    let mut vars = Vec::new();
    for msg in messages {
        for v in msg.variables() {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
    }
    vars
}

/// Convert a `HashMap<String, Value>` to `HashMap<String, String>` for
/// template substitution.
fn to_string_map(vars: &HashMap<String, Value>) -> HashMap<String, String> {
    vars.iter()
        .map(|(k, v)| (k.clone(), value_to_string(v)))
        .collect()
}

/// Parse a JSON value into a list of `Message` objects.
///
/// Expects a JSON array where each element has a `"type"` field indicating
/// the message kind (`"human"`, `"ai"`, `"system"`, etc.) or a simplified
/// object with `"role"` and `"content"` fields.
fn parse_messages_from_value(value: &Value) -> Result<Vec<Message>> {
    match value {
        Value::Array(arr) => {
            let mut messages = Vec::with_capacity(arr.len());
            for item in arr {
                let msg = parse_single_message(item)?;
                messages.push(msg);
            }
            Ok(messages)
        }
        _ => Err(RustChainError::TypeMismatch {
            expected: "Array of messages".into(),
            got: format!("{}", value),
        }),
    }
}

/// Parse a single JSON value into a `Message`.
///
/// Supports two formats:
/// 1. Tagged format: `{"type": "human", "content": "..."}`
/// 2. Role format: `{"role": "human", "content": "..."}`
fn parse_single_message(value: &Value) -> Result<Message> {
    // First try direct deserialization (tagged format)
    if let Ok(msg) = serde_json::from_value::<Message>(value.clone()) {
        return Ok(msg);
    }

    // Fall back to role/content format
    let obj = value
        .as_object()
        .ok_or_else(|| RustChainError::TypeMismatch {
            expected: "Object (message)".into(),
            got: format!("{}", value),
        })?;

    let role = obj
        .get("role")
        .and_then(|r| r.as_str())
        .ok_or_else(|| RustChainError::Other("Message missing 'role' field".into()))?;

    let content = obj.get("content").and_then(|c| c.as_str()).unwrap_or("");

    match role {
        "system" => Ok(Message::system(content)),
        "human" | "user" => Ok(Message::human(content)),
        "ai" | "assistant" => Ok(Message::ai(content)),
        other => Err(RustChainError::Other(format!(
            "Unknown message role: '{}'",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- extract_template_variables ----

    #[test]
    fn test_extract_simple_variables() {
        let vars = extract_template_variables("Hello {name}, you are {age} years old.");
        assert_eq!(vars, vec!["name", "age"]);
    }

    #[test]
    fn test_extract_no_variables() {
        let vars = extract_template_variables("No variables here.");
        assert!(vars.is_empty());
    }

    #[test]
    fn test_extract_escaped_braces() {
        let vars = extract_template_variables("Use {{braces}} and {var}.");
        assert_eq!(vars, vec!["var"]);
    }

    #[test]
    fn test_extract_deduplicates() {
        let vars = extract_template_variables("{x} and {x} again");
        assert_eq!(vars, vec!["x"]);
    }

    // ---- MessagePromptTemplate ----

    #[test]
    fn test_system_message_format() {
        let tmpl = MessagePromptTemplate::System("You are a {role}.".into());
        let mut vars = HashMap::new();
        vars.insert("role".into(), json!("helpful assistant"));
        let msgs = tmpl.format(&vars).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content().text(), "You are a helpful assistant.");
        assert_eq!(msgs[0].message_type().as_str(), "system");
    }

    #[test]
    fn test_human_message_format() {
        let tmpl = MessagePromptTemplate::Human("Tell me about {topic}.".into());
        let mut vars = HashMap::new();
        vars.insert("topic".into(), json!("Rust"));
        let msgs = tmpl.format(&vars).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content().text(), "Tell me about Rust.");
        assert_eq!(msgs[0].message_type().as_str(), "human");
    }

    #[test]
    fn test_ai_message_format() {
        let tmpl = MessagePromptTemplate::Ai("I know about {topic}.".into());
        let mut vars = HashMap::new();
        vars.insert("topic".into(), json!("Rust"));
        let msgs = tmpl.format(&vars).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content().text(), "I know about Rust.");
        assert_eq!(msgs[0].message_type().as_str(), "ai");
    }

    #[test]
    fn test_placeholder_format() {
        let tmpl = MessagePromptTemplate::Placeholder("history".into());
        let mut vars = HashMap::new();
        vars.insert(
            "history".into(),
            json!([
                {"role": "human", "content": "Hi"},
                {"role": "ai", "content": "Hello!"}
            ]),
        );
        let msgs = tmpl.format(&vars).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].message_type().as_str(), "human");
        assert_eq!(msgs[1].message_type().as_str(), "ai");
    }

    #[test]
    fn test_placeholder_missing_variable() {
        let tmpl = MessagePromptTemplate::Placeholder("history".into());
        let vars = HashMap::new();
        let err = tmpl.format(&vars).unwrap_err();
        assert!(format!("{}", err).contains("Missing placeholder variable"));
    }

    #[test]
    fn test_message_template_variables() {
        let tmpl = MessagePromptTemplate::System("Hello {name}, role: {role}".into());
        assert_eq!(tmpl.variables(), vec!["name", "role"]);

        let ph = MessagePromptTemplate::Placeholder("chat_history".into());
        assert_eq!(ph.variables(), vec!["chat_history"]);
    }

    // ---- ChatPromptTemplate ----

    #[test]
    fn test_chat_prompt_new() {
        let prompt = ChatPromptTemplate::new(vec![
            MessagePromptTemplate::System("You are {role}.".into()),
            MessagePromptTemplate::Human("{input}".into()),
        ]);
        assert_eq!(prompt.input_variables(), &["role", "input"]);
    }

    #[test]
    fn test_chat_prompt_from_messages() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "You are a {role}."),
            ("human", "{question}"),
        ]);
        assert_eq!(prompt.input_variables(), &["role", "question"]);
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn test_chat_prompt_from_messages_role_aliases() {
        let prompt =
            ChatPromptTemplate::from_messages(vec![("user", "Hello"), ("assistant", "Hi there")]);
        assert_eq!(prompt.messages.len(), 2);
        let msgs = prompt.format_messages(&HashMap::new()).unwrap();
        assert_eq!(msgs[0].message_type().as_str(), "human");
        assert_eq!(msgs[1].message_type().as_str(), "ai");
    }

    #[test]
    fn test_chat_prompt_format_messages() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "You are a {role}."),
            ("human", "{question}"),
        ]);
        let mut vars = HashMap::new();
        vars.insert("role".into(), json!("helpful assistant"));
        vars.insert("question".into(), json!("What is Rust?"));
        let msgs = prompt.format_messages(&vars).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content().text(), "You are a helpful assistant.");
        assert_eq!(msgs[1].content().text(), "What is Rust?");
    }

    #[test]
    fn test_chat_prompt_format_prompt_string() {
        let prompt =
            ChatPromptTemplate::from_messages(vec![("system", "Be helpful."), ("human", "Hi")]);
        let result = prompt.format_prompt(&HashMap::new()).unwrap();
        assert_eq!(result, "system: Be helpful.\nhuman: Hi");
    }

    #[test]
    fn test_chat_prompt_with_placeholder() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "You are helpful."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ]);
        assert!(prompt.input_variables().contains(&"history".to_string()));
        assert!(prompt.input_variables().contains(&"input".to_string()));

        let mut vars = HashMap::new();
        vars.insert(
            "history".into(),
            json!([
                {"role": "human", "content": "Previous question"},
                {"role": "ai", "content": "Previous answer"}
            ]),
        );
        vars.insert("input".into(), json!("New question"));
        let msgs = prompt.format_messages(&vars).unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].message_type().as_str(), "system");
        assert_eq!(msgs[1].message_type().as_str(), "human");
        assert_eq!(msgs[1].content().text(), "Previous question");
        assert_eq!(msgs[2].message_type().as_str(), "ai");
        assert_eq!(msgs[3].message_type().as_str(), "human");
        assert_eq!(msgs[3].content().text(), "New question");
    }

    #[test]
    fn test_chat_prompt_partial() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "You are a {role} who speaks {language}."),
            ("human", "{input}"),
        ]);
        assert_eq!(prompt.input_variables().len(), 3);

        let prompt = prompt.partial(HashMap::from([
            ("role".into(), json!("translator")),
            ("language".into(), json!("French")),
        ]));
        assert_eq!(prompt.input_variables(), &["input"]);

        let mut vars = HashMap::new();
        vars.insert("input".into(), json!("Translate hello"));
        let msgs = prompt.format_messages(&vars).unwrap();
        assert_eq!(
            msgs[0].content().text(),
            "You are a translator who speaks French."
        );
    }

    #[test]
    fn test_chat_prompt_missing_variable_error() {
        let prompt = ChatPromptTemplate::from_messages(vec![("human", "Hello {name}")]);
        let vars = HashMap::new();
        let err = prompt.format_messages(&vars).unwrap_err();
        assert!(format!("{}", err).contains("Missing variable 'name'"));
    }

    #[test]
    fn test_messages_placeholder_helper() {
        let ph = messages_placeholder("chat_history");
        match &ph {
            MessagePromptTemplate::Placeholder(name) => {
                assert_eq!(name, "chat_history");
            }
            _ => panic!("Expected Placeholder variant"),
        }
    }

    #[test]
    fn test_from_messages_placeholder_strips_braces() {
        let prompt = ChatPromptTemplate::from_messages(vec![("placeholder", "{history}")]);
        match &prompt.messages[0] {
            MessagePromptTemplate::Placeholder(name) => {
                assert_eq!(name, "history");
            }
            _ => panic!("Expected Placeholder variant"),
        }
    }

    #[test]
    fn test_empty_placeholder_list() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "Hello"),
            ("placeholder", "{history}"),
            ("human", "World"),
        ]);
        let mut vars = HashMap::new();
        vars.insert("history".into(), json!([]));
        let msgs = prompt.format_messages(&vars).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content().text(), "Hello");
        assert_eq!(msgs[1].content().text(), "World");
    }

    #[test]
    fn test_numeric_variable_substitution() {
        let prompt = ChatPromptTemplate::from_messages(vec![("human", "Count to {n}")]);
        let mut vars = HashMap::new();
        vars.insert("n".into(), json!(5));
        let msgs = prompt.format_messages(&vars).unwrap();
        assert_eq!(msgs[0].content().text(), "Count to 5");
    }

    #[tokio::test]
    async fn test_runnable_invoke() {
        let prompt = ChatPromptTemplate::from_messages(vec![
            ("system", "You are {role}."),
            ("human", "{input}"),
        ]);
        let result = prompt
            .invoke(json!({"role": "helpful", "input": "Hi"}), None)
            .await
            .unwrap();
        let messages = result.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn test_runnable_invoke_type_error() {
        let prompt = ChatPromptTemplate::from_messages(vec![("human", "Hi")]);
        let err = prompt
            .invoke(json!("not an object"), None)
            .await
            .unwrap_err();
        assert!(format!("{}", err).contains("Type mismatch"));
    }
}
