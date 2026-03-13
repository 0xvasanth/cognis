use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::messages::Message;
use crate::prompt_values::{ChatPromptValue, PromptValue};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::PartialValue;
use super::message::{MessagePromptTemplate, MessagesPlaceholder};

/// An element in a ChatPromptTemplate's message list.
pub enum MessageLike {
    /// A concrete message (no template variables).
    Concrete(Box<Message>),
    /// A message-level template that produces one message.
    Template(MessagePromptTemplate),
    /// A placeholder for injecting a list of messages.
    Placeholder(MessagesPlaceholder),
}

/// A multi-message prompt template for chat models.
///
/// Composes message templates, placeholders, and concrete messages into
/// a sequence of messages. Implements `Runnable` for LCEL chains.
pub struct ChatPromptTemplate {
    pub messages: Vec<MessageLike>,
    pub input_variables: Vec<String>,
    pub partial_variables: HashMap<String, PartialValue>,
}

impl ChatPromptTemplate {
    /// Build a `ChatPromptTemplate` from a list of `(role, template)` tuples
    /// and other `MessageLike` variants.
    ///
    /// Accepts tuples of `(&str, &str)` where role is one of:
    /// `"human"`, `"ai"`, `"system"`, or `"placeholder"`.
    pub fn from_messages(specs: Vec<(&str, &str)>) -> Result<Self> {
        let mut messages = Vec::new();
        let mut input_variables = Vec::new();

        for (role, content) in specs {
            if role == "placeholder" {
                // Content is the variable name, e.g. "{history}" -> "history"
                let var_name = content
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .to_string();
                let placeholder = MessagesPlaceholder::new(&var_name).optional(true);
                messages.push(MessageLike::Placeholder(placeholder));
                // Optional placeholders don't contribute required input variables
            } else {
                let template = MessagePromptTemplate::from_role(role, content)?;
                for v in template.input_variables() {
                    if !input_variables.contains(v) {
                        input_variables.push(v.clone());
                    }
                }
                messages.push(MessageLike::Template(template));
            }
        }

        Ok(Self {
            messages,
            input_variables,
            partial_variables: HashMap::new(),
        })
    }

    /// Create from a list of `MessageLike` entries with explicit input variables.
    pub fn new(messages: Vec<MessageLike>, input_variables: Vec<String>) -> Self {
        Self {
            messages,
            input_variables,
            partial_variables: HashMap::new(),
        }
    }

    /// Create a `ChatPromptTemplate` from a single string template.
    ///
    /// Creates a human message template from the provided string.
    /// Equivalent to Python's `ChatPromptTemplate.from_template(template)`.
    pub fn from_template(template: &str) -> Result<Self> {
        Self::from_messages(vec![("human", template)])
    }

    /// Extend this template by appending multiple `(role, content)` specs.
    ///
    /// Equivalent to Python's `ChatPromptTemplate.extend(messages)`.
    pub fn extend(&mut self, specs: Vec<(&str, &str)>) -> Result<()> {
        for (role, content) in specs {
            self.append(role, content)?;
        }
        Ok(())
    }

    /// Pre-fill some template variables.
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

    /// Format all messages with the given variables.
    pub fn format_messages(&self, kwargs: &HashMap<String, Value>) -> Result<Vec<Message>> {
        let merged = self.merge_variables(kwargs);
        let mut result = Vec::new();

        for msg_like in &self.messages {
            match msg_like {
                MessageLike::Concrete(msg) => {
                    result.push(*msg.clone());
                }
                MessageLike::Template(template) => {
                    result.extend(template.format_messages(&merged)?);
                }
                MessageLike::Placeholder(placeholder) => {
                    result.extend(placeholder.format_messages(&merged)?);
                }
            }
        }

        Ok(result)
    }

    /// Format all messages and wrap as a `ChatPromptValue`.
    pub fn format_prompt(&self, kwargs: &HashMap<String, Value>) -> Result<Box<dyn PromptValue>> {
        let messages = self.format_messages(kwargs)?;
        Ok(Box::new(ChatPromptValue::new(messages)))
    }

    /// Append a new message spec to this template.
    pub fn append(&mut self, role: &str, content: &str) -> Result<()> {
        if role == "placeholder" {
            let var_name = content
                .trim_start_matches('{')
                .trim_end_matches('}')
                .to_string();
            self.messages.push(MessageLike::Placeholder(
                MessagesPlaceholder::new(var_name).optional(true),
            ));
        } else {
            let template = MessagePromptTemplate::from_role(role, content)?;
            for v in template.input_variables() {
                if !self.input_variables.contains(v) {
                    self.input_variables.push(v.clone());
                }
            }
            self.messages.push(MessageLike::Template(template));
        }
        Ok(())
    }
}

#[async_trait]
impl Runnable for ChatPromptTemplate {
    fn name(&self) -> &str {
        "ChatPromptTemplate"
    }

    /// Input: JSON object with template variables.
    /// Output: JSON array of serialized messages.
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
        let messages = self.format_messages(&kwargs)?;
        serde_json::to_value(&messages).map_err(Into::into)
    }
}

impl std::fmt::Display for ChatPromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChatPromptTemplate(input_variables={:?}, messages={})",
            self.input_variables,
            self.messages.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_from_messages_basic() {
        let template = ChatPromptTemplate::from_messages(vec![
            ("system", "You are a helpful assistant"),
            ("human", "{question}"),
        ])
        .unwrap();
        assert_eq!(template.input_variables, vec!["question".to_string()]);
        assert_eq!(template.messages.len(), 2);
    }

    #[test]
    fn test_from_template() {
        let template = ChatPromptTemplate::from_template("Hello {name}!").unwrap();
        assert_eq!(template.input_variables, vec!["name".to_string()]);
        assert_eq!(template.messages.len(), 1);
    }

    #[test]
    fn test_format_messages() {
        let template = ChatPromptTemplate::from_messages(vec![
            ("system", "You are helpful"),
            ("human", "My name is {name}"),
        ])
        .unwrap();
        let mut kwargs = HashMap::new();
        kwargs.insert("name".to_string(), json!("Alice"));
        let messages = template.format_messages(&kwargs).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content().text(), "My name is Alice");
    }

    #[test]
    fn test_append() {
        let mut template =
            ChatPromptTemplate::from_messages(vec![("system", "You are helpful")]).unwrap();
        template.append("human", "{question}").unwrap();
        assert_eq!(template.messages.len(), 2);
        assert!(template.input_variables.contains(&"question".to_string()));
    }

    #[test]
    fn test_extend() {
        let mut template =
            ChatPromptTemplate::from_messages(vec![("system", "You are helpful")]).unwrap();
        template
            .extend(vec![
                ("human", "{question}"),
                ("ai", "Let me help with {question}"),
            ])
            .unwrap();
        assert_eq!(template.messages.len(), 3);
    }

    #[test]
    fn test_partial() {
        let template = ChatPromptTemplate::from_messages(vec![
            ("system", "You are {role}"),
            ("human", "{question}"),
        ])
        .unwrap();
        let partial = template.partial(HashMap::from([(
            "role".to_string(),
            PartialValue::Static(json!("helpful")),
        )]));
        assert!(!partial.input_variables.contains(&"role".to_string()));
        assert!(partial.input_variables.contains(&"question".to_string()));
    }

    #[test]
    fn test_placeholder() {
        let template = ChatPromptTemplate::from_messages(vec![
            ("system", "You are helpful"),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])
        .unwrap();
        // Placeholder variables are optional, not added to input_variables
        assert_eq!(template.input_variables, vec!["question".to_string()]);
    }

    #[tokio::test]
    async fn test_runnable_invoke() {
        let template = ChatPromptTemplate::from_messages(vec![("human", "Hello {name}")]).unwrap();
        let result = template
            .invoke(json!({"name": "World"}), None)
            .await
            .unwrap();
        assert!(result.is_array());
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }
}
