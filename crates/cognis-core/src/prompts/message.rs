use std::collections::HashMap;

use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::messages::{AIMessage, ChatMessage, HumanMessage, Message, SystemMessage};

use super::base::PromptTemplate;

/// A message-level prompt template that produces typed messages.
pub enum MessagePromptTemplate {
    Human(PromptTemplate),
    Ai(PromptTemplate),
    System(PromptTemplate),
    Chat {
        template: PromptTemplate,
        role: String,
    },
}

impl MessagePromptTemplate {
    /// Create from a role string and template text.
    pub fn from_role(role: &str, template: impl Into<String>) -> Result<Self> {
        let pt = PromptTemplate::from_template(template);
        match role {
            "human" | "user" => Ok(Self::Human(pt)),
            "ai" | "assistant" => Ok(Self::Ai(pt)),
            "system" => Ok(Self::System(pt)),
            other => Ok(Self::Chat {
                template: pt,
                role: other.to_string(),
            }),
        }
    }

    /// Get the input variables required by this template.
    pub fn input_variables(&self) -> &[String] {
        match self {
            Self::Human(pt) | Self::Ai(pt) | Self::System(pt) => &pt.input_variables,
            Self::Chat { template, .. } => &template.input_variables,
        }
    }

    /// Format the template into a single message.
    pub fn format_messages(&self, kwargs: &HashMap<String, Value>) -> Result<Vec<Message>> {
        match self {
            Self::Human(pt) => {
                let text = pt.format(kwargs)?;
                Ok(vec![Message::Human(HumanMessage::new(&text))])
            }
            Self::Ai(pt) => {
                let text = pt.format(kwargs)?;
                Ok(vec![Message::Ai(AIMessage::new(&text))])
            }
            Self::System(pt) => {
                let text = pt.format(kwargs)?;
                Ok(vec![Message::System(SystemMessage::new(&text))])
            }
            Self::Chat { template, role } => {
                let text = template.format(kwargs)?;
                Ok(vec![Message::Chat(ChatMessage::new(role, &text))])
            }
        }
    }
}

/// A placeholder for injecting a list of messages into a chat template.
pub struct MessagesPlaceholder {
    pub variable_name: String,
    pub optional: bool,
    pub n_messages: Option<usize>,
}

impl MessagesPlaceholder {
    pub fn new(variable_name: impl Into<String>) -> Self {
        Self {
            variable_name: variable_name.into(),
            optional: false,
            n_messages: None,
        }
    }

    pub fn optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }

    pub fn n_messages(mut self, n: usize) -> Self {
        self.n_messages = Some(n);
        self
    }

    pub fn input_variables(&self) -> Vec<String> {
        if self.optional {
            vec![]
        } else {
            vec![self.variable_name.clone()]
        }
    }

    pub fn format_messages(&self, kwargs: &HashMap<String, Value>) -> Result<Vec<Message>> {
        let value = kwargs.get(&self.variable_name);

        let messages_value = match (value, self.optional) {
            (Some(v), _) => v.clone(),
            (None, true) => Value::Array(vec![]),
            (None, false) => {
                return Err(CognisError::Other(format!(
                    "Missing required variable '{}'",
                    self.variable_name
                )));
            }
        };

        let messages: Vec<Message> = serde_json::from_value(messages_value).map_err(|e| {
            CognisError::Other(format!(
                "Failed to deserialize messages for '{}': {}",
                self.variable_name, e
            ))
        })?;

        let messages = if let Some(n) = self.n_messages {
            if messages.len() > n {
                messages[messages.len() - n..].to_vec()
            } else {
                messages
            }
        } else {
            messages
        };

        Ok(messages)
    }
}
