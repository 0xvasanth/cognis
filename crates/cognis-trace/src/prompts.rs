//! `PromptStore` trait — pull versioned prompts from an external store.
//! Concrete impls live alongside their backend (`exporters/langfuse/prompts.rs`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::TraceError;

/// Two prompt shapes Langfuse supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptBody {
    /// Single string template.
    Text {
        /// Templated string.
        prompt: String,
    },
    /// Sequence of role/content messages, possibly with placeholders.
    Chat {
        /// Templated messages.
        messages: Vec<ChatMessageTemplate>,
    },
}

/// One message in a chat prompt template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatMessageTemplate {
    /// Concrete role/content.
    #[serde(rename = "chatmessage")]
    Message {
        /// Role string ("system", "user", "assistant", ...).
        role: String,
        /// Templated content.
        content: String,
    },
    /// Placeholder for runtime-injected messages (Langfuse's `placeholder`).
    #[serde(rename = "placeholder")]
    Placeholder {
        /// Placeholder name.
        name: String,
    },
}

/// A versioned prompt fetched from a `PromptStore`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// Stable name.
    pub name: String,
    /// Monotonic version number.
    pub version: u32,
    /// The template body.
    pub body: PromptBody,
    /// Free-form config (e.g. `{"model": "gpt-4o", "temperature": 0.7}`).
    #[serde(default)]
    pub config: serde_json::Value,
    /// Deployment labels (e.g. ["production", "experimental"]).
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Fetcher for versioned prompts.
#[async_trait]
pub trait PromptStore: Send + Sync {
    /// Fetch by name; the implementation chooses what "current" means
    /// (latest version, "production" label, etc.).
    async fn get(&self, name: &str) -> Result<Prompt, TraceError>;

    /// Fetch a specific version.
    async fn get_version(&self, name: &str, version: u32) -> Result<Prompt, TraceError>;

    /// Fetch by label (e.g. "production").
    async fn get_label(&self, name: &str, label: &str) -> Result<Prompt, TraceError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_body_text_round_trips() {
        let p = PromptBody::Text {
            prompt: "hi {name}".into(),
        };
        let s = serde_json::to_string(&p).unwrap();
        let p2: PromptBody = serde_json::from_str(&s).unwrap();
        match p2 {
            PromptBody::Text { prompt } => assert_eq!(prompt, "hi {name}"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn prompt_body_chat_with_placeholder_round_trips() {
        let p = PromptBody::Chat {
            messages: vec![
                ChatMessageTemplate::Message {
                    role: "system".into(),
                    content: "you are helpful".into(),
                },
                ChatMessageTemplate::Placeholder {
                    name: "history".into(),
                },
            ],
        };
        let s = serde_json::to_string(&p).unwrap();
        let _: PromptBody = serde_json::from_str(&s).unwrap();
    }
}
