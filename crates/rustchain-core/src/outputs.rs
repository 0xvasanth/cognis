use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use crate::messages::{AIMessage, AIMessageChunk, Message};

/// A single text generation output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Generation {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_info: Option<HashMap<String, Value>>,
}

impl Generation {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            generation_info: None,
        }
    }
}

/// A single chat generation output containing a structured message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatGeneration {
    pub text: String,
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_info: Option<HashMap<String, Value>>,
}

impl ChatGeneration {
    /// Create a new `ChatGeneration` from an `AIMessage`.
    pub fn new(message: AIMessage) -> Self {
        let text = message.base.content.text();
        Self {
            text,
            message: Message::Ai(message),
            generation_info: None,
        }
    }

    /// Create a new `ChatGeneration` from any `Message` variant.
    pub fn from_message(message: Message) -> Self {
        let text = message.content().text();
        Self {
            text,
            message,
            generation_info: None,
        }
    }
}

/// Result of a chat model call with a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatResult {
    pub generations: Vec<ChatGeneration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_output: Option<HashMap<String, Value>>,
}

/// Metadata for a single execution of a chain or model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunInfo {
    pub run_id: Uuid,
}

/// Container for results of an LLM call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMResult {
    pub generations: Vec<Vec<Generation>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_output: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<Vec<RunInfo>>,
}

/// A streaming chunk of a text generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationChunk {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_info: Option<HashMap<String, Value>>,
}

impl GenerationChunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            generation_info: None,
        }
    }

    /// Concatenate two generation chunks.
    pub fn add(&self, other: &GenerationChunk) -> GenerationChunk {
        let mut info = self.generation_info.clone();
        if let Some(other_info) = &other.generation_info {
            info.get_or_insert_with(HashMap::new).extend(
                other_info.iter().map(|(k, v)| (k.clone(), v.clone())),
            );
        }
        GenerationChunk {
            text: format!("{}{}", self.text, other.text),
            generation_info: info,
        }
    }
}

/// A streaming chunk of a chat generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatGenerationChunk {
    pub text: String,
    pub message: AIMessageChunk,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_info: Option<HashMap<String, Value>>,
}

impl ChatGenerationChunk {
    pub fn new(message: AIMessageChunk) -> Self {
        let text = message.base.content.text();
        Self {
            text,
            message,
            generation_info: None,
        }
    }

    /// Concatenate two chat generation chunks.
    pub fn add(&self, other: &ChatGenerationChunk) -> ChatGenerationChunk {
        let mut info = self.generation_info.clone();
        if let Some(other_info) = &other.generation_info {
            info.get_or_insert_with(HashMap::new).extend(
                other_info.iter().map(|(k, v)| (k.clone(), v.clone())),
            );
        }
        ChatGenerationChunk {
            text: format!("{}{}", self.text, other.text),
            message: self.message.clone().add(other.message.clone()),
            generation_info: info,
        }
    }
}

/// Merge a vector of chat generation chunks into a single chunk.
pub fn merge_chat_generation_chunks(
    chunks: Vec<ChatGenerationChunk>,
) -> Option<ChatGenerationChunk> {
    chunks
        .into_iter()
        .reduce(|acc, chunk| acc.add(&chunk))
}

impl LLMResult {
    /// Flatten generations into a list of single-generation LLMResults.
    /// Token usage is kept only for the first result.
    pub fn flatten(&self) -> Vec<LLMResult> {
        self.generations
            .iter()
            .enumerate()
            .map(|(i, gen_list)| {
                let llm_output = if i == 0 {
                    self.llm_output.clone()
                } else {
                    self.llm_output.as_ref().map(|o| {
                        let mut out = o.clone();
                        out.insert("token_usage".into(), Value::Object(Default::default()));
                        out
                    })
                };
                LLMResult {
                    generations: vec![gen_list.clone()],
                    llm_output,
                    run: None,
                }
            })
            .collect()
    }
}
