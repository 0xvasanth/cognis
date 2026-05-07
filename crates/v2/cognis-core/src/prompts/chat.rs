//! `ChatPromptTemplate` — typed templating that produces `Vec<Message>`.

use std::marker::PhantomData;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use crate::content::ContentPart;
use crate::message::Message;
use crate::prompts::template::{render, scan_variables};
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// One element in a `ChatPromptTemplate`.
#[derive(Debug, Clone)]
enum Part {
    /// A templated message at a fixed role.
    Templated { role: Role, template: String },
    /// A templated message that also carries multimodal parts. The text
    /// template is rendered against the input; the parts are passed
    /// through as-is.
    Multimodal {
        role: Role,
        template: String,
        parts: Vec<ContentPart>,
    },
    /// Drop in a `Vec<Message>` from a named field of the input.
    Placeholder { key: String, optional: bool },
}

/// The role assigned to a templated message part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Renders as `Message::System`.
    System,
    /// Renders as `Message::Human`.
    Human,
    /// Renders as `Message::Ai`.
    Ai,
}

/// A typed chat prompt — renders to `Vec<Message>` when invoked.
///
/// Build via the fluent API:
///
/// ```no_run
/// use cognis2_core::prompts::ChatPromptTemplate;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct In { name: String }
///
/// let prompt: ChatPromptTemplate<In> = ChatPromptTemplate::new()
///     .system("You are a helpful assistant.")
///     .placeholder("history")
///     .human("Hello, my name is {name}.");
/// ```
///
/// Placeholders pull a `Vec<Message>` from a named field of the input
/// (the field must serialize to a JSON array of `Message` objects).
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate<I = Value> {
    parts: Vec<Part>,
    _input: PhantomData<fn() -> I>,
}

impl<I> Default for ChatPromptTemplate<I> {
    fn default() -> Self {
        Self {
            parts: Vec::new(),
            _input: PhantomData,
        }
    }
}

impl<I> ChatPromptTemplate<I>
where
    I: Serialize + Send + Sync + 'static,
{
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a system-role templated message.
    pub fn system(mut self, template: impl Into<String>) -> Self {
        self.parts.push(Part::Templated {
            role: Role::System,
            template: template.into(),
        });
        self
    }

    /// Append a human-role templated message.
    pub fn human(mut self, template: impl Into<String>) -> Self {
        self.parts.push(Part::Templated {
            role: Role::Human,
            template: template.into(),
        });
        self
    }

    /// Append an AI-role templated message.
    pub fn ai(mut self, template: impl Into<String>) -> Self {
        self.parts.push(Part::Templated {
            role: Role::Ai,
            template: template.into(),
        });
        self
    }

    /// Append a human-role multimodal message: a text template plus a
    /// pre-built list of [`ContentPart`]s (images, audio). The text
    /// template still receives template-variable substitution against
    /// the call's input.
    pub fn human_with_parts(
        mut self,
        template: impl Into<String>,
        parts: Vec<ContentPart>,
    ) -> Self {
        self.parts.push(Part::Multimodal {
            role: Role::Human,
            template: template.into(),
            parts,
        });
        self
    }

    /// Append an AI-role multimodal message.
    pub fn ai_with_parts(mut self, template: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        self.parts.push(Part::Multimodal {
            role: Role::Ai,
            template: template.into(),
            parts,
        });
        self
    }

    /// Convenience: append a human-role message with one image URL part.
    pub fn human_with_image_url(
        self,
        template: impl Into<String>,
        url: impl Into<String>,
        mime: impl Into<String>,
    ) -> Self {
        self.human_with_parts(
            template,
            vec![ContentPart::Image {
                source: crate::content::ImageSource::url(url),
                mime: mime.into(),
            }],
        )
    }

    /// Append a `Vec<Message>` field from the input.
    ///
    /// Errors at render time if the field is missing.
    pub fn placeholder(mut self, key: impl Into<String>) -> Self {
        self.parts.push(Part::Placeholder {
            key: key.into(),
            optional: false,
        });
        self
    }

    /// Like [`placeholder`](Self::placeholder), but a missing key resolves
    /// to an empty list instead of an error.
    pub fn optional_placeholder(mut self, key: impl Into<String>) -> Self {
        self.parts.push(Part::Placeholder {
            key: key.into(),
            optional: true,
        });
        self
    }

    /// Build from a list of `(role, template)` tuples.
    pub fn from_messages(messages: Vec<(Role, String)>) -> Self {
        let parts = messages
            .into_iter()
            .map(|(role, template)| Part::Templated { role, template })
            .collect();
        Self {
            parts,
            _input: PhantomData,
        }
    }

    /// All `{name}` placeholders across templated parts (excludes
    /// placeholder keys, which are field names not template variables).
    pub fn input_variables(&self) -> Vec<String> {
        let mut out = Vec::new();
        for p in &self.parts {
            let template = match p {
                Part::Templated { template, .. } | Part::Multimodal { template, .. } => template,
                Part::Placeholder { .. } => continue,
            };
            for v in scan_variables(template) {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
        out
    }

    /// Render to `Vec<Message>`.
    pub fn render(&self, input: &I) -> Result<Vec<Message>> {
        let ctx =
            serde_json::to_value(input).map_err(|e| CognisError::Serialization(e.to_string()))?;
        let mut out = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            match part {
                Part::Templated { role, template } => {
                    let text = render(template, &ctx)?;
                    out.push(make_message(*role, text));
                }
                Part::Multimodal {
                    role,
                    template,
                    parts,
                } => {
                    let text = render(template, &ctx)?;
                    out.push(make_multimodal_message(*role, text, parts.clone()));
                }
                Part::Placeholder { key, optional } => {
                    out.extend(pull_messages(&ctx, key, *optional)?);
                }
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl<I> Runnable<I, Vec<Message>> for ChatPromptTemplate<I>
where
    I: Serialize + Send + Sync + 'static,
{
    async fn invoke(&self, input: I, _: RunnableConfig) -> Result<Vec<Message>> {
        self.render(&input)
    }
    fn name(&self) -> &str {
        "ChatPromptTemplate"
    }
}

fn make_message(role: Role, text: String) -> Message {
    match role {
        Role::System => Message::system(text),
        Role::Human => Message::human(text),
        Role::Ai => Message::ai(text),
    }
}

fn make_multimodal_message(role: Role, text: String, parts: Vec<ContentPart>) -> Message {
    match role {
        // System messages are text-only; if the caller asks for a
        // multimodal system message we drop the parts and warn via tracing.
        Role::System => {
            if !parts.is_empty() {
                tracing::warn!(
                    "ChatPromptTemplate: system role doesn't support multimodal parts; dropping"
                );
            }
            Message::system(text)
        }
        Role::Human => Message::human_with_parts(text, parts),
        Role::Ai => Message::ai_with_parts(text, parts),
    }
}

fn pull_messages(ctx: &Value, key: &str, optional: bool) -> Result<Vec<Message>> {
    let v = match ctx.get(key) {
        Some(v) => v,
        None => {
            return if optional {
                Ok(Vec::new())
            } else {
                Err(CognisError::Configuration(format!(
                    "missing required placeholder field `{key}`"
                )))
            };
        }
    };
    serde_json::from_value::<Vec<Message>>(v.clone()).map_err(|e| {
        CognisError::Serialization(format!(
            "placeholder `{key}` did not deserialize as Vec<Message>: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn renders_simple_chat() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new()
            .system("you are {role}")
            .human("hi {name}");
        let out = p
            .invoke(
                json!({"role": "helpful", "name": "ada"}),
                RunnableConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Message::System(_)));
        assert_eq!(out[0].content(), "you are helpful");
        assert!(matches!(out[1], Message::Human(_)));
        assert_eq!(out[1].content(), "hi ada");
    }

    #[test]
    fn placeholder_drops_in_messages() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new()
            .system("sys")
            .placeholder("history")
            .human("now");
        let history = json!([
            {"role": "human", "content": "before-1"},
            {"role": "ai",    "content": "before-2"}
        ]);
        let out = p.render(&json!({"history": history})).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[1].content(), "before-1");
        assert_eq!(out[2].content(), "before-2");
        assert_eq!(out[3].content(), "now");
    }

    #[test]
    fn missing_required_placeholder_errors() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new().placeholder("history");
        let err = p.render(&json!({})).unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[test]
    fn optional_placeholder_accepts_missing() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new()
            .system("hi")
            .optional_placeholder("history");
        let out = p.render(&json!({})).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn input_variables_collects_unique() {
        let p: ChatPromptTemplate<Value> =
            ChatPromptTemplate::new().system("{a} {b}").human("{a} {c}");
        assert_eq!(p.input_variables(), vec!["a", "b", "c"]);
    }

    #[test]
    fn from_messages_constructs_fluently() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::from_messages(vec![
            (Role::System, "sys".into()),
            (Role::Human, "hi {name}".into()),
        ]);
        let out = p.render(&json!({"name": "ada"})).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content(), "hi ada");
    }

    #[test]
    fn human_with_image_url_renders_with_part() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new()
            .system("describe images")
            .human_with_image_url("describe {topic}", "https://x/cat.jpg", "image/jpeg");
        let out = p.render(&json!({"topic": "this cat"})).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content(), "describe this cat");
        let parts = out[1].parts();
        assert_eq!(parts.len(), 1);
        assert!(matches!(
            parts[0],
            crate::content::ContentPart::Image { .. }
        ));
    }

    #[test]
    fn input_variables_includes_multimodal_template_vars() {
        let p: ChatPromptTemplate<Value> = ChatPromptTemplate::new()
            .human("text {a}")
            .human_with_image_url("multimodal {b}", "https://x", "image/png");
        let mut vars = p.input_variables();
        vars.sort();
        assert_eq!(vars, vec!["a", "b"]);
    }
}
