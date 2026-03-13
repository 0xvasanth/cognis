use async_trait::async_trait;
use serde_json::Value;

use crate::error::{CognisError, Result};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

/// Parses XML output into a nested JSON structure.
///
/// Converts `<tag>content</tag>` patterns into `{"tag": "content"}` objects.
/// Nested tags produce nested objects. Repeated tags at the same level
/// produce arrays.
pub struct XmlOutputParser {
    /// Expected tag names for format instructions.
    pub tags: Option<Vec<String>>,
}

impl XmlOutputParser {
    pub fn new() -> Self {
        Self { tags: None }
    }

    pub fn with_tags(tags: Vec<String>) -> Self {
        Self { tags: Some(tags) }
    }
}

impl Default for XmlOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip markdown code fences from XML content.
fn strip_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.starts_with("```") {
        let after_fence = if let Some(rest) = trimmed.strip_prefix("```xml") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```XML") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```") {
            rest
        } else {
            trimmed
        };
        after_fence
            .trim()
            .strip_suffix("```")
            .unwrap_or(after_fence)
            .trim()
    } else {
        trimmed
    }
}

/// Simple recursive XML-to-dict parser.
///
/// Handles `<tag>text</tag>`, `<tag><nested>...</nested></tag>`, and repeated
/// tags at the same level. Does not handle attributes or self-closing tags.
fn xml_to_dict(text: &str) -> Result<Value> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Value::Null);
    }

    // Check if this starts with a tag
    if !text.starts_with('<') {
        // Plain text content
        return Ok(Value::String(text.to_string()));
    }

    let mut result = serde_json::Map::new();
    let mut pos = 0;
    let bytes = text.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }

        // Expect opening tag
        if bytes[pos] != b'<' {
            break;
        }

        // Find tag name
        let tag_start = pos + 1;
        let Some(tag_end) = text[tag_start..].find('>') else {
            return Err(CognisError::OutputParserError {
                message: "Malformed XML: unclosed opening tag".into(),
                observation: Some(text[pos..].chars().take(50).collect()),
                llm_output: None,
            });
        };
        let tag_end = tag_start + tag_end;
        let tag_name = &text[tag_start..tag_end];

        // Skip processing instructions, comments, or closing tags
        if tag_name.starts_with('?') || tag_name.starts_with('!') || tag_name.starts_with('/') {
            pos = tag_end + 1;
            continue;
        }

        // Find matching closing tag
        let closing_tag = format!("</{}>", tag_name);
        let content_start = tag_end + 1;

        let Some(closing_pos) = find_matching_close(text, content_start, tag_name) else {
            return Err(CognisError::OutputParserError {
                message: format!("Malformed XML: no closing tag for <{}>", tag_name),
                observation: Some(text[pos..].chars().take(80).collect()),
                llm_output: None,
            });
        };

        let content = &text[content_start..closing_pos];
        let child_value = xml_to_dict(content)?;

        // Handle repeated tags → array
        if let Some(existing) = result.get(tag_name) {
            match existing {
                Value::Array(arr) => {
                    let mut new_arr = arr.clone();
                    new_arr.push(child_value);
                    result.insert(tag_name.to_string(), Value::Array(new_arr));
                }
                _ => {
                    let arr = vec![existing.clone(), child_value];
                    result.insert(tag_name.to_string(), Value::Array(arr));
                }
            }
        } else {
            result.insert(tag_name.to_string(), child_value);
        }

        pos = closing_pos + closing_tag.len();
    }

    if result.is_empty() {
        Ok(Value::String(text.to_string()))
    } else {
        Ok(Value::Object(result))
    }
}

/// Find the position of the matching closing tag, handling nested same-name tags.
fn find_matching_close(text: &str, start: usize, tag_name: &str) -> Option<usize> {
    let open = format!("<{}>", tag_name);
    let close = format!("</{}>", tag_name);
    let mut depth = 1;
    let mut pos = start;

    while pos < text.len() && depth > 0 {
        if text[pos..].starts_with(&close) {
            depth -= 1;
            if depth == 0 {
                return Some(pos);
            }
            pos += close.len();
        } else if text[pos..].starts_with(&open) {
            depth += 1;
            pos += open.len();
        } else {
            pos += 1;
        }
    }
    None
}

impl OutputParser for XmlOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let cleaned = strip_fences(text);
        xml_to_dict(cleaned)
    }

    fn get_format_instructions(&self) -> Option<String> {
        let mut instructions =
            "Return your response as XML. Do not include any other text or markdown formatting."
                .to_string();
        if let Some(tags) = &self.tags {
            instructions.push_str(&format!(
                "\n\nExpected tags: {}",
                tags.iter()
                    .map(|t| format!("<{}>", t))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Some(instructions)
    }

    fn parser_type(&self) -> &str {
        "xml_output_parser"
    }
}

#[async_trait]
impl Runnable for XmlOutputParser {
    fn name(&self) -> &str {
        "XmlOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
