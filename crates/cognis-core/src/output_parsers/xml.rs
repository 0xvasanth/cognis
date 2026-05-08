//! Lightweight XML-ish parser — flat `<tag>value</tag>` extraction.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Extracts non-nested `<tag>value</tag>` pairs into a flat map.
///
/// Nested tags are not supported — if you need structured output, use
/// [`crate::output_parsers::JsonParser`] instead.
#[derive(Debug, Default, Clone, Copy)]
pub struct XmlParser;

impl XmlParser {
    /// Construct an `XmlParser`.
    pub fn new() -> Self {
        Self
    }
}

impl OutputParser<HashMap<String, String>> for XmlParser {
    fn parse(&self, text: &str) -> Result<HashMap<String, String>> {
        let mut out = HashMap::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'<' {
                i += 1;
                continue;
            }
            let open_start = i + 1;
            let open_end = match bytes[open_start..].iter().position(|&b| b == b'>') {
                Some(p) => open_start + p,
                None => break,
            };
            let tag = std::str::from_utf8(&bytes[open_start..open_end])
                .map_err(|e| CognisError::Serialization(e.to_string()))?;
            if tag.starts_with('/') || tag.is_empty() {
                i = open_end + 1;
                continue;
            }
            let close_marker = format!("</{tag}>");
            let value_start = open_end + 1;
            let close_pos = match text[value_start..].find(&close_marker) {
                Some(p) => value_start + p,
                None => {
                    i = open_end + 1;
                    continue;
                }
            };
            let value = text[value_start..close_pos].trim().to_string();
            out.insert(tag.to_string(), value);
            i = close_pos + close_marker.len();
        }
        Ok(out)
    }

    fn format_instructions(&self) -> Option<String> {
        Some("Wrap each field in XML-style tags: `<field>value</field>`. Do not nest tags.".into())
    }
}

#[async_trait]
impl Runnable<String, HashMap<String, String>> for XmlParser {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<HashMap<String, String>> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "XmlParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_tags() {
        let p = XmlParser::new();
        let out = p.parse("<name>Ada</name><role>scientist</role>").unwrap();
        assert_eq!(out.get("name"), Some(&"Ada".to_string()));
        assert_eq!(out.get("role"), Some(&"scientist".to_string()));
    }

    #[test]
    fn whitespace_around_value_trimmed() {
        let p = XmlParser::new();
        let out = p.parse("<x>\n  hello  \n</x>").unwrap();
        assert_eq!(out.get("x"), Some(&"hello".to_string()));
    }

    #[test]
    fn unclosed_tag_skipped() {
        let p = XmlParser::new();
        let out = p.parse("<a>1</a><b>open").unwrap();
        assert_eq!(out.get("a"), Some(&"1".to_string()));
        assert!(!out.contains_key("b"));
    }
}
