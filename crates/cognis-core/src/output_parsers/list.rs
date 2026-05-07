//! List parsers — comma-separated and numbered/bulleted.

use async_trait::async_trait;

use crate::output_parsers::OutputParser;
use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Parses a separator-delimited string into `Vec<String>`.
///
/// Default separator is `,`. Whitespace around items is trimmed; empty
/// items are dropped.
#[derive(Debug, Clone)]
pub struct CommaListParser {
    separator: String,
}

impl Default for CommaListParser {
    fn default() -> Self {
        Self {
            separator: ",".into(),
        }
    }
}

impl CommaListParser {
    /// Construct a `CommaListParser` with the default `,` separator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the separator.
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
}

impl OutputParser<Vec<String>> for CommaListParser {
    fn parse(&self, text: &str) -> Result<Vec<String>> {
        Ok(text
            .split(&self.separator)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }

    fn format_instructions(&self) -> Option<String> {
        Some(format!(
            "Reply with a `{}`-separated list. No commentary, no numbering.",
            self.separator
        ))
    }
}

#[async_trait]
impl Runnable<String, Vec<String>> for CommaListParser {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<Vec<String>> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "CommaListParser"
    }
}

/// Parses numbered/bulleted lists into `Vec<String>`.
///
/// Recognised line markers (after optional leading whitespace):
/// - `1. `, `12. `, `1) ` (numbered)
/// - `- `, `* ` (bulleted)
///
/// Lines without a marker are still kept after trimming, so plain
/// newline-delimited lists also parse.
#[derive(Debug, Default, Clone, Copy)]
pub struct NumberedListParser;

impl NumberedListParser {
    /// Construct a `NumberedListParser`.
    pub fn new() -> Self {
        Self
    }
}

impl OutputParser<Vec<String>> for NumberedListParser {
    fn parse(&self, text: &str) -> Result<Vec<String>> {
        Ok(text
            .lines()
            .map(|l| strip_list_marker(l.trim_start()).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }

    fn format_instructions(&self) -> Option<String> {
        Some("Reply as a numbered list (`1.`, `2.`, ...). One item per line.".into())
    }
}

#[async_trait]
impl Runnable<String, Vec<String>> for NumberedListParser {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<Vec<String>> {
        OutputParser::parse(self, &input)
    }
    fn name(&self) -> &str {
        "NumberedListParser"
    }
}

fn strip_list_marker(s: &str) -> &str {
    if let Some(rest) = strip_numbered(s) {
        return rest;
    }
    if let Some(rest) = s.strip_prefix("- ") {
        return rest;
    }
    if let Some(rest) = s.strip_prefix("* ") {
        return rest;
    }
    s
}

fn strip_numbered(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() {
        return None;
    }
    if bytes[i] != b'.' && bytes[i] != b')' {
        return None;
    }
    i += 1;
    if i >= bytes.len() || bytes[i] != b' ' {
        return None;
    }
    i += 1;
    Some(&s[i..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_list_basic() {
        let p = CommaListParser::new();
        assert_eq!(p.parse("a, b, c").unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn comma_list_drops_empty_items() {
        let p = CommaListParser::new();
        assert_eq!(p.parse("a, , c, ").unwrap(), vec!["a", "c"]);
    }

    #[test]
    fn comma_list_custom_separator() {
        let p = CommaListParser::new().with_separator(";");
        assert_eq!(p.parse("a;b;c").unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn numbered_list_with_dots() {
        let p = NumberedListParser::new();
        assert_eq!(
            p.parse("1. first\n2. second\n3. third").unwrap(),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn numbered_list_with_parens() {
        let p = NumberedListParser::new();
        assert_eq!(p.parse("1) one\n2) two").unwrap(), vec!["one", "two"]);
    }

    #[test]
    fn numbered_list_with_dashes() {
        let p = NumberedListParser::new();
        assert_eq!(p.parse("- a\n- b").unwrap(), vec!["a", "b"]);
    }
}
