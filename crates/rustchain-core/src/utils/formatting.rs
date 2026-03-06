//! String formatting utilities.

use std::collections::HashMap;
use serde_json::Value;
use crate::error::Result;

/// A strict formatter that only accepts keyword arguments (no positional).
/// Mirrors Python `langchain_core.utils.formatting.StrictFormatter`.
pub fn strict_format(template: &str, kwargs: &HashMap<String, Value>) -> Result<String> {
    crate::prompts::string_formatter::format_template(
        template,
        crate::prompts::string_formatter::TemplateFormat::FString,
        kwargs,
    )
}
