//! JSON parsing utilities.

use crate::error::{Result, CognisError};
use serde_json::Value;

/// Parse potentially incomplete JSON string (useful for streaming).
/// Attempts to fix common issues: missing closing brackets, trailing commas.
pub fn parse_partial_json(s: &str) -> Result<Value> {
    // Try parsing as-is first
    if let Ok(v) = serde_json::from_str(s) {
        return Ok(v);
    }

    let trimmed = s.trim();
    // Try adding closing brackets
    let mut attempt = trimmed.to_string();
    // Count open braces/brackets
    let open_braces = attempt.chars().filter(|c| *c == '{').count();
    let close_braces = attempt.chars().filter(|c| *c == '}').count();
    let open_brackets = attempt.chars().filter(|c| *c == '[').count();
    let close_brackets = attempt.chars().filter(|c| *c == ']').count();

    // Remove trailing comma
    if attempt.ends_with(',') {
        attempt.pop();
    }

    // Add missing closing brackets/braces
    for _ in 0..(open_brackets.saturating_sub(close_brackets)) {
        attempt.push(']');
    }
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        attempt.push('}');
    }

    serde_json::from_str(&attempt)
        .map_err(|e| CognisError::Other(format!("Failed to parse partial JSON: {}", e)))
}

/// Parse JSON from a markdown code block.
pub fn parse_json_markdown(text: &str) -> Result<Value> {
    let trimmed = text.trim();
    let json_str = if trimmed.starts_with("```") {
        // Strip code fence
        let after = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```JSON"))
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        after.trim().strip_suffix("```").unwrap_or(after).trim()
    } else {
        trimmed
    };
    serde_json::from_str(json_str)
        .map_err(|e| CognisError::Other(format!("Failed to parse JSON from markdown: {}", e)))
}

/// Parse JSON from markdown and validate expected keys exist.
pub fn parse_and_check_json_markdown(text: &str, expected_keys: &[&str]) -> Result<Value> {
    let parsed = parse_json_markdown(text)?;
    if let Value::Object(ref map) = parsed {
        for key in expected_keys {
            if !map.contains_key(*key) {
                return Err(CognisError::Other(format!(
                    "Missing expected key '{}' in JSON output",
                    key
                )));
            }
        }
    }
    Ok(parsed)
}
