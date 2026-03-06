//! Patch tool calls middleware — intercepts and corrects malformed tool calls from the model.

use async_trait::async_trait;
use serde_json::Value;

use crate::middleware::{AgentState, Middleware, Result};

/// Middleware that intercepts and corrects malformed tool calls from the model.
///
/// It can fix misspelled tool names (using Levenshtein distance) and repair
/// common JSON issues in tool call arguments.
pub struct PatchToolCallsMiddleware {
    /// List of valid tool names.
    known_tools: Vec<String>,
    /// Whether to attempt JSON repair on malformed args.
    fix_json: bool,
}

impl PatchToolCallsMiddleware {
    /// Create a new `PatchToolCallsMiddleware` with the given list of known tool names.
    ///
    /// JSON fixing is enabled by default.
    pub fn new(known_tools: Vec<String>) -> Self {
        Self {
            known_tools,
            fix_json: true,
        }
    }

    /// Set whether to attempt JSON repair on malformed arguments.
    pub fn set_fix_json(&mut self, fix: bool) -> &mut Self {
        self.fix_json = fix;
        self
    }
}

/// Compute the Levenshtein distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost)
                .min(prev[j + 1] + 1)
                .min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

/// Find the closest matching tool name from the known tools list.
/// Returns `None` if `known` is empty or the best distance exceeds half the name length.
fn find_closest_tool(name: &str, known: &[String]) -> Option<String> {
    if known.is_empty() {
        return None;
    }

    let mut best: Option<(&String, usize)> = None;
    for tool in known {
        let dist = levenshtein(name, tool);
        if best.is_none() || dist < best.unwrap().1 {
            best = Some((tool, dist));
        }
    }

    let (best_tool, best_dist) = best?;
    // Only accept if the distance is reasonable (at most half the name length).
    let threshold = (name.len() / 2).max(2);
    if best_dist <= threshold {
        Some(best_tool.clone())
    } else {
        None
    }
}

/// Attempt to repair common JSON issues in a string.
///
/// Handles:
/// - Trailing commas before `}` or `]`
/// - Single quotes used instead of double quotes (basic replacement)
fn repair_json(input: &str) -> String {
    let mut result = input.to_string();

    // Replace single quotes with double quotes.
    // This is a simple heuristic — it won't handle escaped quotes inside strings,
    // but covers the common model mistake of using single-quoted JSON.
    result = result.replace('\'', "\"");

    // Remove trailing commas before } or ].
    // We do multiple passes to handle nested cases.
    loop {
        let before = result.clone();
        // Remove trailing commas: ", }" or ", ]" with optional whitespace.
        result = remove_trailing_commas(&result);
        if result == before {
            break;
        }
    }

    result
}

/// Remove trailing commas before closing braces/brackets.
fn remove_trailing_commas(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == ',' {
            // Look ahead past whitespace for } or ]
            let mut j = i + 1;
            while j < len && chars[j].is_whitespace() {
                j += 1;
            }
            if j < len && (chars[j] == '}' || chars[j] == ']') {
                // Skip the comma, keep whitespace and closing char.
                i += 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

#[async_trait]
impl Middleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str {
        "patch_tool_calls"
    }

    /// After the model responds, inspect the last AI message for tool calls
    /// and attempt to fix tool names and malformed JSON arguments.
    async fn after_model(&self, state: &mut AgentState) -> Result<()> {
        if self.known_tools.is_empty() {
            return Ok(());
        }

        let messages = match state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            Some(m) => m,
            None => return Ok(()),
        };

        // Find the last AI message.
        let last_ai = messages
            .iter_mut()
            .rev()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("ai"));

        let ai_msg = match last_ai {
            Some(m) => m,
            None => return Ok(()),
        };

        // Get tool_calls array from the AI message.
        let tool_calls = match ai_msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
            Some(tc) => tc,
            None => return Ok(()),
        };

        for tool_call in tool_calls.iter_mut() {
            // Fix tool name if it doesn't match a known tool.
            if let Some(name_val) = tool_call.get_mut("name") {
                if let Some(name) = name_val.as_str().map(|s| s.to_string()) {
                    if !self.known_tools.contains(&name) {
                        if let Some(closest) = find_closest_tool(&name, &self.known_tools) {
                            *name_val = Value::String(closest);
                        }
                    }
                }
            }

            // Fix JSON args if enabled.
            if self.fix_json {
                if let Some(args_val) = tool_call.get_mut("args") {
                    if let Some(args_str) = args_val.as_str() {
                        // Args stored as a JSON string — try to parse and repair.
                        let repaired = repair_json(args_str);
                        if let Ok(parsed) = serde_json::from_str::<Value>(&repaired) {
                            *args_val = parsed;
                        } else {
                            // Even if we can't parse, store the repaired string.
                            *args_val = Value::String(repaired);
                        }
                    } else if args_val.is_object() {
                        // Args already a JSON object — nothing to fix.
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("calculater", "calculator"), 1);
    }

    #[tokio::test]
    async fn test_fix_misspelled_tool_name() {
        let mw = PatchToolCallsMiddleware::new(vec![
            "calculator".to_string(),
            "search".to_string(),
            "read_file".to_string(),
        ]);

        let mut state = json!({
            "messages": [
                { "type": "human", "content": "help" },
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        { "name": "calculater", "args": {"expr": "2+2"} }
                    ]
                }
            ]
        });

        mw.after_model(&mut state).await.unwrap();

        let tool_calls = state["messages"][1]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls[0]["name"], "calculator");
    }

    #[tokio::test]
    async fn test_fix_json_trailing_comma() {
        let mw = PatchToolCallsMiddleware::new(vec!["calculator".to_string()]);

        let mut state = json!({
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "calculator",
                            "args": "{\"expr\": \"2+2\", }"
                        }
                    ]
                }
            ]
        });

        mw.after_model(&mut state).await.unwrap();

        let args = &state["messages"][0]["tool_calls"][0]["args"];
        // Should be parsed into a proper JSON object after repair.
        assert_eq!(args["expr"], "2+2");
    }

    #[tokio::test]
    async fn test_valid_tool_calls_not_modified() {
        let mw = PatchToolCallsMiddleware::new(vec![
            "calculator".to_string(),
            "search".to_string(),
        ]);

        let mut state = json!({
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        { "name": "calculator", "args": {"expr": "2+2"} }
                    ]
                }
            ]
        });

        let original_state = state.clone();
        mw.after_model(&mut state).await.unwrap();

        assert_eq!(state, original_state);
    }

    #[tokio::test]
    async fn test_no_known_tools_no_patching() {
        let mw = PatchToolCallsMiddleware::new(vec![]);

        let mut state = json!({
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        { "name": "nonexistent", "args": "{bad json,}" }
                    ]
                }
            ]
        });

        let original_state = state.clone();
        mw.after_model(&mut state).await.unwrap();

        // Nothing should change since known_tools is empty.
        assert_eq!(state, original_state);
    }

    #[test]
    fn test_repair_json_single_quotes() {
        let input = "{'key': 'value'}";
        let repaired = repair_json(input);
        let parsed: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_find_closest_tool_empty() {
        assert_eq!(find_closest_tool("anything", &[]), None);
    }

    #[test]
    fn test_find_closest_tool_exact_match() {
        let tools = vec!["calculator".to_string()];
        assert_eq!(
            find_closest_tool("calculator", &tools),
            Some("calculator".to_string())
        );
    }
}
