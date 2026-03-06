//! Minimal mustache template engine.
//!
//! Supports basic variable interpolation (`{{var}}`), sections (`{{#section}}`),
//! inverted sections (`{{^section}}`), and HTML escaping.
//!
//! Inspired by Python `langchain_core.utils.mustache` (from chevron).

use serde_json::Value;

use crate::error::{Result, RustChainError};

/// Render a mustache template with the given data.
pub fn render(template: &str, data: &Value) -> Result<String> {
    let scopes = vec![data.clone()];
    render_with_scopes(template, &scopes)
}

fn render_with_scopes(template: &str, scopes: &[Value]) -> Result<String> {
    let mut output = String::new();
    let mut remaining = template;

    while !remaining.is_empty() {
        // Find the next tag
        if let Some(tag_start) = remaining.find("{{") {
            // Add literal text before the tag
            output.push_str(&remaining[..tag_start]);
            let after_open = &remaining[tag_start + 2..];

            if let Some(tag_end) = after_open.find("}}") {
                let tag_content = after_open[..tag_end].trim();
                remaining = &after_open[tag_end + 2..];

                if tag_content.is_empty() {
                    continue;
                }

                let first_char = tag_content.chars().next().unwrap();

                match first_char {
                    // Comment: {{! ... }}
                    '!' => {
                        // Skip comment
                    }
                    // Unescaped: {{{ var }}} or {{& var }}
                    '&' => {
                        let key = tag_content[1..].trim();
                        let val = get_key(key, scopes);
                        output.push_str(&value_to_string(&val));
                    }
                    '{' => {
                        // Triple mustache: {{{ var }}}
                        let key = tag_content[1..].trim();
                        // Need to consume the extra closing brace
                        if remaining.starts_with('}') {
                            remaining = &remaining[1..];
                        }
                        let val = get_key(key, scopes);
                        output.push_str(&value_to_string(&val));
                    }
                    // Section: {{# section }}...{{/ section }}
                    '#' => {
                        let section_name = tag_content[1..].trim();
                        let (section_body, rest) = find_section_end(remaining, section_name)?;
                        remaining = rest;

                        let val = get_key(section_name, scopes);
                        match &val {
                            Value::Bool(false) | Value::Null => {
                                // Falsy: skip section
                            }
                            Value::Array(arr) => {
                                for item in arr {
                                    let mut new_scopes = scopes.to_vec();
                                    new_scopes.push(item.clone());
                                    output
                                        .push_str(&render_with_scopes(section_body, &new_scopes)?);
                                }
                            }
                            Value::Object(_) => {
                                let mut new_scopes = scopes.to_vec();
                                new_scopes.push(val.clone());
                                output.push_str(&render_with_scopes(section_body, &new_scopes)?);
                            }
                            _ => {
                                // Truthy: render once
                                output.push_str(&render_with_scopes(section_body, scopes)?);
                            }
                        }
                    }
                    // Inverted section: {{^ section }}...{{/ section }}
                    '^' => {
                        let section_name = tag_content[1..].trim();
                        let (section_body, rest) = find_section_end(remaining, section_name)?;
                        remaining = rest;

                        let val = get_key(section_name, scopes);
                        let is_falsy = match &val {
                            Value::Bool(false) | Value::Null => true,
                            Value::Array(arr) => arr.is_empty(),
                            _ => false,
                        };

                        if is_falsy {
                            output.push_str(&render_with_scopes(section_body, scopes)?);
                        }
                    }
                    // Partial: {{> partial }} — not supported, skip
                    '>' => {}
                    // Variable
                    _ => {
                        let val = get_key(tag_content, scopes);
                        output.push_str(&html_escape(&value_to_string(&val)));
                    }
                }
            } else {
                // No closing }}, add rest as literal
                output.push_str(&remaining[tag_start..]);
                break;
            }
        } else {
            // No more tags, add remaining literal text
            output.push_str(remaining);
            break;
        }
    }

    Ok(output)
}

/// Find the end of a section, handling nested sections.
fn find_section_end<'a>(template: &'a str, name: &str) -> Result<(&'a str, &'a str)> {
    let open_tag = format!("{{{{#{}}}}}", name);
    let close_tag = format!("{{{{/{}}}}}", name);

    let mut depth = 1;
    let mut search = template;
    let mut body_end = 0;

    while depth > 0 {
        let next_open = search.find(&open_tag);
        let next_close = search.find(&close_tag);

        match (next_open, next_close) {
            (_, Some(close_pos)) => {
                let open_before_close = next_open.is_some_and(|op| op < close_pos);
                if open_before_close {
                    let op = next_open.unwrap();
                    depth += 1;
                    let advance = op + open_tag.len();
                    body_end += advance;
                    search = &search[advance..];
                } else {
                    depth -= 1;
                    if depth == 0 {
                        let body = &template[..body_end + close_pos];
                        let rest = &search[close_pos + close_tag.len()..];
                        return Ok((body, rest));
                    }
                    let advance = close_pos + close_tag.len();
                    body_end += advance;
                    search = &search[advance..];
                }
            }
            (_, None) => {
                return Err(RustChainError::Other(format!(
                    "Unclosed mustache section: {}",
                    name
                )));
            }
        }
    }

    unreachable!()
}

/// Look up a dot-separated key in the scope stack.
fn get_key(key: &str, scopes: &[Value]) -> Value {
    if key == "." {
        return scopes.last().cloned().unwrap_or(Value::Null);
    }

    let parts: Vec<&str> = key.split('.').collect();

    // Search scopes from innermost to outermost.
    for scope in scopes.iter().rev() {
        let mut current = scope;
        let mut found = true;

        for part in &parts {
            match current {
                Value::Object(map) => {
                    if let Some(val) = map.get(*part) {
                        current = val;
                    } else {
                        found = false;
                        break;
                    }
                }
                _ => {
                    found = false;
                    break;
                }
            }
        }

        if found {
            return current.clone();
        }
    }

    Value::Null
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Extract top-level variable names from a mustache template.
pub fn template_vars(template: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut remaining = template;

    while let Some(pos) = remaining.find("{{") {
        let after = &remaining[pos + 2..];
        if let Some(end) = after.find("}}") {
            let tag = after[..end].trim();
            if !tag.is_empty() {
                let first = tag.chars().next().unwrap();
                let var_name = match first {
                    '#' | '^' | '/' | '!' | '>' | '&' => tag[1..].trim().to_string(),
                    '{' => tag[1..].trim().to_string(),
                    _ => tag.to_string(),
                };
                // Only add top-level (no dots), skip section ends and comments
                if first != '/' && first != '!' && !var_name.is_empty() && !vars.contains(&var_name)
                {
                    vars.push(var_name);
                }
            }
            remaining = &after[end + 2..];
        } else {
            break;
        }
    }

    vars
}
