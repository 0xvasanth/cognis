//! String manipulation utilities.

/// Serialize a value to a human-readable string.
pub fn stringify_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Serialize a dict to a human-readable string.
pub fn stringify_dict(dict: &std::collections::HashMap<String, serde_json::Value>) -> String {
    let parts: Vec<String> = dict
        .iter()
        .map(|(k, v)| format!("{}: {}", k, stringify_value(v)))
        .collect();
    parts.join(", ")
}

/// Format a list as a comma-separated string with "and" before the last item.
pub fn comma_list(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let last = &items[items.len() - 1];
            let rest = &items[..items.len() - 1];
            format!("{}, and {}", rest.join(", "), last)
        }
    }
}

/// Remove NUL bytes for PostgreSQL compatibility.
pub fn sanitize_for_postgres(s: &str) -> String {
    s.replace('\0', "")
}
