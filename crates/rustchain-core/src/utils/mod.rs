//! Utility functions and helpers.
//!
//! Mirrors Python `langchain_core.utils`.

pub mod env;
pub mod formatting;
pub mod hashing;
pub mod function_calling;
pub mod html;
pub mod image;
pub mod input;
pub mod iter;
pub mod json;
pub mod json_schema;
pub mod merge;
pub mod mustache;
pub mod strings;
pub mod token_counter;
pub mod retry_policy;
pub mod tokens;
pub mod usage;
pub mod uuid;

use ::uuid::Uuid;
use serde_json::Value;

pub use merge::{merge_dicts, merge_lists, merge_obj, MergeError};

/// Generate a new UUID v4 string.
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Generate an ID with the LangChain prefix format.
pub fn ensure_id(id: Option<String>) -> String {
    match id {
        Some(id) if !id.is_empty() => id,
        _ => format!("lc_{}", Uuid::new_v4()),
    }
}

/// Prefix used for auto-generated LangChain IDs.
pub const LC_AUTO_PREFIX: &str = "lc_";

/// Prefix used for tracing run IDs.
pub const LC_ID_PREFIX: &str = "lc_run-";

/// Get a value from a map or from an environment variable.
pub fn get_from_dict_or_env(
    data: &std::collections::HashMap<String, Value>,
    key: &str,
    env_key: &str,
    default: Option<&str>,
) -> Option<String> {
    // Check in data first
    if let Some(val) = data.get(key) {
        return match val {
            Value::String(s) => Some(s.clone()),
            _ => Some(val.to_string()),
        };
    }
    // Then check environment variable
    if let Ok(val) = std::env::var(env_key) {
        return Some(val);
    }
    // Finally use default
    default.map(|s| s.to_string())
}

/// Get a value from an environment variable.
pub fn get_from_env(key: &str, default: Option<&str>) -> Option<String> {
    std::env::var(key)
        .ok()
        .or_else(|| default.map(|s| s.to_string()))
}

/// Rust/Python type name to JSON Schema type mapping.
pub fn rust_to_json_type(rust_type: &str) -> &str {
    match rust_type {
        "str" | "String" | "&str" => "string",
        "int" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
        | "u128" | "usize" => "integer",
        "float" | "f32" | "f64" => "number",
        "bool" => "boolean",
        "Vec" | "list" => "array",
        _ => "object",
    }
}

/// Python-to-JSON type mapping used for function calling schemas.
pub fn python_to_json_type(py_type: &str) -> &str {
    rust_to_json_type(py_type)
}

/// Create an HTTP error from a status code and body text.
///
/// Helper for building `RustChainError::HttpError` when handling HTTP responses.
/// Equivalent to Python's `raise_for_status_with_text`.
pub fn http_error(status: u16, body: impl Into<String>) -> crate::error::RustChainError {
    crate::error::RustChainError::HttpError {
        status,
        body: body.into(),
    }
}
