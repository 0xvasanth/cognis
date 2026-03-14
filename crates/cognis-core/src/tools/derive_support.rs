//! Support types and trait implementations for the `#[derive(Tool)]` and
//! `#[derive(ToolSchema)]` proc macros.

use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// Trait for types that can produce a JSON Schema representation of themselves.
///
/// This trait is automatically implemented by `#[derive(Tool)]` and
/// `#[derive(ToolSchema)]`. It is also implemented for common Rust primitive
/// types so that the generated code can recursively resolve nested schemas.
///
/// # Example
///
/// ```rust
/// use cognis_core::tools::ToolJsonSchema;
///
/// assert_eq!(
///     String::json_schema(),
///     serde_json::json!({"type": "string"}),
/// );
/// ```
pub trait ToolJsonSchema {
    /// Returns the JSON Schema representation of this type.
    fn json_schema() -> Value;
}

// ---------------------------------------------------------------------------
// Primitive implementations
// ---------------------------------------------------------------------------

impl ToolJsonSchema for String {
    fn json_schema() -> Value {
        json!({"type": "string"})
    }
}

impl ToolJsonSchema for &str {
    fn json_schema() -> Value {
        json!({"type": "string"})
    }
}

impl ToolJsonSchema for f32 {
    fn json_schema() -> Value {
        json!({"type": "number"})
    }
}

impl ToolJsonSchema for f64 {
    fn json_schema() -> Value {
        json!({"type": "number"})
    }
}

macro_rules! impl_integer_schema {
    ($($t:ty),*) => {
        $(
            impl ToolJsonSchema for $t {
                fn json_schema() -> Value {
                    json!({"type": "integer"})
                }
            }
        )*
    };
}

impl_integer_schema!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl ToolJsonSchema for bool {
    fn json_schema() -> Value {
        json!({"type": "boolean"})
    }
}

impl<T: ToolJsonSchema> ToolJsonSchema for Vec<T> {
    fn json_schema() -> Value {
        json!({
            "type": "array",
            "items": T::json_schema()
        })
    }
}

impl<T: ToolJsonSchema> ToolJsonSchema for Option<T> {
    fn json_schema() -> Value {
        T::json_schema()
    }
}

impl<V: ToolJsonSchema> ToolJsonSchema for HashMap<String, V> {
    fn json_schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": V::json_schema()
        })
    }
}

impl<V: ToolJsonSchema> ToolJsonSchema for BTreeMap<String, V> {
    fn json_schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": V::json_schema()
        })
    }
}

impl ToolJsonSchema for Value {
    fn json_schema() -> Value {
        json!({})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_schema() {
        assert_eq!(String::json_schema(), json!({"type": "string"}));
    }

    #[test]
    fn test_f64_schema() {
        assert_eq!(f64::json_schema(), json!({"type": "number"}));
    }

    #[test]
    fn test_i32_schema() {
        assert_eq!(i32::json_schema(), json!({"type": "integer"}));
    }

    #[test]
    fn test_bool_schema() {
        assert_eq!(bool::json_schema(), json!({"type": "boolean"}));
    }

    #[test]
    fn test_vec_schema() {
        assert_eq!(
            Vec::<String>::json_schema(),
            json!({"type": "array", "items": {"type": "string"}})
        );
    }

    #[test]
    fn test_option_schema() {
        assert_eq!(Option::<i32>::json_schema(), json!({"type": "integer"}));
    }

    #[test]
    fn test_hashmap_schema() {
        assert_eq!(
            HashMap::<String, f64>::json_schema(),
            json!({"type": "object", "additionalProperties": {"type": "number"}})
        );
    }

    #[test]
    fn test_value_schema() {
        assert_eq!(Value::json_schema(), json!({}));
    }
}
