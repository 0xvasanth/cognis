//! Schema generation tuned for OpenAI / Anthropic / Ollama tool calling.
//!
//! These providers reject `$ref`, `$defs`, and `$schema` in tool parameter
//! schemas. We configure schemars to inline subschemas and omit the
//! meta-schema field. No vendoring required.

use cognis2_core::schemars::{gen::SchemaSettings, JsonSchema};

/// Generate an LLM-API-compatible JSON Schema for a type.
///
/// Settings:
/// - `inline_subschemas = true` — no `$ref`, no `$defs`
/// - `meta_schema = None` — no `$schema` field
/// - `definitions_path = "/definitions"` — draft-07-style fallback
///
/// These match what OpenAI's strict tool-calling validator accepts.
pub fn schema_for_tool<T: JsonSchema>() -> serde_json::Value {
    let settings = SchemaSettings::draft07().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let gen = settings.into_generator();
    let schema = gen.into_root_schema_for::<T>();
    serde_json::to_value(schema).expect("schemars output is always serializable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis2_core::schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(JsonSchema, Deserialize)]
    struct Nested {
        inner: String,
    }

    #[derive(JsonSchema, Deserialize)]
    struct Outer {
        a: f64,
        nested: Nested,
    }

    #[test]
    fn no_dollar_ref_or_defs_or_schema() {
        let s = schema_for_tool::<Outer>();
        let text = s.to_string();
        assert!(!text.contains("$ref"), "found $ref: {text}");
        assert!(!text.contains("$defs"), "found $defs: {text}");
        assert!(!text.contains("$schema"), "found $schema: {text}");
    }

    #[test]
    fn nested_structs_are_inlined() {
        let s = schema_for_tool::<Outer>();
        // The Nested struct should appear inlined inside `properties.nested`
        assert!(s["properties"]["nested"]["type"].is_string());
        assert!(s["properties"]["nested"]["properties"]["inner"].is_object());
    }
}
