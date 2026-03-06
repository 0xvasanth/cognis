//! JSON Schema utilities for dereferencing $ref pointers.

use std::collections::HashMap;
use serde_json::Value;

/// Dereference all $ref pointers in a JSON schema.
/// Returns a new schema with all references resolved inline.
pub fn dereference_refs(schema: &Value) -> Value {
    let mut definitions = HashMap::new();
    // Collect all definitions
    if let Some(Value::Object(map)) = schema.get("$defs").or_else(|| schema.get("definitions")) {
        for (k, v) in map {
            definitions.insert(format!("#/$defs/{}", k), v.clone());
            definitions.insert(format!("#/definitions/{}", k), v.clone());
        }
    }
    let mut result = dereference_helper(schema, &definitions, &mut Vec::new());
    // Remove definitions from result
    if let Value::Object(ref mut map) = result {
        map.remove("$defs");
        map.remove("definitions");
    }
    result
}

fn dereference_helper(value: &Value, definitions: &HashMap<String, Value>, stack: &mut Vec<String>) -> Value {
    match value {
        Value::Object(map) => {
            if let Some(ref_val) = map.get("$ref").and_then(|r| r.as_str()) {
                if stack.contains(&ref_val.to_string()) {
                    // Circular reference - return empty object
                    return Value::Object(Default::default());
                }
                if let Some(resolved) = definitions.get(ref_val) {
                    stack.push(ref_val.to_string());
                    let result = dereference_helper(resolved, definitions, stack);
                    stack.pop();
                    // Merge any additional properties from the $ref object
                    if map.len() > 1 {
                        let mut merged = result;
                        if let Value::Object(ref mut m) = merged {
                            for (k, v) in map {
                                if k != "$ref" {
                                    m.insert(k.clone(), dereference_helper(v, definitions, stack));
                                }
                            }
                        }
                        return merged;
                    }
                    return result;
                }
            }
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                new_map.insert(k.clone(), dereference_helper(v, definitions, stack));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(|v| dereference_helper(v, definitions, stack)).collect())
        }
        other => other.clone(),
    }
}
