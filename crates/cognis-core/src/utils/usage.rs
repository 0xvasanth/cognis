//! Token usage tracking utilities.

use serde_json::Value;
use std::collections::HashMap;

/// Recursively combine two dictionaries using integer addition.
pub fn dict_int_add(
    a: &HashMap<String, Value>,
    b: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut result = a.clone();
    for (key, b_val) in b {
        let merged = match (result.get(key), b_val) {
            (Some(Value::Number(n1)), Value::Number(n2)) => {
                let sum = n1.as_i64().unwrap_or(0) + n2.as_i64().unwrap_or(0);
                Value::Number(serde_json::Number::from(sum))
            }
            (Some(Value::Object(m1)), Value::Object(m2)) => {
                let h1: HashMap<String, Value> =
                    m1.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let h2: HashMap<String, Value> =
                    m2.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let merged_inner = dict_int_add(&h1, &h2);
                Value::Object(merged_inner.into_iter().collect())
            }
            (_, new_val) => new_val.clone(),
        };
        result.insert(key.clone(), merged);
    }
    result
}
