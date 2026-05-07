//! JSON query tool — apply a path expression to a JSON value.
//!
//! The shipped engine is [`DotPathEngine`] which supports a subset of
//! the common dot/bracket notation:
//!
//! - `a.b.c` — object key navigation
//! - `a[0]` — array index
//! - `a.b[2].c` — mixed
//! - `*` (only as a leaf) — return all values of an object as an array
//!
//! For full JMESPath / JSONPath / `jq` semantics, plug in your own
//! engine via [`QueryEngine`] (e.g. wrap the `jmespath` crate). The
//! tool deliberately ships with the lighter engine to avoid a hard
//! dep.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

/// Pluggable JSON query engine.
pub trait QueryEngine: Send + Sync {
    /// Apply `path` to `value`, returning the matched value (or
    /// `Value::Null` for no match).
    fn query(&self, value: &serde_json::Value, path: &str) -> Result<serde_json::Value>;

    /// Friendly name for diagnostics.
    fn name(&self) -> &str {
        "QueryEngine"
    }
}

/// Closure-based engine.
impl<F> QueryEngine for F
where
    F: Fn(&serde_json::Value, &str) -> Result<serde_json::Value> + Send + Sync,
{
    fn query(&self, value: &serde_json::Value, path: &str) -> Result<serde_json::Value> {
        (self)(value, path)
    }
}

/// Default engine — supports `a.b.c`, `a[0]`, `a.b[2].c`, and a leaf
/// `*` to return all values of the parent object as an array.
#[derive(Debug, Default, Clone, Copy)]
pub struct DotPathEngine;

impl QueryEngine for DotPathEngine {
    fn query(&self, value: &serde_json::Value, path: &str) -> Result<serde_json::Value> {
        let segs = parse_path(path)?;
        let mut cur = value.clone();
        for seg in segs {
            match seg {
                PathSeg::Key(k) => {
                    cur = match cur {
                        serde_json::Value::Object(mut o) => {
                            o.remove(&k).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                }
                PathSeg::Index(i) => {
                    cur = match cur {
                        serde_json::Value::Array(a) => {
                            a.into_iter().nth(i).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                }
                PathSeg::Wildcard => {
                    cur = match cur {
                        serde_json::Value::Object(o) => {
                            serde_json::Value::Array(o.into_values().collect())
                        }
                        // `*` on an array is identity.
                        v @ serde_json::Value::Array(_) => v,
                        _ => serde_json::Value::Null,
                    };
                }
            }
        }
        Ok(cur)
    }

    fn name(&self) -> &str {
        "DotPathEngine"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PathSeg {
    Key(String),
    Index(usize),
    Wildcard,
}

/// Tokenize `a.b[2].*` into a sequence of [`PathSeg`].
fn parse_path(path: &str) -> Result<Vec<PathSeg>> {
    let mut out: Vec<PathSeg> = Vec::new();
    let mut cur = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                flush_key(&mut cur, &mut out);
            }
            '[' => {
                flush_key(&mut cur, &mut out);
                let mut buf = String::new();
                let mut closed = false;
                for nc in chars.by_ref() {
                    if nc == ']' {
                        closed = true;
                        break;
                    }
                    buf.push(nc);
                }
                if !closed {
                    return Err(CognisError::ToolValidationError(format!(
                        "json_query: unclosed bracket in path `{path}`"
                    )));
                }
                let idx: usize = buf.trim().parse().map_err(|_| {
                    CognisError::ToolValidationError(format!(
                        "json_query: non-numeric index `{buf}` in path `{path}`"
                    ))
                })?;
                out.push(PathSeg::Index(idx));
            }
            '*' if cur.is_empty() => out.push(PathSeg::Wildcard),
            _ => cur.push(c),
        }
    }
    flush_key(&mut cur, &mut out);
    Ok(out)
}

fn flush_key(cur: &mut String, out: &mut Vec<PathSeg>) {
    if !cur.is_empty() {
        let k = std::mem::take(cur);
        if k == "*" {
            out.push(PathSeg::Wildcard);
        } else {
            out.push(PathSeg::Key(k));
        }
    }
}

/// Tool input schema.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JsonQueryInput {
    /// JSON value to query (object, array, etc.).
    pub data: serde_json::Value,
    /// Path expression — see the engine's documentation for syntax.
    pub path: String,
}

/// JSON query tool.
pub struct JsonQueryTool {
    engine: Arc<dyn QueryEngine>,
    name: String,
}

impl Default for JsonQueryTool {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonQueryTool {
    /// Build with the default [`DotPathEngine`].
    pub fn new() -> Self {
        Self {
            engine: Arc::new(DotPathEngine),
            name: "json_query".into(),
        }
    }

    /// Use a custom query engine (e.g. a JMESPath wrapper).
    pub fn with_engine<E: QueryEngine + 'static>(engine: E) -> Self {
        Self {
            engine: Arc::new(engine),
            name: "json_query".into(),
        }
    }

    /// Override the registered tool name (e.g. `"jmespath"`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Tool for JsonQueryTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "Query a JSON value using a path expression. Supports object \
         keys (`a.b.c`), array indices (`a[0]`), and `*` for all values \
         of an object."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(JsonQueryInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: JsonQueryInput = serde_json::from_value(input.into_json()).map_err(|e| {
            CognisError::ToolValidationError(format!("json_query: invalid args: {e}"))
        })?;
        let result = self.engine.query(&parsed.data, &parsed.path)?;
        Ok(ToolOutput::Content(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dot_path_navigates_objects() {
        let v = json!({"a": {"b": {"c": 42}}});
        let r = DotPathEngine.query(&v, "a.b.c").unwrap();
        assert_eq!(r, json!(42));
    }

    #[test]
    fn dot_path_array_index() {
        let v = json!({"items": [10, 20, 30]});
        let r = DotPathEngine.query(&v, "items[1]").unwrap();
        assert_eq!(r, json!(20));
    }

    #[test]
    fn dot_path_mixed_index_and_keys() {
        let v = json!({"users": [{"name": "alice"}, {"name": "bob"}]});
        let r = DotPathEngine.query(&v, "users[1].name").unwrap();
        assert_eq!(r, json!("bob"));
    }

    #[test]
    fn wildcard_returns_all_values() {
        let v = json!({"a": 1, "b": 2, "c": 3});
        let r = DotPathEngine.query(&v, "*").unwrap();
        let arr = r.as_array().unwrap();
        // Order across HashMap is non-deterministic; verify membership.
        let mut nums: Vec<i64> = arr.iter().map(|v| v.as_i64().unwrap()).collect();
        nums.sort();
        assert_eq!(nums, vec![1, 2, 3]);
    }

    #[test]
    fn missing_key_returns_null() {
        let v = json!({"a": 1});
        let r = DotPathEngine.query(&v, "missing.deeper").unwrap();
        assert_eq!(r, json!(null));
    }

    #[test]
    fn out_of_range_index_returns_null() {
        let v = json!({"items": [1, 2]});
        let r = DotPathEngine.query(&v, "items[99]").unwrap();
        assert_eq!(r, json!(null));
    }

    #[test]
    fn unclosed_bracket_errors() {
        let err = DotPathEngine.query(&json!({}), "a[1").unwrap_err();
        assert!(matches!(err, CognisError::ToolValidationError(_)));
    }

    #[test]
    fn non_numeric_index_errors() {
        let err = DotPathEngine.query(&json!({}), "a[xyz]").unwrap_err();
        assert!(matches!(err, CognisError::ToolValidationError(_)));
    }

    #[tokio::test]
    async fn tool_runs_with_default_engine() {
        let t = JsonQueryTool::new();
        let mut map = std::collections::HashMap::new();
        map.insert("data".to_string(), json!({"x": [1, 2, 3]}));
        map.insert("path".to_string(), json!("x[2]"));
        let input = ToolInput::Structured(map);
        let out = t._run(input).await.unwrap();
        match out {
            ToolOutput::Content(v) => assert_eq!(v, json!(3)),
            _ => panic!("expected content"),
        }
    }

    #[tokio::test]
    async fn tool_with_custom_engine_via_closure() {
        // Custom engine: always return a constant.
        let t = JsonQueryTool::with_engine(
            |_v: &serde_json::Value, _p: &str| -> Result<serde_json::Value> { Ok(json!("custom")) },
        )
        .with_name("constant");
        assert_eq!(t.name(), "constant");
        let mut map = std::collections::HashMap::new();
        map.insert("data".to_string(), json!({}));
        map.insert("path".to_string(), json!("x"));
        let out = t._run(ToolInput::Structured(map)).await.unwrap();
        match out {
            ToolOutput::Content(v) => assert_eq!(v, json!("custom")),
            _ => panic!(),
        }
    }
}
