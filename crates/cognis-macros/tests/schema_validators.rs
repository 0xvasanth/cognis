use cognis_macros::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(JsonSchema, Serialize, Deserialize)]
#[allow(dead_code)]
struct Sample {
    /// Search query
    #[schema(length(min = 1, max = 100))]
    query: String,
    #[schema(range(min = 1, max = 50))]
    limit: Option<u32>,
    #[schema(enum_values("asc", "desc"))]
    order: String,
    #[schema(format("email"))]
    contact: String,
    #[schema(pattern("^[a-z]+$"))]
    slug: String,
}

#[test]
fn schema_has_length_keys_on_string() {
    let s = Sample::json_schema();
    let query = &s["properties"]["query"];
    assert_eq!(query["minLength"], json!(1));
    assert_eq!(query["maxLength"], json!(100));
    assert_eq!(query["description"], json!("Search query"));
}

#[test]
fn schema_has_range_keys_on_optional_int() {
    let s = Sample::json_schema();
    let limit = &s["properties"]["limit"];
    assert_eq!(limit["minimum"], json!(1));
    assert_eq!(limit["maximum"], json!(50));
}

#[test]
fn schema_has_enum_keys() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["order"]["enum"], json!(["asc", "desc"]));
}

#[test]
fn schema_has_format_key() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["contact"]["format"], json!("email"));
}

#[test]
fn schema_has_pattern_key() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["slug"]["pattern"], json!("^[a-z]+$"));
}

#[test]
fn optional_fields_not_in_required() {
    let s = Sample::json_schema();
    let required: Vec<String> = s["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(required.contains(&"query".to_string()));
    assert!(!required.contains(&"limit".to_string()));
}

#[derive(JsonSchema, Serialize, Deserialize)]
#[allow(dead_code)]
struct Multi {
    #[schema(length(min = 1, max = 100))]
    #[schema(pattern("^[a-z]+$"))]
    slug: String,
}

#[test]
fn multiple_schema_attrs_are_all_applied() {
    let s = Multi::json_schema();
    let slug = &s["properties"]["slug"];
    assert_eq!(slug["minLength"], json!(1));
    assert_eq!(slug["maxLength"], json!(100));
    assert_eq!(slug["pattern"], json!("^[a-z]+$"));
}
