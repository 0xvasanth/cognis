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
    #[schema(length(min = 1, max = 5))]
    tags: Vec<String>,
    #[schema(items(range(min = 0, max = 10)))]
    scores: Vec<u32>,
    #[schema(range(min = 0.5, max = 1.5))]
    ratio: f64,
}

#[test]
fn schema_has_length_keys_on_string() {
    let s = Sample::json_schema();
    let query = &s["properties"]["query"];
    assert_eq!(query["minLength"], json!(1));
    assert_eq!(query["maxLength"], json!(100));
    assert_eq!(query["description"], json!("Search query"));
    // Inverse: the array-branch keys must NOT fire on a string field.
    assert!(
        query.get("minItems").is_none(),
        "minItems should not fire on a string field"
    );
    assert!(
        query.get("maxItems").is_none(),
        "maxItems should not fire on a string field"
    );
}

#[test]
fn schema_has_range_keys_on_optional_int() {
    let s = Sample::json_schema();
    let limit = &s["properties"]["limit"];
    assert_eq!(limit["minimum"], json!(1));
    assert_eq!(limit["maximum"], json!(50));
    assert!(limit.get("minLength").is_none());
    assert!(limit.get("pattern").is_none());
}

#[test]
fn schema_has_enum_keys() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["order"]["enum"], json!(["asc", "desc"]));
    let order = &s["properties"]["order"];
    assert!(order.get("pattern").is_none());
    assert!(order.get("format").is_none());
}

#[test]
fn schema_has_format_key() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["contact"]["format"], json!("email"));
    let contact = &s["properties"]["contact"];
    assert!(contact.get("pattern").is_none());
    assert!(contact.get("minLength").is_none());
}

#[test]
fn schema_has_pattern_key() {
    let s = Sample::json_schema();
    assert_eq!(s["properties"]["slug"]["pattern"], json!("^[a-z]+$"));
    let slug = &s["properties"]["slug"];
    assert!(slug.get("enum").is_none());
    assert!(slug.get("format").is_none());
}

#[test]
fn schema_has_items_keys_on_vec() {
    let s = Sample::json_schema();
    let tags = &s["properties"]["tags"];
    assert_eq!(tags["minItems"], json!(1));
    assert_eq!(tags["maxItems"], json!(5));
    // Inverse: the string-branch keys must NOT fire on an array field.
    assert!(tags.get("minLength").is_none());
    assert!(tags.get("maxLength").is_none());
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

#[test]
fn schema_has_nested_items_keys_on_vec() {
    let s = Sample::json_schema();
    let scores = &s["properties"]["scores"];
    // Base array schema preserved
    assert_eq!(scores["type"], json!("array"));
    // Nested range constraints merged into items subschema
    assert_eq!(scores["items"]["minimum"], json!(0));
    assert_eq!(scores["items"]["maximum"], json!(10));
}

#[test]
fn schema_range_preserves_fractional_f64() {
    let s = Sample::json_schema();
    let ratio = &s["properties"]["ratio"];
    assert_eq!(ratio["minimum"], json!(0.5));
    assert_eq!(ratio["maximum"], json!(1.5));
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

#[derive(JsonSchema, Serialize, Deserialize)]
#[allow(dead_code)]
struct Signed {
    #[schema(range(min = -10, max = 10))]
    offset: i32,
}

#[test]
fn schema_range_emits_negative_minimum() {
    let s = Signed::json_schema();
    let offset = &s["properties"]["offset"];
    assert_eq!(offset["minimum"], json!(-10));
    assert_eq!(offset["maximum"], json!(10));
}
