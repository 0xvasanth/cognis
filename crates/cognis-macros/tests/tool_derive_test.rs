//! Tests for `#[derive(JsonSchema)]` — standalone, zero framework dependencies.
//!
//! These tests verify that the macro generates correct OpenAPI-compatible
//! JSON schemas using only `serde` and `serde_json`.

use cognis_macros::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// =========================================================================
// Test types
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BasicStruct {
    /// A string field
    name: String,
    /// A number field
    score: f64,
    /// An integer field
    count: i32,
    /// A boolean field
    active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct OptionalFields {
    /// Required field
    query: String,
    /// Optional limit
    limit: Option<i32>,
    /// Optional flag
    verbose: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct VecFields {
    /// List of tags
    tags: Vec<String>,
    /// List of scores
    scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MapField {
    /// Key-value metadata
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ValueField {
    /// An action name
    action: String,
    /// Arbitrary data
    params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RenamedFields {
    /// The input text
    #[serde(rename = "inputText")]
    input_text: String,
    /// The output format
    #[serde(rename = "outputFormat")]
    output_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SkippedField {
    /// Visible field
    visible: String,
    /// Internal state
    #[serde(skip)]
    internal: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct DefaultField {
    /// Required name
    name: String,
    /// Count with default
    #[serde(default)]
    count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum SimpleEnum {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum RenamedEnum {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "xml")]
    Xml,
    #[serde(rename = "csv")]
    Csv,
}

// --- Nested structs (3 levels deep) ---

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct GeoCoordinate {
    /// Latitude (-90 to 90)
    latitude: f64,
    /// Longitude (-180 to 180)
    longitude: f64,
    /// Optional altitude in meters
    altitude: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Address {
    /// Street address
    street: String,
    /// City name
    city: String,
    /// Country code
    country: String,
    /// GPS coordinates
    coordinates: GeoCoordinate,
    /// Tags
    tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum Priority {
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "express")]
    Express,
    #[serde(rename = "overnight")]
    Overnight,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LineItem {
    /// Product SKU
    sku: String,
    /// Quantity
    quantity: u32,
    /// Price per unit
    unit_price: f64,
    /// Discount percentage
    discount: Option<f64>,
    /// Item metadata
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Order {
    /// Order ID
    order_id: String,
    /// Customer email
    email: String,
    /// Shipping address (level 2 → GeoCoordinate at level 3)
    address: Address,
    /// Order items (array of nested structs)
    items: Vec<LineItem>,
    /// Shipping priority (enum)
    priority: Priority,
    /// Gift message
    gift_message: Option<String>,
    /// Send confirmation
    #[serde(default)]
    send_confirmation: bool,
}

// =========================================================================
// Tests: basic type mapping
// =========================================================================

#[test]
fn test_string_type() {
    let schema = BasicStruct::json_schema();
    assert_eq!(schema["properties"]["name"]["type"], "string");
}

#[test]
fn test_number_type() {
    let schema = BasicStruct::json_schema();
    assert_eq!(schema["properties"]["score"]["type"], "number");
}

#[test]
fn test_integer_type() {
    let schema = BasicStruct::json_schema();
    assert_eq!(schema["properties"]["count"]["type"], "integer");
}

#[test]
fn test_boolean_type() {
    let schema = BasicStruct::json_schema();
    assert_eq!(schema["properties"]["active"]["type"], "boolean");
}

#[test]
fn test_top_level_is_object() {
    let schema = BasicStruct::json_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"].is_object());
}

#[test]
fn test_all_required() {
    let schema = BasicStruct::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 4);
    assert!(required.contains(&json!("name")));
    assert!(required.contains(&json!("score")));
    assert!(required.contains(&json!("count")));
    assert!(required.contains(&json!("active")));
}

// =========================================================================
// Tests: doc comments → descriptions
// =========================================================================

#[test]
fn test_doc_comments_become_descriptions() {
    let schema = BasicStruct::json_schema();
    assert_eq!(
        schema["properties"]["name"]["description"],
        "A string field"
    );
    assert_eq!(
        schema["properties"]["score"]["description"],
        "A number field"
    );
    assert_eq!(
        schema["properties"]["count"]["description"],
        "An integer field"
    );
    assert_eq!(
        schema["properties"]["active"]["description"],
        "A boolean field"
    );
}

// =========================================================================
// Tests: Option<T> handling
// =========================================================================

#[test]
fn test_optional_not_required() {
    let schema = OptionalFields::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert!(!required.contains(&json!("limit")));
    assert!(!required.contains(&json!("verbose")));
}

#[test]
fn test_optional_still_in_properties() {
    let schema = OptionalFields::json_schema();
    assert_eq!(schema["properties"]["limit"]["type"], "integer");
    assert_eq!(schema["properties"]["verbose"]["type"], "boolean");
}

// =========================================================================
// Tests: Vec<T>, HashMap, Value
// =========================================================================

#[test]
fn test_vec_string() {
    let schema = VecFields::json_schema();
    assert_eq!(schema["properties"]["tags"]["type"], "array");
    assert_eq!(schema["properties"]["tags"]["items"]["type"], "string");
}

#[test]
fn test_vec_number() {
    let schema = VecFields::json_schema();
    assert_eq!(schema["properties"]["scores"]["type"], "array");
    assert_eq!(schema["properties"]["scores"]["items"]["type"], "number");
}

#[test]
fn test_hashmap() {
    let schema = MapField::json_schema();
    assert_eq!(schema["properties"]["metadata"]["type"], "object");
    assert_eq!(
        schema["properties"]["metadata"]["additionalProperties"]["type"],
        "string"
    );
}

#[test]
fn test_value_any() {
    let schema = ValueField::json_schema();
    assert_eq!(schema["properties"]["action"]["type"], "string");
    // Value maps to empty schema (any type)
    assert!(schema["properties"]["params"].is_object());
    assert!(schema["properties"]["params"]["type"].is_null());
}

// =========================================================================
// Tests: serde attributes
// =========================================================================

#[test]
fn test_serde_rename() {
    let schema = RenamedFields::json_schema();
    assert!(schema["properties"]["inputText"].is_object());
    assert!(schema["properties"]["outputFormat"].is_object());
    assert!(schema["properties"]["input_text"].is_null());
}

#[test]
fn test_serde_skip() {
    let schema = SkippedField::json_schema();
    assert!(schema["properties"]["visible"].is_object());
    assert!(schema["properties"]["internal"].is_null());
}

#[test]
fn test_serde_default_not_required() {
    let schema = DefaultField::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("name")));
    assert!(!required.contains(&json!("count")));
    // But still in properties
    assert!(schema["properties"]["count"].is_object());
}

// =========================================================================
// Tests: enums
// =========================================================================

#[test]
fn test_enum_schema() {
    let schema = SimpleEnum::json_schema();
    assert_eq!(schema["type"], "string");
    let vals = schema["enum"].as_array().unwrap();
    assert_eq!(vals.len(), 4);
    assert!(vals.contains(&json!("Add")));
    assert!(vals.contains(&json!("Subtract")));
    assert!(vals.contains(&json!("Multiply")));
    assert!(vals.contains(&json!("Divide")));
}

#[test]
fn test_enum_with_serde_rename() {
    let schema = RenamedEnum::json_schema();
    let vals = schema["enum"].as_array().unwrap();
    assert!(vals.contains(&json!("json")));
    assert!(vals.contains(&json!("xml")));
    assert!(vals.contains(&json!("csv")));
}

// =========================================================================
// Tests: nested structs (3 levels)
// =========================================================================

#[test]
fn test_nested_level1() {
    let schema = Order::json_schema();
    println!("schema - {}", schema);
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["order_id"]["type"], "string");
    assert_eq!(schema["properties"]["email"]["type"], "string");
    assert!(schema["properties"]["address"]["type"].is_string());
    assert!(schema["properties"]["items"]["type"].is_string());

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("order_id")));
    assert!(required.contains(&json!("address")));
    assert!(required.contains(&json!("items")));
    assert!(!required.contains(&json!("gift_message")));
    assert!(!required.contains(&json!("send_confirmation")));
}

#[test]
fn test_nested_level2_address() {
    let schema = Order::json_schema();
    let addr = &schema["properties"]["address"];
    assert_eq!(addr["type"], "object");
    assert_eq!(addr["properties"]["street"]["type"], "string");
    assert_eq!(addr["properties"]["city"]["type"], "string");
    assert_eq!(addr["properties"]["coordinates"]["type"], "object");
    assert_eq!(addr["properties"]["tags"]["type"], "array");
}

#[test]
fn test_nested_level3_coordinates() {
    let schema = Order::json_schema();
    let coords = &schema["properties"]["address"]["properties"]["coordinates"];
    assert_eq!(coords["type"], "object");
    assert_eq!(coords["properties"]["latitude"]["type"], "number");
    assert_eq!(coords["properties"]["longitude"]["type"], "number");
    assert_eq!(
        coords["properties"]["latitude"]["description"],
        "Latitude (-90 to 90)"
    );

    let req = coords["required"].as_array().unwrap();
    assert!(req.contains(&json!("latitude")));
    assert!(req.contains(&json!("longitude")));
    assert!(!req.contains(&json!("altitude")));
}

#[test]
fn test_array_of_nested_structs() {
    let schema = Order::json_schema();
    let items = &schema["properties"]["items"];
    assert_eq!(items["type"], "array");

    let item = &items["items"];
    assert_eq!(item["type"], "object");
    assert_eq!(item["properties"]["sku"]["type"], "string");
    assert_eq!(item["properties"]["quantity"]["type"], "integer");
    assert_eq!(item["properties"]["unit_price"]["type"], "number");
    assert_eq!(item["properties"]["metadata"]["type"], "object");

    let req = item["required"].as_array().unwrap();
    assert!(req.contains(&json!("sku")));
    assert!(!req.contains(&json!("discount")));
}

#[test]
fn test_enum_in_nested() {
    let schema = Order::json_schema();
    let priority = &schema["properties"]["priority"];
    assert_eq!(priority["type"], "string");
    let vals = priority["enum"].as_array().unwrap();
    assert!(vals.contains(&json!("standard")));
    assert!(vals.contains(&json!("express")));
    assert!(vals.contains(&json!("overnight")));
}

// =========================================================================
// Tests: standalone schemas match embedded
// =========================================================================

#[test]
fn test_standalone_matches_embedded() {
    let standalone = Address::json_schema();
    let order_schema = Order::json_schema();
    let mut embedded = order_schema["properties"]["address"].clone();
    embedded.as_object_mut().unwrap().remove("description");
    assert_eq!(standalone, embedded);
}

#[test]
fn test_standalone_coordinates_matches() {
    let standalone = GeoCoordinate::json_schema();
    let order_schema = Order::json_schema();
    let mut embedded = order_schema["properties"]["address"]["properties"]["coordinates"].clone();
    embedded.as_object_mut().unwrap().remove("description");
    assert_eq!(standalone, embedded);
}

#[test]
fn test_standalone_line_item_matches() {
    let standalone = LineItem::json_schema();
    let order_schema = Order::json_schema();
    let embedded = &order_schema["properties"]["items"]["items"];
    assert_eq!(standalone, *embedded);
}

// =========================================================================
// Tests: JSON roundtrip
// =========================================================================

#[test]
fn test_schema_roundtrips_through_json() {
    let schema = Order::json_schema();
    let json_str = serde_json::to_string_pretty(&schema).unwrap();
    let reparsed: Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(schema, reparsed);
}

#[test]
fn test_openapi_structure() {
    let schema = Order::json_schema();

    // Every level has type + properties + required
    assert!(schema["type"].is_string());
    assert!(schema["properties"].is_object());
    assert!(schema["required"].is_array());

    let addr = &schema["properties"]["address"];
    assert!(addr["type"].is_string());
    assert!(addr["properties"].is_object());

    let coords = &addr["properties"]["coordinates"];
    assert!(coords["type"].is_string());
    assert!(coords["properties"].is_object());
    assert!(coords["required"].is_array());
}
