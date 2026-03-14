//! Integration tests for `#[derive(JsonSchema)]` (standalone) and `#[derive(Tool)]` (framework).

use async_trait::async_trait;
use cognis_core::error::Result;
use cognis_core::tools::{ToolInput, ToolJsonSchema, ToolOutput};
use cognis_core::{JsonSchema, Tool, ToolSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// =========================================================================
// Standalone schemas — use #[derive(JsonSchema)], NO framework dependency
// =========================================================================

/// A nested struct — standalone schema, no BaseTool needed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct FilterConfig {
    /// Minimum score threshold
    min_score: f64,
    /// Maximum number of results
    max_results: i32,
    /// Whether to deduplicate
    deduplicate: Option<bool>,
}

/// An enum — standalone schema.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// An enum with serde rename — standalone schema.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum OutputFormat {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "xml")]
    Xml,
    #[serde(rename = "csv")]
    Csv,
}

// =========================================================================
// Tool structs — use #[derive(Tool)], generates BaseTool + json_schema()
// =========================================================================

/// A simple calculator tool.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "calculator", description = "Performs basic arithmetic")]
struct CalculatorTool {
    /// The first operand
    a: f64,
    /// The second operand
    b: f64,
    /// The operation to perform
    operation: String,
}

impl CalculatorTool {
    async fn execute(&self) -> Result<ToolOutput> {
        let result = match self.operation.as_str() {
            "add" => self.a + self.b,
            "sub" => self.a - self.b,
            "mul" => self.a * self.b,
            "div" => self.a / self.b,
            _ => {
                return Err(cognis_core::error::CognisError::ToolException(
                    "unknown op".into(),
                ))
            }
        };
        Ok(ToolOutput::Content(json!(result)))
    }
}

/// A search tool with optional parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct SearchTool {
    /// The search query
    query: String,
    /// Maximum number of results
    limit: Option<i32>,
    /// Whether to include metadata
    include_metadata: Option<bool>,
}

impl SearchTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!({"results": []})))
    }
}

/// Tool with serde rename attributes.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "renamed_tool")]
struct RenamedFieldsTool {
    /// The input text
    #[serde(rename = "inputText")]
    input_text: String,
    /// The output format
    #[serde(rename = "outputFormat")]
    output_format: String,
}

impl RenamedFieldsTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.input_text)))
    }
}

/// Tool with a skipped field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct SkippedFieldTool {
    /// The visible field
    visible: String,
    /// This field is internal
    #[serde(skip)]
    internal_state: i32,
}

impl SkippedFieldTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.visible)))
    }
}

/// Tool with serde default.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct DefaultFieldTool {
    /// Required field
    name: String,
    /// Optional with default
    #[serde(default)]
    count: i32,
}

impl DefaultFieldTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.name)))
    }
}

/// Tool with Vec field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct VecFieldTool {
    /// List of tags
    tags: Vec<String>,
    /// List of scores
    scores: Vec<f64>,
}

impl VecFieldTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.tags)))
    }
}

/// Tool with HashMap field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct MapFieldTool {
    /// Key-value metadata
    metadata: HashMap<String, String>,
}

impl MapFieldTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.metadata)))
    }
}

/// Tool with serde_json::Value field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct DynamicTool {
    /// The action name
    action: String,
    /// Arbitrary parameters
    params: Value,
}

impl DynamicTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.action)))
    }
}

/// Tool with bool field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct BoolTool {
    /// Whether to enable verbose mode
    verbose: bool,
    /// The input text
    text: String,
}

impl BoolTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.text)))
    }
}

/// Tool with integer types.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct IntegerTool {
    /// A 32-bit integer
    count_i32: i32,
    /// A 64-bit integer
    count_i64: i64,
    /// An unsigned integer
    count_u32: u32,
    /// A usize value
    count_usize: usize,
}

impl IntegerTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.count_i32)))
    }
}

/// Tool with a nested struct field.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct NestedTool {
    /// The search query
    query: String,
    /// Filter configuration
    filter: FilterConfig,
}

impl NestedTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.query)))
    }
}

/// Tool with no explicit name/description (uses defaults).
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
struct AutoNamedTool {
    /// The input value
    input: String,
}

impl AutoNamedTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(self.input)))
    }
}

/// Tool that combines many features.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(name = "complex_tool", description = "A tool with many field types")]
struct ComplexTool {
    /// Required string
    name: String,
    /// Optional integer
    age: Option<i32>,
    /// List of tags
    tags: Vec<String>,
    /// Dynamic data
    extra: Value,
    /// Field with default
    #[serde(default)]
    enabled: bool,
    /// Skipped field
    #[serde(skip)]
    _cache: String,
}

impl ComplexTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!({"name": self.name})))
    }
}

// =========================================================================
// Multi-level nested structs (3 layers deep + arrays) — standalone schemas
// =========================================================================

/// Level 3: Geographic coordinate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct GeoCoordinate {
    /// Latitude in degrees (-90 to 90)
    latitude: f64,
    /// Longitude in degrees (-180 to 180)
    longitude: f64,
    /// Optional altitude in meters
    altitude: Option<f64>,
}

/// Level 2: Delivery address.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Address {
    /// Street address line
    street: String,
    /// City name
    city: String,
    /// ZIP or postal code
    zip_code: String,
    /// Country code (ISO 3166-1 alpha-2)
    country: String,
    /// GPS coordinates for the address
    coordinates: GeoCoordinate,
    /// Additional address tags
    tags: Vec<String>,
}

/// Shipping priority enum.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum ShippingPriority {
    #[serde(rename = "standard")]
    Standard,
    #[serde(rename = "express")]
    Express,
    #[serde(rename = "overnight")]
    Overnight,
}

/// Level 2: Order line item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct OrderItem {
    /// Product SKU identifier
    sku: String,
    /// Quantity ordered
    quantity: u32,
    /// Unit price in dollars
    unit_price: f64,
    /// Optional discount percentage
    discount: Option<f64>,
    /// Item-level metadata
    metadata: HashMap<String, String>,
}

/// Level 1: 3-layer deep order placement tool.
#[derive(Debug, Clone, Serialize, Deserialize, Tool)]
#[tool(
    name = "place_order",
    description = "Place a multi-item order with shipping address and delivery options"
)]
struct PlaceOrderTool {
    /// Unique order identifier
    order_id: String,
    /// Customer email address
    customer_email: String,
    /// Shipping destination address
    shipping_address: Address,
    /// List of items in the order
    items: Vec<OrderItem>,
    /// Shipping priority
    priority: ShippingPriority,
    /// Optional gift message
    gift_message: Option<String>,
    /// Whether to send a confirmation email
    #[serde(default)]
    send_confirmation: bool,
}

impl PlaceOrderTool {
    async fn execute(&self) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!({
            "order_id": self.order_id,
            "status": "placed",
            "item_count": self.items.len(),
        })))
    }
}

// =========================================================================
// Tests: Standalone #[derive(JsonSchema)] — no framework, just schemas
// =========================================================================

#[test]
fn test_standalone_struct_schema() {
    let schema = FilterConfig::json_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["min_score"]["type"], "number");
    assert_eq!(schema["properties"]["max_results"]["type"], "integer");

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("min_score")));
    assert!(required.contains(&json!("max_results")));
    assert!(!required.contains(&json!("deduplicate")));
}

#[test]
fn test_standalone_enum_schema() {
    let schema = Operation::json_schema();
    assert_eq!(schema["type"], "string");
    let vals = schema["enum"].as_array().unwrap();
    assert_eq!(vals.len(), 4);
    assert!(vals.contains(&json!("Add")));
    assert!(vals.contains(&json!("Subtract")));
}

#[test]
fn test_standalone_enum_with_serde_rename() {
    let schema = OutputFormat::json_schema();
    let vals = schema["enum"].as_array().unwrap();
    assert!(vals.contains(&json!("json")));
    assert!(vals.contains(&json!("xml")));
    assert!(vals.contains(&json!("csv")));
}

#[test]
fn test_standalone_nested_address() {
    let schema = Address::json_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["coordinates"]["type"], "object");
    assert_eq!(
        schema["properties"]["coordinates"]["properties"]["latitude"]["type"],
        "number"
    );
}

#[test]
fn test_standalone_geo_coordinate() {
    let schema = GeoCoordinate::json_schema();
    assert_eq!(schema["properties"]["latitude"]["type"], "number");
    assert_eq!(schema["properties"]["longitude"]["type"], "number");
    let req = schema["required"].as_array().unwrap();
    assert!(req.contains(&json!("latitude")));
    assert!(!req.contains(&json!("altitude")));
}

#[test]
fn test_standalone_order_item() {
    let schema = OrderItem::json_schema();
    assert_eq!(schema["properties"]["sku"]["type"], "string");
    assert_eq!(schema["properties"]["quantity"]["type"], "integer");
    assert_eq!(schema["properties"]["metadata"]["type"], "object");
    assert_eq!(
        schema["properties"]["metadata"]["additionalProperties"]["type"],
        "string"
    );
}

// =========================================================================
// Tests: #[derive(Tool)] — framework integration via static json_schema()
// =========================================================================

#[test]
fn test_tool_name_explicit() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 1.0,
        b: 2.0,
        operation: "add".into(),
    };
    assert_eq!(tool.name(), "calculator");
}

#[test]
fn test_tool_description_explicit() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 1.0,
        b: 2.0,
        operation: "add".into(),
    };
    assert_eq!(tool.description(), "Performs basic arithmetic");
}

#[test]
fn test_tool_name_auto_generated() {
    use cognis_core::tools::BaseTool;
    let tool = AutoNamedTool {
        input: "test".into(),
    };
    assert_eq!(tool.name(), "auto_named_tool");
}

#[test]
fn test_tool_description_from_doc_comment() {
    use cognis_core::tools::BaseTool;
    let tool = AutoNamedTool {
        input: "test".into(),
    };
    assert_eq!(
        tool.description(),
        "Tool with no explicit name/description (uses defaults)."
    );
}

#[test]
fn test_schema_basic_types() {
    let schema = CalculatorTool::json_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["a"]["type"], "number");
    assert_eq!(schema["properties"]["b"]["type"], "number");
    assert_eq!(schema["properties"]["operation"]["type"], "string");
}

#[test]
fn test_schema_descriptions_from_doc_comments() {
    let schema = CalculatorTool::json_schema();
    assert_eq!(
        schema["properties"]["a"]["description"],
        "The first operand"
    );
    assert_eq!(
        schema["properties"]["b"]["description"],
        "The second operand"
    );
}

#[test]
fn test_schema_all_fields_required() {
    let schema = CalculatorTool::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 3);
    assert!(required.contains(&json!("a")));
    assert!(required.contains(&json!("b")));
    assert!(required.contains(&json!("operation")));
}

#[test]
fn test_schema_optional_fields_not_required() {
    let schema = SearchTool::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert!(!required.contains(&json!("limit")));
}

#[test]
fn test_schema_optional_fields_still_in_properties() {
    let schema = SearchTool::json_schema();
    assert_eq!(schema["properties"]["limit"]["type"], "integer");
    assert_eq!(schema["properties"]["include_metadata"]["type"], "boolean");
}

#[test]
fn test_schema_serde_rename() {
    let schema = RenamedFieldsTool::json_schema();
    assert!(schema["properties"]["inputText"].is_object());
    assert!(schema["properties"]["input_text"].is_null());
}

#[test]
fn test_schema_serde_skip() {
    let schema = SkippedFieldTool::json_schema();
    assert!(schema["properties"]["visible"].is_object());
    assert!(schema["properties"]["internal_state"].is_null());
}

#[test]
fn test_schema_serde_default_not_required() {
    let schema = DefaultFieldTool::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("name")));
    assert!(!required.contains(&json!("count")));
}

#[test]
fn test_schema_vec_field() {
    let schema = VecFieldTool::json_schema();
    assert_eq!(schema["properties"]["tags"]["type"], "array");
    assert_eq!(schema["properties"]["tags"]["items"]["type"], "string");
}

#[test]
fn test_schema_hashmap_field() {
    let schema = MapFieldTool::json_schema();
    assert_eq!(schema["properties"]["metadata"]["type"], "object");
    assert_eq!(
        schema["properties"]["metadata"]["additionalProperties"]["type"],
        "string"
    );
}

#[test]
fn test_schema_value_field() {
    let schema = DynamicTool::json_schema();
    assert_eq!(schema["properties"]["action"]["type"], "string");
    assert!(schema["properties"]["params"]["type"].is_null());
}

#[test]
fn test_schema_bool_field() {
    let schema = BoolTool::json_schema();
    assert_eq!(schema["properties"]["verbose"]["type"], "boolean");
}

#[test]
fn test_schema_integer_types() {
    let schema = IntegerTool::json_schema();
    assert_eq!(schema["properties"]["count_i32"]["type"], "integer");
    assert_eq!(schema["properties"]["count_i64"]["type"], "integer");
    assert_eq!(schema["properties"]["count_u32"]["type"], "integer");
    assert_eq!(schema["properties"]["count_usize"]["type"], "integer");
}

#[test]
fn test_schema_nested_struct() {
    let schema = NestedTool::json_schema();
    let filter = &schema["properties"]["filter"];
    assert_eq!(filter["type"], "object");
    assert_eq!(filter["properties"]["min_score"]["type"], "number");
}

#[test]
fn test_complex_tool_schema() {
    let schema = ComplexTool::json_schema();
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("name")));
    assert!(!required.contains(&json!("age")));
    assert!(!required.contains(&json!("enabled")));
    assert!(schema["properties"]["_cache"].is_null());
}

// --- args_schema(&self) delegates to static json_schema() ---

#[test]
fn test_args_schema_matches_static_json_schema() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 99.0,
        b: -1.0,
        operation: "whatever".into(),
    };
    assert_eq!(tool.args_schema().unwrap(), CalculatorTool::json_schema());
}

// --- ToolJsonSchema bridge works ---

#[test]
fn test_tool_json_schema_bridge() {
    // The Tool derive also generates ToolJsonSchema impl that delegates
    let from_inherent = CalculatorTool::json_schema();
    let from_trait = <CalculatorTool as ToolJsonSchema>::json_schema();
    assert_eq!(from_inherent, from_trait);
}

// --- Execution ---

#[tokio::test]
async fn test_tool_run() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 3.0,
        b: 4.0,
        operation: "add".into(),
    };
    let result = tool._run(ToolInput::Text("ignored".into())).await.unwrap();
    match result {
        ToolOutput::Content(v) => assert_eq!(v, json!(7.0)),
        _ => panic!("expected Content variant"),
    }
}

// =========================================================================
// Multi-level nested tests — all static
// =========================================================================

#[test]
fn test_three_level_top_level() {
    let schema = PlaceOrderTool::json_schema();
    assert_eq!(schema["type"], "object");
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("order_id")));
    assert!(required.contains(&json!("shipping_address")));
    assert!(required.contains(&json!("items")));
    assert!(!required.contains(&json!("gift_message")));
    assert!(!required.contains(&json!("send_confirmation")));
}

#[test]
fn test_three_level_address() {
    let schema = PlaceOrderTool::json_schema();
    let addr = &schema["properties"]["shipping_address"];
    assert_eq!(addr["type"], "object");
    assert_eq!(addr["properties"]["street"]["type"], "string");
    assert_eq!(addr["properties"]["coordinates"]["type"], "object");
}

#[test]
fn test_three_level_coordinates() {
    let schema = PlaceOrderTool::json_schema();
    let coords = &schema["properties"]["shipping_address"]["properties"]["coordinates"];
    assert_eq!(coords["properties"]["latitude"]["type"], "number");
    assert_eq!(coords["properties"]["longitude"]["type"], "number");
    let req = coords["required"].as_array().unwrap();
    assert!(req.contains(&json!("latitude")));
    assert!(!req.contains(&json!("altitude")));
}

#[test]
fn test_array_of_nested_structs() {
    let schema = PlaceOrderTool::json_schema();
    let items = &schema["properties"]["items"];
    assert_eq!(items["type"], "array");
    let item = &items["items"];
    assert_eq!(item["type"], "object");
    assert_eq!(item["properties"]["sku"]["type"], "string");
    assert_eq!(item["properties"]["quantity"]["type"], "integer");
}

#[test]
fn test_enum_in_nested_schema() {
    let schema = PlaceOrderTool::json_schema();
    let priority = &schema["properties"]["priority"];
    assert_eq!(priority["type"], "string");
    let vals = priority["enum"].as_array().unwrap();
    assert!(vals.contains(&json!("standard")));
    assert!(vals.contains(&json!("express")));
    assert!(vals.contains(&json!("overnight")));
}

#[test]
fn test_standalone_matches_embedded() {
    let standalone = Address::json_schema();
    let full = PlaceOrderTool::json_schema();
    let mut embedded = full["properties"]["shipping_address"].clone();
    embedded.as_object_mut().unwrap().remove("description");
    assert_eq!(standalone, embedded);
}

#[test]
fn test_full_schema_roundtrips() {
    let schema = PlaceOrderTool::json_schema();
    let json_str = serde_json::to_string_pretty(&schema).unwrap();
    let reparsed: Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(schema, reparsed);
}

#[tokio::test]
async fn test_three_level_execution() {
    use cognis_core::tools::BaseTool;
    let tool = PlaceOrderTool {
        order_id: "ORD-123".into(),
        customer_email: "user@test.com".into(),
        shipping_address: Address {
            street: "456 Oak Ave".into(),
            city: "Portland".into(),
            zip_code: "97201".into(),
            country: "US".into(),
            coordinates: GeoCoordinate {
                latitude: 45.5152,
                longitude: -122.6784,
                altitude: Some(15.0),
            },
            tags: vec!["residential".into()],
        },
        items: vec![OrderItem {
            sku: "WIDGET-001".into(),
            quantity: 3,
            unit_price: 29.99,
            discount: None,
            metadata: HashMap::new(),
        }],
        priority: ShippingPriority::Express,
        gift_message: None,
        send_confirmation: true,
    };
    let result = tool._run(ToolInput::Text("ignored".into())).await.unwrap();
    match result {
        ToolOutput::Content(v) => {
            assert_eq!(v["order_id"], "ORD-123");
            assert_eq!(v["item_count"], 1);
        }
        _ => panic!("expected Content variant"),
    }
}
