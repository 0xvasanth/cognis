//! Integration tests for `#[derive(Tool)]` and `#[derive(ToolSchema)]`.

use async_trait::async_trait;
use cognis_core::error::Result;
use cognis_core::tools::{ToolInput, ToolJsonSchema, ToolOutput};
use cognis_core::{Tool, ToolSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

// =========================================================================
// Test structs
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

/// A nested struct used as a field type.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
struct FilterConfig {
    /// Minimum score threshold
    min_score: f64,
    /// Maximum number of results
    max_results: i32,
    /// Whether to deduplicate
    deduplicate: Option<bool>,
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

/// An enum that derives ToolSchema.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// An enum with serde rename on variants.
#[derive(Debug, Clone, Serialize, Deserialize, ToolSchema)]
enum OutputFormat {
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "xml")]
    Xml,
    #[serde(rename = "csv")]
    Csv,
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
// Tests
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
    // Doc comment becomes description
    assert_eq!(
        tool.description(),
        "Tool with no explicit name/description (uses defaults)."
    );
}

#[test]
fn test_schema_basic_types() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 0.0,
        b: 0.0,
        operation: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["a"]["type"], "number");
    assert_eq!(schema["properties"]["b"]["type"], "number");
    assert_eq!(schema["properties"]["operation"]["type"], "string");
}

#[test]
fn test_schema_descriptions_from_doc_comments() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 0.0,
        b: 0.0,
        operation: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(
        schema["properties"]["a"]["description"],
        "The first operand"
    );
    assert_eq!(
        schema["properties"]["b"]["description"],
        "The second operand"
    );
    assert_eq!(
        schema["properties"]["operation"]["description"],
        "The operation to perform"
    );
}

#[test]
fn test_schema_all_fields_required() {
    use cognis_core::tools::BaseTool;
    let tool = CalculatorTool {
        a: 0.0,
        b: 0.0,
        operation: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 3);
    assert!(required.contains(&json!("a")));
    assert!(required.contains(&json!("b")));
    assert!(required.contains(&json!("operation")));
}

#[test]
fn test_schema_optional_fields_not_required() {
    use cognis_core::tools::BaseTool;
    let tool = SearchTool {
        query: String::new(),
        limit: None,
        include_metadata: None,
    };
    let schema = tool.args_schema().unwrap();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert!(!required.contains(&json!("limit")));
    assert!(!required.contains(&json!("include_metadata")));
}

#[test]
fn test_schema_optional_fields_still_in_properties() {
    use cognis_core::tools::BaseTool;
    let tool = SearchTool {
        query: String::new(),
        limit: None,
        include_metadata: None,
    };
    let schema = tool.args_schema().unwrap();
    assert!(schema["properties"]["limit"].is_object());
    assert_eq!(schema["properties"]["limit"]["type"], "integer");
    assert!(schema["properties"]["include_metadata"].is_object());
    assert_eq!(schema["properties"]["include_metadata"]["type"], "boolean");
}

#[test]
fn test_schema_serde_rename() {
    use cognis_core::tools::BaseTool;
    let tool = RenamedFieldsTool {
        input_text: String::new(),
        output_format: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    // Should use renamed keys
    assert!(schema["properties"]["inputText"].is_object());
    assert!(schema["properties"]["outputFormat"].is_object());
    // Original names should not exist
    assert!(schema["properties"]["input_text"].is_null());
    assert!(schema["properties"]["output_format"].is_null());
}

#[test]
fn test_schema_serde_skip() {
    use cognis_core::tools::BaseTool;
    let tool = SkippedFieldTool {
        visible: String::new(),
        internal_state: 0,
    };
    let schema = tool.args_schema().unwrap();
    assert!(schema["properties"]["visible"].is_object());
    // Skipped field should not appear
    assert!(schema["properties"]["internal_state"].is_null());
}

#[test]
fn test_schema_serde_default_not_required() {
    use cognis_core::tools::BaseTool;
    let tool = DefaultFieldTool {
        name: String::new(),
        count: 0,
    };
    let schema = tool.args_schema().unwrap();
    let required = schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("name")));
    assert!(!required.contains(&json!("count")));
    // But it should still be in properties
    assert!(schema["properties"]["count"].is_object());
}

#[test]
fn test_schema_vec_field() {
    use cognis_core::tools::BaseTool;
    let tool = VecFieldTool {
        tags: vec![],
        scores: vec![],
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["tags"]["type"], "array");
    assert_eq!(schema["properties"]["tags"]["items"]["type"], "string");
    assert_eq!(schema["properties"]["scores"]["type"], "array");
    assert_eq!(schema["properties"]["scores"]["items"]["type"], "number");
}

#[test]
fn test_schema_hashmap_field() {
    use cognis_core::tools::BaseTool;
    let tool = MapFieldTool {
        metadata: HashMap::new(),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["metadata"]["type"], "object");
    assert_eq!(
        schema["properties"]["metadata"]["additionalProperties"]["type"],
        "string"
    );
}

#[test]
fn test_schema_value_field() {
    use cognis_core::tools::BaseTool;
    let tool = DynamicTool {
        action: String::new(),
        params: json!(null),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["action"]["type"], "string");
    // Value type should produce an empty schema (any type)
    assert!(schema["properties"]["params"].is_object());
    assert!(schema["properties"]["params"]["type"].is_null());
}

#[test]
fn test_schema_bool_field() {
    use cognis_core::tools::BaseTool;
    let tool = BoolTool {
        verbose: false,
        text: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["verbose"]["type"], "boolean");
}

#[test]
fn test_schema_integer_types() {
    use cognis_core::tools::BaseTool;
    let tool = IntegerTool {
        count_i32: 0,
        count_i64: 0,
        count_u32: 0,
        count_usize: 0,
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["count_i32"]["type"], "integer");
    assert_eq!(schema["properties"]["count_i64"]["type"], "integer");
    assert_eq!(schema["properties"]["count_u32"]["type"], "integer");
    assert_eq!(schema["properties"]["count_usize"]["type"], "integer");
}

#[test]
fn test_schema_nested_struct() {
    use cognis_core::tools::BaseTool;
    let tool = NestedTool {
        query: String::new(),
        filter: FilterConfig {
            min_score: 0.0,
            max_results: 10,
            deduplicate: None,
        },
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["properties"]["query"]["type"], "string");
    // Nested struct should be an object with its own properties
    let filter_schema = &schema["properties"]["filter"];
    assert_eq!(filter_schema["type"], "object");
    assert_eq!(filter_schema["properties"]["min_score"]["type"], "number");
    assert_eq!(
        filter_schema["properties"]["max_results"]["type"],
        "integer"
    );
    // Optional field in nested struct should not be required
    let filter_required = filter_schema["required"].as_array().unwrap();
    assert!(filter_required.contains(&json!("min_score")));
    assert!(filter_required.contains(&json!("max_results")));
    assert!(!filter_required.contains(&json!("deduplicate")));
}

#[test]
fn test_enum_tool_schema() {
    let schema = Operation::json_schema();
    assert_eq!(schema["type"], "string");
    let enum_vals = schema["enum"].as_array().unwrap();
    assert_eq!(enum_vals.len(), 4);
    assert!(enum_vals.contains(&json!("Add")));
    assert!(enum_vals.contains(&json!("Subtract")));
    assert!(enum_vals.contains(&json!("Multiply")));
    assert!(enum_vals.contains(&json!("Divide")));
}

#[test]
fn test_enum_with_serde_rename() {
    let schema = OutputFormat::json_schema();
    assert_eq!(schema["type"], "string");
    let enum_vals = schema["enum"].as_array().unwrap();
    assert_eq!(enum_vals.len(), 3);
    assert!(enum_vals.contains(&json!("json")));
    assert!(enum_vals.contains(&json!("xml")));
    assert!(enum_vals.contains(&json!("csv")));
}

#[test]
fn test_complex_tool_schema() {
    use cognis_core::tools::BaseTool;
    let tool = ComplexTool {
        name: String::new(),
        age: None,
        tags: vec![],
        extra: json!(null),
        enabled: false,
        _cache: String::new(),
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["type"], "object");

    // Required: only "name", "tags", and "extra"
    // "age" is Option, "enabled" has serde(default), "_cache" is skipped
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("name")));
    assert!(required.contains(&json!("tags")));
    assert!(required.contains(&json!("extra")));
    assert!(!required.contains(&json!("age")));
    assert!(!required.contains(&json!("enabled")));
    assert!(!required.contains(&json!("_cache")));

    // _cache should not be in properties at all
    assert!(schema["properties"]["_cache"].is_null());
}

#[test]
fn test_tool_json_schema_trait_for_struct() {
    let schema = FilterConfig::json_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["min_score"]["type"], "number");
}

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

#[test]
fn test_tool_schema_type_is_object_at_top_level() {
    use cognis_core::tools::BaseTool;
    let tool = SearchTool {
        query: "test".into(),
        limit: None,
        include_metadata: None,
    };
    let schema = tool.args_schema().unwrap();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"].is_object());
}

#[test]
fn test_primitive_tool_json_schema_impls() {
    assert_eq!(String::json_schema(), json!({"type": "string"}));
    assert_eq!(f64::json_schema(), json!({"type": "number"}));
    assert_eq!(f32::json_schema(), json!({"type": "number"}));
    assert_eq!(i32::json_schema(), json!({"type": "integer"}));
    assert_eq!(i64::json_schema(), json!({"type": "integer"}));
    assert_eq!(u32::json_schema(), json!({"type": "integer"}));
    assert_eq!(bool::json_schema(), json!({"type": "boolean"}));
    assert_eq!(Value::json_schema(), json!({}));
}
