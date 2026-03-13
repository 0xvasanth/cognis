//! Tool Schema Example
//!
//! Demonstrates JSON Schema-based tool parameter definition, validation, and
//! export to OpenAI and Anthropic formats. Covers the full lifecycle from
//! property definitions through schema registration and call validation.
//!
//! Features shown:
//! - PropertySchema static constructors (string, integer, boolean, enum_type, array)
//! - SchemaObject building with required fields
//! - ToolSchema creation and export to OpenAI JSON format
//! - ToolSchema export to Anthropic JSON format
//! - SchemaValidator validation (passing and failing cases)
//! - ToolSchemaGenerator declarative building
//! - SchemaRegistry registration and validate_call
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example tool_schema`

mod shared;
use cognis_core::tools::schema::{
    PropertySchema, SchemaObject, SchemaRegistry, SchemaValidator, ToolSchema, ToolSchemaGenerator,
};
use serde_json::json;

fn main() {
    println!("=== Tool Schema Example ===\n");

    // -----------------------------------------------------------------------
    // 1. PropertySchema Static Constructors
    // -----------------------------------------------------------------------
    println!("--- 1. PropertySchema Static Constructors ---");
    println!("Build property schemas for common JSON Schema types.\n");

    // String with constraints
    let name_prop = PropertySchema::string()
        .with_description("The user's full name")
        .with_min_length(1)
        .with_max_length(100);
    println!("String property: {:?}", name_prop.to_json());

    // Integer with range
    let age_prop = PropertySchema::integer()
        .with_description("Age in years")
        .with_minimum(0.0)
        .with_maximum(150.0);
    println!("Integer property: {:?}", age_prop.to_json());

    // Boolean
    let active_prop = PropertySchema::boolean().with_description("Whether the account is active");
    println!("Boolean property: {:?}", active_prop.to_json());

    // Enum type (string with allowed values)
    let role_prop =
        PropertySchema::enum_type(vec!["admin".into(), "editor".into(), "viewer".into()])
            .with_description("User role");
    println!("Enum property:    {:?}", role_prop.to_json());

    // Array of strings
    let tags_prop =
        PropertySchema::array(PropertySchema::string()).with_description("List of tags");
    println!("Array property:   {:?}", tags_prop.to_json());

    // -----------------------------------------------------------------------
    // 2. SchemaObject Building
    // -----------------------------------------------------------------------
    println!("\n--- 2. SchemaObject Building ---");
    println!("Compose properties into an object schema with required fields.\n");

    let params = SchemaObject::new()
        .with_property(
            "query",
            PropertySchema::string().with_description("Search query text"),
        )
        .with_property(
            "max_results",
            PropertySchema::integer()
                .with_description("Maximum number of results")
                .with_minimum(1.0)
                .with_maximum(100.0),
        )
        .with_property(
            "category",
            PropertySchema::enum_type(vec!["docs".into(), "code".into(), "issues".into()])
                .with_description("Category to search in"),
        )
        .with_required("query")
        .with_required("max_results");

    let schema_json = params.to_json();
    println!("Schema object JSON:");
    println!("{}", serde_json::to_string_pretty(&schema_json).unwrap());

    // -----------------------------------------------------------------------
    // 3. ToolSchema — OpenAI Format
    // -----------------------------------------------------------------------
    println!("\n--- 3. ToolSchema — OpenAI Format ---");
    println!("Create a tool schema and export it in OpenAI function-calling format.\n");

    let search_tool = ToolSchema::new("search_documents", "Search the document index")
        .with_parameters(params.clone())
        .with_strict(false);

    let openai_json = search_tool.to_json();
    println!("OpenAI function-calling format:");
    println!("{}", serde_json::to_string_pretty(&openai_json).unwrap());

    // Verify the structure
    assert_eq!(openai_json["type"], "function");
    assert_eq!(openai_json["function"]["name"], "search_documents");
    println!("\nStructure verified: type=function, name=search_documents");

    // -----------------------------------------------------------------------
    // 4. ToolSchema — Anthropic Format
    // -----------------------------------------------------------------------
    println!("\n--- 4. ToolSchema — Anthropic Format ---");
    println!("Export the same tool schema in Anthropic tool_use format.\n");

    let anthropic_json = search_tool.to_anthropic_json();
    println!("Anthropic tool_use format:");
    println!("{}", serde_json::to_string_pretty(&anthropic_json).unwrap());

    // Verify the structure
    assert_eq!(anthropic_json["name"], "search_documents");
    assert!(anthropic_json.get("input_schema").is_some());
    println!("\nStructure verified: name=search_documents, has input_schema");

    // -----------------------------------------------------------------------
    // 5. SchemaValidator — Passing and Failing Cases
    // -----------------------------------------------------------------------
    println!("\n--- 5. SchemaValidator ---");
    println!("Validate JSON inputs against a schema object.\n");

    let validation_schema = SchemaObject::new()
        .with_property(
            "name",
            PropertySchema::string()
                .with_description("User name")
                .with_min_length(1),
        )
        .with_property(
            "age",
            PropertySchema::integer()
                .with_description("Age")
                .with_minimum(0.0)
                .with_maximum(150.0),
        )
        .with_property(
            "role",
            PropertySchema::enum_type(vec!["admin".into(), "user".into()]).with_description("Role"),
        )
        .with_required("name")
        .with_required("age");

    // Valid input
    let valid_input = json!({
        "name": "Alice",
        "age": 30,
        "role": "admin"
    });
    match SchemaValidator::validate(&validation_schema, &valid_input) {
        Ok(()) => println!("Valid input:   PASS (as expected)"),
        Err(errs) => println!("Valid input:   FAIL (unexpected) {:?}", errs),
    }

    // Missing required field
    let missing_field = json!({
        "name": "Bob"
    });
    match SchemaValidator::validate(&validation_schema, &missing_field) {
        Ok(()) => println!("Missing field: PASS (unexpected)"),
        Err(errs) => {
            println!("Missing field: FAIL (as expected)");
            for e in &errs {
                println!("  -> {}", e);
            }
        }
    }

    // Wrong type
    let wrong_type = json!({
        "name": "Charlie",
        "age": "thirty"
    });
    match SchemaValidator::validate(&validation_schema, &wrong_type) {
        Ok(()) => println!("Wrong type:    PASS (unexpected)"),
        Err(errs) => {
            println!("Wrong type:    FAIL (as expected)");
            for e in &errs {
                println!("  -> {}", e);
            }
        }
    }

    // Invalid enum value
    let bad_enum = json!({
        "name": "Diana",
        "age": 25,
        "role": "superadmin"
    });
    match SchemaValidator::validate(&validation_schema, &bad_enum) {
        Ok(()) => println!("Bad enum:      PASS (unexpected)"),
        Err(errs) => {
            println!("Bad enum:      FAIL (as expected)");
            for e in &errs {
                println!("  -> {}", e);
            }
        }
    }

    // Non-object input
    let not_object = json!("just a string");
    match SchemaValidator::validate(&validation_schema, &not_object) {
        Ok(()) => println!("Non-object:    PASS (unexpected)"),
        Err(errs) => {
            println!("Non-object:    FAIL (as expected)");
            for e in &errs {
                println!("  -> {}", e);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. ToolSchemaGenerator — Declarative Building
    // -----------------------------------------------------------------------
    println!("\n--- 6. ToolSchemaGenerator ---");
    println!("Build tool schemas declaratively with typed parameter helpers.\n");

    let weather_tool = ToolSchemaGenerator::new("get_weather", "Get current weather for a city")
        .add_string_param("city", "City name", true)
        .add_enum_param(
            "units",
            "Temperature units",
            vec!["celsius".into(), "fahrenheit".into()],
            false,
        )
        .add_boolean_param("include_forecast", "Include 5-day forecast", false)
        .add_integer_param("forecast_days", "Number of forecast days", false)
        .add_array_param(
            "fields",
            "Specific weather fields to return",
            PropertySchema::string(),
            false,
        )
        .build();

    println!("Generated tool schema (OpenAI):");
    println!(
        "{}",
        serde_json::to_string_pretty(&weather_tool.to_json()).unwrap()
    );

    // Validate a call against the generated schema
    let valid_call = json!({
        "city": "San Francisco",
        "units": "celsius",
        "include_forecast": true
    });
    match weather_tool.validate_input(&valid_call) {
        Ok(()) => println!("\nWeather tool call validation: PASS"),
        Err(errs) => println!("\nWeather tool call validation: FAIL {:?}", errs),
    }

    // -----------------------------------------------------------------------
    // 7. SchemaRegistry — Registration and validate_call
    // -----------------------------------------------------------------------
    println!("\n--- 7. SchemaRegistry ---");
    println!("Register multiple tools and validate calls by name.\n");

    let mut registry = SchemaRegistry::new();

    // Register the search tool
    registry.register(search_tool);

    // Register the weather tool
    registry.register(weather_tool);

    // Build and register a third tool
    let calc_tool = ToolSchema::new("calculator", "Perform arithmetic operations").with_parameters(
        SchemaObject::new()
            .with_property(
                "expression",
                PropertySchema::string().with_description("Math expression to evaluate"),
            )
            .with_property(
                "precision",
                PropertySchema::integer()
                    .with_description("Decimal places")
                    .with_minimum(0.0)
                    .with_maximum(20.0),
            )
            .with_required("expression"),
    );
    registry.register(calc_tool);

    println!("Registered {} tools in the registry", registry.len());

    // List all registered schemas
    let all = registry.all_schemas();
    println!("Registered tools:");
    for schema in &all {
        println!(
            "  - {} (params: {})",
            schema.name,
            schema.parameters.properties.len()
        );
    }

    // Validate a correct call
    let good_call = json!({
        "query": "vector databases",
        "max_results": 10
    });
    match registry.validate_call("search_documents", &good_call) {
        Ok(()) => println!("\nvalidate_call('search_documents', valid):   PASS"),
        Err(e) => println!("\nvalidate_call('search_documents', valid):   FAIL - {}", e),
    }

    // Validate a call with missing required field
    let bad_call = json!({
        "category": "docs"
    });
    match registry.validate_call("search_documents", &bad_call) {
        Ok(()) => println!("validate_call('search_documents', missing): PASS (unexpected)"),
        Err(e) => println!("validate_call('search_documents', missing): FAIL - {}", e),
    }

    // Validate a call to an unknown tool
    match registry.validate_call("nonexistent_tool", &json!({})) {
        Ok(()) => println!("validate_call('nonexistent_tool'):           PASS (unexpected)"),
        Err(e) => println!("validate_call('nonexistent_tool'):           FAIL - {}", e),
    }

    // Export the full registry as JSON (OpenAI format)
    let registry_json = registry.to_json();
    println!(
        "\nFull registry export ({} tools):",
        registry_json.as_array().map_or(0, |a| a.len())
    );
    println!("{}", serde_json::to_string_pretty(&registry_json).unwrap());

    println!("\n=== Tool Schema Example Complete ===");
}
