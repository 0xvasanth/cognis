//! Tests for round 2 migrations: example_selectors, embeddings_fake,
//! openai_tools parser, schema parser, utils (function_calling, mustache).

use std::collections::HashMap;

use serde_json::{json, Value};

// ============================================================
// Example Selectors (Length-based)
// ============================================================

use cognis_core::prompts::base::PromptTemplate;
use cognis_core::prompts::example_selector::BaseExampleSelector;
use cognis_core::prompts::example_selectors::LengthBasedExampleSelector;

fn make_example(input: &str, output: &str) -> HashMap<String, Value> {
    let mut m = HashMap::new();
    m.insert("input".to_string(), json!(input));
    m.insert("output".to_string(), json!(output));
    m
}

#[tokio::test]
async fn test_length_selector_selects_all_when_fits() {
    let prompt = PromptTemplate::from_template("Input: {input}\nOutput: {output}");
    let examples = vec![make_example("happy", "sad"), make_example("tall", "short")];
    let selector = LengthBasedExampleSelector::new(examples, prompt, 100);

    let input = {
        let mut m = HashMap::new();
        m.insert("input".to_string(), json!("big"));
        m
    };
    let selected = selector.select_examples(&input).await.unwrap();
    assert_eq!(selected.len(), 2);
}

#[tokio::test]
async fn test_length_selector_truncates_when_exceeds() {
    let prompt = PromptTemplate::from_template("Input: {input}\nOutput: {output}");
    let examples = vec![
        make_example("happy", "sad"),
        make_example("tall", "short"),
        make_example("energetic", "lethargic"),
    ];
    // Very short max_length to force truncation
    let selector = LengthBasedExampleSelector::new(examples, prompt, 10);

    let input = {
        let mut m = HashMap::new();
        m.insert("input".to_string(), json!("big"));
        m
    };
    let selected = selector.select_examples(&input).await.unwrap();
    assert!(selected.len() < 3, "Should truncate examples");
}

#[tokio::test]
async fn test_length_selector_empty_when_input_exceeds() {
    let prompt = PromptTemplate::from_template("{input} {output}");
    let examples = vec![make_example("x", "y")];
    // max_length = 2, but input alone might use it all
    let selector = LengthBasedExampleSelector::new(examples, prompt, 1);

    let mut input = HashMap::new();
    input.insert(
        "input".to_string(),
        json!("this is a very long input string that exceeds"),
    );
    let selected = selector.select_examples(&input).await.unwrap();
    assert!(selected.is_empty());
}

#[tokio::test]
async fn test_length_selector_add_example() {
    let prompt = PromptTemplate::from_template("{input} -> {output}");
    let selector = LengthBasedExampleSelector::new(vec![], prompt, 100);

    selector
        .add_example(make_example("hot", "cold"))
        .await
        .unwrap();

    let mut input = HashMap::new();
    input.insert("input".to_string(), json!("x"));
    let selected = selector.select_examples(&input).await.unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].get("input").unwrap(), "hot");
}

// ============================================================
// Embeddings (Fake)
// ============================================================

use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::{DeterministicFakeEmbedding, FakeConstantEmbedding};

#[tokio::test]
async fn test_deterministic_embedding_consistent() {
    let emb = DeterministicFakeEmbedding::new(10);
    let v1 = emb.embed_query("hello world").await.unwrap();
    let v2 = emb.embed_query("hello world").await.unwrap();
    assert_eq!(v1, v2);
    assert_eq!(v1.len(), 10);
}

#[tokio::test]
async fn test_deterministic_embedding_different_texts() {
    let emb = DeterministicFakeEmbedding::new(5);
    let v1 = emb.embed_query("hello").await.unwrap();
    let v2 = emb.embed_query("world").await.unwrap();
    assert_ne!(v1, v2);
}

#[tokio::test]
async fn test_deterministic_embedding_documents() {
    let emb = DeterministicFakeEmbedding::new(8);
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let embeddings = emb.embed_documents(docs).await.unwrap();
    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].len(), 8);
    assert_ne!(embeddings[0], embeddings[1]);
}

#[tokio::test]
async fn test_constant_embedding_all_zeros() {
    let emb = FakeConstantEmbedding::new(4);
    let v = emb.embed_query("anything").await.unwrap();
    assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_constant_embedding_documents() {
    let emb = FakeConstantEmbedding::new(3);
    let embeddings = emb
        .embed_documents(vec!["a".into(), "b".into()])
        .await
        .unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0], vec![0.0, 0.0, 0.0]);
    assert_eq!(embeddings[1], vec![0.0, 0.0, 0.0]);
}

// ============================================================
// OpenAI Tools Output Parser
// ============================================================

use cognis_core::output_parsers::openai_tools::{
    parse_tool_call, parse_tool_calls, JsonOutputKeyToolsParser, OpenAIToolsOutputParser,
};
use cognis_core::output_parsers::OutputParser;

#[test]
fn test_parse_tool_call_basic() {
    let raw = json!({
        "function": {
            "name": "search",
            "arguments": "{\"query\": \"weather\"}"
        },
        "id": "call_123"
    });
    let result = parse_tool_call(&raw, false).unwrap();
    assert_eq!(result["type"], "search");
    assert_eq!(result["args"]["query"], "weather");
    assert!(result.get("id").is_none());
}

#[test]
fn test_parse_tool_call_with_id() {
    let raw = json!({
        "function": {
            "name": "calc",
            "arguments": "{\"x\": 1}"
        },
        "id": "call_456"
    });
    let result = parse_tool_call(&raw, true).unwrap();
    assert_eq!(result["id"], "call_456");
}

#[test]
fn test_parse_tool_calls_multiple() {
    let calls = vec![
        json!({"function": {"name": "a", "arguments": "{}"}, "id": "1"}),
        json!({"function": {"name": "b", "arguments": "{}"}, "id": "2"}),
    ];
    let results = parse_tool_calls(&calls, true).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["type"], "a");
    assert_eq!(results[1]["type"], "b");
}

#[test]
fn test_openai_tools_parser_from_json() {
    let parser = OpenAIToolsOutputParser::new();
    let input = json!({
        "tool_calls": [
            {"function": {"name": "search", "arguments": "{\"q\": \"test\"}"}, "id": "1"}
        ]
    });
    let result = parser.extract_tool_calls(&input).unwrap();
    let arr = result.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["type"], "search");
}

#[test]
fn test_openai_tools_parser_first_only() {
    let parser = OpenAIToolsOutputParser::new().first_only();
    let input = json!({
        "tool_calls": [
            {"function": {"name": "a", "arguments": "{}"}, "id": "1"},
            {"function": {"name": "b", "arguments": "{}"}, "id": "2"}
        ]
    });
    let result = parser.extract_tool_calls(&input).unwrap();
    assert_eq!(result["type"], "a");
}

#[test]
fn test_openai_tools_parser_with_id() {
    let parser = OpenAIToolsOutputParser::new().with_id();
    let input = json!({
        "tool_calls": [
            {"function": {"name": "x", "arguments": "{}"}, "id": "call_99"}
        ]
    });
    let result = parser.extract_tool_calls(&input).unwrap();
    let arr = result.as_array().unwrap();
    assert_eq!(arr[0]["id"], "call_99");
}

#[test]
fn test_openai_tools_parser_nested_additional_kwargs() {
    let parser = OpenAIToolsOutputParser::new();
    let input = json!({
        "additional_kwargs": {
            "tool_calls": [
                {"function": {"name": "fn1", "arguments": "{}"}}
            ]
        }
    });
    let result = parser.extract_tool_calls(&input).unwrap();
    let arr = result.as_array().unwrap();
    assert_eq!(arr[0]["type"], "fn1");
}

#[test]
fn test_json_output_key_tools_parser() {
    let parser = JsonOutputKeyToolsParser::new("answer");
    let input = json!([
        {"type": "get_answer", "args": {"answer": "42"}},
        {"type": "get_answer", "args": {"answer": "hello"}}
    ]);
    let result = parser.parse(&input.to_string()).unwrap();
    let arr = result.as_array().unwrap();
    assert_eq!(arr, &[json!("42"), json!("hello")]);
}

// ============================================================
// Schema Output Parser (Pydantic equivalent)
// ============================================================

use cognis_core::output_parsers::pydantic::SchemaOutputParser;

#[test]
fn test_schema_parser_valid() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let parser = SchemaOutputParser::new("Person", schema);
    let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
    assert_eq!(result["name"], "Alice");
    assert_eq!(result["age"], 30);
}

#[test]
fn test_schema_parser_missing_required() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let parser = SchemaOutputParser::new("Person", schema);
    let result = parser.parse(r#"{"name": "Bob"}"#);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("age"));
}

#[test]
fn test_schema_parser_format_instructions() {
    let schema = json!({
        "type": "object",
        "title": "MyType",
        "properties": {
            "x": {"type": "string"}
        },
        "required": ["x"]
    });
    let parser = SchemaOutputParser::new("MyType", schema);
    let instructions = parser.get_format_instructions().unwrap();
    assert!(instructions.contains("JSON schema"));
    assert!(instructions.contains("properties"));
}

#[test]
fn test_schema_parser_strips_markdown() {
    let schema = json!({"type": "object", "properties": {"v": {"type": "string"}}});
    let parser = SchemaOutputParser::new("Test", schema);
    let result = parser.parse("```json\n{\"v\": \"hello\"}\n```").unwrap();
    assert_eq!(result["v"], "hello");
}

// ============================================================
// Utils: Function Calling
// ============================================================

use cognis_core::utils::function_calling::{
    build_parameters_schema, convert_json_schema_to_openai_function, convert_to_openai_tool,
    set_additional_properties_false, ParameterInfo,
};

#[test]
fn test_convert_json_schema_to_function() {
    let schema = json!({
        "title": "get_weather",
        "description": "Get weather for a location",
        "properties": {
            "location": {"type": "string"}
        }
    });
    let func = convert_json_schema_to_openai_function(&schema, None, None, false);
    assert_eq!(func.name, "get_weather");
    assert_eq!(func.description.unwrap(), "Get weather for a location");
}

#[test]
fn test_convert_json_schema_custom_name() {
    let schema = json!({"properties": {}});
    let func =
        convert_json_schema_to_openai_function(&schema, Some("my_func"), Some("desc"), false);
    assert_eq!(func.name, "my_func");
    assert_eq!(func.description.unwrap(), "desc");
}

#[test]
fn test_convert_json_schema_rm_titles() {
    let schema = json!({
        "title": "Root",
        "properties": {
            "nested": {
                "title": "Nested",
                "type": "object"
            }
        }
    });
    let func = convert_json_schema_to_openai_function(&schema, Some("test"), None, true);
    assert!(func.parameters.get("title").is_none());
    assert!(func.parameters["properties"]["nested"]
        .get("title")
        .is_none());
}

#[test]
fn test_convert_to_openai_tool() {
    let schema = json!({"title": "fn", "properties": {}});
    let func = convert_json_schema_to_openai_function(&schema, None, None, false);
    let tool = convert_to_openai_tool(func);
    assert_eq!(tool.tool_type, "function");
    assert_eq!(tool.function.name, "fn");
}

#[test]
fn test_set_additional_properties_false() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                }
            }
        }
    });
    set_additional_properties_false(&mut schema);
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["nested"]["additionalProperties"],
        false
    );
}

#[test]
fn test_build_parameters_schema() {
    let mut params = HashMap::new();
    params.insert(
        "query".to_string(),
        ParameterInfo {
            json_type: "string".to_string(),
            description: Some("The search query".to_string()),
            enum_values: None,
        },
    );
    params.insert(
        "limit".to_string(),
        ParameterInfo {
            json_type: "integer".to_string(),
            description: None,
            enum_values: None,
        },
    );
    let schema = build_parameters_schema(&params, &["query".to_string()]);
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"], json!(["query"]));
    assert!(schema["properties"]["query"]["description"].is_string());
}

// ============================================================
// Utils: Mustache
// ============================================================

use cognis_core::utils::mustache::{render, template_vars};

#[test]
fn test_mustache_basic_variable() {
    let result = render("Hello {{name}}!", &json!({"name": "World"})).unwrap();
    assert_eq!(result, "Hello World!");
}

#[test]
fn test_mustache_html_escape() {
    let result = render("{{text}}", &json!({"text": "<b>bold</b>"})).unwrap();
    assert_eq!(result, "&lt;b&gt;bold&lt;/b&gt;");
}

#[test]
fn test_mustache_unescaped_ampersand() {
    let result = render("{{&text}}", &json!({"text": "<b>bold</b>"})).unwrap();
    assert_eq!(result, "<b>bold</b>");
}

#[test]
fn test_mustache_triple_braces() {
    let result = render("{{{text}}}", &json!({"text": "<em>hi</em>"})).unwrap();
    assert_eq!(result, "<em>hi</em>");
}

#[test]
fn test_mustache_section_truthy() {
    let result = render("{{#show}}visible{{/show}}", &json!({"show": true})).unwrap();
    assert_eq!(result, "visible");
}

#[test]
fn test_mustache_section_falsy() {
    let result = render("{{#show}}visible{{/show}}", &json!({"show": false})).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_mustache_inverted_section() {
    let result = render("{{^items}}no items{{/items}}", &json!({"items": []})).unwrap();
    assert_eq!(result, "no items");
}

#[test]
fn test_mustache_inverted_section_with_data() {
    let result = render("{{^items}}no items{{/items}}", &json!({"items": [1]})).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_mustache_array_section() {
    let result = render(
        "{{#items}}{{.}} {{/items}}",
        &json!({"items": ["a", "b", "c"]}),
    )
    .unwrap();
    assert_eq!(result, "a b c ");
}

#[test]
fn test_mustache_object_section() {
    let result = render(
        "{{#person}}{{name}} is {{age}}{{/person}}",
        &json!({"person": {"name": "Alice", "age": 30}}),
    )
    .unwrap();
    assert_eq!(result, "Alice is 30");
}

#[test]
fn test_mustache_dot_key() {
    let result = render("{{#items}}({{.}}){{/items}}", &json!({"items": [1, 2, 3]})).unwrap();
    assert_eq!(result, "(1)(2)(3)");
}

#[test]
fn test_mustache_nested_key() {
    let result = render("{{person.name}}", &json!({"person": {"name": "Bob"}})).unwrap();
    assert_eq!(result, "Bob");
}

#[test]
fn test_mustache_missing_key() {
    let result = render("{{missing}}", &json!({"other": 1})).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_mustache_comment() {
    let result = render("before{{! this is a comment }}after", &json!({})).unwrap();
    assert_eq!(result, "beforeafter");
}

#[test]
fn test_template_vars_basic() {
    let vars = template_vars("Hello {{name}}, you are {{age}} years old.");
    assert_eq!(vars, vec!["name", "age"]);
}

#[test]
fn test_template_vars_sections() {
    let vars = template_vars("{{#section}}{{var}}{{/section}}");
    assert!(vars.contains(&"section".to_string()));
    assert!(vars.contains(&"var".to_string()));
    // Should not include the closing tag
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_template_vars_comments_excluded() {
    let vars = template_vars("{{name}}{{! comment }}{{other}}");
    assert_eq!(vars, vec!["name", "other"]);
}

// ============================================================
// Utils: General utilities
// ============================================================

use cognis_core::utils::{
    ensure_id, generate_id, get_from_dict_or_env, merge_dicts, python_to_json_type,
};

#[test]
fn test_merge_dicts_objects() {
    let left = json!({"a": 1, "b": {"x": 10}});
    let right = json!({"b": {"y": 20}, "c": 3});
    let merged = merge_dicts(&left, &[&right]).unwrap();
    assert_eq!(merged["a"], 1);
    assert_eq!(merged["b"]["x"], 10);
    assert_eq!(merged["b"]["y"], 20);
    assert_eq!(merged["c"], 3);
}

#[test]
fn test_generate_id_is_uuid() {
    let id = generate_id();
    assert_eq!(id.len(), 36); // UUID format
    assert!(id.contains('-'));
}

#[test]
fn test_ensure_id_with_value() {
    let id = ensure_id(Some("my_id".into()));
    assert_eq!(id, "my_id");
}

#[test]
fn test_ensure_id_generates() {
    let id = ensure_id(None);
    assert!(id.starts_with("lc_"));
}

#[test]
fn test_get_from_dict_or_env_from_dict() {
    let mut data = HashMap::new();
    data.insert("key".to_string(), json!("value"));
    let result = get_from_dict_or_env(&data, "key", "NONEXISTENT_VAR", None);
    assert_eq!(result, Some("value".to_string()));
}

#[test]
fn test_get_from_dict_or_env_default() {
    let data = HashMap::new();
    let result = get_from_dict_or_env(&data, "missing", "NONEXISTENT_VAR_12345", Some("default"));
    assert_eq!(result, Some("default".to_string()));
}

#[test]
fn test_python_to_json_type() {
    assert_eq!(python_to_json_type("str"), "string");
    assert_eq!(python_to_json_type("String"), "string");
    assert_eq!(python_to_json_type("int"), "integer");
    assert_eq!(python_to_json_type("i64"), "integer");
    assert_eq!(python_to_json_type("float"), "number");
    assert_eq!(python_to_json_type("f64"), "number");
    assert_eq!(python_to_json_type("bool"), "boolean");
    assert_eq!(python_to_json_type("SomeClass"), "object");
}
