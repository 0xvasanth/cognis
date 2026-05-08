use cognis_core::error::CognisError;
use cognis_core::tools::*;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;

// --- Task 1: Error variants ---
#[test]
fn test_tool_validation_error() {
    let e = CognisError::ToolValidationError("bad input".into());
    assert!(e.to_string().contains("bad input"));
}

#[test]
fn test_schema_annotation_error() {
    let e = CognisError::SchemaAnnotationError("wrong type".into());
    assert!(e.to_string().contains("wrong type"));
}

// --- Task 2: Tool types ---
#[test]
fn test_tool_input_from_text() {
    let input = ToolInput::Text("hello".into());
    match input {
        ToolInput::Text(s) => assert_eq!(s, "hello"),
        _ => panic!("Expected Text"),
    }
}

#[test]
fn test_tool_input_from_structured() {
    let mut args = HashMap::new();
    args.insert("query".into(), json!("test"));
    let input = ToolInput::Structured(args);
    match input {
        ToolInput::Structured(m) => assert_eq!(m["query"], json!("test")),
        _ => panic!("Expected Structured"),
    }
}

#[test]
fn test_tool_input_from_tool_call() {
    let tc = ToolCallInput {
        id: "tc_1".into(),
        name: "search".into(),
        args: {
            let mut m = HashMap::new();
            m.insert("q".into(), json!("rust"));
            m
        },
    };
    let input = ToolInput::ToolCall(tc);
    match input {
        ToolInput::ToolCall(tc) => {
            assert_eq!(tc.name, "search");
            assert_eq!(tc.id, "tc_1");
        }
        _ => panic!("Expected ToolCall"),
    }
}

#[test]
fn test_tool_input_deserialize_text() {
    let v: ToolInput = serde_json::from_value(json!("hello")).unwrap();
    match v {
        ToolInput::Text(s) => assert_eq!(s, "hello"),
        _ => panic!("Expected Text"),
    }
}

#[test]
fn test_tool_input_deserialize_structured() {
    let v: ToolInput = serde_json::from_value(json!({"query": "test"})).unwrap();
    // Note: with untagged, a JSON object with "id", "name", "args" would deserialize
    // as ToolCall first, but {"query": "test"} should become Structured.
    match v {
        ToolInput::Structured(m) => assert_eq!(m["query"], json!("test")),
        _ => panic!("Expected Structured"),
    }
}

#[test]
fn test_response_format_default() {
    let rf = ResponseFormat::default();
    assert_eq!(rf, ResponseFormat::Content);
}

#[test]
fn test_error_handler_default() {
    let eh = ErrorHandler::default();
    match eh {
        ErrorHandler::Propagate => {}
        _ => panic!("Expected Propagate"),
    }
}

#[test]
fn test_error_handler_static_message() {
    let eh = ErrorHandler::StaticMessage("oops".into());
    match eh {
        ErrorHandler::StaticMessage(s) => assert_eq!(s, "oops"),
        _ => panic!("Expected StaticMessage"),
    }
}

#[test]
fn test_tool_schema_serialize() {
    let schema = ToolSchema {
        name: "search".into(),
        description: "Search the web".into(),
        parameters: Some(json!({"type": "object"})),
        extras: None,
    };
    let v = serde_json::to_value(&schema).unwrap();
    assert_eq!(v["name"], "search");
    assert!(v.get("extras").is_none());
}

// --- Task 3: FunctionTool ---
#[tokio::test]
async fn test_function_tool_basic() {
    let tool = FunctionTool::new(
        "greet",
        "Greets a person",
        Some(json!({"type": "object", "properties": {"name": {"type": "string"}}})),
        |input| {
            let name = match &input {
                ToolInput::Structured(m) => m
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("world")
                    .to_string(),
                ToolInput::Text(s) => s.clone(),
                _ => "world".into(),
            };
            Ok(ToolOutput::Content(Value::String(format!(
                "Hello, {}!",
                name
            ))))
        },
    );
    assert_eq!(tool.name(), "greet");
    assert_eq!(tool.description(), "Greets a person");
    let mut args = HashMap::new();
    args.insert("name".into(), json!("Alice"));
    let result = tool.run(ToolInput::Structured(args), None).await.unwrap();
    assert_eq!(result, json!("Hello, Alice!"));
}

#[tokio::test]
async fn test_function_tool_text_input() {
    let tool = FunctionTool::new("echo", "Echoes input", None, |input| {
        let text = match &input {
            ToolInput::Text(s) => s.clone(),
            _ => "?".into(),
        };
        Ok(ToolOutput::Content(Value::String(text)))
    });
    let result = tool.run_str("hello").await.unwrap();
    assert_eq!(result, json!("hello"));
}

#[tokio::test]
async fn test_function_tool_error_handling() {
    let tool = FunctionTool::new("fail", "Always fails", None, |_| {
        Err(CognisError::ToolException("broke".into()))
    })
    .with_error_handler(ErrorHandler::DefaultMessage);
    let result = tool.run_str("anything").await.unwrap();
    assert_eq!(result, json!("broke"));
}

#[tokio::test]
async fn test_function_tool_propagate_error() {
    let tool = FunctionTool::new("fail", "Always fails", None, |_| {
        Err(CognisError::ToolException("broke".into()))
    });
    let result = tool.run_str("anything").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_function_tool_return_direct() {
    let tool = FunctionTool::new("t", "d", None, |_| Ok(ToolOutput::Content(json!("ok"))))
        .with_return_direct(true);
    assert!(tool.return_direct());
}

// --- Task 4: Render ---
#[test]
fn test_render_text_description() {
    let t1 = FunctionTool::new("search", "Search the web", None, |_| {
        Ok(ToolOutput::Content(json!("ok")))
    });
    let t2 = FunctionTool::new("calc", "Do math", None, |_| {
        Ok(ToolOutput::Content(json!("ok")))
    });
    let tools: Vec<&dyn BaseTool> = vec![&t1, &t2];
    let text = render_text_description(&tools);
    assert_eq!(text, "search - Search the web\ncalc - Do math");
}

#[test]
fn test_render_text_description_and_args() {
    let t1 = FunctionTool::new(
        "search",
        "Search the web",
        Some(json!({"type": "object", "properties": {"q": {"type": "string"}}})),
        |_| Ok(ToolOutput::Content(json!("ok"))),
    );
    let tools: Vec<&dyn BaseTool> = vec![&t1];
    let text = render_text_description_and_args(&tools);
    assert!(text.contains("search - Search the web, args:"));
    assert!(text.contains("\"q\""));
}

// --- Task 5: convert_to_openai ---
#[test]
fn test_convert_to_openai_tool() {
    let tool = FunctionTool::new(
        "search",
        "Search the web",
        Some(json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})),
        |_| Ok(ToolOutput::Content(json!("ok"))),
    );
    let openai = convert_to_openai_tool(&tool);
    assert_eq!(openai["type"], "function");
    assert_eq!(openai["function"]["name"], "search");
    assert_eq!(openai["function"]["description"], "Search the web");
    assert!(openai["function"]["parameters"]["properties"]["q"].is_object());
}

#[test]
fn test_convert_to_openai_tool_no_schema() {
    let tool = FunctionTool::new("echo", "Echo input", None, |_| {
        Ok(ToolOutput::Content(json!("ok")))
    });
    let openai = convert_to_openai_tool(&tool);
    assert_eq!(openai["type"], "function");
    assert_eq!(openai["function"]["name"], "echo");
}

#[test]
fn test_convert_to_openai_tools_multiple() {
    let t1 = FunctionTool::new("a", "A tool", None, |_| {
        Ok(ToolOutput::Content(json!("ok")))
    });
    let t2 = FunctionTool::new("b", "B tool", None, |_| {
        Ok(ToolOutput::Content(json!("ok")))
    });
    let tools: Vec<&dyn BaseTool> = vec![&t1, &t2];
    let result = convert_to_openai_tools(&tools);
    assert_eq!(result.len(), 2);
}
