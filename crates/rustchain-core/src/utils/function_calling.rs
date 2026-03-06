//! Utilities for converting tool/function descriptions to OpenAI function calling format.
//!
//! Mirrors Python `langchain_core.utils.function_calling`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Description of a function in OpenAI function calling format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDescription {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

/// A tool description wrapping a function description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescription {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDescription,
}

/// Convert a JSON schema into an OpenAI function description.
pub fn convert_json_schema_to_openai_function(
    schema: &Value,
    name: Option<&str>,
    description: Option<&str>,
    rm_titles: bool,
) -> FunctionDescription {
    let func_name = name
        .map(|s| s.to_string())
        .or_else(|| {
            schema
                .get("title")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unnamed_function".to_string());

    let func_desc = description.map(|s| s.to_string()).or_else(|| {
        schema
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });

    let mut params = schema.clone();
    if rm_titles {
        remove_titles(&mut params);
    }

    FunctionDescription {
        name: func_name,
        description: func_desc,
        parameters: params,
    }
}

/// Convert a function description into a tool description.
pub fn convert_to_openai_tool(function: FunctionDescription) -> ToolDescription {
    ToolDescription {
        tool_type: "function".to_string(),
        function,
    }
}

/// Recursively remove `title` keys from a JSON schema.
fn remove_titles(value: &mut Value) {
    if let Value::Object(map) = value {
        map.remove("title");
        for v in map.values_mut() {
            remove_titles(v);
        }
    } else if let Value::Array(arr) = value {
        for v in arr.iter_mut() {
            remove_titles(v);
        }
    }
}

/// Recursively add `"additionalProperties": false` for strict mode.
pub fn set_additional_properties_false(value: &mut Value) {
    if let Value::Object(map) = value {
        if map.contains_key("properties") {
            map.insert("additionalProperties".to_string(), json!(false));
        }
        for v in map.values_mut() {
            set_additional_properties_false(v);
        }
    }
}

/// Build an OpenAI-compatible JSON schema from a map of parameter descriptions.
pub fn build_parameters_schema(
    params: &HashMap<String, ParameterInfo>,
    required: &[String],
) -> Value {
    let mut properties = serde_json::Map::new();
    for (name, info) in params {
        let mut prop = serde_json::Map::new();
        prop.insert("type".to_string(), json!(info.json_type));
        if let Some(desc) = &info.description {
            prop.insert("description".to_string(), json!(desc));
        }
        if let Some(enum_vals) = &info.enum_values {
            prop.insert("enum".to_string(), json!(enum_vals));
        }
        properties.insert(name.clone(), Value::Object(prop));
    }

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// Information about a single function parameter.
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub json_type: String,
    pub description: Option<String>,
    pub enum_values: Option<Vec<String>>,
}

/// Convert an OpenAI tool description back into a JSON Schema.
///
/// Extracts the function schema from a tool description, producing a
/// standard JSON Schema with title and description.
pub fn convert_to_json_schema(tool: &ToolDescription) -> Value {
    let mut schema = serde_json::Map::new();
    schema.insert("title".to_string(), json!(tool.function.name));
    if let Some(desc) = &tool.function.description {
        schema.insert("description".to_string(), json!(desc));
    }
    if let Value::Object(params) = &tool.function.parameters {
        for (k, v) in params {
            schema.insert(k.clone(), v.clone());
        }
    }
    Value::Object(schema)
}

/// Convert a tool call example into a list of messages for few-shot prompting.
///
/// Creates:
/// 1. A `HumanMessage` with the input
/// 2. An `AIMessage` with tool_calls
/// 3. A `ToolMessage` per tool call (with tool output or placeholder)
/// 4. Optionally, a final `AIMessage` with `ai_response`
pub fn tool_example_to_messages(
    input: &str,
    tool_calls: Vec<Value>,
    tool_outputs: Option<Vec<String>>,
    ai_response: Option<&str>,
) -> Vec<crate::messages::Message> {
    use crate::messages::Message;

    let mut messages: Vec<Message> = Vec::new();

    // 1. Human message with the input
    messages.push(Message::human(input));

    // 2. AI message with tool calls
    let tool_call_values: Vec<Value> = tool_calls
        .iter()
        .enumerate()
        .map(|(i, tc)| {
            let mut call = serde_json::Map::new();
            call.insert(
                "name".to_string(),
                tc.get("name").cloned().unwrap_or(json!("tool")),
            );
            call.insert(
                "args".to_string(),
                tc.get("args").cloned().unwrap_or(json!({})),
            );
            call.insert("id".to_string(), json!(format!("call_{}", i)));
            Value::Object(call)
        })
        .collect();

    let ai_msg = Message::ai_with_tool_calls("", tool_call_values.clone());
    messages.push(ai_msg);

    // 3. Tool messages with outputs
    for (i, tc) in tool_call_values.iter().enumerate() {
        let output = tool_outputs
            .as_ref()
            .and_then(|outputs| outputs.get(i))
            .cloned()
            .unwrap_or_else(|| "You have correctly called this tool.".to_string());

        let tool_call_id = tc
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        messages.push(Message::tool(&output, &tool_call_id));
    }

    // 4. Optional final AI response
    if let Some(response) = ai_response {
        messages.push(Message::ai(response));
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_json_schema_to_openai_function() {
        let schema = json!({
            "title": "GetWeather",
            "description": "Get the weather",
            "type": "object",
            "properties": {
                "location": {"type": "string", "title": "Location"}
            }
        });
        let func = convert_json_schema_to_openai_function(&schema, None, None, true);
        assert_eq!(func.name, "GetWeather");
        assert_eq!(func.description.unwrap(), "Get the weather");
        // Titles should be removed
        assert!(func.parameters.get("title").is_none());
    }

    #[test]
    fn test_convert_to_openai_tool() {
        let func = FunctionDescription {
            name: "test".into(),
            description: Some("A test".into()),
            parameters: json!({}),
        };
        let tool = convert_to_openai_tool(func);
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "test");
    }

    #[test]
    fn test_set_additional_properties_false() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        set_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn test_build_parameters_schema() {
        let mut params = HashMap::new();
        params.insert(
            "name".into(),
            ParameterInfo {
                json_type: "string".into(),
                description: Some("The name".into()),
                enum_values: None,
            },
        );
        let schema = build_parameters_schema(&params, &["name".into()]);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["name"]["type"], "string");
    }

    #[test]
    fn test_tool_example_to_messages() {
        let tool_calls = vec![json!({"name": "get_weather", "args": {"location": "NYC"}})];
        let msgs = tool_example_to_messages(
            "What's the weather in NYC?",
            tool_calls,
            Some(vec!["Sunny, 72F".into()]),
            Some("The weather in NYC is sunny and 72F."),
        );
        assert_eq!(msgs.len(), 4); // human, ai w/ tools, tool result, ai response
    }

    #[test]
    fn test_convert_to_json_schema() {
        let tool = ToolDescription {
            tool_type: "function".into(),
            function: FunctionDescription {
                name: "get_weather".into(),
                description: Some("Get the weather".into()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
            },
        };
        let schema = convert_to_json_schema(&tool);
        assert_eq!(schema["title"], "get_weather");
        assert_eq!(schema["description"], "Get the weather");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["location"]["type"], "string");
    }

    #[test]
    fn test_tool_example_to_messages_no_output_no_response() {
        let tool_calls = vec![json!({"name": "search", "args": {"q": "test"}})];
        let msgs = tool_example_to_messages("Search for test", tool_calls, None, None);
        assert_eq!(msgs.len(), 3); // human, ai w/ tools, tool result (placeholder)
    }
}
