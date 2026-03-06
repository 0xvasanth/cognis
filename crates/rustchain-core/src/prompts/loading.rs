//! Load prompt templates from configuration dicts and files (JSON).
//!
//! This module mirrors the Python `langchain_core.prompts.loading` module,
//! providing utilities to deserialize prompt templates from JSON config
//! objects or files on disk.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::PromptTemplate;
use super::chat::ChatPromptTemplate;
use super::few_shot::FewShotPromptTemplate;

/// The types of prompts that can be loaded from config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptType {
    Prompt,
    FewShot,
    Chat,
}

impl PromptType {
    /// Parse a prompt type string (from the `_type` field in config).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "prompt" => Ok(Self::Prompt),
            "few_shot" => Ok(Self::FewShot),
            "chat" => Ok(Self::Chat),
            other => Err(RustChainError::Other(format!(
                "Loading {} prompt not supported",
                other
            ))),
        }
    }
}

/// A loaded prompt template — one of several concrete types.
pub enum LoadedPrompt {
    Prompt(PromptTemplate),
    FewShot(FewShotPromptTemplate),
    Chat(ChatPromptTemplate),
}

impl std::fmt::Debug for LoadedPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prompt(pt) => f
                .debug_struct("LoadedPrompt::Prompt")
                .field("template", &pt.template)
                .finish(),
            Self::FewShot(_) => f.debug_struct("LoadedPrompt::FewShot").finish(),
            Self::Chat(_) => f.debug_struct("LoadedPrompt::Chat").finish(),
        }
    }
}

/// Load a prompt template from a JSON config object.
///
/// The config should contain a `_type` field indicating the prompt type
/// (`"prompt"`, `"few_shot"`, or `"chat"`). If `_type` is missing,
/// defaults to `"prompt"`.
///
/// # Errors
///
/// Returns an error if the prompt type is unsupported or if the config
/// is missing required fields.
pub fn load_prompt_from_config(config: &Value) -> Result<LoadedPrompt> {
    let obj = config
        .as_object()
        .ok_or_else(|| RustChainError::TypeMismatch {
            expected: "Object".into(),
            got: format!("{}", config),
        })?;

    let config_type = obj
        .get("_type")
        .and_then(|v| v.as_str())
        .unwrap_or("prompt");

    let prompt_type = PromptType::parse(config_type)?;

    // Clone the object and remove the _type field for downstream use.
    let mut config_map = obj.clone();
    config_map.remove("_type");

    match prompt_type {
        PromptType::Prompt => load_prompt_config(&config_map),
        PromptType::FewShot => load_few_shot_config(&config_map),
        PromptType::Chat => load_chat_config(&config_map),
    }
}

/// Load a prompt template from a file path.
///
/// Supports `.json` files. Reads the file, parses it as JSON, and
/// delegates to [`load_prompt_from_config`].
///
/// # Errors
///
/// Returns an error if the file cannot be read, has an unsupported
/// extension, or contains invalid config.
pub fn load_prompt<P: AsRef<Path>>(path: P) -> Result<LoadedPrompt> {
    let path = path.as_ref();

    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Check extension before reading the file
    if extension != "json" {
        return Err(RustChainError::Other(format!(
            "Got unsupported file type .{}",
            extension
        )));
    }

    let contents = fs::read_to_string(path).map_err(|e| {
        RustChainError::Other(format!("Failed to read file {}: {}", path.display(), e))
    })?;

    let config: Value = serde_json::from_str(&contents)?;

    load_prompt_from_config(&config)
}

/// Load a template string from a `_path` field if present in the config.
///
/// If `{var_name}_path` exists in the config, reads the `.txt` file at
/// that path and sets `{var_name}` to its contents. It is an error if
/// both `{var_name}_path` and `{var_name}` are present.
fn load_template(var_name: &str, config: &mut serde_json::Map<String, Value>) -> Result<()> {
    let path_key = format!("{}_path", var_name);

    if let Some(path_val) = config.remove(&path_key) {
        if config.contains_key(var_name) {
            return Err(RustChainError::Other(format!(
                "Both `{}_path` and `{}` cannot be provided.",
                var_name, var_name
            )));
        }

        let path_str = path_val
            .as_str()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "String".into(),
                got: format!("{}", path_val),
            })?;

        let path = Path::new(path_str);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "txt" {
            return Err(RustChainError::Other(format!(
                "Unsupported template file extension .{}, expected .txt",
                ext
            )));
        }

        let template = fs::read_to_string(path).map_err(|e| {
            RustChainError::Other(format!(
                "Failed to read template file {}: {}",
                path.display(),
                e
            ))
        })?;

        config.insert(var_name.to_string(), Value::String(template));
    }

    Ok(())
}

/// Load examples from a file path or pass through inline examples.
///
/// If `examples` is a string, it is treated as a file path to a JSON
/// file containing an array of example objects. If it is already an
/// array, it is returned as-is.
fn load_examples(config: &mut serde_json::Map<String, Value>) -> Result<()> {
    if let Some(examples) = config.get("examples") {
        match examples {
            Value::Array(_) => {
                // Already inline — nothing to do.
            }
            Value::String(path_str) => {
                let path = Path::new(path_str);
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

                let contents = fs::read_to_string(path).map_err(|e| {
                    RustChainError::Other(format!(
                        "Failed to read examples file {}: {}",
                        path.display(),
                        e
                    ))
                })?;

                let examples_value: Value = match ext {
                    "json" => serde_json::from_str(&contents)?,
                    _ => {
                        return Err(RustChainError::Other(
                            "Invalid file format. Only json format is supported for examples."
                                .into(),
                        ));
                    }
                };

                config.insert("examples".to_string(), examples_value);
            }
            _ => {
                return Err(RustChainError::Other(
                    "Invalid examples format. Only array or string (file path) are supported."
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

/// Load a basic `PromptTemplate` from config.
fn load_prompt_config(config: &serde_json::Map<String, Value>) -> Result<LoadedPrompt> {
    let mut config = config.clone();
    load_template("template", &mut config)?;

    // Check for jinja2 format (disallowed).
    if let Some(fmt) = config.get("template_format").and_then(|v| v.as_str()) {
        if fmt == "jinja2" {
            return Err(RustChainError::Other(
                "Loading templates with 'jinja2' format is not supported as it can lead \
                 to arbitrary code execution. Please use the 'f-string' template format."
                    .into(),
            ));
        }
    }

    let template = config
        .get("template")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RustChainError::Other("Missing 'template' in prompt config".into()))?
        .to_string();

    let input_variables = extract_string_array(&config, "input_variables");

    let pt = if input_variables.is_empty() {
        PromptTemplate::from_template(template)
    } else {
        PromptTemplate::new(template, input_variables, HashMap::new())
    };

    Ok(LoadedPrompt::Prompt(pt))
}

/// Load a `FewShotPromptTemplate` from config.
fn load_few_shot_config(config: &serde_json::Map<String, Value>) -> Result<LoadedPrompt> {
    let mut config = config.clone();
    load_template("suffix", &mut config)?;
    load_template("prefix", &mut config)?;
    load_examples(&mut config)?;

    // Load the example prompt (inline config or file path).
    let example_prompt = if let Some(path_val) = config.remove("example_prompt_path") {
        if config.contains_key("example_prompt") {
            return Err(RustChainError::Other(
                "Only one of example_prompt and example_prompt_path should be specified.".into(),
            ));
        }
        let path_str = path_val
            .as_str()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "String".into(),
                got: format!("{}", path_val),
            })?;
        match load_prompt(path_str)? {
            LoadedPrompt::Prompt(pt) => pt,
            _ => {
                return Err(RustChainError::Other(
                    "example_prompt must be a basic PromptTemplate".into(),
                ));
            }
        }
    } else if let Some(ep_config) = config.remove("example_prompt") {
        match load_prompt_from_config(&ep_config)? {
            LoadedPrompt::Prompt(pt) => pt,
            _ => {
                return Err(RustChainError::Other(
                    "example_prompt must be a basic PromptTemplate".into(),
                ));
            }
        }
    } else {
        return Err(RustChainError::Other(
            "Missing 'example_prompt' or 'example_prompt_path' in few_shot config".into(),
        ));
    };

    let suffix = config
        .get("suffix")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let prefix = config
        .get("prefix")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let examples = config
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_object().map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<HashMap<String, Value>>()
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let separator = config
        .get("example_separator")
        .and_then(|v| v.as_str())
        .unwrap_or("\n\n")
        .to_string();

    let fs_template = FewShotPromptTemplate::new(examples, example_prompt, suffix)
        .with_prefix(prefix)
        .with_separator(separator);

    Ok(LoadedPrompt::FewShot(fs_template))
}

/// Load a `ChatPromptTemplate` from config.
fn load_chat_config(config: &serde_json::Map<String, Value>) -> Result<LoadedPrompt> {
    let messages = config
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RustChainError::Other("Missing 'messages' in chat config".into()))?;

    // Extract template from the first message if available.
    let template = messages
        .first()
        .and_then(|m| m.get("prompt"))
        .and_then(|p| p.get("template"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| RustChainError::Other("Can't load chat prompt without template".into()))?;

    let chat_template = ChatPromptTemplate::from_messages(vec![("human", template)])?;

    Ok(LoadedPrompt::Chat(chat_template))
}

/// Extract an array of strings from a config map field.
fn extract_string_array(config: &serde_json::Map<String, Value>, key: &str) -> Vec<String> {
    config
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_prompt_from_config_basic() {
        let config = serde_json::json!({
            "_type": "prompt",
            "template": "Hello, {name}!",
            "input_variables": ["name"]
        });

        let loaded = load_prompt_from_config(&config).unwrap();
        match loaded {
            LoadedPrompt::Prompt(pt) => {
                assert_eq!(pt.template, "Hello, {name}!");
                assert_eq!(pt.input_variables, vec!["name"]);
            }
            _ => panic!("Expected LoadedPrompt::Prompt"),
        }
    }

    #[test]
    fn test_load_prompt_from_config_default_type() {
        let config = serde_json::json!({
            "template": "Hello, {name}!",
            "input_variables": ["name"]
        });

        let loaded = load_prompt_from_config(&config).unwrap();
        match loaded {
            LoadedPrompt::Prompt(pt) => {
                assert_eq!(pt.template, "Hello, {name}!");
            }
            _ => panic!("Expected LoadedPrompt::Prompt"),
        }
    }

    #[test]
    fn test_load_prompt_from_config_auto_extract_variables() {
        let config = serde_json::json!({
            "template": "Hello, {name}! Welcome to {place}."
        });

        let loaded = load_prompt_from_config(&config).unwrap();
        match loaded {
            LoadedPrompt::Prompt(pt) => {
                assert!(pt.input_variables.contains(&"name".to_string()));
                assert!(pt.input_variables.contains(&"place".to_string()));
            }
            _ => panic!("Expected LoadedPrompt::Prompt"),
        }
    }

    #[test]
    fn test_load_prompt_from_config_unsupported_type() {
        let config = serde_json::json!({
            "_type": "unknown_type",
            "template": "Hello"
        });

        let result = load_prompt_from_config(&config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("not supported"));
    }

    #[test]
    fn test_load_prompt_from_config_jinja2_rejected() {
        let config = serde_json::json!({
            "_type": "prompt",
            "template": "Hello {{ name }}",
            "template_format": "jinja2"
        });

        let result = load_prompt_from_config(&config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("jinja2"));
    }

    #[test]
    fn test_load_prompt_from_config_not_object() {
        let config = serde_json::json!("not an object");

        let result = load_prompt_from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_prompt_from_json_file() {
        let dir = std::env::temp_dir().join("rustchain_test_loading");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("test_prompt.json");

        let config = serde_json::json!({
            "_type": "prompt",
            "template": "Tell me about {topic}.",
            "input_variables": ["topic"]
        });

        let mut f = fs::File::create(&file_path).unwrap();
        f.write_all(serde_json::to_string_pretty(&config).unwrap().as_bytes())
            .unwrap();

        let loaded = load_prompt(&file_path).unwrap();
        match loaded {
            LoadedPrompt::Prompt(pt) => {
                assert_eq!(pt.template, "Tell me about {topic}.");
                assert_eq!(pt.input_variables, vec!["topic"]);
            }
            _ => panic!("Expected LoadedPrompt::Prompt"),
        }

        let _ = fs::remove_file(&file_path);
    }

    #[test]
    fn test_load_prompt_unsupported_extension() {
        let result = load_prompt("/tmp/test.xml");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unsupported file type"));
    }

    #[test]
    fn test_load_few_shot_from_config() {
        let config = serde_json::json!({
            "_type": "few_shot",
            "examples": [
                {"input": "hello", "output": "world"},
                {"input": "foo", "output": "bar"}
            ],
            "example_prompt": {
                "_type": "prompt",
                "template": "Input: {input}\nOutput: {output}",
                "input_variables": ["input", "output"]
            },
            "suffix": "Input: {query}\nOutput:",
            "prefix": "Translate the following:"
        });

        let loaded = load_prompt_from_config(&config).unwrap();
        match loaded {
            LoadedPrompt::FewShot(fs) => {
                assert_eq!(fs.prefix, "Translate the following:");
                assert!(fs.examples.is_some());
                assert_eq!(fs.examples.as_ref().unwrap().len(), 2);
            }
            _ => panic!("Expected LoadedPrompt::FewShot"),
        }
    }

    #[test]
    fn test_load_chat_from_config() {
        let config = serde_json::json!({
            "_type": "chat",
            "messages": [
                {
                    "prompt": {
                        "template": "Tell me a joke about {topic}."
                    }
                }
            ],
            "input_variables": ["topic"]
        });

        let loaded = load_prompt_from_config(&config).unwrap();
        match loaded {
            LoadedPrompt::Chat(_) => {}
            _ => panic!("Expected LoadedPrompt::Chat"),
        }
    }

    #[test]
    fn test_load_examples_inline() {
        let mut config = serde_json::Map::new();
        config.insert("examples".to_string(), serde_json::json!([{"a": "b"}]));

        load_examples(&mut config).unwrap();
        assert!(config.get("examples").unwrap().is_array());
    }

    #[test]
    fn test_load_examples_from_file() {
        let dir = std::env::temp_dir().join("rustchain_test_loading_examples");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("examples.json");

        let examples = serde_json::json!([
            {"input": "a", "output": "b"},
            {"input": "c", "output": "d"}
        ]);

        let mut f = fs::File::create(&file_path).unwrap();
        f.write_all(serde_json::to_string(&examples).unwrap().as_bytes())
            .unwrap();

        let mut config = serde_json::Map::new();
        config.insert(
            "examples".to_string(),
            Value::String(file_path.to_string_lossy().into_owned()),
        );

        load_examples(&mut config).unwrap();
        let loaded = config.get("examples").unwrap().as_array().unwrap();
        assert_eq!(loaded.len(), 2);

        let _ = fs::remove_file(&file_path);
    }

    #[test]
    fn test_load_template_from_file() {
        let dir = std::env::temp_dir().join("rustchain_test_loading_template");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("template.txt");

        let mut f = fs::File::create(&file_path).unwrap();
        f.write_all(b"Hello, {name}! How are you?").unwrap();

        let mut config = serde_json::Map::new();
        config.insert(
            "template_path".to_string(),
            Value::String(file_path.to_string_lossy().into_owned()),
        );

        load_template("template", &mut config).unwrap();
        assert_eq!(
            config.get("template").unwrap().as_str().unwrap(),
            "Hello, {name}! How are you?"
        );

        let _ = fs::remove_file(&file_path);
    }

    #[test]
    fn test_load_template_conflict_error() {
        let mut config = serde_json::Map::new();
        config.insert("template".to_string(), Value::String("inline".into()));
        config.insert(
            "template_path".to_string(),
            Value::String("/tmp/something.txt".into()),
        );

        let result = load_template("template", &mut config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("cannot be provided"));
    }

    #[test]
    fn test_prompt_type_parse() {
        assert_eq!(PromptType::parse("prompt").unwrap(), PromptType::Prompt);
        assert_eq!(PromptType::parse("few_shot").unwrap(), PromptType::FewShot);
        assert_eq!(PromptType::parse("chat").unwrap(), PromptType::Chat);
        assert!(PromptType::parse("invalid").is_err());
    }
}
