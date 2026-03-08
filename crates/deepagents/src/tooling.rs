//! Built-in tooling system for DeepAgents.
//!
//! Provides tool specifications, a registry for managing tools with handlers,
//! tool chaining for sequential execution, and usage statistics tracking.
//! This mirrors the Python `deepagents` `FilesystemMiddleware` tools
//! (read, write, edit, ls, glob, grep) along with additional utility tools.
//!
//! ## Core types
//!
//! - [`ToolSpec`] -- metadata describing a tool (name, description, parameters, category)
//! - [`ToolParam`] -- a typed parameter accepted by a tool
//! - [`ToolCategory`] -- categorisation of tools (Filesystem, Network, Memory, etc.)
//! - [`ToolResult`] -- the result of a tool invocation
//! - [`BuiltinTools`] -- factory for built-in tool specs and handlers
//! - [`ToolRegistry`] -- manages tool specs and handlers with execution support
//! - [`ToolChain`] -- executes a sequence of tools, piping output between steps
//! - [`ToolUsageStats`] -- tracks tool invocation statistics

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// ToolCategory
// ---------------------------------------------------------------------------

/// Categorisation of a tool.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCategory {
    /// File system operations (read, write, list, etc.).
    Filesystem,
    /// Network operations (HTTP, etc.).
    Network,
    /// Memory / state management.
    Memory,
    /// Computation and math.
    Computation,
    /// Search and query operations.
    Search,
    /// A user-defined category.
    Custom(String),
}

impl ToolCategory {
    /// Returns a string representation of the category.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Filesystem => "filesystem",
            Self::Network => "network",
            Self::Memory => "memory",
            Self::Computation => "computation",
            Self::Search => "search",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// ToolParam
// ---------------------------------------------------------------------------

/// Describes a parameter accepted by a [`ToolSpec`].
#[derive(Debug, Clone)]
pub struct ToolParam {
    /// Parameter name.
    pub name: String,
    /// Expected type as a string (e.g. "string", "number", "boolean").
    pub param_type: String,
    /// Whether this parameter must be provided.
    pub required: bool,
    /// Human-readable description.
    pub description: String,
    /// Default value used when the parameter is not supplied.
    pub default_value: Option<Value>,
}

impl ToolParam {
    /// Create a new required parameter.
    pub fn new(name: impl Into<String>, param_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: param_type.into(),
            required: true,
            description: String::new(),
            default_value: None,
        }
    }

    /// Mark as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set required flag.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }
}

// ---------------------------------------------------------------------------
// ToolSpec
// ---------------------------------------------------------------------------

/// Metadata describing a tool.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Parameters accepted by this tool.
    pub parameters: Vec<ToolParam>,
    /// Description of what the tool returns.
    pub returns: String,
    /// Tool category.
    pub category: ToolCategory,
}

impl ToolSpec {
    /// Create a new spec with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: Vec::new(),
            returns: String::new(),
            category: ToolCategory::Custom("uncategorized".into()),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add a parameter.
    pub fn with_param(mut self, param: ToolParam) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set the return description.
    pub fn with_returns(mut self, returns: impl Into<String>) -> Self {
        self.returns = returns.into();
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: ToolCategory) -> Self {
        self.category = category;
        self
    }

    /// Serialize this spec to a JSON value.
    pub fn to_json(&self) -> Value {
        let params: Vec<Value> = self
            .parameters
            .iter()
            .map(|p| {
                json!({
                    "name": p.name,
                    "type": p.param_type,
                    "required": p.required,
                    "description": p.description,
                    "default": p.default_value,
                })
            })
            .collect();
        json!({
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "returns": self.returns,
            "category": self.category.as_str(),
        })
    }

    /// Validate the provided arguments against parameter definitions.
    ///
    /// Checks that all required parameters are present in the JSON object.
    pub fn validate_args(&self, args: &Value) -> Result<(), String> {
        let obj = args.as_object().ok_or_else(|| "args must be a JSON object".to_string())?;
        for param in &self.parameters {
            if param.required && !obj.contains_key(&param.name) && param.default_value.is_none() {
                return Err(format!("missing required parameter '{}'", param.name));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ToolResult
// ---------------------------------------------------------------------------

/// The result of a tool invocation.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Whether the tool executed successfully.
    pub success: bool,
    /// The output value.
    pub output: Value,
    /// Error message if the execution failed.
    pub error: Option<String>,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Name of the tool that produced this result.
    pub tool_name: String,
}

impl ToolResult {
    /// Serialize to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tool_name": self.tool_name,
        })
    }

    /// Returns `true` if the result represents an error.
    pub fn is_error(&self) -> bool {
        !self.success
    }
}

// ---------------------------------------------------------------------------
// BuiltinTools
// ---------------------------------------------------------------------------

/// Type alias for a tool handler function.
pub type ToolHandler = Box<dyn Fn(Value) -> ToolResult + Send + Sync>;

/// Provides built-in tool implementations.
pub struct BuiltinTools;

impl BuiltinTools {
    /// Returns the spec and handler for a simulated file-read tool.
    pub fn read_file() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("read_file")
            .with_description("Read the contents of a file at the given path")
            .with_param(
                ToolParam::new("path", "string").with_description("Path to the file to read"),
            )
            .with_returns("File contents as a string")
            .with_category(ToolCategory::Filesystem);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if path.is_empty() {
                return ToolResult {
                    success: false,
                    output: Value::Null,
                    error: Some("path is required".into()),
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "read_file".into(),
                };
            }
            // Simulated file read
            ToolResult {
                success: true,
                output: json!(format!("Contents of file: {}", path)),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                tool_name: "read_file".into(),
            }
        });

        (spec, handler)
    }

    /// Returns the spec and handler for a simulated file-write tool.
    pub fn write_file() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("write_file")
            .with_description("Write content to a file at the given path")
            .with_param(
                ToolParam::new("path", "string").with_description("Path to the file to write"),
            )
            .with_param(
                ToolParam::new("content", "string")
                    .with_description("Content to write to the file"),
            )
            .with_returns("Confirmation message")
            .with_category(ToolCategory::Filesystem);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if path.is_empty() {
                return ToolResult {
                    success: false,
                    output: Value::Null,
                    error: Some("path is required".into()),
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "write_file".into(),
                };
            }
            // Simulated file write
            ToolResult {
                success: true,
                output: json!(format!(
                    "Wrote {} bytes to {}",
                    content.len(),
                    path
                )),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                tool_name: "write_file".into(),
            }
        });

        (spec, handler)
    }

    /// Returns the spec and handler for a simulated directory-listing tool.
    pub fn list_directory() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("list_directory")
            .with_description("List the contents of a directory")
            .with_param(
                ToolParam::new("path", "string")
                    .with_description("Path to the directory")
                    .optional()
                    .with_default(json!(".")),
            )
            .with_returns("Array of file/directory names")
            .with_category(ToolCategory::Filesystem);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            // Simulated directory listing
            ToolResult {
                success: true,
                output: json!({
                    "path": path,
                    "entries": ["file1.txt", "file2.rs", "src/", "Cargo.toml"]
                }),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                tool_name: "list_directory".into(),
            }
        });

        (spec, handler)
    }

    /// Returns the spec and handler for a text search tool (actual implementation).
    pub fn search_text() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("search_text")
            .with_description("Search for a pattern in the given text content")
            .with_param(
                ToolParam::new("content", "string")
                    .with_description("The text content to search in"),
            )
            .with_param(
                ToolParam::new("pattern", "string")
                    .with_description("The pattern to search for"),
            )
            .with_returns("Array of matching lines with line numbers")
            .with_category(ToolCategory::Search);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");

            if pattern.is_empty() {
                return ToolResult {
                    success: false,
                    output: Value::Null,
                    error: Some("pattern is required".into()),
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "search_text".into(),
                };
            }

            let matches: Vec<Value> = content
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(pattern))
                .map(|(i, line)| {
                    json!({
                        "line_number": i + 1,
                        "content": line,
                    })
                })
                .collect();

            ToolResult {
                success: true,
                output: json!({
                    "pattern": pattern,
                    "match_count": matches.len(),
                    "matches": matches,
                }),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                tool_name: "search_text".into(),
            }
        });

        (spec, handler)
    }

    /// Returns the spec and handler for a simple calculator tool.
    pub fn calculate() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("calculate")
            .with_description("Evaluate a simple math expression (+, -, *, /)")
            .with_param(
                ToolParam::new("left", "number").with_description("Left operand"),
            )
            .with_param(
                ToolParam::new("operator", "string")
                    .with_description("Operator: +, -, *, /"),
            )
            .with_param(
                ToolParam::new("right", "number").with_description("Right operand"),
            )
            .with_returns("The numeric result")
            .with_category(ToolCategory::Computation);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let left = args.get("left").and_then(|v| v.as_f64());
            let right = args.get("right").and_then(|v| v.as_f64());
            let operator = args.get("operator").and_then(|v| v.as_str()).unwrap_or("");

            let (left, right) = match (left, right) {
                (Some(l), Some(r)) => (l, r),
                _ => {
                    return ToolResult {
                        success: false,
                        output: Value::Null,
                        error: Some("left and right must be numbers".into()),
                        duration_ms: start.elapsed().as_millis() as u64,
                        tool_name: "calculate".into(),
                    };
                }
            };

            let result = match operator {
                "+" => Ok(left + right),
                "-" => Ok(left - right),
                "*" => Ok(left * right),
                "/" => {
                    if right == 0.0 {
                        Err("division by zero".to_string())
                    } else {
                        Ok(left / right)
                    }
                }
                _ => Err(format!("unsupported operator '{}'", operator)),
            };

            match result {
                Ok(val) => ToolResult {
                    success: true,
                    output: json!(val),
                    error: None,
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "calculate".into(),
                },
                Err(e) => ToolResult {
                    success: false,
                    output: Value::Null,
                    error: Some(e),
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "calculate".into(),
                },
            }
        });

        (spec, handler)
    }

    /// Returns the spec and handler for a JSON query tool using dot-notation paths.
    pub fn json_query() -> (ToolSpec, ToolHandler) {
        let spec = ToolSpec::new("json_query")
            .with_description("Extract data from a JSON value using dot-notation paths")
            .with_param(
                ToolParam::new("data", "json").with_description("The JSON data to query"),
            )
            .with_param(
                ToolParam::new("path", "string")
                    .with_description("Dot-notation path (e.g. 'user.name')"),
            )
            .with_returns("The extracted value")
            .with_category(ToolCategory::Search);

        let handler: ToolHandler = Box::new(|args: Value| {
            let start = Instant::now();
            let data = args.get("data").cloned().unwrap_or(Value::Null);
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");

            if path.is_empty() {
                return ToolResult {
                    success: false,
                    output: Value::Null,
                    error: Some("path is required".into()),
                    duration_ms: start.elapsed().as_millis() as u64,
                    tool_name: "json_query".into(),
                };
            }

            let mut current = &data;
            for segment in path.split('.') {
                // Try as object key first
                if let Some(val) = current.get(segment) {
                    current = val;
                } else if let Ok(idx) = segment.parse::<usize>() {
                    // Try as array index
                    if let Some(val) = current.get(idx) {
                        current = val;
                    } else {
                        return ToolResult {
                            success: false,
                            output: Value::Null,
                            error: Some(format!("path segment '{}' not found", segment)),
                            duration_ms: start.elapsed().as_millis() as u64,
                            tool_name: "json_query".into(),
                        };
                    }
                } else {
                    return ToolResult {
                        success: false,
                        output: Value::Null,
                        error: Some(format!("path segment '{}' not found", segment)),
                        duration_ms: start.elapsed().as_millis() as u64,
                        tool_name: "json_query".into(),
                    };
                }
            }

            ToolResult {
                success: true,
                output: current.clone(),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                tool_name: "json_query".into(),
            }
        });

        (spec, handler)
    }
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// Manages tool specs and handlers with execution support.
pub struct ToolRegistry {
    specs: HashMap<String, ToolSpec>,
    handlers: HashMap<String, ToolHandler>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            specs: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool spec with its handler.
    pub fn register(
        &mut self,
        spec: ToolSpec,
        handler: Box<dyn Fn(Value) -> ToolResult + Send + Sync>,
    ) {
        let name = spec.name.clone();
        self.specs.insert(name.clone(), spec);
        self.handlers.insert(name, handler);
    }

    /// Execute a tool by name with the given arguments.
    pub fn execute(&self, name: &str, args: Value) -> Result<ToolResult, String> {
        let spec = self
            .specs
            .get(name)
            .ok_or_else(|| format!("tool '{}' not found", name))?;
        spec.validate_args(&args)?;
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| format!("handler for '{}' not found", name))?;
        Ok(handler(args))
    }

    /// Get a reference to a tool spec by name.
    pub fn get_spec(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.get(name)
    }

    /// Return all tool specs matching a given category.
    pub fn by_category(&self, category: &ToolCategory) -> Vec<&ToolSpec> {
        self.specs
            .values()
            .filter(|s| &s.category == category)
            .collect()
    }

    /// Return references to all registered tool specs.
    pub fn all_specs(&self) -> Vec<&ToolSpec> {
        self.specs.values().collect()
    }

    /// Return the number of registered tools.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Returns `true` if the registry has no tools.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Serialize the registry to a JSON array of tool specs.
    pub fn to_json(&self) -> Value {
        let arr: Vec<Value> = self.specs.values().map(|s| s.to_json()).collect();
        Value::Array(arr)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ToolChain
// ---------------------------------------------------------------------------

/// A step in a tool chain.
#[derive(Debug, Clone)]
struct ToolChainStep {
    tool_name: String,
    args_template: Value,
}

/// Executes a sequence of tool calls, piping output from one step to the next.
pub struct ToolChain {
    steps: Vec<ToolChainStep>,
}

impl ToolChain {
    /// Create an empty tool chain.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the chain.
    pub fn add_step(&mut self, tool_name: String, args_template: Value) {
        self.steps.push(ToolChainStep {
            tool_name,
            args_template,
        });
    }

    /// Execute all steps in sequence, merging the previous step's output into
    /// the next step's args under the key `"_previous"`.
    pub fn execute(
        &self,
        registry: &ToolRegistry,
        initial_input: Value,
    ) -> Result<Vec<ToolResult>, String> {
        let mut results = Vec::new();
        let mut previous_output = initial_input;

        for step in &self.steps {
            let mut args = step.args_template.clone();
            if let Some(obj) = args.as_object_mut() {
                obj.insert("_previous".into(), previous_output.clone());
            }

            let result = registry.execute(&step.tool_name, args)?;
            if result.is_error() {
                results.push(result);
                return Ok(results);
            }
            previous_output = result.output.clone();
            results.push(result);
        }

        Ok(results)
    }

    /// Return the number of steps in the chain.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

impl Default for ToolChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ToolUsageStats
// ---------------------------------------------------------------------------

/// Summary statistics for a single tool.
#[derive(Debug, Clone)]
pub struct ToolUsageSummary {
    /// Total number of calls.
    pub call_count: u64,
    /// Number of successful calls.
    pub success_count: u64,
    /// Number of failed calls.
    pub failure_count: u64,
    /// Total duration in milliseconds across all calls.
    pub total_duration_ms: u64,
    /// Average duration per call in milliseconds.
    pub avg_duration_ms: f64,
}

/// Tracks tool invocation statistics.
pub struct ToolUsageStats {
    stats: HashMap<String, ToolUsageSummary>,
}

impl ToolUsageStats {
    /// Create an empty stats tracker.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Record a tool invocation.
    pub fn record(&mut self, tool_name: &str, duration_ms: u64, success: bool) {
        let entry = self.stats.entry(tool_name.to_string()).or_insert_with(|| {
            ToolUsageSummary {
                call_count: 0,
                success_count: 0,
                failure_count: 0,
                total_duration_ms: 0,
                avg_duration_ms: 0.0,
            }
        });
        entry.call_count += 1;
        if success {
            entry.success_count += 1;
        } else {
            entry.failure_count += 1;
        }
        entry.total_duration_ms += duration_ms;
        entry.avg_duration_ms = entry.total_duration_ms as f64 / entry.call_count as f64;
    }

    /// Get usage summary for a specific tool.
    pub fn by_tool(&self, name: &str) -> Option<&ToolUsageSummary> {
        self.stats.get(name)
    }

    /// Return the name of the most-used tool, or `None` if no calls have been recorded.
    pub fn most_used(&self) -> Option<String> {
        self.stats
            .iter()
            .max_by_key(|(_, s)| s.call_count)
            .map(|(name, _)| name.clone())
    }

    /// Return the total number of calls across all tools.
    pub fn total_calls(&self) -> u64 {
        self.stats.values().map(|s| s.call_count).sum()
    }

    /// Serialize to a JSON value.
    pub fn to_json(&self) -> Value {
        let map: serde_json::Map<String, Value> = self
            .stats
            .iter()
            .map(|(name, s)| {
                (
                    name.clone(),
                    json!({
                        "call_count": s.call_count,
                        "success_count": s.success_count,
                        "failure_count": s.failure_count,
                        "total_duration_ms": s.total_duration_ms,
                        "avg_duration_ms": s.avg_duration_ms,
                    }),
                )
            })
            .collect();
        Value::Object(map)
    }
}

impl Default for ToolUsageStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ToolCategory --

    #[test]
    fn test_category_filesystem_as_str() {
        assert_eq!(ToolCategory::Filesystem.as_str(), "filesystem");
    }

    #[test]
    fn test_category_network_as_str() {
        assert_eq!(ToolCategory::Network.as_str(), "network");
    }

    #[test]
    fn test_category_memory_as_str() {
        assert_eq!(ToolCategory::Memory.as_str(), "memory");
    }

    #[test]
    fn test_category_computation_as_str() {
        assert_eq!(ToolCategory::Computation.as_str(), "computation");
    }

    #[test]
    fn test_category_search_as_str() {
        assert_eq!(ToolCategory::Search.as_str(), "search");
    }

    #[test]
    fn test_category_custom_as_str() {
        assert_eq!(ToolCategory::Custom("my_cat".into()).as_str(), "my_cat");
    }

    #[test]
    fn test_category_display() {
        assert_eq!(format!("{}", ToolCategory::Filesystem), "filesystem");
        assert_eq!(
            format!("{}", ToolCategory::Custom("agent".into())),
            "agent"
        );
    }

    // -- ToolParam --

    #[test]
    fn test_param_new_defaults() {
        let p = ToolParam::new("path", "string");
        assert_eq!(p.name, "path");
        assert_eq!(p.param_type, "string");
        assert!(p.required);
        assert_eq!(p.description, "");
        assert!(p.default_value.is_none());
    }

    #[test]
    fn test_param_builder_chain() {
        let p = ToolParam::new("count", "number")
            .optional()
            .with_description("Number of items")
            .with_default(json!(10));
        assert!(!p.required);
        assert_eq!(p.description, "Number of items");
        assert_eq!(p.default_value, Some(json!(10)));
    }

    #[test]
    fn test_param_with_required() {
        let p = ToolParam::new("x", "string").with_required(false);
        assert!(!p.required);
        let p2 = p.with_required(true);
        assert!(p2.required);
    }

    // -- ToolSpec --

    #[test]
    fn test_spec_new_defaults() {
        let s = ToolSpec::new("test_tool");
        assert_eq!(s.name, "test_tool");
        assert_eq!(s.description, "");
        assert!(s.parameters.is_empty());
        assert_eq!(s.returns, "");
    }

    #[test]
    fn test_spec_builder_chain() {
        let s = ToolSpec::new("read")
            .with_description("Read a file")
            .with_param(ToolParam::new("path", "string"))
            .with_returns("File contents")
            .with_category(ToolCategory::Filesystem);
        assert_eq!(s.name, "read");
        assert_eq!(s.description, "Read a file");
        assert_eq!(s.parameters.len(), 1);
        assert_eq!(s.returns, "File contents");
        assert_eq!(s.category, ToolCategory::Filesystem);
    }

    #[test]
    fn test_spec_to_json() {
        let s = ToolSpec::new("calc")
            .with_description("Calculate")
            .with_param(ToolParam::new("x", "number").with_description("operand"))
            .with_returns("result")
            .with_category(ToolCategory::Computation);
        let j = s.to_json();
        assert_eq!(j["name"], "calc");
        assert_eq!(j["description"], "Calculate");
        assert_eq!(j["category"], "computation");
        assert_eq!(j["returns"], "result");
        assert_eq!(j["parameters"].as_array().unwrap().len(), 1);
        assert_eq!(j["parameters"][0]["name"], "x");
    }

    #[test]
    fn test_spec_validate_args_ok() {
        let s = ToolSpec::new("t")
            .with_param(ToolParam::new("a", "string"))
            .with_param(ToolParam::new("b", "number").optional());
        let args = json!({"a": "hello"});
        assert!(s.validate_args(&args).is_ok());
    }

    #[test]
    fn test_spec_validate_args_missing_required() {
        let s = ToolSpec::new("t").with_param(ToolParam::new("a", "string"));
        let args = json!({});
        let err = s.validate_args(&args);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("a"));
    }

    #[test]
    fn test_spec_validate_args_not_object() {
        let s = ToolSpec::new("t");
        let err = s.validate_args(&json!("not an object"));
        assert!(err.is_err());
    }

    #[test]
    fn test_spec_validate_args_with_default_ok() {
        let s = ToolSpec::new("t")
            .with_param(ToolParam::new("a", "string").with_default(json!("default_val")));
        let args = json!({});
        assert!(s.validate_args(&args).is_ok());
    }

    // -- ToolResult --

    #[test]
    fn test_result_success() {
        let r = ToolResult {
            success: true,
            output: json!("ok"),
            error: None,
            duration_ms: 5,
            tool_name: "test".into(),
        };
        assert!(!r.is_error());
        let j = r.to_json();
        assert_eq!(j["success"], true);
        assert_eq!(j["output"], "ok");
        assert!(j["error"].is_null());
    }

    #[test]
    fn test_result_error() {
        let r = ToolResult {
            success: false,
            output: Value::Null,
            error: Some("boom".into()),
            duration_ms: 1,
            tool_name: "test".into(),
        };
        assert!(r.is_error());
        let j = r.to_json();
        assert_eq!(j["error"], "boom");
    }

    // -- BuiltinTools: read_file --

    #[test]
    fn test_builtin_read_file_success() {
        let (spec, handler) = BuiltinTools::read_file();
        assert_eq!(spec.name, "read_file");
        assert_eq!(spec.category, ToolCategory::Filesystem);
        let result = handler(json!({"path": "/tmp/test.txt"}));
        assert!(result.success);
        assert!(result.output.as_str().unwrap().contains("/tmp/test.txt"));
    }

    #[test]
    fn test_builtin_read_file_empty_path() {
        let (_, handler) = BuiltinTools::read_file();
        let result = handler(json!({"path": ""}));
        assert!(result.is_error());
    }

    #[test]
    fn test_builtin_read_file_missing_path() {
        let (_, handler) = BuiltinTools::read_file();
        let result = handler(json!({}));
        assert!(result.is_error());
    }

    // -- BuiltinTools: write_file --

    #[test]
    fn test_builtin_write_file_success() {
        let (spec, handler) = BuiltinTools::write_file();
        assert_eq!(spec.name, "write_file");
        let result = handler(json!({"path": "/tmp/out.txt", "content": "hello"}));
        assert!(result.success);
        assert!(result.output.as_str().unwrap().contains("5 bytes"));
    }

    #[test]
    fn test_builtin_write_file_empty_path() {
        let (_, handler) = BuiltinTools::write_file();
        let result = handler(json!({"path": "", "content": "data"}));
        assert!(result.is_error());
    }

    // -- BuiltinTools: list_directory --

    #[test]
    fn test_builtin_list_directory_success() {
        let (spec, handler) = BuiltinTools::list_directory();
        assert_eq!(spec.name, "list_directory");
        let result = handler(json!({"path": "/tmp"}));
        assert!(result.success);
        assert!(result.output.get("entries").is_some());
    }

    #[test]
    fn test_builtin_list_directory_default_path() {
        let (_, handler) = BuiltinTools::list_directory();
        let result = handler(json!({}));
        assert!(result.success);
        assert_eq!(result.output["path"], ".");
    }

    // -- BuiltinTools: search_text --

    #[test]
    fn test_builtin_search_text_found() {
        let (spec, handler) = BuiltinTools::search_text();
        assert_eq!(spec.name, "search_text");
        assert_eq!(spec.category, ToolCategory::Search);
        let result = handler(json!({
            "content": "hello world\nfoo bar\nhello again",
            "pattern": "hello"
        }));
        assert!(result.success);
        assert_eq!(result.output["match_count"], 2);
        let matches = result.output["matches"].as_array().unwrap();
        assert_eq!(matches[0]["line_number"], 1);
        assert_eq!(matches[1]["line_number"], 3);
    }

    #[test]
    fn test_builtin_search_text_not_found() {
        let (_, handler) = BuiltinTools::search_text();
        let result = handler(json!({
            "content": "hello world",
            "pattern": "xyz"
        }));
        assert!(result.success);
        assert_eq!(result.output["match_count"], 0);
    }

    #[test]
    fn test_builtin_search_text_empty_pattern() {
        let (_, handler) = BuiltinTools::search_text();
        let result = handler(json!({"content": "hello", "pattern": ""}));
        assert!(result.is_error());
    }

    // -- BuiltinTools: calculate --

    #[test]
    fn test_builtin_calculate_add() {
        let (spec, handler) = BuiltinTools::calculate();
        assert_eq!(spec.name, "calculate");
        assert_eq!(spec.category, ToolCategory::Computation);
        let result = handler(json!({"left": 2, "operator": "+", "right": 3}));
        assert!(result.success);
        assert_eq!(result.output, json!(5.0));
    }

    #[test]
    fn test_builtin_calculate_subtract() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": 10, "operator": "-", "right": 4}));
        assert!(result.success);
        assert_eq!(result.output, json!(6.0));
    }

    #[test]
    fn test_builtin_calculate_multiply() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": 3, "operator": "*", "right": 7}));
        assert!(result.success);
        assert_eq!(result.output, json!(21.0));
    }

    #[test]
    fn test_builtin_calculate_divide() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": 15, "operator": "/", "right": 3}));
        assert!(result.success);
        assert_eq!(result.output, json!(5.0));
    }

    #[test]
    fn test_builtin_calculate_division_by_zero() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": 5, "operator": "/", "right": 0}));
        assert!(result.is_error());
        assert!(result.error.unwrap().contains("division by zero"));
    }

    #[test]
    fn test_builtin_calculate_unsupported_operator() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": 1, "operator": "%", "right": 2}));
        assert!(result.is_error());
        assert!(result.error.unwrap().contains("unsupported operator"));
    }

    #[test]
    fn test_builtin_calculate_invalid_operands() {
        let (_, handler) = BuiltinTools::calculate();
        let result = handler(json!({"left": "not_a_number", "operator": "+", "right": 1}));
        assert!(result.is_error());
    }

    // -- BuiltinTools: json_query --

    #[test]
    fn test_builtin_json_query_simple() {
        let (spec, handler) = BuiltinTools::json_query();
        assert_eq!(spec.name, "json_query");
        let result = handler(json!({
            "data": {"name": "Alice", "age": 30},
            "path": "name"
        }));
        assert!(result.success);
        assert_eq!(result.output, json!("Alice"));
    }

    #[test]
    fn test_builtin_json_query_nested() {
        let (_, handler) = BuiltinTools::json_query();
        let result = handler(json!({
            "data": {"user": {"profile": {"name": "Bob"}}},
            "path": "user.profile.name"
        }));
        assert!(result.success);
        assert_eq!(result.output, json!("Bob"));
    }

    #[test]
    fn test_builtin_json_query_array_index() {
        let (_, handler) = BuiltinTools::json_query();
        let result = handler(json!({
            "data": {"items": ["a", "b", "c"]},
            "path": "items.1"
        }));
        assert!(result.success);
        assert_eq!(result.output, json!("b"));
    }

    #[test]
    fn test_builtin_json_query_missing_path() {
        let (_, handler) = BuiltinTools::json_query();
        let result = handler(json!({"data": {"a": 1}, "path": "b"}));
        assert!(result.is_error());
        assert!(result.error.unwrap().contains("not found"));
    }

    #[test]
    fn test_builtin_json_query_empty_path() {
        let (_, handler) = BuiltinTools::json_query();
        let result = handler(json!({"data": {"a": 1}, "path": ""}));
        assert!(result.is_error());
    }

    // -- ToolRegistry --

    #[test]
    fn test_registry_new_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let reg = ToolRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_register_and_get_spec() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = BuiltinTools::calculate();
        reg.register(spec, handler);
        assert_eq!(reg.len(), 1);
        let s = reg.get_spec("calculate").unwrap();
        assert_eq!(s.name, "calculate");
    }

    #[test]
    fn test_registry_get_spec_missing() {
        let reg = ToolRegistry::new();
        assert!(reg.get_spec("nope").is_none());
    }

    #[test]
    fn test_registry_execute_success() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = BuiltinTools::calculate();
        reg.register(spec, handler);
        let result = reg
            .execute("calculate", json!({"left": 1, "operator": "+", "right": 2}))
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, json!(3.0));
    }

    #[test]
    fn test_registry_execute_missing_tool() {
        let reg = ToolRegistry::new();
        let err = reg.execute("missing", json!({}));
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_registry_execute_invalid_args() {
        let mut reg = ToolRegistry::new();
        let (spec, handler) = BuiltinTools::calculate();
        reg.register(spec, handler);
        let err = reg.execute("calculate", json!({"operator": "+", "right": 2}));
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("left"));
    }

    #[test]
    fn test_registry_by_category() {
        let mut reg = ToolRegistry::new();
        let (s1, h1) = BuiltinTools::read_file();
        let (s2, h2) = BuiltinTools::write_file();
        let (s3, h3) = BuiltinTools::calculate();
        reg.register(s1, h1);
        reg.register(s2, h2);
        reg.register(s3, h3);
        let fs_tools = reg.by_category(&ToolCategory::Filesystem);
        assert_eq!(fs_tools.len(), 2);
        let comp_tools = reg.by_category(&ToolCategory::Computation);
        assert_eq!(comp_tools.len(), 1);
    }

    #[test]
    fn test_registry_all_specs() {
        let mut reg = ToolRegistry::new();
        let (s1, h1) = BuiltinTools::read_file();
        let (s2, h2) = BuiltinTools::calculate();
        reg.register(s1, h1);
        reg.register(s2, h2);
        assert_eq!(reg.all_specs().len(), 2);
    }

    #[test]
    fn test_registry_to_json() {
        let mut reg = ToolRegistry::new();
        let (s, h) = BuiltinTools::search_text();
        reg.register(s, h);
        let j = reg.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
        assert_eq!(j[0]["name"], "search_text");
    }

    // -- ToolChain --

    #[test]
    fn test_chain_empty() {
        let chain = ToolChain::new();
        assert_eq!(chain.step_count(), 0);
    }

    #[test]
    fn test_chain_add_step() {
        let mut chain = ToolChain::new();
        chain.add_step("calculate".into(), json!({"left": 1, "operator": "+", "right": 2}));
        assert_eq!(chain.step_count(), 1);
    }

    #[test]
    fn test_chain_execute_single_step() {
        let mut reg = ToolRegistry::new();
        let (s, h) = BuiltinTools::calculate();
        reg.register(s, h);

        let mut chain = ToolChain::new();
        chain.add_step(
            "calculate".into(),
            json!({"left": 5, "operator": "*", "right": 3}),
        );
        let results = chain.execute(&reg, json!(null)).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(results[0].output, json!(15.0));
    }

    #[test]
    fn test_chain_execute_multiple_steps() {
        let mut reg = ToolRegistry::new();
        let (s1, h1) = BuiltinTools::calculate();
        let (s2, h2) = BuiltinTools::search_text();
        reg.register(s1, h1);
        reg.register(s2, h2);

        let mut chain = ToolChain::new();
        chain.add_step(
            "calculate".into(),
            json!({"left": 2, "operator": "+", "right": 3}),
        );
        chain.add_step(
            "search_text".into(),
            json!({"content": "line one\nline two\nline three", "pattern": "two"}),
        );
        let results = chain.execute(&reg, json!(null)).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(results[1].success);
        assert_eq!(results[1].output["match_count"], 1);
    }

    #[test]
    fn test_chain_stops_on_error() {
        let mut reg = ToolRegistry::new();
        let (s, h) = BuiltinTools::calculate();
        reg.register(s, h);

        let mut chain = ToolChain::new();
        chain.add_step(
            "calculate".into(),
            json!({"left": 1, "operator": "/", "right": 0}),
        );
        chain.add_step(
            "calculate".into(),
            json!({"left": 10, "operator": "+", "right": 20}),
        );
        let results = chain.execute(&reg, json!(null)).unwrap();
        // Should stop after the first error
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error());
    }

    #[test]
    fn test_chain_missing_tool_returns_error() {
        let reg = ToolRegistry::new();
        let mut chain = ToolChain::new();
        chain.add_step("nonexistent".into(), json!({}));
        let err = chain.execute(&reg, json!(null));
        assert!(err.is_err());
    }

    // -- ToolUsageStats --

    #[test]
    fn test_stats_new_empty() {
        let stats = ToolUsageStats::new();
        assert_eq!(stats.total_calls(), 0);
        assert!(stats.most_used().is_none());
    }

    #[test]
    fn test_stats_record_and_by_tool() {
        let mut stats = ToolUsageStats::new();
        stats.record("calc", 10, true);
        stats.record("calc", 20, false);
        let summary = stats.by_tool("calc").unwrap();
        assert_eq!(summary.call_count, 2);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.failure_count, 1);
        assert_eq!(summary.total_duration_ms, 30);
        assert!((summary.avg_duration_ms - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_by_tool_missing() {
        let stats = ToolUsageStats::new();
        assert!(stats.by_tool("nope").is_none());
    }

    #[test]
    fn test_stats_most_used() {
        let mut stats = ToolUsageStats::new();
        stats.record("a", 1, true);
        stats.record("b", 1, true);
        stats.record("b", 1, true);
        stats.record("c", 1, true);
        assert_eq!(stats.most_used().unwrap(), "b");
    }

    #[test]
    fn test_stats_total_calls() {
        let mut stats = ToolUsageStats::new();
        stats.record("a", 1, true);
        stats.record("b", 2, false);
        stats.record("a", 3, true);
        assert_eq!(stats.total_calls(), 3);
    }

    #[test]
    fn test_stats_to_json() {
        let mut stats = ToolUsageStats::new();
        stats.record("calc", 100, true);
        let j = stats.to_json();
        assert!(j.is_object());
        assert_eq!(j["calc"]["call_count"], 1);
        assert_eq!(j["calc"]["success_count"], 1);
        assert_eq!(j["calc"]["total_duration_ms"], 100);
    }

    #[test]
    fn test_stats_default() {
        let stats = ToolUsageStats::default();
        assert_eq!(stats.total_calls(), 0);
    }
}
