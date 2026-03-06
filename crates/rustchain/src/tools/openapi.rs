//! OpenAPI tool auto-generation from API specifications.
//!
//! This module parses simplified OpenAPI 3.0 JSON specs and automatically
//! generates `BaseTool` implementations for each operation. Each operation
//! becomes a standalone tool that constructs and executes HTTP requests
//! based on the tool input parameters.
//!
//! # Example
//!
//! ```rust,ignore
//! use serde_json::json;
//! use rustchain::tools::openapi::OpenAPIToolkit;
//!
//! let spec_json = json!({
//!     "openapi": "3.0.0",
//!     "info": { "title": "Pet Store", "version": "1.0.0" },
//!     "servers": [{ "url": "https://api.example.com" }],
//!     "paths": {
//!         "/pets": {
//!             "get": {
//!                 "operationId": "listPets",
//!                 "summary": "List all pets",
//!                 "parameters": [{
//!                     "name": "limit",
//!                     "in": "query",
//!                     "required": false,
//!                     "schema": { "type": "integer" }
//!                 }]
//!             }
//!         }
//!     }
//! });
//!
//! let toolkit = OpenAPIToolkit::from_json(&spec_json).unwrap();
//! let tools = toolkit.get_tools();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Information about a single parameter in an OpenAPI operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name.
    pub name: String,
    /// Location: "query", "path", "header", or "cookie".
    pub location: String,
    /// Whether the parameter is required.
    pub required: bool,
    /// JSON Schema describing the parameter type.
    pub schema: Option<Value>,
    /// Human-readable description.
    pub description: Option<String>,
}

/// Describes a single API operation extracted from an OpenAPI spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationInfo {
    /// Unique operation identifier (from `operationId`).
    pub operation_id: String,
    /// HTTP method (GET, POST, PUT, DELETE, PATCH).
    pub method: String,
    /// URL path template (e.g. `/pets/{petId}`).
    pub path: String,
    /// Short summary of the operation.
    pub summary: String,
    /// Longer description of the operation.
    pub description: Option<String>,
    /// Parameters (path, query, header).
    pub parameters: Vec<ParameterInfo>,
    /// Request body schema, if any.
    pub request_body: Option<Value>,
    /// Tags associated with the operation.
    pub tags: Vec<String>,
}

/// A parsed OpenAPI 3.0 specification.
#[derive(Debug, Clone)]
pub struct OpenAPISpec {
    /// Title of the API.
    pub title: String,
    /// Version of the API.
    pub version: String,
    /// Base URLs extracted from the `servers` array.
    pub servers: Vec<String>,
    /// All operations extracted from the spec.
    pub operations: Vec<OperationInfo>,
}

impl OpenAPISpec {
    /// Parse an OpenAPI 3.0 spec from a `serde_json::Value`.
    pub fn from_json(spec: &Value) -> Result<Self> {
        let info = spec
            .get("info")
            .ok_or_else(|| RustChainError::ToolValidationError("Missing 'info' in spec".into()))?;

        let title = info
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled API")
            .to_string();

        let version = info
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .to_string();

        let servers = spec
            .get("servers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| {
                        s.get("url").and_then(|u| u.as_str()).map(|u| {
                            // Strip trailing slash for consistency.
                            u.trim_end_matches('/').to_string()
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let paths = spec
            .get("paths")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                RustChainError::ToolValidationError("Missing or invalid 'paths' in spec".into())
            })?;

        let mut operations = Vec::new();

        for (path, path_item) in paths {
            let path_item = match path_item.as_object() {
                Some(obj) => obj,
                None => continue,
            };

            // Shared parameters at path level.
            let shared_params = path_item
                .get("parameters")
                .and_then(|v| v.as_array())
                .map(|arr| Self::parse_parameters(arr))
                .unwrap_or_default();

            for method in &["get", "post", "put", "delete", "patch"] {
                if let Some(op) = path_item.get(*method) {
                    let operation_id = op
                        .get("operationId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    if operation_id.is_empty() {
                        // Generate a fallback ID from method + path.
                        let fallback = format!(
                            "{}_{}",
                            method,
                            path.trim_start_matches('/')
                                .replace('/', "_")
                                .replace(['{', '}'], "")
                        );
                        let mut info = Self::parse_operation(op, &fallback, method, path);
                        // Merge shared parameters (operation-level takes priority).
                        Self::merge_params(&mut info.parameters, &shared_params);
                        operations.push(info);
                    } else {
                        let mut info = Self::parse_operation(op, &operation_id, method, path);
                        Self::merge_params(&mut info.parameters, &shared_params);
                        operations.push(info);
                    }
                }
            }
        }

        Ok(Self {
            title,
            version,
            servers,
            operations,
        })
    }

    /// Return all operations in the spec.
    pub fn list_operations(&self) -> Vec<OperationInfo> {
        self.operations.clone()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn parse_operation(op: &Value, operation_id: &str, method: &str, path: &str) -> OperationInfo {
        let summary = op
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let description = op
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let parameters = op
            .get("parameters")
            .and_then(|v| v.as_array())
            .map(|arr| Self::parse_parameters(arr))
            .unwrap_or_default();

        let request_body = op.get("requestBody").cloned();

        let tags = op
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        OperationInfo {
            operation_id: operation_id.to_string(),
            method: method.to_uppercase(),
            path: path.to_string(),
            summary,
            description,
            parameters,
            request_body,
            tags,
        }
    }

    fn parse_parameters(params: &[Value]) -> Vec<ParameterInfo> {
        params
            .iter()
            .filter_map(|p| {
                let name = p.get("name")?.as_str()?.to_string();
                let location = p
                    .get("in")
                    .and_then(|v| v.as_str())
                    .unwrap_or("query")
                    .to_string();
                let required = p.get("required").and_then(|v| v.as_bool()).unwrap_or(false);
                let schema = p.get("schema").cloned();
                let description = p
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                Some(ParameterInfo {
                    name,
                    location,
                    required,
                    schema,
                    description,
                })
            })
            .collect()
    }

    /// Merge shared (path-level) parameters into operation parameters.
    /// Operation-level parameters with the same name take priority.
    fn merge_params(op_params: &mut Vec<ParameterInfo>, shared: &[ParameterInfo]) {
        for shared_param in shared {
            let exists = op_params
                .iter()
                .any(|p| p.name == shared_param.name && p.location == shared_param.location);
            if !exists {
                op_params.push(shared_param.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAPITool — a BaseTool generated from a single operation
// ---------------------------------------------------------------------------

/// Trait for executing HTTP requests, enabling testability without real network calls.
#[async_trait]
pub trait HttpExecutor: Send + Sync {
    /// Execute an HTTP request and return the response body as a `Value`.
    async fn execute(
        &self,
        method: &str,
        url: &str,
        headers: &HashMap<String, String>,
        body: Option<&Value>,
    ) -> Result<Value>;
}

/// Default HTTP executor that uses `reqwest` to make real HTTP calls.
/// Only available when one of the provider features is enabled.
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
pub struct ReqwestExecutor {
    client: reqwest::Client,
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
impl Default for ReqwestExecutor {
    fn default() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
#[async_trait]
impl HttpExecutor for ReqwestExecutor {
    async fn execute(
        &self,
        method: &str,
        url: &str,
        headers: &HashMap<String, String>,
        body: Option<&Value>,
    ) -> Result<Value> {
        let mut req = match method {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "DELETE" => self.client.delete(url),
            "PATCH" => self.client.patch(url),
            _ => {
                return Err(RustChainError::ToolException(format!(
                    "Unsupported HTTP method: {}",
                    method
                )));
            }
        };

        for (key, value) in headers {
            req = req.header(key, value);
        }

        if let Some(body_val) = body {
            req = req.header("Content-Type", "application/json");
            req = req.json(body_val);
        }

        let response = req
            .send()
            .await
            .map_err(|e| RustChainError::ToolException(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        let response_body = response.text().await.map_err(|e| {
            RustChainError::ToolException(format!("Failed to read response body: {}", e))
        })?;

        if status >= 400 {
            return Err(RustChainError::HttpError {
                status,
                body: response_body,
            });
        }

        let value: Value =
            serde_json::from_str(&response_body).unwrap_or(Value::String(response_body));

        Ok(value)
    }
}

/// A no-op executor that returns a JSON description of the request instead of
/// making a real HTTP call. Useful for testing and dry-run scenarios.
pub struct DryRunExecutor;

#[async_trait]
impl HttpExecutor for DryRunExecutor {
    async fn execute(
        &self,
        method: &str,
        url: &str,
        headers: &HashMap<String, String>,
        body: Option<&Value>,
    ) -> Result<Value> {
        Ok(json!({
            "url": url,
            "method": method,
            "headers": headers,
            "body": body,
        }))
    }
}

/// A tool auto-generated from a single OpenAPI operation.
///
/// When invoked, it constructs the appropriate HTTP request from the tool
/// input and executes it via the configured `HttpExecutor`.
pub struct OpenAPITool {
    operation: OperationInfo,
    base_url: String,
    headers: HashMap<String, String>,
    executor: Arc<dyn HttpExecutor>,
}

impl OpenAPITool {
    /// Create a new tool from an operation, base URL, default headers, and executor.
    pub fn new(
        operation: OperationInfo,
        base_url: String,
        headers: HashMap<String, String>,
        executor: Arc<dyn HttpExecutor>,
    ) -> Self {
        Self {
            operation,
            base_url,
            headers,
            executor,
        }
    }

    /// Build the full URL with path parameter substitution.
    pub fn build_url(&self, args: &HashMap<String, Value>) -> String {
        let mut path = self.operation.path.clone();

        // Substitute path parameters.
        for param in &self.operation.parameters {
            if param.location == "path" {
                if let Some(val) = args.get(&param.name) {
                    let replacement = match val {
                        Value::String(s) => s.clone(),
                        other => other.to_string().trim_matches('"').to_string(),
                    };
                    path = path.replace(&format!("{{{}}}", param.name), &replacement);
                }
            }
        }

        // Build query string.
        let query_params: Vec<String> = self
            .operation
            .parameters
            .iter()
            .filter(|p| p.location == "query")
            .filter_map(|p| {
                args.get(&p.name).map(|val| {
                    let val_str = match val {
                        Value::String(s) => s.clone(),
                        other => other.to_string().trim_matches('"').to_string(),
                    };
                    format!("{}={}", p.name, val_str)
                })
            })
            .collect();

        let base = format!("{}{}", self.base_url, path);
        if query_params.is_empty() {
            base
        } else {
            format!("{}?{}", base, query_params.join("&"))
        }
    }

    /// Build the args schema (JSON Schema) from the operation parameters.
    fn build_args_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.operation.parameters {
            let mut prop = param
                .schema
                .clone()
                .unwrap_or_else(|| json!({"type": "string"}));
            if let Some(desc) = &param.description {
                if let Some(obj) = prop.as_object_mut() {
                    obj.insert("description".to_string(), Value::String(desc.clone()));
                }
            }
            properties.insert(param.name.clone(), prop);
            if param.required {
                required.push(Value::String(param.name.clone()));
            }
        }

        // If there is a request body, add a "body" parameter.
        if self.operation.request_body.is_some() {
            let body_schema = self
                .operation
                .request_body
                .as_ref()
                .and_then(|rb| {
                    rb.get("content")
                        .and_then(|c| c.get("application/json"))
                        .and_then(|j| j.get("schema"))
                        .cloned()
                })
                .unwrap_or_else(|| json!({"type": "object"}));
            properties.insert("body".to_string(), body_schema);
            required.push(Value::String("body".to_string()));
        }

        json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }

    /// Extract header values from tool input arguments.
    fn extract_headers(&self, args: &HashMap<String, Value>) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        for param in &self.operation.parameters {
            if param.location == "header" {
                if let Some(val) = args.get(&param.name) {
                    let val_str = match val {
                        Value::String(s) => s.clone(),
                        other => other.to_string().trim_matches('"').to_string(),
                    };
                    headers.insert(param.name.clone(), val_str);
                }
            }
        }
        headers
    }

    /// Extract the request body from tool input arguments.
    fn extract_body(&self, args: &HashMap<String, Value>) -> Option<Value> {
        if self.operation.request_body.is_some() {
            args.get("body").cloned()
        } else {
            None
        }
    }
}

impl std::fmt::Debug for OpenAPITool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAPITool")
            .field("operation_id", &self.operation.operation_id)
            .field("method", &self.operation.method)
            .field("path", &self.operation.path)
            .finish()
    }
}

#[async_trait]
impl BaseTool for OpenAPITool {
    fn name(&self) -> &str {
        &self.operation.operation_id
    }

    fn description(&self) -> &str {
        if !self.operation.summary.is_empty() {
            &self.operation.summary
        } else {
            self.operation
                .description
                .as_deref()
                .unwrap_or("OpenAPI tool")
        }
    }

    fn args_schema(&self) -> Option<Value> {
        Some(self.build_args_schema())
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args: HashMap<String, Value> = match input {
            ToolInput::Structured(map) => map,
            ToolInput::ToolCall(tc) => tc.args,
            ToolInput::Text(s) => {
                // Try to parse as JSON object; otherwise treat as empty.
                serde_json::from_str(&s).unwrap_or_default()
            }
        };

        let url = self.build_url(&args);
        let headers = self.extract_headers(&args);
        let body = self.extract_body(&args);

        let value = self
            .executor
            .execute(&self.operation.method, &url, &headers, body.as_ref())
            .await?;

        Ok(ToolOutput::Content(value))
    }
}

// ---------------------------------------------------------------------------
// Tool generation
// ---------------------------------------------------------------------------

/// Generate a `BaseTool` for each operation in the spec.
///
/// Optionally filter by tags or specific operation IDs. The `executor`
/// determines how HTTP requests are dispatched (use `DryRunExecutor` for
/// testing or `ReqwestExecutor` for real calls).
pub fn generate_tools(
    spec: &OpenAPISpec,
    base_url: Option<&str>,
    headers: Option<HashMap<String, String>>,
    filter_tags: Option<&[String]>,
    filter_operation_ids: Option<&[String]>,
    executor: Arc<dyn HttpExecutor>,
) -> Vec<Arc<dyn BaseTool>> {
    let url = base_url
        .map(|s| s.trim_end_matches('/').to_string())
        .or_else(|| spec.servers.first().cloned())
        .unwrap_or_default();

    let hdrs = headers.unwrap_or_default();

    spec.operations
        .iter()
        .filter(|op| {
            // Tag filter.
            if let Some(tags) = filter_tags {
                if !op.tags.iter().any(|t| tags.contains(t)) {
                    return false;
                }
            }
            // Operation ID filter.
            if let Some(ids) = filter_operation_ids {
                if !ids.contains(&op.operation_id) {
                    return false;
                }
            }
            true
        })
        .map(|op| {
            Arc::new(OpenAPITool::new(
                op.clone(),
                url.clone(),
                hdrs.clone(),
                executor.clone(),
            )) as Arc<dyn BaseTool>
        })
        .collect()
}

// ---------------------------------------------------------------------------
// OpenAPIToolkit — builder pattern
// ---------------------------------------------------------------------------

/// A toolkit that wraps an OpenAPI spec and produces `BaseTool` instances.
///
/// Use the builder pattern to configure base URL overrides, authentication
/// headers, and operation filters.
pub struct OpenAPIToolkit {
    spec: OpenAPISpec,
    base_url: Option<String>,
    headers: HashMap<String, String>,
    filter_tags: Option<Vec<String>>,
    filter_operation_ids: Option<Vec<String>>,
    executor: Arc<dyn HttpExecutor>,
}

impl OpenAPIToolkit {
    /// Create a toolkit from a parsed `OpenAPISpec`.
    ///
    /// Uses `DryRunExecutor` by default. Call `with_executor` to provide
    /// a real HTTP executor (e.g. `ReqwestExecutor`).
    pub fn new(spec: OpenAPISpec) -> Self {
        Self {
            spec,
            base_url: None,
            headers: HashMap::new(),
            filter_tags: None,
            filter_operation_ids: None,
            executor: Arc::new(DryRunExecutor),
        }
    }

    /// Create a toolkit directly from an OpenAPI JSON spec.
    pub fn from_json(spec: &Value) -> Result<Self> {
        let parsed = OpenAPISpec::from_json(spec)?;
        Ok(Self::new(parsed))
    }

    /// Set a custom HTTP executor.
    pub fn with_executor(mut self, executor: Arc<dyn HttpExecutor>) -> Self {
        self.executor = executor;
        self
    }

    /// Override the base URL for all requests.
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = Some(url.trim_end_matches('/').to_string());
        self
    }

    /// Add a single default header applied to all requests.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Add multiple default headers applied to all requests.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Set an Authorization Bearer token header.
    pub fn with_bearer_token(self, token: &str) -> Self {
        self.with_header("Authorization", &format!("Bearer {}", token))
    }

    /// Only include operations matching the given tags.
    pub fn with_tag_filter(mut self, tags: Vec<String>) -> Self {
        self.filter_tags = Some(tags);
        self
    }

    /// Only include operations with the given operation IDs.
    pub fn with_operation_filter(mut self, ids: Vec<String>) -> Self {
        self.filter_operation_ids = Some(ids);
        self
    }

    /// Generate all tools from the spec, applying configured filters.
    pub fn get_tools(&self) -> Vec<Arc<dyn BaseTool>> {
        generate_tools(
            &self.spec,
            self.base_url.as_deref(),
            Some(self.headers.clone()),
            self.filter_tags.as_deref(),
            self.filter_operation_ids.as_deref(),
            self.executor.clone(),
        )
    }

    /// Return a reference to the underlying parsed spec.
    pub fn spec(&self) -> &OpenAPISpec {
        &self.spec
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: minimal valid OpenAPI spec with a single GET endpoint.
    fn pet_store_spec() -> Value {
        json!({
            "openapi": "3.0.0",
            "info": { "title": "Pet Store", "version": "1.0.0" },
            "servers": [{ "url": "https://api.petstore.com/v1" }],
            "paths": {
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "summary": "List all pets",
                        "tags": ["pets"],
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": false,
                                "description": "Maximum number of pets to return",
                                "schema": { "type": "integer" }
                            },
                            {
                                "name": "status",
                                "in": "query",
                                "required": false,
                                "schema": { "type": "string" }
                            }
                        ]
                    },
                    "post": {
                        "operationId": "createPet",
                        "summary": "Create a pet",
                        "tags": ["pets"],
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": { "type": "string" },
                                            "tag": { "type": "string" }
                                        },
                                        "required": ["name"]
                                    }
                                }
                            }
                        }
                    }
                },
                "/pets/{petId}": {
                    "get": {
                        "operationId": "getPet",
                        "summary": "Get a pet by ID",
                        "tags": ["pets"],
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": true,
                                "schema": { "type": "string" }
                            }
                        ]
                    },
                    "put": {
                        "operationId": "updatePet",
                        "summary": "Update a pet",
                        "tags": ["pets"],
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": true,
                                "schema": { "type": "string" }
                            }
                        ],
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": { "type": "string" },
                                            "tag": { "type": "string" }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "delete": {
                        "operationId": "deletePet",
                        "summary": "Delete a pet",
                        "tags": ["pets"],
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": true,
                                "schema": { "type": "string" }
                            }
                        ]
                    }
                },
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List all users",
                        "tags": ["users"]
                    }
                }
            }
        })
    }

    // ── Test 1: Parse simple OpenAPI spec ──────────────────────────────

    #[test]
    fn test_parse_simple_spec() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        assert_eq!(spec.title, "Pet Store");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.servers, vec!["https://api.petstore.com/v1"]);
        assert!(!spec.operations.is_empty());
    }

    // ── Test 2: List operations from spec ─────────────────────────────

    #[test]
    fn test_list_operations() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let ops = spec.list_operations();
        let ids: Vec<&str> = ops.iter().map(|o| o.operation_id.as_str()).collect();
        assert!(ids.contains(&"listPets"));
        assert!(ids.contains(&"createPet"));
        assert!(ids.contains(&"getPet"));
        assert!(ids.contains(&"updatePet"));
        assert!(ids.contains(&"deletePet"));
        assert!(ids.contains(&"listUsers"));
        assert_eq!(ops.len(), 6);
    }

    // ── Test 3: Generate tool from GET endpoint ───────────────────────

    #[tokio::test]
    async fn test_generate_tool_from_get_endpoint() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let list_pets_tool = tools.iter().find(|t| t.name() == "listPets").unwrap();
        assert_eq!(list_pets_tool.description(), "List all pets");

        // Invoke with query parameters.
        let mut args = HashMap::new();
        args.insert("limit".to_string(), json!(10));

        let result = list_pets_tool
            ._run(ToolInput::Structured(args))
            .await
            .unwrap();

        match result {
            ToolOutput::Content(v) => {
                assert_eq!(v["method"], "GET");
                assert!(v["url"].as_str().unwrap().contains("limit=10"));
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 4: Generate tool from POST endpoint with body ────────────

    #[tokio::test]
    async fn test_generate_tool_from_post_endpoint_with_body() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let create_pet_tool = tools.iter().find(|t| t.name() == "createPet").unwrap();

        let mut args = HashMap::new();
        args.insert("body".to_string(), json!({"name": "Fido", "tag": "dog"}));

        let result = create_pet_tool
            ._run(ToolInput::Structured(args))
            .await
            .unwrap();

        match result {
            ToolOutput::Content(v) => {
                assert_eq!(v["method"], "POST");
                assert_eq!(v["body"]["name"], "Fido");
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 5: Tool name from operationId ────────────────────────────

    #[test]
    fn test_tool_name_from_operation_id() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"listPets"));
        assert!(names.contains(&"createPet"));
        assert!(names.contains(&"getPet"));
        assert!(names.contains(&"updatePet"));
        assert!(names.contains(&"deletePet"));
    }

    // ── Test 6: Tool description from summary ─────────────────────────

    #[test]
    fn test_tool_description_from_summary() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let tool = tools.iter().find(|t| t.name() == "getPet").unwrap();
        assert_eq!(tool.description(), "Get a pet by ID");
    }

    // ── Test 7: Parameter schema generation ───────────────────────────

    #[test]
    fn test_parameter_schema_generation() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let tool = tools.iter().find(|t| t.name() == "listPets").unwrap();
        let schema = tool.args_schema().unwrap();

        // Should have "limit" and "status" properties.
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("limit"));
        assert!(props.contains_key("status"));

        // "limit" should have integer type.
        assert_eq!(props["limit"]["type"], "integer");

        // Neither should be in required (both are optional).
        let required = schema["required"].as_array().unwrap();
        assert!(required.is_empty());
    }

    // ── Test 8: Path parameter substitution ───────────────────────────

    #[tokio::test]
    async fn test_path_parameter_substitution() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let get_pet_tool = tools.iter().find(|t| t.name() == "getPet").unwrap();

        let mut args = HashMap::new();
        args.insert("petId".to_string(), json!("42"));

        let result = get_pet_tool
            ._run(ToolInput::Structured(args))
            .await
            .unwrap();

        match result {
            ToolOutput::Content(v) => {
                let url = v["url"].as_str().unwrap();
                assert!(url.ends_with("/pets/42"), "URL was: {}", url);
                assert!(!url.contains("{petId}"));
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 9: Query parameter appending ─────────────────────────────

    #[tokio::test]
    async fn test_query_parameter_appending() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));

        let tool = tools.iter().find(|t| t.name() == "listPets").unwrap();

        let mut args = HashMap::new();
        args.insert("limit".to_string(), json!(5));
        args.insert("status".to_string(), json!("available"));

        let result = tool._run(ToolInput::Structured(args)).await.unwrap();

        match result {
            ToolOutput::Content(v) => {
                let url = v["url"].as_str().unwrap();
                assert!(url.contains("limit=5"), "URL was: {}", url);
                assert!(url.contains("status=available"), "URL was: {}", url);
                assert!(url.contains('?'), "URL was: {}", url);
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 10: Multiple operations generate multiple tools ──────────

    #[test]
    fn test_multiple_operations_generate_multiple_tools() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));
        assert_eq!(tools.len(), 6);
    }

    // ── Test 11: Invalid spec handling ────────────────────────────────

    #[test]
    fn test_invalid_spec_missing_info() {
        let spec = json!({"paths": {}});
        let result = OpenAPISpec::from_json(&spec);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_spec_missing_paths() {
        let spec = json!({"info": {"title": "Test", "version": "1.0"}});
        let result = OpenAPISpec::from_json(&spec);
        assert!(result.is_err());
    }

    // ── Test 12: Base URL override ────────────────────────────────────

    #[tokio::test]
    async fn test_base_url_override() {
        let toolkit = OpenAPIToolkit::from_json(&pet_store_spec())
            .unwrap()
            .with_base_url("https://custom.api.com/v2");

        let tools = toolkit.get_tools();
        let tool = tools.iter().find(|t| t.name() == "listPets").unwrap();

        let result = tool
            ._run(ToolInput::Structured(HashMap::new()))
            .await
            .unwrap();

        match result {
            ToolOutput::Content(v) => {
                assert!(v["url"]
                    .as_str()
                    .unwrap()
                    .starts_with("https://custom.api.com/v2"));
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 13: Auth headers are included ────────────────────────────

    #[tokio::test]
    async fn test_auth_headers_included() {
        let toolkit = OpenAPIToolkit::from_json(&pet_store_spec())
            .unwrap()
            .with_bearer_token("my-secret-token");

        let tools = toolkit.get_tools();
        let tool = tools.iter().find(|t| t.name() == "listPets").unwrap();

        let result = tool
            ._run(ToolInput::Structured(HashMap::new()))
            .await
            .unwrap();

        match result {
            ToolOutput::Content(v) => {
                let headers = v["headers"].as_object().unwrap();
                assert_eq!(
                    headers["Authorization"].as_str().unwrap(),
                    "Bearer my-secret-token"
                );
            }
            _ => panic!("Expected Content output"),
        }
    }

    // ── Test 14: Tag-based filtering ──────────────────────────────────

    #[test]
    fn test_tag_filtering() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(
            &spec,
            None,
            None,
            Some(&["users".to_string()]),
            None,
            Arc::new(DryRunExecutor),
        );
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "listUsers");
    }

    // ── Test 15: Operation ID filtering ───────────────────────────────

    #[test]
    fn test_operation_id_filtering() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(
            &spec,
            None,
            None,
            None,
            Some(&["getPet".to_string(), "deletePet".to_string()]),
            Arc::new(DryRunExecutor),
        );
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"getPet"));
        assert!(names.contains(&"deletePet"));
    }

    // ── Test 16: Fallback operation ID from method+path ───────────────

    #[test]
    fn test_fallback_operation_id() {
        let spec_json = json!({
            "openapi": "3.0.0",
            "info": { "title": "Test", "version": "1.0.0" },
            "servers": [{ "url": "https://example.com" }],
            "paths": {
                "/items": {
                    "get": {
                        "summary": "List items"
                    }
                }
            }
        });

        let spec = OpenAPISpec::from_json(&spec_json).unwrap();
        assert_eq!(spec.operations.len(), 1);
        assert_eq!(spec.operations[0].operation_id, "get_items");
    }

    // ── Test 17: Request body schema in args ──────────────────────────

    #[test]
    fn test_post_tool_args_schema_includes_body() {
        let spec = OpenAPISpec::from_json(&pet_store_spec()).unwrap();
        let tools = generate_tools(&spec, None, None, None, None, Arc::new(DryRunExecutor));
        let tool = tools.iter().find(|t| t.name() == "createPet").unwrap();

        let schema = tool.args_schema().unwrap();
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("body"));

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("body")));
    }

    // ── Test 18: Toolkit builder pattern ──────────────────────────────

    #[test]
    fn test_toolkit_builder_pattern() {
        let toolkit = OpenAPIToolkit::from_json(&pet_store_spec())
            .unwrap()
            .with_base_url("https://override.com")
            .with_header("X-Api-Key", "key123")
            .with_bearer_token("token456")
            .with_tag_filter(vec!["pets".to_string()]);

        let tools = toolkit.get_tools();
        // "users" tag filtered out, so only pet operations remain.
        assert_eq!(tools.len(), 5);
    }

    // ── Test 19: Servers trailing slash normalization ──────────────────

    #[test]
    fn test_trailing_slash_normalization() {
        let spec_json = json!({
            "openapi": "3.0.0",
            "info": { "title": "Test", "version": "1.0" },
            "servers": [{ "url": "https://api.example.com/" }],
            "paths": {
                "/health": {
                    "get": { "operationId": "healthCheck", "summary": "Health" }
                }
            }
        });

        let spec = OpenAPISpec::from_json(&spec_json).unwrap();
        assert_eq!(spec.servers[0], "https://api.example.com");
    }
}
