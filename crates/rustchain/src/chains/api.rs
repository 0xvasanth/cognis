use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// EndpointSpec & APISpec
// ---------------------------------------------------------------------------

/// Describes a single parameter for an API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSpec {
    /// Parameter name.
    pub name: String,
    /// Parameter type (e.g. `"string"`, `"integer"`, `"boolean"`).
    pub param_type: String,
    /// Whether the parameter is required.
    pub required: bool,
    /// Human-readable description.
    pub description: String,
}

/// Builder for [`ParameterSpec`].
pub struct ParameterSpecBuilder {
    name: String,
    param_type: String,
    required: bool,
    description: String,
}

impl ParameterSpecBuilder {
    /// Create a builder with the required parameter name and type.
    pub fn new(name: impl Into<String>, param_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: param_type.into(),
            required: false,
            description: String::new(),
        }
    }

    /// Mark this parameter as required.
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set the parameter description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Build the [`ParameterSpec`].
    pub fn build(self) -> ParameterSpec {
        ParameterSpec {
            name: self.name,
            param_type: self.param_type,
            required: self.required,
            description: self.description,
        }
    }
}

/// Describes a single API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EndpointSpec {
    /// HTTP method (e.g. `"GET"`, `"POST"`).
    pub method: String,
    /// URL path (e.g. `"/users/{id}"`).
    pub path: String,
    /// Human-readable description of what this endpoint does.
    pub description: String,
    /// Parameters for this endpoint.
    pub parameters: Vec<ParameterSpec>,
}

/// Builder for [`EndpointSpec`].
pub struct EndpointSpecBuilder {
    method: String,
    path: String,
    description: String,
    parameters: Vec<ParameterSpec>,
}

impl EndpointSpecBuilder {
    /// Create a builder with the required method and path.
    pub fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            description: String::new(),
            parameters: Vec::new(),
        }
    }

    /// Set the endpoint description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a parameter to this endpoint.
    pub fn parameter(mut self, param: ParameterSpec) -> Self {
        self.parameters.push(param);
        self
    }

    /// Build the [`EndpointSpec`].
    pub fn build(self) -> EndpointSpec {
        EndpointSpec {
            method: self.method,
            path: self.path,
            description: self.description,
            parameters: self.parameters,
        }
    }
}

/// Describes an API with a base URL and a collection of endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct APISpec {
    /// Base URL for the API (e.g. `"https://api.example.com"`).
    pub base_url: String,
    /// Available endpoints.
    pub endpoints: Vec<EndpointSpec>,
}

/// Builder for [`APISpec`].
pub struct APISpecBuilder {
    base_url: String,
    endpoints: Vec<EndpointSpec>,
}

impl APISpecBuilder {
    /// Create a builder with the required base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            endpoints: Vec::new(),
        }
    }

    /// Add an endpoint to the API spec.
    pub fn endpoint(mut self, endpoint: EndpointSpec) -> Self {
        self.endpoints.push(endpoint);
        self
    }

    /// Build the [`APISpec`].
    pub fn build(self) -> APISpec {
        APISpec {
            base_url: self.base_url,
            endpoints: self.endpoints,
        }
    }
}

impl APISpec {
    /// Start a new builder.
    pub fn builder(base_url: impl Into<String>) -> APISpecBuilder {
        APISpecBuilder::new(base_url)
    }

    /// Generate a human-readable description of the API for inclusion in LLM prompts.
    pub fn to_description(&self) -> String {
        let mut desc = format!("API Base URL: {}\n\nEndpoints:\n", self.base_url);
        for endpoint in &self.endpoints {
            desc.push_str(&format!(
                "\n  {} {} - {}\n",
                endpoint.method, endpoint.path, endpoint.description
            ));
            if !endpoint.parameters.is_empty() {
                desc.push_str("    Parameters:\n");
                for param in &endpoint.parameters {
                    let req = if param.required {
                        "required"
                    } else {
                        "optional"
                    };
                    desc.push_str(&format!(
                        "      - {} ({}, {}): {}\n",
                        param.name, param.param_type, req, param.description
                    ));
                }
            }
        }
        desc
    }

    /// Parse an [`APISpec`] from a simplified JSON format.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "base_url": "https://api.example.com",
    ///   "endpoints": [
    ///     {
    ///       "method": "GET",
    ///       "path": "/users",
    ///       "description": "List all users",
    ///       "parameters": [
    ///         {
    ///           "name": "limit",
    ///           "type": "integer",
    ///           "required": false,
    ///           "description": "Max results"
    ///         }
    ///       ]
    ///     }
    ///   ]
    /// }
    /// ```
    pub fn from_json(value: &Value) -> Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object".into(),
                got: format!("{}", value),
            })?;

        let base_url = obj
            .get("base_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RustChainError::InvalidKey("Missing 'base_url' key".into()))?
            .to_string();

        let endpoints_val = obj
            .get("endpoints")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RustChainError::InvalidKey("Missing 'endpoints' array".into()))?;

        let mut endpoints = Vec::new();
        for ep in endpoints_val {
            let ep_obj = ep.as_object().ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object for endpoint".into(),
                got: format!("{}", ep),
            })?;

            let method = ep_obj
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("GET")
                .to_string();
            let path = ep_obj
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("/")
                .to_string();
            let description = ep_obj
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mut parameters = Vec::new();
            if let Some(params) = ep_obj.get("parameters").and_then(|v| v.as_array()) {
                for p in params {
                    let p_obj = p.as_object().ok_or_else(|| RustChainError::TypeMismatch {
                        expected: "JSON object for parameter".into(),
                        got: format!("{}", p),
                    })?;

                    parameters.push(ParameterSpec {
                        name: p_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        param_type: p_obj
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("string")
                            .to_string(),
                        required: p_obj
                            .get("required")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        description: p_obj
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }

            endpoints.push(EndpointSpec {
                method,
                path,
                description,
                parameters,
            });
        }

        Ok(APISpec {
            base_url,
            endpoints,
        })
    }
}

// ---------------------------------------------------------------------------
// RequestValidator
// ---------------------------------------------------------------------------

/// Validates constructed HTTP requests for safety.
#[derive(Debug, Clone)]
pub struct RequestValidator {
    /// The expected base URL that all requests must start with.
    base_url: String,
    /// Set of allowed HTTP methods (uppercased).
    allowed_methods: HashSet<String>,
}

impl RequestValidator {
    /// Create a new validator for the given base URL and allowed methods.
    pub fn new(base_url: impl Into<String>, allowed_methods: &HashSet<String>) -> Self {
        Self {
            base_url: base_url.into(),
            allowed_methods: allowed_methods.clone(),
        }
    }

    /// Validate an HTTP request described as a JSON object.
    ///
    /// Checks:
    /// - The `method` is in the allowed set.
    /// - The `url` starts with the expected `base_url`.
    /// - The `url` does not contain injection patterns (newlines, backticks, etc.).
    pub fn validate(&self, request: &Value) -> Result<()> {
        let obj = request
            .as_object()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object".into(),
                got: format!("{}", request),
            })?;

        // Validate method
        let method = obj
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RustChainError::InvalidKey("Missing 'method' in request".into()))?
            .to_uppercase();

        if !self.allowed_methods.contains(&method) {
            return Err(RustChainError::Other(format!(
                "HTTP method '{}' is not allowed. Allowed methods: {:?}",
                method, self.allowed_methods
            )));
        }

        // Validate URL
        let url = obj
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RustChainError::InvalidKey("Missing 'url' in request".into()))?;

        if !url.starts_with(&self.base_url) {
            return Err(RustChainError::Other(format!(
                "URL '{}' does not start with expected base URL '{}'",
                url, self.base_url
            )));
        }

        // Check for injection patterns
        Self::check_url_injection(url)?;

        Ok(())
    }

    /// Check a URL for common injection patterns.
    fn check_url_injection(url: &str) -> Result<()> {
        // Disallow newlines (HTTP header injection)
        if url.contains('\n') || url.contains('\r') {
            return Err(RustChainError::Other(
                "URL contains newline characters (possible header injection)".into(),
            ));
        }

        // Disallow backticks (command injection in some contexts)
        if url.contains('`') {
            return Err(RustChainError::Other(
                "URL contains backtick characters (possible injection)".into(),
            ));
        }

        // Disallow javascript: scheme
        let lower = url.to_lowercase();
        if lower.contains("javascript:") {
            return Err(RustChainError::Other(
                "URL contains 'javascript:' scheme (possible injection)".into(),
            ));
        }

        // Disallow data: scheme
        if lower.starts_with("data:") {
            return Err(RustChainError::Other(
                "URL uses 'data:' scheme (possible injection)".into(),
            ));
        }

        // Disallow double-encoded characters that could bypass validation
        let double_encode_re = Regex::new(r"%25[0-9a-fA-F]{2}").unwrap();
        if double_encode_re.is_match(url) {
            return Err(RustChainError::Other(
                "URL contains double-encoded characters (possible injection)".into(),
            ));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Default prompt
// ---------------------------------------------------------------------------

const DEFAULT_API_PROMPT: &str = r#"You are an API request constructor. Given the following API specification and a user question, construct the appropriate HTTP request.

{api_description}

User question: {question}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"method": "GET", "url": "https://...", "headers": {{}}, "body": null}}"#;

// ---------------------------------------------------------------------------
// APIChain
// ---------------------------------------------------------------------------

/// A chain that uses an LLM to construct HTTP requests from natural-language
/// questions based on an API specification.
///
/// The chain:
/// 1. Formats the API spec into a human-readable description
/// 2. Sends the spec + question to the LLM
/// 3. Parses the LLM response as a JSON request object
/// 4. Validates the request (method, URL, injection checks)
/// 5. Optionally executes the HTTP request
pub struct APIChain {
    model: Arc<dyn BaseChatModel>,
    api_spec: APISpec,
    execute_requests: bool,
    allowed_methods: HashSet<String>,
    prompt_template: String,
}

/// Builder for [`APIChain`].
pub struct APIChainBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    api_spec: Option<APISpec>,
    execute_requests: bool,
    allowed_methods: HashSet<String>,
    prompt_template: String,
}

impl APIChainBuilder {
    /// Create a new builder with defaults (GET-only, no request execution).
    pub fn new() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        Self {
            model: None,
            api_spec: None,
            execute_requests: false,
            allowed_methods: allowed,
            prompt_template: DEFAULT_API_PROMPT.to_string(),
        }
    }

    /// Set the chat model (required).
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the API specification (required).
    pub fn api_spec(mut self, spec: APISpec) -> Self {
        self.api_spec = Some(spec);
        self
    }

    /// Enable or disable HTTP request execution. Default: `false`.
    pub fn execute_requests(mut self, execute: bool) -> Self {
        self.execute_requests = execute;
        self
    }

    /// Set the allowed HTTP methods. Default: `{"GET"}`.
    pub fn allowed_methods(mut self, methods: HashSet<String>) -> Self {
        self.allowed_methods = methods;
        self
    }

    /// Add an allowed HTTP method.
    pub fn allow_method(mut self, method: impl Into<String>) -> Self {
        self.allowed_methods.insert(method.into().to_uppercase());
        self
    }

    /// Set a custom prompt template.
    ///
    /// The template should contain `{api_description}` and `{question}` placeholders.
    pub fn prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Build the [`APIChain`]. Panics if model or api_spec is not set.
    pub fn build(self) -> APIChain {
        APIChain {
            model: self.model.expect("model is required for APIChain"),
            api_spec: self.api_spec.expect("api_spec is required for APIChain"),
            execute_requests: self.execute_requests,
            allowed_methods: self.allowed_methods,
            prompt_template: self.prompt_template,
        }
    }
}

impl Default for APIChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl APIChain {
    /// Start a new builder.
    pub fn builder() -> APIChainBuilder {
        APIChainBuilder::new()
    }

    /// Format the prompt template by replacing `{variable}` placeholders.
    fn format_prompt(&self, input: &Value) -> Result<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let obj = input
            .as_object()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object".into(),
                got: format!("{}", input),
            })?;

        let api_description = self.api_spec.to_description();

        let mut missing: Vec<String> = Vec::new();
        let result = re.replace_all(&self.prompt_template, |caps: &regex::Captures| {
            let key = &caps[1];
            match key {
                "api_description" => api_description.clone(),
                _ => match obj.get(key) {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => {
                        missing.push(key.to_string());
                        String::new()
                    }
                },
            }
        });

        if !missing.is_empty() {
            return Err(RustChainError::InvalidKey(format!(
                "Missing input variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(result.into_owned())
    }

    /// Extract a JSON request object from the LLM response text.
    ///
    /// Handles responses that may include markdown code blocks or extra text.
    fn extract_request(response: &str) -> Result<Value> {
        let trimmed = response.trim();

        // Try to extract from ```json ... ``` or ``` ... ``` code block
        let code_block_re = Regex::new(r"(?is)```(?:json)?\s*\n?(.*?)\n?\s*```").unwrap();
        let extracted = code_block_re
            .captures(trimmed)
            .map(|cap| cap[1].trim().to_string());
        let json_str = extracted.as_deref().unwrap_or(trimmed);

        // Try to find the first JSON object in the text
        let start = json_str.find('{').ok_or_else(|| {
            RustChainError::Other(format!(
                "Could not find JSON object in LLM response: {}",
                trimmed
            ))
        })?;

        // Find the matching closing brace
        let mut depth = 0;
        let mut end = start;
        for (i, ch) in json_str[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        let json_slice = &json_str[start..end];
        serde_json::from_str(json_slice).map_err(|e| {
            RustChainError::Other(format!("Failed to parse LLM response as JSON: {}", e))
        })
    }

    /// Create a [`RequestValidator`] for this chain's configuration.
    fn validator(&self) -> RequestValidator {
        RequestValidator::new(&self.api_spec.base_url, &self.allowed_methods)
    }
}

#[async_trait]
impl Runnable for APIChain {
    fn name(&self) -> &str {
        "APIChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let question = input
            .as_object()
            .and_then(|o| o.get("question"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                RustChainError::InvalidKey(
                    "Input must be a JSON object with a 'question' key".into(),
                )
            })?
            .to_string();

        // Format and send prompt to LLM
        let formatted = self.format_prompt(&input)?;
        let messages = vec![Message::Human(HumanMessage::new(&formatted))];
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        let raw_text = ai_msg.base.content.text();

        // Parse the LLM response into a request object
        let request = Self::extract_request(&raw_text)?;

        // Validate the request
        let validator = self.validator();
        validator.validate(&request)?;

        // If execute_requests is enabled and reqwest is available, make the call
        if self.execute_requests {
            #[cfg(feature = "openai")] // reqwest is available with any provider feature
            {
                let response = self.execute_http_request(&request).await?;
                return Ok(json!({
                    "question": question,
                    "request": request,
                    "response": response,
                }));
            }

            #[cfg(not(feature = "openai"))]
            {
                return Err(RustChainError::Other(
                    "HTTP request execution requires the 'reqwest' dependency (enable a provider feature)".into(),
                ));
            }
        }

        Ok(json!({
            "question": question,
            "request": request,
        }))
    }
}

/// HTTP execution support (available when reqwest is present via a provider feature).
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure"
))]
impl APIChain {
    /// Execute an HTTP request described by a JSON object.
    async fn execute_http_request(&self, request: &Value) -> Result<Value> {
        let client = reqwest::Client::new();

        let method_str = request["method"].as_str().unwrap_or("GET").to_uppercase();
        let url = request["url"]
            .as_str()
            .ok_or_else(|| RustChainError::InvalidKey("Missing 'url' in request".into()))?;

        let method = match method_str.as_str() {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "DELETE" => reqwest::Method::DELETE,
            "PATCH" => reqwest::Method::PATCH,
            "HEAD" => reqwest::Method::HEAD,
            "OPTIONS" => reqwest::Method::OPTIONS,
            other => {
                return Err(RustChainError::Other(format!(
                    "Unsupported HTTP method: {}",
                    other
                )));
            }
        };

        let mut req_builder = client.request(method, url);

        // Add headers
        if let Some(headers) = request.get("headers").and_then(|h| h.as_object()) {
            for (key, val) in headers {
                if let Some(v) = val.as_str() {
                    req_builder = req_builder.header(key.as_str(), v);
                }
            }
        }

        // Add body
        if let Some(body) = request.get("body") {
            if !body.is_null() {
                req_builder = req_builder.json(body);
            }
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| RustChainError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        let body_text = response
            .text()
            .await
            .map_err(|e| RustChainError::Other(format!("Failed to read response body: {}", e)))?;

        // Try to parse as JSON, fall back to string
        let body_value =
            serde_json::from_str::<Value>(&body_text).unwrap_or_else(|_| Value::String(body_text));

        Ok(json!({
            "status": status,
            "body": body_value,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::FakeListChatModel;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn sample_api_spec() -> APISpec {
        APISpec::builder("https://api.example.com")
            .endpoint(
                EndpointSpecBuilder::new("GET", "/users")
                    .description("List all users")
                    .parameter(
                        ParameterSpecBuilder::new("limit", "integer")
                            .required(false)
                            .description("Maximum number of results")
                            .build(),
                    )
                    .build(),
            )
            .endpoint(
                EndpointSpecBuilder::new("GET", "/users/{id}")
                    .description("Get a specific user by ID")
                    .parameter(
                        ParameterSpecBuilder::new("id", "integer")
                            .required(true)
                            .description("User ID")
                            .build(),
                    )
                    .build(),
            )
            .endpoint(
                EndpointSpecBuilder::new("POST", "/users")
                    .description("Create a new user")
                    .parameter(
                        ParameterSpecBuilder::new("name", "string")
                            .required(true)
                            .description("User name")
                            .build(),
                    )
                    .parameter(
                        ParameterSpecBuilder::new("email", "string")
                            .required(true)
                            .description("User email")
                            .build(),
                    )
                    .build(),
            )
            .build()
    }

    // 1. Basic API request construction
    #[tokio::test]
    async fn test_basic_api_request_construction() {
        let response = r#"{"method": "GET", "url": "https://api.example.com/users?limit=10", "headers": {}, "body": null}"#;
        let chain = APIChain::builder()
            .model(fake_model(vec![response]))
            .api_spec(sample_api_spec())
            .build();

        let result = chain
            .invoke(json!({"question": "List the first 10 users"}), None)
            .await
            .unwrap();

        assert_eq!(result["request"]["method"], "GET");
        assert_eq!(
            result["request"]["url"],
            "https://api.example.com/users?limit=10"
        );
        assert_eq!(result["question"], "List the first 10 users");
        // No response key when execute_requests is false
        assert!(result.get("response").is_none());
    }

    // 2. API spec to_description formatting
    #[test]
    fn test_api_spec_to_description() {
        let spec = sample_api_spec();
        let desc = spec.to_description();

        assert!(desc.contains("API Base URL: https://api.example.com"));
        assert!(desc.contains("GET /users - List all users"));
        assert!(desc.contains("GET /users/{id} - Get a specific user by ID"));
        assert!(desc.contains("POST /users - Create a new user"));
        assert!(desc.contains("limit (integer, optional)"));
        assert!(desc.contains("id (integer, required)"));
        assert!(desc.contains("name (string, required)"));
        assert!(desc.contains("email (string, required)"));
    }

    // 3. Request validation (valid URL)
    #[test]
    fn test_request_validation_valid_url() {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        let validator = RequestValidator::new("https://api.example.com", &allowed);

        let request = json!({
            "method": "GET",
            "url": "https://api.example.com/users?limit=10",
            "headers": {},
            "body": null
        });

        assert!(validator.validate(&request).is_ok());
    }

    // 4. Request validation (invalid method rejected)
    #[test]
    fn test_request_validation_invalid_method_rejected() {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        let validator = RequestValidator::new("https://api.example.com", &allowed);

        let request = json!({
            "method": "DELETE",
            "url": "https://api.example.com/users/1",
            "headers": {},
            "body": null
        });

        let result = validator.validate(&request);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not allowed"),
            "Error should mention 'not allowed': {err}"
        );
    }

    // 5. APISpec builder
    #[test]
    fn test_api_spec_builder() {
        let spec = APISpec::builder("https://api.test.com")
            .endpoint(
                EndpointSpecBuilder::new("GET", "/health")
                    .description("Health check")
                    .build(),
            )
            .endpoint(
                EndpointSpecBuilder::new("POST", "/data")
                    .description("Submit data")
                    .parameter(
                        ParameterSpecBuilder::new("payload", "object")
                            .required(true)
                            .description("Data payload")
                            .build(),
                    )
                    .build(),
            )
            .build();

        assert_eq!(spec.base_url, "https://api.test.com");
        assert_eq!(spec.endpoints.len(), 2);
        assert_eq!(spec.endpoints[0].method, "GET");
        assert_eq!(spec.endpoints[0].path, "/health");
        assert_eq!(spec.endpoints[1].method, "POST");
        assert_eq!(spec.endpoints[1].parameters.len(), 1);
        assert!(spec.endpoints[1].parameters[0].required);
    }

    // 6. EndpointSpec with parameters
    #[test]
    fn test_endpoint_spec_with_parameters() {
        let endpoint = EndpointSpecBuilder::new("PUT", "/items/{id}")
            .description("Update an item")
            .parameter(
                ParameterSpecBuilder::new("id", "integer")
                    .required(true)
                    .description("Item ID")
                    .build(),
            )
            .parameter(
                ParameterSpecBuilder::new("name", "string")
                    .required(false)
                    .description("New item name")
                    .build(),
            )
            .parameter(
                ParameterSpecBuilder::new("active", "boolean")
                    .required(false)
                    .description("Whether the item is active")
                    .build(),
            )
            .build();

        assert_eq!(endpoint.method, "PUT");
        assert_eq!(endpoint.path, "/items/{id}");
        assert_eq!(endpoint.description, "Update an item");
        assert_eq!(endpoint.parameters.len(), 3);
        assert!(endpoint.parameters[0].required);
        assert!(!endpoint.parameters[1].required);
        assert_eq!(endpoint.parameters[2].param_type, "boolean");
    }

    // 7. Allowed methods filtering
    #[tokio::test]
    async fn test_allowed_methods_filtering() {
        // LLM returns a POST request but only GET is allowed
        let response = r#"{"method": "POST", "url": "https://api.example.com/users", "headers": {}, "body": {"name": "test"}}"#;
        let chain = APIChain::builder()
            .model(fake_model(vec![response]))
            .api_spec(sample_api_spec())
            .build();

        let result = chain
            .invoke(json!({"question": "Create a new user named test"}), None)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not allowed"),
            "Error should mention method not allowed: {err}"
        );
    }

    // 8. from_json parsing
    #[test]
    fn test_from_json_parsing() {
        let json_spec = json!({
            "base_url": "https://api.weather.com",
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/forecast",
                    "description": "Get weather forecast",
                    "parameters": [
                        {
                            "name": "city",
                            "type": "string",
                            "required": true,
                            "description": "City name"
                        },
                        {
                            "name": "days",
                            "type": "integer",
                            "required": false,
                            "description": "Number of forecast days"
                        }
                    ]
                }
            ]
        });

        let spec = APISpec::from_json(&json_spec).unwrap();
        assert_eq!(spec.base_url, "https://api.weather.com");
        assert_eq!(spec.endpoints.len(), 1);
        assert_eq!(spec.endpoints[0].method, "GET");
        assert_eq!(spec.endpoints[0].path, "/forecast");
        assert_eq!(spec.endpoints[0].parameters.len(), 2);
        assert_eq!(spec.endpoints[0].parameters[0].name, "city");
        assert!(spec.endpoints[0].parameters[0].required);
        assert_eq!(spec.endpoints[0].parameters[1].param_type, "integer");
        assert!(!spec.endpoints[0].parameters[1].required);
    }

    // 9. Runnable trait implementation
    #[tokio::test]
    async fn test_runnable_trait_implementation() {
        let response = r#"{"method": "GET", "url": "https://api.example.com/users/42", "headers": {}, "body": null}"#;
        let chain = APIChain::builder()
            .model(fake_model(vec![response]))
            .api_spec(sample_api_spec())
            .build();

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "APIChain");

        let result = runnable
            .invoke(json!({"question": "Get user 42"}), None)
            .await
            .unwrap();
        assert_eq!(result["request"]["method"], "GET");
        assert_eq!(result["request"]["url"], "https://api.example.com/users/42");
    }

    // 10. Request extraction from LLM response
    #[test]
    fn test_request_extraction_from_llm_response() {
        // Plain JSON
        let plain = r#"{"method": "GET", "url": "https://api.example.com/users", "headers": {}, "body": null}"#;
        let parsed = APIChain::extract_request(plain).unwrap();
        assert_eq!(parsed["method"], "GET");

        // Wrapped in markdown code block
        let markdown = "Here is the request:\n```json\n{\"method\": \"GET\", \"url\": \"https://api.example.com/users\", \"headers\": {}, \"body\": null}\n```";
        let parsed = APIChain::extract_request(markdown).unwrap();
        assert_eq!(parsed["method"], "GET");
        assert_eq!(parsed["url"], "https://api.example.com/users");

        // With extra text around it
        let noisy = "Sure! Here you go: {\"method\": \"POST\", \"url\": \"https://api.example.com/data\", \"headers\": {\"Content-Type\": \"application/json\"}, \"body\": {\"key\": \"value\"}} Hope that helps!";
        let parsed = APIChain::extract_request(noisy).unwrap();
        assert_eq!(parsed["method"], "POST");
        assert_eq!(parsed["body"]["key"], "value");
    }

    // 11. URL injection prevention
    #[test]
    fn test_url_injection_prevention() {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        let validator = RequestValidator::new("https://api.example.com", &allowed);

        // Newline injection
        let req_newline = json!({
            "method": "GET",
            "url": "https://api.example.com/users\r\nX-Injected: header",
            "headers": {},
            "body": null
        });
        let result = validator.validate(&req_newline);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("newline"));

        // Backtick injection
        let req_backtick = json!({
            "method": "GET",
            "url": "https://api.example.com/users`whoami`",
            "headers": {},
            "body": null
        });
        let result = validator.validate(&req_backtick);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("backtick"));

        // Wrong base URL
        let req_wrong_base = json!({
            "method": "GET",
            "url": "https://evil.com/api",
            "headers": {},
            "body": null
        });
        let result = validator.validate(&req_wrong_base);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base URL"));

        // JavaScript scheme
        let req_js = json!({
            "method": "GET",
            "url": "https://api.example.com/users?redirect=javascript:alert(1)",
            "headers": {},
            "body": null
        });
        let result = validator.validate(&req_js);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("javascript"));
    }

    // 12. Allow multiple methods
    #[tokio::test]
    async fn test_allow_multiple_methods() {
        let response = r#"{"method": "POST", "url": "https://api.example.com/users", "headers": {"Content-Type": "application/json"}, "body": {"name": "Alice"}}"#;
        let chain = APIChain::builder()
            .model(fake_model(vec![response]))
            .api_spec(sample_api_spec())
            .allow_method("POST")
            .build();

        let result = chain
            .invoke(json!({"question": "Create user Alice"}), None)
            .await
            .unwrap();

        assert_eq!(result["request"]["method"], "POST");
        assert_eq!(result["request"]["body"]["name"], "Alice");
    }

    // 13. from_json missing base_url
    #[test]
    fn test_from_json_missing_base_url() {
        let json_spec = json!({
            "endpoints": []
        });
        let result = APISpec::from_json(&json_spec);
        assert!(result.is_err());
    }

    // 14. Double-encoded URL rejection
    #[test]
    fn test_double_encoded_url_rejection() {
        let mut allowed = HashSet::new();
        allowed.insert("GET".to_string());
        let validator = RequestValidator::new("https://api.example.com", &allowed);

        let req = json!({
            "method": "GET",
            "url": "https://api.example.com/path%252F..%252Fetc%252Fpasswd",
            "headers": {},
            "body": null
        });
        let result = validator.validate(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("double-encoded"));
    }
}
