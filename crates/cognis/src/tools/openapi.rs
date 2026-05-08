//! OpenAPI tool — generate one [`Tool`] per operation in an OpenAPI 3.x
//! spec. Calls the underlying HTTP endpoint with the model-supplied
//! arguments.
//!
//! Supports:
//! - GET / POST / PUT / PATCH / DELETE
//! - Path parameters (`{user_id}` → from `params.user_id`)
//! - Query parameters (`in: query`)
//! - Header parameters (`in: header`)
//! - JSON request body (when `requestBody` is present and the only
//!   content-type is `application/json`)
//!
//! Customization:
//! - [`OpenApiToolset::with_auth`] — pluggable [`AuthScheme`] applied
//!   to every request (Bearer, ApiKey-header, custom signer).
//! - [`OpenApiToolset::with_extra_header`] — additional headers on
//!   every request (correlation id, tenant, etc.).
//! - [`OpenApiToolset::filter_operations`] — only include operations
//!   whose `operationId` matches a predicate.
//!
//! What it does **not** do:
//! - Full OpenAPI semantic validation. We treat the spec as a
//!   `serde_json::Value` and walk the relevant paths. For strict
//!   validation, parse with `openapiv3` externally and pass the
//!   resulting JSON.
//! - cookie / form / multipart bodies. JSON body only.
//!
//! Feature-gated under `tools-http` because it needs `reqwest`.

#![cfg(feature = "tools-http")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;
use serde_json::Value;

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

/// Pluggable auth scheme. The closure receives the outgoing
/// `RequestBuilder` and returns the (possibly modified) builder.
pub trait AuthScheme: Send + Sync {
    /// Apply auth to a request.
    fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder;
}

/// Bearer-token auth.
pub struct BearerAuth(pub String);

impl AuthScheme for BearerAuth {
    fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.bearer_auth(&self.0)
    }
}

/// Custom header-based auth (e.g. `X-API-Key`).
pub struct HeaderAuth {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

impl HeaderAuth {
    /// Build with `(header_name, value)`.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

impl AuthScheme for HeaderAuth {
    fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header(&self.name, &self.value)
    }
}

/// Closure-based auth.
impl<F> AuthScheme for F
where
    F: Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync,
{
    fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        (self)(req)
    }
}

/// Boxed predicate over `operationId`.
pub type OperationFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;
/// `(url, query_pairs, header_pairs)` returned from request assembly.
pub type AssembledRequest = (String, Vec<(String, String)>, Vec<(String, String)>);

/// Toolset built from an OpenAPI spec.
pub struct OpenApiToolset {
    spec: Value,
    base_url: String,
    auth: Option<Arc<dyn AuthScheme>>,
    extra_headers: Vec<(String, String)>,
    operation_filter: Option<OperationFilter>,
    http: reqwest::Client,
}

impl OpenApiToolset {
    /// Build from a parsed spec value (typically loaded via `serde_json`
    /// from a JSON or YAML file).
    pub fn new(spec: Value) -> Result<Self> {
        let base_url = extract_base_url(&spec)?;
        let http = reqwest::ClientBuilder::new()
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(Self {
            spec,
            base_url,
            auth: None,
            extra_headers: Vec::new(),
            operation_filter: None,
            http,
        })
    }

    /// Override the base URL (default: first `servers[0].url` in spec).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the auth scheme.
    pub fn with_auth<A: AuthScheme + 'static>(mut self, auth: A) -> Self {
        self.auth = Some(Arc::new(auth));
        self
    }

    /// Add an extra header on every request.
    pub fn with_extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }

    /// Only include operations whose `operationId` passes `predicate`.
    pub fn filter_operations<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.operation_filter = Some(Arc::new(predicate));
        self
    }

    /// Override the HTTP client.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Generate one [`Tool`] per operation in the spec. Operations
    /// without an `operationId` are skipped (the id is the tool name).
    pub fn into_tools(self) -> Result<Vec<Arc<dyn Tool>>> {
        let mut out: Vec<Arc<dyn Tool>> = Vec::new();
        let paths = match self.spec.get("paths").and_then(|v| v.as_object()) {
            Some(p) => p.clone(),
            None => return Ok(out),
        };
        let toolset = Arc::new(ToolsetCore {
            base_url: self.base_url,
            auth: self.auth,
            extra_headers: self.extra_headers,
            http: self.http,
        });
        for (path, methods) in paths.iter() {
            let methods_obj = match methods.as_object() {
                Some(o) => o,
                None => continue,
            };
            for (method, op) in methods_obj {
                let m = method.to_uppercase();
                if !matches!(m.as_str(), "GET" | "POST" | "PUT" | "PATCH" | "DELETE") {
                    continue;
                }
                let operation_id = match op
                    .get("operationId")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                {
                    Some(id) => id,
                    None => continue,
                };
                if let Some(filter) = &self.operation_filter {
                    if !filter(&operation_id) {
                        continue;
                    }
                }
                let parameters = op
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Array(Vec::new()));
                let request_body = op.get("requestBody").cloned();
                let description = op
                    .get("summary")
                    .or_else(|| op.get("description"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("OpenAPI operation")
                    .to_string();
                out.push(Arc::new(OpenApiOperationTool {
                    toolset: toolset.clone(),
                    operation_id,
                    description,
                    method: m,
                    path: path.to_string(),
                    parameters,
                    request_body,
                }));
            }
        }
        Ok(out)
    }
}

struct ToolsetCore {
    base_url: String,
    auth: Option<Arc<dyn AuthScheme>>,
    extra_headers: Vec<(String, String)>,
    http: reqwest::Client,
}

/// One operation, exposed as a Tool.
struct OpenApiOperationTool {
    toolset: Arc<ToolsetCore>,
    operation_id: String,
    description: String,
    method: String,
    path: String,
    parameters: Value,
    request_body: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GenericInput {
    /// Path / query / header parameters keyed by name.
    #[serde(default)]
    params: HashMap<String, Value>,
    /// JSON body (when the operation declares one).
    #[serde(default)]
    body: Option<Value>,
}

#[async_trait]
impl Tool for OpenApiOperationTool {
    fn name(&self) -> &str {
        &self.operation_id
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        // We could synthesize a per-operation schema from `parameters`,
        // but a generic one is enough to let the LLM call the tool —
        // the per-operation contract is announced in the description.
        Some(serde_json::to_value(schemars::schema_for!(GenericInput)).unwrap_or_default())
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: GenericInput = serde_json::from_value(input.into_json()).map_err(|e| {
            CognisError::ToolValidationError(format!(
                "openapi[{op}]: invalid args: {e}",
                op = self.operation_id
            ))
        })?;
        // Substitute `{name}` placeholders in path with params; remaining
        // params are dispatched to query/header per the spec metadata.
        let (url, query_pairs, header_pairs) = assemble_request(
            &self.path,
            &self.parameters,
            &parsed.params,
            &self.toolset.base_url,
        )?;

        let mut req = match self.method.as_str() {
            "GET" => self.toolset.http.get(&url),
            "POST" => self.toolset.http.post(&url),
            "PUT" => self.toolset.http.put(&url),
            "PATCH" => self.toolset.http.patch(&url),
            "DELETE" => self.toolset.http.delete(&url),
            other => {
                return Err(CognisError::Internal(format!(
                    "openapi: unsupported method `{other}`"
                )))
            }
        };

        if !query_pairs.is_empty() {
            req = req.query(&query_pairs);
        }
        for (k, v) in header_pairs {
            req = req.header(&k, &v);
        }
        for (k, v) in &self.toolset.extra_headers {
            req = req.header(k, v);
        }
        if let Some(auth) = &self.toolset.auth {
            req = auth.apply(req);
        }
        if matches!(self.method.as_str(), "POST" | "PUT" | "PATCH") && self.request_body.is_some() {
            if let Some(b) = parsed.body {
                req = req.json(&b);
            }
        }
        let resp = req
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("openapi[{}]: {e}", self.operation_id)))?;
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        let body_value = serde_json::from_str::<Value>(&body_text)
            .unwrap_or_else(|_| Value::String(body_text.clone()));
        Ok(ToolOutput::Content(serde_json::json!({
            "status": status.as_u16(),
            "ok": status.is_success(),
            "body": body_value,
        })))
    }
}

fn extract_base_url(spec: &Value) -> Result<String> {
    if let Some(servers) = spec.get("servers").and_then(|v| v.as_array()) {
        if let Some(s) = servers
            .first()
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
        {
            return Ok(s.to_string());
        }
    }
    Err(CognisError::Configuration(
        "openapi: spec has no servers[0].url and no base_url override was provided".into(),
    ))
}

/// Build the final URL + query/header pairs for an operation call.
/// Path templates `{name}` are substituted from `params`; consumed
/// names are removed so they don't leak into query/header.
fn assemble_request(
    path: &str,
    parameters: &Value,
    params: &HashMap<String, Value>,
    base_url: &str,
) -> Result<AssembledRequest> {
    let mut consumed_path: Vec<String> = Vec::new();
    let mut query: Vec<(String, String)> = Vec::new();
    let mut headers: Vec<(String, String)> = Vec::new();

    // Substitute path templates first.
    let mut full_path = path.to_string();
    let mut start = 0usize;
    while let Some(open) = full_path[start..].find('{') {
        let abs_open = start + open;
        if let Some(close_rel) = full_path[abs_open..].find('}') {
            let abs_close = abs_open + close_rel;
            let name = full_path[abs_open + 1..abs_close].to_string();
            let value = params.get(&name).map(value_to_string).unwrap_or_default();
            full_path.replace_range(abs_open..=abs_close, &value);
            consumed_path.push(name);
            start = abs_open + value.len();
        } else {
            break;
        }
    }

    // Walk the operation's `parameters` array to dispatch query/header.
    if let Some(params_arr) = parameters.as_array() {
        for p in params_arr {
            let name = match p.get("name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => continue,
            };
            let location = match p.get("in").and_then(|v| v.as_str()) {
                Some(l) => l,
                None => continue,
            };
            // Skip already-substituted path params.
            if location == "path" && consumed_path.iter().any(|n| n == name) {
                continue;
            }
            let value = match params.get(name) {
                Some(v) => v.clone(),
                None => continue,
            };
            let s = value_to_string(&value);
            match location {
                "query" => query.push((name.to_string(), s)),
                "header" => headers.push((name.to_string(), s)),
                "path" => {
                    // Path was already templated above; if the spec
                    // declares a path parameter that the path template
                    // doesn't reference, ignore it (malformed spec).
                }
                _ => {} // cookie / form / unknown — ignored
            }
        }
    }

    let mut url = base_url.to_string();
    if !url.ends_with('/') && !full_path.starts_with('/') {
        url.push('/');
    }
    if url.ends_with('/') && full_path.starts_with('/') {
        url.pop();
    }
    url.push_str(&full_path);
    Ok((url, query, headers))
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_spec() -> Value {
        json!({
            "servers": [{"url": "https://api.example.com/v1"}],
            "paths": {
                "/users/{id}": {
                    "get": {
                        "operationId": "getUser",
                        "summary": "Get user by id",
                        "parameters": [
                            {"name": "id", "in": "path"},
                            {"name": "verbose", "in": "query"}
                        ]
                    }
                },
                "/users": {
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create user",
                        "requestBody": {"content": {"application/json": {}}}
                    }
                }
            }
        })
    }

    #[test]
    fn extracts_base_url_from_servers() {
        let s = sample_spec();
        assert_eq!(extract_base_url(&s).unwrap(), "https://api.example.com/v1");
    }

    #[test]
    fn missing_servers_errors() {
        let bad = json!({"paths": {}});
        assert!(extract_base_url(&bad).is_err());
    }

    #[test]
    fn into_tools_produces_one_tool_per_operation() {
        let toolset = OpenApiToolset::new(sample_spec()).unwrap();
        let tools = toolset.into_tools().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"getUser"));
        assert!(names.contains(&"createUser"));
    }

    #[test]
    fn filter_operations_skips_unmatched() {
        let toolset = OpenApiToolset::new(sample_spec())
            .unwrap()
            .filter_operations(|id| id.starts_with("create"));
        let tools = toolset.into_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "createUser");
    }

    #[test]
    fn assemble_request_substitutes_path_and_classifies_params() {
        let mut params = HashMap::new();
        params.insert("id".to_string(), json!(42));
        params.insert("verbose".to_string(), json!(true));
        params.insert("X-Trace".to_string(), json!("abc"));
        let parameters = json!([
            {"name": "id", "in": "path"},
            {"name": "verbose", "in": "query"},
            {"name": "X-Trace", "in": "header"}
        ]);
        let (url, query, headers) = assemble_request(
            "/users/{id}",
            &parameters,
            &params,
            "https://api.example.com/v1",
        )
        .unwrap();
        assert_eq!(url, "https://api.example.com/v1/users/42");
        assert_eq!(query, vec![("verbose".to_string(), "true".into())]);
        assert_eq!(headers, vec![("X-Trace".to_string(), "abc".into())]);
    }

    #[test]
    fn auth_helpers_construct() {
        // Just verify the structs build; we can't invoke without a real client.
        let _ = BearerAuth("sk-test".into());
        let _ = HeaderAuth::new("X-API-Key", "abc");
    }

    #[test]
    fn with_base_url_overrides_servers() {
        let toolset = OpenApiToolset::new(sample_spec())
            .unwrap()
            .with_base_url("https://staging.example.com");
        assert!(toolset.base_url.contains("staging"));
    }
}
