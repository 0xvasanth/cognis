//! LangSmith-compatible tracing exporter.
//!
//! Captures runs and exports them in a format compatible with the LangSmith API.
//! This module handles serialization/format only — it does NOT make HTTP calls.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

/// Run type for LangSmith API compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LangSmithRunType {
    Llm,
    Chain,
    Tool,
    Retriever,
}

/// A run in LangSmith-compatible format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangSmithRun {
    /// Unique identifier for this run.
    pub id: Uuid,
    /// Human-readable name for the run.
    pub name: String,
    /// Type of run.
    pub run_type: LangSmithRunType,
    /// Inputs provided to the component.
    pub inputs: Value,
    /// Outputs produced by the component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Value>,
    /// Error message if the run failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Start time as milliseconds since UNIX epoch.
    pub start_time: u128,
    /// End time as milliseconds since UNIX epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_time: Option<u128>,
    /// Parent run ID for nested runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<Uuid>,
    /// Additional metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
    /// Tags associated with this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Serialized representation of the component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serialized: Option<Value>,
    /// Session/project name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
}

impl LangSmithRun {
    /// Create a new LangSmith run with the current timestamp.
    pub fn new(
        id: Uuid,
        name: impl Into<String>,
        run_type: LangSmithRunType,
        inputs: Value,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            run_type,
            inputs,
            outputs: None,
            error: None,
            start_time: now_millis(),
            end_time: None,
            parent_run_id: None,
            extra: HashMap::new(),
            tags: Vec::new(),
            serialized: None,
            session_name: None,
        }
    }
}

/// Configuration for LangSmith tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangSmithConfig {
    /// LangSmith API URL.
    pub api_url: String,
    /// LangSmith API key.
    pub api_key: String,
    /// Project name for grouping runs.
    pub project_name: String,
    /// Default tags to apply to all runs.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl LangSmithConfig {
    /// Create a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_url: "https://api.smith.langchain.com".to_string(),
            api_key: api_key.into(),
            project_name: "default".to_string(),
            tags: Vec::new(),
        }
    }

    /// Set the project name.
    pub fn with_project_name(mut self, name: impl Into<String>) -> Self {
        self.project_name = name.into();
        self
    }

    /// Set the API URL.
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Set default tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

impl Default for LangSmithConfig {
    fn default() -> Self {
        Self::new("")
    }
}

/// LangSmith-compatible tracer that captures runs and can export them
/// in LangSmith API format.
///
/// This tracer captures all run events (LLM, chain, tool, retriever)
/// with full timing, input/output, and metadata. It does NOT make
/// actual HTTP calls — use `LangSmithExporter` to serialize runs
/// for the LangSmith API.
pub struct LangSmithTracer {
    config: LangSmithConfig,
    runs: Mutex<Vec<LangSmithRun>>,
}

impl LangSmithTracer {
    /// Create a new LangSmith tracer with the given config.
    pub fn new(config: LangSmithConfig) -> Self {
        Self {
            config,
            runs: Mutex::new(Vec::new()),
        }
    }

    /// Get a copy of all collected runs.
    pub fn get_runs(&self) -> Vec<LangSmithRun> {
        self.runs.lock().unwrap().clone()
    }

    /// Get a specific run by ID.
    pub fn get_run(&self, id: Uuid) -> Option<LangSmithRun> {
        self.runs
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .cloned()
    }

    /// Get all runs of a specific type.
    pub fn get_runs_by_type(&self, run_type: LangSmithRunType) -> Vec<LangSmithRun> {
        self.runs
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.run_type == run_type)
            .cloned()
            .collect()
    }

    /// Get all child runs of a given parent run.
    pub fn get_child_runs(&self, parent_id: Uuid) -> Vec<LangSmithRun> {
        self.runs
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.parent_run_id == Some(parent_id))
            .cloned()
            .collect()
    }

    /// Clear all collected runs.
    pub fn clear(&self) {
        self.runs.lock().unwrap().clear();
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &LangSmithConfig {
        &self.config
    }

    /// Export all runs as a JSON value matching the LangSmith single-run
    /// POST /runs format. Returns an array of run payloads.
    pub fn export_json(&self) -> Value {
        let runs = self.runs.lock().unwrap();
        let payloads: Vec<Value> = runs.iter().map(|r| run_to_api_payload(r, &self.config)).collect();
        Value::Array(payloads)
    }

    /// Export all runs in the LangSmith batch endpoint format.
    ///
    /// Returns a JSON object with `"post"`, `"patch"` arrays matching
    /// the batch API contract. Completed runs (with end_time) go to `"patch"`,
    /// in-progress runs go to `"post"`.
    pub fn export_batch_json(&self) -> Value {
        let runs = self.runs.lock().unwrap();
        let mut post = Vec::new();
        let mut patch = Vec::new();

        for run in runs.iter() {
            let payload = run_to_api_payload(run, &self.config);
            if run.end_time.is_some() {
                patch.push(payload);
            } else {
                post.push(payload);
            }
        }

        serde_json::json!({
            "post": post,
            "patch": patch,
        })
    }

    fn add_run(&self, mut run: LangSmithRun) {
        // Apply default tags from config.
        for tag in &self.config.tags {
            if !run.tags.contains(tag) {
                run.tags.push(tag.clone());
            }
        }
        run.session_name = Some(self.config.project_name.clone());
        self.runs.lock().unwrap().push(run);
    }

    fn finish_run(&self, run_id: Uuid, outputs: Option<Value>, error: Option<String>) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.end_time = Some(now_millis());
            if let Some(out) = outputs {
                run.outputs = Some(out);
            }
            if let Some(err) = error {
                run.error = Some(err);
            }
        }
    }
}

/// Exporter utility for serializing `LangSmithRun` values to LangSmith API format.
pub struct LangSmithExporter;

impl LangSmithExporter {
    /// Serialize a single run to the LangSmith POST /runs payload format.
    pub fn run_to_json(run: &LangSmithRun, config: &LangSmithConfig) -> Value {
        run_to_api_payload(run, config)
    }

    /// Serialize multiple runs to the batch endpoint format.
    pub fn batch_to_json(runs: &[LangSmithRun], config: &LangSmithConfig) -> Value {
        let mut post = Vec::new();
        let mut patch = Vec::new();

        for run in runs {
            let payload = run_to_api_payload(run, config);
            if run.end_time.is_some() {
                patch.push(payload);
            } else {
                post.push(payload);
            }
        }

        serde_json::json!({
            "post": post,
            "patch": patch,
        })
    }
}

/// Convert a `LangSmithRun` to a LangSmith API-compatible JSON payload.
fn run_to_api_payload(run: &LangSmithRun, config: &LangSmithConfig) -> Value {
    let mut payload = serde_json::json!({
        "id": run.id.to_string(),
        "name": run.name,
        "run_type": run.run_type,
        "inputs": run.inputs,
        "start_time": run.start_time,
        "session_name": run.session_name.as_deref().unwrap_or(&config.project_name),
    });

    let obj = payload.as_object_mut().unwrap();

    if let Some(ref outputs) = run.outputs {
        obj.insert("outputs".to_string(), outputs.clone());
    }
    if let Some(ref error) = run.error {
        obj.insert("error".to_string(), Value::String(error.clone()));
    }
    if let Some(end_time) = run.end_time {
        obj.insert("end_time".to_string(), serde_json::json!(end_time));
    }
    if let Some(parent_id) = run.parent_run_id {
        obj.insert(
            "parent_run_id".to_string(),
            Value::String(parent_id.to_string()),
        );
    }
    if !run.extra.is_empty() {
        obj.insert("extra".to_string(), serde_json::to_value(&run.extra).unwrap());
    }
    if !run.tags.is_empty() {
        obj.insert("tags".to_string(), serde_json::to_value(&run.tags).unwrap());
    }
    if let Some(ref serialized) = run.serialized {
        obj.insert("serialized".to_string(), serialized.clone());
    }

    payload
}

/// Get the current time in milliseconds since UNIX epoch.
fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[async_trait]
impl CallbackHandler for LangSmithTracer {
    fn name(&self) -> &str {
        "langsmith_tracer"
    }

    async fn on_llm_start(
        &self,
        serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = LangSmithRun::new(
            run_id,
            "llm",
            LangSmithRunType::Llm,
            serde_json::json!({ "prompts": prompts }),
        );
        run.parent_run_id = parent_run_id;
        run.serialized = Some(serialized.clone());
        self.add_run(run);
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, Some(Value::String("completed".into())), None);
        Ok(())
    }

    async fn on_llm_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, None, Some(error.to_string()));
        Ok(())
    }

    async fn on_chain_start(
        &self,
        serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = LangSmithRun::new(run_id, "chain", LangSmithRunType::Chain, inputs.clone());
        run.parent_run_id = parent_run_id;
        run.serialized = Some(serialized.clone());
        self.add_run(run);
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, Some(outputs.clone()), None);
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, None, Some(error.to_string()));
        Ok(())
    }

    async fn on_tool_start(
        &self,
        serialized: &Value,
        input_str: &str,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = LangSmithRun::new(
            run_id,
            "tool",
            LangSmithRunType::Tool,
            serde_json::json!({ "input": input_str }),
        );
        run.parent_run_id = parent_run_id;
        run.serialized = Some(serialized.clone());
        self.add_run(run);
        Ok(())
    }

    async fn on_tool_end(
        &self,
        output: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(
            run_id,
            Some(serde_json::json!({ "output": output })),
            None,
        );
        Ok(())
    }

    async fn on_tool_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, None, Some(error.to_string()));
        Ok(())
    }

    async fn on_retriever_start(
        &self,
        serialized: &Value,
        query: &str,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = LangSmithRun::new(
            run_id,
            "retriever",
            LangSmithRunType::Retriever,
            serde_json::json!({ "query": query }),
        );
        run.parent_run_id = parent_run_id;
        run.serialized = Some(serialized.clone());
        self.add_run(run);
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        documents: &[Document],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(
            run_id,
            Some(serde_json::to_value(documents).unwrap_or_default()),
            None,
        );
        Ok(())
    }

    async fn on_retriever_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.finish_run(run_id, None, Some(error.to_string()));
        Ok(())
    }

    async fn on_chat_model_start(
        &self,
        serialized: &Value,
        messages: &[Vec<Message>],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = LangSmithRun::new(
            run_id,
            "chat_model",
            LangSmithRunType::Llm,
            serde_json::json!({ "messages": serde_json::to_value(messages).unwrap_or_default() }),
        );
        run.parent_run_id = parent_run_id;
        run.serialized = Some(serialized.clone());
        self.add_run(run);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LangSmithConfig {
        LangSmithConfig::new("test-api-key").with_project_name("test-project")
    }

    fn make_tracer() -> LangSmithTracer {
        LangSmithTracer::new(test_config())
    }

    #[test]
    fn test_config_defaults() {
        let config = LangSmithConfig::default();
        assert_eq!(config.api_url, "https://api.smith.langchain.com");
        assert_eq!(config.api_key, "");
        assert_eq!(config.project_name, "default");
        assert!(config.tags.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = LangSmithConfig::new("key123")
            .with_project_name("my-project")
            .with_api_url("https://custom.langsmith.com")
            .with_tags(vec!["prod".to_string(), "v2".to_string()]);

        assert_eq!(config.api_key, "key123");
        assert_eq!(config.project_name, "my-project");
        assert_eq!(config.api_url, "https://custom.langsmith.com");
        assert_eq!(config.tags, vec!["prod", "v2"]);
    }

    #[tokio::test]
    async fn test_llm_run_capture() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();
        let serialized = serde_json::json!({"name": "gpt-4"});

        tracer
            .on_llm_start(&serialized, &["Hello".to_string()], run_id, None)
            .await
            .unwrap();

        let runs = tracer.get_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run_id);
        assert_eq!(runs[0].run_type, LangSmithRunType::Llm);
        assert_eq!(runs[0].name, "llm");
        assert!(runs[0].end_time.is_none());
        assert!(runs[0].outputs.is_none());
    }

    #[tokio::test]
    async fn test_llm_run_end() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();

        tracer
            .on_llm_start(&Value::Null, &["test".to_string()], run_id, None)
            .await
            .unwrap();

        let result = LLMResult {
            generations: vec![],
            llm_output: None,
            run: None,
        };
        tracer.on_llm_end(&result, run_id, None).await.unwrap();

        let run = tracer.get_run(run_id).unwrap();
        assert!(run.end_time.is_some());
        assert!(run.outputs.is_some());
        assert!(run.error.is_none());
    }

    #[tokio::test]
    async fn test_chain_run_capture() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();
        let inputs = serde_json::json!({"question": "What is 2+2?"});

        tracer
            .on_chain_start(&Value::Null, &inputs, run_id, None)
            .await
            .unwrap();

        let runs = tracer.get_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_type, LangSmithRunType::Chain);
    }

    #[tokio::test]
    async fn test_tool_run_capture() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();

        tracer
            .on_tool_start(&Value::Null, "search query", run_id, None)
            .await
            .unwrap();

        tracer
            .on_tool_end("search result", run_id, None)
            .await
            .unwrap();

        let run = tracer.get_run(run_id).unwrap();
        assert_eq!(run.run_type, LangSmithRunType::Tool);
        assert!(run.end_time.is_some());
        assert!(run.outputs.is_some());
    }

    #[tokio::test]
    async fn test_parent_child_relationship() {
        let tracer = make_tracer();
        let parent_id = Uuid::new_v4();
        let child_id = Uuid::new_v4();

        tracer
            .on_chain_start(&Value::Null, &serde_json::json!({}), parent_id, None)
            .await
            .unwrap();

        tracer
            .on_tool_start(&Value::Null, "child input", child_id, Some(parent_id))
            .await
            .unwrap();

        let child = tracer.get_run(child_id).unwrap();
        assert_eq!(child.parent_run_id, Some(parent_id));

        let children = tracer.get_child_runs(parent_id);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, child_id);
    }

    #[tokio::test]
    async fn test_error_run_tracking() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();

        tracer
            .on_llm_start(&Value::Null, &["test".to_string()], run_id, None)
            .await
            .unwrap();

        tracer
            .on_llm_error("Rate limit exceeded", run_id, None)
            .await
            .unwrap();

        let run = tracer.get_run(run_id).unwrap();
        assert_eq!(run.error.as_deref(), Some("Rate limit exceeded"));
        assert!(run.end_time.is_some());
    }

    #[tokio::test]
    async fn test_timing_capture() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();

        tracer
            .on_chain_start(&Value::Null, &serde_json::json!({}), run_id, None)
            .await
            .unwrap();

        let run_before = tracer.get_run(run_id).unwrap();
        assert!(run_before.start_time > 0);
        assert!(run_before.end_time.is_none());

        tracer
            .on_chain_end(&serde_json::json!({"result": 42}), run_id, None)
            .await
            .unwrap();

        let run_after = tracer.get_run(run_id).unwrap();
        assert!(run_after.end_time.is_some());
        assert!(run_after.end_time.unwrap() >= run_after.start_time);
    }

    #[tokio::test]
    async fn test_json_export_format() {
        let tracer = make_tracer();
        let run_id = Uuid::new_v4();

        tracer
            .on_llm_start(&Value::Null, &["hello".to_string()], run_id, None)
            .await
            .unwrap();

        let result = LLMResult {
            generations: vec![],
            llm_output: None,
            run: None,
        };
        tracer.on_llm_end(&result, run_id, None).await.unwrap();

        let json = tracer.export_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);

        let payload = &arr[0];
        assert_eq!(payload["id"], run_id.to_string());
        assert_eq!(payload["run_type"], "llm");
        assert_eq!(payload["session_name"], "test-project");
        assert!(payload["start_time"].is_number());
        assert!(payload["end_time"].is_number());
        assert!(payload.get("outputs").is_some());
    }

    #[tokio::test]
    async fn test_batch_export() {
        let tracer = make_tracer();
        let completed_id = Uuid::new_v4();
        let pending_id = Uuid::new_v4();

        // Completed run
        tracer
            .on_chain_start(&Value::Null, &serde_json::json!({}), completed_id, None)
            .await
            .unwrap();
        tracer
            .on_chain_end(&serde_json::json!({"done": true}), completed_id, None)
            .await
            .unwrap();

        // Pending run (no end)
        tracer
            .on_tool_start(&Value::Null, "input", pending_id, None)
            .await
            .unwrap();

        let batch = tracer.export_batch_json();
        let post = batch["post"].as_array().unwrap();
        let patch = batch["patch"].as_array().unwrap();

        assert_eq!(post.len(), 1); // pending run
        assert_eq!(patch.len(), 1); // completed run
        assert_eq!(post[0]["id"], pending_id.to_string());
        assert_eq!(patch[0]["id"], completed_id.to_string());
    }

    #[tokio::test]
    async fn test_run_filtering_by_type() {
        let tracer = make_tracer();

        tracer
            .on_llm_start(&Value::Null, &["p".to_string()], Uuid::new_v4(), None)
            .await
            .unwrap();
        tracer
            .on_chain_start(&Value::Null, &serde_json::json!({}), Uuid::new_v4(), None)
            .await
            .unwrap();
        tracer
            .on_tool_start(&Value::Null, "t", Uuid::new_v4(), None)
            .await
            .unwrap();
        tracer
            .on_chain_start(&Value::Null, &serde_json::json!({}), Uuid::new_v4(), None)
            .await
            .unwrap();

        assert_eq!(tracer.get_runs_by_type(LangSmithRunType::Llm).len(), 1);
        assert_eq!(tracer.get_runs_by_type(LangSmithRunType::Chain).len(), 2);
        assert_eq!(tracer.get_runs_by_type(LangSmithRunType::Tool).len(), 1);
        assert_eq!(
            tracer.get_runs_by_type(LangSmithRunType::Retriever).len(),
            0
        );
    }

    #[tokio::test]
    async fn test_clear() {
        let tracer = make_tracer();

        tracer
            .on_llm_start(&Value::Null, &["p".to_string()], Uuid::new_v4(), None)
            .await
            .unwrap();
        assert_eq!(tracer.get_runs().len(), 1);

        tracer.clear();
        assert!(tracer.get_runs().is_empty());
    }

    #[tokio::test]
    async fn test_default_tags_applied() {
        let config = LangSmithConfig::new("key")
            .with_tags(vec!["env:test".to_string(), "version:1".to_string()]);
        let tracer = LangSmithTracer::new(config);

        tracer
            .on_llm_start(&Value::Null, &["p".to_string()], Uuid::new_v4(), None)
            .await
            .unwrap();

        let run = &tracer.get_runs()[0];
        assert!(run.tags.contains(&"env:test".to_string()));
        assert!(run.tags.contains(&"version:1".to_string()));
    }

    #[test]
    fn test_exporter_single_run() {
        let config = test_config();
        let run = LangSmithRun::new(
            Uuid::new_v4(),
            "test-run",
            LangSmithRunType::Chain,
            serde_json::json!({"key": "value"}),
        );

        let json = LangSmithExporter::run_to_json(&run, &config);
        assert_eq!(json["name"], "test-run");
        assert_eq!(json["run_type"], "chain");
        assert_eq!(json["session_name"], "test-project");
    }

    #[test]
    fn test_exporter_batch() {
        let config = test_config();
        let mut completed = LangSmithRun::new(
            Uuid::new_v4(),
            "done",
            LangSmithRunType::Llm,
            Value::Null,
        );
        completed.end_time = Some(now_millis());

        let pending = LangSmithRun::new(
            Uuid::new_v4(),
            "pending",
            LangSmithRunType::Tool,
            Value::Null,
        );

        let batch = LangSmithExporter::batch_to_json(&[completed, pending], &config);
        assert_eq!(batch["post"].as_array().unwrap().len(), 1);
        assert_eq!(batch["patch"].as_array().unwrap().len(), 1);
    }
}
