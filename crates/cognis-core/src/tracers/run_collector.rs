//! Run collector tracer for testing and evaluation.
//!
//! Mirrors Python `langchain_core.tracers.run_collector`.

use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::{CallbackHandler, ToolEndEvent, ToolErrorEvent, ToolStartEvent};
use crate::documents::Document;
use crate::error::Result;
use crate::outputs::LLMResult;

use super::schemas::{Run, RunType};

/// Collects all runs into a list for later inspection.
///
/// Useful for testing, evaluation, and post-hoc analysis of run traces.
/// All runs are stored in an internal `Vec<Run>` protected by a `Mutex`.
pub struct RunCollectorCallbackHandler {
    runs: Mutex<Vec<Run>>,
}

impl RunCollectorCallbackHandler {
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(Vec::new()),
        }
    }

    /// Get a copy of all collected runs.
    pub fn get_runs(&self) -> Vec<Run> {
        self.runs.lock().unwrap().clone()
    }

    /// Clear all collected runs.
    pub fn clear(&self) {
        self.runs.lock().unwrap().clear();
    }

    fn add_run(&self, run: Run) {
        self.runs.lock().unwrap().push(run);
    }
}

impl Default for RunCollectorCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for RunCollectorCallbackHandler {
    async fn on_llm_start(
        &self,
        _serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = Run::new(
            run_id,
            "llm",
            RunType::Llm,
            serde_json::to_value(prompts).unwrap_or_default(),
        );
        run.parent_run_id = parent_run_id;
        self.add_run(run);
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.outputs = Some(Value::String("completed".into()));
        }
        Ok(())
    }

    async fn on_llm_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.error = Some(error.to_string());
        }
        Ok(())
    }

    async fn on_chain_start(
        &self,
        _serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = Run::new(run_id, "chain", RunType::Chain, inputs.clone());
        run.parent_run_id = parent_run_id;
        self.add_run(run);
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.outputs = Some(outputs.clone());
        }
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.error = Some(error.to_string());
        }
        Ok(())
    }

    async fn on_tool_start(&self, event: ToolStartEvent) -> Result<()> {
        let mut run = Run::new(
            event.run_id,
            "tool",
            RunType::Tool,
            Value::String(event.input_str.clone()),
        );
        run.parent_run_id = event.parent_run_id;
        self.add_run(run);
        Ok(())
    }

    async fn on_tool_end(&self, event: ToolEndEvent) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == event.run_id) {
            run.outputs = Some(Value::String(event.output_str.clone()));
        }
        Ok(())
    }

    async fn on_tool_error(&self, event: ToolErrorEvent) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == event.run_id) {
            run.error = Some(event.error.clone());
        }
        Ok(())
    }

    async fn on_retriever_start(
        &self,
        _serialized: &Value,
        query: &str,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut run = Run::new(
            run_id,
            "retriever",
            RunType::Retriever,
            Value::String(query.to_string()),
        );
        run.parent_run_id = parent_run_id;
        self.add_run(run);
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        documents: &[Document],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.outputs = Some(serde_json::to_value(documents).unwrap_or_default());
        }
        Ok(())
    }

    async fn on_retriever_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            run.error = Some(error.to_string());
        }
        Ok(())
    }
}
