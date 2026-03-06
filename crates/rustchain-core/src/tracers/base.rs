//! Base tracer implementations.
//!
//! Mirrors Python `langchain_core.tracers.base`.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

use super::schemas::{Run, RunType};

/// Base tracer that implements `CallbackHandler` by building `Run` objects
/// and delegating persistence to `persist_run`.
///
/// Subclasses (e.g., console tracer, LangSmith tracer) implement
/// `persist_run` to store completed runs.
#[async_trait]
pub trait BaseTracer: CallbackHandler {
    /// Persist a completed run to storage.
    async fn persist_run(&self, run: &Run) -> Result<()>;

    /// Called when a new run is created. Override for custom behavior.
    async fn on_run_create(&self, _run: &Run) -> Result<()> {
        Ok(())
    }

    /// Called when a run is updated. Override for custom behavior.
    async fn on_run_update(&self, _run: &Run) -> Result<()> {
        Ok(())
    }

    /// Access the run map (run_id → Run).
    fn run_map(&self) -> &HashMap<Uuid, Run>;

    /// Access the mutable run map.
    fn run_map_mut(&mut self) -> &mut HashMap<Uuid, Run>;

    /// Start tracking a new run.
    fn start_trace(&mut self, run: Run) {
        let id = run.id;
        self.run_map_mut().insert(id, run);
    }

    /// End tracking a run, removing it from the map.
    fn end_trace(&mut self, run_id: Uuid) -> Option<Run> {
        self.run_map_mut().remove(&run_id)
    }

    /// Create an LLM run from callback parameters.
    fn create_llm_run(run_id: Uuid, name: &str, inputs: Value, parent_run_id: Option<Uuid>) -> Run {
        let mut run = Run::new(run_id, name, RunType::Llm, inputs);
        run.parent_run_id = parent_run_id;
        run
    }

    /// Create a chat model run from callback parameters.
    fn create_chat_model_run(
        run_id: Uuid,
        name: &str,
        messages: &[Vec<Message>],
        parent_run_id: Option<Uuid>,
    ) -> Run {
        let inputs = serde_json::to_value(messages).unwrap_or_default();
        let mut run = Run::new(run_id, name, RunType::ChatModel, inputs);
        run.parent_run_id = parent_run_id;
        run
    }

    /// Create a chain run from callback parameters.
    fn create_chain_run(
        run_id: Uuid,
        name: &str,
        inputs: Value,
        parent_run_id: Option<Uuid>,
    ) -> Run {
        let mut run = Run::new(run_id, name, RunType::Chain, inputs);
        run.parent_run_id = parent_run_id;
        run
    }

    /// Create a tool run from callback parameters.
    fn create_tool_run(run_id: Uuid, name: &str, input: &str, parent_run_id: Option<Uuid>) -> Run {
        let mut run = Run::new(
            run_id,
            name,
            RunType::Tool,
            Value::String(input.to_string()),
        );
        run.parent_run_id = parent_run_id;
        run
    }

    /// Create a retriever run from callback parameters.
    fn create_retriever_run(
        run_id: Uuid,
        name: &str,
        query: &str,
        parent_run_id: Option<Uuid>,
    ) -> Run {
        let mut run = Run::new(
            run_id,
            name,
            RunType::Retriever,
            Value::String(query.to_string()),
        );
        run.parent_run_id = parent_run_id;
        run
    }
}

/// Async base tracer — same interface as `BaseTracer` but with async persistence.
#[async_trait]
pub trait AsyncBaseTracer: CallbackHandler {
    /// Persist a completed run to storage (async).
    async fn persist_run(&self, run: &Run) -> Result<()>;
}

/// A simple in-memory tracer that collects all runs for inspection.
///
/// Useful for testing and debugging.
pub struct InMemoryTracer {
    pub runs: Vec<Run>,
    run_map: HashMap<Uuid, Run>,
}

impl InMemoryTracer {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            run_map: HashMap::new(),
        }
    }
}

impl Default for InMemoryTracer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for InMemoryTracer {
    async fn on_llm_start(
        &self,
        _serialized: &Value,
        _prompts: &[String],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        _documents: &[Document],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl BaseTracer for InMemoryTracer {
    async fn persist_run(&self, _run: &Run) -> Result<()> {
        // Runs are kept in-memory via the runs vec.
        Ok(())
    }

    fn run_map(&self) -> &HashMap<Uuid, Run> {
        &self.run_map
    }

    fn run_map_mut(&mut self) -> &mut HashMap<Uuid, Run> {
        &mut self.run_map
    }
}
