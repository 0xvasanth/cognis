//! Console tracer for debugging.
//!
//! Mirrors Python `langchain_core.tracers.stdout`.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::{CallbackHandler, ToolEndEvent, ToolErrorEvent, ToolStartEvent};
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

/// Colors for console output.
const COLOR_GREEN: &str = "\x1b[32m";
const COLOR_BLUE: &str = "\x1b[34m";
const COLOR_YELLOW: &str = "\x1b[33m";
const COLOR_RED: &str = "\x1b[31m";
const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_RESET: &str = "\x1b[0m";

/// A callback handler that formats events to the console with colors.
///
/// Displays a hierarchical view of chain, LLM, tool, and retriever runs
/// with indentation reflecting the call depth. Useful for local debugging.
pub struct ConsoleCallbackHandler {
    depth: AtomicUsize,
}

impl ConsoleCallbackHandler {
    pub fn new() -> Self {
        Self {
            depth: AtomicUsize::new(0),
        }
    }

    fn indent(&self) -> String {
        "  ".repeat(self.depth.load(Ordering::Relaxed))
    }

    fn print_header(&self, color: &str, title: &str, run_id: Uuid) {
        let indent = self.indent();
        println!(
            "{}{}{}[{}]{} [{}]",
            indent, COLOR_BOLD, color, title, COLOR_RESET, run_id
        );
    }
}

impl Default for ConsoleCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for ConsoleCallbackHandler {
    async fn on_chain_start(
        &self,
        _serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.print_header(COLOR_GREEN, "chain/start", run_id);
        let indent = self.indent();
        println!(
            "{}  inputs: {}",
            indent,
            serde_json::to_string_pretty(inputs).unwrap_or_default()
        );
        self.depth.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_GREEN, "chain/end", run_id);
        let indent = self.indent();
        println!(
            "{}  outputs: {}",
            indent,
            serde_json::to_string_pretty(outputs).unwrap_or_default()
        );
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_RED, "chain/error", run_id);
        let indent = self.indent();
        println!("{}  error: {}", indent, error);
        Ok(())
    }

    async fn on_llm_start(
        &self,
        _serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.print_header(COLOR_BLUE, "llm/start", run_id);
        let indent = self.indent();
        println!("{}  prompts: {:?}", indent, prompts);
        self.depth.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_BLUE, "llm/end", run_id);
        Ok(())
    }

    async fn on_llm_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_RED, "llm/error", run_id);
        let indent = self.indent();
        println!("{}  error: {}", indent, error);
        Ok(())
    }

    async fn on_chat_model_start(
        &self,
        _serialized: &Value,
        messages: &[Vec<Message>],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.print_header(COLOR_CYAN, "chat_model/start", run_id);
        let indent = self.indent();
        println!("{}  messages: {} batch(es)", indent, messages.len());
        self.depth.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_tool_start(&self, event: ToolStartEvent) -> Result<()> {
        self.print_header(COLOR_YELLOW, "tool/start", event.run_id);
        let indent = self.indent();
        println!("{}  input: {}", indent, event.input_str);
        self.depth.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_tool_end(&self, event: ToolEndEvent) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_YELLOW, "tool/end", event.run_id);
        let indent = self.indent();
        println!("{}  output: {}", indent, event.output_str);
        Ok(())
    }

    async fn on_tool_error(&self, event: ToolErrorEvent) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_RED, "tool/error", event.run_id);
        let indent = self.indent();
        println!("{}  error: {}", indent, event.error);
        Ok(())
    }

    async fn on_retriever_start(
        &self,
        _serialized: &Value,
        query: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.print_header(COLOR_CYAN, "retriever/start", run_id);
        let indent = self.indent();
        println!("{}  query: {}", indent, query);
        self.depth.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        documents: &[Document],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.depth.fetch_sub(1, Ordering::Relaxed);
        self.print_header(COLOR_CYAN, "retriever/end", run_id);
        let indent = self.indent();
        println!("{}  {} document(s)", indent, documents.len());
        Ok(())
    }
}
