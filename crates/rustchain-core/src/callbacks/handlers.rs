use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use super::base::CallbackHandler;
use crate::agents::{AgentAction, AgentFinish};
use crate::error::Result;
use crate::messages::ai::UsageMetadata;
use crate::outputs::LLMResult;

/// A callback handler that prints events to stdout.
///
/// Prints chain start/end events and agent actions/finishes.
pub struct StdOutCallbackHandler;

#[async_trait]
impl CallbackHandler for StdOutCallbackHandler {
    async fn on_chain_start(
        &self,
        _serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\n\x1b[1m> Entering new chain run ({})\x1b[0m",
            run_id
        );
        println!("Inputs: {}", serde_json::to_string_pretty(inputs).unwrap_or_default());
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\x1b[1m> Finished chain run ({})\x1b[0m",
            run_id
        );
        println!("Outputs: {}", serde_json::to_string_pretty(outputs).unwrap_or_default());
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\x1b[31m> Chain error ({}): {}\x1b[0m",
            run_id, error
        );
        Ok(())
    }

    async fn on_agent_action(
        &self,
        action: &AgentAction,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\x1b[36m> Agent action ({}): tool={}, input={}\x1b[0m",
            run_id, action.tool, action.tool_input
        );
        Ok(())
    }

    async fn on_agent_finish(
        &self,
        finish: &AgentFinish,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\x1b[32m> Agent finish ({}): {}\x1b[0m",
            run_id,
            serde_json::to_string_pretty(&finish.return_values).unwrap_or_default()
        );
        Ok(())
    }

    async fn on_llm_start(
        &self,
        _serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!(
            "\n\x1b[1m> LLM start ({}): {} prompt(s)\x1b[0m",
            run_id,
            prompts.len()
        );
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!("\n\x1b[1m> LLM end ({})\x1b[0m", run_id);
        Ok(())
    }
}

/// A callback handler that streams LLM tokens to stdout as they arrive.
///
/// Useful for real-time display of LLM output during streaming.
pub struct StreamingStdOutCallbackHandler;

#[async_trait]
impl CallbackHandler for StreamingStdOutCallbackHandler {
    async fn on_llm_new_token(
        &self,
        token: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        use std::io::Write;
        print!("{}", token);
        std::io::stdout().flush().ok();
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        println!();
        Ok(())
    }
}

/// A callback handler that writes events to a file.
///
/// Mirrors Python's `langchain_core.callbacks.file.FileCallbackHandler`.
/// All events are appended to the specified file path.
pub struct FileCallbackHandler {
    /// Path to the output file.
    pub file_path: PathBuf,
}

impl FileCallbackHandler {
    /// Creates a new `FileCallbackHandler` that writes to the given file path.
    pub fn new(file_path: impl Into<PathBuf>) -> Self {
        Self {
            file_path: file_path.into(),
        }
    }

    /// Appends a line to the output file.
    fn write_line(&self, line: &str) {
        use std::fs::OpenOptions;
        use std::io::Write;

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            let _ = writeln!(file, "{}", line);
        }
    }
}

#[async_trait]
impl CallbackHandler for FileCallbackHandler {
    async fn on_chain_start(
        &self,
        _serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[chain/start] [{}] Entering chain run with inputs: {}",
            run_id,
            serde_json::to_string(inputs).unwrap_or_default()
        ));
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[chain/end] [{}] Finished chain run with outputs: {}",
            run_id,
            serde_json::to_string(outputs).unwrap_or_default()
        ));
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[chain/error] [{}] Chain error: {}",
            run_id, error
        ));
        Ok(())
    }

    async fn on_llm_start(
        &self,
        _serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[llm/start] [{}] Entering LLM run with {} prompt(s)",
            run_id,
            prompts.len()
        ));
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!("[llm/end] [{}] Finished LLM run", run_id));
        Ok(())
    }

    async fn on_llm_new_token(
        &self,
        token: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!("[llm/new_token] [{}] {}", run_id, token));
        Ok(())
    }

    async fn on_tool_start(
        &self,
        _serialized: &Value,
        input_str: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[tool/start] [{}] Entering tool run with input: {}",
            run_id, input_str
        ));
        Ok(())
    }

    async fn on_tool_end(
        &self,
        output: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.write_line(&format!(
            "[tool/end] [{}] Finished tool run with output: {}",
            run_id, output
        ));
        Ok(())
    }
}

/// Summary of token usage across LLM calls.
#[derive(Debug, Clone)]
pub struct UsageSummary {
    /// Total input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Total output (completion) tokens generated.
    pub output_tokens: u64,
    /// Total tokens (input + output).
    pub total_tokens: u64,
    /// Number of LLM calls tracked.
    pub call_count: u64,
}

/// A callback handler that tracks cumulative token usage across LLM calls.
///
/// Mirrors Python's `langchain_core.callbacks.usage.UsageMetadataCallbackHandler`.
/// Thread-safe via atomic counters for aggregate totals and a `Mutex`-protected
/// `HashMap` for per-model `UsageMetadata` tracking.
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// use rustchain_core::callbacks::{UsageMetadataCallbackHandler, CallbackManager};
///
/// let handler = Arc::new(UsageMetadataCallbackHandler::new());
/// let manager = CallbackManager::new(vec![handler.clone()], None);
///
/// // ... invoke LLM calls through the manager ...
///
/// // Get aggregate totals
/// println!("Total tokens: {}", handler.total_tokens());
/// println!("Prompt tokens: {}", handler.prompt_tokens());
/// println!("Completion tokens: {}", handler.completion_tokens());
///
/// // Get per-model usage
/// for entry in handler.get_usage() {
///     println!("{:?}", entry);
/// }
///
/// // Get per-model map (model_name -> UsageMetadata)
/// let usage_map = handler.usage_metadata();
/// ```
pub struct UsageMetadataCallbackHandler {
    /// Cumulative input (prompt) tokens.
    total_input_tokens: AtomicU64,
    /// Cumulative output (completion) tokens.
    total_output_tokens: AtomicU64,
    /// Cumulative total tokens.
    total_tokens_counter: AtomicU64,
    /// Number of LLM calls recorded.
    call_count: AtomicU64,
    /// Per-model usage metadata, keyed by model name.
    /// Protected by a Mutex for thread-safe interior mutability.
    per_model_usage: Mutex<HashMap<String, UsageMetadata>>,
}

impl UsageMetadataCallbackHandler {
    /// Creates a new `UsageMetadataCallbackHandler` with all counters at zero.
    pub fn new() -> Self {
        Self {
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
            total_tokens_counter: AtomicU64::new(0),
            call_count: AtomicU64::new(0),
            per_model_usage: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a snapshot of the current aggregate usage statistics.
    pub fn get_summary(&self) -> UsageSummary {
        UsageSummary {
            input_tokens: self.total_input_tokens.load(Ordering::Relaxed),
            output_tokens: self.total_output_tokens.load(Ordering::Relaxed),
            total_tokens: self.total_tokens_counter.load(Ordering::Relaxed),
            call_count: self.call_count.load(Ordering::Relaxed),
        }
    }

    /// Returns all tracked `UsageMetadata` entries (one per model).
    pub fn get_usage(&self) -> Vec<UsageMetadata> {
        let guard = self.per_model_usage.lock().unwrap();
        guard.values().cloned().collect()
    }

    /// Returns a clone of the per-model usage map.
    ///
    /// Keys are model names, values are cumulative `UsageMetadata` for that model.
    pub fn usage_metadata(&self) -> HashMap<String, UsageMetadata> {
        let guard = self.per_model_usage.lock().unwrap();
        guard.clone()
    }

    /// Returns the cumulative total token count across all LLM calls.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens_counter.load(Ordering::Relaxed)
    }

    /// Returns the cumulative prompt (input) token count across all LLM calls.
    pub fn prompt_tokens(&self) -> u64 {
        self.total_input_tokens.load(Ordering::Relaxed)
    }

    /// Returns the cumulative completion (output) token count across all LLM calls.
    pub fn completion_tokens(&self) -> u64 {
        self.total_output_tokens.load(Ordering::Relaxed)
    }

    /// Returns the number of LLM calls recorded.
    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    /// Resets all counters and per-model usage to zero.
    pub fn reset(&self) {
        self.total_input_tokens.store(0, Ordering::Relaxed);
        self.total_output_tokens.store(0, Ordering::Relaxed);
        self.total_tokens_counter.store(0, Ordering::Relaxed);
        self.call_count.store(0, Ordering::Relaxed);
        let mut guard = self.per_model_usage.lock().unwrap();
        guard.clear();
    }

    /// Extract usage data from `llm_output` JSON (the `token_usage` sub-object).
    fn extract_from_llm_output(&self, llm_output: &HashMap<String, Value>) {
        if let Some(token_usage) = llm_output.get("token_usage") {
            let prompt = token_usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let completion = token_usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total = token_usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            if prompt > 0 {
                self.total_input_tokens.fetch_add(prompt, Ordering::Relaxed);
            }
            if completion > 0 {
                self.total_output_tokens
                    .fetch_add(completion, Ordering::Relaxed);
            }
            if total > 0 {
                self.total_tokens_counter
                    .fetch_add(total, Ordering::Relaxed);
            }

            // Build a UsageMetadata and store per-model
            let model_name = llm_output
                .get("model_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let usage = UsageMetadata::new(prompt, completion, total);
            let mut guard = self.per_model_usage.lock().unwrap();
            guard
                .entry(model_name)
                .and_modify(|existing| *existing = existing.add(&usage))
                .or_insert(usage);
        }
    }

    /// Extract usage data from an `AIMessage.usage_metadata` field when available
    /// in `llm_output` as a serialized structure.
    fn extract_usage_metadata(
        &self,
        usage: &UsageMetadata,
        model_name: Option<&str>,
    ) {
        self.total_input_tokens
            .fetch_add(usage.input_tokens, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(usage.output_tokens, Ordering::Relaxed);
        self.total_tokens_counter
            .fetch_add(usage.total_tokens, Ordering::Relaxed);

        let name = model_name.unwrap_or("unknown").to_string();
        let mut guard = self.per_model_usage.lock().unwrap();
        guard
            .entry(name)
            .and_modify(|existing| *existing = existing.add(usage))
            .or_insert_with(|| usage.clone());
    }
}

impl Default for UsageMetadataCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for UsageMetadataCallbackHandler {
    async fn on_llm_end(
        &self,
        response: &LLMResult,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        self.call_count.fetch_add(1, Ordering::Relaxed);

        // Try to extract UsageMetadata from llm_output["usage_metadata"] (structured)
        let mut found_structured = false;
        if let Some(llm_output) = &response.llm_output {
            if let Some(usage_val) = llm_output.get("usage_metadata") {
                if let Ok(usage) = serde_json::from_value::<UsageMetadata>(usage_val.clone()) {
                    let model_name = llm_output
                        .get("model_name")
                        .and_then(|v| v.as_str());
                    self.extract_usage_metadata(&usage, model_name);
                    found_structured = true;
                }
            }
        }

        // Fallback: extract from llm_output["token_usage"] (legacy format)
        if !found_structured {
            if let Some(llm_output) = &response.llm_output {
                self.extract_from_llm_output(llm_output);
            }
        }

        Ok(())
    }
}
