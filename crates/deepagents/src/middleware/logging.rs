//! Structured logging middleware — provides detailed logging for all agent
//! operations including model calls, tool invocations, and state changes.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::middleware::{AgentState, Middleware, Result};

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity level for log entries. Ordered from most to least verbose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogLevel {
    /// Fine-grained diagnostic information.
    Trace = 0,
    /// Debugging information.
    Debug = 1,
    /// Informational messages.
    Info = 2,
    /// Potentially harmful situations.
    Warn = 3,
    /// Error events that might still allow the agent to continue.
    Error = 4,
}

impl LogLevel {
    /// Return the string label used in formatted output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

impl PartialOrd for LogLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LogLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// LogFormat
// ---------------------------------------------------------------------------

/// Output format for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Single-line text: `[timestamp] LEVEL event key=value ...`
    Text,
    /// Machine-readable JSON, one object per entry.
    Json,
    /// Multi-line, human-friendly format.
    Pretty,
}

// ---------------------------------------------------------------------------
// LogDestination & LogSink
// ---------------------------------------------------------------------------

/// Where log entries are written.
pub enum LogDestination {
    /// Store entries in memory (useful for testing).
    InMemory,
    /// Write to standard error.
    Stderr,
    /// Append to a file.
    File(PathBuf),
    /// Delegate to a user-supplied sink.
    Custom(Arc<dyn LogSink + Send + Sync>),
}

impl fmt::Debug for LogDestination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InMemory => write!(f, "InMemory"),
            Self::Stderr => write!(f, "Stderr"),
            Self::File(p) => write!(f, "File({p:?})"),
            Self::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

/// Trait for custom log sinks.
pub trait LogSink {
    /// Write a single log entry.
    fn write_log(&self, entry: &LogEntry);
}

// ---------------------------------------------------------------------------
// LogEntry & LogEvent
// ---------------------------------------------------------------------------

/// A single structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// ISO-8601 timestamp string.
    pub timestamp: String,
    /// Severity level.
    pub level: LogLevel,
    /// The event that triggered this log entry.
    pub event: LogEvent,
    /// Human-readable description.
    pub message: String,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// Duration of the operation (for "end" events).
    pub duration: Option<Duration>,
}

/// Structured event variants.
#[derive(Debug, Clone)]
pub enum LogEvent {
    /// A model call is about to start.
    ModelCallStart {
        /// Model identifier.
        model: String,
        /// Estimated input token count, if available.
        input_tokens: Option<usize>,
    },
    /// A model call has completed.
    ModelCallEnd {
        /// Model identifier.
        model: String,
        /// Output token count, if available.
        output_tokens: Option<usize>,
        /// Wall-clock duration.
        duration: Duration,
    },
    /// A tool invocation is about to start.
    ToolCallStart {
        /// Tool name.
        tool: String,
        /// Serialized input passed to the tool.
        input: String,
    },
    /// A tool invocation has completed.
    ToolCallEnd {
        /// Tool name.
        tool: String,
        /// Serialized output from the tool.
        output: String,
        /// Wall-clock duration.
        duration: Duration,
    },
    /// A state key has changed.
    StateChange {
        /// The key that changed.
        key: String,
        /// Previous value (if any).
        old_value: Option<Value>,
        /// New value.
        new_value: Value,
    },
    /// An error occurred.
    Error {
        /// Component that produced the error.
        source: String,
        /// Error description.
        message: String,
    },
    /// An agent run started.
    AgentStart {
        /// Name of the agent.
        agent_name: String,
    },
    /// An agent run ended.
    AgentEnd {
        /// Name of the agent.
        agent_name: String,
        /// Total wall-clock duration of the agent run.
        total_duration: Duration,
    },
}

impl LogEvent {
    /// Short identifier for the event variant.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ModelCallStart { .. } => "ModelCallStart",
            Self::ModelCallEnd { .. } => "ModelCallEnd",
            Self::ToolCallStart { .. } => "ToolCallStart",
            Self::ToolCallEnd { .. } => "ToolCallEnd",
            Self::StateChange { .. } => "StateChange",
            Self::Error { .. } => "Error",
            Self::AgentStart { .. } => "AgentStart",
            Self::AgentEnd { .. } => "AgentEnd",
        }
    }
}

// ---------------------------------------------------------------------------
// LoggingConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`LoggingMiddleware`].
pub struct LoggingConfig {
    /// Minimum severity to log.
    pub level: LogLevel,
    /// Output format.
    pub format: LogFormat,
    /// Whether to include ISO-8601 timestamps.
    pub include_timestamps: bool,
    /// Whether to include model/tool inputs in log entries.
    pub include_inputs: bool,
    /// Whether to include model/tool outputs in log entries.
    pub include_outputs: bool,
    /// Maximum content length before truncation.
    pub max_content_length: usize,
    /// Regex patterns whose matches will be replaced with `[REDACTED]`.
    pub redact_patterns: Vec<String>,
    /// Where to write log entries.
    pub log_destination: LogDestination,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Text,
            include_timestamps: true,
            include_inputs: true,
            include_outputs: true,
            max_content_length: 500,
            redact_patterns: Vec::new(),
            log_destination: LogDestination::Stderr,
        }
    }
}

impl fmt::Debug for LoggingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingConfig")
            .field("level", &self.level)
            .field("format", &self.format)
            .field("include_timestamps", &self.include_timestamps)
            .field("include_inputs", &self.include_inputs)
            .field("include_outputs", &self.include_outputs)
            .field("max_content_length", &self.max_content_length)
            .field("redact_patterns", &self.redact_patterns)
            .field("log_destination", &self.log_destination)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ContentRedactor
// ---------------------------------------------------------------------------

/// Utility for redacting sensitive content from strings.
pub struct ContentRedactor;

impl ContentRedactor {
    /// Replace all matches of the given regex patterns with `[REDACTED]`.
    ///
    /// Invalid regex patterns are silently skipped.
    pub fn redact(content: &str, patterns: &[String]) -> String {
        let mut result = content.to_string();
        for pat in patterns {
            if let Ok(re) = regex::Regex::new(pat) {
                result = re.replace_all(&result, "[REDACTED]").into_owned();
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Truncate `content` to `max_len` characters, appending "..." if truncated.
pub fn truncate(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        let mut s = content[..max_len].to_string();
        s.push_str("...");
        s
    }
}

/// Format a [`LogEntry`] according to the given [`LogFormat`].
pub fn format_entry(entry: &LogEntry, format: LogFormat) -> String {
    match format {
        LogFormat::Text => format_text(entry),
        LogFormat::Json => format_json(entry),
        LogFormat::Pretty => format_pretty(entry),
    }
}

fn format_text(entry: &LogEntry) -> String {
    let mut parts: Vec<String> = Vec::new();

    if !entry.timestamp.is_empty() {
        parts.push(format!("[{}]", entry.timestamp));
    }
    parts.push(entry.level.as_str().to_string());
    parts.push(entry.event.name().to_string());

    // Append event-specific key=value pairs.
    match &entry.event {
        LogEvent::ModelCallStart { model, input_tokens } => {
            parts.push(format!("model={model}"));
            if let Some(t) = input_tokens {
                parts.push(format!("input_tokens={t}"));
            }
        }
        LogEvent::ModelCallEnd { model, output_tokens, duration } => {
            parts.push(format!("model={model}"));
            if let Some(t) = output_tokens {
                parts.push(format!("output_tokens={t}"));
            }
            parts.push(format!("duration={duration:?}"));
        }
        LogEvent::ToolCallStart { tool, input } => {
            parts.push(format!("tool={tool}"));
            if !input.is_empty() {
                parts.push(format!("input={input}"));
            }
        }
        LogEvent::ToolCallEnd { tool, output, duration } => {
            parts.push(format!("tool={tool}"));
            if !output.is_empty() {
                parts.push(format!("output={output}"));
            }
            parts.push(format!("duration={duration:?}"));
        }
        LogEvent::StateChange { key, .. } => {
            parts.push(format!("key={key}"));
        }
        LogEvent::Error { source, message } => {
            parts.push(format!("source={source}"));
            parts.push(format!("message={message}"));
        }
        LogEvent::AgentStart { agent_name } => {
            parts.push(format!("agent={agent_name}"));
        }
        LogEvent::AgentEnd { agent_name, total_duration } => {
            parts.push(format!("agent={agent_name}"));
            parts.push(format!("total_duration={total_duration:?}"));
        }
    }

    if !entry.message.is_empty() {
        parts.push(format!("msg=\"{}\"", entry.message));
    }

    parts.join(" ")
}

fn format_json(entry: &LogEntry) -> String {
    let mut map = serde_json::Map::new();
    map.insert("timestamp".into(), Value::String(entry.timestamp.clone()));
    map.insert("level".into(), Value::String(entry.level.as_str().to_string()));
    map.insert("event".into(), Value::String(entry.event.name().to_string()));
    map.insert("message".into(), Value::String(entry.message.clone()));

    if let Some(d) = entry.duration {
        map.insert("duration_ms".into(), Value::Number(serde_json::Number::from(d.as_millis() as u64)));
    }

    // Event-specific fields.
    match &entry.event {
        LogEvent::ModelCallStart { model, input_tokens } => {
            map.insert("model".into(), Value::String(model.clone()));
            if let Some(t) = input_tokens {
                map.insert("input_tokens".into(), Value::Number((*t).into()));
            }
        }
        LogEvent::ModelCallEnd { model, output_tokens, duration } => {
            map.insert("model".into(), Value::String(model.clone()));
            if let Some(t) = output_tokens {
                map.insert("output_tokens".into(), Value::Number((*t).into()));
            }
            map.insert("duration_ms".into(), Value::Number(serde_json::Number::from(duration.as_millis() as u64)));
        }
        LogEvent::ToolCallStart { tool, input } => {
            map.insert("tool".into(), Value::String(tool.clone()));
            map.insert("input".into(), Value::String(input.clone()));
        }
        LogEvent::ToolCallEnd { tool, output, duration } => {
            map.insert("tool".into(), Value::String(tool.clone()));
            map.insert("output".into(), Value::String(output.clone()));
            map.insert("duration_ms".into(), Value::Number(serde_json::Number::from(duration.as_millis() as u64)));
        }
        LogEvent::StateChange { key, old_value, new_value } => {
            map.insert("key".into(), Value::String(key.clone()));
            if let Some(old) = old_value {
                map.insert("old_value".into(), old.clone());
            }
            map.insert("new_value".into(), new_value.clone());
        }
        LogEvent::Error { source, message } => {
            map.insert("error_source".into(), Value::String(source.clone()));
            map.insert("error_message".into(), Value::String(message.clone()));
        }
        LogEvent::AgentStart { agent_name } => {
            map.insert("agent_name".into(), Value::String(agent_name.clone()));
        }
        LogEvent::AgentEnd { agent_name, total_duration } => {
            map.insert("agent_name".into(), Value::String(agent_name.clone()));
            map.insert("total_duration_ms".into(), Value::Number(serde_json::Number::from(total_duration.as_millis() as u64)));
        }
    }

    // Merge metadata.
    for (k, v) in &entry.metadata {
        map.insert(k.clone(), v.clone());
    }

    serde_json::to_string(&Value::Object(map)).unwrap_or_default()
}

fn format_pretty(entry: &LogEntry) -> String {
    let mut lines: Vec<String> = Vec::new();

    let level_tag = match entry.level {
        LogLevel::Trace => "[TRACE]",
        LogLevel::Debug => "[DEBUG]",
        LogLevel::Info  => "[ INFO]",
        LogLevel::Warn  => "[ WARN]",
        LogLevel::Error => "[ERROR]",
    };

    let header = if entry.timestamp.is_empty() {
        format!("{level_tag} {}", entry.event.name())
    } else {
        format!("{} {level_tag} {}", entry.timestamp, entry.event.name())
    };
    lines.push(header);

    if !entry.message.is_empty() {
        lines.push(format!("  {}", entry.message));
    }

    match &entry.event {
        LogEvent::ModelCallStart { model, input_tokens } => {
            lines.push(format!("  model: {model}"));
            if let Some(t) = input_tokens {
                lines.push(format!("  input_tokens: {t}"));
            }
        }
        LogEvent::ModelCallEnd { model, output_tokens, duration } => {
            lines.push(format!("  model: {model}"));
            if let Some(t) = output_tokens {
                lines.push(format!("  output_tokens: {t}"));
            }
            lines.push(format!("  duration: {duration:?}"));
        }
        LogEvent::ToolCallStart { tool, input } => {
            lines.push(format!("  tool: {tool}"));
            if !input.is_empty() {
                lines.push(format!("  input: {input}"));
            }
        }
        LogEvent::ToolCallEnd { tool, output, duration } => {
            lines.push(format!("  tool: {tool}"));
            if !output.is_empty() {
                lines.push(format!("  output: {output}"));
            }
            lines.push(format!("  duration: {duration:?}"));
        }
        LogEvent::StateChange { key, old_value, new_value } => {
            lines.push(format!("  key: {key}"));
            if let Some(old) = old_value {
                lines.push(format!("  old: {old}"));
            }
            lines.push(format!("  new: {new_value}"));
        }
        LogEvent::Error { source, message } => {
            lines.push(format!("  source: {source}"));
            lines.push(format!("  message: {message}"));
        }
        LogEvent::AgentStart { agent_name } => {
            lines.push(format!("  agent: {agent_name}"));
        }
        LogEvent::AgentEnd { agent_name, total_duration } => {
            lines.push(format!("  agent: {agent_name}"));
            lines.push(format!("  total_duration: {total_duration:?}"));
        }
    }

    if let Some(d) = entry.duration {
        // Only add if not already present from event-specific rendering.
        if !matches!(entry.event, LogEvent::ModelCallEnd { .. } | LogEvent::ToolCallEnd { .. } | LogEvent::AgentEnd { .. }) {
            lines.push(format!("  duration: {d:?}"));
        }
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// LoggingMiddleware
// ---------------------------------------------------------------------------

/// Middleware that emits structured logs for model calls, tool invocations,
/// and state changes.
pub struct LoggingMiddleware {
    config: LoggingConfig,
    logs: Arc<Mutex<Vec<LogEntry>>>,
}

impl LoggingMiddleware {
    /// Create a new logging middleware with the given configuration.
    pub fn new(config: LoggingConfig) -> Self {
        Self {
            config,
            logs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return all stored log entries (only useful with [`LogDestination::InMemory`]).
    pub async fn get_logs(&self) -> Vec<LogEntry> {
        self.logs.lock().await.clone()
    }

    /// Clear stored log entries.
    pub async fn clear_logs(&self) {
        self.logs.lock().await.clear();
    }

    /// Generate a current timestamp string.
    fn now_timestamp(&self) -> String {
        if self.config.include_timestamps {
            // Use a simple but deterministic format based on SystemTime.
            let now = std::time::SystemTime::now();
            let dur = now
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            // Format as seconds.millis — lightweight without chrono dependency.
            let secs = dur.as_secs();
            let millis = dur.subsec_millis();
            format!("{secs}.{millis:03}")
        } else {
            String::new()
        }
    }

    /// Process content through truncation and redaction.
    fn process_content(&self, content: &str) -> String {
        let truncated = truncate(content, self.config.max_content_length);
        if self.config.redact_patterns.is_empty() {
            truncated
        } else {
            ContentRedactor::redact(&truncated, &self.config.redact_patterns)
        }
    }

    /// Emit a log entry: format, write to destination, and store if InMemory.
    async fn emit(&self, entry: LogEntry) {
        if entry.level < self.config.level {
            return;
        }

        let formatted = format_entry(&entry, self.config.format);

        match &self.config.log_destination {
            LogDestination::InMemory => {
                self.logs.lock().await.push(entry);
            }
            LogDestination::Stderr => {
                eprintln!("{formatted}");
            }
            LogDestination::File(path) => {
                // Best-effort append.
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(f, "{formatted}");
                }
            }
            LogDestination::Custom(sink) => {
                sink.write_log(&entry);
            }
        }
    }
}

#[async_trait]
impl Middleware for LoggingMiddleware {
    fn name(&self) -> &str {
        "logging"
    }

    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let model_name = state
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let input_tokens = state
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let input_preview = if self.config.include_inputs {
            let msgs = state
                .get("messages")
                .map(|v| v.to_string())
                .unwrap_or_default();
            self.process_content(&msgs)
        } else {
            String::new()
        };

        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::ModelCallStart {
                model: model_name.clone(),
                input_tokens,
            },
            message: format!("Starting model call to {model_name}"),
            metadata: if input_preview.is_empty() {
                HashMap::new()
            } else {
                let mut m = HashMap::new();
                m.insert("input_preview".to_string(), Value::String(input_preview));
                m
            },
            duration: None,
        };

        self.emit(entry).await;
        Ok(())
    }

    async fn after_model(&self, state: &mut AgentState) -> Result<()> {
        let model_name = state
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let output_tokens = state
            .get("usage")
            .and_then(|u| u.get("output_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let output_preview = if self.config.include_outputs {
            let msgs = state.get("messages").and_then(|v| v.as_array());
            if let Some(arr) = msgs {
                arr.last()
                    .map(|v| self.process_content(&v.to_string()))
                    .unwrap_or_default()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::ModelCallEnd {
                model: model_name.clone(),
                output_tokens,
                duration: Duration::from_millis(0), // Actual timing would require wrapping the call.
            },
            message: format!("Completed model call to {model_name}"),
            metadata: if output_preview.is_empty() {
                HashMap::new()
            } else {
                let mut m = HashMap::new();
                m.insert("output_preview".to_string(), Value::String(output_preview));
                m
            },
            duration: None,
        };

        self.emit(entry).await;
        Ok(())
    }

    async fn before_tool(&self, _state: &mut AgentState, tool_name: &str) -> Result<()> {
        let input = if self.config.include_inputs {
            self.process_content(tool_name)
        } else {
            String::new()
        };

        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::ToolCallStart {
                tool: tool_name.to_string(),
                input: input.clone(),
            },
            message: format!("Starting tool call: {tool_name}"),
            metadata: HashMap::new(),
            duration: None,
        };

        self.emit(entry).await;
        Ok(())
    }

    async fn after_tool(
        &self,
        _state: &mut AgentState,
        tool_name: &str,
        result: &str,
    ) -> Result<()> {
        let output = if self.config.include_outputs {
            self.process_content(result)
        } else {
            String::new()
        };

        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::ToolCallEnd {
                tool: tool_name.to_string(),
                output: output.clone(),
                duration: Duration::from_millis(0),
            },
            message: format!("Completed tool call: {tool_name}"),
            metadata: HashMap::new(),
            duration: None,
        };

        self.emit(entry).await;
        Ok(())
    }
}

// Helper for creating log entries in tests or external code.
impl LoggingMiddleware {
    /// Log an error event.
    pub async fn log_error(&self, source: &str, message: &str) {
        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Error,
            event: LogEvent::Error {
                source: source.to_string(),
                message: message.to_string(),
            },
            message: message.to_string(),
            metadata: HashMap::new(),
            duration: None,
        };
        self.emit(entry).await;
    }

    /// Log a state change event.
    pub async fn log_state_change(
        &self,
        key: &str,
        old_value: Option<Value>,
        new_value: Value,
    ) {
        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Debug,
            event: LogEvent::StateChange {
                key: key.to_string(),
                old_value,
                new_value,
            },
            message: format!("State key '{key}' changed"),
            metadata: HashMap::new(),
            duration: None,
        };
        self.emit(entry).await;
    }

    /// Log an agent start event.
    pub async fn log_agent_start(&self, agent_name: &str) {
        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::AgentStart {
                agent_name: agent_name.to_string(),
            },
            message: format!("Agent '{agent_name}' started"),
            metadata: HashMap::new(),
            duration: None,
        };
        self.emit(entry).await;
    }

    /// Log an agent end event.
    pub async fn log_agent_end(&self, agent_name: &str, total_duration: Duration) {
        let entry = LogEntry {
            timestamp: self.now_timestamp(),
            level: LogLevel::Info,
            event: LogEvent::AgentEnd {
                agent_name: agent_name.to_string(),
                total_duration,
            },
            message: format!("Agent '{agent_name}' finished"),
            metadata: HashMap::new(),
            duration: Some(total_duration),
        };
        self.emit(entry).await;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    /// Helper: create an InMemory logging middleware with defaults overridden.
    fn in_memory_middleware(config: LoggingConfig) -> LoggingMiddleware {
        LoggingMiddleware::new(config)
    }

    fn default_in_memory_config() -> LoggingConfig {
        LoggingConfig {
            log_destination: LogDestination::InMemory,
            ..Default::default()
        }
    }

    // Test 1: Config defaults
    #[test]
    fn test_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert_eq!(config.format, LogFormat::Text);
        assert!(config.include_timestamps);
        assert!(config.include_inputs);
        assert!(config.include_outputs);
        assert_eq!(config.max_content_length, 500);
        assert!(config.redact_patterns.is_empty());
        assert!(matches!(config.log_destination, LogDestination::Stderr));
    }

    // Test 2: Log entry creation
    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry {
            timestamp: "1234567890.000".to_string(),
            level: LogLevel::Info,
            event: LogEvent::ModelCallStart {
                model: "gpt-4".to_string(),
                input_tokens: Some(100),
            },
            message: "Starting model call".to_string(),
            metadata: HashMap::new(),
            duration: None,
        };

        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.event.name(), "ModelCallStart");
        assert!(entry.duration.is_none());
    }

    // Test 3: Text format output
    #[test]
    fn test_text_format() {
        let entry = LogEntry {
            timestamp: "2026-03-07T10:00:00Z".to_string(),
            level: LogLevel::Info,
            event: LogEvent::ModelCallStart {
                model: "gpt-4".to_string(),
                input_tokens: None,
            },
            message: String::new(),
            metadata: HashMap::new(),
            duration: None,
        };

        let text = format_entry(&entry, LogFormat::Text);
        assert!(text.contains("[2026-03-07T10:00:00Z]"));
        assert!(text.contains("INFO"));
        assert!(text.contains("ModelCallStart"));
        assert!(text.contains("model=gpt-4"));
    }

    // Test 4: JSON format output
    #[test]
    fn test_json_format() {
        let entry = LogEntry {
            timestamp: "2026-03-07T10:00:00Z".to_string(),
            level: LogLevel::Warn,
            event: LogEvent::ToolCallStart {
                tool: "calculator".to_string(),
                input: "2+2".to_string(),
            },
            message: "calling tool".to_string(),
            metadata: HashMap::new(),
            duration: None,
        };

        let json_str = format_entry(&entry, LogFormat::Json);
        let parsed: Value = serde_json::from_str(&json_str).expect("valid JSON");
        assert_eq!(parsed["level"], "WARN");
        assert_eq!(parsed["event"], "ToolCallStart");
        assert_eq!(parsed["tool"], "calculator");
        assert_eq!(parsed["input"], "2+2");
        assert_eq!(parsed["timestamp"], "2026-03-07T10:00:00Z");
    }

    // Test 5: Pretty format output
    #[test]
    fn test_pretty_format() {
        let entry = LogEntry {
            timestamp: "2026-03-07T10:00:00Z".to_string(),
            level: LogLevel::Error,
            event: LogEvent::Error {
                source: "model".to_string(),
                message: "timeout".to_string(),
            },
            message: "An error occurred".to_string(),
            metadata: HashMap::new(),
            duration: None,
        };

        let pretty = format_entry(&entry, LogFormat::Pretty);
        assert!(pretty.contains("[ERROR]"));
        assert!(pretty.contains("source: model"));
        assert!(pretty.contains("message: timeout"));
        assert!(pretty.contains("An error occurred"));
    }

    // Test 6: Content redaction (API key pattern)
    #[test]
    fn test_content_redaction_api_key() {
        let content = "Authorization: Bearer sk-abc123secret key=sk-xyz789";
        let patterns = vec!["sk-[a-zA-Z0-9]+".to_string()];
        let redacted = ContentRedactor::redact(content, &patterns);
        assert!(!redacted.contains("sk-abc123secret"));
        assert!(!redacted.contains("sk-xyz789"));
        assert!(redacted.contains("[REDACTED]"));
        assert!(redacted.contains("Authorization: Bearer"));
    }

    // Test 7: Content truncation
    #[test]
    fn test_content_truncation() {
        let short = "hello";
        assert_eq!(truncate(short, 500), "hello");

        let long = "a".repeat(600);
        let truncated = truncate(&long, 500);
        assert_eq!(truncated.len(), 503); // 500 + "..."
        assert!(truncated.ends_with("..."));
    }

    // Test 8: InMemory log storage
    #[tokio::test]
    async fn test_in_memory_storage() {
        let mw = in_memory_middleware(default_in_memory_config());
        let mut state = json!({"messages": []});

        mw.before_model(&mut state).await.unwrap();
        mw.after_model(&mut state).await.unwrap();

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].event.name(), "ModelCallStart");
        assert_eq!(logs[1].event.name(), "ModelCallEnd");
    }

    // Test 9: Clear logs
    #[tokio::test]
    async fn test_clear_logs() {
        let mw = in_memory_middleware(default_in_memory_config());
        let mut state = json!({"messages": []});

        mw.before_model(&mut state).await.unwrap();
        assert_eq!(mw.get_logs().await.len(), 1);

        mw.clear_logs().await;
        assert!(mw.get_logs().await.is_empty());
    }

    // Test 10: LogLevel ordering / filtering
    #[tokio::test]
    async fn test_log_level_ordering_and_filtering() {
        // Verify ordering.
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);

        // Set level to Warn — Info events should be filtered out.
        let config = LoggingConfig {
            level: LogLevel::Warn,
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);
        let mut state = json!({"messages": []});

        // before_model emits at Info level — should be filtered.
        mw.before_model(&mut state).await.unwrap();
        assert!(mw.get_logs().await.is_empty());

        // Error level should pass through.
        mw.log_error("test", "something broke").await;
        assert_eq!(mw.get_logs().await.len(), 1);
    }

    // Test 11: Model call logging (start + end)
    #[tokio::test]
    async fn test_model_call_logging() {
        let config = LoggingConfig {
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);
        let mut state = json!({
            "model": "claude-sonnet",
            "messages": [{"type": "human", "content": "hello"}]
        });

        mw.before_model(&mut state).await.unwrap();
        mw.after_model(&mut state).await.unwrap();

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 2);

        // Check start event.
        assert!(logs[0].message.contains("claude-sonnet"));
        if let LogEvent::ModelCallStart { model, .. } = &logs[0].event {
            assert_eq!(model, "claude-sonnet");
        } else {
            panic!("Expected ModelCallStart");
        }

        // Check end event.
        if let LogEvent::ModelCallEnd { model, .. } = &logs[1].event {
            assert_eq!(model, "claude-sonnet");
        } else {
            panic!("Expected ModelCallEnd");
        }
    }

    // Test 12: Tool call logging
    #[tokio::test]
    async fn test_tool_call_logging() {
        let config = LoggingConfig {
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);
        let mut state = json!({"messages": []});

        mw.before_tool(&mut state, "web_search").await.unwrap();
        mw.after_tool(&mut state, "web_search", "found 10 results")
            .await
            .unwrap();

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 2);

        if let LogEvent::ToolCallStart { tool, .. } = &logs[0].event {
            assert_eq!(tool, "web_search");
        } else {
            panic!("Expected ToolCallStart");
        }

        if let LogEvent::ToolCallEnd { tool, output, .. } = &logs[1].event {
            assert_eq!(tool, "web_search");
            assert!(output.contains("found 10 results"));
        } else {
            panic!("Expected ToolCallEnd");
        }
    }

    // Test 13: Error logging
    #[tokio::test]
    async fn test_error_logging() {
        let config = LoggingConfig {
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);

        mw.log_error("openai_client", "rate limited by provider")
            .await;

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].level, LogLevel::Error);

        if let LogEvent::Error { source, message } = &logs[0].event {
            assert_eq!(source, "openai_client");
            assert_eq!(message, "rate limited by provider");
        } else {
            panic!("Expected Error event");
        }
    }

    // Test 14: Multiple redaction patterns
    #[test]
    fn test_multiple_redaction_patterns() {
        let content = "key=sk-secret123 password=hunter2 token=ghp_abc999";
        let patterns = vec![
            "sk-[a-zA-Z0-9]+".to_string(),
            "password=[^ ]+".to_string(),
            "ghp_[a-zA-Z0-9]+".to_string(),
        ];
        let redacted = ContentRedactor::redact(content, &patterns);
        assert!(!redacted.contains("sk-secret123"));
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("ghp_abc999"));
        assert_eq!(
            redacted.matches("[REDACTED]").count(),
            3,
            "Expected 3 redacted segments, got: {redacted}"
        );
    }

    // Test 15: LogEvent variants
    #[test]
    fn test_log_event_variants() {
        let events: Vec<LogEvent> = vec![
            LogEvent::ModelCallStart { model: "m".into(), input_tokens: Some(10) },
            LogEvent::ModelCallEnd { model: "m".into(), output_tokens: Some(20), duration: Duration::from_secs(1) },
            LogEvent::ToolCallStart { tool: "t".into(), input: "i".into() },
            LogEvent::ToolCallEnd { tool: "t".into(), output: "o".into(), duration: Duration::from_millis(500) },
            LogEvent::StateChange { key: "k".into(), old_value: None, new_value: json!(42) },
            LogEvent::Error { source: "s".into(), message: "m".into() },
            LogEvent::AgentStart { agent_name: "a".into() },
            LogEvent::AgentEnd { agent_name: "a".into(), total_duration: Duration::from_secs(10) },
        ];

        let expected_names = [
            "ModelCallStart", "ModelCallEnd", "ToolCallStart", "ToolCallEnd",
            "StateChange", "Error", "AgentStart", "AgentEnd",
        ];

        for (event, expected) in events.iter().zip(expected_names.iter()) {
            assert_eq!(event.name(), *expected);
        }
    }

    // Test 16: Middleware name
    #[test]
    fn test_middleware_name() {
        let mw = LoggingMiddleware::new(default_in_memory_config());
        assert_eq!(mw.name(), "logging");
    }

    // Test 17: State change logging
    #[tokio::test]
    async fn test_state_change_logging() {
        let config = LoggingConfig {
            level: LogLevel::Debug, // StateChange is logged at Debug level
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);

        mw.log_state_change("counter", Some(json!(1)), json!(2)).await;

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 1);

        if let LogEvent::StateChange { key, old_value, new_value } = &logs[0].event {
            assert_eq!(key, "counter");
            assert_eq!(*old_value, Some(json!(1)));
            assert_eq!(*new_value, json!(2));
        } else {
            panic!("Expected StateChange event");
        }
    }

    // Test 18: Agent start/end logging
    #[tokio::test]
    async fn test_agent_start_end_logging() {
        let config = LoggingConfig {
            log_destination: LogDestination::InMemory,
            ..Default::default()
        };
        let mw = in_memory_middleware(config);

        mw.log_agent_start("deep-agent-1").await;
        mw.log_agent_end("deep-agent-1", Duration::from_secs(5)).await;

        let logs = mw.get_logs().await;
        assert_eq!(logs.len(), 2);

        if let LogEvent::AgentStart { agent_name } = &logs[0].event {
            assert_eq!(agent_name, "deep-agent-1");
        } else {
            panic!("Expected AgentStart");
        }

        if let LogEvent::AgentEnd { agent_name, total_duration } = &logs[1].event {
            assert_eq!(agent_name, "deep-agent-1");
            assert_eq!(*total_duration, Duration::from_secs(5));
        } else {
            panic!("Expected AgentEnd");
        }
    }
}
