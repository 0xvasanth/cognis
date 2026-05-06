//! Errors for the cognis2 framework. Operational metadata (`is_retryable`,
//! `retry_delay`, `category`) lets retry/fallback middleware consume errors
//! without sniffing strings.

use std::time::Duration;

/// Result alias used throughout cognis2.
pub type Result<T> = std::result::Result<T, CognisError>;

/// All errors produced by cognis2-core and downstream v2 crates.
#[derive(Debug, thiserror::Error)]
pub enum CognisError {
    /// Provider call failed (network, HTTP, parse, etc.).
    #[error("provider `{provider}` error: {message}")]
    Provider {
        /// Provider identifier (e.g. "openai", "ollama").
        provider: String,
        /// Human-readable error message.
        message: String,
    },

    /// LLM provider rate-limited the request.
    #[error("rate limited; retry after {retry_after_ms}ms")]
    RateLimited {
        /// Suggested retry delay in milliseconds.
        retry_after_ms: u64,
    },

    /// Authentication failed (bad API key, expired token, etc.).
    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Tool dispatch or execution failed.
    #[error("tool `{name}` failed: {reason}")]
    Tool {
        /// Tool name.
        name: String,
        /// Failure reason.
        reason: String,
    },

    /// Tool argument failed validation.
    #[error("tool validation: {0}")]
    ToolValidation(String),

    /// Tool argument failed validation (v1 macro compat alias — same as `ToolValidation`).
    #[error("tool validation: {0}")]
    ToolValidationError(String),

    /// Configuration is invalid or incomplete.
    #[error("configuration: {0}")]
    Configuration(String),

    /// Network / transport error.
    #[error("network error{}: {message}", status_code.map(|c| format!(" (status {c})")).unwrap_or_default())]
    Network {
        /// Optional HTTP status code.
        status_code: Option<u16>,
        /// Human-readable error message.
        message: String,
    },

    /// Operation timed out.
    #[error("`{operation}` timed out after {timeout_ms}ms")]
    Timeout {
        /// Operation name.
        operation: String,
        /// Timeout duration in milliseconds.
        timeout_ms: u64,
    },

    /// Operation was cancelled via `RunnableConfig::cancel_token`.
    #[error("operation cancelled")]
    Cancelled,

    /// Graph engine ran past its `recursion_limit`.
    #[error("graph recursion limit ({limit}) exceeded")]
    RecursionLimit {
        /// The configured limit that was hit.
        limit: u32,
    },

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Catch-all for unexpected errors.
    #[error("internal error: {0}")]
    Internal(String),
}

impl CognisError {
    /// Stable category string for telemetry / metrics filtering.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Provider { .. } => "provider",
            Self::RateLimited { .. } => "rate_limit",
            Self::AuthenticationFailed(_) => "auth",
            Self::Tool { .. } => "tool",
            Self::ToolValidation(_) | Self::ToolValidationError(_) => "tool_validation",
            Self::Configuration(_) => "config",
            Self::Network { .. } => "network",
            Self::Timeout { .. } => "timeout",
            Self::Cancelled => "cancelled",
            Self::RecursionLimit { .. } => "recursion_limit",
            Self::Serialization(_) => "serialization",
            Self::Internal(_) => "internal",
        }
    }

    /// Whether retrying this error MAY succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::Network { .. }
                | Self::Timeout { .. }
                | Self::Provider { .. }
        )
    }

    /// Suggested retry delay, if the error type carries one.
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after_ms } => Some(Duration::from_millis(*retry_after_ms)),
            Self::Timeout { timeout_ms, .. } => Some(Duration::from_millis(*timeout_ms / 2)),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for CognisError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}
