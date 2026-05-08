//! Crate-wide error type. Exporter failures are *never* propagated into
//! the user's main code path — they are logged and counted. `TraceError`
//! exists for diagnostic surfaces (`stats()`, builder validation).

use thiserror::Error;

/// Errors surfaced by `cognis-trace`. Exporter-side failures are logged
/// and counted; this enum exists for builder validation and stats.
#[derive(Debug, Error)]
pub enum TraceError {
    /// Required environment variable is unset.
    #[error("missing required env var: {0}")]
    MissingEnvVar(String),

    /// Configuration value is invalid (e.g. malformed URL).
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Network error sending to a backend.
    #[cfg(feature = "langfuse")]
    #[error("network error sending to {backend}: {source}")]
    Network {
        /// Backend name (e.g. "langfuse").
        backend: &'static str,
        /// Underlying reqwest error.
        #[source]
        source: reqwest::Error,
    },

    /// Backend returned a non-2xx status.
    #[error("backend {backend} returned {status}: {body}")]
    BackendStatus {
        /// Backend name.
        backend: &'static str,
        /// HTTP status code.
        status: u16,
        /// Response body (truncated for log safety).
        body: String,
    },

    /// Exporter queue overflowed; events were dropped.
    #[error("queue overflowed; {dropped} events dropped (backend: {backend})")]
    QueueOverflow {
        /// Backend name.
        backend: &'static str,
        /// Number of events dropped.
        dropped: usize,
    },

    /// Exporter does not support a particular feature (scores, prompts, …).
    #[error("backend does not support: {0}")]
    Unsupported(&'static str),

    /// JSON serialization failed.
    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_env_var_displays_name() {
        let e = TraceError::MissingEnvVar("LANGFUSE_PUBLIC_KEY".into());
        assert_eq!(
            e.to_string(),
            "missing required env var: LANGFUSE_PUBLIC_KEY"
        );
    }

    #[test]
    fn unsupported_carries_feature_name() {
        let e = TraceError::Unsupported("scores");
        assert_eq!(e.to_string(), "backend does not support: scores");
    }
}
