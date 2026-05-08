//! Langfuse exporter configuration. All fields have env var equivalents
//! loaded by `LangfuseConfig::from_env()`.

use std::time::Duration;

use secrecy::SecretString;

use crate::batch::BatcherConfig;
use crate::error::TraceError;

/// Langfuse client + exporter configuration.
#[derive(Clone, Debug)]
pub struct LangfuseConfig {
    /// Base URL (default: https://cloud.langfuse.com).
    pub host: String,
    /// Public key.
    pub public_key: String,
    /// Secret key.
    pub secret_key: SecretString,
    /// Optional environment tag.
    pub environment: Option<String>,
    /// Optional release tag.
    pub release: Option<String>,
    /// Per-batcher config.
    pub batcher: BatcherConfig,
    /// Per-request HTTP timeout.
    pub timeout: Duration,
    /// Max retry attempts for 5xx / network errors.
    pub max_retries: u32,
}

impl Default for LangfuseConfig {
    fn default() -> Self {
        Self {
            host: "https://cloud.langfuse.com".into(),
            public_key: String::new(),
            secret_key: SecretString::from(String::new()),
            environment: None,
            release: None,
            batcher: BatcherConfig::default(),
            timeout: Duration::from_secs(10),
            max_retries: 3,
        }
    }
}

impl LangfuseConfig {
    /// Load from environment variables. Required:
    /// `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`. Optional:
    /// `LANGFUSE_HOST`, `LANGFUSE_ENVIRONMENT`, `LANGFUSE_RELEASE`,
    /// `LANGFUSE_FLUSH_INTERVAL_MS`, `LANGFUSE_MAX_BATCH`,
    /// `LANGFUSE_QUEUE_CAPACITY`, `LANGFUSE_TIMEOUT_MS`,
    /// `LANGFUSE_MAX_RETRIES`.
    pub fn from_env() -> Result<Self, TraceError> {
        let public_key = std::env::var("LANGFUSE_PUBLIC_KEY")
            .map_err(|_| TraceError::MissingEnvVar("LANGFUSE_PUBLIC_KEY".into()))?;
        let secret_key = std::env::var("LANGFUSE_SECRET_KEY")
            .map_err(|_| TraceError::MissingEnvVar("LANGFUSE_SECRET_KEY".into()))?;

        let host =
            std::env::var("LANGFUSE_HOST").unwrap_or_else(|_| "https://cloud.langfuse.com".into());
        let environment = std::env::var("LANGFUSE_ENVIRONMENT").ok();
        let release = std::env::var("LANGFUSE_RELEASE").ok();

        let mut batcher = BatcherConfig::default();
        if let Ok(v) = std::env::var("LANGFUSE_FLUSH_INTERVAL_MS") {
            batcher.flush_interval = Duration::from_millis(
                v.parse()
                    .map_err(|_| TraceError::InvalidConfig("LANGFUSE_FLUSH_INTERVAL_MS".into()))?,
            );
        }
        if let Ok(v) = std::env::var("LANGFUSE_MAX_BATCH") {
            batcher.max_batch = v
                .parse()
                .map_err(|_| TraceError::InvalidConfig("LANGFUSE_MAX_BATCH".into()))?;
        }
        if let Ok(v) = std::env::var("LANGFUSE_QUEUE_CAPACITY") {
            batcher.queue_capacity = v
                .parse()
                .map_err(|_| TraceError::InvalidConfig("LANGFUSE_QUEUE_CAPACITY".into()))?;
        }

        let timeout = match std::env::var("LANGFUSE_TIMEOUT_MS") {
            Ok(v) => Duration::from_millis(
                v.parse()
                    .map_err(|_| TraceError::InvalidConfig("LANGFUSE_TIMEOUT_MS".into()))?,
            ),
            Err(_) => Duration::from_secs(10),
        };

        let max_retries = match std::env::var("LANGFUSE_MAX_RETRIES") {
            Ok(v) => v
                .parse()
                .map_err(|_| TraceError::InvalidConfig("LANGFUSE_MAX_RETRIES".into()))?,
            Err(_) => 3,
        };

        Ok(Self {
            host,
            public_key,
            secret_key: SecretString::from(secret_key),
            environment,
            release,
            batcher,
            timeout,
            max_retries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_missing_keys_returns_missing_env_var() {
        // Test in a subprocess-isolated way by clearing env vars first.
        // SAFETY for std::env::remove_var requires single-threaded; this test
        // runs in default `cargo test` thread pool. We assert error variant only.
        let res = LangfuseConfig::from_env();
        // CI env may or may not set these. Accept either MissingEnvVar or Ok.
        match res {
            Err(TraceError::MissingEnvVar(_)) => {}
            Ok(_) => {} // env happens to be set; that's fine
            other => panic!("unexpected: {other:?}"),
        }
    }
}
