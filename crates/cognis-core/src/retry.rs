//! Retry classification utilities for error recovery.
//!
//! Provides the [`RetryClassifier`] trait for determining whether errors are
//! retryable, along with a [`DefaultRetryClassifier`] that classifies HTTP 429
//! (rate limit) and 5xx errors as retryable.

use crate::error::CognisError;

/// Classifies whether an error is retryable.
///
/// Implementations inspect a [`CognisError`] and return `true` if the
/// operation that produced the error should be retried.
pub trait RetryClassifier: Send + Sync {
    /// Returns `true` if the given error is transient and the operation
    /// should be retried.
    fn is_retryable(&self, error: &CognisError) -> bool;
}

/// Default retry classifier: HTTP 429 (rate limit) and 5xx errors are retryable.
///
/// Also classifies errors whose message contains "rate limit", "timeout",
/// or "connection" as retryable.
///
/// # Example
///
/// ```rust
/// use cognis_core::retry::{RetryClassifier, DefaultRetryClassifier};
/// use cognis_core::error::CognisError;
///
/// let classifier = DefaultRetryClassifier;
///
/// let rate_limit = CognisError::HttpError { status: 429, body: "Too Many Requests".into() };
/// assert!(classifier.is_retryable(&rate_limit));
///
/// let not_found = CognisError::HttpError { status: 404, body: "Not Found".into() };
/// assert!(!classifier.is_retryable(&not_found));
/// ```
pub struct DefaultRetryClassifier;

impl RetryClassifier for DefaultRetryClassifier {
    fn is_retryable(&self, error: &CognisError) -> bool {
        match error {
            CognisError::HttpError { status, .. } => *status == 429 || *status >= 500,
            CognisError::Other(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("rate limit")
                    || lower.contains("timeout")
                    || lower.contains("connection")
            }
            CognisError::IoError(_) => true,
            _ => false,
        }
    }
}

/// A retry classifier that always returns `true`.
///
/// Useful for testing or when all errors should be retried.
pub struct AlwaysRetryClassifier;

impl RetryClassifier for AlwaysRetryClassifier {
    fn is_retryable(&self, _error: &CognisError) -> bool {
        true
    }
}

/// A retry classifier that never returns `true`.
///
/// Useful for disabling retries entirely.
pub struct NeverRetryClassifier;

impl RetryClassifier for NeverRetryClassifier {
    fn is_retryable(&self, _error: &CognisError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_classifier_rate_limit() {
        let classifier = DefaultRetryClassifier;
        let err = CognisError::HttpError {
            status: 429,
            body: "Too Many Requests".into(),
        };
        assert!(classifier.is_retryable(&err));
    }

    #[test]
    fn test_retry_classifier_server_errors() {
        let classifier = DefaultRetryClassifier;
        for status in [500, 502, 503, 504] {
            let err = CognisError::HttpError {
                status,
                body: "Server Error".into(),
            };
            assert!(
                classifier.is_retryable(&err),
                "Expected status {} to be retryable",
                status
            );
        }
    }

    #[test]
    fn test_retry_classifier_client_errors_not_retryable() {
        let classifier = DefaultRetryClassifier;
        for status in [400, 401, 403, 404, 422] {
            let err = CognisError::HttpError {
                status,
                body: "Client Error".into(),
            };
            assert!(
                !classifier.is_retryable(&err),
                "Expected status {} to NOT be retryable",
                status
            );
        }
    }

    #[test]
    fn test_retry_classifier_other_with_keywords() {
        let classifier = DefaultRetryClassifier;

        let rate = CognisError::Other("rate limit exceeded".into());
        assert!(classifier.is_retryable(&rate));

        let timeout = CognisError::Other("request timeout after 30s".into());
        assert!(classifier.is_retryable(&timeout));

        let conn = CognisError::Other("connection reset by peer".into());
        assert!(classifier.is_retryable(&conn));
    }

    #[test]
    fn test_retry_classifier_non_retryable_errors() {
        let classifier = DefaultRetryClassifier;

        let parse_err = CognisError::OutputParserError {
            message: "bad format".into(),
            observation: None,
            llm_output: None,
        };
        assert!(!classifier.is_retryable(&parse_err));

        let tool_err = CognisError::ToolException("tool failed".into());
        assert!(!classifier.is_retryable(&tool_err));

        let generic = CognisError::Other("some unknown error".into());
        assert!(!classifier.is_retryable(&generic));
    }

    #[test]
    fn test_always_retry_classifier() {
        let classifier = AlwaysRetryClassifier;
        let err = CognisError::Other("anything".into());
        assert!(classifier.is_retryable(&err));
    }

    #[test]
    fn test_never_retry_classifier() {
        let classifier = NeverRetryClassifier;
        let err = CognisError::HttpError {
            status: 429,
            body: "rate limited".into(),
        };
        assert!(!classifier.is_retryable(&err));
    }
}
