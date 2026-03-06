//! Retry logic with exponential backoff for Pregel node execution.
//!
//! Provides [`run_with_retry`] and [`arun_with_retry`] which wrap an async action
//! with configurable retry policies including exponential backoff and optional jitter.

use std::future::Future;
use std::pin::Pin;

use crate::errors::{LangGraphError, Result};
use crate::types::RetryPolicy;

/// An async action that can be retried. Takes no arguments and returns a `Result<Value>`.
pub type RetryAction =
    Box<dyn FnMut() -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>> + Send>;

/// Compute the sleep duration for a given attempt using exponential backoff.
///
/// The interval is `min(max_interval, initial_interval * backoff_factor^(attempt - 1))`,
/// optionally with uniform jitter in `[0, 1)` added.
fn compute_sleep_duration(policy: &RetryPolicy, attempt: u32, jitter_value: f64) -> f64 {
    let interval = policy
        .max_interval_secs
        .min(policy.initial_interval_secs * policy.backoff_factor.powi((attempt - 1) as i32));

    if policy.jitter {
        interval + jitter_value
    } else {
        interval
    }
}

/// Run an async action with retry logic based on the given [`RetryPolicy`].
///
/// The action closure is called repeatedly until it succeeds or the maximum number
/// of attempts is exhausted. Between retries, the function sleeps for an exponentially
/// increasing interval with optional jitter.
///
/// # Arguments
///
/// * `action` - A closure that returns a future producing a `Result<Value>`.
/// * `policy` - The retry policy governing backoff behavior.
///
/// # Returns
///
/// The result of the first successful invocation, or the last error if all attempts fail.
///
/// # Example
///
/// ```rust,no_run
/// use langgraph::pregel::retry::run_with_retry;
/// use langgraph::types::RetryPolicy;
/// use serde_json::json;
///
/// # async fn example() {
/// let policy = RetryPolicy::default();
/// let mut call_count = 0u32;
///
/// let result = run_with_retry(
///     Box::new(move || {
///         call_count += 1;
///         let count = call_count;
///         Box::pin(async move {
///             if count < 3 {
///                 Err(langgraph::errors::LangGraphError::Other("transient".into()))
///             } else {
///                 Ok(json!({"ok": true}))
///             }
///         })
///     }),
///     &policy,
/// ).await;
/// # }
/// ```
pub async fn run_with_retry(
    mut action: RetryAction,
    policy: &RetryPolicy,
) -> Result<serde_json::Value> {
    let mut attempts: u32 = 0;

    loop {
        match action().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                // Graph interrupts and parent commands should not be retried.
                if matches!(
                    &err,
                    LangGraphError::GraphInterrupt(_) | LangGraphError::ParentCommand(_)
                ) {
                    return Err(err);
                }

                attempts += 1;

                if attempts >= policy.max_attempts {
                    return Err(err);
                }

                let jitter_value = if policy.jitter {
                    // Use a simple deterministic-ish jitter based on attempt number
                    // to avoid pulling in the `rand` crate. In production code you would
                    // use `rand::random::<f64>()`.
                    simple_jitter(attempts)
                } else {
                    0.0
                };

                let sleep_secs = compute_sleep_duration(policy, attempts, jitter_value);
                tokio::time::sleep(tokio::time::Duration::from_secs_f64(sleep_secs)).await;
            }
        }
    }
}

/// Run an action with retry using a list of policies. The first matching policy is used.
///
/// Each policy is tried in order; the first one whose `max_attempts` has not been
/// exhausted will govern the retry. If no policy allows further retries the error
/// propagates immediately.
pub async fn run_with_retry_policies(
    mut action: RetryAction,
    policies: &[RetryPolicy],
) -> Result<serde_json::Value> {
    if policies.is_empty() {
        return action().await;
    }

    let mut attempts: u32 = 0;

    loop {
        match action().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if matches!(
                    &err,
                    LangGraphError::GraphInterrupt(_) | LangGraphError::ParentCommand(_)
                ) {
                    return Err(err);
                }

                attempts += 1;

                // Find the first policy that still has attempts remaining.
                let matching_policy = policies
                    .iter()
                    .find(|p| attempts < p.max_attempts);

                let Some(policy) = matching_policy else {
                    return Err(err);
                };

                let jitter_value = if policy.jitter {
                    simple_jitter(attempts)
                } else {
                    0.0
                };

                let sleep_secs = compute_sleep_duration(policy, attempts, jitter_value);
                tokio::time::sleep(tokio::time::Duration::from_secs_f64(sleep_secs)).await;
            }
        }
    }
}

/// Produce a pseudo-random jitter value in `[0, 1)` without pulling in `rand`.
///
/// This uses a simple hash of the attempt number. It is NOT cryptographically
/// secure and is only meant to spread out retries a bit.
fn simple_jitter(attempt: u32) -> f64 {
    // Use a basic multiplicative hash to spread values.
    let hash = (attempt as u64).wrapping_mul(6_364_136_223_846_793_005);
    (hash % 1_000_000) as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_compute_sleep_duration_no_jitter() {
        let policy = RetryPolicy {
            initial_interval_secs: 1.0,
            backoff_factor: 2.0,
            max_interval_secs: 60.0,
            max_attempts: 5,
            jitter: false,
            retry_on: None,
        };
        // attempt 1: 1.0 * 2^0 = 1.0
        assert!((compute_sleep_duration(&policy, 1, 0.0) - 1.0).abs() < f64::EPSILON);
        // attempt 2: 1.0 * 2^1 = 2.0
        assert!((compute_sleep_duration(&policy, 2, 0.0) - 2.0).abs() < f64::EPSILON);
        // attempt 3: 1.0 * 2^2 = 4.0
        assert!((compute_sleep_duration(&policy, 3, 0.0) - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_sleep_duration_with_jitter() {
        let policy = RetryPolicy {
            initial_interval_secs: 1.0,
            backoff_factor: 2.0,
            max_interval_secs: 60.0,
            max_attempts: 5,
            jitter: true,
            retry_on: None,
        };
        let duration = compute_sleep_duration(&policy, 1, 0.5);
        assert!((duration - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_sleep_duration_capped_at_max() {
        let policy = RetryPolicy {
            initial_interval_secs: 1.0,
            backoff_factor: 10.0,
            max_interval_secs: 5.0,
            max_attempts: 5,
            jitter: false,
            retry_on: None,
        };
        // 1.0 * 10^2 = 100.0, capped at 5.0
        assert!((compute_sleep_duration(&policy, 3, 0.0) - 5.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_run_with_retry_succeeds_first_try() {
        let action: RetryAction = Box::new(|| Box::pin(async { Ok(json!({"ok": true})) }));

        let policy = RetryPolicy {
            initial_interval_secs: 0.001,
            backoff_factor: 2.0,
            max_interval_secs: 1.0,
            max_attempts: 3,
            jitter: false,
            retry_on: None,
        };

        let result = run_with_retry(action, &policy).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"ok": true}));
    }

    #[tokio::test]
    async fn test_run_with_retry_succeeds_after_failures() {
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let action: RetryAction = Box::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(LangGraphError::Other(format!("fail #{}", n)))
                } else {
                    Ok(json!({"attempt": n}))
                }
            })
        });

        let policy = RetryPolicy {
            initial_interval_secs: 0.001,
            backoff_factor: 1.0,
            max_interval_secs: 0.01,
            max_attempts: 5,
            jitter: false,
            retry_on: None,
        };

        let result = run_with_retry(action, &policy).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"attempt": 2}));
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_run_with_retry_exhausts_attempts() {
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let action: RetryAction = Box::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(LangGraphError::Other("always fails".into()))
            })
        });

        let policy = RetryPolicy {
            initial_interval_secs: 0.001,
            backoff_factor: 1.0,
            max_interval_secs: 0.01,
            max_attempts: 3,
            jitter: false,
            retry_on: None,
        };

        let result = run_with_retry(action, &policy).await;
        assert!(result.is_err());
        // 1 initial + 2 retries = 3 total
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_run_with_retry_does_not_retry_graph_interrupt() {
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let action: RetryAction = Box::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(LangGraphError::GraphInterrupt(vec![]))
            })
        });

        let policy = RetryPolicy {
            initial_interval_secs: 0.001,
            backoff_factor: 1.0,
            max_interval_secs: 0.01,
            max_attempts: 5,
            jitter: false,
            retry_on: None,
        };

        let result = run_with_retry(action, &policy).await;
        assert!(result.is_err());
        // Should NOT retry — only 1 call.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_run_with_retry_does_not_retry_parent_command() {
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let action: RetryAction = Box::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(LangGraphError::ParentCommand(
                    crate::types::Command::default(),
                ))
            })
        });

        let policy = RetryPolicy::default();

        let result = run_with_retry(action, &policy).await;
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_run_with_retry_policies_empty_list() {
        let action: RetryAction = Box::new(|| Box::pin(async { Ok(json!(42)) }));

        let result = run_with_retry_policies(action, &[]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!(42));
    }

    #[tokio::test]
    async fn test_run_with_retry_policies_uses_first_matching() {
        let counter = Arc::new(AtomicU32::new(0));

        let counter_clone = counter.clone();
        let action: RetryAction = Box::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(LangGraphError::Other("fail".into()))
                } else {
                    Ok(json!({"ok": true}))
                }
            })
        });

        let policies = vec![RetryPolicy {
            initial_interval_secs: 0.001,
            backoff_factor: 1.0,
            max_interval_secs: 0.01,
            max_attempts: 3,
            jitter: false,
            retry_on: None,
        }];

        let result = run_with_retry_policies(action, &policies).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_jitter_range() {
        for attempt in 1..100 {
            let j = simple_jitter(attempt);
            assert!(j >= 0.0, "jitter should be >= 0 for attempt {attempt}");
            assert!(j < 1.0, "jitter should be < 1 for attempt {attempt}");
        }
    }

    #[test]
    fn test_default_retry_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
        assert!((policy.initial_interval_secs - 0.5).abs() < f64::EPSILON);
        assert!((policy.backoff_factor - 2.0).abs() < f64::EPSILON);
        assert!((policy.max_interval_secs - 128.0).abs() < f64::EPSILON);
        assert!(policy.jitter);
    }
}
