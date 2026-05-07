//! Tests proving that `RunnableConfig::cancellation_token` is observable from
//! `Runnable::invoke` and from an LCEL chain composed of two runnables.
//!
//! This is the contract that tool authors rely on when they need tool-specific
//! cleanup beyond the executor-level future drop.
//!
//! A tool author who wants to kill a child process on cancel writes a
//! `Runnable` (or makes their tool implement `Runnable`) that reads
//! `config.cancellation_token` and pairs its long-running work with
//! `tokio::select!` against `token.cancelled()`. The executor's
//! `run_with_cancel` already forwards the token through its own loop, and any
//! Runnable-composed tool can see it via the config argument.

use std::sync::Arc;
use std::time::Duration;

use cognis_core::error::CognisError;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;
use cognis_core::runnables::lambda::RunnableLambda;
use cognis_core::CancellationToken;
use serde_json::{json, Value};

#[tokio::test]
async fn lambda_can_observe_cancellation_token_from_config() {
    // A lambda that inspects the config for a cancellation token and bails
    // early when it is cancelled. This proves the end-to-end wiring:
    // RunnableConfig carries the token, the lambda observes it, and can
    // synthesise a `CognisError::Cancelled`.
    let lambda = RunnableLambda::with_config("observer", |_input, cfg| async move {
        if let Some(cfg) = cfg {
            if let Some(ref t) = cfg.cancellation_token {
                if t.is_cancelled() {
                    return Err(CognisError::Cancelled("observed by lambda".into()));
                }
            }
        }
        Ok(Value::Null)
    });

    let mut cfg = RunnableConfig::default();
    cfg.cancellation_token = Some(CancellationToken::cancelled_now());

    let err = lambda.invoke(json!({}), Some(&cfg)).await.unwrap_err();
    match err {
        CognisError::Cancelled(reason) => assert_eq!(reason, "observed by lambda"),
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn lambda_without_cancel_token_runs_normally() {
    let lambda = RunnableLambda::with_config("observer", |_input, cfg| async move {
        if let Some(cfg) = cfg {
            if let Some(ref t) = cfg.cancellation_token {
                if t.is_cancelled() {
                    return Err(CognisError::Cancelled("observed".into()));
                }
            }
        }
        Ok(json!("done"))
    });

    // Default config has no token.
    let cfg = RunnableConfig::default();
    let out = lambda.invoke(json!({}), Some(&cfg)).await.unwrap();
    assert_eq!(out, json!("done"));
}

#[tokio::test]
async fn lambda_select_against_cancel_token_drops_inflight_work() {
    // Mimics the pattern a `Runnable`-based tool would use for cancel-aware
    // execution: `tokio::select!` between the slow work and the cancel token.
    let lambda = RunnableLambda::with_config("slow_worker", |_input, cfg| async move {
        let token = cfg
            .as_ref()
            .and_then(|c| c.cancellation_token.clone())
            .unwrap_or_default();

        tokio::select! {
            biased;
            _ = token.cancelled() => Err(CognisError::Cancelled("tool cancelled".into())),
            _ = tokio::time::sleep(Duration::from_secs(10)) => Ok(json!("normal")),
        }
    });

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_clone.cancel();
    });

    let mut cfg = RunnableConfig::default();
    cfg.cancellation_token = Some(cancel);

    let result = tokio::time::timeout(Duration::from_secs(2), lambda.invoke(json!({}), Some(&cfg)))
        .await
        .expect("lambda should abort quickly after cancel fires");

    match result {
        Err(CognisError::Cancelled(reason)) => assert_eq!(reason, "tool cancelled"),
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn patch_config_carries_cancellation_token() {
    // The `ConfigPatch::with_cancellation_token` builder puts the token into
    // a patched config — the usual way callers add a cancel token to an
    // existing config before forwarding to a nested runnable.
    use cognis_core::runnables::config::{patch_config, ConfigPatch};

    let base = RunnableConfig::default();
    let token = CancellationToken::cancelled_now();
    let patch = ConfigPatch::new().with_cancellation_token(token);
    let patched = patch_config(&base, &patch);
    let t = patched
        .cancellation_token
        .as_ref()
        .expect("token should be present");
    assert!(t.is_cancelled());
}

// ---------------------------------------------------------------------------
// End-to-end: LCEL chain propagates the token to every step
// ---------------------------------------------------------------------------

use cognis_core::runnables::sequence::RunnableSequence;

#[tokio::test]
async fn sequence_propagates_cancellation_token_to_all_steps() {
    // Build a two-step sequence: the first step passes through, the second
    // step checks the token. The sequence's `invoke` forwards the caller's
    // config (which carries the token) to each step.
    let step1 = Arc::new(RunnableLambda::new("passthrough", |input| async move {
        Ok(input)
    }));
    let step2 = Arc::new(RunnableLambda::with_config(
        "gate",
        |input, cfg| async move {
            if let Some(cfg) = cfg {
                if let Some(ref t) = cfg.cancellation_token {
                    if t.is_cancelled() {
                        return Err(CognisError::Cancelled("gate saw cancel".into()));
                    }
                }
            }
            Ok(input)
        },
    ));

    let seq = RunnableSequence::new(vec![step1 as Arc<dyn Runnable>, step2 as Arc<dyn Runnable>])
        .expect("sequence should build");

    let mut cfg = RunnableConfig::default();
    cfg.cancellation_token = Some(CancellationToken::cancelled_now());

    let err = seq.invoke(json!({"x": 1}), Some(&cfg)).await.unwrap_err();
    match err {
        CognisError::Cancelled(reason) => assert_eq!(reason, "gate saw cancel"),
        other => panic!("expected Cancelled, got {other:?}"),
    }
}
