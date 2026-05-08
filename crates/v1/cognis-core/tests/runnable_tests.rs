//! Integration tests for core runnable composition primitives.
//!
//! Covers: RunnableLambda, RunnableSequence, RunnableParallel,
//! RunnableBranch, RunnablePassthrough, RunnableWithFallbacks.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use cognis_core::error::CognisError;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::branch::RunnableBranch;
use cognis_core::runnables::config::RunnableConfig;
use cognis_core::runnables::fallbacks::RunnableWithFallbacks;
use cognis_core::runnables::lambda::RunnableLambda;
use cognis_core::runnables::parallel::RunnableParallel;
use cognis_core::runnables::passthrough::RunnablePassthrough;
use cognis_core::runnables::sequence::RunnableSequence;

// ─── RunnableLambda ─────────────────────────────────────────────────

#[tokio::test]
async fn lambda_invoke_basic() {
    let r = RunnableLambda::new("double", |v: Value| async move {
        let n = v.as_i64().unwrap_or(0);
        Ok(json!(n * 2))
    });
    let result = r.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(10));
}

#[tokio::test]
async fn lambda_name_matches() {
    let r = RunnableLambda::new("my_step", |v: Value| async move { Ok(v) });
    assert_eq!(r.name(), "my_step");
}

#[tokio::test]
async fn lambda_error_propagation() {
    let r = RunnableLambda::new("fail", |_v: Value| async move {
        Err(CognisError::Other("intentional error".into()))
    });
    let result = r.invoke(json!(1), None).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("intentional"));
}

#[tokio::test]
async fn lambda_with_config() {
    let r = RunnableLambda::with_config(
        "cfg_reader",
        |v: Value, config: Option<RunnableConfig>| async move {
            let tag = config
                .and_then(|c| c.metadata.get("tag").cloned())
                .unwrap_or(json!("none"));
            Ok(json!({"input": v, "tag": tag}))
        },
    );

    let mut cfg = RunnableConfig::default();
    cfg.metadata.insert("tag".to_string(), json!("hello"));

    let result = r.invoke(json!(42), Some(&cfg)).await.unwrap();
    assert_eq!(result["input"], json!(42));
    assert_eq!(result["tag"], json!("hello"));
}

#[tokio::test]
async fn lambda_batch() {
    let r = RunnableLambda::new("inc", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) + 1))
    });
    let results = r
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    assert_eq!(results, vec![json!(2), json!(3), json!(4)]);
}

// ─── RunnableSequence ───────────────────────────────────────────────

#[tokio::test]
async fn sequence_two_steps() {
    let add_one = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) + 1))
    })) as Arc<dyn Runnable>;
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) * 2))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![add_one, double]).unwrap();
    let result = seq.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(12)); // (5 + 1) * 2
}

#[tokio::test]
async fn sequence_three_steps() {
    let a = Arc::new(RunnableLambda::new("a", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) + 1))
    })) as Arc<dyn Runnable>;
    let b = Arc::new(RunnableLambda::new("b", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) * 3))
    })) as Arc<dyn Runnable>;
    let c = Arc::new(RunnableLambda::new("c", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) - 2))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![a, b, c]).unwrap();
    let result = seq.invoke(json!(4), None).await.unwrap();
    assert_eq!(result, json!(13)); // ((4 + 1) * 3) - 2
}

#[tokio::test]
async fn sequence_error_stops_chain() {
    let ok_step = Arc::new(RunnableLambda::new("ok", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) + 1))
    })) as Arc<dyn Runnable>;
    let fail_step = Arc::new(RunnableLambda::new("fail", |_v: Value| async move {
        Err(CognisError::Other("mid-chain failure".into()))
    })) as Arc<dyn Runnable>;
    let never = Arc::new(RunnableLambda::new("never", |_v: Value| async move {
        panic!("should never be called");
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![ok_step, fail_step, never]).unwrap();
    let result = seq.invoke(json!(1), None).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("mid-chain"));
}

#[tokio::test]
async fn sequence_empty_rejected() {
    let result = RunnableSequence::new(vec![]);
    assert!(result.is_err());
}

#[tokio::test]
async fn sequence_single_step() {
    let step =
        Arc::new(RunnableLambda::new("id", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let seq = RunnableSequence::new(vec![step]).unwrap();
    let result = seq.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result, json!("hello"));
}

#[tokio::test]
async fn sequence_batch() {
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) * 2))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![double]).unwrap();
    let results = seq
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    assert_eq!(results, vec![json!(2), json!(4), json!(6)]);
}

#[tokio::test]
async fn sequence_with_name() {
    let step =
        Arc::new(RunnableLambda::new("s", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let seq = RunnableSequence::new(vec![step])
        .unwrap()
        .with_name("my_chain");
    assert_eq!(seq.name(), "my_chain");
}

#[tokio::test]
async fn sequence_default_name() {
    let step =
        Arc::new(RunnableLambda::new("s", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let seq = RunnableSequence::new(vec![step]).unwrap();
    assert_eq!(seq.name(), "RunnableSequence");
}

// ─── RunnableParallel ───────────────────────────────────────────────

#[tokio::test]
async fn parallel_two_branches() {
    let mut steps = HashMap::new();
    steps.insert(
        "doubled".to_string(),
        Arc::new(RunnableLambda::new("double", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) * 2))
        })) as Arc<dyn Runnable>,
    );
    steps.insert(
        "tripled".to_string(),
        Arc::new(RunnableLambda::new("triple", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) * 3))
        })) as Arc<dyn Runnable>,
    );

    let par = RunnableParallel::new(steps);
    let result = par.invoke(json!(5), None).await.unwrap();
    assert_eq!(result["doubled"], json!(10));
    assert_eq!(result["tripled"], json!(15));
}

#[tokio::test]
async fn parallel_single_branch() {
    let mut steps = HashMap::new();
    steps.insert(
        "identity".to_string(),
        Arc::new(RunnableLambda::new("id", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>,
    );

    let par = RunnableParallel::new(steps);
    let result = par.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result["identity"], json!("hello"));
}

#[tokio::test]
async fn parallel_error_propagates() {
    let mut steps = HashMap::new();
    steps.insert(
        "ok".to_string(),
        Arc::new(RunnableLambda::new("ok", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>,
    );
    steps.insert(
        "fail".to_string(),
        Arc::new(RunnableLambda::new("fail", |_v: Value| async move {
            Err(CognisError::Other("parallel branch failed".into()))
        })) as Arc<dyn Runnable>,
    );

    let par = RunnableParallel::new(steps);
    let result = par.invoke(json!(1), None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn parallel_with_name() {
    let par = RunnableParallel::new(HashMap::new()).with_name("my_parallel");
    assert_eq!(par.name(), "my_parallel");
}

#[tokio::test]
async fn parallel_default_name() {
    let par = RunnableParallel::new(HashMap::new());
    assert_eq!(par.name(), "RunnableParallel");
}

// ─── RunnableBranch ─────────────────────────────────────────────────

#[tokio::test]
async fn branch_first_condition_matches() {
    let cond = Arc::new(RunnableLambda::new("is_positive", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) > 0))
    })) as Arc<dyn Runnable>;
    let action = Arc::new(RunnableLambda::new("pos_action", |_v: Value| async move {
        Ok(json!("positive"))
    })) as Arc<dyn Runnable>;
    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("default"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(cond, action)], default);
    let result = branch.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!("positive"));
}

#[tokio::test]
async fn branch_no_condition_matches_uses_default() {
    let cond = Arc::new(RunnableLambda::new(
        "always_false",
        |_v: Value| async move { Ok(json!(false)) },
    )) as Arc<dyn Runnable>;
    let action = Arc::new(RunnableLambda::new("never", |_v: Value| async move {
        Ok(json!("never"))
    })) as Arc<dyn Runnable>;
    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("fallback"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(cond, action)], default);
    let result = branch.invoke(json!(0), None).await.unwrap();
    assert_eq!(result, json!("fallback"));
}

#[tokio::test]
async fn branch_second_condition_matches() {
    let cond1 = Arc::new(RunnableLambda::new("c1", |_v: Value| async move {
        Ok(json!(false))
    })) as Arc<dyn Runnable>;
    let action1 = Arc::new(RunnableLambda::new("a1", |_v: Value| async move {
        Ok(json!("first"))
    })) as Arc<dyn Runnable>;

    let cond2 = Arc::new(RunnableLambda::new("c2", |_v: Value| async move {
        Ok(json!(true))
    })) as Arc<dyn Runnable>;
    let action2 = Arc::new(RunnableLambda::new("a2", |_v: Value| async move {
        Ok(json!("second"))
    })) as Arc<dyn Runnable>;

    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("default"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(cond1, action1), (cond2, action2)], default);
    let result = branch.invoke(json!(0), None).await.unwrap();
    assert_eq!(result, json!("second"));
}

#[tokio::test]
async fn branch_truthy_values() {
    // Non-empty string is truthy
    let cond = Arc::new(RunnableLambda::new("str_cond", |_v: Value| async move {
        Ok(json!("non-empty"))
    })) as Arc<dyn Runnable>;
    let action = Arc::new(RunnableLambda::new("action", |_v: Value| async move {
        Ok(json!("matched"))
    })) as Arc<dyn Runnable>;
    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("default"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(cond, action)], default);
    let result = branch.invoke(json!(null), None).await.unwrap();
    assert_eq!(result, json!("matched"));
}

#[tokio::test]
async fn branch_falsy_values() {
    // null, 0, empty string, empty array, empty object are falsy
    for falsy in [json!(null), json!(0), json!(""), json!([]), json!({})] {
        let falsy_clone = falsy.clone();
        let cond = Arc::new(RunnableLambda::new("falsy_cond", move |_v: Value| {
            let val = falsy_clone.clone();
            async move { Ok(val) }
        })) as Arc<dyn Runnable>;
        let action = Arc::new(RunnableLambda::new("action", |_v: Value| async move {
            Ok(json!("should_not_match"))
        })) as Arc<dyn Runnable>;
        let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
            Ok(json!("default"))
        })) as Arc<dyn Runnable>;

        let branch = RunnableBranch::new(vec![(cond, action)], default);
        let result = branch.invoke(json!(1), None).await.unwrap();
        assert_eq!(result, json!("default"), "Expected falsy for {:?}", falsy);
    }
}

#[tokio::test]
async fn branch_name() {
    let default =
        Arc::new(RunnableLambda::new("d", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let branch = RunnableBranch::new(vec![], default);
    assert_eq!(branch.name(), "RunnableBranch");
}

#[tokio::test]
async fn branch_input_passed_to_action() {
    // The original input (not condition result) is passed to the matching action
    let cond = Arc::new(RunnableLambda::new("always_true", |_v: Value| async move {
        Ok(json!(true))
    })) as Arc<dyn Runnable>;
    let action = Arc::new(RunnableLambda::new("echo", |v: Value| async move {
        Ok(json!({"received": v}))
    })) as Arc<dyn Runnable>;
    let default =
        Arc::new(RunnableLambda::new("d", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(cond, action)], default);
    let result = branch.invoke(json!(99), None).await.unwrap();
    assert_eq!(result["received"], json!(99));
}

// ─── RunnablePassthrough ────────────────────────────────────────────

#[tokio::test]
async fn passthrough_returns_input() {
    let r = RunnablePassthrough::new();
    let result = r.invoke(json!({"key": "value"}), None).await.unwrap();
    assert_eq!(result, json!({"key": "value"}));
}

#[tokio::test]
async fn passthrough_various_types() {
    let r = RunnablePassthrough::new();

    assert_eq!(r.invoke(json!(42), None).await.unwrap(), json!(42));
    assert_eq!(
        r.invoke(json!("hello"), None).await.unwrap(),
        json!("hello")
    );
    assert_eq!(r.invoke(json!(null), None).await.unwrap(), json!(null));
    assert_eq!(
        r.invoke(json!([1, 2, 3]), None).await.unwrap(),
        json!([1, 2, 3])
    );
    assert_eq!(r.invoke(json!(true), None).await.unwrap(), json!(true));
}

#[tokio::test]
async fn passthrough_with_side_effect() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    let r = RunnablePassthrough::with_side_effect(move |_v: Value| {
        let flag = called_clone.clone();
        async move {
            flag.store(true, Ordering::SeqCst);
            Ok(())
        }
    });

    let result = r.invoke(json!(42), None).await.unwrap();
    assert_eq!(result, json!(42)); // input returned unchanged
    assert!(called.load(Ordering::SeqCst)); // side effect ran
}

#[tokio::test]
async fn passthrough_side_effect_error_propagates() {
    let r = RunnablePassthrough::with_side_effect(|_v: Value| async move {
        Err(CognisError::Other("side effect failed".into()))
    });

    let result = r.invoke(json!(1), None).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("side effect failed"));
}

#[tokio::test]
async fn passthrough_name() {
    let r = RunnablePassthrough::new();
    assert_eq!(r.name(), "RunnablePassthrough");
}

#[tokio::test]
async fn passthrough_default_trait() {
    let r = RunnablePassthrough::default();
    let result = r.invoke(json!("test"), None).await.unwrap();
    assert_eq!(result, json!("test"));
}

// ─── RunnableWithFallbacks ──────────────────────────────────────────

#[tokio::test]
async fn fallbacks_primary_succeeds() {
    let primary = Arc::new(RunnableLambda::new("primary", |v: Value| async move {
        Ok(json!({"from": "primary", "v": v}))
    })) as Arc<dyn Runnable>;
    let fb = Arc::new(RunnableLambda::new("fb", |_v: Value| async move {
        panic!("should not be called");
    })) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(primary).with_fallback(fb);
    let result = chain.invoke(json!(1), None).await.unwrap();
    assert_eq!(result["from"], "primary");
}

#[tokio::test]
async fn fallbacks_primary_fails_uses_fallback() {
    let primary = Arc::new(RunnableLambda::new("primary", |_v: Value| async move {
        Err(CognisError::Other("primary failed".into()))
    })) as Arc<dyn Runnable>;
    let fb = Arc::new(RunnableLambda::new("fb", |v: Value| async move {
        Ok(json!({"from": "fallback", "v": v}))
    })) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(primary).with_fallback(fb);
    let result = chain.invoke(json!(1), None).await.unwrap();
    assert_eq!(result["from"], "fallback");
}

#[tokio::test]
async fn fallbacks_all_fail_returns_last_error() {
    let primary = Arc::new(RunnableLambda::new("p", |_v: Value| async move {
        Err(CognisError::Other("err_primary".into()))
    })) as Arc<dyn Runnable>;
    let fb1 = Arc::new(RunnableLambda::new("fb1", |_v: Value| async move {
        Err(CognisError::Other("err_fb1".into()))
    })) as Arc<dyn Runnable>;
    let fb2 = Arc::new(RunnableLambda::new("fb2", |_v: Value| async move {
        Err(CognisError::Other("err_fb2".into()))
    })) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(primary)
        .with_fallback(fb1)
        .with_fallback(fb2);
    let err = chain.invoke(json!(1), None).await.unwrap_err();
    assert!(err.to_string().contains("err_fb2"));
}

#[tokio::test]
async fn fallbacks_exception_filter() {
    // ToolException is in the filter -> fallback runs
    let primary = Arc::new(RunnableLambda::new("p", |_v: Value| async move {
        Err(CognisError::ToolException("tool broke".into()))
    })) as Arc<dyn Runnable>;
    let fb = Arc::new(RunnableLambda::new("fb", |_v: Value| async move {
        Ok(json!("recovered"))
    })) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(primary)
        .with_fallback(fb)
        .with_exceptions_to_handle(vec!["ToolException".to_string()]);
    let result = chain.invoke(json!(1), None).await.unwrap();
    assert_eq!(result, json!("recovered"));
}

#[tokio::test]
async fn fallbacks_exception_filter_non_matching_propagates() {
    // Other is NOT in the filter -> error propagates immediately
    let primary = Arc::new(RunnableLambda::new("p", |_v: Value| async move {
        Err(CognisError::Other("unfiltered".into()))
    })) as Arc<dyn Runnable>;
    let fb = Arc::new(RunnableLambda::new("fb", |_v: Value| async move {
        Ok(json!("should not reach"))
    })) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(primary)
        .with_fallback(fb)
        .with_exceptions_to_handle(vec!["ToolException".to_string()]);
    let err = chain.invoke(json!(1), None).await.unwrap_err();
    assert!(err.to_string().contains("unfiltered"));
}

#[tokio::test]
async fn fallbacks_display_name() {
    let p =
        Arc::new(RunnableLambda::new("gpt4", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let fb = Arc::new(RunnableLambda::new(
        "claude",
        |v: Value| async move { Ok(v) },
    )) as Arc<dyn Runnable>;

    let chain = RunnableWithFallbacks::new(p).with_fallback(fb);
    assert_eq!(chain.display_name(), "gpt4 with fallbacks [claude]");
}

// ─── Composition: combining primitives ──────────────────────────────

#[tokio::test]
async fn sequence_of_parallel_and_lambda() {
    // Step 1: parallel splits input into two branches
    let mut par_steps = HashMap::new();
    par_steps.insert(
        "a".to_string(),
        Arc::new(RunnableLambda::new("times2", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) * 2))
        })) as Arc<dyn Runnable>,
    );
    par_steps.insert(
        "b".to_string(),
        Arc::new(RunnableLambda::new("times3", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) * 3))
        })) as Arc<dyn Runnable>,
    );
    let parallel = Arc::new(RunnableParallel::new(par_steps)) as Arc<dyn Runnable>;

    // Step 2: sum the parallel outputs
    let sum = Arc::new(RunnableLambda::new("sum", |v: Value| async move {
        let a = v["a"].as_i64().unwrap_or(0);
        let b = v["b"].as_i64().unwrap_or(0);
        Ok(json!(a + b))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![parallel, sum]).unwrap();
    let result = seq.invoke(json!(10), None).await.unwrap();
    assert_eq!(result, json!(50)); // (10*2) + (10*3) = 50
}

#[tokio::test]
async fn passthrough_in_sequence() {
    let pt = Arc::new(RunnablePassthrough::new()) as Arc<dyn Runnable>;
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap_or(0) * 2))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![pt, double]).unwrap();
    let result = seq.invoke(json!(7), None).await.unwrap();
    assert_eq!(result, json!(14));
}

#[tokio::test]
async fn branch_in_sequence() {
    // classify -> branch -> transform
    let classify = Arc::new(RunnableLambda::new("classify", |v: Value| async move {
        let n = v.as_i64().unwrap_or(0);
        Ok(json!({"value": n, "positive": n > 0}))
    })) as Arc<dyn Runnable>;

    let cond = Arc::new(RunnableLambda::new("check", |v: Value| async move {
        Ok(json!(v["positive"].as_bool().unwrap_or(false)))
    })) as Arc<dyn Runnable>;
    let pos_action = Arc::new(RunnableLambda::new("pos", |v: Value| async move {
        Ok(json!(v["value"].as_i64().unwrap_or(0) * 10))
    })) as Arc<dyn Runnable>;
    let neg_action = Arc::new(RunnableLambda::new("neg", |v: Value| async move {
        Ok(json!(v["value"].as_i64().unwrap_or(0) * -1))
    })) as Arc<dyn Runnable>;

    let branch =
        Arc::new(RunnableBranch::new(vec![(cond, pos_action)], neg_action)) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![classify, branch]).unwrap();

    let positive = seq.invoke(json!(5), None).await.unwrap();
    assert_eq!(positive, json!(50));

    let negative = seq.invoke(json!(-3), None).await.unwrap();
    assert_eq!(negative, json!(3));
}

// ─── chain! macro ───────────────────────────────────────────────────

#[tokio::test]
async fn chain_macro() {
    let seq = cognis_core::chain!(
        RunnableLambda::new("add1", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) + 1))
        }),
        RunnableLambda::new("mul2", |v: Value| async move {
            Ok(json!(v.as_i64().unwrap_or(0) * 2))
        }),
    )
    .unwrap();

    let result = seq.invoke(json!(3), None).await.unwrap();
    assert_eq!(result, json!(8)); // (3 + 1) * 2
}
