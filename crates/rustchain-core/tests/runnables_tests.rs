use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::{json, Value};

use rustchain_core::chain;
use rustchain_core::error::RustChainError;
use rustchain_core::runnables::schema::{EventData, StreamEvent};
use rustchain_core::runnables::*;

// ─── Config Tests ───

#[test]
fn test_config_default() {
    let cfg = RunnableConfig::default();
    assert_eq!(cfg.recursion_limit, 25);
    assert!(cfg.tags.is_empty());
    assert!(cfg.metadata.is_empty());
    assert!(cfg.run_name.is_none());
    assert!(cfg.max_concurrency.is_none());
    assert!(cfg.run_id.is_none());
    assert!(cfg.callbacks.is_empty());
}

#[test]
fn test_ensure_config_none() {
    let cfg = ensure_config(None);
    assert_eq!(cfg.recursion_limit, 25);
}

#[test]
fn test_ensure_config_some() {
    let mut base = RunnableConfig::default();
    base.recursion_limit = 10;
    let cfg = ensure_config(Some(&base));
    assert_eq!(cfg.recursion_limit, 10);
}

#[test]
fn test_merge_configs() {
    let mut base = RunnableConfig::default();
    base.tags = vec!["a".into()];
    base.metadata.insert("key1".into(), json!("val1"));
    base.run_name = Some("base_run".into());

    let mut overlay = RunnableConfig::default();
    overlay.tags = vec!["b".into()];
    overlay.metadata.insert("key2".into(), json!("val2"));
    overlay.run_name = Some("overlay_run".into());
    overlay.recursion_limit = 10;

    let merged = merge_configs(&base, &overlay);
    assert_eq!(merged.tags, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(merged.metadata.get("key1"), Some(&json!("val1")));
    assert_eq!(merged.metadata.get("key2"), Some(&json!("val2")));
    assert_eq!(merged.run_name, Some("overlay_run".into()));
    assert_eq!(merged.recursion_limit, 10);
}

// ─── Lambda Tests ───

#[tokio::test]
async fn test_lambda_invoke() {
    let lambda = RunnableLambda::new("double", |input: Value| async move {
        let n = input.as_i64().unwrap();
        Ok(json!(n * 2))
    });
    let result = lambda.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(10));
}

#[tokio::test]
async fn test_lambda_with_config() {
    let lambda = RunnableLambda::with_config(
        "tag_check",
        |input, config: Option<RunnableConfig>| async move {
            let cfg = config.unwrap();
            Ok(json!({
                "input": input,
                "tags": cfg.tags,
            }))
        },
    );
    let mut cfg = RunnableConfig::default();
    cfg.tags = vec!["test_tag".into()];
    let result = lambda.invoke(json!("hello"), Some(&cfg)).await.unwrap();
    assert_eq!(result["tags"][0], "test_tag");
}

#[tokio::test]
async fn test_lambda_batch() {
    let lambda = RunnableLambda::new("inc", |input: Value| async move {
        Ok(json!(input.as_i64().unwrap() + 1))
    });
    let results = lambda
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    assert_eq!(results, vec![json!(2), json!(3), json!(4)]);
}

// ─── Sequence Tests ───

#[tokio::test]
async fn test_sequence_two_steps() {
    let add_one = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    }));
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }));

    let seq = RunnableSequence::new(vec![add_one, double]).unwrap();
    // (5 + 1) * 2 = 12
    let result = seq.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(12));
}

#[tokio::test]
async fn test_sequence_three_steps() {
    let a = Arc::new(RunnableLambda::new("a", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    }));
    let b = Arc::new(RunnableLambda::new("b", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 3))
    }));
    let c = Arc::new(RunnableLambda::new("c", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() - 2))
    }));

    let seq = RunnableSequence::new(vec![a, b, c]).unwrap();
    // ((1 + 1) * 3) - 2 = 4
    let result = seq.invoke(json!(1), None).await.unwrap();
    assert_eq!(result, json!(4));
}

#[tokio::test]
async fn test_sequence_batch() {
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }));
    let add_one = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    }));

    let seq = RunnableSequence::new(vec![double, add_one]).unwrap();
    let results = seq
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    // (1*2)+1=3, (2*2)+1=5, (3*2)+1=7
    assert_eq!(results, vec![json!(3), json!(5), json!(7)]);
}

#[tokio::test]
async fn test_sequence_stream() {
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }));
    let add_one = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    }));

    let seq = RunnableSequence::new(vec![double, add_one]).unwrap();
    let mut stream = seq.stream(json!(5), None).await.unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, json!(11)); // (5*2)+1
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_sequence_recursion_limit() {
    let passthrough = Arc::new(RunnableLambda::new("pass", |v: Value| async move { Ok(v) }));
    let seq = RunnableSequence::new(vec![passthrough]).unwrap();

    let mut cfg = RunnableConfig::default();
    cfg.recursion_limit = 0;

    let result = seq.invoke(json!(1), Some(&cfg)).await;
    assert!(matches!(
        result.unwrap_err(),
        RustChainError::RecursionLimitExceeded(_)
    ));
}

#[tokio::test]
async fn test_sequence_empty_fails() {
    let result = RunnableSequence::new(vec![]);
    assert!(result.is_err());
}

// ─── Parallel Tests ───

#[tokio::test]
async fn test_parallel_fan_out() {
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    })) as Arc<dyn Runnable>;
    let triple = Arc::new(RunnableLambda::new("triple", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 3))
    })) as Arc<dyn Runnable>;

    let mut steps = HashMap::new();
    steps.insert("doubled".into(), double);
    steps.insert("tripled".into(), triple);

    let parallel = RunnableParallel::new(steps);
    let result = parallel.invoke(json!(5), None).await.unwrap();

    assert_eq!(result["doubled"], json!(10));
    assert_eq!(result["tripled"], json!(15));
}

#[tokio::test]
async fn test_parallel_error_propagation() {
    let fail = Arc::new(RunnableLambda::new("fail", |_v: Value| async move {
        Err(RustChainError::Other("boom".into()))
    })) as Arc<dyn Runnable>;

    let mut steps = HashMap::new();
    steps.insert("will_fail".into(), fail);

    let parallel = RunnableParallel::new(steps);
    let result = parallel.invoke(json!(1), None).await;
    assert!(result.is_err());
}

// ─── Passthrough & Assign Tests ───

#[tokio::test]
async fn test_passthrough_returns_input() {
    let pt = RunnablePassthrough::new();
    let result = pt.invoke(json!({"a": 1}), None).await.unwrap();
    assert_eq!(result, json!({"a": 1}));
}

#[tokio::test]
async fn test_assign_merges_keys() {
    let double_a = Arc::new(RunnableLambda::new("double_a", |v: Value| async move {
        let a = v["a"].as_i64().unwrap();
        Ok(json!(a * 2))
    })) as Arc<dyn Runnable>;

    let assign = RunnableAssign::new().assign("b", double_a);
    let result = assign.invoke(json!({"a": 5}), None).await.unwrap();

    assert_eq!(result["a"], json!(5));
    assert_eq!(result["b"], json!(10));
}

#[tokio::test]
async fn test_assign_non_object_errors() {
    let noop =
        Arc::new(RunnableLambda::new("noop", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;

    let assign = RunnableAssign::new().assign("x", noop);
    let result = assign.invoke(json!(42), None).await;
    assert!(matches!(
        result.unwrap_err(),
        RustChainError::TypeMismatch { .. }
    ));
}

// ─── Branch Tests ───

#[tokio::test]
async fn test_branch_first_match_wins() {
    let is_positive = Arc::new(RunnableLambda::new("is_pos", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() > 0))
    })) as Arc<dyn Runnable>;
    let pos_action = Arc::new(RunnableLambda::new("pos", |_v: Value| async move {
        Ok(json!("positive"))
    })) as Arc<dyn Runnable>;
    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("default"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(is_positive, pos_action)], default);
    let result = branch.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!("positive"));
}

#[tokio::test]
async fn test_branch_default_fallthrough() {
    let always_false = Arc::new(RunnableLambda::new("false", |_v: Value| async move {
        Ok(json!(false))
    })) as Arc<dyn Runnable>;
    let action = Arc::new(RunnableLambda::new("action", |_v: Value| async move {
        Ok(json!("should not reach"))
    })) as Arc<dyn Runnable>;
    let default = Arc::new(RunnableLambda::new("default", |_v: Value| async move {
        Ok(json!("fell through"))
    })) as Arc<dyn Runnable>;

    let branch = RunnableBranch::new(vec![(always_false, action)], default);
    let result = branch.invoke(json!(1), None).await.unwrap();
    assert_eq!(result, json!("fell through"));
}

// ─── Fallbacks Tests ───

#[tokio::test]
async fn test_fallbacks_primary_succeeds() {
    let primary = Arc::new(RunnableLambda::new("primary", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    })) as Arc<dyn Runnable>;
    let fallback = Arc::new(RunnableLambda::new("fallback", |_v: Value| async move {
        Ok(json!(-1))
    })) as Arc<dyn Runnable>;

    let with_fallbacks = RunnableWithFallbacks::new(primary).with_fallbacks(vec![fallback]);
    let result = with_fallbacks.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(6));
}

#[tokio::test]
async fn test_fallbacks_primary_fails() {
    let primary = Arc::new(RunnableLambda::new("primary", |_v: Value| async move {
        Err(RustChainError::Other("primary failed".into()))
    })) as Arc<dyn Runnable>;
    let fallback = Arc::new(RunnableLambda::new("fallback", |v: Value| async move {
        Ok(json!({"fallback": v}))
    })) as Arc<dyn Runnable>;

    let with_fallbacks = RunnableWithFallbacks::new(primary).with_fallbacks(vec![fallback]);
    let result = with_fallbacks.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!({"fallback": 5}));
}

#[tokio::test]
async fn test_fallbacks_all_fail() {
    let primary = Arc::new(RunnableLambda::new("primary", |_v: Value| async move {
        Err(RustChainError::Other("primary failed".into()))
    })) as Arc<dyn Runnable>;
    let fallback = Arc::new(RunnableLambda::new("fallback", |_v: Value| async move {
        Err(RustChainError::Other("fallback also failed".into()))
    })) as Arc<dyn Runnable>;

    let with_fallbacks = RunnableWithFallbacks::new(primary).with_fallbacks(vec![fallback]);
    let result = with_fallbacks.invoke(json!(5), None).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("fallback also failed"),
        "Expected last error, got: {}",
        err_msg
    );
}

// ─── Router Tests ───

#[tokio::test]
async fn test_router_correct_dispatch() {
    let upper = Arc::new(RunnableLambda::new("upper", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().to_uppercase()))
    })) as Arc<dyn Runnable>;
    let lower = Arc::new(RunnableLambda::new("lower", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().to_lowercase()))
    })) as Arc<dyn Runnable>;

    let mut runnables = HashMap::new();
    runnables.insert("upper".into(), upper);
    runnables.insert("lower".into(), lower);

    let router = RouterRunnable::new(runnables);
    let result = router
        .invoke(json!({"key": "upper", "input": "hello"}), None)
        .await
        .unwrap();
    assert_eq!(result, json!("HELLO"));
}

#[tokio::test]
async fn test_router_missing_key() {
    let noop =
        Arc::new(RunnableLambda::new("noop", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let mut runnables = HashMap::new();
    runnables.insert("exists".into(), noop);

    let router = RouterRunnable::new(runnables);
    let result = router
        .invoke(json!({"key": "missing", "input": 1}), None)
        .await;
    assert!(matches!(result.unwrap_err(), RustChainError::InvalidKey(_)));
}

// ─── Binding Tests ───

#[tokio::test]
async fn test_binding_kwargs_merged() {
    let echo =
        Arc::new(RunnableLambda::new("echo", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;

    let mut kwargs = HashMap::new();
    kwargs.insert("extra".into(), json!("bound_value"));

    let binding = RunnableBinding::new(echo, kwargs, None);
    let result = binding
        .invoke(json!({"original": "data"}), None)
        .await
        .unwrap();
    assert_eq!(result["original"], "data");
    assert_eq!(result["extra"], "bound_value");
}

// ─── Each Tests ───

#[tokio::test]
async fn test_each_maps_over_array() {
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    })) as Arc<dyn Runnable>;

    let each = RunnableEach::new(double);
    let result = each.invoke(json!([1, 2, 3]), None).await.unwrap();
    assert_eq!(result, json!([2, 4, 6]));
}

#[tokio::test]
async fn test_each_non_array_errors() {
    let noop =
        Arc::new(RunnableLambda::new("noop", |v: Value| async move { Ok(v) })) as Arc<dyn Runnable>;
    let each = RunnableEach::new(noop);
    let result = each.invoke(json!("not_array"), None).await;
    assert!(matches!(
        result.unwrap_err(),
        RustChainError::TypeMismatch { .. }
    ));
}

// ─── Integration Tests ───

#[tokio::test]
async fn test_complex_chain() {
    // Build: input → parallel(double, triple) → extract sum
    let double = Arc::new(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    })) as Arc<dyn Runnable>;
    let triple = Arc::new(RunnableLambda::new("triple", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 3))
    })) as Arc<dyn Runnable>;

    let mut par_steps = HashMap::new();
    par_steps.insert("doubled".into(), double);
    par_steps.insert("tripled".into(), triple);
    let parallel = Arc::new(RunnableParallel::new(par_steps)) as Arc<dyn Runnable>;

    let sum = Arc::new(RunnableLambda::new("sum", |v: Value| async move {
        let d = v["doubled"].as_i64().unwrap();
        let t = v["tripled"].as_i64().unwrap();
        Ok(json!(d + t))
    })) as Arc<dyn Runnable>;

    let seq = RunnableSequence::new(vec![parallel, sum]).unwrap();
    // 5*2=10, 5*3=15, 10+15=25
    let result = seq.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(25));
}

#[tokio::test]
async fn test_chain_macro() {
    let add_one = RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    });
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    let seq = chain!(add_one, double).unwrap();
    // (3 + 1) * 2 = 8
    let result = seq.invoke(json!(3), None).await.unwrap();
    assert_eq!(result, json!(8));
}

// ─── Retry Tests ───

use rustchain_core::runnables::RunnableExt;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn test_runnable_retry_succeeds_first_try() {
    let lambda = RunnableLambda::new("echo", |input| async move { Ok(input) });
    let retry = RunnableRetry::new(Arc::new(lambda), 3);
    let result = retry.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result, json!("hello"));
}

#[tokio::test]
async fn test_runnable_retry_succeeds_on_second_try() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    let lambda = RunnableLambda::new("flaky", move |input| {
        let c = c.clone();
        async move {
            let count = c.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Err(rustchain_core::error::RustChainError::Other(
                    "transient".into(),
                ))
            } else {
                Ok(input)
            }
        }
    });
    let retry = RunnableRetry::new(Arc::new(lambda), 3).with_wait(1, 10);
    let result = retry.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result, json!("hello"));
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_runnable_retry_exhausts_attempts() {
    let lambda = RunnableLambda::new("always_fail", |_input| async {
        Err::<serde_json::Value, _>(rustchain_core::error::RustChainError::Other(
            "permanent".into(),
        ))
    });
    let retry = RunnableRetry::new(Arc::new(lambda), 2).with_wait(1, 10);
    let result = retry.invoke(json!("hello"), None).await;
    assert!(result.is_err());
}

// ─── Configurable Tests ───

#[tokio::test]
async fn test_configurable_default() {
    let default_lambda = Arc::new(RunnableLambda::new("default", |input| async move {
        Ok(json!(format!("default: {}", input)))
    }));
    let configurable = RunnableConfigurableFields::new(
        default_lambda as Arc<dyn Runnable>,
        vec![ConfigurableField::new(
            "model",
            "Model",
            rustchain_core::runnables::ConfigurableFieldType::String,
        )],
    );
    let result = configurable.invoke(json!("test"), None).await.unwrap();
    assert_eq!(result, json!("default: \"test\""));
}

#[tokio::test]
async fn test_configurable_with_alternative() {
    let default_lambda = Arc::new(RunnableLambda::new("default", |_input| async {
        Ok(json!("default"))
    }));
    let alt_lambda = Arc::new(RunnableLambda::new("alt", |_input| async {
        Ok(json!("alternative"))
    }));
    let mut alts = std::collections::HashMap::new();
    alts.insert("gpt4".into(), alt_lambda as Arc<dyn Runnable>);
    let configurable = RunnableConfigurableFields::new(
        default_lambda as Arc<dyn Runnable>,
        vec![ConfigurableField::new(
            "model",
            "model",
            rustchain_core::runnables::ConfigurableFieldType::String,
        )],
    )
    .with_alternatives("model", alts);

    // Without config: default
    let result = configurable.invoke(json!("x"), None).await.unwrap();
    assert_eq!(result, json!("default"));

    // With config selecting alternative
    let mut config = rustchain_core::runnables::RunnableConfig::default();
    config.configurable.insert("model".into(), json!("gpt4"));
    let result = configurable
        .invoke(json!("x"), Some(&config))
        .await
        .unwrap();
    assert_eq!(result, json!("alternative"));
}

// ─── StreamEvent Tests ───

#[test]
fn test_stream_event_serialize() {
    let event = StreamEvent::Standard {
        event: "on_chain_start".into(),
        name: "MyChain".into(),
        run_id: "abc123".into(),
        tags: vec!["tag1".into()],
        metadata: Default::default(),
        parent_ids: vec![],
        data: EventData {
            input: Some(json!({"query": "hello"})),
            output: None,
            chunk: None,
            error: None,
        },
    };
    let v = serde_json::to_value(&event).unwrap();
    assert_eq!(v["event"], "on_chain_start");
    assert_eq!(v["name"], "MyChain");
    assert!(v["data"]["input"].is_object());
}

#[test]
fn test_event_data_default() {
    let data = EventData::default();
    assert!(data.input.is_none());
    assert!(data.output.is_none());
    assert!(data.chunk.is_none());
    assert!(data.error.is_none());
}

#[test]
fn test_stream_event_custom() {
    let event = StreamEvent::Custom {
        event: "on_custom_event".into(),
        name: "my_event".into(),
        run_id: "def456".into(),
        tags: vec![],
        metadata: Default::default(),
        parent_ids: vec![],
        data: json!({"custom": "data"}),
    };
    let v = serde_json::to_value(&event).unwrap();
    assert_eq!(v["event"], "on_custom_event");
    assert_eq!(v["data"]["custom"], "data");
}

// ─── RunnableExt: pipe Tests ───

#[tokio::test]
async fn test_ext_pipe_two_lambdas() {
    let add_one = RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    });
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    let chain = add_one.pipe(double).unwrap();
    // (5 + 1) * 2 = 12
    let result = chain.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(12));
}

#[tokio::test]
async fn test_ext_pipe_chained_three() {
    let a = RunnableLambda::new(
        "a",
        |v: Value| async move { Ok(json!(v.as_i64().unwrap() + 1)) },
    );
    let b = RunnableLambda::new(
        "b",
        |v: Value| async move { Ok(json!(v.as_i64().unwrap() * 3)) },
    );

    // Pipe a into b, then pipe the result into another lambda
    let ab = a.pipe(b).unwrap();
    let c = RunnableLambda::new(
        "c",
        |v: Value| async move { Ok(json!(v.as_i64().unwrap() - 2)) },
    );
    let abc = RunnableSequence::new(vec![
        Arc::new(ab) as Arc<dyn Runnable>,
        Arc::new(c) as Arc<dyn Runnable>,
    ])
    .unwrap();

    // ((1 + 1) * 3) - 2 = 4
    let result = abc.invoke(json!(1), None).await.unwrap();
    assert_eq!(result, json!(4));
}

// ─── RunnableExt: map Tests ───

#[tokio::test]
async fn test_ext_map_over_array() {
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    let mapped = double.map();
    let result = mapped.invoke(json!([1, 2, 3, 4]), None).await.unwrap();
    assert_eq!(result, json!([2, 4, 6, 8]));
}

#[tokio::test]
async fn test_ext_map_non_array_errors() {
    let noop = RunnableLambda::new("noop", |v: Value| async move { Ok(v) });
    let mapped = noop.map();
    let result = mapped.invoke(json!("not_array"), None).await;
    assert!(matches!(
        result.unwrap_err(),
        RustChainError::TypeMismatch { .. }
    ));
}

// ─── RunnableExt: with_fallbacks Tests ───

#[tokio::test]
async fn test_ext_with_fallbacks_primary_ok() {
    let primary = RunnableLambda::new("primary", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 10))
    });
    let fallback = Arc::new(RunnableLambda::new("fallback", |_v: Value| async move {
        Ok(json!(-1))
    })) as Arc<dyn Runnable>;

    let chain = primary.with_fallbacks(vec![fallback]);
    let result = chain.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(15));
}

#[tokio::test]
async fn test_ext_with_fallbacks_primary_fails() {
    let primary = RunnableLambda::new("primary", |_v: Value| async move {
        Err(RustChainError::Other("fail".into()))
    });
    let fallback = Arc::new(RunnableLambda::new("fallback", |v: Value| async move {
        Ok(json!({"recovered": v}))
    })) as Arc<dyn Runnable>;

    let chain = primary.with_fallbacks(vec![fallback]);
    let result = chain.invoke(json!(42), None).await.unwrap();
    assert_eq!(result, json!({"recovered": 42}));
}

// ─── RunnableExt: with_retry Tests ───

#[tokio::test]
async fn test_ext_with_retry_succeeds() {
    let lambda = RunnableLambda::new("ok", |v: Value| async move { Ok(v) });
    let retried = lambda.with_retry(3, 1);
    let result = retried.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result, json!("hello"));
}

#[tokio::test]
async fn test_ext_with_retry_recovers() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    let lambda = RunnableLambda::new("flaky", move |v| {
        let c = c.clone();
        async move {
            let count = c.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Err(RustChainError::Other("transient".into()))
            } else {
                Ok(v)
            }
        }
    });
    let retried = lambda.with_retry(5, 1);
    let result = retried.invoke(json!("data"), None).await.unwrap();
    assert_eq!(result, json!("data"));
    assert_eq!(counter.load(Ordering::SeqCst), 3); // failed twice, succeeded on third
}

#[tokio::test]
async fn test_ext_with_retry_exhausted() {
    let lambda = RunnableLambda::new("fail", |_v: Value| async move {
        Err::<Value, _>(RustChainError::Other("always fails".into()))
    });
    let retried = lambda.with_retry(2, 1);
    let result = retried.invoke(json!("x"), None).await;
    assert!(result.is_err());
}

// ─── RunnableExt: assign Tests ───

#[tokio::test]
async fn test_ext_assign_merges_computed_keys() {
    let passthrough = RunnableLambda::new("pass", |v: Value| async move { Ok(v) });

    let compute_b = Arc::new(RunnableLambda::new("compute_b", |v: Value| async move {
        let a = v["a"].as_i64().unwrap();
        Ok(json!(a * 10))
    })) as Arc<dyn Runnable>;

    let mut mapping = HashMap::new();
    mapping.insert("b".into(), compute_b);

    let chain = passthrough.assign(mapping).unwrap();
    let result = chain.invoke(json!({"a": 3}), None).await.unwrap();

    assert_eq!(result["a"], json!(3));
    assert_eq!(result["b"], json!(30));
}

#[tokio::test]
async fn test_ext_assign_multiple_keys() {
    let passthrough = RunnableLambda::new("pass", |v: Value| async move { Ok(v) });

    let upper = Arc::new(RunnableLambda::new("upper", |v: Value| async move {
        let name = v["name"].as_str().unwrap().to_uppercase();
        Ok(json!(name))
    })) as Arc<dyn Runnable>;
    let length = Arc::new(RunnableLambda::new("length", |v: Value| async move {
        let len = v["name"].as_str().unwrap().len();
        Ok(json!(len))
    })) as Arc<dyn Runnable>;

    let mut mapping = HashMap::new();
    mapping.insert("upper_name".into(), upper);
    mapping.insert("name_length".into(), length);

    let chain = passthrough.assign(mapping).unwrap();
    let result = chain.invoke(json!({"name": "alice"}), None).await.unwrap();

    assert_eq!(result["name"], json!("alice"));
    assert_eq!(result["upper_name"], json!("ALICE"));
    assert_eq!(result["name_length"], json!(5));
}

// ─── Stream Tests ───

#[tokio::test]
async fn test_stream_single_runnable_returns_result() {
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    let mut stream = double.stream(json!(5), None).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, json!(10));
    assert!(
        stream.next().await.is_none(),
        "default stream should yield exactly one item"
    );
}

#[tokio::test]
async fn test_stream_sequence_propagates_last_step() {
    let add_one = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    }));
    let triple = Arc::new(RunnableLambda::new("triple", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 3))
    }));

    let seq = RunnableSequence::new(vec![add_one, triple]).unwrap();
    let mut stream = seq.stream(json!(4), None).await.unwrap();

    // (4 + 1) * 3 = 15
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, json!(15));
    assert!(stream.next().await.is_none());
}

/// A mock runnable whose `stream()` returns multiple chunks.
struct MultiChunkRunnable {
    chunks: Vec<Value>,
}

impl MultiChunkRunnable {
    fn new(chunks: Vec<Value>) -> Self {
        Self { chunks }
    }
}

#[async_trait::async_trait]
impl Runnable for MultiChunkRunnable {
    fn name(&self) -> &str {
        "MultiChunkRunnable"
    }

    async fn invoke(
        &self,
        _input: Value,
        _config: Option<&RunnableConfig>,
    ) -> rustchain_core::error::Result<Value> {
        // For invoke, concatenate all chunks into a single string
        let combined: String = self.chunks.iter().filter_map(|v| v.as_str()).collect();
        Ok(json!(combined))
    }

    async fn stream(
        &self,
        _input: Value,
        _config: Option<&RunnableConfig>,
    ) -> rustchain_core::error::Result<rustchain_core::runnables::RunnableStream> {
        let chunks: Vec<rustchain_core::error::Result<Value>> =
            self.chunks.iter().map(|c| Ok(c.clone())).collect();
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

#[tokio::test]
async fn test_stream_multi_chunk_runnable() {
    let chunker = MultiChunkRunnable::new(vec![
        json!("Hello"),
        json!(", "),
        json!("world"),
        json!("!"),
    ]);

    let mut stream = chunker.stream(json!(null), None).await.unwrap();
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    assert_eq!(
        collected,
        vec![json!("Hello"), json!(", "), json!("world"), json!("!")]
    );
}

#[tokio::test]
async fn test_stream_sequence_with_multi_chunk_last_step() {
    // First step transforms input, last step streams multiple chunks
    let add_prefix = Arc::new(RunnableLambda::new("add_prefix", |v: Value| async move {
        let s = v.as_str().unwrap_or("unknown");
        Ok(json!(format!("prefix_{}", s)))
    }));

    let chunker = Arc::new(MultiChunkRunnable::new(vec![
        json!("chunk1"),
        json!("chunk2"),
        json!("chunk3"),
    ]));

    let seq = RunnableSequence::new(vec![
        add_prefix as Arc<dyn Runnable>,
        chunker as Arc<dyn Runnable>,
    ])
    .unwrap();

    let mut stream = seq.stream(json!("input"), None).await.unwrap();
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    // The multi-chunk runnable ignores input for streaming, returns its fixed chunks
    assert_eq!(collected.len(), 3);
    assert_eq!(
        collected,
        vec![json!("chunk1"), json!("chunk2"), json!("chunk3")]
    );
}

// ─── RunnableExt: batch (via trait) Tests ───

#[tokio::test]
async fn test_ext_batch_on_lambda() {
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    // Use the Runnable trait batch method directly
    let results = double
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    assert_eq!(results, vec![json!(2), json!(4), json!(6)]);
}

#[tokio::test]
async fn test_ext_batch_on_pipe_chain() {
    let add_one = RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    });
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    let chain = add_one.pipe(double).unwrap();
    let results = chain
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();
    // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert_eq!(results, vec![json!(4), json!(6), json!(8)]);
}

// ─── RunnableExt: composition combination Tests ───

#[tokio::test]
async fn test_ext_pipe_then_map() {
    let add_one = RunnableLambda::new("add_one", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    });
    let double = RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    });

    // Create a chain, then map it over an array
    let chain = add_one.pipe(double).unwrap();
    let mapped = chain.map();
    let result = mapped.invoke(json!([1, 2, 3]), None).await.unwrap();
    // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert_eq!(result, json!([4, 6, 8]));
}

#[tokio::test]
async fn test_ext_pipe_with_fallbacks() {
    let fail_step = RunnableLambda::new("fail", |_v: Value| async move {
        Err(RustChainError::Other("step failed".into()))
    });
    let safe_step = RunnableLambda::new("safe", |v: Value| async move { Ok(json!({"safe": v})) });

    let fallback = Arc::new(RunnableLambda::new("fallback", |v: Value| async move {
        Ok(json!({"fallback_used": v}))
    })) as Arc<dyn Runnable>;

    // fail_step piped to safe_step will fail at fail_step,
    // so fallbacks should kick in
    let chain = fail_step.pipe(safe_step).unwrap();
    let with_fb = chain.with_fallbacks(vec![fallback]);
    let result = with_fb.invoke(json!(99), None).await.unwrap();
    assert_eq!(result, json!({"fallback_used": 99}));
}
