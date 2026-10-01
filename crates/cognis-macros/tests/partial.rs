//! Behavior tests for `#[derive(Partial)]`.

use cognis_core::Partial;
use cognis_macros::Partial as DerivePartial;

#[derive(DerivePartial)]
#[allow(dead_code)]
struct Report {
    title: String,
    score: u32,
    tags: Vec<String>,
}

type P = <Report as Partial>::Partial;

#[test]
fn partial_mirror_deserializes_missing_fields_as_none() {
    let p: P = serde_json::from_str("{\"title\":\"x\"}").unwrap();
    assert_eq!(p.title.as_deref(), Some("x"));
    assert_eq!(p.score, None);
    assert_eq!(p.tags, None);
}

#[test]
fn partial_mirror_deserializes_full_object() {
    let p: P = serde_json::from_str("{\"title\":\"x\",\"score\":5,\"tags\":[\"a\"]}").unwrap();
    assert_eq!(p.title.as_deref(), Some("x"));
    assert_eq!(p.score, Some(5));
    assert_eq!(p.tags, Some(vec!["a".to_string()]));
}

#[test]
fn partial_mirror_default_and_debug_work() {
    let p = P::default();
    assert!(p.title.is_none() && p.score.is_none() && p.tags.is_none());
    assert!(format!("{p:?}").contains("score: None"));
}

#[test]
fn partial_mirror_rejects_wrong_field_type() {
    let r = serde_json::from_str::<P>("{\"score\":\"high\"}");
    assert!(r.is_err());
}
