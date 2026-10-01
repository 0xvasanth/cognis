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

#[derive(DerivePartial)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct CamelReport {
    report_title: String,
    total_score: u32,
}

#[test]
fn rename_all_mirror_reads_camel_case_keys() {
    type C = <CamelReport as Partial>::Partial;
    let p: C = serde_json::from_str("{\"reportTitle\":\"Q3\",\"totalScore\":92}").unwrap();
    assert_eq!(p.report_title.as_deref(), Some("Q3"));
    assert_eq!(p.total_score, Some(92));

    // The snake_case spelling is not the wire name any more, so it is an
    // unknown key and leaves the field unset.
    let p: C = serde_json::from_str("{\"report_title\":\"Q3\"}").unwrap();
    assert_eq!(p.report_title, None);
}

#[derive(DerivePartial)]
#[allow(dead_code)]
struct RenamedFields {
    #[serde(rename = "name")]
    title: String,
    #[serde(alias = "points", alias = "pts")]
    score: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[test]
fn field_rename_mirror_reads_renamed_key_only() {
    type R = <RenamedFields as Partial>::Partial;
    let p: R = serde_json::from_str("{\"name\":\"x\"}").unwrap();
    assert_eq!(p.title.as_deref(), Some("x"));
    let p: R = serde_json::from_str("{\"title\":\"x\"}").unwrap();
    assert_eq!(p.title, None, "the Rust field name is no longer a key");
}

#[test]
fn field_alias_mirror_reads_primary_and_alias_keys() {
    type R = <RenamedFields as Partial>::Partial;
    for json in ["{\"score\":5}", "{\"points\":5}", "{\"pts\":5}"] {
        let p: R = serde_json::from_str(json).unwrap();
        assert_eq!(p.score, Some(5), "input: {json}");
    }
}

#[test]
fn option_field_mirror_is_doubly_optional() {
    type R = <RenamedFields as Partial>::Partial;
    let absent: R = serde_json::from_str("{}").unwrap();
    assert_eq!(absent.note, None);
    let present: R = serde_json::from_str("{\"note\":\"n\"}").unwrap();
    assert_eq!(present.note, Some(Some("n".to_string())));
    let null: R = serde_json::from_str("{\"note\":null}").unwrap();
    assert_eq!(null.note, None, "null is indistinguishable from absent");
}

/// The derive must coexist with serde's own derives on the same struct,
/// which register the same `serde` helper attribute.
#[derive(serde::Serialize, serde::Deserialize, DerivePartial)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
struct Both {
    #[serde(rename = "ID")]
    id: u32,
    display_name: String,
}

#[test]
fn derive_coexists_with_serde_derives_and_agrees_on_keys() {
    let full = Both {
        id: 7,
        display_name: "n".into(),
    };
    let wire = serde_json::to_string(&full).unwrap();
    assert_eq!(wire, "{\"ID\":7,\"DISPLAY_NAME\":\"n\"}");
    let p: <Both as Partial>::Partial = serde_json::from_str(&wire).unwrap();
    assert_eq!(p.id, Some(7));
    assert_eq!(p.display_name.as_deref(), Some("n"));
    let back: Both = serde_json::from_str(&wire).unwrap();
    assert_eq!(back.id, 7);
}

mod elsewhere {
    pub use cognis_core as core_reexport;
}

#[derive(DerivePartial)]
#[partial(crate = "crate::elsewhere::core_reexport")]
#[allow(dead_code)]
struct ViaReexport {
    title: String,
}

#[test]
fn partial_crate_attribute_resolves_trait_and_serde_through_given_path() {
    let p: <ViaReexport as elsewhere::core_reexport::Partial>::Partial =
        serde_json::from_str("{\"title\":\"x\"}").unwrap();
    assert_eq!(p.title.as_deref(), Some("x"));
}

/// A public source struct yields a public, documented mirror: this module
/// would fail to compile under `deny(missing_docs)` otherwise.
#[deny(missing_docs)]
pub mod documented {
    /// A report.
    #[derive(cognis_macros::Partial)]
    pub struct PublicReport {
        /// Title.
        pub title: String,
    }
}

#[test]
fn public_mirror_compiles_under_deny_missing_docs() {
    let p = documented::PublicReportPartial::default();
    assert!(p.title.is_none());
}
