//! trybuild-driven compile-failure tests for `#[cognis::tool]`.
//!
//! Each fixture in `tests/ui/` must fail to compile with a specific
//! error message. The `.stderr` files are captured on first run
//! (`TRYBUILD=overwrite cargo test -p cognis-core --test tool_attr_ui`)
//! and then committed alongside the `.rs` fixtures.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/tool_attr/*.rs");
}
