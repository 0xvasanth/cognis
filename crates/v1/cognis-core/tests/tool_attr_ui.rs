//! trybuild-driven compile-failure tests for #[cognis::tool], #[tools_impl],
//! and #[derive(GraphStateV2)].

#[test]
fn ui_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/tool_attr/*.rs");
    t.compile_fail("tests/ui/tools_impl_*.rs");
    t.compile_fail("tests/ui/graph_state_v2_*.rs");
}
