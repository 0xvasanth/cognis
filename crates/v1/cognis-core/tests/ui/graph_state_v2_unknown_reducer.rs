use cognis_macros::GraphStateV2;

#[derive(GraphStateV2)]
#[graph_state(crate_path = "crate::stub")]
pub struct Bad {
    #[reducer(unknown_op)]
    pub field: u32,
}

mod stub {
    pub trait GraphState { type Update; fn apply(&mut self, _: Self::Update); }
    pub fn __merge_json(_: &mut serde_json::Value, _: serde_json::Value) {}
}

fn main() {}
