//! `#[tool]` impl-form with `&mut self` — must fail because the
//! generated `BaseTool::_run` takes `&self`.

use cognis_core::tool;

pub struct Counter;

#[tool]
impl Counter {
    async fn bump(&mut self) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
        Ok(cognis_core::tools::ToolOutput::Content(
            serde_json::json!(null),
        ))
    }
}

fn main() {}
