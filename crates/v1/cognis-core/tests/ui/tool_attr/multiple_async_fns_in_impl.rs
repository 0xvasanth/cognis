//! `#[tool]` impl-form with multiple async fns — must fail; exactly
//! one async fn is required per impl block.

use cognis_core::tool;

pub struct Searcher;

#[tool]
impl Searcher {
    async fn search(&self, _q: String) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
        Ok(cognis_core::tools::ToolOutput::Content(serde_json::json!(null)))
    }
    async fn also_search(&self, _q: String) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
        Ok(cognis_core::tools::ToolOutput::Content(serde_json::json!(null)))
    }
}

fn main() {}
