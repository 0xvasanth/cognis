//! V2 file-management tools, backed by an in-memory backend.
//! All six tools share an `Arc<dyn Backend>` so they see the same
//! virtual filesystem.

use std::collections::HashMap;
use std::sync::Arc;

use cognis::prelude::*;
use cognis::tools::{
    FileEditTool, FileExistsTool, FileGlobTool, FileListTool, FileReadTool, FileWriteTool,
};
use cognis::{Backend, MemoryBackend};
use cognis_llm::tools::{Tool, ToolInput};
use serde_json::{json, Value};

fn structured(pairs: &[(&str, Value)]) -> ToolInput {
    let map: HashMap<String, Value> = pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    ToolInput::Structured(map)
}

#[tokio::main]
async fn main() -> Result<()> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let writer = FileWriteTool::new(backend.clone());
    let reader = FileReadTool::new(backend.clone());
    let lister = FileListTool::new(backend.clone());
    let exists = FileExistsTool::new(backend.clone());
    let editor = FileEditTool::new(backend.clone());
    let globber = FileGlobTool::new(backend.clone());

    writer._run(structured(&[("path", json!("hello.txt")), ("contents", json!("hi there"))])).await?;
    println!("wrote hello.txt");

    let read = reader._run(structured(&[("path", json!("hello.txt"))])).await?;
    println!("read: {read:?}");

    let listing = lister._run(structured(&[("path", json!("."))])).await?;
    println!("ls: {listing:?}");

    let ex = exists._run(structured(&[("path", json!("hello.txt"))])).await?;
    println!("exists: {ex:?}");

    editor._run(structured(&[
        ("path", json!("hello.txt")),
        ("find", json!("hi there")),
        ("replace", json!("hello world")),
    ])).await?;
    println!("edited hello.txt");

    let glob = globber._run(structured(&[("pattern", json!("*.txt"))])).await?;
    println!("glob: {glob:?}");

    Ok(())
}
