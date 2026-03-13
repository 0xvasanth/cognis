//! File Management Tools Example
//!
//! Demonstrates the file management toolkit: read, write, list, search, and
//! file info operations, all sandboxed within a temporary directory.
//!
//! No API keys required -- operates on a local temp directory.

mod shared;
use std::collections::HashMap;

use serde_json::{json, Value};

use cognis::tools::file_management::{
    create_file_toolkit, FileInfoTool, FileSystemConfig, ListDirectoryTool, ReadFileTool,
    SearchFilesTool, WriteFileTool,
};
use cognis_core::tools::base::{BaseTool, BaseToolkit};
use cognis_core::tools::types::ToolInput;

/// Helper to create a structured tool input from key-value pairs.
fn structured_input(pairs: &[(&str, &str)]) -> ToolInput {
    let map: HashMap<String, Value> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect();
    ToolInput::Structured(map)
}

/// Helper to create a text tool input.
fn text_input(s: &str) -> ToolInput {
    ToolInput::Text(s.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== File Management Tools Example ===\n");

    // Create a temporary directory that will be automatically cleaned up.
    let tmp_dir = tempfile::TempDir::new()?;
    let root = tmp_dir.path();
    println!("  Working directory: {}\n", root.display());

    let config = FileSystemConfig::new(root);

    // =========================================================================
    // 1. WriteFileTool: create files
    // =========================================================================
    println!("--- 1. Writing Files ---\n");

    let write_tool = WriteFileTool::new(config.clone());

    // Write a simple text file
    let result = write_tool
        ._run(structured_input(&[
            ("path", "hello.txt"),
            ("content", "Hello from Cognis!"),
        ]))
        .await?;
    println!("  {}", extract_content_str(&result));

    // Write a Rust source file
    let rust_code = r#"fn main() {
    println!("Hello, world!");
}
"#;
    let result = write_tool
        ._run(structured_input(&[
            ("path", "src/main.rs"),
            ("content", rust_code),
        ]))
        .await?;
    println!("  {}", extract_content_str(&result));

    // Write a config file in a nested directory
    let result = write_tool
        ._run(structured_input(&[
            ("path", "config/settings.toml"),
            ("content", "[app]\nname = \"demo\"\nversion = \"1.0\""),
        ]))
        .await?;
    println!("  {}", extract_content_str(&result));

    // Write another file for search demo
    let result = write_tool
        ._run(structured_input(&[
            ("path", "src/lib.rs"),
            ("content", "pub fn add(a: i32, b: i32) -> i32 { a + b }"),
        ]))
        .await?;
    println!("  {}", extract_content_str(&result));

    // =========================================================================
    // 2. ReadFileTool: read file contents
    // =========================================================================
    println!("\n--- 2. Reading Files ---\n");

    let read_tool = ReadFileTool::new(config.clone());

    let result = read_tool._run(text_input("hello.txt")).await?;
    println!("  hello.txt content: \"{}\"", extract_content_str(&result));

    let result = read_tool._run(text_input("src/main.rs")).await?;
    println!(
        "  src/main.rs content:\n    {}",
        extract_content_str(&result).replace('\n', "\n    ")
    );

    // Using run_json (the JSON-based interface)
    let result = read_tool
        .run_json(&json!({"path": "config/settings.toml"}))
        .await?;
    println!(
        "  config/settings.toml (via run_json): \"{}\"",
        result.as_str().unwrap_or("")
    );

    // =========================================================================
    // 3. ListDirectoryTool: list directory contents
    // =========================================================================
    println!("\n--- 3. Listing Directories ---\n");

    let list_tool = ListDirectoryTool::new(config.clone());

    let result = list_tool._run(text_input(".")).await?;
    println!("  Root directory:");
    for line in extract_content_str(&result).lines() {
        println!("    {}", line);
    }

    let result = list_tool._run(text_input("src")).await?;
    println!("\n  src/ directory:");
    for line in extract_content_str(&result).lines() {
        println!("    {}", line);
    }

    // =========================================================================
    // 4. SearchFilesTool: find files by glob pattern
    // =========================================================================
    println!("\n--- 4. Searching Files ---\n");

    let search_tool = SearchFilesTool::new(config.clone());

    let result = search_tool
        ._run(structured_input(&[("pattern", "**/*.rs")]))
        .await?;
    println!("  Pattern '**/*.rs' matches:");
    for line in extract_content_str(&result).lines() {
        println!("    {}", line);
    }

    let result = search_tool
        ._run(structured_input(&[("pattern", "*.txt")]))
        .await?;
    println!("\n  Pattern '*.txt' matches:");
    for line in extract_content_str(&result).lines() {
        println!("    {}", line);
    }

    let result = search_tool
        ._run(structured_input(&[("pattern", "*.xyz")]))
        .await?;
    println!("\n  Pattern '*.xyz': {}", extract_content_str(&result));

    // =========================================================================
    // 5. FileInfoTool: get file metadata
    // =========================================================================
    println!("\n--- 5. File Info ---\n");

    let info_tool = FileInfoTool::new(config.clone());

    let result = info_tool._run(text_input("hello.txt")).await?;
    let info = extract_content_value(&result);
    println!("  hello.txt metadata:");
    println!("    size: {} bytes", info["size"]);
    println!("    is_file: {}", info["is_file"]);
    println!("    is_dir: {}", info["is_dir"]);
    println!("    readonly: {}", info["readonly"]);

    let result = info_tool._run(text_input("src")).await?;
    let info = extract_content_value(&result);
    println!("\n  src/ metadata:");
    println!("    is_file: {}", info["is_file"]);
    println!("    is_dir: {}", info["is_dir"]);

    // =========================================================================
    // 6. FileToolkit: get all tools at once
    // =========================================================================
    println!("\n--- 6. FileToolkit ---\n");

    let toolkit = create_file_toolkit(config.clone());
    let tools = toolkit.get_tools();
    println!("  Toolkit provides {} tools:", tools.len());
    for tool in &tools {
        println!("    - {} : {}", tool.name(), tool.description());
    }

    // =========================================================================
    // 7. Error handling demos
    // =========================================================================
    println!("\n--- 7. Error Handling ---\n");

    // Read non-existent file
    let result = read_tool._run(text_input("nonexistent.txt")).await;
    match result {
        Err(e) => println!("  Reading missing file: Error - {}", e),
        Ok(_) => println!("  Reading missing file: (unexpected success)"),
    }

    // Read-only config
    let mut ro_config = config.clone();
    ro_config.read_only = true;
    let ro_write = WriteFileTool::new(ro_config);
    let result = ro_write
        ._run(structured_input(&[
            ("path", "test.txt"),
            ("content", "data"),
        ]))
        .await;
    match result {
        Err(e) => println!("  Writing in read-only mode: Error - {}", e),
        Ok(_) => println!("  Writing in read-only mode: (unexpected success)"),
    }

    println!("\n=== File Management Tools Example Complete ===");
    Ok(())
}

/// Extract string content from a ToolOutput.
fn extract_content_str(output: &cognis_core::tools::types::ToolOutput) -> String {
    match output {
        cognis_core::tools::types::ToolOutput::Content(Value::String(s)) => s.clone(),
        cognis_core::tools::types::ToolOutput::Content(v) => v.to_string(),
        other => format!("{:?}", other),
    }
}

/// Extract a JSON value from a ToolOutput.
fn extract_content_value(output: &cognis_core::tools::types::ToolOutput) -> Value {
    match output {
        cognis_core::tools::types::ToolOutput::Content(v) => v.clone(),
        _ => json!(null),
    }
}
