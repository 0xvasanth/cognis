//! File Management Tools Example
//!
//! Demonstrates file tools (read, write, list, search, info) sandboxed in a temp directory,
//! then uses an LLM to generate content and write it to a file.

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use serde_json::{json, Value};

use cognis::tools::file_management::{
    create_file_toolkit, FileInfoTool, FileSystemConfig, ListDirectoryTool, ReadFileTool,
    SearchFilesTool, WriteFileTool,
};
use cognis_core::tools::base::{BaseTool, BaseToolkit};
use cognis_core::tools::types::{ToolInput, ToolOutput};

fn structured_input(pairs: &[(&str, &str)]) -> ToolInput {
    let map: HashMap<String, Value> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect();
    ToolInput::Structured(map)
}

fn text_input(s: &str) -> ToolInput {
    ToolInput::Text(s.to_string())
}

fn content_str(output: &ToolOutput) -> String {
    match output {
        ToolOutput::Content(Value::String(s)) => s.clone(),
        ToolOutput::Content(v) => v.to_string(),
        other => format!("{:?}", other),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = tempfile::TempDir::new()?;
    let root = tmp_dir.path();
    let config = FileSystemConfig::new(root);
    println!("Working directory: {}\n", root.display());

    // Write files
    let write_tool = WriteFileTool::new(config.clone());
    for (path, content) in [
        ("hello.txt", "Hello from Cognis!"),
        (
            "src/main.rs",
            "fn main() {\n    println!(\"Hello, world!\");\n}\n",
        ),
        (
            "config/settings.toml",
            "[app]\nname = \"demo\"\nversion = \"1.0\"",
        ),
    ] {
        let r = write_tool
            ._run(structured_input(&[("path", path), ("content", content)]))
            .await?;
        println!("Write: {}", content_str(&r));
    }

    // Read a file back
    let read_tool = ReadFileTool::new(config.clone());
    let r = read_tool._run(text_input("hello.txt")).await?;
    println!("\nRead hello.txt: \"{}\"", content_str(&r));

    // List directory
    let list_tool = ListDirectoryTool::new(config.clone());
    let r = list_tool._run(text_input(".")).await?;
    println!(
        "\nRoot listing:\n  {}",
        content_str(&r).replace('\n', "\n  ")
    );

    // Search by glob
    let search_tool = SearchFilesTool::new(config.clone());
    let r = search_tool
        ._run(structured_input(&[("pattern", "**/*.rs")]))
        .await?;
    println!(
        "\nGlob '**/*.rs':\n  {}",
        content_str(&r).replace('\n', "\n  ")
    );

    // File info
    let info_tool = FileInfoTool::new(config.clone());
    let r = info_tool._run(text_input("hello.txt")).await?;
    if let ToolOutput::Content(v) = &r {
        println!(
            "\nhello.txt info: size={} is_file={}",
            v["size"], v["is_file"]
        );
    }

    // Toolkit overview
    let toolkit = create_file_toolkit(config.clone());
    println!("\nToolkit: {} tools", toolkit.get_tools().len());

    // Error handling: read missing file
    if let Err(e) = read_tool._run(text_input("nonexistent.txt")).await {
        println!("Missing file error: {}", e);
    }

    // LLM-generated content written to file
    let model = shared::get_chat_model(vec![
        "# Cognis\n\nRust LLM framework.\n\n- Chat models\n- File tools\n- Graph engine".into(),
    ]);
    let messages = vec![
        cognis_core::messages::Message::system("You are a technical writer."),
        cognis_core::messages::Message::human("Write a short README for Cognis."),
    ];
    if let Ok(resp) = model.invoke_messages(&messages, None).await {
        let text = resp.base.content.text();
        write_tool
            ._run(structured_input(&[
                ("path", "llm_generated.md"),
                ("content", &text),
            ]))
            .await?;
        let r = read_tool._run(text_input("llm_generated.md")).await?;
        println!(
            "\nLLM-generated file:\n  {}",
            content_str(&r).replace('\n', "\n  ")
        );
    }

    Ok(())
}
