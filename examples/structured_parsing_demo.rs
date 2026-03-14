//! Structured Output Parsing Demo
//!
//! Demonstrates structured output parsing from `cognis::output_parsers::structured`:
//! - StructuredParser for parsing JSON blocks, key-value pairs, and lists
//! - SchemaEnforcer for validating parsed output against a schema
//! - OutputRepairer for fixing malformed JSON
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example structured_parsing_demo`

mod shared;
use cognis::output_parsers::structured::{
    JsonType, OutputRepairer, SchemaEnforcer, StructuredParser,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Structured Output Parsing Demo ===\n");

    let parser = StructuredParser::new();

    // -----------------------------------------------------------------------
    // 1. Parsing JSON blocks from LLM-style output
    // -----------------------------------------------------------------------
    println!("--- 1. Parsing JSON Blocks ---\n");

    // 1a. JSON wrapped in markdown code fences (common LLM output)
    let llm_output_fenced = r#"
Here is the information you requested:

```json
{
  "name": "Rust",
  "year": 2010,
  "paradigm": "systems programming",
  "memory_safe": true
}
```

I hope that helps!
"#;

    println!("Input (fenced JSON):");
    println!("{llm_output_fenced}");

    match parser.parse_json_block(llm_output_fenced) {
        Ok(value) => println!(
            "Parsed: {}\n",
            serde_json::to_string_pretty(&value).unwrap()
        ),
        Err(e) => println!("Error: {e}\n"),
    }

    // 1b. JSON embedded in prose (no code fences)
    let llm_output_inline = r#"Based on my analysis, the result is {"score": 95, "grade": "A", "passed": true} which indicates excellent performance."#;

    println!("Input (inline JSON):");
    println!("{llm_output_inline}\n");

    match parser.parse_json_block(llm_output_inline) {
        Ok(value) => println!(
            "Parsed: {}\n",
            serde_json::to_string_pretty(&value).unwrap()
        ),
        Err(e) => println!("Error: {e}\n"),
    }

    // 1c. Using parse_json for clean JSON with metadata
    let clean_json = r#"{"temperature": 72.5, "unit": "fahrenheit", "location": "San Francisco"}"#;

    println!("Input (clean JSON):");
    println!("{clean_json}\n");

    match parser.parse_json::<serde_json::Value>(clean_json) {
        Ok(result) => {
            println!(
                "Parsed value: {}",
                serde_json::to_string_pretty(&result.value).unwrap()
            );
            println!("Format: {}", result.format);
            println!("Parse duration: {} us", result.parse_duration_us);
            println!("Has warnings: {}\n", result.has_warnings());
        }
        Err(e) => println!("Error: {e}\n"),
    }

    // -----------------------------------------------------------------------
    // 2. Parsing key-value pairs
    // -----------------------------------------------------------------------
    println!("--- 2. Parsing Key-Value Pairs ---\n");

    let kv_output = "\
Name: Claude
Version: 3.5
Type: Large Language Model
Provider: Anthropic
Release Year: 2024";

    println!("Input:");
    println!("{kv_output}\n");

    let kv_map = parser.parse_key_value(kv_output);
    println!("Parsed key-value pairs:");
    let mut sorted_keys: Vec<_> = kv_map.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        println!("  {key}: {}", kv_map[key]);
    }

    // -----------------------------------------------------------------------
    // 3. Parsing lists
    // -----------------------------------------------------------------------
    println!("\n--- 3. Parsing Lists ---\n");

    let list_output = "\
Here are the top programming languages:
- Rust
- Python
- TypeScript
* Go
* Java
1. C++
2. Kotlin
3. Swift";

    println!("Input:");
    println!("{list_output}\n");

    let items = parser.parse_list(list_output);
    println!("Parsed list ({} items):", items.len());
    for (i, item) in items.iter().enumerate() {
        println!("  [{i}] {item}");
    }

    // -----------------------------------------------------------------------
    // 4. SchemaEnforcer — validating parsed output
    // -----------------------------------------------------------------------
    println!("\n--- 4. SchemaEnforcer — Validating Output ---\n");

    let enforcer = SchemaEnforcer::new()
        .require_field("name")
        .require_field("age")
        .require_type("name", JsonType::String)
        .require_type("age", JsonType::Number)
        .require_type("active", JsonType::Bool)
        .optional_field("role", serde_json::json!("user"));

    // 4a. Valid input
    let valid_json: serde_json::Value = serde_json::json!({
        "name": "Alice",
        "age": 30,
        "active": true
    });

    println!(
        "Validating: {}",
        serde_json::to_string(&valid_json).unwrap()
    );
    match enforcer.validate(&valid_json) {
        Ok(result) => {
            println!("  Valid! Result with defaults applied:");
            println!("  {}\n", serde_json::to_string_pretty(&result).unwrap());
        }
        Err(violations) => {
            println!("  Invalid:");
            for v in &violations {
                println!("    - {v}");
            }
            println!();
        }
    }

    // 4b. Missing required field
    let missing_field: serde_json::Value = serde_json::json!({
        "name": "Bob"
    });

    println!(
        "Validating: {}",
        serde_json::to_string(&missing_field).unwrap()
    );
    match enforcer.validate(&missing_field) {
        Ok(result) => println!("  Valid: {result}\n"),
        Err(violations) => {
            println!("  Invalid ({} violation(s)):", violations.len());
            for v in &violations {
                println!("    - {v}");
            }
            println!();
        }
    }

    // 4c. Wrong type
    let wrong_type: serde_json::Value = serde_json::json!({
        "name": 12345,
        "age": "not a number",
        "active": true
    });

    println!(
        "Validating: {}",
        serde_json::to_string(&wrong_type).unwrap()
    );
    match enforcer.validate(&wrong_type) {
        Ok(result) => println!("  Valid: {result}\n"),
        Err(violations) => {
            println!("  Invalid ({} violation(s)):", violations.len());
            for v in &violations {
                println!("    - {v}");
            }
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // 5. OutputRepairer — fixing malformed JSON
    // -----------------------------------------------------------------------
    println!("--- 5. OutputRepairer — Fixing Malformed JSON ---\n");

    let repairer = OutputRepairer::new();

    // 5a. Trailing commas
    let trailing_commas = r#"{"name": "Alice", "age": 30, "hobbies": ["reading", "coding",],}"#;
    println!("Input (trailing commas):");
    println!("  {trailing_commas}");
    match repairer.repair_json(trailing_commas) {
        Ok(fixed) => {
            let parsed: serde_json::Value = serde_json::from_str(&fixed).unwrap();
            println!(
                "  Fixed: {}\n",
                serde_json::to_string_pretty(&parsed).unwrap()
            );
        }
        Err(e) => println!("  Repair failed: {e}\n"),
    }

    // 5b. Single quotes instead of double quotes
    let single_quotes = "{'name': 'Bob', 'score': 42}";
    println!("Input (single quotes):");
    println!("  {single_quotes}");
    match repairer.repair_json(single_quotes) {
        Ok(fixed) => {
            let parsed: serde_json::Value = serde_json::from_str(&fixed).unwrap();
            println!(
                "  Fixed: {}\n",
                serde_json::to_string_pretty(&parsed).unwrap()
            );
        }
        Err(e) => println!("  Repair failed: {e}\n"),
    }

    // 5c. JSON inside markdown fences
    let fenced_json = r#"```json
{"tool": "rustc", "version": "1.75.0"}
```"#;
    println!("Input (markdown fences):");
    println!("  {fenced_json}");
    let stripped = repairer.strip_markdown_fences(fenced_json);
    println!("  Stripped: {stripped}");
    match repairer.repair_json(fenced_json) {
        Ok(fixed) => {
            let parsed: serde_json::Value = serde_json::from_str(&fixed).unwrap();
            println!(
                "  Parsed: {}\n",
                serde_json::to_string_pretty(&parsed).unwrap()
            );
        }
        Err(e) => println!("  Repair failed: {e}\n"),
    }

    // 5d. Extract text between delimiters
    let xml_like = "<result>The answer is 42</result>";
    println!("Input (delimited text):");
    println!("  {xml_like}");
    if let Some(extracted) = repairer.extract_between(xml_like, "<result>", "</result>") {
        println!("  Extracted: \"{extracted}\"");
    }

    // -----------------------------------------------------------------------
    // 6. End-to-end: parse, validate, and repair
    // -----------------------------------------------------------------------
    println!("\n--- 6. End-to-End Pipeline ---\n");

    let llm_response = "{'username': 'rustacean', 'level': 5, 'verified': true,}";

    println!("Raw LLM output (malformed JSON):");
    println!("  {llm_response}\n");

    // Step 1: Repair the JSON
    println!("Step 1 — Repair JSON:");
    let repaired = repairer.repair_json(llm_response).unwrap();
    println!("  Repaired: {repaired}");

    // Step 2: Parse the repaired JSON
    println!("\nStep 2 — Parse:");
    let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    println!(
        "  Parsed: {}",
        serde_json::to_string_pretty(&parsed).unwrap()
    );

    // Step 3: Validate against schema
    println!("\nStep 3 — Validate:");
    let profile_schema = SchemaEnforcer::new()
        .require_field("username")
        .require_field("level")
        .require_type("username", JsonType::String)
        .require_type("level", JsonType::Number)
        .require_type("verified", JsonType::Bool)
        .optional_field("bio", serde_json::json!(""));

    match profile_schema.validate(&parsed) {
        Ok(validated) => {
            println!("  Validated with defaults:");
            println!("  {}", serde_json::to_string_pretty(&validated).unwrap());
        }
        Err(violations) => {
            println!("  Validation failed:");
            for v in &violations {
                println!("    - {v}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Real LLM Demo — LLM-powered structured parsing
    // -----------------------------------------------------------------------
    println!("\n--- 7. Real LLM Demo ---");
    println!("Ask an LLM to generate structured JSON, then parse and validate it.\n");

    let model = shared::get_chat_model(vec![r#"```json
{"language": "Rust", "year": 2015, "paradigm": "systems", "memory_safe": true}
```"#
        .into()]);
    let messages = vec![
        Message::system("Always respond with a JSON object inside a markdown code block."),
        Message::human("Give me structured data about the Rust programming language: name, first stable release year, paradigm, and whether it is memory safe."),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let raw = gen.message.content().text();
        println!("Raw LLM output:\n{}\n", raw);

        match parser.parse_json_block(&raw) {
            Ok(parsed) => {
                println!(
                    "Parsed JSON: {}",
                    serde_json::to_string_pretty(&parsed).unwrap()
                );

                let llm_schema = SchemaEnforcer::new()
                    .require_field("language")
                    .require_type("language", JsonType::String)
                    .require_type("year", JsonType::Number);
                match llm_schema.validate(&parsed) {
                    Ok(validated) => println!(
                        "Validation: PASS\n{}",
                        serde_json::to_string_pretty(&validated).unwrap()
                    ),
                    Err(violations) => {
                        println!("Validation: FAIL");
                        for v in &violations {
                            println!("  - {v}");
                        }
                    }
                }
            }
            Err(e) => println!("Parse error: {e}"),
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
