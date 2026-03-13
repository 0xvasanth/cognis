//! Graph Validation Demo
//!
//! Demonstrates the graph validation utilities from `cognisgraph::graph::validation`:
//!
//! - **GraphValidator** — structural checks (unreachable nodes, missing entry/finish,
//!   dangling edges, self-loops, duplicate edges, orphan nodes)
//! - **StateValidator** with **StateRule** — required keys, immutable keys, type constraints
//! - **SchemaValidator** — node input/output key compatibility across edges
//! - **QuickCheck** — fast boolean pre-flight checks
//! - **ValidationReport** — unified error/warning/info display
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_validation_demo`

use cognisgraph::graph::validation::{
    GraphValidator, QuickCheck, SchemaValidator, StateRule, StateValidator, ValidationReport,
};
use serde_json::json;

/// Helper to build a `Vec<String>` from a slice of `&str`.
fn nodes(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Helper to build edge tuples.
fn edges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn print_report(label: &str, report: &ValidationReport) {
    println!("  {}", label);
    println!("  {}", report.summary());
    for issue in report.issues() {
        println!("    {}", issue);
    }
    println!();
}

fn main() {
    println!("=== Graph Validation Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. GraphValidator — well-formed graph
    // -----------------------------------------------------------------------
    println!("--- 1. Validate a well-formed graph ---");
    println!("A simple pipeline: start -> process -> enrich -> end\n");

    let validator = GraphValidator::new();
    let n = nodes(&["start", "process", "enrich", "end"]);
    let e = edges(&[
        ("start", "process"),
        ("process", "enrich"),
        ("enrich", "end"),
    ]);

    let report = validator.validate(&n, &e, Some("start"), Some("end"));
    print_report("Well-formed graph:", &report);
    println!("  is_valid = {}", report.is_valid());
    println!();

    // -----------------------------------------------------------------------
    // 2. GraphValidator — malformed graph with multiple issues
    // -----------------------------------------------------------------------
    println!("--- 2. Validate a malformed graph ---");
    println!("Issues: no entry, dangling edge target, self-loop, unreachable node, orphan node\n");

    let n2 = nodes(&["a", "b", "c", "orphan"]);
    let e2 = edges(&[
        ("a", "b"),
        ("b", "ghost"), // dangling edge — "ghost" is not a node
        ("b", "b"),     // self-loop
                        // "c" is unreachable (no edge leading to it)
                        // "orphan" has no edges at all
    ]);

    // No entry, no finish
    let report2 = validator.validate(&n2, &e2, None, None);
    print_report("Malformed graph:", &report2);
    println!("  is_valid = {}", report2.is_valid());
    println!("  error count   = {}", report2.errors().len());
    println!("  warning count = {}", report2.warnings().len());
    println!();

    // -----------------------------------------------------------------------
    // 3. QuickCheck — fast pre-flight checks
    // -----------------------------------------------------------------------
    println!("--- 3. QuickCheck — fast pre-flight checks ---\n");

    let n3 = nodes(&["entry", "middle", "finish"]);
    let e3 = edges(&[("entry", "middle"), ("middle", "finish")]);

    println!(
        "  has_entry(\"entry\")  = {}",
        QuickCheck::has_entry(&n3, "entry")
    );
    println!(
        "  has_entry(\"nope\")   = {}",
        QuickCheck::has_entry(&n3, "nope")
    );
    println!(
        "  has_finish(\"finish\") = {}",
        QuickCheck::has_finish(&n3, "finish")
    );
    println!(
        "  is_connected        = {}",
        QuickCheck::is_connected(&n3, &e3)
    );
    println!(
        "  has_cycles          = {}",
        QuickCheck::has_cycles(&n3, &e3)
    );

    // Add a cycle
    let e3_cycle = edges(&[
        ("entry", "middle"),
        ("middle", "finish"),
        ("finish", "entry"),
    ]);
    println!(
        "  has_cycles (with cycle) = {}",
        QuickCheck::has_cycles(&n3, &e3_cycle)
    );
    println!();

    // -----------------------------------------------------------------------
    // 4. StateValidator with StateRules
    // -----------------------------------------------------------------------
    println!("--- 4. StateValidator — state transition rules ---\n");

    let mut state_validator = StateValidator::new();
    state_validator.add_rule(StateRule::required_key("user_id"));
    state_validator.add_rule(StateRule::immutable_key("session_id"));
    state_validator.add_rule(StateRule::type_constraint("score", "number"));

    println!("  Registered {} rule(s)\n", state_validator.rule_count());

    // Valid transition
    let from = json!({"user_id": "u1", "session_id": "s1", "score": 42});
    let to = json!({"user_id": "u1", "session_id": "s1", "score": 95});
    let report_valid = state_validator.validate_transition(&from, &to);
    print_report("Valid transition:", &report_valid);

    // Invalid transition — missing required key, immutable key changed, wrong type
    let from_bad = json!({"user_id": "u1", "session_id": "s1", "score": 10});
    let to_bad = json!({"session_id": "s2", "score": "not-a-number"});
    let report_bad = state_validator.validate_transition(&from_bad, &to_bad);
    print_report("Invalid transition:", &report_bad);

    // -----------------------------------------------------------------------
    // 5. SchemaValidator — node input/output compatibility
    // -----------------------------------------------------------------------
    println!("--- 5. SchemaValidator — node schema compatibility ---\n");

    let mut schema_validator = SchemaValidator::new();

    // Register schemas: each node declares what keys it consumes and produces
    schema_validator.register_node_schema(
        "fetch",
        vec![],                                     // no inputs (entry)
        vec!["raw_text".into(), "metadata".into()], // outputs
    );
    schema_validator.register_node_schema(
        "parse",
        vec!["raw_text".into()],                       // needs raw_text
        vec!["parsed_data".into(), "metadata".into()], // outputs
    );
    schema_validator.register_node_schema(
        "summarize",
        vec!["parsed_data".into(), "context".into()], // needs parsed_data AND context
        vec!["summary".into()],                       // outputs
    );

    println!(
        "  Registered {} node schema(s)\n",
        schema_validator.node_count()
    );

    let schema_edges = edges(&[("fetch", "parse"), ("parse", "summarize")]);
    let schema_report = schema_validator.validate_connections(&schema_edges);
    print_report("Schema validation:", &schema_report);
    // "parse" does not produce "context", which "summarize" requires — expect an error

    // -----------------------------------------------------------------------
    // 6. ValidationReport — JSON output and summary
    // -----------------------------------------------------------------------
    println!("--- 6. ValidationReport — JSON summary ---\n");

    let json_output = report2.to_json();
    println!(
        "  JSON report (malformed graph):\n  {}",
        serde_json::to_string_pretty(&json_output).unwrap()
    );
    println!();

    println!("=== Graph Validation Demo Complete ===");
}
