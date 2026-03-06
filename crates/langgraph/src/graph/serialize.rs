//! Graph definition serialization and persistence.
//!
//! This module provides [`GraphDefinition`], a serializable representation of a
//! graph's topology (nodes, edges, routing) that can be persisted to JSON files,
//! validated, and used to generate Mermaid diagrams without a compiled graph.
//!
//! It also provides [`GraphRegistry`] for storing and retrieving named graph
//! definitions.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::constants::{END, START};
use crate::errors::{LangGraphError, Result};

/// A serializable representation of a conditional edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalEdgeDef {
    /// The source node name.
    pub from: String,
    /// The possible target node names.
    pub targets: Vec<String>,
    /// Optional mapping of route labels to target node names.
    pub labels: Option<HashMap<String, String>>,
}

/// A serializable representation of a graph's topology.
///
/// This captures the structure (nodes, edges, routing) but NOT the node actions
/// (which are closures and can't be serialized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDefinition {
    /// Node names in the graph.
    pub nodes: Vec<String>,
    /// Direct edges: (from, to) pairs.
    pub edges: Vec<(String, String)>,
    /// Conditional edges: (from, possible_targets) pairs.
    pub conditional_edges: Vec<ConditionalEdgeDef>,
    /// The entry point node name.
    pub entry_point: String,
    /// Nodes that trigger interrupt before execution.
    pub interrupt_before: Vec<String>,
    /// Nodes that trigger interrupt after execution.
    pub interrupt_after: Vec<String>,
}

impl GraphDefinition {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| LangGraphError::Other(format!("Failed to serialize graph definition: {e}")))
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| LangGraphError::Other(format!("Failed to deserialize graph definition: {e}")))
    }

    /// Save the definition to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| LangGraphError::Other(format!("Failed to write graph definition to file: {e}")))
    }

    /// Load a definition from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| LangGraphError::Other(format!("Failed to read graph definition from file: {e}")))?;
        Self::from_json(&json)
    }

    /// Validate the definition for consistency.
    ///
    /// Returns a list of warning messages. If the definition has fatal
    /// inconsistencies, returns an error instead.
    ///
    /// Checks performed:
    /// - Entry point exists in the node list.
    /// - All edge sources and targets reference valid nodes (or START/END).
    /// - No orphan nodes (nodes with no incoming or outgoing edges).
    /// - Interrupt nodes exist in the node list.
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings: Vec<String> = Vec::new();
        let valid_names: std::collections::HashSet<&str> = self
            .nodes
            .iter()
            .map(|s| s.as_str())
            .chain([START, END])
            .collect();

        // Check entry point exists.
        if !self.nodes.contains(&self.entry_point) {
            return Err(LangGraphError::Other(format!(
                "Entry point '{}' is not in the node list",
                self.entry_point
            )));
        }

        // Check direct edges reference valid nodes.
        for (from, to) in &self.edges {
            if !valid_names.contains(from.as_str()) {
                return Err(LangGraphError::Other(format!(
                    "Edge source '{}' is not a valid node",
                    from
                )));
            }
            if !valid_names.contains(to.as_str()) {
                return Err(LangGraphError::Other(format!(
                    "Edge target '{}' is not a valid node",
                    to
                )));
            }
        }

        // Check conditional edges reference valid nodes.
        for cond in &self.conditional_edges {
            if !valid_names.contains(cond.from.as_str()) {
                return Err(LangGraphError::Other(format!(
                    "Conditional edge source '{}' is not a valid node",
                    cond.from
                )));
            }
            for target in &cond.targets {
                if !valid_names.contains(target.as_str()) {
                    return Err(LangGraphError::Other(format!(
                        "Conditional edge target '{}' is not a valid node",
                        target
                    )));
                }
            }
        }

        // Check interrupt nodes exist.
        for name in &self.interrupt_before {
            if !self.nodes.contains(name) {
                return Err(LangGraphError::Other(format!(
                    "Interrupt-before node '{}' is not in the node list",
                    name
                )));
            }
        }
        for name in &self.interrupt_after {
            if !self.nodes.contains(name) {
                return Err(LangGraphError::Other(format!(
                    "Interrupt-after node '{}' is not in the node list",
                    name
                )));
            }
        }

        // Warn about orphan nodes (no incoming and no outgoing edges).
        let mut has_incoming: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_outgoing: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for (from, to) in &self.edges {
            has_outgoing.insert(from.as_str());
            has_incoming.insert(to.as_str());
        }
        for cond in &self.conditional_edges {
            has_outgoing.insert(cond.from.as_str());
            for target in &cond.targets {
                has_incoming.insert(target.as_str());
            }
        }

        for node in &self.nodes {
            if !has_incoming.contains(node.as_str()) && !has_outgoing.contains(node.as_str()) {
                warnings.push(format!("Node '{}' is orphaned (no edges)", node));
            }
        }

        Ok(warnings)
    }

    /// Generate a Mermaid diagram from this definition.
    ///
    /// This produces the same style of diagram as
    /// [`CompiledStateGraph::draw_mermaid`](super::state::CompiledStateGraph::draw_mermaid)
    /// but works from a serialized definition without needing a compiled graph.
    pub fn to_mermaid(&self) -> String {
        let mut lines: Vec<String> = vec!["graph TD".to_string()];

        // Emit node declarations. Always include __start__ and __end__.
        lines.push(format!("    {}([{}])", START, START));
        let mut sorted_nodes: Vec<&String> = self.nodes.iter().collect();
        sorted_nodes.sort();
        for name in &sorted_nodes {
            lines.push(format!("    {}([{}])", name, name));
        }
        lines.push(format!("    {}([{}])", END, END));

        // Emit direct edges.
        for (from, to) in &self.edges {
            lines.push(format!("    {} --> {}", from, to));
        }

        // Emit conditional edges.
        for cond in &self.conditional_edges {
            if let Some(ref labels) = cond.labels {
                let mut sorted_labels: Vec<(&String, &String)> = labels.iter().collect();
                sorted_labels.sort_by_key(|(k, _)| (*k).clone());
                for (label, target) in sorted_labels {
                    lines.push(format!("    {} -. {} .-> {}", cond.from, label, target));
                }
            } else {
                for target in &cond.targets {
                    lines.push(format!("    {} -. condition .-> {}", cond.from, target));
                }
            }
        }

        // Style interrupt nodes.
        let mut interrupt_nodes: Vec<&String> = self
            .interrupt_before
            .iter()
            .chain(self.interrupt_after.iter())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        interrupt_nodes.sort();
        for name in interrupt_nodes {
            lines.push(format!("    style {} fill:#f9f,stroke:#333", name));
        }

        lines.join("\n")
    }
}

/// A registry for storing and retrieving named graph definitions.
#[derive(Debug, Default)]
pub struct GraphRegistry {
    definitions: HashMap<String, GraphDefinition>,
}

impl GraphRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
        }
    }

    /// Register a graph definition with the given name.
    pub fn register(&mut self, name: &str, def: GraphDefinition) {
        self.definitions.insert(name.to_string(), def);
    }

    /// Get a graph definition by name.
    pub fn get(&self, name: &str) -> Option<&GraphDefinition> {
        self.definitions.get(name)
    }

    /// List all registered graph definition names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.definitions.keys().cloned().collect();
        names.sort();
        names
    }

    /// Save all definitions to individual JSON files in the given directory.
    ///
    /// Each definition is saved as `<name>.json`.
    pub fn save_all(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)
            .map_err(|e| LangGraphError::Other(format!("Failed to create directory: {e}")))?;
        for (name, def) in &self.definitions {
            let file_path = dir.join(format!("{}.json", name));
            def.save_to_file(&file_path)?;
        }
        Ok(())
    }

    /// Load all `.json` files from the given directory as graph definitions.
    ///
    /// The file stem (name without extension) is used as the definition name.
    pub fn load_all(dir: &Path) -> Result<Self> {
        let mut registry = Self::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| LangGraphError::Other(format!("Failed to read directory: {e}")))?;
        for entry in entries {
            let entry = entry
                .map_err(|e| LangGraphError::Other(format!("Failed to read directory entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let def = GraphDefinition::load_from_file(&path)?;
                registry.register(&name, def);
            }
        }
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_definition() -> GraphDefinition {
        GraphDefinition {
            nodes: vec![
                "agent".to_string(),
                "tools".to_string(),
            ],
            edges: vec![
                (START.to_string(), "agent".to_string()),
                ("tools".to_string(), "agent".to_string()),
            ],
            conditional_edges: vec![ConditionalEdgeDef {
                from: "agent".to_string(),
                targets: vec!["tools".to_string(), END.to_string()],
                labels: Some(HashMap::from([
                    ("call_tool".to_string(), "tools".to_string()),
                    ("done".to_string(), END.to_string()),
                ])),
            }],
            entry_point: "agent".to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        }
    }

    #[test]
    fn test_serialize_to_definition_captures_nodes_and_edges() {
        let def = sample_definition();
        assert_eq!(def.nodes, vec!["agent", "tools"]);
        assert_eq!(def.edges.len(), 2);
        assert_eq!(def.edges[0], (START.to_string(), "agent".to_string()));
        assert_eq!(def.conditional_edges.len(), 1);
        assert_eq!(def.conditional_edges[0].from, "agent");
        assert_eq!(def.entry_point, "agent");
    }

    #[test]
    fn test_serialize_json_round_trip() {
        let def = sample_definition();
        let json = def.to_json().unwrap();
        let restored = GraphDefinition::from_json(&json).unwrap();

        assert_eq!(def.nodes, restored.nodes);
        assert_eq!(def.edges, restored.edges);
        assert_eq!(def.entry_point, restored.entry_point);
        assert_eq!(def.conditional_edges.len(), restored.conditional_edges.len());
        assert_eq!(
            def.conditional_edges[0].from,
            restored.conditional_edges[0].from
        );
        assert_eq!(
            def.conditional_edges[0].targets,
            restored.conditional_edges[0].targets
        );
    }

    #[test]
    fn test_serialize_save_load_file() {
        let dir = std::env::temp_dir().join("langgraph_serialize_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let def = sample_definition();
        let file_path = dir.join("test_graph.json");
        def.save_to_file(&file_path).unwrap();

        let loaded = GraphDefinition::load_from_file(&file_path).unwrap();
        assert_eq!(def.nodes, loaded.nodes);
        assert_eq!(def.edges, loaded.edges);
        assert_eq!(def.entry_point, loaded.entry_point);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_serialize_validate_catches_invalid_entry_point() {
        let def = GraphDefinition {
            nodes: vec!["a".to_string()],
            edges: vec![(START.to_string(), "a".to_string())],
            conditional_edges: vec![],
            entry_point: "nonexistent".to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };
        let result = def.validate();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("nonexistent"));
    }

    #[test]
    fn test_serialize_validate_catches_invalid_edge_target() {
        let def = GraphDefinition {
            nodes: vec!["a".to_string()],
            edges: vec![
                (START.to_string(), "a".to_string()),
                ("a".to_string(), "ghost".to_string()),
            ],
            conditional_edges: vec![],
            entry_point: "a".to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };
        let result = def.validate();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("ghost"));
    }

    #[test]
    fn test_serialize_validate_warns_orphan_nodes() {
        let def = GraphDefinition {
            nodes: vec!["a".to_string(), "orphan".to_string()],
            edges: vec![
                (START.to_string(), "a".to_string()),
                ("a".to_string(), END.to_string()),
            ],
            conditional_edges: vec![],
            entry_point: "a".to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };
        let warnings = def.validate().unwrap();
        assert!(warnings.iter().any(|w| w.contains("orphan")));
    }

    #[test]
    fn test_serialize_to_mermaid_from_definition() {
        let def = sample_definition();
        let mermaid = def.to_mermaid();

        assert!(mermaid.starts_with("graph TD"));
        assert!(mermaid.contains("agent"));
        assert!(mermaid.contains("tools"));
        assert!(mermaid.contains(START));
        assert!(mermaid.contains(END));
        // Should contain the conditional edge labels.
        assert!(mermaid.contains("call_tool"));
        assert!(mermaid.contains("done"));
    }

    #[test]
    fn test_serialize_graph_registry() {
        let mut registry = GraphRegistry::new();
        assert!(registry.list().is_empty());

        let def1 = sample_definition();
        let def2 = GraphDefinition {
            nodes: vec!["single".to_string()],
            edges: vec![
                (START.to_string(), "single".to_string()),
                ("single".to_string(), END.to_string()),
            ],
            conditional_edges: vec![],
            entry_point: "single".to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };

        registry.register("react_agent", def1);
        registry.register("simple", def2);

        assert_eq!(registry.list(), vec!["react_agent", "simple"]);
        assert!(registry.get("react_agent").is_some());
        assert!(registry.get("simple").is_some());
        assert!(registry.get("unknown").is_none());

        // Test save_all and load_all round-trip.
        let dir = std::env::temp_dir().join("langgraph_registry_test");
        let _ = std::fs::remove_dir_all(&dir);
        registry.save_all(&dir).unwrap();

        let loaded = GraphRegistry::load_all(&dir).unwrap();
        assert_eq!(loaded.list(), vec!["react_agent", "simple"]);
        assert_eq!(
            loaded.get("react_agent").unwrap().nodes,
            vec!["agent", "tools"]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
