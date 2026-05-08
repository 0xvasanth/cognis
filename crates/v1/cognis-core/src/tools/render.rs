use super::base::BaseTool;

/// Type alias for a function that renders a list of tools as a string.
pub type ToolsRenderer = fn(&[&dyn BaseTool]) -> String;

/// Render tools as a simple "name - description" list.
pub fn render_text_description(tools: &[&dyn BaseTool]) -> String {
    tools
        .iter()
        .map(|t| format!("{} - {}", t.name(), t.description()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render tools as "name - description, args: <schema>" list.
pub fn render_text_description_and_args(tools: &[&dyn BaseTool]) -> String {
    tools
        .iter()
        .map(|t| {
            let schema = t.tool_call_schema();
            format!("{} - {}, args: {}", t.name(), t.description(), schema)
        })
        .collect::<Vec<_>>()
        .join("\n")
}
