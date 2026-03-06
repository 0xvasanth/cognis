//! Type mappings for serialization.
//!
//! Mirrors Python `langchain_core.load.mapping`.

use std::collections::HashMap;

/// Get the default mapping of class paths to their modules.
/// This is used during deserialization to validate allowed types.
pub fn get_default_mappings() -> HashMap<String, String> {
    let mut m = HashMap::new();
    // Core types
    m.insert(
        "rustchain_core.prompts.base.PromptTemplate".to_string(),
        "prompts".to_string(),
    );
    m.insert(
        "rustchain_core.prompts.chat.ChatPromptTemplate".to_string(),
        "prompts".to_string(),
    );
    m.insert(
        "rustchain_core.prompts.few_shot.FewShotPromptTemplate".to_string(),
        "prompts".to_string(),
    );
    m.insert(
        "rustchain_core.output_parsers.json.JsonOutputParser".to_string(),
        "output_parsers".to_string(),
    );
    m.insert(
        "rustchain_core.output_parsers.string.StrOutputParser".to_string(),
        "output_parsers".to_string(),
    );
    m.insert(
        "rustchain_core.output_parsers.list.CommaSeparatedListOutputParser".to_string(),
        "output_parsers".to_string(),
    );
    m.insert(
        "rustchain_core.documents.Document".to_string(),
        "documents".to_string(),
    );
    m.insert(
        "rustchain_core.messages.HumanMessage".to_string(),
        "messages".to_string(),
    );
    m.insert(
        "rustchain_core.messages.AIMessage".to_string(),
        "messages".to_string(),
    );
    m.insert(
        "rustchain_core.messages.SystemMessage".to_string(),
        "messages".to_string(),
    );
    m.insert(
        "rustchain_core.messages.ToolMessage".to_string(),
        "messages".to_string(),
    );
    // LangChain compatibility
    m.insert(
        "langchain_core.prompts.prompt.PromptTemplate".to_string(),
        "prompts".to_string(),
    );
    m.insert(
        "langchain_core.prompts.chat.ChatPromptTemplate".to_string(),
        "prompts".to_string(),
    );
    m.insert(
        "langchain_core.output_parsers.json.JsonOutputParser".to_string(),
        "output_parsers".to_string(),
    );
    m.insert(
        "langchain_core.output_parsers.string.StrOutputParser".to_string(),
        "output_parsers".to_string(),
    );
    m.insert(
        "langchain_core.documents.base.Document".to_string(),
        "documents".to_string(),
    );
    m
}

/// Check if a class path is in the default allowed mappings.
pub fn is_allowed_class_path(path: &str) -> bool {
    get_default_mappings().contains_key(path)
}
