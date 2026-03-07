use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message, SystemMessage};

/// Output format for the extraction result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON output format.
    Json,
    /// YAML output format.
    Yaml,
    /// Markdown table output format.
    Markdown,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Json
    }
}

/// Type of a schema field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// A string value.
    String,
    /// An integer value.
    Integer,
    /// A floating-point value.
    Float,
    /// A boolean value.
    Boolean,
    /// An array of values.
    Array,
    /// A nested object.
    Object,
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldType::String => write!(f, "string"),
            FieldType::Integer => write!(f, "integer"),
            FieldType::Float => write!(f, "float"),
            FieldType::Boolean => write!(f, "boolean"),
            FieldType::Array => write!(f, "array"),
            FieldType::Object => write!(f, "object"),
        }
    }
}

/// A single field in an extraction schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    /// Name of the field.
    pub name: String,
    /// Type of the field.
    pub field_type: FieldType,
    /// Description of what this field represents.
    pub description: String,
    /// Whether this field is required.
    pub required: bool,
    /// Allowed values for this field (enum constraint).
    pub enum_values: Option<Vec<String>>,
    /// Default value if the field is not found.
    pub default: Option<Value>,
}

/// Builder for [`SchemaField`].
pub struct SchemaFieldBuilder {
    name: String,
    field_type: FieldType,
    description: String,
    required: bool,
    enum_values: Option<Vec<String>>,
    default: Option<Value>,
}

impl SchemaFieldBuilder {
    /// Create a new field builder.
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            description: String::new(),
            required: false,
            enum_values: None,
            default: None,
        }
    }

    /// Set the field description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Mark the field as required.
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set allowed enum values.
    pub fn enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }

    /// Set a default value.
    pub fn default_value(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Build the [`SchemaField`].
    pub fn build(self) -> SchemaField {
        SchemaField {
            name: self.name,
            field_type: self.field_type,
            description: self.description,
            required: self.required,
            enum_values: self.enum_values,
            default: self.default,
        }
    }
}

/// A few-shot example for extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionExample {
    /// The input text for this example.
    pub input: String,
    /// The expected extraction output.
    pub output: Value,
}

impl ExtractionExample {
    /// Create a new extraction example.
    pub fn new(input: impl Into<String>, output: Value) -> Self {
        Self {
            input: input.into(),
            output,
        }
    }
}

/// Schema describing what entities and fields to extract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    /// Entity name (e.g., "Person", "Event").
    pub name: String,
    /// Description of the entity to extract.
    pub description: String,
    /// Fields to extract.
    pub fields: Vec<SchemaField>,
}

impl ExtractionSchema {
    /// Create a new builder for an extraction schema.
    pub fn builder() -> ExtractionSchemaBuilder {
        ExtractionSchemaBuilder::new()
    }

    /// Generate extraction instructions for the LLM prompt.
    pub fn to_prompt_instruction(&self) -> String {
        let mut instruction = format!(
            "Extract {} entities from the text.\n\nEntity: {}\nDescription: {}\n\nFields:\n",
            self.name, self.name, self.description
        );

        let required_fields: Vec<&SchemaField> =
            self.fields.iter().filter(|f| f.required).collect();
        let optional_fields: Vec<&SchemaField> =
            self.fields.iter().filter(|f| !f.required).collect();

        for field in &self.fields {
            instruction.push_str(&format!(
                "- {} ({}): {}",
                field.name, field.field_type, field.description
            ));
            if field.required {
                instruction.push_str(" [REQUIRED]");
            } else {
                instruction.push_str(" [OPTIONAL]");
            }
            if let Some(ref enum_vals) = field.enum_values {
                instruction.push_str(&format!(" Allowed values: {}", enum_vals.join(", ")));
            }
            if let Some(ref default) = field.default {
                instruction.push_str(&format!(" Default: {}", default));
            }
            instruction.push('\n');
        }

        if !required_fields.is_empty() {
            instruction.push_str(&format!(
                "\nRequired fields: {}\n",
                required_fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if !optional_fields.is_empty() {
            instruction.push_str(&format!(
                "Optional fields: {}\n",
                optional_fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        instruction
    }
}

/// Builder for [`ExtractionSchema`].
pub struct ExtractionSchemaBuilder {
    name: Option<String>,
    description: Option<String>,
    fields: Vec<SchemaField>,
}

impl ExtractionSchemaBuilder {
    /// Create a new schema builder.
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            fields: Vec::new(),
        }
    }

    /// Set the entity name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the entity description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a field to the schema.
    pub fn field(mut self, field: SchemaField) -> Self {
        self.fields.push(field);
        self
    }

    /// Add a required field with basic parameters.
    pub fn required_field(
        mut self,
        name: impl Into<String>,
        field_type: FieldType,
        description: impl Into<String>,
    ) -> Self {
        self.fields.push(SchemaField {
            name: name.into(),
            field_type,
            description: description.into(),
            required: true,
            enum_values: None,
            default: None,
        });
        self
    }

    /// Build the [`ExtractionSchema`].
    ///
    /// # Panics
    ///
    /// Panics if `name` or `description` have not been set.
    pub fn build(self) -> ExtractionSchema {
        ExtractionSchema {
            name: self.name.expect("name is required for ExtractionSchema"),
            description: self
                .description
                .expect("description is required for ExtractionSchema"),
            fields: self.fields,
        }
    }
}

impl Default for ExtractionSchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an extraction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities as JSON values.
    pub entities: Vec<Value>,
    /// The raw LLM response text.
    pub raw_response: String,
    /// Optional confidence score.
    pub confidence: Option<f64>,
    /// Additional metadata.
    pub metadata: HashMap<String, Value>,
}

/// A chain that extracts structured data from text using an LLM with schema-guided prompts.
///
/// The chain builds a prompt from the extraction schema, optional few-shot examples,
/// and the input text, then sends it to the LLM and parses the response into
/// structured entities.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use rustchain::chains::extraction::{ExtractionChain, ExtractionSchema, FieldType};
/// use rustchain_core::language_models::fake::FakeListChatModel;
///
/// # async fn example() {
/// let model = Arc::new(FakeListChatModel::new(vec![
///     r#"[{"name": "Alice", "age": 30}]"#.to_string(),
/// ]));
///
/// let schema = ExtractionSchema::builder()
///     .name("Person")
///     .description("A person entity")
///     .required_field("name", FieldType::String, "Person's name")
///     .required_field("age", FieldType::Integer, "Person's age")
///     .build();
///
/// let chain = ExtractionChain::builder()
///     .llm(model)
///     .schema(schema)
///     .build();
///
/// let result = chain.extract("Alice is 30 years old").await.unwrap();
/// assert_eq!(result.entities.len(), 1);
/// # }
/// ```
pub struct ExtractionChain {
    /// The LLM to use for extraction.
    llm: Arc<dyn BaseChatModel>,
    /// The schema describing what to extract.
    schema: ExtractionSchema,
    /// Optional custom system prompt.
    system_prompt: Option<String>,
    /// Few-shot examples for the LLM.
    examples: Vec<ExtractionExample>,
    /// Output format for the extraction.
    output_format: OutputFormat,
}

/// Builder for [`ExtractionChain`].
pub struct ExtractionChainBuilder {
    llm: Option<Arc<dyn BaseChatModel>>,
    schema: Option<ExtractionSchema>,
    system_prompt: Option<String>,
    examples: Vec<ExtractionExample>,
    output_format: OutputFormat,
}

impl ExtractionChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            llm: None,
            schema: None,
            system_prompt: None,
            examples: Vec::new(),
            output_format: OutputFormat::Json,
        }
    }

    /// Set the LLM (required).
    pub fn llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the extraction schema (required).
    pub fn schema(mut self, schema: ExtractionSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set a custom system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a few-shot example.
    pub fn add_example(mut self, example: ExtractionExample) -> Self {
        self.examples.push(example);
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Build the [`ExtractionChain`].
    ///
    /// # Panics
    ///
    /// Panics if `llm` or `schema` have not been set.
    pub fn build(self) -> ExtractionChain {
        ExtractionChain {
            llm: self.llm.expect("llm is required for ExtractionChain"),
            schema: self.schema.expect("schema is required for ExtractionChain"),
            system_prompt: self.system_prompt,
            examples: self.examples,
            output_format: self.output_format,
        }
    }
}

impl Default for ExtractionChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractionChain {
    /// Create a new builder.
    pub fn builder() -> ExtractionChainBuilder {
        ExtractionChainBuilder::new()
    }

    /// Get a reference to the schema.
    pub fn schema(&self) -> &ExtractionSchema {
        &self.schema
    }

    /// Get the output format.
    pub fn output_format(&self) -> OutputFormat {
        self.output_format
    }

    /// Build the complete prompt messages for the LLM.
    fn build_messages(&self, text: &str) -> Vec<Message> {
        let mut messages = Vec::new();

        // System message
        let system_text = if let Some(ref custom) = self.system_prompt {
            custom.clone()
        } else {
            "You are an expert extraction algorithm. Your task is to extract structured \
             information from text according to a given schema. Only extract information \
             that is explicitly stated in the text. If a field cannot be determined, omit it \
             or set it to null."
                .to_string()
        };
        messages.push(Message::System(SystemMessage::new(&system_text)));

        // Schema instructions
        let schema_instruction = self.schema.to_prompt_instruction();
        let format_instruction = match self.output_format {
            OutputFormat::Json => {
                "Return the extracted entities as a JSON array. Each element should be an object \
                 with the fields described above. If no entities are found, return an empty array []."
                    .to_string()
            }
            OutputFormat::Yaml => {
                "Return the extracted entities in YAML format as a list. Each item should contain \
                 the fields described above. If no entities are found, return an empty list."
                    .to_string()
            }
            OutputFormat::Markdown => {
                "Return the extracted entities as a Markdown table with columns matching the field \
                 names described above. If no entities are found, return an empty table."
                    .to_string()
            }
        };

        let instruction_msg = format!("{}\n\n{}", schema_instruction, format_instruction);
        messages.push(Message::Human(HumanMessage::new(&instruction_msg)));
        messages.push(Message::Ai(rustchain_core::messages::AIMessage::new(
            "Understood. I will extract entities according to the schema. Please provide the text.",
        )));

        // Few-shot examples
        for example in &self.examples {
            messages.push(Message::Human(HumanMessage::new(&format!(
                "Extract from this text:\n{}",
                example.input
            ))));
            let example_output = match self.output_format {
                OutputFormat::Json => serde_json::to_string_pretty(&example.output)
                    .unwrap_or_else(|_| example.output.to_string()),
                _ => example.output.to_string(),
            };
            messages.push(Message::Ai(rustchain_core::messages::AIMessage::new(
                &example_output,
            )));
        }

        // Actual input text
        messages.push(Message::Human(HumanMessage::new(&format!(
            "Extract from this text:\n{}",
            text
        ))));

        messages
    }

    /// Parse the LLM response into entities.
    fn parse_response(&self, raw: &str) -> Result<Vec<Value>> {
        let trimmed = raw.trim();

        // Strip markdown code fences if present
        let cleaned = if trimmed.starts_with("```") {
            let without_prefix = if let Some(rest) = trimmed.strip_prefix("```json") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix("```yaml") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix("```") {
                rest
            } else {
                trimmed
            };
            without_prefix
                .strip_suffix("```")
                .unwrap_or(without_prefix)
                .trim()
        } else {
            trimmed
        };

        match self.output_format {
            OutputFormat::Json => {
                let parsed: Value = serde_json::from_str(cleaned).map_err(|e| {
                    RustChainError::OutputParserError {
                        message: format!("Failed to parse extraction JSON: {}", e),
                        observation: Some(raw.to_string()),
                        llm_output: None,
                    }
                })?;
                match parsed {
                    Value::Array(arr) => Ok(arr),
                    Value::Object(_) => Ok(vec![parsed]),
                    _ => Err(RustChainError::OutputParserError {
                        message: format!("Expected JSON array or object, got: {}", raw),
                        observation: None,
                        llm_output: None,
                    }),
                }
            }
            OutputFormat::Yaml | OutputFormat::Markdown => {
                // For non-JSON formats, return the raw text as a single entity
                Ok(vec![json!({ "raw": cleaned })])
            }
        }
    }

    /// Extract structured data from the given text.
    pub async fn extract(&self, text: &str) -> Result<ExtractionResult> {
        let messages = self.build_messages(text);
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let raw_response = ai_msg.base.content.text();
        let entities = self.parse_response(&raw_response)?;

        Ok(ExtractionResult {
            entities,
            raw_response,
            confidence: None,
            metadata: HashMap::new(),
        })
    }

    /// Extract structured data from multiple texts.
    pub async fn extract_batch(&self, texts: &[String]) -> Result<Vec<ExtractionResult>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.extract(text).await?);
        }
        Ok(results)
    }

    /// Extract structured data from a document.
    pub async fn extract_from_document(&self, doc: &Document) -> Result<ExtractionResult> {
        let mut result = self.extract(&doc.page_content).await?;
        // Carry over document metadata
        if let Some(ref id) = doc.id {
            result.metadata.insert("document_id".to_string(), json!(id));
        }
        for (key, value) in &doc.metadata {
            result
                .metadata
                .insert(format!("doc_{}", key), value.clone());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::FakeListChatModel;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn person_schema() -> ExtractionSchema {
        ExtractionSchema::builder()
            .name("Person")
            .description("A person entity with name and age")
            .required_field("name", FieldType::String, "The person's full name")
            .required_field("age", FieldType::Integer, "The person's age in years")
            .build()
    }

    // 1. Schema construction with builder
    #[test]
    fn test_schema_builder() {
        let schema = ExtractionSchema::builder()
            .name("Person")
            .description("A person entity")
            .required_field("name", FieldType::String, "Name")
            .required_field("age", FieldType::Integer, "Age")
            .build();

        assert_eq!(schema.name, "Person");
        assert_eq!(schema.description, "A person entity");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.fields[0].required);
        assert!(schema.fields[1].required);
    }

    // 2. Field types
    #[test]
    fn test_field_types() {
        let field_str = SchemaFieldBuilder::new("name", FieldType::String)
            .description("A name")
            .build();
        assert_eq!(field_str.field_type, FieldType::String);
        assert_eq!(field_str.field_type.to_string(), "string");

        let field_int = SchemaFieldBuilder::new("count", FieldType::Integer).build();
        assert_eq!(field_int.field_type, FieldType::Integer);
        assert_eq!(field_int.field_type.to_string(), "integer");

        let field_float = SchemaFieldBuilder::new("score", FieldType::Float).build();
        assert_eq!(field_float.field_type, FieldType::Float);

        let field_bool = SchemaFieldBuilder::new("active", FieldType::Boolean).build();
        assert_eq!(field_bool.field_type, FieldType::Boolean);

        let field_arr = SchemaFieldBuilder::new("tags", FieldType::Array).build();
        assert_eq!(field_arr.field_type, FieldType::Array);

        let field_obj = SchemaFieldBuilder::new("address", FieldType::Object).build();
        assert_eq!(field_obj.field_type, FieldType::Object);
    }

    // 3. Prompt instruction generation
    #[test]
    fn test_prompt_instruction_generation() {
        let schema = person_schema();
        let instruction = schema.to_prompt_instruction();

        assert!(instruction.contains("Person"));
        assert!(instruction.contains("name"));
        assert!(instruction.contains("age"));
        assert!(instruction.contains("[REQUIRED]"));
        assert!(instruction.contains("Required fields:"));
        assert!(instruction.contains("string"));
        assert!(instruction.contains("integer"));
    }

    // 4. Extraction with mock LLM
    #[tokio::test]
    async fn test_extraction_with_mock_llm() {
        let response = r#"[{"name": "Alice", "age": 30}]"#;
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![response]))
            .schema(person_schema())
            .build();

        let result = chain.extract("Alice is 30 years old").await.unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Alice");
        assert_eq!(result.entities[0]["age"], 30);
    }

    // 5. Batch extraction
    #[tokio::test]
    async fn test_batch_extraction() {
        let model = fake_model(vec![
            r#"[{"name": "Alice", "age": 30}]"#,
            r#"[{"name": "Bob", "age": 25}]"#,
        ]);
        let chain = ExtractionChain::builder()
            .llm(model)
            .schema(person_schema())
            .build();

        let texts = vec![
            "Alice is 30 years old".to_string(),
            "Bob is 25 years old".to_string(),
        ];
        let results = chain.extract_batch(&texts).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entities[0]["name"], "Alice");
        assert_eq!(results[1].entities[0]["name"], "Bob");
    }

    // 6. Few-shot examples in prompt
    #[tokio::test]
    async fn test_few_shot_examples_in_prompt() {
        let example =
            ExtractionExample::new("John is 40 years old", json!([{"name": "John", "age": 40}]));

        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![r#"[{"name": "Jane", "age": 35}]"#]))
            .schema(person_schema())
            .add_example(example)
            .build();

        // Verify the messages include the example
        let messages = chain.build_messages("Jane is 35");
        let message_texts: Vec<String> = messages.iter().map(|m| m.content().text()).collect();

        // Should contain the example input and output
        assert!(message_texts.iter().any(|t| t.contains("John is 40")));
        assert!(message_texts.iter().any(|t| t.contains("John")));

        let result = chain.extract("Jane is 35").await.unwrap();
        assert_eq!(result.entities[0]["name"], "Jane");
    }

    // 7. Custom system prompt
    #[tokio::test]
    async fn test_custom_system_prompt() {
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![r#"[{"name": "Test", "age": 1}]"#]))
            .schema(person_schema())
            .system_prompt("You are a specialized person extractor.")
            .build();

        let messages = chain.build_messages("Test is 1");
        let system_msg = &messages[0];
        assert_eq!(
            system_msg.content().text(),
            "You are a specialized person extractor."
        );

        let result = chain.extract("Test is 1").await.unwrap();
        assert_eq!(result.entities.len(), 1);
    }

    // 8. JSON output format
    #[tokio::test]
    async fn test_json_output_format() {
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![r#"[{"name": "Alice", "age": 30}]"#]))
            .schema(person_schema())
            .output_format(OutputFormat::Json)
            .build();

        assert_eq!(chain.output_format(), OutputFormat::Json);
        let result = chain.extract("Alice is 30").await.unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Alice");
    }

    // 9. Required vs optional fields
    #[test]
    fn test_required_vs_optional_fields() {
        let schema = ExtractionSchema::builder()
            .name("Person")
            .description("A person")
            .required_field("name", FieldType::String, "Name")
            .field(
                SchemaFieldBuilder::new("nickname", FieldType::String)
                    .description("Nickname")
                    .required(false)
                    .build(),
            )
            .build();

        let instruction = schema.to_prompt_instruction();
        assert!(instruction.contains("[REQUIRED]"));
        assert!(instruction.contains("[OPTIONAL]"));
        assert!(instruction.contains("Required fields: name"));
        assert!(instruction.contains("Optional fields: nickname"));
    }

    // 10. Enum field values
    #[test]
    fn test_enum_field_values() {
        let field = SchemaFieldBuilder::new("status", FieldType::String)
            .description("Employment status")
            .enum_values(vec![
                "employed".to_string(),
                "unemployed".to_string(),
                "student".to_string(),
            ])
            .required(true)
            .build();

        assert_eq!(
            field.enum_values,
            Some(vec![
                "employed".to_string(),
                "unemployed".to_string(),
                "student".to_string()
            ])
        );

        let schema = ExtractionSchema::builder()
            .name("Person")
            .description("A person")
            .field(field)
            .build();

        let instruction = schema.to_prompt_instruction();
        assert!(instruction.contains("Allowed values: employed, unemployed, student"));
    }

    // 11. ExtractionResult structure
    #[test]
    fn test_extraction_result_structure() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!("test"));

        let result = ExtractionResult {
            entities: vec![json!({"name": "Alice"})],
            raw_response: r#"[{"name": "Alice"}]"#.to_string(),
            confidence: Some(0.95),
            metadata,
        };

        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Alice");
        assert_eq!(result.raw_response, r#"[{"name": "Alice"}]"#);
        assert_eq!(result.confidence, Some(0.95));
        assert_eq!(result.metadata["source"], "test");
    }

    // 12. Empty text extraction
    #[tokio::test]
    async fn test_empty_text_extraction() {
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec!["[]"]))
            .schema(person_schema())
            .build();

        let result = chain.extract("").await.unwrap();
        assert!(result.entities.is_empty());
    }

    // 13. Document extraction
    #[tokio::test]
    async fn test_document_extraction() {
        let mut doc_metadata = HashMap::new();
        doc_metadata.insert("source".to_string(), json!("test_file.txt"));

        let doc = Document {
            page_content: "Alice is 30 years old".to_string(),
            id: Some("doc-123".to_string()),
            metadata: doc_metadata,
            doc_type: None,
        };

        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![r#"[{"name": "Alice", "age": 30}]"#]))
            .schema(person_schema())
            .build();

        let result = chain.extract_from_document(&doc).await.unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Alice");
        assert_eq!(result.metadata["document_id"], "doc-123");
        assert_eq!(result.metadata["doc_source"], "test_file.txt");
    }

    // 14. Schema with nested objects
    #[tokio::test]
    async fn test_schema_with_nested_objects() {
        let schema = ExtractionSchema::builder()
            .name("Company")
            .description("A company entity")
            .required_field("name", FieldType::String, "Company name")
            .field(
                SchemaFieldBuilder::new("address", FieldType::Object)
                    .description("Company address")
                    .required(false)
                    .build(),
            )
            .field(
                SchemaFieldBuilder::new("employees", FieldType::Array)
                    .description("List of employees")
                    .required(false)
                    .build(),
            )
            .build();

        let response = r#"[{"name": "Acme Corp", "address": {"city": "New York", "state": "NY"}, "employees": ["Alice", "Bob"]}]"#;
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![response]))
            .schema(schema)
            .build();

        let result = chain
            .extract("Acme Corp is based in New York, NY with employees Alice and Bob")
            .await
            .unwrap();

        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Acme Corp");
        assert_eq!(result.entities[0]["address"]["city"], "New York");
        assert_eq!(result.entities[0]["employees"][0], "Alice");
    }

    // Additional: test single object response is wrapped in array
    #[tokio::test]
    async fn test_single_object_response() {
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![r#"{"name": "Solo", "age": 99}"#]))
            .schema(person_schema())
            .build();

        let result = chain.extract("Solo is 99 years old").await.unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Solo");
    }

    // Additional: test response with markdown code fences
    #[tokio::test]
    async fn test_response_with_code_fences() {
        let response = "```json\n[{\"name\": \"Fenced\", \"age\": 42}]\n```";
        let chain = ExtractionChain::builder()
            .llm(fake_model(vec![response]))
            .schema(person_schema())
            .build();

        let result = chain.extract("Fenced is 42").await.unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0]["name"], "Fenced");
    }
}
