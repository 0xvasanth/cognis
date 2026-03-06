use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// Column & Table schema types
// ---------------------------------------------------------------------------

/// Describes a single column in a database table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// SQL type (e.g. `"INTEGER"`, `"VARCHAR(255)"`).
    pub column_type: String,
    /// Whether the column is nullable. Default `true`.
    pub nullable: bool,
    /// Whether this column is part of the primary key.
    pub primary_key: bool,
}

/// Builder for [`ColumnSchema`].
pub struct ColumnSchemaBuilder {
    name: String,
    column_type: String,
    nullable: bool,
    primary_key: bool,
}

impl ColumnSchemaBuilder {
    /// Create a builder with the required column name and type.
    pub fn new(name: impl Into<String>, column_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            column_type: column_type.into(),
            nullable: true,
            primary_key: false,
        }
    }

    /// Mark this column as NOT NULL.
    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self
    }

    /// Mark this column as nullable (the default).
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Mark this column as a primary key.
    pub fn primary_key(mut self) -> Self {
        self.primary_key = true;
        self
    }

    /// Build the [`ColumnSchema`].
    pub fn build(self) -> ColumnSchema {
        ColumnSchema {
            name: self.name,
            column_type: self.column_type,
            nullable: self.nullable,
            primary_key: self.primary_key,
        }
    }
}

/// Describes a database table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TableSchema {
    /// Table name.
    pub name: String,
    /// Columns in the table.
    pub columns: Vec<ColumnSchema>,
    /// Optional human-readable description.
    pub description: Option<String>,
}

/// Builder for [`TableSchema`].
pub struct TableSchemaBuilder {
    name: String,
    columns: Vec<ColumnSchema>,
    description: Option<String>,
}

impl TableSchemaBuilder {
    /// Create a builder with the required table name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: Vec::new(),
            description: None,
        }
    }

    /// Add a column to the table.
    pub fn column(mut self, column: ColumnSchema) -> Self {
        self.columns.push(column);
        self
    }

    /// Set the table description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Build the [`TableSchema`].
    pub fn build(self) -> TableSchema {
        TableSchema {
            name: self.name,
            columns: self.columns,
            description: self.description,
        }
    }
}

// ---------------------------------------------------------------------------
// DatabaseSchema
// ---------------------------------------------------------------------------

/// Represents the schema of a database (a collection of tables).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatabaseSchema {
    /// All tables in the database.
    pub tables: Vec<TableSchema>,
}

/// Builder for [`DatabaseSchema`].
pub struct DatabaseSchemaBuilder {
    tables: Vec<TableSchema>,
}

impl DatabaseSchemaBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Add a table to the schema.
    pub fn table(mut self, table: TableSchema) -> Self {
        self.tables.push(table);
        self
    }

    /// Build the [`DatabaseSchema`].
    pub fn build(self) -> DatabaseSchema {
        DatabaseSchema {
            tables: self.tables,
        }
    }
}

impl Default for DatabaseSchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DatabaseSchema {
    /// Start a new builder.
    pub fn builder() -> DatabaseSchemaBuilder {
        DatabaseSchemaBuilder::new()
    }

    /// Generate DDL (`CREATE TABLE` statements) for all tables.
    pub fn to_ddl(&self) -> String {
        let mut ddl = String::new();
        for (i, table) in self.tables.iter().enumerate() {
            if i > 0 {
                ddl.push_str("\n\n");
            }
            if let Some(ref desc) = table.description {
                ddl.push_str(&format!("-- {}\n", desc));
            }
            ddl.push_str(&format!("CREATE TABLE {} (\n", table.name));
            let pk_cols: Vec<&str> = table
                .columns
                .iter()
                .filter(|c| c.primary_key)
                .map(|c| c.name.as_str())
                .collect();

            for (j, col) in table.columns.iter().enumerate() {
                ddl.push_str(&format!("  {} {}", col.name, col.column_type));
                if !col.nullable {
                    ddl.push_str(" NOT NULL");
                }
                if col.primary_key && pk_cols.len() == 1 {
                    ddl.push_str(" PRIMARY KEY");
                }
                if j < table.columns.len() - 1 || pk_cols.len() > 1 {
                    ddl.push(',');
                }
                ddl.push('\n');
            }
            if pk_cols.len() > 1 {
                ddl.push_str(&format!("  PRIMARY KEY ({})\n", pk_cols.join(", ")));
            }
            ddl.push_str(");");
        }
        ddl
    }

    /// Parse a DDL string into a `DatabaseSchema`.
    ///
    /// This is a basic parser that handles simple `CREATE TABLE` statements.
    /// It does not support all SQL DDL features.
    pub fn from_ddl(ddl: &str) -> Result<Self> {
        let table_re =
            Regex::new(r"(?is)CREATE\s+TABLE\s+(\w+)\s*\(\s*(.*?)\s*\)\s*;").unwrap();
        let mut tables = Vec::new();

        for cap in table_re.captures_iter(ddl) {
            let table_name = cap[1].to_string();
            let body = &cap[2];
            let mut columns = Vec::new();

            for line in body.split(',') {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // Skip composite PRIMARY KEY constraints
                let upper = line.to_uppercase();
                if upper.starts_with("PRIMARY KEY") || upper.starts_with("FOREIGN KEY")
                    || upper.starts_with("UNIQUE") || upper.starts_with("CHECK")
                    || upper.starts_with("CONSTRAINT")
                {
                    continue;
                }
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() < 2 {
                    continue;
                }
                let col_name = tokens[0].to_string();
                let col_type = tokens[1].to_string();
                let rest = upper;
                let nullable = !rest.contains("NOT NULL");
                let primary_key = rest.contains("PRIMARY KEY");

                columns.push(ColumnSchema {
                    name: col_name,
                    column_type: col_type,
                    nullable,
                    primary_key,
                });
            }

            tables.push(TableSchema {
                name: table_name,
                columns,
                description: None,
            });
        }

        if tables.is_empty() {
            return Err(RustChainError::Other(
                "No CREATE TABLE statements found in DDL".into(),
            ));
        }

        Ok(DatabaseSchema { tables })
    }
}

// ---------------------------------------------------------------------------
// SQL Query Validator
// ---------------------------------------------------------------------------

/// Validates SQL queries for safety by checking for dangerous operations.
#[derive(Debug, Clone)]
pub struct SQLQueryValidator {
    /// Set of disallowed SQL operations (uppercased keywords).
    disallowed_operations: HashSet<String>,
}

impl SQLQueryValidator {
    /// Create a validator with the default disallowed operations:
    /// `DROP`, `DELETE`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`.
    pub fn new() -> Self {
        let ops = ["DROP", "DELETE", "ALTER", "TRUNCATE", "INSERT", "UPDATE"];
        Self {
            disallowed_operations: ops.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create a validator with a custom set of disallowed operations.
    pub fn with_disallowed(operations: &[&str]) -> Self {
        Self {
            disallowed_operations: operations
                .iter()
                .map(|s| s.to_uppercase())
                .collect(),
        }
    }

    /// Create a validator that allows everything (no restrictions).
    pub fn permissive() -> Self {
        Self {
            disallowed_operations: HashSet::new(),
        }
    }

    /// Validate the given SQL query string.
    ///
    /// Returns `Ok(())` if valid, or an error describing the violation.
    pub fn validate(&self, sql: &str) -> Result<()> {
        let upper = sql.to_uppercase();
        // Tokenize on whitespace and common SQL punctuation for matching
        let tokens: Vec<&str> = upper.split_whitespace().collect();
        for token in &tokens {
            // Strip trailing punctuation like ( ; etc.
            let clean = token.trim_matches(|c: char| !c.is_alphanumeric());
            if self.disallowed_operations.contains(clean) {
                return Err(RustChainError::Other(format!(
                    "SQL validation error: disallowed operation '{}' found in query",
                    clean,
                )));
            }
        }
        Ok(())
    }
}

impl Default for SQLQueryValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Default prompt
// ---------------------------------------------------------------------------

const DEFAULT_TEXT_TO_SQL_PROMPT: &str = r#"Given the following database schema:

{schema}

Generate a SQL query that answers the following question:
{question}

Return ONLY the SQL query, without any explanation or markdown formatting."#;

// ---------------------------------------------------------------------------
// TextToSQLChain
// ---------------------------------------------------------------------------

/// A chain that converts natural-language questions into SQL queries.
///
/// It sends a prompt containing the database schema and the user question to
/// an LLM, then extracts and optionally validates the resulting SQL.
pub struct TextToSQLChain {
    model: Arc<dyn BaseChatModel>,
    schema: DatabaseSchema,
    prompt_template: String,
    validate_sql: bool,
    include_schema_in_prompt: bool,
    validator: SQLQueryValidator,
}

/// Builder for [`TextToSQLChain`].
pub struct TextToSQLChainBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    schema: Option<DatabaseSchema>,
    prompt_template: String,
    validate_sql: bool,
    include_schema_in_prompt: bool,
    validator: SQLQueryValidator,
}

impl TextToSQLChainBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            model: None,
            schema: None,
            prompt_template: DEFAULT_TEXT_TO_SQL_PROMPT.to_string(),
            validate_sql: true,
            include_schema_in_prompt: true,
            validator: SQLQueryValidator::new(),
        }
    }

    /// Set the chat model (required).
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the database schema (required).
    pub fn schema(mut self, schema: DatabaseSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set a custom prompt template.
    ///
    /// The template should contain `{schema}` and `{question}` placeholders.
    pub fn prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Enable or disable SQL validation. Default: `true`.
    pub fn validate_sql(mut self, validate: bool) -> Self {
        self.validate_sql = validate;
        self
    }

    /// Whether to include the schema DDL in the prompt. Default: `true`.
    pub fn include_schema_in_prompt(mut self, include: bool) -> Self {
        self.include_schema_in_prompt = include;
        self
    }

    /// Set a custom SQL validator.
    pub fn validator(mut self, validator: SQLQueryValidator) -> Self {
        self.validator = validator;
        self
    }

    /// Build the [`TextToSQLChain`]. Panics if model or schema is not set.
    pub fn build(self) -> TextToSQLChain {
        TextToSQLChain {
            model: self.model.expect("model is required for TextToSQLChain"),
            schema: self.schema.expect("schema is required for TextToSQLChain"),
            prompt_template: self.prompt_template,
            validate_sql: self.validate_sql,
            include_schema_in_prompt: self.include_schema_in_prompt,
            validator: self.validator,
        }
    }
}

impl Default for TextToSQLChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TextToSQLChain {
    /// Start a new builder.
    pub fn builder() -> TextToSQLChainBuilder {
        TextToSQLChainBuilder::new()
    }

    /// Format the prompt template by replacing `{variable}` placeholders.
    fn format_prompt(&self, input: &Value) -> Result<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let obj = input
            .as_object()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object".into(),
                got: format!("{}", input),
            })?;

        let schema_ddl = if self.include_schema_in_prompt {
            self.schema.to_ddl()
        } else {
            String::new()
        };

        let mut missing: Vec<String> = Vec::new();
        let result = re.replace_all(&self.prompt_template, |caps: &regex::Captures| {
            let key = &caps[1];
            match key {
                "schema" => schema_ddl.clone(),
                _ => match obj.get(key) {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => {
                        missing.push(key.to_string());
                        String::new()
                    }
                },
            }
        });

        if !missing.is_empty() {
            return Err(RustChainError::InvalidKey(format!(
                "Missing input variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(result.into_owned())
    }

    /// Extract SQL from a model response, handling markdown code blocks.
    fn extract_sql(response: &str) -> String {
        let trimmed = response.trim();

        // Try to extract from ```sql ... ``` code block
        let sql_block_re = Regex::new(r"(?is)```(?:sql)?\s*\n?(.*?)\n?\s*```").unwrap();
        if let Some(cap) = sql_block_re.captures(trimmed) {
            return cap[1].trim().to_string();
        }

        // Otherwise return the raw response trimmed
        trimmed.to_string()
    }
}

#[async_trait]
impl Runnable for TextToSQLChain {
    fn name(&self) -> &str {
        "TextToSQLChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let question = input
            .as_object()
            .and_then(|o| o.get("question"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| RustChainError::InvalidKey(
                "Input must be a JSON object with a 'question' key".into(),
            ))?
            .to_string();

        let formatted = self.format_prompt(&input)?;
        let messages = vec![Message::Human(HumanMessage::new(&formatted))];
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        let raw_text = ai_msg.base.content.text();
        let sql = Self::extract_sql(&raw_text);

        if self.validate_sql {
            self.validator.validate(&sql)?;
        }

        Ok(json!({
            "sql": sql,
            "question": question,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::FakeListChatModel;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn sample_schema() -> DatabaseSchema {
        DatabaseSchema::builder()
            .table(
                TableSchemaBuilder::new("users")
                    .column(
                        ColumnSchemaBuilder::new("id", "INTEGER")
                            .not_null()
                            .primary_key()
                            .build(),
                    )
                    .column(
                        ColumnSchemaBuilder::new("name", "VARCHAR(255)")
                            .not_null()
                            .build(),
                    )
                    .column(
                        ColumnSchemaBuilder::new("email", "VARCHAR(255)")
                            .build(),
                    )
                    .description("User accounts")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn test_basic_question_to_sql() {
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["SELECT COUNT(*) FROM users"]))
            .schema(sample_schema())
            .build();

        let result = chain
            .invoke(json!({"question": "How many users are there?"}), None)
            .await
            .unwrap();

        assert_eq!(result["sql"], "SELECT COUNT(*) FROM users");
        assert_eq!(result["question"], "How many users are there?");
    }

    #[test]
    fn test_schema_ddl_generation() {
        let schema = sample_schema();
        let ddl = schema.to_ddl();
        assert!(ddl.contains("CREATE TABLE users"));
        assert!(ddl.contains("id INTEGER NOT NULL PRIMARY KEY"));
        assert!(ddl.contains("name VARCHAR(255) NOT NULL"));
        assert!(ddl.contains("email VARCHAR(255)"));
        assert!(ddl.contains("-- User accounts"));
    }

    #[tokio::test]
    async fn test_sql_extraction_from_markdown_code_blocks() {
        let response = "Here is the SQL:\n```sql\nSELECT * FROM users WHERE id = 1;\n```";
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec![response]))
            .schema(sample_schema())
            .build();

        let result = chain
            .invoke(json!({"question": "Get user 1"}), None)
            .await
            .unwrap();

        assert_eq!(result["sql"], "SELECT * FROM users WHERE id = 1;");
    }

    #[tokio::test]
    async fn test_sql_validation_rejects_drop() {
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["DROP TABLE users"]))
            .schema(sample_schema())
            .build();

        let result = chain
            .invoke(json!({"question": "delete everything"}), None)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("DROP"), "Error should mention DROP: {err}");
    }

    #[tokio::test]
    async fn test_sql_validation_allows_select() {
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["SELECT * FROM users"]))
            .schema(sample_schema())
            .build();

        let result = chain
            .invoke(json!({"question": "show all users"}), None)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap()["sql"], "SELECT * FROM users");
    }

    #[tokio::test]
    async fn test_custom_prompt_template() {
        let custom_prompt = "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:";
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["SELECT name FROM users"]))
            .schema(sample_schema())
            .prompt_template(custom_prompt)
            .build();

        let result = chain
            .invoke(json!({"question": "list user names"}), None)
            .await
            .unwrap();

        assert_eq!(result["sql"], "SELECT name FROM users");
    }

    #[test]
    fn test_database_schema_builder() {
        let schema = DatabaseSchema::builder()
            .table(
                TableSchemaBuilder::new("products")
                    .column(
                        ColumnSchemaBuilder::new("id", "INTEGER")
                            .primary_key()
                            .not_null()
                            .build(),
                    )
                    .column(
                        ColumnSchemaBuilder::new("price", "DECIMAL(10,2)")
                            .not_null()
                            .build(),
                    )
                    .description("Product catalog")
                    .build(),
            )
            .build();

        assert_eq!(schema.tables.len(), 1);
        assert_eq!(schema.tables[0].name, "products");
        assert_eq!(schema.tables[0].columns.len(), 2);
        assert_eq!(schema.tables[0].description.as_deref(), Some("Product catalog"));
    }

    #[test]
    fn test_table_schema_all_column_types() {
        let table = TableSchemaBuilder::new("test_table")
            .column(ColumnSchemaBuilder::new("pk", "INTEGER").primary_key().not_null().build())
            .column(ColumnSchemaBuilder::new("text_col", "TEXT").build())
            .column(ColumnSchemaBuilder::new("bool_col", "BOOLEAN").not_null().build())
            .column(ColumnSchemaBuilder::new("ts_col", "TIMESTAMP").build())
            .column(ColumnSchemaBuilder::new("float_col", "FLOAT").nullable(false).build())
            .build();

        assert_eq!(table.columns.len(), 5);
        assert!(table.columns[0].primary_key);
        assert!(!table.columns[0].nullable);
        assert!(table.columns[1].nullable);
        assert!(!table.columns[2].nullable);
        assert!(table.columns[3].nullable);
        assert!(!table.columns[4].nullable);
    }

    #[tokio::test]
    async fn test_runnable_trait_implementation() {
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["SELECT 1"]))
            .schema(sample_schema())
            .build();

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "TextToSQLChain");

        let result = runnable
            .invoke(json!({"question": "test"}), None)
            .await
            .unwrap();
        assert_eq!(result["sql"], "SELECT 1");
    }

    #[test]
    fn test_custom_allowed_operations() {
        let validator = SQLQueryValidator::with_disallowed(&["DROP", "TRUNCATE"]);

        // DELETE should now be allowed
        assert!(validator.validate("DELETE FROM users WHERE id = 1").is_ok());
        // DROP should still be disallowed
        assert!(validator.validate("DROP TABLE users").is_err());
        // TRUNCATE should still be disallowed
        assert!(validator.validate("TRUNCATE TABLE users").is_err());
        // SELECT is fine
        assert!(validator.validate("SELECT * FROM users").is_ok());
    }

    #[test]
    fn test_from_ddl_parsing() {
        let ddl = r#"
CREATE TABLE users (
  id INTEGER NOT NULL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email TEXT
);
        "#;

        let schema = DatabaseSchema::from_ddl(ddl).unwrap();
        assert_eq!(schema.tables.len(), 1);
        assert_eq!(schema.tables[0].name, "users");
        assert_eq!(schema.tables[0].columns.len(), 3);
        assert_eq!(schema.tables[0].columns[0].name, "id");
        assert!(!schema.tables[0].columns[0].nullable);
        assert!(schema.tables[0].columns[0].primary_key);
        assert_eq!(schema.tables[0].columns[1].name, "name");
        assert!(!schema.tables[0].columns[1].nullable);
        assert!(schema.tables[0].columns[2].nullable);
    }

    #[test]
    fn test_multi_table_schema() {
        let schema = DatabaseSchema::builder()
            .table(
                TableSchemaBuilder::new("orders")
                    .column(ColumnSchemaBuilder::new("id", "INTEGER").primary_key().not_null().build())
                    .column(ColumnSchemaBuilder::new("user_id", "INTEGER").not_null().build())
                    .column(ColumnSchemaBuilder::new("total", "DECIMAL(10,2)").build())
                    .description("Customer orders")
                    .build(),
            )
            .table(
                TableSchemaBuilder::new("order_items")
                    .column(ColumnSchemaBuilder::new("id", "INTEGER").primary_key().not_null().build())
                    .column(ColumnSchemaBuilder::new("order_id", "INTEGER").not_null().build())
                    .column(ColumnSchemaBuilder::new("product", "VARCHAR(255)").build())
                    .column(ColumnSchemaBuilder::new("quantity", "INTEGER").not_null().build())
                    .build(),
            )
            .build();

        let ddl = schema.to_ddl();
        assert!(ddl.contains("CREATE TABLE orders"));
        assert!(ddl.contains("CREATE TABLE order_items"));
        assert!(ddl.contains("-- Customer orders"));
        assert_eq!(schema.tables.len(), 2);
        assert_eq!(schema.tables[0].columns.len(), 3);
        assert_eq!(schema.tables[1].columns.len(), 4);
    }

    #[tokio::test]
    async fn test_sql_extraction_from_plain_code_block() {
        let response = "```\nSELECT name FROM users;\n```";
        let sql = TextToSQLChain::extract_sql(response);
        assert_eq!(sql, "SELECT name FROM users;");
    }

    #[tokio::test]
    async fn test_validation_disabled() {
        let chain = TextToSQLChain::builder()
            .model(fake_model(vec!["DROP TABLE users"]))
            .schema(sample_schema())
            .validate_sql(false)
            .build();

        let result = chain
            .invoke(json!({"question": "drop users"}), None)
            .await;

        // Should succeed when validation is disabled
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["sql"], "DROP TABLE users");
    }
}
