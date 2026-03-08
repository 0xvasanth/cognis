pub mod base;
pub mod format;
pub mod json;
pub mod list;
pub mod openai_functions;
pub mod openai_tools;
pub mod pydantic;
pub mod string;
pub mod tools;
pub mod transform;
pub mod xml;

pub use base::{default_transform_stream, OutputParser, TransformOutputParser};
pub use json::JsonOutputParser;
pub use list::{
    CommaSeparatedListOutputParser, MarkdownListOutputParser, NumberedListOutputParser,
};
pub use openai_functions::{
    JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser, OutputFunctionsParser,
    SchemaAttrOutputFunctionsParser, SchemaOutputFunctionsParser,
};
pub use openai_tools::{JsonOutputKeyToolsParser, OpenAIToolsOutputParser};
pub use pydantic::SchemaOutputParser;
pub use string::StrOutputParser;
pub use tools::ToolCallOutputParser;
pub use transform::CumulativeTransformParser;
pub use xml::XmlOutputParser;

pub use format::{
    CommaSeparatedParser, DatetimeOutputParser, ExtractingJsonParser, ListOutputParser,
    OutputFixingParser, ParseError, PydanticOutputParser, RetryParser,
};

/// Standard JSON format instructions template.
///
/// Contains `{schema}` placeholder to be replaced with the actual JSON schema.
pub const JSON_FORMAT_INSTRUCTIONS: &str = r#"STRICT OUTPUT FORMAT:
- Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
- Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
- Do not prepend or append any text (e.g., do not write "Here is the JSON:").
- The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema (shown in a code block for readability only — do not include any backticks or Markdown in your output):
```
{schema}
```"#;

/// Pydantic-style format instructions template.
///
/// Mirrors Python's `PydanticOutputParser.get_format_instructions()`.
/// Contains `{schema}` placeholder to be replaced with the actual JSON schema.
pub const PYDANTIC_FORMAT_INSTRUCTIONS: &str = r#"The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"#;
