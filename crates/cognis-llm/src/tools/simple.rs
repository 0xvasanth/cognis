//! Tier-3 ergonomic: `simple_tool!` declarative macro for inline tool
//! definitions. Useful for prototyping or short-lived tools that don't
//! need typed Params via `SchemaBasedTool`.
//!
//! # Example
//!
//! ```ignore
//! let echo = simple_tool! {
//!     name: "echo",
//!     description: "Echo the input",
//!     parameters: serde_json::json!({"type": "object"}),
//!     execute: |input| async move {
//!         Ok(cognis_llm::tools::ToolOutput::Content(input.into_json()))
//!     }
//! };
//! ```

/// Build an `Arc<dyn Tool>` inline.
#[macro_export]
macro_rules! simple_tool {
    (
        name: $name:expr,
        description: $description:expr,
        parameters: $params:expr,
        execute: $execute:expr $(,)?
    ) => {{
        struct __Inline {
            name: ::std::string::String,
            description: ::std::string::String,
            parameters: ::serde_json::Value,
        }
        #[$crate::tools::__simple_async_trait]
        impl $crate::tools::Tool for __Inline {
            fn name(&self) -> &str {
                &self.name
            }
            fn description(&self) -> &str {
                &self.description
            }
            fn args_schema(&self) -> ::core::option::Option<::serde_json::Value> {
                ::core::option::Option::Some(self.parameters.clone())
            }
            async fn _run(
                &self,
                input: $crate::tools::ToolInput,
            ) -> $crate::error::Result<$crate::tools::ToolOutput> {
                let f = $execute;
                f(input).await
            }
        }
        ::std::sync::Arc::new(__Inline {
            name: ::std::string::String::from($name),
            description: ::std::string::String::from($description),
            parameters: $params,
        }) as ::std::sync::Arc<dyn $crate::tools::Tool>
    }};
}

// Re-export async_trait under a private name so the macro doesn't pollute the
// caller's namespace.
#[doc(hidden)]
pub use async_trait::async_trait as __simple_async_trait;

#[cfg(test)]
mod tests {
    use crate::tools::ToolInput;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn simple_tool_macro_works() {
        let echo = simple_tool! {
            name: "echo",
            description: "echoes input",
            parameters: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            execute: |input: ToolInput| async move {
                Ok(crate::tools::ToolOutput::Content(input.into_json()))
            },
        };
        assert_eq!(echo.name(), "echo");
        assert_eq!(echo.description(), "echoes input");
        let _: &dyn crate::tools::Tool = echo.as_ref();

        let mut m = std::collections::HashMap::new();
        m.insert("x".into(), json!("hi"));
        let out = echo._run(ToolInput::Structured(m)).await.unwrap();
        if let crate::tools::ToolOutput::Content(v) = out {
            assert_eq!(v["x"], "hi");
        }
        let _ = Arc::clone(&echo); // Arc returned by macro, cloneable
    }
}
