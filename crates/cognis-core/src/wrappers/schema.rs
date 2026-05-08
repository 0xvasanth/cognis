//! Schema introspection wrapper — fills [`Runnable::input_schema`] and
//! [`Runnable::output_schema`] for any `(I, O)` whose types implement
//! [`schemars::JsonSchema`].
//!
//! Why a wrapper rather than a blanket impl? Many runnables are generic
//! over `(I, O)` without bounding `I/O: JsonSchema`, and the trait
//! itself is object-safe. The wrapper opts the runnable in at the call
//! site without polluting the trait.
//!
//! Customization: pass user-supplied JSON Schema values via
//! [`WithSchema::with_schemas`] to override the schemars-derived ones
//! (useful when the runnable accepts a wider type than its schema
//! advertises, or when schema generation is too expensive).

use std::marker::PhantomData;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Wraps a runnable so that `input_schema()` / `output_schema()` return
/// real [`schemars::JsonSchema`]-derived JSON Schema values.
pub struct WithSchema<R, I, O> {
    inner: R,
    input_schema: serde_json::Value,
    output_schema: serde_json::Value,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> WithSchema<R, I, O>
where
    R: Runnable<I, O>,
    I: schemars::JsonSchema + Send + 'static,
    O: schemars::JsonSchema + Send + 'static,
{
    /// Wrap with schemas auto-derived from `I` and `O` via `schemars`.
    pub fn new(inner: R) -> Self {
        let input_schema =
            serde_json::to_value(schemars::schema_for!(I)).unwrap_or(serde_json::Value::Null);
        let output_schema =
            serde_json::to_value(schemars::schema_for!(O)).unwrap_or(serde_json::Value::Null);
        Self {
            inner,
            input_schema,
            output_schema,
            _phantom: PhantomData,
        }
    }
}

impl<R, I, O> WithSchema<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap with caller-supplied schemas (no `JsonSchema` bound on
    /// `I`/`O`). Use when the runnable accepts a more permissive type
    /// than its public schema advertises.
    pub fn with_schemas(
        inner: R,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self {
            inner,
            input_schema,
            output_schema,
            _phantom: PhantomData,
        }
    }

    /// Replace just the input schema (e.g. after a [`Self::new`]).
    pub fn override_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    /// Replace just the output schema.
    pub fn override_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = schema;
        self
    }

    /// Borrow the configured schemas (read-only).
    pub fn schemas(&self) -> (&serde_json::Value, &serde_json::Value) {
        (&self.input_schema, &self.output_schema)
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for WithSchema<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        self.inner.invoke(input, config).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_schema(&self) -> Option<serde_json::Value> {
        Some(self.input_schema.clone())
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        Some(self.output_schema.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, JsonSchema)]
    struct In {
        topic: String,
    }
    #[derive(Serialize, Deserialize, JsonSchema)]
    struct Out {
        summary: String,
    }

    struct R;

    #[async_trait]
    impl Runnable<In, Out> for R {
        async fn invoke(&self, input: In, _: RunnableConfig) -> Result<Out> {
            Ok(Out {
                summary: format!("about {}", input.topic),
            })
        }
    }

    #[tokio::test]
    async fn auto_derives_schemas_from_jsonschema() {
        let wrapped: WithSchema<R, In, Out> = WithSchema::new(R);
        let inp = wrapped.input_schema().unwrap();
        let out = wrapped.output_schema().unwrap();
        // schemars emits a top-level "title" referencing the type name.
        assert!(inp.to_string().contains("topic"));
        assert!(out.to_string().contains("summary"));
    }

    #[tokio::test]
    async fn override_schemas_replaces_derived() {
        let wrapped: WithSchema<R, In, Out> =
            WithSchema::new(R).override_input_schema(serde_json::json!({"custom": true}));
        let inp = wrapped.input_schema().unwrap();
        assert_eq!(inp["custom"], true);
    }

    #[tokio::test]
    async fn invoke_pass_through() {
        let wrapped: WithSchema<R, In, Out> = WithSchema::new(R);
        let out = wrapped
            .invoke(
                In {
                    topic: "rust".into(),
                },
                RunnableConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(out.summary, "about rust");
    }

    #[tokio::test]
    async fn with_schemas_skips_jsonschema_bound() {
        // When I/O don't impl JsonSchema, with_schemas still works.
        struct PlainIn(#[allow(dead_code)] String);
        struct PlainOut(#[allow(dead_code)] String);
        struct P;
        #[async_trait]
        impl Runnable<PlainIn, PlainOut> for P {
            async fn invoke(&self, input: PlainIn, _: RunnableConfig) -> Result<PlainOut> {
                Ok(PlainOut(input.0))
            }
        }
        let wrapped: WithSchema<P, PlainIn, PlainOut> = WithSchema::with_schemas(
            P,
            serde_json::json!({"type": "string"}),
            serde_json::json!({"type": "string"}),
        );
        assert_eq!(wrapped.input_schema().unwrap()["type"], "string");
    }
}
