//! Integration tests for `#[cognis::tool]` on an `impl` block — the
//! stateful form where the tool struct holds configuration such as
//! API keys or HTTP clients.

use cognis_core::error::Result;
use cognis_core::tool;
use cognis_core::tools::{BaseTool, ToolInput, ToolOutput};
use serde_json::json;
use std::collections::HashMap;

pub struct Calculator {
    offset: i64,
}

#[tool(name = "add_with_offset")]
impl Calculator {
    /// Add two numbers, biased by the offset configured on the Calculator.
    async fn add(&self, a: i64, b: i64) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!(a + b + self.offset)))
    }
}

#[tokio::test]
async fn impl_form_uses_receiver_state() {
    let calc = Calculator { offset: 100 };
    assert_eq!(calc.name(), "add_with_offset");
    assert_eq!(
        calc.description(),
        "Add two numbers, biased by the offset configured on the Calculator."
    );

    let mut m = HashMap::new();
    m.insert("a".to_string(), json!(2));
    m.insert("b".to_string(), json!(3));
    let out = calc._run(ToolInput::Structured(m)).await.unwrap();
    match out {
        ToolOutput::Content(v) => assert_eq!(v, json!(105)),
        _ => panic!("expected Content"),
    }
}

#[tokio::test]
async fn impl_form_preserves_inherent_method() {
    // The original async fn must still be callable as an inherent method,
    // independent of the generated BaseTool impl.
    let calc = Calculator { offset: 10 };
    let out = calc.add(1, 2).await.unwrap();
    match out {
        ToolOutput::Content(v) => assert_eq!(v, json!(13)),
        _ => panic!("expected Content"),
    }
}

// ---------------------------------------------------------------------------
// Impl form with schema validators on args
// ---------------------------------------------------------------------------

pub struct Translator {
    default_lang: String,
}

#[tool(name = "translate")]
impl Translator {
    /// Translate a piece of text into a target language.
    async fn translate(
        &self,
        #[schema(length(min = 1, max = 500))] text: String,
        #[schema(enum_values("en", "fr", "es", "de", "ja"))] target: Option<String>,
    ) -> Result<ToolOutput> {
        let target = target.unwrap_or_else(|| self.default_lang.clone());
        Ok(ToolOutput::Content(
            json!({ "text": text, "target": target }),
        ))
    }
}

#[tokio::test]
async fn impl_form_validates_enum_and_falls_back_to_state_default() {
    let t = Translator {
        default_lang: "fr".into(),
    };
    let mut m = HashMap::new();
    m.insert("text".to_string(), json!("hello"));
    let out = t._run(ToolInput::Structured(m)).await.unwrap();
    match out {
        ToolOutput::Content(v) => {
            assert_eq!(v["text"], json!("hello"));
            assert_eq!(v["target"], json!("fr"));
        }
        _ => panic!("expected Content"),
    }
}

#[tokio::test]
async fn impl_form_rejects_invalid_enum() {
    let t = Translator {
        default_lang: "fr".into(),
    };
    let mut m = HashMap::new();
    m.insert("text".to_string(), json!("hello"));
    m.insert("target".to_string(), json!("xx"));
    let err = t._run(ToolInput::Structured(m)).await.unwrap_err();
    assert!(err.to_string().contains("xx"), "got {err}");
}

#[tokio::test]
async fn impl_form_rejects_short_text() {
    let t = Translator {
        default_lang: "fr".into(),
    };
    let mut m = HashMap::new();
    m.insert("text".to_string(), json!(""));
    let err = t._run(ToolInput::Structured(m)).await.unwrap_err();
    assert!(err.to_string().contains("minimum"), "got {err}");
}

#[tokio::test]
async fn impl_form_exposes_schema() {
    let t = Translator {
        default_lang: "fr".into(),
    };
    let schema = t.args_schema().unwrap();
    assert_eq!(schema["properties"]["text"]["minLength"], json!(1));
    assert_eq!(schema["properties"]["text"]["maxLength"], json!(500));
    assert_eq!(
        schema["properties"]["target"]["enum"],
        json!(["en", "fr", "es", "de", "ja"])
    );
}
