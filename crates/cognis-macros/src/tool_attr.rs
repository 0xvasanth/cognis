//! `#[cognis::tool]` attribute macro.
//!
//! Wraps an `async fn` (either standalone or inside an `impl` block) to
//! generate a `BaseTool` implementation with typed, schema-validated args.
//!
//! Two forms are supported:
//!
//! - **Standalone** — `#[tool]` on a free `async fn`. The macro emits a
//!   unit struct with the fn's PascalCase name; that struct is the
//!   `BaseTool` implementation.
//! - **Impl-block** — `#[tool]` on an `impl` block containing exactly one
//!   `async fn`. The macro preserves the original inherent method and
//!   emits a separate `BaseTool` impl whose `_run` dispatches into that
//!   method (letting tools hold state like HTTP clients or API keys).
//!
//! Generated code references `cognis_core` by absolute path, so callers
//! only need `cognis_core` and (for pattern validators) `regex` in scope.
//! The macro re-exports regex via `cognis_core::tools::validation::__regex`
//! — users do not need to add `regex` to their `Cargo.toml`.

use proc_macro2::TokenStream as TokenStream2;
use syn::{
    parse::{Parse, ParseStream},
    LitStr, Token,
};

/// Parsed arguments of the `#[tool(...)]` attribute.
#[derive(Default)]
pub(crate) struct ToolArgs {
    pub name: Option<String>,
    pub description: Option<String>,
    pub return_direct: Option<bool>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ToolArgs::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "name" => args.name = Some(input.parse::<LitStr>()?.value()),
                "description" => args.description = Some(input.parse::<LitStr>()?.value()),
                "return_direct" => {
                    let b: syn::LitBool = input.parse()?;
                    args.return_direct = Some(b.value);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[tool] argument `{other}`; expected name, description, or return_direct"
                        ),
                    ))
                }
            }
            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
            }
        }
        Ok(args)
    }
}

/// Entry point — dispatches to the correct form based on what `input`
/// parses as. During scaffolding this is a no-op; real expansion lands
/// in subsequent commits.
pub(crate) fn expand(_args: ToolArgs, input: TokenStream2) -> syn::Result<TokenStream2> {
    Ok(input)
}
