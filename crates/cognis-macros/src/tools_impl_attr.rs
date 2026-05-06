//! `#[tools_impl]` — outer attribute that scans an `impl` block for inner
//! `#[tool]`-marked async methods and generates one `BaseTool`-implementing
//! wrapper struct per method, plus an `into_tools()` collector on the user's
//! struct.
//!
//! Methods without `#[tool]` are passed through unchanged. The receiver
//! must be `&self` (consuming `self` is rejected; `&mut self` is rejected).
//!
//! Generated paths route through the configurable `crate_path` argument,
//! defaulting to `cognis_core` for backward compatibility with v1.

use proc_macro2::TokenStream as TokenStream2;
use syn::{
    parse::{Parse, ParseStream},
    ItemImpl, LitStr, Token,
};

#[derive(Default)]
pub(crate) struct ToolsImplArgs {
    pub crate_path: String,
}

impl Parse for ToolsImplArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ToolsImplArgs {
            crate_path: "cognis_core".to_string(),
        };
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "crate_path" => args.crate_path = input.parse::<LitStr>()?.value(),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[tools_impl] argument `{other}`; expected crate_path"),
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

pub(crate) fn expand(_args: ToolsImplArgs, _input: TokenStream2) -> syn::Result<TokenStream2> {
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "#[tools_impl] not yet implemented (Task 5)",
    ))
}

#[allow(dead_code)]
fn _unused(_: ItemImpl) {}
