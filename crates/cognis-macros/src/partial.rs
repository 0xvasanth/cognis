//! Implementation of `#[derive(Partial)]`.
//!
//! Generates `<Name>Partial`, a mirror of a named-field struct with every
//! field wrapped in `Option`, plus the `cognis_core::Partial` impl linking the
//! two. Used so streamed JSON snapshots, which lack not-yet-emitted fields,
//! still deserialize.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

pub fn derive_partial(ast: DeriveInput) -> TokenStream {
    match expand(&ast) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error(),
    }
}

fn expand(ast: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &ast.ident;
    if !ast.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &ast.generics,
            "Partial does not support generic structs",
        ));
    }
    let fields = match &ast.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "Partial supports named-field structs only",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Partial supports structs only",
            ))
        }
    };

    let vis = &ast.vis;
    let partial_name = format_ident!("{}Partial", name);
    let decls = fields.iter().filter_map(|f| {
        let id = f.ident.as_ref()?;
        let ty = &f.ty;
        let fvis = &f.vis;
        Some(quote! {
            #[serde(default)]
            #fvis #id: ::std::option::Option<#ty>
        })
    });

    Ok(quote! {
        #[derive(::std::fmt::Debug, ::std::default::Default, ::serde::Deserialize)]
        #vis struct #partial_name {
            #(#decls),*
        }

        impl ::cognis_core::Partial for #name {
            type Partial = #partial_name;
        }
    })
}
