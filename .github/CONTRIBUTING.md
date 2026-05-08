# Contributing to Cognis

Thanks for your interest in contributing to Cognis! This document covers the basics. The full contributor docs live at the docs site under [Contribute](https://github.com/0xvasanth/cognis/tree/main/docs/mintlify/contribute).

## Getting started

1. **Fork the repository** and clone your fork.
2. **Install Rust** (stable, 1.75+).
3. **Build the workspace**: `cargo build --workspace`.
4. **Run tests**: `cargo test --workspace`.

## Project structure

```
cognis/
├── crates/
│   ├── cognis-core/    # Foundation. Zero internal-crate deps.
│   ├── cognis-llm/     # LLM clients + providers (feature-gated).
│   ├── cognis-rag/     # Embeddings, vector stores, retrievers, splitters.
│   ├── cognisgraph/    # Stateful Graph<S>, Pregel engine, checkpointers.
│   ├── cognis-trace/   # Pluggable observability.
│   ├── cognis-macros/  # Proc macros: #[tool], #[derive(GraphState)].
│   └── cognis/         # Umbrella + agent layer. Re-exports the four siblings.
├── examples/           # Runnable demos under examples/<category>/
├── docs/mintlify/      # The docs site.
└── Cargo.toml          # Workspace root.
```

## Development workflow

### Before you start

- Check existing [issues](https://github.com/0xvasanth/cognis/issues) and [discussions](https://github.com/0xvasanth/cognis/discussions).
- For large changes, open an issue or discussion first to align on approach.

### Making changes

1. Create a feature branch from `main`.
2. Make your changes.
3. Ensure the pre-push checklist passes:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --features all-providers -- -D warnings
   cargo test --workspace
   ```

4. Write tests for new functionality.
5. Submit a pull request.

### Code guidelines

- **Dependency boundaries**: `cognis-core` must have zero internal-crate dependencies. The four sibling capability crates (`cognis-llm`, `cognis-rag`, `cognisgraph`, `cognis-trace`) depend only on `cognis-core` (and `cognis-macros` where they use a derive).
- **Feature flags**: Use them for any external integration (providers, vector stores, exporters).
- **Error handling**: `thiserror` per crate; cross-crate via `From` conversions. The umbrella `cognis` crate hand-rolls errors instead.
- **Async**: All I/O traits via `#[async_trait]`.
- **Documentation**: `///` doc comments on every public type, trait, and function.
- **Testing**: Tests next to the code; integration tests behind `#[cfg(feature = "integration_tests")]`.

### Commit messages

Conventional commits:

- `feat(crate): description` — new features
- `fix(crate): description` — bug fixes
- `docs(crate): description` — documentation
- `refactor(crate): description` — code refactoring
- `test(crate): description` — tests
- `chore: description` — maintenance

For changes spanning multiple crates, drop the `(crate)` suffix.

### Provider integrations

Each LLM provider should:

- Be gated behind a feature flag (e.g. `features = ["anthropic"]`).
- Implement the `LLMProvider` trait from `cognis-llm`.
- Include tests with mocked HTTP responses.
- Include an example under `examples/models/`.

The full step-by-step is in the docs: [Contribute → Adding a new provider](https://github.com/0xvasanth/cognis/blob/main/docs/mintlify/contribute/adding-a-provider.mdx).

## Reporting issues

- **Bugs**: Use the Bug Report template.
- **Features**: Use the Feature Request template.
- **Questions**: Use [Discussions](https://github.com/0xvasanth/cognis/discussions).

## Code of conduct

Be respectful and constructive. We're building something together. The full code of conduct is in [`docs/mintlify/contribute/code-of-conduct.mdx`](https://github.com/0xvasanth/cognis/blob/main/docs/mintlify/contribute/code-of-conduct.mdx).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).
