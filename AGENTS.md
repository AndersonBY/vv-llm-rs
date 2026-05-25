# AGENTS.md

This file is the entry point for coding agents working in this repository. Keep it short; deeper project knowledge belongs in `docs/`.

## Start Here

- Read `docs/README.md` for the documentation map.
- Read `docs/ARCHITECTURE.md` before changing public types, settings resolution, provider routing, streaming, embeddings, rerank, or tokenizer behavior.
- Read `docs/PROVIDERS.md` before touching OpenAI-compatible, Anthropic, Bedrock, or Vertex behavior.
- Read `docs/TESTING.md` before adding tests or running live integration tests.
- Read `docs/SECURITY.md` before handling credentials, logs, fixtures, auth, or provider error payloads.

## Project Rules

- This is a Rust workspace for the `vv-llm` crate. Run commands from the repository root unless a doc says otherwise.
- Treat `crates/vv-llm/tests/fixtures/dev_settings.json` as secret-bearing local state. Do not print, paste, commit, or summarize its values.
- Prefer typed SDK surfaces over hand-parsed provider protocols. Current SDKs include `async-openai`, `anthropic`, and AWS Bedrock SDK crates.
- Keep public request/response types provider-neutral. Provider-specific payload shapes should stay inside provider adapter modules.
- Preserve `LlmSettings` compatibility with the shared VectorVein settings shape: `endpoints`, `backends`, `embedding_backends`, `rerank_backends`, string/object endpoint bindings, and transport metadata.
- Add or update tests with behavior changes. Synthetic tests should cover mapping and normalization; live tests should cover real provider paths when credentials are available.
- After public API, provider behavior, settings, tests, or significant docs changes, check whether README and `docs/` need updates.

## Verification

Default local verification:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Documentation-only changes can use:

```bash
cargo test --doc
git diff --check
```

Live integration tests are opt-in and call real provider APIs:

```bash
VV_LLM_RUN_LIVE_TESTS=1 ./scripts/run_live_tests.sh
```

Run live tests only when the user asks for them or has explicitly confirmed that usable credentials are configured.
