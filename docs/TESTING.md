# Testing

Tests are split into local deterministic tests and opt-in live integration tests.

## Local Commands

Run from the repository root:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Documentation-only changes can usually use:

```bash
cargo test --doc
git diff --check
```

Run the full local suite after code changes that touch public types, settings, provider adapters, streaming, auth, tokenizers, or retrieval clients.

## Local Test Files

- `crates/vv-llm/tests/public_api.rs` checks exported API types and construction.
- `crates/vv-llm/tests/settings.rs` checks settings parsing, endpoint binding, provider model override, and transport metadata preservation.
- `crates/vv-llm/tests/chat.rs` checks chat adapter request shapes, factory routing, multimodal mapping, tools, multi-turn tool messages, stream normalization, and Vertex token cache behavior.
- `crates/vv-llm/tests/retrieval.rs` checks embedding and rerank request mapping.
- `crates/vv-llm/tests/utilities.rs` checks message normalization, tokenizer behavior, fallback counting, and retry metadata.

Prefer deterministic tests for mapping logic. A provider feature should not require a live API call just to verify that local request construction is correct.

## Live Integration Tests

Live tests are ignored by default and call real provider APIs. They run through:

```bash
VV_LLM_RUN_LIVE_TESTS=1 ./scripts/run_live_tests.sh
```

The script runs:

```bash
cargo test --test live_tests -- --ignored --test-threads=1
```

The single-threaded mode reduces provider rate-limit noise and makes failures easier to read.

## Live Settings

Live settings load from the first available source:

1. `VV_LLM_SETTINGS_JSON`
2. `crates/vv-llm/tests/fixtures/dev_settings.json`
3. `crates/vv-llm/tests/fixtures/sample_settings.json`

`dev_settings.json` is gitignored and may contain real credentials. Do not print it, paste it into logs, include it in diffs, or summarize actual key values.

Use `crates/vv-llm/tests/fixtures/dev_settings.example.json` and `sample_settings.json` for documented shapes and committed examples.

Credential detection accepts:

- Plain API keys in endpoint `api_key`.
- Bedrock `credentials.access_key` and `credentials.secret_key`.
- Vertex refresh-token credentials: `refresh_token`, `client_id`, and `client_secret`.
- Vertex service-account credentials: `private_key` and `client_email`.

`VV_LLM_ALLOW_EMPTY_KEYS=1` can bypass credential validation for local fixture-debugging only. It does not make live API calls succeed.

## Current Live Coverage

`live_tests.rs` covers:

- DeepSeek OpenAI-compatible chat completion.
- Qwen OpenAI-compatible chat with system prompt.
- ZhiPuAI OpenAI-compatible chat completion.
- Anthropic resolved client chat.
- Anthropic Bedrock image understanding.
- Anthropic Bedrock tool call.
- Anthropic Bedrock tool-result multi-turn conversation.
- Anthropic Bedrock streaming text.
- Anthropic Bedrock streaming tool call.
- Gemini OpenAI-compatible Vertex chat completion when configured.
- SiliconFlow embedding.
- SiliconFlow rerank.

When adding a provider feature, add a deterministic local test first. Add a live test when behavior depends on real provider contracts, auth, or streaming event sequences.

## Failure Handling

- For local test failures, inspect the failing assertion and provider adapter mapping before changing public types.
- For live failures, first distinguish configuration/auth/rate-limit errors from adapter bugs.
- Do not paste secret-bearing provider error payloads into docs or commit messages.
- If a live provider is temporarily unavailable, leave deterministic coverage in place and state exactly which live command failed.
