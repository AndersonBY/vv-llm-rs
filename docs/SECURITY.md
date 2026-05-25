# Security And Credentials

`vv-llm-rs` interacts with real LLM provider credentials during live tests and in settings-based clients. Keep secrets out of commits, logs, docs, and chat transcripts.

## Secret-Bearing Files

`crates/vv-llm/tests/fixtures/dev_settings.json` is local-only and gitignored. It may contain:

- Provider API keys.
- AWS Bedrock access keys.
- Google OAuth client secrets and refresh tokens.
- Google service-account private keys.

Rules:

- Do not print this file.
- Do not include its values in test output, docs, commit messages, or issue text.
- Do not copy values into examples.
- Use `dev_settings.example.json` and `sample_settings.json` for committed shapes.
- If a command accidentally prints secrets, stop and treat the output as sensitive.

## Environment Variables

`VV_LLM_SETTINGS_JSON` may point to an alternate settings file. Treat that target as secret-bearing unless proven otherwise.

`VV_LLM_RUN_LIVE_TESTS=1` enables real API calls.

`VV_LLM_ALLOW_EMPTY_KEYS=1` only bypasses local credential validation. It should not be used as proof that live tests are meaningful.

## Logging

Provider errors can include request ids, model ids, quota details, or partial request context. Avoid dumping full error payloads when they may include credentials or raw request bodies.

Safe logging usually includes:

- Test name.
- Provider name.
- Model alias, when not sensitive.
- High-level error class.
- HTTP status code.

Unsafe logging includes:

- `Authorization` headers.
- Raw endpoint credentials.
- Google refresh tokens, client secrets, service-account private keys, or access tokens.
- AWS access keys or secret keys.
- Full settings JSON.

## Vertex Credentials

Vertex OpenAI-compatible endpoints use `endpoint_type: "openai_vertex"` and credentials in endpoint config.

Supported credential forms:

- User refresh-token credentials: `refresh_token`, `client_id`, `client_secret`.
- Service-account credentials: `private_key`, `client_email`.

The adapter exchanges credentials for an access token and caches fresh tokens in memory. Do not write refreshed access tokens back to settings files.

## Bedrock Credentials

Anthropic Bedrock endpoints use `endpoint_type: "anthropic_bedrock"` or `is_bedrock: true`, a region, and AWS credentials.

The adapter passes credentials to AWS SDK configuration. Do not log AWS credential values or generated signed request details.

## Test Fixtures

Committed fixtures should use placeholders such as `sk-...` or `YOUR_KEY`. Live tests should load real credentials from ignored local files or environment-selected files.

When adding a new live fixture shape:

- Add the shape to `dev_settings.example.json` or `sample_settings.json`.
- Keep real values out of git.
- Update `docs/TESTING.md`.
