# vv-llm-rs Examples

The network examples use a settings JSON file. Copy
llm_settings.example.json, replace the placeholder key, and select the
backend and model explicitly:

```powershell
$env:VV_LLM_SETTINGS_JSON = 'C:\path\to\llm_settings.json'
$env:VV_LLM_BACKEND = 'deepseek'
$env:VV_LLM_MODEL = 'deepseek-v4-flash'
```

The settings path may also be passed as the first argument. The shared loader
accepts either form and does not select a provider or model implicitly.

```powershell
cargo run -p vv-llm --example basic_chat
cargo run -p vv-llm --example streaming
cargo run -p vv-llm --example tools
cargo run -p vv-llm --example multimodal
cargo run -p vv-llm --example typed_thinking
cargo run -p vv-llm --example middleware_metadata
```

Set VV_LLM_IMAGE_URL for multimodal; data URLs are useful when the provider
requires inline image bytes.

`contract_json` is a deterministic codec example. It covers object
`tool_choice`, canonical `x_*` extensions, and options:

```powershell
cargo run -p vv-llm --example contract_json
```

`registry_fallback` is also deterministic and does not call a real API:

```powershell
cargo run -p vv-llm --example registry_fallback
```

streaming calls ChatClient::create_stream explicitly. Retry or fallback is
allowed only before the first visible chunk; once output begins, a later error
is propagated without replaying the request.
