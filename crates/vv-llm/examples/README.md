# vv-llm-rs Examples

Copy `llm_settings.example.json`, replace the placeholder API key, then run
examples from the repository root:

```powershell
cargo run -p vv-llm --example typed_thinking -- 'C:\path\to\llm_settings.json'
cargo run -p vv-llm --example streaming -- 'C:\path\to\llm_settings.json'
cargo run -p vv-llm --example middleware_metadata -- 'C:\path\to\llm_settings.json'
```

`registry_fallback` is deterministic and does not call a real API:

```powershell
cargo run -p vv-llm --example registry_fallback
```

`streaming` uses `MiddlewareChatClient::create_stream`. Retry or fallback is
allowed only before the first visible chunk. Once output begins, a later error
is propagated without replaying the request.
