# Changelog

## 0.5.0
The package is named `ricochet-llm`: the module is imported as `ricochet_llm` and the command is `ricochet-llm`. The dashboard becomes an optional dependency (`ricochet-llm[server]`).
Lessons from real-provider usage since 0.4.0.
- **Web search** (`ricochet_llm.search`, `web_search`, `register_search_tool`, `python -m ricochet_llm search`, `GET /api/search/status`): Exa, Tavily and LangSearch behind one interface, failover on refusal, quota tracked on disk with a stop at 90% of the free allowance, `search_web` tool for models. `docs/WEB_SEARCH.md`: the providers kept and those set aside, with the reason. Adapters tested against simulated responses only.
- **Better-classified refusals**: 402 and a per-minute limit of zero (plan not enabled) = one-hour pause; 429 on a daily cap = one-hour pause instead of one minute; `x-ratelimit-reset-tokens` read when there is no `Retry-After`.
- **Per-model token budget** read from the headers of successful responses: a model that can't handle the request is not called; the advertised limit is learned without waiting for a 413.
- **`read_json`**: tolerant reading of a JSON answer (written twice, extra field, stray `null`) that never guesses at a truncated answer.
- **`find_personal_data` / `check_conversation`**: spots emails, phone numbers, IBANs, card numbers and social security numbers (check digits) before sending to a free tier.
- `<think>...</think>` blocks are no longer returned as the answer. Gemini: chain `gemini-3.6-flash`, `3.5-flash`, `3.1-pro-preview`. Gonka now comes after OpenRouter. `RICOCHET_ORDER` sets the head of the list without a file. `httpx` becomes a direct dependency.
- **OpenAI-compatible endpoint**: `/v1/chat/completions` (with `tools`, `stream`) and `/v1/models` in the server, `complete()` in the library, optional `RICOCHET_API_KEY`. The failover loop of `generate` is now shared with `complete`.
- **Async**: `achat`, `achat_detailed`, `agenerate`, `acomplete`.
- **Persistent state** (opt-in, `RICOCHET_STATE`): provider pauses, model pauses and learned token limits survive a restart.
- `py.typed`, issue templates, Dependabot, CI on Linux, Windows and macOS.
- 235 tests (no network).
- Out of scope: the native Gemini client (tool calls and live streaming, `thoughtSignature`), which assumes an async loop and a Gemini-specific format; Gemini is still called in OpenAI format, without tools.

## 0.4.0
- **Batch reading**: `generate()` (single call: failover between providers and between models, never an empty answer, adjustable timeout that only ever grows, `only` / `prefer`) and `run_parallel()` (workers per provider, concurrency cap, delay between starts, provider in pause gets no work, clean stop).
- **Per-model state**: a 429, a 413 or an empty answer only sets one model aside; the provider is paused only when none of its models answers. A 413 teaches the model's limit.
- **Catalog**: `extra_models`, `model_options` (turn off Nemotron reasoning: `enable_thinking: false`), `token_limits`, `max_concurrency`, `min_interval_s`, `env_required`, `env_aliases`, templated address `{VARIABLE}`.
- **Providers**: NVIDIA updated (ultra-550b, super-vl-120b, bolt-e4b), Groq (3 models, per-model budget), Ollama, **Cloudflare Workers AI** and **Gonka** added.
- `python -m ricochet_llm models <provider>`: which listed models actually answer. `docs/GUIDE.md`: batch-reading pitfalls.
- 149 tests (no network).

## 0.3.0
- Providers configurable without touching the code (`providers.json`, validated); `@tool` registry; plugins (`RICOCHET_PLUGIN`); system prompt via `set_system_prompt`.
- New interface: beige background, one row per provider (quota bar + KPIs), details on click.
- The server refuses to listen outside the local machine without `--allow-remote`.
- Removed the sample business code (`rechercher_client`); added `examples/`, CI, lint (ruff), 88+ tests.

## 0.2.0
- Python package, web dashboard, NVIDIA NIM / Ollama Cloud / OpenRouter providers, statistics and event log.

## 0.1.0
- Multi-provider router with quota failover and tool calling (single file).
