# Multi-provider "free tier" LLM router: technical guide

> For anyone contributing to the project or integrating it. Getting started and usage: see the [README](../README.md).

## 1. Principle

An agentic loop (chat + tool calling) that depends on no single LLM vendor. It relies on the OpenAI-compatible API exposed by several free providers (a single SDK: `openai`, pointed at each `base_url`) and fails over from one to the next.

1. `catalog.py` lists the providers in order of preference (overridable by `providers.json`), each with a **heavy** model, a **light** model and, if needed, **fallback models** (`extra_models`) from the same provider.
2. Before each call, `_choose_provider_and_model()` selects:
   - **HEAVY**: the first available provider (not in cooldown) whose quota is unknown or ≥ `quota_min`, favoring those that handle tools; within that provider, the **first usable model** (heavy, then `extra_models`);
   - **LIGHT** (all below the threshold): the one with the most quota left, on its light model.
   A model set aside does **not** sideline the whole provider: state (pause, learned token limit) is kept **per model**. The provider is paused only when none of its models can answer, until the first one comes back.
3. Every call error is **classified** (`_classify_error`); the conversation is never lost:

   | Error | Reaction |
   |---|---|
   | `429` | the **model** is set aside (`Retry-After`, 60 s by default); another model of the provider is tried right away; with no other usable model, the provider is paused until the nearest model comes back, quota at 0 |
   | `413` "Limit N" | the limit N is **learned** for this model, which is set aside for 30 min (larger requests are no longer sent to it); same rules as 429 |
   | **empty** answer | model set aside for 5 min, next model (a reasoning model spent all its tokens before writing) |
   | network / 5xx / timeout | 1 retry with backoff, then 30 s cooldown |
   | `404` | model removed from the catalog: set aside for 15 min if there are others, otherwise provider cooldown for 15 min |
   | `401` / `403` | provider cooldown for 15 min (key to fix) |
   | `400` / `410` / `422` | failover for this call only, no cooldown |
   | other exception (local bug) | raised as is |

   A provider set aside during a call is never retried within that same call: the loop always terminates (otherwise `LLMRouterError` with each provider's last error). On failure, the history is restored to its original state.
4. Tool calls are executed locally (`tools.py` registry) and fed back until `finish_reason == "stop"` (8 rounds max). An unknown tool, invalid arguments or a tool exception are returned **to the model** as the result. During a tool exchange, the router never switches to a provider without tool calling.

## 2. Architecture

- `catalog.py` (data) → `router.py` (engine and state) → `server.py` / `__main__.py` (entry points) → `web/` (interface). `parallel.py` spreads batch jobs across providers (section 6).
- Three entry points into the engine: `chat` (conversation with tools), `generate` (a single call: extraction, classification, document reading) and `probe_models` (diagnostics: which listed models actually answer).
- Observable state per provider, kept in `router.py` under a lock: remaining quota, cooldown, last error, latest `x-ratelimit-*` headers, quota probe, statistics (ok, errors, tokens, latency), call window, plus an event log. `quota_status()` returns it, enriched (gauges, key expiry); the interface polls it through `GET /api/status` every 3 s.
- **Quota gauges**: each `x-ratelimit-remaining-X` / `x-ratelimit-limit-X` pair becomes a gauge (`quota_labels` provides the labels). OpenRouter returns nothing in headers: a probe (`probe`) reads `GET /key` (`free_model_daily_requests`, `expires_at`), at most once a minute.
- **Key expiry**: `<KEY_NAME>_EXPIRES=YYYY-MM-DD` in `.env`, or `expires_at` supplied by the probe.
- **Security**: the server listens on `127.0.0.1` by default and has no authentication (its actions consume quotas). Don't expose it without adding authentication.

## 3. Adding or changing a provider

Without touching the code: `providers.json` (see `providers.example.json` and the docstring of `catalog.py`) overrides a provider, removes one, adds one, changes the order. The file is validated on load (explicit `ConfigError`). For a provider shipped with the package, add an entry to `BUILTIN_SPECS`. **Before choosing a model, test it with a real call** (see pitfalls). With no quota header, leave `quota_header` at `null`: only 429s will then trigger the cooldown. Run `python -m ricochet_llm models <provider>`: **a free provider's list doesn't tell you what answers**. For a reasoning model, set `model_options` (turn reasoning off); for a provider with a per-model budget, `token_limits`; for a rate-limited provider, `max_concurrency` and `min_interval_s`. For a dedicated probe, add a function to `_PROBES` (`router.py`).

## 4. Integrating into another project

`pip install -e .` then `from ricochet_llm import chat, tool, set_system_prompt`. Declare your tools with `@tool` (schema inferred from the signature and docstring). The `openai` client is synchronous: in an async application, call `chat` through `run_in_threadpool` / `run_in_executor` (like `server.py`). Server sessions are in memory (enough for a single-instance internal tool; with several instances, plan for Redis/Firestore). Don't send two messages in parallel on the same session (the server returns 409). With no registered tool, no `tools` parameter is sent to providers.

## 5. Provider behaviors to know

Free tiers change fast (models retired, limits changed): **test a model with a real call before configuring it** (`python -m ricochet_llm models <provider>`), a provider's `/models` list is not reliable.

**Refusals and pauses**
- **Which refusal means what**: a **402** or a per-minute limit of zero (`x-ratelimit-limit-req-minute = 0`) = plan not enabled, one-hour pause. A **429 on a daily cap** (Gemini `GenerateRequestsPerDayPerProjectPerModel`, Cloudflare "daily free allocation", OpenRouter `free-models-per-day`) doesn't come back for hours: retrying it every minute would only waste a call. A **429 without `Retry-After`** but with `x-ratelimit-reset-tokens` ("46.5s", "1m30s"): wait for the announced reset.
- **The advertised token budget** (`x-ratelimit-remaining-tokens` / `-reset-tokens`) is kept per model after each successful response: a model that can't handle the request is not called. `x-ratelimit-limit-tokens` sets the largest acceptable request, without waiting for a 413.
- **Time-limited trial keys** (NVIDIA, OpenRouter): past the date, 401/403 → 15-minute pause in a loop, with no automatic repair. `<KEY_NAME>_EXPIRES=YYYY-MM-DD` shows the expiry.
- **Quotas with different windows** (Groq: tokens/min and requests/day; Cerebras: tokens/day): the catalog's `quota_min` values are calibrated on these windows, to be revisited if you change model. A quota read more than 5 min ago goes back to "unknown" and the provider is probed again.
- **The SDK must not retry on its own**: clients are created with `max_retries=0`, otherwise it would sleep on a 429 before the router fails over.

**Model answers**
- **Reasoning models** (gpt-oss, Nemotron...): `max_tokens` includes reasoning tokens; an empty answer triggers a failover. For NVIDIA's Nemotron, `model_options` turns reasoning off (`enable_thinking: false`); a longer timeout changes nothing there.
- **`<think>...</think>` in `content`**: never returned as the answer; an answer made only of reasoning counts as empty (failover).
- **Small models**: they sometimes return a tool call's JSON as plain text. `cleaning.strip_tool_artifacts` removes **only** JSON recognized as tool calls (known name, or `{"type": "function", ...}`) and `<function=...>` / `<tool_call>` tags; legitimate code and JSON are kept. If the cleanup empties the answer, the router fails over and retries once with "Answer my previous question."
- **Gemini and tools**: Gemini 3.x requires a `thought_signature` field on the second round of a tool exchange, which the router doesn't replay: `supports_tools: False`. Fallback chain `gemini-3.6-flash`, `3.5-flash`, `3.1-pro-preview` (the most capable, but its free daily quota is very short).
- **Mistral**: a valid key may have no chat rights until the free plan is activated; the router then pauses Mistral for an hour and uses the others.
- **OpenRouter**: a 429 "temporarily rate-limited upstream" often comes from the free model's upstream provider, not from your quota.
- **Cerebras**: no permanent free tier any more according to recent sources: card "disabled" without a key.

**Default order**: Groq, NVIDIA, Ollama, Cloudflare, OpenRouter, Gonka (6 requests/min and 1M tokens per month: last of the free ones), Gemini, Mistral. `RICOCHET_ORDER=groq,gemini` changes the head of the list without a file.

**Privacy**: several free tiers log prompts or use them to train their models. Don't send them personal data without checking their terms (`ricochet_llm.privacy` spots the most obvious).

## 5 bis. Best practices for batch reading

- **Valid but empty JSON is not "no result".** A model that is too weak returns well-formed JSON without extracting anything. The router can't see it (the answer isn't empty): **validate the content on the caller's side** (a quote found in the source, for example) and compare a model to a reference on a few inputs BEFORE enabling it.
- **Small models read less.** Good as a last fallback, not as a first choice.
- **A rate limit is not a concurrency limit.** "40 requests per minute" applies to the START rate: set `max_concurrency` and `min_interval_s` to stay under the limit rather than capping yourself at 1 or 2 workers. A burst goes through, a sustained flow hits the limit.
- **A token budget per minute PER MODEL** (Groq): a 429 or a 413 on one model says nothing about the others, hence the per-model state.
- **The chat timeout isn't a reading timeout.** The catalog timeout is calibrated for chat (fail over fast), not for a long JSON answer. `generate(timeout=60)` lengthens the timeout for this call and never shortens it.
- **Count by the provider that ANSWERED**, not the one you preferred: read `GenerateResult.provider` and `.model`. Durations measured around a call include the failovers.
- **Compare models on the same inputs**: the gap between different samples is often larger than the gap between models.
- **Quotas and licenses**: NVIDIA returns no quota header and its free tier's license is "research, development and testing"; Gonka caps requests per minute and tokens per month; Cloudflare caps per day. Don't let a batch job use up what chat needs as a fallback: run it at night, or exclude those providers (`only=`).
- **Keys pasted into a `.env`**: a value that is far too long (token pasted several times, comment, quotes) gives 401 or 400 "Invalid format for Authorization header". Check the length and characters without printing the key. A misnamed variable is accepted through `env_aliases`.
- **Windows: a click in the console can freeze everything.** "Quick edit" mode puts the console in selection mode: the first `print` stays blocked until you press Enter, and a program that writes to the console, even in a thread, can freeze all its calls in flight. Turn off quick edit, or use Windows Terminal.
- **Save as you go**: write each item's result as soon as it is read (atomic write: temporary file then rename). An interruption then loses only the requests in flight.
- **Tests: `load_dotenv()` modifies `os.environ` for the whole suite.** A test that imports a module that loads the real `.env` injects real keys into the following tests. Neutralize it (`mock.patch("dotenv.load_dotenv")`) before the import.

## 6. Batch reading: `generate` and `run_parallel`

```python
from ricochet_llm import generate, run_parallel

def read(document, provider):                      # runs in a thread
    r = generate(prompt(document), system=INSTRUCTIONS, max_tokens=8192, timeout=60, prefer=provider)
    if r.truncated:                                # stopped on max_tokens: the JSON may be cut
        raise ValueError("truncated answer")
    save(document, r)                              # as you go; record r.provider and r.model

summary = run_parallel(documents, read, limits={"nvidia": 10})
# {"processed": ..., "errors": ..., "remaining": ..., "interrupted": False, "per_provider": {"groq": ..., ...}}
```

- `generate` fails over between providers **and** between models like `chat`, without tools. It **never** returns an empty answer (the model is set aside, the next one tried). `timeout` lengthens a call's timeout; `only` restricts the providers; `prefer` names one to try first.
- `run_parallel` gives each provider **its own** workers (the catalog's `max_concurrency`, or `limits`), all drawing from the same queue (order is kept), with a minimum gap between starts (`min_interval_s`). A provider in pause takes no work; with no provider left available, or `Interrupted` raised by `work`: clean stop, the remaining items are returned (`remaining`). An error on one item is counted without stopping the others.
- **Choose the batch's providers** (`only=`): a provider with a short monthly quota (Gonka) or a daily cap (Cloudflare) is a chat fallback, not a catch-up engine.
- Before launching, a slice of ~100 items, then a quality check **per provider and per model** (items per document, share of quotes found, unreadable answers).

## 6 bis. Imperfect JSON answers: `read_json`

Some models write their answer twice (the complete object, a quote mark, a second copy), follow it with an extra field (`...}}"products":[]}`) or slip a `null` between two key/value pairs. `read_json(text)` reads strictly first (unchanged for a valid answer), then only on failure keeps the first complete object. A truncated answer (`max_tokens`) or one that loops stays `None`: read it again. It says nothing about the **content**: valid JSON can contain made-up values, and the substantive check (is the quote in the source?) remains up to you.

## 7. Checklist

1. `pip install -e ".[server]"`, create `.env` from `.env.example`, fill in at least one key.
2. Run `python -m ricochet_llm serve --open`, click **Test** on each card: valid key and decent latency?
3. Declare your tools (`@tool`) in a plugin (`RICOCHET_PLUGIN`) or in your code.
4. `python -m unittest discover -s tests -v` after any change to the engine.
5. Before any batch job: `python -m ricochet_llm models <provider>` (models that can really be called), a slice of ~100 items, a summary per provider AND per model.
6. Before sending text typed by a user to a free tier: `find_personal_data(text)`. For web search: `python -m ricochet_llm search "test"` with a real key (the adapters have only been tested against simulated responses).
7. Test the failover for real: set a very high `quota_min` on the first provider (`catalog.py`) and check in the log that the next one takes over.
