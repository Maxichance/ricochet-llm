<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/logo-wordmark-dark.png">
  <img src="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/logo-wordmark.png" alt="ricochet-llm" width="440">
</picture>

**One call. Several free APIs. No quota ever runs dry.**

The LLM router that bounces from one free provider to the next: quota hit, model retired, outage, empty answer... your code never notices.

[![tests](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-1d3a8a.svg)](https://github.com/Maxichance/ricochet-llm/blob/main/pyproject.toml)
[![MIT license](https://img.shields.io/badge/license-MIT-1d3a8a.svg)](https://github.com/Maxichance/ricochet-llm/blob/main/LICENSE)
[![providers](https://img.shields.io/badge/providers-8%2B%20free-1d3a8a.svg)](#providers)

[Getting started](#getting-started) · [Features](#features) · [In your code](#in-your-code) · [Providers](#providers) · [Configuration](#configuration) · [Documentation](#documentation)

<br>

<img src="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/dashboard.png" alt="Ricochet dashboard: status of each provider, quotas, test console and event log" width="860">

<sub>The dashboard (demo data): each provider's quota, pauses, a console showing the live failover, and the event log.</sub>

</div>

<br>

## Why

Free LLM API tiers are generous, but each one has its limits: per-minute or per-day quotas, models that disappear, 429 errors, empty answers, latency spikes. Chaining them by hand makes for fragile code.

Ricochet puts all of it behind **a single interface**. It picks the provider and the model, tracks quotas, sets aside whatever stops answering and retries elsewhere, without ever losing the conversation.

```
 your code ──►  chat()  ·  generate()  ·  run_parallel()  ·  web_search()
                                   │
                          ┌────────▼────────┐
                          │    Ricochet     │   429 · 402 · 5xx · empty answer · low quota
                          │  picks · tracks │──────────────────────┐
                          │  fails over     │                      │ bounces: next model,
                          └────────┬────────┘                      │ then next provider
        ┌───────┬────────┬─────────┼──────────┬─────────┐          │
      Groq   NVIDIA   Ollama   Cloudflare  OpenRouter  Gemini ... ◄┘
```

## Features

| | |
|---|---|
| **Automatic failover** | Per provider **and per model**: a 429 only sets one model aside, and a provider is paused only once none of its models answers. Refusals are classified (quota, plan not enabled, daily cap, key to fix) with the right pause length. |
| **Quota tracking** | Token budget read from response headers, learned limits, no call wasted on a model that is about to refuse. |
| **Tool calling** | Describe a Python function with `@tool` and the JSON schema is inferred. Tool errors are sent back to the model instead of crashing the conversation. |
| **Parallel batch reading** | `run_parallel` gives each provider its own workers, with a concurrency cap and a delay between starts matched to its limits. |
| **Free web search** | Exa, Tavily and LangSearch behind one interface, quota tracked on disk, stop at 90% of the free allowance. A ready-made tool for your models. |
| **Imperfect JSON** | `read_json` recovers an answer written twice or containing a stray `null`, and never guesses at a truncated one. |
| **Personal data** | `find_personal_data` spots emails, phone numbers, IBANs, card numbers and social security numbers (with their check digits) before sending. |
| **Dashboard** | Quota, latency, pauses, event log and test console, local, with no browser-side dependency. |
| **No-code configuration** | `providers.json`: change a model, remove or add a provider (any OpenAI-compatible endpoint), change the order. |

## Getting started

```bash
pip install "ricochet-llm[server]"        # without the dashboard: pip install ricochet-llm
```

Add at least one [free key](#providers) to a `.env` file:

```bash
GROQ_API_KEY=...
NVIDIANIM_API_KEY=...        # one key is enough, several give more headroom
```

```bash
ricochet-llm serve --open                 # dashboard at http://127.0.0.1:8765
ricochet-llm chat "Hello!"                # one question, one answer (plus the provider that replied)
```

| Command | Effect |
|---|---|
| `ricochet-llm serve [--port N] [--open]` | web server and dashboard (default command) |
| `ricochet-llm status` | provider status in the terminal |
| `ricochet-llm chat "..."` | one question, one answer |
| `ricochet-llm models nvidia` | **which listed models actually answer** (a free provider's model list doesn't tell you) |
| `ricochet-llm search "..."` | web search through a free API |

The dashboard only listens on your machine. It has no authentication and its buttons consume quota: listening on another address is refused unless you pass `--allow-remote`.

From source: `git clone https://github.com/Maxichance/ricochet-llm.git && cd ricochet-llm && pip install -e ".[server]"`, then `cp .env.example .env`.

## In your code

### A conversation with tools

```python
from ricochet_llm import chat_detailed, tool

@tool
def weather(city: str) -> dict:
    """Get the weather for a city.

    Args:
        city: Name of the city
    """
    return {"city": city, "sky": "clear", "temperature": 18}

history = []                                       # one list per conversation
r = chat_detailed(history, "What's the weather in Nantes?")
print(r.answer)
print(r.provider, r.model, r.trail)                # who answered, and any failovers on the way
```

### A single call, then a parallel batch

```python
from ricochet_llm import generate, run_parallel

def read(document, provider):                      # runs in a thread
    r = generate(prompt(document), max_tokens=8192, timeout=60, prefer=provider)
    save(document, r.text, r.provider, r.model)    # as you go: an interruption loses nothing

summary = run_parallel(documents, read, limits={"nvidia": 10})
# {"processed": 812, "errors": 3, "remaining": 0, "per_provider": {...}}
```

`generate` never returns an empty answer, accepts a longer timeout than chat and fails over between providers and between models. `run_parallel` stops cleanly when no provider is left and hands back what remains.

### Search the web, read imperfect JSON, protect data

```python
from ricochet_llm import (generate, web_search, register_search_tool,
                          read_json, find_personal_data)

register_search_tool()                             # the `search_web` tool becomes callable by your models
pages = web_search("free llm router", n=5, content=True).results

data = read_json(generate(prompt, max_tokens=8192).text)    # a dict, or None if the answer is truncated
if find_personal_data(message):                    # email, phone, IBAN, card, social security number
    ...                                            # don't send it to a free tier
```

Checking the **content** of an answer remains up to you: `read_json` only repairs the shape.

## Providers

| Provider | Where to get a free key | Variable |
|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) (trial key) | `NVIDIANIM_API_KEY` |
| Ollama Cloud | [ollama.com/settings/keys](https://ollama.com/settings/keys) | `OLLAMACLOUD_API_KEY` |
| Cloudflare Workers AI | [dash.cloudflare.com](https://dash.cloudflare.com/profile/api-tokens) (token **and** account ID) | `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` |
| Gonka | [gonkabroker.com](https://gonkabroker.com) | `GONKA_API_KEY` |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` |
| Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) (check that the free plan is active) | `MISTRAL_API_KEY` |
| *Web search*: Exa | [dashboard.exa.ai](https://dashboard.exa.ai) | `EXA_API_KEY` |
| *Web search*: Tavily | [app.tavily.com](https://app.tavily.com) | `TAVILY_API_KEY` |
| *Web search*: LangSearch | [langsearch.com](https://langsearch.com) | `LANGSEARCH_API_KEY` |

A provider without a key is simply ignored. `NVIDIANIM_API_KEY_EXPIRES=YYYY-MM-DD` shows a trial key's expiry date in the dashboard, with a warning 30 days before.

<details>
<summary><b>See a provider's details in the dashboard</b></summary>
<br>
<img src="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/providers.png" alt="Expanded rows: models, quota gauges, pause with the last error and a resume button" width="820">
</details>

> **Free tiers change fast** (models retired, limits changed). Test a model with a real call before configuring it: `ricochet-llm models <provider>`.
>
> **Privacy.** Several free tiers log your prompts or use them to train their models. Don't send them personal or confidential data without checking their terms. NVIDIA NIM's free-tier license covers research, development and testing.

## Configuration

Copy `providers.example.json` to `providers.json` (or point to it with `RICOCHET_PROVIDERS`) to change a model, remove a provider, add one or change the order:

```json
{
  "providers": [
    { "name": "groq", "model_heavy": "openai/gpt-oss-120b" },
    { "name": "cerebras", "enabled": false },
    { "name": "myprovider", "env_key": "MYPROVIDER_API_KEY",
      "base_url": "https://api.example.com/v1", "model_heavy": "big-model" }
  ],
  "order": ["myprovider"]
}
```

The file is validated: a typo gives an explicit error. Without a file, `RICOCHET_ORDER=groq,gemini` changes the head of the list. All fields (fallback models, per-model options, token limits, concurrency cap...) are documented at the top of [`ricochet_llm/catalog.py`](https://github.com/Maxichance/ricochet-llm/blob/main/ricochet_llm/catalog.py).

Your tools and your system prompt can live in a plugin, loaded by the server and the command line: `RICOCHET_PLUGIN=examples/demo_tools.py`. A plugin is code run as is: only load code you trust.

## Documentation

- [Technical guide](https://github.com/Maxichance/ricochet-llm/blob/main/docs/GUIDE.md): routing in detail, how each error is handled, provider behaviors, best practices for batch reading.
- [Web search](https://github.com/Maxichance/ricochet-llm/blob/main/docs/WEB_SEARCH.md): providers kept and set aside, quotas, adding an engine.
- [Examples](https://github.com/Maxichance/ricochet-llm/tree/main/examples): a tool plugin, a library usage.
- [Changelog](https://github.com/Maxichance/ricochet-llm/blob/main/CHANGELOG.md).
- [Contributing](https://github.com/Maxichance/ricochet-llm/blob/main/CONTRIBUTING.md): propose providers, models or any other change.

## Development

```bash
pip install -e ".[dev]" ruff
ruff check .
python -m unittest discover -s tests -v          # no network, no quota consumed
```

The tests use fake providers: they consume nothing.

## Disclaimers

- **No affiliation.** Ricochet is an independent project. It is not affiliated with, endorsed or sponsored by Groq, NVIDIA, Ollama, Cloudflare, OpenRouter, Gonka, Google, Mistral, Cerebras, Exa, Tavily, LangSearch or any other provider mentioned. Their names and trademarks belong to their owners and are used only to indicate compatibility.
- **Provider terms.** You alone are responsible for complying with the terms of use and licenses of the providers whose keys you use (NVIDIA NIM's trial plan, for example, is limited to research, development and testing). Ricochet uses one key per provider and is not designed to get around their limits: don't use it to multiply accounts or keys, or to resell or share access.
- **Offers change.** Models, quotas, prices and terms of free tiers change without notice. The information in this repository may be out of date, and no availability is guaranteed.
- **Your data.** What you send goes to third parties, sometimes outside the European Union, sometimes logged or reused. You remain responsible for the data you transmit, including under the GDPR. The `find_personal_data` filter is a limited aid, not a guarantee.
- **Generated content and search results.** A model's answers may be inaccurate. Pages returned by web search belong to their authors: you are responsible for how you use them.
- **No warranty.** The software is provided "as is", without warranty of any kind, and its author is not liable for any damage arising from its use, to the extent permitted by law (see the MIT license). This repository is not legal advice.

## Security and license

Report a vulnerability privately: see [SECURITY.md](https://github.com/Maxichance/ricochet-llm/blob/main/SECURITY.md). Code released under the [MIT](https://github.com/Maxichance/ricochet-llm/blob/main/LICENSE) license.
