"""
Provider catalog: data + loading/validation. Order = decreasing preference.
A provider with no key in the environment is ignored (shown as "disabled").

Customize WITHOUT touching the code: a `providers.json` file (current directory, or path in
$RICOCHET_PROVIDERS), see providers.example.json:
  { "providers": [
      {"name": "groq", "model_heavy": "other-model"},                          # modifies an existing provider
      {"name": "cerebras", "enabled": false},                                 # removes a provider
      {"name": "myprovider", "env_key": "MYPROVIDER_API_KEY",
       "base_url": "https://api.example.com/v1", "model_heavy": "model"}      # adds one
    ],
    "order": ["myprovider", "groq"] }                                         # optional: providers to move to the front
  (or, without a file: RICOCHET_ORDER=myprovider,groq)

Provider fields:
  name / label      technical identifier / display name
  env_key           environment variable holding the API key
                    (optional: <env_key>_EXPIRES=YYYY-MM-DD to show the key's expiry date)
  base_url          OpenAI-compatible endpoint
  model_heavy/light "heavy" model (HEAVY mode) and "light" model (LIGHT mode, when every quota is low)
  quota_header      response header giving the remaining quota that drives routing (None if absent)
  quota_min         threshold below which the provider is skipped in HEAVY mode
  supports_tools    False if the endpoint does not handle multi-turn tool calling
  timeout           HTTP timeout in seconds (short, to fail over quickly if the provider hangs)
  extra_models      fallback models of the same provider, tried in order after model_heavy when a model is
                    set aside (429, 413, 404, empty response). A model set aside does NOT pause the whole provider.
                    model_light is only part of it if listed here: it serves LIGHT mode (low quota).
  model_options     per-model settings: {"nvidia/nemotron-3-ultra-550b-a55b":
                    {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}}.
                    Needed to turn off reasoning on models that "think" (see GUIDE.md).
  token_limits      largest request accepted per model, when the provider announces it (Groq: 8,000 tokens/min
                    per model). Avoids calling a model that will refuse (413). Completed by learning.
  max_concurrency   maximum simultaneous requests for this provider (batch reading, see parallel.py)
  min_interval_s    minimum seconds between two request starts at this provider (6 req/min = 10 s)
  env_required      ADDITIONAL environment variables required (e.g. the Cloudflare account ID)
  env_aliases       alternative names accepted for a variable: {"CLOUDFLARE_API_KEY": ["CLOUDFLARE_API_TOKEN"]}
                    `base_url` may contain {VARIABLE_NAME}, replaced by its environment value.
  quota_labels      readable labels for the x-ratelimit-remaining-<suffix> headers
  probe             dedicated probe to read the quota outside the headers (see router.py)
  console_url       where to get a free key
  note              quirk to display
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

REQUEST_TIMEOUT_S = 60
CONFIG_FILENAME = "providers.json"
CONFIG_ENV_VAR = "RICOCHET_PROVIDERS"
ORDER_ENV_VAR = "RICOCHET_ORDER"          # "groq,gemini": providers to move to the front, without a file (unknown names ignored)


class ConfigError(ValueError):
    """Invalid provider configuration."""


BUILTIN_SPECS = [
    {
        "name":           "groq",
        "label":          "Groq",
        "env_key":        "GROQ_API_KEY",
        "base_url":       "https://api.groq.com/openai/v1",
        "model_heavy":    "openai/gpt-oss-120b",
        "model_light":    "openai/gpt-oss-20b",
        # One token-per-minute budget PER MODEL: several models multiply the calls before the first 429,
        # and a 429 on one says nothing about the others.
        "extra_models":   ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "token_limits":   {"openai/gpt-oss-120b": 8000, "qwen/qwen3.8-27b": 7000, "openai/gpt-oss-20b": 8000},
        "quota_header":   "x-ratelimit-remaining-tokens",
        "quota_min":      2_000,
        "timeout":        30,
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "quota_labels":   {"tokens": "tokens / min", "requests": "requests / day"},
        "console_url":    "https://console.groq.com/keys",
    },
    {
        "name":           "nvidia",
        "label":          "NVIDIA NIM",
        "env_key":        "NVIDIANIM_API_KEY",
        "base_url":       "https://integrate.api.nvidia.com/v1",
        # The catalog lists many more models than the ones that answer with a trial key (`python -m ricochet_llm
        # models nvidia` gives the real list). The Nemotron models REASON: without an explicit setting they spend all of
        # `max_tokens` in `reasoning_content` and return an empty `content`. `enable_thinking: false` makes them answer
        # directly. Order: the most capable first, the small model as the last fallback.
        "model_heavy":    "nvidia/nemotron-3-ultra-550b-a55b",
        "extra_models":   ["nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-bolt-e4b"],
        "model_light":    "nvidia/nemotron-3.5-bolt-e4b",         # 4B, does not reason: no setting needed
        "model_options": {
            "nvidia/nemotron-3-ultra-550b-a55b": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "nvidia/nemotron-3.5-super-vl-120b-a12b": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        },
        "timeout":        20,     # chat: fail over quickly. A long read passes `timeout=60` to generate()
        # A requests-PER-MINUTE limit applies to the start rate, not to simultaneity. A burst gets through, a sustained
        # flow hits the limit: so we set both a worker cap AND a gap between starts (2 s = 30/min at most). Beyond that,
        # harmless 429s: the router switches to the next model.
        "max_concurrency": 6,
        "min_interval_s": 2.0,
        "console_url":    "https://build.nvidia.com",
        "note":           "Trial: development and testing only (license), no limit announced in headers",
    },
    {
        "name":           "ollama",
        "label":          "Ollama Cloud",
        "env_key":        "OLLAMACLOUD_API_KEY",
        "base_url":       "https://ollama.com/v1",
        "model_heavy":    "gpt-oss:120b",
        "model_light":    "gpt-oss:20b",
        "extra_models":   ["gpt-oss:20b", "nemotron-3-nano:30b"],   # free tier: the other models answer 402 (Pro plan)
        "timeout":        45,
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "console_url":    "https://ollama.com/settings/keys",
        "note":           "Unpublished limits (5 h session / weekly)",
    },
    {
        "name":           "cloudflare",
        "label":          "Cloudflare Workers AI",
        "env_key":        "CLOUDFLARE_API_KEY",
        "env_required":   ["CLOUDFLARE_ACCOUNT_ID"],
        "env_aliases":    {"CLOUDFLARE_API_KEY": ["CLOUDFLARE_API_TOKEN"],
                           "CLOUDFLARE_ACCOUNT_ID": ["CLOUDFLAIRE_ACCOUNT_ID", "CF_ACCOUNT_ID"]},
        "base_url":       "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        # Models kept: those on the free tier that call the expected tool. Models on a paid plan answer
        # 403; some small models answer in text instead of calling a tool.
        "model_heavy":    "@cf/openai/gpt-oss-120b",
        "extra_models":   ["@cf/nvidia/nemotron-3-120b-a12b", "@cf/google/gemma-4-26b-a4b-it", "@cf/zai-org/glm-4.7-flash"],
        "model_light":    "@cf/zai-org/glm-4.7-flash",
        "timeout":        45,     # slow: allow a generous timeout for a long read
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "console_url":    "https://dash.cloudflare.com/profile/api-tokens",
        "note":           "Daily cap (neurons); \"Workers AI\" token ~53 characters",
    },
    {
        "name":           "openrouter",
        "label":          "OpenRouter",
        "env_key":        "OPENROUTER_API_KEY",
        "base_url":       "https://openrouter.ai/api/v1",
        "model_heavy":    "nvidia/nemotron-3-ultra-550b-a55b:free",
        "model_light":    "google/gemma-4-26b-a4b-it:free",
        "timeout":        45,
        "probe":          "openrouter",
        "console_url":    "https://openrouter.ai/keys",
        "note":           "50 free requests per day",
    },
    {
        "name":           "gonka",
        "label":          "Gonka Broker",
        "env_key":        "GONKA_API_KEY",
        "base_url":       "https://proxy.gonkabroker.com/v1",
        # From fastest to slowest.
        "model_heavy":    "deepseek-ai/DeepSeek-V4-Flash-0731",
        "extra_models":   ["zai-org/GLM-5.3-Flash", "MiniMaxAI/MiniMax-M2.7"],
        "model_light":    "MiniMaxAI/MiniMax-M2.7",
        "timeout":        45,
        # Free tier: 6 requests per minute and 1,000,000 tokens per MONTH: keep it for chat, not for batch
        # reading.
        "max_concurrency": 1,
        "min_interval_s": 11.0,
        "console_url":    "https://gonkabroker.com",
        "note":           "6 requests/min, 1 M tokens/month",
    },
    {
        "name":           "gemini",
        "label":          "Google Gemini",
        "env_key":        "GEMINI_API_KEY",
        "base_url":       "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Fallback chain: a model that times out or whose daily quota is exhausted is set aside, the next one takes over
        # (`gemini-3.1-pro-preview` is the most capable but its free daily quota is very short).
        "model_heavy":    "gemini-3.6-flash",
        "extra_models":   ["gemini-3.5-flash", "gemini-3.1-pro-preview"],
        "model_light":    "gemini-flash-lite-latest",
        "supports_tools": False,
        "timeout":        45,
        "console_url":    "https://aistudio.google.com/apikey",
        "note":           "No tool calling (thought_signature); 20 requests/day/model on the free tier",
    },
    {
        "name":           "mistral",
        "label":          "Mistral",
        "env_key":        "MISTRAL_API_KEY",
        "base_url":       "https://api.mistral.ai/v1",
        "model_heavy":    "mistral-small-latest",
        "quota_header":   "x-ratelimit-remaining-tokens-minute",
        "quota_min":      500,
        "timeout":        30,
        "quota_labels":   {"tokens-minute": "tokens / min", "req-minute": "requests / min"},
        "console_url":    "https://console.mistral.ai/api-keys",
        "note":           "Free tier to be verified: limit is zero without an activated plan, paused 1 h after the first refusal",
    },
    {
        "name":           "cerebras",
        "label":          "Cerebras",
        "env_key":        "CEREBRAS_API_KEY",
        "base_url":       "https://api.cerebras.ai/v1",
        "model_heavy":    "llama3.1-8b",
        "quota_header":   "x-ratelimit-remaining-tokens-day",
        "quota_min":      5_000,
        "timeout":        30,
        "quota_labels":   {"tokens-day": "tokens / day", "requests-day": "requests / day", "req-minute": "requests / min"},
        "console_url":    "https://cloud.cerebras.ai",
        "note":           "No permanent free tier any more (trial with a card)",
    },
]

# ───────── validation ─────────

_DEFAULTS = {
    "quota_header": None, "quota_min": 1, "supports_tools": True, "timeout": REQUEST_TIMEOUT_S,
    "quota_labels": None, "probe": None, "console_url": None, "note": None,
    "extra_models": None, "model_options": None, "token_limits": None,
    "max_concurrency": 2, "min_interval_s": 0.0, "env_required": None, "env_aliases": None,
}
_MODEL_OPTION_KEYS = {"extra_body"}
_REQUIRED = ("name", "env_key", "base_url", "model_heavy")
_ALLOWED = set(_REQUIRED) | set(_DEFAULTS) | {"label", "model_light"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalize(spec: dict, where: str) -> dict:
    """Validates a provider and fills in the defaults. Raises ConfigError with an actionable message."""
    unknown = sorted(set(spec) - _ALLOWED)
    if unknown:
        raise ConfigError(f"{where}: unknown field(s) {unknown}. Allowed fields: {sorted(_ALLOWED | {'enabled'})}")
    missing = [k for k in _REQUIRED if not spec.get(k)]
    if missing:
        raise ConfigError(f"{where}: required field(s) missing: {missing}")
    out = {**_DEFAULTS, **spec}
    for key in ("name", "env_key", "base_url", "model_heavy"):
        if not isinstance(out[key], str):
            raise ConfigError(f"{where}: \"{key}\" must be a string")
    if not _NAME_RE.match(out["name"]):
        raise ConfigError(f"{where}: \"name\" must be lowercase (a-z, 0-9, - and _): {out['name']!r}")
    if not out["base_url"].startswith(("http://", "https://")):
        raise ConfigError(f"{where}: \"base_url\" must start with http:// or https://")
    out["label"] = out.get("label") or out["name"]
    out["model_light"] = out.get("model_light") or out["model_heavy"]
    if not isinstance(out["quota_min"], int) or isinstance(out["quota_min"], bool) or out["quota_min"] < 0:
        raise ConfigError(f"{where}: \"quota_min\" must be an integer ≥ 0")
    if not isinstance(out["supports_tools"], bool):
        raise ConfigError(f"{where}: \"supports_tools\" must be true or false")
    if isinstance(out["timeout"], bool) or not isinstance(out["timeout"], (int, float)) or out["timeout"] <= 0:
        raise ConfigError(f"{where}: \"timeout\" must be a number of seconds > 0")
    if out["quota_header"] is not None and not isinstance(out["quota_header"], str):
        raise ConfigError(f"{where}: \"quota_header\" must be a string or null")
    labels = out["quota_labels"]
    if labels is not None and not (isinstance(labels, dict) and all(isinstance(v, str) for v in labels.values())):
        raise ConfigError(f"{where}: \"quota_labels\" must be an object {{suffix: label}}")
    out["quota_labels"] = dict(labels or {})

    extra = out["extra_models"]
    if extra is not None and not (isinstance(extra, list) and all(isinstance(m, str) and m for m in extra)):
        raise ConfigError(f"{where}: \"extra_models\" must be a list of model names")
    out["extra_models"] = list(extra or [])

    options = out["model_options"]
    if options is not None:
        if not (isinstance(options, dict) and all(isinstance(m, str) and isinstance(o, dict) for m, o in options.items())):
            raise ConfigError(f"{where}: \"model_options\" must be an object {{model: {{settings}}}}")
        for model, opt in options.items():
            unknown_opt = sorted(set(opt) - _MODEL_OPTION_KEYS)
            if unknown_opt:
                raise ConfigError(f"{where}: \"model_options\" → {model}: unknown setting(s) {unknown_opt} "
                                  f"(allowed: {sorted(_MODEL_OPTION_KEYS)})")
            if "extra_body" in opt and not isinstance(opt["extra_body"], dict):
                raise ConfigError(f"{where}: \"model_options\" → {model} → extra_body must be an object")
    out["model_options"] = {m: dict(o) for m, o in (options or {}).items()}

    limits = out["token_limits"]
    if limits is not None and not (isinstance(limits, dict) and all(
            isinstance(m, str) and isinstance(n, int) and not isinstance(n, bool) and n > 0 for m, n in limits.items())):
        raise ConfigError(f"{where}: \"token_limits\" must be an object {{model: integer > 0}}")
    out["token_limits"] = dict(limits or {})

    mc = out["max_concurrency"]
    if isinstance(mc, bool) or not isinstance(mc, int) or not 1 <= mc <= 30:
        raise ConfigError(f"{where}: \"max_concurrency\" must be an integer between 1 and 30")
    mi = out["min_interval_s"]
    if isinstance(mi, bool) or not isinstance(mi, (int, float)) or mi < 0:
        raise ConfigError(f"{where}: \"min_interval_s\" must be a number of seconds ≥ 0")

    required = out["env_required"]
    if required is not None and not (isinstance(required, list) and all(isinstance(v, str) and v for v in required)):
        raise ConfigError(f"{where}: \"env_required\" must be a list of variable names")
    out["env_required"] = list(required or [])
    aliases = out["env_aliases"]
    if aliases is not None and not (isinstance(aliases, dict) and all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(a, str) for a in v) for k, v in aliases.items())):
        raise ConfigError(f"{where}: \"env_aliases\" must be an object {{variable: [alternative names]}}")
    out["env_aliases"] = {k: list(v) for k, v in (aliases or {}).items()}
    return out


def _config_path(path: Optional[str]) -> Optional[Path]:
    explicit = path or os.getenv(CONFIG_ENV_VAR)
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ConfigError(f"Providers file not found: {p}")
        return p
    default = Path.cwd() / CONFIG_FILENAME
    return default if default.is_file() else None


def _env_first(order: List[str]) -> List[str]:
    """`$RICOCHET_ORDER` moves its providers to the front (handy in deployment: change the order without touching the file).
    An unknown name is ignored: a typo must not stop the router from starting."""
    wanted = [n.strip().lower() for n in (os.getenv(ORDER_ENV_VAR) or "").split(",")]
    return list(dict.fromkeys([n for n in wanted if n in order] + order))


def load_specs(path: Optional[str] = None) -> List[dict]:
    """Built-in catalog, then overrides from providers.json (if present). Returns validated, ordered specs."""
    specs = {s["name"]: _normalize(dict(s), f"built-in catalog → {s['name']}") for s in BUILTIN_SPECS}
    order = [s["name"] for s in BUILTIN_SPECS]

    cfg = _config_path(path)
    if cfg is None:
        return [specs[n] for n in _env_first(order)]
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{cfg.name} unreadable: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("providers", []), list):
        raise ConfigError(f"{cfg.name}: expected {{\"providers\": [ … ]}}")

    for i, entry in enumerate(data.get("providers", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ConfigError(f"{cfg.name}: entry #{i + 1} must be an object with a \"name\" field")
        name = entry["name"]
        if entry.get("enabled", True) is False:
            specs.pop(name, None)
            if name in order:
                order.remove(name)
            continue
        merged = {**specs.get(name, {}), **{k: v for k, v in entry.items() if k != "enabled"}}
        specs[name] = _normalize(merged, f"{cfg.name} → {name}")
        if name not in order:
            order.append(name)

    first = data.get("order", [])
    if not isinstance(first, list) or any(n not in specs for n in first):
        unknown = [n for n in first if n not in specs] if isinstance(first, list) else first
        raise ConfigError(f"{cfg.name}: \"order\" must list known providers (unknown: {unknown})")
    order = list(dict.fromkeys(first + order))
    return [specs[n] for n in _env_first(order)]


PROVIDER_SPECS = load_specs()
