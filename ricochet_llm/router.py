"""
Core of the multi-provider free-tier LLM router.
Picks a provider, fails over on exhausted quota / outage, runs the agentic loop (tool calling),
and keeps the observable state (quotas, cooldowns, statistics, event log) shown by the UI.

Pure data            : catalog.py (providers), cleaning.py (cleanup)
Application side     : tools.py (tool registry) + set_system_prompt(), loaded via plugins.py
"""
import atexit
import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI

from . import cleaning
from .catalog import PROVIDER_SPECS, REQUEST_TIMEOUT_S
from .tools import registry

logger = logging.getLogger(__name__)
load_dotenv()


class LLMRouterError(RuntimeError):
    """No provider could answer (or the tool loop ran too long)."""


# ============================================================
# SETTINGS
# ============================================================

MAX_TOOL_ROUNDS       = 8      # max number of "the model calls tools" rounds
TRANSIENT_RETRIES     = 1      # retries on the same provider (network error / 5xx)
TRANSIENT_BACKOFF_S   = 0.5    # base of the exponential backoff (with jitter)

COOLDOWN_RATE_LIMIT_S = 60     # 429 without a Retry-After header
COOLDOWN_TRANSIENT_S  = 30     # provider chaining network errors / 5xx
COOLDOWN_CONFIG_S     = 900    # 401 / 403 / 404 : key or model to fix, no point retrying soon
COOLDOWN_PAYMENT_S    = 3600   # 402, zero limit, DAILY cap reached: won't be back for hours
COOLDOWN_MAX_S        = 3600   # ceiling for any cooldown (Retry-After included)
COOLDOWN_SIZE_S       = 1800   # a model too small for the request (413): its limit won't change within 30 min
COOLDOWN_EMPTY_S      = 300    # a model returning an empty answer (reasoning without an answer): move on to the next
SIZE_MARGIN           = 0.9    # the token estimate slightly overestimates: keep a margin
QUOTA_STALE_S         = 300    # a quota read more than 5 min ago is considered "unknown" and probed again
PROBE_MAX_AGE_S       = 60     # max age of a quota probe (e.g. OpenRouter /key) before re-reading
EVENTS_KEPT           = 200
STATE_ENV_VAR         = "RICOCHET_STATE"   # path of a JSON file: pauses and learned limits survive a restart (opt-in)
STATE_WRITE_GAP_S     = 2.0               # at most one write every 2 s; the last changes are flushed at exit

DEFAULT_SYSTEM_PROMPT = "You are an assistant. You answer clearly and in a structured way."
_system_prompt = DEFAULT_SYSTEM_PROMPT


def set_system_prompt(text: str) -> None:
    """Set the system prompt sent with every call."""
    global _system_prompt
    _system_prompt = text


def get_system_prompt() -> str:
    return _system_prompt


# ============================================================
# ACTIVE PROVIDERS
# ============================================================

_RE_VARIABLE = re.compile(r"\{([A-Z0-9_]+)\}")


def _env_value(spec: dict, variable: str) -> str:
    """Value of one of the provider's environment variables, under its name or one of its aliases (`env_aliases`):
    the typo `CLOUDFLAIRE_ACCOUNT_ID` exists in the wild."""
    for name in (variable, *(spec.get("env_aliases") or {}).get(variable, ())):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _is_configured(spec: dict) -> bool:
    """The key AND the variables the URL requires (account id) are set."""
    return bool(_env_value(spec, spec["env_key"])) and all(_env_value(spec, v) for v in spec.get("env_required") or ())


def _resolve_base_url(spec: dict) -> str:
    return _RE_VARIABLE.sub(lambda m: _env_value(spec, m.group(1)), spec["base_url"])


def _build_providers() -> list:
    providers = []
    for spec in PROVIDER_SPECS:
        if not _is_configured(spec):
            logger.info("LLM %s skipped: %s missing", spec["name"], spec["env_key"])
            continue
        provider = dict(spec)
        provider["expires_at"] = (os.getenv(spec["env_key"] + "_EXPIRES") or "").strip() or None
        # max_retries=0: the SDK must not retry (and sleep) on a 429, the router does the failover.
        provider["client"] = OpenAI(
            base_url=_resolve_base_url(spec), api_key=_env_value(spec, spec["env_key"]), max_retries=0,
            timeout=spec.get("timeout", REQUEST_TIMEOUT_S),
        )
        providers.append(provider)
    return providers


def _disabled_specs() -> list:
    return [dict(s) for s in PROVIDER_SPECS if not _is_configured(s)]


_PROVIDERS = _build_providers()
_DISABLED  = _disabled_specs()      # catalog providers without a key: shown as "disabled" by the UI

_now   = time.monotonic             # indirections for the tests
_sleep = time.sleep


def _provider_by_name(name: str) -> Optional[dict]:
    return next((p for p in _PROVIDERS if p["name"] == name), None)


# ============================================================
# OBSERVABLE STATE (quotas, cooldowns, statistics, event log)
# ============================================================

_quota_lock = threading.Lock()
_provider_remaining:      Dict[str, Optional[int]] = {}
_provider_seen:           Dict[str, float] = {}   # time of the last quota reading
_provider_cooldown_until: Dict[str, float] = {}   # 0 = no cooldown
_provider_last_error:     Dict[str, Optional[str]] = {}
_provider_headers:        Dict[str, dict] = {}    # last x-ratelimit-* headers: {"at": t, "values": {...}}
_provider_probe:          Dict[str, dict] = {}    # last quota probe (OpenRouter /key…)
_provider_stats:          Dict[str, dict] = {}
_calls_window:            Dict[str, deque] = {}   # call timestamps (for "calls / minute")
# PER-MODEL STATE. A 429 at Groq sets ONE model aside (each has its own tokens-per-minute budget), not the whole provider;
# a 413 teaches a model's limit. The provider is only paused when none of its models can answer any more.
_model_cooldown_until:    Dict[Tuple[str, str], float] = {}
_model_limit:             Dict[Tuple[str, str], int] = {}
_model_budget:            Dict[Tuple[str, str], Tuple[int, float]] = {}   # remaining tokens, instant of the reset

_events_lock = threading.Lock()
_events: deque = deque(maxlen=EVENTS_KEPT)


def _new_stats() -> dict:
    return {"ok": 0, "errors": 0, "tokens": 0, "last_latency_ms": None, "avg_latency_ms": None,
            "last_ok_at": None, "last_error_at": None}


def _reset_state() -> None:
    with _quota_lock:
        for d in (_provider_remaining, _provider_seen, _provider_cooldown_until, _provider_last_error,
                  _provider_headers, _provider_probe, _provider_stats, _calls_window,
                  _model_cooldown_until, _model_limit, _model_budget):
            d.clear()
        for p in _PROVIDERS:
            name = p["name"]
            _provider_remaining[name] = None
            _provider_seen[name] = 0.0
            _provider_cooldown_until[name] = 0.0
            _provider_last_error[name] = None
            _provider_stats[name] = _new_stats()
            _calls_window[name] = deque(maxlen=500)
    with _events_lock:
        _events.clear()


_reset_state()


# ============================================================
# STATE THAT SURVIVES A RESTART (opt-in: RICOCHET_STATE=path/to/state.json)
# ============================================================
# Without it, a restart forgets every pause and sends the first calls to providers that were out of quota.
# Only what costs a wasted call is kept: provider pauses, model pauses, learned token limits. Pauses are stored as
# wall-clock times (the monotonic clock does not outlive the process) and are dropped once over.

_state_lock = threading.Lock()
_state_written = 0.0
_state_dirty = False


def _state_file() -> Optional[Path]:
    value = os.getenv(STATE_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


def _snapshot() -> dict:
    now, wall = _now(), time.time()
    with _quota_lock:
        providers = {n: {"until": wall + (u - now), "error": _provider_last_error.get(n)}
                     for n, u in _provider_cooldown_until.items() if u > now}
        models = [{"provider": p, "model": m, "until": wall + (u - now)}
                  for (p, m), u in _model_cooldown_until.items() if u > now]
        limits = [{"provider": p, "model": m, "tokens": n} for (p, m), n in _model_limit.items()]
    return {"saved_at": wall, "providers": providers, "models": models, "limits": limits}


def save_state() -> bool:
    """Write the state file now (no-op without RICOCHET_STATE). Returns True if a file was written."""
    global _state_written, _state_dirty
    path = _state_file()
    if path is None:
        return False
    data = json.dumps(_snapshot(), indent=1)
    with _state_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("State file not written (%s): %s", path, e)
            return False
        _state_written, _state_dirty = time.time(), False
    return True


def _state_changed() -> None:
    """Something worth keeping changed: write soon (rate-limited)."""
    global _state_dirty
    if _state_file() is None:
        return
    if time.time() - _state_written >= STATE_WRITE_GAP_S:
        save_state()
    else:
        _state_dirty = True


def _flush_state() -> None:
    if _state_dirty:
        save_state()


def load_state() -> bool:
    """Restore pauses and learned limits from the state file. Entries for unknown providers, finished pauses and
    malformed entries are ignored. Returns True if a file was read."""
    path = _state_file()
    if path is None or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("State file ignored (%s): %s", path, e)
        return False
    if not isinstance(data, dict):
        return False
    now, wall = _now(), time.time()
    known = {p["name"] for p in _PROVIDERS}

    def left(until) -> float:
        return min(float(until) - wall, COOLDOWN_MAX_S)

    with _quota_lock:
        for name, entry in (data.get("providers") or {}).items():
            try:
                if name in known and left(entry["until"]) > 0:
                    _provider_cooldown_until[name] = now + left(entry["until"])
                    _provider_remaining[name] = 0
                    _provider_seen[name] = now
                    _provider_last_error[name] = str(entry.get("error") or "")[:200] or None
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        for entry in data.get("models") or []:
            try:
                if entry["provider"] in known and left(entry["until"]) > 0:
                    _model_cooldown_until[(entry["provider"], str(entry["model"]))] = now + left(entry["until"])
            except (KeyError, TypeError, ValueError):
                continue
        for entry in data.get("limits") or []:
            try:
                if entry["provider"] in known and int(entry["tokens"]) > 0:
                    _model_limit[(entry["provider"], str(entry["model"]))] = int(entry["tokens"])
            except (KeyError, TypeError, ValueError):
                continue
    return True


load_state()
atexit.register(_flush_state)


def _record_event(provider: str, kind: str, message: str) -> None:
    with _events_lock:
        _events.append({"t": time.time(), "provider": provider, "kind": kind, "message": message[:240]})


def recent_events(limit: int = 60) -> List[dict]:
    """Event log, newest first."""
    with _events_lock:
        return list(_events)[::-1][:limit]


def _usage_tokens(response) -> int:
    return getattr(getattr(response, "usage", None), "total_tokens", 0) or 0


def _record_call_result(name: str, ok: bool, seconds: float, tokens: int = 0) -> None:
    with _quota_lock:
        st = _provider_stats.get(name)
        if st is None:
            return
        _calls_window[name].append(_now())
        if ok:
            ms = int(seconds * 1000)
            st["ok"] += 1
            st["tokens"] += tokens
            st["last_latency_ms"] = ms
            st["avg_latency_ms"] = ms if st["avg_latency_ms"] is None else int(0.7 * st["avg_latency_ms"] + 0.3 * ms)
            st["last_ok_at"] = time.time()
        else:
            st["errors"] += 1
            st["last_error_at"] = time.time()


# ============================================================
# QUOTAS / COOLDOWNS
# ============================================================

def _update_quota_from_headers(provider_name: str, headers, routing: bool = True) -> None:
    """Remember the quota headers (display) and, if `routing`, the quota that drives the provider choice."""
    provider = _provider_by_name(provider_name)
    if provider is None or headers is None:
        return
    captured = {}
    try:
        for k, v in headers.items():
            kl = str(k).lower()
            if "ratelimit" in kl or kl == "retry-after":
                captured[kl] = str(v)
    except Exception:
        pass
    if captured:
        with _quota_lock:
            _provider_headers[provider_name] = {"at": _now(), "values": captured}
    quota_header = provider.get("quota_header")
    if not routing or not quota_header:
        return
    val = headers.get(quota_header)
    if val is not None:
        try:
            with _quota_lock:
                _provider_remaining[provider_name] = int(val)
                _provider_seen[provider_name] = _now()
        except (ValueError, TypeError):
            pass


_RE_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _duration_text(text) -> Optional[float]:
    """'46.5s', '1m30.5s', '250ms', '2m0s', '12' -> seconds. None if unreadable."""
    if text is None:
        return None
    text = str(text).strip()
    try:
        return float(text)
    except ValueError:
        pass
    total, found = 0.0, False
    for value, unit in _RE_DURATION.findall(text):
        found = True
        total += float(value) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total if found else None


def _note_budget(provider_name: str, model: str, headers) -> None:
    """Remember the token budget announced by the provider (`x-ratelimit-*-tokens`): this tells us, BEFORE calling,
    that a model can no longer handle the request (no avoidable 429, no second wasted). The announced limit
    becomes the largest acceptable request: beyond it, the request will never succeed, even on an idle model."""
    if not headers:
        return
    key = (provider_name, model)
    limit = headers.get("x-ratelimit-limit-tokens")
    remaining = headers.get("x-ratelimit-remaining-tokens")
    delay = _duration_text(headers.get("x-ratelimit-reset-tokens"))
    changed = False
    with _quota_lock:
        try:
            if limit is not None:
                learned = int(float(limit))
                changed = _model_limit.get(key) != learned
                _model_limit[key] = learned
            if remaining is not None and delay is not None:
                _model_budget[key] = (int(float(remaining)), _now() + delay)
        except (TypeError, ValueError):
            return
    if changed:
        _state_changed()


def _put_in_cooldown(provider_name: str, seconds: float, exhausted: bool = False, reason: str = "") -> None:
    seconds = max(1.0, min(float(seconds), COOLDOWN_MAX_S))
    with _quota_lock:
        _provider_cooldown_until[provider_name] = _now() + seconds
        if exhausted:
            _provider_remaining[provider_name] = 0
            _provider_seen[provider_name] = _now()
        if reason:
            _provider_last_error[provider_name] = reason[:200]
    logger.warning("LLM %s unavailable for %.0fs (%s)", provider_name, seconds, reason[:120])
    _state_changed()


def _mark_provider_exhausted(provider_name: str, cooldown: float = COOLDOWN_RATE_LIMIT_S, reason: str = "") -> None:
    _put_in_cooldown(provider_name, cooldown, exhausted=True, reason=reason)


def _clear_cooldown(provider_name: str) -> None:
    with _quota_lock:
        _provider_cooldown_until[provider_name] = 0.0
        _provider_remaining[provider_name] = None
    _state_changed()


def _refresh_locked(now: float) -> None:
    """Cooldown over or quota stale → quota 'unknown' (the provider is probed again). Call with the lock held."""
    for p in _PROVIDERS:
        name = p["name"]
        until = _provider_cooldown_until[name]
        if until and until <= now:
            _provider_cooldown_until[name] = 0.0
            _provider_remaining[name] = None
        elif _provider_remaining[name] is not None and now - _provider_seen[name] > QUOTA_STALE_S:
            _provider_remaining[name] = None


def _models_for(provider: dict, mode: str) -> List[str]:
    """Models that can be tried, in order. HEAVY: the heavy one then the DECLARED fallback models (`extra_models`); the
    light one is only part of it if listed there. LIGHT (low quota): the light one alone."""
    if mode == "LIGHT":
        return [provider["model_light"]]
    out: List[str] = []
    for m in [provider["model_heavy"], *(provider.get("extra_models") or [])]:
        if m and m not in out:
            out.append(m)
    return out


def _estimate_tokens(messages: list) -> int:
    """Rough estimate of the input tokens (~3 characters per token)."""
    return sum(len(str(m.get("content") or "")) for m in messages) // 3 + 4 * len(messages)


def _model_wait(provider: dict, model: str, need_tokens: int = 0, now: Optional[float] = None) -> float:
    """Seconds until this model is usable for a request of `need_tokens` tokens, 0 if it is usable now."""
    now = _now() if now is None else now
    key = (provider["name"], model)
    with _quota_lock:
        limit = _model_limit.get(key) or (provider.get("token_limits") or {}).get(model)
        until = _model_cooldown_until.get(key, 0.0)
        budget = _model_budget.get(key)
        if budget is not None and now >= budget[1]:
            _model_budget.pop(key, None)       # the window has reset: the announced budget no longer applies
            budget = None
    if limit and need_tokens * SIZE_MARGIN > limit:
        return float(COOLDOWN_SIZE_S)          # the request won't fit, no point calling just to get a 413
    wait = max(0.0, until - now)
    if budget is not None and budget[0] < need_tokens:
        wait = max(wait, budget[1] - now)      # the announced token budget is not enough: wait for its reset
    return wait


def _first_usable_model(provider: dict, mode: str, need_tokens: int, now: float) -> Optional[str]:
    return next((m for m in _models_for(provider, mode) if _model_wait(provider, m, need_tokens, now) <= 0), None)


def _soonest_model_return(provider: dict, need_tokens: int = 0) -> float:
    """Seconds until the provider's first model is usable again (at least 1)."""
    waits = [_model_wait(provider, m, need_tokens) for m in _models_for(provider, "HEAVY")]
    return max(1.0, min(waits)) if waits else float(COOLDOWN_RATE_LIMIT_S)


def _bench_model(provider: dict, model: str, seconds: float, reason: str) -> None:
    seconds = max(1.0, min(float(seconds), COOLDOWN_MAX_S))
    with _quota_lock:
        _model_cooldown_until[(provider["name"], model)] = _now() + seconds
        _provider_last_error[provider["name"]] = reason[:200]
    _record_event(provider["name"], "model_paused", f"{model} set aside {seconds:.0f}s: {reason}"[:240])
    logger.warning("LLM %s / %s set aside for %.0fs (%s)", provider["name"], model, seconds, reason[:120])
    _state_changed()


def _other_model_usable(provider: dict, model: str, need_tokens: int = 0) -> bool:
    return any(m != model and _model_wait(provider, m, need_tokens) <= 0 for m in _models_for(provider, "HEAVY"))


def _call_options(provider: dict, model: str) -> dict:
    """Settings specific to this MODEL (`model_options`): `extra_body` to turn off the reasoning of Nemotron models."""
    opt = (provider.get("model_options") or {}).get(model) or {}
    return {"extra_body": opt["extra_body"]} if opt.get("extra_body") else {}


def _choose_provider_and_model(exclude=(), require_tools: bool = False, need_tokens: int = 0,
                               only=None, prefer: Optional[str] = None) -> Tuple[dict, str, str]:
    """HEAVY: first provider whose quota is sufficient. LIGHT: the one with the most remaining quota.
    `exclude` = providers already tried during this call; `require_tools` = the conversation contains tool calls;
    `need_tokens` = estimated size of the request (a model too small for it is skipped); `only` = allowed provider
    names; `prefer` = provider tried first if eligible.
    Within a provider, the first usable model: a model set aside does not sideline the whole provider."""
    if not _PROVIDERS:
        raise LLMRouterError("No provider configured: set at least one API key in .env")
    now = _now()
    with _quota_lock:
        _refresh_locked(now)
        remaining = dict(_provider_remaining)
        cooldown  = dict(_provider_cooldown_until)
        last_err  = dict(_provider_last_error)

    candidates = [
        p for p in _PROVIDERS
        if p["name"] not in exclude
        and cooldown[p["name"]] <= now
        and (p["supports_tools"] or not require_tools)
        and (only is None or p["name"] in only)
    ]
    healthy = [
        p for p in candidates
        if remaining[p["name"]] is None or remaining[p["name"]] >= p["quota_min"]
    ]
    prefer_tools = len(registry) > 0          # with no registered tools, the catalog order is enough
    healthy.sort(key=lambda p: (p["name"] != prefer, prefer_tools and not p["supports_tools"], _PROVIDERS.index(p)))
    for provider in healthy:
        model = _first_usable_model(provider, "HEAVY", need_tokens, now)
        if model is not None:
            return provider, model, "HEAVY"
    candidates.sort(key=lambda p: (
        p["name"] != prefer,
        prefer_tools and not p["supports_tools"],
        -(remaining[p["name"]] or 0),
        _PROVIDERS.index(p),
    ))
    for provider in candidates:
        model = _first_usable_model(provider, "LIGHT", need_tokens, now)
        if model is not None:
            return provider, model, "LIGHT"

    waits = [cooldown[p["name"]] - now for p in _PROVIDERS if cooldown[p["name"]] > now]
    waits += [_soonest_model_return(p, need_tokens) for p in candidates]
    hint = f" Retry in ~{int(min(waits)) + 1}s." if waits else ""
    causes = "; ".join(f"{p['name']} → {last_err[p['name']]}" for p in _PROVIDERS if last_err[p["name"]])
    raise LLMRouterError("No provider available." + hint + (f" Last errors: {causes}" if causes else ""))


def available_providers(only=None, need_tokens: int = 0) -> List[str]:
    """Names of the providers that can answer RIGHT NOW (not paused, at least one usable model). Used by the
    parallel executor: a paused provider must not take work."""
    now = _now()
    with _quota_lock:
        _refresh_locked(now)
        cooldown = dict(_provider_cooldown_until)
    return [p["name"] for p in _PROVIDERS
            if cooldown[p["name"]] <= now and (only is None or p["name"] in only)
            and _first_usable_model(p, "HEAVY", need_tokens, now) is not None]


def next_provider(require_tools: bool = False) -> dict:
    """Which provider/model would be chosen now (without calling anything)."""
    try:
        provider, model, mode = _choose_provider_and_model(require_tools=require_tools)
        return {"provider": provider["name"], "label": provider.get("label", provider["name"]),
                "model": model, "mode": mode}
    except LLMRouterError as e:
        return {"provider": None, "error": str(e)}


# ----- quota probes outside headers ---------------------------------------------------------

def _probe_openrouter(provider: dict) -> dict:
    data = provider["client"].get("/key", cast_to=object)
    data = data.get("data", data) if isinstance(data, dict) else {}
    limits = []
    free = data.get("free_model_daily_requests")
    if isinstance(free, dict) and free.get("limit") is not None:
        limits.append({"label": "free requests / day", "remaining": free.get("remaining"),
                       "limit": free.get("limit"), "source": "API /key"})
    return {"limits": limits, "expires_at": data.get("expires_at")}


_PROBES = {"openrouter": _probe_openrouter}


def refresh_probes(force: bool = False) -> None:
    """Re-read the quotas that do not come through headers (blocking: call outside the event loop)."""
    for provider in list(_PROVIDERS):
        probe = _PROBES.get(provider.get("probe"))
        if probe is None:
            continue
        name = provider["name"]
        with _quota_lock:
            last = _provider_probe.get(name, {}).get("at")
        if not force and last is not None and _now() - last < PROBE_MAX_AGE_S:
            continue
        try:
            data = probe(provider)
        except Exception as e:
            data = {"limits": [], "error": f"{type(e).__name__}: {str(e)[:120]}"}
        data["at"] = _now()
        with _quota_lock:
            _provider_probe[name] = data


def _limits_from_headers(provider: dict, values: dict) -> List[dict]:
    """Turn the x-ratelimit-remaining-X / x-ratelimit-limit-X pairs into readable gauges."""
    labels = provider.get("quota_labels") or {}
    out = []
    for header, raw in values.items():
        if not header.startswith("x-ratelimit-remaining"):
            continue
        suffix = header[len("x-ratelimit-remaining"):].strip("-")
        try:
            remaining = int(float(raw))
        except (TypeError, ValueError):
            continue
        raw_limit = values.get(header.replace("remaining", "limit", 1))
        try:
            limit = int(float(raw_limit)) if raw_limit is not None else None
        except (TypeError, ValueError):
            limit = None
        label = labels.get(suffix) or suffix.replace("-", " ") or "requests"
        out.append({"label": label, "remaining": remaining, "limit": limit, "source": "header"})
    return sorted(out, key=lambda x: x["label"])


def _key_expiry(provider: dict, probe: Optional[dict]) -> Tuple[Optional[str], Optional[int]]:
    raw = (probe or {}).get("expires_at") or provider.get("expires_at")
    if not raw:
        return None, None
    try:
        day = date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None, None
    return day.isoformat(), (day - date.today()).days


def _catalog_index(name: str) -> int:
    names = [s["name"] for s in PROVIDER_SPECS]
    return names.index(name) if name in names else len(names)


def quota_status() -> Dict[str, dict]:
    """Readable snapshot of each provider (status, quotas, statistics) — consumed by the UI and /api/status."""
    now = _now()
    with _quota_lock:
        _refresh_locked(now)
        remaining = dict(_provider_remaining)
        cooldown  = dict(_provider_cooldown_until)
        last_err  = dict(_provider_last_error)
        headers   = {n: dict(v) for n, v in _provider_headers.items()}
        probes    = {n: dict(v) for n, v in _provider_probe.items()}
        stats     = {n: dict(v) for n, v in _provider_stats.items()}
        calls_min = {n: sum(1 for t in w if now - t <= 60) for n, w in _calls_window.items()}

    out = {}
    for p in _PROVIDERS:
        name = p["name"]
        if cooldown[name] > now:
            status = "cooldown"
        elif remaining[name] is None:
            status = "unknown"
        else:
            status = "ok" if remaining[name] >= p["quota_min"] else "limit"

        hdr = headers.get(name)
        probe = probes.get(name)
        limits = (_limits_from_headers(p, hdr["values"]) if hdr else []) + list((probe or {}).get("limits") or [])
        expires_at, expires_in = _key_expiry(p, probe)
        out[name] = {
            "label":        p.get("label", name),
            "configured":   True,
            "status":       status,
            "remaining":    remaining[name],
            "min":          p["quota_min"],
            "cooldown_s":   max(0, int(cooldown[name] - now)) if cooldown[name] > now else 0,
            "last_error":   last_err[name],
            "model_heavy":  p.get("model_heavy"),
            "model_light":  p.get("model_light"),
            "models":       _models_for(p, "HEAVY"),
            "model_pauses": {m: int(w) for m in _models_for(p, "HEAVY") if (w := _model_wait(p, m, 0, now)) > 0},
            "supports_tools": p["supports_tools"],
            "note":         p.get("note"),
            "console_url":  p.get("console_url"),
            "limits":       limits,
            "limits_age_s": int(now - hdr["at"]) if hdr else None,
            "probe_error":  (probe or {}).get("error"),
            "stats":        {**stats.get(name, _new_stats()), "calls_last_minute": calls_min.get(name, 0)},
            "key_expires_at":      expires_at,
            "key_expires_in_days": expires_in,
        }
    for spec in _DISABLED:
        out[spec["name"]] = {
            "label": spec.get("label", spec["name"]), "configured": False, "status": "disabled",
            "env_key": spec["env_key"], "console_url": spec.get("console_url"), "note": spec.get("note"),
            "model_heavy": spec.get("model_heavy"), "model_light": spec.get("model_light"),
            "supports_tools": spec["supports_tools"],
        }
    return dict(sorted(out.items(), key=lambda kv: _catalog_index(kv[0])))


# ============================================================
# ERRORS: classification and reaction
# ============================================================

def _classify_error(e: Exception) -> str:
    """rate_limit | payment | transient | config | rejected | fatal"""
    if isinstance(e, openai.RateLimitError):
        return "rate_limit"
    if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", None) == 402:
        return "payment"                       # offer not activated or model on a paid plan: won't be fixed within a minute
    if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):  # inclut APITimeoutError
        return "transient"
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return "config"
    if isinstance(e, openai.APIStatusError):   # 400, 410, 413, 422…: this provider refuses this request
        return "rejected"
    err = str(e).lower()
    if "429" in err or "rate_limit" in err or "quota" in err:
        return "rate_limit"
    return "fatal"                             # local bug (TypeError, KeyError…): don't mask it


def _describe_error(e: Exception) -> str:
    """Readable error message: 'RateLimitError 429: …' with the real reason extracted from the response body."""
    body = getattr(e, "body", None)
    if isinstance(body, list) and body:        # Gemini returns a list
        body = body[0]
    msg = None
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        msg = err.get("message") or err.get("detail") or body.get("detail")
        raw = (err.get("metadata") or {}).get("raw") if isinstance(err.get("metadata"), dict) else None
        if raw and (not msg or msg == "Provider returned error"):   # OpenRouter: the cause is in metadata.raw
            msg = raw
    if not msg:
        msg = re.sub(r"^Error code: \d+ - ", "", str(e))
    code = getattr(e, "status_code", None)
    return f"{type(e).__name__}{' ' + str(code) if code else ''}: {' '.join(str(msg).split())}"


def _retry_after_seconds(e: Exception) -> Optional[float]:
    headers = getattr(getattr(e, "response", None), "headers", None)
    if not headers:
        return None
    for name, factor in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        val = headers.get(name)
        if val is not None:
            try:
                return float(val) * factor
            except (TypeError, ValueError):
                pass   # Retry-After in HTTP date format: ignored
    return None


def _response_header(e: Exception, name: str) -> Optional[str]:
    headers = getattr(getattr(e, "response", None), "headers", None)
    return headers.get(name) if headers else None


def _limit_is_zero(e: Exception) -> bool:
    """A per-minute limit of ZERO means 'free offer not activated' (valid key but no chat entitlement):
    no point retrying for a long time."""
    value = _response_header(e, "x-ratelimit-limit-req-minute")
    return value is not None and str(value).strip() == "0"


_DAILY_MARKERS = ("perday", "per day", "per-day", "daily", "par jour", "quotidien")


def _is_daily_cap(e: Exception) -> bool:
    """Does the 429 come from an exhausted DAILY allocation (Gemini: `GenerateRequestsPerDayPerProjectPerModel`,
    Cloudflare: "daily free allocation", OpenRouter: `free-models-per-day`)? It won't be back for hours:
    retrying every minute only wastes a call."""
    return any(marker in f"{_describe_error(e)} {getattr(e, 'body', '')}".lower() for marker in _DAILY_MARKERS)


def _rate_limit_delay(e: Exception) -> float:
    """How long to set aside after a 429: `Retry-After`, otherwise the token budget reset announced by the
    provider, otherwise one hour for a daily cap, otherwise the default pause."""
    return (_retry_after_seconds(e)
            or _duration_text(_response_header(e, "x-ratelimit-reset-tokens"))
            or (COOLDOWN_PAYMENT_S if _is_daily_cap(e) else COOLDOWN_RATE_LIMIT_S))


_RE_LIMIT_413 = re.compile(r"Limit (\d+)")


def _limit_from_413(e: Exception) -> Optional[int]:
    """'Request too large ... tokens per minute (TPM): Limit 8000, Requested 8285' -> 8000."""
    found = _RE_LIMIT_413.search(_describe_error(e))
    return int(found.group(1)) if found else None


def _handle_call_failure(provider: dict, e: Exception, excluded: set, retries: Dict[str, int],
                         model: Optional[str] = None, need_tokens: int = 0) -> str:
    """React to a call exception and return its category. If the provider is not added to `excluded`,
    the caller must retry the same one; raises if the error is a local bug.

    Returns "model_limited" when ONLY the model was set aside and another model of the same provider can answer:
    the provider stays in play, the caller picks another model (a single 413 must not sideline a whole provider
    while its other models are available)."""
    name, kind = provider["name"], _classify_error(e)
    reason = _describe_error(e)[:200]
    if kind == "fatal":
        raise e
    _update_quota_from_headers(name, getattr(getattr(e, "response", None), "headers", None), routing=False)
    _record_event(name, kind, reason)
    status = getattr(e, "status_code", None)
    multi = model is not None and model in _models_for(provider, "HEAVY") and len(_models_for(provider, "HEAVY")) > 1

    if kind == "payment" or (kind == "rate_limit" and _limit_is_zero(e)):
        _put_in_cooldown(name, COOLDOWN_PAYMENT_S, reason=reason)
        excluded.add(name)
    elif kind == "rate_limit":
        delay = _rate_limit_delay(e)
        if multi:
            _bench_model(provider, model, delay, reason)
            if _other_model_usable(provider, model, need_tokens):
                return "model_limited"
            delay = _soonest_model_return(provider, need_tokens)      # the provider comes back with its first model
        _mark_provider_exhausted(name, delay, reason)
        excluded.add(name)
    elif kind == "transient":
        if retries.get(name, 0) < TRANSIENT_RETRIES:
            attempt = retries[name] = retries.get(name, 0) + 1
            delay = TRANSIENT_BACKOFF_S * (2 ** (attempt - 1)) * (0.5 + random.random())
            logger.warning("LLM %s transient error (%s) → retrying in %.1fs", name, reason, delay)
            _sleep(delay)
            return kind                 # same provider
        _put_in_cooldown(name, COOLDOWN_TRANSIENT_S, reason=reason)
        excluded.add(name)
    elif kind == "config":
        if multi and status == 404:     # a model removed from the catalog says nothing about the others
            _bench_model(provider, model, COOLDOWN_CONFIG_S, reason)
            if _other_model_usable(provider, model, need_tokens):
                return "model_limited"
        _put_in_cooldown(name, COOLDOWN_CONFIG_S, reason=reason)
        excluded.add(name)
    else:   # rejected: specific to this request
        if status == 413 and model is not None:
            limit = _limit_from_413(e)
            if limit is not None:
                with _quota_lock:
                    _model_limit[(name, model)] = limit
                _bench_model(provider, model, COOLDOWN_SIZE_S, f"request larger than the {limit}-token limit (413)")
                if _other_model_usable(provider, model, need_tokens):
                    return "model_limited"
        logger.warning("LLM %s refused the request (%s) → failing over", name, reason)
        with _quota_lock:
            _provider_last_error[name] = reason
        excluded.add(name)
    return kind


def _handle_empty_answer(provider: dict, model: str, excluded: set, need_tokens: int = 0) -> None:
    """Empty answer: a reasoning model spent all its tokens before writing. Moves on to the provider's next
    model if one is usable, otherwise sets the provider aside for this call."""
    models = _models_for(provider, "HEAVY")
    if model in models and len(models) > 1:
        _bench_model(provider, model, COOLDOWN_EMPTY_S, "empty response (reasoning without an answer?)")
        if _other_model_usable(provider, model, need_tokens):
            return
    excluded.add(provider["name"])


# ============================================================
# MANUAL ACTIONS (UI)
# ============================================================

def ping_provider(name: str) -> dict:
    """Minimal call on the light model: checks the key, measures latency, lifts the cooldown if it answers."""
    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    model = provider["model_light"]
    t0 = time.perf_counter()
    try:
        raw = provider["client"].chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "Reply only: OK"}],
            max_tokens=32, temperature=0, **_call_options(provider, model))
        response = raw.parse()
    except Exception as e:
        dt = time.perf_counter() - t0
        _record_call_result(name, False, dt)
        try:
            kind = _handle_call_failure(provider, e, set(), {name: TRANSIENT_RETRIES})
        except Exception:
            kind = "fatal"
        return {"ok": False, "provider": name, "model": model, "kind": kind,
                "latency_ms": int(dt * 1000), "error": _describe_error(e)[:200]}
    dt = time.perf_counter() - t0
    _clear_cooldown(name)
    _update_quota_from_headers(name, raw.headers)
    _record_call_result(name, True, dt, _usage_tokens(response))
    _record_event(name, "ok", f"test {model} · {int(dt * 1000)} ms")
    return {"ok": True, "provider": name, "model": model, "latency_ms": int(dt * 1000)}


def reset_provider(name: str) -> None:
    """Lift the cooldown and forget the quota: the provider is eligible again right away."""
    if _provider_by_name(name) is None:
        raise KeyError(name)
    _clear_cooldown(name)
    _record_event(name, "info", "cooldown lifted manually")


# ============================================================
# AGENTIC LOOP
# ============================================================

def _strip_tool_artifacts(text: str) -> str:
    return cleaning.strip_tool_artifacts(cleaning.strip_think(text), registry.names())


@dataclass
class ChatResult:
    """Answer + what happened to obtain it."""
    answer: str
    provider: str
    model: str
    mode: str
    latency_ms: int
    tool_rounds: int = 0
    trail: List[dict] = field(default_factory=list)   # providers tried before the one that answered

    def to_dict(self) -> dict:
        return asdict(self)


def chat(conversation_history: list, user_message: str) -> str:
    """Append user_message to the history, run the loop (tool calls included), return the answer.
    On failure, the history is restored to its state before the call (the same message can be sent again)."""
    return chat_detailed(conversation_history, user_message).answer


def chat_detailed(conversation_history: list, user_message: str) -> ChatResult:
    """Like `chat`, but also returns the provider/model used, the latency and the failovers."""
    start = len(conversation_history)
    conversation_history.append({"role": "user", "content": user_message})
    try:
        return _chat_loop(conversation_history)
    except BaseException:
        del conversation_history[start:]
        raise


def _chat_loop(conversation_history: list) -> ChatResult:
    t_start = time.perf_counter()
    messages = [{"role": "system", "content": _system_prompt}]
    tool_schemas = registry.schemas()
    relevant = [
        msg for msg in conversation_history
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str)
    ]
    for msg in relevant[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    excluded: set = set()            # providers abandoned for THIS call (guarantees the loop terminates)
    retries: Dict[str, int] = {}
    trail: List[dict] = []
    tool_rounds = 0
    nudged = False
    in_tool_exchange = False         # tool calls are in progress: only providers with tools can continue

    need = _estimate_tokens(messages)
    provider, model, mode = _choose_provider_and_model(excluded, need_tokens=need)
    logger.info("LLM [%s] %s / %s", mode, provider["name"], model)

    while True:
        t_call = time.perf_counter()
        try:
            kwargs = dict(model=model, messages=messages, max_tokens=4096, temperature=0.2,
                          **_call_options(provider, model))
            if provider["supports_tools"] and tool_schemas:
                kwargs["tools"]       = tool_schemas
                kwargs["tool_choice"] = "auto"
            raw = provider["client"].chat.completions.with_raw_response.create(**kwargs)
            _update_quota_from_headers(provider["name"], raw.headers)
            _note_budget(provider["name"], model, raw.headers)
            response = raw.parse()
        except Exception as e:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            kind = _handle_call_failure(provider, e, excluded, retries, model, need)
            trail.append({"provider": provider["name"], "model": model, "outcome": kind,
                          "detail": _describe_error(e)[:160]})
            if kind != "model_limited" and provider["name"] not in excluded:
                continue                                  # retry on the same provider and the same model
            provider, model, mode = _choose_provider_and_model(excluded, in_tool_exchange, need)
            logger.info("LLM failover → [%s] %s / %s", mode, provider["name"], model)
            continue

        with _quota_lock:
            rem = _provider_remaining.get(provider["name"])
        if rem is not None:
            logger.info("LLM [%s] remaining quota: %s tokens", provider["name"], rem)

        if not response.choices:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            _record_event(provider["name"], "no_choices", "response without choices")
            trail.append({"provider": provider["name"], "outcome": "no_choices", "detail": "response without choices"})
            _mark_provider_exhausted(provider["name"], COOLDOWN_RATE_LIMIT_S, "response without choices")
            excluded.add(provider["name"])
            provider, model, mode = _choose_provider_and_model(excluded, in_tool_exchange, need)
            continue

        _record_call_result(provider["name"], True, time.perf_counter() - t_call, _usage_tokens(response))
        choice  = response.choices[0]
        message = choice.message

        if choice.finish_reason == "stop" or not message.tool_calls:
            final   = message.content or ""
            cleaned = _strip_tool_artifacts(final)
            if not cleaned:
                logger.warning("LLM %s: empty response after cleaning → failing over", provider["name"])
                _record_event(provider["name"], "empty", "empty response after cleaning")
                trail.append({"provider": provider["name"], "model": model, "outcome": "empty", "detail": "empty response"})
                _handle_empty_answer(provider, model, excluded, need)
                provider, model, mode = _choose_provider_and_model(excluded, in_tool_exchange, need)
                if not nudged:
                    messages.append({"role": "user", "content": "Answer my previous question."})
                    nudged = True
                continue
            conversation_history.append({"role": "assistant", "content": cleaned})
            latency_ms = int((time.perf_counter() - t_start) * 1000)
            _record_event(provider["name"], "ok", f"{model} · {mode} · {latency_ms} ms")
            return ChatResult(answer=cleaned, provider=provider["name"], model=model, mode=mode,
                              latency_ms=latency_ms, tool_rounds=tool_rounds, trail=trail)

        tool_rounds += 1
        if tool_rounds > MAX_TOOL_ROUNDS:
            raise LLMRouterError(f"More than {MAX_TOOL_ROUNDS} tool rounds without a final answer: giving up.")
        in_tool_exchange = True

        messages.append({
            "role":       "assistant",
            "content":    message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            result_str = registry.execute(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
            conversation_history.append({"role": "tool_result", "tool_call_id": tc.id, "content": result_str})


# ============================================================
# STANDALONE CALL (no conversation, no tools): batch reads, extraction, classification…
# ============================================================

def provider_names(only=None) -> List[str]:
    """Names of the configured providers (key present), in order of preference."""
    return [p["name"] for p in _PROVIDERS if only is None or p["name"] in only]


def provider_settings(name: str) -> dict:
    """Batch-read settings of a provider: max simultaneous requests, gap between two starts."""
    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    return {"max_concurrency": provider.get("max_concurrency", 2), "min_interval_s": provider.get("min_interval_s", 0.0)}


@dataclass
class GenerateResult:
    """Answer of a standalone call, with what happened to obtain it."""
    text: str
    provider: str
    model: str
    mode: str
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    finish_reason: Optional[str] = None
    truncated: bool = False            # the answer stopped on `max_tokens`: a JSON may be cut off
    reasoning_used: bool = False       # the model wrote some reasoning (reasoning_content)
    trail: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _client_for(provider: dict, timeout: Optional[float]):
    """The provider's client, with a LONGER timeout for this call. It never shortens the provider's own: the catalog
    timeout is tuned for chat (20 s at NVIDIA), not for a long JSON answer."""
    client = provider["client"]
    if timeout and timeout > provider.get("timeout", REQUEST_TIMEOUT_S) and hasattr(client, "with_options"):
        client = client.with_options(timeout=float(timeout))
    return client


def _messages_for(prompt, system: Optional[str]) -> list:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else [dict(m) for m in prompt]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def generate(prompt, system: Optional[str] = None, *, max_tokens: int = 4096, temperature: float = 0.0,
             timeout: Optional[float] = None, only=None, prefer: Optional[str] = None) -> GenerateResult:
    """A call without conversation: fails over between providers AND between models, like `chat`, without tools.

    prompt       text, or list of {"role", "content"} messages
    max_tokens   output cap. It INCLUDES the reasoning of reasoning models: too low, and the answer is empty
    timeout      seconds granted to each provider FOR THIS CALL (lengthens the catalog one, never shortens it)
    only         allowed provider names (None = all)
    prefer       provider tried first if eligible (load spreading: see parallel.py)

    An EMPTY answer is never returned: the model is set aside and the next one tried. Raises LLMRouterError if nothing
    can answer any more. Checking the CONTENT (valid JSON, quotes found in the source) is left to the caller:
    a valid but empty JSON is indistinguishable from a real absence."""
    done = _run_completion(_messages_for(prompt, system), max_tokens=max_tokens, temperature=temperature,
                           timeout=timeout, only=only, prefer=prefer)
    choice = done.response.choices[0]
    usage = getattr(done.response, "usage", None)
    return GenerateResult(
        text=done.text, provider=done.provider, model=done.model, mode=done.mode, latency_ms=done.latency_ms,
        tokens_in=getattr(usage, "prompt_tokens", None), tokens_out=getattr(usage, "completion_tokens", None),
        finish_reason=choice.finish_reason, truncated=choice.finish_reason == "length",
        reasoning_used=done.reasoning, trail=done.trail)


@dataclass
class _Completion:
    """What `_run_completion` found: the provider's response, the cleaned text and how it was obtained."""
    response: object
    text: str
    provider: str
    model: str
    mode: str
    latency_ms: int
    reasoning: bool
    trail: List[dict]


def _run_completion(messages: list, *, max_tokens: int, temperature: float, timeout: Optional[float] = None,
                    only=None, prefer: Optional[str] = None, tools: Optional[list] = None,
                    tool_choice=None, params: Optional[dict] = None) -> _Completion:
    """The failover loop shared by `generate` and `complete`. With `tools`, only providers that handle tools are used
    and an answer made of tool calls is a valid answer; without, an empty answer is set aside and the next model tried."""
    t_start = time.perf_counter()
    need = _estimate_tokens(messages)
    excluded: set = set()
    retries: Dict[str, int] = {}
    trail: List[dict] = []
    wants_tools = bool(tools)
    provider, model, mode = _choose_provider_and_model(excluded, wants_tools, need, only=only, prefer=prefer)

    while True:
        t_call = time.perf_counter()
        try:
            kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                          **_call_options(provider, model), **(params or {}))
            if wants_tools:
                kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
            raw = _client_for(provider, timeout).chat.completions.with_raw_response.create(**kwargs)
            _update_quota_from_headers(provider["name"], raw.headers)
            _note_budget(provider["name"], model, raw.headers)
            response = raw.parse()
        except Exception as e:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            kind = _handle_call_failure(provider, e, excluded, retries, model, need)
            trail.append({"provider": provider["name"], "model": model, "outcome": kind,
                          "detail": _describe_error(e)[:160]})
            if kind != "model_limited" and provider["name"] not in excluded:
                continue
            provider, model, mode = _choose_provider_and_model(excluded, wants_tools, need, only=only, prefer=prefer)
            continue

        if not response.choices:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            _record_event(provider["name"], "no_choices", "response without choices")
            trail.append({"provider": provider["name"], "model": model, "outcome": "no_choices", "detail": "response without choices"})
            _mark_provider_exhausted(provider["name"], COOLDOWN_RATE_LIMIT_S, "response without choices")
            excluded.add(provider["name"])
            provider, model, mode = _choose_provider_and_model(excluded, wants_tools, need, only=only, prefer=prefer)
            continue

        choice = response.choices[0]
        message = choice.message
        text = cleaning.strip_think(message.content or "")
        reasoning = bool(getattr(message, "reasoning_content", None)
                         or (getattr(message, "model_extra", None) or {}).get("reasoning_content"))
        if not text and not (wants_tools and getattr(message, "tool_calls", None)):
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            why = ("max_tokens reached during reasoning" if choice.finish_reason == "length" and reasoning
                   else "reasoning without an answer" if reasoning else "empty response")
            _record_event(provider["name"], "empty", f"{model} : {why}")
            trail.append({"provider": provider["name"], "model": model, "outcome": "empty", "detail": why})
            _handle_empty_answer(provider, model, excluded, need)
            provider, model, mode = _choose_provider_and_model(excluded, wants_tools, need, only=only, prefer=prefer)
            continue

        _record_call_result(provider["name"], True, time.perf_counter() - t_call, _usage_tokens(response))
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        _record_event(provider["name"], "ok", f"{model} · {mode} · {latency_ms} ms")
        return _Completion(response=response, text=text, provider=provider["name"], model=model, mode=mode,
                           latency_ms=latency_ms, reasoning=reasoning, trail=trail)


# Request parameters that `complete` passes through to the provider as they are.
PASSTHROUGH_PARAMS = ("top_p", "stop", "seed", "presence_penalty", "frequency_penalty", "response_format")


def complete(messages: list, *, tools: Optional[list] = None, tool_choice=None, max_tokens: int = 4096,
             temperature: float = 0.2, timeout: Optional[float] = None, only=None, prefer: Optional[str] = None,
             **params) -> dict:
    """A chat completion in the OpenAI response format, with failover between providers and models.

    Unlike `chat`, the caller owns the conversation AND the tools: `messages` and `tools` go to the provider as they
    are, and tool calls come back to the caller instead of being run here. With `tools`, only providers that handle
    tool calling are used. Extra keyword arguments among PASSTHROUGH_PARAMS are forwarded.
    The result carries an `x_ricochet` field: provider, model, mode, latency and failovers."""
    unknown = sorted(set(params) - set(PASSTHROUGH_PARAMS))
    if unknown:
        raise TypeError(f"unsupported parameter(s): {', '.join(unknown)}")
    done = _run_completion(list(messages), max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                           only=only, prefer=prefer, tools=tools, tool_choice=tool_choice,
                           params={k: v for k, v in params.items() if v is not None})
    choice = done.response.choices[0]
    tool_calls = [{"id": tc.id, "type": "function",
                   "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                  for tc in (getattr(choice.message, "tool_calls", None) or [])] if tools else []
    message: dict = {"role": "assistant", "content": done.text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = getattr(done.response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "id": f"chatcmpl-{random.getrandbits(64):016x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": done.model,
        "choices": [{"index": 0, "message": message, "finish_reason": choice.finish_reason or "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": getattr(usage, "total_tokens", None) or prompt_tokens + completion_tokens},
        "x_ricochet": {"provider": done.provider, "model": done.model, "mode": done.mode,
                       "latency_ms": done.latency_ms, "trail": done.trail},
    }


# ============================================================
# DIAGNOSTIC: which listed models REALLY answer?
# ============================================================

# Models that are not text (embeddings, reranking, vision only, speech, images, biology…): left out of the probe.
_NON_TEXT = re.compile(
    r"embed|rerank|vision|clip|paligemma|neva|fuyu|parse|safety|guard|riva|asr|tts|translate|reward|retriev|diffusion|"
    r"cosmos|sdxl|flux|stable|genmol|molmim|esm|bionemo|alphafold|deplot|kosmos|llava|vila|pii|whisper|parakeet|"
    r"calibration|synthetic|wan-ai|proteina", re.IGNORECASE)


def _probe_outcome(e: Exception) -> str:
    if isinstance(e, openai.NotFoundError):
        return "404 not callable"            # listed by the provider but "Function not found"
    if isinstance(e, openai.APITimeoutError):
        return "timeout"
    if isinstance(e, openai.RateLimitError):
        return "429 quota"
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "key or access denied"
    if isinstance(e, openai.APIStatusError):
        return f"HTTP {getattr(e, 'status_code', '?')}"
    return type(e).__name__


def probe_models(name: str, *, timeout: float = 40.0, include_all: bool = False, workers: int = 4,
                 prompt: str = "Reply with the word ok.", max_tokens: int = 40) -> List[dict]:
    """Probe each model listed by `GET /models` with a trivial request: which one really answers?

    A free provider's catalog is not reliable: being LISTED guarantees nothing (NVIDIA: 32 models out of 47 return
    404, others time out). Diagnostic only: touches neither pauses, nor statistics, nor learned quotas.
    Each row: model, status (ok | reasoning | 404 not callable | timeout | …), latency_s, content_chars,
    reasoning (the model writes some reasoning). "reasoning" = empty answer at this small max_tokens: its reasoning
    must be turned off (`model_options`) or max_tokens raised before using it."""
    from concurrent.futures import ThreadPoolExecutor

    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    client = provider["client"].with_options(timeout=float(timeout))
    try:
        listed = sorted(m.id for m in client.models.list().data)
    except Exception:
        listed = _models_for(provider, "HEAVY")          # list unavailable: at least the catalog models
    models = listed if include_all else [m for m in listed if not _NON_TEXT.search(m)]

    def probe(model: str) -> dict:
        t0 = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0,
                **_call_options(provider, model))
            response = raw.parse()
        except Exception as e:                           # noqa: BLE001  diagnostic: every error is a result
            return {"model": model, "status": _probe_outcome(e), "latency_s": round(time.perf_counter() - t0, 1),
                    "content_chars": 0, "reasoning": False}
        message = response.choices[0].message if response.choices else None
        text = ((message.content if message else "") or "").strip()
        reasoning = bool(message and (getattr(message, "reasoning_content", None)
                                      or (getattr(message, "model_extra", None) or {}).get("reasoning_content")))
        return {"model": model, "status": "ok" if text else ("reasoning" if reasoning else "empty response"),
                "latency_s": round(time.perf_counter() - t0, 1), "content_chars": len(text), "reasoning": reasoning}

    with ThreadPoolExecutor(max_workers=max(1, min(workers, provider.get("max_concurrency", 2) + 2))) as pool:
        rows = list(pool.map(probe, models))
    order = {"ok": 0, "reasoning": 1}
    return sorted(rows, key=lambda r: (order.get(r["status"], 2), r["latency_s"], r["model"]))
