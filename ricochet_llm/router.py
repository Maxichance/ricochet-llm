"""
Moteur du routeur LLM multi-providers free tier.
Choisit un provider, bascule sur quota épuisé / panne, exécute la boucle agentique (tool calling),
et tient l'état observable (quotas, cooldowns, statistiques, journal) affiché par l'interface.

Données pures        : catalog.py (providers), cleaning.py (nettoyage)
Partie métier        : tools.py (registre d'outils) + set_system_prompt(), chargés via plugins.py
"""
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date
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
    """Aucun provider n'a pu répondre (ou boucle d'outils trop longue)."""


# ============================================================
# RÉGLAGES
# ============================================================

MAX_TOOL_ROUNDS       = 8      # nombre max de tours « le modèle appelle des outils »
TRANSIENT_RETRIES     = 1      # nouvelle tentative sur le même provider (erreur réseau / 5xx)
TRANSIENT_BACKOFF_S   = 0.5    # base du backoff exponentiel (avec jitter)

COOLDOWN_RATE_LIMIT_S = 60     # 429 sans en-tête Retry-After
COOLDOWN_TRANSIENT_S  = 30     # provider qui enchaîne les erreurs réseau / 5xx
COOLDOWN_CONFIG_S     = 900    # 401 / 403 / 404 : clé ou modèle à corriger, inutile de réessayer vite
COOLDOWN_PAYMENT_S    = 3600   # 402, limite à zéro, plafond JOURNALIER atteint : ne revient pas avant des heures
COOLDOWN_MAX_S        = 3600   # plafond de tout cooldown (y compris Retry-After)
COOLDOWN_SIZE_S       = 1800   # un modèle trop petit pour la demande (413) : sa limite ne change pas en 30 min
COOLDOWN_EMPTY_S      = 300    # un modèle qui rend une réponse vide (raisonnement sans réponse) : on passe au suivant
SIZE_MARGIN           = 0.9    # l'estimation de jetons surestime légèrement : on garde une marge
QUOTA_STALE_S         = 300    # un quota lu depuis plus de 5 min est considéré « inconnu » et re-sondé
PROBE_MAX_AGE_S       = 60     # âge max d'une sonde de quota (ex. OpenRouter /key) avant relecture
EVENTS_KEPT           = 200

DEFAULT_SYSTEM_PROMPT = "Tu es un assistant. Tu réponds en français, de façon claire et structurée."
_system_prompt = DEFAULT_SYSTEM_PROMPT


def set_system_prompt(text: str) -> None:
    """Définit le prompt système envoyé à chaque appel."""
    global _system_prompt
    _system_prompt = text


def get_system_prompt() -> str:
    return _system_prompt


# ============================================================
# PROVIDERS ACTIFS
# ============================================================

_RE_VARIABLE = re.compile(r"\{([A-Z0-9_]+)\}")


def _env_value(spec: dict, variable: str) -> str:
    """Valeur d'une variable d'environnement du provider, sous son nom ou sous l'un de ses alias (`env_aliases`) :
    la faute de frappe `CLOUDFLAIRE_ACCOUNT_ID` existe dans la nature."""
    for name in (variable, *(spec.get("env_aliases") or {}).get(variable, ())):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _is_configured(spec: dict) -> bool:
    """La clé ET les variables que l'adresse exige (identifiant de compte) sont renseignées."""
    return bool(_env_value(spec, spec["env_key"])) and all(_env_value(spec, v) for v in spec.get("env_required") or ())


def _resolve_base_url(spec: dict) -> str:
    return _RE_VARIABLE.sub(lambda m: _env_value(spec, m.group(1)), spec["base_url"])


def _build_providers() -> list:
    providers = []
    for spec in PROVIDER_SPECS:
        if not _is_configured(spec):
            logger.info("LLM %s ignoré : %s absente", spec["name"], spec["env_key"])
            continue
        provider = dict(spec)
        provider["expires_at"] = (os.getenv(spec["env_key"] + "_EXPIRES") or "").strip() or None
        # max_retries=0 : le SDK ne doit pas re-tenter (et dormir) sur un 429, c'est le routeur qui bascule.
        provider["client"] = OpenAI(
            base_url=_resolve_base_url(spec), api_key=_env_value(spec, spec["env_key"]), max_retries=0,
            timeout=spec.get("timeout", REQUEST_TIMEOUT_S),
        )
        providers.append(provider)
    return providers


def _disabled_specs() -> list:
    return [dict(s) for s in PROVIDER_SPECS if not _is_configured(s)]


_PROVIDERS = _build_providers()
_DISABLED  = _disabled_specs()      # providers du catalogue sans clé : affichés « désactivé » par l'interface

_now   = time.monotonic             # indirections pour les tests
_sleep = time.sleep


def _provider_by_name(name: str) -> Optional[dict]:
    return next((p for p in _PROVIDERS if p["name"] == name), None)


# ============================================================
# ÉTAT OBSERVABLE (quotas, cooldowns, statistiques, journal)
# ============================================================

_quota_lock = threading.Lock()
_provider_remaining:      Dict[str, Optional[int]] = {}
_provider_seen:           Dict[str, float] = {}   # date de la dernière lecture du quota
_provider_cooldown_until: Dict[str, float] = {}   # 0 = pas de cooldown
_provider_last_error:     Dict[str, Optional[str]] = {}
_provider_headers:        Dict[str, dict] = {}    # derniers en-têtes x-ratelimit-* : {"at": t, "values": {...}}
_provider_probe:          Dict[str, dict] = {}    # dernière sonde de quota (OpenRouter /key…)
_provider_stats:          Dict[str, dict] = {}
_calls_window:            Dict[str, deque] = {}   # horodatages des appels (pour « appels / minute »)
# ÉTAT PAR MODÈLE. Un 429 chez Groq écarte UN modèle (chacun a son budget de jetons par minute), pas tout le provider ;
# un 413 apprend la limite d'un modèle. Le provider n'est en pause que lorsque plus aucun de ses modèles ne peut répondre.
_model_cooldown_until:    Dict[Tuple[str, str], float] = {}
_model_limit:             Dict[Tuple[str, str], int] = {}
_model_budget:            Dict[Tuple[str, str], Tuple[int, float]] = {}   # jetons restants, instant de remise à zéro

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


def _record_event(provider: str, kind: str, message: str) -> None:
    with _events_lock:
        _events.append({"t": time.time(), "provider": provider, "kind": kind, "message": message[:240]})


def recent_events(limit: int = 60) -> List[dict]:
    """Journal des événements, le plus récent d'abord."""
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
    """Mémorise les en-têtes de quota (affichage) et, si `routing`, le quota qui pilote le choix du provider."""
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
    """« 46.5s », « 1m30.5s », « 250ms », « 2m0s », « 12 » -> secondes. None si illisible."""
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
    """Retient le budget de jetons annoncé par le provider (`x-ratelimit-*-tokens`) : on sait ainsi, AVANT d'appeler,
    qu'un modèle n'a plus de quoi traiter la demande (pas de 429 évitable, pas de seconde perdue). La limite annoncée
    devient la plus grande demande acceptable : au-delà, elle n'aboutira jamais, même à vide."""
    if not headers:
        return
    key = (provider_name, model)
    limit = headers.get("x-ratelimit-limit-tokens")
    remaining = headers.get("x-ratelimit-remaining-tokens")
    delay = _duration_text(headers.get("x-ratelimit-reset-tokens"))
    with _quota_lock:
        try:
            if limit is not None:
                _model_limit[key] = int(float(limit))
            if remaining is not None and delay is not None:
                _model_budget[key] = (int(float(remaining)), _now() + delay)
        except (TypeError, ValueError):
            return


def _put_in_cooldown(provider_name: str, seconds: float, exhausted: bool = False, reason: str = "") -> None:
    seconds = max(1.0, min(float(seconds), COOLDOWN_MAX_S))
    with _quota_lock:
        _provider_cooldown_until[provider_name] = _now() + seconds
        if exhausted:
            _provider_remaining[provider_name] = 0
            _provider_seen[provider_name] = _now()
        if reason:
            _provider_last_error[provider_name] = reason[:200]
    logger.warning("LLM %s indisponible %.0fs (%s)", provider_name, seconds, reason[:120])


def _mark_provider_exhausted(provider_name: str, cooldown: float = COOLDOWN_RATE_LIMIT_S, reason: str = "") -> None:
    _put_in_cooldown(provider_name, cooldown, exhausted=True, reason=reason)


def _clear_cooldown(provider_name: str) -> None:
    with _quota_lock:
        _provider_cooldown_until[provider_name] = 0.0
        _provider_remaining[provider_name] = None


def _refresh_locked(now: float) -> None:
    """Fin de cooldown ou quota périmé → quota « inconnu » (le provider est re-sondé). Appeler sous verrou."""
    for p in _PROVIDERS:
        name = p["name"]
        until = _provider_cooldown_until[name]
        if until and until <= now:
            _provider_cooldown_until[name] = 0.0
            _provider_remaining[name] = None
        elif _provider_remaining[name] is not None and now - _provider_seen[name] > QUOTA_STALE_S:
            _provider_remaining[name] = None


def _models_for(provider: dict, mode: str) -> List[str]:
    """Modèles essayables, par ordre. HEAVY : le lourd puis les modèles de repli DÉCLARÉS (`extra_models`) ; le léger n'en
    fait partie que s'il y est listé. LIGHT (quota bas) : le léger seul."""
    if mode == "LIGHT":
        return [provider["model_light"]]
    out: List[str] = []
    for m in [provider["model_heavy"], *(provider.get("extra_models") or [])]:
        if m and m not in out:
            out.append(m)
    return out


def _estimate_tokens(messages: list) -> int:
    """Estimation grossière des jetons d'entrée (~3 caractères par jeton en français)."""
    return sum(len(str(m.get("content") or "")) for m in messages) // 3 + 4 * len(messages)


def _model_wait(provider: dict, model: str, need_tokens: int = 0, now: Optional[float] = None) -> float:
    """Secondes avant que ce modèle soit utilisable pour une demande de `need_tokens` jetons, 0 s'il l'est maintenant."""
    now = _now() if now is None else now
    key = (provider["name"], model)
    with _quota_lock:
        limit = _model_limit.get(key) or (provider.get("token_limits") or {}).get(model)
        until = _model_cooldown_until.get(key, 0.0)
        budget = _model_budget.get(key)
        if budget is not None and now >= budget[1]:
            _model_budget.pop(key, None)       # la fenêtre est repartie à zéro : le budget annoncé n'a plus cours
            budget = None
    if limit and need_tokens * SIZE_MARGIN > limit:
        return float(COOLDOWN_SIZE_S)          # la demande ne passera pas, inutile d'appeler pour recevoir un 413
    wait = max(0.0, until - now)
    if budget is not None and budget[0] < need_tokens:
        wait = max(wait, budget[1] - now)      # le budget de jetons annoncé ne suffit pas : attendre sa remise à zéro
    return wait


def _first_usable_model(provider: dict, mode: str, need_tokens: int, now: float) -> Optional[str]:
    return next((m for m in _models_for(provider, mode) if _model_wait(provider, m, need_tokens, now) <= 0), None)


def _soonest_model_return(provider: dict, need_tokens: int = 0) -> float:
    """Secondes avant que le premier modèle du provider redevienne utilisable (au moins 1)."""
    waits = [_model_wait(provider, m, need_tokens) for m in _models_for(provider, "HEAVY")]
    return max(1.0, min(waits)) if waits else float(COOLDOWN_RATE_LIMIT_S)


def _bench_model(provider: dict, model: str, seconds: float, reason: str) -> None:
    seconds = max(1.0, min(float(seconds), COOLDOWN_MAX_S))
    with _quota_lock:
        _model_cooldown_until[(provider["name"], model)] = _now() + seconds
        _provider_last_error[provider["name"]] = reason[:200]
    _record_event(provider["name"], "model_paused", f"{model} écarté {seconds:.0f}s : {reason}"[:240])
    logger.warning("LLM %s / %s écarté %.0fs (%s)", provider["name"], model, seconds, reason[:120])


def _other_model_usable(provider: dict, model: str, need_tokens: int = 0) -> bool:
    return any(m != model and _model_wait(provider, m, need_tokens) <= 0 for m in _models_for(provider, "HEAVY"))


def _call_options(provider: dict, model: str) -> dict:
    """Réglages propres à ce MODÈLE (`model_options`) : `extra_body` pour couper le raisonnement des Nemotron."""
    opt = (provider.get("model_options") or {}).get(model) or {}
    return {"extra_body": opt["extra_body"]} if opt.get("extra_body") else {}


def _choose_provider_and_model(exclude=(), require_tools: bool = False, need_tokens: int = 0,
                               only=None, prefer: Optional[str] = None) -> Tuple[dict, str, str]:
    """HEAVY : 1er provider dont le quota est suffisant. LIGHT : celui qui a le plus de quota restant.
    `exclude` = providers déjà essayés pendant cet appel ; `require_tools` = la conversation contient des tool calls ;
    `need_tokens` = taille estimée de la demande (un modèle trop petit pour elle est sauté) ; `only` = noms de
    providers permis ; `prefer` = provider essayé en premier s'il est éligible.
    Dans un provider, le premier modèle utilisable : un modèle écarté ne met pas tout le provider de côté."""
    if not _PROVIDERS:
        raise LLMRouterError("Aucun provider configuré : renseignez au moins une clé API dans .env")
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
    prefer_tools = len(registry) > 0          # sans outils enregistrés, l'ordre du catalogue suffit
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
    hint = f" Réessayez dans ~{int(min(waits)) + 1}s." if waits else ""
    causes = "; ".join(f"{p['name']} → {last_err[p['name']]}" for p in _PROVIDERS if last_err[p["name"]])
    raise LLMRouterError("Aucun provider disponible." + hint + (f" Dernières erreurs : {causes}" if causes else ""))


def available_providers(only=None, need_tokens: int = 0) -> List[str]:
    """Noms des providers qui peuvent répondre MAINTENANT (hors pause, au moins un modèle utilisable). Sert à
    l'exécuteur parallèle : un provider en pause ne doit pas prendre de travail."""
    now = _now()
    with _quota_lock:
        _refresh_locked(now)
        cooldown = dict(_provider_cooldown_until)
    return [p["name"] for p in _PROVIDERS
            if cooldown[p["name"]] <= now and (only is None or p["name"] in only)
            and _first_usable_model(p, "HEAVY", need_tokens, now) is not None]


def next_provider(require_tools: bool = False) -> dict:
    """Quel provider/modèle serait choisi maintenant (sans rien appeler)."""
    try:
        provider, model, mode = _choose_provider_and_model(require_tools=require_tools)
        return {"provider": provider["name"], "label": provider.get("label", provider["name"]),
                "model": model, "mode": mode}
    except LLMRouterError as e:
        return {"provider": None, "error": str(e)}


# ----- sondes de quota hors en-têtes ---------------------------------------------------------

def _probe_openrouter(provider: dict) -> dict:
    data = provider["client"].get("/key", cast_to=object)
    data = data.get("data", data) if isinstance(data, dict) else {}
    limits = []
    free = data.get("free_model_daily_requests")
    if isinstance(free, dict) and free.get("limit") is not None:
        limits.append({"label": "requêtes gratuites / jour", "remaining": free.get("remaining"),
                       "limit": free.get("limit"), "source": "API /key"})
    return {"limits": limits, "expires_at": data.get("expires_at")}


_PROBES = {"openrouter": _probe_openrouter}


def refresh_probes(force: bool = False) -> None:
    """Relit les quotas qui ne passent pas par des en-têtes (bloquant : à appeler hors de la boucle d'événements)."""
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
    """Transforme les paires x-ratelimit-remaining-X / x-ratelimit-limit-X en jauges lisibles."""
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
        label = labels.get(suffix) or suffix.replace("-", " ") or "requêtes"
        out.append({"label": label, "remaining": remaining, "limit": limit, "source": "en-tête"})
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
    """Snapshot lisible de chaque provider (statut, quotas, statistiques) — consommé par l'interface et /api/status."""
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
# ERREURS : classification et réaction
# ============================================================

def _classify_error(e: Exception) -> str:
    """rate_limit | payment | transient | config | rejected | fatal"""
    if isinstance(e, openai.RateLimitError):
        return "rate_limit"
    if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", None) == 402:
        return "payment"                       # offre non activée ou modèle d'un plan payant : ne se règle pas en une minute
    if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):  # inclut APITimeoutError
        return "transient"
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return "config"
    if isinstance(e, openai.APIStatusError):   # 400, 410, 413, 422… : ce provider refuse cette requête
        return "rejected"
    err = str(e).lower()
    if "429" in err or "rate_limit" in err or "quota" in err:
        return "rate_limit"
    return "fatal"                             # bug local (TypeError, KeyError…) : on ne masque pas


def _describe_error(e: Exception) -> str:
    """Message d'erreur lisible : « RateLimitError 429: … » avec la vraie raison extraite du corps de la réponse."""
    body = getattr(e, "body", None)
    if isinstance(body, list) and body:        # Gemini renvoie une liste
        body = body[0]
    msg = None
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        msg = err.get("message") or err.get("detail") or body.get("detail")
        raw = (err.get("metadata") or {}).get("raw") if isinstance(err.get("metadata"), dict) else None
        if raw and (not msg or msg == "Provider returned error"):   # OpenRouter : la cause est dans metadata.raw
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
                pass   # Retry-After au format date HTTP : ignoré
    return None


def _response_header(e: Exception, name: str) -> Optional[str]:
    headers = getattr(getattr(e, "response", None), "headers", None)
    return headers.get(name) if headers else None


def _limit_is_zero(e: Exception) -> bool:
    """Une limite par minute à ZÉRO veut dire « offre gratuite non activée » (clé valide mais sans droit de chat) :
    inutile de réessayer avant longtemps."""
    value = _response_header(e, "x-ratelimit-limit-req-minute")
    return value is not None and str(value).strip() == "0"


_DAILY_MARKERS = ("perday", "per day", "per-day", "daily", "par jour", "quotidien")


def _is_daily_cap(e: Exception) -> bool:
    """Le 429 vient-il d'une allocation QUOTIDIENNE épuisée (Gemini : `GenerateRequestsPerDayPerProjectPerModel`,
    Cloudflare : « daily free allocation », OpenRouter : `free-models-per-day`) ? Elle ne revient pas avant des heures :
    la retenter chaque minute ne fait que perdre un appel."""
    return any(marker in f"{_describe_error(e)} {getattr(e, 'body', '')}".lower() for marker in _DAILY_MARKERS)


def _rate_limit_delay(e: Exception) -> float:
    """Combien de temps écarter après un 429 : `Retry-After`, sinon la remise à zéro du budget de jetons annoncée par
    le provider, sinon une heure pour un plafond journalier, sinon la pause par défaut."""
    return (_retry_after_seconds(e)
            or _duration_text(_response_header(e, "x-ratelimit-reset-tokens"))
            or (COOLDOWN_PAYMENT_S if _is_daily_cap(e) else COOLDOWN_RATE_LIMIT_S))


_RE_LIMIT_413 = re.compile(r"Limit (\d+)")


def _limit_from_413(e: Exception) -> Optional[int]:
    """« Request too large ... tokens per minute (TPM): Limit 8000, Requested 8285 » -> 8000."""
    found = _RE_LIMIT_413.search(_describe_error(e))
    return int(found.group(1)) if found else None


def _handle_call_failure(provider: dict, e: Exception, excluded: set, retries: Dict[str, int],
                         model: Optional[str] = None, need_tokens: int = 0) -> str:
    """Réagit à une exception d'appel et retourne sa catégorie. Si le provider n'est pas ajouté à `excluded`,
    l'appelant doit re-tenter le même ; lève si l'erreur est un bug local.

    Retourne « model_limited » quand SEUL le modèle a été écarté et qu'un autre modèle du même provider peut répondre :
    le provider reste en jeu, l'appelant choisit un autre modèle (un seul 413 ne doit pas écarter tout un provider
    alors que ses autres modèles sont disponibles)."""
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
            delay = _soonest_model_return(provider, need_tokens)      # le provider revient avec son premier modèle
        _mark_provider_exhausted(name, delay, reason)
        excluded.add(name)
    elif kind == "transient":
        if retries.get(name, 0) < TRANSIENT_RETRIES:
            attempt = retries[name] = retries.get(name, 0) + 1
            delay = TRANSIENT_BACKOFF_S * (2 ** (attempt - 1)) * (0.5 + random.random())
            logger.warning("LLM %s erreur transitoire (%s) → nouvelle tentative dans %.1fs", name, reason, delay)
            _sleep(delay)
            return kind                 # même provider
        _put_in_cooldown(name, COOLDOWN_TRANSIENT_S, reason=reason)
        excluded.add(name)
    elif kind == "config":
        if multi and status == 404:     # un modèle retiré du catalogue ne dit rien des autres
            _bench_model(provider, model, COOLDOWN_CONFIG_S, reason)
            if _other_model_usable(provider, model, need_tokens):
                return "model_limited"
        _put_in_cooldown(name, COOLDOWN_CONFIG_S, reason=reason)
        excluded.add(name)
    else:   # rejected : propre à cette requête
        if status == 413 and model is not None:
            limit = _limit_from_413(e)
            if limit is not None:
                with _quota_lock:
                    _model_limit[(name, model)] = limit
                _bench_model(provider, model, COOLDOWN_SIZE_S, f"demande plus grande que la limite de {limit} jetons (413)")
                if _other_model_usable(provider, model, need_tokens):
                    return "model_limited"
        logger.warning("LLM %s a refusé la requête (%s) → bascule", name, reason)
        with _quota_lock:
            _provider_last_error[name] = reason
        excluded.add(name)
    return kind


def _handle_empty_answer(provider: dict, model: str, excluded: set, need_tokens: int = 0) -> None:
    """Réponse vide : un modèle qui raisonne a dépensé tous ses jetons avant d'écrire. Passe au modèle suivant du
    provider s'il en a un utilisable, sinon écarte le provider pour cet appel."""
    models = _models_for(provider, "HEAVY")
    if model in models and len(models) > 1:
        _bench_model(provider, model, COOLDOWN_EMPTY_S, "réponse vide (raisonnement sans réponse ?)")
        if _other_model_usable(provider, model, need_tokens):
            return
    excluded.add(provider["name"])


# ============================================================
# ACTIONS MANUELLES (interface)
# ============================================================

def ping_provider(name: str) -> dict:
    """Appel minimal sur le modèle léger : vérifie la clé, mesure la latence, lève le cooldown si ça répond."""
    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    model = provider["model_light"]
    t0 = time.perf_counter()
    try:
        raw = provider["client"].chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "Réponds uniquement : OK"}],
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
    """Lève le cooldown et oublie le quota : le provider est de nouveau éligible tout de suite."""
    if _provider_by_name(name) is None:
        raise KeyError(name)
    _clear_cooldown(name)
    _record_event(name, "info", "cooldown levé manuellement")


# ============================================================
# BOUCLE AGENTIQUE
# ============================================================

def _strip_tool_artifacts(text: str) -> str:
    return cleaning.strip_tool_artifacts(cleaning.strip_think(text), registry.names())


@dataclass
class ChatResult:
    """Réponse + ce qui s'est passé pour l'obtenir."""
    answer: str
    provider: str
    model: str
    mode: str
    latency_ms: int
    tool_rounds: int = 0
    trail: List[dict] = field(default_factory=list)   # providers essayés avant celui qui a répondu

    def to_dict(self) -> dict:
        return asdict(self)


def chat(conversation_history: list, user_message: str) -> str:
    """Ajoute user_message à l'historique, exécute la boucle (tool calls inclus), retourne la réponse.
    En cas d'échec, l'historique est remis dans son état d'avant l'appel (on peut renvoyer le même message)."""
    return chat_detailed(conversation_history, user_message).answer


def chat_detailed(conversation_history: list, user_message: str) -> ChatResult:
    """Comme `chat`, mais retourne aussi le provider/modèle utilisé, la latence et les bascules."""
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

    excluded: set = set()            # providers abandonnés pour CET appel (garantit que la boucle se termine)
    retries: Dict[str, int] = {}
    trail: List[dict] = []
    tool_rounds = 0
    nudged = False
    in_tool_exchange = False         # des tool calls sont en cours : seuls les providers avec tools peuvent continuer

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
                continue                                  # nouvelle tentative sur le même provider et le même modèle
            provider, model, mode = _choose_provider_and_model(excluded, in_tool_exchange, need)
            logger.info("LLM bascule → [%s] %s / %s", mode, provider["name"], model)
            continue

        with _quota_lock:
            rem = _provider_remaining.get(provider["name"])
        if rem is not None:
            logger.info("LLM [%s] quota restant : %s tokens", provider["name"], rem)

        if not response.choices:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            _record_event(provider["name"], "no_choices", "réponse sans choices")
            trail.append({"provider": provider["name"], "outcome": "no_choices", "detail": "réponse sans choices"})
            _mark_provider_exhausted(provider["name"], COOLDOWN_RATE_LIMIT_S, "réponse sans choices")
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
                logger.warning("LLM %s : réponse vide après nettoyage → bascule", provider["name"])
                _record_event(provider["name"], "empty", "réponse vide après nettoyage")
                trail.append({"provider": provider["name"], "model": model, "outcome": "empty", "detail": "réponse vide"})
                _handle_empty_answer(provider, model, excluded, need)
                provider, model, mode = _choose_provider_and_model(excluded, in_tool_exchange, need)
                if not nudged:
                    messages.append({"role": "user", "content": "Réponds à ma question précédente."})
                    nudged = True
                continue
            conversation_history.append({"role": "assistant", "content": cleaned})
            latency_ms = int((time.perf_counter() - t_start) * 1000)
            _record_event(provider["name"], "ok", f"{model} · {mode} · {latency_ms} ms")
            return ChatResult(answer=cleaned, provider=provider["name"], model=model, mode=mode,
                              latency_ms=latency_ms, tool_rounds=tool_rounds, trail=trail)

        tool_rounds += 1
        if tool_rounds > MAX_TOOL_ROUNDS:
            raise LLMRouterError(f"Plus de {MAX_TOOL_ROUNDS} tours d'outils sans réponse finale : abandon.")
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
# APPEL ISOLÉ (sans conversation ni outils) : lectures par lot, extraction, classification…
# ============================================================

def provider_names(only=None) -> List[str]:
    """Noms des providers configurés (clé présente), dans l'ordre de préférence."""
    return [p["name"] for p in _PROVIDERS if only is None or p["name"] in only]


def provider_settings(name: str) -> dict:
    """Réglages de lecture par lot d'un provider : requêtes simultanées au plus, écart entre deux départs."""
    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    return {"max_concurrency": provider.get("max_concurrency", 2), "min_interval_s": provider.get("min_interval_s", 0.0)}


@dataclass
class GenerateResult:
    """Réponse d'un appel isolé, avec ce qui s'est passé pour l'obtenir."""
    text: str
    provider: str
    model: str
    mode: str
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    finish_reason: Optional[str] = None
    truncated: bool = False            # la réponse s'est arrêtée sur `max_tokens` : un JSON peut être coupé
    reasoning_used: bool = False       # le modèle a écrit un raisonnement (reasoning_content)
    trail: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _client_for(provider: dict, timeout: Optional[float]):
    """Client du provider, avec un délai ALLONGÉ pour cet appel. Il ne raccourcit jamais celui du provider : le délai
    du catalogue est calibré pour le chat (20 s chez NVIDIA), pas pour une longue réponse JSON."""
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
    """Un appel sans conversation : bascule entre providers ET entre modèles, comme `chat`, sans outils.

    prompt       texte, ou liste de messages {"role", "content"}
    max_tokens   plafond de sortie. Il INCLUT le raisonnement des modèles qui raisonnent : trop bas, la réponse est vide
    timeout      secondes accordées à chaque provider POUR CET APPEL (allonge celui du catalogue, ne le raccourcit pas)
    only         noms de providers permis (None = tous)
    prefer       provider essayé en premier s'il est éligible (répartition de charge : voir parallel.py)

    Une réponse VIDE n'est jamais renvoyée : le modèle est écarté et le suivant essayé. Lève LLMRouterError si plus rien
    ne peut répondre. Le contrôle du CONTENU (JSON valide, citations retrouvées dans la source) reste à l'appelant :
    un JSON valide mais vide ne se distingue pas d'une absence réelle."""
    t_start = time.perf_counter()
    messages = _messages_for(prompt, system)
    need = _estimate_tokens(messages)
    excluded: set = set()
    retries: Dict[str, int] = {}
    trail: List[dict] = []
    provider, model, mode = _choose_provider_and_model(excluded, need_tokens=need, only=only, prefer=prefer)

    while True:
        t_call = time.perf_counter()
        try:
            raw = _client_for(provider, timeout).chat.completions.with_raw_response.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                **_call_options(provider, model))
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
            provider, model, mode = _choose_provider_and_model(excluded, need_tokens=need, only=only, prefer=prefer)
            continue

        if not response.choices:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            _record_event(provider["name"], "no_choices", "réponse sans choices")
            trail.append({"provider": provider["name"], "model": model, "outcome": "no_choices", "detail": "réponse sans choices"})
            _mark_provider_exhausted(provider["name"], COOLDOWN_RATE_LIMIT_S, "réponse sans choices")
            excluded.add(provider["name"])
            provider, model, mode = _choose_provider_and_model(excluded, need_tokens=need, only=only, prefer=prefer)
            continue

        choice = response.choices[0]
        message = choice.message
        text = cleaning.strip_think(message.content or "")
        reasoning = bool(getattr(message, "reasoning_content", None)
                         or (getattr(message, "model_extra", None) or {}).get("reasoning_content"))
        if not text:
            _record_call_result(provider["name"], False, time.perf_counter() - t_call)
            why = ("max_tokens atteint pendant le raisonnement" if choice.finish_reason == "length" and reasoning
                   else "raisonnement sans réponse" if reasoning else "réponse vide")
            _record_event(provider["name"], "empty", f"{model} : {why}")
            trail.append({"provider": provider["name"], "model": model, "outcome": "empty", "detail": why})
            _handle_empty_answer(provider, model, excluded, need)
            provider, model, mode = _choose_provider_and_model(excluded, need_tokens=need, only=only, prefer=prefer)
            continue

        usage = getattr(response, "usage", None)
        _record_call_result(provider["name"], True, time.perf_counter() - t_call, _usage_tokens(response))
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        _record_event(provider["name"], "ok", f"{model} · {mode} · {latency_ms} ms")
        return GenerateResult(
            text=text, provider=provider["name"], model=model, mode=mode, latency_ms=latency_ms,
            tokens_in=getattr(usage, "prompt_tokens", None), tokens_out=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason, truncated=choice.finish_reason == "length",
            reasoning_used=reasoning, trail=trail)


# ============================================================
# DIAGNOSTIC : quels modèles listés répondent VRAIMENT ?
# ============================================================

# Modèles qui ne sont pas du texte (embeddings, reranking, vision seule, parole, images, biologie…) : écartés du sondage.
_NON_TEXT = re.compile(
    r"embed|rerank|vision|clip|paligemma|neva|fuyu|parse|safety|guard|riva|asr|tts|translate|reward|retriev|diffusion|"
    r"cosmos|sdxl|flux|stable|genmol|molmim|esm|bionemo|alphafold|deplot|kosmos|llava|vila|pii|whisper|parakeet|"
    r"calibration|synthetic|wan-ai|proteina", re.IGNORECASE)


def _probe_outcome(e: Exception) -> str:
    if isinstance(e, openai.NotFoundError):
        return "404 non appelable"           # listé par le provider mais « Function not found »
    if isinstance(e, openai.APITimeoutError):
        return "délai dépassé"
    if isinstance(e, openai.RateLimitError):
        return "429 quota"
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "clé ou droit refusé"
    if isinstance(e, openai.APIStatusError):
        return f"HTTP {getattr(e, 'status_code', '?')}"
    return type(e).__name__


def probe_models(name: str, *, timeout: float = 40.0, include_all: bool = False, workers: int = 4,
                 prompt: str = "Réponds par le mot ok.", max_tokens: int = 40) -> List[dict]:
    """Sonde chaque modèle listé par `GET /models` avec une requête triviale : lequel répond vraiment ?

    Le catalogue d'un provider gratuit n'est pas fiable : être LISTÉ ne garantit rien (NVIDIA : 32 modèles sur 47 en
    404, d'autres expirent). Diagnostic seul : ne touche ni aux pauses, ni aux statistiques, ni aux quotas appris.
    Chaque ligne : model, status (ok | raisonne | 404 non appelable | délai dépassé | …), latency_s, content_chars,
    reasoning (le modèle écrit un raisonnement). « raisonne » = réponse vide à ce petit max_tokens : il faudra couper son
    raisonnement (`model_options`) ou relever max_tokens avant de s'en servir."""
    from concurrent.futures import ThreadPoolExecutor

    provider = _provider_by_name(name)
    if provider is None:
        raise KeyError(name)
    client = provider["client"].with_options(timeout=float(timeout))
    try:
        listed = sorted(m.id for m in client.models.list().data)
    except Exception:
        listed = _models_for(provider, "HEAVY")          # liste indisponible : au moins les modèles du catalogue
    models = listed if include_all else [m for m in listed if not _NON_TEXT.search(m)]

    def probe(model: str) -> dict:
        t0 = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0,
                **_call_options(provider, model))
            response = raw.parse()
        except Exception as e:                           # noqa: BLE001  diagnostic : toute erreur est un résultat
            return {"model": model, "status": _probe_outcome(e), "latency_s": round(time.perf_counter() - t0, 1),
                    "content_chars": 0, "reasoning": False}
        message = response.choices[0].message if response.choices else None
        text = ((message.content if message else "") or "").strip()
        reasoning = bool(message and (getattr(message, "reasoning_content", None)
                                      or (getattr(message, "model_extra", None) or {}).get("reasoning_content")))
        return {"model": model, "status": "ok" if text else ("raisonne" if reasoning else "réponse vide"),
                "latency_s": round(time.perf_counter() - t0, 1), "content_chars": len(text), "reasoning": reasoning}

    with ThreadPoolExecutor(max_workers=max(1, min(workers, provider.get("max_concurrency", 2) + 2))) as pool:
        rows = list(pool.map(probe, models))
    order = {"ok": 0, "raisonne": 1}
    return sorted(rows, key=lambda r: (order.get(r["status"], 2), r["latency_s"], r["model"]))
