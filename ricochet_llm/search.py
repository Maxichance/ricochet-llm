"""
Web search through free APIs, for any agent of the router: models cannot search, they only call the tools the
application gives them. This module provides that tool, with the model router's logic.

    from ricochet_llm import web_search, register_search_tool       # web_search = ricochet_llm.search.search

    r = web_search("comparison of free LLM routers", n=5, content=True)   # never raises: `r.status` says what happened
    for page in r.results:                                        # {"title", "url", "snippet", "content" (if provided)}
        ...
    register_search_tool()    # the `search_web` tool becomes callable by the models of chat()

- **Several providers, one interface.** `search()` tries those whose key is present, in order, fails over on a
  refusal (quota, key, outage) and always returns the same result.
- **Lasting free tiers only.** A provider is listed only if its free offer RENEWS and requires neither a card nor
  billing (official documentation, snapshot of 2026-10-28, see `PROVIDERS`). A trial credit that runs out is not one:
  Jina, Serper and Linkup are excluded, with the reason in docs/WEB_SEARCH.md.
- **The quota is tracked BEFORE it is reached.** The counters (credits, dollars) are written to disk and we stop at
  90% of the free ceiling, instead of depending on what the provider does once exhausted. The real cost is read from
  the response when it provides one. File: `~/.ricochet/search_usage.json` (`RICOCHET_SEARCH_STATE`): in the user's
  home folder, because the quota belongs to the KEY, not to the project using it.
- **No network call to know the state**: `status()` reads the keys, the counters and the pauses.
- **An optional backup engine** (`fallback`) for when no API is usable.

What comes back is never an AI answer: it is pages. An agent that uses it produces a simulation, not a measure of what
a consumer assistant answers.

Variables: EXA_API_KEY, TAVILY_API_KEY, LANGSEARCH_API_KEY (at least one); RICOCHET_SEARCH_ORDER ("tavily,exa");
RICOCHET_SEARCH_STATE; RICOCHET_SEARCH_COUNTRY (Tavily, e.g. "france"); RICOCHET_EXA_TYPE (default "auto").
"""
import calendar
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

SUCCESS, EMPTY, BLOCKED, RATE_LIMITED, ERROR = "SUCCESS", "EMPTY", "BLOCKED", "RATE_LIMITED", "ERROR"

REQUEST_TIMEOUT_S = 20.0
SNIPPET_MAX = 300
CONTENT_MAX = 4000
RESULTS_MAX = 10
QUOTA_MARGIN = 0.9               # stop at 90% of the free quota
COOLDOWN_RATE_LIMIT_S = 60.0
COOLDOWN_TRANSIENT_S = 30.0
COOLDOWN_CONFIG_S = 900.0        # 401 / 403: the key needs fixing
STATE_ENV_VAR = "RICOCHET_SEARCH_STATE"
ORDER_ENV_VAR = "RICOCHET_SEARCH_ORDER"

# Snapshot of 2026-10-28 from each provider's official documentation. These terms change: re-check them.
#   exa        : $10 of credit, reset to $10 on the 1st of each month, "No payment method required"; 402 when exhausted;
#                10 requests per second; the price of a search is in `costDollars` (auto: $7 per 1,000).
#   tavily     : 1,000 credits per month, "No credit card required"; 1 credit per basic search; 432 at the plan ceiling.
#   langsearch : free plan at $0; daily token allowance reset at 00:00 UTC; 5 requests per second;
#                the token ceiling is not published (see the dashboard), so we rely on the 429.
# Terms of use (storage, reuse of results): NOT verified for these three providers, to be read before scaling up
# volume.
PROVIDERS: Dict[str, dict] = {
    "exa": {
        "label": "Exa", "env_key": "EXA_API_KEY", "url": "https://api.exa.ai/search",
        "period": "month", "unit": "dollars", "quota": 10.0, "default_cost": 0.007,
        "note": "$10 of credit per month, no payment method. Terms for reusing results to be read.",
    },
    "tavily": {
        "label": "Tavily", "env_key": "TAVILY_API_KEY", "url": "https://api.tavily.com/search",
        "period": "month", "unit": "credits", "quota": 1000.0, "default_cost": 1.0,
        "note": "1,000 credits per month, no card. Behavior at exhaustion not documented: the local counter stops first.",
    },
    "langsearch": {
        "label": "LangSearch", "env_key": "LANGSEARCH_API_KEY", "url": "https://api.langsearch.com/v1/web-search",
        "period": "day", "unit": "tokens", "quota": None, "default_cost": 0.0,
        "note": "Daily token allowance (00:00 UTC), ceiling visible in the dashboard. No terms page found.",
    },
}
DEFAULT_ORDER = ("exa", "tavily", "langsearch")


@dataclass
class SearchResult:
    status: str                                            # SUCCESS | EMPTY | BLOCKED | RATE_LIMITED | ERROR
    query: str
    results: List[dict] = field(default_factory=list)      # {"title", "url", "snippet", "content" (if the provider gives it)}
    error: Optional[str] = None
    engine: str = ""                                       # name of the provider that answered
    cost: float = 0.0                                      # credits, dollars or tokens depending on the provider

    def to_dict(self) -> dict:
        return asdict(self)


class _Refusal(Exception):
    """The provider returned no results: fail over to the next one."""

    def __init__(self, status: str, reason: str, pause: float, exhausted: bool = False):
        super().__init__(reason)
        self.status, self.reason, self.pause, self.exhausted = status, reason, pause, exhausted


# ───────────────────────── State: counters on disk, pauses in memory ─────────────────────────

_lock = threading.Lock()
_cooldown: Dict[str, float] = {}
_last_error: Dict[str, str] = {}


def state_file() -> Path:
    return Path(os.getenv(STATE_ENV_VAR) or Path.home() / ".ricochet" / "search_usage.json")


def _period_key(spec: dict, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m") if spec["period"] == "month" else now.strftime("%Y-%m-%d")


def _seconds_until_renewal(spec: dict, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    if spec["period"] == "day":
        end = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp() + 86400
    else:
        last_day = calendar.monthrange(now.year, now.month)[1]
        end = datetime(now.year, now.month, last_day, tzinfo=timezone.utc).timestamp() + 86400
    return max(60.0, end - now.timestamp())


def _read_state() -> dict:
    try:
        data = json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: dict) -> None:
    """Atomic write: a half-written counter would lose the whole month's consumption."""
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def used(name: str) -> float:
    """Amount already consumed in the current period (month or day) at this provider."""
    return float(_read_state().get(name, {}).get(_period_key(PROVIDERS[name]), 0.0) or 0.0)


def _count(name: str, amount: float, exhausted: bool = False) -> None:
    """Adds `amount` to the period counter; `exhausted` pushes it to the ceiling (the provider said stop)."""
    spec = PROVIDERS[name]
    with _lock:
        state = _read_state()
        period = _period_key(spec)
        current = float(state.get(name, {}).get(period, 0.0) or 0.0)
        new = max(current + amount, float(spec["quota"])) if exhausted and spec["quota"] else current + amount
        state[name] = {period: round(new, 6)}              # past periods are forgotten: the file stays tiny
        try:
            _write_state(state)
        except OSError:
            logger.warning("search: counter for %s not written to disk", name)


def _quota_reached(name: str) -> bool:
    quota = PROVIDERS[name]["quota"]
    return bool(quota) and used(name) >= quota * QUOTA_MARGIN


def reset() -> None:
    """Resets the state: pauses AND counters (tests)."""
    _cooldown.clear()
    _last_error.clear()
    try:
        state_file().unlink(missing_ok=True)
    except OSError:
        pass


def _key(name: str) -> str:
    return (os.getenv(PROVIDERS[name]["env_key"]) or "").strip()


def _pause(name: str, seconds: float, reason: str) -> None:
    _cooldown[name] = time.monotonic() + max(1.0, seconds)
    _last_error[name] = reason[:200]
    logger.warning("search: %s benched for %.0f s (%s)", name, seconds, reason[:120])


def _paused(name: str) -> bool:
    until = _cooldown.get(name)
    if until is None:
        return False
    if time.monotonic() >= until:
        _cooldown.pop(name, None)
        return False
    return True


def _order() -> List[str]:
    wanted = [n.strip().lower() for n in (os.getenv(ORDER_ENV_VAR) or "").split(",")]
    wanted = [n for n in wanted if n in PROVIDERS]
    return wanted or list(DEFAULT_ORDER)


def available(only=None) -> List[str]:
    """Providers usable now: key present, not paused, free quota not reached."""
    base = [n for n in (only if only is not None else _order()) if n in PROVIDERS]
    return [n for n in base if _key(n) and not _paused(n) and not _quota_reached(n)]


def status() -> dict:
    """State of the providers, for an admin screen. No network call, no key in the response."""
    out = []
    ready = available()
    for name in _order():
        spec = PROVIDERS[name]
        out.append({
            "name": name, "label": spec["label"], "configured": bool(_key(name)), "env_key": spec["env_key"],
            "period": spec["period"], "unit": spec["unit"], "quota": spec["quota"], "used": round(used(name), 4),
            "available": name in ready, "paused": _paused(name), "last_error": _last_error.get(name),
            "note": spec["note"],
        })
    return {"providers": out, "order": _order()}


# ───────────────────────── Adapters: one per provider ─────────────────────────

def _text(value: Any, maxi: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:maxi]


def _page(title: Any, url: Any, snippet: Any, content: Any = "") -> Optional[dict]:
    url = str(url or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    out = {"title": _text(title, 200), "url": url, "snippet": _text(snippet, SNIPPET_MAX)}
    if content:
        out["content"] = _text(content, CONTENT_MAX)
    return out


def _bearer(name: str) -> dict:
    return {"Authorization": "Bearer " + _key(name), "Content-Type": "application/json"}


def _build_exa(query: str, n: int, content: bool) -> Tuple[dict, dict]:
    body: Dict[str, Any] = {"query": query, "numResults": n, "type": os.getenv("RICOCHET_EXA_TYPE") or "auto"}
    if content:
        body["contents"] = {"highlights": True}
    return body, {"x-api-key": _key("exa"), "Content-Type": "application/json"}


def _read_exa(data: dict) -> Tuple[List[dict], float]:
    pages = []
    for r in data.get("results") or []:
        highlights = " ... ".join(h for h in (r.get("highlights") or []) if isinstance(h, str))
        text = highlights or r.get("text") or ""
        page = _page(r.get("title"), r.get("url"), text, text)
        if page:
            pages.append(page)
    cost = data["costDollars"].get("total") if isinstance(data.get("costDollars"), dict) else None
    return pages, float(cost) if isinstance(cost, (int, float)) else PROVIDERS["exa"]["default_cost"]


def _build_tavily(query: str, n: int, content: bool) -> Tuple[dict, dict]:
    body: Dict[str, Any] = {"query": query, "search_depth": "basic", "max_results": n, "include_answer": False,
                            "include_usage": True}
    country = (os.getenv("RICOCHET_SEARCH_COUNTRY") or "").strip().lower()
    if country:
        body["country"] = country
    if content:
        body["include_raw_content"] = "text"
    return body, _bearer("tavily")


def _read_tavily(data: dict) -> Tuple[List[dict], float]:
    pages = []
    for r in data.get("results") or []:
        page = _page(r.get("title"), r.get("url"), r.get("content"), r.get("raw_content"))
        if page:
            pages.append(page)
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    credits = usage.get("credits")
    return pages, float(credits) if isinstance(credits, (int, float)) else PROVIDERS["tavily"]["default_cost"]


def _build_langsearch(query: str, n: int, content: bool) -> Tuple[dict, dict]:
    body: Dict[str, Any] = {"query": query, "count": n}
    if content:
        body["contents"] = {"text": {"maxCharacters": CONTENT_MAX}}
    return body, _bearer("langsearch")


def _read_langsearch(data: dict) -> Tuple[List[dict], float]:
    if data.get("code") not in (None, 200):
        raise _Refusal(ERROR, f"code {data.get('code')}", COOLDOWN_TRANSIENT_S)
    values = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    pages = []
    for r in values:
        # `text` replaces `snippet` when requested: the snippet then comes from the text
        page = _page(r.get("name"), r.get("url"), r.get("snippet") or r.get("text"), r.get("text"))
        if page:
            pages.append(page)
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return pages, float((usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0))


ADAPTERS: Dict[str, Tuple[Callable, Callable]] = {
    "exa": (_build_exa, _read_exa),
    "tavily": (_build_tavily, _read_tavily),
    "langsearch": (_build_langsearch, _read_langsearch),
}


def _retry_after(response: httpx.Response) -> Optional[float]:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _classify(name: str, response: httpx.Response) -> _Refusal:
    """Turns a failing HTTP status into a decision: how long to bench the provider, and why."""
    code = response.status_code
    if code in (401, 403):
        return _Refusal(BLOCKED, f"key refused (HTTP {code})", COOLDOWN_CONFIG_S)
    if code in (402, 432, 433):     # Exa: credits exhausted; Tavily: plan or key ceiling
        return _Refusal(RATE_LIMITED, f"quota exhausted (HTTP {code})", _seconds_until_renewal(PROVIDERS[name]), exhausted=True)
    if code == 429:
        return _Refusal(RATE_LIMITED, "rate limit (HTTP 429)", _retry_after(response) or COOLDOWN_RATE_LIMIT_S)
    return _Refusal(ERROR, f"HTTP {code}", COOLDOWN_TRANSIENT_S)


def _call(name: str, query: str, n: int, content: bool, client: httpx.Client) -> SearchResult:
    build, read = ADAPTERS[name]
    body, headers = build(query, n, content)
    try:
        response = client.post(PROVIDERS[name]["url"], json=body, headers=headers)
    except httpx.TimeoutException:
        raise _Refusal(ERROR, "timeout", COOLDOWN_TRANSIENT_S) from None
    except httpx.HTTPError as exc:
        raise _Refusal(ERROR, f"network: {type(exc).__name__}", COOLDOWN_TRANSIENT_S) from None
    if response.status_code != 200:
        raise _classify(name, response)
    try:
        data = response.json()
        pages, cost = read(data if isinstance(data, dict) else {})
    except ValueError:
        raise _Refusal(ERROR, "unreadable response", COOLDOWN_TRANSIENT_S) from None
    seen: set = set()
    unique = [p for p in pages if not (p["url"] in seen or seen.add(p["url"]))]
    return SearchResult(SUCCESS if unique else EMPTY, query, unique, None, name, cost)


# ───────────────────────── Interface ─────────────────────────

def search(query: str, n: int = 6, content: bool = False, only=None,
           fallback: Optional[Callable[[str, int], SearchResult]] = None,
           client: Optional[httpx.Client] = None) -> SearchResult:
    """One search on the first free provider that answers.

    content   asks, when the provider can give it, for a text excerpt of each page (saves downloading it)
    only      allowed provider names (None = all those whose key is present, in order)
    fallback  function `(query, n)` called when no API is usable; it also returns a `SearchResult`
    client    `httpx.Client` to reuse (otherwise a client is opened for the call)

    Never raises: `status` says what happened."""
    query = " ".join((query or "").split())[:300]
    if not query:
        return SearchResult(ERROR, query, error="empty query")
    n = max(1, min(int(n), RESULTS_MAX))
    last: Optional[_Refusal] = None
    own = client is None
    client = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    try:
        for name in available(only):
            try:
                result = _call(name, query, n, content, client)
            except _Refusal as refusal:
                last = refusal
                _pause(name, refusal.pause, refusal.reason)
                if refusal.exhausted:
                    _count(name, 0.0, exhausted=True)
                continue
            _count(name, result.cost)
            return result
    finally:
        if own:
            client.close()
    if fallback is not None:
        return fallback(query, n)
    if last is not None:
        return SearchResult(last.status, query, error=last.reason)
    return SearchResult(ERROR, query, error="no search engine configured or available")


def search_web(query: str) -> dict:
    """Searches the Web and returns up to 6 results (title, url, snippet, sometimes the page content).

    Args:
        query: Search terms
    """
    r = search(query, n=6, content=True)
    if r.status == SUCCESS:
        return {"results": r.results}
    if r.status == EMPTY:
        return {"results": []}
    return {"error": "search unavailable", "status": r.status}    # never the reason nor the engine name


def register_search_tool(name: str = "search_web", replace: bool = True) -> None:
    """Makes `search_web` callable by the models of `chat()` (registry tool). Of no real use if no search key is
    configured: the tool then answers "search unavailable"."""
    from .tools import registry
    registry.register(search_web, name=name, replace=replace)


__all__ = ["search", "search_web", "register_search_tool", "status", "available", "reset", "used", "SearchResult",
           "PROVIDERS", "SUCCESS", "EMPTY", "BLOCKED", "RATE_LIMITED", "ERROR"]
