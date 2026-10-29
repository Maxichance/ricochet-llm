"""
Recherche Web par API gratuite, pour tout agent du routeur : les modèles ne savent pas chercher, ils n'appellent que les
outils que l'application leur donne. Cette brique fournit cet outil, avec la logique du routeur de modèles.

    from ricochet import web_search, register_search_tool       # web_search = ricochet.search.search

    r = web_search("comparatif des routeurs LLM gratuits", n=5, content=True)   # ne lève jamais : `r.status` dit ce qui s'est passé
    for page in r.results:                                        # {"title", "url", "snippet", "content" (si fourni)}
        ...
    register_search_tool()    # l'outil `search_web` devient appelable par les modèles de chat()

- **Plusieurs providers, une seule interface.** `search()` essaie ceux dont la clé est présente, dans l'ordre, bascule sur
  un refus (quota, clé, panne) et rend toujours le même résultat.
- **Gratuité durable seulement.** Un provider n'entre dans la liste que si son offre gratuite se RENOUVELLE et ne demande ni
  carte ni facturation (documentation officielle, relevé du 28/10/2026, voir `PROVIDERS`). Un crédit d'essai qui s'épuise
  n'en est pas une : Jina, Serper et Linkup sont écartés, avec la raison dans docs/RECHERCHE_WEB.md.
- **Le quota est suivi AVANT d'être atteint.** Les compteurs (crédits, dollars) sont écrits sur disque et on s'arrête à 90 %
  du plafond gratuit, au lieu de dépendre de ce que fait le provider quand il est épuisé. Le coût réel est lu dans la
  réponse quand il la donne. Fichier : `~/.ricochet/search_usage.json` (`RICOCHET_SEARCH_STATE`) : dans le dossier de
  l'utilisateur, car le quota appartient à la CLÉ, pas au projet qui l'utilise.
- **Aucun appel réseau pour connaître l'état** : `status()` lit les clés, les compteurs et les pauses.
- **Un moteur de secours optionnel** (`fallback`) pour quand aucune API n'est utilisable.

Ce qui revient n'est jamais une réponse d'IA : ce sont des pages. Un agent qui s'en sert produit une simulation, pas la
mesure de ce que répond un assistant grand public.

Variables : EXA_API_KEY, TAVILY_API_KEY, LANGSEARCH_API_KEY (au moins une) ; RICOCHET_SEARCH_ORDER (« tavily,exa ») ;
RICOCHET_SEARCH_STATE ; RICOCHET_SEARCH_COUNTRY (Tavily, ex. « france ») ; RICOCHET_EXA_TYPE (défaut « auto »).
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
QUOTA_MARGIN = 0.9               # on s'arrête à 90 % du quota gratuit
COOLDOWN_RATE_LIMIT_S = 60.0
COOLDOWN_TRANSIENT_S = 30.0
COOLDOWN_CONFIG_S = 900.0        # 401 / 403 : clé à corriger
STATE_ENV_VAR = "RICOCHET_SEARCH_STATE"
ORDER_ENV_VAR = "RICOCHET_SEARCH_ORDER"

# Relevé du 28/10/2026 dans la documentation officielle de chaque provider. Ces conditions changent : à revérifier.
#   exa        : 10 $ de crédit, remis à 10 $ le 1er de chaque mois, « No payment method required » ; 402 à l'épuisement ;
#                10 requêtes par seconde ; le prix d'une recherche est dans `costDollars` (auto : 7 $ les 1 000).
#   tavily     : 1 000 crédits par mois, « No credit card required » ; 1 crédit par recherche simple ; 432 au plafond du plan.
#   langsearch : plan gratuit à 0 $ ; allocation de jetons par jour remise à zéro à 00:00 UTC ; 5 requêtes par seconde ;
#                le plafond en jetons n'est pas publié (il se lit dans le tableau de bord), on s'appuie sur le 429.
# Conditions d'utilisation (stockage, réutilisation des résultats) : NON vérifiées pour ces trois providers, à faire lire
# avant une montée en volume.
PROVIDERS: Dict[str, dict] = {
    "exa": {
        "label": "Exa", "env_key": "EXA_API_KEY", "url": "https://api.exa.ai/search",
        "period": "month", "unit": "dollars", "quota": 10.0, "default_cost": 0.007,
        "note": "10 $ de crédit par mois, sans moyen de paiement. Conditions de réutilisation des résultats à lire.",
    },
    "tavily": {
        "label": "Tavily", "env_key": "TAVILY_API_KEY", "url": "https://api.tavily.com/search",
        "period": "month", "unit": "credits", "quota": 1000.0, "default_cost": 1.0,
        "note": "1 000 crédits par mois, sans carte. Comportement à l'épuisement non documenté : le compteur local s'arrête avant.",
    },
    "langsearch": {
        "label": "LangSearch", "env_key": "LANGSEARCH_API_KEY", "url": "https://api.langsearch.com/v1/web-search",
        "period": "day", "unit": "jetons", "quota": None, "default_cost": 0.0,
        "note": "Allocation de jetons par jour (00:00 UTC), plafond visible dans le tableau de bord. Aucune page de conditions trouvée.",
    },
}
DEFAULT_ORDER = ("exa", "tavily", "langsearch")


@dataclass
class SearchResult:
    status: str                                            # SUCCESS | EMPTY | BLOCKED | RATE_LIMITED | ERROR
    query: str
    results: List[dict] = field(default_factory=list)      # {"title", "url", "snippet", "content" (si le provider le donne)}
    error: Optional[str] = None
    engine: str = ""                                       # nom du provider qui a répondu
    cost: float = 0.0                                      # crédits, dollars ou jetons selon le provider

    def to_dict(self) -> dict:
        return asdict(self)


class _Refusal(Exception):
    """Le provider n'a pas rendu de résultats : on bascule sur le suivant."""

    def __init__(self, status: str, reason: str, pause: float, exhausted: bool = False):
        super().__init__(reason)
        self.status, self.reason, self.pause, self.exhausted = status, reason, pause, exhausted


# ───────────────────────── État : compteurs sur disque, pauses en mémoire ─────────────────────────

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
    """Écriture atomique : un compteur à moitié écrit ferait perdre toute la consommation du mois."""
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
    """Quantité déjà consommée dans la période en cours (mois ou jour) chez ce provider."""
    return float(_read_state().get(name, {}).get(_period_key(PROVIDERS[name]), 0.0) or 0.0)


def _count(name: str, amount: float, exhausted: bool = False) -> None:
    """Ajoute `amount` au compteur de la période ; `exhausted` le pousse au plafond (le provider a dit stop)."""
    spec = PROVIDERS[name]
    with _lock:
        state = _read_state()
        period = _period_key(spec)
        current = float(state.get(name, {}).get(period, 0.0) or 0.0)
        new = max(current + amount, float(spec["quota"])) if exhausted and spec["quota"] else current + amount
        state[name] = {period: round(new, 6)}              # les périodes passées sont oubliées : le fichier reste minuscule
        try:
            _write_state(state)
        except OSError:
            logger.warning("recherche : compteur de %s non écrit sur disque", name)


def _quota_reached(name: str) -> bool:
    quota = PROVIDERS[name]["quota"]
    return bool(quota) and used(name) >= quota * QUOTA_MARGIN


def reset() -> None:
    """Remet l'état à zéro : pauses ET compteurs (tests)."""
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
    logger.warning("recherche : %s écarté %.0f s (%s)", name, seconds, reason[:120])


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
    """Providers utilisables maintenant : clé présente, pas en pause, quota gratuit non atteint."""
    base = [n for n in (only if only is not None else _order()) if n in PROVIDERS]
    return [n for n in base if _key(n) and not _paused(n) and not _quota_reached(n)]


def status() -> dict:
    """État des providers, pour un écran d'administration. Aucun appel réseau, aucune clé dans la réponse."""
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


# ───────────────────────── Adaptateurs : un par provider ─────────────────────────

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
        # `text` remplace `snippet` quand il est demandé : l'extrait vient alors du texte
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
    """Traduit un statut HTTP d'échec en décision : combien de temps écarter le provider, et pourquoi."""
    code = response.status_code
    if code in (401, 403):
        return _Refusal(BLOCKED, f"clé refusée (HTTP {code})", COOLDOWN_CONFIG_S)
    if code in (402, 432, 433):     # Exa : crédits épuisés ; Tavily : plafond du plan ou de la clé
        return _Refusal(RATE_LIMITED, f"quota épuisé (HTTP {code})", _seconds_until_renewal(PROVIDERS[name]), exhausted=True)
    if code == 429:
        return _Refusal(RATE_LIMITED, "limite de débit (HTTP 429)", _retry_after(response) or COOLDOWN_RATE_LIMIT_S)
    return _Refusal(ERROR, f"HTTP {code}", COOLDOWN_TRANSIENT_S)


def _call(name: str, query: str, n: int, content: bool, client: httpx.Client) -> SearchResult:
    build, read = ADAPTERS[name]
    body, headers = build(query, n, content)
    try:
        response = client.post(PROVIDERS[name]["url"], json=body, headers=headers)
    except httpx.TimeoutException:
        raise _Refusal(ERROR, "délai dépassé", COOLDOWN_TRANSIENT_S) from None
    except httpx.HTTPError as exc:
        raise _Refusal(ERROR, f"réseau : {type(exc).__name__}", COOLDOWN_TRANSIENT_S) from None
    if response.status_code != 200:
        raise _classify(name, response)
    try:
        data = response.json()
        pages, cost = read(data if isinstance(data, dict) else {})
    except ValueError:
        raise _Refusal(ERROR, "réponse illisible", COOLDOWN_TRANSIENT_S) from None
    seen: set = set()
    unique = [p for p in pages if not (p["url"] in seen or seen.add(p["url"]))]
    return SearchResult(SUCCESS if unique else EMPTY, query, unique, None, name, cost)


# ───────────────────────── Interface ─────────────────────────

def search(query: str, n: int = 6, content: bool = False, only=None,
           fallback: Optional[Callable[[str, int], SearchResult]] = None,
           client: Optional[httpx.Client] = None) -> SearchResult:
    """Une recherche sur le premier provider gratuit qui répond.

    content   demande, quand le provider sait le donner, un extrait de texte de chaque page (évite de la télécharger)
    only      noms de providers permis (None = tous ceux dont la clé est présente, dans l'ordre)
    fallback  fonction `(query, n)` appelée quand aucune API n'est utilisable ; elle rend aussi un `SearchResult`
    client    `httpx.Client` à réutiliser (sinon un client est ouvert pour l'appel)

    Ne lève pas d'exception : `status` dit ce qui s'est passé."""
    query = " ".join((query or "").split())[:300]
    if not query:
        return SearchResult(ERROR, query, error="requête vide")
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
    return SearchResult(ERROR, query, error="aucun moteur de recherche configuré ou disponible")


def search_web(query: str) -> dict:
    """Cherche sur le Web et rend jusqu'à 6 résultats (titre, url, extrait, parfois le contenu de la page).

    Args:
        query: Termes de la recherche
    """
    r = search(query, n=6, content=True)
    if r.status == SUCCESS:
        return {"results": r.results}
    if r.status == EMPTY:
        return {"results": []}
    return {"error": "recherche indisponible", "status": r.status}    # jamais la raison ni le nom du moteur


def register_search_tool(name: str = "search_web", replace: bool = True) -> None:
    """Rend `search_web` appelable par les modèles de `chat()` (outil du registre). Sans effet utile si aucune clé de
    recherche n'est configurée : l'outil répond alors « recherche indisponible »."""
    from .tools import registry
    registry.register(search_web, name=name, replace=replace)


__all__ = ["search", "search_web", "register_search_tool", "status", "available", "reset", "used", "SearchResult",
           "PROVIDERS", "SUCCESS", "EMPTY", "BLOCKED", "RATE_LIMITED", "ERROR"]
