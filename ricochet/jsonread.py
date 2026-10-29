"""
Lecture tolérante de la réponse JSON d'un modèle : fonction pure, aucun état.

    from ricochet import generate, read_json

    r = generate(prompt, max_tokens=8192)
    data = read_json(r.text)          # dict, ou None si la réponse est vraiment illisible

Ce que les modèles font parfois à un JSON valide :

- le sortir dans un bloc ```json, derrière un <think>…</think>, ou avec du texte autour ;
- laisser une virgule finale ;
- écrire la réponse DEUX FOIS (« …}}"{"items":[… ») ou la faire suivre d'un champ en trop (« …}}"extra":[]} ») ;
- glisser un `null` parasite entre deux paires clé/valeur (« …"scope":"all",null,"start":null… »).

Lecture STRICTE d'abord, inchangée pour toute réponse valide. On ne répare qu'en cas d'échec, et seulement ce qui est sans
ambiguïté : le premier objet COMPLET est gardé, le reste est ignoré. Une réponse coupée (max_tokens atteint) ou qui boucle
(« "urgences": null, » cent fois) reste illisible : il faut la relire, jamais la deviner.

Ne vérifie PAS le contenu : un JSON valide peut contenir des valeurs inventées. Le contrôle de fond (une citation est-elle
retrouvée dans la source ?) reste à l'appelant, et s'applique à l'objet récupéré comme à un autre.
"""
import json
import re
from typing import Optional

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
# `null` suivi d'une clé : n'a aucun sens dans un objet. Un `null` légitime (valeur, ou élément d'un tableau) n'est pas
# suivi d'une clé entre guillemets.
_STRAY_NULL = re.compile(r',\s*null\s*(?=,\s*")')
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _first_object(text: str) -> Optional[dict]:
    """Le PREMIER objet JSON complet de `text` ; tout ce qui suit est ignoré. Jamais un objet coupé."""
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def read_json(raw: Optional[str]) -> Optional[dict]:
    """Objet JSON de la réponse, même entouré de texte ou d'un bloc de code ; None si rien d'exploitable.

    Un objet dans un tableau (`[{"a": 1}]`) est lu comme avant : la lecture stricte va de la première accolade à la
    dernière."""
    if not raw:
        return None
    text = _FENCE.sub("", _THINK.sub("", raw).strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    chunk = text[start:end + 1]
    for attempt in (chunk, _TRAILING_COMMA.sub(r"\1", chunk)):
        try:
            data = json.loads(attempt)
            return data if isinstance(data, dict) else None
        except ValueError:
            continue
    from_start = text[start:]
    for variant in (from_start, _TRAILING_COMMA.sub(r"\1", from_start), _STRAY_NULL.sub("", from_start)):
        data = _first_object(variant)
        if data is not None:
            return data
    return None
