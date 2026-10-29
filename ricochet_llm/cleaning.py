"""
Nettoyage des réponses : les petits modèles laissent parfois un tool call en texte brut.
Fonctions pures (aucun état), donc testables seules.
"""
import json
import re
from typing import Iterable

_JSON_DECODER = json.JSONDecoder()
_TAGGED_CALLS = re.compile(r'<function=[^>]*>.*?</function>|<tool_call>.*?</tool_call>', re.DOTALL)
_EMPTY_FENCES = re.compile(r'```[a-zA-Z]*\s*```')
_JSON_RESIDUE = re.compile(r'^[\s\[\],]+$', re.MULTILINE)


_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)


def strip_think(text: str) -> str:
    """Retire les blocs <think>…</think> que certains modèles laissent dans `content` : ce n'est pas la réponse.
    Un bloc jamais refermé (réponse coupée) est conservé : on ne devine pas où le raisonnement s'arrête."""
    return _THINK_BLOCK.sub("", text or "").strip()


def looks_like_tool_call(obj, tool_names: Iterable[str]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        return True
    return obj.get("name") in set(tool_names) and ("arguments" in obj or "parameters" in obj)


def strip_tool_artifacts(text: str, tool_names: Iterable[str]) -> str:
    """Retire uniquement les tool calls qui ont fuité en texte (JSON complet reconnu ou balises
    <function=…>), jamais le reste : le code et le JSON légitimes de la réponse sont conservés."""
    names = set(tool_names)
    out, i, removed = [], 0, False
    while True:
        j = text.find("{", i)
        if j == -1:
            out.append(text[i:])
            break
        try:
            obj, end = _JSON_DECODER.raw_decode(text, j)
        except ValueError:
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if looks_like_tool_call(obj, names):
            out.append(text[i:j])
            removed = True
        else:
            out.append(text[i:end])
        i = end
    cleaned = "".join(out)
    cleaned, n = _TAGGED_CALLS.subn("", cleaned)
    if removed or n:
        cleaned = _EMPTY_FENCES.sub("", cleaned)
        cleaned = _JSON_RESIDUE.sub("", cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
