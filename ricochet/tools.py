"""
Registre d'outils (tool calling) : décrire une fonction Python suffit, le schéma JSON est déduit de sa signature.

    from ricochet import tool

    @tool
    def rechercher_client(email: str) -> dict:
        \"\"\"Recherche un client par email.

        Args:
            email: Email du client
        \"\"\"
        return {"email": email, "nom": "Dupont"}

Le modèle ne voit que la description et le schéma ; le retour doit être sérialisable en JSON.
Toute erreur levée par l'outil est renvoyée au modèle (qui peut se corriger) au lieu de planter la conversation.
"""
import inspect
import json
import logging
import re
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")     # contrainte de l'API OpenAI sur les noms de fonctions
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", tuple: "array", set: "array", dict: "object"}
_UNION_TYPES = tuple(t for t in (Union, getattr(types, "UnionType", None)) if t is not None)


def _json_schema(annotation) -> dict:
    """Annotation Python → fragment de JSON Schema (repli : string)."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    origin = get_origin(annotation)
    if origin in _UNION_TYPES:                                    # Optional[X], X | None
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _json_schema(args[0]) if len(args) == 1 else {"type": "string"}
    if origin is Literal:
        values = list(get_args(annotation))
        return {"type": _JSON_TYPES.get(type(values[0]), "string"), "enum": values}
    if origin in (list, tuple, set):
        inner = get_args(annotation)
        return {"type": "array", **({"items": _json_schema(inner[0])} if inner else {})}
    if origin is dict:
        return {"type": "object"}
    return {"type": _JSON_TYPES.get(annotation, "string")}


def _split_doc(doc: str):
    """(description, {paramètre: description}) depuis une docstring de style Google (« Args: »)."""
    lines = doc.strip().splitlines()
    description, arg_docs, in_args, current = [], {}, False, None
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Param(è|e)tres|Parameters)\s*:\s*$", stripped, re.IGNORECASE):
            in_args = True
            continue
        if not in_args:
            description.append(stripped)
            continue
        if line and not line[0].isspace():                        # nouvelle section : fin de « Args: »
            in_args = False
            continue
        m = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
        if m:
            current = m.group(1)
            arg_docs[current] = m.group(2)
        elif current and stripped:
            arg_docs[current] += " " + stripped
    text = " ".join(" ".join(description).split())
    return text, arg_docs


def _schema_from_function(func: Callable, arg_docs: Dict[str, str]) -> dict:
    hints = {}
    try:
        from typing import get_type_hints
        hints = get_type_hints(func)
    except Exception:                                             # annotations non résolubles : repli sur string
        pass
    props, required = {}, []
    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _json_schema(hints.get(name, param.annotation))
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        props[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, func: Optional[Callable] = None, *, name: Optional[str] = None,
                 description: Optional[str] = None, parameters: Optional[dict] = None, replace: bool = False):
        """Décorateur : `@tool` ou `@tool(name=…, description=…, parameters=…)`. Sans `parameters`, le schéma
        est déduit de la signature et de la docstring."""
        def decorate(f: Callable) -> Callable:
            tool_name = name or f.__name__
            if not _NAME_RE.match(tool_name):
                raise ValueError(f"Nom d'outil invalide : {tool_name!r} (lettres, chiffres, _ et - ; 64 caractères max)")
            if tool_name in self._tools and not replace:
                raise ValueError(f"Outil déjà enregistré : {tool_name!r} (utilisez replace=True pour le remplacer)")
            doc_text, arg_docs = _split_doc(inspect.getdoc(f) or "")
            desc = description or doc_text
            if not desc:
                raise ValueError(f"L'outil {tool_name!r} n'a pas de description : ajoutez une docstring ou description=…")
            self._tools[tool_name] = Tool(tool_name, desc, parameters or _schema_from_function(f, arg_docs), f)
            return f
        return decorate(func) if func is not None else decorate

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear(self) -> None:
        self._tools.clear()

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[dict]:
        """Liste à passer au paramètre `tools` de l'API (vide si aucun outil)."""
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, raw_arguments: str) -> str:
        """Exécute un outil à partir des arguments JSON du modèle ; retourne toujours une chaîne JSON.
        Outil inconnu, arguments invalides ou exception de l'outil → {"error": …} renvoyé au modèle."""
        tool = self._tools.get(name)
        if tool is None:
            result: Any = {"error": f"Outil inconnu : {name}. Outils disponibles : {sorted(self._tools)}"}
        else:
            try:
                args = json.loads(raw_arguments) if raw_arguments else {}
                if not isinstance(args, dict):
                    raise ValueError("les arguments doivent être un objet JSON")
                result = tool.func(**args)
            except Exception as e:
                logger.warning("Outil %s en erreur : %s", name, e)
                result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result, ensure_ascii=False, default=str)


registry = ToolRegistry()
tool = registry.register
