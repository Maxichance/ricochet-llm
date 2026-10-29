"""
Tool registry (tool calling): describing a Python function is enough, the JSON schema is inferred from its signature.

    from ricochet_llm import tool

    @tool
    def find_customer(email: str) -> dict:
        \"\"\"Looks up a customer by email.

        Args:
            email: Customer email
        \"\"\"
        return {"email": email, "name": "Smith"}

The model only sees the description and the schema; the return value must be JSON-serializable.
Any error raised by the tool is returned to the model (which can correct itself) instead of crashing the conversation.
"""
import inspect
import json
import logging
import re
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")     # OpenAI API constraint on function names
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", tuple: "array", set: "array", dict: "object"}
_UNION_TYPES = tuple(t for t in (Union, getattr(types, "UnionType", None)) if t is not None)


def _json_schema(annotation) -> dict:
    """Python annotation → JSON Schema fragment (fallback: string)."""
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
    """(description, {parameter: description}) from a Google-style docstring ("Args:")."""
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
        if line and not line[0].isspace():                        # new section: end of "Args:"
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
    except Exception:                                             # unresolvable annotations: fall back to string
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
        """Decorator: `@tool` or `@tool(name=…, description=…, parameters=…)`. Without `parameters`, the schema
        is inferred from the signature and the docstring."""
        def decorate(f: Callable) -> Callable:
            tool_name = name or f.__name__
            if not _NAME_RE.match(tool_name):
                raise ValueError(f"Invalid tool name: {tool_name!r} (letters, digits, _ and - ; 64 characters max)")
            if tool_name in self._tools and not replace:
                raise ValueError(f"Tool already registered: {tool_name!r} (use replace=True to replace it)")
            doc_text, arg_docs = _split_doc(inspect.getdoc(f) or "")
            desc = description or doc_text
            if not desc:
                raise ValueError(f"Tool {tool_name!r} has no description: add a docstring or description=…")
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
        """List to pass as the API `tools` parameter (empty if there are no tools)."""
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, raw_arguments: str) -> str:
        """Runs a tool from the model's JSON arguments; always returns a JSON string.
        Unknown tool, invalid arguments or tool exception → {"error": …} returned to the model."""
        tool = self._tools.get(name)
        if tool is None:
            result: Any = {"error": f"Unknown tool: {name}. Available tools: {sorted(self._tools)}"}
        else:
            try:
                args = json.loads(raw_arguments) if raw_arguments else {}
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                result = tool.func(**args)
            except Exception as e:
                logger.warning("Tool %s failed: %s", name, e)
                result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result, ensure_ascii=False, default=str)


registry = ToolRegistry()
tool = registry.register
