"""
Tolerant reading of a model's JSON response: pure function, no state.

    from ricochet_llm import generate, read_json

    r = generate(prompt, max_tokens=8192)
    data = read_json(r.text)          # dict, or None if the response is truly unreadable

What models sometimes do to valid JSON:

- output it in a ```json block, after a <think>…</think>, or with text around it;
- leave a trailing comma;
- write the response TWICE ("…}}"{"items":[…") or follow it with a stray field ("…}}"extra":[]}");
- slip a stray `null` between two key/value pairs ("…"scope":"all",null,"start":null…").

STRICT reading first, unchanged for any valid response. We only repair on failure, and only what is unambiguous:
the first COMPLETE object is kept, the rest is ignored. A truncated response (max_tokens reached) or one that loops
(`"urgent": null,` a hundred times) stays unreadable: it must be re-read, never guessed.

Does NOT check the content: valid JSON may contain invented values. The substantive check (is a quote found in the
source?) is left to the caller, and applies to a recovered object like any other.
"""
import json
import re
from typing import Optional

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
# `null` followed by a key: makes no sense in an object. A legitimate `null` (a value, or an array element) is not
# followed by a quoted key.
_STRAY_NULL = re.compile(r',\s*null\s*(?=,\s*")')
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _first_object(text: str) -> Optional[dict]:
    """The FIRST complete JSON object in `text`; everything after it is ignored. Never a truncated object."""
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def read_json(raw: Optional[str]) -> Optional[dict]:
    """JSON object of the response, even when surrounded by text or a code block; None if nothing usable.

    An object inside an array (`[{"a": 1}]`) is read as before: the strict reading goes from the first brace to the
    last."""
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
