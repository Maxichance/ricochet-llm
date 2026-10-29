"""
Parallel executor: several calls at once, spread across the available providers.

For batch jobs (reading thousands of documents, classifying, extracting) where the "one provider, then the next if it
fails" cascade leaves the other, configured providers idle.

    from ricochet_llm import generate, run_parallel

    def read_doc(document, provider):                   # runs in a thread
        r = generate(prompt(document), prefer=provider, timeout=60, max_tokens=8192)
        save(document, r.text)                          # SAVE AS YOU GO: an interruption loses nothing

    summary = run_parallel(documents, read_doc, limits={"nvidia": 10})
    # {"processed": 812, "errors": 3, "remaining": 0, "interrupted": False, "per_provider": {"groq": 301, ...}}

What the executor enforces, and why:

- A cap on SIMULTANEOUS requests per provider (`max_concurrency` from the catalog, or `limits`). A limit of "40 requests
  per minute" applies to the START RATE, not to concurrency: at 20 s per response, 2 workers make ~5 requests
  per minute, ~13 are needed to reach 40. A burst gets through, a sustained flow hits the limit.
- A minimum gap between two STARTS on the same provider (`min_interval_s`): 10 s for 6 requests per minute.
- A paused provider (quota reached) takes no more work: its workers wait for it to come back instead of shifting
  their load onto the others beyond their cap.
- No provider available anymore: everything stops cleanly, the remaining items are returned (`remaining`), nothing is lost.
- An error on ONE item (odd response, local bug) is counted and does not stop the others.

Queue order is preserved (chronological if your items are).
"""
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional

from . import router


class Interrupted(Exception):
    """Raise from `work` to stop the whole run cleanly (the current item is returned to the queue)."""


def parse_limits(text: Optional[str]) -> Dict[str, int]:
    """"nvidia=10,groq=3" -> {"nvidia": 10, "groq": 3}. Raises ValueError on a malformed entry: a setting silently
    ignored would make you believe the limit had changed."""
    out: Dict[str, int] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        if not sep or not name.strip() or not value.strip().isdigit() or not 1 <= int(value) <= 30:
            raise ValueError(f"invalid setting: {part.strip()!r} (expected: provider=number between 1 and 30)")
        out[name.strip().lower()] = int(value)
    return out


def run_parallel(items: Iterable, work: Callable, *, only=None, limits: Optional[Dict[str, int]] = None,
                 intervals: Optional[Dict[str, float]] = None, pause_wait: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Runs `work(item, provider_name)` for each item, in parallel across the configured providers.

    `limits` / `intervals` override the catalog's `max_concurrency` / `min_interval_s`. `work` must call the router
    with `prefer=provider_name` for the spread to take effect (the router keeps the failover if that provider fails).
    Returns {"processed", "errors", "remaining", "interrupted", "per_provider"}."""
    names = router.provider_names(only)
    queue: deque = deque(items)
    lock = threading.Lock()
    stop = threading.Event()
    summary = {"processed": 0, "errors": 0, "per_provider": {}}
    next_start: Dict[str, float] = {}
    spacing = {n: threading.Lock() for n in names}

    def interval(name: str) -> float:
        return float((intervals or {}).get(name, router.provider_settings(name)["min_interval_s"]))

    def wait_turn(name: str) -> None:
        """One start at a time per provider, spaced by `interval(name)` seconds."""
        with spacing[name]:
            delay = next_start.get(name, 0.0) - clock()
            if delay > 0:
                sleep(delay)
            next_start[name] = clock() + interval(name)

    def worker(name: str) -> None:
        while not stop.is_set():
            with lock:
                if not queue:
                    return
            available = router.available_providers(only=names)
            if not available:                       # no provider left: stop, the queue is returned
                stop.set()
                return
            if name not in available:               # paused: wait for it to come back without taking work
                sleep(pause_wait)
                continue
            with lock:
                if not queue:
                    return
                item = queue.popleft()
            wait_turn(name)
            try:
                work(item, name)
            except (Interrupted, router.LLMRouterError):
                with lock:
                    queue.appendleft(item)
                stop.set()
                return
            except Exception:                       # noqa: BLE001  an error on one item does not stop the others
                with lock:
                    summary["errors"] += 1
                continue
            with lock:
                summary["processed"] += 1
                summary["per_provider"][name] = summary["per_provider"].get(name, 0) + 1

    threads = [threading.Thread(target=worker, args=(name,), daemon=True)
               for name in names for _ in range(int((limits or {}).get(name, router.provider_settings(name)["max_concurrency"])))]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.2)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(2)
        raise
    return {**summary, "remaining": len(queue), "interrupted": stop.is_set()}
