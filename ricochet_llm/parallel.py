"""
Exécuteur parallèle : plusieurs appels à la fois, répartis entre les providers disponibles.

Pour les traitements par lot (lire des milliers de documents, classer, extraire) où la cascade « un provider, puis le
suivant s'il échoue » laisse les autres providers, pourtant configurés, au repos.

    from ricochet_llm import generate, run_parallel

    def lire(document, provider):                       # exécutée dans un thread
        r = generate(prompt(document), prefer=provider, timeout=60, max_tokens=8192)
        sauvegarder(document, r.text)                   # ENREGISTRER AU FIL DE L'EAU : une interruption ne perd rien

    bilan = run_parallel(documents, lire, limits={"nvidia": 10})
    # {"processed": 812, "errors": 3, "remaining": 0, "interrupted": False, "per_provider": {"groq": 301, ...}}

Ce que l'exécuteur tient, et pourquoi :

- Un plafond de requêtes SIMULTANÉES par provider (`max_concurrency` du catalogue, ou `limits`). Une limite « 40 requêtes
  par minute » porte sur le RYTHME DE DÉPART, pas sur la simultanéité : à 20 s la réponse, 2 travailleurs font ~5 requêtes
  par minute, il en faut ~13 pour atteindre 40. Une salve passe, un flux soutenu atteint la limite.
- Un écart minimal entre deux DÉPARTS du même provider (`min_interval_s`) : 10 s pour 6 requêtes par minute.
- Un provider en pause (quota atteint) ne prend plus de travail : ses travailleurs attendent son retour au lieu de reporter
  leur charge sur les autres au-delà de leur plafond.
- Plus aucun provider disponible : tout s'arrête proprement, les éléments restants sont rendus (`remaining`), rien n'est perdu.
- Une erreur sur UN élément (réponse bizarre, bug local) est comptée et n'arrête pas les autres.

L'ordre de la file est conservé (chronologique si vos éléments le sont).
"""
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional

from . import router


class Interrupted(Exception):
    """À lever depuis `work` pour arrêter tout le traitement proprement (l'élément en cours est rendu à la file)."""


def parse_limits(text: Optional[str]) -> Dict[str, int]:
    """« nvidia=10,groq=3 » -> {"nvidia": 10, "groq": 3}. Lève ValueError sur une entrée mal formée : un réglage ignoré
    en silence ferait croire que la limite a changé."""
    out: Dict[str, int] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        if not sep or not name.strip() or not value.strip().isdigit() or not 1 <= int(value) <= 30:
            raise ValueError(f"réglage invalide : {part.strip()!r} (attendu : provider=nombre entre 1 et 30)")
        out[name.strip().lower()] = int(value)
    return out


def run_parallel(items: Iterable, work: Callable, *, only=None, limits: Optional[Dict[str, int]] = None,
                 intervals: Optional[Dict[str, float]] = None, pause_wait: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Exécute `work(item, provider_name)` pour chaque élément, en parallèle entre les providers configurés.

    `limits` / `intervals` surchargent `max_concurrency` / `min_interval_s` du catalogue. `work` doit appeler le routeur
    avec `prefer=provider_name` pour que la répartition soit effective (le routeur garde la bascule si ce provider échoue).
    Renvoie {"processed", "errors", "remaining", "interrupted", "per_provider"}."""
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
        """Un seul départ à la fois par provider, espacé de `interval(name)` secondes."""
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
            if not available:                       # plus aucun provider : on s'arrête, la file est rendue
                stop.set()
                return
            if name not in available:               # en pause : on attend son retour sans prendre de travail
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
            except Exception:                       # noqa: BLE001  une erreur sur un élément n'arrête pas les autres
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
