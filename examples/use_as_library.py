"""Utiliser le routeur comme une bibliothèque, sans serveur ni interface."""
import logging

from ricochet import chat_detailed, quota_status, tool

logging.basicConfig(level=logging.INFO)


@tool
def jeter_de(faces: int = 6) -> int:
    """Lance un dé.

    Args:
        faces: Nombre de faces du dé
    """
    import random
    return random.randint(1, faces)


history = []                                    # une liste par conversation (en mémoire)
result = chat_detailed(history, "Lance un dé à 20 faces et commente le résultat.")
print(result.answer)
print(f"→ {result.provider} · {result.model} · {result.mode} · {result.latency_ms} ms · bascules : {result.trail}")
print({name: p["status"] for name, p in quota_status().items()})
