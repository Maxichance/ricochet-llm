"""Use the router as a library, with no server or dashboard."""
import logging

from ricochet_llm import chat_detailed, quota_status, tool

logging.basicConfig(level=logging.INFO)


@tool
def roll_die(faces: int = 6) -> int:
    """Roll a die.

    Args:
        faces: Number of faces on the die
    """
    import random
    return random.randint(1, faces)


history = []                                    # one list per conversation (in memory)
result = chat_detailed(history, "Roll a 20-sided die and comment on the result.")
print(result.answer)
print(f"→ {result.provider} · {result.model} · {result.mode} · {result.latency_ms} ms · failovers: {result.trail}")
print({name: p["status"] for name, p in quota_status().items()})
