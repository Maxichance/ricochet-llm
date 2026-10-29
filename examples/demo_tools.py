"""
Exemple de plugin : des outils + un prompt système, sans toucher au paquet.

    # PowerShell
    $env:RICOCHET_PLUGIN = "examples/demo_tools.py"
    python -m ricochet_llm serve --open

Puis demandez dans la console : « Quelle heure est-il ? » ou « Combien font 17 * 23 ? ».
"""
from datetime import datetime, timezone

from ricochet_llm import set_system_prompt, tool

set_system_prompt("Tu es un assistant concis. Tu réponds en français et tu utilises les outils quand ils aident.")


@tool
def heure_actuelle() -> dict:
    """Donne la date et l'heure actuelles de la machine (heure locale et UTC)."""
    now = datetime.now().astimezone()
    return {"locale": now.isoformat(timespec="seconds"), "utc": now.astimezone(timezone.utc).isoformat(timespec="seconds")}


@tool
def calculer(a: float, operation: str, b: float) -> dict:
    """Effectue une opération arithmétique simple.

    Args:
        a: Premier nombre
        operation: L'une de + - * /
        b: Second nombre
    """
    operations = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}
    if operation not in operations:
        return {"erreur": f"opération inconnue : {operation}"}
    return {"resultat": operations[operation]}
