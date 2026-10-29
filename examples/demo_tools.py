"""
Example plugin: tools + a system prompt, without touching the package.

    # PowerShell
    $env:RICOCHET_PLUGIN = "examples/demo_tools.py"
    python -m ricochet_llm serve --open

Then ask in the console: "What time is it?" or "What is 17 * 23?".
"""
from datetime import datetime, timezone

from ricochet_llm import set_system_prompt, tool

set_system_prompt("You are a concise assistant. You answer in English and use the tools when they help.")


@tool
def current_time() -> dict:
    """Gives the current date and time of the machine (local time and UTC)."""
    now = datetime.now().astimezone()
    return {"local": now.isoformat(timespec="seconds"), "utc": now.astimezone(timezone.utc).isoformat(timespec="seconds")}


@tool
def calculate(a: float, operation: str, b: float) -> dict:
    """Performs a simple arithmetic operation.

    Args:
        a: First number
        operation: One of + - * /
        b: Second number
    """
    operations = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}
    if operation not in operations:
        return {"error": f"unknown operation: {operation}"}
    return {"result": operations[operation]}
