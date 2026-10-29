"""
Async entry points for applications built on asyncio.

The router itself is synchronous (the `openai` client, one thread per call): each function here runs the matching
synchronous one in a worker thread, so the event loop is never blocked and failover works exactly the same.

    answer = await achat(history, "Hello")
    result = await agenerate("Summarize...", max_tokens=1024)
"""
import asyncio

from . import router


async def achat(conversation_history: list, user_message: str) -> str:
    """Async `chat`."""
    return await asyncio.to_thread(router.chat, conversation_history, user_message)


async def achat_detailed(conversation_history: list, user_message: str) -> router.ChatResult:
    """Async `chat_detailed`."""
    return await asyncio.to_thread(router.chat_detailed, conversation_history, user_message)


async def agenerate(prompt, system=None, **options) -> router.GenerateResult:
    """Async `generate` (same arguments)."""
    return await asyncio.to_thread(lambda: router.generate(prompt, system, **options))


async def acomplete(messages: list, **options) -> dict:
    """Async `complete` (same arguments)."""
    return await asyncio.to_thread(lambda: router.complete(messages, **options))
