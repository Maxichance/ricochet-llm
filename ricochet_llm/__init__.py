"""Routeur LLM multi-providers free tier : bascule automatique, tool calling, lecture par lot, recherche Web, tableau de bord."""
from .jsonread import read_json
from .parallel import Interrupted, parse_limits, run_parallel
from .plugins import PluginError, load_plugin
from .privacy import check_conversation, find_personal_data
from .router import (
    ChatResult,
    GenerateResult,
    LLMRouterError,
    available_providers,
    chat,
    chat_detailed,
    generate,
    get_system_prompt,
    next_provider,
    ping_provider,
    probe_models,
    provider_names,
    provider_settings,
    quota_status,
    recent_events,
    refresh_probes,
    reset_provider,
    set_system_prompt,
)
from .search import register_search_tool
from .search import search as web_search  # `ricochet_llm.search` reste le module (statut, compteurs)
from .tools import registry, tool

__version__ = "0.5.0"

__all__ = [
    "ChatResult", "GenerateResult", "Interrupted", "LLMRouterError", "PluginError", "available_providers", "chat",
    "chat_detailed", "check_conversation", "find_personal_data", "generate", "get_system_prompt", "load_plugin",
    "next_provider", "parse_limits", "ping_provider", "probe_models", "provider_names", "provider_settings",
    "quota_status", "read_json", "recent_events", "refresh_probes", "register_search_tool", "registry",
    "reset_provider", "run_parallel", "set_system_prompt", "tool", "web_search",
]
