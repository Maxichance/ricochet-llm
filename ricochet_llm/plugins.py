"""
Loads your own code (tools, system prompt) without modifying the package.

  RICOCHET_PLUGIN=my_module          importable module (on PYTHONPATH or in the current directory)
  RICOCHET_PLUGIN=path/to/tools.py   Python file

The plugin registers its tools with `@tool` and may call `set_system_prompt(...)`.
Warning: the plugin is run as-is — only load trusted code.
"""
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)
_loaded: Set[str] = set()


class PluginError(RuntimeError):
    """Plugin not found or failing to load."""


def load_plugin(target: Optional[str] = None) -> Optional[str]:
    """Loads `target` (or $RICOCHET_PLUGIN) once. Returns what was loaded, or None."""
    target = (target or os.getenv("RICOCHET_PLUGIN") or "").strip()
    if not target or target in _loaded:
        return None
    try:
        if target.endswith(".py") or "/" in target or "\\" in target:
            path = Path(target).expanduser().resolve()
            if not path.is_file():
                raise PluginError(f"Plugin not found: {path}")
            spec = importlib.util.spec_from_file_location(f"ricochet_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(target)
    except PluginError:
        raise
    except Exception as e:
        raise PluginError(f"Failed to load plugin {target!r}: {type(e).__name__}: {e}") from e
    _loaded.add(target)
    logger.info("Plugin loaded: %s", target)
    return target
