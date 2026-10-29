"""
Chargement de votre code métier (outils, prompt système) sans modifier le paquet.

  RICOCHET_PLUGIN=mon_module            module importable (dans le PYTHONPATH ou le dossier courant)
  RICOCHET_PLUGIN=chemin/vers/outils.py fichier Python

Le plugin enregistre ses outils avec `@tool` et peut appeler `set_system_prompt(...)`.
Attention : le plugin est du code exécuté tel quel — ne chargez que du code de confiance.
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
    """Plugin introuvable ou qui échoue au chargement."""


def load_plugin(target: Optional[str] = None) -> Optional[str]:
    """Charge `target` (ou $RICOCHET_PLUGIN) une seule fois. Retourne ce qui a été chargé, ou None."""
    target = (target or os.getenv("RICOCHET_PLUGIN") or "").strip()
    if not target or target in _loaded:
        return None
    try:
        if target.endswith(".py") or "/" in target or "\\" in target:
            path = Path(target).expanduser().resolve()
            if not path.is_file():
                raise PluginError(f"Plugin introuvable : {path}")
            spec = importlib.util.spec_from_file_location(f"ricochet_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(target)
    except PluginError:
        raise
    except Exception as e:
        raise PluginError(f"Échec du chargement du plugin {target!r} : {type(e).__name__}: {e}") from e
    _loaded.add(target)
    logger.info("Plugin chargé : %s", target)
    return target
