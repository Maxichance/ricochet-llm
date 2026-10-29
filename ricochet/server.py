"""
Serveur web local : tableau de bord (statut + quota de chaque provider), console de chat et API JSON.
Lancer :  python -m ricochet serve        puis ouvrir http://127.0.0.1:8765
Écoute en local uniquement par défaut : les actions (chat, test, reprise) consomment des quotas.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import router, search
from .plugins import load_plugin

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
NO_STORE = {"Cache-Control": "no-store"}
MAX_SESSIONS = 200


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


def create_app() -> FastAPI:
    load_plugin()                                   # $RICOCHET_PLUGIN : vos outils et votre prompt système
    app = FastAPI(title="Ricochet", docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    sessions: Dict[str, list] = {}                  # session_id -> historique (en mémoire, mono-instance)
    session_locks: Dict[str, threading.Lock] = {}
    registry_lock = threading.Lock()
    started = time.time()

    def get_session(session_id: str):
        with registry_lock:
            if session_id not in sessions and len(sessions) >= MAX_SESSIONS:
                oldest = next(iter(sessions))
                sessions.pop(oldest, None)
                session_locks.pop(oldest, None)
            return sessions.setdefault(session_id, []), session_locks.setdefault(session_id, threading.Lock())

    def known_provider(name: str) -> bool:
        return router._provider_by_name(name) is not None

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers=NO_STORE)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/status")
    async def status():
        await run_in_threadpool(router.refresh_probes)      # sondes de quota (réseau) hors de la boucle d'événements
        providers = router.quota_status()
        return JSONResponse({
            "providers":   providers,
            "order":       list(providers),
            "next":        router.next_provider(),
            "events":      router.recent_events(60),
            "server_time": time.time(),
            "uptime_s":    int(time.time() - started),
        }, headers=NO_STORE)

    @app.get("/api/search/status")
    async def search_status():
        """Moteurs de recherche Web : clés présentes, consommation du mois ou du jour, pauses. Aucun appel réseau."""
        return JSONResponse(search.status(), headers=NO_STORE)

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        history, lock = get_session(req.session_id)
        if not lock.acquire(blocking=False):            # l'historique d'une session n'est pas thread-safe
            return JSONResponse({"status": "error", "response": "Une requête est déjà en cours pour cette session."},
                                status_code=409)
        try:
            result = await run_in_threadpool(router.chat_detailed, history, req.message)
            return {"status": "ok", "response": result.answer, "meta": result.to_dict()}
        except router.LLMRouterError as e:
            return JSONResponse({"status": "error", "response": str(e)}, status_code=503)
        except Exception as e:
            logger.exception("Erreur pendant le chat")
            return JSONResponse({"status": "error", "response": f"Erreur LLM : {str(e)[:300]}"}, status_code=500)
        finally:
            lock.release()

    @app.post("/api/reset")
    async def api_reset(req: ResetRequest):
        with registry_lock:
            sessions.pop(req.session_id, None)
            session_locks.pop(req.session_id, None)
        return {"status": "ok"}

    @app.post("/api/providers/{name}/ping")
    async def api_ping(name: str):
        if not known_provider(name):
            return JSONResponse({"status": "error", "response": f"Provider inconnu ou sans clé : {name}"},
                                status_code=404)
        return await run_in_threadpool(router.ping_provider, name)

    @app.post("/api/providers/{name}/reset")
    async def api_provider_reset(name: str):
        if not known_provider(name):
            return JSONResponse({"status": "error", "response": f"Provider inconnu ou sans clé : {name}"},
                                status_code=404)
        router.reset_provider(name)
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
