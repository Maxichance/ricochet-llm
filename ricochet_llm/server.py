"""
Local web server: dashboard (status + quota of each provider), chat console and JSON API.
Run:  python -m ricochet_llm serve        then open http://127.0.0.1:8765
Listens on localhost only by default: the actions (chat, test, resume) consume quota.
"""
import hmac
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import router, search
from .plugins import load_plugin

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
NO_STORE = {"Cache-Control": "no-store"}
MAX_SESSIONS = 200
SSE_END = chr(10) * 2                       # a blank line ends each server-sent event
API_KEY_ENV_VAR = "RICOCHET_API_KEY"        # if set, the /v1 endpoints require "Authorization: Bearer <value>"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


def _openai_error(message: str, status: int, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _stream_chunks(result: dict):
    """The finished completion as OpenAI-style SSE chunks. The router needs the whole answer to tell a good one from
    an empty one before failing over, so the stream delivers it in one piece, then the end marker."""
    choice = result["choices"][0]
    message = choice["message"]
    base = {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"],
            "model": result["model"]}
    delta: dict = {"role": "assistant"}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
    first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
    for chunk in (first, last):
        yield f"data: {json.dumps(chunk)}" + SSE_END
    yield "data: [DONE]" + SSE_END


def create_app() -> FastAPI:
    load_plugin()                                   # $RICOCHET_PLUGIN: your tools and your system prompt
    app = FastAPI(title="Ricochet", docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    sessions: Dict[str, list] = {}                  # session_id -> history (in memory, single instance)
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
        await run_in_threadpool(router.refresh_probes)      # quota probes (network) kept off the event loop
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
        """Web search engines: keys present, usage for the month or day, pauses. No network call."""
        return JSONResponse(search.status(), headers=NO_STORE)

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        history, lock = get_session(req.session_id)
        if not lock.acquire(blocking=False):            # a session's history is not thread-safe
            return JSONResponse({"status": "error", "response": "A request is already in progress for this session."},
                                status_code=409)
        try:
            result = await run_in_threadpool(router.chat_detailed, history, req.message)
            return {"status": "ok", "response": result.answer, "meta": result.to_dict()}
        except router.LLMRouterError as e:
            return JSONResponse({"status": "error", "response": str(e)}, status_code=503)
        except Exception as e:
            logger.exception("Error during chat")
            return JSONResponse({"status": "error", "response": f"LLM error: {str(e)[:300]}"}, status_code=500)
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
            return JSONResponse({"status": "error", "response": f"Unknown provider or no key: {name}"},
                                status_code=404)
        return await run_in_threadpool(router.ping_provider, name)

    @app.post("/api/providers/{name}/reset")
    async def api_provider_reset(name: str):
        if not known_provider(name):
            return JSONResponse({"status": "error", "response": f"Unknown provider or no key: {name}"},
                                status_code=404)
        router.reset_provider(name)
        return {"status": "ok"}

    def key_refused(request: Request) -> Optional[JSONResponse]:
        key = os.getenv(API_KEY_ENV_VAR, "")
        if key and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {key}"):
            return _openai_error("Invalid or missing API key.", 401, "authentication_error")
        return None

    @app.get("/v1/models")
    async def v1_models(request: Request):
        """OpenAI-compatible model list: `auto` (any provider) and one entry per configured provider."""
        refused = key_refused(request)
        if refused:
            return refused
        names = ["auto"] + router.provider_names()
        return {"object": "list", "data": [{"id": n, "object": "model", "created": int(started), "owned_by": "ricochet"}
                                           for n in names]}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request):
        """OpenAI-compatible chat completions with failover. `model` is `auto` (or anything) to use every provider,
        or a provider name to restrict the call to it. Tools are the caller's: tool calls come back unexecuted."""
        refused = key_refused(request)
        if refused:
            return refused
        try:
            body = await request.json()
        except ValueError:
            return _openai_error("The body must be JSON.", 400)
        messages = body.get("messages") if isinstance(body, dict) else None
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages)):
            return _openai_error("`messages` must be a non-empty list of {role, content} objects.", 400)
        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            return _openai_error("`tools` must be a list.", 400)
        limit = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
        temperature = body.get("temperature")
        if (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
                or not (temperature is None or isinstance(temperature, (int, float)))):
            return _openai_error("`max_tokens` must be a positive integer and `temperature` a number.", 400)
        model = body.get("model")
        options = {k: body[k] for k in router.PASSTHROUGH_PARAMS if k in body}
        try:
            result = await run_in_threadpool(lambda: router.complete(
                messages, tools=tools or None, tool_choice=body.get("tool_choice"), max_tokens=limit,
                temperature=0.2 if temperature is None else temperature,
                only=[model] if model in router.provider_names() else None, **options))
        except router.LLMRouterError as e:
            return _openai_error(str(e), 503, "server_error")
        except Exception as e:
            logger.exception("Error during /v1/chat/completions")
            return _openai_error(f"LLM error: {str(e)[:300]}", 500, "server_error")
        if body.get("stream"):
            return StreamingResponse(_stream_chunks(result), media_type="text/event-stream", headers=NO_STORE)
        return JSONResponse(result, headers=NO_STORE)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
