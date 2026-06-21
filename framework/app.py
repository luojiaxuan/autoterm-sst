"""FastAPI transport for the framework.

This module owns the user-facing protocol (REST + WebSocket) and nothing else.
It is backward compatible with the existing ``serve/static/index.html`` UI and
``scripts/smoke_p0_protocol.py``:

* ``POST /init``      -> ``router.open_session`` (accepts query params OR JSON)
* ``WS   /wss/{id}``  -> ``submit_audio`` / control; streams text/events back
* ``GET  /config``    -> aggregated agent capabilities
* ``GET  /health``    -> aggregated agent health
* ``POST /ping`` / ``/delete_session`` / ``/update_latency`` /
  ``/reset_translation`` / ``/glossary/build`` / ``GET /queue_status/{id}``

Control endpoints are thin shims that forward an opaque message to the agent via
``router.on_control`` -- the framework does not interpret them.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import starlette.websockets
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from framework.agent import EVENT_ERROR, EVENT_STATUS, TranslationEvent
from framework.router import AgentRouter

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "serve" / "static"


def _to_wire(event: TranslationEvent) -> str:
    if event.type == EVENT_ERROR:
        return f"ERROR: {event.text or (event.meta or {}).get('error', 'unknown error')}"
    # status/partial/final are all sent as plain text; the agent is responsible
    # for any prefix (e.g. "INITIALIZING:") it wants the client to see.
    return event.text or ""


def _to_wire_json(event: TranslationEvent) -> Dict[str, Any]:
    """Structured form of an event for ``?event_format=json`` clients.

    Backward compatible: a client only sees this when it opts in via the query
    param. ``meta`` is forwarded verbatim (retrieved terms, retrieval/generation
    latency, term-memory snapshot, ...) so richer UIs can render evidence
    without the framework interpreting any of it.
    """

    text = event.text or ""
    if event.type == EVENT_ERROR and not text:
        text = (event.meta or {}).get("error", "unknown error")
    payload: Dict[str, Any] = {"type": event.type, "text": text}
    if event.meta:
        payload["meta"] = event.meta
    return payload


async def _read_init_payload(request: Request) -> Dict[str, Any]:
    """Merge query params and (optional) JSON body into one config dict.

    The UI posts JSON; ``scripts/smoke_p0_protocol.py`` posts query params.
    """

    payload: Dict[str, Any] = dict(request.query_params)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if isinstance(body, dict):
            payload.update(body)
    if "latency_multiplier" in payload:
        try:
            payload["latency_multiplier"] = int(payload["latency_multiplier"])
        except (TypeError, ValueError):
            payload["latency_multiplier"] = 2
    return payload


def create_app(router: AgentRouter) -> FastAPI:
    app = FastAPI(title="RASST-Demo SST Framework")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def _startup() -> None:
        await router.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await router.shutdown()

    @app.exception_handler(Exception)
    async def _json_errors(request: Request, exc: Exception):  # noqa: ANN001
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    # ------------------------------------------------------------------ static
    @app.get("/")
    async def read_index():  # noqa: ANN202
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.get("/config")
    async def get_config():  # noqa: ANN202
        return router.config()

    @app.get("/health")
    async def health_check():  # noqa: ANN202
        return await router.health()

    # ------------------------------------------------------------------- init
    @app.post("/init")
    async def initialize_translation(request: Request):  # noqa: ANN202
        config = await _read_init_payload(request)
        if not config.get("agent_type") or not config.get("language_pair"):
            raise HTTPException(status_code=400, detail="agent_type and language_pair are required")
        session_id, info = await router.open_session(config)
        response = {
            "session_id": session_id,
            "queued": bool(info.queued),
            "queue_position": int(info.queue_position),
        }
        response.update(info.meta or {})
        return response

    # ------------------------------------------------------------- session ctl
    @app.post("/ping")
    async def ping_session(session_id: str):  # noqa: ANN202
        if not router.touch_ping(session_id):
            return {"success": False, "error": "Invalid session ID"}
        return {"success": True}

    @app.post("/delete_session")
    async def delete_session(request: Request, session_id: Optional[str] = None):  # noqa: ANN202
        if session_id is None:
            try:
                form = await request.form()
                session_id = form.get("session_id")
            except Exception:  # noqa: BLE001
                session_id = None
        if not session_id:
            return {"success": False, "error": "No session_id provided"}
        ok = await router.close_session(session_id)
        return {"success": ok} if ok else {"success": False, "error": "Invalid session ID"}

    @app.get("/queue_status/{session_id}")
    async def queue_status(session_id: str):  # noqa: ANN202
        return router.queue_status(session_id)

    @app.post("/update_latency")
    async def update_latency(session_id: str, latency_multiplier: int):  # noqa: ANN202
        try:
            result = await router.on_control(
                session_id,
                {"type": "update_latency", "latency_multiplier": int(latency_multiplier)},
            )
        except KeyError:
            return {"success": False, "error": "Invalid session ID"}
        return result or {"success": True}

    @app.post("/reset_translation")
    async def reset_translation(session_id: str):  # noqa: ANN202
        try:
            result = await router.on_control(session_id, {"type": "reset"})
        except KeyError:
            return {"success": False, "error": "Invalid session ID"}
        return result or {"success": True, "message": "Translation reset successfully."}

    @app.post("/glossary/build")
    async def build_glossary(request: Request):  # noqa: ANN202
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        session_id = body.get("session_id")
        message = {"type": "glossary_build", **body}
        try:
            result = await router.on_control(session_id, message)
        except KeyError:
            raise HTTPException(status_code=400, detail="Unknown session")
        return result or {"success": True}

    @app.post("/download_youtube")
    async def download_youtube(request: Request, background_tasks: BackgroundTasks):  # noqa: ANN202
        query = dict(request.query_params)
        url = query.get("url")
        session_id = query.get("session_id") or "anon"
        if not url:
            return {"error": "Missing URL parameter"}
        out_dir = Path(os.environ.get("RASST_YT_TMP_DIR", tempfile.gettempdir()))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"video_{session_id}.mp4"
        cmd = [sys.executable, "-m", "yt_dlp"]
        cookies = os.environ.get("RASST_YT_COOKIES")
        if cookies:
            cmd.append(f"--cookies={cookies}")
        cmd += [
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "--no-continue", "--no-part",
            "-o", str(output_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists():
            return {"error": f"yt-dlp error: {result.stderr[:2000]}"}
        background_tasks.add_task(os.remove, str(output_path))
        return FileResponse(str(output_path), media_type="video/mp4", filename=output_path.name)

    # -------------------------------------------------------------- websocket
    @app.websocket("/wss/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):  # noqa: ANN202
        await websocket.accept()
        record = router.get_session(session_id)
        if record is None:
            await websocket.close(code=4000, reason="Invalid session ID")
            return

        # Opt-in structured protocol. Default ("plain") keeps the original
        # text-only wire so existing web/Electron clients work unchanged.
        json_mode = websocket.query_params.get("event_format", "plain").lower() == "json"

        def serialize(event: TranslationEvent) -> str:
            if json_mode:
                return json.dumps(_to_wire_json(event), ensure_ascii=False)
            return _to_wire(event)

        async def send_status(text: str) -> None:
            await websocket.send_text(
                serialize(TranslationEvent(session_id=session_id, type=EVENT_STATUS, text=text))
            )

        await send_status("READY: framework ready")

        import asyncio

        async def sender() -> None:
            while True:
                try:
                    event = await record.queue.get()
                except asyncio.CancelledError:
                    break
                if websocket.client_state.name != "CONNECTED":
                    break
                try:
                    await websocket.send_text(serialize(event))
                except Exception:  # noqa: BLE001
                    break

        sender_task = asyncio.create_task(sender())
        try:
            while True:
                try:
                    message = await websocket.receive()
                except (starlette.websockets.WebSocketDisconnect, RuntimeError):
                    break
                router.touch_ping(session_id)
                if "bytes" in message and message["bytes"] is not None:
                    pcm = np.frombuffer(message["bytes"], dtype=np.float32)
                    if pcm.size == 0:
                        continue
                    try:
                        await router.submit_audio(session_id, pcm, final=False)
                    except KeyError:
                        break
                elif "text" in message and message["text"] is not None:
                    text = message["text"]
                    if text == "EOF":
                        try:
                            await router.submit_audio(session_id, np.zeros(0, dtype=np.float32), final=True)
                        except KeyError:
                            pass
                        if websocket.client_state.name == "CONNECTED":
                            await send_status("PROCESSING_COMPLETE: File processing finished")
                    elif text.startswith("LATENCY:"):
                        value = text.split(":", 1)[1].strip()
                        try:
                            await router.on_control(
                                session_id,
                                {"type": "update_latency", "latency_multiplier": int(value)},
                            )
                        except (KeyError, ValueError):
                            pass
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except Exception:  # noqa: BLE001
                pass

    return app
