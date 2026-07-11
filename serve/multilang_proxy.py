"""Multilingual demo gateway.

One tunnel, one origin. Serves the static demo UI and fronts three
language-specific AutoTerm hosts (each a single-model host). Routing:

* GET  /config          -> zh backend's config with all pairs marked available
* POST /init            -> pick backend by ``language_pair``; remember session
* WS   /wss/{sid}        -> relay to the session's backend (binary + text)
* other session calls    -> forward to the session's backend
* GET  / and /static/*   -> the demo UI

Backends bind 0.0.0.0 on their hosts; taurus reaches aries over the network.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

BACKENDS = {
    "English -> Chinese": os.environ.get("PROXY_ZH", "http://127.0.0.1:8014"),
    "English -> Japanese": os.environ.get("PROXY_JA", "http://aries:8031"),
    "English -> German": os.environ.get("PROXY_DE", "http://aries:8032"),
}
DEFAULT = "English -> Chinese"
STATIC_DIR = os.environ.get("PROXY_STATIC", "/mnt/taurus/home/jiaxuanluo/rasst-demo/serve/static")

app = FastAPI()
_sessions: dict[str, str] = {}  # session_id -> backend base url


def _ws(url: str) -> str:
    return url.replace("https://", "wss://").replace("http://", "ws://")


def _backend_for(sid: str | None) -> str:
    return _sessions.get(sid or "", BACKENDS[DEFAULT])


@app.get("/health")
async def health():
    return {"status": "healthy", "proxy": True, "backends": list(BACKENDS)}


@app.get("/config")
async def config():
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(BACKENDS[DEFAULT] + "/config", timeout=10)
            cfg = r.json()
        except Exception:
            cfg = {}
    cfg["language_pairs"] = [{"id": k, "label": k, "available": True} for k in BACKENDS]
    cfg["loaded_language_pair"] = DEFAULT
    return JSONResponse(cfg)


@app.post("/init")
async def init(request: Request):
    body = await request.body()
    try:
        data = json.loads(body or b"{}")
    except Exception:
        data = {}
    lp = data.get("language_pair") or DEFAULT
    backend = BACKENDS.get(lp, BACKENDS[DEFAULT])
    async with httpx.AsyncClient() as c:
        r = await c.post(
            backend + "/init",
            content=body,
            headers={"content-type": "application/json"},
            timeout=180,
        )
    try:
        resp = r.json()
    except Exception:
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    sid = resp.get("session_id")
    if sid:
        _sessions[sid] = backend
    return JSONResponse(resp, status_code=r.status_code)


async def _forward(request: Request, path: str):
    sid = request.query_params.get("session_id")
    body = await request.body()
    if not sid and body:
        try:
            sid = json.loads(body).get("session_id")
        except Exception:
            pass
    backend = _backend_for(sid)
    url = backend + path
    if request.url.query:
        url += "?" + request.url.query
    async with httpx.AsyncClient() as c:
        r = await c.request(
            request.method,
            url,
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
            timeout=180,
        )
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.post("/ping")
async def ping(request: Request):
    return await _forward(request, "/ping")


@app.post("/update_latency")
async def update_latency(request: Request):
    return await _forward(request, "/update_latency")


@app.post("/delete_session")
async def delete_session(request: Request):
    return await _forward(request, "/delete_session")


@app.post("/reset_translation")
async def reset_translation(request: Request):
    return await _forward(request, "/reset_translation")


@app.post("/glossary/build")
async def glossary_build(request: Request):
    return await _forward(request, "/glossary/build")


@app.post("/download_youtube")
async def download_youtube(request: Request):
    return await _forward(request, "/download_youtube")


@app.get("/queue_status/{session_id}")
async def queue_status(request: Request, session_id: str):
    backend = _backend_for(session_id)
    async with httpx.AsyncClient() as c:
        r = await c.get(backend + f"/queue_status/{session_id}", timeout=30)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.websocket("/wss/{session_id}")
async def wss(ws: WebSocket, session_id: str):
    backend = _backend_for(session_id)
    burl = _ws(backend) + f"/wss/{session_id}"
    if ws.url.query:
        burl += "?" + ws.url.query
    await ws.accept()
    _dbg = os.environ.get("PROXY_DEBUG")
    _c = {"c2b": 0, "b2c": 0}
    if _dbg:
        print(f"[wss] sid={session_id[:30]} -> backend={backend} burl={burl}", flush=True)
    try:
        async with websockets.connect(burl, max_size=None, ping_interval=None) as bws:
            if _dbg:
                print(f"[wss] backend WS connected: {backend}", flush=True)

            async def c2b():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("bytes") is not None:
                            _c["c2b"] += 1
                            await bws.send(msg["bytes"])
                        elif msg.get("text") is not None:
                            _c["c2b"] += 1
                            await bws.send(msg["text"])
                except Exception:
                    pass
                finally:
                    try:
                        await bws.close()
                    except Exception:
                        pass

            async def b2c():
                try:
                    async for m in bws:
                        _c["b2c"] += 1
                        if isinstance(m, (bytes, bytearray)):
                            await ws.send_bytes(m)
                        else:
                            await ws.send_text(m)
                except Exception as e:
                    if _dbg:
                        print(f"[wss] b2c ended: {type(e).__name__} {e}", flush=True)

            t1 = asyncio.create_task(c2b())
            t2 = asyncio.create_task(b2c())
            _, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if _dbg:
                print(f"[wss] done sid={session_id[:30]} c2b={_c['c2b']} b2c={_c['b2c']}", flush=True)
    except Exception as e:
        if _dbg:
            print(f"[wss] relay error: {type(e).__name__} {e}", flush=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PROXY_PORT", "8080")), log_level="info")
