"""Render the web UI while it replays a recorded AutoTerm-SST run, as a high-DPI screenshot.

The server runs the mock RASST agent so that the UI, the wire protocol, and the
session lifecycle are the real ones. Every partial translation the mock agent
emits is replaced by the next record of a recorded run: its translation text,
retrieved term pairs, active slices, router state, and retrieval latency. The
page is driven by headless Chrome through Playwright and captured at a chosen
device scale factor, so the figure shows the outputs of an evaluated run in the
interface users see.

Usage:
    python scripts/render_ui_replay.py --records run.json --start-event 600 \
        --count 40 --out ui.png --scale 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("RASST_DEMO_MOCK", "1")
os.environ.setdefault("RASST_DEMO_FAKE_GPUS", "0")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import uvicorn  # noqa: E402

from framework.agent import EVENT_PARTIAL, EVENT_STATUS, TranslationEvent  # noqa: E402
from framework.agents.omni import OmniAgent  # noqa: E402
from framework.app import create_app  # noqa: E402
from framework.router import AgentRouter  # noqa: E402


def topic_of(record):
    return {
        "active_domain": record["active_domain"],
        "confidence": record["router_confidence"],
        "active_glossary_preset": record["active_preset"],
        "switch_count": record["switch_count"],
        "last_reason": record["router_reason"],
    }


class ReplayAgent(OmniAgent):
    """Mock RASST agent whose partial outputs come from a recorded run."""

    def __init__(self, records, **kwargs):
        super().__init__(**kwargs)
        self.records = records
        self.cursor = {}
        self.delivered = 0

    def current(self, session_id):
        return self.records[max(0, self.cursor.get(session_id, 0) - 1)]

    async def start(self, emit):
        def replay(event):
            if event.type == EVENT_STATUS and (event.text or "").startswith("AUTO_GLOSSARY_SWITCH"):
                return
            if event.type == EVENT_PARTIAL:
                index = self.cursor.get(event.session_id, 0)
                if index >= len(self.records):
                    return
                self.cursor[event.session_id] = index + 1
                record = self.records[index]
                meta = dict(event.meta or {})
                meta.update(
                    references=record["references"],
                    retrieve_s=record["retrieve_s"],
                    prompt_reference_count=record["prompt_reference_count"],
                    active_slices=record["selected_slice_presets"],
                    active_terms=record["selected_term_count"],
                    topic=topic_of(record),
                    cursor_samples=record["cursor_samples"],
                    start_sample=record["start_sample"],
                )
                meta.pop("topic_router", None)
                event = TranslationEvent(
                    session_id=event.session_id, type=EVENT_PARTIAL, text=record["text"], meta=meta
                )
                self.delivered += 1
            emit(event)

        await super().start(replay)

    async def health(self):
        health = await super().health()
        # The translations shown are recorded model outputs, so the UI's
        # placeholder-translation banner for mock mode does not apply.
        health["mock_mode"] = False
        sessions = {}
        for session_id in self.cursor:
            sessions[session_id] = topic_of(self.current(session_id))
        record = self.records[0] if not self.cursor else self.current(next(iter(self.cursor)))
        health["term_memory"] = {
            "enabled": True,
            "backend": "maxsim",
            "active_terms": record["selected_term_count"],
            "snapshot_id": " + ".join(record["selected_slice_presets"]),
            "sessions": sessions,
        }
        return health


def serve(agent, port):
    app = create_app(AgentRouter({"RASST": agent}, default_agent="RASST"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError("server did not start")


STAGE_CSS = """
.translation-panel{top:%dpx !important;bottom:auto !important;}
#mediaContainer, .media-player, audio, video{visibility:hidden !important;}
"""


async def capture(args, agent):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        page = await browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.scale,
        )
        await page.goto(f"http://127.0.0.1:{args.port}/", wait_until="networkidle")
        await page.click("#modeAutoBtn")
        await page.click(".sample-btn-primary")
        await page.wait_for_function(
            "() => { const m = document.querySelector('#mediaContainer audio, #mediaContainer video, audio, video');"
            " return m && m.readyState >= 2 && m.duration > 0; }",
            timeout=60000,
        )
        await page.evaluate(
            "() => { const m = document.querySelector('#mediaContainer audio, #mediaContainer video, audio, video');"
            " m.playbackRate = 1.0; return m.play(); }"
        )
        deadline = time.time() + args.count * 1.92 * 3 + 120
        while agent.delivered < args.count and time.time() < deadline:
            await asyncio.sleep(1.0)
        await asyncio.sleep(2.0)
        chips_bottom = await page.evaluate(
            "() => { const lang = document.getElementById('languagePair');"
            " const card = lang.closest('.control-group') || lang;"
            " window.scrollTo(0, Math.max(0, card.getBoundingClientRect().top + window.scrollY - 16));"
            " return Math.round(document.getElementById('adaptiveStatus').getBoundingClientRect().bottom); }"
        )
        await page.add_style_tag(content=STAGE_CSS % (chips_bottom + 10) + args.extra_css)
        await page.evaluate(
            "() => document.querySelectorAll('.toast, .notification, .status-toast').forEach(e => e.remove())"
        )
        await asyncio.sleep(1.0)
        box = await page.evaluate(
            "() => { const p = document.querySelector('.translation-panel').getBoundingClientRect();"
            " const c = document.getElementById('adaptiveStatus').getBoundingClientRect();"
            " return {x0: Math.min(p.left, c.left), x1: Math.max(p.right, c.right), y1: p.bottom}; }"
        )
        margin = 16
        x0 = max(0, box["x0"] - margin)
        await page.screenshot(
            path=args.out,
            clip={
                "x": x0,
                "y": 0,
                "width": min(args.width, box["x1"] + margin) - x0,
                "height": min(args.height, box["y1"] + margin),
            },
        )
        await browser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, required=True, help="run JSON with a 'records' list")
    ap.add_argument("--start-event", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=820)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--extra-css", default="")
    args = ap.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))["records"]
    records = [r for r in records if r["event_idx"] >= args.start_event][: args.count]
    agent = ReplayAgent(records, name="RASST", model_id="qwen3_omni")
    server = serve(agent, args.port)
    try:
        asyncio.run(capture(args, agent))
    finally:
        server.should_exit = True
    print(f"delivered {agent.delivered} records; wrote {args.out}")


if __name__ == "__main__":
    main()
