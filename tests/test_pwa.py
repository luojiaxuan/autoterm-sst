import struct

from fastapi.testclient import TestClient

from framework.app import STATIC_DIR, create_app
from framework.config import build_router


def test_pwa_assets(monkeypatch, tmp_path):
    monkeypatch.setenv("RASST_DEMO_MOCK", "1")
    monkeypatch.setenv("RASST_TMP_DIR", str(tmp_path))

    with TestClient(create_app(build_router(agents="RASST"))) as client:
        response = client.get("/manifest.webmanifest")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/manifest+json"
        manifest = response.json()
        expected = {
            "name": "AutoTerm-SST",
            "short_name": "AutoTerm",
            "start_url": "/",
            "scope": "/",
            "display": "standalone",
            "orientation": "portrait",
            "background_color": "#1a1a2e",
            "theme_color": "#1a1a2e",
        }
        assert {key: manifest[key] for key in expected} == expected

        icons = {icon["sizes"]: icon for icon in manifest["icons"]}
        for size, purpose in ((192, "any"), (512, "maskable")):
            icon = icons[f"{size}x{size}"]
            assert purpose in icon["purpose"].split()
            assert icon["type"] == "image/png"
            png = (STATIC_DIR / icon["src"].removeprefix("/static/")).read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", png[16:24]) == (size, size)

        worker = client.get("/sw.js")
        assert worker.status_code == 200
        assert worker.headers["content-type"].split(";")[0] == "text/javascript"
        assert worker.headers["service-worker-allowed"] == "/"
