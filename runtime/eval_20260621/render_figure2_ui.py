#!/usr/bin/env python3
"""Render Figure 2 from the current web UI plus a live Taurus sample."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "runtime/eval_20260621/figure2_live_sample.json"
OUT = ROOT / "runtime/eval_20260621/figure2_ui.png"
HEADLESS = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> None:
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    refs = sample.get("all_references") or sample.get("references") or []
    preferred = [
        "Annotated Corpus",
        "corpus",
        "lexical",
        "dataset",
        "NLP task",
        "NLP",
        "models",
        "model",
    ]
    ref_by_term = {str(ref.get("term") or ""): ref for ref in refs}
    chosen = []
    for term in preferred:
        ref = ref_by_term.get(term)
        if ref:
            chosen.append(ref)
    for ref in refs:
        if len(chosen) >= 8:
            break
        if ref not in chosen:
            chosen.append(ref)

    payload = {
        "translation": sample.get("translation", ""),
        "refs": chosen[:8],
        "topic": sample.get("topic") or {},
        "router": sample.get("topic_router") or {},
        "retrieve_ms": sample.get("retrieve_ms"),
        "elapsed_ms": sample.get("elapsed_ms"),
        "active_terms": (sample.get("init") or {}).get("preset_terms", 10000),
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=HEADLESS,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        page.goto((ROOT / "serve/static/index.html").as_uri(), wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        page.evaluate(
            """data => {
                const $ = id => document.getElementById(id);
                const important = (el, prop, value) => el && el.style.setProperty(prop, value, 'important');
                document.body.style.background = '#f4f7fb';
                const title = document.querySelector('.header h1');
                if (title) title.textContent = 'AutoTerm-SST';
                const subtitle = document.querySelector('.header p');
                if (subtitle) subtitle.textContent = 'Retrieval-aware streaming speech translation';
                const gp = $('glossaryPreset');
                if (gp) {
                    if (!Array.from(gp.options).some(o => o.value === 'auto_working')) {
                        const opt = document.createElement('option');
                        opt.value = 'auto_working';
                        opt.textContent = 'Automatic terminology';
                        gp.insertBefore(opt, gp.firstChild);
                    }
                    gp.value = 'auto_working';
                }
                if ($('glossaryMeta')) $('glossaryMeta').textContent = 'Preset: Automatic terminology · Manual: 0 · Index ready';
                if ($('adaptiveMode')) $('adaptiveMode').textContent = 'Automatic';
                if ($('adaptiveTopic')) $('adaptiveTopic').textContent = data.topic.active_domain || 'general';
                if ($('adaptiveConfidence')) $('adaptiveConfidence').textContent = Number(data.topic.confidence || 0).toFixed(2);
                const activePreset = data.topic.active_glossary_preset || 'common_10k';
                const activeLabel = activePreset === 'common_10k' ? 'common-terms diagnostic' : activePreset;
                if ($('adaptiveGlossary')) $('adaptiveGlossary').textContent = activeLabel;
                if ($('adaptiveTerms')) $('adaptiveTerms').textContent = Number(data.active_terms || 10000).toLocaleString();
                if ($('adaptiveSwitches')) $('adaptiveSwitches').textContent = String(data.topic.switch_count || 0);
                if ($('adaptiveRouter')) $('adaptiveRouter').textContent = `${data.router.action || 'stay'}: ${data.router.reason || 'embedding_refs'}`;
                if ($('statusText')) $('statusText').textContent = 'Streaming';
                if ($('statusIndicator')) $('statusIndicator').className = 'status-indicator processing';
                const header = document.querySelector('.translation-header h3');
                if (header) header.textContent = 'Live Translation';
                const panel = $('translationPanel');
                if (panel) {
                    important(panel, 'display', 'block');
                    important(panel, 'position', 'fixed');
                    important(panel, 'width', '920px');
                    important(panel, 'max-width', '920px');
                    important(panel, 'height', '265px');
                    important(panel, 'left', '80px');
                    important(panel, 'right', 'auto');
                    important(panel, 'top', '625px');
                    important(panel, 'bottom', 'auto');
                    important(panel, 'transform', 'none');
                    important(panel, 'z-index', '3000');
                    important(panel, 'border-radius', '8px');
                }
                const content = $('translationContent');
                if (content) {
                    important(content, 'max-height', '195px');
                    important(content, 'padding', '12px 18px');
                    important(content, 'overflow', 'hidden');
                }
                const output = $('translationOutput');
                if (output) {
                    output.textContent = data.translation;
                    important(output, 'font-size', '16px');
                    important(output, 'line-height', '1.5');
                    important(output, 'min-height', '72px');
                    important(output, 'max-height', '82px');
                }
                const status = $('termMemoryStatus');
                if (status) {
                    status.textContent = `memory: maxsim · ${Number(data.active_terms || 10000).toLocaleString()} terms · retrieve ${data.retrieve_ms}ms · gen ${data.elapsed_ms}ms`;
                }
                const evidence = $('evidenceList');
                if (evidence) {
                    evidence.innerHTML = '';
                    for (const ref of data.refs || []) {
                        const row = document.createElement('div');
                        row.className = 'evidence-row used';
                        const score = Number(ref.score || 0).toFixed(2);
                        const source = ref.source || 'diagnostic:common_terms';
                        const sourceLabel = source === 'auto:common_10k' ? 'diagnostic:common_terms' : source;
                        row.innerHTML =
                          `<span class="ev-term" title="${ref.term || ''}">${ref.term || ''}</span>` +
                          `<span class="ev-arrow">→</span>` +
                          `<span class="ev-tr" title="${ref.translation || ''}">${ref.translation || ''}</span>` +
                          `<span class="ev-score">${score}</span>` +
                          `<span class="ev-source">${sourceLabel}</span>`;
                        evidence.appendChild(row);
                    }
                }
            }""",
            payload,
        )
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(OUT)


if __name__ == "__main__":
    main()
