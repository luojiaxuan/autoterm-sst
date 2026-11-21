#!/usr/bin/env python3
"""Generate each slice's router prototype texts from its own glossary with an LLM.

For every ``<slice>/glossary.json`` under ``--slices-root``, all remaining English terms and
their Chinese translations are given to the model, which returns the same three texts the
router embeds for a slice: a one-line description, one English sentence listing the field's
core topics, and its Chinese translation. Entries whose ``source`` is listed in
``--exclude-source`` (benchmark-derived terms) are dropped first, so the prototypes never see
evaluation vocabulary. ``--samples`` independent generations per slice measure how much the
routing result depends on one particular wording.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """Below is a bilingual glossary of one subject field. Each line is an English term and its Chinese translation.

Write three short texts that describe this field. A multilingual sentence encoder will compare them with the running translation of a talk to decide whether the talk is about this field.
1. description: one English line of at most 12 words, of the form "<kind>, <kind>, ..., and <kind> terms.", naming the main kinds of terms in the glossary.
2. en: one English sentence that names the field and then lists 7 to 10 of its core topics, separated by commas.
3. zh: the Chinese translation of the en sentence.

Base the texts only on the glossary. Return JSON with the keys description, en and zh.

Glossary:
{glossary}"""

SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "en": {"type": "string"}, "zh": {"type": "string"}},
    "required": ["description", "en", "zh"],
    "additionalProperties": False,
}


def build_prompt(rows: list[dict]) -> str:
    return PROMPT.format(glossary="\n".join(f"{r['term']} => {r['target_translations']['zh']}" for r in rows))


def generate(model: str, prompt: str, api_key: str) -> tuple[dict, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "field_texts", "strict": True, "schema": SCHEMA}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    return json.loads(payload["choices"][0]["message"]["content"]), payload["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slices-root", type=Path, required=True)
    parser.add_argument("--exclude-source", action="append", default=[], help="glossary source to drop; repeatable")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api_key = os.environ["OPENAI_API_KEY"]

    slices = {}
    for path in sorted(args.slices_root.glob("*/glossary.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        slices[path.parent.name] = [r for r in rows if r["source"] not in args.exclude_source]

    jobs = [(name, i) for name in slices for i in range(args.samples)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: generate(args.model, build_prompt(slices[job[0]]), api_key), jobs))

    out = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "requested_model": args.model,
        "served_models": sorted({served for _, served in results}),
        "excluded_sources": args.exclude_source,
        "prompt_template": PROMPT,
        "slices": {name: {"terms_used": len(rows), "samples": []} for name, rows in slices.items()},
    }
    for (name, _), (texts, _) in zip(jobs, results):
        out["slices"][name]["samples"].append(texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, entry in out["slices"].items():
        print(f"{name} ({entry['terms_used']} terms): {entry['samples'][0]['en']}")


if __name__ == "__main__":
    main()
