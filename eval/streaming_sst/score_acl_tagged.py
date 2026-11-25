#!/usr/bin/env python3
"""ACL-tagged term-accuracy through the rasst-demo framework (validation).

Non-circular gold: the ACL-tagged glossary lists, per term, the `sentence_indices`
where it occurs (with its zh). For each tagged sentence we feed that sentence's
audio through the framework agent at the given latency multiplier (lm → chunk =
0.96·lm s, retriever lookback 1.92 s → varctx encode window), then score
term-recall = fraction of that sentence's tagged terms whose zh appears in the
output. Audio is fed in lm-sized chunks at ~real-time so segmentation matches the
trained streaming form.

    python eval/streaming_sst/score_acl_tagged.py --base-url http://aries:8011 \
        --preset acl_tagged_gs10k --lm 2 --limit 60 \
        --glossary <acl6060_tagged_gt_union_gs10000_..._sentence_ids.json> \
        --seg-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments/seg
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import urllib.parse
import urllib.request
import wave
from typing import Any, Dict, List, Tuple

import numpy as np
import websockets

UNIT = 0.96  # base segment seconds; chunk = UNIT * lm


def read_wav(p: str) -> np.ndarray:
    with wave.open(p) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def strip_tags(t: str) -> str:
    return re.sub(r"</?t>", "", t or "")


def build_gold(glossary_path: str, lang: str) -> Dict[int, List[Tuple[str, str]]]:
    """sentence_index -> [(term, zh)] for tagged terms that occur there."""
    glo = json.load(open(glossary_path, encoding="utf-8"))
    entries = glo if isinstance(glo, list) else list(glo.values())
    gold: Dict[int, List[Tuple[str, str]]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        zh = str((e.get("target_translations") or {}).get(lang) or "").strip()
        term = str(e.get("term") or "").strip()
        if not zh or not term:
            continue
        for si in e.get("sentence_indices") or []:
            try:
                gold.setdefault(int(si), []).append((term, zh))
            except (TypeError, ValueError):
                continue
    return gold


def build_gt_keys(glossary_path: str) -> set:
    """The curated domain set (term_key, lowercased) = entries that carry
    `sentence_indices` (the 238 ACL GT terms). Used as the "relevant" set for
    retrieval precision: a retrieved ref is relevant iff its key is in here;
    everything else injected is distractor noise."""
    glo = json.load(open(glossary_path, encoding="utf-8"))
    entries = glo if isinstance(glo, list) else list(glo.values())
    keys = set()
    for e in entries:
        if isinstance(e, dict) and e.get("sentence_indices"):
            k = str(e.get("term_key") or e.get("term") or "").strip().lower()
            if k:
                keys.add(k)
    return keys


def ref_key(r: Dict[str, Any]) -> str:
    return str(r.get("key") or r.get("term_key") or r.get("term") or "").strip().lower()


async def translate_sentence(base_url: str, lang: str, preset: str, lm: int, pcm: np.ndarray
                             ) -> Tuple[str, List[Dict[str, Any]], int]:
    q = urllib.parse.urlencode({
        "agent_type": "RASST", "language_pair": lang, "glossary_preset": preset, "latency_multiplier": lm,
    })
    with urllib.request.urlopen(urllib.request.Request(f"{base_url}/init?{q}", data=b""), timeout=30) as r:
        sid = json.load(r)["session_id"]
    parts: List[str] = []
    refs: List[Dict[str, Any]] = []   # flattened injected refs across all chunks
    n_chunks = [0]                    # partials (= generated increments) that carried refs
    chunk = int(UNIT * lm * 16000)

    async def drain(ws, timeout: float) -> bool:
        """Collect partials until one arrives (return True) or timeout (False)."""
        got = False
        try:
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=timeout)
                o = json.loads(m)
                if o.get("type") == "partial":
                    parts.append(strip_tags(o.get("text") or ""))
                    rr = (o.get("meta") or {}).get("references")
                    if rr is not None:
                        refs.extend(rr)
                        n_chunks[0] += 1
                    got = True
                    return got
        except asyncio.TimeoutError:
            return got

    try:
        async with websockets.connect(f"{base_url.replace('http','ws')}/wss/{sid}?event_format=json", max_size=None) as ws:
            await ws.recv()
            # lock-step: send one segment, wait for its partial (paced by generation,
            # guarantees increment == one segment == the trained chunk form)
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i:i + chunk].tobytes())
                await drain(ws, timeout=12.0)
            await ws.send("EOF")
            # final flush
            for _ in range(3):
                if not await drain(ws, timeout=6.0):
                    break
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{base_url}/delete_session?{urllib.parse.urlencode({'session_id': sid})}", data=b""), timeout=15)
        except Exception:  # noqa: BLE001
            pass
    return "".join(parts), refs, n_chunks[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://aries:8011")
    ap.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments/seg")
    ap.add_argument("--glossary", default="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill_sentence_ids.json")
    ap.add_argument("--preset", default="acl_tagged_gs10k")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--lm", type=int, default=2)
    ap.add_argument("--limit", type=int, default=60, help="number of tagged sentences to eval (0 = all)")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    gold = build_gold(args.glossary, args.target_lang)
    gt_keys = build_gt_keys(args.glossary)  # the 238 curated domain terms = relevant set
    segs = sorted(glob.glob(f"{args.seg_dir}/*.wav"))
    sids = sorted(si for si in gold if si < len(segs))  # tagged sentences with audio
    if args.limit:
        sids = sids[: args.limit]
    n_terms = sum(len(gold[si]) for si in sids)
    print(f"[acl-tagged] preset={args.preset} lm={args.lm} | {len(sids)} tagged sentences, "
          f"{n_terms} gold term-occurrences | {len(gt_keys)} curated GT terms (relevant set)")

    hit = 0
    total = 0
    refs_total = 0     # all injected refs across all chunks/sentences
    refs_gt = 0        # injected refs whose key is a curated GT term (relevant)
    chunks_total = 0   # generated increments that carried a reference list
    per_sent = []
    for si in sids:
        pcm = read_wav(segs[si]).astype(np.float32)
        out, refs, nchunks = asyncio.run(
            translate_sentence(args.base_url, args.language_pair, args.preset, args.lm, pcm))
        got = [(t, z) for (t, z) in gold[si] if z in out]
        hit += len(got)
        total += len(gold[si])
        refs_total += len(refs)
        refs_gt += sum(1 for r in refs if ref_key(r) in gt_keys)
        chunks_total += nchunks
        per_sent.append({"sentence_index": si, "gold": len(gold[si]), "hit": len(got),
                         "refs": len(refs), "refs_gt": sum(1 for r in refs if ref_key(r) in gt_keys),
                         "terms": [f"{t}->{z}{'✓' if (t,z) in got else '✗'}" for (t, z) in gold[si]]})
        print(f"  sent#{si}: {len(got)}/{len(gold[si])}  refs={len(refs)}  {[f'{t}->{z}' for (t,z) in got][:6]}")

    precision = round(refs_gt / refs_total, 4) if refs_total else 0.0
    refs_per_chunk = round(refs_total / chunks_total, 4) if chunks_total else 0.0
    acc = round(hit / total, 4) if total else 0.0
    print(f"\n[acl-tagged] TERM_RECALL (occurrence-weighted) = {hit}/{total} = {acc}  (lm={args.lm}, preset={args.preset})")
    print(f"[acl-tagged] RETRIEVAL_PRECISION (refs in curated GT / all injected refs) = "
          f"{refs_gt}/{refs_total} = {precision}  | refs/chunk = {refs_per_chunk}")
    if args.out_json:
        json.dump({"preset": args.preset, "lm": args.lm, "term_recall": acc, "hit": hit, "total": total,
                   "retrieval_precision": precision, "refs_gt": refs_gt, "refs_total": refs_total,
                   "refs_per_chunk": refs_per_chunk, "chunks": chunks_total,
                   "sentences": per_sent}, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
