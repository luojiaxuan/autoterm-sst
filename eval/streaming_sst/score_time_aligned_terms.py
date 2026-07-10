#!/usr/bin/env python3
"""Time-aligned occurrence-level term accuracy using MFA alignments.

Protocol (see eval/streaming_sst/mfa_alignments/README.md): every gold term
occurrence is anchored to source-audio seconds — ACL occurrences by matching
the term's token sequence in the MFA ``words`` tier, medicine occurrences by
the oracle rows' start/end seconds. A hit requires a distinct appearance of an
acceptable target variant inside the streaming output emitted for the window
[t_start - PRE_S, t_end + POST_S]; output appearances are assigned greedily
one-to-one in time order, so a translation rendered once can satisfy at most
one occurrence regardless of how the windows overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.streaming_sst.score_mixed_audio_terms import (  # noqa: E402
    _ALNUM_TERM_RE,
    _LATIN_TRANSLATION_TARGETS,
    _latin_stem,
    allowed_identity_retention_source,
    contains_cjk_or_kana,
    load_gold_entries,
    normalise_space,
)

TARGET_SAMPLE_RATE = 16000
PRE_S = 2.0
POST_S = 30.0


# --------------------------------------------------------------------------- TextGrid
def parse_textgrid_words(path: Path) -> List[tuple[float, float, str]]:
    """Parse a short-format TextGrid and return the ``words`` tier intervals."""
    tokens: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            tokens.append(line)
    out: List[tuple[float, float, str]] = []
    idx = 0
    n = len(tokens)
    while idx < n:
        if tokens[idx] == '"IntervalTier"' and idx + 1 < n:
            name = tokens[idx + 1].strip('"')
            count = int(float(tokens[idx + 4]))
            pos = idx + 5
            intervals: List[tuple[float, float, str]] = []
            for _ in range(count):
                xmin = float(tokens[pos])
                xmax = float(tokens[pos + 1])
                text = tokens[pos + 2].strip('"')
                intervals.append((xmin, xmax, text))
                pos += 3
            if name == "words":
                out = intervals
                break
            idx = pos
        else:
            idx += 1
    if not out:
        raise ValueError(f"no words tier found in {path}")
    return out


_WORD_CLEAN_RE = re.compile(r"^[^0-9a-z]+|[^0-9a-z]+$")


def _clean_token(token: str) -> str:
    return _WORD_CLEAN_RE.sub("", token.lower())


def term_tokens(term: str) -> List[str]:
    parts = re.split(r"[\s\-/]+", normalise_space(term).lower())
    return [_clean_token(p) for p in parts if _clean_token(p)]


def find_term_occurrences(
    words: Sequence[tuple[float, float, str]], term: str
) -> List[tuple[float, float]]:
    """All (t_start, t_end) matches of the term's token sequence in the tier."""
    toks = term_tokens(term)
    if not toks:
        return []
    stream = [(s, e, _clean_token(t)) for s, e, t in words if _clean_token(t)]
    out: List[tuple[float, float]] = []
    k = len(toks)
    for i in range(len(stream) - k + 1):
        window = stream[i : i + k]
        ok = True
        for (_, _, w), want in zip(window, toks):
            if w == want:
                continue
            # light inflection tolerance mirroring term_in_source leniency
            if w.rstrip("s") == want.rstrip("s") and min(len(w), len(want)) >= 3:
                continue
            ok = False
            break
        if ok:
            out.append((window[0][0], window[-1][1]))
    return out


# --------------------------------------------------------------------------- matches in output
@dataclass
class OutputIndex:
    text: str
    char_to_time: List[tuple[int, float]]  # (char_end_offset, cursor_s)

    def time_at(self, char_pos: int) -> float:
        ends = [c for c, _ in self.char_to_time]
        i = min(bisect_right(ends, char_pos), len(ends) - 1)
        return self.char_to_time[i][1]


def build_output_index(payload: Dict[str, Any], block_index: int) -> OutputIndex:
    spans = {int(s["block_index"]): s for s in payload.get("block_spans") or []}
    span = spans[block_index]
    parts: List[str] = []
    char_to_time: List[tuple[int, float]] = []
    total = 0
    for record in payload.get("records") or []:
        cursor = int(record.get("cursor_samples") or 0)
        if int(span["start_sample"]) < cursor <= int(span["end_sample"]):
            text = str(record.get("text") or record.get("text_preview") or "")
            parts.append(text)
            total += len(text)
            char_to_time.append((total, cursor / TARGET_SAMPLE_RATE))
    return OutputIndex("".join(parts), char_to_time)


def variant_match_times(index: OutputIndex, term: str, variants: Sequence[str], target_lang: str) -> List[float]:
    """Global times of distinct acceptable-variant appearances (best variant kind)."""
    latin_target = target_lang in _LATIN_TRANSLATION_TARGETS

    def positions(pattern: str, regex: bool) -> List[int]:
        if regex:
            return [m.start() for m in re.finditer(pattern, index.text, re.IGNORECASE)]
        out, start = [], 0
        while True:
            i = index.text.find(pattern, start)
            if i < 0:
                return out
            out.append(i)
            start = i + max(len(pattern), 1)

    best: List[int] = []
    for variant in variants:
        v = normalise_space(variant)
        if not v:
            continue
        if contains_cjk_or_kana(v):
            pos = positions(v, regex=False)
        elif latin_target and v.casefold() != normalise_space(term).casefold():
            if len(v) < 4 or _ALNUM_TERM_RE.fullmatch(v):
                pos = positions(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", regex=True)
            else:
                low = index.text.casefold()
                pos, start = [], 0
                needle = v.casefold()
                while True:
                    i = low.find(needle, start)
                    if i < 0:
                        break
                    pos.append(i)
                    start = i + len(needle)
                if not pos:
                    stems = [_latin_stem(w) for w in needle.split()]
                    if stems and all(len(s) >= 4 for s in stems) and all(s in low for s in stems):
                        pos = [low.find(stems[0])]
        else:
            continue
        if len(pos) > len(best):
            best = pos
    if not best and allowed_identity_retention_source(term):
        for variant in variants:
            v = normalise_space(variant)
            if v and not contains_cjk_or_kana(v):
                pos = positions(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", regex=True)
                if len(pos) > len(best):
                    best = pos
    return sorted(index.time_at(p) for p in best)


# --------------------------------------------------------------------------- gold building
@dataclass
class TimedOccurrence:
    domain: str
    block_index: int
    term: str
    variants: Sequence[str]
    t_start: float  # global playlist seconds
    t_end: float


def build_timed_gold(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[TimedOccurrence]]:
    blocks = payload.get("blocks") or []
    spans = {int(s["block_index"]): s for s in payload.get("block_spans") or []}
    acl_tech = load_gold_entries(args.acl_technical_gold, target_lang=args.target_lang)
    acl_raw = load_gold_entries(args.acl_raw_glossary, target_lang=args.target_lang)
    mfa_root = Path(args.mfa_root)
    gold: Dict[str, List[TimedOccurrence]] = {"technical_plus_medicine": [], "raw_plus_medicine": []}
    for block_index, block in enumerate(blocks, start=1):
        span = spans[block_index]
        block_start = int(span["start_sample"]) / TARGET_SAMPLE_RATE
        duration = int(span["sample_count"]) / TARGET_SAMPLE_RATE
        item = str(block["item_id"])
        if block.get("corpus") == "acl":
            tg = mfa_root / "acl6060" / item / f"{item}.TextGrid"
            words = parse_textgrid_words(tg)
            for gold_key, entries in (("technical_plus_medicine", acl_tech), ("raw_plus_medicine", acl_raw)):
                for term, variants in entries:
                    for ts, te in find_term_occurrences(words, term):
                        if ts >= duration:
                            continue
                        gold[gold_key].append(
                            TimedOccurrence("nlp", block_index, term, variants, block_start + ts, block_start + te)
                        )
        elif block.get("corpus") == "medicine":
            mid = item.removeprefix("medicine_")
            rows = json.load(open(Path(args.medicine_oracle_dir) / f"hard_medicine.oracle_term_map__medicine_{mid}.json"))
            for row in rows:
                ts = float(row.get("start_sec") or 0.0)
                te = float(row.get("end_sec") or ts)
                if ts >= duration:
                    continue
                for ref in row.get("references") or []:
                    term = normalise_space(ref.get("term") or "")
                    translation = normalise_space(ref.get("translation") or "")
                    if term and translation:
                        occ = TimedOccurrence("medicine", block_index, term, [translation], block_start + ts, block_start + te)
                        gold["technical_plus_medicine"].append(occ)
                        gold["raw_plus_medicine"].append(occ)
    return gold


# --------------------------------------------------------------------------- scoring
def score_run(payload: Dict[str, Any], gold: Sequence[TimedOccurrence], target_lang: str) -> Dict[str, Any]:
    by_block_term: Dict[tuple[int, str, tuple[str, ...]], List[TimedOccurrence]] = {}
    for occ in gold:
        by_block_term.setdefault((occ.block_index, occ.term, tuple(occ.variants)), []).append(occ)
    hits = 0
    dom_tot: Dict[str, int] = {}
    dom_hit: Dict[str, int] = {}
    output_cache: Dict[int, OutputIndex] = {}
    for (block_index, term, variants), occs in by_block_term.items():
        if block_index not in output_cache:
            output_cache[block_index] = build_output_index(payload, block_index)
        times = variant_match_times(output_cache[block_index], term, variants, target_lang)
        occs = sorted(occs, key=lambda o: o.t_start)
        used = [False] * len(times)
        for occ in occs:
            dom_tot[occ.domain] = dom_tot.get(occ.domain, 0) + 1
            lo, hi = occ.t_start - PRE_S, occ.t_end + POST_S
            got = False
            for i, mt in enumerate(times):
                if used[i] or mt < lo:
                    continue
                if mt > hi:
                    break
                used[i] = True
                got = True
                break
            hits += int(got)
            dom_hit[occ.domain] = dom_hit.get(occ.domain, 0) + int(got)
    total = len(gold)
    return {
        "gold_occurrences": total,
        "hits": hits,
        "term_acc": round(hits / total, 4) if total else None,
        "by_domain": {
            d: {"gold_occurrences": dom_tot[d], "hits": dom_hit.get(d, 0),
                "term_acc": round(dom_hit.get(d, 0) / dom_tot[d], 4)}
            for d in sorted(dom_tot)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--mfa-root", required=True)
    ap.add_argument("--acl-technical-gold", required=True)
    ap.add_argument("--acl-raw-glossary", required=True)
    ap.add_argument("--medicine-oracle-dir", required=True)
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--post-s", type=float, default=POST_S)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    global POST_S
    POST_S = args.post_s

    results: Dict[str, Any] = {"post_s": POST_S, "pre_s": PRE_S, "runs": {}}
    gold_cache: Dict[str, List[TimedOccurrence]] | None = None
    for spec in args.run:
        name, path = spec.split("=", 1)
        payload = json.load(open(path, encoding="utf-8"))
        if gold_cache is None:
            gold_cache = build_timed_gold(payload, args)
            for key, occs in gold_cache.items():
                doms: Dict[str, int] = {}
                for o in occs:
                    doms[o.domain] = doms.get(o.domain, 0) + 1
                results[f"gold_{key}"] = {"total": len(occs), **doms}
        results["runs"][name] = {
            key: score_run(payload, occs, args.target_lang) for key, occs in gold_cache.items()
        }
    text = json.dumps(results, ensure_ascii=False, indent=1)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
