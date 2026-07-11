#!/usr/bin/env python3
"""Time-aligned occurrence-level term accuracy using MFA alignments.

Protocol (see eval/streaming_sst/mfa_alignments/README.md): every gold term
annotation is anchored to source-audio seconds — ACL annotations by matching
the term's token sequence in the MFA ``words`` tier, medicine annotations by
the oracle rows' start/end seconds. Exact-span source aliases with the same
normalized target variants are one spoken occurrence in the TERM_ACC
denominator. A hit requires a distinct appearance of an acceptable target
variant inside the streaming output emitted for the window [t_start - PRE_S,
t_end + POST_S]; output appearances are assigned greedily one-to-one in time
order, so a translation rendered once can satisfy at most one occurrence
regardless of how the windows overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, replace
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
ALIAS_DEDUP_RULE_ID = "exact_span_same_normalized_target_variants_v1"
ALIAS_DEDUP_RULE = (
    "collapse annotations with identical domain, block_index, exact t_start/t_end, "
    "and normalized sorted target variants; preserve all source aliases"
)


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
    source_aliases: Sequence[str] = ()
    raw_annotation_rows: int = 1


def _normalised_target_variants(variants: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalise_space(unicodedata.normalize("NFKC", variant)).casefold()
                for variant in variants
                if normalise_space(unicodedata.normalize("NFKC", variant))
            }
        )
    )


def _normalised_source_aliases(occurrence: TimedOccurrence) -> tuple[str, ...]:
    terms = (occurrence.term, *occurrence.source_aliases)
    by_key: Dict[str, str] = {}
    for term in terms:
        normalized = normalise_space(unicodedata.normalize("NFKC", term))
        if normalized:
            by_key.setdefault(normalized.casefold(), normalized)
    canonical_key = normalise_space(
        unicodedata.normalize("NFKC", occurrence.term)
    ).casefold()
    return tuple(
        by_key[key]
        for key in sorted(by_key)
        if key != canonical_key
    )


def deduplicate_alias_occurrences(
    occurrences: Sequence[TimedOccurrence],
) -> List[TimedOccurrence]:
    """Collapse exact-span aliases while retaining every source annotation."""

    deduplicated: List[TimedOccurrence] = []
    key_to_index: Dict[tuple[str, int, float, float, tuple[str, ...]], int] = {}
    for occurrence in occurrences:
        key = (
            occurrence.domain,
            occurrence.block_index,
            occurrence.t_start,
            occurrence.t_end,
            _normalised_target_variants(occurrence.variants),
        )
        existing_index = key_to_index.get(key)
        if existing_index is None:
            key_to_index[key] = len(deduplicated)
            deduplicated.append(
                replace(
                    occurrence,
                    source_aliases=_normalised_source_aliases(occurrence),
                )
            )
            continue

        existing = deduplicated[existing_index]
        merged_alias_holder = replace(
            existing,
            source_aliases=(
                *existing.source_aliases,
                occurrence.term,
                *occurrence.source_aliases,
            ),
        )
        deduplicated[existing_index] = replace(
            existing,
            source_aliases=_normalised_source_aliases(merged_alias_holder),
            raw_annotation_rows=(
                existing.raw_annotation_rows + occurrence.raw_annotation_rows
            ),
        )
    return deduplicated


def raw_annotation_count(occurrences: Sequence[TimedOccurrence]) -> int:
    return sum(occurrence.raw_annotation_rows for occurrence in occurrences)


def load_acl_segment_map(acl_root: str) -> Dict[str, List[tuple[float, float, float]]]:
    """Per talk: list of (orig_offset, orig_offset+duration, played_start) in play order.

    The playlist concatenates the talk's segment wavs back-to-back (gaps
    removed), so a TextGrid time (original-talk seconds) maps to played seconds
    only through the segment it falls in.
    """
    meta_path = Path(acl_root) / "segments.meta.jsonl"
    rows = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_talk: Dict[str, List[dict]] = {}
    for r in rows:
        by_talk.setdefault(str(r.get("talk")), []).append(r)
    out: Dict[str, List[tuple[float, float, float]]] = {}
    for talk, segs in by_talk.items():
        segs.sort(key=lambda r: int(r.get("index", 0)))
        played = 0.0
        spans: List[tuple[float, float, float]] = []
        for r in segs:
            off = float(r["offset"]); dur = float(r["duration"])
            spans.append((off, off + dur, played))
            played += float(r.get("seg_duration") or dur)
        out[talk] = spans
    return out


def map_orig_to_played(spans: List[tuple[float, float, float]], t_orig: float) -> float | None:
    for off, end, played_start in spans:
        if off <= t_orig < end:
            return played_start + (t_orig - off)
    return None


def block_source_window(
    block: Dict[str, Any],
    span: Dict[str, Any],
) -> tuple[str, float, float]:
    original_item_id = str(block.get("original_item_id") or block.get("item_id") or "")
    source_offset_samples = int(block.get("source_offset_samples") or 0)
    raw_source_end = int(block.get("source_end_samples") or 0)
    if raw_source_end:
        source_end_samples = raw_source_end
    else:
        source_end_samples = source_offset_samples + int(span["sample_count"])
    if source_end_samples <= source_offset_samples:
        raise ValueError(
            f"{original_item_id}: invalid source window "
            f"[{source_offset_samples}, {source_end_samples})"
        )
    expected_samples = source_end_samples - source_offset_samples
    if expected_samples != int(span["sample_count"]):
        raise ValueError(
            f"{original_item_id}: source window has {expected_samples} samples, "
            f"span has {span['sample_count']}"
        )
    return (
        original_item_id,
        source_offset_samples / TARGET_SAMPLE_RATE,
        source_end_samples / TARGET_SAMPLE_RATE,
    )


def build_timed_gold(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[TimedOccurrence]]:
    blocks = payload.get("blocks") or []
    spans = {int(s["block_index"]): s for s in payload.get("block_spans") or []}
    acl_tech = load_gold_entries(args.acl_technical_gold, target_lang=args.target_lang)
    acl_raw = load_gold_entries(args.acl_raw_glossary, target_lang=args.target_lang)
    mfa_root = Path(args.mfa_root)
    seg_map = load_acl_segment_map(args.acl_root)
    gold: Dict[str, List[TimedOccurrence]] = {"technical_plus_medicine": [], "raw_plus_medicine": []}
    for block_index, block in enumerate(blocks, start=1):
        span = spans[block_index]
        block_start = int(span["start_sample"]) / TARGET_SAMPLE_RATE
        item, source_offset, source_end = block_source_window(block, span)
        if block.get("corpus") == "acl":
            tg = mfa_root / "acl6060" / item / f"{item}.TextGrid"
            words = parse_textgrid_words(tg)
            talk_spans = seg_map.get(item, [])
            for gold_key, entries in (("technical_plus_medicine", acl_tech), ("raw_plus_medicine", acl_raw)):
                for term, variants in entries:
                    for ts, te in find_term_occurrences(words, term):
                        ps = map_orig_to_played(talk_spans, ts)
                        pe = map_orig_to_played(talk_spans, te)
                        if ps is None:  # word fell in a removed inter-segment gap
                            continue
                        if pe is None:
                            pe = ps
                        if ps < source_offset or ps >= source_end:
                            continue
                        local_start = ps - source_offset
                        local_end = min(
                            max(pe - source_offset, local_start),
                            source_end - source_offset,
                        )
                        gold[gold_key].append(
                            TimedOccurrence(
                                "nlp",
                                block_index,
                                term,
                                variants,
                                block_start + local_start,
                                block_start + local_end,
                            )
                        )
        elif block.get("corpus") == "medicine":
            mid = item.removeprefix("medicine_")
            oracle_path = (
                Path(args.medicine_oracle_dir)
                / f"hard_medicine.oracle_term_map__medicine_{mid}.json"
            )
            rows = json.loads(oracle_path.read_text(encoding="utf-8"))
            for row in rows:
                ts = float(row.get("start_sec") or 0.0)
                te = float(row.get("end_sec") or ts)
                if ts < source_offset or ts >= source_end:
                    continue
                for ref in row.get("references") or []:
                    term = normalise_space(ref.get("term") or "")
                    translation = normalise_space(ref.get("translation") or "")
                    if term and translation:
                        occ = TimedOccurrence(
                            "medicine",
                            block_index,
                            term,
                            [translation],
                            block_start + (ts - source_offset),
                            block_start
                            + min(
                                max(te - source_offset, ts - source_offset),
                                source_end - source_offset,
                            ),
                        )
                        gold["technical_plus_medicine"].append(occ)
                        gold["raw_plus_medicine"].append(occ)
    return {
        gold_key: deduplicate_alias_occurrences(occurrences)
        for gold_key, occurrences in gold.items()
    }


# --------------------------------------------------------------------------- scoring
def score_run(payload: Dict[str, Any], gold: Sequence[TimedOccurrence], target_lang: str) -> Dict[str, Any]:
    gold = deduplicate_alias_occurrences(gold)
    annotation_rows = raw_annotation_count(gold)
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
        "raw_annotation_rows": annotation_rows,
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
    ap.add_argument("--acl-root", required=True, help="dir with segments.meta.jsonl (segment offsets)")
    ap.add_argument("--acl-technical-gold", required=True)
    ap.add_argument("--acl-raw-glossary", required=True)
    ap.add_argument("--medicine-oracle-dir", required=True)
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--post-s", type=float, default=30.0)
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
                results[f"gold_{key}"] = {
                    "total": len(occs),
                    "raw_annotation_rows": raw_annotation_count(occs),
                    "alias_dedup_denominator": len(occs),
                    "alias_dedup_rule_id": ALIAS_DEDUP_RULE_ID,
                    **doms,
                }
        results["runs"][name] = {
            key: score_run(payload, occs, args.target_lang) for key, occs in gold_cache.items()
        }
    text = json.dumps(results, ensure_ascii=False, indent=1)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
