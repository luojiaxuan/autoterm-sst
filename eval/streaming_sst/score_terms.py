#!/usr/bin/env python3
"""Terminology accuracy and masked-term BLEU over the framework JSON WS.

For each preset, streams the segments and concatenates the model output, then
scores against gold terms = glossary entries whose English term appears in the
aligned source text. Measures the terminology axis the open memory targets:

* term_recall  = gold terms whose target translation appears in the output
* false_copy   = gold terms whose English form is copied verbatim into the output
                 (i.e. left untranslated)
* masked_terms_bleu = BLEU after target-side glossary translations are removed
                 from both the hypothesis and reference, matching the RASST
                 main-result MASKED_TERMS_BLEU definition

    python eval/streaming_sst/score_terms.py --base-url http://aries:8011 \
        --seg-dir <smoke>/seg --source-text <smoke>/source_text.txt \
        --reference-text <smoke>/ref.txt --glossary <ACL glossary.json> \
        --presets none,acl_tagged_raw,open_wiki_academic
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover - only collect_output needs this optional dependency
    websockets = None


TARGET_SAMPLE_RATE = 16000


def resolve_chunk_samples(chunk: int, base_segment_sec: float, latency_multiplier: int) -> int:
    try:
        explicit = int(chunk)
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return explicit
    try:
        lm = max(1, min(4, int(latency_multiplier)))
    except (TypeError, ValueError):
        lm = 2
    try:
        base_sec = float(base_segment_sec)
    except (TypeError, ValueError):
        base_sec = 0.96
    return max(1, int(round(base_sec * lm * TARGET_SAMPLE_RATE)))


_ALNUM_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+/#&%()-]*$")
_CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_wav(p: str) -> np.ndarray:
    with wave.open(p) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_gold(glossary_path: str, source_text: str, lang: str) -> List[Tuple[str, List[str]]]:
    """Circular gold: glossary's own terms appearing in the source (kept for ref)."""
    glo = json.load(open(glossary_path, encoding="utf-8"))
    entries = glo if isinstance(glo, list) else list(glo.values())
    src = source_text.lower()
    gold: List[Tuple[str, List[str]]] = []
    seen = set()
    for e in entries:
        term = str(e.get("term") or "").strip()
        zh = str((e.get("target_translations") or {}).get(lang) or "").strip()
        if not term or not zh:
            continue
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", src):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                gold.append((term, [zh]))
    return gold


def load_gold_file(path: str) -> List[Tuple[str, List[str]]]:
    """Independent gold: [{"en": "...", "zh": ["variant", ...]}]. Glossary-agnostic."""
    data = json.load(open(path, encoding="utf-8"))
    gold: List[Tuple[str, List[str]]] = []
    for e in data:
        en = str(e.get("en") or e.get("term") or "").strip()
        zh = e.get("zh") or e.get("variants") or []
        if isinstance(zh, str):
            zh = [zh]
        zh = [str(v).strip() for v in zh if str(v).strip()]
        if en and zh:
            gold.append((en, zh))
    return gold


def load_target_terms_for_masking(glossary_path: str, target_lang: str) -> List[str]:
    """Load target-side glossary translations for RASST-style masked BLEU."""
    data = json.load(open(glossary_path, encoding="utf-8"))
    if isinstance(data, dict):
        raw_entries: Iterable[Any] = data.values()
    elif isinstance(data, list):
        raw_entries = data
    else:
        raise ValueError(f"Unsupported glossary format for masked BLEU: {glossary_path}")

    terms: List[str] = []
    seen = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        translations = entry.get("target_translations")
        translation = ""
        if isinstance(translations, dict):
            translation = normalise_space(translations.get(target_lang) or "")
        if not translation:
            translation = normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(target_lang)
                or ""
            )
        if not translation:
            continue
        key = translation.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(translation)

    # Longer terms must be removed before shorter overlapping terms.
    terms.sort(key=lambda text: (len(text), text), reverse=True)
    return terms


def target_terms_from_gold(gold: List[Tuple[str, List[str]]]) -> List[str]:
    terms: List[str] = []
    seen = set()
    for _, variants in gold:
        for variant in variants:
            term = normalise_space(variant)
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    terms.sort(key=lambda text: (len(text), text), reverse=True)
    return terms


def term_to_mask_regex(term: str) -> re.Pattern[str]:
    term_norm = normalise_space(term)
    if not term_norm:
        raise ValueError("Cannot build a mask regex for an empty term.")
    escaped = re.escape(term_norm).replace(r"\ ", r"\s+")
    if _ALNUM_TERM_RE.fullmatch(term_norm):
        return re.compile(
            r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
    if len(term_norm) == 1 and _CJK_OR_KANA_RE.search(term_norm):
        return re.compile(
            r"(?<![\u3040-\u30ff\u3400-\u9fff])"
            + escaped
            + r"(?![\u3040-\u30ff\u3400-\u9fff])"
        )
    flags = 0 if _CJK_OR_KANA_RE.search(term_norm) else re.IGNORECASE
    return re.compile(escaped, flags=flags)


def compile_term_mask_patterns(target_terms: Sequence[str]) -> List[re.Pattern[str]]:
    return [term_to_mask_regex(term) for term in target_terms]


def mask_target_terms(text: str, term_patterns: Sequence[re.Pattern[str]]) -> Tuple[str, int]:
    masked = str(text or "")
    removed = 0
    for pattern in term_patterns:
        masked, count = pattern.subn(" ", masked)
        removed += count
    return normalise_space(masked), removed


def compute_bleu_scores(
    *,
    hypothesis: str,
    reference: str,
    target_terms: Sequence[str],
    sacrebleu_tokenizer: str,
) -> Dict[str, Any]:
    try:
        import sacrebleu
    except ImportError as exc:
        return {"bleu_error": f"sacrebleu is required for BLEU metrics: {exc}"}

    bleu = sacrebleu.corpus_bleu(
        [normalise_space(hypothesis)],
        [[normalise_space(reference)]],
        tokenize=sacrebleu_tokenizer,
    ).score
    patterns = compile_term_mask_patterns(target_terms)
    masked_hyp, hyp_removed = mask_target_terms(hypothesis, patterns)
    masked_ref, ref_removed = mask_target_terms(reference, patterns)
    masked_bleu = sacrebleu.corpus_bleu(
        [masked_hyp],
        [[masked_ref]],
        tokenize=sacrebleu_tokenizer,
    ).score
    return {
        "bleu": round(float(bleu), 4),
        "masked_terms_bleu": round(float(masked_bleu), 4),
        "delta_masked_minus_bleu": round(float(masked_bleu) - float(bleu), 4),
        "masked_terms_hyp_removed": hyp_removed,
        "masked_terms_ref_removed": ref_removed,
        "masked_terms_types": len(target_terms),
    }


def coverage(glossary_path: str, gold: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    """How many gold English terms exist in this glossary (any case)."""
    glo = json.load(open(glossary_path, encoding="utf-8"))
    entries = glo if isinstance(glo, list) else list(glo.values())
    terms = set(str(e.get("term", "")).lower() for e in entries if isinstance(e, dict))
    have = [en for en, _ in gold if en.lower() in terms]
    return {"covered": len(have), "of": len(gold), "missing": [en for en, _ in gold if en.lower() not in terms]}


def strip_tags(t: str) -> str:
    return re.sub(r"</?t>", "", t or "")


async def collect_output(
    base_url: str,
    lang: str,
    preset: str,
    pcm: np.ndarray,
    chunk: int = 0,
    feed_sleep: float = 0.45,
    latency_multiplier: int = 2,
    base_segment_sec: float = 0.96,
) -> Dict[str, Any]:
    if websockets is None:
        raise RuntimeError("websockets is required to collect framework JSON-WS output")
    chunk_samples = resolve_chunk_samples(chunk, base_segment_sec, latency_multiplier)
    try:
        lm = max(1, min(4, int(latency_multiplier)))
    except (TypeError, ValueError):
        lm = 2
    q = urllib.parse.urlencode(
        {"agent_type": "RASST", "language_pair": lang, "glossary_preset": preset, "latency_multiplier": lm}
    )
    with urllib.request.urlopen(urllib.request.Request(f"{base_url}/init?{q}", data=b""), timeout=30) as r:
        sid = json.load(r)["session_id"]
    parts: List[str] = []
    refs: List[Tuple[str, str]] = []
    ref_events: List[Dict[str, Any]] = []
    prompt_counts: List[int] = []
    pool_counts: List[int] = []
    shortfalls: List[int] = []
    rescue_events = 0
    chunks = 0
    try:
        async with websockets.connect(f"{base_url.replace('http','ws')}/wss/{sid}?event_format=json", max_size=None) as ws:
            await ws.recv()

            async def feed() -> None:
                for i in range(0, len(pcm), chunk_samples):
                    await ws.send(pcm[i:i + chunk_samples].tobytes())
                    await asyncio.sleep(feed_sleep)
                await ws.send("EOF")

            ft = asyncio.create_task(feed())
            idle = 0
            while idle < 1:
                try:
                    timeout_s = 10.0 if ft.done() and parts else 60.0
                    m = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    idle += 1
                    continue
                o = json.loads(m)
                if o.get("type") == "partial":
                    chunks += 1
                    parts.append(strip_tags(o.get("text") or ""))
                    meta = o.get("meta") or {}
                    prompt_counts.append(int(meta.get("prompt_reference_count") or 0))
                    pool_counts.append(int(meta.get("candidate_pool_count") or 0))
                    shortfalls.append(int(meta.get("prompt_candidate_shortfall") or 0))
                    if meta.get("open_wiki_rescue_triggered"):
                        rescue_events += 1
                    for r in meta.get("references") or []:
                        term = str(r.get("term") or "")
                        translation = str(r.get("translation") or "")
                        refs.append((term, translation))
                        event = dict(r)
                        event["chunk_index"] = chunks - 1
                        ref_events.append(event)
            await ft
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{base_url}/delete_session?{urllib.parse.urlencode({'session_id': sid})}", data=b""), timeout=15)
        except Exception:  # noqa: BLE001
            pass
    return {
        "text": "".join(parts),
        "refs": refs,
        "ref_events": ref_events,
        "chunks": chunks,
        "prompt_counts": prompt_counts,
        "candidate_pool_counts": pool_counts,
        "prompt_shortfalls": shortfalls,
        "rescue_events": rescue_events,
        "chunk_samples": chunk_samples,
        "latency_multiplier": lm,
    }


def contains_cjk_or_kana(text: str) -> bool:
    return bool(_CJK_OR_KANA_RE.search(str(text or "")))


def allowed_identity_retention_source(term: str) -> bool:
    clean = normalise_space(term)
    if not clean:
        return False
    compact = clean.replace("-", "").replace("_", "")
    if len(compact) >= 2 and compact.upper() == compact and re.search(r"[A-Z]", compact):
        return True
    if " " not in clean and re.search(r"[A-Z].*[A-Z]", compact) and not clean.islower():
        return True
    if re.search(r"[A-Za-z]+\d|\d+[A-Za-z]", compact):
        return True
    return False


def output_contains_variant(output: str, variant: str) -> bool:
    variant = normalise_space(variant)
    if not variant:
        return False
    if _ALNUM_TERM_RE.fullmatch(variant):
        return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])", output, re.IGNORECASE))
    return variant in output


def classify_output_hit(term: str, variants: Sequence[str], output: str) -> Tuple[bool, Optional[str], Optional[str]]:
    for variant in variants:
        if contains_cjk_or_kana(variant) and output_contains_variant(output, variant):
            return True, variant, "zh_translation"
    if allowed_identity_retention_source(term):
        for variant in variants:
            if not contains_cjk_or_kana(variant) and output_contains_variant(output, variant):
                kind = "acronym_retention" if term.upper() == term else "identity_retention"
                return True, variant, kind
    return False, None, None


def score(output: str, gold: List[Tuple[str, List[str]]], surfaced_terms: Optional[set[str]] = None) -> Dict[str, Any]:
    rec, fc, got = 0, 0, []
    zh_rec = 0
    identity_rec = 0
    surfaced_total = 0
    surfaced_rec = 0
    not_surfaced_total = 0
    not_surfaced_rec = 0
    traces: List[Dict[str, Any]] = []
    surfaced_terms = surfaced_terms or set()
    for term, variants in gold:
        hit, hit_variant, hit_kind = classify_output_hit(term, variants, output)
        if hit:
            rec += 1
            got.append(f"{term}->{hit_variant}")
            if hit_kind == "zh_translation":
                zh_rec += 1
            else:
                identity_rec += 1
        if not hit and re.search(r"\b" + re.escape(term.lower()) + r"\b", output.lower()):
            fc += 1
        surfaced = term.lower() in surfaced_terms
        if surfaced:
            surfaced_total += 1
            surfaced_rec += int(hit)
        else:
            not_surfaced_total += 1
            not_surfaced_rec += int(hit)
        traces.append(
            {
                "gold_source": term,
                "surfaced_in_prompt_top10": surfaced,
                "output_hit": hit,
                "output_variant": hit_variant,
                "output_hit_kind": hit_kind,
                "allow_identity_retention_proxy": allowed_identity_retention_source(term),
            }
        )
    n = len(gold) or 1
    return {
        "gold": len(gold),
        "term_recall": round(rec / n, 3),
        "false_copy": round(fc / n, 3),
        "translation_term_recall": round(zh_rec / n, 3),
        "identity_retention_recall": round(identity_rec / n, 3),
        "term_recall_surfaced": round(surfaced_rec / surfaced_total, 3) if surfaced_total else None,
        "term_recall_not_surfaced": round(not_surfaced_rec / not_surfaced_total, 3) if not_surfaced_total else None,
        "surfaced_gold_terms": surfaced_total,
        "not_surfaced_gold_terms": not_surfaced_total,
        "recovered": got,
        "gold_traces": traces,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://aries:8011")
    ap.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg")
    ap.add_argument("--source-text", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/source_text.txt")
    ap.add_argument("--glossary", default="/mnt/taurus/data2/jiaxuanluo/RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json")
    ap.add_argument("--reference-text", default="", help="target-language references; enables BLEU/masked_terms_bleu")
    ap.add_argument("--mask-glossary", default="", help="glossary whose target translations are masked; defaults to --glossary")
    ap.add_argument("--sacrebleu-tokenizer", default="zh", help="sacreBLEU tokenizer for BLEU/masked_terms_bleu")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--presets", default="none,acl_tagged_raw,open_wiki_academic")
    ap.add_argument("--max-segs", type=int, default=8)
    ap.add_argument("--gold-file", default="", help="independent gold JSON [{en, zh:[variants]}]; avoids circular gold")
    ap.add_argument("--coverage", default="", help="comma-sep glossary JSONs to report gold coverage for")
    ap.add_argument("--latency-multiplier", type=int, default=2)
    ap.add_argument("--base-segment-sec", type=float, default=0.96)
    ap.add_argument("--chunk", type=int, default=0, help="PCM samples per send; 0 uses base_segment_sec * latency_multiplier")
    ap.add_argument("--feed-sleep", type=float, default=0.45, help="per-chunk send delay (lower = faster than realtime)")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    src_lines = open(args.source_text, encoding="utf-8").read().splitlines()[: args.max_segs]
    source_text = "\n".join(src_lines)
    reference_text: Optional[str] = None
    if args.reference_text:
        reference_text = "\n".join(open(args.reference_text, encoding="utf-8").read().splitlines()[: args.max_segs])
    if args.gold_file:
        gold = load_gold_file(args.gold_file)
        print(f"[score] independent gold ({len(gold)} terms): {[g[0] for g in gold]}")
        for cg in [c for c in args.coverage.split(',') if c.strip()]:
            cov = coverage(cg, gold)
            print(f"[coverage] {cg.split('/')[-1]}: {cov['covered']}/{cov['of']} (missing: {cov['missing']})")
    else:
        gold = load_gold(args.glossary, source_text, args.target_lang)
    fs = sorted(glob.glob(f"{args.seg_dir}/*.wav"))[: args.max_segs]
    pcm = np.concatenate([read_wav(f) for f in fs]).astype(np.float32)
    print(f"[score] {len(fs)} segs; {len(gold)} gold terms: {[g[0] for g in gold]}")
    target_terms: List[str] = []
    if reference_text is not None:
        mask_glossary = args.mask_glossary or args.glossary
        if mask_glossary:
            target_terms = load_target_terms_for_masking(mask_glossary, args.target_lang)
            print(f"[score] masked BLEU terms from {mask_glossary}: {len(target_terms)}")
        else:
            target_terms = target_terms_from_gold(gold)
            print(f"[score] masked BLEU terms from gold variants: {len(target_terms)}")

    gold_en = set(en.lower() for en, _ in gold)
    rows = []
    for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
        try:
            res = asyncio.run(
                collect_output(
                    args.base_url,
                    args.language_pair,
                    preset,
                    pcm,
                    chunk=args.chunk,
                    feed_sleep=args.feed_sleep,
                    latency_multiplier=args.latency_multiplier,
                    base_segment_sec=args.base_segment_sec,
                )
            )
            refs = res["refs"]
            surfaced_terms = {en.lower() for en, _ in refs if en}
            row = {"preset": preset, **score(res["text"], gold, surfaced_terms=surfaced_terms)}
            gold_refs = sum(1 for en, _ in refs if en.lower() in gold_en)
            row["latency_multiplier"] = res.get("latency_multiplier")
            row["streaming_chunk_samples"] = res.get("chunk_samples")
            row["chunks"] = res["chunks"]
            row["refs_total"] = len(refs)
            row["refs_per_chunk"] = round(len(refs) / res["chunks"], 2) if res["chunks"] else 0.0
            row["avg_prompt_candidates"] = round(sum(res["prompt_counts"]) / len(res["prompt_counts"]), 3) if res["prompt_counts"] else 0.0
            row["avg_candidate_pool"] = round(sum(res["candidate_pool_counts"]) / len(res["candidate_pool_counts"]), 3) if res["candidate_pool_counts"] else 0.0
            row["prompt_shortfall_chunks"] = sum(1 for item in res["prompt_shortfalls"] if item)
            row["open_wiki_rescue_chunks"] = int(res.get("rescue_events") or 0)
            # precision: fraction of surfaced top-10 terms that are gold, not distractors
            row["retrieval_precision"] = round(gold_refs / len(refs), 3) if refs else None
            row["retrieval_precision_at_10"] = row["retrieval_precision"]
            # gold-retrieved recall: distinct gold terms surfaced at least once / |gold|
            row["gold_retrieved"] = (
                round(len(set(en.lower() for en, _ in refs if en.lower() in gold_en)) / len(gold), 3)
                if gold
                else None
            )
            row["prompt_gold_retrieved_at_10"] = row["gold_retrieved"]
            if reference_text is not None:
                row.update(
                    compute_bleu_scores(
                        hypothesis=res["text"],
                        reference=reference_text,
                        target_terms=target_terms,
                        sacrebleu_tokenizer=args.sacrebleu_tokenizer,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            row = {"preset": preset, "error": str(exc)[:200]}
        rows.append(row)
        print(f"[score] {preset}: recall@gold={row.get('term_recall')} "
              f"retr_precision@10={row.get('retrieval_precision_at_10')} "
              f"prompt_gold@10={row.get('prompt_gold_retrieved_at_10')} "
              f"surfaced_recall={row.get('term_recall_surfaced')} "
              f"not_surfaced_recall={row.get('term_recall_not_surfaced')} "
              f"refs/chunk={row.get('refs_per_chunk')} masked_bleu={row.get('masked_terms_bleu')}")

    cols = [
        "preset",
        "gold",
        "term_recall",
        "translation_term_recall",
        "identity_retention_recall",
        "term_recall_surfaced",
        "term_recall_not_surfaced",
        "false_copy",
        "bleu",
        "masked_terms_bleu",
        "masked_terms_hyp_removed",
        "masked_terms_ref_removed",
        "masked_terms_types",
        "prompt_gold_retrieved_at_10",
        "retrieval_precision_at_10",
        "avg_prompt_candidates",
        "avg_candidate_pool",
        "prompt_shortfall_chunks",
        "open_wiki_rescue_chunks",
        "gold_retrieved",
        "retrieval_precision",
        "refs_per_chunk",
        "chunks",
    ]
    print("\n=== glossary-scale terminology eval (fixed gold denominator) ===")
    print(" | ".join(c.ljust(13) for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(13) for c in cols))
    if args.out_json:
        json.dump(rows, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
