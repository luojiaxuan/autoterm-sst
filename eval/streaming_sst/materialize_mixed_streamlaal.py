#!/usr/bin/env python3
"""Materialize canonical StreamLAAL inputs from a mixed-session run.

The mixed-session evaluator persists one additive translation fragment per
decoder window.  RASST's release scorer instead consumes one SimulEval row per
talk plus sentence-level audio/reference manifests.  This tool bridges those
formats without changing decoder output: a fragment is assigned to the talk
whose span contains its endpoint, using ``start < cursor <= end`` exactly as
the frozen terminology scorer does.

It can also convert historical ACL/medicine InfiniSST ``instances.log`` files
back into the mixed-run record format, allowing the same MFA-aligned TERM_ACC
denominator to be applied to the no-RAG baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SAMPLE_RATE = 16_000
SCHEMA_VERSION = "mixed_streamlaal_bundle.v1"
ASSIGNMENT_RULE = "block.start_sample < record.cursor_samples <= block.end_sample"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _spans_by_index(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    spans = payload.get("block_spans") or []
    result = {int(span["block_index"]): dict(span) for span in spans}
    blocks = payload.get("blocks") or []
    if len(result) != len(spans) or len(result) != len(blocks):
        raise ValueError("blocks and unique block_spans must have equal lengths")
    return result


def _records_for_span(
    payload: Mapping[str, Any], span: Mapping[str, Any]
) -> list[dict[str, Any]]:
    start = int(span["start_sample"])
    end = int(span["end_sample"])
    return [
        dict(record)
        for record in payload.get("records") or []
        if start < int(record.get("cursor_samples") or 0) <= end
    ]


def _target_translation(entry: Mapping[str, Any], target_lang: str) -> str:
    translations = entry.get("target_translations")
    if isinstance(translations, Mapping):
        value = str(translations.get(target_lang) or "").strip()
        if value:
            return value
    return str(
        entry.get("translation")
        or entry.get("target_translation")
        or entry.get(target_lang)
        or ""
    ).strip()


def merge_mask_glossaries(
    glossary_paths: Sequence[Path], *, target_lang: str
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in glossary_paths:
        data = _read_json(path)
        entries = data.values() if isinstance(data, dict) else data
        if not isinstance(entries, Iterable):
            raise ValueError(f"unsupported glossary format: {path}")
        for raw in entries:
            if not isinstance(raw, Mapping):
                continue
            entry = dict(raw)
            source = str(entry.get("term") or entry.get("source") or "").strip()
            target = _target_translation(entry, target_lang)
            if not target:
                continue
            key = (source.casefold(), target.casefold())
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    if not merged:
        raise ValueError("merged masking glossary is empty")
    return merged


def _select_acl_rows(
    *,
    item_id: str,
    audio_rows: Sequence[Mapping[str, Any]],
    references: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    selected_audio: list[dict[str, Any]] = []
    selected_refs: list[str] = []
    wanted = f"{item_id}.wav"
    for audio, reference in zip(audio_rows, references):
        if Path(str(audio.get("wav") or "")).name != wanted:
            continue
        selected_audio.append(dict(audio))
        selected_refs.append(reference)
    if not selected_audio:
        raise ValueError(f"no ACL reference rows found for {item_id}")
    return selected_audio, selected_refs


def _medicine_files(
    medicine_dir: Path, item_id: str, target_lang: str
) -> tuple[Path, Path, Path]:
    medicine_id = item_id.removeprefix("medicine_")
    return (
        medicine_dir / f"medicine.audio__medicine_{medicine_id}.yaml",
        medicine_dir / f"medicine.ref.{target_lang}__medicine_{medicine_id}.txt",
        medicine_dir / f"medicine.source_text.en__medicine_{medicine_id}.txt",
    )


def materialize_bundle(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.run_json)
    if not isinstance(payload, dict):
        raise ValueError("run JSON must be an object")
    spans = _spans_by_index(payload)

    acl_audio = yaml.safe_load(args.acl_audio_yaml.read_text(encoding="utf-8"))
    acl_refs = args.acl_reference.read_text(encoding="utf-8").splitlines()
    acl_sources = args.acl_source.read_text(encoding="utf-8").splitlines()
    if (
        not isinstance(acl_audio, list)
        or len(acl_audio) != len(acl_refs)
        or len(acl_refs) != len(acl_sources)
    ):
        raise ValueError("ACL audio/source/reference rows must align")

    out_audio: list[dict[str, Any]] = []
    out_refs: list[str] = []
    out_sources: list[str] = []
    instances: list[dict[str, Any]] = []
    block_manifest: list[dict[str, Any]] = []

    for block_index, block in enumerate(payload.get("blocks") or [], start=1):
        span = spans[block_index]
        item_id = str(block.get("original_item_id") or block.get("item_id") or "")
        corpus = str(block.get("corpus") or "")
        if corpus == "acl":
            block_audio, block_refs = _select_acl_rows(
                item_id=item_id,
                audio_rows=acl_audio,
                references=acl_refs,
            )
            _, block_sources = _select_acl_rows(
                item_id=item_id,
                audio_rows=acl_audio,
                references=acl_sources,
            )
        elif corpus == "medicine":
            audio_path, ref_path, source_path = _medicine_files(
                args.medicine_input_dir, item_id, args.target_lang
            )
            block_audio = yaml.safe_load(audio_path.read_text(encoding="utf-8"))
            block_refs = ref_path.read_text(encoding="utf-8").splitlines()
            block_sources = source_path.read_text(encoding="utf-8").splitlines()
            if (
                not isinstance(block_audio, list)
                or len(block_audio) != len(block_refs)
                or len(block_refs) != len(block_sources)
            ):
                raise ValueError(
                    f"medicine audio/source/reference rows do not align: {item_id}"
                )
        else:
            raise ValueError(f"unsupported corpus for block {block_index}: {corpus}")

        wav_names = {Path(str(row.get("wav") or "")).name for row in block_audio}
        if len(wav_names) != 1:
            raise ValueError(f"block {block_index} maps to multiple wav basenames: {wav_names}")
        wav_path = str(block_audio[0].get("wav") or "")
        records = _records_for_span(payload, span)
        prediction_parts: list[str] = []
        delays_ms: list[float] = []
        start_sample = int(span["start_sample"])
        for record in records:
            text = str(record.get("text") or record.get("text_preview") or "")
            delay_ms = (
                int(record.get("cursor_samples") or 0) - start_sample
            ) * 1000.0 / SAMPLE_RATE
            prediction_parts.append(text)
            delays_ms.extend([delay_ms] * len(text))
        prediction = "".join(prediction_parts)
        if not prediction:
            raise ValueError(f"block {block_index} has no emitted prediction")
        source_length_ms = int(span["sample_count"]) * 1000.0 / SAMPLE_RATE
        instances.append(
            {
                "index": block_index - 1,
                "prediction": prediction,
                "delays": delays_ms,
                # note (luojiaxuan): elapsed is populated only to satisfy the
                # SimulEval schema. Computation-aware latency from this bundle
                # is intentionally not reported; BLEU uses mWER resegmentation.
                "elapsed": delays_ms,
                "prediction_length": len(prediction),
                "reference": "".join(block_refs),
                "source": [wav_path],
                "source_length": source_length_ms,
            }
        )
        out_audio.extend(dict(row) for row in block_audio)
        out_refs.extend(block_refs)
        out_sources.extend(block_sources)
        block_manifest.append(
            {
                "block_index": block_index,
                "item_id": item_id,
                "corpus": corpus,
                "wav": Path(wav_path).name,
                "reference_segments": len(block_refs),
                "prediction_chars": len(prediction),
                "decoder_records": len(records),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    instances_path = args.out_dir / "instances.log"
    audio_path = args.out_dir / "audio.yaml"
    audio_json_path = args.out_dir / "audio.json"
    reference_path = args.out_dir / "ref.txt"
    source_path = args.out_dir / "source.txt"
    glossary_path = args.out_dir / "raw_mask_glossary.json"
    _write_jsonl(instances_path, instances)
    audio_path.write_text(
        yaml.safe_dump(out_audio, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(audio_json_path, out_audio)
    reference_path.write_text("\n".join(out_refs) + "\n", encoding="utf-8")
    source_path.write_text("\n".join(out_sources) + "\n", encoding="utf-8")
    _write_json(
        glossary_path,
        merge_mask_glossaries(args.mask_glossary, target_lang=args.target_lang),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_json": str(args.run_json),
        "run_sha256": _sha256(args.run_json),
        "assignment_rule": ASSIGNMENT_RULE,
        "sample_rate": SAMPLE_RATE,
        "blocks": block_manifest,
        "instances": len(instances),
        "reference_segments": len(out_refs),
        "mask_glossary_entries": len(_read_json(glossary_path)),
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                instances_path,
                audio_path,
                audio_json_path,
                source_path,
                reference_path,
                glossary_path,
            )
        },
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    return manifest


def _instance_by_wav(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            source = row.get("source")
            if not isinstance(source, list) or not source:
                raise ValueError(f"instance lacks source[0]: {path}")
            wav = Path(str(source[0])).name
            if wav in result:
                raise ValueError(f"duplicate historical instance for wav={wav}")
            result[wav] = row
    return result


def _group_prediction_by_delay(row: Mapping[str, Any]) -> list[tuple[float, str]]:
    prediction = str(row.get("prediction") or "")
    delays = row.get("delays")
    if not isinstance(delays, list) or len(delays) != len(prediction):
        raise ValueError(
            f"prediction/delay mismatch: chars={len(prediction)} delays="
            f"{len(delays) if isinstance(delays, list) else 'missing'}"
        )
    groups: list[tuple[float, str]] = []
    for char, raw_delay in zip(prediction, delays):
        delay = float(raw_delay)
        if groups and groups[-1][0] == delay:
            previous_delay, text = groups[-1]
            groups[-1] = (previous_delay, text + char)
        else:
            groups.append((delay, char))
    return groups


def materialize_norag_payload(args: argparse.Namespace) -> dict[str, Any]:
    template = _read_json(args.template_run_json)
    if not isinstance(template, dict):
        raise ValueError("template run JSON must be an object")
    spans = _spans_by_index(template)
    historical = _instance_by_wav(args.norag_instances)
    records: list[dict[str, Any]] = []

    for block_index, block in enumerate(template.get("blocks") or [], start=1):
        span = spans[block_index]
        wav_paths = list(block.get("wav_paths") or [])
        if not wav_paths:
            raise ValueError(f"template block {block_index} has no wav_paths")
        if str(block.get("corpus") or "") == "acl":
            wav = f"{block.get('item_id')}.wav"
        else:
            wav = Path(str(wav_paths[0])).name
        row = historical.get(wav)
        if row is None:
            raise ValueError(f"historical no-RAG output missing wav={wav}")
        start = int(span["start_sample"])
        end = int(span["end_sample"])
        for delay_ms, text in _group_prediction_by_delay(row):
            cursor = start + int(round(delay_ms * SAMPLE_RATE / 1000.0))
            cursor = min(max(cursor, start + 1), end)
            records.append(
                {
                    "cursor_samples": cursor,
                    "start_sample": max(start, cursor - 1),
                    "expected_domain": str(block.get("expected_domain") or ""),
                    "translation_status": "partial",
                    "text": text,
                    "text_preview": text,
                    "prompt_reference_count": 0,
                    "references": [],
                }
            )

    records.sort(key=lambda record: int(record["cursor_samples"]))
    payload = {
        "config": {
            "role": "infinisst_no_rag",
            "source": [str(path) for path in args.norag_instances],
            "conversion": SCHEMA_VERSION,
            "assignment_rule": ASSIGNMENT_RULE,
        },
        "blocks": template["blocks"],
        "block_spans": template["block_spans"],
        "records": records,
        "summary": {
            "records": len(records),
            "prompt_reference_count": 0,
            "refs_per_chunk": 0.0,
        },
    }
    _write_json(args.out_json, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export StreamLAAL/mWER inputs")
    export.add_argument("--run-json", type=Path, required=True)
    export.add_argument("--acl-audio-yaml", type=Path, required=True)
    export.add_argument("--acl-source", type=Path, required=True)
    export.add_argument("--acl-reference", type=Path, required=True)
    export.add_argument("--medicine-input-dir", type=Path, required=True)
    export.add_argument("--mask-glossary", type=Path, action="append", required=True)
    export.add_argument("--target-lang", default="zh")
    export.add_argument("--out-dir", type=Path, required=True)

    norag = subparsers.add_parser(
        "import-norag", help="convert historical SimulEval output for MFA TERM_ACC"
    )
    norag.add_argument("--template-run-json", type=Path, required=True)
    norag.add_argument("--norag-instances", type=Path, action="append", required=True)
    norag.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        manifest = materialize_bundle(args)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        payload = materialize_norag_payload(args)
        print(f"wrote {args.out_json}: {len(payload['records'])} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
