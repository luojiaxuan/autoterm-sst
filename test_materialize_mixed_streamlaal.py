from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.streaming_sst.materialize_mixed_streamlaal import (
    materialize_bundle,
    materialize_norag_payload,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_materialize_bundle_preserves_endpoint_assignment(tmp_path: Path) -> None:
    run = {
        "blocks": [
            {"item_id": "acl-a", "corpus": "acl", "wav_paths": ["acl-a.wav"]},
            {
                "item_id": "medicine_7",
                "corpus": "medicine",
                "wav_paths": ["7_v2.wav"],
            },
        ],
        "block_spans": [
            {"block_index": 1, "start_sample": 0, "end_sample": 100, "sample_count": 100},
            {"block_index": 2, "start_sample": 100, "end_sample": 220, "sample_count": 120},
        ],
        "records": [
            {"cursor_samples": 100, "text": "甲"},
            {"cursor_samples": 120, "text": "乙"},
        ],
    }
    run_path = _write_json(tmp_path / "run.json", run)
    acl_audio_path = tmp_path / "acl.yaml"
    acl_audio_path.write_text(
        yaml.safe_dump([{"wav": "acl-a.wav", "offset": 0, "duration": 1}]),
        encoding="utf-8",
    )
    acl_ref_path = tmp_path / "acl.ref"
    acl_ref_path.write_text("参考一\n", encoding="utf-8")
    acl_source_path = tmp_path / "acl.source"
    acl_source_path.write_text("source one\n", encoding="utf-8")
    med_dir = tmp_path / "med"
    med_dir.mkdir()
    (med_dir / "medicine.audio__medicine_7.yaml").write_text(
        yaml.safe_dump([{"wav": "7_v2.wav", "offset": 0, "duration": 1}]),
        encoding="utf-8",
    )
    (med_dir / "medicine.ref.zh__medicine_7.txt").write_text(
        "参考二\n", encoding="utf-8"
    )
    (med_dir / "medicine.source_text.en__medicine_7.txt").write_text(
        "source two\n", encoding="utf-8"
    )
    g1 = _write_json(tmp_path / "g1.json", [{"term": "x", "translation": "甲"}])
    g2 = _write_json(tmp_path / "g2.json", [{"term": "y", "translation": "乙"}])
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        run_json=run_path,
        acl_audio_yaml=acl_audio_path,
        acl_source=acl_source_path,
        acl_reference=acl_ref_path,
        medicine_input_dir=med_dir,
        mask_glossary=[g1, g2],
        target_lang="zh",
        out_dir=out_dir,
    )

    manifest = materialize_bundle(args)
    instances = [json.loads(line) for line in (out_dir / "instances.log").read_text().splitlines()]
    assert [row["prediction"] for row in instances] == ["甲", "乙"]
    assert instances[0]["delays"] == [6.25]
    assert instances[1]["delays"] == [1.25]
    assert manifest["reference_segments"] == 2
    assert manifest["mask_glossary_entries"] == 2
    assert (out_dir / "source.txt").read_text().splitlines() == [
        "source one",
        "source two",
    ]


def test_import_norag_reconstructs_mixed_records(tmp_path: Path) -> None:
    template = {
        "blocks": [
            {
                "item_id": "acl-a",
                "corpus": "acl",
                "expected_domain": "nlp",
                "wav_paths": ["acl-a.wav"],
            },
            {
                "item_id": "medicine_7",
                "corpus": "medicine",
                "expected_domain": "medicine",
                "wav_paths": ["7_v2.wav"],
            },
        ],
        "block_spans": [
            {"block_index": 1, "start_sample": 0, "end_sample": 16000, "sample_count": 16000},
            {"block_index": 2, "start_sample": 16000, "end_sample": 32000, "sample_count": 16000},
        ],
    }
    template_path = _write_json(tmp_path / "template.json", template)
    acl = tmp_path / "acl.log"
    acl.write_text(
        json.dumps(
            {"prediction": "甲乙", "delays": [100.0, 100.0], "source": ["acl-a.wav"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    med = tmp_path / "med.log"
    med.write_text(
        json.dumps(
            {"prediction": "丙", "delays": [200.0], "source": ["7_v2.wav"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "norag.json"
    args = argparse.Namespace(
        template_run_json=template_path,
        norag_instances=[acl, med],
        out_json=out,
    )

    payload = materialize_norag_payload(args)
    assert [row["text"] for row in payload["records"]] == ["甲乙", "丙"]
    assert [row["cursor_samples"] for row in payload["records"]] == [1600, 19200]
    assert all(row["prompt_reference_count"] == 0 for row in payload["records"])
