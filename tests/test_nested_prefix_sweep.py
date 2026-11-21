from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from framework.agents.term_memory.manifest import TermMemoryManifest
from scripts.term_memory.build_nested_prefix_sweep import (
    build_nested_prefix_sweep,
    derived_preset_id,
    file_sha256,
)


def _entry(term: str, translation: str) -> dict:
    return {
        "term": term,
        "target_translations": {"zh": translation},
        "source": "toy",
    }


def _write_source(
    root: Path,
    preset_id: str,
    entries: list[dict],
    embeddings: torch.Tensor,
    *,
    index_terms: list[dict] | None = None,
) -> tuple[Path, Path]:
    glossary_path = root / preset_id / "glossary.json"
    index_path = root / preset_id / "en-zh" / "maxsim.pt"
    glossary_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    glossary_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    terms = index_terms or [
        {
            "key": row["term"].casefold(),
            "term": row["term"],
            "target_translations": row["target_translations"],
        }
        for row in entries
    ]
    torch.save(
        {
            "text_embs": embeddings,
            "term_list": terms,
            "build_metadata": {
                "embedding_checkpoint_path": "/models/toy.pt",
                "embedding_checkpoint_sha256": "a" * 64,
            },
        },
        index_path,
    )
    return glossary_path, index_path


def _write_manifest(
    root: Path,
    sources: dict[str, tuple[Path, Path]],
    *,
    count: int,
) -> Path:
    scales = {}
    preset_meta = {}
    for domain, (glossary_path, index_path) in sources.items():
        preset_id = f"{domain}_core_10k"
        scales[preset_id] = {
            "en-zh": {
                "terms_path": str(glossary_path),
                "indexes": {"maxsim": str(index_path)},
                "num_terms": count,
            }
        }
        preset_meta[preset_id] = {
            "label": f"{domain.title()} glossary",
            "domain": domain,
            "domain_id": domain,
            "description": f"Stable {domain} routing description.",
            "enabled_for_auto_router": True,
        }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": "toy_10domain_10k",
                "source": "toy",
                "scales": scales,
                "preset_meta": preset_meta,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


class NestedPrefixSweepTests(unittest.TestCase):
    def test_aligned_nested_outputs_manifest_hashes_and_gold_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            nlp_entries = [
                _entry("Alpha", "甲"),
                _entry("Beta", "乙"),
                _entry("Gamma", "丙"),
                _entry("Delta", "丁"),
                _entry("Epsilon", "戊"),
            ]
            medicine_entries = [
                _entry("Heart", "心脏"),
                _entry("Lung", "肺"),
                _entry("Drug", "药物"),
                _entry("Clinic", "诊所"),
                _entry("Scan", "扫描"),
            ]
            nlp_embeddings = torch.arange(15, dtype=torch.float32).reshape(5, 3)
            medicine_embeddings = torch.arange(15, 30, dtype=torch.float32).reshape(5, 3)
            nlp_paths = _write_source(
                root / "inputs", "nlp_core_10k", nlp_entries, nlp_embeddings
            )
            medicine_paths = _write_source(
                root / "inputs",
                "medicine_core_10k",
                medicine_entries,
                medicine_embeddings,
            )
            manifest_path = _write_manifest(
                root,
                {"nlp": nlp_paths, "medicine": medicine_paths},
                count=5,
            )
            nlp_gold = root / "nlp_gold.json"
            nlp_gold.write_text(
                json.dumps(
                    [
                        {"term": "Alpha", "target_translations": {"zh": "甲"}},
                        {"term": "Gamma", "target_translations": {"zh": "丙"}},
                    ]
                ),
                encoding="utf-8",
            )
            medicine_gold = root / "medicine_gold.json"
            medicine_gold.write_text(
                json.dumps(
                    {
                        "heart": {"en": "Heart", "zh": "心脏"},
                        "lung": {"term": "Lung", "target_translations": {"zh": "肺"}},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"

            report = build_nested_prefix_sweep(
                manifest_path=manifest_path,
                out_root=out,
                domains=["nlp", "medicine"],
                sizes=[4, 2, 3],
                expected_source_size=5,
                gold_paths={"nlp": [nlp_gold], "medicine": [medicine_gold]},
            )

            self.assertEqual(report["prefix_sizes"], [2, 3, 4])
            self.assertEqual(len(report["outputs"]), 6)
            nlp_2_index = out / "nlp_core_2" / "en-zh" / "maxsim.pt"
            nlp_3_index = out / "nlp_core_3" / "en-zh" / "maxsim.pt"
            nlp_2_glossary = out / "nlp_core_2" / "glossary.json"
            payload_2 = torch.load(
                nlp_2_index, map_location="cpu", weights_only=True
            )
            payload_3 = torch.load(
                nlp_3_index, map_location="cpu", weights_only=True
            )
            self.assertTrue(torch.equal(payload_2["text_embs"], nlp_embeddings[:2]))
            self.assertTrue(torch.equal(payload_3["text_embs"][:2], payload_2["text_embs"]))
            self.assertEqual(
                [row["term"] for row in payload_2["term_list"]],
                ["Alpha", "Beta"],
            )
            self.assertEqual(
                [row["term"] for row in json.loads(nlp_2_glossary.read_text())],
                ["Alpha", "Beta"],
            )
            metadata = payload_2["build_metadata"]
            self.assertEqual(metadata["parent_preset_id"], "nlp_core_10k")
            self.assertEqual(metadata["parent_index_sha256"], file_sha256(nlp_paths[1]))
            self.assertEqual(metadata["embedding_checkpoint_sha256"], "a" * 64)

            audit = json.loads((out / "gold_prefix_coverage.json").read_text())
            nlp_audit = audit["domains"]["nlp"]
            self.assertEqual(nlp_audit["gold_unique_term_count"], 2)
            self.assertEqual(nlp_audit["minimum_prefix_size_for_all_source_gold"], 3)
            self.assertEqual(
                nlp_audit["prefixes"]["2"]["coverage_of_all_gold"], 0.5
            )
            self.assertTrue(nlp_audit["prefixes"]["3"]["full_gold_coverage"])
            self.assertTrue(
                audit["domains"]["medicine"]["prefixes"]["2"][
                    "full_gold_coverage"
                ]
            )

            fragment_path = out / "manifest_fragment.json"
            fragment = json.loads(fragment_path.read_text())
            self.assertEqual(
                fragment["scales"]["nlp_core_2"]["en-zh"]["num_terms"], 2
            )
            self.assertEqual(
                fragment["preset_meta"]["nlp_core_2"]["description"],
                "Stable nlp routing description.",
            )
            loaded_fragment = TermMemoryManifest.load(str(fragment_path))
            snapshot = loaded_fragment.snapshot_for("nlp_core_2", "zh")
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.num_terms, 2)
            output_row = next(
                row for row in report["outputs"] if row["preset_id"] == "nlp_core_2"
            )
            self.assertEqual(
                report["manifest_fragment_sha256"], file_sha256(fragment_path)
            )
            self.assertEqual(
                report["gold_prefix_coverage_sha256"],
                file_sha256(out / "gold_prefix_coverage.json"),
            )
            self.assertEqual(output_row["index_sha256"], file_sha256(nlp_2_index))
            self.assertEqual(
                output_row["glossary_sha256"], file_sha256(nlp_2_glossary)
            )

    def test_rejects_glossary_index_term_misalignment_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            entries = [_entry("Alpha", "甲"), _entry("Beta", "乙")]
            bad_terms = [
                {"term": "Alpha", "target_translations": {"zh": "甲"}},
                {"term": "Gamma", "target_translations": {"zh": "丙"}},
            ]
            paths = _write_source(
                root / "inputs",
                "nlp_core_10k",
                entries,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                index_terms=bad_terms,
            )
            manifest_path = _write_manifest(root, {"nlp": paths}, count=2)
            out = root / "out"

            with self.assertRaisesRegex(ValueError, "glossary/index term mismatch"):
                build_nested_prefix_sweep(
                    manifest_path=manifest_path,
                    out_root=out,
                    domains=["nlp"],
                    sizes=[1],
                    expected_source_size=2,
                )
            self.assertFalse(out.exists())

    def test_optional_full_gold_requirement_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            entries = [
                _entry("Alpha", "甲"),
                _entry("Beta", "乙"),
                _entry("Gamma", "丙"),
            ]
            paths = _write_source(
                root / "inputs",
                "nlp_core_10k",
                entries,
                torch.tensor([[1.0], [2.0], [3.0]]),
            )
            manifest_path = _write_manifest(root, {"nlp": paths}, count=3)
            gold = root / "gold.json"
            gold.write_text(json.dumps([{"term": "Gamma"}]), encoding="utf-8")
            out = root / "out"

            with self.assertRaisesRegex(ValueError, "nlp@2: 0/1"):
                build_nested_prefix_sweep(
                    manifest_path=manifest_path,
                    out_root=out,
                    domains=["nlp"],
                    sizes=[2],
                    expected_source_size=3,
                    gold_paths={"nlp": [gold]},
                    require_full_gold_prefix_coverage=True,
                )
            self.assertFalse(out.exists())

    def test_fractional_k_preset_name_is_filesystem_safe(self) -> None:
        self.assertEqual(derived_preset_id("nlp_core_10k", 1000), "nlp_core_1k")
        self.assertEqual(derived_preset_id("nlp_core_10k", 2500), "nlp_core_2p5k")
        self.assertEqual(derived_preset_id("nlp_core_10k", 5000), "nlp_core_5k")


if __name__ == "__main__":
    unittest.main()
