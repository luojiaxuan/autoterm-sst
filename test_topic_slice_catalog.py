from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.term_memory.build_topic_slice_catalog import (
    _iter_json_array,
    build_catalog,
    load_taxonomy,
    validate_catalog,
)


def _category_row(
    term: str,
    qid: str,
    categories: list[str],
    *,
    zh: str,
    de: str = "",
) -> dict:
    translations = {"zh": zh}
    if de:
        translations["de"] = de
    return {
        "term": term,
        "term_key": term.casefold(),
        "target_translations": translations,
        "source": "wikipedia_deep_category",
        "wikidata_qid": qid,
        "wikipedia_pageid": int(qid[1:]),
        "category_path": categories,
        "category_query": f'deepcat:"{categories[-1].removeprefix("Category:")}"',
    }


def _p31_row(term: str, qid: str, type_qid: str, *, zh: str, de: str) -> dict:
    return {
        "term": term,
        "target_translations": {"zh": zh, "de": de},
        "source": "wikidata_exact_p31",
        "wikidata_qid": qid,
        "wikidata_type_qid": type_qid,
        "rdf_path": "P31",
    }


def _write_taxonomy(path: Path, *, duplicate_description: bool = False) -> None:
    medicine_description = "Banking instruments and financial institutions." if duplicate_description else "Cancer diagnosis and oncology treatment."
    path.write_text(
        json.dumps(
            {
                "taxonomy_id": "toy-topics-v1",
                "expected_topic_count": 2,
                "slice_capacity": 3,
                "topics": [
                    {
                        "domain_id": "finance_banking",
                        "preset_id": "topic_finance_banking",
                        "domain_description": "Banking instruments and financial institutions.",
                        "priority": 20,
                        "primary_match": [{"category_any": ["Category:Banking"]}],
                        "filler_match": [{"category_any": ["Category:Financial services"]}],
                    },
                    {
                        "domain_id": "medicine_oncology",
                        "preset_id": "topic_medicine_oncology",
                        "domain_description": medicine_description,
                        "capacity": 1,
                        "priority": 10,
                        "primary_match": [
                            {"category_any": ["Category:Oncology"]},
                            {"wikidata_type_qid_any": ["Q12136"]},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class TopicSliceCatalogTests(unittest.TestCase):
    def test_streams_json_array_across_small_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "rows.json"
            rows = [{"term": "alpha", "n": 1}, {"term": "长术语", "n": 2}]
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(_iter_json_array(path, chunk_size=7)), rows)

    def test_builds_deterministic_unique_slices_and_reports_filler(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            taxonomy_path = root / "taxonomy.json"
            _write_taxonomy(taxonomy_path)
            input_path = root / "candidates.json"
            rows = [
                _category_row("Bank account", "Q1", ["Category:Finance", "Category:Banking"], zh="银行账户", de="Bankkonto"),
                _category_row(
                    "Clinical bank",
                    "Q2",
                    ["Category:Finance", "Category:Banking", "Category:Oncology"],
                    zh="临床银行",
                    de="Klinische Bank",
                ),
                _p31_row("Oncology", "Q3", "Q12136", zh="肿瘤学", de="Onkologie"),
                _category_row(
                    "Payment network",
                    "Q4",
                    ["Category:Finance", "Category:Financial services"],
                    zh="支付网络",
                    de="Zahlungsnetz",
                ),
                _category_row("Unassigned concept", "Q5", ["Category:Unknown"], zh="未分配概念", de="Nicht zugeordnet"),
                _category_row("Bank account", "Q999", ["Category:Banking"], zh="银行账户", de="Bankkonto"),
                _category_row("Missing German", "Q6", ["Category:Banking"], zh="缺少德语"),
                {
                    "term": "No QID",
                    "target_translations": {"zh": "无QID", "de": "Ohne QID"},
                    "source": "wikidata",
                    "category_path": ["Category:Banking"],
                },
            ]
            input_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            taxonomy = load_taxonomy(taxonomy_path)

            outputs = []
            for name in ("out-a", "out-b"):
                out = root / name
                report = build_catalog(
                    taxonomy=taxonomy,
                    inputs=[("toy", input_path)],
                    out_dir=out,
                    snapshot_id="toy-catalog-v1",
                    target_languages=("zh", "de"),
                    require_full_slices=True,
                    emit_merged=True,
                )
                outputs.append(out)
                self.assertEqual(report["stats"]["selected_primary"], 3)
                self.assertEqual(report["stats"]["selected_filler"], 1)
                self.assertEqual(report["stats"]["primary_overlap_rows"], 1)
                self.assertEqual(report["stats"]["rejected_missing_translation"], 1)
                self.assertEqual(report["stats"]["rejected_missing_structured_provenance"], 1)
                self.assertEqual(report["stats"]["source_term_qid_collisions"], 1)
                self.assertEqual(report["stats"]["unassigned_unique_source_terms"], 1)
                self.assertEqual(report["per_topic"]["finance_banking"]["filler_term_count"], 1)
                self.assertEqual(
                    report["per_topic"]["finance_banking"]["provenance_by_role"]["filler"],
                    {"wikipedia_deep_category": 1},
                )
                validation = validate_catalog(
                    out / "catalog_manifest.json", expected_topic_count=2, require_full_slices=True
                )
                self.assertEqual(validation["global_term_count"], 4)

            self.assertEqual(
                (outputs[0] / "catalog_manifest.json").read_bytes(),
                (outputs[1] / "catalog_manifest.json").read_bytes(),
            )
            for relative in (
                "slices/topic_finance_banking.json",
                "slices/topic_medicine_oncology.json",
                "merged_topics.json",
            ):
                self.assertEqual((outputs[0] / relative).read_bytes(), (outputs[1] / relative).read_bytes())

    def test_rejects_duplicate_router_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "taxonomy.json"
            _write_taxonomy(path, duplicate_description=True)
            with self.assertRaisesRegex(ValueError, "domain_description"):
                load_taxonomy(path)

    def test_allows_and_validates_empty_underfilled_slice(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            taxonomy_path = root / "taxonomy.json"
            _write_taxonomy(taxonomy_path)
            input_path = root / "candidates.json"
            input_path.write_text(
                json.dumps(
                    [
                        _category_row(
                            "Bank account",
                            "Q1",
                            ["Category:Finance", "Category:Banking"],
                            zh="银行账户",
                            de="Bankkonto",
                        )
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            build_catalog(
                taxonomy=load_taxonomy(taxonomy_path),
                inputs=[("toy", input_path)],
                out_dir=out,
                snapshot_id="underfilled-v1",
                target_languages=("zh", "de"),
                require_full_slices=False,
                emit_merged=True,
            )
            result = validate_catalog(
                out / "catalog_manifest.json", expected_topic_count=2, require_full_slices=False
            )
            self.assertEqual(result["global_term_count"], 1)
            self.assertEqual(result["per_topic"]["medicine_oncology"]["term_count"], 0)


if __name__ == "__main__":
    unittest.main()
