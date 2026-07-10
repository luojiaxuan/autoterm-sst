from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.term_memory.build_realsi_domain_catalog import build_catalog, write_catalog
from scripts.term_memory.collect_wikimedia_domain_glossary import (
    acceptable_title,
    merge_candidates,
    traversable_category,
)


def _row(term: str, translation: str, source: str = "wikidata_p31_p279") -> dict:
    row = {
        "term": term,
        "term_key": term.casefold(),
        "target_translations": {"zh": translation},
        "source": source,
    }
    if source == "wikidata_p31_p279":
        row.update(
            {
                "wikidata_qid": "Q1",
                "domain_root_qid": "Q2",
                "rdf_path": "P31/P279*|P279+",
            }
        )
    elif source == "wikidata_exact_p31":
        row.update(
            {
                "wikidata_qid": "Q5",
                "wikidata_type_qid": "Q6",
                "rdf_path": "P31",
            }
        )
    elif source == "wikipedia_category":
        row.update(
            {
                "wikidata_qid": "Q3",
                "wikipedia_pageid": 3,
                "category_path": ["Category:Education"],
                "category_depth": 0,
            }
        )
    elif source == "wikipedia_deep_category":
        row.update(
            {
                "wikidata_qid": "Q4",
                "wikipedia_pageid": 4,
                "category_path": ["Category:Education"],
                "category_depth": 0,
                "category_query": 'deepcat:"Education"',
            }
        )
    return row


class WikimediaCollectionTests(unittest.TestCase):
    def test_rejects_meta_titles_and_categories(self) -> None:
        self.assertTrue(acceptable_title("Educational psychology"))
        self.assertFalse(acceptable_title("List of education journals"))
        self.assertFalse(acceptable_title("2024 in sports"))
        self.assertTrue(traversable_category("Category:Environmental science"))
        self.assertFalse(traversable_category("Category:Wikipedia articles about law"))

    def test_rdf_rows_precede_category_fallback(self) -> None:
        rdf = [_row("Banking", "银行业")]
        category = [
            _row("Banking", "银行业", "wikipedia_category"),
            _row("Credit risk", "信用风险", "wikipedia_category"),
        ]
        rows = merge_candidates(rdf, category, domain="finance", limit=2)
        self.assertEqual([row["term"] for row in rows], ["Banking", "Credit risk"])
        self.assertEqual(rows[0]["source"], "wikidata_p31_p279")


class RealSIDomainCatalogTests(unittest.TestCase):
    def test_preserves_evaluated_seed_rows_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            seed_rows = [
                {"term": "ACL", "target_translations": {"zh": "ACL"}, "marker": 1},
                {"term": "acl", "target_translations": {"zh": "ACL"}, "marker": 2},
            ]
            seed = root / "seed.json"
            seed.write_text(json.dumps(seed_rows), encoding="utf-8")
            rows, _ = build_catalog(
                domains=("nlp",),
                seeds={"nlp": (seed,)},
                domain_sources={"nlp": ()},
                target_lang="zh",
                limit=2,
            )
            self.assertEqual(rows["nlp"], seed_rows)

    def test_builds_exact_slices_without_keyword_inference(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            nlp_seed = root / "nlp.json"
            nlp_seed.write_text(json.dumps([_row("BERT", "BERT")]), encoding="utf-8")
            medicine_seed = root / "medicine.json"
            medicine_seed.write_text(json.dumps([_row("oncology", "肿瘤学")]), encoding="utf-8")
            education_source = root / "education.json"
            education_source.write_text(
                json.dumps(
                    [
                        _row("pedagogy", "教育学", "wikipedia_category"),
                        _row("curriculum", "课程", "wikipedia_category"),
                        _row("teaching method", "教学法", "wikipedia_category"),
                    ]
                ),
                encoding="utf-8",
            )

            rows, report = build_catalog(
                domains=("nlp", "medicine", "education"),
                seeds={"nlp": (nlp_seed,), "medicine": (medicine_seed,), "education": ()},
                domain_sources={"nlp": (), "medicine": (), "education": (education_source,)},
                target_lang="zh",
                limit=1,
            )
            report = write_catalog(
                rows,
                report,
                out_dir=root / "out",
                target_lang="zh",
                limit=1,
            )

            self.assertEqual({domain: len(items) for domain, items in rows.items()}, {"nlp": 1, "medicine": 1, "education": 1})
            self.assertEqual(report["merged_rows"], 3)
            self.assertFalse(report["domain_inference_from_substrings"])
            self.assertEqual(report["per_domain"]["education"]["source_roles"]["wikipedia_category"], 1)


if __name__ == "__main__":
    unittest.main()
