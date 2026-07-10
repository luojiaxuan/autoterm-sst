from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.term_memory.collect_wikimedia_domain_glossary import (
    CollectionUnit,
    legacy_domain_units,
    load_topic_specs,
    preflight_topic_capacity,
    validate_topic_catalog,
)


SPEC_PATH = Path(__file__).parent / "configs" / "autoterm_100_topics_v1.json"


class FrozenTopicCatalogTests(unittest.TestCase):
    def test_frozen_catalog_is_exactly_ten_by_ten(self) -> None:
        payload, units = load_topic_specs(SPEC_PATH)

        self.assertEqual(payload["target_topics"], 100)
        self.assertEqual(len(units), 100)
        self.assertEqual(len({unit.unit_id for unit in units}), 100)
        self.assertEqual(
            Counter(unit.macro_domain for unit in units),
            Counter({macro_domain: 10 for macro_domain in payload["macro_domains"]}),
        )
        for unit in units:
            self.assertTrue(unit.seed_terms)
            self.assertTrue(unit.wikidata_root_qids)
            self.assertTrue(unit.wikipedia_query_categories)
            self.assertTrue(unit.wikipedia_reserve_categories)
            self.assertTrue(unit.router_label)
            self.assertTrue(unit.router_description)

    def test_duplicate_topic_id_is_rejected(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["topics"][1]["topic_id"] = payload["topics"][0]["topic_id"]

        with self.assertRaisesRegex(ValueError, "duplicate topic_id"):
            validate_topic_catalog(payload)

    def test_missing_required_field_is_rejected(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        del payload["topics"][0]["wikipedia_reserve_category_roots"]

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_topic_catalog(payload)

    def test_macro_domain_must_keep_ten_topics(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["topics"][0]["macro_domain"] = "medicine"
        payload["topics"][0]["topic_id"] = "medicine_natural_language_processing"

        with self.assertRaisesRegex(ValueError, "exactly 10 topics"):
            validate_topic_catalog(payload)

    def test_candidate_policy_freezes_ten_thousand_rows_per_topic(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["candidate_policy"]["target_rows_per_topic"] = 9_999

        with self.assertRaisesRegex(ValueError, "exactly 10000"):
            validate_topic_catalog(payload)

    def test_legacy_domain_mapping_keeps_old_roots(self) -> None:
        units = legacy_domain_units(("finance", "science"))

        self.assertEqual([unit.unit_id for unit in units], ["finance", "science"])
        self.assertEqual(units[0].wikidata_root_qids, ("Q43015",))
        self.assertIn("Banking", units[0].wikipedia_query_categories)
        self.assertTrue(units[1].skip_rdf_root)


class _PreflightClient:
    def __init__(self, hits: dict[str, int]) -> None:
        self.hits = hits

    def get_json(self, _endpoint: str, params: dict[str, object]) -> dict[str, object]:
        query = str(params["srsearch"])
        category = query.removeprefix('deepcat:"').removesuffix('"')
        return {"query": {"searchinfo": {"totalhits": self.hits.get(category, 0)}}}


def _unit(unit_id: str, primary: str, reserve: str) -> CollectionUnit:
    return CollectionUnit(
        unit_id=unit_id,
        macro_domain="nlp",
        router_label=unit_id,
        router_description="test router description",
        seed_terms=("one", "two", "three"),
        wikidata_root_qids=("Q1",),
        wikidata_exact_p31_qids=(),
        wikipedia_root_categories=(primary, reserve),
        wikipedia_query_categories=(primary,),
        wikipedia_reserve_categories=(reserve,),
    )


class TopicCapacityPreflightTests(unittest.TestCase):
    def test_preflight_reports_raw_upper_bound_and_shortfall(self) -> None:
        units = (
            _unit("nlp_large", "Large", "Large reserve"),
            _unit("nlp_small", "Small", "Small reserve"),
        )
        client = _PreflightClient(
            {"Large": 12_000, "Large reserve": 7_000, "Small": 3_000, "Small reserve": 2_000}
        )

        report = preflight_topic_capacity(client, units, minimum_raw_candidates=15_000)

        self.assertEqual(report["failed_topics"], ["nlp_small"])
        self.assertEqual(report["topics"]["nlp_large"]["raw_hit_upper_bound"], 19_000)
        self.assertFalse(report["topics"]["nlp_small"]["passes_cheap_preflight"])
        self.assertIn("upper bound", report["interpretation"])


if __name__ == "__main__":
    unittest.main()
