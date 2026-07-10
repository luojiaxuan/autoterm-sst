from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.term_memory.build_deduped_merged_index import (
    SourceSpec,
    build_deduped_merged_index,
)


def _entry(term: str, zh: str, *, qid: str) -> dict:
    return {
        "term": term,
        "target_translations": {"zh": zh},
        "source": "toy",
        "wikidata_qid": qid,
        "category_path": ["Category:Toy"],
    }


def _write_pair(
    root: Path,
    role: str,
    entries: list[dict],
    embeddings: torch.Tensor,
    *,
    index_terms: list[dict] | None = None,
    checkpoint_sha256: str = "",
) -> SourceSpec:
    glossary = root / f"{role}.json"
    index = root / f"{role}.pt"
    glossary.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    terms = index_terms or [
        {"key": row["term"].casefold(), "term": row["term"], "target_translations": row["target_translations"]}
        for row in entries
    ]
    payload = {"text_embs": embeddings, "term_list": terms}
    if checkpoint_sha256:
        payload["build_metadata"] = {"embedding_checkpoint_sha256": checkpoint_sha256}
    torch.save(payload, index)
    return SourceSpec(role, glossary, index)


class DedupedMergedIndexTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        path = root / "retriever.pt"
        path.write_bytes(b"toy retrieval checkpoint")
        return path

    def test_first_source_wins_nfkc_duplicate_and_tensor_rows_stay_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = self._checkpoint(root)
            first = _write_pair(
                root,
                "first",
                [_entry("Alpha", "甲", qid="Q1"), _entry("Beta", "乙", qid="Q2")],
                torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            )
            second = _write_pair(
                root,
                "second",
                [_entry("Ａｌｐｈａ", "阿尔法", qid="Q3"), _entry("Gamma", "丙", qid="Q4")],
                torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
            )
            out = root / "out"
            report = build_deduped_merged_index(
                sources=[first, second],
                topup_source=None,
                target_size=None,
                out_dir=out,
                preset_id="toy_dedup",
                language_pair="en-zh",
                embedding_checkpoint=checkpoint,
            )

            payload = torch.load(out / "maxsim.pt", map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(
                    payload["text_embs"],
                    torch.tensor([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]),
                )
            )
            self.assertEqual([row["term"] for row in payload["term_list"]], ["Alpha", "Beta", "Gamma"])
            self.assertEqual(report["base_input_rows"], 4)
            self.assertEqual(report["base_unique_terms"], 3)
            self.assertEqual(report["base_duplicate_rows"], 1)
            self.assertEqual(report["base_target_variant_conflict_term_count"], 1)

            audit = json.loads((out / "duplicate_audit.json").read_text(encoding="utf-8"))
            duplicate = audit["duplicate_terms"][0]
            self.assertEqual(duplicate["normalized_source"], "alpha")
            self.assertEqual(duplicate["output_winner"]["source_role"], "first")
            self.assertEqual(
                [row["glossary_entry"]["wikidata_qid"] for row in duplicate["occurrences"]],
                ["Q1", "Q3"],
            )

    def test_topup_reaches_exact_unique_target_without_counting_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = self._checkpoint(root)
            base = _write_pair(
                root,
                "base",
                [_entry("Alpha", "甲", qid="Q1"), _entry("Beta", "乙", qid="Q2")],
                torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            )
            topup = _write_pair(
                root,
                "distractor",
                [
                    _entry("alpha", "甲", qid="Q10"),
                    _entry("Delta", "丁", qid="Q11"),
                    _entry("DELTA", "丁", qid="Q12"),
                    _entry("Epsilon", "戊", qid="Q13"),
                    _entry("Zeta", "己", qid="Q14"),
                ],
                torch.tensor(
                    [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0]]
                ),
            )
            topup = SourceSpec(
                topup.role,
                topup.glossary_path,
                topup.index_path,
                kind="topup",
            )
            out = root / "out"
            report = build_deduped_merged_index(
                sources=[base],
                topup_source=topup,
                target_size=4,
                out_dir=out,
                preset_id="toy_4",
                language_pair="en-zh",
                embedding_checkpoint=checkpoint,
            )

            glossary = json.loads((out / "glossary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["term"] for row in glossary], ["Alpha", "Beta", "Delta", "Epsilon"])
            payload = torch.load(out / "maxsim.pt", map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(
                    payload["text_embs"],
                    torch.tensor([[1.0, 1.0], [2.0, 2.0], [11.0, 11.0], [13.0, 13.0]]),
                )
            )
            self.assertEqual(report["output_term_count"], 4)
            self.assertEqual(report["topup_term_count"], 2)
            topup_report = report["source_roles"][-1]
            self.assertEqual(topup_report["collisions_with_base_rows"], 1)
            self.assertEqual(topup_report["duplicate_rows_within_topup"], 1)
            self.assertEqual(topup_report["eligible_unique_terms_not_selected"], 1)

    def test_rejects_misaligned_index_term_list(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = self._checkpoint(root)
            spec = _write_pair(
                root,
                "bad",
                [_entry("Alpha", "甲", qid="Q1")],
                torch.tensor([[1.0, 2.0]]),
                index_terms=[{"term": "Beta", "target_translations": {"zh": "乙"}}],
            )
            with self.assertRaisesRegex(ValueError, "glossary/index term mismatch"):
                build_deduped_merged_index(
                    sources=[spec],
                    topup_source=None,
                    target_size=None,
                    out_dir=root / "out",
                    preset_id="bad",
                    language_pair="en-zh",
                    embedding_checkpoint=checkpoint,
                )

    def test_rejects_incompatible_embedding_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = self._checkpoint(root)
            first = _write_pair(
                root,
                "first",
                [_entry("Alpha", "甲", qid="Q1")],
                torch.tensor([[1.0, 2.0]]),
            )
            second = _write_pair(
                root,
                "second",
                [_entry("Beta", "乙", qid="Q2")],
                torch.tensor([[1.0, 2.0, 3.0]]),
            )
            with self.assertRaisesRegex(ValueError, "incompatible embedding shape"):
                build_deduped_merged_index(
                    sources=[first, second],
                    topup_source=None,
                    target_size=None,
                    out_dir=root / "out",
                    preset_id="bad_shape",
                    language_pair="en-zh",
                    embedding_checkpoint=checkpoint,
                )

    def test_rejects_embedded_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            checkpoint = self._checkpoint(root)
            spec = _write_pair(
                root,
                "bad_checkpoint",
                [_entry("Alpha", "甲", qid="Q1")],
                torch.tensor([[1.0, 2.0]]),
                checkpoint_sha256="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                build_deduped_merged_index(
                    sources=[spec],
                    topup_source=None,
                    target_size=None,
                    out_dir=root / "out",
                    preset_id="bad_checkpoint",
                    language_pair="en-zh",
                    embedding_checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
