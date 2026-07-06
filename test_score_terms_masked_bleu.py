from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest

from eval.streaming_sst.score_terms import (
    compile_term_mask_patterns,
    compute_bleu_scores,
    mask_target_terms,
    resolve_chunk_samples,
    target_terms_from_gold,
)


class ScoreTermsMaskedBleuTests(unittest.TestCase):
    def test_default_streaming_chunk_follows_latency_multiplier(self) -> None:
        self.assertEqual(resolve_chunk_samples(0, 0.96, 1), 15360)
        self.assertEqual(resolve_chunk_samples(0, 0.96, 2), 30720)
        self.assertEqual(resolve_chunk_samples(8000, 0.96, 2), 8000)
        self.assertEqual(resolve_chunk_samples(0, 0.96, 9), 61440)
        self.assertEqual(resolve_chunk_samples(0, 0.96, -2), 15360)
        self.assertEqual(resolve_chunk_samples(0, 0.96, "bad"), 30720)

    def test_longer_terms_are_masked_before_overlaps(self) -> None:
        terms = target_terms_from_gold(
            [
                ("language model", ["语言模型"]),
                ("model", ["模型"]),
            ]
        )
        masked, removed = mask_target_terms(
            "这个语言模型比旧模型更好。",
            compile_term_mask_patterns(terms),
        )
        self.assertEqual(removed, 2)
        self.assertNotIn("语言模型", masked)
        self.assertNotIn("旧模型", masked)

    def test_single_cjk_term_does_not_match_inside_cjk_word(self) -> None:
        masked, removed = mask_target_terms(
            "语言 语",
            compile_term_mask_patterns(["语"]),
        )
        self.assertEqual(removed, 1)
        self.assertEqual(masked, "语言")

    def test_alnum_terms_use_word_boundaries(self) -> None:
        masked, removed = mask_target_terms(
            "The App in application is an app.",
            compile_term_mask_patterns(["app"]),
        )
        self.assertEqual(removed, 2)
        self.assertIn("application", masked)
        self.assertNotIn(" App ", f" {masked} ")

    def test_compute_bleu_scores_reports_mask_counts(self) -> None:
        old_sacrebleu = sys.modules.get("sacrebleu")

        def fake_corpus_bleu(hypotheses, references, tokenize):
            del references, tokenize
            return SimpleNamespace(score=float(len(hypotheses[0])))

        sys.modules["sacrebleu"] = SimpleNamespace(corpus_bleu=fake_corpus_bleu)
        try:
            scores = compute_bleu_scores(
                hypothesis="神经网络 提升性能。",
                reference="神经网络 可以提升性能。",
                target_terms=["神经网络"],
                sacrebleu_tokenizer="zh",
            )
        finally:
            if old_sacrebleu is None:
                del sys.modules["sacrebleu"]
            else:
                sys.modules["sacrebleu"] = old_sacrebleu

        self.assertEqual(scores["masked_terms_hyp_removed"], 1)
        self.assertEqual(scores["masked_terms_ref_removed"], 1)
        self.assertEqual(scores["masked_terms_types"], 1)
        self.assertIn("masked_terms_bleu", scores)


if __name__ == "__main__":
    unittest.main()
