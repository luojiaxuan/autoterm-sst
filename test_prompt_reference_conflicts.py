from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

from framework.agents.omni import OmniAgent
from framework.agents.plugins.prompt import PromptBuilder, merge_references


class PromptReferenceConflictTests(unittest.TestCase):
    def test_merge_drops_exact_pairs_but_preserves_translation_conflicts(self) -> None:
        references = [
            {"term": "bank", "translation": "银行", "source": "finance"},
            {"term": "BANK", "translation": "银行", "source": "duplicate"},
            {"term": "bank", "translation": "河岸", "source": "environment"},
        ]

        merged = merge_references(references)

        self.assertEqual(len(merged), 2)
        self.assertEqual({item["translation"] for item in merged}, {"银行", "河岸"})

    def test_fixed_merged_prompt_records_the_actual_pair_deduped_rows(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=False,
            glossary_preset="merged_raw_100k",
            active_glossary_preset="merged_raw_100k",
            active_domain="merged",
            recent_references=deque(maxlen=16),
            last_prompt_reference_count=0,
        )
        references = [
            {"term": "bank", "translation": "银行", "score": 0.90},
            {"term": "BANK", "translation": "银行", "score": 0.80},
            {"term": "bank", "translation": "河岸", "score": 0.70},
        ]

        prompt_refs = agent._prompt_references(session, references)
        term_map = PromptBuilder(term_map_format="tagged").term_map([], prompt_refs)

        self.assertEqual(len(prompt_refs), 2)
        self.assertEqual(session.last_prompt_reference_count, 2)
        self.assertEqual(term_map.count("[TERM]"), 2)
        self.assertIn("bank => 银行", term_map)
        self.assertIn("bank => 河岸", term_map)


if __name__ == "__main__":
    unittest.main()
