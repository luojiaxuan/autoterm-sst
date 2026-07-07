from __future__ import annotations

import unittest

from eval.streaming_sst.eval_auto_glossary_switch import (
    ACL_FIXTURE,
    MEDICINE_FIXTURE,
    TextWindow,
    run_all_scenarios,
)


class AutoGlossarySwitchEvalTests(unittest.TestCase):
    def test_fixture_scenarios_pass_text_topic_regression(self) -> None:
        acl = [TextWindow("nlp", text) for text in ACL_FIXTURE]
        medicine = [TextWindow("medicine", text) for text in MEDICINE_FIXTURE]

        rows = run_all_scenarios(
            acl_windows=acl,
            medicine_windows=medicine,
            scenarios=("acl_only", "medicine_only", "acl_to_medicine", "medicine_to_acl"),
        )

        self.assertTrue(all(row["regression_pass"] for row in rows))
        by_name = {row["scenario"]: row for row in rows}
        self.assertEqual(by_name["acl_only"]["false_medicine_on_acl"], 0)
        self.assertEqual(by_name["acl_to_medicine"]["switch_latency_windows"], 2)
        self.assertEqual(by_name["medicine_to_acl"]["switch_latency_windows"], 2)

    def test_probe_assisted_scenarios_pass(self) -> None:
        acl = [TextWindow("nlp", text) for text in ACL_FIXTURE[:2]]
        medicine = [TextWindow("medicine", text) for text in MEDICINE_FIXTURE[:2]]

        rows = run_all_scenarios(
            acl_windows=acl,
            medicine_windows=medicine,
            scenarios=("acl_to_medicine",),
            with_probe=True,
        )

        self.assertEqual(rows[0]["scenario"], "acl_to_medicine")
        self.assertTrue(rows[0]["regression_pass"])


if __name__ == "__main__":
    unittest.main()
