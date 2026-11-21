from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.streaming_sst.replay_domain_router_sweep import (
    choose_pareto,
    evaluate_top_k,
    load_dataset,
    load_router_specs,
    replay_hard_top1,
    run_sweep,
    score_margins,
)


def _records():
    raw = [
        ("a", {"a": 0.9, "b": 0.1}),
        ("a", {"a": 0.4, "b": 0.6}),
        ("b", {"a": 0.1, "b": 0.9}),
        ("b", {"a": 0.7, "b": 0.3}),
    ]
    rows = []
    for index, (gold, scores) in enumerate(raw):
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            {
                "index": index,
                "gold": gold,
                "scores": dict(ranked),
                "ranked": ranked,
                "text": "",
                "start_segment": index,
                "end_segment": index + 1,
            }
        )
    return rows


def _dataset(label: str = "toy"):
    return {
        "label": label,
        "path": f"/{label}.json",
        "sha256": "0" * 64,
        "domains": ["a", "b"],
        "records": _records(),
        "source_settings": {},
    }


def _router_spec(name: str = "streak2", streak: int = 2):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "configs.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "name": name,
                        "router_config": {
                            "min_confidence": 0.0,
                            "min_margin": 0.0,
                            "min_current_margin": 0.0,
                            "min_consistent_windows_generated_target": streak,
                            "audio_ema_alpha": 0.0,
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        return load_router_specs(path)[0]


class DomainRouterReplaySweepTests(unittest.TestCase):
    def test_top_k_coverage_churn_and_transition_delay(self) -> None:
        top1 = evaluate_top_k(_records(), 1)
        top2 = evaluate_top_k(_records(), 2)

        self.assertEqual(top1["gold_included"], 2)
        self.assertEqual(top1["gold_slice_coverage"], 0.5)
        self.assertEqual(top1["transitions"]["details"][0]["delay_windows"], 0)
        self.assertEqual(top2["gold_included"], 4)
        self.assertEqual(top2["gold_slice_coverage"], 1.0)
        self.assertEqual(top2["mean_active_slices"], 2.0)
        self.assertEqual(top2["churn"]["selection_changes"], 0)

    def test_margin_summary_reports_gold_rank_and_signed_margin(self) -> None:
        summary = score_margins(_records())

        self.assertEqual(summary["top1_minus_top2"]["count"], 4)
        self.assertEqual(summary["gold_rank"]["mean"], 1.5)
        self.assertEqual(summary["nonpositive_gold_margin"], 2)
        self.assertLess(summary["gold_minus_best_non_gold"]["min"], 0.0)

    def test_hard_top1_replays_real_router_hysteresis(self) -> None:
        dataset = _dataset()
        dataset["records"][1]["scores"] = {"a": 0.8, "b": 0.2}
        dataset["records"][1]["ranked"] = [("a", 0.8), ("b", 0.2)]
        dataset["records"][3]["scores"] = {"b": 0.8, "a": 0.2}
        dataset["records"][3]["ranked"] = [("b", 0.8), ("a", 0.2)]

        result = replay_hard_top1(dataset, _router_spec(streak=2))

        self.assertEqual(result["gold_active"], 3)
        self.assertEqual(result["gold_slice_coverage"], 0.75)
        self.assertEqual(result["switches"], 1)
        self.assertEqual(result["useful_switches"], 1)
        self.assertEqual(result["wrong_target_switches"], 0)
        self.assertEqual(result["transitions"]["details"][0]["delay_windows"], 1)

    def test_predeclared_floor_selects_smallest_eligible_policy(self) -> None:
        candidates = [
            {
                "policy_id": "top_k:1",
                "minimum_dataset_coverage": 0.97,
                "gold_slice_coverage": 0.98,
                "mean_active_slices": 1.0,
                "churn_per_100_windows": 1.0,
                "mean_transition_delay_windows": 0.0,
                "unresolved_transitions": 0,
            },
            {
                "policy_id": "top_k:2",
                "minimum_dataset_coverage": 0.99,
                "gold_slice_coverage": 0.99,
                "mean_active_slices": 2.0,
                "churn_per_100_windows": 2.0,
                "mean_transition_delay_windows": 0.0,
                "unresolved_transitions": 0,
            },
            {
                "policy_id": "top_k:4",
                "minimum_dataset_coverage": 1.0,
                "gold_slice_coverage": 1.0,
                "mean_active_slices": 4.0,
                "churn_per_100_windows": 0.0,
                "mean_transition_delay_windows": 0.0,
                "unresolved_transitions": 0,
            },
        ]

        selection = choose_pareto(candidates, min_gold_coverage=0.99)

        self.assertFalse(selection["fallback_used"])
        self.assertEqual(selection["recommended"]["policy_id"], "top_k:2")
        self.assertFalse(selection["selection_policy"]["translation_metrics_used"])

    def test_sweep_uses_minimum_dataset_coverage(self) -> None:
        second = _dataset("toy2")
        for record in second["records"]:
            gold = record["gold"]
            other = "b" if gold == "a" else "a"
            record["scores"] = {other: 0.9, gold: 0.1}
            record["ranked"] = [(other, 0.9), (gold, 0.1)]

        result = run_sweep(
            [_dataset(), second],
            top_k_values=[1, 2],
            router_specs=[],
            min_gold_coverage=0.99,
        )

        top1 = next(row for row in result["aggregate_candidates"] if row["policy_id"] == "top_k:1")
        self.assertEqual(top1["minimum_dataset_coverage"], 0.0)
        self.assertEqual(result["selection"]["all_policies"]["recommended"]["policy_id"], "top_k:2")

    def test_loader_accepts_real_evaluator_schema_and_hashes_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text(
                json.dumps(
                    {
                        "domains": ["a", "b"],
                        "settings": {"window_segments": 6},
                        "records": [
                            {
                                "expected_domain": "a",
                                "scores": {"a": 0.8, "b": 0.2},
                                "text": "example",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_dataset(f"base={path}")

        self.assertEqual(dataset["label"], "base")
        self.assertEqual(dataset["records"][0]["gold"], "a")
        self.assertEqual(len(dataset["sha256"]), 64)
        self.assertEqual(dataset["source_settings"]["window_segments"], 6)

    def test_router_config_rejects_non_hard_top1_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "configs.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "name": "bad",
                            "router_config": {"slice_selection_mode": "budgeted_top_slices"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hard_top1"):
                load_router_specs(path)


if __name__ == "__main__":
    unittest.main()
