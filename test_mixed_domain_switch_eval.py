from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.streaming_sst.eval_mixed_domain_switch import (
    PlaylistBlock,
    build_schedule,
    evaluate_playlist,
    read_acl_blocks,
)


NLP_TARGET_WINDOWS = (
    "该语言模型在机器翻译数据集和基准测试上表现稳定。",
    "研究使用预训练和微调来改进注意力与嵌入表示。",
    "系统评估 BLEU、语料库标注和实体识别结果。",
    "解码器和编码器结构提升了自然语言处理任务表现。",
)

MEDICINE_TARGET_WINDOWS = (
    "患者在医院接受临床治疗，医生根据诊断调整药物剂量。",
    "癌症和感染相关症状需要结合临床试验结果分析。",
    "手术后需要监测血压、心率以及疫苗治疗反应。",
    "医生根据糖尿病和高血压情况开具处方。",
)


def _blocks() -> tuple[list[PlaylistBlock], list[PlaylistBlock]]:
    acl = [
        PlaylistBlock(f"acl_{idx}", "nlp", NLP_TARGET_WINDOWS, corpus="acl")
        for idx in range(2)
    ]
    medicine = [
        PlaylistBlock(f"medicine_{idx}", "medicine", MEDICINE_TARGET_WINDOWS, corpus="medicine")
        for idx in range(2)
    ]
    return acl, medicine


class MixedDomainSwitchEvalTests(unittest.TestCase):
    def test_alternating_generated_target_playlist_switches_with_expected_probe(self) -> None:
        acl, medicine = _blocks()
        payload = evaluate_playlist(
            build_schedule(acl, medicine, schedule="alternating"),
            schedule_name="alternating",
            router_text_source="generated_target",
            probe_mode="expected",
        )

        summary = payload["summary"]
        self.assertTrue(summary["regression_pass"])
        self.assertEqual(summary["block_count"], 4)
        self.assertEqual(summary["domain_transition_count"], 3)
        self.assertEqual(summary["max_observed_switch_latency_windows"], 3)
        self.assertEqual(summary["steady_state_mismatch_count"], 0)

    def test_random_playlist_counts_only_domain_transition_boundaries(self) -> None:
        acl, medicine = _blocks()
        blocks = build_schedule(acl, medicine, schedule="random", seed=7)
        payload = evaluate_playlist(
            blocks,
            schedule_name="random",
            router_text_source="generated_target",
            probe_mode="expected",
        )
        expected_transitions = sum(
            1
            for prev, current in zip(blocks, blocks[1:])
            if prev.expected_domain != current.expected_domain
        )

        self.assertTrue(payload["summary"]["regression_pass"])
        self.assertEqual(payload["summary"]["domain_transition_count"], expected_transitions)

    def test_generated_target_without_probe_is_diagnostic_not_deployable_pass(self) -> None:
        acl, medicine = _blocks()
        payload = evaluate_playlist(
            build_schedule(acl[:1], medicine[:1], schedule="alternating"),
            schedule_name="alternating",
            router_text_source="generated_target",
            probe_mode="none",
        )

        self.assertFalse(payload["summary"]["regression_pass"])
        self.assertEqual(payload["summary"]["switch_count"], 0)

    def test_inverted_probe_diagnostic_fails(self) -> None:
        acl, medicine = _blocks()
        payload = evaluate_playlist(
            build_schedule(acl[:1], medicine[:1], schedule="alternating"),
            schedule_name="alternating",
            router_text_source="generated_target",
            probe_mode="inverted",
        )

        self.assertFalse(payload["summary"]["regression_pass"])
        self.assertLess(payload["summary"]["steady_state_accuracy"], 1.0)

    def test_acl_reader_preserves_meta_alignment_with_blank_text_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "segments.target").write_text("第一条\n\n第三条\n", encoding="utf-8")
            (root / "segments.source").write_text("one\n\nt hree\n", encoding="utf-8")
            (root / "segments.meta.jsonl").write_text(
                '{"talk":"talk_a"}\n{"talk":"talk_blank"}\n{"talk":"talk_c"}\n',
                encoding="utf-8",
            )

            blocks = read_acl_blocks(
                str(root),
                limit_items=3,
                windows_per_item=1,
                text_field="target",
            )

        self.assertEqual([block.item_id for block in blocks], ["talk_a", "talk_c"])
        self.assertEqual([list(block.windows) for block in blocks], [["第一条"], ["第三条"]])


if __name__ == "__main__":
    unittest.main()
