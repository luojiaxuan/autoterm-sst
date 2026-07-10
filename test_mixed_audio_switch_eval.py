from __future__ import annotations

import asyncio
import io
import json
import sys
import unittest
import urllib.error
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from eval.streaming_sst.eval_mixed_audio_switch import (
    TARGET_SAMPLE_RATE,
    AudioBlock,
    AudioBlockSpan,
    CursorBackpressure,
    build_schedule,
    build_spans,
    domain_transitions,
    extract_record,
    expected_domain_at,
    iter_oracle_chunk_plan,
    parse_oracle_preset_map,
    read_acl_audio_blocks,
    read_medicine_audio_blocks,
    resolve_max_switch_events,
    run_streaming_eval,
    summarize_run,
    switch_session_glossary,
    validate_oracle_preset_map,
)


def _write_wav(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((frames,), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(data.tobytes())


class MixedAudioSwitchEvalTests(unittest.TestCase):
    def test_oracle_preset_map_parser_and_missing_domain_validation(self) -> None:
        mapping = parse_oracle_preset_map(
            "nlp=nlp_core_10k, medicine=medicine_core_10k"
        )
        spans = [
            AudioBlockSpan(1, "acl", "acl", "nlp", 0, 16, 16, 1),
            AudioBlockSpan(2, "med", "medicine", "medicine", 16, 32, 16, 1),
        ]

        validate_oracle_preset_map(mapping, spans)
        with self.assertRaisesRegex(ValueError, "medicine"):
            validate_oracle_preset_map({"nlp": "nlp_core_10k"}, spans)
        with self.assertRaisesRegex(ValueError, "DOMAIN=PRESET"):
            parse_oracle_preset_map("nlp_core_10k")

    def test_oracle_chunk_plan_preserves_acl_medicine_acl_chunk_sequence(self) -> None:
        chunks = [
            np.full((16,), 1.0, dtype=np.float32),
            np.full((16,), 2.0, dtype=np.float32),
            np.full((16,), 3.0, dtype=np.float32),
        ]
        spans = [
            AudioBlockSpan(1, "acl-a", "acl", "nlp", 0, 16, 16, 1),
            AudioBlockSpan(2, "med", "medicine", "medicine", 16, 32, 16, 1),
            AudioBlockSpan(3, "acl-b", "acl", "nlp", 32, 48, 16, 1),
        ]
        mapping = {"nlp": "nlp_core_10k", "medicine": "medicine_core_10k"}

        plans = list(iter_oracle_chunk_plan(iter(chunks), spans=spans, oracle_preset_map=mapping))

        self.assertEqual([plan.expected_domain for plan in plans], ["nlp", "medicine", "nlp"])
        self.assertEqual(
            [plan.glossary_preset for plan in plans],
            ["nlp_core_10k", "medicine_core_10k", "nlp_core_10k"],
        )
        self.assertEqual([plan.future_cursor_samples for plan in plans], [16, 32, 48])
        for original, plan in zip(chunks, plans):
            self.assertIs(plan.chunk, original)
            np.testing.assert_array_equal(plan.chunk, original)

    def test_oracle_glossary_http_failure_names_requested_preset(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8012/glossary/build",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"detail":"unknown preset"}'),
        )
        with (
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "medicine_core_10k.*HTTP 400"),
        ):
            switch_session_glossary(
                "http://127.0.0.1:8012",
                "session-1",
                "medicine_core_10k",
                language_pair="English -> Chinese",
            )

    def test_cursor_backpressure_waits_for_status_or_partial_cursor(self) -> None:
        async def scenario():
            pacing = CursorBackpressure(
                chunk_samples=16,
                max_unacked_chunks=1,
                stall_timeout_sec=1.0,
            )
            pacing.record_sent(16)
            waiting = asyncio.create_task(pacing.wait_to_send(16))
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())

            await pacing.observe_event("status", {"cursor_samples": 16})
            self.assertFalse(await waiting)
            pacing.record_sent(16)
            await pacing.observe_event("partial", {"cursor_samples": 32})
            return pacing.snapshot()

        stats = asyncio.run(scenario())

        self.assertEqual(stats["sent_samples"], 32)
        self.assertEqual(stats["acknowledged_cursor_samples"], 32)
        self.assertEqual(stats["wait_count"], 1)
        self.assertEqual(stats["status_cursor_event_count"], 1)
        self.assertEqual(stats["partial_cursor_event_count"], 1)
        self.assertEqual(stats["timeout_release_count"], 0)
        self.assertLessEqual(stats["max_observed_unacked_chunks"], 1.0)

    def test_cursor_backpressure_silence_timeout_releases_only_one_chunk(self) -> None:
        async def scenario():
            pacing = CursorBackpressure(
                chunk_samples=16,
                max_unacked_chunks=1,
                stall_timeout_sec=0.01,
            )
            pacing.record_sent(16)
            first_released = await pacing.wait_to_send(16)
            pacing.record_sent(16, timeout_released=first_released)
            second_waiting = asyncio.create_task(pacing.wait_to_send(16))
            await asyncio.sleep(0)
            self.assertFalse(second_waiting.done())
            await pacing.observe_event("status", {"cursor_samples": 32})
            second_released = await second_waiting
            pacing.record_sent(16, timeout_released=second_released)
            return pacing.snapshot()

        stats = asyncio.run(scenario())

        self.assertEqual(stats["wait_count"], 2)
        self.assertEqual(stats["timeout_release_count"], 1)
        self.assertEqual(stats["timeout_release_samples"], 16)

    def test_resolve_max_switch_events_from_seconds(self) -> None:
        chunk_samples = int(1.92 * TARGET_SAMPLE_RATE)

        self.assertEqual(resolve_max_switch_events(3, 30.0, chunk_samples), 16)
        self.assertEqual(resolve_max_switch_events(3, 0.0, chunk_samples), 3)

    def test_audio_block_readers_and_schedule(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            acl_wav_a = acl_root / "seg" / "000.wav"
            acl_wav_b = acl_root / "seg" / "001.wav"
            _write_wav(acl_wav_a, TARGET_SAMPLE_RATE)
            _write_wav(acl_wav_b, TARGET_SAMPLE_RATE)
            (acl_root / "segments.meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"talk": "acl_a", "seg_wav": str(acl_wav_a)}),
                        json.dumps({"talk": "acl_b", "seg_wav": str(acl_wav_b)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            med_dir = root / "medicine"
            med_wav = med_dir / "sample_404_v2" / "404_v2.wav"
            _write_wav(med_wav, TARGET_SAMPLE_RATE)

            acl = read_acl_audio_blocks(str(acl_root), limit_items=2)
            medicine = read_medicine_audio_blocks(str(med_dir), limit_items=1)
            schedule = build_schedule(acl, medicine, schedule="alternating")

        self.assertEqual([item.item_id for item in schedule], ["acl_a", "medicine_404", "acl_b"])
        self.assertEqual([item.expected_domain for item in schedule], ["nlp", "medicine", "nlp"])

    def test_medicine_reader_can_select_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            med_dir = root / "medicine"
            med_wav_404 = med_dir / "sample_404_v2" / "404_v2.wav"
            med_wav_606 = med_dir / "sample_606_v2" / "606_v2.wav"
            _write_wav(med_wav_404, TARGET_SAMPLE_RATE)
            _write_wav(med_wav_606, TARGET_SAMPLE_RATE)

            blocks = read_medicine_audio_blocks(
                str(med_dir),
                limit_items=2,
                medicine_ids=["606", "medicine_404"],
            )

            with self.assertRaisesRegex(FileNotFoundError, "999"):
                read_medicine_audio_blocks(str(med_dir), limit_items=1, medicine_ids=["999"])

        self.assertEqual([item.item_id for item in blocks], ["medicine_606", "medicine_404"])

    def test_acl_reader_filters_missing_wavs_before_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            existing_a = acl_root / "seg" / "000.wav"
            existing_b = acl_root / "seg" / "002.wav"
            _write_wav(existing_a, TARGET_SAMPLE_RATE)
            _write_wav(existing_b, TARGET_SAMPLE_RATE)
            (acl_root / "segments.meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"talk": "missing_first", "seg_wav": str(acl_root / "seg" / "missing.wav")}),
                        json.dumps({"talk": "acl_a", "seg_wav": str(existing_a)}),
                        json.dumps({"talk": "acl_b", "seg_wav": str(existing_b)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            blocks = read_acl_audio_blocks(str(acl_root), limit_items=2)

        self.assertEqual([block.item_id for block in blocks], ["acl_a", "acl_b"])

    def test_audio_block_readers_respect_zero_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            acl_wav = acl_root / "seg" / "000.wav"
            _write_wav(acl_wav, TARGET_SAMPLE_RATE)
            (acl_root / "segments.meta.jsonl").write_text(
                json.dumps({"talk": "acl_a", "seg_wav": str(acl_wav)}) + "\n",
                encoding="utf-8",
            )
            med_dir = root / "medicine"
            med_wav = med_dir / "sample_404_v2" / "404_v2.wav"
            _write_wav(med_wav, TARGET_SAMPLE_RATE)

            acl = read_acl_audio_blocks(str(acl_root), limit_items=0)
            medicine = read_medicine_audio_blocks(str(med_dir), limit_items=0)

        self.assertEqual(acl, [])
        self.assertEqual(medicine, [])

    def test_spans_and_expected_domain_lookup(self) -> None:
        blocks = [
            AudioBlock("acl", "nlp", "acl", ["mock-a.wav"]),
            AudioBlock("med", "medicine", "medicine", ["mock-b.wav"]),
        ]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)

        self.assertEqual(spans[0].start_sample, 0)
        self.assertEqual(spans[0].end_sample, TARGET_SAMPLE_RATE)
        self.assertEqual(spans[1].start_sample, TARGET_SAMPLE_RATE)
        self.assertEqual(expected_domain_at(spans, 1), "nlp")
        self.assertEqual(expected_domain_at(spans, TARGET_SAMPLE_RATE), "nlp")
        self.assertEqual(expected_domain_at(spans, TARGET_SAMPLE_RATE + 1), "medicine")

    def test_summary_detects_transition_latency_and_steady_state(self) -> None:
        blocks = [
            AudioBlock("acl", "nlp", "acl", ["mock-a.wav"]),
            AudioBlock("med", "medicine", "medicine", ["mock-b.wav"]),
        ]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)
        records = [
            {
                "event_idx": 1,
                "cursor_samples": TARGET_SAMPLE_RATE // 2,
                "expected_domain": "nlp",
                "active_domain": "nlp",
                "router_action": "stay",
                "router_target_domain": "nlp",
                "domain_probe_top_domain": "nlp",
            },
            {
                "event_idx": 2,
                "cursor_samples": TARGET_SAMPLE_RATE + TARGET_SAMPLE_RATE // 4,
                "expected_domain": "medicine",
                "active_domain": "nlp",
                "router_action": "stay",
                "router_target_domain": "medicine",
                "domain_probe_top_domain": "medicine",
            },
            {
                "event_idx": 3,
                "cursor_samples": TARGET_SAMPLE_RATE + TARGET_SAMPLE_RATE // 2,
                "expected_domain": "medicine",
                "active_domain": "medicine",
                "router_action": "switch",
                "router_target_domain": "medicine",
                "domain_probe_top_domain": "medicine",
                "switch_count": 1,
            },
        ]

        transitions = domain_transitions(spans, records, max_switch_events=2)
        summary = summarize_run(
            schedule_name="alternating",
            preset="auto_working",
            spans=spans,
            records=records,
            chunk_samples=TARGET_SAMPLE_RATE // 2,
            max_switch_events=2,
        )

        self.assertEqual(transitions[0]["latency_events"], 2)
        self.assertTrue(transitions[0]["passed"])
        self.assertTrue(summary["regression_pass"])
        self.assertEqual(summary["steady_state_mismatch_count"], 0)
        self.assertEqual(summary["probe_top_accuracy"], 1.0)

    def test_extract_record_requires_cursor_samples_and_runtime_schema(self) -> None:
        blocks = [AudioBlock("acl", "nlp", "acl", ["mock-a.wav"])]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)
        event = {
            "type": "partial",
            "text": "测试",
            "meta": {
                "cursor_samples": TARGET_SAMPLE_RATE,
                "start_sample": TARGET_SAMPLE_RATE // 2,
                "topic": {
                    "active_domain": "nlp",
                    "active_glossary_preset": "nlp_core_10k",
                    "switch_count": 0,
                },
                "topic_router": {
                    "action": "stay",
                    "to_preset": "nlp_core_10k",
                    "confidence": 0.9,
                    "margin": 0.5,
                    "evidence": {
                        "slice_selection": {
                            "selected_slice_presets": [
                                "nlp_core_10k",
                                "science_core_10k",
                            ],
                            "selected_slice_count": 2,
                            "selected_term_count": 20000,
                        }
                    },
                },
                "domain_probe_scores": {
                    "nlp": 0.8,
                },
                "router_text_source": "generated_target",
                "prompt_reference_count": 10,
                "fixed_prompt_k": 10,
                "candidate_pool_count": 50,
                "retrieval_candidate_cost": {
                    "candidate_budget": 100,
                    "index_query_count": 2,
                    "scored_inventory_terms": 20000,
                },
                "references": [
                    {"term": "SRL", "translation": "语义角色标注", "score": 0.91},
                ],
            },
        }

        record = extract_record(event, event_idx=1, spans=spans)
        self.assertEqual(record["cursor_samples"], TARGET_SAMPLE_RATE)
        self.assertEqual(record["expected_domain"], "nlp")
        self.assertEqual(record["router_target_domain"], "nlp")
        self.assertEqual(record["domain_probe_top_domain"], "nlp")
        self.assertEqual(record["start_sample"], TARGET_SAMPLE_RATE // 2)
        self.assertEqual(record["references"][0]["term"], "SRL")
        self.assertEqual(
            record["selected_slice_presets"],
            ["nlp_core_10k", "science_core_10k"],
        )
        self.assertEqual(record["selected_slice_count"], 2)
        self.assertEqual(record["selected_term_count"], 20000)
        self.assertEqual(record["retrieval_candidate_cost"]["candidate_budget"], 100)

        bad = dict(event)
        bad["meta"] = dict(event["meta"])
        del bad["meta"]["cursor_samples"]
        with self.assertRaisesRegex(RuntimeError, "cursor_samples"):
            extract_record(bad, event_idx=1, spans=spans)

        fixed_meta = dict(event["meta"])
        del fixed_meta["topic_router"]
        del fixed_meta["domain_probe_scores"]
        fixed_event = dict(event)
        fixed_event["meta"] = fixed_meta
        fixed_record = extract_record(fixed_event, event_idx=1, spans=spans, require_router_meta=False)
        self.assertEqual(fixed_record["router_target_domain"], "general")
        with self.assertRaisesRegex(RuntimeError, "topic_router"):
            extract_record(fixed_event, event_idx=1, spans=spans)

    def test_streaming_oracle_switches_acl_medicine_acl_before_sending_chunks(self) -> None:
        chunks = [
            np.full((16,), 1.0, dtype=np.float32),
            np.full((16,), 2.0, dtype=np.float32),
            np.full((16,), 3.0, dtype=np.float32),
        ]
        blocks = [
            AudioBlock("acl-a", "nlp", "acl", ["mock-a.wav"]),
            AudioBlock("med", "medicine", "medicine", ["mock-b.wav"]),
            AudioBlock("acl-b", "nlp", "acl", ["mock-c.wav"]),
        ]
        spans = [
            AudioBlockSpan(1, "acl-a", "acl", "nlp", 0, 16, 16, 1),
            AudioBlockSpan(2, "med", "medicine", "medicine", 16, 32, 16, 1),
            AudioBlockSpan(3, "acl-b", "acl", "nlp", 32, 48, 16, 1),
        ]
        partial = {
            "type": "partial",
            "text": "done",
            "meta": {
                "cursor_samples": 48,
                "topic": {
                    "active_domain": "nlp",
                    "active_glossary_preset": "nlp_core_10k",
                    "switch_count": 0,
                },
                "router_text_source": "none",
                "prompt_reference_count": 0,
                "fixed_prompt_k": 10,
                "candidate_pool_count": 100,
            },
        }

        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent = []
                self.initial_sent = False
                self.status_16_sent = False
                self.status_32_sent = False
                self.partial_sent = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def send(self, payload):
                self.sent.append(payload)

            async def recv(self):
                if not self.initial_sent:
                    self.initial_sent = True
                    return json.dumps({"type": "status", "text": "READY"})
                while not self.partial_sent:
                    binary_count = sum(isinstance(item, bytes) for item in self.sent)
                    if binary_count >= 1 and not self.status_16_sent:
                        self.status_16_sent = True
                        return json.dumps(
                            {"type": "status", "text": "ACK", "meta": {"cursor_samples": 16}}
                        )
                    if binary_count >= 2 and not self.status_32_sent:
                        self.status_32_sent = True
                        return json.dumps(
                            {"type": "status", "text": "ACK", "meta": {"cursor_samples": 32}}
                        )
                    if binary_count >= 3:
                        self.partial_sent = True
                        return json.dumps(partial)
                    await asyncio.sleep(0.001)
                await asyncio.sleep(1.0)
                return json.dumps({"type": "status", "text": "unused"})

        fake_ws = FakeWebSocket()
        fake_module = SimpleNamespace(connect=lambda *args, **kwargs: fake_ws)
        previous_module = sys.modules.get("websockets")
        sys.modules["websockets"] = fake_module

        def fake_switch(base_url, session_id, glossary_preset, **kwargs):
            domain = "medicine" if glossary_preset == "medicine_core_10k" else "nlp"
            return {
                "success": True,
                "session_updated": True,
                "active_domain": domain,
                "active_glossary_preset": glossary_preset,
                "auto_glossary_enabled": False,
            }

        try:
            with (
                patch(
                    "eval.streaming_sst.eval_mixed_audio_switch.init_session",
                    return_value={
                        "session_id": "s1",
                        "active_domain": "nlp",
                        "active_glossary_preset": "nlp_core_10k",
                        "auto_glossary_enabled": False,
                    },
                ) as init_mock,
                patch("eval.streaming_sst.eval_mixed_audio_switch.delete_session"),
                patch(
                    "eval.streaming_sst.eval_mixed_audio_switch.iter_pcm_chunks",
                    return_value=iter(chunks),
                ),
                patch(
                    "eval.streaming_sst.eval_mixed_audio_switch.switch_session_glossary",
                    side_effect=fake_switch,
                ) as switch_mock,
            ):
                payload = asyncio.run(
                    run_streaming_eval(
                        base_url="http://127.0.0.1:8012",
                        language_pair="English -> Chinese",
                        preset="auto_working",
                        blocks=blocks,
                        spans=spans,
                        chunk_samples=16,
                        feed_sleep=0.0,
                        latency_multiplier=2,
                        idle_timeout_sec=0.02,
                        idle_timeouts_after_eof=1,
                        require_router_meta=False,
                        max_unacked_chunks=1,
                        backpressure_stall_timeout_sec=1.0,
                        oracle_preset_map={
                            "nlp": "nlp_core_10k",
                            "medicine": "medicine_core_10k",
                        },
                        oracle_switch_timeout_sec=1.0,
                    )
                )
        finally:
            if previous_module is None:
                sys.modules.pop("websockets", None)
            else:
                sys.modules["websockets"] = previous_module

        init_mock.assert_called_once_with(
            "http://127.0.0.1:8012",
            "English -> Chinese",
            "nlp_core_10k",
            2,
        )
        self.assertEqual(
            [call.args[2] for call in switch_mock.call_args_list],
            ["medicine_core_10k", "nlp_core_10k"],
        )
        self.assertEqual(fake_ws.sent[:3], [chunk.tobytes() for chunk in chunks])
        self.assertEqual(fake_ws.sent[-1], "EOF")
        self.assertTrue(payload["oracle"]["enabled"])
        self.assertEqual(payload["oracle"]["switch_count"], 2)
        self.assertEqual(
            [row["to_domain"] for row in payload["oracle"]["switches"]],
            ["nlp", "medicine", "nlp"],
        )
        self.assertEqual(payload["pacing"]["cursor_barrier_timeout_count"], 0)

    def test_streaming_eval_drains_partials_after_processing_complete(self) -> None:
        blocks = [AudioBlock("acl", "nlp", "acl", ["mock-a.wav"])]
        spans = [
            SimpleNamespace(
                start_sample=0,
                end_sample=TARGET_SAMPLE_RATE,
                expected_domain="nlp",
            )
        ]
        partial = {
            "type": "partial",
            "text": "测试",
            "meta": {
                "cursor_samples": TARGET_SAMPLE_RATE,
                "topic": {
                    "active_domain": "nlp",
                    "active_glossary_preset": "nlp_core_10k",
                    "switch_count": 0,
                },
                "topic_router": {
                    "action": "stay",
                    "to_preset": "nlp_core_10k",
                    "confidence": 0.9,
                    "margin": 0.5,
                },
                "domain_probe_scores": {"nlp": 0.8},
                "router_text_source": "generated_target",
                "prompt_reference_count": 10,
                "fixed_prompt_k": 10,
                "candidate_pool_count": 50,
            },
        }

        class FakeWebSocket:
            def __init__(self) -> None:
                self.sent = []
                self.messages = [
                    json.dumps({"type": "status", "text": "READY: framework ready"}),
                    json.dumps({"type": "status", "text": "PROCESSING_COMPLETE: File processing finished"}),
                    json.dumps(partial),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def send(self, payload):
                self.sent.append(payload)

            async def recv(self):
                if self.messages:
                    if len(self.messages) == 2:
                        await asyncio.sleep(0.01)
                    return self.messages.pop(0)
                await asyncio.sleep(1.0)
                return json.dumps({"type": "status", "text": "unused"})

        fake_ws = FakeWebSocket()
        fake_module = SimpleNamespace(connect=lambda *args, **kwargs: fake_ws)
        previous_module = sys.modules.get("websockets")
        sys.modules["websockets"] = fake_module
        try:
            with (
                patch("eval.streaming_sst.eval_mixed_audio_switch.init_session", return_value={"session_id": "s1"}),
                patch("eval.streaming_sst.eval_mixed_audio_switch.delete_session"),
                patch(
                    "eval.streaming_sst.eval_mixed_audio_switch.iter_pcm_chunks",
                    return_value=iter([np.zeros((16,), dtype=np.float32)]),
                ),
            ):
                payload = asyncio.run(
                    run_streaming_eval(
                        base_url="http://127.0.0.1:8012",
                        language_pair="English -> Chinese",
                        preset="auto_working",
                        blocks=blocks,
                        spans=spans,
                        chunk_samples=16,
                        feed_sleep=0.0,
                        latency_multiplier=2,
                        idle_timeout_sec=0.02,
                        idle_timeouts_after_eof=1,
                    )
                )
        finally:
            if previous_module is None:
                sys.modules.pop("websockets", None)
            else:
                sys.modules["websockets"] = previous_module

        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["records"][0]["active_domain"], "nlp")
        self.assertFalse(payload["pacing"]["enabled"])
        self.assertEqual(payload["pacing"]["sent_samples"], 16)
        self.assertEqual(payload["pacing"]["partial_cursor_event_count"], 1)
        self.assertEqual(fake_ws.sent[-1], "EOF")


if __name__ == "__main__":
    unittest.main()
