"""Frozen talk-local windows for the inexpensive four-block smoke protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

TARGET_SAMPLE_RATE = 16_000
PROTOCOL_ID = "frozen4_selected_window_v1"
EXPECTED_RAW_DENOMINATOR = 179


@dataclass(frozen=True)
class SelectedWindow:
    original_item_id: str
    corpus: str
    expected_domain: str
    source_offset_samples: int
    source_end_samples: int

    @property
    def sample_count(self) -> int:
        return self.source_end_samples - self.source_offset_samples

    @property
    def item_id(self) -> str:
        return (
            f"{self.original_item_id}__window_"
            f"{self.source_offset_samples}_{self.source_end_samples}"
        )

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "item_id": self.item_id,
                "source_offset_s": self.source_offset_samples / TARGET_SAMPLE_RATE,
                "source_end_s": self.source_end_samples / TARGET_SAMPLE_RATE,
                "sample_count": self.sample_count,
            }
        )
        return row


FROZEN4_WINDOWS: tuple[SelectedWindow, ...] = (
    SelectedWindow(
        "2022.acl-long.268",
        "acl",
        "nlp",
        int(round(278.4 * TARGET_SAMPLE_RATE)),
        int(round(398.4 * TARGET_SAMPLE_RATE)),
    ),
    SelectedWindow(
        "medicine_545006",
        "medicine",
        "medicine",
        0,
        int(round(180.0 * TARGET_SAMPLE_RATE)),
    ),
    SelectedWindow(
        "2022.acl-long.367",
        "acl",
        "nlp",
        int(round(180.48 * TARGET_SAMPLE_RATE)),
        int(round(270.48 * TARGET_SAMPLE_RATE)),
    ),
    SelectedWindow(
        "medicine_606",
        "medicine",
        "medicine",
        int(round(1409.28 * TARGET_SAMPLE_RATE)),
        int(round(1589.28 * TARGET_SAMPLE_RATE)),
    ),
)


def protocol_manifest() -> Dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "sample_rate": TARGET_SAMPLE_RATE,
        "expected_raw_denominator": EXPECTED_RAW_DENOMINATOR,
        "windows": [window.as_dict() for window in FROZEN4_WINDOWS],
    }


def validate_payload(payload: Mapping[str, Any]) -> None:
    config = payload.get("config") or {}
    if not isinstance(config, Mapping) or config.get("selected_window_protocol") != PROTOCOL_ID:
        raise ValueError(f"run is not marked as selected-window protocol {PROTOCOL_ID}")
    blocks = payload.get("blocks") or []
    spans = payload.get("block_spans") or []
    if len(blocks) != len(FROZEN4_WINDOWS) or len(spans) != len(FROZEN4_WINDOWS):
        raise ValueError(
            f"selected-window protocol requires {len(FROZEN4_WINDOWS)} blocks/spans"
        )
    cursor = 0
    for block_index, (block, span, expected) in enumerate(
        zip(blocks, spans, FROZEN4_WINDOWS),
        start=1,
    ):
        for row_name, row in (("block", block), ("span", span)):
            if not isinstance(row, Mapping):
                raise ValueError(f"selected-window {row_name} {block_index} is not an object")
            actual = (
                str(row.get("original_item_id") or ""),
                str(row.get("corpus") or ""),
                str(row.get("expected_domain") or ""),
                int(row.get("source_offset_samples") or 0),
                int(row.get("source_end_samples") or 0),
            )
            wanted = (
                expected.original_item_id,
                expected.corpus,
                expected.expected_domain,
                expected.source_offset_samples,
                expected.source_end_samples,
            )
            if actual != wanted:
                raise ValueError(
                    f"selected-window {row_name} {block_index} is {actual}, expected {wanted}"
                )
        for row_name, row in (("block", block), ("span", span)):
            if str(row.get("item_id") or "") != expected.item_id:
                raise ValueError(
                    f"selected-window {row_name} {block_index} item_id does not "
                    "encode its window"
                )
        if int(span.get("sample_count") or 0) != expected.sample_count:
            raise ValueError(
                f"selected-window span {block_index} has sample_count="
                f"{span.get('sample_count')}, expected {expected.sample_count}"
            )
        expected_span = (
            block_index,
            cursor,
            cursor + expected.sample_count,
        )
        actual_span = (
            int(span.get("block_index") or 0),
            int(span.get("start_sample") or 0),
            int(span.get("end_sample") or 0),
        )
        if actual_span != expected_span:
            raise ValueError(
                f"selected-window span {block_index} is {actual_span}, "
                f"expected {expected_span}"
            )
        cursor += expected.sample_count
