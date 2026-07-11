from __future__ import annotations

from pathlib import Path

from eval.streaming_sst.score_resegmented_masked_bleu import (
    parse_domain_glossaries,
    wav_domains,
)


def test_wav_domains_uses_corpus_labels() -> None:
    assert wav_domains(
        {
            "blocks": [
                {"wav": "acl.wav", "corpus": "acl"},
                {"wav": "medicine.wav", "corpus": "medicine"},
            ]
        }
    ) == {"acl.wav": "acl", "medicine.wav": "medicine"}


def test_parse_domain_glossaries_rejects_duplicates(tmp_path: Path) -> None:
    glossary = tmp_path / "g.json"
    glossary.write_text("[]", encoding="utf-8")
    try:
        parse_domain_glossaries([f"acl={glossary}", f"acl={glossary}"])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate glossary domain was accepted")
