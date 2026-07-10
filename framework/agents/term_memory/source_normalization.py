"""Canonical identity normalization for English glossary source terms."""

from __future__ import annotations

import unicodedata
from typing import Any


SOURCE_NORMALIZATION_POLICY = "NFKC + casefold + collapsed whitespace"


def normalize_english_source(value: Any) -> str:
    """Return the catalog identity key used for English source labels."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())
