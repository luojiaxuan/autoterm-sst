"""Pair-level catalog operations for the same-catalog evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Entry = Dict[str, object]
Pair = Tuple[str, str]


def pair_of(entry: Entry) -> Pair:
    """A pair is a case-folded source term with its Chinese translation."""
    return str(entry["term"]).strip().lower(), str(entry["zh"]).strip()


def distinct(entries: Iterable[Entry]) -> List[Entry]:
    """Keep the first entry of every pair, in order."""
    seen = set()
    kept = []
    for entry in entries:
        pair = pair_of(entry)
        if pair not in seen:
            seen.add(pair)
            kept.append(entry)
    return kept


def fill(catalog: Sequence[Entry], pool: Iterable[Entry], size: int) -> List[Entry]:
    """The pairs of ``pool``, in order, that bring ``catalog`` to ``size`` distinct pairs."""
    seen = {pair_of(entry) for entry in catalog}
    added: List[Entry] = []
    for entry in pool:
        if len(seen) == size:
            break
        pair = pair_of(entry)
        if pair not in seen:
            seen.add(pair)
            added.append(entry)
    if len(seen) != size:
        raise ValueError(f"the pool yields {len(seen)} distinct pairs, {size} requested")
    return added


def extend(slices: Dict[str, List[Entry]], added: Sequence[Entry], topics: Sequence[str]) -> Dict[str, List[Entry]]:
    """Append every added pair to the slice of its topic."""
    if len(added) != len(topics):
        raise ValueError("one topic is needed per added pair")
    grown = {domain: list(entries) for domain, entries in slices.items()}
    for entry, topic in zip(added, topics):
        grown[topic].append(entry)
    return grown


def conflicts(entries: Iterable[Entry]) -> int:
    """The number of source terms that carry more than one translation."""
    translations: Dict[str, set] = {}
    for entry in entries:
        term, zh = pair_of(entry)
        translations.setdefault(term, set()).add(zh)
    return sum(len(found) > 1 for found in translations.values())
