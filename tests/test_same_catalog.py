import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts" / "same_catalog" / "catalog.py"
_spec = importlib.util.spec_from_file_location("same_catalog_catalog", _path)
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)


def entry(term, zh):
    return {"term": term, "zh": zh}


def test_distinct_keeps_conflicting_translations_and_first_case():
    kept = catalog.distinct([entry("Paper", "论文"), entry("paper", "论文"), entry("paper", "纸")])
    assert kept == [entry("Paper", "论文"), entry("paper", "纸")]


def test_fill_skips_known_pairs_and_stops_at_size():
    pool = [entry("a", "甲"), entry("b", "乙"), entry("B", "乙"), entry("c", "丙"), entry("d", "丁")]
    assert catalog.fill([entry("A", "甲")], pool, 3) == [entry("b", "乙"), entry("c", "丙")]


def test_fill_refuses_a_short_pool():
    with pytest.raises(ValueError):
        catalog.fill([], [entry("a", "甲")], 2)


def test_extended_slices_cover_exactly_the_merged_catalog():
    slices = {"x": [entry("a", "甲")], "y": [entry("b", "乙")]}
    added = [entry("c", "丙"), entry("d", "丁")]
    grown = catalog.extend(slices, added, ["y", "x"])
    assert grown["x"] == [entry("a", "甲"), entry("d", "丁")]
    assert slices["x"] == [entry("a", "甲")]
    everything = [e for v in grown.values() for e in v]
    assert {catalog.pair_of(e) for e in everything} == {catalog.pair_of(e) for e in [*slices["x"], *slices["y"], *added]}


def test_conflicts_counts_source_terms_not_pairs():
    assert catalog.conflicts([entry("paper", "论文"), entry("Paper", "纸"), entry("a", "甲"), entry("A", "甲")]) == 1


def test_balanced_topics_have_equal_sizes_and_keep_clear_choices():
    torch = pytest.importorskip("torch")
    _bspec = importlib.util.spec_from_file_location("same_catalog_build", _path.parent / "build.py")
    import sys
    sys.path.insert(0, str(_path.parent))
    build = importlib.util.module_from_spec(_bspec)
    _bspec.loader.exec_module(build)
    gen = torch.Generator().manual_seed(0)
    sim = torch.rand(6000, 3, generator=gen) * 0.1
    sim[:4000, 0] += 0.5
    sim[:10, 2] += 1.0
    pick = build.balance(sim, 2000.0)
    assert torch.bincount(pick, minlength=3).tolist() == pytest.approx([2000, 2000, 2000], rel=0.011)
    assert (pick[:10] == 2).all()
