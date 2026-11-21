import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval" / "streaming_sst" / "evidence"))

from framework.agents.term_memory.language_model_router import filter_step  # noqa: E402
from llm_hmm_filter import forward  # noqa: E402

DOMAINS = ["nlp", "medicine", "finance", "legal"]


def test_filter_step_matches_offline_forward():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(len(DOMAINS)), size=50)
    offline = forward(probs, rho=1e-3, beta=0.25)
    posterior = {}
    for t, row in enumerate(probs):
        posterior = filter_step(posterior, dict(zip(DOMAINS, row)), switch_rate=1e-3, evidence_weight=0.25)
        assert np.allclose([posterior[d] for d in DOMAINS], offline[t], atol=1e-9)


def test_one_contrary_window_does_not_flip_a_confident_topic():
    posterior = {"nlp": 0.97, "medicine": 0.01, "finance": 0.01, "legal": 0.01}
    window = {"nlp": 0.1, "medicine": 0.8, "finance": 0.05, "legal": 0.05}
    after = filter_step(posterior, window, switch_rate=1e-3, evidence_weight=0.25)
    assert max(after, key=after.get) == "nlp"


def test_repeated_evidence_moves_the_topic():
    posterior = {"nlp": 0.97, "medicine": 0.01, "finance": 0.01, "legal": 0.01}
    window = {"nlp": 0.05, "medicine": 0.9, "finance": 0.025, "legal": 0.025}
    for _ in range(12):
        posterior = filter_step(posterior, window, switch_rate=1e-3, evidence_weight=0.25)
    assert max(posterior, key=posterior.get) == "medicine"


def test_topic_follows_the_posterior_top_slice():
    from types import SimpleNamespace

    from framework.agents.omni import OmniAgent
    from framework.agents.term_memory.slice_registry import RetrievalSlice

    def slice_(domain):
        return RetrievalSlice(preset_id=f"{domain}_core_1k", slice_id=domain, role="topic", domain=domain,
                              index_path=f"/{domain}.pt", term_count=1000)

    agent = SimpleNamespace(config=SimpleNamespace(router_context_switch_rate=1e-3))
    session = SimpleNamespace(active_retrieval_slices=[], active_slice_presets=[], active_slice_terms={},
                              last_retrieval_plan=[], top_slice_changes=0,
                              router_posterior={"medicine": 0.7, "nlp": 0.2},
                              active_domain="nlp", topic_confidence=1.0, active_glossary_preset="nlp_core_1k",
                              glossary_switch_count=3, last_topic_reason="rules")
    for first in ("nlp", "nlp", "medicine"):
        OmniAgent._record_active_slices(agent, session, [slice_(first), slice_("finance")])
    topic = OmniAgent._topic_meta(agent, session)
    assert topic["active_glossary_preset"] == "medicine_core_1k"
    assert topic["active_domain"] == "medicine"
    assert topic["confidence"] == 0.7
    assert topic["switch_count"] == 1
    assert topic["router"] == "language_model"
    assert topic["top_slices"] == [
        {"domain": "medicine", "preset": "medicine_core_1k", "probability": 0.7},
        {"domain": "finance", "preset": "finance_core_1k", "probability": 0.0},
    ]


def _router_agent(scorer, sessions):
    import asyncio
    from types import SimpleNamespace

    from framework.agents.omni import OmniAgent

    agent = SimpleNamespace(
        config=SimpleNamespace(router_context_switch_rate=1e-3, router_generated_target_window_chunks=4),
        context_similarity=scorer,
        sessions={s.session_id: s for s in sessions},
        _router_wakeup=asyncio.Event(),
        _domain_probe_refresh_sec=lambda session: 5.0,
    )
    agent._due_context_sessions = lambda batch, now: OmniAgent._due_context_sessions(agent, batch, now)
    return agent


def _router_session(session_id, text, segment):
    from types import SimpleNamespace

    return SimpleNamespace(
        session_id=session_id, router_text_window=text, router_text_source="generated_target",
        auto_glossary_enabled=True, last_context_similarity_scores={}, last_context_similarity_text="",
        last_context_similarity_at_s=0.0, last_context_similarity_s=None, last_context_similarity_cached=False,
        router_posterior={}, router_posterior_segment=-1, segment_idx=segment, language_pair="English -> Chinese",
    )


class _FakeScorer:
    enabled = True
    prototypes = {"nlp": "", "medicine": ""}

    def __init__(self):
        self.calls = []

    async def score_batch(self, texts, *, allowed_domains):
        self.calls.append(list(texts))
        return [{"nlp": 0.2, "medicine": 0.8} for _ in texts]


def test_chunks_read_cached_scores_and_the_worker_refreshes_them():
    import asyncio

    from framework.agents.omni import OmniAgent

    scorer = _FakeScorer()
    session = _router_session("s1", "癌症治疗", segment=6)
    agent = _router_agent(scorer, [session])
    out = asyncio.run(OmniAgent._context_similarity_scores_batch(agent, [session]))
    assert out == [{}] and scorer.calls == [] and agent._router_wakeup.is_set()

    asyncio.run(OmniAgent._refresh_context_scores(agent))
    assert scorer.calls == [["癌症治疗"]]
    assert session.router_posterior_segment == 6
    expected = filter_step({}, {"nlp": 0.2, "medicine": 0.8}, switch_rate=1e-3, evidence_weight=1.0)
    assert session.last_context_similarity_scores == expected

    # Same text: nothing is due, so the worker does not score again.
    asyncio.run(OmniAgent._refresh_context_scores(agent))
    assert len(scorer.calls) == 1
