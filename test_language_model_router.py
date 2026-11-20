import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval" / "streaming_sst" / "evidence"))

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
