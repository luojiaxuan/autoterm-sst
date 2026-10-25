"""Online Bayesian filtering over a latent domain, in audio time, with an explicit unknown state.

State space: K catalog domains plus an "unknown" state u. Between two observations separated by
dt seconds of audio the domain persists with probability exp(-hazard * dt); otherwise it jumps to a
state drawn from the prior pi (a continuous-time Markov chain, so hazard = 1 / expected segment
length and no wall-clock timers are involved).

Emission: the routing context embedding e_t is scored against each slice prototype mu_k by cosine.
The unknown state scores a fixed reference similarity c_u, i.e. it wins exactly when no prototype
is closer than c_u. With temperature kappa this is a von Mises-Fisher likelihood up to a shared
constant: log p(e | k) = kappa * cos(e, mu_k), log p(e | u) = kappa * c_u. Consecutive contexts of
w chunks overlap in w - 1 chunks, so each observation is tempered by 1 / w (composite likelihood),
which keeps each emitted chunk counted once.

Slice selection is a Bayes decision with a per-slice cost: slice k is loaded iff its posterior
exceeds the cost ratio tau, capped at the memory budget B (highest posterior first).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouterParams:
    kappa: float
    hazard: float
    c_unknown: float
    tau: float
    budget: int
    window: int


def filter_posteriors(sim, cursor_s, observed, p, prior=None):
    """sim: (T, K) cosine to prototypes; observed: (T,) bool (window carries new text)."""
    t_len, k = sim.shape
    log_lik = np.concatenate([sim, np.full((t_len, 1), p.c_unknown)], axis=1) * (p.kappa / p.window)
    pi = np.full(k + 1, 1.0 / (k + 1)) if prior is None else prior
    post = np.empty((t_len, k + 1))
    belief = pi.copy()
    last_t = cursor_s[0]
    for t in range(t_len):
        stay = np.exp(-p.hazard * max(cursor_s[t] - last_t, 0.0))
        belief = stay * belief + (1.0 - stay) * pi
        last_t = cursor_s[t]
        if observed[t]:
            lw = np.log(belief) + log_lik[t]
            lw -= lw.max()
            belief = np.exp(lw)
            belief /= belief.sum()
        post[t] = belief
    return post


def select_slices(post, p):
    """Returns a (T, K) bool mask of loaded slices; the unknown state never loads a slice."""
    dom = post[:, :-1]
    order = np.argsort(-dom, axis=1)[:, : p.budget]
    mask = np.zeros_like(dom, dtype=bool)
    rows = np.arange(len(dom))[:, None]
    mask[rows, order] = np.take_along_axis(dom, order, axis=1) >= p.tau
    return mask
