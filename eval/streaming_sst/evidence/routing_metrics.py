"""Ranking metrics shared by the routing replays."""

import numpy as np

DOMAINS = ["nlp", "medicine", "education", "finance", "legal", "environment", "entertainment", "science", "sports", "art"]


def ranking_metrics(scores, names, gold, valid, ks):
    order = np.argsort(-scores, axis=1)
    gold_idx = np.array([names.index(g) for g in gold])
    rank = np.argmax(order == gold_idx[:, None], axis=1)
    top1 = order[:, 0]
    flips = int((top1[1:] != top1[:-1]).sum())
    out = {"top1_acc": round(float((rank[valid] == 0).mean()), 4), "top1_changes": flips}
    for k in ks:
        sets = [frozenset(row[:k]) for row in order]
        changes = sum(a != b for a, b in zip(sets[1:], sets[:-1]))
        out[f"cover@{k}"] = round(float((rank[valid] < k).mean()), 4)
        out[f"set_changes@{k}"] = int(changes)
    return out
