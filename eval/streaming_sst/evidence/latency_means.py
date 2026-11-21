"""Mean per-chunk compute time of single-session latency runs.

Reads the concurrency_full_pipeline.py outputs of each configuration, keeps the one-session level,
drops the first three chunks as the client does, and reports the mean retrieval time (speech
encoding, routing and term retrieval), the mean generation time, and the real-time factor: their
sum over the 1.92-second chunk length.
"""

import argparse
import json
from pathlib import Path

import numpy as np

CHUNK_S = 1.92
WARMUP_CHUNKS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_JSON")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = {"chunk_s": CHUNK_S, "warmup_chunks_dropped": WARMUP_CHUNKS, "configs": {}}
    for spec in args.run:
        name, path = spec.split("=", 1)
        payload = json.load(open(path, encoding="utf-8"))
        level = next(lv for lv in payload["levels"] if lv["sessions"] == 1)
        events = [e for s in level["raw_sessions"] for e in s["events"] if e["chunk"] >= WARMUP_CHUNKS]
        retrieve = np.array([e["retrieve_s"] for e in events if e["retrieve_s"] is not None])
        generation = np.array([e["elapsed_s"] for e in events if e["elapsed_s"] is not None])
        result["configs"][name] = {
            "chunks": len(events), "retrieve_chunks": len(retrieve), "generation_chunks": len(generation),
            "retrieve_mean_s": round(float(retrieve.mean()), 4),
            "generation_mean_s": round(float(generation.mean()), 4),
            "rtf": round(float((retrieve.mean() + generation.mean()) / CHUNK_S), 4),
        }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
