#!/usr/bin/env python3
"""Drive an N-way concurrent SimulEval streaming-SST run against ONE remote omni
engine (vLLM or sglang-omni) and emit a merged ``instances.log`` ready for the
canonical FBK StreamLAAL/BLEU scorer.

Concurrency model
-----------------
We shard the source list (default: the 468 ACL6060 segments, audio.yaml order)
round-robin across ``--concurrency`` workers and launch one ``simuleval`` process
per worker, all hitting the same engine endpoint. At any instant ~N streams are
in flight, so the engine sees N concurrent streaming sessions -- the exact load
whose scheduling efficiency we are A/B-ing (sglang-omni vs vLLM, both TP=2).

Each worker streams its segments sequentially in ``--source-segment-size`` ms
chunks via ``remote_omni_agent.py``. SimulEval records per-char source delays and
computation-aware elapsed (which absorbs queueing under load), so the merged log
scores for BLEU / StreamLAAL / StreamLAAL_CA -- StreamLAAL_CA is where the engine
gap shows up as N grows.

The runner is protocol-agnostic: pass ``--source segments.source`` for the
per-segment sweep, or the canonical ``source.list`` (5 talk wavs) for a
whole-talk reference run.

Example (per-segment, N=8, sglang):

  python eval/streaming_sst/run_concurrency.py \
    --engine sglang --base-url http://127.0.0.1:8100 --model-name qwen3-omni \
    --concurrency 8 \
    --data-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments \
    --out-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/runs/sglang_n8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_NUM = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
_TRIPLE_RE = re.compile(rf"^\s*({_NUM})\s+({_NUM})\s+({_NUM})\s*$")


def _score_instances(
    *, fbk_tool: str, mwer_root: str, instances: Path, audio_yaml: Path,
    ref: Path, source: Optional[Path], tokenizer: str, latency_unit: str,
) -> Dict[str, object]:
    """Run FBK stream_laal_term.py directly (no glossary) and parse the metrics."""
    env = dict(os.environ)
    env["MWERSEGMENTER_ROOT"] = mwer_root
    cmd = [
        SPACY_PY, fbk_tool,
        "--simuleval-instances", str(instances),
        "--reference", str(ref),
        "--audio-yaml", str(audio_yaml),
        "--sacrebleu-tokenizer", tokenizer,
        "--latency-unit", latency_unit,
    ]
    if source is not None and source.is_file():
        cmd += ["--source-reference", str(source)]
    # stdin from /dev/null: mwerSegmenter blocks if it inherits a live stdin.
    with open(os.devnull) as devnull:
        p = subprocess.run(cmd, env=env, stdin=devnull, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    bleu = stream_laal = stream_laal_ca = None
    for line in out.splitlines():
        m = _TRIPLE_RE.match(line)
        if m:
            bleu, stream_laal, stream_laal_ca = float(m.group(1)), float(m.group(2)), float(m.group(3))
            break
    return {
        "score_returncode": p.returncode,
        "BLEU": bleu,
        "StreamLAAL": stream_laal,
        "StreamLAAL_CA": stream_laal_ca,
        "score_raw_tail": "\n".join(out.splitlines()[-6:]),
    }

SPACY_PY = "/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python"
REPO_ROOT = "/mnt/taurus/home/jiaxuanluo/rasst-demo"
AGENT_REL = "eval/streaming_sst/remote_omni_agent.py"


def _read_lines(path: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _shard_round_robin(items: List[str], n: int) -> List[List[int]]:
    shards: List[List[int]] = [[] for _ in range(n)]
    for idx in range(len(items)):
        shards[idx % n].append(idx)
    return [s for s in shards if s]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model-name", default="qwen3-omni")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out-dir", required=True)

    # Either a prepared data dir (segments.source/segments.target) or explicit files.
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N source lines (smoke)")

    ap.add_argument("--source-segment-size", type=int, default=1920)
    ap.add_argument("--min-start-sec", type=float, default=0.96)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--keep-cache-chunks", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--sacrebleu-tokenizer", default="zh")
    ap.add_argument("--latency-unit", choices=["char", "word"], default="char")
    ap.add_argument("--source-lang", default="English")
    ap.add_argument("--target-lang", default="Chinese")

    ap.add_argument("--simuleval-bin", default=f"{os.path.dirname(SPACY_PY)}/simuleval")
    ap.add_argument("--seg-tmp-base", default="/dev/shm/remote_omni_eval")

    # Optional inline StreamLAAL/BLEU scoring (FBK stream_laal_term.py, no glossary).
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fbk-tool", default="/mnt/taurus/home/jiaxuanluo/FBK-fairseq/"
                    "examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py")
    ap.add_argument("--mwer-root", default="/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    args = ap.parse_args()

    if args.data_dir:
        src_path = Path(args.data_dir) / "segments.source"
        tgt_path = Path(args.data_dir) / "segments.target"
    else:
        src_path = Path(args.source)
        tgt_path = Path(args.target)
    if not src_path.is_file() or not tgt_path.is_file():
        print(f"[runner] missing source/target: {src_path} {tgt_path}", file=sys.stderr)
        return 2

    sources = _read_lines(src_path)
    targets = _read_lines(tgt_path)
    if len(sources) != len(targets):
        print(f"[runner] length mismatch source={len(sources)} target={len(targets)}", file=sys.stderr)
        return 2
    if args.limit and args.limit < len(sources):
        sources = sources[: args.limit]
        targets = targets[: args.limit]

    n = max(1, min(args.concurrency, len(sources)))
    if n != args.concurrency:
        print(f"[runner] capping concurrency {args.concurrency} -> {n} (only {len(sources)} sources)")

    out_dir = Path(args.out_dir)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    shards = _shard_round_robin(sources, n)
    n = len(shards)

    procs = []
    logs = []
    for wi, idxs in enumerate(shards):
        w_src = work_dir / f"w{wi}.source"
        w_tgt = work_dir / f"w{wi}.target"
        w_src.write_text("\n".join(sources[i] for i in idxs) + "\n", encoding="utf-8")
        w_tgt.write_text("\n".join(targets[i] for i in idxs) + "\n", encoding="utf-8")
        w_out = work_dir / f"w{wi}"
        seg_tmp = f"{args.seg_tmp_base}/w{wi}"
        os.makedirs(seg_tmp, exist_ok=True)

        cmd = [
            args.simuleval_bin,
            "--agent", AGENT_REL,
            "--source", str(w_src),
            "--target", str(w_tgt),
            "--output", str(w_out),
            "--source-segment-size", str(args.source_segment_size),
            "--quality-metrics", "BLEU",
            "--eval-latency-unit", args.latency_unit,
            "--sacrebleu-tokenizer", args.sacrebleu_tokenizer,
            "--remote-engine", args.engine,
            "--remote-base-url", args.base_url,
            "--remote-model-name", args.model_name,
            "--source-lang", args.source_lang,
            "--target-lang", args.target_lang,
            "--min-start-sec", str(args.min_start_sec),
            "--max-new-tokens", str(args.max_new_tokens),
            "--keep-cache-chunks", str(args.keep_cache_chunks),
            "--request-timeout", str(args.request_timeout),
            "--segment-tmp-dir", seg_tmp,
            "--worker-id", str(wi),
        ]
        log_f = open(work_dir / f"w{wi}.log", "w", encoding="utf-8")
        logs.append(log_f)
        procs.append(
            subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
        )

    print(f"[runner] engine={args.engine} base_url={args.base_url} concurrency={n} "
          f"segments={len(sources)} seg_size_ms={args.source_segment_size}")
    t0 = time.time()
    rcs = [p.wait() for p in procs]
    wall = time.time() - t0
    for f in logs:
        f.close()

    # Merge per-worker instances.log -> out_dir/instances.log
    merged = out_dir / "instances.log"
    n_inst = 0
    with merged.open("w", encoding="utf-8") as out:
        for wi in range(n):
            inst = work_dir / f"w{wi}" / "instances.log"
            if not inst.is_file():
                print(f"[runner] WARN missing instances.log for worker {wi} (rc={rcs[wi]})")
                continue
            for line in inst.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    n_inst += 1

    summary = {
        "engine": args.engine,
        "base_url": args.base_url,
        "concurrency": n,
        "segments_total": len(sources),
        "instances_merged": n_inst,
        "worker_returncodes": rcs,
        "wall_clock_sec": round(wall, 2),
        "segments_per_sec": round(len(sources) / wall, 4) if wall > 0 else 0.0,
        "instances_log": str(merged),
    }

    if args.score:
        if n_inst != len(sources):
            summary["score_skipped"] = f"instances {n_inst} != segments {len(sources)}"
        else:
            data_root = Path(args.data_dir) if args.data_dir else src_path.parent
            audio_yaml = data_root / "audio.yaml"
            ref = data_root / "ref.txt"
            source_txt = data_root / "source_text.txt"
            if args.limit:
                summary["score_skipped"] = "scoring needs the full set; --limit was used"
            elif not (audio_yaml.is_file() and ref.is_file()):
                summary["score_skipped"] = f"missing {audio_yaml} or {ref}"
            else:
                summary.update(_score_instances(
                    fbk_tool=args.fbk_tool, mwer_root=args.mwer_root, instances=merged,
                    audio_yaml=audio_yaml, ref=ref,
                    source=source_txt if source_txt.is_file() else None,
                    tokenizer=args.sacrebleu_tokenizer, latency_unit=args.latency_unit,
                ))

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[runner] " + json.dumps(summary))
    if any(rc != 0 for rc in rcs):
        print("[runner] WARN: one or more workers returned non-zero; check work/w*.log", file=sys.stderr)
    if n_inst != len(sources):
        print(f"[runner] WARN: merged instances {n_inst} != segments {len(sources)} "
              f"(scorer requires the full set to match audio.yaml)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
