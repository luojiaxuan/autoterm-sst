"""Check that base + recovered LoRA reproduces a merged checkpoint under vLLM.

Two OpenAI-compatible vLLM servers are compared on the same prompts. The merged
server's greedy translation of each prompt becomes the target; both servers then
score that target with prompt log-probabilities, and the script reports the mean
and maximum absolute difference of the per-token log-probabilities and the share
of prompts whose greedy outputs are identical.

Modes:
  generate  --url merged server, writes greedy targets and their token log-probs
  compare   --url multi-LoRA server and --model adapter name, reads the targets
"""

import argparse
import json
from pathlib import Path

import requests

SYSTEM = "You are a professional simultaneous interpreter. Translate the English input into {lang}."
LANG = {"zh": "Chinese", "de": "German", "ja": "Japanese"}


def chat(url, model, lang, text, max_tokens):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(lang=LANG[lang])},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def target_logprobs(url, model, lang, text, target):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(lang=LANG[lang])},
            {"role": "user", "content": text},
            {"role": "assistant", "content": target},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    pl = r.json()["prompt_logprobs"]
    return [next(iter(x.values()))["logprob"] if x else None for x in pl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "compare"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", choices=sorted(LANG), required=True)
    ap.add_argument("--sentences", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sents = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()][: args.n]

    if args.mode == "generate":
        rows = []
        for s in sents:
            t = chat(args.url, args.model, args.lang, s, args.max_tokens)
            rows.append({"source": s, "target": t, "logprobs": target_logprobs(args.url, args.model, args.lang, s, t)})
        args.targets.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    rows = json.loads(args.targets.read_text(encoding="utf-8"))
    diffs, same = [], 0
    for row in rows:
        t = chat(args.url, args.model, args.lang, row["source"], args.max_tokens)
        same += int(t == row["target"])
        lp = target_logprobs(args.url, args.model, args.lang, row["source"], row["target"])
        assert len(lp) == len(row["logprobs"])
        diffs += [abs(a - b) for a, b in zip(lp, row["logprobs"]) if a is not None and b is not None]
    result = {
        "lang": args.lang,
        "prompts": len(rows),
        "identical_greedy_outputs": same,
        "mean_abs_logprob_diff": sum(diffs) / len(diffs),
        "max_abs_logprob_diff": max(diffs),
        "tokens": len(diffs),
    }
    print(json.dumps(result), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
