"""Recover a LoRA adapter from a merged fine-tuned checkpoint and its base model.

The RASST language models were trained as rank-32 LoRA on every linear layer of
the Qwen3-Omni thinker and then merged into full weights. For each changed weight,
the difference to the base model therefore has rank at most 32 plus the rounding
noise of bf16 storage. The top singular directions of that difference give the
adapter: lora_B = U_r * S_r and lora_A = V_r^T, with lora_alpha = r so the scale is
one. Expert weights stay per expert, the 2D layout that vLLM loads with
is_3d_lora_weight false. The script reports, per module type, the share of the
difference's energy that the adapter keeps.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET = re.compile(
    r"^thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--base-model-id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    bidx = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    midx = json.loads((args.merged / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(bidx) == set(midx)

    by_shard = defaultdict(list)
    for key, shard in midx.items():
        if TARGET.match(key):
            by_shard[shard].append(key)
    tensors, kept = {}, defaultdict(lambda: [0.0, 0.0])
    r = args.rank
    for shard, keys in sorted(by_shard.items()):
        with safe_open(args.merged / shard, framework="pt") as fm:
            merged = {k: fm.get_tensor(k) for k in keys}
        base = {}
        for k in keys:
            with safe_open(args.base / bidx[k], framework="pt") as fb:
                base[k] = fb.get_tensor(k)
        # Batch weights of the same shape so the SVD runs once per group.
        groups = defaultdict(list)
        for k in keys:
            groups[tuple(merged[k].shape)].append(k)
        for shape, ks in groups.items():
            delta = torch.stack([merged[k].float() - base[k].float() for k in ks])
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            b = (u[:, :, :r] * s[:, None, :r]).to(torch.bfloat16)
            a = vh[:, :r, :].to(torch.bfloat16)
            energy = (s**2).sum(-1)
            top = (s[:, :r] ** 2).sum(-1)
            for i, k in enumerate(ks):
                module = k[: -len(".weight")]
                tensors[f"base_model.model.{module}.lora_A.weight"] = a[i].contiguous()
                tensors[f"base_model.model.{module}.lora_B.weight"] = b[i].contiguous()
                kind = module.rsplit(".", 1)[-1]
                kept[kind][0] += float(top[i])
                kept[kind][1] += float(energy[i])
        print(json.dumps({"shard": shard, "modules": len(keys)}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base_model_id,
        "r": r,
        "lora_alpha": r,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "rank": r,
        "modules": len(tensors) // 2,
        "energy_kept": {k: round(v[0] / v[1], 4) for k, v in sorted(kept.items())},
    }
    (args.out / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
