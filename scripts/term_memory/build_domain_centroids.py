#!/usr/bin/env python3
"""Build audio-router domain centroids from MaxSim text indexes.

Each active working-glossary preset already has a MaxSim text index whose
``text_embs`` live in the same retrieval space as the speech-side query
embeddings. This tool computes one normalized centroid per preset and records
the centroid paths in manifest ``preset_meta``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.agents.term_memory.domain_taxonomy import domain_for_preset  # noqa: E402
from framework.agents.term_memory.manifest import TermMemoryManifest  # noqa: E402


def _load_index_embeddings(index_path: str, device: str):
    import torch  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433

    data = torch.load(index_path, map_location=device)
    if "text_embs" not in data:
        raise RuntimeError(f"index has no text_embs: {index_path}")
    text_embs = data["text_embs"].float()
    if text_embs.ndim == 3:
        text_embs = text_embs.mean(dim=1)
    if text_embs.ndim != 2:
        raise RuntimeError(f"expected 2D text_embs, got shape={tuple(text_embs.shape)}")
    text_embs = F.normalize(text_embs, p=2, dim=-1)
    centroid = F.normalize(text_embs.mean(dim=0), p=2, dim=-1).detach().cpu()
    return centroid, int(text_embs.shape[0]), int(text_embs.shape[-1])


def _preset_list(raw: str, manifest: TermMemoryManifest) -> List[str]:
    presets = [item.strip() for item in (raw or "").split(",") if item.strip()]
    return presets or manifest.preset_ids()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="term-memory manifest JSON")
    ap.add_argument("--out-dir", required=True, help="directory for <preset>.pt centroid files")
    ap.add_argument("--presets", default="", help="comma-separated preset ids; default = all manifest presets")
    ap.add_argument("--target-lang", default="zh", help="target language code or manifest lang key")
    ap.add_argument("--device", default="cpu", help="torch load/compute device, e.g. cpu or cuda:0")
    ap.add_argument("--update-manifest", action="store_true", help="write centroid metadata back to a manifest JSON")
    ap.add_argument("--manifest-out", default="", help="output manifest path; default overwrites --manifest")
    args = ap.parse_args()

    manifest_path = Path(os.path.expandvars(args.manifest))
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = TermMemoryManifest.from_dict(raw, base_dir=manifest_path.parent, path=str(manifest_path))
    out_dir = Path(os.path.expandvars(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: WPS433

    updated_meta: Dict[str, Dict[str, Any]] = dict(raw.get("preset_meta") or raw.get("slice_meta") or {})
    for preset_id in _preset_list(args.presets, manifest):
        snapshot = manifest.snapshot_for(preset_id, args.target_lang)
        if snapshot is None:
            print(f"[centroid] skip {preset_id}: no snapshot for {args.target_lang}", file=sys.stderr)
            continue
        index_path = snapshot.index_path("maxsim")
        if not index_path or not Path(index_path).is_file():
            print(f"[centroid] skip {preset_id}: missing maxsim index {index_path}", file=sys.stderr)
            continue

        centroid, num_terms, dim = _load_index_embeddings(index_path, args.device)
        meta = manifest.meta_for_preset(preset_id)
        domain_id = str(meta.get("domain_id") or meta.get("domain") or domain_for_preset(preset_id))
        payload = {
            "preset_id": preset_id,
            "domain_id": domain_id,
            "centroid": centroid,
            "num_terms": num_terms,
            "source_index": index_path,
            "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "embedding_dim": dim,
        }
        centroid_path = out_dir / f"{preset_id}.pt"
        torch.save(payload, str(centroid_path))
        print(f"[centroid] {preset_id}: terms={num_terms} dim={dim} -> {centroid_path}")

        meta_out = dict(updated_meta.get(preset_id) or {})
        meta_out.update(
            {
                "preset_id": preset_id,
                "domain_id": domain_id,
                "fallback_preset_id": meta_out.get("fallback_preset_id") or ("common_10k" if preset_id != "common_10k" else ""),
                "term_count": int(meta_out.get("term_count") or meta_out.get("terms") or snapshot.num_terms or num_terms),
                "maxsim_index_path": index_path,
                "centroid_path": str(centroid_path),
                "enabled_for_auto_router": meta_out.get("enabled_for_auto_router", True),
            }
        )
        updated_meta[preset_id] = meta_out

    if args.update_manifest:
        raw["preset_meta"] = updated_meta
        out_path = Path(os.path.expandvars(args.manifest_out)) if args.manifest_out else manifest_path
        _atomic_write_json(out_path, raw)
        print(f"[centroid] wrote manifest metadata: {out_path}")


if __name__ == "__main__":
    main()
