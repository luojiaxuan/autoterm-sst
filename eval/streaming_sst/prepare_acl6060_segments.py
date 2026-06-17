#!/usr/bin/env python3
"""Prepare ACL 60-60 segment wavs + SimulEval source/target lists.

The ACL 60-60 dev set ships as a SimulEval-style manifest: ``audio.yaml`` lists
468 utterance segments (``wav`` + ``offset`` + ``duration``) over 5 ACL-2022
talks, with line-aligned ``ref.txt`` (zh) and ``source_text.txt`` (en).

For a SimulEval-native run we want one independent audio file per segment, in
manifest order, so ``--source`` is a flat list of wavs that aligns 1:1 with the
references. This script crops each segment to a 16 kHz mono wav and emits an
``out_dir`` that is self-consistent for both the SimulEval run and the canonical
``offline_streamlaal_eval.py`` scorer:

  out_dir/
    seg/00000_2022.acl-long.268.wav ...   # cropped 16 kHz mono segments
    segments.source                        # abs seg wav paths (SimulEval --source)
    segments.target                        # zh references     (SimulEval --target)
    audio.yaml                             # {wav: seg_wav, offset: 0, duration}
    ref.txt                                # zh references (scorer --ref-file)
    source_text.txt                        # en source text (scorer --source-file)
    segments.meta.jsonl                    # per-segment provenance

Usage (defaults point at the canonical RASST en->zh ACL 60-60 set):

  python eval/streaming_sst/prepare_acl6060_segments.py \
    --out-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments

  # smoke subset (first 8 segments):
  python eval/streaming_sst/prepare_acl6060_segments.py --limit 8 \
    --out-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import yaml

TARGET_SR = 16000

RASST_ROOT = "/mnt/taurus/data2/jiaxuanluo/RASST"
ACL_ZH_DIR = f"{RASST_ROOT}/data/main_result/inputs/acl_zh"


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        import librosa  # noqa: WPS433

        return librosa.resample(
            audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr
        ).astype(np.float32)


def _load_talk(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasst-root", default=RASST_ROOT, help="root the audio.yaml wav paths are relative to")
    ap.add_argument("--audio-yaml", default=f"{ACL_ZH_DIR}/audio.yaml")
    ap.add_argument("--ref-file", default=f"{ACL_ZH_DIR}/ref.txt")
    ap.add_argument("--source-text-file", default=f"{ACL_ZH_DIR}/source_text.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N segments (0 = all)")
    ap.add_argument("--talk", default="", help="keep only segments whose wav basename contains this (e.g. 590)")
    args = ap.parse_args()

    rasst_root = Path(args.rasst_root)
    entries: List[Dict[str, Any]] = yaml.safe_load(Path(args.audio_yaml).read_text(encoding="utf-8"))
    refs = Path(args.ref_file).read_text(encoding="utf-8").splitlines()
    srcs = Path(args.source_text_file).read_text(encoding="utf-8").splitlines()
    if not (len(entries) == len(refs) == len(srcs)):
        raise SystemExit(
            f"length mismatch: audio.yaml={len(entries)} ref={len(refs)} source={len(srcs)}"
        )

    out_dir = Path(args.out_dir)
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    talk_cache: Dict[str, tuple[np.ndarray, int]] = {}
    kept_audio: List[Dict[str, Any]] = []
    kept_refs: List[str] = []
    kept_srcs: List[str] = []
    source_lines: List[str] = []
    meta_lines: List[str] = []

    n_written = 0
    for idx, entry in enumerate(entries):
        wav_rel = str(entry["wav"]).strip()
        talk_id = Path(wav_rel).stem  # e.g. 2022.acl-long.590
        if args.talk and args.talk not in Path(wav_rel).name:
            continue
        if args.limit and n_written >= args.limit:
            break

        if wav_rel not in talk_cache:
            wav_path = Path(wav_rel)
            if not wav_path.is_absolute():
                wav_path = rasst_root / wav_rel
            talk_cache[wav_rel] = _load_talk(wav_path)
        talk_audio, talk_sr = talk_cache[wav_rel]

        offset = float(entry.get("offset", 0.0))
        duration = float(entry.get("duration", 0.0))
        start = int(round(offset * talk_sr))
        end = int(round((offset + duration) * talk_sr))
        clip = talk_audio[max(0, start):max(0, end)]
        clip = _resample(clip, talk_sr, TARGET_SR)

        seg_name = f"{n_written:05d}_{talk_id}.wav"
        # abspath (not resolve) so we keep the host-qualified Taurus path and do
        # not collapse the /mnt/taurus -> /mnt symlink.
        seg_path = Path(os.path.abspath(seg_dir / seg_name))
        sf.write(str(seg_path), clip, TARGET_SR)

        seg_dur = round(len(clip) / TARGET_SR, 6)
        kept_audio.append({"wav": str(seg_path), "offset": 0.0, "duration": seg_dur})
        kept_refs.append(refs[idx])
        kept_srcs.append(srcs[idx])
        source_lines.append(str(seg_path))
        meta_lines.append(
            json.dumps(
                {
                    "index": n_written,
                    "orig_index": idx,
                    "talk": talk_id,
                    "offset": offset,
                    "duration": duration,
                    "seg_wav": str(seg_path),
                    "seg_duration": seg_dur,
                },
                ensure_ascii=False,
            )
        )
        n_written += 1

    if n_written == 0:
        raise SystemExit("no segments selected (check --talk/--limit)")

    (out_dir / "audio.yaml").write_text(
        yaml.safe_dump(kept_audio, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "ref.txt").write_text("\n".join(kept_refs) + "\n", encoding="utf-8")
    (out_dir / "source_text.txt").write_text("\n".join(kept_srcs) + "\n", encoding="utf-8")
    (out_dir / "segments.source").write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    (out_dir / "segments.target").write_text("\n".join(kept_refs) + "\n", encoding="utf-8")
    (out_dir / "segments.meta.jsonl").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    talks = sorted({Path(e["wav"]).name.split("_", 1)[-1] for e in kept_audio})
    total_sec = round(sum(e["duration"] for e in kept_audio), 1)
    print(
        f"prepared segments={n_written} talks={len(talk_cache)} total_audio_sec={total_sec} "
        f"out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
