# Internal Operations Notes

Internal-only pointers moved out of the public `README.md` (2026-07-08). These
paths refer to lab machines and are meaningless outside the group.

## Source of truth

- Code and lightweight project records: GitHub
  `git@github.com:luojiaxuan/autoterm-sst.git`, branch `main` (renamed from rasst-demo/framework on 2026-07-09).
- AutoTerm progress and eval summaries live in Git docs:
  `docs/adaptive_working_glossary_eval.md`,
  `docs/auto_glossary_mixed_switch_20260707.md`, and
  `docs/auto_glossary_routing_probe_20260707.md`. The aligned-window
  AutoTerm-10k vs merged-40k xCOMET-lite sensitivity and its local artifact
  checksums are in `docs/autoterm_40k_xcomet_eval_20260710.md`. The provisional
  10-domain AutoTerm top-4/40k vs raw merged-100k stress result, including the
  negative quality result and router slice-coverage failure, is in
  `docs/autoterm_100k_stress_eval_20260710.md`; its machine-readable lightweight
  record is `runtime/eval_20260710/autoterm_100k_3talk_summary.json`.
- Lightweight ACL 60/60 and ESO medicine MFA TextGrid annotations are tracked
  in Git under `eval/streaming_sst/mfa_alignments/`; that directory records the
  original Taurus paths and SHA-256 checksums. Audio and large derived chunk
  datasets are not vendored.
- Current Taurus local staging for real E2E mixed-router outputs:
  `/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012`.
  These raw JSON/MD outputs are local staging artifacts, not reusable canonical
  datasets yet.
- Reusable datasets, generated glossaries/index bundles, and larger artifacts
  should be uploaded to Hugging Face and recorded here with repo URLs and
  revisions before being treated as canonical.
- Canonical eval data on HF: `gavinlaw/rasst-main-result-data`
  (audio, per-talk inputs, gold glossaries). `glossaries/README.md` there
  documents the fixed-denominator `gs` union recipe; the union-ready compact
  medicine GT is `glossaries/hard_medicine_gt_raw_unique212.json`
  (added in revision `204ba141`, 2026-07-08).

## Cluster paths and environments

- Canonical repo root on Taurus: `/mnt/taurus/home/jiaxuanluo/rasst-demo`.
- Resident real run (Taurus):

  ```bash
  cd /mnt/taurus/home/jiaxuanluo/rasst-demo
  setsid nohup bash scripts/run_taurus_framework_vllm.sh \
    > logs/framework_vllm_live.out 2>&1 < /dev/null &
  curl -s http://127.0.0.1:8011/health | python -m json.tool
  ```

- Conda envs: `spaCyEnv` (vLLM >= 0.13, native Qwen3-Omni multimodal) for the
  real path; `infinisst` for mock mode and the legacy scheduler agent. The
  default `infinisst` env's vLLM 0.9.x loads the checkpoint text-only and
  rejects audio.
- Runtime data root (indexes, manifests, snapshots):
  `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory` — never in the
  repo.
- Retriever root: `RASST_ROOT=/mnt/taurus/data2/jiaxuanluo/RASST`.
- `scripts/slurm_framework_vllm_aries.sh` runs the framework on the aries SLURM
  node.
- Keep large artifacts (checkpoints, logs, recordings, benchmark dumps) out of
  the repo; prefer a runtime root under `/mnt/taurus/data2/jiaxuanluo`. Use
  host-qualified Taurus paths in scripts and docs (see `AGENTS.md`).
