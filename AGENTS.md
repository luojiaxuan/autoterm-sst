# RASST-Demo Agent Instructions

This repository is the demo-system continuation of the previous InfiniSST
EMNLP System Demonstration draft. The current working name is `rasst-demo`.
Treat it as a live demo project for retrieval-aware, terminology-sensitive
streaming speech translation, not as a generic InfiniSST archive.

## Communication

- Prefer English for user-facing explanations unless Chinese is clearly faster,
  more precise, or requested.
- If the user's English or mixed English/Chinese wording is unnatural,
  ambiguous, or unprofessional, include a concise `Better English:` correction.
- Keep language feedback separate from engineering work.
- If an instruction is ambiguous enough to affect paths, code, experiments, or
  paper claims, ask a targeted clarification question before acting.

## Canonical Paths

- Repo root: `/mnt/taurus/home/jiaxuanluo/rasst-demo`
- Demo paper draft: `/mnt/taurus/home/jiaxuanluo/rasst-demo/demo_paper_emnlp`
- Main RASST reference: `/mnt/data2/jiaxuanluo/RASST`
- User-expected RASST home path, if later symlinked or moved:
  `/mnt/taurus/home/jiaxuanluo/RASST`
- Optional RL/post-training reference: `/mnt/taurus/home/jiaxuanluo/RAPO`

Use host-qualified Taurus paths in scripts, docs, and reports. Do not write new
commands using bare `/home/...` or Aries-only `/mnt/aries/...` paths unless the
task explicitly concerns Aries storage.

Large generated files, model checkpoints, logs, recordings, and benchmark dumps
should stay out of the repo unless the user explicitly asks to commit or package
them. Prefer a separate runtime/data root under `/mnt/taurus/data2/jiaxuanluo`
for new large artifacts.

## Current Demo Scope

The baseline system is an InfiniSST-style multi-user simultaneous speech
translation demo with:

- web and Electron frontends;
- microphone, file, YouTube/web, and desktop system-audio input paths;
- a scheduler/back-end split for concurrent sessions;
- prefill/decode batching, KV-cache reuse, attention-sink/sliding-window
  inference, and paged attention support;
- prior 32-sessions-per-GPU positioning from the old draft.

For a new demo-track submission, do not rely on "long speech translation with an
LLM" or "32 sessions per GPU" as the main novelty. Treat those as baseline
system strengths. The refreshed claim should focus on term-dense vertical-domain
translation: domain glossary retrieval, live terminology control, term-recall
diagnostics, and a demo experience that makes terminology corrections visible.

## Extension Direction

Prioritize a practical RASST-Demo story:

```text
streaming speech -> candidate terms/retrieved evidence -> terminology-aware
translation -> live UI diagnostics for latency, term recall, and corrections
```

Good demo features include:

- domain or glossary selection for medical, legal, academic, or financial
  speech;
- live display of retrieved terms/evidence next to the translation;
- user-visible term correction or acceptance controls;
- per-session diagnostics for latency, throughput, term recall, and false-copy
  behavior;
- comparison between plain InfiniSST and terminology-aware/RAG modes.

Avoid making RL the core demo requirement unless experiments already show a
clear user-facing win. RAPO-style RL/post-training is optional and should be
framed as a low-latency 4B student or adaptive policy extension. A 30B model can
remain the stronger teacher, reference model, or quality baseline when that is
more compelling for a demo.

## Relationship To RAPO

RAPO is a post-training reference, not the main identity of this demo repo. If
used, frame RAPO-style work as optional retrieval-aware or terminology-aware
post-training for a smaller student model.

The Speech LLM is the agent. Terminology retrieval, glossary lookup, or domain
evidence is a tool/evidence source. DPO, SimPO, GRPO, and OPD are training
backends; they are not the demo contribution by themselves.

For a compute-conscious extension, prefer:

```text
30B RASST/Qwen-Omni system as demo engine or teacher
4B student as low-latency optional variant
offline teacher/rubric feedback for term recall, false-copy behavior, latency,
and correction quality
```

Do not plan full 30B online GRPO unless the user explicitly changes the scope.

## Paper Framing

The paper in `demo_paper_emnlp` is a starting point, not a final submission.
When revising it:

- update the title away from pure high-throughput InfiniSST scheduling if the
  demo is now RASST/terminology centered;
- keep the 32-session scheduling result as a systems capability, not the whole
  contribution;
- add concrete evaluation, because EMNLP demo submissions can be desk rejected
  if they report no evaluation;
- include screenshots or diagrams of the actual updated demo;
- include a live demo link or installable package plan;
- keep the submission within the current demo-track page limit and style rules;
- state licensing and deployment constraints clearly.

The likely contribution shape is:

1. an interactive streaming speech translation demo for term-dense domains;
2. retrieval-aware terminology assistance integrated into a low-latency SST UI;
3. system evidence showing the quality-latency-terminology trade-off under
   concurrent use.

## Engineering Rules

- Read the existing code before editing; this repo contains a web server,
  Electron app, model wrappers, scheduler, and paper draft.
- Keep changes scoped to the demo path unless the user explicitly asks to sync
  with another repo.
- Prefer small smoke tests before reporting success.
- After committing material code, config, docs, evaluation summaries, or
  progress records, push the branch to the GitHub remote before reporting the
  task complete unless the user explicitly says not to push. Local-only commits
  are not a source of truth for this project.
- Do not commit generated logs, caches, model weights, downloaded checkpoints,
  large media, or temporary LaTeX build products.
- If borrowing code from InfiniSST, RASST, or RAPO, record the source path and
  keep the demo-facing behavior clear.

## Useful Entry Points

- Backend API: `serve/api.py`
- Inference engine: `serve/inference_engine.py`
- Scheduler: `serve/scheduler.py`
- Main agents: `agents/infinisst.py`, `agents/infinisst_fast.py`,
  `agents/infinisst_faster.py`
- Web/static frontend: `serve/static/`
- Electron frontend: `electron/`
- Old demo paper: `demo_paper_emnlp/latex/`
- Integration tests: `tests/` (e.g. `tests/test_integrated_system.py`, `tests/test_scheduler_system.py`)

When evaluating submission readiness, inspect both the runnable demo and the
paper draft. The submission should be judged by what a reviewer can see in the
live demo/video, not only by model-training claims.
