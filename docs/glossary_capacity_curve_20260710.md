# Glossary capacity curve protocol（2026-07-10）

## Objective

测量固定目标术语在 glossary candidate universe 从 10k 扩展到 100k、500k、
1M 时的检索与翻译质量变化，并确定 merged glossary 出现副作用的容量区间。

本阶段是 controlled capacity stress：四档都包含相同 ACL-238 target entries，
只追加 general-Wikidata distractors。它不能被描述为 `100 topics × 10k`；真正的
topic-union catalog 将在 capacity crossover 确认后独立构建和审计。

## Frozen protocol

- Audio：5 个 ACL talks，固定顺序
  `268 → 367 → 590 → 110 → 117`，总计 468 segments / 3,107.332s。
- Target language：English → Chinese。
- Glossaries：严格前缀嵌套的
  `acl_tagged_gs10k/100k/500k/1m`；每档前 238 entries 完全相同。
- Streaming：latency multiplier 2，30,720 samples / 1.92s per chunk，
  feed sleep 1.6s。
- Retriever：MaxSim，最终 prompt top-k 10，score threshold 0.78，tagged term
  map；四档统一设置 pre-rerank candidate budget = 10，并记录实际 scored
  inventory，避免候选预算随 glossary 规模变化。
- Decoder：同一 zh Qwen3-Omni checkpoint，Hyper H200 单卡 TP=1，固定
  sampling/config。每个 capacity server 只见一张物理 GPU；不同 capacity 档可在
  两张卡上并行，但每档内部不做跨卡切分。
- 每档重启 server，只加载一个 capacity index，避免 1M index 与前面档位在
  retriever cache 中叠加。
- 四档全部报告，不根据结果只保留一个所谓 sweet point。

## Metrics

Headline：

1. MFA time-aligned occurrence-level `TERM_ACC`：technical142 与 raw238。
2. Prompt Precision：保存每个 output event 的完整 prompt references，再按
   reference source key 与当前 MFA-aligned source occurrence window 匹配。
3. Retrieved references per emitted translation chunk；空输出、未持久化 partial 的
   decoder tick 不进入分母。四档另保存 `(start_sample, cursor_samples)` signature，
   用于检查 glossary latency 是否改变 chunk coalescing。
4. BLEU、technical-masked BLEU、raw-masked BLEU。

同时保留 corpus-gold retrieval precision（reference key 是否属于全局 ACL gold
inventory），但它只作较弱的诊断指标，不冒充 time-local Prompt Precision。

xCOMET-lite 不作为本轮 headline 或完成条件。其约 15s aligned-window 评分仅保留为
后续可选的质量补充，等四档主指标完整且确有必要时再运行；30s 版本只作
segmentation sensitivity。

## Budget-aware router implementation

容量曲线给出的 crossover 用来确定检索预算 $B$，而不是直接作为 AutoTerm 的
topic taxonomy。实验分支 `explore/multidomain-routing` 已实现显式
`budgeted_top_slices` 模式：router 对 accumulated generated-target sliding-window
context 与 manifest 中每个 topic description 计算语义相似度，再按分数从高到低
激活总 term count 不超过 $B$ 的 slices。若 slices 均为 10k、$B=100k$，即激活
top-10 slices；随后 MaxSim 在每个激活 slice 内召回，并沿用现有全局 rerank，最终
仍只向 LLM 注入全局 top-10 references。

- 代码状态：GitHub 分支 `explore/multidomain-routing`，核心实现 commit
  `40f71b6`。
- 默认兼容：`slice_selection_mode=hard_top1`；只有显式设置
  `budgeted_top_slices` 才改变线上行为。
- Paper 评测固定使用严格的 `score_prompt_precision.py`：只有保存的 UI references
  与实际 `prompt_reference_count` 完全相等时才评分，并按 MFA source-time window
  计算 Prompt Precision。较弱的 corpus-gold type precision 只作诊断。
- 100-topic catalog：尚在构建和 provenance 审计中；在完成前不把 nested
  distractor glossary 描述为 topic-union catalog。

## Inputs and integrity

Manifest：

```text
/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/manifests/current.json
```

| scale | glossary SHA-256 | MaxSim index SHA-256 |
|---|---|---|
| 10k | `df54826723b33806eb722736b896aabdb84001b9fe7eca6956d44b6781510a42` | `0c3ab712955836deab70d3d617c9e24da33ee3d7c9ce87f6ff446f9d8d1764b5` |
| 100k | `a3999ca008c7e9e6c3b4a5741ac5d388fee608e1e5c7fb19eecb27d531b9b048` | `fa6b84c997444fb984efa7a30fd2abb1f0d3137c4f8cd11f8c5325c1669482b8` |
| 500k | `36fbea2a764c9e7f3a3b5e7616d70ad7d23fadc57557334a8f77b203a5be49e5` | `1a2a5a71659cf3aedec83e95308418afc893d759b6712459af866842b8408fc3` |
| 1M | `2354c0a8be9a614c430cf4afc49ae63a3b52bb970c7c4ebba2cf5b49c990480b` | `60e5af275df460327854070bfffa588c21d4992d19d119256d1fb00b18e20f70` |

ACL inputs：

```text
source_text.txt       aa37f443c0e1c6d23fac0ef285230c29e9a25e7941d45cb796c622a5c15452d1
ref.txt               0b1e32490472b7ff98c8a3d0a83092d10b633283ebc7b4daf1888e3a344f4f72
segments.meta.jsonl   b8c911aab3cde27190e331f7769b353aecff2ff856ffa2fa31a2a0426194b8a8
```

## Artifact status

Hyper00 paused staging（2026-07-10）：

```text
/data02/jaxan/autoterm-capacity-zh-20260710/
```

- Code：Git branch `explore/multidomain-routing` at `82d6394`。
- Integrity：RAG、四个 glossary 和四个 MaxSim index 的 SHA-256 均与上表一致；
  copied Conda runtime file count 与 Taurus source 同为 92,523。
- Smoke：`outputs/smoke-10k` 已完成 60.0s / 28 emitted events，health 显示
  `active_terms=10000`，`tail_gap_samples=0`，保存的 112 条 prompt references 与
  `prompt_reference_count` 完全一致。
- Full run：按用户要求已暂停并释放 GPU。`outputs/full-zh` 的 10k / 100k
  controller 分别在 1,172 / 1,171 batches 时收到 SIGTERM；
  `run_manifest_{a,b}.json` 明确记录
  `TerminationRequested: received signal 15`。由于 raw run JSON 只在一个 preset
  完整结束后原子写入，当前 `runs_{a,b}.json` 均为空，不能把 partial server log
  当成正式 metric 输入。两个 server log、完整启动命令和 exact expected term count
  均已保留；500k / 1M 尚未启动。
- Cleanup：`sglang-omni-jaxan-07110219` / `07110220` 已删除，GPU 2/3 回到
  4 MiB；Hyper00 没有遗留本项目的 GPU container。
- xCOMET-lite staging：`/data02/jaxan/floras_qe_eval/`，NL2G code commit
  `e21e291b2a09a5a854c55b0c01c53ab580692beb`，model revision
  `8d628ebffb4e3f20f53f52f9570d19dee38b9b9a`。Hyper runtime 显式固定
  `huggingface_hub==0.36.0`、`transformers==4.57.1`，CPU checkpoint load 已通过。
  xCOMET 评分按用户要求延期；额外的 `--network none` 检查发现上游
  DeBERTa encoder 仍会尝试 Hub HEAD request，因此当前不能声称 fully offline，
  后续运行前应显式传 local encoder snapshot 或启用 upstream offline mode。
- 一个共享机 failure mode：首次 preflight 后 GPU 0/1 被已有的他人容器恢复任务
  抢占，两个 server 在 health 前因显存不足失败；失败输出隔离在 `outputs/full`，
  不进入正式结果。重新 preflight 后正式任务使用 GPU 2/3。

### Restart note

以后重跑时先重新执行 GPU preflight，再复用
`run_manifest_{a,b}.json` 中保存的 controller 参数。Partial preset 不能续接，必须从
音频开头重跑；建议使用新的 `outputs/full-zh-rerun-<date>/` 和对应 tmp 目录，避免
把新日志 append 到本次暂停日志。等某一 preset 完整写出并通过 validator 后，才可用
`--resume` 跳过该 completed preset。

### B200 completed exploratory run（2026-07-10）

另一条 B200 exploratory run 已完整落盘，但尚未计算任何质量指标：

```text
/data02/jaxan/autoterm-capacity-sweep-20260710/run/
```

- Host / compute：`b200`（`innomatrix-us-adc-smb200-0003`），GPU 0，单卡 TP=1；
  container `sglang-omni-jaxan-07101717`。四个 client 同时串流同一 playlist，分别固定
  `acl_tagged_gs10k/100k/500k/1m`。
- Runtime code：`explore/multidomain-routing` at
  `5a1df512cf7e4c3dcc42b0c017e1051380aff3da`。该 exploratory run 复用一个 server，
  四个 session 各自绑定独立 MaxSim index；它与上面的 frozen final protocol“每档重启
  server”不同，不能把两者混写成同一个 run。
- Streaming：`chunk_samples=30,720`、`chunk_seconds=1.92`、
  `feed_sleep=1.6`、latency multiplier 2、top-k 10、threshold 0.78；5 个 talk
  实际总长 3,107.332s。所有 run 使用同一 block/span playlist，playlist signature
  SHA-256 为
  `4280ffe8fd4d703b0cbeb0d041dd428091d3c68197f31457c4f9ae4bfeb8e9a4`。
- Completion：四个 session 均正常删除后，guard 看到四个 JSON 已落盘并停止 container
  释放 GPU。Container 的最终 `Exited (137)` 来自完成后的强制收尾，不是生成中 OOM；
  server 的 1,724 个 batches 中 `generation_ok != batch_size` 为 0。

| scale | events | prompt refs | refs/event | last cursor / span end | JSON SHA-256 |
|---|---:|---:|---:|---:|---|
| 10k | 1,221 | 4,580 | 3.751 | 49,704,960 / 49,717,305 | `dc694f5c28ee7dfe2c43e55f80248e2f825698618737a92bfe8b74db9580f90f` |
| 100k | 1,213 | 7,445 | 6.138 | 49,704,960 / 49,717,305 | `a2d1aac1b8bff6f115ac2a142442e5200e46856957abfcf2a029c451f026b18a` |
| 500k | 1,214 | 10,825 | 8.917 | 49,704,960 / 49,717,305 | `219122263a9c4d0db66237730c698847ae57b96fa642eec29e154c2f09e6296a` |
| 1M | 1,214 | 11,603 | 9.558 | 49,704,960 / 49,717,305 | `081e45725d3a78bd20cad944ac1a628a1874523c738438f3b69989a31f782199` |

四档的 `event_count == len(records)`、cursor/start 均单调、完整 prompt reference capture
的 mismatch chunks 均为 0；尾部 gap 为 12,345 samples，小于一个 30,720-sample
chunk。并行 continuous batching 导致 event timing signatures 不完全相同，事件数最大
相差 8（0.66%）；后续不得把 event index 当作配对样本，应使用按 source-audio time
对齐的窗口 scorer。当前状态严格为 `complete_unscored`：尚未运行 TERM_ACC、BLEU、
masked BLEU、Prompt Precision 或 xCOMET。

Git 中的轻量完整性清单为
`runtime/eval_20260621/glossary_capacity_full_acl_20260710_integrity.json`。Raw JSON 与完整
reference events 仍只在上述 B200 staging，Hugging Face dataset repo / revision 为
`pending / TBD`；在上传前不能把本地路径当作 canonical artifact。四个 `full_acl_*.log`
与对应 JSON byte-identical，只是 stdout duplicate，不是独立数据版本。

Taurus local staging：

```text
/mnt/data1/jiaxuanluo/rasst_eval/autoterm_capacity_100topics_20260710/
```

Aries earlier staging：

```text
/mnt/data6/jiaxuanluo/autoterm_capacity_curve_20260710/
```

这些目录当前是 local staging，不是 canonical reusable artifact。实验完成后，
轻量 summaries 和命令记录进 Git；raw run JSON/reference events 后续与真正的
100-topic catalog 一起选择 Hugging Face dataset repo，当前 upload status 为
`pending / repo TBD`。

## Multilingual follow-up

现有 nested 1M 只有 9,999 条 ja translation、567,230 条 de translation，并且
只有 en-zh MaxSim index，因此不能直接当作 ja/de 1M baseline。ja/de endpoint
必须等待 source-term-identical、target mappings 完整的 multilingual catalog；
否则测到的是 translation missing / prompt shortfall，而不是 glossary scale。
