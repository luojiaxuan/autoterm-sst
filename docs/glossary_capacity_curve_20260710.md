# Glossary capacity curve protocol（2026-07-10）

## Objective

测量固定目标术语在 glossary candidate universe 从 10k 扩展到 100k、500k、
1M 时的检索与翻译质量变化，并确定 merged glossary 出现副作用的容量区间。

本阶段是 controlled capacity stress：四档都包含相同 ACL-238 target entries，
只追加 general-Wikidata distractors。它不能被描述为 `100 topics × 10k`；真正的
topic-union catalog 将在 capacity crossover 确认后独立构建和审计。

## Frozen protocol

- Audio：5 个 ACL talks，固定顺序
  `268 → 367 → 590 → 110 → 117`，总计 468 segments / 3,441.718s。
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
5. 约 15s aligned-window xCOMET-lite；30s 只作 segmentation sensitivity。

同时保留 corpus-gold retrieval precision（reference key 是否属于全局 ACL gold
inventory），但它只作较弱的诊断指标，不冒充 time-local Prompt Precision。

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

Hyper00 active run（2026-07-10）：

```text
/data02/jaxan/autoterm-capacity-zh-20260710/
```

- Code：Git branch `explore/multidomain-routing` at `82d6394`。
- Integrity：RAG、四个 glossary 和四个 MaxSim index 的 SHA-256 均与上表一致；
  copied Conda runtime file count 与 Taurus source 同为 92,523。
- Smoke：`outputs/smoke-10k` 已完成 60.0s / 28 emitted events，health 显示
  `active_terms=10000`，`tail_gap_samples=0`，保存的 112 条 prompt references 与
  `prompt_reference_count` 完全一致。
- Full run：`outputs/full-zh` 正在运行；GPU 2 执行 10k -> 1M，GPU 3 执行
  100k -> 500k。每档完成后 server 会重启，并由 health gate 检查精确 term count。
- 一个共享机 failure mode：首次 preflight 后 GPU 0/1 被已有的他人容器恢复任务
  抢占，两个 server 在 health 前因显存不足失败；失败输出隔离在 `outputs/full`，
  不进入正式结果。重新 preflight 后正式任务使用 GPU 2/3。

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
