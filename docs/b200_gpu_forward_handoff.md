# B200 GPU Forward 优化 — 交接文档

> **用途：** 从 Taurus/Aries A6000 上的 #760 thinker forward 诊断，切换到 B200
> 项目继续 **MoE expert weight-traffic / FP8 量化** 优化。B200 环境从零开始，
> 本文档覆盖：背景结论、git 拉取、环境、建议迁入 sglang fork 的内容、实验流程。
>
> **关联 issue：** [sglang-omni#760](https://github.com/sgl-project/sglang-omni/issues/760)
> （P1 decode-only nsys 结论已发：
> [#760#issuecomment-4703214587](https://github.com/sgl-project/sglang-omni/issues/760#issuecomment-4703214587)）

---

## 1. 背景：A6000 上已完成的结论

### 1.1 优化优先级（已决策）

| 方向 | A6000 结论 | B200 是否继续 |
|------|-----------|--------------|
| **MoE expert GEMM / weight traffic** | decode 69% GPU-busy，59% critical path；bs24→32 仅 +9% wall → **memory-bound** | **主目标** |
| **FP8 / int8 experts** | 理论上限最大（减半 weight traffic） | **主实验** |
| mixed-chunk (P2c) | +14% seg/s，已 ship PR [#789](https://github.com/sgl-project/sglang-omni/pull/789) | 默认开，B200 上保持 |
| custom-AR (TP comm) | 真实 comm ~1.3 ms/step；custom-AR 最多 ~3% decode wall | 次要；PR [#783](https://github.com/sgl-project/sglang-omni/pull/783) |
| host/scheduling overhead | scheduler decode step ~26–30 ms vs GPU wall ~18.6 ms（~8–11 ms gap） | **不在 B200 forward 项目范围** |

### 1.2 A6000 decode-only nsys baseline（job 46210，对照组）

**配置：** aries, GPU 0,1 (NV4), `Qwen3-Omni-30B-A3B-Instruct`, thinker TP=2,
mixed-chunk **off**, CUDA graph **on**, N=32 steady decode, 30 s nsys window。

| 指标 | bs≈32 (`graphId=2`) | bs≈24 (`graphId=5`) |
|------|---------------------|---------------------|
| decode wall / step | **18.6 ms** | 16.9 ms |
| GPU-busy (union) | 18.2 ms (98%) | — |
| inter-kernel gap | 0.40 ms (2.1%) | — |
| MoE expert GEMM (% GPU-busy) | **69%** | 69% |
| all-reduce (% GPU-busy) | 15% | — |
| dense GEMM | 8% | — |
| attention + KV | 2% | — |
| MoE critical path (% wall) | **59%** | — |
| exposed all-reduce (% wall) | 13% (~2.4 ms) | — |

**Thinker 模型维度（decode 相关）：**

| 参数 | 值 |
|------|-----|
| hidden_size | 2048 |
| num_hidden_layers | 48 |
| num_experts | 128 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 768 |
| all-reduce msg (bs=32) | 2048×32×2 = **128 KB** × 97 AR/step |

**Artifacts（Taurus NFS，可 rsync 到 B200 作对照）：**

```
/mnt/taurus/data2/jiaxuanluo/rasst_eval/nsys/decode_tp2_46210.nsys-rep
/mnt/taurus/data2/jiaxuanluo/rasst_eval/nsys/decode_tp2_46210.sqlite
/mnt/taurus/data2/jiaxuanluo/rasst_eval/logs/p1nsys_46210.log
/mnt/taurus/data2/jiaxuanluo/rasst_eval/runs/aries_p1nsys_46210.server.log
```

---

## 2. B200 项目目标

在 B200 上验证 **MoE expert FP8（或 int8）** 能否显著降低 decode forward wall time，
并用与 A6000 相同的 nsys 方法论做 apples-to-apples 对比。

**预期 B200 优势：**

- ~8 TB/s HBM3e → 同样 memory-bound MoE 应有更大 absolute/relative 收益
- 原生 FP8 Tensor Core → `fused_moe_kernel` / CUTLASS FP8 MoE 路径
- 192 GB VRAM → 可能 **TP=1** 跑 30B MoE（消除 all-reduce），或更大 decode batch

**不在本项目范围：** host-side scheduling、de-GIL、relay、端到端 StreamLAAL（除非
需要 sanity check）。

---

## 3. Git：从零拉取（唯一入口）

B200 **不要**依赖 Taurus/Aries NFS 或本机路径。所有脚本、分析工具、baseline
数字都在 fork 里：

```bash
git clone https://github.com/luojiaxuan/sglang-omni.git
cd sglang-omni
git checkout perf/b200-moe-fp8
```

**自包含诊断包（B200 主入口）：**

```
benchmarks/diagnostics/thinker_decode_forward/
├── README.md              # quickstart
├── FINDINGS_A6000.md      # BF16 baseline 对照
├── scripts/               # serve, nsys, fwd-by-bs, car_bench
├── analysis/              # decode_split.py, overlap.py
└── slurm/                 # 可选 SLURM 模板
```

Quickstart 见 fork 内
[`benchmarks/diagnostics/thinker_decode_forward/README.md`](https://github.com/luojiaxuan/sglang-omni/blob/perf/b200-moe-fp8/benchmarks/diagnostics/thinker_decode_forward/README.md)。

### 3.1 两个 repo（可选）

| repo | 用途 | 是否必需 |
|------|------|---------|
| **luojiaxuan/sglang-omni** | engine + 全部 profiling/FP8 实验 | **必需** |
| **rasst-demo** | 旧 Taurus eval harness | **不必**（已迁入 fork） |

### 3.2 分支

```bash
# sglang-omni
git clone https://github.com/luojiaxuan/sglang-omni.git
cd sglang-omni
git remote add upstream https://github.com/sgl-project/sglang-omni.git

# 诊断 + profiling 基础（A6000 全部工作在此分支）
git checkout perf/thinker-decode-opt
# 或等价 fork 名：diag/qwen3-omni-tp-vllm-gap（同一 HEAD）

# B200 实验新分支（含 thinker_decode_forward 包）
git checkout perf/b200-moe-fp8

# 或从诊断基础分支切出
git checkout perf/thinker-decode-opt
git checkout -b perf/b200-moe-fp8

# 可选：合并已提交的 mixed-chunk default
git fetch upstream
git merge feat/qwen3-omni-mixed-chunk-default-760   # PR #789

# 可选：custom-AR topology gate
git merge feat/thinker-tp-custom-allreduce-p2p-gate  # PR #783
```

**关键 commit（A6000 诊断链）：**

```
3e9399f  P2c prefill-attribution + mixed-chunk
cd7376b  P1 forward attribution (memory-bound)
eafb78f  env-gated step-phase + decode-stats profiling
```

### 3.3 rasst-demo（eval harness，可选）

```bash
git clone git@github.com:luojiaxuan/rasst-demo.git
cd rasst-demo
# 只需 eval/streaming_sst/ 下的 server + concurrency 脚本
```

B200 若只跑 engine 级 microbench / nsys，可以 **不 clone rasst-demo**；
完整 N=32 streaming load 才需要。

---

## 4. 已在 fork 中的内容（无需从 Taurus 拷贝）

以下内容已 commit 到 `benchmarks/diagnostics/thinker_decode_forward/`，**git pull 即可**：

| 文件 | 作用 |
|------|------|
| `scripts/serve_thinker.sh` | 本地/docker 起 thinker（无 host 硬编码路径） |
| `scripts/p1_nsys_decode.sh` | steady decode nsys + 自动 analysis |
| `scripts/p1_fwd_by_bs.sh` | forward vs batch-size |
| `scripts/car_bench.py` | custom-AR vs NCCL microbench |
| `analysis/decode_split.py` | decode-only kernel 分类 |
| `analysis/overlap.py` | critical-path / AR exposure |
| `FINDINGS_A6000.md` | A6000 BF16 baseline 数字 |
| `slurm/p1_nsys_decode.slurm` | SLURM 模板（改 partition） |

**Sibling（同 repo）：** `benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/` 提供
TP server 脚本 + N=32 streaming load。

~~§4.1 建议目录结构~~ — 已实现，见 fork README。

~~§4.2 从 Taurus NFS 拷贝~~ — **已废弃**，勿用。

## 5. 环境搭建（B200）

### 5.1 Docker 镜像

A6000 用的是 `frankleeeee/sglang-omni:dev`。**B200 (Blackwell) 需要新镜像**，
至少满足：

- CUDA ≥ 12.8（Blackwell 支持）
- PyTorch 支持 sm_100 / Blackwell
- `sgl_kernel` + nsys 2025.x 内置
- 与 fork 的 `sglang` / `sglang_omni` 版本匹配

**建议路径：**

1. 先在 B200 上试跑 upstream `sglang-omni` 官方 Dockerfile / CI image
2. 不行则基于 `frankleeeee/sglang-omni:dev` 换 CUDA base 重建
3. 把最终 image tag 写进 `scripts/serve_qwen3omni_thinker.sh` 的 `IMAGE=` 默认值

### 5.2 模型权重

| 用途 | checkpoint | 说明 |
|------|-----------|------|
| **BF16 baseline（对照 A6000）** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | A6000 nsys 用的同一模型 |
| **FP8 主实验** | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` | fork 已有 FP8 CI + colocated config |
| int8（若做） | 需确认是否有官方 checkpoint 或 runtime quant | 次选 |

```bash
# 下载（B200 上设 HF cache 路径）
export HF_HOME=/path/to/hf_cache
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
huggingface-cli download marksverdhei/Qwen3-Omni-30B-A3B-FP8
```

**FP8 注意（来自 `model_worker.py` + unit tests）：**

- Qwen3-Omni native FP8 需要 checkpoint 内 `quantization_config.weight_block_size = [128, 128]`
- MoE runner：`moe_runner_backend=auto` → 在支持 CUTLASS FP8 MoE 的 GPU 上选 `cutlass`
- 不要用 `flashinfer_cutlass` backend 跑 native FP8 checkpoint（会 raise）
- 默认 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`（避免 post-ready 长 compile）

### 5.3 Eval 数据（N=32 streaming load）

```
acl6060_zh_segments/   # 468 个 16 kHz mono segment wav + SimulEval source/target
```

推荐从公开 HF dataset 下载：

```bash
huggingface-cli download gavinlaw/rasst-demo-acl6060-zh-segments \
  --repo-type dataset \
  --local-dir /path/to/acl6060_zh_segments

export DATA_DIR=/path/to/acl6060_zh_segments
```

Taurus 备份路径：
`/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments`

也可从 RASST release data 重新生成：

```bash
python benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/eval/prepare_acl6060_segments.py \
  --out-dir benchmarks/diagnostics/thinker_decode_forward/artifacts/data/acl6060_zh_segments
export DATA_DIR=benchmarks/diagnostics/thinker_decode_forward/artifacts/data/acl6060_zh_segments
```

### 5.4 诊断 env vars（engine 内置，无需改代码）

| env | 作用 |
|-----|------|
| `SGLANG_OMNI_DECODE_STATS=1` | `[decode stats]` bs 直方图 + CUDA graph hit rate |
| `SGLANG_OMNI_DECODE_STATS_INTERVAL=2` | 日志间隔 (s) |
| `SGLANG_OMNI_PHASE_PROFILE=1` | `[step phases]` CPU/GPU 分阶段 |
| `SGLANG_OMNI_PHASE_SYNC=1` | forward 后 sync，测 true GPU time |
| `ENABLE_MIXED_CHUNK=1` | P2c 默认开（端到端 sanity 时） |
| `ENABLE_MIXED_CHUNK=` (empty) | **nsys decode-only 对照时关掉** |
| `NSYS_PREFIX=...` | server launch 外包 nsys（见 §6.2） |

---

## 6. 实验流程（推荐顺序）

### Phase 0 — 环境 smoke

```bash
# 1. 容器 + 2 GPU TP=2 起 server
GPUS=0,1 PORT=8100 bash scripts/serve_qwen3omni_thinker.sh

# 2. health check
curl http://127.0.0.1:8100/health

# 3. 看 thinker args + decode stats
SGLANG_OMNI_DECODE_STATS=1 GPUS=0,1 PORT=8100 bash scripts/serve_qwen3omni_thinker.sh
# 日志里应有 [thinker args] 和 [decode stats]
```

### Phase 1 — BF16 baseline nsys（复现 A6000 方法论）

**目的：** 在 B200 上建立 BF16 decode forward baseline，与 A6000 18.6 ms 对比。

```bash
# mixed-chunk OFF, CUDA graph ON, N=32 steady decode
ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0 \
  sbatch slurm/b200_p1_nsys_decode.sh
```

**nsys 采集要点（已从 A6000 踩坑总结）：**

- 用 **session control**（`nsys launch → start → stop`），不要 fixed `--delay/--duration`
- `nsys launch` 上 **不要** 加 `--sample/--cpuctxsw/--backtrace`（放到 `nsys start`）
- 必须 `--cuda-graph-trace=node` 才能在 graph replay 里看到 kernel
- 等 server healthy + 30 s ramp 后再 `nsys start`
- 分析时用 sqlite 的 `graphId IS NOT NULL` 隔离 decode

**输出对照表：**

| 指标 | A6000 (46210) | B200 BF16 |
|------|--------------|-----------|
| decode wall / step (bs≈32) | 18.6 ms | _待填_ |
| MoE % GPU-busy | 69% | _待填_ |
| MoE % critical path | 59% | _待填_ |
| exposed AR % wall | 13% | _待填_ |

### Phase 2 — FP8 thinker decode nsys

**目的：** 同一 load / 同一 nsys 方法，只换 FP8 checkpoint + quantization path。

```bash
# FP8 colocated config（参考 examples/configs/qwen3_omni_fp8_colocated.yaml）
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-FP8 \
  QUANTIZATION=fp8 \
  ENABLE_MIXED_CHUNK= \
  sbatch slurm/b200_p1_nsys_decode.sh
```

**关注 kernel 变化：**

- `fused_moe_kernel` → 是否变成 FP8/CUTLASS variant？
- MoE ms/step 下降多少？
- dense GEMM / attention 是否也走 FP8？
- CUDA graph 是否仍 100% hit？

**成功标准（初版）：**

- decode wall / step 相对 BF16 baseline 下降 **≥15%**（memory-bound 场景下 FP8
  理论 ~2× weight BW，实际受 kernel efficiency 限制）
- BLEU/质量不回归（若跑端到端）

### Phase 3 — TP=1 探索（B200 192GB 独有）

A6000 上 TP=1 不可行（48 GB × 2 放不下 30B MoE ~60 GB）。B200 192 GB 可能单卡
或 TP=1 可行：

```bash
TP_SIZE=1 GPUS=0 bash scripts/serve_qwen3omni_thinker.sh
```

若 TP=1 可行 → 额外 nsys 对比 **消除 all-reduce** 后的 decode wall。

### Phase 4 — bs scaling + forward curve

```bash
SGLANG_OMNI_PHASE_PROFILE=1 SGLANG_OMNI_PHASE_SYNC=1 SGLANG_OMNI_DECODE_STATS=1 \
  sbatch scripts/p1_fwd_by_bs.sh
# N sweep: 8, 16, 24, 32 → [fwd-by-bs] decode curve
```

验证 FP8 下 MoE 是否仍 memory-bound（bs 增大时 wall 亚线性增长）。

---

## 7. Server 启动参考（B200 改路径版）

从 `rasst-demo/eval/streaming_sst/servers/serve_sglang_qwen3omni.sh` 精简。
迁入 fork 后把 `REPO_ROOT` / `SGLANG_OMNI_SRC` / `MODEL_PATH` / `IMAGE` 改成
B200 路径。

```bash
# 关键 env
REPO_ROOT=/path/to/sglang-omni          # fork root
SGLANG_OMNI_SRC=/path/to/sglang-omni    # PYTHONPATH
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct   # 或 FP8 路径
IMAGE=your-b200-sglang-omni:latest
GPUS=0,1
TP_SIZE=2
ENABLE_MIXED_CHUNK=1                    # 端到端; nsys decode-only 时设空
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.75                # B200 192GB 可调高
MAX_RUNNING_REQUESTS=32

# nsys hook（P1 profiling）
NSYS_PREFIX="nsys launch --session-new=sstprof --trace=cuda,nvtx,nccl --cuda-graph-trace=node"

# 启动
GPUS=0,1 PORT=8100 bash scripts/serve_qwen3omni_thinker.sh
```

**FP8 额外 flags（通过 server args overrides 或 CLI）：**

```bash
--quantization fp8
# moe_runner_backend 保持 auto（会选 cutlass）
# 参考 examples/configs/qwen3_omni_fp8_colocated.yaml
```

---

## 8. nsys 分析 quickstart

```bash
# 1. 导出 sqlite（若 nsys rep 已有）
nsys export --type sqlite decode_tp2_JOB.nsys-rep -o decode_tp2_JOB.sqlite

# 2. 快速 top kernels
nsys stats --report cuda_gpu_kern_sum decode_tp2_JOB.nsys-rep | head -40

# 3. decode-only split
python analysis/decode_split.py decode_tp2_JOB.sqlite
python analysis/overlap.py decode_tp2_JOB.sqlite

# 4. 确认 decode graph bucket
sqlite3 decode_tp2_JOB.sqlite \
  "SELECT graphId, count(*) k, printf('%.1f', sum(end-start)/1e6/count(*)) ms_per_launch
   FROM CUPTI_ACTIVITY_KIND_KERNEL
   WHERE deviceId=0 AND graphNodeId IS NOT NULL
   GROUP BY graphId ORDER BY k DESC;"
```

---

## 9. 已知问题 / 注意事项

1. **mixed-chunk ON 时 decode graph 不 pure** — nsys decode-only 分析必须
   `ENABLE_MIXED_CHUNK=` (empty) 关掉。
2. **A6000 nsys 里 decode graph 走 NCCL 而非 custom-AR** — 尽管 hardware 支持
   custom-AR；PR #783 修 topology gate，但 capture path 可能仍 route 到 NCCL。
   B200 上先不 blocking 在此；AR 本身不是主瓶颈。
3. **`SGLANG_OMNI_PHASE_SYNC=1` 会扰动 absolute throughput** — 只读
   `[fwd-by-bs]` / forward split，不读 seg/s。
4. **FP8 首次 launch 可能 long compile** — 设 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`，
   预热后再 profile。
5. **Docker 不要用 `--privileged`** — 会破坏 `--gpus device=` 隔离（见
   serve script 注释）。
6. **路径一律用 host-qualified 绝对路径** — 写脚本/docs 时避免 bare `/home/...`。

---

## 10. 相关 PR / 分支 / 文档索引

| 项 | 链接 / 路径 |
|----|------------|
| Issue #760 | https://github.com/sgl-project/sglang-omni/issues/760 |
| P1 nsys comment | https://github.com/sgl-project/sglang-omni/issues/760#issuecomment-4703214587 |
| P2c mixed-chunk PR | https://github.com/sgl-project/sglang-omni/pull/789 |
| custom-AR topology PR | https://github.com/sgl-project/sglang-omni/pull/783 |
| 诊断分支 | `perf/thinker-decode-opt` @ `luojiaxuan/sglang-omni` |
| vLLM gap 诊断包 | `benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/` |
| FP8 用法 | `docs/basic_usage/qwen3_omni.md` § Single-GPU FP8 |
| FP8 config | `examples/configs/qwen3_omni_fp8_colocated.yaml` |
| FP8 模型 | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` |
| ACL6060 prepared segments | https://huggingface.co/datasets/gavinlaw/rasst-demo-acl6060-zh-segments |
| rasst-demo server script | `eval/streaming_sst/servers/serve_sglang_qwen3omni.sh` |
| A6000 nsys 脚本 | `/mnt/taurus/data2/jiaxuanluo/rasst_eval/aries_p1_nsys_decode.sh` |
| 本地 comment 草稿 | `rasst-demo/docs/sglang_omni_760_p1_decode_nsys_comment.md` |

---

## 11. B200 项目 TODO checklist

- [ ] B200 节点 access + SLURM partition 确认
- [ ] Blackwell-compatible Docker image 构建/验证
- [ ] `git clone` fork + checkout `perf/b200-moe-fp8`
- [ ] 迁入 `benchmarks/diagnostics/thinker_decode_forward/`（§4.1）
- [ ] 下载 BF16 + FP8 checkpoint
- [ ] 下载 `gavinlaw/rasst-demo-acl6060-zh-segments` 并设置 `DATA_DIR`
- [ ] Phase 0 smoke test
- [ ] Phase 1 BF16 nsys → 填 §6 对照表
- [ ] Phase 2 FP8 nsys → 与 BF16 对比 MoE ms/step
- [ ] Phase 3 TP=1 可行性（可选）
- [ ] 更新 #760 comment / 开新 issue 报 B200 结果

---

*文档版本：2026-06-14。基于 A6000/Aries job 46210 + fork `perf/thinker-decode-opt`。*
