# 全局 source-dedup merged baseline（2026-07-10）

## 为什么不能直接使用 100k rows

当前 10 个 broad-domain slices 每个各有 10,000 rows，但跨域保留了 source-term
overlap。把它们直接拼成 100k 会让重复 term 重复占用 retrieval catalog 容量，也会
让 `merged-100k` 这个名称高估实际 candidate universe。

`scripts/term_memory/build_deduped_merged_index.py` 以 frozen catalog policy
`NFKC + casefold + collapsed whitespace` 规范化 English source term，并同时对
glossary rows、MaxSim `term_list` 与 `text_embs` 做一致切片。冲突策略是显式的：

1. base sources 按 CLI 顺序决定 priority，最早 occurrence 获胜；
2. 相同 normalized source 的后续 rows 不进入输出，但完整 glossary/index row、
   target variants、provenance 与原始 row index 全部写入 `duplicate_audit.json`；
3. optional distractor top-up 只补此前未出现的 source keys，直到 exact target size；
4. glossary/index row 数、term alignment、target translation、embedding trailing shape、
   dtype 和可用的 checkpoint provenance 任一不兼容时直接失败。

输出目录包含 `glossary.json`、`maxsim.pt`、`duplicate_audit.json`、
`manifest_fragment.json` 和 `build_report.json`。输出 index 内嵌 retriever checkpoint
SHA-256 与 source identity fingerprint，后续再合并 index 时可开启
`--strict-checkpoint-evidence`。

## Taurus real-asset validation

2026-07-10 使用以下 staged assets 做了 CPU-only validation（没有启动 GPU job）：

- base glossaries：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/catalog_10domain_100k_deepcat_20260710/`
- base indexes：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/indexes_10domain_100k_deepcat_20260710/`
- optional top-up glossary：
  `/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/glossaries/acl_tagged_gs100k_zh.json`
- optional top-up index：
  `/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/acl_tagged_gs100k/en-zh/maxsim.pt`
- retriever checkpoint：现有 `q3rag_scale_lora-...acl6060_recallat10.pt`，SHA-256
  `ba20040789d409436c7a77c4e3bcc87cd78611fd815d491977555bc64ab7f4bc`。

结果：

| Item | Count |
| --- | ---: |
| base input rows | 100,000 |
| globally unique base terms | 88,622 |
| duplicate rows removed | 11,378 |
| distinct duplicated base keys | 9,802 |
| base keys with different zh targets | 758 |
| deterministic top-up selected | 11,378 |
| exact top-up output | 100,000 |

Validated 100k output 暂存在：
`/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/dedup_baselines_20260710/dedup_topup_100k/`。
这是 local staging，不是 canonical release artifact；若论文采用，应上传 Hugging Face
dataset repo 并在 Git docs 记录 revision。

## Build commands

两个 baseline 使用相同、按 priority 排列的十组 `--source` 参数：

```text
nlp, medicine, education, finance, legal, environment,
entertainment, science, sports, art
```

每组参数格式为：

```bash
--source <role> \
  /mnt/data1/jiaxuanluo/rasst_autoterm_10domain/catalog_10domain_100k_deepcat_20260710/<role>_core_10k.json \
  /mnt/data1/jiaxuanluo/rasst_autoterm_10domain/indexes_10domain_100k_deepcat_20260710/<role>_core_10k/en-zh/maxsim.pt
```

其中每个 base role 还应重复提供同一个 build report，以验证 checkpoint path：

```bash
--build-report <role>=/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/indexes_10domain_100k_deepcat_20260710/index_build_report.json
```

完成十组参数后，88,622 exact union 使用：

```bash
--embedding-checkpoint /mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt \
--language-pair en-zh \
--preset-id merged_10domain_dedup_88622 \
--out-dir /mnt/data1/jiaxuanluo/rasst_autoterm_10domain/dedup_baselines_20260710/dedup_union_88622
```

100k exact unique baseline 在同一命令增加：

```bash
--topup-source general_capacity \
  /mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/glossaries/acl_tagged_gs100k_zh.json \
  /mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/acl_tagged_gs100k/en-zh/maxsim.pt \
--target-size 100000 \
--preset-id merged_10domain_dedup_topup_100k \
--out-dir /mnt/data1/jiaxuanluo/rasst_autoterm_10domain/dedup_baselines_20260710/dedup_topup_100k
```

Legacy capacity index 没有内嵌 checkpoint SHA，因此 report 会将它明确标为
`caller_declared_legacy_payload`；base indexes 则通过 shared `index_build_report.json`
验证 checkpoint path。正式发布前应重建或补充 top-up index provenance，再用
`--strict-checkpoint-evidence` 复验。
