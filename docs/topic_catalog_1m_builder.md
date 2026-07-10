# 100-topic × 10k catalog builder 与资产审计（2026-07-10）

## 目标与约束

`scripts/term_memory/build_topic_slice_catalog.py` 为 100-topic AutoTerm
实验构造可审计的 topic slices。它只接受显式 taxonomy 规则，不使用 term、description
关键词或语义模型猜 domain：

- Wikipedia records 通过 exact `category_path` / `category_query` 分配；
- Wikidata records 通过 `domain_root_qid`、`wikidata_type_qid` 或显式 QID 分配；
- 每条选中记录必须有 QID，并保留 RDF/category provenance；
- 每个 normalized English source term 在整个 catalog 中只出现一次；
- 要求的所有 target languages 必须同时存在，保证 zh/ja/de 等 language view
  使用完全相同的 source-term inventory；
- primary assignment 先于显式 `filler_match`，两者在 report 中分开统计；
- topic overlap 按 taxonomy `priority` 和配置顺序确定性消解，并保留 overlap samples；
- 每个 slice 输出 `domain_id`、`domain_description`、`term_count`、capacity、
  primary/filler count 和 source-term SHA-256 fingerprint。

`catalog_manifest.json` 使用现有 term-memory manifest 的 `scales` / `preset_meta`
结构，但 index 留空并标为 `pending`。完成 catalog audit 后再构建 MaxSim indexes，
不能把 candidate manifest 直接发布为 serving `current.json`。

示例 taxonomy 在 `configs/autoterm_topic_taxonomy.example.json`。正式实验必须另行
提交、审阅一个恰好 100 topics 的 taxonomy；example 不是实验 taxonomy。

## 运行方式

```bash
python3 scripts/term_memory/build_topic_slice_catalog.py \
  --taxonomy /path/to/autoterm_100_topics.json \
  --expected-topic-count 100 \
  --input wikimedia-batch-a=/path/to/candidates-a.json \
  --input wikimedia-batch-b=/path/to/candidates-b.jsonl \
  --target-languages zh \
  --snapshot-id autoterm_100topics_1m_v1 \
  --out-dir /path/to/catalog \
  --emit-merged
```

默认要求每个 slice 达到 capacity；不足时生成 `build_failure_report.json` 并失败。
探索数据覆盖率时可加 `--allow-underfilled`，但 underfilled catalog 不能作为论文中的
1M baseline。独立校验：

```bash
python3 scripts/term_memory/build_topic_slice_catalog.py \
  --validate-manifest /path/to/catalog/catalog_manifest.json \
  --expected-topic-count 100
```

## 当前 Taurus staged assets 能支持什么

已审计路径：

```text
/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/
/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/glossaries/
```

现有 8 组 structured candidates（Education、Science、Finance、Legal、Environment、
Entertainment、Sports、Art）共有 95,100 rows、81,144 个 normalized unique source
terms、81,084 个 unique QIDs，且 zh translation 完整。具体 provenance：

- 六个 deep-category files 各 12,000 rows，全部有 `wikidata_qid`、
  `wikipedia_pageid`、`category_path` 和 `category_query`；合计只有 42 个实际产生
  records 的 deepcat query buckets。
- Education 有 10,600 rows：9,424 个 `wikidata_exact_p31` 与 1,176 个
  `wikidata_p31_p279`；只有 9 个 distinct direct type QIDs / root QID buckets。
- Science 有 12,500 rows：3,919 个 exact-P31 与 8,581 个 category rows；category
  tree 较细，但绝大多数 leaf bucket 只有几十条，不能直接形成 10k slice。
- 已构造的 10-domain 100k catalog 允许 cross-domain overlap，因此只有 88,622
  unique source terms；它不能直接拆成 100 个全局 source-term-unique 10k slices。

现成 `wiki_general_zh_1m.json` / `acl_tagged_gs1m_zh.json` 可以做 controlled
capacity stress，但通用 1M 的 1,000,000 rows 中没有 `wikidata_qid`、`qid`、
`category_path`、`domain_root_qid` 或 `wikidata_type_qid` 字段。它只有 term、
translations、short description 和粗粒度 `source=wikidata`，因此不能在不重新关联
Wikidata/category provenance 的前提下声称是 100-topic catalog。

## 当前 blocker

当前 staged structured pool 的 unique terms 只有约目标 1M 的 8.1%，而且 bucket
分布高度不均衡。即使放松 broad domain 定义，也无法诚实填满 100 × 10k。需要的
下一步不是把通用 1M 用 description keyword 硬切，而是：

1. 先冻结 100-topic taxonomy（每个 topic 有可解释的 QID roots / category roots
   和不同的 router description）；
2. 按 topic 独立采集至少 12k--15k bilingual candidates，为 global dedup、质量过滤、
   overlap 和 underfill 留余量；
3. 或回到生成 `wiki_general_zh_1m.json` 的 upstream Wikidata join，恢复每个 row 的
   QID/P31/category fields，再用本 builder 分片；当前 Taurus staged paths 未找到该
   upstream provenance-rich 12.4M source；
4. 对 100 个 slices 做人工抽样 audit，并将 taxonomy、report、artifact revision
   固定后再跑 AutoTerm vs merged-1M。

因此，代码现在可以确定性构造和验证 catalog，也能诚实量化 coverage/overlap/filler；
但现有数据不足以产出可用于论文 claim 的 100 × 10k catalog。

## Source of Truth 与 artifact status

| 内容 | 当前 source of truth / staging | 状态与目标 |
|---|---|---|
| Builder、validator、example taxonomy、tests | Git 中的 `scripts/term_memory/build_topic_slice_catalog.py`、`configs/autoterm_topic_taxonomy.example.json`、`test_topic_slice_catalog.py` | `build/topic-catalog-1m` topic branch 已本地 commit，等待合入实验分支；未 push |
| 当前 structured candidate sources | Taurus `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/` | local staging；尚未上传 Hugging Face；若成为正式 100-topic dataset，HF dataset repo/revision 仍为 TBD |
| Real-field builder smoke | Taurus `/mnt/data1/jiaxuanluo/autoterm_topic_catalog_smoke_20260710/` | scratch validation，仅验证字段兼容性；不是正式 catalog，不计划作为论文 artifact 发布 |
| 正式 100-topic taxonomy 与 1M catalog | 尚不存在 | blocked；完成采集和人工 audit 后应上传 Hugging Face dataset，并把 repo URL/revision 回填 Git docs |

当前没有新的 model/checkpoint artifact。本文件与轻量测试结果属于 Git；大规模
candidate/catalog 数据仍只能视为 local staging，不能将 Taurus 路径当作 canonical
release location。
