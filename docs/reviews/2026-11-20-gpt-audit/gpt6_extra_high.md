# GPT-6 Extra High fact audit, 2026-11-20

ChatGPT temporary chat, Unpersonalized, GPT-6 at Extra High ("worked for 6m 32s"). Input: the 9-page PDF built at `a9175c6` and `audit_bundle.zip` (experiment-branch `9afc056` result JSONs, router and scoring code, `docs/decisions.md`, `latex/README.md`). Prompt reproduced below. Math that the page rendered twice is kept once. The 226-row numerical ledger TSV it attached was not downloaded; its summary is in section 3.

## Prompt

SETUP: autoterm_sst.pdf is our NAACL 2027 demo paper: a Qwen3-4B router plus sticky HMM filter picks topic slices for RASST glossary retrieval during streaming translation. MY POSITION: every number traces to audit_bundle.zip; Merged-100k/1M rows come from an earlier run paired talk by talk with AutoTerm-SST (same audio order and chunk boundaries, see paper_sources_README.md). ATTACHED zip: llm_router_20261120/ (run3/*.json, latency/summary.json, replay JSONs), realsi_20261120/results/routing_coverage.json, run1_evidence_20261024/, code/ (router, serving agent, scoring scripts), paper_decisions.md, paper_sources_README.md. TASKS: (1) check every number, interval and p-value in the PDF against the files; list mismatches with file, key, correct value; (2) flag statistical errors, confounds, and sentences that are false given the evidence; (3) top three attacks a demo-track reviewer would make. Read the attachments first and cite files and keys. Be blunt. FORMAT: Your reader is an AI agent. Be terse: no preamble, no restating my setup, no hedging boilerplate. Every sentence must carry a claim, a number, a citation, or an instruction. Prefer numbered claims and compact tables.

## Response (verbatim from the page text)

1. Statistical errors and significance risks

1.1. Merged-1M terminology significance fails the bundled multiple-testing correction. PDF pp. 5–6 and Table 4 report TERM_ACC +3.27 points, p=0.031, marked †. The source has:

run3/vs_run1_merged_paired_significance.json → contrasts["lm_hmm_vs_merged1m:term_acc"].talk_permutation_p = 0.03125

Same contrast, talk_permutation_p_holm = 0.0625

The unadjusted result is correct; the Holm-adjusted result is not significant at 0.05. Report the correction or explicitly define the unadjusted tests as the inferential basis.

1.2. “On par” is not an equivalence result. PDF §6.2 reports AutoTerm-SST versus Known-domain-1k as +0.20 TERM_ACC points, 95% CI [−0.83,+0.99], p=0.82. The interval includes zero but permits a disadvantage of approximately 0.83 points. No equivalence margin or equivalence test is supplied. Replace “on par” with “no statistically detected difference.” Source: run3/paired_significance.json → contrasts["lm_hmm_vs_known:term_acc"]; run1_evidence_20261024/README.md → §1.

1.3. InfiniSST uses an incompatible “All” denominator. Table 1 reports 61.4%, the unweighted mean of NLP 76.6% and medicine 46.3%. Weighting by the paper's 1,368 NLP and 679 medicine occurrences gives approximately 66.5% instead. AutoTerm-SST's 87.7% uses pooled occurrences. The footnote acknowledges averaging, but placing both figures in the same “All” column invites an invalid comparison. Sources: PDF §6.2/Table 1; run1_evidence_20261024/README.md → Inputs; paper_sources_README.md → InfiniSST.

2. Provenance and experimental confounds

2.1. The main catalog-scale comparison mixes runs and hardware without disclosure. Table 1's AutoTerm-SST, Known-domain-1k, similarity, Merged-2k and Merged-10k rows come from November 20 run 3 on two H200 GPUs. Merged-100k and Merged-1M come from July 11 run 1 on one H200. Identical audio order and chunk boundaries permit paired talk-level comparisons; they do not establish that the catalog size alone caused the differences. Sources: paper_sources_README.md → Table 1; run3/results.json → hardware; paper_decisions.md → 大合并基线不在 run 3 重跑.

2.2. The original million-entry baseline cannot currently be reproduced from the retained catalog. paper_decisions.md → 大合并基线不在 run 3 重跑 records that the original 1M index is missing. The attempted reconstruction covers 27,752 of 30,716 originally injected distinct mappings, leaving 2,964 mappings (9.65%) unaccounted for. The PDF neither discloses this nor supplies equivalent inputs.

2.3. The reported processing speed does not validate the deployed background router. llm_router_20261120/latency/summary.json → code_revision identifies the earlier synchronous implementation. At 16 concurrent sessions, its chunk-latency median is 3.38 s, versus 0.81 s for similarity routing; 80.3% of AutoTerm-SST chunks exceed the 1.92-second audio interval. paper_decisions.md → 路由打分改为后台进行 records the subsequent asynchronous fix, but the bundle contains no post-fix concurrency measurement. PDF §6's real-time claim is therefore not established for the final implementation.

2.4. The flagship router has a substantially worse displayed top topic than the similarity baseline. llm_router_20261120/run3/routing_coverage.json reports:

| Ten-talk closed-loop metric | AutoTerm-SST | Similarity |
| --- | --- | --- |
| Correct slice in working set | 97.99% | 98.70% |
| Correct slice ranked first | 81.49% | 95.84% |
| First-ranked slice changes | 231 | 15 |

The retriever uses the top four, so terminology quality need not suffer. But PDF §5 presents the top-ranked slice as a meaningful live status indicator; on the main stream, that indicator is markedly less stable and less accurate.

2.5. RealSI uses evaluation-derived glossary entries. realsi_20261120/README.md → 术语标注, Slice says an LLM extracted 439 term occurrences from the evaluated talks and inserted 19–46 corresponding entries into each relevant slice. Against the one manually annotated talk, term-annotation precision was 80.0%. Domain-label routing coverage is still a legitimate metric; these results do not establish generalization to independently constructed glossaries.

2.6. Thirty RealSI talks are not thirty independent talks. realsi_20261120/README.md → 播放顺序 specifies 10 recordings repeated in three orders. realsi_20261120/routing_coverage.json confirms 29/30 AutoTerm-SST talks versus 11/30 similarity talks ever reach correct top-1. Its medians of 15.64 s and 68.15 s exclude the 1 and 19 failures, respectively. Report the exclusion wherever comparing adaptation delays.

2.7. RealSI does not demonstrate improved downstream terminology accuracy. paper_decisions.md → RealSI 在论文里的定位 records a +1.37-point TERM_ACC difference at p=0.50, omitted from the paper. Better routing coverage has not translated into a statistically established improvement in the ten-domain translation test.

2.8. “Sentences without terms” is a false description of the evaluated subset. code/evidence/term_free_quality.py → term_free selects sentences whose Chinese references lack a glossary target string. It does not establish that their source audio contains no terminology. run3/vs_run1_merged_term_free_quality.json → by_domain shows 888 of 935 such sentences (95.0%) are medicine. The +1.85 BLEU finding is real for this reference-string-defined, medicine-heavy subset, not proof of degradation on genuinely terminology-free speech.

3. Numerical reconciliation

| PDF claim | Source file and key | Audit verdict |
| --- | --- | --- |
| Table 2 similarity top-1 delay 68.1 s | realsi_20261120/routing_coverage.json → cosine.median_delay_to_top_s_among_reached = 68.15 | Rounding mismatch: 68.2 s at one decimal. Minor. |
| Table 3 retrieval/generation: Known .059/.359, similarity .064/.343, AutoTerm .094/.333 seconds | llm_router_20261120/latency/summary.json → configs.{fixed,routed,lm}.levels[0].{retrieve_s,generation_s} | Six unauditable cells: file contains p50/p95, not the reported averages or raw observations. Do not substitute medians for means. |
| Table 1 Merged-100k and Merged-1M NLP/medicine TERM_ACC, prompt precision, pairs/chunk | paper_sources_README.md → Table 1; referenced docs/autoterm_1m_budget_search_20260711.md | Eight cells lack original supporting score records in the ZIP. Pooled TERM_ACC and BLEU differences have corroborating summary evidence. |
| Table 1 InfiniSST 76.6/46.3/61.4, BLEU 41.84, MT-BLEU 38.32 | paper_sources_README.md → InfiniSST | Five cells not independently auditable: the referenced RASST baseline record is absent. The inconsistent “All” averaging is addressed in §1.3. |
| §6 prior RASST retrieval 0.042 s/chunk | Prior RASST benchmark, absent from bundle | External figure unverified. |

Verified rather than mismatched: All 84 numerical cells of PDF Table 4's paired-difference, bootstrap-interval and permutation-test comparisons agree with run3/paired_significance.json, run3/vs_run1_merged_paired_significance.json and their term-free counterparts to published precision. Table 5's replay coverage cells agree with llm_router_20261120/live_cadence_replay.json. Table 1's five run-3 configurations agree with run3/results.json and run3/prompt_precision.json. Table 6's principal settings agree with the configuration documentation and code.

Download the complete 226-claim numerical audit ledger (TSV)

4. Additional incorrect or unsupported inferences

4.1. Catalog-scale causality is not isolated. PDF §6.2 attributes the Merged-1M gap to irrelevant mappings occupying a fixed prompt. Different run conditions, the missing original index, and the absence of a same-run catalog-scale ablation prevent identifying that mechanism as the cause. paper_sources_README.md → Table 1; paper_decisions.md → 大合并基线不在 run 3 重跑.

4.2. The deployed 10k merged baseline already matches or beats AutoTerm-SST. run3/results.json → systems.merged10k reports 88.08% TERM_ACC and 45.11 BLEU, versus AutoTerm-SST's 87.69% and 45.10. run3/paired_significance.json → lm_hmm_vs_merged10k gives no significant difference. The demonstrated benefit is not superior quality at the actual ten-slice catalog size.

4.3. “Reversing the order removes position preference” is stronger than the algorithm warrants. code/language_model_router.py → _score_batch_sync averages two letter-order scores; averaging may mitigate positional bias, but cannot guarantee its removal. Change “removes” to “reduces potential.”

4.4. The HMM overlap correction is heuristic, not a proof of independent evidence. code/language_model_router.py → filter_step applies a fractional likelihood exponent based on new context chunks. That does not establish that dependence between successive LLM outputs is eliminated. Change PDF §4's “counts each chunk's evidence once” to an explicitly approximate description.

4.5. Unknown-topic handling fails in the retained negative test. llm_router_20261120/unknown_domain.json → holdouts.{nlp,medicine}.fallback_when_topic_missing yields only 3.83% and 0% rejection when the true topic is not registered. “Zero setup” applies to an offline-prepared catalog, not arbitrary unseen domains.

4.6. TERM_ACC is a lexical-window metric, not proof of correct interpretation. code/score_time_aligned_terms.py → PRE_S=2, POST_S=30 accepts reference-target strings in a broad output-time window. The reported 87.7% is valid under that definition; it does not independently establish correct term sense, alignment or fluent use.

5. Top three demo-track reviewer attacks

1. “Your headline million-entry comparison cannot be reproduced.” The original 1M catalog is missing, its reconstruction omits 9.65% of previously injected mappings, and the strongest comparison mixes July and November runs without paper disclosure. Required fix: restore the actual catalog and rerun on the same configuration, or prominently disclose provenance and limit causal claims. Sources: paper_decisions.md → 大合并基线不在 run 3 重跑, 论文不提 run.

2. “Why do I need this router?” On the headline translation stream, Merged-10k slightly outperforms AutoTerm-SST; on RealSI, the benefit is routing coverage, while the retained terminology comparison is p=0.50. Required fix: show downstream translation gains in independently evaluated, genuinely multi-domain streams, including a same-run larger-merged baseline. Sources: run3/results.json; paper_decisions.md → RealSI 在论文里的定位.

3. “Is this a trustworthy live demo or a favorable replay?” The interface's current-topic indicator flips 231 times versus 15 for similarity routing, the synchronous router fails at 16-session concurrency, and the revised background worker lacks a new latency measurement. Required fix: report final-code latency under concurrency, show topic-transition failures, and label the UI's top slice as a probabilistic estimate rather than established ground truth. Sources: run3/routing_coverage.json; latency/summary.json; paper_decisions.md → 路由打分改为后台进行.

Priority: Repair the missing 1M provenance and run disclosure first; measure the final worker second; strengthen the evaluation beyond top-four coverage third.
