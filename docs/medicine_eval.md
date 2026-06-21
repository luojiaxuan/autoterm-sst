# Medicine terminology eval — the REAL pipeline + data (recorded)

The validated terminology metric is **TERM_ACC from the InfiniSST SimulEval
pipeline**, not the ad-hoc WebSocket `score_terms.py` scorer. Reference numbers:
medicine hardraw lm1–4 TERM_ACC = **0.7905 / 0.8336 / 0.8470 / 0.8514**.

> Why the earlier WS recall was wrong: it fed whole variable-length utterance
> wavs through the streaming scheduler (arbitrary segmentation) and matched zh by
> incidental substring. The real eval uses **fixed variable-context chunks with
> MFA-aligned term spans** and a proper term-adoption metric. Different speech
> form, different metric — not comparable.

## Correct speech form (the key fix)

Per-chunk dataset: each row is a **fixed-context audio chunk** of
2.88 / 3.84 / 4.80 / 5.76 s (`context_duration_tag` 2p88/3p84/4p8/5p76), with the
target term's MFA span inside the chunk (`mfa_term_start_in_chunk`,
`mfa_term_end_in_chunk`). `chunk_audio_path` → a pre-cut context wav. Example row
keys: `chunk_src_text, utter_id, sample_id, chunk_idx, chunk_audio_path,
chunk_duration_sec, context_duration_sec, source_chunk_idx_1p92, term, term_key,
mfa_term_start_in_chunk, mfa_term_end_in_chunk, mfa_locate_method`.

## Vendored code (this repo) — `eval/infinisst_eval/`

Mirrors the InfiniSST layout so imports resolve with `PYTHONPATH=$PWD/eval/infinisst_eval`
(the agent does `REPO_ROOT=parents[4]; from agents.streaming_maxsim_retriever import …`).

| role | path (under `eval/infinisst_eval/`) |
|---|---|
| SimulEval agent (omni vLLM + MaxSim RAG) | `agents/infinisst_omni_vllm_maxsim_rag.py` |
| MaxSim retriever (differs from RASST's!) | `agents/streaming_maxsim_retriever.py` |
| agent CLI options | `agents/options.py` |
| eval orchestrator | `documents/code/simuleval/eval_density_unified.sh` |
| medicine input prep | `documents/code/simuleval/prepare_medicine_one_talk_inputs.py` |
| batched vLLM RAG engine | `documents/code/simuleval/src/batched_vllm_rag_eval.py` |
| StreamLAAL + TERM_ACC scorer | `documents/code/offline_sst_eval/offline_streamlaal_eval.py` |
| aggregation | `documents/code/simuleval/aggregate_medicine_retriever_lm_sweep.py`, `instances_log_to_tsv.py` |
| medicine varctx data builder | `documents/code/data_pre/training_terms_for_retriever/prepare_medicine_variable_context.py` |
| glossary translation enrich | `documents/code/data_pre/.../src/enrich_medicine_glossary_translations_from_eso.py` |
| maxsim index builder | `retriever/gigaspeech/build_maxsim_index.py` |

**External runtime deps (NOT vendored):** `simuleval` pkg, `vllm`, `qwen_omni_utils`,
`transformers`; fairseq `examples/.../stream_laal_term.py`; the InfiniSST conda env.
MFA TextGrids default: `/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids`.

## Data (on /mnt/gemini — NOT copied, paths recorded)

**1. medicine terms + MFA-strict eval data**
`/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/`
- `medicine_dev_dataset.jsonl` — 11,071 rows, 2,408 term rows, 571 unique medicine
  terms; `mfa_exact` only (dropped 204 unmatched + 241 char-proportional)
- `medicine_dev_dataset_stats.json`, `medicine_dev_dataset_dropped_terms.json`
- `audio_chunks/{2p88,3p84,4p8,5p76}/`
- `medicine_glossary_gt_plus_medicine_wiki_gs10000.json`
- builder: `prepare_medicine_variable_context.py`

**2. translated medicine glossary** (10,000/10,000 translated; 571 ESO + 9,429 wiki filler)
`…/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json` (+ `_stats.json`)
- enrich script: `enrich_medicine_glossary_translations_from_eso.py`
- upstream wiki glossary: `documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json`

**3. medicine hardraw RASST eval (zh, successful)**
- eval root: `/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242`
- manifest: `eval/infinisst_eval/documents/code/simuleval/manifests/2026/05/20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch.json`
- launcher: `eval/infinisst_eval/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh`
- hard medicine glossary (212 entries; samples 404, 545006, 596001, 605000, 606):
  `/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json`
- per-lm outputs under eval root: `eval_results.tsv / scores.tsv / instances.log / term_adoption.json`
- **TERM_ACC lm1–4 = 0.7905 / 0.8336 / 0.8470 / 0.8514**

## Model + retriever + RAG params (from the launcher)

- speech-LLM (new_v9 termtag delay): `/mnt/gemini/data1/jiaxuanluo/slm_exports/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8/keep1.0_r32/v0-20260524-062743-hf`
- hn1024 retriever ckpt: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- RAG: `top_k=10, tau=0.78, lora_r=128, text_lora_r=128, lookback_sec=1.92`; GPU_PAIR=0,1, RAG on cuda:1.

## Related (de/ja/abbrev/promptfix — review with care)

`eval/infinisst_eval/documents/code/simuleval/{launchers,manifests,notes}/2026/05/`
keywords: `medicine_norag_abbrev_restored, medicine_hardraw, medicine_de,
medicine_ja, promptfix, cap16_denoise`.
