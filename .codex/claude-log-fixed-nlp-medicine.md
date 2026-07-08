Verification already run:

Local:
python3 -m py_compile eval/streaming_sst/score_mixed_audio_terms.py framework/agents/omni.py test_score_mixed_audio_terms.py test_auto_working_fixed_top10.py && python3 -m unittest test_score_mixed_audio_terms
Result: OK, 1 test.

Taurus host:
python3 -m unittest test_score_mixed_audio_terms test_mixed_audio_switch_eval
Result: OK, 9 tests.

Taurus container sglang-omni-jaxan-qe-0706, copied checkout in /data/tmp/rasst-demo-test:
python3 -m unittest test_auto_working_fixed_top10.AutoWorkingFixedTop10Tests.test_fixed_glossary_preset_also_forces_prompt_top10 test_auto_working_fixed_top10.AutoWorkingFixedTop10Tests.test_none_glossary_does_not_backfill_prompt_candidates
Result: OK, 2 tests.

Known pre-existing drift:
Running the whole test_auto_working_fixed_top10 file in that container has one existing failure unrelated to this patch: config.auto_glossary_current_margin reads 0.3 from current config while the old test expected 0.1.

Existing mixed output rescored with new type diagnostics:
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_termacc_4block/term_acc_compare_type_diagnostics.md
Medicine type_acc_any: fixed_nlp 5/6, fixed_medicine 3/6, auto_working 5/6.
