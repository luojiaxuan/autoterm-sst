Verification before second Claude review:

Git:

```text
Branch: framework
HEAD: f7d0b27 Update mixed-audio dry-run reference
Remote: git@github.com:luojiaxuan/rasst-demo.git
origin/framework: f7d0b27
Taurus checkout was tested at 57262e9 for code; docs-only final commit f7d0b27 updates the dry-run reference path.
```

Local tests:

```text
$ /Users/luojiaxuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m py_compile eval/streaming_sst/eval_mixed_audio_switch.py test_mixed_audio_switch_eval.py
OK

$ /Users/luojiaxuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest test_mixed_audio_switch_eval test_mixed_domain_switch_eval test_hybrid_window_topic_router test_auto_glossary_switch_eval
...........................
----------------------------------------------------------------------
Ran 27 tests in 0.029s

OK
```

Taurus tests:

```text
$ cd /mnt/taurus/home/jiaxuanluo/rasst-demo && git rev-parse --short HEAD
57262e9

$ python3 -m unittest test_mixed_audio_switch_eval test_mixed_domain_switch_eval test_hybrid_window_topic_router test_auto_glossary_switch_eval
...........................
----------------------------------------------------------------------
Ran 27 tests in 0.043s

OK
```

Accepted Claude findings patched:

```text
1. cursor_samples fallback removed:
   extract_record now requires meta.cursor_samples and raises RuntimeError if missing.
2. Runtime schema validation added:
   Required keys include cursor_samples, topic, topic_router, domain_probe_scores,
   router_text_source, prompt_reference_count, fixed_prompt_k, candidate_pool_count,
   plus topic.active_domain, topic.active_glossary_preset, topic.switch_count.
3. Long-run receive loop changed:
   The client no longer exits on the first 60s timeout before EOF. It only stops
   after explicit final/done/complete/eof events or a bounded number of idle
   timeouts after the feed task has sent EOF.
4. PCM wire format documented inline:
   The client sends float32 PCM in [-1, 1), matching eval_auto_glossary.py.
5. ACL file existence filtering fixed before limit counting.
6. Tests added for schema extraction, missing cursor_samples, and ACL missing
   wav filtering.
```

Taurus dry-run outputs at final code ref 57262e9:

```text
Output directory:
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_57262e9

alternating_audio_playlist_dryrun.json:
- block_count=10
- audio_seconds=16848.115

random_seed20260707_audio_playlist_dryrun.json:
- block_count=10
- audio_seconds=16848.115
```

Server/GPU blocker remains:

```text
Taurus 127.0.0.1:8011 is healthy but router_mode=embedding_refs, so it is not
valid for the target hybrid_window_topic generated-target/probe eval.

Taurus preflight selected only GPU4 free. Aries has root filesystem full and
insufficient clean free GPUs. No valid real E2E run was started.
```
