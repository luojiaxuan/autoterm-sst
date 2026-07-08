Verification run before second Claude review:

Local bundled Python:

```text
$ /Users/luojiaxuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest test_mixed_domain_switch_eval test_hybrid_window_topic_router test_auto_glossary_switch_eval
....................
----------------------------------------------------------------------
Ran 20 tests in 0.029s

OK
```

Taurus final checkout:

```text
$ cd /mnt/taurus/home/jiaxuanluo/rasst-demo && git rev-parse --short HEAD
ce8fbc5

$ python3 -m unittest test_mixed_domain_switch_eval test_hybrid_window_topic_router test_auto_glossary_switch_eval
....................
----------------------------------------------------------------------
Ran 20 tests in 0.034s

OK
```

Benchmark outputs on Taurus from git ref c62b523:

```text
Output directory:
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523

Fixed 64 windows/item:
- alternating_target64_expected_probe.json:
  regression_pass=true, windows=640, transitions=9, switches=9,
  domain_accuracy=0.9719, steady_state_accuracy=1.0,
  steady_state_mismatch_count=0, wrong_switch_count=0,
  max_observed_switch_latency_windows=3.
- random_seed20260707_target64_expected_probe.json:
  regression_pass=true, windows=640, transitions=7, switches=7,
  domain_accuracy=0.9781, steady_state_accuracy=1.0,
  steady_state_mismatch_count=0, wrong_switch_count=0,
  max_observed_switch_latency_windows=3.
- alternating_target64_no_probe_diagnostic.json:
  regression_pass=false, windows=640, transitions=9, switches=0,
  domain_accuracy=0.5, steady_state_accuracy=0.5024,
  steady_state_mismatch_count=305, wrong_switch_count=0.
- alternating_target64_inverted_probe_diagnostic.json:
  regression_pass=false, windows=640, transitions=9, switches=10,
  domain_accuracy=0.0766, steady_state_accuracy=0.0424,
  steady_state_mismatch_count=587, wrong_switch_count=10.
- alternating_target64_contested_probe_diagnostic.json:
  regression_pass=false, windows=640, transitions=9, switches=7,
  domain_accuracy=0.8125, steady_state_accuracy=0.8434,
  steady_state_mismatch_count=96, wrong_switch_count=0,
  max_observed_switch_latency_windows=13.

Full-window clean expected probe control:
- alternating_all_target_expected_probe.json:
  regression_pass=true, windows=1905, transitions=9, switches=9,
  domain_accuracy=0.9906, steady_state_accuracy=1.0,
  steady_state_mismatch_count=0, wrong_switch_count=0,
  max_observed_switch_latency_windows=3.
- random_seed20260707_all_target_expected_probe.json:
  regression_pass=true, windows=1905, transitions=7, switches=7,
  domain_accuracy=0.9927, steady_state_accuracy=1.0,
  steady_state_mismatch_count=0, wrong_switch_count=0,
  max_observed_switch_latency_windows=3.
```

Accepted Claude findings patched:

```text
1. Benchmark validity wording and failure coverage:
   Added inverted and contested probe diagnostic modes and documented that clean
   expected-probe runs validate the state machine given clean probe evidence,
   not real MaxSim domain discrimination.
2. Probe-only effective signal guard:
   Added has_text_topic_signal to observe evidence and require the stricter
   audio-probe raw floor when text exists but has no topic evidence.
3. Path docs:
   Updated benchmark doc to use /mnt/taurus/... paths.
```
