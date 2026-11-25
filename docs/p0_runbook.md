# P0 Runbook

P0 has two separate acceptance targets:

1. Run the current prototype end to end through the UI/API/WebSocket protocol.
2. Validate the scheduler/protocol path with 32 concurrent sessions per GPU.

The launcher defaults to real inference. Protocol-level mock inference remains
available only when `RASST_DEMO_MOCK=1` is explicitly set.

## Start The Prototype

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo
HOST=127.0.0.1 PORT=8000 ./start_demo.sh
```

Then open:

```text
http://127.0.0.1:8000
```

The startup path uses:

```text
PYTHON=/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python
FAIRSEQ=/mnt/taurus/data2/jiaxuanluo/fairseq-0.12.2
```

Real GPU runs should use the cluster-provided `CUDA_VISIBLE_DEVICES` and should
not set `RASST_DEMO_FAKE_GPUS`. For protocol-only UI development, use
`RASST_DEMO_MOCK=1`; only then will the launcher create a fake GPU id if needed.

## Health Check

```bash
curl -sS http://127.0.0.1:8000/health | \
  /mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python -m json.tool
```

Expected real-mode fields:

```text
scheduler_enabled: true
mock_mode: false
supported_languages: ["English -> Chinese"]
```

## 32-Session Protocol Smoke

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo
/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python \
  scripts/smoke_p0_protocol.py \
  --base-url http://127.0.0.1:8000 \
  --sessions 32 \
  --timeout 20
```

Validated on 2026-06-01 against port `18000`:

```text
sessions=32 successes=32 failures=0 elapsed_s=0.069
sample_result=[mock-p0_00] terminology-aware translation after 0.10s audio
```

## Real Backend Boundary

The mock path is not a quality or throughput claim for the real speech LLM. It
only proves the protocol and scheduler surface. The next real-backend step is:

1. launch inside a Slurm GPU allocation;
2. run `RASST_DEMO_MOCK=0` with real `CUDA_VISIBLE_DEVICES`;
3. verify model load, one WebSocket translation, then 32 sessions;
4. record GPU memory, per-step latency, and failure mode.

The current repo is still an InfiniSST-era half-prototype. The P1 framework
rewrite should treat this P0 protocol as the behavioral contract while replacing
the backend with SGLang/SGLang-omni style serving and keeping VeRL/RAPO as an
optional post-training path.
