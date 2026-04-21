# Exp23 CUDA Graph Probe

Date: 2026-04-21

Hardware: 1x H100 SXM, `runpod/parameter-golf:latest`, PyTorch 2.9.1 + CUDA 12.8.

Purpose: test whether CUDA graph capture is worth enabling for the current
fused-CE single-GPU fast path. This was a speed-path diagnostic only: it used
synthetic uint16 shards with `vocab_size=16384`, `batch_size=1024`,
`seq_len=512`, `chunk_size=64`, `lm_head_backward_mode=fused`, and
`eval_batches=0`.

| mode | steps | tok/s | peak VRAM | graph decision |
| --- | ---: | ---: | ---: | --- |
| eager | 56 | 2,550,800 | 30,865 MB | n/a |
| CUDA graph probe | 56 | 2,343,472 | 31,026 MB | rejected |

The graph path captured successfully, but the measured replay step time only
improved from `0.2126s` to `0.2096s`, while capture plus warmup cost `0.772s`.
The gate rejected it for `projected_speedup_below_minimum`. Break-even was about
`259` steps / `55s`, but the steady-state speedup was only about `1.4%`, so this
is not the load-bearing speed lever.

Artifacts:

```text
experiments/23_fast_path/results_graph_probe/graph_none.json
experiments/23_fast_path/results_graph_probe/graph_probe.json
experiments/23_fast_path/results_graph_probe/logs/
```
