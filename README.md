# EPIC

A [verl](https://github.com/verl-project/verl)-based implementation of
**Exact-KL Prefix Importance-Ratio Clipping (EPIC)** for off-policy LLM
reinforcement learning.

## Implementation

The policy loss is implemented in
[`core_algos.py`](vendor/verl/verl/trainer/ppo/core_algos.py) and registered as
`prefix_geometric_probability_weighted_exact_kl_dual_clip`.

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: prefix_geometric_probability_weighted_exact_kl_dual_clip
      prefix_exact_kl_delta_low: 0.0005
      prefix_exact_kl_delta_high: 0.002
    loss_agg_mode: seq-mean-token-mean
```

EPIC uses prefix ratios, geometric-mean probability calibration and a
current-token-only gradient path. It does not require expert-routing replay.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for code locations.

## Installation

Prepare a Linux/CUDA environment with compatible PyTorch, Megatron, vLLM and
Ray versions as described in [ENVIRONMENT.md](ENVIRONMENT.md), then install
the included verl implementation:

```bash
uv pip install --no-deps -e vendor/verl
```

## Model and data

The default configuration uses Qwen3-30B-A3B-Base and DAPO-Math-17K.
Prepare the model and processed Parquet files following [DATA.md](DATA.md).
Validate the input schema with:

```bash
uv run --with pyarrow python scripts/check_data.py /shared/data
```

## Training

The default configuration uses 32 A100 GPUs across four nodes. On a prepared
Ray cluster, run:

```bash
python scripts/run_experiment.py epic \
  --model /shared/model/Qwen3-30B-A3B-Base \
  --data /shared/data --output /shared/outputs/epic \
  --ray-address auto --execute
```

Omit `--execute` to print the resolved configuration without starting training.
Use a fresh output directory. Model, data and output paths must be accessible
at the same absolute locations on all nodes.

Configs are in [`configs/`](configs/). Replace `epic` with `grpo`, `dapo`,
`gspo`, `ctpo`, or `ripo` to run a baseline. Each method has one editable
configuration; adjust it for your experiment and hardware.

## Evaluation and outputs

Validation is integrated into training. The default setup evaluates before
training and every five steps on AIME 2024, AIME 2025, AMC 2023 and HMMT 2025.
The metric is mean@8: average correctness over eight responses, not pass@8.

The output directory contains the resolved configuration, `metrics.jsonl`,
validation outputs and retained checkpoints. External tracking is disabled
by default. Checkpoint retention is configured under `trainer` in the JSON
configuration.

## Tests

```bash
uv run --with torch --with numpy python -m unittest discover -s tests -v
```

Tests cover loss computation, prefix gradients, KL-coordinate roots and launch
configuration checks. The optional `epic/loss.py` is a standalone CPU reference;
training uses the verl implementation.

## License

EPIC additions use **Apache-2.0**. Upstream verl notices are retained; see
[LICENSE](LICENSE) and [NOTICE](NOTICE). Models and datasets retain their own
licenses.
