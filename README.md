# EPIC

A [verl](https://github.com/verl-project/verl)-based implementation of
**Exact-KL Prefix Importance-Ratio Clipping (EPIC)** for off-policy LLM
reinforcement learning.

## Highlights

- **Prefix-aware policy optimization.** EPIC accounts for changes in both the
  generated context and the current token, using a cumulative prefix ratio
  instead of a token-only ratio or a shared full-response ratio.
- **Exact-KL clipping geometry.** Starting from an autoregressive
  performance-difference identity and a TRPO-inspired prefix trust region,
  EPIC derives asymmetric clipping bounds from the exact KL coordinate
  $\phi(r)=r-1-\log r$, without a local quadratic approximation.
- **Strong off-policy reasoning results.** On Qwen3-30B-A3B-Base, EPIC reaches
  **39.34%** average mean@8 across four mathematical-reasoning benchmarks:
  **+6.50 percentage points over GSPO** and **+10.69 over GRPO**.
- **No expert-routing replay.** In a separate routing-replay comparison, EPIC
  achieves competitive performance with **77.5% / 93.0% lower median
  driver-process CPU memory** than R2 / R3, respectively. These are not GPU
  memory or total-cluster memory reductions.

## Method at a glance

For a prompt $x$ and generated prefix $y_{1:t}$, EPIC uses

$$
r_t^{\mathrm{pre}}
=\frac{\pi_\theta(y_{1:t}\mid x)}{\mu(y_{1:t}\mid x)}
=\prod_{k=1}^{t}
\frac{\pi_\theta(y_k\mid x,y_{<k})}{\mu(y_k\mid x,y_{<k})}.
$$

This is the exact change-of-measure factor for each stepwise term in the
autoregressive performance-difference identity. For policies with common
support, the behavior-policy expectation of $\phi(r_t^{\mathrm{pre}})$ equals
the KL divergence $D_{\mathrm{KL}}(\mu_{1:t}\|\pi_{\theta,1:t})$ between their
prefix distributions. EPIC uses this coordinate to construct asymmetric
prefix-ratio bounds, then implements a PPO-style clipped surrogate with
geometric-mean normalization, probability-calibrated budgets, and a
current-token-only gradient path. The exact identity motivates the clipping
design; the practical surrogate does not enforce a hard distribution-level
KL constraint.

## Main results

Off-policy training on **Qwen3-30B-A3B-Base** with an **update depth of eight**.
Scores are **mean@8 (%)**, averaged over independent runs. For each run, one
checkpoint is selected by the equal-weight average across the four validation
benchmarks; all four scores come from that same checkpoint.

| Method | AIME 2024 | AIME 2025 | AMC 2023 | HMMT 2025 | Average |
|:--|--:|--:|--:|--:|--:|
| GRPO | 23.61 | 16.81 | 69.48 | 4.72 | 28.65 |
| DAPO | 31.67 | 24.03 | 76.25 | 9.86 | 35.45 |
| GSPO | 30.00 | 20.56 | 73.44 | 7.36 | 32.84 |
| CTPO | 30.14 | 20.97 | 76.04 | 8.75 | 33.98 |
| RIPO | 26.94 | 21.81 | 74.69 | 8.47 | 32.98 |
| **EPIC** | **38.06** | **25.28** | **77.92** | **16.11** | **39.34** |

EPIC ranks first on all four benchmarks in this comparison. Mean@8 measures
average correctness across eight sampled responses, **not pass@8**.

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
