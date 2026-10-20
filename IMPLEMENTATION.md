# Implementation

Policy objectives are registered in
`vendor/verl/verl/trainer/ppo/core_algos.py`.

| Configuration | Registered loss |
|---|---|
| `epic` | `prefix_geometric_probability_weighted_exact_kl_dual_clip` |
| `grpo`, `dapo` | `vanilla` (different clipping parameters) |
| `gspo` | `gspo` |
| `ctpo` | `cum-token-cumprod-la` |
| `ripo` | `ripo_clip` |

## EPIC

- `_solve_exact_prefix_kl_log_bounds` inverts `exp(z) - 1 - z` in log space.
- `_compute_policy_loss_prefix_exact_kl_clip` implements the shared clipping path.
- `compute_policy_loss_prefix_geometric_probability_weighted_exact_kl_dual_clip`
  selects EPIC's probability calibration and optimization ratio.

For a prefix of length `t`, the budget uses `mu(prefix) ** (1/t)` and the
optimization ratio uses `r_prefix ** (1/t)`. Forward prefix weights are detached;
gradients flow through the current token. The negative-advantage dual clip uses
the prefix-dependent upper endpoint. Prefix variants aggregate token means
within each response, then average responses.

The numerical implementation includes a small-budget series to avoid
cancellation and caps optimization log-weights to `[-10, 10]`. The optional
standalone `epic/loss.py` is tested against the training loss but does not
implement rollout, reward computation or distributed scaling.

## Training and evaluation

- `verl/trainer/main_ppo.py`: training entry point used by the launcher.
- `verl/trainer/ppo/ray_trainer.py`: validation and checkpointing.
- `verl/workers/engine/megatron/`: actor training.
- `verl/workers/rollout/vllm_rollout/`: rollout generation.
- `verl/utils/reward_score/__init__.py`: reward dispatch by `data_source`.
- `math_reward.py` and `math_benchmark.py` in that reward directory: training
  verification and benchmark answer checking.

The framework retains upstream functionality beyond these example configs.
Use the included snapshot when running them; compatibility with other verl
versions is not implied.
