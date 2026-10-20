# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import json
import os
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
_PREFIX_CLIP_DIAG_CALL_COUNT = 0
_IS_LOG_RATIO_SAFETY_BOUND = 20.0


def _dump_prefix_clip_response_diagnostics(
    *,
    loss_name: str,
    response_mask: torch.Tensor,
    clipped: torch.Tensor,
    lower_clipped: torch.Tensor,
    prefix_avg_log_ratio: torch.Tensor,
    eps_pos: torch.Tensor,
    eps_neg: torch.Tensor,
) -> None:
    """Dump per-response prefix clipping diagnostics to a sidecar JSONL file.

    The normal metrics path is shared by console/file/tensorboard loggers and
    therefore should stay scalar-only. This sidecar keeps complete per-response
    values for offline analysis without breaking TensorBoard.
    """

    if os.getenv("VERL_PREFIX_CLIP_DIAG_DISABLE", "0") == "1":
        return

    metrics_path = os.getenv("VERL_FILE_LOGGER_PATH")
    diag_dir = os.getenv("VERL_PREFIX_CLIP_DIAG_DIR")
    if diag_dir is None:
        if not metrics_path:
            return
        diag_dir = os.path.join(os.path.dirname(metrics_path), "prefix_clip_diagnostics")

    global _PREFIX_CLIP_DIAG_CALL_COUNT
    _PREFIX_CLIP_DIAG_CALL_COUNT += 1

    try:
        os.makedirs(diag_dir, exist_ok=True)
        pid = os.getpid()
        path = os.path.join(diag_dir, f"{loss_name}_pid{pid}.jsonl")

        with torch.no_grad():
            mask = response_mask.bool()
            lengths = response_mask.sum(dim=-1).clamp(min=1.0)
            clipped_valid = clipped & mask
            lower_clipped_valid = lower_clipped & mask
            positions = torch.cumsum(response_mask, dim=-1)
            rel_positions = (positions / lengths.unsqueeze(-1)).clamp(min=0.0, max=1.0)

            clipped_count = clipped_valid.float().sum(dim=-1)
            lower_clipped_count = lower_clipped_valid.float().sum(dim=-1)
            any_clip = clipped_valid.any(dim=-1)
            any_lower_clip = lower_clipped_valid.any(dim=-1)

            clipped_rel = torch.where(clipped_valid, rel_positions, torch.zeros_like(rel_positions))
            first_clip_rel = torch.where(
                any_clip,
                torch.where(clipped_valid, rel_positions, torch.full_like(rel_positions, 2.0)).min(dim=-1).values,
                torch.full_like(lengths, -1.0),
            )
            last_clip_rel = torch.where(
                any_clip,
                clipped_rel.max(dim=-1).values,
                torch.full_like(lengths, -1.0),
            )
            mean_clip_rel = torch.where(
                any_clip,
                clipped_rel.sum(dim=-1) / clipped_count.clamp(min=1.0),
                torch.full_like(lengths, -1.0),
            )

            masked_prefix_abs = torch.where(
                mask, prefix_avg_log_ratio.detach().abs(), torch.zeros_like(prefix_avg_log_ratio)
            )
            masked_eps_pos = torch.where(mask, eps_pos.detach(), torch.zeros_like(eps_pos))
            masked_eps_neg = torch.where(mask, eps_neg.detach(), torch.zeros_like(eps_neg))

            payload: dict[str, Any] = {
                "loss_name": loss_name,
                "pid": pid,
                "call_index": _PREFIX_CLIP_DIAG_CALL_COUNT,
                "batch_size": int(response_mask.shape[0]),
                "response_length": lengths.detach().cpu().to(torch.int64).tolist(),
                "clipped_token_count": clipped_count.detach().cpu().to(torch.int64).tolist(),
                "lower_clipped_token_count": lower_clipped_count.detach().cpu().to(torch.int64).tolist(),
                "clip_token_fraction": (clipped_count / lengths).detach().cpu().tolist(),
                "lower_clip_token_fraction": (lower_clipped_count / lengths).detach().cpu().tolist(),
                "any_clip": any_clip.detach().cpu().to(torch.int64).tolist(),
                "any_lower_clip": any_lower_clip.detach().cpu().to(torch.int64).tolist(),
                "first_clip_rel_pos": first_clip_rel.detach().cpu().tolist(),
                "last_clip_rel_pos": last_clip_rel.detach().cpu().tolist(),
                "mean_clip_rel_pos": mean_clip_rel.detach().cpu().tolist(),
                "avg_log_ratio_abs_mean": (
                    masked_prefix_abs.sum(dim=-1) / lengths
                ).detach().cpu().tolist(),
                "eps_pos_mean": (masked_eps_pos.sum(dim=-1) / lengths).detach().cpu().tolist(),
                "eps_neg_mean": (masked_eps_neg.sum(dim=-1) / lengths).detach().cpu().tolist(),
            }

            # Optional raw clipped-token positions. This can become very large
            # for long responses, so keep it opt-in.
            if os.getenv("VERL_PREFIX_CLIP_DIAG_FULL_POSITIONS", "0") == "1":
                payload["clip_rel_positions"] = [
                    rel_positions[row_idx][clipped_valid[row_idx]].detach().cpu().tolist()
                    for row_idx in range(response_mask.shape[0])
                ]
                payload["lower_clip_rel_positions"] = [
                    rel_positions[row_idx][lower_clipped_valid[row_idx]].detach().cpu().tolist()
                    for row_idx in range(response_mask.shape[0])
                ]

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        # Diagnostics must never affect training.
        return


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name == 'unclipped_is' and loss_name not in POLICY_LOSS_REGISTRY:
        from verl.trainer.ppo.unclipped_is import compute_policy_loss_unclipped_is
        POLICY_LOSS_REGISTRY[loss_name] = compute_policy_loss_unclipped_is
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    VTRACE = "vtrace"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.VTRACE)
def compute_vtrace_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    gamma: float,
    rho_bar: float = 10.0,
    c_bar: float = 1.0,
):
    """Compute V-trace targets using token-level current/behavior policy ratios."""
    with torch.no_grad():
        log_ratio = torch.clamp(
            current_log_probs - rollout_log_probs,
            min=-_IS_LOG_RATIO_SAFETY_BOUND,
            max=_IS_LOG_RATIO_SAFETY_BOUND,
        )
        rho = torch.exp(log_ratio) * response_mask
        rho_bar_t = torch.clamp(rho, max=float(rho_bar))
        c_t = torch.clamp(rho, max=float(c_bar))
        vtrace_reversed = []
        advantage_reversed = []
        next_value = torch.zeros_like(values[:, 0])
        next_vtrace = torch.zeros_like(values[:, 0])
        for t in reversed(range(values.shape[-1])):
            valid = response_mask[:, t]
            delta = rho_bar_t[:, t] * (token_level_rewards[:, t] + gamma * next_value - values[:, t])
            vtrace_t = values[:, t] + delta + gamma * c_t[:, t] * (next_vtrace - next_value)
            vtrace_t = vtrace_t * valid
            # The actor applies token-level IS in the policy loss. Keep the
            # advantage itself unweighted here to avoid applying rho twice.
            advantage_t = token_level_rewards[:, t] + gamma * next_vtrace - values[:, t]
            advantage_t = advantage_t * valid
            vtrace_reversed.append(vtrace_t)
            advantage_reversed.append(advantage_t)
            next_value = values[:, t] * valid
            next_vtrace = vtrace_t
        returns = torch.stack(vtrace_reversed[::-1], dim=1)
        advantages = torch.stack(advantage_reversed[::-1], dim=1)
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GDPO)  # or simply: @register_adv_est("gdpo")
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Group reward-Decoupled Normalization Policy Optimization.

    Instead of summing all reward dimensions first (like GRPO), GDPO normalizes
    each reward dimension independently within each group before aggregation.
    This prevents a dominant reward signal from drowning out weaker ones.

    Mathematical formulation:
        Step 1 – Group-wise decoupled normalization (via GRPO per dimension):
            For each reward dimension k, within each group g:
            A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)

        Step 2 – Weighted aggregation:
            A_sum = Σ_k w_k · A_k

        Step 3 – Batch-level normalization (via masked_whiten):
            A_final = whiten(A_sum, response_mask)

    Args:
        token_level_rewards: (bs, response_length) – standard token-level rewards.
            Used as fallback when per-dimension rewards are not provided.
        response_mask: (bs, response_length)
        index: (bs,) – group id per sample (from ``uid``).
        epsilon: Numerical stability constant.
        norm_adv_by_std_in_grpo: Whether to normalize by std in GRPO.
        config: Algorithm configuration (optional).
        non_tensor_batch: Non-tensor batch data containing per-dimension reward scores.
        batch: Batch data containing prompts, attention_mask, etc.

    Note:
        Ref GDPO (https://arxiv.org/abs/2601.05242).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) – same as advantages (outcome-only).
    """
    score_list = None
    reward_weights = None

    if config is not None and non_tensor_batch is not None and batch is not None:
        gdpo_reward_keys = config.get("gdpo_reward_keys", None)
        assert gdpo_reward_keys, (
            "GDPO requires 'algorithm.gdpo_reward_keys' listing the individual reward "
            "component keys returned by compute_score (e.g. ['format_reward', 'accuracy_reward'])."
        )
        device = token_level_rewards.device
        prompt_length = batch["prompts"].size(1)
        valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        score_list = []
        for key in gdpo_reward_keys:
            assert key in non_tensor_batch, (
                f"GDPO reward key '{key}' not found in non_tensor_batch. "
                f"Available keys: {list(non_tensor_batch.keys())}. "
                f"Make sure your compute_score returns a dict containing '{key}'."
            )
            comp = non_tensor_batch[key]
            rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0), device=device), valid_response_length] = rm_score
            score_list.append(rm_scores)

        gdpo_weights = config.get("gdpo_reward_weights", None)
        if gdpo_weights is not None:
            reward_weights = list(gdpo_weights)

    if score_list is None:
        score_list = [token_level_rewards]

    num_scores = len(score_list)

    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=token_level_rewards.device)
    else:
        weights = torch.ones(num_scores, dtype=torch.float32, device=token_level_rewards.device)

    new_advantage = None

    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score

    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("ripo_clip")
def compute_policy_loss_ripo_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the paper-exact Riemannian Isometric Policy Optimization loss.

    RIPO replaces PPO's fixed ratio deviation with the probability-dependent
    Riemannian Isometric Clip from Eq. 11 of the paper:

        epsilon(pi_old(a|s)) = sqrt(delta / pi_old(a|s)).

    The paper uses a symmetric ``delta=0.05`` by default and additionally
    truncates token importance ratios to ``[0.5, 10]``. Its LLM objective uses
    token-level ratios, group-relative advantages, and token-mean aggregation.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    if loss_agg_mode != "token-mean":
        raise ValueError(f"RIPO requires paper token-mean aggregation, got {loss_agg_mode!r}.")

    policy_loss_cfg = config.policy_loss
    delta_low = float(policy_loss_cfg.get("ripo_delta_low", 0.05))
    delta_high = float(policy_loss_cfg.get("ripo_delta_high", 0.05))
    ratio_lower = float(policy_loss_cfg.get("ripo_ratio_lower", 0.5))
    ratio_upper = float(policy_loss_cfg.get("ripo_ratio_upper", 10.0))
    if delta_low <= 0.0 or delta_high <= 0.0:
        raise ValueError("ripo_delta_low/high must both be positive.")
    if not 0.0 < ratio_lower < 1.0 < ratio_upper:
        raise ValueError("RIPO ratio bounds must satisfy 0 < lower < 1 < upper.")

    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    raw_ratio = torch.exp(log_ratio)
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    # Compute the action probability and Riemannian radius in float32. This
    # avoids BF16 underflow for rare tokens; the paper's finite ratio bounds
    # then keep the effective clipping interval well-conditioned.
    old_prob = torch.exp(old_log_prob.detach().float()).clamp_min(torch.finfo(torch.float32).tiny)
    epsilon_low = torch.sqrt(delta_low / old_prob)
    epsilon_high = torch.sqrt(delta_high / old_prob)
    dynamic_lower = 1.0 - epsilon_low
    dynamic_upper = 1.0 + epsilon_high
    effective_lower = torch.maximum(dynamic_lower, torch.full_like(dynamic_lower, ratio_lower))
    effective_upper = torch.minimum(dynamic_upper, torch.full_like(dynamic_upper, ratio_upper))

    # The paper applies dual-sided truncation [0.5, 10] to the token IS ratio,
    # followed by the probability-dependent Riemannian clipping interval.
    truncated_ratio = torch.clamp(raw_ratio, min=ratio_lower, max=ratio_upper)
    clipped_ratio = torch.minimum(torch.maximum(truncated_ratio, effective_lower), effective_upper)

    pg_losses_unclipped = -advantages * truncated_ratio
    pg_losses_clipped = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses_unclipped, pg_losses_clipped)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="token-mean",
        **config.global_batch_info,
    )

    ratio_lower_clipped = raw_ratio < ratio_lower
    ratio_upper_clipped = raw_ratio > ratio_upper
    dynamic_lower_clipped = truncated_ratio < effective_lower
    dynamic_upper_clipped = truncated_ratio > effective_upper
    lower_clipped = ratio_lower_clipped | dynamic_lower_clipped
    upper_clipped = ratio_upper_clipped | dynamic_upper_clipped
    clipped = lower_clipped | upper_clipped

    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(lower_clipped.float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ripo_clip/delta_low": delta_low,
        "actor/ripo_clip/delta_high": delta_high,
        "actor/ripo_clip/ratio_lower": ratio_lower,
        "actor/ripo_clip/ratio_upper": ratio_upper,
        "actor/ripo_clip/old_action_prob_mean": verl_F.masked_mean(old_prob, response_mask).detach().item(),
        "actor/ripo_clip/epsilon_low_mean": verl_F.masked_mean(epsilon_low, response_mask).detach().item(),
        "actor/ripo_clip/epsilon_high_mean": verl_F.masked_mean(epsilon_high, response_mask).detach().item(),
        "actor/ripo_clip/lower_bound_mean": verl_F.masked_mean(effective_lower, response_mask).detach().item(),
        "actor/ripo_clip/upper_bound_mean": verl_F.masked_mean(effective_upper, response_mask).detach().item(),
        "actor/ripo_clip/ratio_lower_clipfrac": verl_F.masked_mean(
            ratio_lower_clipped.float(), response_mask
        )
        .detach()
        .item(),
        "actor/ripo_clip/ratio_upper_clipfrac": verl_F.masked_mean(
            ratio_upper_clipped.float(), response_mask
        )
        .detach()
        .item(),
        "actor/ripo_clip/dynamic_lower_clipfrac": verl_F.masked_mean(
            dynamic_lower_clipped.float(), response_mask
        )
        .detach()
        .item(),
        "actor/ripo_clip/dynamic_upper_clipfrac": verl_F.masked_mean(
            dynamic_upper_clipped.float(), response_mask
        )
        .detach()
        .item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finally exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("prefix_dynamic_clip")
def compute_policy_loss_prefix_dynamic_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO-style loss with dynamic clipping on average prefix log-ratios.

    Compared with GSPO, which uses one sequence-level geometric-mean ratio for
    every token in a response, this loss uses a token-dependent prefix ratio:

        r_pre,i,t(theta) =
            exp((1 / t) * sum_{k <= t}
                log pi_theta(o_i,k | q_i, o_i,<k)
              - log pi_old(o_i,k | q_i, o_i,<k))

    The clipped ratio is obtained by clipping the average prefix log-ratio in
    log space. The clip boundary is token-position dependent: it starts from a
    DAPO/GRPO-like wide token-level boundary at the first response token and
    decays as 1/t to a GSPO-like narrow boundary at the last response token.

    As in GSPO, the numerical importance-ratio value is detached from the
    gradient path. The gradient is routed only through the current-token
    log-probability:

        log r_for_grad,i,t =
            log pi_theta(o_i,t | q_i, o_i,<t)
          - sg[log pi_theta(o_i,t | q_i, o_i,<t)]
          + sg[log r_pre,i,t(theta)]

    This keeps the forward value equal to the prefix ratio, while avoiding
    repeated backpropagation through all earlier prefix tokens.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    eps_1_low = float(policy_loss_cfg.get("prefix_clip_first_low", 0.2))
    eps_1_high = float(policy_loss_cfg.get("prefix_clip_first_high", 0.28))
    eps_t_low = float(policy_loss_cfg.get("prefix_clip_final_low", 3e-4))
    eps_t_high = float(policy_loss_cfg.get("prefix_clip_final_high", 4e-4))

    negative_approx_kl = log_prob - old_log_prob

    # compute token-level log-ratio:
    # l_i,t(theta) =
    #   log pi_theta(o_i,t | q_i, o_i,<t)
    # - log pi_old(o_i,t | q_i, o_i,<t)
    # Only response tokens should contribute to prefix statistics.
    log_ratio = negative_approx_kl * response_mask

    # compute average prefix log-ratio:
    # z_i,t(theta) = (1 / t) * sum_{k <= t} l_i,k(theta)
    #
    # prefix_len is the valid response-token prefix length at each token
    # position. Padding positions are clamped to one to avoid division by zero;
    # they are masked out again in loss/metric aggregation.
    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)
    prefix_avg_log_ratio = prefix_log_ratio_sum / prefix_len

    # First-token boundaries are specified in ratio space, matching the usual
    # asymmetric PPO/DAPO-style clip interval [1 - eps_low, 1 + eps_high].
    # Convert them to positive log-space magnitudes so the actual interval is
    # [-eps_1_neg, eps_1_pos].
    eps_1_pos = torch.log(torch.tensor(1.0 + eps_1_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_1_neg = -torch.log(torch.tensor(1.0 - eps_1_low, device=log_prob.device, dtype=log_prob.dtype))

    # Final-token boundaries follow the same ratio-space semantics as GSPO's
    # clip interval [1 - eps_low, 1 + eps_high]. Convert them to positive
    # log-space magnitudes before interpolating in log-ratio space.
    eps_t_pos = torch.log(torch.tensor(1.0 + eps_t_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_neg = -torch.log(torch.tensor(1.0 - eps_t_low, device=log_prob.device, dtype=log_prob.dtype))

    # Interpolate clip boundaries from the first-token boundary to the
    # final-token boundary with normalized 1/t decay:
    #
    # lambda_t = (1/t - 1/T) / (1 - 1/T)
    #
    # Thus lambda_1 = 1 and lambda_T = 0. The response_mask multiplication keeps
    # padding positions inert.
    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    lambda_t = (1.0 / prefix_len - 1.0 / seq_len) / (1.0 - 1.0 / seq_len).clamp(min=1e-8)
    lambda_t = lambda_t.clamp(min=0.0, max=1.0) * response_mask

    eps_pos = eps_t_pos + (eps_1_pos - eps_t_pos) * lambda_t
    eps_neg = eps_t_neg + (eps_1_neg - eps_t_neg) * lambda_t

    # Clip the average prefix log-ratio in log space:
    #
    # z_clip,i,t = clamp(z_i,t, -eps_neg,i,t, eps_pos,i,t)
    #
    # The outer [-10, 10] clamp is only a numerical guard before exp().
    prefix_avg_log_ratio = torch.clamp(prefix_avg_log_ratio, min=-10.0, max=10.0)
    prefix_avg_log_ratio_clip = torch.minimum(torch.maximum(prefix_avg_log_ratio, -eps_neg), eps_pos)

    # Route gradients through only the current-token log-probability, as in
    # GSPO, while using the average prefix log-ratio as the detached numerical
    # importance-ratio value. The clipped branch uses torch.clamp so out-of-bound
    # samples stop contributing gradients in the clipped direction.
    prefix_log_importance_ratio = log_prob - log_prob.detach() + prefix_avg_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-10.0, max=10.0)
    prefix_log_importance_ratio_clip = torch.minimum(torch.maximum(prefix_log_importance_ratio, -eps_neg), eps_pos)

    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)

    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Match the GSPO baseline aggregation so the experiment only changes the
    # importance-ratio clipping rule.
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    clipped = torch.ne(prefix_avg_log_ratio, prefix_avg_log_ratio_clip)
    lower_clipped = clipped & (prefix_avg_log_ratio < -eps_neg)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(lower_clipped.float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    _dump_prefix_clip_response_diagnostics(
        loss_name="prefix_dynamic_clip",
        response_mask=response_mask,
        clipped=clipped,
        lower_clipped=lower_clipped,
        prefix_avg_log_ratio=prefix_avg_log_ratio,
        eps_pos=eps_pos,
        eps_neg=eps_neg,
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/prefix_clip/eps_first_low": eps_1_low,
        "actor/prefix_clip/eps_first_high": eps_1_high,
        "actor/prefix_clip/eps_final_low": eps_t_low,
        "actor/prefix_clip/eps_final_high": eps_t_high,
        "actor/prefix_clip/avg_log_ratio_mean": verl_F.masked_mean(prefix_avg_log_ratio.detach(), response_mask).item(),
        "actor/prefix_clip/avg_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), response_mask
        ).item(),
    }

    # Diagnostics only: where does prefix clipping happen along the response?
    #
    # Buckets are based on normalized response position t / T. These metrics
    # help distinguish "middle tokens are over-constrained" from "only the
    # final segment approaches the GSPO-like endpoint bound".
    relative_pos = (prefix_len / seq_len).clamp(min=0.0, max=1.0)
    valid_response_mask = response_mask.bool()
    for bucket_idx in range(10):
        left = bucket_idx / 10.0
        right = (bucket_idx + 1) / 10.0
        if bucket_idx == 0:
            bucket_mask = valid_response_mask & (relative_pos >= left) & (relative_pos <= right)
        else:
            bucket_mask = valid_response_mask & (relative_pos > left) & (relative_pos <= right)
        bucket_mask_float = bucket_mask.float()
        bucket_prefix = f"actor/prefix_clip/pos_bucket_{bucket_idx:02d}_{bucket_idx + 1:02d}"
        pg_metrics[f"{bucket_prefix}/token_frac"] = verl_F.masked_mean(
            bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac_lower"] = verl_F.masked_mean(
            lower_clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/avg_log_ratio_abs_mean"] = verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/eps_pos_mean"] = verl_F.masked_mean(
            eps_pos.detach(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/eps_neg_mean"] = verl_F.masked_mean(
            eps_neg.detach(), bucket_mask_float
        ).detach().item()

    # Diagnostics only: do shorter and longer responses experience different
    # clipping pressure? For each response-length bucket, record both token-level
    # clip fraction and sequence-level "any token clipped" fraction.
    seq_len_flat = seq_len.squeeze(-1)
    clipped_valid = clipped & valid_response_mask
    clipped_lower_valid = clipped_valid & (prefix_avg_log_ratio < -eps_neg)
    length_buckets = [
        (0, 1024, "00000_01024"),
        (1024, 2048, "01024_02048"),
        (2048, 4096, "02048_04096"),
        (4096, 8192, "04096_08192"),
        (8192, 12288, "08192_12288"),
        (12288, 16000, "12288_16000"),
        (16000, None, "16000_plus"),
    ]
    for lower, upper, label in length_buckets:
        if upper is None:
            seq_bucket_mask = seq_len_flat >= lower
        else:
            seq_bucket_mask = (seq_len_flat >= lower) & (seq_len_flat < upper)
        seq_bucket_mask_float = seq_bucket_mask.float()
        token_bucket_mask = valid_response_mask & seq_bucket_mask.unsqueeze(-1)
        token_bucket_mask_float = token_bucket_mask.float()
        seq_bucket_denom = seq_bucket_mask_float.sum().clamp(min=1.0)
        len_prefix = f"actor/prefix_clip/response_len_bucket_{label}"
        pg_metrics[f"{len_prefix}/response_frac"] = seq_bucket_mask_float.mean().detach().item()
        pg_metrics[f"{len_prefix}/token_frac"] = verl_F.masked_mean(
            token_bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_len_mean"] = (
            (seq_len_flat.detach() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac_lower"] = verl_F.masked_mean(
            clipped_lower_valid.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_clipfrac"] = (
            (clipped_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_lower_clipfrac"] = (
            (clipped_lower_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()
    return pg_loss, pg_metrics


def _solve_exact_prefix_kl_log_bounds(log_budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``exp(z) - 1 - z = exp(log_budget)`` on both branches.

    ``z`` is the cumulative prefix log-ratio. The equation has one negative
    and one positive root for every positive budget. A direct ratio-space
    solve is numerically unusable for long prefixes because
    ``t * delta / pi_old(prefix)`` quickly exceeds the floating-point range.
    This routine therefore works from its logarithm and uses:

    - a local series for very small budgets;
    - Newton iterations while the budget is directly representable;
    - asymptotically exact log-space updates for large budgets.

    The returned tensors are detached trust-region boundaries.
    """

    with torch.no_grad():
        work_dtype = torch.float32 if log_budget.dtype in (torch.float16, torch.bfloat16) else log_budget.dtype
        log_c = log_budget.detach().to(work_dtype)

        # The local roots are z+ = sqrt(2c) - c/3 + O(c^(3/2)) and
        # z- = -sqrt(2c) - c/3 + O(c^(3/2)).
        c_for_newton = torch.exp(torch.clamp(log_c, max=12.0))
        sqrt_2c = torch.sqrt(2.0 * c_for_newton)
        use_series = c_for_newton < 1e-4
        upper_series = (sqrt_2c - c_for_newton / 3.0).clamp_min(0.0)
        lower_series = -(sqrt_2c + c_for_newton / 3.0)

        # Positive root. expm1 keeps the Newton residual accurate near zero.
        upper_newton = torch.where(c_for_newton < 1.0, sqrt_2c, torch.log1p(c_for_newton) + 0.5)
        for _ in range(4):
            expm1_upper = torch.expm1(upper_newton)
            residual = expm1_upper - upper_newton - c_for_newton
            upper_newton = (upper_newton - residual / expm1_upper.clamp_min(1e-12)).clamp_min(0.0)

        # For large c, exp(z) = c + 1 + z. Fixed-point updates in log-space
        # avoid ever materializing c and converge rapidly because z << c.
        upper_large = log_c.clamp_min(0.0)
        for _ in range(4):
            upper_large = torch.logaddexp(log_c, torch.log1p(upper_large))
        upper = torch.where(log_c > 12.0, upper_large, torch.where(use_series, upper_series, upper_newton))

        # Negative root with u = -z: exp(-u) - 1 + u = c. For large c,
        # u = c + 1 - exp(-u) is indistinguishable from c + 1. Overflow to
        # -inf is intentional: it denotes an inactive lower ratio boundary.
        lower_u = torch.where(c_for_newton < 1.0, sqrt_2c, c_for_newton + 1.0)
        for _ in range(4):
            residual = torch.exp(-lower_u) - 1.0 + lower_u - c_for_newton
            derivative = (-torch.expm1(-lower_u)).clamp_min(1e-12)
            lower_u = (lower_u - residual / derivative).clamp_min(0.0)
        lower_newton = -lower_u
        lower_large = -(torch.exp(log_c) + 1.0)
        lower = torch.where(log_c > 12.0, lower_large, torch.where(use_series, lower_series, lower_newton))

        return lower.to(log_budget.dtype), upper.to(log_budget.dtype)


@register_policy_loss("prefix_ripo_clip")
def compute_policy_loss_prefix_ripo_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average-prefix surrogate with RIPO-like prefix likelihood-ratio clipping.

    This variant keeps the same surrogate semantics as ``prefix_dynamic_clip``:
    the forward importance ratio for token ``t`` is the average prefix ratio

        r_pre,i,t = exp((1 / t) * sum_{k <= t} log r_i,k),

    and the gradient is locked to the current token as in GSPO. The only change
    is the clipping rule. Instead of prescribing a position-only schedule for
    the average prefix log-ratio, it follows the RIPO-like section of the design
    note and clips the prefix likelihood ratio

        R_pre,i,t = exp(L_i,t),  L_i,t = sum_{k <= t} log r_i,k,

    with a per-prefix coordinate bound determined by the old-policy prefix
    probability and a length-scaled prefix KL budget. Here
    ``prefix_ripo_delta_low/high`` are average-token KL budgets, so the prefix
    budget at length ``t`` is ``t * delta``:

        0.5 * pi_old(o_i,<=t | q) * (R_pre,i,t - 1)^2 <= t * delta.

    In log space this gives

        L_i,t <= log(1 + sqrt(2 * t * delta_high / pi_old_prefix))

    and, when the lower bound is positive,

        L_i,t >= log(1 - sqrt(2 * t * delta_low / pi_old_prefix)).

    The clipped likelihood-ratio bound is converted back to the average-prefix
    ratio by dividing by the prefix length before exponentiation.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    delta_low = float(policy_loss_cfg.get("prefix_ripo_delta_low", 1e-5))
    delta_high = float(policy_loss_cfg.get("prefix_ripo_delta_high", 3e-5))
    if delta_low <= 0.0 or delta_high <= 0.0:
        raise ValueError("prefix_ripo_delta_low/high must both be positive.")

    negative_approx_kl = log_prob - old_log_prob
    log_ratio = negative_approx_kl * response_mask

    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)
    prefix_avg_log_ratio = prefix_log_ratio_sum / prefix_len

    # log pi_old(o_i,<=t | q) = sum_{k <= t} log pi_old(o_i,k | q,o_i,<k).
    # Padding positions are inert because response_mask zeros them before cumsum.
    old_prefix_log_prob = torch.cumsum(old_log_prob * response_mask, dim=-1)

    log_2_delta_high = torch.tensor(
        np.log(2.0 * delta_high), device=log_prob.device, dtype=log_prob.dtype
    )
    log_2_delta_low = torch.tensor(
        np.log(2.0 * delta_low), device=log_prob.device, dtype=log_prob.dtype
    )

    # stable log(1 + sqrt(2 * t * delta_high / pi_old_prefix)).
    # prefix_len is t for valid response positions. Padding positions are
    # ignored by response_mask downstream, so their bound values are inert.
    log_prefix_len = torch.log(prefix_len)
    high_radius_log = 0.5 * (log_2_delta_high + log_prefix_len - old_prefix_log_prob)
    upper_log_bound = torch.logaddexp(torch.zeros_like(high_radius_log), high_radius_log)

    # lower bound is active only when 1 - sqrt(2 * t * delta_low / pi_old_prefix) > 0.
    low_radius = torch.exp(torch.clamp(0.5 * (log_2_delta_low + log_prefix_len - old_prefix_log_prob), max=80.0))
    lower_ratio_bound = 1.0 - low_radius
    lower_active = lower_ratio_bound > 0.0
    neg_inf = torch.full_like(lower_ratio_bound, -torch.inf)
    lower_log_bound = torch.where(
        lower_active,
        torch.log(torch.clamp(lower_ratio_bound, min=torch.finfo(log_prob.dtype).tiny)),
        neg_inf,
    )

    # Clip the original prefix cumulative log-ratio L_i,t directly. Do not add
    # a fixed cap here: L_i,t naturally grows with prefix length, while the
    # RIPO-like bounds already account for both t and pi_old(o_i,<=t | q).
    # A fixed cumulative cap would change the derived trust region and can also
    # make clip diagnostics disagree with the surrogate for long responses.
    prefix_log_ratio_sum_clip = torch.minimum(torch.maximum(prefix_log_ratio_sum, lower_log_bound), upper_log_bound)

    # Gradient locking follows GSPO/prefix_dynamic_clip: the numerical ratio is
    # detached prefix information, while gradients flow through the current
    # token log-probability only. The clipped branch clamps the token-gradient
    # expression so clipped samples stop contributing in the clipped direction.
    lower_avg_log_bound = lower_log_bound / prefix_len
    upper_avg_log_bound = upper_log_bound / prefix_len
    prefix_log_importance_ratio = log_prob - log_prob.detach() + prefix_avg_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-10.0, max=10.0)
    prefix_log_importance_ratio_clip = torch.minimum(
        torch.maximum(prefix_log_importance_ratio, lower_avg_log_bound), upper_avg_log_bound
    )
    prefix_log_importance_ratio_clip = torch.clamp(prefix_log_importance_ratio_clip, min=-10.0, max=10.0)

    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)

    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    clipped = torch.ne(prefix_log_ratio_sum, prefix_log_ratio_sum_clip)
    lower_clipped = clipped & (prefix_log_ratio_sum < lower_log_bound)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(lower_clipped.float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    upper_avg_log_bound_detached = upper_avg_log_bound.detach()
    eps_pos = upper_avg_log_bound_detached
    eps_neg = torch.where(lower_active, -lower_avg_log_bound.detach(), torch.zeros_like(lower_avg_log_bound))

    _dump_prefix_clip_response_diagnostics(
        loss_name="prefix_ripo_clip",
        response_mask=response_mask,
        clipped=clipped,
        lower_clipped=lower_clipped,
        prefix_avg_log_ratio=prefix_avg_log_ratio,
        eps_pos=eps_pos,
        eps_neg=eps_neg,
    )

    lower_active_float = lower_active.float()
    finite_lower_avg_log_bound = torch.where(
        lower_active, lower_avg_log_bound.detach(), torch.zeros_like(lower_avg_log_bound)
    )
    response_mask_float = response_mask
    lower_active_count = (lower_active_float * response_mask_float).sum().clamp(min=1.0)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/prefix_ripo_clip/delta_low": delta_low,
        "actor/prefix_ripo_clip/delta_high": delta_high,
        "actor/prefix_ripo_clip/prefix_kl_budget_low_mean": verl_F.masked_mean(
            (prefix_len.detach() * delta_low), response_mask
        ).item(),
        "actor/prefix_ripo_clip/prefix_kl_budget_high_mean": verl_F.masked_mean(
            (prefix_len.detach() * delta_high), response_mask
        ).item(),
        "actor/prefix_ripo_clip/avg_log_ratio_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach(), response_mask
        ).item(),
        "actor/prefix_ripo_clip/avg_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), response_mask
        ).item(),
        "actor/prefix_ripo_clip/sum_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_log_ratio_sum.detach().abs(), response_mask
        ).item(),
        "actor/prefix_ripo_clip/upper_avg_log_bound_mean": verl_F.masked_mean(
            upper_avg_log_bound_detached, response_mask
        ).item(),
        "actor/prefix_ripo_clip/lower_bound_active_frac": verl_F.masked_mean(
            lower_active_float, response_mask
        ).item(),
        "actor/prefix_ripo_clip/lower_avg_log_bound_active_mean": (
            (finite_lower_avg_log_bound * lower_active_float * response_mask_float).sum() / lower_active_count
        )
        .detach()
        .item(),
    }

    relative_pos = (prefix_len / response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)).clamp(min=0.0, max=1.0)
    valid_response_mask = response_mask.bool()
    for bucket_idx in range(10):
        left = bucket_idx / 10.0
        right = (bucket_idx + 1) / 10.0
        if bucket_idx == 0:
            bucket_mask = valid_response_mask & (relative_pos >= left) & (relative_pos <= right)
        else:
            bucket_mask = valid_response_mask & (relative_pos > left) & (relative_pos <= right)
        bucket_mask_float = bucket_mask.float()
        bucket_prefix = f"actor/prefix_ripo_clip/pos_bucket_{bucket_idx:02d}_{bucket_idx + 1:02d}"
        pg_metrics[f"{bucket_prefix}/token_frac"] = verl_F.masked_mean(
            bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac_lower"] = verl_F.masked_mean(
            lower_clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/avg_log_ratio_abs_mean"] = verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/sum_log_ratio_abs_mean"] = verl_F.masked_mean(
            prefix_log_ratio_sum.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/upper_avg_log_bound_mean"] = verl_F.masked_mean(
            upper_avg_log_bound_detached, bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/lower_bound_active_frac"] = verl_F.masked_mean(
            lower_active_float, bucket_mask_float
        ).detach().item()

    # Diagnostics only: record clipping pressure by full response length. This
    # mirrors prefix_dynamic_clip and prefix_sqrt_dynamic_clip so RIPO-like runs
    # can be compared in the same post-hoc length-vs-clip plots.
    seq_len_flat = response_mask.sum(dim=-1)
    clipped_valid = clipped & valid_response_mask
    lower_clipped_valid = lower_clipped & valid_response_mask
    length_buckets = [
        (0, 1024, "00000_01024"),
        (1024, 2048, "01024_02048"),
        (2048, 4096, "02048_04096"),
        (4096, 8192, "04096_08192"),
        (8192, 12288, "08192_12288"),
        (12288, 16000, "12288_16000"),
        (16000, None, "16000_plus"),
    ]
    for lower, upper, label in length_buckets:
        if upper is None:
            seq_bucket_mask = seq_len_flat >= lower
        else:
            seq_bucket_mask = (seq_len_flat >= lower) & (seq_len_flat < upper)
        seq_bucket_mask_float = seq_bucket_mask.float()
        token_bucket_mask = valid_response_mask & seq_bucket_mask.unsqueeze(-1)
        token_bucket_mask_float = token_bucket_mask.float()
        seq_bucket_denom = seq_bucket_mask_float.sum().clamp(min=1.0)
        len_prefix = f"actor/prefix_ripo_clip/response_len_bucket_{label}"
        pg_metrics[f"{len_prefix}/response_frac"] = seq_bucket_mask_float.mean().detach().item()
        pg_metrics[f"{len_prefix}/token_frac"] = verl_F.masked_mean(
            token_bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_len_mean"] = (
            (seq_len_flat.detach() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac_lower"] = verl_F.masked_mean(
            lower_clipped_valid.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_clipfrac"] = (
            (clipped_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_lower_clipfrac"] = (
            (lower_clipped_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()

    return pg_loss, pg_metrics


def _compute_policy_loss_prefix_exact_kl_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    *,
    probability_weighted: bool,
    geometric_average_surrogate: bool,
    dual_clip_negative_advantage: bool,
    loss_name: str,
    geometric_probability_weighted: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prefix surrogate with exact cumulative-prefix clipping.

    The unweighted strategy applies

        R - 1 - log(R) <= t * delta.

    The probability-weighted strategy instead applies

        R - 1 - log(R) <= t * delta / pi_old(prefix).

    The geometric-probability-weighted strategy length-normalizes the old
    prefix probability before applying the same coordinate constraint:

        R - 1 - log(R) <= t * delta / pi_old(prefix) ** (1 / t).

    This retains a per-token typicality correction without making the budget
    grow exponentially with prefix length.

    where ``R = exp(sum_{k<=t} log r_k)`` is the cumulative prefix likelihood
    ratio. Separate low/high deltas define the two roots. The surrogate may
    use either the cumulative prefix ratio directly or its geometric average;
    both use the same exact cumulative-prefix clipping decisions.

    The prefix ratio value is detached and gradients flow only through the
    current-token log probability. The optional negative-advantage dual clip
    places a lower bound on the maximized surrogate (an upper bound on the
    minimized loss), preventing ``A < 0, R_prefix >> 1`` from selecting the
    unbounded raw PPO branch.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    delta_low = float(policy_loss_cfg.get("prefix_exact_kl_delta_low", 3.09912028333e-4))
    delta_high = float(policy_loss_cfg.get("prefix_exact_kl_delta_high", 5.17091807565e-3))
    if delta_low <= 0.0 or delta_high <= 0.0:
        raise ValueError("prefix_exact_kl_delta_low/high must both be positive.")
    if probability_weighted and geometric_probability_weighted:
        raise ValueError("full and geometric prefix-probability weighting are mutually exclusive.")

    negative_approx_kl = log_prob - old_log_prob
    log_ratio = negative_approx_kl * response_mask
    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)
    prefix_avg_log_ratio = prefix_log_ratio_sum / prefix_len
    old_prefix_log_prob = torch.cumsum(old_log_prob * response_mask, dim=-1)

    # Work with the logarithm of the right-hand side. For the weighted
    # coordinate constraint this also avoids materializing pi_old(prefix),
    # which underflows rapidly for long responses.
    log_prefix_len = torch.log(prefix_len)
    log_budget_low = log_prefix_len + np.log(delta_low)
    log_budget_high = log_prefix_len + np.log(delta_high)
    if probability_weighted:
        log_budget_low = log_budget_low - old_prefix_log_prob
        log_budget_high = log_budget_high - old_prefix_log_prob
    elif geometric_probability_weighted:
        old_prefix_avg_log_prob = old_prefix_log_prob / prefix_len
        log_budget_low = log_budget_low - old_prefix_avg_log_prob
        log_budget_high = log_budget_high - old_prefix_avg_log_prob
    lower_log_bound, _ = _solve_exact_prefix_kl_log_bounds(log_budget_low)
    _, upper_log_bound = _solve_exact_prefix_kl_log_bounds(log_budget_high)

    lower_avg_log_bound = lower_log_bound / prefix_len
    upper_avg_log_bound = upper_log_bound / prefix_len
    surrogate_log_ratio = prefix_avg_log_ratio if geometric_average_surrogate else prefix_log_ratio_sum
    surrogate_lower_log_bound = lower_avg_log_bound if geometric_average_surrogate else lower_log_bound
    surrogate_upper_log_bound = upper_avg_log_bound if geometric_average_surrogate else upper_log_bound
    surrogate_log_ratio_clip = torch.minimum(
        torch.maximum(surrogate_log_ratio, surrogate_lower_log_bound), surrogate_upper_log_bound
    )

    # The detached average-prefix term supplies the numerical importance
    # weight R_pre ** (1 / t). The zero-valued log_prob - log_prob.detach()
    # term preserves the existing current-token-only gradient route.
    prefix_log_importance_ratio = log_prob - log_prob.detach() + surrogate_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-10.0, max=10.0)
    prefix_log_importance_ratio_clip = torch.minimum(
        torch.maximum(prefix_log_importance_ratio, surrogate_lower_log_bound), surrogate_upper_log_bound
    )
    prefix_log_importance_ratio_clip = torch.clamp(prefix_log_importance_ratio_clip, min=-10.0, max=10.0)

    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)
    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    dual_clipped = torch.zeros_like(response_mask, dtype=torch.bool)
    if dual_clip_negative_advantage:
        dual_log_cap = torch.clamp(surrogate_upper_log_bound, min=-10.0, max=10.0)
        dual_ratio_cap = torch.exp(dual_log_cap)
        dual_loss_cap = -advantages * dual_ratio_cap
        dual_clipped = (advantages < 0) & (ratio_for_loss > dual_ratio_cap)
        pg_losses = torch.where(advantages < 0, torch.minimum(pg_losses, dual_loss_cap), pg_losses)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    clipped = torch.ne(surrogate_log_ratio, surrogate_log_ratio_clip)
    lower_clipped = clipped & (surrogate_log_ratio < surrogate_lower_log_bound)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(lower_clipped.float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    finite_lower = torch.isfinite(lower_log_bound)
    eps_pos = upper_avg_log_bound.detach()
    eps_neg = torch.where(finite_lower, -lower_avg_log_bound.detach(), torch.zeros_like(lower_avg_log_bound))

    _dump_prefix_clip_response_diagnostics(
        loss_name=loss_name,
        response_mask=response_mask,
        clipped=clipped,
        lower_clipped=lower_clipped,
        prefix_avg_log_ratio=prefix_avg_log_ratio,
        eps_pos=eps_pos,
        eps_neg=eps_neg,
    )

    response_mask_float = response_mask.float()
    finite_lower_float = finite_lower.float()
    finite_lower_count = (finite_lower_float * response_mask_float).sum().clamp(min=1.0)
    finite_lower_mean = (
        torch.where(finite_lower, lower_avg_log_bound.detach(), torch.zeros_like(lower_avg_log_bound))
        * finite_lower_float
        * response_mask_float
    ).sum() / finite_lower_count
    valid_raw_ratios = ratio_for_loss.detach()[response_mask.bool()].float()
    raw_ratio_p95 = (
        torch.quantile(valid_raw_ratios, 0.95)
        if valid_raw_ratios.numel()
        else torch.zeros((), device=ratio_for_loss.device)
    )

    metric_prefix = f"actor/{loss_name}"
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        f"{metric_prefix}/delta_low": delta_low,
        f"{metric_prefix}/delta_high": delta_high,
        f"{metric_prefix}/probability_weighted": float(probability_weighted),
        f"{metric_prefix}/geometric_probability_weighted": float(geometric_probability_weighted),
        f"{metric_prefix}/geometric_average_surrogate": float(geometric_average_surrogate),
        f"{metric_prefix}/cumulative_surrogate": float(not geometric_average_surrogate),
        f"{metric_prefix}/dual_clip_negative_advantage": float(dual_clip_negative_advantage),
        f"{metric_prefix}/dual_clipfrac": verl_F.masked_mean(
            dual_clipped.float(), response_mask
        ).detach().item(),
        f"{metric_prefix}/raw_ratio_mean": verl_F.masked_mean(
            ratio_for_loss.detach(), response_mask
        ).item(),
        f"{metric_prefix}/raw_ratio_p95": raw_ratio_p95.item(),
        f"{metric_prefix}/raw_ratio_max": torch.where(
            response_mask.bool(), ratio_for_loss.detach(), torch.zeros_like(ratio_for_loss)
        ).max().item(),
        f"{metric_prefix}/avg_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), response_mask
        ).item(),
        f"{metric_prefix}/sum_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_log_ratio_sum.detach().abs(), response_mask
        ).item(),
        f"{metric_prefix}/old_prefix_nll_per_token_mean": verl_F.masked_mean(
            (-old_prefix_log_prob.detach() / prefix_len), response_mask
        ).item(),
        f"{metric_prefix}/log_budget_low_mean": verl_F.masked_mean(
            log_budget_low.detach(), response_mask
        ).item(),
        f"{metric_prefix}/log_budget_high_mean": verl_F.masked_mean(
            log_budget_high.detach(), response_mask
        ).item(),
        f"{metric_prefix}/upper_avg_log_bound_mean": verl_F.masked_mean(
            upper_avg_log_bound.detach(), response_mask
        ).item(),
        f"{metric_prefix}/finite_lower_bound_frac": verl_F.masked_mean(
            finite_lower_float, response_mask
        ).item(),
        f"{metric_prefix}/finite_lower_avg_log_bound_mean": finite_lower_mean.detach().item(),
    }

    # Position metrics expose how the exact bound evolves along the response.
    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    relative_pos = (prefix_len / seq_len).clamp(min=0.0, max=1.0)
    valid_response_mask = response_mask.bool()
    for bucket_idx in range(10):
        left = bucket_idx / 10.0
        right = (bucket_idx + 1) / 10.0
        if bucket_idx == 0:
            bucket_mask = valid_response_mask & (relative_pos >= left) & (relative_pos <= right)
        else:
            bucket_mask = valid_response_mask & (relative_pos > left) & (relative_pos <= right)
        bucket_mask_float = bucket_mask.float()
        bucket_prefix = f"{metric_prefix}/pos_bucket_{bucket_idx:02d}_{bucket_idx + 1:02d}"
        pg_metrics[f"{bucket_prefix}/pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac_lower"] = verl_F.masked_mean(
            lower_clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/upper_avg_log_bound_mean"] = verl_F.masked_mean(
            upper_avg_log_bound.detach(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/finite_lower_bound_frac"] = verl_F.masked_mean(
            finite_lower_float, bucket_mask_float
        ).detach().item()

    return pg_loss, pg_metrics


@register_policy_loss("prefix_probability_weighted_exact_kl_clip")
def compute_policy_loss_prefix_probability_weighted_exact_kl_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact prefix KL-coordinate clip weighted by old prefix probability."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=True,
        geometric_average_surrogate=True,
        dual_clip_negative_advantage=False,
        loss_name="prefix_probability_weighted_exact_kl_clip",
    )


@register_policy_loss("prefix_geometric_probability_weighted_exact_kl_clip")
def compute_policy_loss_prefix_geometric_probability_weighted_exact_kl_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact-prefix clip weighted by geometric-mean old prefix probability."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=False,
        geometric_probability_weighted=True,
        geometric_average_surrogate=True,
        dual_clip_negative_advantage=False,
        loss_name="prefix_geometric_probability_weighted_exact_kl_clip",
    )


@register_policy_loss("prefix_geometric_probability_weighted_exact_kl_dual_clip")
def compute_policy_loss_prefix_geometric_probability_weighted_exact_kl_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Geometric-probability exact-prefix loss with negative-advantage dual clip."""
    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=False,
        geometric_probability_weighted=True,
        geometric_average_surrogate=True,
        dual_clip_negative_advantage=True,
        loss_name="prefix_geometric_probability_weighted_exact_kl_dual_clip",
    )


@register_policy_loss("prefix_geometric_probability_weighted_exact_kl_cumulative_dual_clip")
def compute_policy_loss_prefix_geometric_probability_weighted_exact_kl_cumulative_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep geom-old exact budgets and dual clipping; use cumulative update weights."""
    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=False,
        geometric_probability_weighted=True,
        geometric_average_surrogate=False,
        dual_clip_negative_advantage=True,
        loss_name="prefix_geometric_probability_weighted_exact_kl_cumulative_dual_clip",
    )


@register_policy_loss("prefix_probability_weighted_exact_kl_dual_clip")
def compute_policy_loss_prefix_probability_weighted_exact_kl_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Raw old-prefix-probability exact bounds with geometric surrogate and dual clip."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=True,
        geometric_probability_weighted=False,
        geometric_average_surrogate=True,
        dual_clip_negative_advantage=True,
        loss_name="prefix_probability_weighted_exact_kl_dual_clip",
    )


@register_policy_loss("prefix_probability_weighted_exact_kl_cumulative_dual_clip")
def compute_policy_loss_prefix_probability_weighted_exact_kl_cumulative_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Probability-weighted exact bounds with cumulative-prefix surrogate and dual clip."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=True,
        geometric_average_surrogate=False,
        dual_clip_negative_advantage=True,
        loss_name="prefix_probability_weighted_exact_kl_cumulative_dual_clip",
    )


@register_policy_loss("prefix_exact_kl_clip")
def compute_policy_loss_prefix_exact_kl_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact prefix KL-coordinate clip with ``R - 1 - log(R) <= t * delta``."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=False,
        geometric_average_surrogate=True,
        dual_clip_negative_advantage=False,
        loss_name="prefix_exact_kl_clip",
    )


@register_policy_loss("prefix_exact_kl_cumulative_dual_clip")
def compute_policy_loss_prefix_exact_kl_cumulative_dual_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Cumulative exact-prefix surrogate with negative-advantage dual clip."""

    return _compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        probability_weighted=False,
        geometric_average_surrogate=False,
        dual_clip_negative_advantage=True,
        loss_name="prefix_exact_kl_cumulative_dual_clip",
    )


@register_policy_loss("prefix_sqrt_dynamic_clip")
def compute_policy_loss_prefix_sqrt_dynamic_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average-prefix PPO-style loss with sqrt-position dynamic clipping.

    This is intentionally the same surrogate as ``prefix_dynamic_clip``: the
    numerical ratio is the detached average prefix ratio and gradients are
    routed only through the current-token log-probability. The only algorithmic
    change is the boundary schedule:

        lambda_t = (t^{-1/2} - T^{-1/2}) / (1 - T^{-1/2})

    instead of the original normalized 1/t schedule. This treats the prefix
    cumulative log-ratio as a sum of token-level increments whose natural scale
    grows like sqrt(t), so the average prefix log-ratio bound decays like
    1/sqrt(t) between the first-token and final-token endpoints.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    eps_1_low = float(policy_loss_cfg.get("prefix_clip_first_low", 0.2))
    eps_1_high = float(policy_loss_cfg.get("prefix_clip_first_high", 0.28))
    eps_t_low = float(policy_loss_cfg.get("prefix_clip_final_low", 3e-4))
    eps_t_high = float(policy_loss_cfg.get("prefix_clip_final_high", 4e-4))

    negative_approx_kl = log_prob - old_log_prob
    log_ratio = negative_approx_kl * response_mask

    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)
    prefix_avg_log_ratio = prefix_log_ratio_sum / prefix_len

    eps_1_pos = torch.log(torch.tensor(1.0 + eps_1_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_1_neg = -torch.log(torch.tensor(1.0 - eps_1_low, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_pos = torch.log(torch.tensor(1.0 + eps_t_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_neg = -torch.log(torch.tensor(1.0 - eps_t_low, device=log_prob.device, dtype=log_prob.dtype))

    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    sqrt_prefix_len = torch.sqrt(prefix_len)
    sqrt_seq_len = torch.sqrt(seq_len)
    lambda_t = (1.0 / sqrt_prefix_len - 1.0 / sqrt_seq_len) / (1.0 - 1.0 / sqrt_seq_len).clamp(min=1e-8)
    lambda_t = lambda_t.clamp(min=0.0, max=1.0) * response_mask

    eps_pos = eps_t_pos + (eps_1_pos - eps_t_pos) * lambda_t
    eps_neg = eps_t_neg + (eps_1_neg - eps_t_neg) * lambda_t

    prefix_avg_log_ratio = torch.clamp(prefix_avg_log_ratio, min=-10.0, max=10.0)
    prefix_avg_log_ratio_clip = torch.minimum(torch.maximum(prefix_avg_log_ratio, -eps_neg), eps_pos)

    prefix_log_importance_ratio = log_prob - log_prob.detach() + prefix_avg_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-10.0, max=10.0)
    prefix_log_importance_ratio_clip = torch.minimum(torch.maximum(prefix_log_importance_ratio, -eps_neg), eps_pos)

    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)

    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    clipped = torch.ne(prefix_avg_log_ratio, prefix_avg_log_ratio_clip)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped & (prefix_avg_log_ratio < -eps_neg)).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    lower_clipped = clipped & (prefix_avg_log_ratio < -eps_neg)
    _dump_prefix_clip_response_diagnostics(
        loss_name="prefix_sqrt_dynamic_clip",
        response_mask=response_mask,
        clipped=clipped,
        lower_clipped=lower_clipped,
        prefix_avg_log_ratio=prefix_avg_log_ratio,
        eps_pos=eps_pos,
        eps_neg=eps_neg,
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/prefix_sqrt_clip/eps_first_low": eps_1_low,
        "actor/prefix_sqrt_clip/eps_first_high": eps_1_high,
        "actor/prefix_sqrt_clip/eps_final_low": eps_t_low,
        "actor/prefix_sqrt_clip/eps_final_high": eps_t_high,
        "actor/prefix_sqrt_clip/avg_log_ratio_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach(), response_mask
        ).item(),
        "actor/prefix_sqrt_clip/avg_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), response_mask
        ).item(),
        "actor/prefix_sqrt_clip/eps_pos_mean": verl_F.masked_mean(eps_pos.detach(), response_mask).item(),
        "actor/prefix_sqrt_clip/eps_neg_mean": verl_F.masked_mean(eps_neg.detach(), response_mask).item(),
        "actor/prefix_sqrt_clip/lambda_mean": verl_F.masked_mean(lambda_t.detach(), response_mask).item(),
    }

    relative_pos = (prefix_len / seq_len).clamp(min=0.0, max=1.0)
    valid_response_mask = response_mask.bool()
    for bucket_idx in range(10):
        left = bucket_idx / 10.0
        right = (bucket_idx + 1) / 10.0
        if bucket_idx == 0:
            bucket_mask = valid_response_mask & (relative_pos >= left) & (relative_pos <= right)
        else:
            bucket_mask = valid_response_mask & (relative_pos > left) & (relative_pos <= right)
        bucket_mask_float = bucket_mask.float()
        bucket_prefix = f"actor/prefix_sqrt_clip/pos_bucket_{bucket_idx:02d}_{bucket_idx + 1:02d}"
        pg_metrics[f"{bucket_prefix}/token_frac"] = verl_F.masked_mean(
            bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac_lower"] = verl_F.masked_mean(
            lower_clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/avg_log_ratio_abs_mean"] = verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/eps_pos_mean"] = verl_F.masked_mean(
            eps_pos.detach(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/eps_neg_mean"] = verl_F.masked_mean(
            eps_neg.detach(), bucket_mask_float
        ).detach().item()

    seq_len_flat = seq_len.squeeze(-1)
    clipped_valid = clipped & valid_response_mask
    clipped_lower_valid = lower_clipped & valid_response_mask
    length_buckets = [
        (0, 1024, "00000_01024"),
        (1024, 2048, "01024_02048"),
        (2048, 4096, "02048_04096"),
        (4096, 8192, "04096_08192"),
        (8192, 12288, "08192_12288"),
        (12288, 16000, "12288_16000"),
        (16000, None, "16000_plus"),
    ]
    for lower, upper, label in length_buckets:
        if upper is None:
            seq_bucket_mask = seq_len_flat >= lower
        else:
            seq_bucket_mask = (seq_len_flat >= lower) & (seq_len_flat < upper)
        seq_bucket_mask_float = seq_bucket_mask.float()
        token_bucket_mask = valid_response_mask & seq_bucket_mask.unsqueeze(-1)
        token_bucket_mask_float = token_bucket_mask.float()
        seq_bucket_denom = seq_bucket_mask_float.sum().clamp(min=1.0)
        len_prefix = f"actor/prefix_sqrt_clip/response_len_bucket_{label}"
        pg_metrics[f"{len_prefix}/response_frac"] = seq_bucket_mask_float.mean().detach().item()
        pg_metrics[f"{len_prefix}/token_frac"] = verl_F.masked_mean(
            token_bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_len_mean"] = (
            (seq_len_flat.detach() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/token_pg_clipfrac_lower"] = verl_F.masked_mean(
            clipped_lower_valid.float(), token_bucket_mask_float
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_clipfrac"] = (
            (clipped_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()
        pg_metrics[f"{len_prefix}/response_any_lower_clipfrac"] = (
            (clipped_lower_valid.any(dim=-1).float() * seq_bucket_mask_float).sum() / seq_bucket_denom
        ).detach().item()

    return pg_loss, pg_metrics


@register_policy_loss("prefix_sum_dynamic_clip")
def compute_policy_loss_prefix_sum_dynamic_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO-style loss with dynamic clipping on prefix likelihood ratios.

    This variant uses the prefix cumulative log-ratio directly:

        L_i,t(theta) = sum_{k <= t}
            log pi_theta(o_i,k | q_i, o_i,<k)
          - log pi_old(o_i,k | q_i, o_i,<k)

    Therefore the numerical importance-ratio value is the prefix likelihood
    ratio exp(L_i,t), rather than the average-prefix ratio exp(L_i,t / t). To
    keep this sum-log-ratio parameterization on the same trust-region scale as
    average-prefix clipping, the average log-space boundary is multiplied by the
    prefix length t and then capped by alpha:

        B_i,t^+ = min(t * eps_i,t^+, alpha)
        B_i,t^- = min(t * eps_i,t^-, alpha)

    The alpha cap is applied in log space before exp(), so alpha=2 constrains
    the usable prefix likelihood ratio to approximately [exp(-2), exp(2)]. As
    in GSPO, gradients are routed only through the current-token log-probability.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    eps_1_low = float(policy_loss_cfg.get("prefix_clip_first_low", 0.2))
    eps_1_high = float(policy_loss_cfg.get("prefix_clip_first_high", 0.28))
    eps_t_low = float(policy_loss_cfg.get("prefix_clip_final_low", 3e-4))
    eps_t_high = float(policy_loss_cfg.get("prefix_clip_final_high", 4e-4))
    alpha = float(policy_loss_cfg.get("prefix_clip_sum_alpha", 2.0))

    negative_approx_kl = log_prob - old_log_prob

    # token-level log-ratio l_i,t(theta), masked to response tokens only.
    log_ratio = negative_approx_kl * response_mask

    # prefix cumulative log-ratio L_i,t(theta) = sum_{k <= t} l_i,k(theta).
    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)

    # Reuse the average-prefix dynamic schedule, then convert its boundary to
    # the sum-log-ratio scale by multiplying with prefix_len.
    eps_1_pos = torch.log(torch.tensor(1.0 + eps_1_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_1_neg = -torch.log(torch.tensor(1.0 - eps_1_low, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_pos = torch.log(torch.tensor(1.0 + eps_t_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_neg = -torch.log(torch.tensor(1.0 - eps_t_low, device=log_prob.device, dtype=log_prob.dtype))

    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    lambda_t = (1.0 / prefix_len - 1.0 / seq_len) / (1.0 - 1.0 / seq_len).clamp(min=1e-8)
    lambda_t = lambda_t.clamp(min=0.0, max=1.0) * response_mask

    avg_eps_pos = eps_t_pos + (eps_1_pos - eps_t_pos) * lambda_t
    avg_eps_neg = eps_t_neg + (eps_1_neg - eps_t_neg) * lambda_t

    alpha_tensor = torch.tensor(alpha, device=log_prob.device, dtype=log_prob.dtype)
    sum_eps_pos = torch.minimum(prefix_len * avg_eps_pos, alpha_tensor)
    sum_eps_neg = torch.minimum(prefix_len * avg_eps_neg, alpha_tensor)

    # Numerical guard for prefix likelihood ratios. This is the sum-log-ratio
    # value used by the unclipped surrogate branch.
    prefix_sum_log_ratio = torch.clamp(prefix_log_ratio_sum, min=-alpha, max=alpha)
    prefix_sum_log_ratio_clip = torch.minimum(torch.maximum(prefix_sum_log_ratio, -sum_eps_neg), sum_eps_pos)

    # Stop-gradient trick: forward value equals prefix_sum_log_ratio, but the
    # gradient enters only through the current token log-probability. The clipped
    # branch keeps clamp's zero-gradient behavior outside the active interval.
    prefix_log_importance_ratio = log_prob - log_prob.detach() + prefix_sum_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-alpha, max=alpha)
    prefix_log_importance_ratio_clip = torch.minimum(
        torch.maximum(prefix_log_importance_ratio, -sum_eps_neg), sum_eps_pos
    )

    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)

    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    clipped = torch.ne(prefix_sum_log_ratio, prefix_sum_log_ratio_clip)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped & (prefix_sum_log_ratio < -sum_eps_neg)).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/prefix_sum_clip/alpha": alpha,
        "actor/prefix_sum_clip/eps_first_low": eps_1_low,
        "actor/prefix_sum_clip/eps_first_high": eps_1_high,
        "actor/prefix_sum_clip/eps_final_low": eps_t_low,
        "actor/prefix_sum_clip/eps_final_high": eps_t_high,
        "actor/prefix_sum_clip/log_ratio_mean": verl_F.masked_mean(prefix_sum_log_ratio.detach(), response_mask).item(),
        "actor/prefix_sum_clip/log_ratio_abs_mean": verl_F.masked_mean(
            prefix_sum_log_ratio.detach().abs(), response_mask
        ).item(),
        "actor/prefix_sum_clip/bound_pos_mean": verl_F.masked_mean(sum_eps_pos.detach(), response_mask).item(),
        "actor/prefix_sum_clip/bound_neg_mean": verl_F.masked_mean(sum_eps_neg.detach(), response_mask).item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("prefix_sum_gate_avg_clip")
def compute_policy_loss_prefix_sum_gate_avg_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average-prefix ratio loss with prefix-sum clipping gates.

    This variant is a GSPO-style stop-gradient policy loss that uses a
    token-dependent prefix ratio instead of one sequence-level ratio for every
    token. Let

        l_i,t(theta) =
            log pi_theta(o_i,t | q_i, o_i,<t)
          - log pi_old(o_i,t | q_i, o_i,<t)

    be the token-level log-ratio, and let

        L_i,t(theta) = sum_{k <= t} l_i,k(theta)
        z_i,t(theta) = L_i,t(theta) / t

    be the cumulative prefix log-ratio and average-prefix log-ratio. The
    numerical importance ratio used by the surrogate remains the stable
    average-prefix ratio exp(z_i,t). Clipping, however, is decided on the prefix
    likelihood-ratio scale exp(L_i,t). We first construct a sum-log-ratio bound

        B_i,t^+ = min(t * eps_i,t^+, alpha)
        B_i,t^- = min(t * eps_i,t^-, alpha)

    and then convert it back to the average-log-ratio scale for the clipped
    branch:

        z_i,t = L_i,t / t
        z_clip,i,t = clamp(z_i,t, -B_i,t^- / t, B_i,t^+ / t)

    This preserves the stable average-prefix weighting while using the prefix
    likelihood ratio exp(L_i,t) as the trust-region gate. Gradients are routed
    only through the current-token log-probability, as in GSPO.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of response tokens under the old policy, shape
            (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of response tokens under the current policy,
            shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each response token, shape
            (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating valid response tokens, shape
            (batch_size, response_length).
        loss_agg_mode (str, optional):
            Unused here. This loss matches GSPO and aggregates with
            "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    policy_loss_cfg = config.policy_loss
    eps_1_low = float(policy_loss_cfg.get("prefix_clip_first_low", 0.2))
    eps_1_high = float(policy_loss_cfg.get("prefix_clip_first_high", 0.28))
    eps_t_low = float(policy_loss_cfg.get("prefix_clip_final_low", 3e-4))
    eps_t_high = float(policy_loss_cfg.get("prefix_clip_final_high", 4e-4))
    alpha = float(policy_loss_cfg.get("prefix_clip_sum_alpha", 2.0))

    # compute token-level log-ratio:
    # l_i,t(theta) =
    #   log pi_theta(o_i,t | q_i, o_i,<t)
    # - log pi_old(o_i,t | q_i, o_i,<t)
    negative_approx_kl = log_prob - old_log_prob

    # Only response tokens should contribute to prefix statistics. Padding
    # positions are masked out again in loss/metric aggregation.
    log_ratio = negative_approx_kl * response_mask

    # compute prefix statistics:
    # L_i,t(theta) = sum_{k <= t} l_i,k(theta)
    # z_i,t(theta) = L_i,t(theta) / t
    #
    # prefix_len is the valid response-token prefix length at each position.
    # Padding positions are clamped to one to avoid division by zero.
    prefix_len = torch.cumsum(response_mask, dim=-1).clamp(min=1.0)
    prefix_log_ratio_sum = torch.cumsum(log_ratio, dim=-1)
    prefix_avg_log_ratio = prefix_log_ratio_sum / prefix_len

    # First-token boundaries are specified in ratio space, matching the usual
    # asymmetric PPO/DAPO-style clip interval [1 - eps_low, 1 + eps_high].
    # Convert them to positive log-space magnitudes so the actual interval is
    # [-eps_1_neg, eps_1_pos].
    eps_1_pos = torch.log(torch.tensor(1.0 + eps_1_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_1_neg = -torch.log(torch.tensor(1.0 - eps_1_low, device=log_prob.device, dtype=log_prob.dtype))

    # Final-token boundaries follow the same ratio-space semantics as GSPO's
    # clip interval [1 - eps_low, 1 + eps_high]. Convert them to positive
    # log-space magnitudes before interpolating in log-ratio space.
    eps_t_pos = torch.log(torch.tensor(1.0 + eps_t_high, device=log_prob.device, dtype=log_prob.dtype))
    eps_t_neg = -torch.log(torch.tensor(1.0 - eps_t_low, device=log_prob.device, dtype=log_prob.dtype))

    # Interpolate average-log-ratio clip boundaries from the first-token
    # boundary to the final-token boundary with normalized 1/t decay:
    #
    # lambda_t = (1/t - 1/T) / (1 - 1/T)
    #
    # Thus lambda_1 = 1 and lambda_T = 0. The response_mask multiplication keeps
    # padding positions inert.
    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    lambda_t = (1.0 / prefix_len - 1.0 / seq_len) / (1.0 - 1.0 / seq_len).clamp(min=1e-8)
    lambda_t = lambda_t.clamp(min=0.0, max=1.0) * response_mask

    avg_eps_pos = eps_t_pos + (eps_1_pos - eps_t_pos) * lambda_t
    avg_eps_neg = eps_t_neg + (eps_1_neg - eps_t_neg) * lambda_t

    # Convert the average-log-ratio boundary to the prefix-sum scale:
    #
    # B_i,t^+ = min(t * eps_i,t^+, alpha)
    # B_i,t^- = min(t * eps_i,t^-, alpha)
    #
    # alpha caps the prefix likelihood-ratio gate in log space. Dividing the
    # capped bound by t maps it back to the average-prefix log-ratio scale used
    # by the surrogate value.
    alpha_tensor = torch.tensor(alpha, device=log_prob.device, dtype=log_prob.dtype)
    sum_eps_pos = torch.minimum(prefix_len * avg_eps_pos, alpha_tensor)
    sum_eps_neg = torch.minimum(prefix_len * avg_eps_neg, alpha_tensor)
    gate_avg_eps_pos = sum_eps_pos / prefix_len
    gate_avg_eps_neg = sum_eps_neg / prefix_len

    # Clip the average-prefix log-ratio with bounds derived from the prefix-sum
    # gate:
    #
    # z_clip,i,t = clamp(z_i,t, -B_i,t^- / t, B_i,t^+ / t)
    #
    # The outer [-10, 10] clamp is only a numerical guard before exp().
    prefix_avg_log_ratio = torch.clamp(prefix_avg_log_ratio, min=-10.0, max=10.0)
    prefix_avg_log_ratio_clip = torch.minimum(torch.maximum(prefix_avg_log_ratio, -gate_avg_eps_neg), gate_avg_eps_pos)

    # Combined prefix ratio at token level:
    # r_i,t(theta) = sg[exp(z_i,t(theta))]
    #              * pi_theta(o_i,t | q_i, o_i,<t)
    #              / sg[pi_theta(o_i,t | q_i, o_i,<t)]
    #
    # In log space:
    # log r_i,t(theta) = log_prob - sg[log_prob] + sg[z_i,t(theta)]
    #
    # This keeps the forward value equal to exp(z_i,t), while routing gradients
    # only through the current-token log-probability as in GSPO. The clipped
    # branch keeps clamp's zero-gradient behavior outside the active interval.
    prefix_log_importance_ratio = log_prob - log_prob.detach() + prefix_avg_log_ratio.detach()
    prefix_log_importance_ratio = torch.clamp(prefix_log_importance_ratio, min=-10.0, max=10.0)
    prefix_log_importance_ratio_clip = torch.minimum(
        torch.maximum(prefix_log_importance_ratio, -gate_avg_eps_neg), gate_avg_eps_pos
    )

    # finaly exp() to remove log
    ratio_for_loss = torch.exp(prefix_log_importance_ratio)
    ratio_clip_for_loss = torch.exp(prefix_log_importance_ratio_clip)

    pg_losses1 = -advantages * ratio_for_loss
    pg_losses2 = -advantages * ratio_clip_for_loss
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Match the GSPO baseline aggregation so this variant changes only the
    # importance-ratio and clipping rule.
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )

    # Metrics report both the average-ratio clipped fraction used by the loss
    # and the prefix-sum gate clipped fraction used to diagnose trust-region
    # pressure on exp(L_i,t).
    sum_log_ratio_for_metrics = torch.clamp(prefix_log_ratio_sum, min=-alpha, max=alpha)
    sum_log_ratio_clip_for_metrics = torch.minimum(torch.maximum(sum_log_ratio_for_metrics, -sum_eps_neg), sum_eps_pos)
    clipped = torch.ne(prefix_avg_log_ratio, prefix_avg_log_ratio_clip)
    sum_gate_clipped = torch.ne(sum_log_ratio_for_metrics, sum_log_ratio_clip_for_metrics)
    pg_clipfrac = verl_F.masked_mean(clipped.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        (clipped & (prefix_avg_log_ratio < -gate_avg_eps_neg)).float(), response_mask
    )
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/prefix_sum_gate_avg_clip/alpha": alpha,
        "actor/prefix_sum_gate_avg_clip/eps_first_low": eps_1_low,
        "actor/prefix_sum_gate_avg_clip/eps_first_high": eps_1_high,
        "actor/prefix_sum_gate_avg_clip/eps_final_low": eps_t_low,
        "actor/prefix_sum_gate_avg_clip/eps_final_high": eps_t_high,
        "actor/prefix_sum_gate_avg_clip/avg_log_ratio_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach(), response_mask
        ).item(),
        "actor/prefix_sum_gate_avg_clip/avg_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), response_mask
        ).item(),
        "actor/prefix_sum_gate_avg_clip/sum_log_ratio_abs_mean": verl_F.masked_mean(
            sum_log_ratio_for_metrics.detach().abs(), response_mask
        ).item(),
        "actor/prefix_sum_gate_avg_clip/sum_bound_pos_mean": verl_F.masked_mean(
            sum_eps_pos.detach(), response_mask
        ).item(),
        "actor/prefix_sum_gate_avg_clip/sum_bound_neg_mean": verl_F.masked_mean(
            sum_eps_neg.detach(), response_mask
        ).item(),
        "actor/prefix_sum_gate_avg_clip/sum_gate_clipfrac": verl_F.masked_mean(
            sum_gate_clipped.float(), response_mask
        ).item(),
    }

    # Position-bucket diagnostics for checking whether the dynamic prefix bound
    # becomes too tight in the middle/end of long responses. These metrics do
    # not affect the loss; they only expose where clipping happens along the
    # normalized response position.
    relative_pos = (prefix_len / seq_len).clamp(min=0.0, max=1.0)
    valid_response_mask = response_mask.bool()
    for bucket_idx in range(10):
        left = bucket_idx / 10.0
        right = (bucket_idx + 1) / 10.0
        if bucket_idx == 0:
            bucket_mask = valid_response_mask & (relative_pos >= left) & (relative_pos <= right)
        else:
            bucket_mask = valid_response_mask & (relative_pos > left) & (relative_pos <= right)
        bucket_mask_float = bucket_mask.float()
        bucket_prefix = f"actor/prefix_sum_gate_avg_clip/pos_bucket_{bucket_idx:02d}_{bucket_idx + 1:02d}"
        pg_metrics[f"{bucket_prefix}/token_frac"] = verl_F.masked_mean(
            bucket_mask_float, response_mask
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/pg_clipfrac"] = verl_F.masked_mean(
            clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/sum_gate_clipfrac"] = verl_F.masked_mean(
            sum_gate_clipped.float(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/avg_log_ratio_abs_mean"] = verl_F.masked_mean(
            prefix_avg_log_ratio.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/sum_log_ratio_abs_mean"] = verl_F.masked_mean(
            sum_log_ratio_for_metrics.detach().abs(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/sum_bound_pos_mean"] = verl_F.masked_mean(
            sum_eps_pos.detach(), bucket_mask_float
        ).detach().item()
        pg_metrics[f"{bucket_prefix}/sum_bound_neg_mean"] = verl_F.masked_mean(
            sum_eps_neg.detach(), bucket_mask_float
        ).detach().item()
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    # Strip the optional '+' suffix so e.g. "k3+" dispatches to "k3".
    base_kl_penalty = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
    forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


@register_policy_loss("cum-token-cumprod-la")
def compute_policy_loss_cum_token_cumprod_la(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CTPO with cumulative-token ratios and position-adaptive clipping."""

    assert config is not None
    assert isinstance(config, ActorConfig)

    la_clip_low = float(config.get("la_clip_low", 0.025))
    la_clip_high = float(config.get("la_clip_high", 0.05))
    la_clip_c = float(config.get("la_clip_c", 0.05))
    la_power = float(config.get("clip_ratio_la_power", 0.5))
    if la_clip_low <= 0 or la_clip_high <= 0 or la_clip_c <= 0:
        raise ValueError("CTPO la_clip_low, la_clip_high, and la_clip_c must be positive.")

    token_log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    mask = response_mask.to(dtype=token_log_ratio.dtype)
    prefix_log_ratio = torch.cumsum(token_log_ratio * mask, dim=-1)

    # Forward value: cumulative prefix ratio. Backward: current token only.
    log_importance_ratio = prefix_log_ratio.detach() + log_prob - log_prob.detach()
    log_importance_ratio = torch.clamp(log_importance_ratio, max=10.0)
    ratio = torch.where(mask > 0, torch.exp(log_importance_ratio), torch.zeros_like(log_importance_ratio))

    position = torch.cumsum(mask, dim=-1)
    scale = position.clamp(min=1.0).pow(la_power)
    log_clip_low = la_clip_low * scale
    log_clip_high = la_clip_high * scale
    log_clip_c = la_clip_c * scale

    clipped_log_ratio = torch.clamp(log_importance_ratio, min=-log_clip_low, max=log_clip_high)
    clipped_ratio = torch.where(mask > 0, torch.exp(clipped_log_ratio), torch.zeros_like(clipped_log_ratio))
    dual_clip_threshold = torch.where(mask > 0, torch.exp(log_clip_c), torch.ones_like(log_clip_c))

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_losses3 = -advantages * dual_clip_threshold
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) & (advantages < 0)).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    valid_mask = response_mask.bool()
    high_clipped = valid_mask & (log_importance_ratio > log_clip_high) & (advantages > 0)
    low_clipped = valid_mask & (log_importance_ratio < -log_clip_low) & (advantages < 0)
    dual_clipped = valid_mask & torch.gt(clip_pg_losses1, pg_losses3) & (advantages < 0)
    clipped = high_clipped | low_clipped | dual_clipped
    ratio_valid = ratio[valid_mask]
    scale_valid = scale[valid_mask]
    valid_count = int(valid_mask.sum().item())

    _dump_prefix_clip_response_diagnostics(
        loss_name="cum-token-cumprod-la",
        response_mask=response_mask,
        clipped=clipped,
        lower_clipped=low_clipped | dual_clipped,
        prefix_avg_log_ratio=prefix_log_ratio,
        eps_pos=log_clip_high,
        eps_neg=log_clip_low,
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": verl_F.masked_mean(-token_log_ratio, response_mask).detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ctpo/la_clip_low": la_clip_low,
        "actor/ctpo/la_clip_high": la_clip_high,
        "actor/ctpo/la_clip_c": la_clip_c,
        "actor/ctpo/la_power": la_power,
        "actor/ctpo/prefix_log_ratio_abs_mean": verl_F.masked_mean(
            prefix_log_ratio.detach().abs(), response_mask
        ).item(),
        "actor/ctpo/scale_mean": scale_valid.mean().item() if valid_count else 0.0,
        "actor/ctpo/ratio_mean": ratio_valid.mean().item() if valid_count else 0.0,
        "actor/ctpo/ratio_p95": torch.quantile(ratio_valid, 0.95).item() if valid_count else 0.0,
        "actor/ctpo/ratio_max": ratio_valid.max().item() if valid_count else 0.0,
    }
    return pg_loss, pg_metrics
