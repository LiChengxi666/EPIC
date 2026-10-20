# Copyright 2026 EPIC contributors.
# Derived from the supplied verl experiment implementation.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal local-batch reference loss; not a distributed training adapter."""
import math
import torch
from ._bounds import _solve_exact_prefix_kl_log_bounds


def epic_loss(old_log_prob, log_prob, advantages, response_mask, *,
              delta_low=5e-4, delta_high=2e-3):
    """Compute EPIC's minimized loss for tensors of shape [B, T].

    Behavior log probabilities and advantages are fixed data. Mask values must
    be 0 or 1; padding values must be finite. Empty rows are excluded from the
    response mean; at least one response must contain a valid token. This
    function uses local sequence-mean/token-mean reduction, not DDP scaling.
    """
    values = (old_log_prob, log_prob, advantages, response_mask)
    if log_prob.ndim != 2 or any(v.shape != log_prob.shape for v in values):
        raise ValueError("Inputs must share shape [batch, response_tokens].")
    if any(v.device != log_prob.device for v in values):
        raise ValueError("All inputs must share one device.")
    if not log_prob.is_floating_point():
        raise ValueError("log_prob must be floating point.")
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError("Inputs, including padded positions, must be finite.")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response_mask must contain only zeros and ones.")
    if not response_mask.any():
        raise ValueError("At least one valid response token is required.")
    if any(not math.isfinite(x) or x <= 0 for x in (delta_low, delta_high)):
        raise ValueError("Both KL-coordinate budgets must be finite and positive.")
    mask = response_mask.to(log_prob.dtype)
    old = old_log_prob.detach().to(log_prob.dtype)
    adv = advantages.detach().to(log_prob.dtype)
    length = mask.cumsum(-1).clamp_min(1)
    prefix_log_ratio = ((log_prob - old) * mask).cumsum(-1)
    old_prefix_log_prob = (old * mask).cumsum(-1)
    calibrated_log_scale = length.log() - old_prefix_log_prob / length
    lower, _ = _solve_exact_prefix_kl_log_bounds(calibrated_log_scale + math.log(delta_low))
    _, upper = _solve_exact_prefix_kl_log_bounds(calibrated_log_scale + math.log(delta_high))
    lower, upper = lower / length, upper / length
    # Forward geometric-prefix ratio; backward current-token-only path.
    log_weight = (log_prob - log_prob.detach() + (prefix_log_ratio / length).detach()).clamp(-10, 10)
    clipped_log_weight = torch.minimum(torch.maximum(log_weight, lower), upper).clamp(-10, 10)
    losses = torch.maximum(-adv * log_weight.exp(), -adv * clipped_log_weight.exp())
    # Prefix-dependent negative-advantage cap, not a fixed PPO dual-clip constant.
    negative_cap = -adv * upper.clamp(-10, 10).exp()
    losses = torch.where(adv < 0, torch.minimum(losses, negative_cap), losses)
    tokens = mask.sum(-1)
    per_response = (losses * mask).sum(-1) / (tokens + 1e-8)
    valid = (tokens > 0).to(log_prob.dtype)
    return (per_response * valid).sum() / valid.sum()
