"""PPO importance-ratio objective with both clipping branches removed."""
import torch
from verl.utils import torch_functional as F


def compute_policy_loss_unclipped_is(old_log_prob, log_prob, advantages, response_mask,
                                     loss_agg_mode='token-mean', config=None, rollout_is_weights=None):
    from verl.trainer.ppo.core_algos import agg_loss
    if config is None:
        raise ValueError('Actor config is required')
    raw = log_prob - old_log_prob
    # Retain vanilla's existing exponent safety guard, not a trust-region clip.
    safe = raw.clamp(-20., 20.)
    ratio = safe.exp()
    losses = -advantages * ratio
    if rollout_is_weights is not None:
        losses = losses * rollout_is_weights
    loss = agg_loss(losses, response_mask, loss_agg_mode, **config.global_batch_info)
    metrics = {
        'actor/pg_clipfrac': 0.0,
        'actor/pg_clipfrac_lower': 0.0,
        'actor/ppo_kl': F.masked_mean(-safe, response_mask).detach().item(),
        'actor/is_ratio_mean': F.masked_mean(ratio, response_mask).detach().item(),
        'actor/is_log_ratio_safety_frac': F.masked_mean((raw.abs() > 20).float(), response_mask).detach().item(),
    }
    return loss, metrics
