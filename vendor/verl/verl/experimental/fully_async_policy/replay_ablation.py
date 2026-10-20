"""Replay transport for stale trajectories; never replace the behavior denominator."""
import time

import torch


def attach_replay(batch, mode, record=None):
    probs = batch.batch['rollout_log_probs']
    mask = batch.batch['response_mask'].bool()
    if probs.shape != mask.shape or not torch.isfinite(probs[mask]).all():
        raise ValueError('Invalid generation-time log probabilities')
    if mode == 'R2':
        if record is None:
            raise ValueError('R2 requires a learner routing-record forward')
        output = record(batch)
        if 'routed_experts' not in output.batch:
            raise ValueError('R2 forward did not return routed_experts')
        batch.batch['routed_experts'] = output.batch['routed_experts']
    elif mode not in ('disabled', 'R3'):
        raise ValueError(f'Unsupported replay mode: {mode}')
    if mode != 'disabled' and batch.batch.get('routed_experts') is None:
        raise ValueError(f'{mode} requires routed_experts; refusing silent fallback')
    batch.batch['old_log_probs'] = probs
    return batch


def fit_compute_log_prob(trainer, batch):
    actor = trainer.config.actor_rollout_ref.actor
    mode = actor.get('megatron', {}).get('router_replay', {}).get('mode', 'disabled')
    start = time.perf_counter()
    def record(b):
        # Use the current learner solely to record R2 routes. Do NOT invoke
        # FullyAsyncTrainer's snapshot/recompute denominator machinery.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        output, _ = RayPPOTrainer._compute_old_log_prob(trainer, b)
        return output
    result = attach_replay(batch, mode, record=record if mode == 'R2' else None)
    if mode != 'disabled':
        trainer.metrics.update({
            'replay/routes_present': 1.0,
            'replay/r2_record_seconds': time.perf_counter() - start if mode == 'R2' else 0.0,
            'replay/behavior_denominator_preserved': 1.0,
        })
    return result


def merge_routes(previous, incoming, new_tokens, prompt_length, previous_tokens, required):
    """Routes are predictor-input rows: prompt + response length minus one."""
    if new_tokens == 0:
        return previous
    if incoming is None:
        if required or previous is not None:
            raise ValueError('Missing routing for generated tokens')
        return None
    incoming = torch.as_tensor(incoming)
    expected = prompt_length + previous_tokens + new_tokens - 1
    if incoming.ndim != 3 or incoming.shape[0] != expected:
        raise ValueError(f'Invalid routing length: expected {expected}, got {tuple(incoming.shape)}')
    if previous is None:
        if previous_tokens:
            raise ValueError('Missing historical routing before resumed segment')
        return incoming
    previous = torch.as_tensor(previous)
    if previous.shape[0] != prompt_length + previous_tokens - 1 or previous.shape[1:] != incoming.shape[1:]:
        raise ValueError('Inconsistent historical routing shape')
    return torch.cat((previous, incoming[-new_tokens:]), dim=0)
