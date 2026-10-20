"""Opt-in whole-trajectory stale-data experiment, using the existing model engines.

No oversampling, no cross-version partial responses, no change to registered losses.
Checkpoint boundaries drain generation and preserve the completed FIFO plus data cursor.
"""

import asyncio
import os
import random
import time
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import RolloutSample, prepare_single_generation_data
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue
from verl.experimental.fully_async_policy.stale_probe_config import (
    atomic_write,
    lag_metrics,
    load_probe_state,
    save_probe_state,
)


class ProbeQueue(MessageQueue.__ray_metadata__.modified_class):
    async def put_sample(self, sample):
        # Unlike the default queue, never silently evict training data.
        async with self._lock:
            if len(self.queue) >= self.max_queue_size:
                raise RuntimeError("Stale-probe queue overflow; refusing to discard trajectories")
            self.queue.append(sample)
            self.total_produced += 1
            self._consumer_condition.notify_all()
        return True

    async def snapshot(self):
        async with self._lock:
            return {"items": list(self.queue), "produced": self.total_produced, "consumed": self.total_consumed}

    async def restore(self, state):
        async with self._lock:
            if self.queue or len(state["items"]) > self.max_queue_size:
                raise ValueError("Cannot restore queue into occupied or smaller storage")
            self.queue.extend(state["items"])
            self.total_produced, self.total_consumed = state["produced"], state["consumed"]
            self._consumer_condition.notify_all()


def _signature(config):
    c = OmegaConf.to_container(config, resolve=True)
    return {
        "run_id": c["trainer"]["experiment_name"],
        "actor": c["actor_rollout_ref"]["actor"],
        "async": c["async_training"],
        "model": c["actor_rollout_ref"]["model"],
        "data": c["data"],
        "algorithm": c["algorithm"],
        "reward": c.get("reward"),
        "world_size": c["trainer"]["nnodes"] * c["trainer"]["n_gpus_per_node"],
        "sampling": {
            k: c["actor_rollout_ref"]["rollout"].get(k) for k in ("n", "temperature", "top_p", "top_k", "val_kwargs")
        },
    }


def _resume_signatures_match(saved, current, allow_offload_change=False):
    """Opt-in placement-only resume; retain every algorithm/identity check."""
    import copy
    saved, current = copy.deepcopy(saved), copy.deepcopy(current)
    if allow_offload_change:
        for signature in (saved, current):
            engine = signature.get("actor", {}).get("megatron", {})
            for key in ("param_offload", "optimizer_offload", "grad_offload"):
                engine.pop(key, None)
    return saved == current


def _read_resume(config):
    from verl.utils import hdfs_io
    from verl.utils.checkpoint.hdfs_checkpoint import _copy_required_to_local, download_remote_metadata

    if config.trainer.resume_mode == "disable":
        return None, None
    if config.trainer.resume_mode != "auto":
        raise ValueError("Stale probe uses resume_mode=auto with its own immutable checkpoint namespace")
    root = config.trainer.default_hdfs_dir
    if root:
        local = None
        committed_version = 0
        if hdfs_io.exists(root + "/latest_checkpointed_iteration.txt"):
            local, _ = download_remote_metadata(root, config.trainer.default_local_dir)
            committed_version = int(Path(local).name.removeprefix("global_step_"))
        # The shared publisher copies _SUCCESS before the latest pointer. A
        # write-ahead hint allows read-only adoption after pointer-copy failure.
        pending_name = "pending_checkpointed_iteration.txt"
        if hdfs_io.exists(root + "/" + pending_name):
            pending = Path(config.trainer.default_local_dir) / pending_name
            _copy_required_to_local(root + "/" + pending_name, pending)
            version = int(pending.read_text())
            remote_step = f"{root}/global_step_{version}"
            if version > committed_version and hdfs_io.exists(remote_step + "/_SUCCESS"):
                local = pending.parent / f"global_step_{version}"
                for name in ("_SUCCESS", "data.pt"):
                    _copy_required_to_local(remote_step + "/" + name, local / name)
        if local is None:
            return None, None
    else:
        tracker = Path(config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return None, None
        local = tracker.parent / f"global_step_{int(tracker.read_text())}"
    state = load_probe_state(local)
    if not 0 < state["version"] <= config.trainer.total_training_steps:
        raise ValueError("Checkpoint version is outside the configured training range")
    if not _resume_signatures_match(
        state["signature"], _signature(config),
        allow_offload_change=os.environ.get("VERL_STALE_RESUME_OFFLOAD_CHANGE") == "1",
    ):
        raise ValueError("Resume configuration differs from the saved stale-probe experiment")
    return Path(local), state


class ProbeRollouter(FullyAsyncRollouter.__ray_metadata__.modified_class):
    def _init_async_objects(self):
        super()._init_async_objects()
        self._admission_lock = asyncio.Lock()
        self._quiesced = False
        self._finished = False
        self._resume_epoch = 0

    async def pause_admission(self):
        """Stop new groups without waiting for native partial-rollout requests."""
        async with self._admission_lock:
            self._quiesced = True

    async def quiesce(self):
        await self.pause_admission()
        # A publisher never holds admission_lock: draining cannot deadlock on it.
        tasks = list(self.active_tasks)
        if tasks:
            await asyncio.gather(*tasks)
            self.active_tasks.difference_update(tasks)

    async def release(self):
        async with self._admission_lock:
            self._quiesced = False

    async def export_state(self):
        if not self._quiesced or self.active_tasks:
            raise RuntimeError("Freeze and drain rollout before saving its cursor")
        return {
            "data": self.train_dataloader.state_dict(),
            "next_sample": self.global_steps,
            "epoch": self._resume_epoch,
            "finished": self._finished,
            "generated": self.total_generated_samples,
            "processed": self.processed_sample_count,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

    def load_checkpoint(self):
        _, state = _read_resume(self.config)
        if state is None:
            return 0
        s = state["rollouter"]
        self.train_dataloader.load_state_dict(s["data"])
        self.global_steps, self._resume_epoch, self._finished = s["next_sample"], s["epoch"], s["finished"]
        self.total_generated_samples, self.processed_sample_count = s["generated"], s["processed"]
        random.setstate(s["python_rng"])
        np.random.set_state(s["numpy_rng"])
        torch.set_rng_state(s["torch_rng"])
        return state["version"]

    async def _process_single_sample_streaming(self, sample):
        # The streaming manager bypasses AgentLoopManager's ordinary batch setup.
        sample.full_batch.non_tensor_batch["__rollout_source_index__"] = np.arange(len(sample.full_batch))
        output = await self.async_rollout_manager.generate_sequences_single(sample.full_batch)
        # Like the ordinary manager, consume worker-local source metadata before
        # concatenation. This probe never resamples or discards prompt groups.
        indices = output.meta_info.pop("rollout_source_indices", None)
        if indices is not None and not np.array_equal(indices, np.arange(len(sample.full_batch))):
            raise ValueError("Whole-trajectory rollout changed prompt-group order")
        if output.meta_info.pop("oversample_discarded", 0):
            raise ValueError("Whole-trajectory rollout unexpectedly discarded responses")
        sample.full_batch = output
        output.non_tensor_batch["uid"] = np.array([f"uid_{sample.sample_id}"] * len(output), dtype=object)
        sample.rollout_status = await self.get_statistics()
        success = await self.message_queue_client.put_sample(sample=ray.cloudpickle.dumps(sample))
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        # One bounded producer replaces the upstream read-ahead/pending queue. A
        # checkpoint then needs only the dataloader cursor and completed FIFO.
        if self._finished:
            return  # The restored FIFO already contains the terminal marker.
        iterator = None
        self.running = True
        exhausted = False
        try:
            while not exhausted:
                for task in list(self.active_tasks):
                    if task.done():
                        self.active_tasks.remove(task)
                        await task  # Propagate generation failures; do not swallow them.
                async with self._admission_lock:
                    blocked = self._quiesced or await self._should_pause_generation()
                    if not blocked and len(self.active_tasks) < self.max_concurrent_samples:
                        if self.global_steps > self.total_rollout_steps:
                            exhausted = True
                        else:
                            if iterator is None:
                                iterator = iter(self.train_dataloader)
                            try:
                                batch = next(iterator)
                            except StopIteration:
                                self._resume_epoch += 1
                                if self._resume_epoch >= self.config.trainer.total_epochs:
                                    raise RuntimeError(
                                        "Dataset epochs exhausted before the requested sample budget"
                                    ) from None
                                iterator = iter(self.train_dataloader)
                                continue
                            sample = RolloutSample(
                                prepare_single_generation_data(batch, self.config),
                                f"sample_{self._resume_epoch}_{self.global_steps}",
                                self._resume_epoch,
                                {},
                            )
                            task = asyncio.create_task(self._process_single_sample_streaming(sample))
                            self.active_tasks.add(task)
                            self.global_steps += 1
                            self.staleness_samples += 1
                if not exhausted:
                    await asyncio.sleep(0.01)  # Yield to control RPCs and completion publishers.
            # Serialize the end marker with snapshot admission, after all outputs.
            async with self._admission_lock:
                if self.active_tasks:
                    await asyncio.gather(*self.active_tasks)
                    self.active_tasks.clear()
                await self.message_queue_client.put_sample(None)
                self._finished = True
        finally:
            self.running = False
            if self.active_tasks:
                tasks = list(self.active_tasks)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.active_tasks.clear()

    async def fit(self):
        # The upstream fit swallows gather(return_exceptions=True) failures.
        await self._streaming_generation_main()

    async def validate_version(self, version):
        if not self._quiesced:
            raise RuntimeError("Validation requires quiesced rollout")
        next_sample = self.global_steps
        try:
            self.global_steps = version
            return self.do_validate()
        finally:
            self.global_steps = next_sample


class ProbeTrainer(FullyAsyncTrainer.__ray_metadata__.modified_class):
    def set_total_train_steps(self, total_training_steps):
        if total_training_steps != self.config.trainer.total_training_steps:
            raise ValueError("Dataset epochs do not cover the requested full training cycles")
        super().set_total_train_steps(total_training_steps)

    def _fit_compute_log_prob(self, batch):
        from verl.experimental.fully_async_policy.replay_ablation import fit_compute_log_prob
        return fit_compute_log_prob(self, batch)

    def _fit_update_actor(self, batch):
        # Preserve the original 8 optimizer updates / one scheduler tick cadence.
        batch.meta_info["advance_lr_scheduler"] = self.local_trigger_step == self.trigger_parameter_sync_step
        return super()._fit_update_actor(batch)

    def _collect_metrics_from_samples(self, batch, metrics):
        super()._collect_metrics_from_samples(batch, metrics)
        starts = batch.non_tensor_batch["min_global_steps"]
        ends = batch.non_tensor_batch["max_global_steps"]
        if any(a != b for a, b in zip(starts, ends, strict=True)) and not self.config.async_training.partial_rollout:
            raise ValueError("A response crossed behavior-policy versions in whole-trajectory mode")
        # End-version lag alone understates the age of a resumed prefix. Keep
        # both ranges; token denominators remain the native generation log-probs.
        oldest = lag_metrics(list(starts), current_version=self.current_param_version,
                             completed_updates=self.global_steps - 1,
                             sync_every=self.trigger_parameter_sync_step)
        if any(a > b for a, b in zip(starts, ends, strict=True)):
            raise ValueError("Behavior policy versions moved backwards")
        metrics.update({
            "staleness/cross_version_response_fraction": float(np.mean(starts != ends)),
            "staleness/response_version_span_mean": float(np.mean(ends - starts)),
            "staleness/response_version_span_max": float(np.max(ends - starts)),
            "staleness/oldest_version_lag_mean": oldest["staleness/version_lag_mean"],
            "staleness/oldest_version_lag_max": oldest["staleness/version_lag_max"],
        })
        metrics.update(
            lag_metrics(
                list(ends),
                current_version=self.current_param_version,
                completed_updates=self.global_steps - 1,
                sync_every=self.trigger_parameter_sync_step,
            )
        )
        _, counts = np.unique(batch.non_tensor_batch["uid"], return_counts=True)
        if len(counts) != self.required_samples or np.any(counts != self.config.actor_rollout_ref.rollout.n):
            raise ValueError("Incomplete or mixed GRPO prompt groups")

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return
        wait_start = time.monotonic()
        if self.config.async_training.partial_rollout:
            await self.rollouter.pause_admission.remote()
        else:
            await self.rollouter.quiesce.remote()
        self.timing_raw["sync_admission_wait"] = time.monotonic() - wait_start
        try:
            start = time.monotonic()
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
            self.timing_raw["timing_s/param_sync"] = time.monotonic() - start
            self.metrics.update(await self.rollouter.reset_staleness.remote())
        finally:
            await self.rollouter.release.remote()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return
        freq = self.config.trainer.test_freq
        if not val_before_train and not (
            freq > 0 and self.current_param_version > 0 and self.current_param_version % freq == 0
        ):
            return
        await self.rollouter.quiesce.remote()
        try:
            values = await self.rollouter.validate_version.remote(self.current_param_version)
            values["training/global_step"] = self.current_param_version
            self.logger.log(data=values, step=self.current_param_version)
        finally:
            await self.rollouter.release.remote()

    def _fit_postprocess_step(self):
        super()._fit_postprocess_step()
        if self.local_trigger_step == 1:
            values = self.metrics_aggregator.get_aggregated_metrics()
            values.update(
                {
                    "training/global_step": self.current_param_version,
                    "training/optimizer_updates": self.global_steps - 1,
                    "training/prompts_consumed": (self.global_steps - 1) * self.required_samples,
                }
            )
            self.logger.log(data=values, step=self.current_param_version)
            self.metrics_aggregator.reset()
        if getattr(self, "_deferred_checkpoint", False):
            # Include the completed cycle's training AND validation log in HDFS.
            self._deferred_checkpoint = False
            super()._fit_save_checkpoint()

    def _fit_save_checkpoint(self, force=False):
        if self.local_trigger_step != 1:
            if force:
                raise RuntimeError("Refusing to publish a partial parameter-sync cycle")
            return
        if force:
            super()._fit_save_checkpoint(force=True)
        else:
            self._deferred_checkpoint = True

    def _save_checkpoint(self):
        from verl.trainer.ppo.ray_trainer import prune_complete_checkpoints
        from verl.utils import hdfs_io
        from verl.utils.checkpoint.hdfs_checkpoint import (
            _copy_required,
            publish_remote_metadata,
            verify_remote_actor_checkpoint,
        )

        v = self.current_param_version
        local = Path(self.config.trainer.default_local_dir) / f"global_step_{v}"
        root = self.config.trainer.default_hdfs_dir
        remote = f"{root}/global_step_{v}" if root else None
        if remote and hdfs_io.exists(remote + "/_SUCCESS"):
            raise RuntimeError("Refusing to overwrite a published checkpoint")
        ray.get(self.rollouter.quiesce.remote())
        try:
            state = {
                "version": v,
                "signature": _signature(self.config),
                "rollouter": ray.get(self.rollouter.export_state.remote()),
                "queue": ray.get(self.message_queue_client.queue_actor.snapshot.remote()),
                "stale_trajectory_processed": self.stale_trajectory_processed,
            }
            if remote:
                pending = local.parent / "pending_checkpointed_iteration.txt"
                atomic_write(pending, str(v).encode())
                hdfs_io.makedirs(root, exist_ok=True)
                _copy_required(str(pending), root + "/" + pending.name)
            self.actor_rollout_wg.save_checkpoint(
                str(local / "actor"), remote + "/actor" if remote else None, v, max_ckpt_to_keep=1
            )
            if remote:
                verify_remote_actor_checkpoint(
                    remote + "/actor",
                    self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node,
                    strategy=self.config.actor_rollout_ref.actor.strategy,
                )
            save_probe_state(local, state)
            log = Path(self.config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH)
            if log.is_file():
                atomic_write(local / "metrics.jsonl", log.read_bytes())
            tracker = local.parent / "latest_checkpointed_iteration.txt"
            atomic_write(tracker, str(v).encode())
            if remote:
                publish_remote_metadata(root, local, tracker)
            prune_complete_checkpoints(local.parent, 1)
        finally:
            ray.get(self.rollouter.release.remote())

    async def load_checkpoint(self):
        local, state = _read_resume(self.config)
        if state is None:
            return 0
        self.current_param_version = state["version"]
        # Parent fit() adds one before consuming its first batch.
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step
        self.local_trigger_step = 1
        self.last_ckpt_version = self.current_param_version
        self.stale_trajectory_processed = state["stale_trajectory_processed"]
        root = self.config.trainer.default_hdfs_dir
        remote = f"{root}/global_step_{self.current_param_version}/actor" if root else None
        self.actor_rollout_wg.load_checkpoint(
            str(local / "actor"), hdfs_path=remote, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if root:
            from verl.utils.checkpoint.hdfs_checkpoint import _copy_required

            # Commit an adopted orphan before replacing its pending hint at the
            # next save. Only the pointer changes; checkpoint payloads are immutable.
            tracker = local.parent / "latest_checkpointed_iteration.txt"
            # A current LOCAL tracker does not prove remote publication succeeded.
            atomic_write(tracker, str(self.current_param_version).encode())
            _copy_required(str(tracker), root + "/" + tracker.name)
        await self.message_queue_client.queue_actor.restore.remote(state["queue"])
        return self.current_param_version

    async def fit(self):
        if self.current_param_version == self.config.trainer.total_training_steps:
            return  # A terminal checkpoint may already have consumed its None marker.
        await super().fit()
        expected = self.config.trainer.total_training_steps * self.trigger_parameter_sync_step
        if self.global_steps - 1 != expected:
            raise RuntimeError(f"Consumed {self.global_steps - 1} optimizer batches; expected {expected}")


class ProbeTaskRunner(FullyAsyncTaskRunner.__ray_metadata__.modified_class):
    trainer_class = ray.remote(num_cpus=10)(ProbeTrainer)
    rollouter_class = ray.remote(num_cpus=10)(ProbeRollouter)
    queue_class = ray.remote(num_cpus=2, max_concurrency=20)(ProbeQueue)


def run(config):
    """Use the established run_ppo environment and job setup, with opt-in actors."""
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    if config.actor_rollout_ref.rollout.over_sample_rate != 0:
        raise ValueError("Stale probe requires no oversampling")
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("The probe has no value-function/critic checkpoint path")
    auto_set_device(config)
    # Successful resolved recipes have already migrated and removed legacy keys.
    if "reward_model" in config:
        config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ProbeTaskRunner))
