"""Opt-in configuration for a no-oversampling Qwen off8 staleness experiment."""

import copy
import hashlib
import json
import math
import os
import pickle
import re
import uuid
from pathlib import Path


def build_probe_config(base, *, staleness, run_id, state_dir, hdfs_dir, rollout_nodes=1):
    if not math.isfinite(staleness) or staleness < 0:
        raise ValueError("staleness must be finite and nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not hdfs_dir.startswith("hdfs://") or rollout_nodes < 1:
        raise ValueError("A new run identity, HDFS directory and rollout resources are required")
    if run_id == base["trainer"].get("experiment_name") or hdfs_dir.rstrip("/") == str(
        base["trainer"].get("default_hdfs_dir", "")
    ).rstrip("/"):
        raise ValueError("The async probe must not reuse the baseline run or checkpoint namespace")
    c = copy.deepcopy(base)
    actor = c["actor_rollout_ref"]["actor"]
    rollout = c["actor_rollout_ref"]["rollout"]
    batch, mini = c["data"]["train_batch_size"], actor["ppo_mini_batch_size"]
    if batch != 64 or mini != 8 or actor["ppo_epochs"] != 1 or rollout["n"] != 8:
        raise ValueError("This probe expects the established b64 n8 mb8 ep1 off8 recipe")
    if actor["policy_loss"]["loss_mode"] not in (
        "prefix_geometric_probability_weighted_exact_kl_dual_clip", "gspo", "vanilla"
    ):
        raise ValueError("Only Exact Geom Dual, GSPO and vanilla GRPO/ClipHigher are supported by this probe")
    if actor.get("megatron", {}).get("router_replay", {}).get("mode", "disabled") != "disabled" or rollout.get(
        "enable_rollout_routing_replay", False
    ):
        raise ValueError("Use the router-replay-disabled baseline for this probe")
    cycles = c["trainer"]["total_training_steps"]
    if cycles < 1:
        raise ValueError("Training cycles must be positive")
    c["actor_rollout_ref"]["hybrid_engine"] = False
    c["actor_rollout_ref"]["_checkpoint_hdfs_dir"] = hdfs_dir
    actor["calculate_entropy"] = True
    actor.setdefault("checkpoint", {})["async_save"] = False
    rollout.update(
        over_sample_rate=0.0, calculate_log_probs=True, mode="async", nnodes=rollout_nodes, n_gpus_per_node=8
    )
    rollout.setdefault("checkpoint_engine", {})["backend"] = "nccl"
    rollout.setdefault("skip", {})["enable"] = False
    if "trace" in rollout:
        rollout["trace"]["experiment_name"] = run_id
    c["data"].update(train_batch_size=0, gen_batch_size=1)
    # The adapter supplies the behavior denominator without replacing the loss.
    c["algorithm"]["rollout_correction"] = None
    c["async_training"] = dict(
        staleness_threshold=staleness,
        trigger_parameter_sync_step=8,
        require_batches=1,
        partial_rollout=False,
        use_trainer_do_validate=False,
    )
    c["rollout"] = dict(nnodes=rollout_nodes, n_gpus_per_node=8, n=8, total_rollout_steps=batch * cycles)
    c["trainer"].update(
        experiment_name=run_id,
        default_hdfs_dir=hdfs_dir,
        resume_hdfs_dir=hdfs_dir,
        default_local_dir=str(Path(state_dir) / "checkpoints"),
        validation_data_dir=str(Path(state_dir) / "validation"),
        resume_mode="auto",
        max_actor_ckpt_to_keep=1,
        max_critic_ckpt_to_keep=1,
        max_complete_ckpt_to_keep=1,
    )
    c["trainer"]["rollout_buffer"] = {"enable": False, "delay_steps": 0}
    runtime = c.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
    env = runtime.setdefault("env_vars", {})
    env.update(
        WANDB_RUN_ID=run_id,
        WANDB_NAME=run_id,
        WANDB_RESUME="allow",
        WANDB_RUN_GROUP="PR2_Qwen30BA3B_off8_staleness_nooversample",
        VERL_FILE_LOGGER_PATH=str(Path(state_dir) / "metrics.jsonl"),
        WANDB_DIR=str(Path(state_dir) / "wandb"),
        TENSORBOARD_DIR=str(Path(state_dir) / "tensorboard"),
    )
    return c


def lag_metrics(versions, *, current_version, completed_updates, sync_every):
    if not versions or any(v is None or int(v) != v or v < 0 or v > current_version for v in versions):
        raise ValueError("Missing, invalid, or future behavior policy version")
    ages = [current_version - v for v in versions]
    updates = [completed_updates - v * sync_every for v in versions]
    if min(updates) < 0:
        raise ValueError("Behavior version is newer than the completed learner updates")
    return {
        "staleness/version_lag_mean": sum(ages) / len(ages),
        "staleness/version_lag_max": max(ages),
        "staleness/stale_fraction": sum(a > 0 for a in ages) / len(ages),
        "staleness/lag_2plus_fraction": sum(a >= 2 for a in ages) / len(ages),
        "staleness/update_lag_mean": sum(updates) / len(updates),
        "staleness/update_lag_max": max(updates),
    }


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def save_probe_state(directory, state):
    directory = Path(directory)
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write(directory / "data.pt", raw)
    atomic_write(
        directory / "_SUCCESS",
        json.dumps(
            {
                "format": "stale-probe-v1",
                "schema_version": 1,
                "global_step": state["version"],
                "version": state["version"],
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ).encode(),
    )


def load_probe_state(directory):
    directory = Path(directory)
    marker = json.loads((directory / "_SUCCESS").read_text())
    raw = (directory / "data.pt").read_bytes()
    if marker.get("format") != "stale-probe-v1" or hashlib.sha256(raw).hexdigest() != marker.get("sha256"):
        raise ValueError("Incomplete or incompatible stale-probe checkpoint")
    # Only load checkpoints produced by this experiment in its trusted HDFS namespace.
    state = pickle.loads(raw)
    if state["version"] != marker["version"]:
        raise ValueError("Checkpoint policy version mismatch")
    return state
