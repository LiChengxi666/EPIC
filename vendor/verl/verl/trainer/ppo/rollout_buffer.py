# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Small synchronous rollout buffer for controlled policy-lag experiments."""

import hashlib
import json
import os
import pickle
import re
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from verl import DataProto


_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKSUM_CHUNK_SIZE = 1024 * 1024
_BATCH_FILENAME_PATTERN = re.compile(r"batch_([0-9a-f]{32})_(\d+)\.pkl")


class RolloutBufferCheckpointError(RuntimeError):
    """Raised when a persisted rollout buffer cannot be safely restored."""


@dataclass(frozen=True)
class BufferedRollout:
    """A rollout batch and the policy version that generated it."""

    batch: DataProto
    policy_version: int
    generation_step: int


class FixedDelayRolloutBuffer:
    """FIFO buffer that releases a rollout after a fixed number of generations.

    A delay of zero is the ordinary synchronous path. With ``delay_steps=2``,
    the first two pushes warm up the queue and every later push releases the
    batch generated two rollout iterations earlier.
    """

    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be non-negative, got {delay_steps}")
        self.delay_steps = delay_steps
        self._queue: deque[BufferedRollout] = deque()

    def push(
        self,
        batch: DataProto,
        *,
        policy_version: int,
        generation_step: int,
    ) -> BufferedRollout | None:
        self._queue.append(
            BufferedRollout(batch=batch, policy_version=policy_version, generation_step=generation_step)
        )
        if len(self._queue) <= self.delay_steps:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)

    def save_to_directory(
        self,
        directory: str | os.PathLike,
        checkpoint_step: int,
        generation_step: int,
    ) -> None:
        """Persist the buffered rollouts and publish their manifest atomically."""
        checkpoint_directory = Path(directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        if len(self._queue) != self.delay_steps:
            raise RolloutBufferCheckpointError("rollout buffer queue length does not match configured delay")

        batches = []
        snapshot_id = uuid.uuid4().hex
        for index, rollout in enumerate(self._queue):
            filename = f"batch_{snapshot_id}_{index}.pkl"
            batch_path = checkpoint_directory / filename
            temporary_path = checkpoint_directory / f"{filename}.tmp"
            with temporary_path.open("wb") as file:
                pickle.dump(rollout, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, batch_path)
            batches.append(
                {
                    "filename": filename,
                    "policy_version": rollout.policy_version,
                    "generation_step": rollout.generation_step,
                    "size_bytes": batch_path.stat().st_size,
                    "sha256": _sha256(batch_path),
                }
            )

        # Persist every published batch directory entry before making a
        # manifest that references those entries authoritative.
        _fsync_directory(checkpoint_directory)

        manifest = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_step": checkpoint_step,
            "delay_steps": self.delay_steps,
            "generation_step": generation_step,
            "batches": batches,
        }
        temporary_manifest_path = checkpoint_directory / "manifest.json.tmp"
        with temporary_manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_manifest_path, checkpoint_directory / "manifest.json")
        _fsync_directory(checkpoint_directory)
        _remove_obsolete_batch_files(checkpoint_directory, {batch["filename"] for batch in batches})

    def load_from_directory(
        self,
        directory: str | os.PathLike,
        expected_checkpoint_step: int,
    ) -> int:
        """Restore buffered rollouts after validating the complete checkpoint."""
        checkpoint_directory = Path(directory)
        manifest = _load_manifest(checkpoint_directory)
        _validate_manifest(manifest, expected_checkpoint_step, self.delay_steps)

        restored_queue = deque()
        snapshot_id = None
        for index, batch_metadata in enumerate(manifest["batches"]):
            snapshot_id = _validate_batch_metadata(batch_metadata, index, snapshot_id)
            batch_path = checkpoint_directory / batch_metadata["filename"]
            _validate_batch_file(batch_path, batch_metadata)
            restored_queue.append(_load_buffered_rollout(batch_path, batch_metadata))

        self._queue = restored_queue
        return manifest["generation_step"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(checkpoint_directory: Path) -> dict:
    try:
        with (checkpoint_directory / "manifest.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RolloutBufferCheckpointError("could not read rollout buffer manifest") from error

    if not isinstance(manifest, dict):
        raise RolloutBufferCheckpointError("rollout buffer manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: dict, expected_checkpoint_step: int, delay_steps: int) -> None:
    expected_fields = {"schema_version", "checkpoint_step", "delay_steps", "generation_step", "batches"}
    if set(manifest) != expected_fields:
        raise RolloutBufferCheckpointError("rollout buffer manifest has unexpected fields")
    if not _is_int(manifest["schema_version"]):
        raise RolloutBufferCheckpointError("rollout buffer manifest schema version must be an integer")
    if manifest["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise RolloutBufferCheckpointError("unsupported rollout buffer manifest schema")
    if not _is_int(manifest["checkpoint_step"]):
        raise RolloutBufferCheckpointError("rollout buffer checkpoint step must be an integer")
    if manifest["checkpoint_step"] != expected_checkpoint_step:
        raise RolloutBufferCheckpointError("rollout buffer checkpoint step does not match")
    if not _is_int(manifest["delay_steps"]):
        raise RolloutBufferCheckpointError("rollout buffer delay steps must be an integer")
    if manifest["delay_steps"] != delay_steps:
        raise RolloutBufferCheckpointError("rollout buffer delay steps do not match")
    if not _is_int(manifest["generation_step"]):
        raise RolloutBufferCheckpointError("rollout buffer generation step must be an integer")
    if not isinstance(manifest["batches"], list):
        raise RolloutBufferCheckpointError("rollout buffer batches must be a list")
    if len(manifest["batches"]) != delay_steps:
        raise RolloutBufferCheckpointError("rollout buffer queue length does not match configured delay")
    expected_policy_versions = list(range(expected_checkpoint_step - delay_steps, expected_checkpoint_step))
    policy_versions = [batch.get("policy_version") for batch in manifest["batches"] if isinstance(batch, dict)]
    if policy_versions != expected_policy_versions:
        raise RolloutBufferCheckpointError("rollout buffer policy versions do not match checkpoint step")
    expected_generation_steps = list(range(manifest["generation_step"] - delay_steps + 1, manifest["generation_step"] + 1))
    generation_steps = [batch.get("generation_step") for batch in manifest["batches"] if isinstance(batch, dict)]
    if generation_steps != expected_generation_steps:
        raise RolloutBufferCheckpointError("rollout buffer generation steps are not the latest contiguous window")


def _validate_batch_metadata(batch_metadata: object, index: int, snapshot_id: str | None) -> str:
    if not isinstance(batch_metadata, dict):
        raise RolloutBufferCheckpointError("rollout buffer batch metadata must be an object")

    expected_fields = {"filename", "policy_version", "generation_step", "size_bytes", "sha256"}
    if set(batch_metadata) != expected_fields:
        raise RolloutBufferCheckpointError("rollout buffer batch metadata has unexpected fields")
    filename = batch_metadata["filename"]
    if not isinstance(filename, str):
        raise RolloutBufferCheckpointError("rollout buffer batch filename is invalid")
    filename_match = _BATCH_FILENAME_PATTERN.fullmatch(filename)
    if filename_match is None or int(filename_match.group(2)) != index:
        raise RolloutBufferCheckpointError("rollout buffer batch filename is out of order")
    batch_snapshot_id = filename_match.group(1)
    if snapshot_id is not None and batch_snapshot_id != snapshot_id:
        raise RolloutBufferCheckpointError("rollout buffer batches belong to different snapshots")
    if not _is_int(batch_metadata["policy_version"]):
        raise RolloutBufferCheckpointError("rollout buffer policy version must be an integer")
    if not _is_int(batch_metadata["generation_step"]):
        raise RolloutBufferCheckpointError("rollout buffer batch generation step must be an integer")
    if not _is_int(batch_metadata["size_bytes"]) or batch_metadata["size_bytes"] < 0:
        raise RolloutBufferCheckpointError("rollout buffer batch size must be a non-negative integer")
    checksum = batch_metadata["sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise RolloutBufferCheckpointError("rollout buffer batch checksum is invalid")
    try:
        int(checksum, 16)
    except ValueError as error:
        raise RolloutBufferCheckpointError("rollout buffer batch checksum is invalid") from error
    return batch_snapshot_id


def _validate_batch_file(batch_path: Path, batch_metadata: dict) -> None:
    try:
        if batch_path.stat().st_size != batch_metadata["size_bytes"]:
            raise RolloutBufferCheckpointError("rollout buffer batch size does not match manifest")
        if _sha256(batch_path) != batch_metadata["sha256"]:
            raise RolloutBufferCheckpointError("rollout buffer batch checksum does not match manifest")
    except OSError as error:
        raise RolloutBufferCheckpointError("could not read rollout buffer batch") from error


def _load_buffered_rollout(batch_path: Path, batch_metadata: dict) -> BufferedRollout:
    try:
        with batch_path.open("rb") as file:
            buffered_rollout = pickle.load(file)
    except Exception as error:
        raise RolloutBufferCheckpointError("could not deserialize rollout buffer batch") from error

    if not isinstance(buffered_rollout, BufferedRollout) or not isinstance(buffered_rollout.batch, DataProto):
        raise RolloutBufferCheckpointError("rollout buffer batch payload is invalid")
    if not _is_int(buffered_rollout.policy_version) or not _is_int(buffered_rollout.generation_step):
        raise RolloutBufferCheckpointError("rollout buffer batch payload metadata is invalid")
    if (
        buffered_rollout.policy_version != batch_metadata["policy_version"]
        or buffered_rollout.generation_step != batch_metadata["generation_step"]
    ):
        raise RolloutBufferCheckpointError("rollout buffer batch payload metadata does not match manifest")
    return buffered_rollout


def _remove_obsolete_batch_files(checkpoint_directory: Path, current_filenames: set[str]) -> None:
    for batch_path in checkpoint_directory.glob("batch_*.pkl"):
        if batch_path.name not in current_filenames:
            try:
                batch_path.unlink()
            except OSError:
                pass


def _fsync_directory(directory: Path) -> None:
    """Make preceding directory-entry updates durable before publication continues."""
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
