# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os
import shutil
import uuid
from pathlib import Path

from verl.utils import hdfs_io


def _copy_required(src: str, dst: str) -> None:
    if not hdfs_io.copy(src=src, dst=dst):
        raise RuntimeError(f"HDFS checkpoint copy failed: {src} -> {dst}")


def _copy_required_to_local(src: str, dst: str | Path) -> None:
    """Download via a fresh sibling path, then atomically replace stale local state."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.download")
    try:
        _copy_required(src, str(temporary))
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_required_directory_to_local(src: str, dst: str | Path) -> None:
    """Download an HDFS directory atomically and retain a completed local copy."""
    dst = Path(dst)
    marker_name = ".hdfs_restore_complete"
    marker = dst / marker_name
    if marker.is_file() and marker.read_text(encoding="utf-8") == src:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.download")
    stale = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.stale")
    try:
        _copy_required(src, str(temporary))
        (temporary / marker_name).write_text(src, encoding="utf-8")
        if dst.exists():
            os.replace(dst, stale)
        os.replace(temporary, dst)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(stale, ignore_errors=True)


MEGATRON_HDFS_MANIFEST = "megatron_hdfs_manifest.json"


def _read_remote_json(remote_path: str, local_parent: str | Path) -> dict:
    local_parent = Path(local_parent)
    local_parent.mkdir(parents=True, exist_ok=True)
    temporary = local_parent / f".{Path(remote_path).name}.{uuid.uuid4().hex}.download"
    try:
        _copy_required(remote_path, str(temporary))
        return json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def _validate_megatron_manifest(manifest: dict, world_size: int) -> None:
    if manifest.get("format") != "megatron-node-shards-v1":
        raise RuntimeError("Unsupported or missing Megatron HDFS checkpoint manifest")
    if manifest.get("world_size") != world_size:
        raise RuntimeError(
            f"Megatron HDFS checkpoint world size mismatch: saved={manifest.get('world_size')} requested={world_size}"
        )
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("Megatron HDFS checkpoint manifest has no node shards")
    for node in nodes:
        hostname = node.get("hostname")
        files = node.get("files")
        if not isinstance(hostname, str) or not hostname or "/" in hostname:
            raise RuntimeError(f"Invalid Megatron checkpoint hostname: {hostname!r}")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"Megatron checkpoint node {hostname} has no files")
        for relative in files:
            if not isinstance(relative, str) or relative in ("", "."):
                raise RuntimeError(f"Invalid Megatron checkpoint relative path: {relative!r}")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Invalid Megatron checkpoint relative path: {relative!r}")


def _select_megatron_restore_files(manifest: dict, leader_rank: int) -> list[tuple[str, str]]:
    """Select shared metadata plus the saved shards for one current physical node."""
    nodes_by_leader = {node["leader_rank"]: node for node in manifest["nodes"]}
    if 0 not in nodes_by_leader:
        raise RuntimeError("Megatron HDFS checkpoint manifest has no rank-zero node")
    if leader_rank not in nodes_by_leader:
        raise RuntimeError(f"Megatron HDFS checkpoint has no shards for node leader rank {leader_rank}")

    shared_node = nodes_by_leader[0]
    local_node = nodes_by_leader[leader_rank]
    selected = [
        (shared_node["hostname"], relative)
        for relative in shared_node["files"]
        if not Path(relative).name.startswith("__")
    ]
    selected.extend(
        (local_node["hostname"], relative) for relative in local_node["files"] if Path(relative).name.startswith("__")
    )
    return selected


def build_megatron_manifest(hostnames: list[str], files_by_rank: list[list[str]]) -> dict:
    if not hostnames or len(hostnames) != len(files_by_rank):
        raise RuntimeError("Megatron checkpoint host/file inventories do not match world size")
    nodes = []
    for rank, hostname in enumerate(hostnames):
        if hostname in hostnames[:rank]:
            continue
        files = sorted(set(files_by_rank[rank]))
        for peer_rank, peer_hostname in enumerate(hostnames):
            if peer_hostname == hostname and sorted(set(files_by_rank[peer_rank])) != files:
                raise RuntimeError(
                    f"Megatron checkpoint inventory differs within node {hostname}: ranks {rank} and {peer_rank}"
                )
        nodes.append(
            {
                "hostname": hostname,
                "leader_rank": rank,
                "files": files,
            }
        )
    manifest = {
        "format": "megatron-node-shards-v1",
        "world_size": len(hostnames),
        "nodes": nodes,
    }
    _validate_megatron_manifest(manifest, len(hostnames))
    return manifest


def upload_megatron_node_checkpoint(
    local_dist: str | Path,
    remote_actor: str,
    rank: int,
    manifest: dict,
) -> bool:
    node = next((item for item in manifest["nodes"] if item["leader_rank"] == rank), None)
    if node is None:
        return False
    local_dist = Path(local_dist)
    remote_dist = os.path.join(
        remote_actor,
        "node_shards",
        node["hostname"],
        "dist_ckpt",
    )
    for relative in node["files"]:
        local_file = local_dist / relative
        if not local_file.is_file():
            raise RuntimeError(f"Missing local Megatron checkpoint file: {local_file}")
        remote_file = os.path.join(remote_dist, relative)
        hdfs_io.makedirs(os.path.dirname(remote_file), exist_ok=True)
        _copy_required(str(local_file), remote_file)
    return True


def publish_megatron_manifest(
    local_actor: str | Path,
    remote_actor: str,
    manifest: dict,
) -> None:
    _validate_megatron_manifest(manifest, manifest.get("world_size"))
    local_actor = Path(local_actor)
    local_actor.mkdir(parents=True, exist_ok=True)
    local_manifest = local_actor / MEGATRON_HDFS_MANIFEST
    local_manifest.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    hdfs_io.makedirs(remote_actor, exist_ok=True)
    _copy_required(
        str(local_manifest),
        os.path.join(remote_actor, MEGATRON_HDFS_MANIFEST),
    )


def _load_megatron_manifest(remote_actor: str, local_parent: str | Path, world_size: int) -> dict:
    manifest = _read_remote_json(
        os.path.join(remote_actor, MEGATRON_HDFS_MANIFEST),
        local_parent,
    )
    _validate_megatron_manifest(manifest, world_size)
    return manifest


def restore_remote_megatron_checkpoint(
    remote_actor: str,
    local_actor: str | Path,
    rank: int,
    hostnames: list[str],
) -> bool:
    """Materialize the merged Megatron dist checkpoint once on each physical node."""
    hostname = hostnames[rank]
    if rank != hostnames.index(hostname):
        return False

    local_actor = Path(local_actor)
    manifest = _load_megatron_manifest(remote_actor, local_actor.parent, len(hostnames))
    leader_rank = hostnames.index(hostname)

    local_actor.mkdir(parents=True, exist_ok=True)
    remote_huggingface = os.path.join(remote_actor, "huggingface")
    if hdfs_io.exists(remote_huggingface):
        _copy_required_directory_to_local(
            remote_huggingface,
            local_actor / "huggingface",
        )

    local_dist = local_actor / "dist_ckpt"
    temporary_dist = local_actor / f".dist_ckpt.{uuid.uuid4().hex}.download"
    stale_dist = local_actor / f".dist_ckpt.{uuid.uuid4().hex}.stale"
    try:
        temporary_dist.mkdir()
        for saved_hostname, relative in _select_megatron_restore_files(manifest, leader_rank):
            remote_dist = os.path.join(
                remote_actor,
                "node_shards",
                saved_hostname,
                "dist_ckpt",
            )
            _copy_required_to_local(
                os.path.join(remote_dist, relative),
                temporary_dist / relative,
            )
        if local_dist.exists():
            os.replace(local_dist, stale_dist)
        os.replace(temporary_dist, local_dist)
    finally:
        shutil.rmtree(temporary_dist, ignore_errors=True)
        shutil.rmtree(stale_dist, ignore_errors=True)
    return True


def verify_remote_actor_checkpoint(remote_actor: str, world_size: int, strategy: str = "fsdp") -> None:
    """Refuse to publish a checkpoint whose distributed actor state is incomplete."""
    if strategy == "megatron":
        manifest = _load_megatron_manifest(remote_actor, Path("/tmp"), world_size)
        missing = []
        for node in manifest["nodes"]:
            remote_dist = os.path.join(
                remote_actor,
                "node_shards",
                node["hostname"],
                "dist_ckpt",
            )
            for relative in node["files"]:
                remote_file = os.path.join(remote_dist, relative)
                if not hdfs_io.exists(remote_file):
                    missing.append(remote_file)
        if missing:
            raise RuntimeError(
                "HDFS Megatron actor checkpoint is incomplete; refusing to publish latest pointer. "
                f"Missing {len(missing)} file(s), first missing: {missing[0]}"
            )
        return
    if strategy not in ("fsdp", "fsdp2"):
        raise ValueError(f"Unsupported actor checkpoint strategy: {strategy}")
    missing = []
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world_size):
            filename = f"{kind}_world_size_{world_size}_rank_{rank}.pt"
            remote_file = os.path.join(remote_actor, filename)
            if not hdfs_io.exists(remote_file):
                missing.append(remote_file)
    if missing:
        raise RuntimeError(
            "HDFS actor checkpoint is incomplete; refusing to publish latest pointer. "
            f"Missing {len(missing)} shard(s), first missing: {missing[0]}"
        )


def publish_remote_metadata(remote_root: str, local_step_folder: str | Path, local_tracker: str | Path) -> None:
    local_step_folder = Path(local_step_folder)
    remote_step = os.path.join(remote_root, local_step_folder.name)
    hdfs_io.makedirs(remote_step, exist_ok=True)
    for name in ("data.pt", "_SUCCESS"):
        _copy_required(str(local_step_folder / name), os.path.join(remote_step, name))
    metrics = local_step_folder / "metrics.jsonl"
    if metrics.exists():
        _copy_required(str(metrics), os.path.join(remote_step, metrics.name))
    _copy_required(str(local_tracker), os.path.join(remote_root, "latest_checkpointed_iteration.txt"))


def download_remote_metadata(remote_root: str, local_root: str | Path) -> tuple[str, str]:
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    local_tracker = local_root / "latest_checkpointed_iteration.txt"
    _copy_required_to_local(os.path.join(remote_root, local_tracker.name), local_tracker)
    step = int(local_tracker.read_text(encoding="utf-8").strip())
    local_step = local_root / f"global_step_{step}"
    remote_step = os.path.join(remote_root, local_step.name)
    local_step.mkdir(parents=True, exist_ok=True)
    remote_success = os.path.join(remote_step, "_SUCCESS")
    _copy_required_to_local(remote_success, local_step / "_SUCCESS")
    remote_data = os.path.join(remote_step, "data.pt")
    if hdfs_io.exists(remote_data):
        _copy_required_to_local(remote_data, local_step / "data.pt")
    return str(local_step), remote_step
