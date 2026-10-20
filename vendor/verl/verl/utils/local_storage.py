# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Safety helpers for bounding recoverable local training artifacts."""

from __future__ import annotations

import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


def _require_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Refusing to remove path outside checkpoint root: {path}") from error


def _remove_empty_parents(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


class BoundedFileCache:
    """Keep at most ``max_files`` matching files under a cache root.

    Files are ordered by modification time on startup and by access via
    :meth:`record` afterwards. Only files matching ``pattern`` are managed;
    manifests, locks, and unrelated user files are left untouched.
    """

    def __init__(self, root: str | Path, *, pattern: str, max_files: int) -> None:
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pattern = pattern
        self.max_files = max_files
        existing = sorted(
            (path.resolve() for path in self.root.rglob(pattern) if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        self._files: OrderedDict[Path, None] = OrderedDict((path, None) for path in existing)
        self._prune()

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)

    def record(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        _require_within(path, self.root)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._files.pop(path, None)
        self._files[path] = None
        self._prune()

    def _prune(self) -> None:
        while len(self._files) > self.max_files:
            path, _ = self._files.popitem(last=False)
            path.unlink(missing_ok=True)
            _remove_empty_parents(path, self.root)


def remove_uploaded_checkpoint_files(paths: Iterable[str | Path], *, root: str | Path) -> None:
    """Delete only explicitly listed local files after synchronous HDFS upload."""

    root = Path(root).expanduser().resolve()
    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    for path in resolved_paths:
        _require_within(path, root)
    for path in resolved_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        _remove_empty_parents(path, root)


def remove_published_checkpoint_tree(
    checkpoint_path: str | Path,
    *,
    checkpoint_root: str | Path,
) -> None:
    """Remove a local step only after its durable HDFS pointer is published."""

    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.parent != checkpoint_root or re.fullmatch(r"global_step_\d+", checkpoint_path.name) is None:
        raise ValueError(f"Checkpoint must be a direct global_step child of {checkpoint_root}: {checkpoint_path}")
    shutil.rmtree(checkpoint_path, ignore_errors=False)
