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

import gzip
import json
import math
import os
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.local_storage import BoundedFileCache
from verl.utils.model import compute_position_id_with_mask


class TeacherTrajectoryDataset(Dataset):
    """Read fixed teacher trajectories without invoking a rollout engine.

    The schema-v2 rollout corpus stores one gzip JSONL shard for a fixed number
    of prompts and keeps the behavior-policy token log probabilities alongside
    the exact generated token IDs. Dataset indices follow shard/line order so a
    sequential StatefulDataLoader resumes deterministically.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: DictConfig,
        processor: Optional[Any] = None,
        max_samples: int = -1,
    ) -> None:
        del processor
        if isinstance(data_files, str):
            data_files = [data_files]
        offline_config = config.get("offline_trajectory")
        if not offline_config or not offline_config.get("enable", False):
            raise ValueError("TeacherTrajectoryDataset requires data.offline_trajectory.enable=true")

        self.config = config
        self.offline_config = offline_config
        self.tokenizer = tokenizer
        self.root = str(offline_config.get("root") or data_files[0]).rstrip("/")
        self.cache_dir = os.path.expanduser(str(offline_config.get("cache_dir", "/tmp/verl_teacher_trajectory")))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_cached_shards = max(1, int(offline_config.get("max_cached_shards", 2)))
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._disk_shard_cache = BoundedFileCache(
            self.cache_dir,
            pattern="part-*.jsonl.gz",
            max_files=self.max_cached_shards,
        )

        manifest_path = copy_to_local(f"{self.root}/manifest.json", cache_dir=self.cache_dir)
        with open(manifest_path) as handle:
            self.manifest = json.load(handle)
        if int(self.manifest.get("schema_version", -1)) != 2:
            raise ValueError(f"Expected teacher trajectory schema_version=2, got {self.manifest.get('schema_version')}")

        self.samples_per_prompt = int(self.manifest["samples_per_prompt"])
        self.shard_prompt_count = int(self.manifest["shard_prompt_count"])
        self.records_per_full_shard = self.samples_per_prompt * self.shard_prompt_count
        manifest_trajectories = int(self.manifest["unique_prompts"]) * self.samples_per_prompt
        self.real_num_trajectories = (
            min(manifest_trajectories, max_samples) if max_samples is not None and max_samples > 0 else manifest_trajectories
        )
        self.num_shards = math.ceil(manifest_trajectories / self.records_per_full_shard)
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.max_response_length = int(
            offline_config.get(
                "max_response_length",
                config.get("max_response_length", self.manifest["sampling"]["max_tokens"]),
            )
        )

        pad_to_multiple = max(1, int(offline_config.get("pad_to_multiple", 1)))
        self.num_trajectories = math.ceil(self.real_num_trajectories / pad_to_multiple) * pad_to_multiple
        self.max_resp_len = int(offline_config.get("max_resp_len", self.max_response_length))
        self.overlong_buffer_len = int(offline_config.get("overlong_buffer_len", 0))
        self.overlong_penalty_factor = float(offline_config.get("overlong_penalty_factor", 1.0))

    def __len__(self) -> int:
        return self.num_trajectories

    def _shard_path(self, shard_index: int) -> str:
        return f"{self.root}/raw/part-{shard_index:05d}-of-{self.num_shards:05d}.jsonl.gz"

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached = self._shard_cache.pop(shard_index, None)
        if cached is not None:
            self._shard_cache[shard_index] = cached
            return cached

        local_path = copy_to_local(self._shard_path(shard_index), cache_dir=self.cache_dir)
        with gzip.open(local_path, "rt") as handle:
            records = [json.loads(line) for line in handle]
        local_shard_path = os.path.realpath(local_path)
        if os.path.commonpath((local_shard_path, os.path.realpath(self.cache_dir))) == os.path.realpath(self.cache_dir):
            self._disk_shard_cache.record(local_shard_path)
        if not records:
            raise ValueError(f"Teacher trajectory shard is empty: {self._shard_path(shard_index)}")
        self._shard_cache[shard_index] = records
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return records

    def _get_real_record(self, index: int) -> dict[str, Any]:
        shard_index, line_index = divmod(index, self.records_per_full_shard)
        records = self._load_shard(shard_index)
        if line_index >= len(records):
            raise IndexError(
                f"Trajectory index {index} maps past shard {shard_index}: "
                f"line={line_index}, records={len(records)}"
            )
        return records[line_index]

    def _soft_overlong_penalty(self, response_length: int) -> float:
        if self.overlong_buffer_len <= 0:
            return 0.0
        expected_len = self.max_resp_len - self.overlong_buffer_len
        exceed_len = response_length - expected_len
        return min(-exceed_len / self.overlong_buffer_len * self.overlong_penalty_factor, 0.0)

    def _build_item(self, record: dict[str, Any], *, padding: bool) -> dict[str, Any]:
        if int(record.get("schema_version", -1)) != 2:
            raise ValueError(f"Expected record schema_version=2, got {record.get('schema_version')}")
        prompt_ids = list(record["prompt_token_ids"])
        response_ids = list(record["response_token_ids"])
        behavior_logprobs = list(record["behavior_logprobs"])
        if len(behavior_logprobs) != len(response_ids):
            raise ValueError(
                "behavior_logprobs and response_token_ids must have identical lengths: "
                f"{len(behavior_logprobs)} != {len(response_ids)} for "
                f"{record.get('prompt_id')}:{record.get('sample_index')}"
            )
        if not response_ids:
            raise ValueError("Teacher trajectory response_token_ids must not be empty")
        if len(prompt_ids) > self.max_prompt_length:
            raise ValueError(f"Teacher prompt length {len(prompt_ids)} exceeds {self.max_prompt_length}")
        if len(response_ids) > self.max_response_length:
            raise ValueError(f"Teacher response length {len(response_ids)} exceeds {self.max_response_length}")

        pad_token_id = int(self.tokenizer.pad_token_id)
        prompt_padding = self.max_prompt_length - len(prompt_ids)
        response_padding = self.max_response_length - len(response_ids)
        prompts = torch.tensor([pad_token_id] * prompt_padding + prompt_ids, dtype=torch.long)
        responses = torch.tensor(response_ids + [pad_token_id] * response_padding, dtype=torch.long)
        prompt_attention_mask = torch.tensor([0] * prompt_padding + [1] * len(prompt_ids), dtype=torch.long)
        response_attention_mask = torch.tensor(
            [1] * len(response_ids) + [0] * response_padding,
            dtype=torch.long,
        )
        response_mask = response_attention_mask.clone()
        if padding:
            response_mask.zero_()
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask))
        input_ids = torch.cat((prompts, responses))
        position_ids = compute_position_id_with_mask(attention_mask)
        rollout_log_probs = torch.tensor(behavior_logprobs + [0.0] * response_padding, dtype=torch.float32)

        base_reward = float(record["base_reward"])
        overlong_reward = self._soft_overlong_penalty(len(response_ids))
        final_reward = base_reward + overlong_reward
        rm_scores = torch.zeros(self.max_response_length, dtype=torch.float32)
        if not padding:
            rm_scores[len(response_ids) - 1] = final_reward

        source_rows = record.get("source_rows") or []
        source_row = source_rows[0] if source_rows else {}
        reward_model = dict(source_row.get("reward_model") or {})
        reward_model["ground_truth"] = record.get("ground_truth")
        trajectory_id = f"{record['prompt_id']}:{record['sample_index']}"
        return {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "rollout_log_probs": rollout_log_probs,
            "rm_scores": rm_scores,
            "raw_prompt": record.get("prompt", source_row.get("prompt")),
            "data_source": source_row.get("data_source", "lighteval/MATH"),
            "ability": source_row.get("ability", "MATH"),
            "reward_model": reward_model,
            "extra_info": source_row.get("extra_info", {}),
            "multi_modal_data": {},
            "multi_modal_inputs": {},
            "mm_processor_kwargs": {},
            "trajectory_id": trajectory_id,
            "prompt_id": record["prompt_id"],
            "sample_index": int(record["sample_index"]),
            "offline_trajectory_index": int(record.get("unique_prompt_index", -1)),
            "base_reward": base_reward,
            "acc": base_reward,
            "is_correct": bool(record.get("is_correct", base_reward > 0)),
            "finish_reason": record.get("finish_reason"),
            "overlong_reward": overlong_reward,
            "overlong": overlong_reward < 0,
            "__offline_trajectory__": True,
            "__offline_padding__": padding,
            "__num_turns__": 1,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.real_num_trajectories:
            return self._build_item(self._get_real_record(index), padding=False)
        # Padding samples only make the final global batch divisible by the PPO
        # mini-batch size. They retain a valid input shape but have a zero
        # response mask and therefore contribute no actor or critic gradient.
        return self._build_item(self._get_real_record(self.real_num_trajectories - 1), padding=True)


def consume_offline_trajectory_batch(batch: DataProto) -> Optional[DataProto]:
    """Return a pre-generated batch, or ``None`` for an ordinary online batch."""

    marker = batch.non_tensor_batch.get("__offline_trajectory__")
    if marker is None:
        return None
    if not bool(np.asarray(marker, dtype=bool).all()):
        raise ValueError("A batch must not mix offline teacher trajectories with online prompts")
    required = {"responses", "response_mask", "rollout_log_probs", "rm_scores"}
    missing = required - set(batch.batch.keys())
    if missing:
        raise ValueError(f"Offline trajectory batch is missing tensors: {sorted(missing)}")
    batch.meta_info["timing"] = {}
    batch.meta_info["reward_extra_keys"] = [
        "acc",
        "base_reward",
        "finish_reason",
        "is_correct",
        "overlong",
        "overlong_reward",
    ]
    return batch
