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

from .math_dapo import normalize_final_answer
from .math_reward import last_boxed_only_string, remove_boxed


def compute_score(solution_str: str, ground_truth: str) -> dict[str, float | bool | str]:
    """Score boxed-answer math benchmarks with binary accuracy."""
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is None:
        prediction = "[INVALID]"
        correct = False
    else:
        try:
            prediction = normalize_final_answer(remove_boxed(boxed_answer))
            reference = normalize_final_answer(ground_truth)
            correct = prediction == reference
        except Exception:
            prediction = "[INVALID]"
            correct = False

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": prediction,
    }
