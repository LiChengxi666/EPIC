"""Recipe-scoped GSM8K scoring for the pretrained Moonlight checkpoint."""

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.gsm8k import compute_score as compute_gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        return compute_gsm8k_score(solution_str, ground_truth, method="flexible")
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
