# Model and data

Use [Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
with the default configurations. Model weights and datasets are not bundled.

The launcher expects these processed Parquet files below `--data`:

| File | Public source |
|---|---|
| `math-17k.parquet` | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k |
| `evals/amc23.parquet` | https://huggingface.co/datasets/math-ai/amc23 |
| `evals/aime24.parquet` | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 |
| `evals/aime25.parquet` | https://huggingface.co/datasets/math-ai/aime25 |
| `evals/hmmt25.parquet` | https://huggingface.co/datasets/MathArena/hmmt_feb_2025 |

Each row requires:

- `data_source`: `lighteval/MATH` for training, or `amc23`, `aime24`, `aime25`, `hmmt25` for validation.
- `prompt`: a list of `{role, content}` messages.
- `ability`: `MATH`.
- `reward_model`: a mapping containing `ground_truth` and `style` (the default uses `rule-lighteval/MATH_v2`).
- `extra_info`: optional metadata.

Prompts should request step-by-step reasoning and a final answer in `\boxed{}`
to match the supplied answer verifier. Remove duplicate AMC23 prompts.

```bash
uv run --with pyarrow python scripts/check_data.py /shared/data
```

This checks existing input files; it is not a downloader or raw-data preparation
script. Add `--check-counts` to also check the default dataset sizes: 17,917
training rows and 40/30/30/30 validation rows. The checker does not compare
against private or historical dataset fingerprints.

For different datasets, update the configuration paths, source dispatch and
reward function together. Respect each dataset's license.
