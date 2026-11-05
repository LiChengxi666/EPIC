# Model and data

Use [Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
with the default configurations. Model weights and datasets are not bundled. Download the public model into the
shared directory using the training image:

```bash
docker run --rm -v /shared:/shared epic:local \
  hf download Qwen/Qwen3-30B-A3B-Base \
  --local-dir /shared/model/Qwen3-30B-A3B-Base
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for building the image. The model directory
must contain all weight shards, configuration and tokenizer files.

## Download and prepare

Prepare data using the training image on a CPU machine; these commands do not
request GPUs:

```bash
docker run --rm -v /shared:/shared epic:local \
  python scripts/prepare_data.py --output /shared/data
docker run --rm -v /shared:/shared epic:local \
  python scripts/check_data.py /shared/data --check-counts
```

The output directory must not already exist. Public files are downloaded from
Hugging Face at pinned revisions and cached under `~/.cache/epic-data`. Use
`--cache-dir /path/to/cache` to choose its location or `--offline` to reuse
cached files without network access. For standalone CPU use, install PyArrow in your Python environment and run
the same scripts directly; the preparation code does not require CUDA or model
weights.

The five upstream revisions are pinned in `configs/data/sources.json`.
Preparation checks row counts and schema before making the output directory
available. `preparation_manifest.json` records upstream commits, source-file
SHA-256 hashes, output hashes, normalized-content hashes, and tool versions.
Keep this manifest with experiment outputs. Generated data are excluded from
Git and source release packages.

The script preserves problem text, replaces DAPO's `Answer:` instruction with
the boxed-answer instruction required by the configured verifier, and applies
that same instruction to evaluation questions. Evaluation solutions are never
included in prompts. AMC23 duplicates are removed only when their answers
agree; conflicting answers cause an error. Answers remain strings, preserving
fractions and other mathematical expressions.

The pinned DAPO file contains 1,791,700 rows (about 299 MB), including repeated
source IDs. The script reads it in batches and keeps the first occurrence of
each ID after checking that repeated entries agree, producing 17,917 training
examples. It does not load the expanded file into Python objects all at once.

This is a versioned public preparation recipe. Its outputs have not been
established as byte-for-byte or prompt-identical copies of historical training
inputs. Matching counts and schema alone does not establish that equivalence.

## Sources and schema

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
docker run --rm -v /shared:/shared epic:local \
  python scripts/check_data.py /shared/data
```

The checker validates existing input files. Add `--check-counts` to also check the default dataset sizes: 17,917
training rows and 40/30/30/30 validation rows. The checker does not compare
against private or historical dataset fingerprints.

For different datasets, update the configuration paths, source dispatch and
reward function together. Respect each dataset's license.
In particular, the pinned HMMT source declares CC-BY-NC-SA-4.0; repository
Apache-2.0 licensing does not replace dataset terms. Consult the upstream cards
for each dataset's current terms.
