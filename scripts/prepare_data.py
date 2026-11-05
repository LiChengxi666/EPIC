# Copyright 2026 EPIC contributors.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned public datasets and prepare the five EPIC Parquet inputs.

Run with: uv run --no-project --with pyarrow python scripts/prepare_data.py --output data
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'configs/data/sources.json'
INSTRUCTION = r'Please reason step by step, and put your final answer within \boxed{}.'
DAPO_PREFIX = ('Solve the following math problem step by step. The last line of your response '
               'should be of the form Answer: $Answer (without quotes) where $Answer is the '
               'answer to the problem.\n\n')
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def download_source(spec, cache_dir, offline=False):
    """Cache immutable public source URLs; publish only fully downloaded files."""
    target = Path(cache_dir) / spec['repo'] / spec['revision'] / spec['filename']
    if target.is_file():
        return target
    if offline:
        raise FileNotFoundError(f'Pinned source is not cached: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"
    for attempt in range(3):
        part = None
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    part = Path(out.name)
                    shutil.copyfileobj(response, out)
                expected = response.headers.get('Content-Length')
            if expected is not None and part.stat().st_size != int(expected):
                raise OSError('Incomplete source download')
            part.replace(target)
            return target
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if part is not None:
                part.unlink(missing_ok=True)


def answer_text(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('Missing or invalid ground-truth answer')
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError('Non-integral numeric answer; preserve it explicitly as text')
        value = int(value)
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError('Expected a non-empty string or integer answer')
    return str(value).strip()


def normalize_rows(rows, spec):
    """Preserve problem text, replacing DAPO's Answer: wrapper for the boxed verifier."""
    output, seen = [], {}
    duplicates = 0
    for index, row in enumerate(rows):
        if spec['name'] == 'train':
            messages = row['prompt']
            answer = answer_text(row['reward_model']['ground_truth'])
            if (not isinstance(messages, list) or len(messages) != 1
                    or messages[0].get('role') != 'user'):
                raise ValueError('Expected one DAPO user message')
            content = messages[0]['content']
            if not content.startswith(DAPO_PREFIX) or not content.endswith(DAPO_SUFFIX):
                raise ValueError('Unrecognized DAPO prompt wrapper; review the pinned source')
            problem = content[len(DAPO_PREFIX):-len(DAPO_SUFFIX)].strip()
            if not problem:
                raise ValueError('Empty DAPO problem')
            prompt = [{'role': 'user', 'content': problem + '\n\n' + INSTRUCTION}]
        else:
            problem = row[spec['problem_field']]
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError('Expected a non-empty problem')
            prompt = [{'role': 'user', 'content': problem.strip() + '\n\n' + INSTRUCTION}]
            answer = answer_text(row['answer'])
        prompt_key = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        source_id = str(row.get('id', row.get('problem_idx', row.get('extra_info', {}).get('index', index))))
        key = source_id if spec['name'] == 'train' else prompt_key
        value = (prompt_key, answer)
        if spec['name'] in ('train', 'amc23') and key in seen:
            if seen[key] != value:
                raise ValueError(f"{spec['name']}: duplicate has conflicting problem text or answers")
            duplicates += 1
            continue
        seen[key] = value
        output.append({
            'data_source': spec['data_source'],
            'prompt': prompt,
            'ability': 'MATH',
            'reward_model': {'ground_truth': answer, 'style': 'rule-lighteval/MATH_v2'},
            'extra_info': {'index': index, 'split': spec['split'],
                           'source_id': source_id},
        })
    return output, duplicates


def prepare(output, cache_dir=None, offline=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from check_data import inspect_data

    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SOURCES.read_text())
    schema = pa.schema([
        ('data_source', pa.string()),
        ('prompt', pa.list_(pa.struct([('role', pa.string()), ('content', pa.string())]))),
        ('ability', pa.string()),
        ('reward_model', pa.struct([('ground_truth', pa.string()), ('style', pa.string())])),
        ('extra_info', pa.struct([('index', pa.int64()), ('split', pa.string()), ('source_id', pa.string())])),
    ])
    staging = Path(tempfile.mkdtemp(prefix='.epic-data-', dir=output.parent))
    manifest = {
        'format_version': 1,
        'preparation_script_sha256': digest(__file__),
        'source_config_sha256': digest(SOURCES),
        'evaluation_instruction': INSTRUCTION,
        'packages': {'pyarrow': importlib.metadata.version('pyarrow')},
        'datasets': [],
    }
    try:
        for spec in specs:
            print(f"Preparing {spec['name']} from {spec['repo']} @ {spec['revision']}", flush=True)
            source = download_source(spec, cache_dir or Path.home() / '.cache/epic-data', offline)
            if spec['filename'].endswith('.parquet'):
                parquet = pq.ParquetFile(source)
                raw_count = parquet.metadata.num_rows
                raw = (row for batch in parquet.iter_batches(batch_size=4096) for row in batch.to_pylist())
            else:
                raw = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
                raw_count = len(raw)
            rows, duplicates = normalize_rows(raw, spec)
            if len(rows) != spec['expected_rows']:
                raise ValueError(f"{spec['name']}: expected {spec['expected_rows']} rows; found {len(rows)}")
            target = staging / spec['output']
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
            canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            manifest['datasets'].append({**spec, 'raw_rows': raw_count, 'rows': len(rows),
                'duplicates_removed': duplicates, 'source_sha256': digest(source),
                'parquet_sha256': digest(target), 'content_sha256': hashlib.sha256(canonical).hexdigest()})
        inspect_data(staging, check_counts=True)
        (staging / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during preparation: {output}')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f'Validated all five datasets: {output}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for prepared data')
    parser.add_argument('--cache-dir', type=Path, help='Download cache (default: ~/.cache/epic-data)')
    parser.add_argument('--offline', action='store_true', help='Use already cached pinned source files')
    args = parser.parse_args()
    prepare(args.output, args.cache_dir, args.offline)


if __name__ == '__main__':
    main()
