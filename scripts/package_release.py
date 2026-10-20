# Copyright 2026 EPIC contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build a source ZIP without caches, credentials, or generated training outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist',
            'outputs', 'checkpoints', 'wandb', '.mypy_cache', '.ruff_cache'}


def files():
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(p in EXCLUDED or p.endswith('.egg-info') for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not included: {rel}')
        if not path.is_file() or path.name in {'.DS_Store', 'CHECKSUMS.sha256'}:
            continue
        if path.suffix in {'.pyc', '.pyo', '.zip', '.key', '.pem', '.csv'} or path.name.startswith('.env'):
            continue
        yield path


def audit(paths):
    # These are targeted leak checks, not a guarantee of legal/publication clearance.
    forbidden = ['/' + 'Users/', 'hou-zg-' + 'meituan', 'nwuser' + 'houzg',
                 '/opt/' + 'tiger/', '/mnt/' + 'bn/']
    secret = re.compile(r'(?:AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for p in paths:
        try:
            text = p.read_text()
        except UnicodeError:
            continue
        if any(s in text for s in forbidden) or secret.search(text):
            raise ValueError(f'Potential private content: {p.relative_to(ROOT)}')
        if p.suffix == '.py':
            ast.parse(text, filename=str(p.relative_to(ROOT)))
    assert {p.stem for p in (ROOT/'configs').glob('*.json')} == {'epic','grpo','dapo','gspo','ctpo','ripo'}
    assert not (ROOT/'experiments.json').exists()
    assert not (ROOT/'results').exists()
    assert (ROOT / 'LICENSE').read_bytes() == (ROOT / 'vendor/verl/LICENSE').read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = list(files())
    audit(paths)
    checksums = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in paths)
    (ROOT / 'CHECKSUMS.sha256').write_text(checksums)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [*paths, ROOT / 'CHECKSUMS.sha256']:
            z.write(p, Path('epic') / p.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for line in checksums.splitlines():
            digest, rel = line.split('  ', 1)
            assert hashlib.sha256(z.read('epic/' + rel)).hexdigest() == digest
    print(f'Checked {len(paths)} files; archive integrity and checksums pass.')
    print(args.output)


if __name__ == '__main__':
    main()
