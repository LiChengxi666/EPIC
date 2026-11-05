"""Correct TE 2.9's CUDA-only wheel tag to match its published py3-none filename.

The wheel contains libtransformer_engine.so, not a CPython extension. Its PyPI
filename is py3-none-manylinux_2_28_x86_64, but WHEEL incorrectly says cp310.
The separately compiled transformer_engine_torch extension keeps its ABI tag.
"""
import base64
import csv
import hashlib
from importlib.metadata import distribution
import io


def fix_metadata():
    dist = distribution('transformer-engine-cu12')
    if dist.version != '2.9.0':
        return
    entry = next(p for p in dist.files if str(p).endswith('.dist-info/WHEEL'))
    path = dist.locate_file(entry)
    before = path.read_text()
    old = 'Tag: cp310-cp310-manylinux_2_28_x86_64'
    new = 'Tag: py3-none-manylinux_2_28_x86_64'
    if old not in before:
        return
    after = before.replace(old, new).encode()
    path.write_bytes(after)
    record = path.with_name('RECORD')
    rows = list(csv.reader(io.StringIO(record.read_text())))
    digest = base64.urlsafe_b64encode(hashlib.sha256(after).digest()).decode().rstrip('=')
    for row in rows:
        if row[0] == str(entry):
            row[1:] = ['sha256=' + digest, str(len(after))]
    with record.open('w', newline='') as target:
        csv.writer(target).writerows(rows)
    print('Corrected TE CUDA-only WHEEL metadata to its published py3-none tag')


if __name__ == '__main__':
    fix_metadata()
