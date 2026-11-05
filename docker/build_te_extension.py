"""Build the public TE 2.9 PyTorch source with its force-build packaging fix."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

SDIST_SHA256 = 'abbc59f6acf635abf865085ecdf90e7d4ca9a3782bc91a9845e38adb2655a547'


def patch_force_build(source):
    # TE 2.9 otherwise calls bdist_wheel twice after a successful source build.
    before = '        if FORCE_BUILD:\n            super().run()\n'
    after = '        if FORCE_BUILD:\n            return super().run()\n'
    if after in source:
        return source
    if source.count(before) != 1:
        raise RuntimeError('Unexpected TE 2.9 setup.py; force-build patch cannot be applied')
    return source.replace(before, after, 1)


def build(constraints):
    with tempfile.TemporaryDirectory(prefix='te-source-') as directory:
        staging = Path(directory)
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
            '--no-build-isolation', '--no-binary=:all:', '--dest', directory,
            'transformer-engine-torch==2.9.0'], check=True)
        archive = next(staging.glob('*.tar.gz'))
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SDIST_SHA256:
            raise RuntimeError('TE source archive differs from the pinned public PyPI release')
        with tarfile.open(archive) as source:
            source.extractall(staging / 'source', filter='data')
        setup = next((staging / 'source').glob('*/setup.py'))
        setup.write_text(patch_force_build(setup.read_text()))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
            '--no-build-isolation', '--constraint', constraints, str(setup.parent)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constraints', required=True)
    build(parser.parse_args().constraints)
