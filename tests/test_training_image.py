"""Regression coverage for the CUDA-only TE wheel's incorrect CPython tag."""
import base64
import csv
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('te_metadata', ROOT / 'docker/fix_te_wheel_metadata.py')
te_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te_metadata)


class TrainingImageTests(unittest.TestCase):
    def test_correct_tag_and_record_and_allow_repeated_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = Path('transformer_engine_cu12-2.9.0.dist-info/WHEEL')
            wheel = root / entry
            wheel.parent.mkdir()
            wheel.write_text('Wheel-Version: 1.0\nTag: cp310-cp310-manylinux_2_28_x86_64\n')
            record = wheel.with_name('RECORD')
            record.write_text(f'{entry},sha256=old,1\nlibtransformer_engine.so,sha256=untouched,123\n')
            dist = SimpleNamespace(version='2.9.0', files=[entry], locate_file=lambda p: root / p)
            with patch.object(te_metadata, 'distribution', return_value=dist):
                te_metadata.fix_metadata()
                first = wheel.read_bytes(), record.read_bytes()
                te_metadata.fix_metadata()
            self.assertEqual((wheel.read_bytes(), record.read_bytes()), first)
            self.assertIn('Tag: py3-none-manylinux_2_28_x86_64', wheel.read_text())
            rows = list(csv.reader(record.read_text().splitlines()))
            digest = base64.urlsafe_b64encode(hashlib.sha256(wheel.read_bytes()).digest()).decode().rstrip('=')
            self.assertEqual(rows[0], [str(entry), 'sha256=' + digest, str(wheel.stat().st_size)])
            self.assertEqual(rows[1], ['libtransformer_engine.so', 'sha256=untouched', '123'])

    def test_te_force_build_returns_after_one_wheel(self):
        spec = importlib.util.spec_from_file_location('te_source', ROOT / 'docker/build_te_extension.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        original = ('class Builder:\n    def run(self):\n'
            '        if FORCE_BUILD:\n            super().run()\n'
            '        download_prebuilt()\n')
        patched = builder.patch_force_build(original)
        self.assertIn('return super().run()', patched)
        self.assertEqual(builder.patch_force_build(patched), patched)
        namespace = {'FORCE_BUILD': True, 'download_prebuilt': lambda: self.fail('Unexpected download')}
        class Parent:
            def run(self):
                return 'built'
        namespace['Parent'] = Parent
        exec(patched.replace('class Builder:', 'class Builder(Parent):'), namespace)
        self.assertEqual(namespace['Builder']().run(), 'built')
        with self.assertRaises(RuntimeError):
            builder.patch_force_build('unexpected source')

    def test_other_versions_are_unchanged(self):
        with patch.object(te_metadata, 'distribution', return_value=SimpleNamespace(version='2.10.0')):
            te_metadata.fix_metadata()


if __name__ == '__main__':
    unittest.main()
