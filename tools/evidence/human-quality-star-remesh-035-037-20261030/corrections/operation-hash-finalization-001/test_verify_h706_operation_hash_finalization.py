import importlib.util
import json
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).with_name('verify_h706_operation_hash_finalization.py')
spec = importlib.util.spec_from_file_location('verify_h706', SCRIPT)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class FinalizationCorrectionTests(unittest.TestCase):
    def test_reconstruction_removes_only_finalizer_fields_and_is_canonical(self):
        doc = {
            'z': 9,
            'patches': [{'patch_id': 'p'}],
            'new_output_face_rows': [7, 8],
            'a': 1,
        }
        actual, removed = verify.canonical_without_finalization(doc)
        self.assertEqual(removed, ['new_output_face_rows', 'patches'])
        self.assertEqual(actual, b'{\n  "a": 1,\n  "z": 9\n}\n')
        self.assertEqual(doc['patches'], [{'patch_id': 'p'}])

    def test_reconstruction_fails_if_either_finalizer_field_is_missing(self):
        with self.assertRaisesRegex(ValueError, 'expected exactly'):
            verify.canonical_without_finalization({'new_output_face_rows': []})

    def test_report_patch_map_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, 'duplicate patch ID'):
            verify.json_by_patch([
                {'patch_id': 'same'},
                {'patch_id': 'same'},
            ])

    def test_report_patch_map_requires_a_list(self):
        with self.assertRaisesRegex(ValueError, 'must be a list'):
            verify.json_by_patch({})


if __name__ == '__main__':
    unittest.main(verbosity=2)
