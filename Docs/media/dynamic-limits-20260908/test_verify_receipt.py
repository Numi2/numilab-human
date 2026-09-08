import gzip
import json
import unittest

from verify_receipt import HERE, audit_payload


class AuditMutationTests(unittest.TestCase):
    def test_authored_rows_and_mutations(self):
        source = json.loads(gzip.decompress((HERE / "source-export.json.gz").read_bytes()))
        manifest = json.loads(gzip.decompress((HERE / "source-manifest.json.gz").read_bytes()))
        original = (HERE / "myosim-fullbody-joint-limits.nhlim").read_bytes()
        self.assertEqual(audit_payload(original, source, manifest)["joint_count"], 122)
        for offset in (0, 8, 20, 36, 48, 80, 88, 92, 96, 108, 112, 128):
            changed = bytearray(original); changed[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                audit_payload(changed, source, manifest)
        with self.assertRaises(ValueError): audit_payload(original[:-80], source, manifest)


if __name__ == "__main__":
    unittest.main()
