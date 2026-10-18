import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_waist_separated_skin_candidate import _remap_nhcnt_source_geometry
from numilab_human.resting_scene import NHCNT_HEADER, NHCNT_MAGIC, NHCNT_RECORD


class NHCNTVertexRemapTests(unittest.TestCase):
    def test_remaps_only_one_based_source_geometry_indices(self):
        original = NHCNT_HEADER.pack(
            NHCNT_MAGIC, 1, 157, 3, 0, bytes.fromhex("ab" * 32),
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.55,
        )
        records = b"".join(
            NHCNT_RECORD.pack(body, geometry, *([0.125] * 10))
            for body, geometry in ((23, 3), (20, 6), (42, 9))
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.nhcnt"
            target = root / "candidate.nhcnt"
            receipt = root / "receipt.json"
            source.write_bytes(original + records)
            result = _remap_nhcnt_source_geometry(
                source, target, receipt,
                rigid={"body_count": 157, "source_sha256": "ab" * 32},
                source_vertex_count=10,
                source_to_output={2: 7, 5: 5, 8: 1},
                candidate_skin_sha="cd" * 32,
            )
            remapped = target.read_bytes()
            self.assertEqual(len(remapped), len(original + records))
            self.assertEqual(remapped[:NHCNT_HEADER.size], original)
            self.assertEqual(
                [NHCNT_RECORD.unpack_from(remapped, NHCNT_HEADER.size + i * NHCNT_RECORD.size)[1]
                 for i in range(3)],
                [8, 6, 2],
            )
            allowed_changed = {
                NHCNT_HEADER.size + i * NHCNT_RECORD.size + 4 + lane
                for i, (old, new) in enumerate(((3, 8), (6, 6), (9, 2)))
                if old != new for lane in range(4)
            }
            changed = {
                i for i, (a, b) in enumerate(zip(original + records, remapped)) if a != b
            }
            self.assertTrue(changed)
            self.assertTrue(changed <= allowed_changed)
            self.assertTrue(all((original + records)[i] == remapped[i]
                                for i in range(len(remapped)) if i not in allowed_changed))
            self.assertEqual(result["source_geometry_ids_remapped"], 2)
            self.assertTrue(result["all_non_geometry_id_bytes_identical"])
            self.assertEqual(json.loads(receipt.read_text())["record_count"], 3)

    def test_missing_source_ancestry_fails_closed(self):
        payload = NHCNT_HEADER.pack(
            NHCNT_MAGIC, 1, 157, 1, 0, bytes.fromhex("ab" * 32),
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.55,
        ) + NHCNT_RECORD.pack(23, 10, *([0.0] * 10))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target, receipt = root / "src", root / "dst", root / "rpt"
            source.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "absent from the cut output"):
                _remap_nhcnt_source_geometry(
                    source, target, receipt,
                    rigid={"body_count": 157, "source_sha256": "ab" * 32},
                    source_vertex_count=10, source_to_output={},
                    candidate_skin_sha="cd" * 32,
                )


if __name__ == "__main__":
    unittest.main()
