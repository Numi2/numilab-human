"""Source-byte composition guards; these fixtures do not qualify anatomy."""
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from numilab_human.passive_attachment_composition import _read_nhtiss4, _row_slices
from numilab_human.resting_regional_tissue_composition import (
    merge_fixed_allocation_tissue_rows,
)


def tissue(offsets=None, material_change=False, binding_change=False):
    offsets = offsets or {}
    records, vertices, indices = [], [], []
    for index, sid in enumerate((72, 134, 157)):
        records.append(struct.pack("<8I", index, 1, index * 3, 3, index * 3, 3, sid, 1))
        for vertex in range(3):
            vertices.append(struct.pack("<6f4I4f", vertex + offsets.get(sid, 0), vertex % 2, 0,
                            0, 0, 1, index, 0xffffffff, 0xffffffff, 0xffffffff,
                            .5 if material_change and sid == 72 else 1, 0, 0, 0))
        indices.append(struct.pack("<3I", index * 3, index * 3 + 1, index * 3 + 2))
    bindings = bytearray(3 * 36)
    if binding_change:
        bindings[0] = 1
    return (struct.pack("<8s6I32s", (b"NHTISS4" + bytes([0])), 5, 3, 3, 9, 9, 123, b"a" * 32)
            + b"".join(records) + bindings + b"".join(vertices) + b"".join(indices))


class RegionalTissueCompositionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.write("base.nhtissue", tissue())
        self.a = self.write("a.nhtissue", tissue({72: .01}))
        self.b = self.write("b.nhtissue", tissue({134: .02}))

    def write(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return path

    def merge(self, replacements, pins=None):
        pins = pins or {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (self.base, *(p for p, _ in replacements))}
        return merge_fixed_allocation_tissue_rows(self.base, replacements, expected_sha256=pins)

    def test_disjoint_repairs_keep_back_and_exact_parent_records(self):
        raw, proof = self.merge([(self.a, [72]), (self.b, [134])])
        final = _read_nhtiss4(self.write("merged.nhtissue", raw))
        expected = {72: _read_nhtiss4(self.a), 134: _read_nhtiss4(self.b),
                    157: _read_nhtiss4(self.base)}
        for sid, owner in expected.items():
            actual_row = next(r for r in final["records"] if int(r[6]) == sid)
            owner_row = next(r for r in owner["records"] if int(r[6]) == sid)
            self.assertEqual(_row_slices(final, actual_row), _row_slices(owner, owner_row))
        self.assertEqual(proof["changed_stable_ids"], [72, 134])
        self.assertEqual(final["vertex_count"], 9)

    def test_face_count_change_rebases_indices_without_reassigning_vertices(self):
        raw = bytearray(self.a.read_bytes())
        struct.pack_into("<I", raw, 24, 12)
        struct.pack_into("<I", raw, 64 + 5 * 4, 6)
        struct.pack_into("<I", raw, 64 + 32 + 4 * 4, 6)
        struct.pack_into("<I", raw, 64 + 64 + 4 * 4, 9)
        index_start = 64 + 3 * 32 + 3 * 36 + 9 * 56
        raw[index_start + 12:index_start + 12] = struct.pack("<3I", 1, 2, 0)
        expanded = self.write("expanded.nhtissue", raw)
        merged, _proof = self.merge([(expanded, [72]), (self.b, [134])])
        final = _read_nhtiss4(self.write("expanded-merge.nhtissue", merged))
        self.assertEqual(final["index_count"], 12)
        self.assertEqual(final["records"][:, 2].tolist(), [0, 3, 6])
        self.assertEqual(final["records"][:, 4].tolist(), [0, 6, 9])
        back = _row_slices(final, final["records"][2])
        original = _read_nhtiss4(self.base)
        old_back = _row_slices(original, original["records"][2])
        for key in ("vertex_bytes", "binding_bytes", "local_faces"):
            self.assertEqual(back[key], old_back[key])

    def test_undeclared_geometry_change_is_rejected(self):
        path = self.write("extra.nhtissue", tissue({72: .01, 157: .02}))
        with self.assertRaisesRegex(ValueError, "undeclared row"):
            self.merge([(path, [72])])

    def test_material_or_attachment_edits_are_rejected(self):
        for kwargs, message in (({"material_change": True}, "material parent"),
                                ({"binding_change": True}, "attachment body")):
            with self.subTest(kwargs=kwargs):
                path = self.write("invalid.nhtissue", tissue({72: .01}, **kwargs))
                with self.assertRaisesRegex(ValueError, message):
                    self.merge([(path, [72])])

    def test_overlapping_regions_and_boolean_ids_are_rejected(self):
        for replacements, message in (([(self.a, [72]), (self.a, [72])], "overlap"),
                                      ([(self.a, [True])], "anatomical integers")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.merge(replacements)

    def test_missing_or_changed_reviewed_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.merge([(self.a, [72])], pins={"unrelated": "0" * 64})


if __name__ == "__main__":
    unittest.main()
