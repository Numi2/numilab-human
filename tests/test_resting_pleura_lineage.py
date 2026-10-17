"""Focused source-lineage tests for local pleura surface edge flips."""
import unittest

import numpy as np

from numilab_human.resting_pleura_proxy import (
    LOBE_IDS, _normalize_source_face_ancestry, derive_lung_union_exterior,
)


class SourceFaceAncestryTests(unittest.TestCase):
    def test_default_contract_rejects_a_two_parent_output_face(self):
        with self.assertRaisesRegex(ValueError, "one unique source face"):
            _normalize_source_face_ancestry(
                [(0,), (1, 2)], 3, 308, allow_two_parent_flip=False,
            )

    def test_explicit_local_flip_keeps_both_parents_on_both_output_faces(self):
        result = _normalize_source_face_ancestry(
            [(0,), (1,), (3, 2), (2, 3)], 4, 308,
            allow_two_parent_flip=True,
        )
        self.assertEqual(result, ((0,), (1,), (2, 3), (2, 3)))

    def test_local_flip_rejects_unpaired_or_reused_parent_lineage(self):
        bad_maps = (
            [(0,), (1,), (2, 3)],
            [(0,), (1,), (2, 3), (2, 4)],
            [(0,), (1,), (2, 3), (2, 3), (2,)],
        )
        for mapping in bad_maps:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                _normalize_source_face_ancestry(
                    mapping, 5, 308, allow_two_parent_flip=True,
                )


    def test_owner_requires_explicit_source_and_accepts_authenticated_flip_pair(self):
        lobes = {}
        source = {}
        face_ancestry = {}
        vertex_ancestry = {}
        for index, stable_id in enumerate(LOBE_IDS):
            origin = np.asarray([0.01 * index, 0.0, 0.0], dtype=np.float32)
            if stable_id == 308:
                positions = np.asarray([
                    origin + [0.0, 0.0, 0.0], origin + [1e-3, 0.0, 0.0],
                    origin + [0.0, 1e-3, 0.0], origin + [1e-3, 1e-3, 0.0],
                ], dtype=np.float32)
                faces = np.asarray([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
                face_ancestry[stable_id] = [(0, 1), (0, 1)]
            else:
                positions = np.asarray([
                    origin + [0.0, 0.0, 0.0], origin + [1e-3, 0.0, 0.0],
                    origin + [0.0, 1e-3, 0.0],
                ], dtype=np.float32)
                faces = np.asarray([[0, 1, 2]], dtype=np.int64)
                face_ancestry[stable_id] = [(0,)]
            vertices6 = np.column_stack((positions, np.zeros_like(positions))).astype(np.float32)
            lobes[stable_id] = (vertices6, faces)
            source[stable_id] = (vertices6.copy(), faces.copy())
            vertex_ancestry[stable_id] = [(i,) for i in range(len(vertices6))]

        with self.assertRaisesRegex(ValueError, "requires explicit source ancestry"):
            derive_lung_union_exterior(lobes, allow_multi_parent_source_face_ancestry=(308,))
        with self.assertRaisesRegex(ValueError, "one unique source face"):
            derive_lung_union_exterior(
                lobes, source_lineage_lobes=source,
                source_face_ancestry=face_ancestry,
                source_vertex_ancestry=vertex_ancestry,
            )
        with self.assertRaisesRegex(ValueError, "detached lung-union component"):
            derive_lung_union_exterior(
                lobes, source_lineage_lobes=source,
                source_face_ancestry=face_ancestry,
                source_vertex_ancestry=vertex_ancestry,
                allow_multi_parent_source_face_ancestry=(308,),
            )

    def test_local_flip_rejects_fractional_duplicate_and_out_of_range_parents(self):
        bad_maps = (
            [(0,), (1,), (2, 3), (2.0, 3.0)],
            [(0,), (1,), (2, 2), (2, 2)],
            [(0,), (1,), (2, 5), (2, 5)],
        )
        for mapping in bad_maps:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                _normalize_source_face_ancestry(
                    mapping, 5, 308, allow_two_parent_flip=True,
                )


if __name__ == "__main__":
    unittest.main()
