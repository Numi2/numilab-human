"""Focused source-lineage tests for local pleura surface edge flips."""
import unittest

import numpy as np

from numilab_human.resting_pleura_proxy import (
    LOBE_IDS, _normalize_source_face_ancestry,
    _source_face_subdivision_patch_report, derive_lung_union_exterior,
)


class SourceFaceSubdivisionPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture(positions=None, faces=None):
        if positions is None:
            positions=np.asarray([[0,0,0],[4,0,0],[0,4,0],[2,0,0]],dtype=np.float32)
        if faces is None:
            faces=np.asarray([[0,3,2],[3,1,2]],dtype=np.int64)
        vertices6=np.column_stack((positions,np.zeros_like(positions))).astype(np.float32)
        nums=np.asarray(positions,dtype=np.int64)
        dens=np.ones_like(nums,dtype=np.int64)
        evidence={
            "support_source_face_ids":(0,),
            "raw_vertex_lattice_numerators":nums,
            "raw_vertex_lattice_denominators":dens,
            "lattice_scale":1,
        }
        source_vertices6=np.asarray([[0,0,0,0,0,0],[4,0,0,0,0,0],[0,4,0,0,0,0]],dtype=np.float32)
        source_faces=np.asarray([[0,1,2]],dtype=np.int64)
        return vertices6,np.asarray(faces,dtype=np.int64),source_vertices6,source_faces,evidence

    def _check(self, fixture):
        vertices,faces,source_vertices,source_faces,evidence=fixture
        return _source_face_subdivision_patch_report(
            vertices,faces,source_vertices,source_faces,[(0,)]*len(faces),evidence,(0,),
        )

    def test_exact_subdivision_requires_parent_coverage_orientation_and_patch_boundary(self):
        report=self._check(self._fixture())
        self.assertTrue(report["passed"])
        self.assertEqual(report["exact_child_orientation_and_containment_checked"],2)
        self.assertEqual(report["source_parent_area_coverage_fraction_by_face"]["0"],1.0)
        self.assertTrue(report["exact_raw_and_float32_patch_boundary_conserved"])

    def test_subdivision_rejects_gap_even_when_parent_ids_repeat(self):
        positions=np.asarray([[0,0,0],[4,0,0],[0,4,0],[2,0,0],[1,1,0]],dtype=np.float32)
        faces=np.asarray([[0,4,2],[4,3,2]],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"patch boundary differs"):
            self._check(self._fixture(positions,faces))

    def test_subdivision_rejects_child_outside_parent(self):
        positions=np.asarray([[0,0,0],[4,0,0],[0,4,0],[5,0,0]],dtype=np.float32)
        faces=np.asarray([[0,3,2],[3,1,2]],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"outside its exact parent"):
            self._check(self._fixture(positions,faces))

    def test_subdivision_rejects_reversed_child_orientation(self):
        faces=np.asarray([[0,2,3],[3,2,1]],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"reversed parent orientation"):
            self._check(self._fixture(faces=faces))

    def test_subdivision_rejects_overlapping_children(self):
        faces=np.asarray([[0,3,2],[0,3,2]],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"overlap in parent"):
            self._check(self._fixture(faces=faces))

    def test_subdivision_requires_raw_geometry_to_match_float32_emission(self):
        fixture=list(self._fixture())
        fixture[0]=fixture[0].copy()
        fixture[0][3,0]=np.nextafter(fixture[0][3,0],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,"does not roundtrip"):
            self._check(tuple(fixture))


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

    def test_one_to_many_source_face_subdivision_requires_exact_opt_in(self):
        mapping = [(0,), (0,), (0,), (0,), (1,)]
        with self.assertRaisesRegex(ValueError, "one unique source face"):
            _normalize_source_face_ancestry(
                mapping, 2, 305, allow_two_parent_flip=False,
            )
        self.assertEqual(
            _normalize_source_face_ancestry(
                mapping, 2, 305, allow_two_parent_flip=False,
                allowed_subdivision_sources=(0,),
            ),
            ((0,), (0,), (0,), (0,), (1,)),
        )

    def test_subdivision_opt_in_must_match_all_and_only_repeated_sources(self):
        mapping = [(0,), (0,), (1,)]
        bad = (
            ((), "one unique source face"),
            ((1,), "declaration differs"),
            ((0, 1), "declaration differs"),
            ((2,), "out-of-range"),
            ((0, 0), "duplicate or out-of-range"),
            ((0.0,), "requires integer"),
            ((True,), "requires integer"),
        )
        for allowed, message in bad:
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, message):
                _normalize_source_face_ancestry(
                    mapping, 2, 305, allow_two_parent_flip=False,
                    allowed_subdivision_sources=allowed,
                )

    def test_subdivision_and_flip_optins_cannot_reuse_a_source_face(self):
        with self.assertRaisesRegex(ValueError, "source-face subdivision declaration"):
            _normalize_source_face_ancestry(
                [(0,), (0,), (1, 2), (1, 2)], 3, 305,
                allow_two_parent_flip=True,
                allowed_subdivision_sources=(1,),
            )

    def test_owner_subdivision_opt_in_requires_explicit_source_lineage(self):
        lobes = {}
        for index, stable_id in enumerate(LOBE_IDS):
            origin = np.asarray([0.01 * index, 0.0, 0.0], dtype=np.float32)
            positions = np.asarray([
                origin + [0.0, 0.0, 0.0], origin + [1e-3, 0.0, 0.0],
                origin + [0.0, 1e-3, 0.0],
            ], dtype=np.float32)
            vertices6 = np.column_stack((positions, np.zeros_like(positions))).astype(np.float32)
            lobes[stable_id] = (vertices6, np.asarray([[0, 1, 2]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "requires explicit source ancestry"):
            derive_lung_union_exterior(
                lobes, allow_source_face_subdivisions={305: (0,)},
            )

    def test_owner_accepts_named_subdivision_before_later_geometry_rejection(self):
        lobes = {}
        source = {}
        face_ancestry = {}
        vertex_ancestry = {}
        for index, stable_id in enumerate(LOBE_IDS):
            origin = np.asarray([0.01 * index, 0.0, 0.0], dtype=np.float32)
            source_positions = np.asarray([
                origin + [0.0, 0.0, 0.0], origin + [1e-3, 0.0, 0.0],
                origin + [0.0, 1e-3, 0.0],
            ], dtype=np.float32)
            source_vertices6 = np.column_stack((source_positions, np.zeros_like(source_positions))).astype(np.float32)
            source_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
            source[stable_id] = (source_vertices6, source_faces)
            if stable_id == 305:
                candidate_positions = np.vstack((source_positions, origin + [0.5e-3, 0.0, 0.0])).astype(np.float32)
                candidate_vertices6 = np.column_stack((candidate_positions, np.zeros_like(candidate_positions))).astype(np.float32)
                candidate_faces = np.asarray([[0, 3, 2], [3, 1, 2]], dtype=np.int64)
                face_ancestry[stable_id] = [(0,), (0,)]
                vertex_ancestry[stable_id] = [(0,), (1,), (2,), (0, 1)]
            else:
                candidate_vertices6 = source_vertices6.copy()
                candidate_faces = source_faces.copy()
                face_ancestry[stable_id] = [(0,)]
                vertex_ancestry[stable_id] = [(0,), (1,), (2,)]
            lobes[stable_id] = (candidate_vertices6, candidate_faces)
        subdivision_positions=lobes[305][0][:,:3]
        subdivision_numerators=np.asarray([
            [str(int(round(float(value)*(1<<149))) ) for value in point]
            for point in subdivision_positions
        ],dtype="U256")
        evidence={305:{
            "support_source_face_ids":(0,),
            "raw_vertex_lattice_numerators":subdivision_numerators,
            "raw_vertex_lattice_denominators":np.ones_like(subdivision_numerators),
            "lattice_scale":1<<149,
        }}
        with self.assertRaisesRegex(ValueError, "matching exact patch geometry evidence"):
            derive_lung_union_exterior(
                lobes, source_lineage_lobes=source,
                source_face_ancestry=face_ancestry,
                source_vertex_ancestry=vertex_ancestry,
                allow_source_face_subdivisions={305: (0,)},
            )
        with self.assertRaisesRegex(ValueError, "detached lung-union component"):
            derive_lung_union_exterior(
                lobes, source_lineage_lobes=source,
                source_face_ancestry=face_ancestry,
                source_vertex_ancestry=vertex_ancestry,
                allow_source_face_subdivisions={305: (0,)},
                source_face_subdivision_patch_geometry=evidence,
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
