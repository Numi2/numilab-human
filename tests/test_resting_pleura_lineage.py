"""Focused source-lineage tests for local pleura surface repairs."""
import hashlib
import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import encode_payload
from numilab_human.resting_pleura_proxy import (
    LOBE_IDS, _lineage_map_sha256, _normalize_source_face_ancestry,
    _parent_surface_sha256, _source_face_subdivision_patch_report,
    _validate_source_face_edge_collapse_parent_chain, derive_lung_union_exterior,
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


class SourceFaceEdgeCollapseTests(unittest.TestCase):
    @staticmethod
    def _surface_and_maps():
        source_xyz = np.asarray([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=np.float32)
        source_vertices6 = np.column_stack((source_xyz, np.zeros_like(source_xyz))).astype(np.float32)
        source_faces = np.asarray([
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ], dtype=np.int64)
        # The immediate parent is an exact 1-to-3 split of source face 0.
        center = (source_xyz[[4, 0, 2]].astype(np.float64).mean(axis=0)).astype(np.float32)
        parent_vertices6 = np.vstack((source_vertices6, np.r_[center, np.zeros(3, dtype=np.float32)]))
        parent_faces = np.vstack((
            np.asarray([[4, 0, 6], [0, 2, 6], [2, 4, 6]], dtype=np.int64), source_faces[1:]
        ))
        parent_face_to_source = [(0,), (0,), (0,)] + [(i,) for i in range(1, len(source_faces))]
        parent_vertex_to_source = [(i,) for i in range(len(source_xyz))] + [(0, 2, 4)]

        # Collapse a disjoint edge (1, 5), preserving the subdivided face patch.
        removed, retained = 1, 5
        retired = tuple(map(int, np.flatnonzero(np.sum(np.isin(parent_faces, (removed, retained)), axis=1) == 2)))
        old_to_new = {}
        candidate_vertices6 = []
        candidate_vertex_to_parent = []
        for parent_id, vertex in enumerate(parent_vertices6):
            if parent_id == removed:
                continue
            output_id = len(candidate_vertices6)
            old_to_new[parent_id] = output_id
            if parent_id == retained:
                xyz = ((parent_vertices6[removed, :3].astype(np.float64)
                        + parent_vertices6[retained, :3].astype(np.float64)) * 0.5).astype(np.float32)
                candidate_vertices6.append(np.r_[xyz, parent_vertices6[parent_id, 3:]])
                candidate_vertex_to_parent.append((removed, retained))
            else:
                candidate_vertices6.append(vertex.copy())
                candidate_vertex_to_parent.append((parent_id,))
        old_to_new[removed] = old_to_new[retained]
        candidate_face_to_parent = []
        candidate_faces = []
        for parent_face_id, face in enumerate(parent_faces):
            if parent_face_id in retired:
                continue
            candidate_faces.append([old_to_new[int(vertex)] for vertex in face])
            candidate_face_to_parent.append((parent_face_id,))
        candidate_vertices6 = np.asarray(candidate_vertices6, dtype=np.float32)
        candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
        candidate_to_source_faces = [
            tuple(sorted({source for parent_id in row for source in parent_face_to_source[parent_id]}))
            for row in candidate_face_to_parent
        ]
        candidate_to_source_vertices = [
            tuple(sorted({source for parent_id in row for source in parent_vertex_to_source[parent_id]}))
            for row in candidate_vertex_to_parent
        ]
        return {
            "source_vertices6": source_vertices6, "source_faces": source_faces,
            "parent_vertices6": parent_vertices6, "parent_faces": parent_faces,
            "parent_face_to_source": parent_face_to_source,
            "parent_vertex_to_source": parent_vertex_to_source,
            "candidate_vertices6": candidate_vertices6, "candidate_faces": candidate_faces,
            "candidate_face_to_parent": candidate_face_to_parent,
            "candidate_vertex_to_parent": candidate_vertex_to_parent,
            "candidate_to_source_faces": candidate_to_source_faces,
            "candidate_to_source_vertices": candidate_to_source_vertices,
            "retired_parent_faces": retired, "endpoints": (removed, retained),
        }

    @staticmethod
    def _map_hashes(stable_id, evidence):
        evidence["candidate_face_ancestry_sha256"] = _lineage_map_sha256(
            stable_id, evidence["candidate_face_ancestry"], kind="candidate-face-to-parent-face")
        evidence["candidate_vertex_ancestry_sha256"] = _lineage_map_sha256(
            stable_id, evidence["candidate_vertex_ancestry"], kind="candidate-vertex-to-parent-vertex")
        evidence["parent_face_ancestry_sha256"] = _lineage_map_sha256(
            stable_id, evidence["parent_face_ancestry"], kind="parent-face-to-source-face")
        evidence["parent_vertex_ancestry_sha256"] = _lineage_map_sha256(
            stable_id, evidence["parent_vertex_ancestry"], kind="parent-vertex-to-source-vertex")

    @classmethod
    def _evidence(cls, fixture):
        stable_id = 305
        source_vertices6 = fixture["source_vertices6"]
        source_faces = fixture["source_faces"]
        parent_vertices6 = fixture["parent_vertices6"]
        parent_faces = fixture["parent_faces"]
        header = (b"NHANAT1\0", 5, 1, len(parent_vertices6), parent_faces.size, 0, bytes(32))
        parent_payload = encode_payload(header, {stable_id: {
            "body_index": 20, "layer": 8, "flags": 0,
            "vertices6": parent_vertices6, "faces": parent_faces,
        }})
        source_parent_payload = encode_payload(
            (b"NHANAT1\0", 5, 1, len(source_vertices6), source_faces.size, 0, bytes(32)),
            {stable_id: {"body_index": 20, "layer": 8, "flags": 0,
                         "vertices6": source_vertices6, "faces": source_faces}},
        )
        identity_faces = [(i,) for i in range(len(source_faces))]
        identity_vertices = [(i,) for i in range(len(source_vertices6))]
        nested = {
            "parent_vertices6": source_vertices6,
            "parent_faces": source_faces,
            "parent_payload_bytes": source_parent_payload,
            "parent_payload_sha256": hashlib.sha256(source_parent_payload).hexdigest(),
            "parent_surface_sha256": _parent_surface_sha256(stable_id, source_vertices6, source_faces),
            "candidate_face_ancestry": fixture["parent_face_to_source"],
            "candidate_vertex_ancestry": fixture["parent_vertex_to_source"],
            "parent_face_ancestry": identity_faces,
            "parent_vertex_ancestry": identity_vertices,
            "immediate_subdivision_face_ids": (0,),
        }
        cls._map_hashes(stable_id, nested)
        scale = 3
        parent_xyz = fixture["parent_vertices6"][:, :3]
        raw_lattice = np.rint(parent_xyz.astype(np.float64) * scale).astype(np.int64)
        parent_subdivision = {
            "subdivided_source_face_ids": (0,),
            "source_patch_support_face_ids": (0,),
            "patch_geometry": {
                "support_source_face_ids": (0,),
                "raw_vertex_lattice_numerators": raw_lattice,
                "raw_vertex_lattice_denominators": np.ones_like(raw_lattice),
                "lattice_scale": scale,
            },
            "subdivision_parent_evidence": nested,
        }
        evidence = {
            "parent_vertices6": parent_vertices6,
            "parent_faces": parent_faces,
            "parent_payload_bytes": parent_payload,
            "parent_payload_sha256": hashlib.sha256(parent_payload).hexdigest(),
            "parent_surface_sha256": _parent_surface_sha256(stable_id, parent_vertices6, parent_faces),
            "candidate_face_ancestry": fixture["candidate_face_to_parent"],
            "candidate_vertex_ancestry": fixture["candidate_vertex_to_parent"],
            "parent_face_ancestry": fixture["parent_face_to_source"],
            "parent_vertex_ancestry": fixture["parent_vertex_to_source"],
            "edge_endpoint_parent_vertex_ids": fixture["endpoints"],
            "retired_parent_face_ids": fixture["retired_parent_faces"],
            "retained_position_policy": "midpoint",
            "collapse_position_f32_m": fixture["candidate_vertices6"][
                next(i for i, row in enumerate(fixture["candidate_vertex_to_parent"])
                     if row == tuple(sorted(fixture["endpoints"])))
            ][:3].copy(),
            "maximum_endpoint_displacement_bound_m": 0.8,
            "maximum_metric_deviation_bound_m": 1.5,
            "parent_source_subdivision_evidence": parent_subdivision,
        }
        cls._map_hashes(stable_id, evidence)
        return evidence

    def test_collapse_after_exact_parent_subdivision_composes_and_keeps_gates_pending(self):
        fixture = self._surface_and_maps()
        evidence = self._evidence(fixture)
        report = _validate_source_face_edge_collapse_parent_chain(
            305, fixture["candidate_faces"], fixture["candidate_vertices6"],
            fixture["candidate_to_source_faces"], fixture["candidate_to_source_vertices"],
            len(fixture["source_faces"]), len(fixture["source_vertices6"]), evidence,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["retired_parent_face_ids"], list(fixture["retired_parent_faces"]))
        self.assertEqual(report["parent_source_subdivision_evidence"][
            "parent_exact_subdivision_patch_geometry"]["exact_raw_and_float32_patch_boundary_conserved"], True)
        self.assertTrue(report["exact_source_and_world_self_intersection_rechecks_required"])

    def test_collapse_rejects_off_star_edit_and_missing_ancestry(self):
        fixture = self._surface_and_maps()
        evidence = self._evidence(fixture)
        changed = fixture["candidate_vertices6"].copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "changed vertex geometry outside"):
            _validate_source_face_edge_collapse_parent_chain(
                305, fixture["candidate_faces"], changed,
                fixture["candidate_to_source_faces"], fixture["candidate_to_source_vertices"],
                len(fixture["source_faces"]), len(fixture["source_vertices6"]), evidence,
            )
        missing = self._evidence(fixture)
        missing["parent_vertex_ancestry"] = missing["parent_vertex_ancestry"][:-1]
        self._map_hashes(305, missing)
        with self.assertRaisesRegex(ValueError, "length differs"):
            _validate_source_face_edge_collapse_parent_chain(
                305, fixture["candidate_faces"], fixture["candidate_vertices6"],
                fixture["candidate_to_source_faces"], fixture["candidate_to_source_vertices"],
                len(fixture["source_faces"]), len(fixture["source_vertices6"]), missing,
            )

    def test_collapse_rejects_forged_exact_subdivision_parent_patch(self):
        fixture = self._surface_and_maps()
        evidence = self._evidence(fixture)
        evidence["parent_source_subdivision_evidence"]["patch_geometry"][
            "raw_vertex_lattice_numerators"][6, 2] += 1
        with self.assertRaisesRegex(ValueError, "raw geometry does not roundtrip|outside its exact parent"):
            _validate_source_face_edge_collapse_parent_chain(
                305, fixture["candidate_faces"], fixture["candidate_vertices6"],
                fixture["candidate_to_source_faces"], fixture["candidate_to_source_vertices"],
                len(fixture["source_faces"]), len(fixture["source_vertices6"]), evidence,
            )

    def test_collapse_is_not_accepted_without_the_parent_exact_subdivision_proof(self):
        fixture = self._surface_and_maps()
        evidence = self._evidence(fixture)
        del evidence["parent_source_subdivision_evidence"]
        with self.assertRaisesRegex(ValueError, "without its independently checked exact subdivision chain"):
            _validate_source_face_edge_collapse_parent_chain(
                305, fixture["candidate_faces"], fixture["candidate_vertices6"],
                fixture["candidate_to_source_faces"], fixture["candidate_to_source_vertices"],
                len(fixture["source_faces"]), len(fixture["source_vertices6"]), evidence,
            )


if __name__ == "__main__":
    unittest.main()
