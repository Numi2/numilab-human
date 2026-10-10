"""Exact predicate regressions; no simulator or physical stepping."""
from __future__ import annotations
import copy
from fractions import Fraction
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from numilab_human import cardiac_cavity_intersections as audit
from numilab_human.cardiac_cavity_geometry import extract_cavity_surfaces
from numilab_human.model import ImportError as HumanImportError


def tetra(name, offset=(0., 0., 0.), scale=1.):
    vertices = [[offset[k]+scale*p[k] for k in range(3)] for p in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
    return {'source_id': name, 'exact_coordinate_quotient': {'vertices_m': vertices,
            'triangles': [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]}}


def run(*chambers):
    return audit.audit_cavity_intersections({'chambers': list(chambers)})


class PredicateTests(unittest.TestCase):
    def pair(self, vertices, faces):
        records = audit._records(vertices, faces)
        return audit._audit_pair(records, records, same_surface=True)

    def test_shared_edge_contact_allowed_but_same_side_overlap_rejected(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0),(0,-2,0),(1,1,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(1,0,3)])['count'], 0)
        self.assertEqual(self.pair(vertices, [(0,1,2),(1,0,4)])['triangle_pairs'], [[0,1]])

    def test_shared_vertex_is_not_blanket_overlap_exemption(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0),(-2,0,0),(0,-2,0),(2,1,0),(1,2,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,3,4)])['count'], 0)
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,5,6)])['count'], 1)

    def test_duplicate_or_unrelated_coincident_faces_rejected(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,2,1)])['count'], 1)
        self.assertEqual(self.pair(vertices+vertices, [(0,1,2),(3,4,5)])['count'], 1)

    def test_transverse_and_coplanar_crossing_without_contained_vertices(self):
        first = ((0,0,0),(4,0,0),(0,4,0))
        transverse = ((1,1,-1),(1,1,1),(2,1,0))
        self.assertTrue(audit.triangle_intersection_points(first, transverse))
        # Star-of-David triangles: intersection polygon comes from edge crossings.
        a = ((0,3,0),(-3,-2,0),(3,-2,0))
        b = ((0,-3,0),(-3,2,0),(3,2,0))
        self.assertEqual(len(audit.triangle_intersection_points(a, b)), 6)

    def test_exact_one_bit_separation_is_not_tolerance_contact(self):
        first = ((0,0,0),(1,0,0),(0,1,0))
        second = ((0,0,1),(1,0,1),(0,1,1))
        self.assertFalse(audit.triangle_intersection_points(first, second))
        a, b = tetra('a'), tetra('b', (math.nextafter(1., math.inf),0.,0.))
        self.assertTrue(run(a,b)['all_domains_disjoint'])
        self.assertFalse(run(a,tetra('b',(1.,0.,0.)))['all_domains_disjoint'])

    def test_float32_shared_face_keys_use_one_exact_integer_lattice(self):
        f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
        vertices = [(f32(.1),f32(.2),f32(.3)), (f32(.8),f32(.2),f32(.3)),
                    (f32(.1),f32(.9),f32(.3)), (f32(.5),f32(.5),f32(.3))]
        forward = audit.float32_triangle_lattice_key(vertices, (0,1,2))
        reverse = audit.float32_triangle_lattice_key(vertices, (2,1,0))
        self.assertEqual(forward, reverse)
        second_mesh = [vertices[index] for index in (2,0,1)]
        self.assertEqual(forward, audit.float32_triangle_lattice_key(second_mesh, (0,1,2)))
        self.assertNotEqual(forward, audit.float32_triangle_lattice_key(vertices, (0,1,3)))
        with self.assertRaisesRegex(HumanImportError, 'exact Float32'):
            audit.float32_point_lattice_key((.1,.2,.3))

    def test_float32_patch_membership_uses_one_scalar_denominator(self):
        f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
        vertices = [(f32(0),f32(0),f32(0)), (f32(1),f32(0),f32(0)),
                    (f32(0),f32(1),f32(0))]
        triangle = tuple(audit.float32_point_lattice_key(vertices[index]) for index in (0,1,2))
        lattice = 1 << 149
        self.assertTrue(audit.float32_lattice_point_on_triangle((lattice,lattice,0),4,triangle))
        self.assertFalse(audit.float32_lattice_point_on_triangle((3*lattice,3*lattice,0),4,triangle))
        with self.assertRaisesRegex(HumanImportError, 'scalar'):
            audit.float32_lattice_point_on_triangle((lattice,lattice,0),(4,4,4),triangle)

    def test_exact_ray_retries_vertex_hit_and_reports_boundary(self):
        t = tetra('test')['exact_coordinate_quotient']
        records = audit._records([tuple(map(int,p)) for p in t['vertices_m']],t['triangles'])
        outside = audit.point_location((-1,-1,-1), records)
        self.assertEqual(outside['location'], 'outside')
        self.assertGreater(outside['attempts'], 1)
        self.assertEqual(audit.point_location((0,0,0), records)['location'], 'boundary')
        self.assertEqual(audit.point_location((Fraction(1,8),)*3, records)['location'], 'inside')

    def test_degenerate_triangles_are_not_admitted(self):
        with self.assertRaisesRegex(HumanImportError,'degenerate'):
            audit.triangle_intersection_points(((0,0,0),(1,0,0),(2,0,0)), ((0,0,1),(1,0,1),(0,1,1)))



class PreparedPointLocationTests(unittest.TestCase):
    @staticmethod
    def tetra_records(scale=12):
        vertices = ((0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        return audit._records(vertices, faces)

    def assert_legacy_parity(self, records, points):
        prepared = audit.prepare_point_location(records)
        for point in points:
            with self.subTest(point=point):
                self.assertEqual(audit.point_location(point, records),
                                 audit.point_location_prepared(point, prepared))
        return prepared

    def test_full_result_inside_outside_boundary_edge_and_vertex(self):
        records = self.tetra_records()
        edge = (Fraction(6), Fraction(6), Fraction(0))
        prepared = self.assert_legacy_parity(
            records, ((1, 1, 1), (-1, -1, -1), (0, 0, 0), edge, (12, 0, 0)))
        self.assertGreater(audit.point_location((-1, -1, -1), records)["attempts"], 1)
        self.assertIn(2, prepared._fallback_ray_data)

    def test_coplanar_ambiguity_behind_forward_aabb_is_preserved(self):
        # The first face lies in x=y, contains the origin in its plane, and is
        # wholly behind the query point. The primary (1,1,1) ray is coplanar.
        # Legacy point_location retries that ray even though no triangle AABB
        # is reachable in the positive direction.
        vertices = ((-1, -1, -1), (-2, -2, -1), (-1, -1, -2), (-3, -4, -3))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        records = audit._records(vertices, faces)
        self.assertTrue(all(any(bound < 0 for bound in row[2]) for row in records))
        prepared = audit.prepare_point_location(records)
        expected = audit.point_location((0, 0, 0), records)
        self.assertEqual(expected["attempts"], 2)
        self.assertEqual(audit.point_location_prepared((0, 0, 0), prepared), expected)
        self.assertIn(2, prepared._fallback_ray_data)


    def test_all_sixty_four_coplanar_rays_reach_indeterminate(self):
        records = []
        for attempt in range(1, 65):
            triangle = ((-1, -attempt, 0), (-2, -2 * attempt, 0),
                        (-1, -attempt, -1))
            lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
            upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
            vertex_base = 3 * (attempt - 1)
            records.append((triangle, lower, upper, attempt - 1,
                            (vertex_base, vertex_base + 1, vertex_base + 2)))
        prepared = audit.prepare_point_location(records)
        expected = audit.point_location((0, 0, 0), records)
        self.assertEqual(expected, {"location": "indeterminate", "ray": None,
                                    "crossings": None, "attempts": 64})
        self.assertEqual(audit.point_location_prepared((0, 0, 0), prepared), expected)
        self.assertEqual(set(prepared._fallback_ray_data), set(range(2, 65)))


    def test_large_float32_lattice_integers_remain_exact(self):
        scale = 1 << 149
        records = self.tetra_records(scale)
        points = ((scale // 8, scale // 8, scale // 8),
                  (scale, 0, 0), (2 * scale, 2 * scale, 2 * scale))
        self.assert_legacy_parity(records, points)

    def test_snapshot_is_immutable_after_caller_mutates_records(self):
        records = self.tetra_records()
        expected = audit.point_location((1, 1, 1), records)
        mutable_records = [
            ([list(point) for point in row[0]], list(row[1]), list(row[2]), row[3], list(row[4]))
            for row in records
        ]
        prepared = audit.prepare_point_location(mutable_records)
        identity = prepared.record_identity_sha256
        mutable_records[0][0][0][0] = 500
        mutable_records[0][1][0] = -500
        mutable_records[0][4][0] = 99
        self.assertEqual(audit.point_location_prepared((1, 1, 1), prepared), expected)
        self.assertEqual(prepared.record_identity_sha256, identity)
        self.assertIsInstance(prepared.records, tuple)
        self.assertIsInstance(prepared.records[0][0], tuple)
        self.assertIsInstance(prepared.records[0][0][0], tuple)
        self.assertEqual(prepared.records, tuple(records))

    def test_empty_degenerate_malformed_and_inexact_inputs_reject(self):
        with self.assertRaisesRegex(HumanImportError, "non-empty"):
            audit.prepare_point_location([])
        degenerate = [(((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                       (0, 0, 0), (2, 0, 0), 0, (0, 1, 2))]
        with self.assertRaisesRegex(HumanImportError, "degenerate"):
            audit.prepare_point_location(degenerate)
        records = self.tetra_records()
        malformed_bounds = list(records)
        row = records[0]
        malformed_bounds[0] = (row[0], (row[1][0] - 1, *row[1][1:]), row[2], row[3], row[4])
        with self.assertRaisesRegex(HumanImportError, "bounds"):
            audit.prepare_point_location(malformed_bounds)
        prepared = audit.prepare_point_location(records)
        with self.assertRaisesRegex(HumanImportError, "exact integer or Fraction"):
            audit.point_location_prepared((0.25, 0, 0), prepared)
        with self.assertRaisesRegex(HumanImportError, "snapshot"):
            audit.point_location_prepared((0, 0, 0), records)



class PreparedSignedWindingTests(unittest.TestCase):
    @staticmethod
    def tetra_records(offset=(0, 0, 0), scale=12, reverse=False):
        base = ((0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale))
        vertices = tuple(tuple(offset[axis] + point[axis] for axis in range(3))
                         for point in base)
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        if reverse:
            faces = tuple((a, c, b) for a, b, c in faces)
        return vertices, faces, audit._records(vertices, faces)

    def test_outward_and_reversed_tetra_have_signed_unit_winding(self):
        _vertices, _faces, outward = self.tetra_records()
        prepared = audit.prepare_signed_winding(outward)
        self.assertEqual(prepared.face_component_count, 1)
        result = audit.signed_winding_number((1, 1, 1), prepared)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["winding_number"], 1)
        self.assertEqual(result["crossings"], 1)

        _vertices, _faces, inward = self.tetra_records(reverse=True)
        reversed_result = audit.signed_winding_number(
            (1, 1, 1), audit.prepare_signed_winding(inward))
        self.assertEqual(reversed_result["winding_number"], -1)

    def test_overlapping_closed_tetrahedra_sum_to_winding_two(self):
        outer_vertices, outer_faces, _outer = self.tetra_records(scale=12)
        inner_vertices, inner_faces, _inner = self.tetra_records(
            offset=(2, 2, 2), scale=12)
        faces = outer_faces + tuple(tuple(index + 4 for index in face)
                                    for face in inner_faces)
        records = audit._records(outer_vertices + inner_vertices, faces)
        first_records = audit._records(outer_vertices, outer_faces)
        second_records = audit._records(inner_vertices, inner_faces)
        self.assertGreater(audit._audit_pair(first_records, second_records,
                                             same_surface=False)["count"], 0)
        prepared = audit.prepare_signed_winding(records)
        self.assertEqual(prepared.face_component_count, 2)
        self.assertEqual(audit.signed_winding_number((3, 3, 3), prepared)["winding_number"], 2)

    def test_nested_opposite_shell_is_zero_in_cavity_and_one_in_material(self):
        outer_vertices, outer_faces, _outer = self.tetra_records(scale=12)
        inner_vertices, inner_faces, _inner = self.tetra_records(
            offset=(1, 1, 1), scale=4, reverse=True)
        faces = outer_faces + tuple(tuple(index + 4 for index in face)
                                    for face in inner_faces)
        records = audit._records(outer_vertices + inner_vertices, faces)
        prepared = audit.prepare_signed_winding(records)
        self.assertEqual(prepared.face_component_count, 2)
        self.assertEqual(audit.signed_winding_number((2, 2, 2), prepared)["winding_number"], 0)
        self.assertEqual(audit.signed_winding_number((6, 1, 1), prepared)["winding_number"], 1)

    def test_boundary_has_no_winding_and_primary_vertex_ray_retries(self):
        _vertices, _faces, records = self.tetra_records()
        prepared = audit.prepare_signed_winding(records)
        boundary = audit.signed_winding_number((0, 0, 0), prepared)
        self.assertEqual(boundary["status"], "boundary")
        self.assertIsNone(boundary["winding_number"])
        outside = audit.signed_winding_number((-1, -1, -1), prepared)
        self.assertEqual(outside["status"], "resolved")
        self.assertEqual(outside["winding_number"], 0)
        self.assertGreater(outside["attempts"], 1)

    def test_open_inconsistently_oriented_and_degenerate_shells_are_rejected(self):
        _vertices, faces, records = self.tetra_records()
        with self.assertRaisesRegex(HumanImportError, "exactly two incident faces"):
            audit.prepare_signed_winding(records[:-1])
        inconsistent_faces = list(faces)
        inconsistent_faces[0] = tuple(reversed(inconsistent_faces[0]))
        inconsistent = audit._records(_vertices, inconsistent_faces)
        with self.assertRaisesRegex(HumanImportError, "consistently oriented"):
            audit.prepare_signed_winding(inconsistent)
        duplicate = records + [(records[0][0], records[0][1], records[0][2],
                                len(records), records[0][4])]
        with self.assertRaisesRegex(HumanImportError, "duplicate faces"):
            audit.prepare_signed_winding(duplicate)

        vertices_a, faces_a, _ = self.tetra_records(scale=4)
        vertices_b, faces_b, _ = self.tetra_records(
            offset=(0, 0, 0), scale=-4)
        shared_vertex_mesh = vertices_a + vertices_b[1:]
        second_map = (0, 4, 5, 6)
        shared_vertex_faces = faces_a + tuple(
            tuple(second_map[index] for index in face) for face in faces_b)
        pinch = audit._records(shared_vertex_mesh, shared_vertex_faces)
        with self.assertRaisesRegex(HumanImportError, "multiple cycles"):
            audit.prepare_signed_winding(pinch)

        degenerate = [(((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                       (0, 0, 0), (2, 0, 0), 0, (0, 1, 2))]
        with self.assertRaisesRegex(HumanImportError, "degenerate"):
            audit.prepare_signed_winding(degenerate)

    def test_unresolved_ray_returns_no_winding(self):
        _vertices, _faces, records = self.tetra_records()
        prepared = audit.prepare_signed_winding(records)
        # Force the exact parallel/coplanar retry path for every deterministic
        # direction while the query lies in the z=0 plane, outside the face.
        cached = (tuple(0 for _ in prepared.point_locator.normals), (0,))
        with patch.object(audit, "_prepared_ray_data", return_value=cached):
            result = audit.signed_winding_number((13, 1, 0), prepared)
        self.assertEqual(result["status"], "indeterminate")
        self.assertIsNone(result["winding_number"])
        self.assertEqual(result["attempts"], 64)


class ClosedDomainTests(unittest.TestCase):
    def test_disjoint_nested_and_intersecting_solids_are_distinguished(self):
        disjoint = run(tetra('a'),tetra('b',(2.,0.,0.)))
        self.assertTrue(disjoint['all_surfaces_embedded'])
        self.assertTrue(disjoint['all_domains_disjoint'])
        nested = run(tetra('outer',scale=2.),tetra('inner',(.25,.25,.25),.25))
        pair = nested['per_pair'][0]
        self.assertEqual(pair['count'], 0)
        self.assertFalse(nested['all_domains_disjoint'])
        self.assertEqual(pair['containment']['first_in_second']['location'], 'inside')
        crossing = run(tetra('a'),tetra('b',(.25,.25,.25)))
        self.assertGreater(crossing['per_pair'][0]['count'],0)
        self.assertEqual(crossing['per_pair'][0]['containment']['status'],'not_checked_intersecting_surfaces')

    def test_cached_topology_claim_cannot_hide_open_or_disconnected_surface(self):
        open_surface = tetra('open')
        open_surface['exact_coordinate_quotient']['triangles'].pop()
        open_surface['exact_coordinate_quotient']['topology'] = {'closed_oriented_manifold_candidate':True}
        self.assertFalse(run(open_surface)['all_domains_disjoint'])
        a, b = tetra('disconnected'), tetra('unused',(2.,0.,0.))
        q, r = a['exact_coordinate_quotient'], b['exact_coordinate_quotient']
        q['vertices_m'] += r['vertices_m']
        q['triangles'] += [[k+4 for k in face] for face in r['triangles']]
        self.assertFalse(run(a)['all_surfaces_embedded'])

    def test_unused_first_vertex_cannot_turn_nested_domains_into_disjoint_solids(self):
        outer = tetra('outer', scale=2.)
        inner = tetra('inner', (.25, .25, .25), .25)
        nested = run(outer, inner)
        self.assertTrue(nested['all_surfaces_embedded'])
        self.assertFalse(nested['all_domains_disjoint'])
        self.assertEqual(nested['per_pair'][0]['containment']['first_in_second']['location'], 'inside')
        q = inner['exact_coordinate_quotient']
        q['vertices_m'].insert(0, [9., 9., 9.])
        q['triangles'] = [[index + 1 for index in face] for face in q['triangles']]
        # The actual inner surface is unchanged, but vertex zero now lies
        # outside the outer shell. It must never serve as a containment witness.
        result = run(outer, inner)
        self.assertEqual(result['per_pair'][0]['count'], 0)
        self.assertFalse(result['per_surface']['inner']['embedded_closed_surface'])
        self.assertFalse(result['all_surfaces_embedded'])
        self.assertFalse(result['all_domains_disjoint'])
        self.assertEqual(result['per_pair'][0]['containment']['status'], 'not_checked_invalid_surface')

    def test_translation_and_reversed_orientation_preserve_disjointness(self):
        a,b=tetra('a'),tetra('b',(2.,0.,0.))
        expected=run(a,b)
        for chamber in (a,b):
            q=chamber['exact_coordinate_quotient']
            q['vertices_m']=[[p[0]+1024,p[1]-2048,p[2]+4096] for p in q['vertices_m']]
            q['triangles']=[list(reversed(face)) for face in q['triangles']]
        actual=run(a,b)
        self.assertEqual(expected['all_domains_disjoint'],actual['all_domains_disjoint'])
        self.assertEqual([r['triangle_pairs'] for r in expected['per_pair']], [r['triangle_pairs'] for r in actual['per_pair']])

    def test_duplicate_ids_nonfinite_and_wrong_arity_rejected(self):
        with self.assertRaisesRegex(HumanImportError,'duplicate'):
            run(tetra('a'),tetra('a'))
        for value in (float('nan'),float('inf'),True,10**1000):
            bad=tetra('a');bad['exact_coordinate_quotient']['vertices_m'][0][0]=value
            with self.assertRaises(HumanImportError):run(bad)
        bad=tetra('a');bad['exact_coordinate_quotient']['triangles'][0]=[0,1,2,3]
        with self.assertRaises(HumanImportError):run(bad)


class SourceGeometryTests(unittest.TestCase):
    def test_pinned_source_is_embedded_but_right_chambers_intersect(self):
        cavities=extract_cavity_surfaces()
        original=copy.deepcopy(cavities)
        result=audit.audit_cavity_intersections(cavities)
        self.assertEqual(cavities,original)
        self.assertTrue(result['all_surfaces_embedded'])
        self.assertFalse(result['all_domains_disjoint'])
        self.assertTrue(all(row['count']==0 for row in result['per_surface'].values()))
        pairs={(p['first'],p['second']):p for p in result['per_pair']}
        overlap=pairs[('right_atrium','right_ventricle')]
        self.assertEqual(overlap['count'],42)
        self.assertIn([748,1916],overlap['triangle_pairs'])
        self.assertTrue(all(p['disjoint_closed_domains'] for key,p in pairs.items() if key!=('right_atrium','right_ventricle')))
        self.assertEqual(result['coordinate_semantics'],'exact_rational_value_of_published_binary64_metres')
        self.assertFalse(result['source_coordinates_modified'])
        self.assertFalse(result['overlap_repair'])


if __name__ == '__main__':
    unittest.main()
