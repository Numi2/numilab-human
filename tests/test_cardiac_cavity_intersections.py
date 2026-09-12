"""Exact predicate regressions; no simulator or physical stepping."""
from __future__ import annotations
import copy
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

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
