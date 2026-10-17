"""Independent exact tetrahedral fixtures for arrangement and ownership proofs."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from numilab_human import cardiac_partition_certificate as cert
from numilab_human.model import ImportError as HumanImportError

A,B=cert.SOURCES
FACES=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]


def fixture():
    av=[(F(0),F(0),F(0)),(F(2),F(0),F(0)),(F(0),F(2),F(0)),(F(0),F(0),F(2))]
    bv=[tuple(x+F(1,2) for x in p) for p in av]
    sources={A:{'vertices':av,'triangles':FACES.copy(),'source_sha256':'1'*64},
             B:{'vertices':bv,'triangles':FACES.copy(),'source_sha256':'2'*64}}
    rows=[]
    def add(source,parent,tri,location):
        rows.append({'source':source,'source_face':parent,'vertices':tri,'other_location':location})
    for i,face in enumerate(FACES[:3]):add(A,i,tuple(av[k] for k in face),'outside')
    x,y,z=av[1:];p=(F(1),F(1,2),F(1,2));q=(F(1,2),F(1),F(1,2));r=(F(1,2),F(1,2),F(1))
    for tri in ((x,y,q),(x,q,p),(y,z,r),(y,r,q),(z,x,p),(z,p,r)):add(A,3,tri,'outside')
    add(A,3,(p,q,r),'inside')
    for i,face in enumerate(FACES[:3]):
        o,u,v=[bv[k] for k in face]
        near=lambda point:tuple(o[k]+(point[k]-o[k])/4 for k in range(3))
        nu,nv=near(u),near(v)
        add(B,i,(o,nu,nv),'inside');add(B,i,(nu,u,v),'outside');add(B,i,(nu,v,nv),'outside')
    add(B,3,tuple(bv[k] for k in FACES[3]),'outside')
    return sources,rows


def check(sources,rows,**kwargs):
    return cert.certify_partition(sources,rows,expected_geometry_sha256={s:cert.source_geometry_sha256(v) for s,v in sources.items()},**kwargs)


def decode(value):return F(int(value[0],16),int(value[1],16))


def emitted(rows,priority=A):
    p=cert.build_triangle_sets(rows)['partitions'][priority+'_priority']
    pool={v:tuple(float(x) for x in v) for row in rows for v in row['vertices']}
    convert=lambda tris:[tuple(pool[v] for v in t) for t in tris]
    return {s:convert(t) for s,t in p['regions'].items()},convert(p['shared_interface'])


def legacy_prepared_location(prepared, point):
    """Reference the pre-filter _Prepared.location loop for parity tests."""
    if any(point[k] < prepared.lo[k] or point[k] > prepared.hi[k] for k in range(3)):
        return "outside", 0
    for row in prepared.records:
        if all(row[1][k] <= point[k] <= row[2][k] for k in range(3)):
            normal = prepared.normals[row[3]]
            if cert._dot(normal, cert._sub(point, row[0][0])) == 0 and cert.predicates._inside(
                    cert.predicates._signs(point, 1, row[0], normal)):
                return "boundary", 0
    for attempt, direction in enumerate(cert._RAY_DIRECTIONS, 1):
        crossings, ambiguous = 0, False
        for row in prepared.records:
            lo = max(F(row[1][k] - point[k], direction[k]) for k in range(3))
            hi = min(F(row[2][k] - point[k], direction[k]) for k in range(3))
            if hi <= 0 or lo > hi:
                continue
            tri = row[0]
            normal = prepared.normals[row[3]]
            numerator = cert._dot(normal, cert._sub(tri[0], point))
            denominator = cert._dot(normal, direction)
            if not denominator:
                if not numerator:
                    ambiguous = True
                    break
                continue
            if numerator * denominator <= 0:
                continue
            hit = tuple(point[k] * denominator + numerator * direction[k] for k in range(3))
            if denominator < 0:
                denominator, hit = -denominator, tuple(-x for x in hit)
            signs = cert.predicates._signs(hit, denominator, tri, normal)
            if cert.predicates._inside(signs):
                if 0 in signs:
                    ambiguous = True
                    break
                crossings += 1
        if not ambiguous:
            return ("inside" if crossings % 2 else "outside"), attempt
    return "indeterminate", 64


class PositiveRayUpperOrthantTests(unittest.TestCase):
    @staticmethod
    def tetra(vertices, faces):
        return cert._Prepared([tuple(vertices[index] for index in face) for face in faces])

    def test_boundary_edge_inside_outside_and_negative_queries_match_full_face_loop(self):
        vertices = ((F(0), F(0), F(0)), (F(12), F(0), F(0)),
                    (F(0), F(12), F(0)), (F(0), F(0), F(12)))
        prepared = self.tetra(vertices, FACES)
        points = ((F(1, 4), F(1, 4), F(1, 4)), (-1, -1, -1),
                  (F(0), F(0), F(0)), (F(6), F(6), F(0)),
                  (F(12), F(0), F(0)), (F(30), F(-4), F(2)))
        for point in points:
            with self.subTest(point=point):
                self.assertEqual(prepared.location(point), legacy_prepared_location(prepared, point)[0])

    def test_exact_rational_query_grid_matches_unfiltered_closed_surfaces(self):
        base = ((F(0), F(0), F(0)), (F(4), F(0), F(0)),
                (F(0), F(4), F(0)), (F(0), F(0), F(4)))
        # Translation and a nonsingular rational shear exercise negative origins,
        # oblique faces, exact boundary points and ambiguous-ray retries.
        for shift in (F(0), F(-3)):
            vertices = tuple((x + y / 2 + shift, y + z / 3 + shift, z + shift)
                             for x, y, z in base)
            prepared = self.tetra(vertices, FACES)
            for x in (F(-1), F(0), F(1, 3), F(2), F(4)):
                for y in (F(-1), F(0), F(1, 3), F(2), F(4)):
                    for z in (F(-1), F(0), F(1, 3), F(2), F(4)):
                        point = (x + y / 2 + shift, y + z / 3 + shift, z + shift)
                        with self.subTest(shift=shift, point=point):
                            self.assertEqual(prepared.location(point),
                                             legacy_prepared_location(prepared, point)[0])

    def test_parallel_coplanar_ray_retries_with_identical_location(self):
        # The face x=y is coplanar with the first (1,1,1) ray. This negative
        # query remains inside the mesh whole bounds, lies before that face,
        # and the ray crosses the face interior; the unfiltered ray must retry.
        vertices = ((F(0), F(0), F(0)), (F(4), F(4), F(0)),
                    (F(0), F(0), F(4)), (F(-2), F(-1), F(-2)))
        prepared = self.tetra(vertices, FACES)
        point = (F(-1), F(-1), F(-1, 2))
        expected, attempts = legacy_prepared_location(prepared, point)
        self.assertEqual(attempts, 2)
        self.assertEqual(prepared.location(point), expected)

    def test_upper_orthant_filter_keeps_equality_and_original_face_order(self):
        vertices = ((F(0), F(0), F(0)), (F(2), F(0), F(0)),
                    (F(0), F(2), F(0)), (F(0), F(0), F(2)))
        prepared = self.tetra(vertices, FACES)
        point = (F(2), F(0), F(0))
        expected = tuple(row for row in prepared.records
                         if all(row[2][axis] >= point[axis] for axis in range(3)))
        actual = cert._positive_ray_upper_orthant_candidates(prepared.records, point)
        self.assertEqual(actual, expected)
        self.assertTrue(any(row[2][0] == point[0] for row in actual))
        self.assertEqual([row[3] for row in actual], sorted(row[3] for row in actual))
        self.assertTrue(any(any(row[2][axis] < point[axis] for axis in range(3))
                            for row in prepared.records))
        self.assertLess(len(actual), len(prepared.records))

    def test_empty_orthant_inside_global_bounds_matches_full_loop(self):
        triangles = (
            ((F(0), F(-2), F(-2)), (F(2), F(-2), F(-2)), (F(1), F(0), F(-2))),
            ((F(-2), F(0), F(0)), (F(-2), F(2), F(0)), (F(-2), F(1), F(2))),
        )
        prepared = cert._Prepared(triangles)
        point = (F(0), F(0), F(0))
        self.assertTrue(all(prepared.lo[k] <= point[k] <= prepared.hi[k] for k in range(3)))
        self.assertEqual(cert._positive_ray_upper_orthant_candidates(prepared.records, point), ())
        self.assertEqual(prepared.location(point), legacy_prepared_location(prepared, point)[0])


class PreparedCertificateSpatialIndexTests(unittest.TestCase):
    def test_triangle_candidates_match_legacy_closed_box_scan(self):
        base = ((F(0), F(0), F(0)), (F(2), F(0), F(0)),
                (F(0), F(2), F(0)), (F(0), F(0), F(2)))
        triangles = [tuple((base[v][0]+shift, base[v][1], base[v][2]) for v in face)
                     for shift in (F(-8), F(0), F(9)) for face in FACES]
        prepared = cert._Prepared(triangles)
        queries = triangles + [
            ((F(-1, 7), F(0), F(0)), (F(0), F(1, 3), F(0)),
             (F(1, 5), F(0), F(2))),
            ((F(2), F(0), F(0)), (F(2), F(1), F(0)), (F(2), F(0), F(1)))]
        for tri in queries:
            lo=tuple(min(p[k] for p in tri) for k in range(3))
            hi=tuple(max(p[k] for p in tri) for k in range(3))
            expected=[row for row in prepared.ordered
                      if all(lo[k]<=row[2][k] and row[1][k]<=hi[k] for k in range(3))]
            with self.subTest(triangle=tri):
                self.assertEqual(prepared.candidates(tri), expected)

    def test_disconnected_shells_and_thin_rational_queries_match_full_ray_loop(self):
        triangles=[]
        for shift, width in ((F(-7), F(1, 2**18)), (F(0), F(2)), (F(8), F(3, 7))):
            vertices=((shift, F(0), F(0)), (shift+width, F(0), F(0)),
                      (shift, width, F(0)), (shift, F(0), width))
            triangles.extend(tuple(vertices[i] for i in face) for face in FACES)
        prepared=cert._Prepared(triangles)
        for tri in triangles:
            center=tuple(sum(p[k] for p in tri)/3 for k in range(3))
            for offset in (F(0), F(1, 2**24), F(-1, 2**24)):
                point=tuple(x+offset for x in center)
                with self.subTest(point=point):
                    self.assertEqual(prepared.location(point),
                                     legacy_prepared_location(prepared,point)[0])


class InterfaceEdgeIndexTests(unittest.TestCase):
    def test_pruned_contacts_match_exhaustive_exact_segment_predicate(self):
        edges = {((F(0), F(0), F(0)), (F(4), F(2), F(0))),
                 ((F(-3), F(1), F(-2)), (F(-3), F(1), F(5))),
                 ((F(7), F(-4), F(2)), (F(9), F(-4), F(2)))}
        vertices = {p for edge in edges for p in edge}
        index = cert._prepare_interface_edges(edges)
        contacts = [
            [(F(1), F(1, 2), F(0)), (F(3), F(3, 2), F(0))],
            [(F(-3), F(1), F(0)), (F(-3), F(1), F(4))],
            [(F(1), F(3, 4), F(0)), (F(3), F(3, 2), F(0))],
            [(F(0), F(0), F(0))] * 2,
            [(F(4), F(2), F(0)), (F(7), F(-4), F(2))],
            [(F(8), F(-4), F(2))],
            [(F(4), F(2), F(0)), (F(4), F(2), F(1))],
        ]
        for points in contacts:
            expected = ((len(set(points)) == 1 and all(p in vertices for p in points))
                        or any(all(cert._on_segment(p, *edge) for p in points)
                               for edge in edges))
            with self.subTest(points=points):
                self.assertEqual(cert._contact_on_interface(points, vertices, index), expected)

    def test_exact_box_index_discards_remote_segments_before_predicates(self):
        from unittest.mock import patch
        edges = {((i, 0, 0), (i, 1, 0)) for i in range(1024)}
        points = [(F(500), F(1, 4), F(0)), (F(500), F(3, 4), F(0))]
        index = cert._prepare_interface_edges(edges)
        original = cert._on_segment
        with patch.object(cert, "_on_segment", wraps=original) as tested:
            self.assertTrue(cert._contact_on_interface(points, set(), index))
            self.assertEqual(tested.call_count, 2)

    def test_off_segment_contact_in_overlapping_box_still_fails(self):
        edge = ((0, 0, 0), (2, 2, 2))
        index = cert._prepare_interface_edges({edge})
        self.assertFalse(cert._contact_on_interface([(F(1), F(1), F(1, 2))],
                                                   set(edge), index))


class ExactCoverageTests(unittest.TestCase):
    def test_custom_source_names_must_be_two_distinct_nonempty_strings(self):
        for names in (("same","same"), ("", "other"), (1, "other")):
            with self.assertRaisesRegex(HumanImportError, "distinct named sources"):
                cert.build_triangle_sets([], source_names=names)
        sources, rows = fixture()
        expected={A:cert.source_geometry_sha256(sources[A]), B:cert.source_geometry_sha256(sources[B])}
        with self.assertRaisesRegex(HumanImportError, "source coverage differs"):
            cert.certify_partition(sources, rows, expected_geometry_sha256=expected,
                                   source_names=("same","same"))
    @staticmethod
    def append_component(sources, rows, name, origin, edge=F(1), reverse=False):
        points = [origin, tuple(origin[k] + (edge if k == 0 else 0) for k in range(3)),
                  tuple(origin[k] + (edge if k == 1 else 0) for k in range(3)),
                  tuple(origin[k] + (edge if k == 2 else 0) for k in range(3))]
        offset = len(sources[name]['vertices'])
        parent = len(sources[name]['triangles'])
        faces = [(a,c,b) if reverse else (a,b,c) for a,b,c in FACES]
        sources[name]['vertices'].extend(points)
        sources[name]['triangles'].extend(tuple(offset+i for i in face) for face in faces)
        rows.extend({'source':name,'source_face':parent+i,
                     'vertices':tuple(points[k] for k in face),'other_location':'outside'}
                    for i,face in enumerate(faces))

    def test_disconnected_source_requires_exact_declared_component_inventory(self):
        sources,rows=fixture()
        self.append_component(sources,rows,A,(F(10),F(10),F(10)))
        with self.assertRaisesRegex(HumanImportError,'one connected boundary'):
            check(sources,rows)
        proof=check(sources,rows,expected_source_component_counts={A:2,B:1})
        self.assertEqual(proof['source_topology'][A]['face_components'],2)
        self.assertEqual(proof['expected_source_component_counts'],{A:2,B:1})
        self.assertTrue(proof['source_face_coverage_exact'])
        self.assertEqual(decode(proof['source_moments'][A]['volume_m3']),F(3,2))
        for partition in proof['partitions'].values():
            self.assertTrue(partition['interiors_disjoint_by_source_membership'])
            self.assertTrue(partition['shared_interface_opposite_chain'])
        for counts in ({A:1,B:1},{A:3,B:1}):
            with self.assertRaisesRegex(HumanImportError,'component count differs'):
                check(sources,rows,expected_source_component_counts=counts)
        for counts in ({A:True,B:1},{A:0,B:1},{A:2},[2,1]):
            with self.assertRaisesRegex(HumanImportError,'invalid expected source'):
                check(sources,rows,expected_source_component_counts=counts)

    def test_declared_negative_cavity_preserves_full_source_coverage_and_signed_volume(self):
        sources,rows=fixture()
        self.append_component(sources,rows,B,(F(3,4),F(3,4),F(9,8)),F(1,16),reverse=True)
        counts={A:1,B:2}
        proof=check(sources,rows,expected_source_component_counts=counts)
        self.assertEqual(decode(proof['source_moments'][B]['volume_m3']),F(4,3)-F(1,24576))
        self.assertEqual(proof['source_topology'][B]['face_components'],2)
        with self.assertRaises(HumanImportError):
            check(sources,rows[:-1],expected_source_component_counts=counts)
        altered=deepcopy(rows);altered[-1]['other_location']='inside'
        with self.assertRaisesRegex(HumanImportError,'classification'):
            check(sources,altered,expected_source_component_counts=counts)

    def test_component_inventory_rejects_wrong_nested_or_isolated_shell_orientation(self):
        for reverse,origin in ((False,(F(3,4),F(3,4),F(9,8))),
                               (True,(F(10),F(10),F(10)))):
            sources,rows=fixture()
            self.append_component(sources,rows,B,origin,F(1,16),reverse=reverse)
            with self.assertRaisesRegex(HumanImportError,'component winding'):
                check(sources,rows,expected_source_component_counts={A:1,B:2})
            regions,interface=emitted(rows)
            with self.assertRaisesRegex(HumanImportError,'component winding'):
                cert.audit_shared_interface_partition(
                    regions,interface,expected_region_component_counts={A:1,B:2})

    def test_component_inventory_does_not_bypass_source_embedding(self):
        sources,rows=fixture()
        self.append_component(sources,rows,A,(F(1,4),F(1,4),F(1,4)),edge=F(2))
        with self.assertRaisesRegex(HumanImportError,'source self-intersection'):
            check(sources,rows,expected_source_component_counts={A:2,B:1})

    def test_arrangement_over_explicit_50000_resource_ceiling_is_rejected(self):
        sources, rows = fixture()
        oversized = [rows[0]] * (cert.MAX_CERTIFIED_SUBTRIANGLES + 1)
        expected = {s: cert.source_geometry_sha256(v) for s, v in sources.items()}
        with self.assertRaisesRegex(HumanImportError, 'invalid arrangement size'):
            cert.certify_partition(sources, oversized, expected_geometry_sha256=expected)

    def test_hand_authored_arrangement_has_exact_known_volume_and_raw_moments(self):
        sources,rows=fixture();proof=check(sources,rows)
        self.assertTrue(proof['source_face_coverage_exact'])
        self.assertTrue(proof['independent_classification_exact'])
        self.assertEqual(decode(proof['intersection']['moments']['volume_m3']),F(1,48))
        self.assertEqual(decode(proof['union']['moments']['volume_m3']),F(127,48))
        self.assertEqual([decode(v) for v in proof['intersection']['moments']['first_volume_moment_m4']],[F(5,384)]*3)
        second=proof['intersection']['moments']['second_volume_moment_m5']
        self.assertEqual(decode(second[0][0]),F(1,120))
        self.assertEqual(decode(second[0][1]),F(31,3840))
        for name,partition in proof['partitions'].items():
            self.assertTrue(partition['shared_interface_opposite_chain'])
            self.assertTrue(partition['interiors_disjoint_by_source_membership'])
            volumes=[decode(r['moments']['volume_m3']) for r in partition['regions'].values()]
            self.assertEqual(sum(volumes),F(127,48))
        self.assertFalse(proof['biological_valve_interface'])
        self.assertFalse(proof['mechanical_mass_assigned'])

    def test_gap_duplicate_overlap_and_reversed_child_are_rejected(self):
        sources,rows=fixture()
        for altered in (rows[1:],rows+[deepcopy(rows[0])]):
            with self.assertRaises(HumanImportError):check(sources,altered)
        altered=deepcopy(rows);altered[0]['vertices']=tuple(reversed(altered[0]['vertices']))
        with self.assertRaisesRegex(HumanImportError,'winding'):check(sources,altered)

    def test_wrong_parent_plane_and_outside_parent_bounds_are_rejected(self):
        sources,rows=fixture();altered=deepcopy(rows);altered[0]['source_face']=1;altered[1]['source_face']=0
        with self.assertRaisesRegex(HumanImportError,'source-face plane'):check(sources,altered)
        altered=deepcopy(rows);tri=list(altered[0]['vertices']);tri[1]=tuple(x*2 for x in tri[1]);altered[0]['vertices']=tuple(tri)
        with self.assertRaisesRegex(HumanImportError,'outside source face'):check(sources,altered)

    def test_centroid_label_cannot_hide_an_uncut_transverse_intersection(self):
        sources,rows=fixture()
        altered=[r for r in rows if not(r['source']==A and r['source_face']==3)]
        altered.append({'source':A,'source_face':3,'vertices':tuple(sources[A]['vertices'][i] for i in FACES[3]),'other_location':'outside'})
        with self.assertRaisesRegex(HumanImportError,'child interior crosses'):check(sources,altered)

    def test_false_classification_and_forged_parent_rejected(self):
        sources,rows=fixture();altered=deepcopy(rows);altered[0]['other_location']='inside'
        with self.assertRaisesRegex(HumanImportError,'classification'):check(sources,altered)
        altered=deepcopy(rows);altered[0]['source_face']=99
        with self.assertRaisesRegex(HumanImportError,'forged source parent'):check(sources,altered)
        altered=deepcopy(rows);altered[0]['source']='forged'
        with self.assertRaisesRegex(HumanImportError,'forged source parent'):check(sources,altered)

    def test_source_geometry_digest_cannot_be_reused_after_coordinate_drift(self):
        sources,rows=fixture();expected={s:cert.source_geometry_sha256(v) for s,v in sources.items()}
        sources[A]['vertices'][0]=(F(1,8),F(0),F(0))
        with self.assertRaisesRegex(HumanImportError,'identity'):
            cert.certify_partition(sources,rows,expected_geometry_sha256=expected)
        sources,rows=fixture();sources[A]['source_sha256']='3'*64
        with self.assertRaisesRegex(HumanImportError,'identity'):
            cert.certify_partition(sources,rows,expected_geometry_sha256=expected)

    def test_arbitrary_rationals_are_not_claimed_as_published_binary64_source(self):
        sources,rows=fixture();sources[A]['vertices'][0]=(F(1,3),F(0),F(0))
        with self.assertRaisesRegex(HumanImportError,'binary64'):check(sources,rows)

    def test_refined_positive_parent_coverage_preserves_exact_moments(self):
        sources,rows=fixture();index=next(i for i,r in enumerate(rows) if r['source']==A and r['other_location']=='inside')
        row=rows.pop(index);t=row['vertices'];center=tuple(sum(p[k] for p in t)/3 for k in range(3))
        for i in range(3):rows.append(dict(row,vertices=(t[i],t[(i+1)%3],center)))
        proof=check(sources,rows)
        self.assertEqual(decode(proof['intersection']['moments']['volume_m3']),F(1,48))

    def test_raw_moments_follow_exact_translation_identity(self):
        sources,rows=fixture();base=check(sources,rows)['union']['moments'];shift=(F(1024),F(-2048),F(4096))
        move=lambda p:tuple(p[k]+shift[k] for k in range(3))
        for surface in sources.values():surface['vertices']=[move(p) for p in surface['vertices']]
        for row in rows:row['vertices']=tuple(move(p) for p in row['vertices'])
        result=check(sources,rows)['union']['moments'];v=decode(base['volume_m3']);m=[decode(x) for x in base['first_volume_moment_m4']]
        self.assertEqual(decode(result['volume_m3']),v)
        for i in range(3):
            self.assertEqual(decode(result['first_volume_moment_m4'][i]),m[i]+v*shift[i])
            for j in range(3):
                self.assertEqual(decode(result['second_volume_moment_m5'][i][j]),decode(base['second_volume_moment_m5'][i][j])+shift[i]*m[j]+shift[j]*m[i]+v*shift[i]*shift[j])

    def test_hex_encoding_roundtrips_large_exact_moments_without_global_changes(self):
        limit=sys.get_int_max_str_digits();value=F((1<<20000)+1,(1<<20001)+3)
        self.assertEqual(decode(cert.encode_rational(value)),value)
        self.assertEqual(sys.get_int_max_str_digits(),limit)


class EmittedInterfaceTests(unittest.TestCase):
    def test_emitted_self_failure_retains_exact_face_pair_audit(self):
        from unittest.mock import patch
        _, rows = fixture()
        regions, interface = emitted(rows, A)
        failed = {"count": 1, "triangle_pairs": [(0, 2)],
                  "aabb_candidate_pairs": 3, "allowed_shared_index_contacts": 2}
        with (
            patch.object(cert.predicates, "_audit_pair", return_value=failed),
            self.assertRaisesRegex(HumanImportError, "emitted region self-intersects") as caught,
        ):
            cert.audit_shared_interface_partition(regions, interface)
        self.assertEqual(caught.exception.diagnostic,
                         {"stage": "emitted_self_intersection", "region": A, "audit": failed})

    def test_emitted_cross_contact_failure_retains_faces_and_exact_metre_points(self):
        from unittest.mock import patch
        _, rows = fixture()
        regions, interface = emitted(rows, A)
        # Force rejection at the unchanged contact guard, after valid self checks.
        with (
            patch.object(cert, "_contact_on_interface", return_value=False),
            self.assertRaisesRegex(HumanImportError, "outside their declared interface") as caught,
        ):
            cert.audit_shared_interface_partition(regions, interface)
        detail = caught.exception.diagnostic
        self.assertEqual(detail["stage"], "emitted_cross_domain_contact")
        self.assertEqual((detail["first_region"], detail["second_region"]), (A, B))
        self.assertIs(type(detail["first_face"]), int)
        self.assertIs(type(detail["second_face"]), int)
        self.assertFalse(detail["contact_on_declared_interface"])
        first = {tuple(decode(x) for x in p) for p in detail["first_triangle_m"]}
        second = {tuple(decode(x) for x in p) for p in detail["second_triangle_m"]}
        self.assertEqual(first, {tuple(F.from_float(x) for x in p)
                                 for p in regions[A][detail["first_face"]]})
        self.assertEqual(second, {tuple(F.from_float(x) for x in p)
                                  for p in regions[B][detail["second_face"]]})
        self.assertTrue(detail["points_m"])
        for point in detail["points_m"]:
            value = tuple(decode(x) for x in point)
            self.assertTrue(all(isinstance(x, F) for x in value))

    def test_bounded_contact_collection_stops_at_limit_without_admission(self):
        from unittest.mock import patch
        _, rows = fixture()
        regions, interface = emitted(rows, A)
        with (
            patch.object(cert, "_contact_on_interface", return_value=False),
            self.assertRaises(HumanImportError) as caught,
        ):
            cert.audit_shared_interface_partition(
                regions, interface, diagnostic_contact_limit=2)
        detail = caught.exception.diagnostic
        self.assertEqual(detail["diagnostic_contact_limit"], 2)
        self.assertFalse(detail["cross_domain_contact_scan_complete"])
        witnesses = detail["unexpected_contacts"]
        self.assertEqual(len(witnesses), 2)
        self.assertEqual(len({(x["first_face"], x["second_face"]) for x in witnesses}), 2)
        for key, value in witnesses[0].items():
            if key != "completed_cross_domain_contact_count":
                self.assertEqual(detail[key], value)

    def test_bounded_contact_collection_reports_complete_failed_scan(self):
        from unittest.mock import patch
        _, rows = fixture()
        regions, interface = emitted(rows, A)
        with (
            patch.object(cert, "_contact_on_interface", return_value=False),
            self.assertRaises(HumanImportError) as caught,
        ):
            cert.audit_shared_interface_partition(
                regions, interface, diagnostic_contact_limit=10000,
                classify_connected_patches=True)
        detail = caught.exception.diagnostic
        self.assertTrue(detail["cross_domain_contact_scan_complete"])
        self.assertGreater(len(detail["unexpected_contacts"]), 2)
        self.assertGreaterEqual(detail["completed_cross_domain_contact_count"],
                                detail["unexpected_contacts"][-1]["completed_cross_domain_contact_count"])

    def test_diagnostic_limit_preserves_valid_certificate(self):
        _, rows = fixture()
        regions, interface = emitted(rows, A)
        self.assertEqual(
            cert.audit_shared_interface_partition(regions, interface),
            cert.audit_shared_interface_partition(
                regions, interface, diagnostic_contact_limit=10000))

    def test_invalid_diagnostic_limit_is_rejected_before_geometry(self):
        for value in (0, -1, True, 1.5, 10001):
            with self.subTest(value=value):
                with self.assertRaisesRegex(HumanImportError, "diagnostic contact limit"):
                    cert.audit_shared_interface_partition(
                        {}, [], diagnostic_contact_limit=value)

    def test_both_priorities_admit_shared_interface_with_disjoint_interiors(self):
        _,rows=fixture()
        for priority in (A,B):
            regions,interface=emitted(rows,priority);proof=cert.audit_shared_interface_partition(regions,interface)
            self.assertTrue(proof['interiors_disjoint'])
            self.assertTrue(proof['shared_interface_opposite_chain'])
            self.assertEqual(decode(proof['union_moments']['volume_m3']),F(127,48))
            self.assertTrue(all(v['outside_noninterface_face_count']>0 for v in proof['per_region'].values()))

    def test_inventory_preserves_emitted_negative_cavity_without_waiving_contacts(self):
        sources,rows=fixture()
        ExactCoverageTests.append_component(
            sources,rows,B,(F(3,4),F(3,4),F(9,8)),F(1,16),reverse=True)
        for priority in (A,B):
            regions,interface=emitted(rows,priority)
            with self.assertRaisesRegex(HumanImportError,'one connected boundary'):
                cert.audit_shared_interface_partition(regions,interface)
            for patch_mode in (False,True):
                proof=cert.audit_shared_interface_partition(
                    regions,interface,expected_region_component_counts={A:1,B:2},
                    classify_connected_patches=patch_mode)
                self.assertTrue(proof['interiors_disjoint'])
                self.assertEqual(proof['per_region'][B]['topology']['face_components'],2)
                self.assertEqual(decode(proof['union_moments']['volume_m3']),F(127,48)-F(1,24576))
            with self.assertRaisesRegex(HumanImportError,'component count differs'):
                cert.audit_shared_interface_partition(
                    regions,interface,expected_region_component_counts={A:1,B:1})
            with self.assertRaises(HumanImportError):
                cert.audit_shared_interface_partition(
                    regions,interface[:-1],expected_region_component_counts={A:1,B:2})
            for counts in ({A:True,B:2},{A:1,B:0},{B:2},[1,2]):
                with self.assertRaisesRegex(HumanImportError,'invalid expected emitted'):
                    cert.audit_shared_interface_partition(
                        regions,interface,expected_region_component_counts=counts)

    def test_connected_patch_mode_matches_full_proof_after_exact_refinement(self):
        _, rows = fixture()
        def refine(tris):
            result = []
            for a, b, c in tris:
                ab, bc, ca = (tuple((x+y)/2 for x,y in zip(u,v))
                              for u,v in ((a,b),(b,c),(c,a)))
                result.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
            return result
        for priority in (A, B):
            regions, interface = emitted(rows, priority)
            for level in range(2):
                full = cert.audit_shared_interface_partition(regions, interface)
                fast = cert.audit_shared_interface_partition(
                    regions, interface, classify_connected_patches=True)
                self.assertEqual(fast.pop('outside_classification'),
                                 'exact_uncut_connected_boundary_patches')
                for row in fast['per_region'].values():
                    count = row.pop('outside_patch_count')
                    witnesses = row.pop('outside_patch_witnesses')
                    self.assertEqual(count, len(witnesses))
                    self.assertEqual(sum(x['face_count'] for x in witnesses),
                                     row['outside_noninterface_face_count'])
                    self.assertLess(count, row['outside_noninterface_face_count'])
                self.assertEqual(fast, full)
                regions = {k:refine(v) for k,v in regions.items()}
                interface = refine(interface)

    def test_connected_patch_mode_rejects_overlap_and_forged_interface(self):
        _, rows = fixture()
        regions, interface = emitted(rows)
        old, new = (2.5,.5,.5), (.75,.75,.25)
        regions[B] = [tuple(new if p==old else p for p in t) for t in regions[B]]
        with self.assertRaises(HumanImportError):
            cert.audit_shared_interface_partition(
                regions, interface, classify_connected_patches=True)
        regions, interface = emitted(rows)
        target = tuple(sorted(interface[0]))
        i = next(i for i,t in enumerate(regions[B]) if tuple(sorted(t))==target)
        regions[B][i] = tuple(reversed(regions[B][i]))
        with self.assertRaisesRegex(HumanImportError, 'opposite winding'):
            cert.audit_shared_interface_partition(
                regions, interface, classify_connected_patches=True)
        with self.assertRaisesRegex(HumanImportError, 'Boolean'):
            cert.audit_shared_interface_partition(regions, interface,
                                                  classify_connected_patches=1)

    def test_missing_or_reversed_interface_copy_is_rejected(self):
        _,rows=fixture();regions,interface=emitted(rows)
        bad=deepcopy(regions);target=tuple(sorted(interface[0]));index=next(i for i,t in enumerate(bad[B]) if tuple(sorted(t))==target)
        bad[B][index]=tuple(reversed(bad[B][index]))
        with self.assertRaisesRegex(HumanImportError,'opposite winding'):cert.audit_shared_interface_partition(bad,interface)
        with self.assertRaises(HumanImportError):cert.audit_shared_interface_partition(regions,[])

    def test_partial_overlap_cannot_be_hidden_by_declaring_an_interface(self):
        _,rows=fixture();regions,interface=emitted(rows)
        # Move a non-interface B corner into A while preserving index closure.
        old=(2.5,.5,.5);new=(.75,.75,.25)
        regions[B]=[tuple(new if p==old else p for p in t) for t in regions[B]]
        with self.assertRaises(HumanImportError):cert.audit_shared_interface_partition(regions,interface)

    def test_distinct_vertices_can_become_collinear_after_float64_conversion(self):
        # Exact positive tetrahedron: conversion loses the tiny transverse
        # coordinate without identifying any pair of its four vertices.
        o=(F(0),F(0),F(0));b=(F(1),F(1),F(0))
        c=(F(2),F(2)+F(1,10**80),F(0));d=(F(0),F(0),F(1))
        av=[o,b,c,d];bv=[o,b,(F(1),F(-1),F(0)),d]
        exact_a=[tuple(av[k] for k in face) for face in FACES]
        exact_b=[tuple(bv[k] for k in reversed(face)) for face in FACES]
        cert._topology(exact_a)
        self.assertEqual(cert._moment_vector(exact_a)[0],F(1,6*10**80))
        self.assertEqual(len({tuple(float(x) for x in p) for p in av}),4)
        convert=lambda triangles:[tuple(tuple(float(x) for x in p) for p in t) for t in triangles]
        regions={A:convert(exact_a),B:convert(exact_b)}
        interface=convert([(o,b,d)])
        with self.assertRaisesRegex(HumanImportError,'degenerate'):
            cert.audit_shared_interface_partition(regions,interface)

    def test_finite_float_requirement_and_duplicate_faces_rejected(self):
        _,rows=fixture();regions,interface=emitted(rows);regions[A].append(regions[A][0])
        with self.assertRaises(HumanImportError):cert.audit_shared_interface_partition(regions,interface)
        regions,interface=emitted(rows);tri=list(regions[A][0]);tri[0]=(float('nan'),0.,0.);regions[A][0]=tuple(tri)
        with self.assertRaisesRegex(HumanImportError,'finite Float64'):cert.audit_shared_interface_partition(regions,interface)


if __name__=='__main__':unittest.main()
