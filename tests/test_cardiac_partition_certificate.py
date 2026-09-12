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


def check(sources,rows):
    return cert.certify_partition(sources,rows,expected_geometry_sha256={s:cert.source_geometry_sha256(v) for s,v in sources.items()})


def decode(value):return F(int(value[0],16),int(value[1],16))


def emitted(rows,priority=A):
    p=cert.build_triangle_sets(rows)['partitions'][priority+'_priority']
    pool={v:tuple(float(x) for x in v) for row in rows for v in row['vertices']}
    convert=lambda tris:[tuple(pool[v] for v in t) for t in tris]
    return {s:convert(t) for s,t in p['regions'].items()},convert(p['shared_interface'])


class ExactCoverageTests(unittest.TestCase):
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
    def test_both_priorities_admit_shared_interface_with_disjoint_interiors(self):
        _,rows=fixture()
        for priority in (A,B):
            regions,interface=emitted(rows,priority);proof=cert.audit_shared_interface_partition(regions,interface)
            self.assertTrue(proof['interiors_disjoint'])
            self.assertTrue(proof['shared_interface_opposite_chain'])
            self.assertEqual(decode(proof['union_moments']['volume_m3']),F(127,48))
            self.assertTrue(all(v['outside_noninterface_face_count']>0 for v in proof['per_region'].values()))

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
