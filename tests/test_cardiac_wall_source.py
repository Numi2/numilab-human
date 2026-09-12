from array import array
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from numilab_human.cardiac_wall_source import (
    CONFIG, CONFIG_SHA256, Mesh, array_bytes, audit_boundary, audit_frames,
    load_config, parse_vtk, prepare_topology, write_immutable,
)
from numilab_human.model import ImportError as HumanImportError


def mesh(points, cells, labels=None):
    return Mesh(array('d', itertools.chain.from_iterable(points)),
                array('I', itertools.chain.from_iterable(cells)),
                array('I', labels or [1]*len(cells)),
                array('d', [1, 0, 0]*len(cells)), array('d', [0, 1, 0]*len(cells)),
                {key: array('d', [-10]*len(points)) for key in ('RHO.dat','PHI.dat','Z.dat','V.dat')}, '')


POINTS = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]


def vtk_bytes():
    lines = ['# vtk DataFile Version 3.0', 'test fixture', 'ASCII',
             'DATASET UNSTRUCTURED_GRID', 'POINTS 4 float',
             '0 0 0', '1000 0 0', '0 1000 0', '0 0 1000',
             'CELL_TYPES 1', '10', 'CELLS 1 5', '4 0 1 2 3',
             'CELL_DATA 1', 'SCALARS ID int 1', 'LOOKUP_TABLE default', '1',
             'VECTORS fibres float', '0.99999 0 0', 'VECTORS sheets float', '0 1 0',
             'POINT_DATA 4']
    for key in ('RHO.dat','PHI.dat','Z.dat','V.dat'):
        lines.extend([f'SCALARS {key} float 1', 'LOOKUP_TABLE default','-10','0','0.5','1'])
    return ('\n'.join(lines)+'\n').encode()


class CardiacWallSourceTests(unittest.TestCase):
    def test_ascii_source_retains_fields_and_only_converts_units(self):
        data = vtk_bytes()
        result = parse_vtk(io.BytesIO(data))
        self.assertEqual(list(result.points), list(itertools.chain.from_iterable(POINTS)))
        self.assertEqual(list(result.cells), [0,1,2,3])
        self.assertEqual(list(result.fibres), [0.99999,0,0])
        self.assertEqual(list(result.uvc['V.dat']), [-10,0,0.5,1])
        self.assertEqual(result.member_sha256, hashlib.sha256(data).hexdigest())

    def test_invalid_vtk_is_rejected_without_silent_repair(self):
        good = vtk_bytes()
        changes = [(b'ASCII',b'BINARY'), (b'POINTS 4 float',b'POINTS 4 double'),
                   (b'1000 0 0',b'nan 0 0'),(b'1000 0 0',b'1e999 0 0'),
                   (b'CELL_TYPES 1\n10',b'CELL_TYPES 1\n12'),
                   (b'4 0 1 2 3',b'4 0 1 2 8'),(b'4 0 1 2 3',b'4 0 1 2 2'),
                   (b'4 0 1 2 3',b'4 -1 1 2 3'),
                   (b'SCALARS ID int 1\nLOOKUP_TABLE default\n1',b'SCALARS ID int 1\nLOOKUP_TABLE default\n25'),
                   (b'VECTORS sheets float',b'VECTORS directions float')]
        for before, after in changes:
            with self.subTest(after=after), self.assertRaises(HumanImportError):
                parse_vtk(io.BytesIO(good.replace(before,after)))
        for data in (good[:-10], good+b'junk\n'):
            with self.assertRaises(HumanImportError): parse_vtk(io.BytesIO(data))

    def test_orientation_permutation_is_reversible_and_geometry_is_conserved(self):
        original = mesh(POINTS,[(1,0,2,3)])
        saved = copy.deepcopy(original)
        report,cells,reversed_ids,boundary,owners = prepare_topology(original)
        self.assertEqual(list(reversed_ids),[0])
        self.assertEqual(list(cells),[0,1,2,3])
        self.assertEqual(original,saved)
        recovered = array('I',cells)
        for cell in reversed_ids: recovered[4*cell],recovered[4*cell+1]=recovered[4*cell+1],recovered[4*cell]
        self.assertEqual(recovered,original.cells)
        self.assertAlmostEqual(report['regional_geometric_volume_m3']['1'],1/6)
        audit,ids = audit_boundary(original,boundary,owners)
        self.assertEqual(len(audit['components']),1)
        self.assertEqual(audit['edge_manifold_orientation_defect_count'],0)
        self.assertAlmostEqual(audit['components'][0]['signed_material_outward_volume_m3'],1/6)
        self.assertIsNone(audit['components'][0]['candidate_chamber'])
        self.assertFalse(audit['components'][0]['native_cavity_admitted'])

    def test_shared_face_is_removed_but_owner_labels_survive(self):
        original = mesh(POINTS+[(0,0,-1)],[(0,1,2,3),(0,2,1,4)],[1,7])
        report,_,_,boundary,owners=prepare_topology(original)
        self.assertEqual(report['interior_face_count'],1)
        self.assertEqual(len(owners),6)
        audit,_=audit_boundary(original,boundary,owners)
        self.assertEqual(audit['components'][0]['numerical_closure_face_count'],3)
        self.assertEqual(audit['components'][0]['owner_label_face_counts'],{'1':3,'7':3})

    def test_duplicate_degenerate_and_overlapping_adjacent_tetrahedra_rejected(self):
        for value in (mesh(POINTS,[(0,1,2,3),(1,0,2,3)]),
                      mesh(POINTS[:3]+[(1,1,0)],[(0,1,2,3)]),
                      mesh(POINTS+[(0,0,2)],[(0,1,2,3),(0,1,2,4)]),
                      mesh(POINTS+[(0,0,1)],[(0,1,2,3)])):
            with self.assertRaises(HumanImportError): prepare_topology(value)

    def test_hollow_wall_inner_surface_has_negative_material_volume(self):
        points=list(itertools.product(range(4),repeat=3))
        ids={point:i for i,point in enumerate(points)}
        cells=[]
        for base in itertools.product(range(3),repeat=3):
            if base==(1,1,1): continue
            for permutation in itertools.permutations(range(3)):
                path=[list(base)]
                for axis in permutation:
                    p=path[-1].copy();p[axis]+=1;path.append(p)
                cells.append([ids[tuple(p)] for p in path])
        original=mesh(points,cells)
        report,_,_,boundary,owners=prepare_topology(original)
        audit,_=audit_boundary(original,boundary,owners)
        self.assertEqual(report['positive_oriented_tetrahedra'],156)
        self.assertEqual(report['regional_geometric_volume_m3']['1'],26)
        self.assertEqual(audit['edge_manifold_orientation_defect_count'],0)
        self.assertEqual(sorted(c['signed_material_outward_volume_m3'] for c in audit['components']),[-1,27])
        cavity=next(c for c in audit['components'] if c['candidate_chamber'])
        self.assertEqual(cavity['candidate_chamber'],'left_ventricle')
        self.assertFalse(cavity['native_cavity_admitted'])

    def test_nonmanifold_touching_solids_are_retained_as_unqualified(self):
        original=mesh(POINTS+[(0,-1,0),(0,0,-1)],[(0,1,2,3),(0,1,4,5)])
        _,_,_,boundary,owners=prepare_topology(original)
        audit,_=audit_boundary(original,boundary,owners)
        self.assertEqual(audit['edge_manifold_orientation_defect_count'],1)
        self.assertFalse(any(c['native_cavity_admitted'] for c in audit['components']))

    def test_frames_are_measured_without_normalization(self):
        original=mesh(POINTS,[(0,1,2,3)])
        original.fibres[0]=0.9
        original.sheets=array('d',[0.9,0,0])
        saved=copy.deepcopy(original)
        report=audit_frames(original)
        self.assertEqual(original,saved)
        self.assertEqual(report['regions']['1']['parallel_frames'],1)
        self.assertFalse(report['coordinate_edits_or_orthonormalization'])

    def test_all_source_semantics_are_hash_pinned(self):
        config,sha=load_config(CONFIG)
        self.assertEqual(sha,CONFIG_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            for path,value in ((['source','metres_per_source_length_unit'],1),
                               (['source','archive','sha256'],'0'*64),(['mesh','points'],4)):
                changed=copy.deepcopy(config);row=changed
                for key in path[:-1]: row=row[key]
                row[path[-1]]=value
                target=Path(tmp)/'config.json';target.write_text(json.dumps(changed))
                with self.assertRaisesRegex(HumanImportError,'configuration hash'): load_config(target)
            target.write_bytes(CONFIG.read_bytes().replace(b'"schema":',b'"schema":"forged","schema":',1))
            with self.assertRaises(HumanImportError):load_config(target)

    def test_buffer_endian_and_immutable_output(self):
        self.assertEqual(array_bytes(array('I',[1,256])),b'\x01\0\0\0\0\x01\0\0')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'buffer';write_immutable(path,b'abc');write_immutable(path,b'abc')
            with self.assertRaises(HumanImportError):write_immutable(path,b'changed')
            link=Path(tmp)/'link';link.symlink_to(path)
            with self.assertRaises(HumanImportError):write_immutable(link,b'abc')


if __name__=='__main__':unittest.main()
