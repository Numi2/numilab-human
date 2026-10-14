import unittest
import json,hashlib,importlib.util,io,contextlib
import tempfile
from pathlib import Path
import numpy as np
from numilab_human.resting_anatomy_interface_patch import (
    HEADER, encode_payload, parse_payload, remove_isolated_opposed_fragments,
)

class SourceFragmentRepairTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0,0,0,0,0,1], [1,0,0,0,0,1], [0,1,0,0,0,1],
                           [3,0,0,0,0,1], [4,0,0,0,0,1], [3,1,0,0,0,1]], dtype=np.float32)
        self.f = np.array([[0,1,2],[0,2,1],[3,4,5]],dtype=np.int64)
    def test_removes_both_zero_volume_facets_and_preserves_source_slots(self):
        before=self.v.tobytes()
        f,r=remove_isolated_opposed_fragments(self.v,self.f)
        np.testing.assert_array_equal(f,self.f[2:])
        self.assertEqual(r['removed_source_face_ids'],[0,1])
        self.assertEqual(r['retained_source_face_index_ranges'],[[2,3]])
        self.assertEqual(r['exact_signed_volume_change_m3'],'0')
        self.assertEqual(self.v.tobytes(),before)
    def test_exact_coordinate_seams_and_cyclic_winding_are_supported(self):
        v=np.concatenate([self.v,self.v[:3]])
        f=self.f.copy();f[1]=[8,7,6]
        out,r=remove_isolated_opposed_fragments(v,f)
        self.assertEqual(r['removed_source_face_ids'],[0,1])
        np.testing.assert_array_equal(out,self.f[2:])
    def test_same_winding_is_rejected(self):
        f=self.f.copy();f[1]=[1,2,0]
        with self.assertRaisesRegex(ValueError,'opposite winding'):
            remove_isolated_opposed_fragments(self.v,f)
    def test_touching_vertex_or_edge_is_rejected(self):
        for tri in ([1,4,5],[0,1,5]):
            f=self.f.copy();f[2]=tri
            with self.subTest(tri=tri), self.assertRaisesRegex(ValueError,'touches'):
                remove_isolated_opposed_fragments(self.v,f)
    def test_larger_duplicate_group_and_degenerate_facet_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'exactly two'):
            remove_isolated_opposed_fragments(self.v,np.concatenate([self.f,self.f[:1]]))
        f=self.f.copy();f[0]=[0,0,2]
        with self.assertRaisesRegex(ValueError,'degenerate'):
            remove_isolated_opposed_fragments(self.v,f)
    def test_collinear_duplicate_pair_is_rejected(self):
        v=self.v.copy();v[2,:3]=[2,0,0]
        with self.assertRaisesRegex(ValueError,'collinear'):
            remove_isolated_opposed_fragments(v,self.f)
    def test_float64_is_not_silently_rounded(self):
        with self.assertRaisesRegex(ValueError,'Float32'):
            remove_isolated_opposed_fragments(self.v.astype(float),self.f)
    def test_real_preparation_updates_null_provenance_without_changing_mass_or_owners(self):
        path=Path(__file__).resolve().parents[1]/'tools/repair_intercostal_source_fragments.py'
        spec=importlib.util.spec_from_file_location('fragment_tool',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        h=(b'NHANAT1\0',5,3,18,27,123,bytes(32))
        rows={sid:{'body_index':20,'layer':1,'flags':0,'vertices6':self.v,'faces':self.f} for sid in (313,315,317)}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);payload=root/'source.nhanatomy';payload.write_bytes(encode_payload(h,rows))
            digest=hashlib.sha256(payload.read_bytes()).hexdigest()
            receipt={'payload':{'sha256':digest},
                     'functional_bindings':{'anatomy_payload_sha256':digest,'owner':'retained'},
                     'provenance':{'source_id_map':{str(sid):{'repair':None,'name':str(sid),'source_sha256':'immutable'} for sid in rows}},
                     'qualification':'retained native scene incomplete','mass_geometry_accounting':{'physical_mass_kg':72,'passive_extra_mass_kg':0}}
            rp=root/'receipt.json';rp.write_text(json.dumps(receipt))
            with contextlib.redirect_stdout(io.StringIO()):mod.repair(payload,rp,None,root/'out')
            out=json.loads((root/'out/resting-anatomy-receipt.json').read_text())
            self.assertEqual(out['qualification'],receipt['qualification'])
            self.assertEqual(out['mass_geometry_accounting'],receipt['mass_geometry_accounting'])
            self.assertEqual(out['functional_bindings']['owner'],'retained')
            for sid in rows:
                self.assertEqual(out['provenance']['source_id_map'][str(sid)]['source_sha256'],'immutable')
                self.assertEqual(out['provenance']['source_id_map'][str(sid)]['repair']['isolated_opposed_fragments']['removed_source_face_ids'],[0,1])
            self.assertTrue((root/'out/tool-source.py').exists())
            self.assertTrue((root/'out/codec-source.py').exists())
    def test_existing_payload_roundtrip_preserves_order_owners_and_vertex_bytes(self):
        h=(b'NHANAT1\0',5,2,12,18,123,bytes(32))
        rows={317:{'body_index':20,'layer':1,'flags':0,'vertices6':self.v,'faces':self.f},
              313:{'body_index':22,'layer':4,'flags':7,'vertices6':self.v,'faces':self.f}}
        payload=encode_payload(h,rows)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.nhanatomy';path.write_bytes(payload)
            hh,rr=parse_payload(path)
        self.assertEqual(list(rr),[317,313]);self.assertEqual(hh,h)
        self.assertEqual(encode_payload(hh,rr),payload)
        rr[317]['faces'],_=remove_isolated_opposed_fragments(rr[317]['vertices6'],rr[317]['faces'])
        repaired=encode_payload(hh,rr)
        self.assertEqual(HEADER.unpack_from(repaired)[3],12)
        self.assertEqual(HEADER.unpack_from(repaired)[4],12)

if __name__=='__main__':unittest.main()
