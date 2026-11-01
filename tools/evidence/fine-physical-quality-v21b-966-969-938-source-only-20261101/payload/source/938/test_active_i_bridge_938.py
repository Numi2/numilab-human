#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import unittest

HERE=Path(__file__).resolve().parent
OWNER_PATH=HERE/"current_fine_217_owner_938.py"
BRIDGE_PATH=HERE/"current_fine_volume_to_217_938.py"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

owner=load("owner_active_i_938_tests",OWNER_PATH)
bridge=load("bridge_active_i_938_tests",BRIDGE_PATH)

class ActiveInterfaceMapTests(unittest.TestCase):
    def setUp(self):
        self.points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[9,9,9]],np.float32)
        self.faces=np.asarray([[0,1,2]],np.int64)
        self.inner={"points":self.points,"faces":self.faces}
        self.skin=self.shell(np.asarray([0,1,2,-1],np.int64))
        self.fat=self.shell(np.asarray([1,2,0,-1],np.int64))

    def shell(self,rows):
        # Role-local shell row numbers may differ, while the mapped Float32
        # source coordinates must be identical for all active source-I rows.
        rows=np.asarray(rows,np.int64)
        valid=np.flatnonzero(rows>=0)
        order=np.argsort(rows[valid],kind="stable")
        shell_points=self.points[valid[order]].copy()
        shell_row_for_source=np.full(len(rows),-1,np.int64)
        for new,src in enumerate(valid[order]):shell_row_for_source[src]=new
        return {"points_world_m":shell_points,
          "registered_I_source_vertex_rows":np.arange(len(rows),dtype=np.int64),
          "registered_I_shell_node_rows":shell_row_for_source,
          "registered_I_source_faces":self.faces.copy()}

    def test_exact_bridge_and_owner_pins_load(self):
        loaded=bridge.import_owner()
        self.assertEqual(Path(bridge.OWNER_PATH).resolve(),OWNER_PATH.resolve())
        self.assertEqual(bridge.OWNER_SHA,bridge.sha(OWNER_PATH))
        self.assertEqual(loaded.__file__,str(OWNER_PATH))
        self.assertEqual(bridge.OWNER_BASE_SHA,"d7f411c18f91905c6e1d91ab75592ec5782a8852892ac8a1fdb566ee2206df09")

    def test_only_unreferenced_historical_row_can_be_minus_one(self):
        bridge.validate_shared_i(None,self.skin,self.fat,self.inner)
        skin_point_map=np.asarray([10,11,12],np.int64)
        fat_point_map=np.asarray([20,21,22],np.int64)
        expected={0:10,1:11,2:12}
        a=owner.registered_i_source_node_map(self.skin,skin_point_map,"skin",np.asarray([0,1,2]))
        b=owner.registered_i_source_node_map(self.fat,fat_point_map,"fat",np.asarray([0,1,2]))
        self.assertEqual(a,expected)
        self.assertEqual(set(a),set(b))
        self.assertEqual(b,{0:21,1:22,2:20})

    def test_referenced_minus_one_rejects_in_both_layers(self):
        bad_skin=self.shell(np.asarray([0,-1,2,-1],np.int64))
        with self.assertRaisesRegex(ValueError,"face-referenced"):
            bridge.validate_shared_i(None,bad_skin,self.fat,self.inner)
        with self.assertRaisesRegex(ValueError,"face-referenced"):
            owner.registered_i_source_node_map(bad_skin,np.asarray([10,11,12]),"skin",np.asarray([0,1,2]))

    def test_arbitrary_negative_sentinel_rejected_even_when_unreferenced(self):
        bad_fat=self.shell(np.asarray([1,2,0,-1],np.int64))
        bad_fat["registered_I_shell_node_rows"][3]=-2
        with self.assertRaisesRegex(ValueError,"negative sentinel"):
            bridge.validate_shared_i(None,self.skin,bad_fat,self.inner)
        with self.assertRaisesRegex(ValueError,"negative sentinel"):
            owner.registered_i_source_node_map(bad_fat,np.asarray([20,21,22]),"fat",np.asarray([0,1,2]))

    def test_active_coordinate_mismatch_rejected(self):
        bad=self.shell(np.asarray([0,1,2,-1],np.int64))
        bad["points_world_m"][1,0]=np.float32(1.0001)
        with self.assertRaisesRegex(ValueError,"exact Float32 coordinate bytes"):
            bridge.validate_shared_i(None,bad,self.fat,self.inner)

    def test_float32_signed_zero_is_not_an_exact_coordinate_match(self):
        bad=self.shell(np.asarray([0,1,2,-1],np.int64))
        bad["points_world_m"][0,0]=np.float32(-0.0)
        with self.assertRaisesRegex(ValueError,"coordinate bytes"):
            bridge.validate_shared_i(None,bad,self.fat,self.inner)

    def test_full_source_identity_and_face_rows_remain_required(self):
        bad=self.shell(np.asarray([0,1,2,-1],np.int64))
        bad["registered_I_source_vertex_rows"][3]=30
        with self.assertRaisesRegex(ValueError,"source vertex identity"):
            bridge.validate_shared_i(None,bad,self.fat,self.inner)
        bad=self.shell(np.asarray([0,1,2,-1],np.int64))
        bad["registered_I_source_faces"]=np.asarray([[0,1,3]],np.int64)
        with self.assertRaisesRegex(ValueError,"source face identity"):
            bridge.validate_shared_i(None,bad,self.fat,self.inner)

if __name__=="__main__":unittest.main(verbosity=2)
