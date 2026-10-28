#!/usr/bin/env python3
import copy
import unittest
import numpy as np
from validate_connected_skin_local_repair import validate_local_repair

PIN="a"*64

def reference():
    p=np.array([[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0],[.005,.005,0]],np.float32)
    f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],np.int64)
    return dict(points_world_m=p,source_O_faces=f,candidate_vertex_to_O033_row=np.arange(5),
        O033_to_candidate_vertex_row=np.arange(5),face_ancestry_offsets=np.arange(5),
        face_ancestry_parent_face_rows=np.arange(4),face_ancestry_source_original_ids=np.ones(4,int),
        face_ancestry_source_face_rows=np.arange(4))

def repair(r):
    x=copy.deepcopy(r);x["local_repair_parent_sha256"]=np.array(PIN)
    x["local_repair_allowed_vertex_rows"]=np.array([4],np.int64);x["points_world_m"][4,2]=3e-6
    return x

class LocalRepair(unittest.TestCase):
    def test_bounded_interior(self):
        r=reference();q=repair(r);v=validate_local_repair(r,q,PIN)
        self.assertEqual(v["moved_vertices"],1);self.assertFalse(v["admitted"])
    def test_wrong_pin(self):
        r=reference();q=repair(r)
        with self.assertRaisesRegex(ValueError,"hash"):validate_local_repair(r,q,"b"*64)
    def test_outside_patch(self):
        r=reference();q=repair(r);q["points_world_m"][0,2]=1e-6
        with self.assertRaisesRegex(ValueError,"outside"):validate_local_repair(r,q,PIN)
    def test_oversize(self):
        r=reference();q=repair(r);q["points_world_m"][4,2]=51e-6
        with self.assertRaisesRegex(ValueError,"bound"):validate_local_repair(r,q,PIN)
    def test_boundary(self):
        r=reference();q=repair(r);q["local_repair_allowed_vertex_rows"]=np.arange(5);q["points_world_m"][0,2]=1e-6
        with self.assertRaisesRegex(ValueError,"boundary coordinates"):validate_local_repair(r,q,PIN)
    def test_identity(self):
        r=reference();q=repair(r);q["candidate_vertex_to_O033_row"][0]=3
        with self.assertRaisesRegex(ValueError,"identity"):validate_local_repair(r,q,PIN)
    def test_unchanged_face_ancestry(self):
        r=reference();q=repair(r);q["face_ancestry_source_face_rows"][0]=10
        with self.assertRaisesRegex(ValueError,"unaffected"):validate_local_repair(r,q,PIN)
    def test_nonfinite(self):
        r=reference();q=repair(r);q["points_world_m"][4,2]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):validate_local_repair(r,q,PIN)
    def test_topology_lineage(self):
        r=reference();r["source_O_faces"]=np.array([[0,1,2],[0,2,3]])
        r["face_ancestry_offsets"]=np.arange(3)
        for k in ("face_ancestry_parent_face_rows","face_ancestry_source_original_ids","face_ancestry_source_face_rows"):r[k]=r[k][:2]
        q=copy.deepcopy(r);q["source_O_faces"]=np.array([[0,1,3],[1,2,3]])
        q["local_repair_allowed_vertex_rows"]=np.arange(4);q["local_repair_parent_sha256"]=np.array(PIN)
        q["face_ancestry_offsets"]=np.array([0,2,4])
        for k in ("face_ancestry_parent_face_rows","face_ancestry_source_original_ids","face_ancestry_source_face_rows"):q[k]=np.tile(r[k],2)
        self.assertEqual(validate_local_repair(r,q,PIN)["topology_changed_faces"],2)
        q["face_ancestry_parent_face_rows"][:]=0
        with self.assertRaisesRegex(ValueError,"ancestry"):validate_local_repair(r,q,PIN)

if __name__=="__main__":unittest.main()
