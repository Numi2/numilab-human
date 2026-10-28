#!/usr/bin/env python3
import unittest
import numpy as np
from prepare_conditioned_skin_collar import prepare
class Packet(dict):
    @property
    def files(self):return list(self)
class CollarTests(unittest.TestCase):
    def setUp(self):
        o=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],np.float32)
        cp=np.vstack([o,o[:4]+[0,0,-.1],o[:4]+[0,0,-.2]]).astype(np.float32)
        skin=[];fat=[]
        for i in range(4):
            j=(i+1)%4;skin.extend([[i,j,5+j],[i,5+j,5+i]]);fat.extend([[5+i,5+j,9+j],[5+i,9+j,9+i]])
        self.c=Packet(points_world_m=cp,outer_faces=np.array([[0,1,2],[0,2,3]]),
          skin_sidewall_faces=np.array(skin),fat_sidewall_faces=np.array(fat),donor_I_parent_node_rows=np.arange(4)+5,
          donor_D_parent_node_rows=np.arange(4)+9,donor_O_parent_node_rows=np.arange(4),donor_O_to_candidate_vertex=np.arange(4))
        self.o=Packet(points_world_m=o,source_O_faces=np.array([[0,1,3],[1,2,3]]),
          candidate_vertex_to_O033_row=np.arange(5),O033_to_candidate_vertex_row=np.array([0,1,2,3,2]),
          face_ancestry_offsets=np.array([0,2,4]),face_ancestry_parent_face_rows=np.array([0,1,0,1]),
          face_ancestry_source_original_ids=np.array([1,149,1,149]),face_ancestry_source_face_rows=np.array([8,9,8,9]))
        self.rows=[dict(zip(["face","v0","v1","v2","O033_source_face_parent_row","baseline806_source_face_row_exact"],v)) for v in ([0,0,1,3,0,-1],[1,1,2,3,1,-1])]
    def test_compaction_preserves_wall_and_mixed_ancestry(self):
        outer,c,r=prepare(self.o,self.c,self.rows)
        self.assertEqual(r["current_O_point_rows"],4);self.assertEqual(r["source_original_ids"],[1,149]);self.assertFalse(r["admitted"])
        for key in ("skin_sidewall_faces","fat_sidewall_faces"):
            np.testing.assert_array_equal(self.c["points_world_m"][self.c[key]],c["points_world_m"][c[key]])
        np.testing.assert_array_equal(c["face_ancestry_source_original_ids"],[1,149,1,149])
        np.testing.assert_array_equal(outer["O033_to_candidate_vertex_row"],[0,1,2,3,2])
    def test_whole_boundary_reversal_is_explicit(self):
        self.o["source_O_faces"]=self.o["source_O_faces"][:,[0,2,1]]
        for row,f in zip(self.rows,self.o["source_O_faces"]):
            for c,v in enumerate(f):row[f"v{c}"]=v
        self.assertTrue(prepare(self.o,self.c,self.rows)[2]["O_boundary_globally_reversed"])
    def test_partial_orientation_change_rejected(self):
        self.o["source_O_faces"][0]=self.o["source_O_faces"][0,[0,2,1]]
        with self.assertRaisesRegex(ValueError,"part of"):prepare(self.o,self.c,self.rows)
    def test_changed_original_point_rejected(self):
        self.o["points_world_m"]=self.o["points_world_m"].copy();self.o["points_world_m"][0,0]=.001
        with self.assertRaisesRegex(ValueError,"exact original point"):prepare(self.o,self.c,self.rows)
    def test_stale_face_map_rejected(self):
        self.rows[1]["v0"]=0
        with self.assertRaisesRegex(ValueError,"exact current compact"):prepare(self.o,self.c,self.rows)
    def test_truncated_typed_ancestry_rejected(self):
        self.o["face_ancestry_source_original_ids"]=np.array([1])
        with self.assertRaisesRegex(ValueError,"CSR malformed"):prepare(self.o,self.c,self.rows)
if __name__=="__main__":unittest.main()
