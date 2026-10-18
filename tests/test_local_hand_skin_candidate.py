"""Regression gates for local skin repair receipts, using tiny closed surfaces."""
import importlib.util,struct,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location("local_skin",Path(__file__).resolve().parents[1]/"tools/compose_resting_local_hand_skin_candidate.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def payload(*,move=None,binding=False,open_surface=False,bad_weight=False):
    binding_rows=np.zeros((86,9),dtype="<u4")
    binding_rows[:,0]=np.arange(86);binding_rows[:,7]=np.float32(1).view("<u4");binding_rows[:,8]=np.float32(1).view("<u4")
    if binding:binding_rows[2,1]=np.float32(.01).view("<u4")
    vertices=np.zeros((4,14),dtype="<f4");vertices[:,:3]=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]];vertices[:,5]=1;vertices[:,10]=1
    if move is not None:vertices[move,0]+=.1
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],dtype="<u4")
    if open_surface:faces=faces[:-1]
    weights=np.zeros((4,86),dtype="<f4");weights[:,0]=1
    if bad_weight:weights[0,0]=-.1;weights[0,1]=1.1
    return struct.pack("<8s5I32s",b"NHSKIN1\0",5,86,4,faces.size,7,b"s"*32)+binding_rows.tobytes()+vertices.tobytes()+faces.tobytes()+weights.tobytes()

class LocalSkinRepairTests(unittest.TestCase):
    def test_local_coordinate_edit_keeps_closed_surface(self):
        _,_,r=m.validate_local_rows(payload(),payload(move=0),[0])
        self.assertEqual(r["boundary_edges"],0)
        self.assertEqual(r["unchanged_outside_vertex_count"],3)
    def test_unrelated_vertex_edit_rejected(self):
        with self.assertRaisesRegex(ValueError,"unrelated"):m.validate_local_rows(payload(),payload(move=2),[0])
    def test_changed_atlas_frame_rejected(self):
        with self.assertRaisesRegex(ValueError,"atlas"):m.validate_local_rows(payload(),payload(binding=True),[0])
    def test_new_skin_hole_rejected(self):
        with self.assertRaisesRegex(ValueError,"apertures"):m.validate_local_rows(payload(),payload(open_surface=True),[0])
    def test_negative_influence_rejected_even_when_sum_is_one(self):
        with self.assertRaisesRegex(ValueError,"partition"):m.validate_local_rows(payload(),payload(bad_weight=True),[0])
    def test_duplicate_edit_ancestry_rejected(self):
        with self.assertRaisesRegex(ValueError,"distinct"):m.validate_local_rows(payload(),payload(),[0,0])

if __name__=="__main__":unittest.main()
