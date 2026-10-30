from pathlib import Path
import importlib.util,unittest,tempfile,numpy as np
s=importlib.util.spec_from_file_location("microedge",Path(__file__).with_name("prepare_microedge_boundary_706.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Contract(unittest.TestCase):
 def setUp(self):
  self.parent={"points_world_m":np.asarray([[0,0,0],[1e-7,0,0],[0,1e-3,0],[1e-3,0,0]],np.float32),"faces":np.asarray([[0,1,2],[1,3,2]])}
  self.new={"points_world_m":self.parent["points_world_m"][[0,2,3]],"faces":np.asarray([[0,2,1]]),
   "microedge_parent_candidate_sha256":np.asarray(["a"*64]),"microedge_input_point_to_output_row":np.asarray([0,0,1,2]),
   "microedge_parent_input_point_offsets":np.asarray([0,2,3,4]),"microedge_parent_input_point_rows":np.asarray([0,1,2,3]),
   "microedge_max_displacement_m":np.asarray([1e-6]),"microedge_parent_input_face_offsets":np.asarray([0,2]),
   "microedge_parent_input_face_rows":np.asarray([0,1])}
 def test_valid_contraction_and_unmerged_target(self):
  mp,off,stats=m.validate_contraction(self.parent,self.new,"a"*64)
  self.assertEqual(m.preserved_target(self.parent,self.new,mp,off,self.parent["points_world_m"][2]),1)
  self.assertLess(stats["maximum_original_point_displacement_m"],1e-6)
 def test_merged_protected_target_rejected(self):
  mp,off,_=m.validate_contraction(self.parent,self.new,"a"*64)
  with self.assertRaisesRegex(ValueError,"merged"):m.preserved_target(self.parent,self.new,mp,off,self.parent["points_world_m"][0])
 def test_stale_parent_rejected(self):
  with self.assertRaisesRegex(ValueError,"SHA"):m.validate_contraction(self.parent,self.new,"b"*64)
 def test_omitted_face_parent_rejected(self):
  self.new["microedge_parent_input_face_rows"]=np.asarray([0]);self.new["microedge_parent_input_face_offsets"]=np.asarray([0,1])
  with self.assertRaisesRegex(ValueError,"face ancestry"):m.validate_contraction(self.parent,self.new,"a"*64)
 def test_nonendpoint_coordinate_rejected(self):
  self.new["points_world_m"][0,0]=5e-8
  with self.assertRaisesRegex(ValueError,"endpoint"):m.validate_contraction(self.parent,self.new,"a"*64)
 def test_duplicate_point_parent_rejected(self):
  self.new["microedge_parent_input_point_rows"]=np.asarray([0,0,2,3])
  with self.assertRaisesRegex(ValueError,"partition"):m.validate_contraction(self.parent,self.new,"a"*64)
 def test_collar_geometry_preserved_after_reindex(self):
  points=np.vstack((self.parent["points_world_m"],np.arange(270,dtype=np.float32).reshape(90,3)*.01))
  c={"points_world_m":points,"donor_O_to_candidate_vertex":np.asarray([0,2]),"skin_sidewall_faces":np.asarray([[0,4,5]]),"fat_sidewall_faces":np.asarray([[4,49,50]])}
  out=m.reindex_collar(c,self.new,self.parent,np.asarray([0,0,1,2]))
  self.assertEqual(out["skin_sidewall_faces"].tolist(),[[0,3,4]])
  self.assertTrue(np.array_equal(points[c["skin_sidewall_faces"]],out["points_world_m"][out["skin_sidewall_faces"]]))
 def test_wall_parent_pin_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"input.csv";p.write_text("original")
   keys=("outer","inner","deep","collar","wall_csv","protected_targets","canonical_d_proof","wall_parent_cells","wall_parent_nodes")
   pins={k:{"path":str(p),"sha256":m.sha(p)} for k in keys}
   m.validate_parent_plan_inputs(pins)
   bad=Path(td)/"changed.csv";bad.write_text("changed")
   for key in ("wall_parent_cells","wall_parent_nodes"):
    trial={k:dict(v) for k,v in pins.items()};trial[key]["path"]=str(bad)
    with self.assertRaisesRegex(ValueError,key):m.validate_parent_plan_inputs(trial)
if __name__=="__main__":unittest.main()
