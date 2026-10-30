import importlib.util,json,os,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location("tested_role_adapter",HERE/"tetgen-217-role-adapter.py")
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
CONFIG=json.loads((HERE/"adapter-config.json").read_text())
OWNER_PATH=Path(CONFIG["pins"]["native217_source004"]["path"])
OWNER=adapter.load_module("test_owner_for_adapter",OWNER_PATH,[OWNER_PATH.parent,adapter.P.parent/"authoring-deps"])
HELPER_PATH=Path(CONFIG["pins"]["region_seed_helper_056"]["path"])
helper=adapter.load_module("tested_seed_helper",HELPER_PATH,[])
POINTS=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
FACES=np.asarray([[0,2,1],[0,1,3],[1,2,3],[0,3,2]],dtype=np.int64)
MARKERS=np.ones(4,dtype=np.int32)

def make_fixture(root,role="skin",material=15):
 poly=root/f"{role}.poly"
 region=np.asarray([0.1,0.1,0.1],dtype=np.float64)
 OWNER.write_poly(poly,POINTS,FACES,MARKERS,[],[(region,material)])
 np.savez_compressed(root/f"{role}-shell.npz",points_world_m=POINTS,faces=FACES,face_marker=MARKERS)
 return poly

class AdapterTests(unittest.TestCase):
 def test_frozen_dependency_pins_match(self):
  actual=adapter.verify_pins(CONFIG)
  self.assertEqual(actual["tetgen054_binary"],CONFIG["pins"]["tetgen054_binary"]["sha256"])
 def test_hardcoded_config_hash_fails_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"tampered.json";path.write_bytes((HERE/"adapter-config.json").read_bytes()+b" ")
   with self.assertRaisesRegex(ValueError,"config SHA mismatch"):adapter.load_config(path)
 def test_required_config_policy_cannot_be_disabled(self):
  tampered=dict(CONFIG);tampered["region_seed_completion"]=dict(CONFIG["region_seed_completion"])
  tampered["region_seed_completion"]["preserves_existing_region_rows_literally"]=False
  with self.assertRaisesRegex(ValueError,"required source004 region-seed gate disabled"):
   adapter.verify_pins(tampered)
 def test_default_seed_retry_is_off_and_command_is_explicit_zero(self):
  self.assertFalse(adapter.region_retry_enabled({}))
  self.assertFalse(adapter.region_retry_enabled({adapter.REGION_ENV:"0"}))
  cfg={"required_parent_argv_option":"-pYq1.2/10AS512","zero_improvement_suffix":"O0/0/0"}
  cmd=adapter.compose_tetgen_command(Path("/tmp/tetgen"),[cfg["required_parent_argv_option"],"/tmp/skin.poly"],cfg)
  self.assertEqual(cmd,["/tmp/tetgen","-pYq1.2/10AS512O0/0/0","/tmp/skin.poly"])
 def test_seed_retry_requires_explicit_opt_in(self):
  self.assertTrue(adapter.region_retry_enabled({adapter.REGION_ENV:"1"}))
  with self.assertRaises(ValueError):adapter.region_retry_enabled({adapter.REGION_ENV:"yes"})
 def test_signed_shell_volume_retains_orientation(self):
  self.assertAlmostEqual(adapter.signed_surface_volume(POINTS,FACES),1/6)
  self.assertAlmostEqual(adapter.signed_surface_volume(POINTS,FACES[:,::-1]),-1/6)
 def test_volume_closure_uses_positive_signed_shell_volume(self):
  policy={"source_shell_volume_abs_tolerance_m3":1e-15,"source_shell_volume_relative_tolerance":1e-9}
  self.assertTrue(adapter.volume_closure(1.0,1.0,policy)["pass"])
  with self.assertRaisesRegex(ValueError,"nonpositive signed volume"):adapter.volume_closure(1.0,-1.0,policy)
  with self.assertRaisesRegex(ValueError,"differs from source closed signed shell"):adapter.volume_closure(1.01,1.0,policy)
 def test_default_wrapper_pass_through_does_not_retry(self):
  deps=adapter.verify_pins(CONFIG)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poly=make_fixture(root)
   calls=[]
   def fake_run(cmd,**kwargs):
    calls.append(cmd)
    (root/"skin.1.node").write_text("4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n")
    (root/"skin.1.ele").write_text("1 4 1\n1 1 2 3 4 15\n")
    return adapter.subprocess.CompletedProcess(cmd,0,"stub first pass\n","")
   with patch.object(adapter.subprocess,"run",side_effect=fake_run):
    with patch.dict(os.environ,{adapter.REGION_ENV:"0"},clear=False):result=adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
   self.assertEqual(result,0);self.assertEqual(len(calls),1)
   self.assertEqual(calls[0][1],"-pYq1.2/10AS512O0/0/0")
   receipt=json.loads((root/"tetgen-adapter-skin.json").read_text())
   self.assertEqual(receipt["status"],"FIRST_TETGEN_COMPLETE_REGION_RETRY_DISABLED_NO_PHYSICAL_ADMISSION")
   self.assertEqual(receipt["source_poly_shell_match"]["points_faces_markers_match_shell_npz"],True)
   self.assertIn("DELEGATED_TO_UNCHANGED_OWNER",receipt["source004_final_cell_gate"])
 def test_opted_in_retry_preserves_shell_and_copies_final_tables(self):
  deps=adapter.verify_pins(CONFIG)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poly=make_fixture(root)
   calls=[]
   def fake_run(cmd,**kwargs):
    calls.append(cmd);stem=Path(cmd[-1]).stem;attr=0 if len(calls)==1 else 15
    (root/(stem+".1.node")).write_text("4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n")
    (root/(stem+".1.ele")).write_text(f"1 4 1\n1 1 2 3 4 {attr}\n")
    return adapter.subprocess.CompletedProcess(cmd,0,f"stub pass {len(calls)}\n","")
   with patch.object(adapter.subprocess,"run",side_effect=fake_run):
    with patch.dict(os.environ,{adapter.REGION_ENV:"1"},clear=False):result=adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
   self.assertEqual(result,0);self.assertEqual(len(calls),2)
   self.assertEqual(calls[1][1],"-pYq1.2/10AS512O0/0/0");self.assertEqual(Path(calls[1][-1]).name,"skin-seed-completed.poly")
   receipt=json.loads((root/"tetgen-adapter-skin.json").read_text())
   self.assertEqual(receipt["status"],"ONE_REGION_SEED_RERUN_COMPLETE_NO_PHYSICAL_ADMISSION")
   self.assertEqual(receipt["unowned_component_count"],1)
   self.assertTrue(all(receipt["final_expected_tables_match_second_pass"].values()))
   self.assertEqual(receipt["volume_closure"]["shell_signed_volume_m3"],1/6)
   self.assertTrue((root/"tetgen-region-seed-completion-skin/first-skin.1.node").is_file())
 def test_volume_closure_failure_keeps_first_pass_outputs_and_rejects_retry(self):
  deps=adapter.verify_pins(CONFIG)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poly=make_fixture(root);calls=[]
   first_node="4 3 0 0\n1 0 0 0\n2 2 0 0\n3 0 1 0\n4 0 0 1\n"
   first_ele="1 4 1\n1 1 2 3 4 0\n"
   def fake_run(cmd,**kwargs):
    calls.append(cmd);(root/"skin.1.node").write_text(first_node);(root/"skin.1.ele").write_text(first_ele)
    return adapter.subprocess.CompletedProcess(cmd,0,"stub pass\n","")
   with patch.object(adapter.subprocess,"run",side_effect=fake_run):
    with patch.dict(os.environ,{adapter.REGION_ENV:"1"},clear=False):
     with self.assertRaisesRegex(ValueError,"differs from source closed signed shell"):
      adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
   self.assertEqual(len(calls),1)
   snap=root/"tetgen-region-seed-completion-skin/first-skin.1.ele"
   self.assertEqual(snap.read_text(),first_ele)
   receipt=json.loads((root/"tetgen-adapter-skin.json").read_text())
   self.assertEqual(receipt["status"],"REGION_SEED_RETRY_REJECTED_FIRST_OUTPUTS_RETAINED_NO_PHYSICAL_ADMISSION")
   self.assertEqual(receipt["first_output_snapshots"]["first-skin.1.ele"],adapter.sha256(snap))
 def test_poly_point_shell_mismatch_fails_before_tetgen(self):
  deps=adapter.verify_pins(CONFIG)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poly=make_fixture(root)
   np.savez_compressed(root/"skin-shell.npz",points_world_m=POINTS+np.asarray([0.01,0,0],dtype=np.float32),faces=FACES,face_marker=MARKERS)
   with patch.object(adapter.subprocess,"run") as run:
    with patch.dict(os.environ,{adapter.REGION_ENV:"0"},clear=False):
     with self.assertRaisesRegex(ValueError,"poly points differ from the loaded source shell"):adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
   run.assert_not_called()
   receipt=json.loads((root/"tetgen-adapter-skin.json").read_text())
   self.assertEqual(receipt["status"],"SOURCE_PLC_PREFLIGHT_REJECTED_NO_TETGEN_RUN")
 def test_poly_face_or_marker_mismatch_fails_closed(self):
  deps=adapter.verify_pins(CONFIG)
  for changed,reason in [(FACES[:,::-1].copy(),"poly facets differ"),(MARKERS+1,"poly facet markers differ")]:
   with self.subTest(reason=reason),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);poly=make_fixture(root)
    if reason=="poly facets differ":np.savez_compressed(root/"skin-shell.npz",points_world_m=POINTS,faces=changed,face_marker=MARKERS)
    else:np.savez_compressed(root/"skin-shell.npz",points_world_m=POINTS,faces=FACES,face_marker=changed)
    with patch.object(adapter.subprocess,"run") as run:
     with patch.dict(os.environ,{adapter.REGION_ENV:"0"},clear=False):
      with self.assertRaisesRegex(ValueError,reason):adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
    run.assert_not_called()
 def test_region_material_must_match_requested_role(self):
  deps=adapter.verify_pins(CONFIG)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poly=make_fixture(root,role="fat",material=15)
   with patch.object(adapter.subprocess,"run") as run:
    with patch.dict(os.environ,{adapter.REGION_ENV:"0"},clear=False):
     with self.assertRaisesRegex(ValueError,"source poly region material differs from requested role"):adapter.run_wrapper(["-pYq1.2/10AS512",str(poly)],CONFIG,deps)
   run.assert_not_called()
 def test_disconnected_unowned_components_get_one_seed_each(self):
  pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[3,0,0],[4,0,0],[3,1,0],[3,0,1]],dtype=np.float64)
  tets=np.asarray([[0,1,2,3],[4,5,6,7]],dtype=np.int64);attrs=np.asarray([0,0],dtype=np.int64)
  seeds=helper.seeds_from_unowned(pts,tets,attrs,15,lambda p:True)
  self.assertEqual(len(seeds),2);self.assertEqual([len(x["unowned_cell_rows"]) for x in seeds],[1,1])
  self.assertEqual([x["material"] for x in seeds],[15,15])
 def test_every_unowned_component_cell_must_be_inside_shell(self):
  pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=np.float64)
  tets=np.asarray([[0,1,2,3],[0,2,1,4]],dtype=np.int64);attrs=np.asarray([0,0],dtype=np.int64)
  visited=[]
  def inside(point):visited.append(point.copy());return point[2]>0
  with self.assertRaisesRegex(ValueError,"outside existing material shell"):
   helper.seeds_from_unowned(pts,tets,attrs,15,inside)
  self.assertEqual(len(visited),2)
 def test_forbidden_material_attribute_rejected(self):
  with self.assertRaisesRegex(ValueError,"unexpected material attribute"):
   helper.seeds_from_unowned(POINTS,np.asarray([[0,1,2,3]]),np.asarray([16]),15,lambda p:True)
 def test_collapsed_component_rejected(self):
  pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],dtype=np.float64)
  with self.assertRaisesRegex(ValueError,"collapsed F32 cell"):
   helper.seeds_from_unowned(pts,np.asarray([[0,1,2,3]]),np.asarray([0]),15,lambda p:True)
 def test_mismatched_dependency_pin_fails_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   paths={k:Path(tmp)/(k+".bin") for k in CONFIG["pins"]}
   for k,path in paths.items():path.write_bytes(k.encode())
   paths["tetgen054_binary"].write_bytes(b"wrong")
   with self.assertRaisesRegex(ValueError,"SHA mismatch"):adapter.verify_pins(CONFIG,paths)
 def test_completed_poly_preserves_shell_and_original_region_rows(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=make_fixture(root);dst=root/"skin-seed-completed.poly"
   seed=[{"point_world_m":[0.2,0.2,0.2],"material":15}]
   helper.complete_poly(src,dst,seed,OWNER)
   section_hash=adapter.assert_same_shell_and_existing_regions(src,dst,seed,15)
   self.assertTrue(section_hash)
   parsed=adapter.poly_sections(dst)
   self.assertEqual([r["material"] for r in parsed["regions"]],[15,15])
if __name__=="__main__":unittest.main()
