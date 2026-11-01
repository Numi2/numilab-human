from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("shell1022",HERE/"build_current_shell_1022.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
owner=runner.import_pinned(runner.OWNER_PATH,runner.OWNER_SHA,"test_shell_owner_938v2")
owner.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
bridge=runner.import_pinned(runner.BRIDGE_PATH,runner.BRIDGE_SHA,"test_shell_bridge_938v2")
validator_owner=bridge.import_owner()

def surface(points, faces, offset, layer):
    faces=np.asarray(faces,np.int64)
    return {
      "path":f"synthetic-{layer}","points":np.asarray(points,np.float32),"faces":faces,
      "source_rows":np.array([offset],np.int64),"exact_source_rows":np.array([offset+10],np.int64),
      "source_original_ids":np.array([offset+20],np.int64),"source_layers":np.array([layer],np.int8),
      "source_bodies":np.array([layer+10],np.int32),"source_bones":np.array([layer+20],np.int32),
      "source_run_original_ids":np.array([offset+30],np.int64),"source_face_ids":np.array([offset+40],np.int64),
      "ancestry":[[offset+10]],"ancestry_original_ids":[[offset+20]],
      "ancestry_parent_rows":[[offset+50]],"ancestry_kind":np.array([6],np.int8),
      "raw_npz_keys":["face_ancestry_offsets","face_ancestry_source_face_rows",
        "face_ancestry_source_original_ids","face_ancestry_parent_face_rows"]}

def make_case(current_faces=None):
    oold=np.array([[0,0,2],[1,0,2],[0,1,2],[4,4,2]],np.float32)
    o=np.array([[0,0,2],[1,0,2],[0,1,2]],np.float32)
    i=np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32)
    d=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    oldfaces=np.array([[0,1,2]],np.int64)
    curfaces=np.array([[0,2,1]],np.int64) if current_faces is None else np.asarray(current_faces,np.int64)
    # Collar point order is historical O block, I donor ring, D donor ring.
    points=np.vstack((oold,i,d))
    skin=[];fat=[]
    for a,b in ((0,1),(1,2),(2,0)):
        ia,ib=4+a,4+b
        da,db=7+a,7+b
        skin.extend(([a,ia,ib],[a,ib,b]))
        fat.extend(([ia,da,db],[ia,db,ib]))
    collar=runner.NPZView({
      "points_world_m":points,"outer_faces":oldfaces,
      "donor_O_to_candidate_vertex":np.array([0,1,2],np.uint32),
      "donor_O_parent_node_rows":np.array([100,101,102],np.int64),
      "donor_I_parent_node_rows":np.array([200,201,202],np.int64),
      "donor_D_parent_node_rows":np.array([300,301,302],np.int64),
      "skin_sidewall_faces":np.asarray(skin,np.int64),
      "fat_sidewall_faces":np.asarray(fat,np.int64),
      "exact_candidate_face_source_rows":np.array([44],np.int64),
      "skin_sidewall_parent_tet_rows":np.arange(6,dtype=np.int64)+500,
      "fat_sidewall_parent_tet_rows":np.arange(6,dtype=np.int64)+600})
    outer=surface(o,curfaces,0,0)
    inner=surface(i,[[0,1,2]],100,1)
    deep=surface(d,[[0,1,2]],200,2)
    return oold,o,i,d,collar,outer,inner,deep

class CurrentShellTests(unittest.TestCase):
    def test_rebind_assemble_serialize_and_validate_shared_I(self):
        _,o,i,d,c,outer,inner,deep=make_case()
        rebound,proof=runner.rebind_collar_to_current_surfaces(c,o,outer["faces"],i,d,True)
        self.assertEqual(proof["outer_face_ancestry_mode"],"current_O_complete_typed_CSR")
        np.testing.assert_array_equal(rebound["exact_candidate_face_source_rows"],[-1])
        np.testing.assert_array_equal(rebound["outer_faces"],outer["faces"])
        self.assertTrue(proof["wall_triangle_coordinates_byte_identical"])
        wall={
          "skin_parent_tets":np.arange(6,dtype=np.int64)+700,
          "skin_parent_nodes":np.tile([100,101,102],(6,1)),
          "skin_materials":np.full(6,15,np.int64),
          "fat_parent_tets":np.arange(6,dtype=np.int64)+800,
          "fat_parent_nodes":np.tile([200,201,202],(6,1)),
          "fat_materials":np.full(6,16,np.int64)}
        skin,fat,report=owner.assemble_shells(outer,inner,deep,rebound,wall)
        self.assertEqual(report["skin_shell"]["edge_report"]["boundary_edges"],0)
        self.assertEqual(report["fat_shell"]["edge_report"]["boundary_edges"],0)
        self.assertEqual(report["skin_shell"]["edge_report"]["nonmanifold_edges"],0)
        self.assertEqual(report["fat_shell"]["edge_report"]["same_direction_twoface_edges"],0)
        runner.validate_shell_maps(owner,skin,fat,outer,inner,deep)
        with tempfile.TemporaryDirectory() as td:
            sp=Path(td)/"skin.npz";fp=Path(td)/"fat.npz"
            owner.write_shell_npz(sp,skin,"skin");owner.write_shell_npz(fp,fat,"fat")
            sl=bridge.load_shell_npz(sp);fl=bridge.load_shell_npz(fp)
            bridge.validate_shared_i(validator_owner,sl,fl,inner)
            np.testing.assert_array_equal(sl["registered_I_source_faces"],inner["faces"])
            np.testing.assert_array_equal(fl["registered_I_source_faces"],inner["faces"])
            sm=sl["points_world_m"][sl["registered_I_shell_node_rows"]]
            fm=fl["points_world_m"][fl["registered_I_shell_node_rows"]]
            self.assertEqual(sm.view(np.uint32).tobytes(),fm.view(np.uint32).tobytes())

    def test_loads_current_typed_face_aliases_without_dropping_ancestry(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float64)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"current-I.npz"
            np.savez_compressed(path,
              points_world_m=points,points_world_m_f32=points.astype(np.float32),
              faces=np.array([[0,1,2]],np.int32),
              source_input_face_rows=np.array([5],np.int64),
              face_source_original_id=np.array([77],np.int64),
              face_source_layer=np.array([1],np.int8),
              face_source_body=np.array([2],np.int32),
              face_source_bone=np.array([3],np.int32),
              face_source_face_id=np.array([88],np.int64),
              face_ancestry_offsets=np.array([0,2],np.int64),
              face_ancestry_source_face_rows=np.array([10,11],np.int64),
              face_ancestry_source_original_ids=np.array([101,102],np.int64),
              face_ancestry_parent_face_rows=np.array([20,21],np.int64))
            got=runner.load_current_surface(owner,path,"I")
            np.testing.assert_array_equal(got["source_rows"],[5])
            np.testing.assert_array_equal(got["source_original_ids"],[77])
            np.testing.assert_array_equal(got["source_face_ids"],[88])
            self.assertEqual(got["ancestry"],[[10,11]])
            self.assertEqual(got["ancestry_original_ids"],[[101,102]])
            self.assertEqual(got["ancestry_parent_rows"],[[20,21]])

    def test_rejects_ambiguous_donor_coordinate(self):
        _,o,i,d,c,outer,_,_=make_case()
        dup=np.vstack((o,o[0]))
        with self.assertRaisesRegex(ValueError,"exact-F32 current matches"):
            runner.rebind_collar_to_current_surfaces(c,dup,outer["faces"],i,d,True)

    def test_rejects_changed_O_faces_without_typed_ancestry(self):
        _,o,i,d,c,outer,_,_=make_case()
        with self.assertRaisesRegex(ValueError,"without complete typed face ancestry"):
            runner.rebind_collar_to_current_surfaces(c,o,outer["faces"],i,d,False)

    def test_rejects_wall_referencing_unmapped_O_row(self):
        _,o,i,d,c,outer,_,_=make_case()
        bad=runner.NPZView(c)
        bad["skin_sidewall_faces"]=np.array(bad["skin_sidewall_faces"],copy=True)
        bad["skin_sidewall_faces"][0,0]=3
        with self.assertRaisesRegex(ValueError,"outside donor correspondence"):
            runner.rebind_collar_to_current_surfaces(bad,o,outer["faces"],i,d,True)

    def test_shared_I_validator_rejects_face_namespace_mismatch(self):
        _,o,i,d,c,outer,inner,deep=make_case()
        rebound,_=runner.rebind_collar_to_current_surfaces(c,o,outer["faces"],i,d,True)
        wall={"skin_parent_tets":np.arange(6),"skin_parent_nodes":np.tile([1,2,3],(6,1)),
          "skin_materials":np.full(6,15),"fat_parent_tets":np.arange(6),"fat_parent_nodes":np.tile([4,5,6],(6,1)),
          "fat_materials":np.full(6,16)}
        skin,fat,_=owner.assemble_shells(outer,inner,deep,rebound,wall)
        bad=dict(fat);bad["registered_I_source_faces"]=np.array([[0,2,1]],np.int64)
        with self.assertRaisesRegex(ValueError,"current I surface"):
            bridge.validate_shared_i(validator_owner,skin,bad,inner)

if __name__=="__main__":
    unittest.main()
