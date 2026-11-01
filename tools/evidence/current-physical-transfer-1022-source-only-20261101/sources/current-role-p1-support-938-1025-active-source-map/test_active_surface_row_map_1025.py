from __future__ import annotations
import importlib.util, unittest
from pathlib import Path
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OWNER=H/"current-role-p1-support-938-1025-active-source-map/current_fine_217_owner_938_v3_active_source_rows.py"
spec=importlib.util.spec_from_file_location("owner_active_rows_test",OWNER)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
class ActiveSurfaceRowTests(unittest.TestCase):
    def test_point_map_keeps_unused_source_as_minus_one(self):
        mesh={"points":np.asarray([[1,2,3],[4,5,6]],np.float32)}
        source=np.asarray([[1,2,3],[99,98,97]],np.float32)
        np.testing.assert_array_equal(owner._mesh_source_point_map(mesh,source,"fixture",np.asarray([0])),[0,-1])
    def test_active_missing_or_ambiguous_rows_fail(self):
        source=np.asarray([[1,2,3],[9,9,9]],np.float32)
        with self.assertRaisesRegex(ValueError,"active input surface node 1 has 0"):
            owner._mesh_source_point_map({"points":source[:1]},source,"fixture",np.asarray([1]))
        dup=np.vstack((source[0],source[0]))
        with self.assertRaisesRegex(ValueError,"active input surface node 0 has 2"):
            owner._mesh_source_point_map({"points":dup},source,"fixture",np.asarray([0]))
    def _shell(self):
        return {"registered_O_source_vertex_rows":np.arange(4,dtype=np.int64),
          "registered_O_shell_node_rows":np.arange(4,dtype=np.int64),
          "registered_O_source_faces":np.asarray([[0,1,2]],np.int64),
          "registered_O_shell_face_rows":np.asarray([0],np.int64),
          "points_world_m":np.asarray([[0,0,0],[1,0,0],[0,1,0],[8,8,8]],np.float32),
          "faces":np.asarray([[0,1,2]],np.int64),"face_marker":np.asarray([1],np.int64),
          "face_ancestry":{"face_source_role":np.asarray(["outer_O"])}}
    def test_registered_face_map_preserves_full_rows_with_inactive_sentinel(self):
        shell=self._shell()
        pmap=owner._mesh_source_point_map({"points":shell["points_world_m"][:3]},shell["points_world_m"],"fixture",[0,1,2])
        tet,joint=owner._registered_surface_tetgen_rows(shell,{"points":shell["points_world_m"][:3]},pmap,np.arange(3,dtype=np.int64),
          "O",owner.MARKER["skin_outer_O"],"outer_O","fixture")
        np.testing.assert_array_equal(tet,[0,1,2,-1])
        np.testing.assert_array_equal(joint,[0,1,2,-1])
    def test_active_face_row_cannot_be_unmapped(self):
        shell=self._shell()
        with self.assertRaisesRegex(ValueError,"active input surface node 0 has 0"):
            pmap=owner._mesh_source_point_map({"points":shell["points_world_m"][1:3]},shell["points_world_m"],"fixture",[0,1,2])
            owner._registered_surface_tetgen_rows(shell,{"points":shell["points_world_m"][1:3]},pmap,np.arange(2,dtype=np.int64),
              "O",owner.MARKER["skin_outer_O"],"outer_O","fixture")
    def test_owner_csv_surface_map_skips_only_minus_one(self):
        got={};owner._add_surface_owner_rows(got,np.asarray([8,-1,9]),"O","fixture")
        self.assertEqual(got,{8:("O",0),9:("O",2)})
        with self.assertRaisesRegex(ValueError,"unsupported negative"):
            owner._add_surface_owner_rows({},np.asarray([-2]),"O","fixture")
if __name__=="__main__":unittest.main(verbosity=2)
