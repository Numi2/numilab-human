#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
S=importlib.util.spec_from_file_location("build_current_manifest_849", HERE/"build_current_manifest_849d_explicit_reference_base.py")
M=importlib.util.module_from_spec(S); S.loader.exec_module(M)


def write_volume(path, points, tets, mats, maps=None):
    data={"points_world_m":np.asarray(points,np.float32),
          "tetrahedra":np.asarray(tets,np.int64),
          "material_indices":np.asarray(mats,np.int32)}
    if maps: data.update(maps)
    np.savez_compressed(path,**data)
    return M.exact_pin(path)


def write_state(path, skin_pin, fat_pin, parent=None, **extra):
    d={"schema":"h706.combined-current-test.v1","Skin_current":skin_pin,"Fat_current":fat_pin,
       "physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
    if parent is not None: d["source_state"]=parent
    d.update(extra); path.write_text(json.dumps(d)); return M.exact_pin(path)


class AutoManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.pts=np.arange(21,dtype=np.float32).reshape(7,3)
        self.tets=np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]],np.int64)
        self.mats=np.array([15,15,16],np.int32)
        self.base_skin=write_volume(self.root/"base-skin.npz",self.pts,self.tets,self.mats)
        self.base_fat=write_volume(self.root/"base-fat.npz",self.pts,self.tets,self.mats)
        self.base=write_state(self.root/"base.json",self.base_skin,self.base_fat)
    def tearDown(self): self.tmp.cleanup()

    def test_auto_chain_accepts_point_only_domain_change_and_composes(self):
        pts2=np.vstack([self.pts,np.array([[99,98,97]],np.float32)])
        s1=write_volume(self.root/"s1-skin.npz",pts2,self.tets,self.mats)
        f1=write_volume(self.root/"s1-fat.npz",pts2,self.tets,self.mats)
        s1state=write_state(self.root/"s1.json",s1,f1,self.base)
        manifest,report=M.build_manifest(self.base["path"],s1state["path"])
        self.assertEqual(report["chain_state_count"],2)
        self.assertEqual(report["roles"]["Skin"]["identity_cell_steps"],1)
        self.assertTrue(report["roles"]["Skin"]["point_row_domain_changed_identity_steps"])
        replay=M.validate_manifest(manifest)
        self.assertEqual(replay["status"],"PASS_SOURCE_ONLY_CANONICAL_CURRENT_VOLUME_PARENT_UNIONS")
        self.assertEqual(replay["roles"]["Skin"]["canonical_parent_references"],3)

    def test_pending_single_role_proposal_rejected(self):
        bad=self.root/"pending.json"
        bad.write_text(json.dumps({"schema":"pending","status":"CANDIDATE_PENDING_775_697_217",
                                  "Skin_current":self.base_skin,"source_state":self.base}))
        with self.assertRaises(ValueError): M.build_manifest(self.base["path"],bad)

    def test_wrong_source_pin_hash_rejected(self):
        pts2=np.vstack([self.pts,np.array([[99,98,97]],np.float32)])
        s1=write_volume(self.root/"s1-skin.npz",pts2,self.tets,self.mats)
        f1=write_volume(self.root/"s1-fat.npz",pts2,self.tets,self.mats)
        bad_parent={"path":self.base["path"],"sha256":"0"*64}
        s1state=write_state(self.root/"s1.json",s1,f1,bad_parent)
        with self.assertRaises(ValueError): M.build_manifest(self.base["path"],s1state["path"])

    def test_mapped_cell_step_requires_and_composes_full_parent_union(self):
        # Three base cells; output keeps the first two and replaces the third.
        points=np.vstack([self.pts,np.array([[21,22,23]],np.float32)])
        maps={
          "retained_input_cell_rows":np.array([0,1],np.int64),
          "replacement_tetrahedra":np.array([[2,3,4,7]],np.int64),
          "replacement_parent_offsets":np.array([0,2],np.int64),
          "replacement_parent_cell_rows":np.array([1,2],np.int64),
          "output_cell_to_input_cell_rows":np.array([0,1,1],np.int64),
          "input_cell_to_output_cell_rows":np.array([0,1,-1],np.int64),
        }
        s=write_volume(self.root/"mapped-skin.npz",points,np.vstack([self.tets[:2],maps["replacement_tetrahedra"]]),self.mats,maps)
        f=write_volume(self.root/"mapped-fat.npz",points,np.vstack([self.tets[:2],maps["replacement_tetrahedra"]]),self.mats,maps)
        cur=write_state(self.root/"mapped.json",s,f,self.base)
        manifest,report=M.build_manifest(self.base["path"],cur["path"])
        self.assertEqual(report["roles"]["Skin"]["mapped_steps"],1)
        replay=M.validate_manifest(manifest)
        self.assertEqual(replay["roles"]["Skin"]["canonical_parent_references"],4)

if __name__=="__main__": unittest.main()
