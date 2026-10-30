"""Portable registered-surface metadata tests; these do not admit physical tissue."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

owner_override = os.environ.get("NUMI_REGISTERED_SURFACE_OWNER")
if owner_override:
    spec = importlib.util.spec_from_file_location("registered_surface_owner_test", owner_override)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
else:
    from numilab_human import registered_surface_handoff as owner


def typed_quad():
    a = {
        "points_world_m": np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], np.float32),
        "faces": np.asarray([[0,1,2],[0,2,3]], np.int64),
        "face_source_layer": np.asarray([1,1],np.int8),
        "face_source_body": np.asarray([-1,-1],np.int32),
        "face_source_bone": np.asarray([-1,-1],np.int32),
        "face_source_original_id": np.asarray([23,23],np.int64),
        "face_source_face_id": np.asarray([7,8],np.int64),
        "face_source_identity0": np.asarray([4,4],np.int64),
        "face_source_identity1": np.asarray([5,5],np.int64),
        "face_source_identity_schema_by_original_id_json": np.asarray([json.dumps({"23":{"source":"fixture"}})]),
    }
    return a


def prepared_quad():
    a = typed_quad()
    a.update({
        "face_group_id": np.asarray([0,0],np.int64),
        "source_group_offsets": np.asarray([0,3],np.int64),
        "source_group_face_rows": np.asarray([0,1,2],np.int64),
        "source_group_keys_json": np.asarray([json.dumps([dict(face_source_layer=1,face_source_body=-1,face_source_bone=-1,face_source_original_id=23)])]),
        "face_parent_source_face_rows": np.asarray([0,2],np.int64),
        "face_exact_source_row": np.asarray([-1,-1],np.int64),
        "face_ancestry_offsets": np.asarray([0,2,3],np.int64),
        "face_ancestry_source_face_rows": np.asarray([0,1,2],np.int64),
        "face_ancestry_source_original_ids": np.asarray([23,23,23],np.int64),
        "face_ancestry_parent_face_rows": np.asarray([7,6,8],np.int64),
        "lineage_preparation_input_sha256": np.asarray(["fixture"]),
        "face_ancestry_method": np.asarray(["fixture derivation; no full overlap claim"]),
    })
    return a


class RegisteredSurfaceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"surface.npz"
    def tearDown(self):
        self.tmp.cleanup()
    def write(self,a):
        np.savez_compressed(self.path,**a)
    def prepared(self,a):
        self.write(a)
        return owner.load_prepared_reduced_surface_217(None,self.path,"I")
    def test_prepared_derivation_retains_all_three_lanes(self):
        result=self.prepared(prepared_quad())
        self.assertEqual(result["ancestry"],[[0,1],[2]])
        self.assertEqual(result["ancestry_original_ids"],[[23,23],[23]])
        self.assertEqual(result["ancestry_parent_rows"],[[7,6],[8]])
        self.assertEqual(result["source_rows"].tolist(),[0,2])
        self.assertTrue(np.all(result["exact_source_rows"] == -1))
        self.assertEqual(result["typed_source_npz_sha256"],owner.sha(self.path))
    def test_unreduced_source_adapts_its_own_declared_rows(self):
        a=typed_quad();self.write(a)
        result=owner.load_direct_typed_surface_217(None,self.path,"I")
        self.assertEqual(result["ancestry"],[[0],[1]])
        self.assertEqual(result["ancestry_parent_rows"],[[7],[8]])
        self.assertEqual(result["source_rows"].tolist(),[0,1])
        self.assertTrue(np.all(result["exact_source_rows"] == -1))
    def test_reduced_asset_cannot_use_direct_fallback(self):
        self.write(prepared_quad())
        with self.assertRaisesRegex(ValueError,"prepared lineage"):
            owner.load_direct_typed_surface_217(None,self.path,"I")
    def test_reject_missing_provenance(self):
        a=prepared_quad();del a["face_source_identity0"]
        with self.assertRaisesRegex(ValueError,"typed provenance"): self.prepared(a)
    def test_reject_index_outside_geometry(self):
        a=prepared_quad();a["faces"][1,2]=4
        with self.assertRaisesRegex(ValueError,"geometry malformed"): self.prepared(a)
    def test_reject_collapsed_float32_face(self):
        a=prepared_quad();a["points_world_m"][3]=a["points_world_m"][2]
        with self.assertRaisesRegex(ValueError,"collapsed F32"): self.prepared(a)
    def test_reject_duplicate_source_csr_rows(self):
        a=prepared_quad();a["source_group_face_rows"][:]=[0,1,1]
        with self.assertRaisesRegex(ValueError,"source-group CSR"): self.prepared(a)
    def test_reject_empty_ancestry_row(self):
        a=prepared_quad();a["face_ancestry_offsets"][:]=[0,0,3]
        with self.assertRaisesRegex(ValueError,"ancestry CSR"): self.prepared(a)
    def test_reject_exact_source_fiction(self):
        a=prepared_quad();a["face_exact_source_row"][0]=0
        with self.assertRaisesRegex(ValueError,"exact source"): self.prepared(a)
    def test_reject_incomplete_identity_schema(self):
        a=prepared_quad();a["face_source_identity_schema_by_original_id_json"]=np.asarray(["{}"])
        with self.assertRaisesRegex(ValueError,"schema incomplete"): self.prepared(a)


if __name__ == "__main__":
    unittest.main()
