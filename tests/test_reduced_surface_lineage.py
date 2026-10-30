"""Source-row versus declared parent-face identity regression for tissue transfer."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

override=os.environ.get("NUMI_REDUCED_LINEAGE_OWNER")
if override:
    spec=importlib.util.spec_from_file_location("reduced_lineage_test",override)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
else:
    from numilab_human import reduced_surface_lineage as owner


class ReducedLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/"source.npz";self.candidate=self.root/"candidate.npz"
        self.output=self.root/"prepared.npz";self.receipt=self.root/"receipt.json"
    def tearDown(self):self.tmp.cleanup()
    def packet(self,full=False):
        source=dict(points_world_m=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32),
                    faces=np.asarray([[0,1,2],[0,2,3]],np.int64))
        for key,value in dict(face_source_layer=1,face_source_body=-1,face_source_bone=-1,
                              face_source_original_id=23,face_source_identity0=4,face_source_identity1=5).items():
            source[key]=np.full(2,value,np.int64)
        source["face_source_face_id"]=np.asarray([7,8],np.int64)
        source["face_source_identity_schema_by_original_id_json"]=np.asarray([json.dumps({"23":{"source":"fixture"}})])
        if full:
            source.update(face_ancestry_offsets=np.asarray([0,2,3],np.int64),
                          face_ancestry_source_face_rows=np.asarray([50,51,52],np.int64),
                          face_ancestry_source_original_ids=np.asarray([40,40,41],np.int64),
                          face_ancestry_parent_face_rows=np.asarray([500,501,502],np.int64))
        np.savez_compressed(self.source,**source)
        candidate={k:v.copy() for k,v in source.items()}
        candidate.update(source_candidate_sha256=np.asarray([owner.sha(self.source)]),
                         source_group_keys_json=np.asarray([json.dumps([dict(face_source_layer=1,face_source_body=-1,face_source_bone=-1,face_source_original_id=23)])]),
                         source_group_offsets=np.asarray([0,2],np.int64),
                         source_group_face_rows=np.asarray([0,1],np.int64),
                         face_group_id=np.asarray([0,0],np.int64),
                         face_parent_source_face_rows=np.asarray([0,1],np.int64),
                         face_repair_parent_source_offsets=np.asarray([0,2,4],np.int64),
                         face_repair_parent_source_rows=np.asarray([0,1,0,1],np.int64))
        np.savez_compressed(self.candidate,**candidate)
        return source,candidate
    def prepare(self):
        receipt=owner.prepare(self.candidate,self.source,self.output,self.receipt)
        with np.load(self.output,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        return receipt,a
    def test_unreduced_source_uses_declared_face_id_not_array_row(self):
        s,c=self.packet();r,a=self.prepare()
        self.assertEqual(a["face_ancestry_source_face_rows"].tolist(),[0,1,0,1])
        self.assertEqual(a["face_ancestry_parent_face_rows"].tolist(),[7,8,7,8])
        self.assertFalse(r["source_had_full_declared_typed_CSR"])
        np.testing.assert_array_equal(a["points_world_m"],c["points_world_m"])
        np.testing.assert_array_equal(a["faces"],c["faces"])
    def test_full_source_csr_composes_all_ancestry_lanes(self):
        self.packet(full=True);r,a=self.prepare()
        self.assertTrue(r["source_had_full_declared_typed_CSR"])
        self.assertEqual(a["face_ancestry_source_face_rows"].tolist(),[50,51,52,50,51,52])
        self.assertEqual(a["face_ancestry_source_original_ids"].tolist(),[40,40,41,40,40,41])
        self.assertEqual(a["face_ancestry_parent_face_rows"].tolist(),[500,501,502,500,501,502])
    def test_missing_primary_uses_only_its_declared_union_representative(self):
        s,c=self.packet();c["face_parent_source_face_rows"][:]=-1
        np.savez_compressed(self.candidate,**c)
        r,a=self.prepare()
        self.assertEqual(a["face_parent_source_face_rows"].tolist(),[0,0])
        self.assertEqual(a["lineage_history_face_parent_source_face_rows"].tolist(),[-1,-1])
        self.assertEqual(a["face_ancestry_parent_face_rows"].tolist(),[7,8,7,8])
        self.assertEqual(r["repair_union_representatives_selected"],2)
        self.assertTrue(np.all(a["face_exact_source_row"]==-1))
    def test_missing_primary_without_declared_union_is_rejected(self):
        s,c=self.packet();c["face_parent_source_face_rows"][:]=-1
        del c["face_repair_parent_source_offsets"];del c["face_repair_parent_source_rows"]
        np.savez_compressed(self.candidate,**c)
        with self.assertRaisesRegex(ValueError,"without a declared repair-parent"):self.prepare()
    def test_D_union_source_and_flip_parent_sets_compose_without_loss(self):
        s,c=self.packet(full=True)
        names={"face_ancestry_offsets":"union_face_ancestry_offsets",
               "face_ancestry_source_face_rows":"union_face_ancestry_rows",
               "face_ancestry_source_original_ids":"union_face_ancestry_run_original_ids",
               "face_ancestry_parent_face_rows":"union_face_ancestry_face_ids"}
        for old,new in names.items():s[new]=s.pop(old)
        np.savez_compressed(self.source,**s)
        c["source_candidate_sha256"]=np.asarray([owner.sha(self.source)])
        c["topology_flip_parent_D624_face_offsets"]=c.pop("face_repair_parent_source_offsets")
        c["topology_flip_parent_D624_face_rows"]=c.pop("face_repair_parent_source_rows")
        np.savez_compressed(self.candidate,**c)
        r,a=self.prepare()
        self.assertTrue(r["source_had_full_declared_typed_CSR"])
        self.assertEqual(a["face_ancestry_source_face_rows"].tolist(),[50,51,52,50,51,52])
        self.assertEqual(a["face_ancestry_source_original_ids"].tolist(),[40,40,41,40,40,41])
        self.assertEqual(a["face_ancestry_parent_face_rows"].tolist(),[500,501,502,500,501,502])
        self.assertEqual(a["face_repair_parent_source_rows"].tolist(),[0,1,0,1])
    def test_missing_candidate_schema_comes_from_authenticated_source(self):
        s,c=self.packet();del c["face_source_identity_schema_by_original_id_json"]
        np.savez_compressed(self.candidate,**c)
        r,a=self.prepare()
        self.assertEqual(a["face_source_identity_schema_by_original_id_json"].tolist(),
                         s["face_source_identity_schema_by_original_id_json"].tolist())
    def test_reject_incomplete_source_schema(self):
        s,c=self.packet();s["face_source_identity_schema_by_original_id_json"]=np.asarray(["{}"])
        np.savez_compressed(self.source,**s)
        c["source_candidate_sha256"]=np.asarray([owner.sha(self.source)])
        np.savez_compressed(self.candidate,**c)
        with self.assertRaisesRegex(ValueError,"schema is incomplete"):self.prepare()
    def test_reject_partial_source_csr(self):
        s,c=self.packet();s["face_ancestry_source_face_rows"]=np.asarray([0,1])
        np.savez_compressed(self.source,**s)
        c["source_candidate_sha256"]=np.asarray([owner.sha(self.source)])
        np.savez_compressed(self.candidate,**c)
        with self.assertRaisesRegex(ValueError,"partial typed ancestry"):self.prepare()
    def test_reject_wrong_source_hash(self):
        s,c=self.packet();c["source_candidate_sha256"]=np.asarray(["wrong"])
        np.savez_compressed(self.candidate,**c)
        with self.assertRaisesRegex(ValueError,"source hash"):self.prepare()
    def test_reject_repair_parent_outside_source_domain(self):
        s,c=self.packet();c["face_repair_parent_source_rows"][1]=2
        np.savez_compressed(self.candidate,**c)
        with self.assertRaisesRegex(ValueError,"outside source face domain"):self.prepare()
    def test_refuse_to_replace_existing_checkpoint(self):
        self.packet();self.output.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual(self.output.read_bytes(),b"preserve")

if __name__=="__main__":unittest.main()
