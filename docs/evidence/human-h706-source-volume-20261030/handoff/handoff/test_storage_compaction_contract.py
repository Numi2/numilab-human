from pathlib import Path
import hashlib, importlib.util, json, tempfile, unittest
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
BASE=P/'paired-head-reference-fit-684/run-055-D-seam-disk-remesh/d010-ring1-composition-002/compact-unused-storage-001'
CAND=BASE/'candidate-D-d010-ring1-composed-schema62000204-compact-unused.npz'
CERT=BASE/'storage-compaction-certificate.json'
ROWMAP=BASE/'old-to-compact-row-map.npz'
PROOF=BASE/'H706-rebound-003/candidate-preflight-receipt.json'
HELPER=P/'fine-physical-volume-706/handoff/canonical_d_operational_targets.py'
spec=importlib.util.spec_from_file_location('canonical_d_operational_targets_test',HELPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class StorageCompactionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.proof=json.loads(PROOF.read_text())
  with np.load(CAND,allow_pickle=False) as z: cls.points=np.asarray(z['points_world_m'],np.float32)
  cls.packet=cls.proof['point_preservation']
 def test_authenticated_compaction_pass(self):
  out=module.validate_storage_compaction(self.packet,CAND,self.points)
  self.assertEqual(out['point_count'],128293)
  self.assertEqual(out['removed_unreferenced_source_rows'],[72893,82725])
  self.assertTrue(out['all_ordered_face_coordinate_bits_identical'])
  self.assertTrue(out['existing_output_vertex_map_preserved'])
 def test_stale_old_to_new_map_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   bad=Path(td)/'bad-map.npz'
   with np.load(ROWMAP,allow_pickle=False) as z:
    old=np.asarray(z['old_to_compact_vertex_row']).copy();new=np.asarray(z['compact_to_old_vertex_row']).copy()
   old[0],old[1]=old[1],old[0]
   np.savez_compressed(bad,old_to_compact_vertex_row=old,compact_to_old_vertex_row=new)
   packet=dict(self.packet,row_map_path=str(bad),row_map_sha256=sha(bad))
   with self.assertRaisesRegex(ValueError,'row map is stale'):
    module.validate_storage_compaction(packet,CAND,self.points)
 def test_tampered_retained_coordinate_rejected_even_with_rehashed_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);badcand=td/'tampered.npz';badcert=td/'certificate.json'
   with np.load(CAND,allow_pickle=False) as z: arrays={k:np.array(z[k],copy=True) for k in z.files}
   arrays['points_world_m'][100,0]=np.nextafter(arrays['points_world_m'][100,0],np.float32(np.inf))
   np.savez_compressed(badcand,**arrays)
   cert=json.loads(CERT.read_text());cert['compact_candidate_path']=str(badcand);cert['compact_candidate_sha256']=sha(badcand)
   badcert.write_text(json.dumps(cert,sort_keys=True)+'\n')
   packet=dict(self.packet,certificate_path=str(badcert),certificate_sha256=sha(badcert),compact_candidate_sha256=sha(badcand))
   with self.assertRaisesRegex(ValueError,'altered a retained point'):
    module.validate_storage_compaction(packet,badcand,arrays['points_world_m'])
if __name__=='__main__': unittest.main(verbosity=2)
