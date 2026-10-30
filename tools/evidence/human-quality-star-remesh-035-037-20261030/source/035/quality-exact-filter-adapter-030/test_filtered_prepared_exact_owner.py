from pathlib import Path
import importlib,hashlib,sys,unittest
import numpy as np
from filtered_prepared_exact_owner import FilteredPreparedExactOwner
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py")
assert hashlib.sha256(OWNER.read_bytes()).hexdigest()=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
sys.path.insert(0,str(OWNER.parent.parent))
owner=importlib.import_module("numilab_human.cardiac_cavity_intersections")
def records(points,faces,ids):
 rows,_=owner.float32_surface_records(np.asarray(points,np.float32).astype(np.float64),np.asarray(faces,np.int64),source_seam_groups=())
 return [(r[0],r[1],r[2],int(ids[i]),r[4]) for i,r in enumerate(rows)]
def duplicate(offset,faceid):
 p=[[offset,0,0],[offset+1,0,0],[offset,1,0]]
 r=records(p,[[0,1,2]],[faceid])[0]
 return (r[0],r[1],r[2],r[3],tuple(10+v for v in r[4]))
class Tests(unittest.TestCase):
 def setUp(self):
  self.base=records([[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]],[[0,1,2],[3,4,5]],[100,200])
  self.index=owner._prepare_surface_aabb(self.base)
  self.proxy=FilteredPreparedExactOwner(owner,{100,200})
 def test_low_query_id_does_not_hide_overlap(self):
  q=[duplicate(0,5)]
  self.assertEqual(owner._audit_pair_prepared_first(self.index,q,same_surface=True)["count"],0)
  self.assertEqual(self.proxy._audit_pair_prepared_first(self.index,q,same_surface=True)["count"],1)
 def test_only_retired_records_excluded(self):
  self.proxy.set_excluded({100})
  r=self.proxy._audit_pair_prepared_first(self.index,[duplicate(0,5),duplicate(3,6)],same_surface=True)
  self.assertEqual(r["triangle_pairs"],[[200,6]])
  self.assertEqual(r["filtered_retired_aabb_pairs"],1)
  self.assertEqual(r["aabb_candidate_pairs"],1)
 def test_cross_domain_overlap_is_not_authorized(self):
  self.assertEqual(self.proxy._audit_pair_prepared_first(self.index,[duplicate(0,5)],same_surface=False)["count"],1)
 def test_mask_cannot_remove_bone_barrier(self):
  with self.assertRaises(ValueError):self.proxy.set_excluded({-1})
  with self.assertRaises(ValueError):FilteredPreparedExactOwner(owner,{-1})
 def test_mask_cannot_remove_another_role(self):
  with self.assertRaises(ValueError):self.proxy.set_excluded({300})
 def test_empty_queries(self):
  r=self.proxy._audit_pair_prepared_first(self.index,[],same_surface=True)
  self.assertEqual((r["count"],r["aabb_candidate_pairs"],r["complete_query_records"]),(0,0,0))
 def test_duplicate_query_ids_rejected(self):
  with self.assertRaises(ValueError):self.proxy._audit_pair_prepared_first(self.index,[duplicate(0,5),duplicate(3,5)])
 def test_shared_index_edge_stays_allowed(self):
  p=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
  rr=records(p,[[0,1,2],[1,0,3]],[100,5])
  proxy=FilteredPreparedExactOwner(owner,{100})
  r=proxy._audit_pair_prepared_first(owner._prepare_surface_aabb(rr[:1]),rr[1:],same_surface=True)
  self.assertEqual(r["count"],0)
  self.assertEqual(r["allowed_shared_vertex_or_edge_pairs"],1)
 def test_index_and_records_unchanged(self):
  before=repr(self.index.records);q=duplicate(0,5)
  self.proxy.set_excluded({100});self.proxy._audit_pair_prepared_first(self.index,[q],same_surface=True)
  self.assertEqual(repr(self.index.records),before)
  self.proxy.set_excluded(())
  self.assertEqual(self.proxy._audit_pair_prepared_first(self.index,[q],same_surface=True)["count"],1)
if __name__=="__main__":unittest.main(verbosity=2)
