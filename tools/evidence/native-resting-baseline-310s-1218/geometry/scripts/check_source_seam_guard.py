#!/usr/bin/env python3
"""Bounded regression checks for rejecting unrelated posed coincidences."""
import hashlib, importlib.util, json, struct, sys
from pathlib import Path
import numpy as np
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-accepted-geometry-audit-001")
P=B/"audit_native_candidate_1218_root003.py"
EXPECTED="246b9c28eeb4c80ded2271dfa3cd340e63ee059568052416748d5c7791761568"
def record(xyz=(0.,0.,0.), indices=(0,1,2,3), weights=(1.,0.,0.,0.), normal=(0.,1.,0.)):
 return struct.pack("<6f4I4f",*(xyz+normal+indices+weights))
def main():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED
 spec=importlib.util.spec_from_file_location("_root003_seam_guard_checks",P)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 xyz=np.zeros((7,3),dtype=np.float64); ids=np.array([5,6],dtype=np.int64)
 tests=[]
 r=m.verify_source_seam_groups(xyz,ids,5,{"vertex_bytes":record()+record(normal=(1.,0.,0.))})
 assert r["posed_duplicate_group_count"]==1 and r["identified_duplicate_coordinate_records"]==1
 tests.append({"case":"identical_source_xyz_and_route_with_distinct_normals","result":"accepted_as_source_seam"})
 for label,b in [
  ("different_source_position",record(xyz=(0.001,0.,0.))),
  ("different_sparse_route_index",record(indices=(1,1,2,3))),
  ("different_sparse_route_weight",record(weights=(0.5,0.5,0.,0.)))]:
  try:m.verify_source_seam_groups(xyz,ids,5,{"vertex_bytes":record()+b})
  except ValueError as exc:
   assert "distinct source positions or muscle routes" in str(exc)
   tests.append({"case":label,"result":"rejected_unrelated_posed_coincidence"})
  else:raise AssertionError(label+" silently accepted")
 distinct=xyz.copy();distinct[6,0]=float(np.float32(0.001))
 r=m.verify_source_seam_groups(distinct,ids,5,{"vertex_bytes":record()+record(xyz=(0.001,0.,0.))})
 assert r["posed_duplicate_group_count"]==0
 tests.append({"case":"distinct_captured_coordinates","result":"no_identification"})
 out=B/"source-seam-guard-checks.json"
 assert not out.exists()
 out.write_text(json.dumps({"status":"passed","audit_script_sha256":EXPECTED,
  "test_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  "cases":tests,"qualification":"Synthetic guard regressions only, not native anatomical evidence."},indent=2,sort_keys=True)+"\n")
 print(out)
if __name__=="__main__":main()
