#!/usr/bin/env python3
"""Independent rim-seam correction for paired inferred-envelope retirement 293."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"native-apex-D-ears-282-root/run-001/typed-001/candidate-D-typed-current.npz"
S=H/"fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz"
SR=H/"fat-inferred-boundary-shell-assembly-140-v2/receipt.json"
CR=H/"microedge-collar-014-saved-op-replay/receipt.json"
OUTD=H/"paired-inferred-envelope-resolution-reduction-293-root/candidate-D-paired-resolution-reduced.npz"
MAP=H/"paired-inferred-envelope-resolution-reduction-293-root/paired-retirement-maps.npz"
OP=H/"paired-inferred-envelope-resolution-reduction-293-root/operation.json"
REC=H/"paired-inferred-envelope-resolution-reduction-293-root/receipt.json"
OUT=H/"paired-inferred-envelope-resolution-reduction-293-root/rim-seam-correction-001.json"
PINS={
 str(D):"a8845bc3ab5106fa08e77ecd37f7bc65ffe9991961f68351c444e2cf6b1a6866",
 str(S):"65a13549560123b31ecbc698680edaff48ab843e834ef127e72f59d6aaffc04a",
 str(OUTD):"9a66be5a75d097647d8ff6c0b8a7ee447faa687f2434d146709d6ae5fa316ff5",
 str(MAP):"6c9a38306195d0cce8d96d869a9e2cbdede87c0c9a477cb399e9ee24a3b1abca",
 str(OP):"b73348842c2a8653e9ce8f641c93fa836e38315da8aa00173e8173c781b6c1ba",
 str(REC):"7ef042cea2ced053824261505d37454fcf64353249080e23beeb2402c9d23183",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def census(faces):
 out=defaultdict(lambda:[0,0])
 owner=defaultdict(list)
 for fi,t in enumerate(faces):
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):
   e=tuple(sorted((a,b)));out[e][0]+=1;out[e][1]+=1 if a<b else -1;owner[e].append((fi,a,b))
 return {e:(v[0],v[1]) for e,v in out.items()},owner
def edge_in_face(t,u,v):
 for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):
  if {a,b}=={u,v}:return (a,b)
 raise AssertionError("edge absent")
def cyclic_same(a,b):
 a=tuple(map(int,a));b=tuple(map(int,b))
 return any(tuple(a[i:]+a[:i])==b for i in range(3))
def cyclic_reverse(a,b):
 ar=(int(a[0]),int(a[2]),int(a[1]))
 return cyclic_same(ar,b)
def main():
 for p,h in PINS.items():
  got=sha(p)
  if got!=h:raise RuntimeError(f"pin mismatch {p}: {got}")
 d=load(D);sh=load(S);od=load(OUTD);maps=load(MAP)
 c=json.loads(CR.read_text());sr=json.loads(SR.read_text())
 if c["outputs"]["wall_csv"]["sha256"]!="62373bf5fc8e7ebcdf03261acbf4a564957485eaa058db3d1140dd47df3e97a7":
  raise RuntimeError("collar014 wall CSV pin mismatch")
 if not c.get("wall_ancestry_existing_owner_pass"):raise RuntimeError("collar014 wall ancestry owner did not pass")
 if sr.get("status")!="PASS_EXISTING_697_ASSEMBLY_AND_EXACT_CANDIDATE135_BOUNDARY":
  raise RuntimeError("shell140 owner receipt status mismatch")
 f=d["faces"]; pts=d["points_world_m"]
 before,owners=census(f)
 # Ensure retired K4 is closed so its removal does not alter boundary.
 k4=np.array([62075,62077,162298,162299],dtype=np.int64)
 keep=np.ones(len(f),dtype=bool);keep[k4]=False
 kept=f[keep]
 old_boundary={e:v for e,v in before.items() if v[0]!=2 or v[1]!=0}
 retained,retained_owners=census(kept)
 # Map output faces back to immutable D282 point rows and compare exact directed census.
 point_map=maps["D_output_point_to_parent282_row"].astype(np.int64)
 mapped=point_map[od["faces"]]
 outc,outowners=census(mapped)
 if retained!=outc:raise RuntimeError("retained D directed edge census changed through output map")
 new_boundary={e:v for e,v in outc.items() if v[0]!=2 or v[1]!=0}
 if old_boundary!=new_boundary:raise RuntimeError("D boundary edge witness changed after paired retirement")
 open_edges=[(e,v,owners[e][0]) for e,v in old_boundary.items() if v[0]==1]
 if len(open_edges)!=45:raise RuntimeError(f"expected 45 inherited open D edges, found {len(open_edges)}")
 # Exact coordinate map for only those rim vertices.
 coord=defaultdict(list)
 for r,p in enumerate(sh["points_world_m"]):coord[tuple(np.asarray(p).tolist())].append(r)
 d2s={}
 for e,_,_ in open_edges:
  for q in e:
   hits=coord.get(tuple(np.asarray(pts[q]).tolist()),[])
   if len(hits)!=1:raise RuntimeError(f"D rim point {q} has {len(hits)} exact shell140 point matches")
   d2s[q]=hits[0]
 roles=sh["face_source_role"]
 inner_rows=np.flatnonzero(roles=="inner_D")
 wall_rows=np.flatnonzero(roles=="donor_fat_ID_headwall")
 def build_edges(rows):
  em=defaultdict(list)
  for fr in rows:
   t=sh["faces"][fr]
   for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
    em[tuple(sorted((int(a),int(b))))].append((int(fr),int(a),int(b)))
  return em
 inner=build_edges(inner_rows);wall=build_edges(wall_rows)
 recs=[]
 for e,stat,drec in open_edges:
  du,dv=drec[1],drec[2]
  su,sv=d2s[du],d2s[dv]
  key=tuple(sorted((su,sv)))
  ir=inner.get(key,[]);wr=wall.get(key,[])
  if len(ir)!=1 or len(wr)!=1:raise RuntimeError(f"rim edge {e}: inner_D hits={len(ir)}, wall hits={len(wr)}")
  if_edge=(ir[0][1],ir[0][2]);wf_edge=(wr[0][1],wr[0][2])
  if if_edge!=(sv,su):raise RuntimeError(f"inner_D winding does not reverse D edge {e}")
  if wf_edge!=(if_edge[1],if_edge[0]):raise RuntimeError(f"wall winding does not cancel inner_D edge {e}")
  itri=sh["faces"][ir[0][0]].astype(int)
  recs.append({"D282_edge":[int(du),int(dv)],"D282_face_row":int(drec[0]),"D282_face_vertices":f[drec[0]].astype(int).tolist(),"shell140_inner_D_face_row":ir[0][0],"shell140_inner_D_vertices":itri.tolist(),"shell140_wall_face_row":wr[0][0],"shell140_wall_vertices":sh["faces"][wr[0][0]].astype(int).tolist(),"D_to_inner_D":"opposite","inner_D_to_wall":"opposite","full_face_triangle_equivalence":"not_asserted"})
 result={
  "schema":"numi.paired-retirement-inherited-rim-seam-correction.v1",
  "status":"PASS_INHERITED_45_EDGE_RIM_SEAM_EXACTLY_REPRESENTED_IN_SHELL140",
  "scope":"only the inherited D rim seam and its unchanged edge census; this does not claim the whole current I/D/Fat assembly is closed",
  "supersedes_diagnostic_field":"operation.json effect.inherited_registered_rim_wall_pairing.orientation_relation_counts was computed from sorted edge endpoints and is superseded here",
  "inputs":{p:{"sha256":sha(p)} for p in PINS},
  "owner_receipts":{"collar014_receipt":{"path":str(CR),"sha256":sha(CR),"wall_csv_path":c["outputs"]["wall_csv"]["path"],"wall_csv_sha256":c["outputs"]["wall_csv"]["sha256"],"wall_ancestry_existing_owner_pass":True},"shell140_receipt":{"path":str(SR),"sha256":sha(SR),"status":sr["status"]}},
  "checks":{"source_D_directed_edge_census_equals_retained_output_mapped_to_D282":True,"D_boundary_edge_counter_unchanged":True,"inherited_D_open_edges":45,"exact_unique_shell140_coordinate_match_for_all_rim_endpoints":True,"each_edge_has_one_inner_D_and_one_headwall_face":True,"D_to_inner_D_opposite_winding":True,"inner_D_to_wall_opposite_winding":True,"rim_edge_level_orientation_chain_verified":True,"full_face_triangle_equivalence_checked":False,"D_itself_globally_closed":False,"whole_current_I_D_and_walls_closure_claimed":False},
  "counts":{"D282_faces":len(f),"paired_reduced_D_faces":len(od["faces"]),"D_open_edges_before_after":[len(open_edges),sum(1 for e,v in new_boundary.items() if v[0]==1)],"shell140_inner_D_faces":len(inner_rows),"shell140_headwall_faces":len(wall_rows),"verified_rim_edge_pairs":len(recs)},
  "pairs":recs,
  "limitations":["This verifies the inherited rim edge chain only; the final current I/D plus wall assembly must still pass its existing complete assembly/closure owner after singleton repair and final surface rebinding.","No mass redistribution, physical installation, or native qualification is claimed."]
 }
 OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":result["status"],"path":str(OUT),"sha256":sha(OUT),"pairs":len(recs),"D_boundary_unchanged":True,"wall_pairing":True}))
if __name__=="__main__":main()
