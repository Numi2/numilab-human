#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,sys,time,struct
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009"); E=Path("/Users/n/numi-human-resting-evidence-20261005")
D=R/"skin-resting-multipose-clearance-1206/inside-vertex-diagnosis-001"
PACK=R/"native-flat-reference-40s-1201/native-run/accepted-geometry/step-20000.mrvpack"
RECEIPT=PACK.with_suffix(".receipt.json"); SKIN=E/"native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
BASE_AUDIT=R/"native-flat-reference-40s-1201/skin-audit-1201-terminal"
START=time.monotonic(); key=(51005,63)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p): return {"path":str(p),"sha256":sha(p)}
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human import common_atlas_skin_clearance as c
from numilab_human import cardiac_cavity_intersections as ci
# The target shell was exactly welded by coordinate only in the quotient gate; reconstruct that same quotient.
pos,surfs,_=c._pack_surfaces(PACK,{key}); skin_raw=SKIN.read_bytes()
import struct as st
magic,abi,nb,nv,ni,_,_=st.unpack_from("<8s5I32s",skin_raw); assert magic==b"NHSKIN1\0" and abi==5
io=60+36*nb+56*nv; skin_faces=np.frombuffer(skin_raw,"<u4",ni,io).reshape(-1,3).astype(np.int64)
base=int(surfs[(51007,1)]["faces"].min()); skin_points=np.asarray(pos[base:base+nv],dtype="<f4").copy()
target_ids=np.unique(surfs[key]["faces"]); target_xyz=np.ascontiguousarray(np.asarray(pos[target_ids],dtype="<f4"))
key_to_q={}; q_vertices=[]; orig_to_q={}
for idx,xyz in zip(target_ids,target_xyz):
 k=xyz.tobytes(); qid=key_to_q.get(k)
 if qid is None: qid=len(q_vertices);key_to_q[k]=qid;q_vertices.append(xyz.copy())
 orig_to_q[int(idx)]=qid
q_vertices=np.asarray(q_vertices,dtype="<f4")
q_faces=np.asarray([[orig_to_q[int(v)] for v in tri] for tri in surfs[key]["faces"]],dtype=np.int64)
lattice_vertices=[ci.float32_point_lattice_key(p) for p in q_vertices]
records=ci._records(lattice_vertices,q_faces)
# Exact owner uses the common 2**149 integer lattice for both surface and tested points.
triangles=np.asarray(pos[surfs[key]["faces"]],dtype="<f4").astype(np.float64)
lo=triangles.min((0,1)); hi=triangles.max((0,1))
ids=np.flatnonzero(np.all((skin_points>=lo)&(skin_points<=hi),axis=1))
tri=triangles
w=np.empty(len(ids),dtype=np.float64)
for st0 in range(0,len(ids),20):
 p=skin_points[ids[st0:st0+20]].astype(np.float64); v=tri[None,:,:,:]-p[:,None,None,:]
 a,b,d=v[:,:,0],v[:,:,1],v[:,:,2]; la=np.linalg.norm(a,axis=2);lb=np.linalg.norm(b,axis=2);ld=np.linalg.norm(d,axis=2)
 num=np.einsum("bfi,bfi->bf",a,np.cross(b,d));ab=np.einsum("bfi,bfi->bf",a,b);bd=np.einsum("bfi,bfi->bf",b,d);da=np.einsum("bfi,bfi->bf",d,a)
 w[st0:st0+len(p)]=np.sum(2*np.arctan2(num,la*lb*ld+ab*ld+bd*la+da*lb),axis=1)/(4*np.pi)

rows=[]; counts={"inside_winding":0,"outside_winding":0,"ambiguous_winding":0,"exact_inside":0,"exact_outside":0,"exact_boundary":0,"exact_indeterminate":0}
for vid,wn in zip(ids,w):
 aw=abs(float(wn)); wc="inside" if aw>.5 else "outside" if aw<.45 else "ambiguous"
 counts[{"inside":"inside_winding","outside":"outside_winding","ambiguous":"ambiguous_winding"}[wc]]+=1
 exact=ci.point_location(ci.float32_point_lattice_key(skin_points[int(vid)]),records)
 counts["exact_"+exact["location"]]+=1
 if (wc=="inside" and exact["location"]!="inside") or (wc=="outside" and exact["location"]!="outside") or wc=="ambiguous":
  pass
 rows.append({"skin_vertex_id":int(vid),"world_xyz_f32_m":skin_points[int(vid)].tolist(),"world_xyz_f32_bits_le_hex":struct.pack("<3f",*map(float,skin_points[int(vid)])).hex(),"generalized_winding":float(wn),"winding_class":wc,"exact_point_location":exact})
inside_rows=[r for r in rows if r["winding_class"]=="inside"]
for row in inside_rows:
 if row["exact_point_location"]["location"]!="inside":
  pass
summary={"schema":"numi.human.right-vastus-winding-exact-parity-check.v1","status":"exact_parity_crosscheck_complete","target":"51005:63 right vastus lateralis","input_pack_pose":{"path":str(PACK),"accepted_step":20000},"target_shell_gate":{"quotient_report":pin(D/"report-coordinate-quotient-stop.json"),"coordinate_equivalence":"bit-exact little-endian Float32 xyz bytes","quotient_vertex_count":len(q_vertices),"face_count":len(q_faces),"exact_self_unallowed_pairs":0},"method":{"candidate_point_domain":"all skin source vertex rows in inclusive right-vastus target triangle-soup AABB; no face-pair scan","winding":"same generalized solid-angle formula as previous report; |w|>0.5 inside, |w|<0.45 outside, otherwise ambiguous","exact_parity":"existing cardiac_cavity_intersections.point_location using exact 2**149 Float32 coordinate lattice for both skin point and quotient target triangles"},"counts":{"aabb_skin_vertices":len(ids),**counts},"winding_exact_disagreements":[r for r in rows if (r["winding_class"]=="inside" and r["exact_point_location"]["location"]!="inside") or (r["winding_class"]=="outside" and r["exact_point_location"]["location"]!="outside")],"winding_ambiguous_rows":[r for r in rows if r["winding_class"]=="ambiguous"],"baseline_inside_vertices_exact_parity_rows":inside_rows,"all_aabb_vertex_rows":rows,"inputs":[pin(PACK),pin(RECEIPT),pin(SKIN),pin(BASE_AUDIT/"step-20000.crossing-witnesses.jsonl"),pin(D/"report-coordinate-quotient-stop.json"),pin(D/"report-coordinate-quotient-winding.json"),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/cardiac_cavity_intersections.py"))],"elapsed_seconds":time.monotonic()-START}
out=D/"right-point-location-check.json";out.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(out),"sha256":sha(out),"counts":summary["counts"],"disagreements":len(summary["winding_exact_disagreements"]),"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)
