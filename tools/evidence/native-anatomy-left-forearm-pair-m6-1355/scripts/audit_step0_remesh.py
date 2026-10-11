#!/usr/bin/env python3
import hashlib,importlib.util,json,pickle,sys,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
W=Path("/Users/n/numi-human-fds132-m6-1350"); A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
CAND=Path(sys.argv[1]) if len(sys.argv)>1 else W/"attempt-059-paired-cut-sliver-contraction-001/candidate.pkl"
CREPORT=CAND.parent/"report.json"; CHECK=CAND.parent/"exact-check-001/report.json"
REF=W/"attempt-050/arrangement-001/result-001/source-parent-refs-054/source-parent-refs-and-shared-interface.json"
REFREPORT=W/"attempt-050/arrangement-001/result-001/source-parent-refs-054/report.json"
REST=A/"forearm054-native-diagnostic-006/rest-registration-preflight.json"
PACK=A/"forearm054-native-diagnostic-006/baseline/native-run/accepted-geometry/step-0.mrvpack"
RECEIPT=A/"forearm054-native-diagnostic-006/baseline/native-run/accepted-geometry/step-0.receipt.json"
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
CI=Path("/Users/n/numi-human-anatomy-completion-1276/src/numilab_human/cardiac_cavity_intersections.py")
CA=Path("/Users/n/numi-human-anatomy-completion-1276/src/numilab_human/common_atlas_skin_clearance.py")
OUT=W/"attempt-061-step0-capture-remesh-diagnostic"; OUT.mkdir(parents=True,exist_ok=True); START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(v,m):
 if not v: raise RuntimeError(m)
def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def xyz(v): return tuple(q if isinstance(q,F) else F(int(q[0]),int(q[1])) for q in v)
def terms_for(rec):
 p=rec["source_parent"];sid=int(p["parent_stable_id"]);ids=list(map(int,p["parent_candidate_vertex_indices"]));alpha=[F(int(x[0]),int(x[1])) for x in p["barycentric_exact"]]
 req(len(ids)==3 and sum(alpha,F(0))==1 and min(alpha)>=0,("bad parent simplex",sid,ids,alpha))
 z=tuple(sorted([(0,sid,vi,a) for vi,a in zip(ids,alpha) if a>0],key=lambda z:(z[0],z[1],z[2])))
 req(sum((x[3] for x in z),F(0))==1 and len({x[:3] for x in z})==len(z),"parent terms not canonical positive simplex")
 return z
def exact_self(points,faces,original_ids,parent_refs,ci):
 pp=[tuple(F.from_float(float(v)) for v in p) for p in np.asarray(points,dtype=np.float32)];deg=[]
 for i,f in enumerate(faces):
  a,b,c=(pp[int(k)] for k in f)
  if not any(ci._cross(ci._sub(b,a),ci._sub(c,a))): deg.append(int(i))
 if deg: return {"degenerate_count":len(deg),"degenerate_indices":deg[:100],"self_count":None,"self_pairs":[]}
 au=ci._audit_pair(ci._records(pp,np.asarray(faces,dtype=np.int64).tolist()),ci._records(pp,np.asarray(faces,dtype=np.int64).tolist()),same_surface=True);pairs=[]
 for i,j in au["triangle_pairs"]:
  i=int(i);j=int(j);pairs.append({"remeshed_face_indices":[i,j],"original_054_face_indices":[int(original_ids[i]),int(original_ids[j])],"parent_face_refs":[parent_refs[int(original_ids[i])],parent_refs[int(original_ids[j])] ]})
 return {"degenerate_count":0,"degenerate_indices":[],"self_count":int(au["count"]),"self_pairs":pairs,"self_aabb_candidate_pairs":int(au.get("aabb_candidate_pairs",0))}
def capture_row(pos,surf,sid,ref):
 q=np.asarray(surf[(51005,sid)]["faces"],dtype=np.int64);nf=len(ref["faces"]);nv=len(ref["vertices"]);req(len(q)==nf,("capture face count",sid,len(q),nf));off=int(q.min())
 req(np.array_equal(q-off,np.asarray(ref["faces"],dtype=np.int64)),("capture face order differs from 054 row",sid))
 req(int(q.max()-off+1)==nv,("capture vertex count differs",sid,int(q.max()-off+1),nv))
 return np.asarray(pos[off:off+nv],dtype=np.float32).copy()
req(sha(CAND)==json.loads(CREPORT.read_text())["candidate_sha256"],"candidate/report hash mismatch")
ref=json.loads(REF.read_text());rest=json.loads(REST.read_text());creport=json.loads(CREPORT.read_text());check=json.loads(CHECK.read_text())
req(rest["status"]=="passed" and rest["source_parent_map_sha256"]==sha(REF),"054 rest preflight mismatch")
sys.path.insert(0,str(CI.parent.parent));from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
d=pickle.loads(CAND.read_bytes());regions=d["regions"];gpts=[xyz(p) for p in regions["132"]["vertices"]]
req(all(xyz(p)==gpts[i] for i,p in enumerate(regions["134"]["vertices"])),"global point tables differ")
req(len({p:i for i,p in enumerate(gpts)})==len(gpts),"duplicate global exact points")
ret=np.asarray(d["original_to_retained_vertex"],dtype=np.int64);req(len(ret)==len(gpts) and np.array_equal(ret[ret],ret),"invalid retained endpoint map")
terms={};local_gid={};capture={}
for sid in (132,134):
 s=str(sid);mr=ref["rows"][s];reg=regions[s];lp=[xyz(v["exact_xyz_m"]) for v in mr["vertices"]];l2g=list(map(int,reg["original_local_to_global"]))
 req(len(lp)==len(l2g) and all(gpts[g]==p for p,g in zip(lp,l2g)),("row global map mismatch",sid))
 for li,rec in enumerate(mr["vertices"]):
  tm=terms_for(rec);old=terms.setdefault(lp[li],tm);req(old==tm,("duplicate point map mismatch",sid,li))
 local_gid[s]=l2g
pos,surf,_=ca._pack_surfaces(PACK,{(51005,132),(51005,134)})
for sid in (132,134): capture[str(sid)]=capture_row(pos,surf,sid,ref["rows"][str(sid)])
gid_pos={}
for s in ("132","134"):
 for li,g in enumerate(local_gid[s]):
  p=capture[s][li]
  if g in gid_pos: req(np.array_equal(gid_pos[g],p),("duplicated shared capture differs",g))
  else: gid_pos[g]=p
old_shared=[];new_shared=[]
for x in ref["shared_owner_patch_vertices"]:
 i=int(x["row_local_vertex_indices"]["132"]);j=int(x["row_local_vertex_indices"]["134"]);a=local_gid["132"][i];b=local_gid["134"][j]
 req(a==b,("054 shared IDs differ",a,b));old_shared.append(a);req(int(ret[a])==int(ret[b]),"paired endpoint contraction differs");new_shared.append(int(ret[a]))
compact={};audits={}
for sid in (132,134):
 s=str(sid);reg=regions[s];ofs=[tuple(map(int,f)) for f in reg["faces"]];orig=list(map(int,reg["original_face_indices"]));anc=reg["arrangement_row_ancestry"]
 req(len(ofs)==len(orig)==len(anc),("face lineage sizes",sid))
 for oi,pf in zip(orig,anc): req(pf==ref["rows"][s]["face_parent_refs"][oi],("face parent changed",sid,oi))
 used=sorted({int(v) for f in ofs for v in f});req(all(int(ret[g])==g and g in gid_pos for g in used),("final face ID absent from old capture",sid))
 ix={g:i for i,g in enumerate(used)};faces=np.asarray([[ix[g] for g in f] for f in ofs],dtype=np.int64);pts=np.asarray([gid_pos[g] for g in used],dtype=np.float32);maps=[]
 for g in used:
  tm=terms[gpts[g]];req(len(tm)<=3 and all(v[3]>0 for v in tm) and sum((v[3] for v in tm),F(0))==1,("invalid output map",sid,g));maps.append(tm)
 compact[s]={"global_vertex_ids":used,"faces":faces,"points":pts,"raw_source_terms":maps}
 audits[s]=exact_self(pts,faces,orig,ref["rows"][s]["face_parent_refs"],ci)
seam_equal=all(np.array_equal(capture["132"][local_gid["132"].index(g)],capture["134"][local_gid["134"].index(g)]) for g in set(old_shared))
req(seam_equal,"actual native006 seam positions differ")
passed=all(audits[str(s)]["degenerate_count"]==0 and audits[str(s)]["self_count"]==0 for s in (132,134))
report={"schema":"numi.human.left-fds132-fdp134-step0-native-capture-remesh-diagnostic.v1","status":"step0_capture_remesh_self_clear" if passed else "step0_capture_remesh_self_failed","candidate_path":str(CAND),"candidate_sha256":sha(CAND),"candidate_construction_report_path":str(CREPORT),"candidate_construction_report_sha256":sha(CREPORT),"candidate_exact_check_path":str(CHECK),"candidate_exact_check_sha256":sha(CHECK),"candidate_exact_check_union":check["checks"]["union"],"candidate_admitted":False,"source_parent_manifest_sha256":sha(REF),"rest_registration_preflight_path":str(REST),"rest_registration_preflight_sha256":sha(REST),"rest_registration_preflight_status":rest["status"],"full_054_dominant_map_max_reproduction_um":max(float(rest["provisional_child_weight_rules"]["dominant"]["rows"][str(s)]["max_um"]) for s in (132,134)),"rest_gate_subset_reason":"Only exact original 054 points survive; retained endpoints reuse existing exact source-parent maps; no new point or coefficient.","actual_native006_step0_capture":{"pack_path":str(PACK),"pack_sha256":sha(PACK),"receipt_path":str(RECEIPT),"receipt_sha256":sha(RECEIPT),"accepted_body_state_sha256":json.loads(RECEIPT.read_text()).get("accepted_body_state_sha256"),"source_054_face_order_verified":True,"shared_seam_bitwise_equal":seam_equal,"remeshed_surface_audits":audits,"counts":{s:{"vertices":len(compact[s]["global_vertex_ids"]),"faces":len(compact[s]["faces"]),"max_parent_terms":max(len(x) for x in compact[s]["raw_source_terms"])} for s in ("132","134")}},"map_adaptation":{"global_points":len(gpts),"mapped_points":len(terms),"original_shared_vertices":len(ref["shared_owner_patch_vertices"]),"retained_shared_vertices":len(set(new_shared)),"all_terms_positive_normalized":True,"cross_row_face_endpoints_supported":True},"evidence_boundary":"CPU geometric remesh of actual native006 step0 Float32 child positions. Not execution of candidate bindings or a native run. Candidate exact union failure remains disqualifying.","pins":{str(p):sha(p) for p in [CAND,CREPORT,CHECK,REF,REFREPORT,REST,PACK,RECEIPT]},"elapsed_seconds":time.monotonic()-START,"cpu_only":True,"gpu_used":False}
outfile=OUT/(CAND.parent.name+"-step0-report.json");outfile.write_text(json.dumps(report,indent=2,sort_keys=True,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+"\n")
print(json.dumps({"report":str(outfile),"report_sha256":sha(outfile),"status":report["status"],"self_counts":{s:audits[s]["self_count"] for s in audits},"degenerates":{s:audits[s]["degenerate_count"] for s in audits},"union_self_pairs":check["checks"]["union"]["self_count"],"elapsed_s":report["elapsed_seconds"]},sort_keys=True),flush=True)
