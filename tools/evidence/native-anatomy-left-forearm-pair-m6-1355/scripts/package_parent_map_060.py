#!/usr/bin/env python3
import hashlib,json,pickle
from fractions import Fraction as F
from pathlib import Path
W=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
C=W/"attempt-060-paired-cut-sliver-contraction-001/candidate.pkl"
CR=C.parent/"report.json"; CK=C.parent/"exact-check-001/report.json"
REF=W/"attempt-050/arrangement-001/result-001/source-parent-refs-054/source-parent-refs-and-shared-interface.json"
REST=A/"forearm054-native-diagnostic-006/rest-registration-preflight.json"
PARENT=A/"left-fds-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
AUD=W/"attempt-061-step0-capture-remesh-diagnostic/attempt-060-paired-cut-sliver-contraction-001-step0-report-corrected.json"
OUT=W/"attempt-061-step0-capture-remesh-diagnostic"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
def xyz(v):
 return tuple(q if isinstance(q,F) else F(int(q[0]),int(q[1])) for q in v)
def terms_for(rec):
 p=rec["source_parent"]; sid=int(p["parent_stable_id"])
 ids=list(map(int,p["parent_candidate_vertex_indices"]))
 aa=[F(int(z[0]),int(z[1])) for z in p["barycentric_exact"]]
 req(len(ids)==3 and sum(aa,F(0))==1 and min(aa)>=0,("invalid parent simplex",sid))
 return tuple(sorted([(0,sid,vi,a) for vi,a in zip(ids,aa) if a>0],key=lambda z:(z[0],z[1],z[2])))
req(json.loads(CR.read_text())["candidate_sha256"]==sha(C),"candidate/report hash mismatch")
ck=json.loads(CK.read_text());req(ck["complete"] and ck["checks"]["union"]["self_count"]==0 and ck["checks"]["union"]["topology_error"] is None,"candidate exact union did not pass")
audit=json.loads(AUD.read_text());req(audit["candidate_sha256"]==sha(C) and audit["status"]=="step0_capture_remesh_self_clear","step0 audit mismatch")
ref=json.loads(REF.read_text());rr=pickle.load(open(C,"rb"));regions=rr["regions"]
gpts=[xyz(v) for v in regions["132"]["vertices"]]
req(all(xyz(v)==gpts[i] for i,v in enumerate(regions["134"]["vertices"])),"global vertex tables differ")
point_terms={}
for sid in (132,134):
 s=str(sid);mr=ref["rows"][s];reg=regions[s];lp=[xyz(v["exact_xyz_m"]) for v in mr["vertices"]];l2g=list(map(int,reg["original_local_to_global"]))
 req(len(lp)==len(l2g) and all(gpts[g]==p for p,g in zip(lp,l2g)),("source point correspondence",sid))
 for p,v in zip(lp,mr["vertices"]):
  t=terms_for(v);old=point_terms.setdefault(p,t);req(old==t,("duplicate source map conflict",sid))
outregions={};surfaces=[];exact_payload={}
for sid in (132,134):
 s=str(sid);reg=regions[s]; faces=[tuple(map(int,f)) for f in reg["faces"]];used=sorted({g for f in faces for g in f})
 req(all(gpts[g] in point_terms for g in used),"output vertex lacks parent map")
 ix={g:i for i,g in enumerate(used)};cf=[[ix[g] for g in f] for f in faces]
 vertices=[];exact_vertices=[];raw=[]
 for g in used:
  tm=point_terms[gpts[g]]
  # canonical receipt order, with exact rational retained separately
  vertices.append([[int(pi),int(rsid),int(vi),float(a)] for pi,rsid,vi,a in tm])
  exact_vertices.append([[int(pi),int(rsid),int(vi),int(a.numerator),int(a.denominator)] for pi,rsid,vi,a in tm])
  raw.append([[int(pi),int(rsid),int(vi),float(a)] for pi,rsid,vi,a in tm])
 parentfaces=list(map(int,reg["original_face_indices"]))
 req(len(parentfaces)==len(cf),"face lineage size mismatch")
 pfrefs=reg["arrangement_row_ancestry"]
 req(all(pfrefs[i]==ref["rows"][s]["face_parent_refs"][parentfaces[i]] for i in range(len(parentfaces))),"face parent ancestry mismatch")
 surfaces.append({"stable_id":sid,"vertices":vertices})
 exact_payload[s]={"stable_id":sid,"global_vertex_ids":used,"rest_vertices_exact_m":[gpts[g] for g in used],"faces":cf,"source_parent_terms_exact":exact_vertices,"source_parent_terms_float":raw,"original_054_face_indices":parentfaces,"original_054_face_parent_refs":pfrefs}
out={"schema":"numi.human.passive-tissue-parent-map-template.v1","candidate_path":str(C),"candidate_sha256":sha(C),"source_parent_manifest_path":str(REF),"source_parent_manifest_sha256":sha(REF),"parent_source_payloads":[{"path":str(PARENT),"sha256":sha(PARENT),"source_payload_index":0}],"coordinate_source":"Exact retained vertices from candidate060, each mapped by the unchanged exact 054 source_parent simplex; no new points or fitted coefficients.","surface_order":"stable IDs sorted ascending; within each surface vertices sorted by retained global candidate ID; faces preserve candidate060 output face order with indices remapped to that vertex order.","surfaces":exact_payload,"candidate_exact_check_path":str(CK),"candidate_exact_check_sha256":sha(CK),"step0_capture_remesh_report_path":str(AUD),"step0_capture_remesh_report_sha256":sha(AUD),"rest_registration_preflight_path":str(REST),"rest_registration_preflight_sha256":sha(REST),"rest_registration_preflight_status":json.loads(REST.read_text())["status"],"rest_reproduction_upper_bound_um":max(float(json.loads(REST.read_text())["provisional_child_weight_rules"]["dominant"]["rows"][s]["max_um"]) for s in ("132","134")),"native_receipt_template":{"child_payload_sha256":"FILL_AFTER_MATERIALIZING_EXACT_SURFACE_PAYLOAD","source_payloads":[{"path":str(PARENT),"sha256":sha(PARENT)}],"surfaces":surfaces},"native_receipt_note":"Template only. Bind child_payload_sha256 to the exact materialized child NHTISS bytes before native loading; this file is not itself an executable anatomy receipt."}
pkl=OUT/"candidate060-adapted-source-parent-map-v2.pkl";pkl.write_bytes(pickle.dumps(out,protocol=5))
j=OUT/"candidate060-passive-tissue-parent-bindings-template-v2.json";j.write_text(json.dumps(out["native_receipt_template"],indent=2,sort_keys=True)+"\n")
r={"schema":"numi.human.left-fds132-fdp134-adapted-parent-map-delivery.v1","candidate_path":str(C),"candidate_sha256":sha(C),"candidate_exact_check_sha256":sha(CK),"step0_capture_remesh_report_sha256":sha(AUD),"map_path":str(pkl),"map_sha256":sha(pkl),"receipt_template_path":str(j),"receipt_template_sha256":sha(j),"surfaces":{s:{"vertices":len(exact_payload[s]["global_vertex_ids"]),"faces":len(exact_payload[s]["faces"]),"max_parent_simplex_terms":max(len(x) for x in exact_payload[s]["source_parent_terms_exact"])} for s in ("132","134")},"shared_retained_vertices":audit["map_adaptation"]["retained_shared_vertices"],"all_coefficients_positive_normalized":True,"native_receipt_bound":False,"elapsed_seconds_note":"fast map packaging; source SHA hashes streamed"}
rp=OUT/"candidate060-adapted-source-parent-map-report-v2.json";rp.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
print(json.dumps({"map":str(pkl),"map_sha256":sha(pkl),"receipt_template":str(j),"receipt_sha256":sha(j),"report":str(rp),"report_sha256":sha(rp),"surfaces":r["surfaces"]},sort_keys=True))
