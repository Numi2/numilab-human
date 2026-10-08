#!/usr/bin/env python3
import collections,hashlib,importlib.util,json,pathlib,struct
import numpy as np
E=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
OUT=E/"native-lung-seam-captured-envelope-953"
D=E/"native-lung-pleura-self-seam-diagnosis-952"
spec=importlib.util.spec_from_file_location("seam952",D/"analyze.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=json.loads((D/"report.json").read_text())
witness_path=E/"native-lung-seam-patch-linkage-933/report.json"
witness=json.loads(witness_path.read_text())
N=E/"lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy"
M=E/"lung-choroid-composition-924/choroid-v2/resting-anatomy-manifest.json"
PARAM=E/"native-terminal-cycle-931/skin-source-anatomy-parameters.bin"
GEOM=E/"native-terminal-cycle-931/accepted-geometry"
assert m.sha(N)==witness["inputs"]["source_nha"]["sha256"]==base["inputs"][str(N)]["sha256"]
params=PARAM.read_bytes();assert len(params)==1008
words=struct.unpack_from("<4I",params,0);assert words[0]==20
anchor=np.asarray(struct.unpack_from("<3f",params,16),dtype=np.float64)
volume,=struct.unpack_from("<f",params,28)
axis=np.asarray(struct.unpack_from("<3f",params,32),dtype=np.float64)
area,=struct.unpack_from("<f",params,104)
selected=witness["events"]
steps=sorted({int(e["step"]) for e in selected})
capture_step={s:(10000 if s==9999 else s) for s in steps}
rows=m.rows
def dot(a,b):return float(np.dot(a,b))
def point_triangle_distance(p,t):
 a,b,c=t;ab=b-a;ac=c-a;ap=p-a;d1=dot(ab,ap);d2=dot(ac,ap)
 if d1<=0 and d2<=0:return float(np.linalg.norm(ap))
 bp=p-b;d3=dot(ab,bp);d4=dot(ac,bp)
 if d3>=0 and d4<=d3:return float(np.linalg.norm(bp))
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return float(np.linalg.norm(p-(a+(d1/(d1-d3))*ab)))
 cp=p-c;d5=dot(ab,cp);d6=dot(ac,cp)
 if d6>=0 and d5<=d6:return float(np.linalg.norm(cp))
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return float(np.linalg.norm(p-(a+(d2/(d2-d6))*ac)))
 va=d3*d6-d5*d4
 if va<=0 and d4>=d3 and d5>=d6:
  w=(d4-d3)/((d4-d3)+(d5-d6));return float(np.linalg.norm(p-(b+w*(c-b))))
 den=va+vb+vc
 if den==0:return min(float(np.linalg.norm(p-q)) for q in (a,b,c))
 v=vb/den;w=vc/den
 return float(np.linalg.norm(p-(a+ab*v+ac*w)))
def segment_distance(p1,q1,p2,q2):
 d1=q1-p1;d2=q2-p2;r=p1-p2;a=dot(d1,d1);e=dot(d2,d2);f=dot(d2,r);tiny=1e-300
 if a<=tiny and e<=tiny:return float(np.linalg.norm(p1-p2))
 if a<=tiny:s=0.;t=min(1.,max(0.,f/e))
 else:
  c=dot(d1,r)
  if e<=tiny:t=0.;s=min(1.,max(0.,-c/a))
  else:
   b=dot(d1,d2);den=a*e-b*b
   s=min(1.,max(0.,(b*f-c*e)/den)) if den>tiny*a*e else 0.;t=(b*s+f)/e
   if t<0.:t=0.;s=min(1.,max(0.,-c/a))
   elif t>1.:t=1.;s=min(1.,max(0.,(b-c)/a))
 return float(np.linalg.norm((p1+s*d1)-(p2+t*d2)))
def tri_distance(a,b):
 d=min(point_triangle_distance(p,b) for p in a)
 d=min(d,min(point_triangle_distance(p,a) for p in b))
 for i in range(3):
  for j in range(3):d=min(d,segment_distance(a[i],a[(i+1)%3],b[j],b[(j+1)%3]))
 return d
results=[];pins={}
for requested_step in steps:
 step=capture_step[requested_step]
 pack=GEOM/f"step-{step}.mrvpack";receipt_path=GEOM/f"step-{step}.receipt.json"
 if not pack.is_file() or not receipt_path.is_file():raise FileNotFoundError(f"missing actual 931 capture {step}")
 rec=json.loads(receipt_path.read_text())
 if int(rec["accepted_step"])!=step:raise RuntimeError("receipt accepted step mismatch")
 body=next(x for x in rec["accepted_registered_body_poses"] if int(x["body_index"])==20)
 motion=np.asarray([rec["accepted_respiratory_motion"]["diaphragm_swept_volume_m3"],rec["accepted_respiratory_motion"]["rib_swept_volume_m3"]],dtype=np.float32)
 mp,stream,vo,surfs=m.pr.read_pack(pack)
 try:
  m.pr.validate_accepted_receipt(pack,receipt_path,step,mp,vo,surfs)
  pins[str(step)]={"pack_sha256":m.sha(pack),"receipt_sha256":m.sha(receipt_path),"accepted_time_s":rec["accepted_time_s"],"body20":body,"motion_f32":motion.tolist()}
  for old in [e for e in selected if int(e["step"])==requested_step]:
   owners=list(map(int,old["owners"]));face_ids=list(map(int,old["face_ids"]))
   source={};ideal={};replay={};actual={}
   for owner,fid in zip(owners,face_ids):
    row=rows[owner];ids=np.asarray(row["faces"][fid],dtype=np.int64)
    src=np.asarray(row["vertices6"][ids,:3],dtype=np.float32);source[owner]=src
    ideal[owner]=np.asarray([m.map64(p,m.cfg,axis,anchor,volume,area,motion,body) for p in src],dtype=np.float64)
    replay[owner]=np.asarray([m.map32(p,m.cfg,np.asarray(axis,dtype=np.float32),np.asarray(anchor,dtype=np.float32),np.float32(volume),np.float32(area),motion,body) for p in src],dtype=np.float32)
    packed_ids=surfs[(51023,owner)]["faces"][fid]
    actual[owner]=np.asarray([struct.unpack_from("<3f",mp,vo+int(i)*80) for i in packed_ids],dtype=np.float32)
   srel=m.hit(m.tri32(source[owners[0]]),m.tri32(source[owners[1]]),lattice=True)
   irel=m.hit(m.tri64(ideal[owners[0]]),m.tri64(ideal[owners[1]]))
   frel=m.hit(m.tri32(replay[owners[0]]),m.tri32(replay[owners[1]]),lattice=True)
   nrel=m.hit(m.tri32(actual[owners[0]]),m.tri32(actual[owners[1]]),lattice=True)
   eps={o:max(float(np.linalg.norm(actual[o][i].astype(np.float64)-ideal[o][i])) for i in range(3)) for o in owners}
   gap=0. if irel["hit"] else tri_distance(ideal[owners[0]],ideal[owners[1]])
   bound=eps[owners[0]]+eps[owners[1]]
   if nrel["hit"] and gap>bound+1e-15:raise RuntimeError(f"event outside envelope {step} {owners} {face_ids}")
   results.append({"witness_933_step":requested_step,"capture_931_step":step,"same_step_capture":requested_step==step,
    "owners":owners,"face_ids":face_ids,"witness_933_source_shared_coordinate_count":old.get("shared_source_coordinate_count"),
    "source_relation_931_NHA":srel,"ideal_map_relation_931_parameters":irel,"ideal_map_minimum_triangle_distance_m":gap,
    "f32_transcription_relation":frel,"actual_931_capture_relation":nrel,
    "per_triangle_max_captured_vs_ideal_vertex_error_m":{str(o):eps[o] for o in owners},
    "pairwise_hausdorff_displacement_bound_m":bound,"ideal_gap_within_bound":gap<=bound,
    "captured_intersection_is_unallowed":nrel["hit"],"captured_pack_sha256":pins[str(step)]["pack_sha256"],
    "captured_receipt_sha256":pins[str(step)]["receipt_sha256"],
    "source_vertex_max_error_m":max(float(np.linalg.norm(actual[o][i].astype(np.float64)-ideal[o][i])) for o in owners for i in range(3))})
 finally:
  stream.close();mp.close()
summary={"witness_event_count":len(results),"matched_step_count":sum(x["same_step_capture"] for x in results),
 "terminal_substitution_count":sum(not x["same_step_capture"] for x in results),
 "actual_931_exact_intersection_count":sum(x["actual_931_capture_relation"]["hit"] for x in results),
 "ideal_map_exact_intersection_count":sum(x["ideal_map_relation_931_parameters"]["hit"] for x in results),
 "relation_counts":{"native_and_ideal_hit":sum(x["actual_931_capture_relation"]["hit"] and x["ideal_map_relation_931_parameters"]["hit"] for x in results),
  "native_hit_ideal_disjoint":sum(x["actual_931_capture_relation"]["hit"] and not x["ideal_map_relation_931_parameters"]["hit"] for x in results),
  "native_and_ideal_disjoint":sum(not x["actual_931_capture_relation"]["hit"] and not x["ideal_map_relation_931_parameters"]["hit"] for x in results),
  "native_disjoint_ideal_hit":sum(not x["actual_931_capture_relation"]["hit"] and x["ideal_map_relation_931_parameters"]["hit"] for x in results)},
 "same_step_relation_counts":{"native_and_ideal_hit":sum(x["same_step_capture"] and x["actual_931_capture_relation"]["hit"] and x["ideal_map_relation_931_parameters"]["hit"] for x in results),
  "native_hit_ideal_disjoint":sum(x["same_step_capture"] and x["actual_931_capture_relation"]["hit"] and not x["ideal_map_relation_931_parameters"]["hit"] for x in results)},
 "terminal_substitution_relation_counts":{"native_and_ideal_hit":sum(not x["same_step_capture"] and x["actual_931_capture_relation"]["hit"] and x["ideal_map_relation_931_parameters"]["hit"] for x in results),
  "native_and_ideal_disjoint":sum(not x["same_step_capture"] and not x["actual_931_capture_relation"]["hit"] and not x["ideal_map_relation_931_parameters"]["hit"] for x in results)},
 "pair_envelope_covers_all_actual_intersections":sum(x["ideal_gap_within_bound"] for x in results if x["actual_931_capture_relation"]["hit"])==sum(x["actual_931_capture_relation"]["hit"] for x in results),
 "max_ideal_disjoint_gap_m":max(x["ideal_map_minimum_triangle_distance_m"] for x in results),
 "max_pairwise_hausdorff_displacement_bound_m":max(x["pairwise_hausdorff_displacement_bound_m"] for x in results),
 "max_actual_vs_ideal_vertex_error_m":max(x["source_vertex_max_error_m"] for x in results),
 "counts_by_step":{str(s):{"event_count":sum(x["capture_931_step"]==s for x in results),"actual_hits":sum(x["capture_931_step"]==s and x["actual_931_capture_relation"]["hit"] for x in results),"ideal_hits":sum(x["capture_931_step"]==s and x["ideal_map_relation_931_parameters"]["hit"] for x in results)} for s in sorted(set(x["capture_931_step"] for x in results))}}
out={"schema":"numi.human.native-931-selected-lung-seam-displacement-envelope.v1","status":"complete_selected_prior_witnesses",
 "scope":"93 pair/phase witnesses from the earlier 925 report 933 were rechecked only at their matching accepted 931 phase capture; six step-9999 witnesses use the 931 step-10000 terminal capture because 931 has no step-9999 pack. This is not a full all-pairs scan or a carry-forward of 925 intersections.",
 "inputs":{"witness_report_933":{"path":str(witness_path),"sha256":m.sha(witness_path)},"witness_scan_938":{"path":str(E/"native-lung-seam-patch-linkage-938/report.json"),"sha256":m.sha(E/"native-lung-seam-patch-linkage-938/report.json")},"source_nha":{"path":str(N),"sha256":m.sha(N)},"manifest":{"path":str(M),"sha256":m.sha(M)},"anatomy_parameters":{"path":str(PARAM),"sha256":m.sha(PARAM)},"source_map_analysis_952":{"path":str(D/"analyze.py"),"sha256":m.sha(D/"analyze.py")},"captured_displacement_envelope_952":{"path":str(D/"captured-displacement-envelope.json"),"sha256":m.sha(D/"captured-displacement-envelope.json")},"native_931_captures":pins},
 "method":{"selection":"The 93 entries are selected by a previous exact 925 scan (933); this check verifies only those existing owner/face/phase witnesses against actual 931 packs and does not discover new pairs.","map":"Frozen source map replay is evaluated per actual 931 accepted pose, using captured float32 anatomy parameters, respiratory motion, and body_index 20 pose. The same mapped source face vertices are compared to exact actual pack vertices.","bound":"Maximum corresponding-vertex displacement for each triangle bounds equal-barycentric point displacement; the pair bound is the sum of its two triangle bounds. This is a per-captured-state geometry envelope, not a formal shader error bound.","step_mismatch":"933's selected step 9999 witnesses are evaluated on step 10000 because 931 only retains step 10000 at that terminal time; these six are explicitly not same-step comparisons.","limits":["Selected prior witnesses only; no full-scene or full-pair scan was repeated.","The envelope is not a formal upper bound over shader arithmetic and provides no continuous-time guarantee.","Exact captured intersections remain unallowed; source or topology is not modified."]},
 "summary":summary,"events":results}
dst=OUT/"report.json";dst.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
print(json.dumps(summary,sort_keys=True,indent=2));print("report_sha256",m.sha(dst))
