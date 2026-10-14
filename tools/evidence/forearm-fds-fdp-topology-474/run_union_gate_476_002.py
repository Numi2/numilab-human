#!/usr/bin/env python3
import hashlib,json,pathlib,signal,sys,time,traceback
import numpy as np
BASE=pathlib.Path("/Users/n/numi-human-fds132-m6-1350")
AUDIT=pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/forearm470-actual-gpu-self-gate-001")
PUB=pathlib.Path("/Users/n/numi-human-final-integration-publish-1397/src")
ATTEMPT=BASE/"attempt-476-fds-fdp-union-material-gate-001"
OUT=ATTEMPT/"result-002"
PIN_JSON=ATTEMPT/"frozen.pins.json"
PIN_JSON_SHA="53f3308df9b831c3ed373eff65663716109f942a04247347cd3a097832952f3d"
CAP=BASE/"attempt-474-collapse-neighborhood-edgeflip-001/candidate-474-trial-02.npz"
LIN474=BASE/"attempt-474-collapse-neighborhood-edgeflip-001/face-lineage-474-trial-02.json"
HAND=BASE/"attempt-471-native-material-handoff-001/result-001"
LIN471=HAND/"forearm-native-material-lineage.json"
LIN466=BASE/"attempt-466-local-shared-edge-classification-001/result-001/joint-cross-lineage-466-v7.json"
FDSCAP=AUDIT/"row-132-actual-gpu.npz"; FDPCAP=AUDIT/"row-134-actual-gpu.npz"
H132=HAND/"row-132-fds-material-handoff.npz"; H134=HAND/"row-134-fdp-material-handoff.npz"
AV_PATH=PUB/"numilab_human/resting_anatomy_verify.py"
CI_PATH=PUB/"numilab_human/cardiac_cavity_intersections.py"
EXPECTED_AV="2c3ed5799d73d3076b5e6c8497b1095efad31419691c0469535bb7ef47983015"
EXPECTED_CI="4e6fb1f4ee75ea0ba9cdd06a072702a238f066fc8e63a61aa9eae32f2356a384"
sys.path.insert(0,str(PUB))
from numilab_human import resting_anatomy_verify as av
from numilab_human import cardiac_cavity_intersections as ci
t0=time.monotonic()
report={"schema":"numi.human.forearm.476.actual-gpu-fds-fdp-union-material-gate.v1",
 "status":"running","actual_frame":"accepted native one-step capture step 0, actual emitted Float32 positions",
 "typed_identities":[[51005,132],[51005,134]],
 "candidate_topology":"474 trial 2 topology-only edge flip; points and 21-core material rows unchanged",
 "started_unix":time.time()}
if OUT.exists(): raise RuntimeError(f"result directory already exists: {OUT}")
OUT.mkdir(parents=True,exist_ok=False); rp=OUT/"report.json"
def flush():
 report["elapsed_s"]=time.monotonic()-t0
 tmp=rp.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");tmp.replace(rp)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jread(p):return json.loads(pathlib.Path(p).read_text())
def canonical(p):return str(pathlib.Path(p).resolve())
def norm(x):
 if isinstance(x,dict):
  if set(x)=={"numerator","denominator"}:
   n,d=int(x["numerator"]),int(x["denominator"]);return n if d==1 else float(n)/float(d)
  return {k:norm(v) for k,v in x.items()}
 if isinstance(x,list):return [norm(v) for v in x]
 return x
def norm3(x):
 r=norm(x)
 if "material_mapping_kind" in r:r["material_mapping_kind"]=int(r["material_mapping_kind"])
 return r
def mats(z):
 ids=np.asarray(z["core_weight_gids"],dtype=np.int64);o=np.argsort(ids,kind="stable")
 return ids[o],np.asarray(z["core_weights_float64"],dtype=np.float64)[o]
def parity(x):
 return ((x[:,0]>x[:,1]).astype(np.int8)+(x[:,0]>x[:,2])+(x[:,1]>x[:,2]))%2
def keyed_rows(ids,values,query,label):
 ids=np.asarray(ids,dtype=np.int64); values=np.asarray(values)
 order=np.argsort(ids,kind="stable"); ordered=ids[order]
 query=np.asarray(query,dtype=np.int64)
 at=np.searchsorted(ordered,query)
 if np.any(at>=len(ordered)) or not np.array_equal(ordered[at],query):
  raise RuntimeError(f"{label}: missing queried GIDs")
 return values[order[at]]
def pyvalue(x):
 if isinstance(x,dict): return {str(k):pyvalue(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [pyvalue(v) for v in x]
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 return x
def alarm(_s,_f):
 report["status"]="timeout_1200s";report["failure_stage"]=report.get("stage");flush()
 raise TimeoutError("1200-second CPU bound reached")
signal.signal(signal.SIGALRM,alarm);signal.alarm(1200)
try:
 report["stage"]="pin_inputs"
 if sha(PIN_JSON)!=PIN_JSON_SHA: raise RuntimeError("frozen pin manifest SHA mismatch")
 manifest=jread(PIN_JSON)
 if manifest.get("schema")!="numi.human.forearm.476.frozen-input-pins.v1" or manifest.get("typed_identities")!=[[51005,132],[51005,134]]:
  raise RuntimeError("frozen pin manifest schema/typed identity mismatch")
 frozen={}
 for row in manifest["inputs"]:
  p=pathlib.Path(row["path"])
  if not p.is_file(): raise RuntimeError(f"pinned input missing: {p}")
  actual=sha(p)
  if actual!=row["sha256"] or p.stat().st_size!=row["size_bytes"]:
   raise RuntimeError(f"pinned input changed: {p}")
  frozen[str(p)]=actual
 if frozen[str(AV_PATH)]!=EXPECTED_AV or frozen[str(CI_PATH)]!=EXPECTED_CI:
  raise RuntimeError("frozen HP97 verifier/CI hash mismatch")
 if canonical(av.__file__)!=canonical(AV_PATH) or canonical(ci.__file__)!=canonical(CI_PATH):
  raise RuntimeError("mixed import namespace")
 capmeta=jread(AUDIT/"report.json")
 stages={int(x["row"]):x for x in capmeta["stages"] if x.get("stage")=="preparation"}
 for sid in (132,134):
  if stages[sid]["diagnostics"].get("identity")!=[51005,sid]:
   raise RuntimeError(f"actual capture report typed identity mismatch for row {sid}")
 frozen_by_canonical={canonical(k):v for k,v in frozen.items()}
 capture_pins_by_canonical={canonical(k):v for k,v in capmeta.get("pins",{}).items()}
 for p in (AUDIT.parent/"run_forearm470_actual_gpu_self_gate_001.py",
           pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/joint-native-emission-diagnostic-001/invocation.json"),
           pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.mrvpack"),
           pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.receipt.json")):
  key=canonical(p)
  if capture_pins_by_canonical.get(key)!=frozen_by_canonical.get(key):
   raise RuntimeError(f"actual capture report does not bind canonical pinned provenance: {p}")
 report["sha256"]=frozen
 report["pin_manifest_sha256"]=PIN_JSON_SHA
 report["verifier"]={"resting_anatomy_verify_sha256":EXPECTED_AV,"cardiac_ci_sha256":EXPECTED_CI}
 report["native_capture_provenance"]={
  "accepted_pack_sha256":frozen["/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.mrvpack"],
  "accepted_receipt_sha256":frozen["/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/joint-native-emission-diagnostic-001/native-run/accepted-geometry/step-0.receipt.json"],
  "capture_report_status":capmeta.get("status"),
  "capture_geometry_basis":capmeta.get("geometry_basis")}
 flush()
 report["stage"]="load_and_material_preflight"
 c=np.load(CAP,allow_pickle=False);f=np.load(FDSCAP,allow_pickle=False);p=np.load(FDPCAP,allow_pickle=False)
 h1=np.load(H132,allow_pickle=False);h2=np.load(H134,allow_pickle=False)
 l471=jread(LIN471);l466=jread(LIN466);l474=jread(LIN474)
 capmeta=jread(AUDIT/"report.json")
 stages={int(x["row"]):x for x in capmeta["stages"] if x.get("stage")=="preparation"}
 for sid,z in ((132,f),(134,p)):
  digest=hashlib.sha256(np.ascontiguousarray(z["positions"],dtype=np.float32).tobytes()).hexdigest()
  if digest!=stages[sid]["positions_sha256"]:
   raise RuntimeError(f"actual GPU position bytes differ from pinned stage report for {sid}")
 fids=np.asarray(f["gids"],dtype=np.int64);pids=np.asarray(p["gids"],dtype=np.int64)
 ffaces=np.asarray(f["faces_local"],dtype=np.int64)
 cf_gid=np.asarray(c["gids"],dtype=np.int64)[np.asarray(c["faces_local"],dtype=np.int64)]
 active=np.isin(cf_gid,pids).all(axis=1)
 if not active.all():raise RuntimeError(f"candidate faces without actual GPU vertices: {int((~active).sum())}")
 active_face_ids=np.flatnonzero(active);porder=np.argsort(pids,kind="stable")
 pfaces=porder[np.searchsorted(pids[porder],cf_gid[active])].astype(np.int64)
 if not np.array_equal(pids[pfaces],cf_gid[active]):raise RuntimeError("candidate GID lookup failed")
 fgids=fids[ffaces];pgids=cf_gid[active]
 if len(l466["FDS_face_ancestry"])!=len(ffaces):raise RuntimeError("FDS face ancestry cardinality mismatch")
 if len(l474["rows"])!=len(pfaces) or any(int(x["output_face_index"])!=i for i,x in enumerate(l474["rows"])):
  raise RuntimeError("474 face ancestry order/cardinality mismatch")
 if not np.array_equal(active_face_ids,np.arange(len(active_face_ids))):raise RuntimeError("non-prefix active faces need explicit lineage subsetting")
 sg,ia,ib=np.intersect1d(fids,pids,return_indices=True)
 posbad=np.any(f["positions"][ia].view(np.uint32)!=p["positions"][ib].view(np.uint32),axis=1)
 if posbad.any():raise RuntimeError(f"actual positions differ on {int(posbad.sum())} common active GIDs")
 id1,w1=mats(h1);id2,w2=mats(h2)
 if not np.array_equal(h1["core_body_indices"],h2["core_body_indices"]):raise RuntimeError("ordered body index lists differ")
 if (np.asarray(h1["core_body_indices"]).shape!=(21,) or np.asarray(h2["core_body_indices"]).shape!=(21,)
     or w1.ndim!=2 or w2.ndim!=2 or w1.shape[1]!=21 or w2.shape[1]!=21):
  raise RuntimeError("handoff does not carry the ordered 21-core field")
 if not np.array_equal(np.sort(np.asarray(h1["gids"],dtype=np.int64)),id1) or not np.array_equal(np.sort(np.asarray(h2["gids"],dtype=np.int64)),id2):
  raise RuntimeError("handoff GID and core_weight_gids coverage differs")
 used_w1=keyed_rows(id1,w1,fids,"FDS used GID weights")
 used_w2=keyed_rows(id2,w2,pids,"FDP used GID weights")
 invalid1=np.any(~np.isfinite(used_w1),axis=1)|np.any(used_w1<=0,axis=1)|(np.abs(used_w1.sum(axis=1)-1)>1e-12)
 invalid2=np.any(~np.isfinite(used_w2),axis=1)|np.any(used_w2<=0,axis=1)|(np.abs(used_w2.sum(axis=1)-1)>1e-12)
 if invalid1.any() or invalid2.any():
  raise RuntimeError(f"used 21-core rows not finite, strictly positive, normalized: FDS={int(invalid1.sum())}, FDP={int(invalid2.sum())}")
 if not np.array_equal(np.sort(np.asarray(h1["gids"],dtype=np.int64)),id1) or not np.array_equal(np.sort(np.asarray(h2["gids"],dtype=np.int64)),id2):
  raise RuntimeError("handoff GID and core_weight_gids coverage differs")
 source1=keyed_rows(h1["gids"],h1["source_positions_f32"],sg,"FDS source positions")
 source2=keyed_rows(h2["gids"],h2["source_positions_f32"],sg,"FDP source positions")
 sourcebad=np.any(source1.view(np.uint32)!=source2.view(np.uint32),axis=1)
 if sourcebad.any():raise RuntimeError(f"shared keyed source positions differ on {int(sourcebad.sum())} GIDs")
 gc=np.intersect1d(id1,id2);activec=np.intersect1d(sg,gc)
 j1=np.searchsorted(id1,activec);j2=np.searchsorted(id2,activec)
 wdiff=np.any(w1[j1].view(np.uint64)!=w2[j2].view(np.uint64),axis=1)
 badw=np.any(w1[j1]<=0,axis=1)|np.any(w2[j2]<=0,axis=1)|(np.abs(w1[j1].sum(axis=1)-1)>1e-12)|(np.abs(w2[j2].sum(axis=1)-1)>1e-12)
 if wdiff.any() or badw.any():raise RuntimeError(f"keyed active material gate failed: mismatch={int(wdiff.sum())},invalid={int(badw.sum())}")
 sf=np.sort(fgids,axis=1);sp=np.sort(pgids,axis=1)
 uf,fi,fc=np.unique(sf,axis=0,return_index=True,return_counts=True)
 up,pi,pc=np.unique(sp,axis=0,return_index=True,return_counts=True)
 dt=np.dtype([("a","<i8"),("b","<i8"),("c","<i8")])
 _,ixf,ixp=np.intersect1d(np.ascontiguousarray(uf).view(dt).ravel(),np.ascontiguousarray(up).view(dt).ravel(),return_indices=True)
 frows=fi[ixf];prows=active_face_ids[pi[ixp]]
 facekeys=uf[ixf]
 badface=(fc[ixf]!=1)|(pc[ixp]!=1)|(parity(fgids[frows])==parity(pgids[prows]))
 if badface.any():raise RuntimeError(f"invalid reciprocal face rows: {int(badface.sum())}")
 # Verify exact weight equality and positivity for each shared face vertex.
 fgw=np.searchsorted(id1,np.unique(fgids[frows]))
 pgw=np.searchsorted(id2,np.unique(fgids[frows]))
 fvals=w1[fgw];pvals=w2[pgw]
 if not np.array_equal(fvals.view(np.uint64),pvals.view(np.uint64)):raise RuntimeError("reciprocal face material rows differ")
 report["material_and_interface_preflight"]={
  "FDS_actual_vertices":len(fids),"FDS_faces":len(ffaces),"FDP_actual_vertices":len(pids),
  "FDP_candidate_faces":len(pfaces),"common_active_GIDs":len(sg),
  "actual_position_byte_mismatches":int(posbad.sum()),"keyed_source_position_byte_mismatches":int(sourcebad.sum()),"keyed_material_row_mismatches":int(wdiff.sum()),
  "invalid_positive_normalized_shared_material_rows":int(badw.sum()),
  "ordered_body_indices_equal":True,"shared_reciprocal_face_pairs":len(frows),
  "shared_face_duplicate_or_winding_errors":int(badface.sum()),"material_join":"keyed by core_weight_gids"}
 flush()
 report["stage"]="assemble_source_lineage_and_prepare"
 births466={int(k):norm3(v) for k,v in l466["new_vertices"].items()}
 b132={int(k):norm3(v) for k,v in l471["rows"]["132"].get("new_vertices",{}).items()}
 b134={}
 for cat,rows in l471["rows"]["134"].get("birth_lineage",{}).items():
  if isinstance(rows,dict):
   for k,v in rows.items():
    if isinstance(v,dict) and len(v.get("core_weights",[]))==21:b134[int(k)]=norm3(v)
 vf={};vp={}
 for gid in map(int,fids):
  vf[gid]=births466.get(gid,b132.get(gid,{"source_gid":gid,"source_vertex_kind":"retained_source_gid"}))
 for gid in map(int,pids):
  vp[gid]=births466.get(gid,b134.get(gid,{"source_gid":gid,"source_vertex_kind":"retained_source_gid"}))
 conflicts=[int(g) for g in sg if json.dumps(vf[int(g)],sort_keys=True,separators=(",",":"))!=json.dumps(vp[int(g)],sort_keys=True,separators=(",",":"))]
 if conflicts:raise RuntimeError(f"shared birth lineage mismatch: {conflicts[:8]} ({len(conflicts)} total)")
 unsupported_births=[int(g) for g in sg if int(g) not in vf or int(g) not in vp or not av._has_source_vertex_lineage(vf[int(g)]) or not av._has_source_vertex_lineage(vp[int(g)])]
 if unsupported_births:raise RuntimeError(f"shared GIDs lack supported source birth/alias lineage: {unsupported_births[:16]} ({len(unsupported_births)} total)")
 kind3_mismatch=[]
 for gid,j in zip(activec,j1):
  row=vf[int(gid)]
  if row.get("material_mapping_kind")==3 and (row.get("material_binding")!="independent_existing_core_carrier" or tuple(map(int,row.get("core_body_indices",[])))!=tuple(map(int,h1["core_body_indices"])) or not np.array_equal(np.asarray(row.get("core_weights",[]),dtype=np.float64),w1[j])):
   kind3_mismatch.append(int(gid))
 for gid,j in zip(activec,j2):
  row=vp[int(gid)]
  if row.get("material_mapping_kind")==3 and (row.get("material_binding")!="independent_existing_core_carrier" or tuple(map(int,row.get("core_body_indices",[])))!=tuple(map(int,h2["core_body_indices"])) or not np.array_equal(np.asarray(row.get("core_weights",[]),dtype=np.float64),w2[j])):
   kind3_mismatch.append(int(gid))
 if kind3_mismatch:raise RuntimeError(f"shared kind3 birth owner rows differ from the actual ordered owner table: {kind3_mismatch[:16]} ({len(kind3_mismatch)} total)")
 fullpairs=list(zip(map(int,frows),map(int,prows)))
 first=av.prepare_surface((51005,132),np.asarray(f["positions"],dtype=np.float32),ffaces,
  normals=np.asarray(f["normals"],dtype=np.float32),source_vertex_ids=fids,
  source_face_lineage=l466["FDS_face_ancestry"],source_vertex_lineage=vf,
  ordered_core_body_indices=tuple(map(int,h1["core_body_indices"])),core_weight_gids=h1["core_weight_gids"],core_weights_float64=h1["core_weights_float64"])
 second=av.prepare_surface((51005,134),np.asarray(p["positions"],dtype=np.float32),pfaces,
  normals=np.asarray(p["normals"],dtype=np.float32),source_vertex_ids=pids,
  source_face_lineage=[l474["rows"][int(i)] for i in active_face_ids],source_vertex_lineage=vp,
  ordered_core_body_indices=tuple(map(int,h2["core_body_indices"])),core_weight_gids=h2["core_weight_gids"],core_weights_float64=h2["core_weights_float64"])
 bad_fds_face_lineage=[i for i,row in enumerate(first.source_face_lineage) if not av._has_source_face_lineage(row)]
 bad_fdp_face_lineage=[i for i,row in enumerate(second.source_face_lineage) if not av._has_source_face_lineage(row)]
 report["source_lineage_preflight"]={"shared_GIDs_with_supported_birth_lineage":len(sg)-len(unsupported_births),
  "FDS_faces_with_source_lineage":len(first.source_face_lineage)-len(bad_fds_face_lineage),
  "FDS_faces_missing_source_lineage":len(bad_fds_face_lineage),
  "FDP_faces_with_source_lineage":len(second.source_face_lineage)-len(bad_fdp_face_lineage),
  "FDP_faces_missing_source_lineage":len(bad_fdp_face_lineage)}
 if bad_fds_face_lineage or bad_fdp_face_lineage:raise RuntimeError("active face lacks supported source-face ancestry")
 report["prepared_surface_diagnostics"]={"FDS":first.diagnostics(),"FDP":second.diagnostics()}
 if not first.closed or not second.closed:
  report["status"]="failed_surface_preparation";report["failure_stage"]="referenced_surface_not_closed";flush()
  raise RuntimeError("actual emitted reference surfaces not closed")
 flush()
 report["stage"]="build_shared_feature_declarations"
 # Candidate declarations are keyed by actual shared GID incidence. Keep only
 # conservative triangle-AABB overlaps; exact intersection is still proved by HP97.
 f_by_gid={}
 p_by_gid={}
 for fi0,tri in enumerate(fgids):
  for gid in set(map(int,tri)): f_by_gid.setdefault(gid,[]).append(fi0)
 for pi0,tri in enumerate(pgids):
  for gid in set(map(int,tri)): p_by_gid.setdefault(gid,[]).append(pi0)
 fullset={(int(a),int(b)) for a,b in fullpairs}
 pair_gids={}
 for gid in sg:
  aa=f_by_gid.get(int(gid),());bb=p_by_gid.get(int(gid),())
  for ai in aa:
   for bi in bb:
    key=(int(ai),int(bi))
    if key not in fullset: pair_gids.setdefault(key,set()).add(int(gid))
 candidate_keys=[key for key,gids in pair_gids.items() if len(gids) in (1,2)]
 if candidate_keys:
  candidate_arr=np.asarray(candidate_keys,dtype=np.int64)
  fa=ffaces[candidate_arr[:,0]];pb=pfaces[candidate_arr[:,1]]
  amin=f["positions"][fa].min(axis=1);amax=f["positions"][fa].max(axis=1)
  bmin=p["positions"][pb].min(axis=1);bmax=p["positions"][pb].max(axis=1)
  aabb=np.all((amin<=bmax)&(bmin<=amax),axis=1)
  candidate_arr=candidate_arr[aabb]
 else:
  candidate_arr=np.empty((0,2),dtype=np.int64)
 feature_declarations=[]
 feature_flat=[];feature_offsets=[0]
 for ai,bi in candidate_arr:
  gids=sorted(pair_gids[(int(ai),int(bi))])
  feature_declarations.append({"first_identity":[51005,132],"second_identity":[51005,134],
   "first_face_index":int(ai),"second_face_index":int(bi),"feature_gids":gids})
  feature_flat.extend(gids);feature_offsets.append(len(feature_flat))
 np.savez_compressed(OUT/"interface-declaration-input.npz",
  reciprocal_full_face_pairs=np.asarray(fullpairs,dtype=np.int64).reshape((-1,2)),
  feature_face_pairs=candidate_arr,
  feature_gids=np.asarray(feature_flat,dtype=np.int64),
  feature_offsets=np.asarray(feature_offsets,dtype=np.int64))
 report["interface_declarations"]={"reciprocal_full_face_pairs":len(fullpairs),
  "shared_gid_feature_candidates_after_conservative_aabb":len(feature_declarations),
  "feature_candidates_with_one_or_two_shared_GIDs":len(feature_declarations),
  "feature_candidate_GID_count":len(feature_flat),
  "candidate_file_sha256":sha(OUT/"interface-declaration-input.npz"),
  "proof":"actual typed surfaces, exact shared GID, keyed actual 21-core owner row, keyed source lineage, byte-identical emitted F32 point, and exact triangle intersection must agree"}
 flush()
 report["stage"]="exact_shared_interface_union_and_containment"
 result=av.audit_surface_pair(first,second,intended_face_pairs=fullpairs,
  intended_shared_feature_pairs=feature_declarations)
 unexpected=np.asarray(result.get("unexpected_triangle_pairs",[]),dtype=np.int64).reshape((-1,2))
 rejected=result.get("rejected_interface_feature_contacts",[])
 rejected_hits=[x for x in rejected if x.get("reason")!="declared_pair_has_no_exact_intersection"]
 declared=result.get("declared_interface_triangle_pairs",[])
 feature_hits=result.get("declared_interface_feature_contacts",[])
 fullmatch=len(declared)==len(fullpairs)
 np.savez_compressed(OUT/"exact-union-witnesses.npz",
  unexpected_pairs=unexpected,
  unexpected_first_face_gids=(fgids[unexpected[:,0]] if len(unexpected) else np.empty((0,3),dtype=np.int64)),
  unexpected_second_face_gids=(pgids[unexpected[:,1]] if len(unexpected) else np.empty((0,3),dtype=np.int64)))
 result_path=OUT/"exact-union-result.json"
 result_path.write_text(json.dumps(pyvalue(result),indent=2,sort_keys=True)+"\n")
 json.loads(result_path.read_text())
 contained=result.get("containment",[])
 unresolved=(len(contained)!=2 or any(x.get("status")!="checked_exact_rational_metres" for x in contained))
 inside=sum(len(x.get("inside_component_witnesses",[])) for x in contained)
 report["exact_union"]={"exact_triangle_pair_count":int(result.get("triangle_pair_count",0)),
  "reciprocal_full_face_pairs_expected":len(fullpairs),
  "reciprocal_full_face_pairs_exact":len(declared),
  "shared_gid_feature_candidates":len(feature_declarations),
  "accepted_exact_feature_contacts":len(feature_hits),
  "nonhit_feature_declarations":sum(x.get("reason")=="declared_pair_has_no_exact_intersection" for x in rejected),
  "rejected_exact_feature_contacts":len(rejected_hits),
  "unexpected_exact_triangle_pairs":int(len(unexpected)),
  "unexpected_pair_witness_file":"exact-union-witnesses.npz",
  "full_helper_result_file":"exact-union-result.json",
  "containment_count":len(contained),
  "containment_statuses":[x.get("status") for x in contained],
  "inside_component_witness_count":int(inside),
  "containment_unresolved":bool(unresolved),
  "helper_passed":bool(result.get("passed",False)),
  "root_gate":"all full reciprocal faces declared from keyed GIDs and positive actual owners; all remaining contacts must be exact supported vertex/edge features; both-direction closed-solid containment must resolve with no inside witness"}
 gate=(bool(result.get("passed",False)) and fullmatch and not rejected_hits and len(unexpected)==0 and len(contained)==2 and not unresolved and inside==0)
 report["status"]="passed_frame0_union" if gate else "rejected_union_or_containment"
 report["final_gate"]={"frame0_union_clear":bool(gate),
  "full_reciprocal_face_set_exact":bool(fullmatch),
  "unsupported_feature_contacts":len(rejected_hits),
  "unexpected_exact_pairs":int(len(unexpected)),
  "containment_unresolved":bool(unresolved),
  "inside_component_witness_count":int(inside),
  "scope":"frame 0 only; later accepted poses, bone/PQ/skin/nearby targets and native qualification remain required"}
 report["finished_unix"]=time.time();flush()
except Exception as e:
 if report.get("status")=="running":report["status"]="failed";report["failure_stage"]=report.get("stage")
 report["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
 flush()
 raise
finally:signal.alarm(0)

