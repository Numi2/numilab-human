#!/usr/bin/env python3
"""Exact nine-pose offline audit of the unadmitted 1218 source reduction."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,os,struct,sys,time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
import numpy as np
E=Path("/Users/n/numi-human-resting-evidence-20261005");R=Path("/Users/n/numi-human-retained-delivery-20261009")
B=R/"skin-resting-multipose-clearance-1218";FIT=B/"fit-attempt-001";R4=B/"local-self-reduction-004";R5=B/"local-self-reduction-005"
OUT=B/"local-self-reduction-audit-001/scan-001"
SKIN=E/"native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
NHA=E/"native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"
OLD=E/"common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin"
ORIENT=E/"local-skin-orientation-892/local-orientation-report.json";SCENE=E/"common-atlas-skin-composition-907/resting-scene/resting-supine-scene.manifest.json"
INV=E/"native-complete-skin-containment-audit-890/pair-summary-v3.csv"
ROOT=Path("/Users/n/numi-human-conforming-composition-source-1216");REV=ROOT/"source-revision.json"
CLEAR=ROOT/"src/numilab_human/common_atlas_skin_clearance.py";CARD=ROOT/"src/numilab_human/cardiac_cavity_intersections.py"
FORWARD=E/"native-common-skin-multipose-forward-model-915.py"
RUNNER=E/"native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py"
INDEX=R/"skin-resting-multipose-clearance-1206/prepared-index-predicate-comparison.json"
INDEX_EQ=R/"skin-prepared-first-equivalence-1210-attempt003/equivalence-report.json"
NATIVE=R/"muscle-conforming-refinement-1216/native-refinement-1217-attempt2/native-run"
HELD=R/"native-lung1178-thumb1187-smoke-1191/native-run"
CAND=FIT/"candidate-source-positions-f32.npy";START=FIT/"starting-source-positions-f32.npy";PREF=FIT/"preflight.json";FITREP=FIT/"candidate-report.json"
P4=R4/"unadmitted-source-proposal.npy";Q4=R4/"feasibility.json";S4=R4/"feasibility.py"
P5=R5/"unadmitted-source-proposal.npy";Q5=R5/"feasibility.json";S5=R5/"feasibility.py"
R6=B/"local-self-reduction-006";P6=R6/"unadmitted-source-proposal.npy";Q6=R6/"feasibility.json";S6=R6/"feasibility.py"
A5=R5/"constraint-matrix.npz";B5=R5/"constraint-rhs.npy";L5=R5/"constraint-labels.json"
STEPS=[0,9983,20000,4991,5375,5759,6111,6495,7743]
POSES=[(s,NATIVE if s in (0,9983,20000) else HELD,"1217_native_refinement" if s in (0,9983,20000) else "1191_retained_holdout") for s in STEPS]
RIGHT=(49688,49720,49721,49723,49759,49760,49761,49796);LEFT=(60436,60437,60478,60479,60515,60516)
RESP_ACTUAL=ROOT/"src/numilab_human/resting_respiratory_conforming_field.py"
CLOSED={(51005,63):6688,(51005,64):6690};NHA_SHA="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
PINS={
 Q6:"cebee14945800d8f41a55c19fcedc33d69b805dbe51c4355ea99723f04fa5e16",
 S6:"6c062737cd7247273be5877de9f394e1544b961b65b673aae7b3038eb1c71316",
 P6:"2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258",
 RESP_ACTUAL:"980b368d8de0cc6cd7e25a93e61e5b6f304275dbd18511207da8cc9919410bb5",
 SKIN:"b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b",NHA:NHA_SHA,
 OLD:"c98e72605f78dad832d5106283f0cb0dca9564d50b1873fc78c883476a093392",
 ORIENT:"4380f2e36a759f4f8f79da1d6f06f5f0011980c2ff51ffb4f199ffb45a034be9",
 SCENE:"8802260236813440c200c7723bd78a36112f7722b007f4160d0ac9c5aaf16f36",
 INV:"a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3",
 CAND:"48464230377c811f2cb313a13a8dbcbf7b50d008c2cec18fe4ef59f258d92e35",
 START:"365cfa072fee6e7ba868adf0c3eaa7659b69a2c50d6dbaca9a8cea8b8bbdad3e",
 PREF:"b9f3a555e054cb6afc9f3aa2c55d955270e4163744360d00815d2cdd57dcd630",
 FITREP:"c497fcc209ba183761cf97425bc060ee91fa7767b98234e433dd0d26f8f05265",
 FORWARD:"81a5532f634278a45aa9aaf1457d0fdb3be71906256f3490941a4885b74d4cdc",
 REV:"51e3a732e18fbdb6922871b435d98f42d01c00511d2d08a0fc4cac1009faba6f",
 CLEAR:"ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f",
 CARD:"934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4",
 INDEX:"44245299282ce64cd6f118d35b75f4bf5cb250f20853363b08b2392ec4694c98",
 INDEX_EQ:"0714743b609f2f540998afefb658a91cd0ccc1e0798111eb5ac545344e0abaa3",
 RUNNER:"b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb",
 S4:"dda535152cc285b4e47ef1738400d6276e6061a6de89788a7574e76b37aad0ef",
 Q4:"dd053cb43d5e992a36385baa0552573ffed84e70cb42fb3025ed21790a57afc6",
 P4:"2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258",
 S5:"eda511bd8fd49fe5c7e997180c4afbea15fd0bf5a03b4f43948633c958550b8c",
 Q5:"3d679bd05d9a215347cf50c369eaebc31a614808bb9916bc179f0f2e89dbf03c",
 P5:"2a7b416c166b034030c84ac5098dcb5aa6735573726e250e92bdd7a21d2ab258",
 A5:"7ab50d714411cc96eec8f1a8e0f0047d8e9a135230744d7b39e401cb834bbb83",
 B5:"6764c2fb61dcae5b7c2cb38aaef92b5700d282079552de365287f5cafd95b09f",
 L5:"f25d03c630e958c64490d8fa15ccb2780ad79f012cfaad1a736c11879917bc0e"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for z in iter(lambda:f.read(4*1024*1024),b""):h.update(z)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve()
 if not p.is_file() or p.is_symlink():raise RuntimeError("missing/symlink input "+str(p))
 return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def j(p):return json.loads(Path(p).read_text())
def wj(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
def req(ok,msg):
 if not ok:raise RuntimeError(msg)
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p)
 req(s is not None and s.loader is not None,"cannot import "+str(p))
 m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def skinread(p):
 b=Path(p).read_bytes();magic,abi,nb,nv,ni,fp,ar=struct.unpack_from("<8s5I32s",b);vo=60+36*nb;io=vo+56*nv;wo=io+4*ni
 req(magic==b"NHSKIN1\0" and abi==5 and len(b)==wo+4*nv*nb,"invalid NHSKIN")
 v=np.frombuffer(b,"<f4",14*nv,vo).reshape(nv,14);f=np.frombuffer(b,"<u4",ni,io).reshape(-1,3).astype(np.int64)
 return {"raw":b,"nv":nv,"nb":nb,"pos":v[:,:3].astype(float),"faces":f,"bind":np.frombuffer(b,"<f4",9*nb,60).reshape(nb,9).copy(),"bind_u":np.frombuffer(b,"<u4",9*nb,60).reshape(nb,9).copy(),"weights":np.frombuffer(b,"<f4",nv*nb,wo).reshape(nv,nb).astype(float)}
def report_inputs(p,h,prop,ph):
 req(sha(p)==h and sha(prop)==ph,"pinned QP report/proposal mismatch");d=j(p);before=d.get("inputs_before")
 req(d.get("status")=="feasible_pending_exact_full_checks" and d.get("solver_success") is True and d.get("inputs_unchanged") is True and isinstance(before,dict) and before==d.get("inputs_after"),"QP replay incomplete")
 for q,v in before.items():req(Path(q).is_file() and not Path(q).is_symlink() and sha(q)==v,"QP input changed: "+q)
 req(Path(d["proposal"]["path"]).resolve()==prop.resolve() and d["proposal"]["sha256"]==ph,"QP proposal binding mismatch")
 return d,before
def loadmod(path,name):
 spec=importlib.util.spec_from_file_location(name,path);req(spec and spec.loader,"module load failure")
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--scan",action="store_true");ap.add_argument("--prepare-check",action="store_true");a=ap.parse_args()
 req(a.scan != a.prepare_check,"select --scan or --prepare-check")
 req(not OUT.exists() and OUT.parent.is_dir(),"output must be fresh dedicated scan-001")
 for p,h in PINS.items():req(p.is_file() and not p.is_symlink() and sha(p)==h,"input hash drift: "+str(p))
 q4,i4=report_inputs(Q4,PINS[Q4],P4,PINS[P4]);q5,i5=report_inputs(Q5,PINS[Q5],P5,PINS[P5])
 q6,i6=report_inputs(Q6,PINS[Q6],P6,PINS[P6])
 req(q6["dependency_complete_replay"]["array_bit_exact"] is True and len(i6)==547,"006 receipt-map/import dependency replay incomplete")
 for k,v in i5.items():req(i6.get(k)==v or k==str(S5),"006 does not preserve005 dependency "+k)
 dep=q5["dependency_complete_replay"]
 req(dep["array_bit_exact"] is True and dep["inherited_preflight_pins_revalidated"]==49 and dep["loaded_module_files_pinned"]==542,"005 complete replay proof missing")
 for k,v in i4.items():req(i5.get(k)==v,"005 does not preserve004 input "+k)
 pre=j(PREF);base=skinread(SKIN);old=skinread(OLD);faces=base["faces"]
 cand=np.load(CAND,allow_pickle=False);proposal=np.load(P4,allow_pickle=False);replay=np.load(P5,allow_pickle=False)
 req(base["nv"]==54949 and base["nb"]==86 and len(faces)==109211 and np.array_equal(faces,old["faces"]),"skin/topology mismatch")
 req(cand.shape==proposal.shape==replay.shape==(54949,3) and cand.dtype==proposal.dtype==replay.dtype==np.dtype("<f4") and np.isfinite(proposal).all() and np.array_equal(proposal,replay),"proposal arrays invalid or not byte-identical")
 affected=np.asarray(q5["affected_vertices"],dtype=np.int64);x=np.asarray(q5["reduction_fractions"],dtype=float)
 req(len(x)==len(affected) and np.isfinite(x).all() and np.all((x>=0)&(x<=1)),"QP variables invalid")
 masks=[]
 for rows in (RIGHT,LEFT):
  seeds=np.unique(faces[np.asarray(rows,dtype=np.int64)]);d=np.full(54949,4,dtype=np.int32);d[seeds]=0
  for ring in range(1,4):
   ids=np.unique(faces[np.any(d[faces]<ring,axis=1)]);d[ids]=np.minimum(d[ids],ring)
  t=np.clip(1-d/3.,0.,1.);masks.append(t*t*(3.-2.*t))
 D=np.stack([-(cand.astype(float)-base["pos"])*m[:,None] for m in masks],axis=-1)
 expected=cand.copy();expected[affected]=np.asarray(cand[affected].astype(float)+D[affected].sum(axis=-1)*x[:,None],dtype="<f4")
 req(np.array_equal(expected,proposal),"proposal does not replay from the recorded QP fractions")
 changed=np.flatnonzero(np.any(cand!=proposal,axis=1));req(set(changed).issubset(set(affected)) and len(changed)>0,"proposal has unrecorded source edits")
 fixed=np.asarray(pre["fixed_support_source_vertex_ids"],dtype=np.int64);anchors=np.asarray(pre["preserved_source_anchor_vertex_ids"],dtype=np.int64)
 req(len(fixed)==32 and len(anchors)==15 and np.array_equal(cand[fixed],proposal[fixed]) and np.array_equal(cand[anchors],proposal[anchors]),"support or anchor changed")
 from scipy.sparse import load_npz
 A=load_npz(A5).tocsr();rhs=np.load(B5,allow_pickle=False);labs=j(L5)
 req(A.shape==(len(rhs),len(affected)) and len(rhs)==q5["constraint_count"] and len(labs)==len(rhs) and float(np.max(A@x-rhs))<=1e-10,"unreduced full QP constraints fail")
 sys.path.insert(0,str(ROOT/"src"))
 from numilab_human import common_atlas_skin_clearance as cc
 from numilab_human import cardiac_cavity_intersections as pci
 req(Path(cc.__file__).resolve()==CLEAR and sha(CLEAR)==PINS[CLEAR] and Path(pci.__file__).resolve()==CARD,"1216 owner mismatch")
 rr=imp(RUNNER,"runner1172_selfreduction");ci,cp,validator,core=rr.load_predicates();keys,expected_by=rr.inventory()
 req(len(keys)==859 and len(set(keys))==859,"target inventory not859")
 ori=j(ORIENT);req(ori["inputs"]["nhskin"]["sha256"]==sha(OLD) and len(ori["per_face"])==980,"orientation basis drift")
 seed=np.asarray([int(s["face"]) for s in ori["per_face"]],dtype=np.int64)
 req(all(np.array_equal(faces[int(s["face"])],np.asarray(s["source_vertex_ids"],dtype=np.int64)) for s in ori["per_face"]),"orientation source face mismatch")
 signs=cc._propagate_source_face_orientation(faces,seed);ref=np.unique(faces);cf=np.searchsorted(ref,faces)
 # Preserve captured old stable64 points and append only the refined midpoint.
 tvpath=B/"heldout-001/validate-refined-target-1217.py"
 trpath=B/"heldout-001/refined-target-1217-validation.json"
 tr=j(trpath)
 req(tr.get("status")=="pass_within_1um" and tr.get("inputs_unchanged") is True,"refined target validation failed")
 for p,h in tr["inputs"].items():req(sha(p)==h,"refined target validation input changed "+p)
 tv=imp(tvpath,"target64_ninepose")
 tissue=R/"muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
 oldt=E/"passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
 cr=tv.row(tissue,64);br=tv.row(oldt,64)
 req(cr["counts"]==[3975,6690] and br["counts"]==[3974,6688],"target counts changed")
 req(np.array_equal(cr["v"][:-1].view("u1"),br["v"].view("u1")) and np.array_equal(cr["binds"].view("u1"),br["binds"].view("u1")),"old target vertices or bindings changed")
 target_injections=[]
 def read_refined(pack,requested):
  pos,surf,counts=cp._pack_surfaces(pack,requested)
  if Path(pack).parent!=HELD/"accepted-geometry":return pos,surf,counts
  key=(51005,64);rd=j(Path(pack).with_suffix(".receipt.json"))
  oldmap=tv.vertex_map(np.asarray(br["f"],dtype=np.int64),np.asarray(surf[key]["faces"]),3974)
  midpoint=tv.mapped_vertices(cr["v"][-1:],cr["binds"],rd)[0]
  newid=len(pos);newpos=np.concatenate([pos,midpoint.astype(pos.dtype)[None,:]],axis=0)
  req(np.array_equal(newpos[:-1].view("u1"),pos.view("u1")),"old captured positions changed")
  newfaces=np.concatenate([oldmap,np.asarray([newid],dtype=np.int64)])[np.asarray(cr["f"],dtype=np.int64)]
  req(newfaces.shape==(6690,3),"refined target face count changed")
  surf=dict(surf);surf[key]=dict(surf[key]);surf[key]["faces"]=newfaces
  target_injections.append({"step":rd["accepted_step"],"preserved_captured_vertices":3974,"appended_offline_midpoints":1,"faces":6690})
  return newpos,surf,counts
 # Hash window begins before any capture decoding or candidate forward evaluation.
 tracked={};paths={Path(p) for p in i4}|{Path(p) for p in i5}|{Path(p) for p in i6}|set(PINS)
 paths.update(Path(p) for p in tr["inputs"])
 paths.update([Path(__file__),tvpath,trpath,tissue,oldt,Path(rr.INV),Path(rr.AUDIT_CORE),Path(rr.HELPER)])
 paths.update(Path(m.__file__) for m in (ci,cp,validator,core,cc,pci))
 paths.update(Path(p) for p in rr.PREDICATE_SOURCE_PINS)
 for step,run,_ in POSES:
  paths.add(run/"accepted-geometry"/("step-%d.mrvpack"%step));paths.add(run/"accepted-geometry"/("step-%d.receipt.json"%step))
  receipt_for_maps=j(run/"accepted-geometry"/("step-%d.receipt.json"%step))
  for field in ("vertex_map","anatomy_parameters"):
   record=receipt_for_maps["skin_source_mapping"][field]
   req(sha(record["path"])==record["sha256"],"receipt-bound map hash mismatch")
   paths.add(Path(record["path"]))
 for p in sorted(paths,key=str):tracked[str(p.resolve())]=sha(p)
 req(len(tracked)>=542,"tracked inputs do not include full005 dependencies")
 if a.prepare_check:
  print(json.dumps({"status":"prepared_not_scanned","steps":STEPS,"targets":len(keys),"input_count":len(tracked),"proposal":pin(P4),"script":pin(Path(__file__))},sort_keys=True))
  return 0
 scan_start=time.monotonic()
 captures=[];states=[];items=[];initial=None;maps_by_cohort={};initial_by_cohort={}
 for step,run,cohort in POSES:
  pack=run/"accepted-geometry"/("step-%d.mrvpack"%step);rec=pack.with_suffix(".receipt.json");rd=j(rec)
  rr.verify_pack(step,pack,rec,validator,NHA_SHA)
  req(rd.get("common_field_source_anatomy_payload_sha256")==NHA_SHA,"capture anatomy mismatch")
  if initial is None:initial=rd
  else:req(rd["initial_anatomical_registration"]["body_poses"]==initial["initial_anatomical_registration"]["body_poses"],"initial registration changed")
  ident=(rd["skin_source_mapping"]["vertex_map"]["sha256"],rd["skin_source_mapping"]["anatomy_parameters"]["sha256"])
  if cohort not in maps_by_cohort:maps_by_cohort[cohort]=ident;initial_by_cohort[cohort]=rd
  else:req(maps_by_cohort[cohort]==ident,"skin/resp map mismatch within cohort")
  pos,surf,_=read_refined(pack,set(keys));rr.validate_pack_surface_keys(surf,keys)
  sf=surf[rr.EXPECTED_SKIN_KEY]["faces"];b=int(sf.min())
  req(len(sf)==len(faces) and np.array_equal(sf-b,faces),"captured skin face topology changed")
  captures.append(np.asarray(pos[b:b+54949][ref],dtype="<f4").astype(float))
  states.append({"step":step,"body_poses":rd["accepted_registered_body_poses"],"respiratory_motion":rd["accepted_respiratory_motion"]})
  items.append((step,cohort,pack,rec,surf,pos))
  print(json.dumps({"phase":"capture_loaded","step":step,"cohort":cohort}),flush=True)
 captures=np.asarray(captures,dtype="<f4").astype(float);fm=loadmod(FORWARD,"forward915_selfreduction")
 world=np.empty(captures.shape,dtype="<f4");jac=np.empty((9,len(ref),3,3));maprep={}
 for cohort in ("1217_native_refinement","1191_retained_holdout"):
  indices=[i for i,x in enumerate(POSES) if x[2]==cohort]
  map_receipt=initial_by_cohort[cohort]
  forward,cohort_report,_=fm.build_forward(skin=base,source_positions=base["pos"],referenced_ids=ref,captured_by_pose=captures[indices],state_receipts=[states[i] for i in indices],
   initial_body_poses=map_receipt["initial_anatomical_registration"]["body_poses"],map_receipt=map_receipt,
   anatomy_parameters_path=map_receipt["skin_source_mapping"]["anatomy_parameters"]["path"],
   respiration_source=Path("/Users/n/numi-human-resting-resp-source-20261006"),clearance_module=cc)
  from numilab_human import resting_respiratory_conforming_field as loaded_field
  req(Path(loaded_field.__file__).resolve()==RESP_ACTUAL and sha(RESP_ACTUAL)==PINS[RESP_ACTUAL],"wrong respiratory field module loaded")
  orig=forward(np.asarray(base["pos"],dtype="<f4"))
  req(orig["diagnostics"]["admissible"] is True and np.array_equal(np.asarray(orig["world_positions_by_pose"],dtype="<f4"),captures[indices].astype("<f4")),"cohort zero-delta replay not capture-exact")
  fw=forward(proposal);req(fw["diagnostics"]["admissible"] is True,"cohort candidate forward inadmissible")
  world[indices]=np.asarray(fw["world_positions_by_pose"],dtype="<f4");jac[indices]=np.asarray(fw["jacobians_by_pose"],dtype=float)
  maprep[cohort]={"steps":[STEPS[i] for i in indices],"map_receipt":map_receipt["skin_source_mapping"],"forward_model_report":cohort_report,"qualification":"Each cohort uses its own actual receipt-bound map; no cross-cohort byte-identity assumption."}
 req(world.shape==captures.shape and jac.shape==(9,len(ref),3,3) and np.isfinite(world).all() and np.isfinite(jac).all(),"candidate forward invalid")
 tri=world[:,cf];areas=.5*np.linalg.norm(np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]),axis=2)
 stri=proposal[faces];sa=.5*np.linalg.norm(np.cross(stri[:,1]-stri[:,0],stri[:,2]-stri[:,0]),axis=1)
 winding=cc._verify_source_winding_in_accepted_poses(proposal[ref],cf,signs,jac,np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]))
 scene=j(SCENE);plane=np.asarray(scene["bed"]["plane_point_m"]);normal=np.asarray(scene["bed"]["normal"])
 req(abs(np.linalg.norm(normal)-1)<1e-8,"bed normal is invalid");gaps=(world.astype(float)-plane)@normal
 base_tri=captures[:,cf].astype(float);new_tri=world[:,cf].astype(float)
 base_area=np.cross(base_tri[:,:,1]-base_tri[:,:,0],base_tri[:,:,2]-base_tri[:,:,0])
 new_area=np.cross(new_tri[:,:,1]-new_tri[:,:,0],new_tri[:,:,2]-new_tri[:,:,0])
 baseline_alignment=np.einsum("pfi,pfi->pf",base_area,new_area)/(np.linalg.norm(base_area,axis=2)*np.linalg.norm(new_area,axis=2))
 baseline_gaps=(captures.astype(float)-plane)@normal
 req(np.isfinite(baseline_alignment).all(),"nonfinite captured-baseline orientation")
 decl={"schema":"numi.human.skin-local-self-reduction-nine-pose-audit.declaration.v1","status":"running_exact_nine_pose_offline_candidate_audit",
  "steps":STEPS,"cohorts":[{"step":s,"cohort":c} for s,_,c in POSES],
  "candidate_qualification":"Unadmitted source-position proposal; offline forward audit only.",
  "proposal_004":pin(P4),"complete_dependency_replay_005":pin(Q5),"proposal_replay_005":pin(P5),"receipt_map_dependency_replay_006":pin(Q6),"proposal_replay_006":pin(P6),
  "fit_preflight":pin(PREF),"skin":pin(SKIN),"NHA":pin(NHA),"forward915":pin(FORWARD),
  "clearance_owner_1216":pin(CLEAR),"self_predicate_runner_1172":pin(RUNNER),"prepared_index_owner_1216":pin(CARD),
  "index_proof":pin(INDEX),"index_equivalence":pin(INDEX_EQ),"orientation":pin(ORIENT),"scene_bed_plane":pin(SCENE),
  "target_inventory":pin(INV),"target_surface_count":859,"protected_source_vertices":{"fixed_support":len(fixed),"thorax_anchors":len(anchors)},
  "source_reduction":{"changed_source_vertex_count":int(len(changed)),"affected_variable_count":int(len(affected)),
    "max_delta_mm":float(np.linalg.norm(proposal.astype(float)-cand,axis=1).max()*1000),"constraint_count":len(rhs),
    "max_unreduced_violation":float(np.max(A@x-rhs))},
  "inputs_before_scan":tracked,"target_coordinate_policy":"1191 old3974 stable64 vertices are captured byte-for-byte; only the new midpoint is predicted offline. All6690 refined faces are retained;1217 positions are actual captures.","scope":"Steps 0/9983/20000 from accepted 1217 captures plus 4991/5375/5759/6111/6495/7743 from retained 1191 captures. Offline forward candidate only; exact all-859 target and full skin self predicates. Not native admission, continuous-time, physiology, or biological acceptance.","output_directory":str(OUT)}
 OUT.mkdir();wj(OUT/"declaration.json",decl)
 ocular=set(rr.OCULAR);results=[]
 for pi,(step,cohort,pack,rec,surf,basepos) in enumerate(items):
  pos=np.asarray(basepos).copy();sf=surf[rr.EXPECTED_SKIN_KEY]["faces"];b=int(sf.min());pos[b+ref]=world[pi]
  vids=np.unique(sf);local=np.searchsorted(vids,sf);srec,srows,sdeg=core.exact_records(pos[vids],local,ci);sindex=pci._prepare_surface_aabb(srec)
  targets=OUT/("step-%d.targets.jsonl"%step);cross=OUT/("step-%d.crossing-witnesses.jsonl"%step)
  selfout=OUT/("step-%d.self-witnesses.jsonl"%step);invalid=OUT/("step-%d.invalid-triangles.jsonl"%step)
  total={"targets":0,"pairs":0,"ocular":0,"nonocular":0,"bad_surfaces":0,"bad_faces":0};inside={}
  for key,nface in CLOSED.items():
   tf=surf[key]["faces"];req(len(tf)==nface,"closed 63/64 target face count mismatch")
   ids=np.unique(tf);prepared=cc._prepare_closed_clearance_target(pos[ids].astype("<f4"),np.searchsorted(ids,tf),allow_nested_enclosure=True)
   inside["%d:%d"%key]=cc._closed_target_inside_vertices(world[pi],prepared)
  with targets.open("x") as ts,cross.open("x") as xs,selfout.open("x") as ss,invalid.open("x") as bs:
   for fi in sdeg:
    ids=[int(v) for v in sf[int(fi)]]
    bs.write(json.dumps({"role":"skin_degenerate_face","skin_source_face_row":int(fi),"pack_vertex_ids":ids,"triangle_xyz_f32_m":pos[ids].astype(float).tolist()},separators=(",",":"),allow_nan=False)+"\n")
   for ord,key in enumerate(keys,1):
    tf=surf[key]["faces"];ids=np.unique(tf);tr,rrr,td=core.exact_records(pos[ids],np.searchsorted(ids,tf),ci)
    au=pci._audit_pair_prepared_first(sindex,tr,same_surface=False);n=int(au["count"])
    total["targets"]+=1;total["pairs"]+=n
    if key in ocular:total["ocular"]+=n
    else:total["nonocular"]+=n
    if td:
     total["bad_surfaces"]+=1;total["bad_faces"]+=len(td)
     for fi in td:
      vv=[int(v) for v in tf[int(fi)]]
      bs.write(json.dumps({"role":"target_degenerate_face","target_surface":list(key),"target_surface_face_row":int(fi),"pack_vertex_ids":vv,"triangle_xyz_f32_m":pos[vv].astype(float).tolist()},separators=(",",":"),allow_nan=False)+"\n")
    ts.write(json.dumps({"surface":list(key),"source_owner_or_label":expected_by[key]["source_owner_or_label"],
      "face_count":int(len(tf)),"pinned_890_face_count":int(expected_by[key]["face_count_other"]),
      "face_count_delta_from_890":int(len(tf))-int(expected_by[key]["face_count_other"]),
      "aabb_candidate_pairs":int(au["aabb_candidate_pairs"]),"intersecting_triangle_pairs":n,
      "degenerate_face_rows":[int(z) for z in td],"pair_coverage_complete":not bool(td or sdeg),"ocular_monitor":key in ocular},
      separators=(",",":"),allow_nan=False)+"\n")
    for si,ti in au["triangle_pairs"]:
     core.crossing_witness(xs,semantic=key[0],stable_id=key[1],skin_pair_index=si,target_pair_index=ti,
      skin_record=srec[si],target_record=tr[ti],skin_row=srows[si],target_row=rrr[ti],skin_faces=sf,target_faces=tf,positions=pos,ci=ci)
    if ord%100==0 or ord==len(keys):print(json.dumps({"phase":"target_scan","step":step,"done":ord,"targets":len(keys),"pairs":total["pairs"]}),flush=True)
   selfaudit=ci._audit_pair(srec,srec,same_surface=True)
   for si,ti in selfaudit["triangle_pairs"]:
    core.crossing_witness(ss,semantic=rr.EXPECTED_SKIN_KEY[0],stable_id=rr.EXPECTED_SKIN_KEY[1],skin_pair_index=si,target_pair_index=ti,
     skin_record=srec[si],target_record=srec[ti],skin_row=srows[si],target_row=srows[ti],skin_faces=sf,target_faces=sf,positions=pos,ci=ci,role="skin_self_intersection")
  cov=total["targets"]==859 and not sdeg and total["bad_faces"]==0
  row={"accepted_step":step,"cohort":cohort,"status":"complete_pair_coverage" if cov else "incomplete_degenerate_input_fail_closed",
   "all_skin_crossing_pair_count":total["pairs"],"nonocular_crossing_pair_count":total["nonocular"],"ocular_crossing_pair_count":total["ocular"],
   "surface_target_count":total["targets"],"skin_self_crossing_pair_count":int(selfaudit["count"]),
   "skin_degenerate_face_rows":[int(v) for v in sdeg],"invalid_target_surface_count":total["bad_surfaces"],"invalid_target_triangle_count":total["bad_faces"],
   "pair_coverage_complete":cov,
   "closed_target_inside_counts":{"51005:63":{"inside_vertex_count":len(inside["51005:63"])},
    "51005:64":{"inside_vertex_count":len(inside["51005:64"])}},
   "minimum_bed_signed_gap_m":float(gaps[pi].min()),"negative_bed_gap_vertex_count":int(np.count_nonzero(gaps[pi]<0)),
   "baseline_minimum_bed_signed_gap_m":float(baseline_gaps[pi].min()),"minimum_captured_baseline_normal_alignment":float(baseline_alignment[pi].min()),
   "baseline_orientation_and_bed_gates_pass":bool(baseline_alignment[pi].min()>0 and gaps[pi].min()>=baseline_gaps[pi].min()-1e-6 and sa.min()>0 and areas[pi].min()>0),
   "source_surface_min_triangle_area_m2":float(sa.min()),"mapped_surface_min_triangle_area_m2":float(areas[pi].min()),
   "source_to_pose_winding":winding,"targets_sha256":sha(targets),"crossing_witnesses_sha256":sha(cross),
   "self_witnesses_sha256":sha(selfout),"invalid_triangles_sha256":sha(invalid),
   "predicate":"exact_float32_lattice_triangle_intersection","contact_exemptions":[]}
  wj(OUT/("step-%d.result.json"%step),row);results.append(row)
  print(json.dumps({"phase":"pose_complete","step":step,"target_pairs":total["pairs"],"self_pairs":row["skin_self_crossing_pair_count"],
   "degenerate_skin":len(sdeg),"inside63":len(inside["51005:63"]),"inside64":len(inside["51005:64"])}),flush=True)
  del pos,srec,srows,sindex,selfaudit;gc.collect()
 after={p:sha(Path(p)) for p in tracked};unchanged=tracked==after
 cov=len(results)==9 and all(r["pair_coverage_complete"] for r in results)
 passed=cov and unchanged and all(r["all_skin_crossing_pair_count"]==r["skin_self_crossing_pair_count"]==0 and
   not r["skin_degenerate_face_rows"] and r["invalid_target_triangle_count"]==0 and r["baseline_orientation_and_bed_gates_pass"] and
   r["closed_target_inside_counts"]["51005:63"]["inside_vertex_count"]==r["closed_target_inside_counts"]["51005:64"]["inside_vertex_count"]==0 for r in results)
 summary={"schema":"numi.human.skin-local-self-reduction-nine-pose-audit.summary.v1",
  "status":"complete_exact_intersection_free" if passed else "complete_with_intersections_or_invalid_geometry" if cov and unchanged else "incomplete_or_input_changed",
  "steps":STEPS,"pose_result_count":len(results),"target_surface_count":859,"pair_coverage_complete_all_nine":cov,
  "all_nine_target_self_degenerate_and_closed_envelope_gates_pass":passed,"inputs_unchanged":unchanged,
  "inputs_before":tracked,"inputs_after":after,"source_reduction_check":decl["source_reduction"],
  "forward_model_report":maprep,"pose_results":results,"target_injection_calls":target_injections,"elapsed_wall_seconds":time.monotonic()-scan_start,
  "qualification":"Nine retained accepted states only; offline source-proposal forward and exact predicates. Not native admission, continuous-time assurance, physiology, or biological acceptance."}
 wj(OUT/"summary.json",summary);print(json.dumps({"phase":"audit_complete","status":summary["status"],"steps":STEPS,
  "pair_coverage_complete":cov,"all_nine_pass":passed,"inputs_unchanged":unchanged,"summary":str(OUT/"summary.json"),"summary_sha256":sha(OUT/"summary.json")},sort_keys=True))
 return 0 if cov and unchanged else 2
if __name__=="__main__":raise SystemExit(main())
