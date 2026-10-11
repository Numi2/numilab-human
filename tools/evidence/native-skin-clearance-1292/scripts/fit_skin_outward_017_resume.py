#!/usr/bin/env python3
from __future__ import annotations
import gc, gzip, hashlib, importlib.util, json, os, signal, struct, sys, time
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
ROOT=A/"stable152-skin-fit-001"
PREV=ROOT/"result-012"
OUT=ROOT/"result-014-resume-004"
SCRIPT=Path(__file__).resolve()
SKIN=R/"skin-resting-multipose-clearance-1218/package-preparation-005/composed-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_MAN=SKIN.with_name("common-atlas-skin-geometry-registration.manifest.json")
TISS=R/"passive-neck-back-coverage-1281/candidate-005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS_MAN=TISS.with_suffix(".manifest.json")
PATCH=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/right-scm-local-reference-002/stable-151-alpha-0.3.npz")
PATCH_REPORT=PATCH.parent/"report.json"
BASE_REFRESH=ROOT/"corrected151-baseline-target-refresh-001"
BASE_REFRESH_REPORT=BASE_REFRESH/"report.json"
PARTIAL_AUDIT=ROOT/"corrected151-partialskin-audit-017"
PARTIAL_AUDIT_REPORT=PARTIAL_AUDIT/"report.json"
FIT_PREV=ROOT/"result-013"
FIT_EVENT_PATH=FIT_PREV/"attempts.jsonl"
FIT_SOURCE_PATH=FIT_PREV/"attempts/attempt-0001-source.npy"
FIT_TARGET_PATH=FIT_PREV/"attempts/attempt-0001-target-audits.json.gz"
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
SCENE=SKIN.parent/"resting-supine-scene.manifest.json"
ORIENT=Path("/Users/n/numi-human-resting-evidence-20261005/local-skin-orientation-892/local-orientation-report.json")
ORIENT_SKIN=Path("/Users/n/numi-human-resting-evidence-20261005/common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin")
FORWARD=Path("/Users/n/numi-human-resting-evidence-20261005/native-common-skin-multipose-forward-model-915.py")
TISS_FORWARD=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/neck-back-retained-pose-audit-002/forward_all_positive_weights.py")
RESP=Path("/Users/n/numi-human-resting-resp-source-20261006")
CA=Path("/Users/n/numi-human-self-audit-resume-1265/src/numilab_human/common_atlas_skin_clearance.py")
CI=CA.with_name("cardiac_cavity_intersections.py")
PACKDIR=A/"twenty-two-surface-native-composition-002/baseline/native-run/accepted-geometry"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
NEW_KEYS={(51005,151),(51005,152)}
SKIN_KEY=(51007,1)
MAX_SECONDS=3600
MAX_CUMULATIVE_SOURCE_DISPLACEMENT_MM=4.0
if OUT.exists():
    raise SystemExit("refuse existing fit output "+str(OUT))
OUT.mkdir(parents=True)
(OUT/"attempts").mkdir()
START=time.monotonic()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def f32sha(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype="<f4").tobytes()).hexdigest()
def pin(path):
    p=Path(path).resolve()
    if not p.is_file(): raise RuntimeError("missing pinned input "+str(p))
    return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def write_json(path,doc):
    Path(path).write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
def log(event,**kw):
    row={"event":event,"elapsed_s":time.monotonic()-START,**kw}
    with (OUT/"progress.jsonl").open("a") as f:
        f.write(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())
    print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def timeout(_sig,_frame):
    raise TimeoutError("bounded 8-pose outward skin fit exceeded 3600 seconds")
signal.signal(signal.SIGALRM,timeout)
signal.alarm(MAX_SECONDS)

sys.path.insert(0,str(CA.parent.parent))
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci
def load_skin(path):
    raw=Path(path).read_bytes()
    magic,abi,nb,nv,ni,fp,arch=struct.unpack_from("<8s5I32s",raw)
    vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
    require(magic==b"NHSKIN1\0" and abi==5 and len(raw)==wo+4*nv*nb,"unsupported NHSKIN ABI/layout")
    vr=np.frombuffer(raw,"<f4",14*nv,vo).reshape(nv,14)
    bind_u=np.frombuffer(raw,"<u4",9*nb,60).reshape(nb,9).copy()
    bind=np.frombuffer(raw,"<f4",9*nb,60).reshape(nb,9).copy()
    faces=np.frombuffer(raw,"<u4",ni,io).reshape(-1,3).astype(np.int64)
    weights=np.frombuffer(raw,"<f4",nv*nb,wo).reshape(nv,nb).copy()
    return raw,{"nb":nb,"nv":nv,"ni":ni,"fp":fp,"pos":vr[:,:3].copy(),
                "normals":vr[:,3:6].copy(),"bind_u":bind_u,"bind":bind,
                "faces":faces,"weights":weights}
def rotate32(q,p):
    f=np.float32; q=np.asarray(q,dtype=np.float32); p=np.asarray(p,dtype=np.float32); a=q[:3]
    cr=np.asarray([f(f(a[1]*p[2])-f(a[2]*p[1])),
                   f(f(a[2]*p[0])-f(a[0]*p[2])),
                   f(f(a[0]*p[1])-f(a[1]*p[0]))],dtype=np.float32)
    tw=np.asarray([f(f(2)*v) for v in cr],dtype=np.float32)
    co=np.asarray([f(f(f(q[3]*tw[0])+f(a[1]*tw[2]))-f(a[2]*tw[1])),
                   f(f(f(q[3]*tw[1])+f(a[2]*tw[0]))-f(a[0]*tw[2])),
                   f(f(f(q[3]*tw[2])+f(a[0]*tw[1]))-f(a[1]*tw[0]))],dtype=np.float32)
    return np.asarray([f(p[i]+co[i]) for i in range(3)],dtype=np.float32)
def load_tissue(path):
    raw=Path(path).read_bytes()
    magic,abi,nr,nb,nv,ni,fp,src=struct.unpack_from("<8s6I32s",raw)
    require(magic==b"NHTISS4\0" and abi==5,"unsupported NHTISS ABI")
    bo=64+32*nr; vo=bo+36*nb; io=vo+56*nv
    require(len(raw)==io+4*ni,"NHTISS byte length mismatch")
    rec=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(-1,8)
    bind=np.frombuffer(raw,dtype=np.dtype([("core","<u4"),("value","<f4",(8,))]),count=nb,offset=bo)
    ind=np.frombuffer(raw,"<u4",count=ni,offset=io)
    return {"path":Path(path),"raw":raw,"records":rec,"bindings":bind,"indices":ind,"bo":bo,"vo":vo}
def tissue_row(t,sid):
    hits=[r for r in t["records"] if int(r[6])==sid]
    require(len(hits)==1,f"stable {sid} row cardinality {len(hits)}")
    r=hits[0]; fb,bc,fv,vc,fi,ic=map(int,r[:6]); raw=t["raw"]; vo=t["vo"]
    pos=np.ndarray((vc,3),dtype="<f4",buffer=raw,offset=vo+fv*56,strides=(56,4)).copy()
    local=np.ndarray((vc,4),dtype="<u4",buffer=raw,offset=vo+fv*56+24,strides=(56,4)).copy()
    weights=np.ndarray((vc,4),dtype="<f4",buffer=raw,offset=vo+fv*56+40,strides=(56,4)).copy()
    faces=t["indices"][fi:fi+ic].reshape(-1,3).astype(np.int64)-fv
    return {"sid":sid,"fb":fb,"bc":bc,"fv":fv,"vc":vc,"fi":fi,"ic":ic,"positions":pos,"local":local,"weights":weights,"faces":faces}
def tissue_forward(points,local,weights,row,tissue,poses):
    f=np.float32; out=np.zeros_like(points,dtype=np.float32)
    for vi in range(len(points)):
        accum=np.zeros(3,dtype=np.float32)
        for slot in range(4):
            w=f(weights[vi,slot])
            if w<=f(0): continue
            li=int(local[vi,slot]); require(li<row["bc"],"SCM local binding out of range")
            b=tissue["bindings"][row["fb"]+li]; core=int(b["core"])
            require(core in poses,"missing SCM body pose "+str(core))
            val=b["value"]; rr=rotate32(val[3:7],points[vi])
            loc=np.asarray([f(val[j]+f(f(rr[j])*val[7])) for j in range(3)],dtype=np.float32)
            bp,bq=poses[core]; wr=rotate32(bq,loc)
            world=np.asarray([f(bp[j]+wr[j]) for j in range(3)],dtype=np.float32)
            accum=np.asarray([f(accum[j]+f(w*world[j])) for j in range(3)],dtype=np.float32)
        out[vi]=accum
    return out

pins=[pin(p) for p in [SCRIPT,SKIN,SKIN_MAN,TISS,TISS_MAN,INV,SCENE,ORIENT,ORIENT_SKIN,
    FORWARD,TISS_FORWARD,CA,CI,RESP/"src/numilab_human/resting_respiratory_conforming_field.py"]]
prehash={p["path"]:p["sha256"] for p in pins}
skin_raw,skin=load_skin(SKIN)
require((skin["nb"],skin["nv"],len(skin["faces"]))==(86,54949,109211),"current skin layout changed")
require(sha(ORIENT_SKIN)=="c98e72605f78dad832d5106283f0cb0dca9564d50b1873fc78c883476a093392","orientation topology source pin changed")
_,orientation_skin=load_skin(ORIENT_SKIN)
require(np.array_equal(skin["faces"],orientation_skin["faces"]),"current skin face topology differs from orientation source")
tiss_manifest=json.loads(TISS_MAN.read_text())
surface_rows={int(x["stable_id"]):x for x in tiss_manifest["source"]["surfaces"]}
require(surface_rows[151]["member_id"]=="FJ1595" and surface_rows[152]["member_id"]=="FJ1573","SCM stable/member mapping changed")
tissue=load_tissue(TISS)
rows={sid:tissue_row(tissue,sid) for sid in (151,152)}
patch_npz=np.load(PATCH,allow_pickle=False)
patch_vertices=np.asarray(patch_npz["vertices6"],dtype="<f4")
patch_faces=np.asarray(patch_npz["faces"],dtype=np.int64)
patch_local=np.asarray(patch_npz["binding_indices"],dtype="<u4")
patch_weights=np.asarray(patch_npz["weights"],dtype="<f4")
require(patch_vertices.shape==(5160,6) and patch_faces.shape==(10324,3)
        and patch_local.shape==(5160,4) and patch_weights.shape==(5160,4),"corrected151 patch layout changed")
require(np.isfinite(patch_vertices).all() and np.isfinite(patch_weights).all(),"corrected151 patch is non-finite")
require(np.all(patch_local[patch_weights>0] < rows[151]["bc"]),"corrected151 active local binding index is out of range")
require(sha(PATCH_REPORT)=="bb59b07299f557e2dfd6b3489c4c4ddcf34e3d5adb7a6581d0da61a3ba02db43","corrected151 patch report changed")
rows[151]={**rows[151],"positions":patch_vertices[:,:3].copy(),"local":patch_local.copy(),
            "weights":patch_weights.copy(),"faces":patch_faces.copy(),"vc":len(patch_vertices)}
require(rows[151]["faces"].shape==(10324,3) and rows[152]["faces"].shape==(10224,3),"SCM corrected face inventory changed")
require(rows[151]["vc"]==5160 and rows[152]["vc"]==6210,"SCM corrected vertex inventory changed")
legacy_keys,nonocular_keys,_,_=ca._load_target_inventory(INV)
all_target_keys=set(legacy_keys)|NEW_KEYS
all_text={f"{a}:{b}" for a,b in all_target_keys}
ocular={f"{a}:{b}" for a,b in ca._OCULAR_MONITOR_KEYS}
require(len(legacy_keys)==859 and len(all_target_keys)==861 and ocular.issubset(all_text),"full target inventory mismatch")

fs=importlib.util.spec_from_file_location("forward915_outward_skin",FORWARD)
fm=importlib.util.module_from_spec(fs); fs.loader.exec_module(fm)
ts=importlib.util.spec_from_file_location("forward_all_positive_weights_scm",TISS_FORWARD)
th=importlib.util.module_from_spec(ts); ts.loader.exec_module(th)
require(Path(fm.__file__).resolve()==FORWARD.resolve(),"unexpected 915 forward module")
require(Path(th.__file__).resolve()==TISS_FORWARD.resolve(),"unexpected SCM forward module")
ref=np.unique(skin["faces"]); faces=skin["faces"]; compact=np.searchsorted(ref,faces)
require(len(ref)==54663,"skin referenced vertex inventory changed")
orient=json.loads(ORIENT.read_text()); seeds=orient.get("per_face",[])
require(orient.get("inputs",{}).get("nhskin",{}).get("sha256")==sha(ORIENT_SKIN) and len(seeds)==980,"source orientation seed report changed")
seed_ids=np.asarray([int(x["face"]) for x in seeds],dtype=np.int64)
require(all(np.array_equal(faces[int(x["face"])],np.asarray(x["source_vertex_ids"],dtype=np.int64)) for x in seeds),"orientation face/vertex seed mapping changed")
signs=ca._propagate_source_face_orientation(faces,seed_ids)

states=[]; captured=[]; state_receipts=[]; initial_receipt=None
baseline_self_proof_reuse=[]
target_cache=set()
# Reuse the complete exact baseline target tables emitted by run 012 only after
# validating that run's executable/input pins, every pose-specific target world
# identity, complete semantic key coverage, and pair-table digests.
prior_preflight_path=PREV/"preflight.json"
prior_preflight=json.loads(prior_preflight_path.read_text())
require(prior_preflight.get("status")=="ready_for_bounded_outward_skin_clearance_trial","prior baseline preflight is not ready")
for prior_pin in prior_preflight["input_pins"]:
    prior_path=Path(prior_pin["path"])
    require(prior_path.is_file() and sha(prior_path)==prior_pin["sha256"],"prior baseline input pin changed: "+str(prior_path))
prior_pin_map={str(Path(x["path"]).resolve()):x["sha256"] for x in prior_preflight["input_pins"]}
require(prior_preflight.get("source_positions_f32_sha256")==f32sha(skin["pos"]),"cached baseline self proof source positions differ")
require(prior_preflight.get("source_face_index_sha256")==ca._face_index_sha256(faces),"cached baseline self proof face topology differs")
require(prior_preflight.get("baseline_self_pairs_by_pose")==[0]*len(STEPS),"cached baseline self proof is not zero at all poses")
require(len(prior_preflight.get("pose_steps",[]))==len(STEPS) and [int(x) for x in prior_preflight["pose_steps"]]==STEPS,"cached baseline self proof pose coverage differs")
prior_executor=ROOT/"fit_skin_outward_012.py"
require(prior_pin_map.get(str(prior_executor.resolve()))==sha(prior_executor),"cached baseline self proof executor identity is absent/stale")
require(prior_pin_map.get(str(CA.resolve()))==sha(CA) and prior_pin_map.get(str(CI.resolve()))==sha(CI),"cached baseline self proof owner/predicate pins differ")
prior_cache=PREV/"baseline-target-tables.json.gz"
require(sha(prior_cache)==prior_preflight.get("target_tables_sha256"),"prior exact baseline table container pin mismatch")
with gzip.open(prior_cache,"rt",encoding="utf-8") as f: prior_tables=json.load(f)
require(set(prior_tables)=={str(step) for step in STEPS},"prior exact baseline tables do not cover all eight poses")
prior_events=[json.loads(line) for line in (PREV/"progress.jsonl").read_text().splitlines()]
prior_target_hash={int(e["step"]):e["target_geometry_sha256"] for e in prior_events if e.get("event")=="baseline_target_scan_start"}
prior_scan_done={int(e["step"]):e for e in prior_events if e.get("event")=="baseline_target_scan_done"}
require(set(prior_target_hash)==set(STEPS) and set(prior_scan_done)==set(STEPS),"prior progress does not prove complete baseline scans")
for step in STEPS:
    tab=prior_tables[str(step)]
    require(set(tab)==all_text and len(tab)==861,"prior baseline table has incomplete target coverage")
    for key,row in tab.items():
        require(isinstance(row,dict) and isinstance(row.get("triangle_pairs"),list) and type(row.get("count")) is int
                and row["count"]==len(row["triangle_pairs"]) and not row.get("degenerate_face_rows"),
                "prior baseline table malformed at pose "+str(step)+" target "+key)
    table_sha=ca._baseline_target_pair_table_sha256(tab)
    require(table_sha==prior_preflight["baseline_target_pair_table_sha256_by_pose"][STEPS.index(step)]
            and table_sha==prior_scan_done[step]["table_sha256"],"prior baseline pair-table identity mismatch at pose "+str(step))
base_refresh_report=json.loads(BASE_REFRESH_REPORT.read_text())
require(base_refresh_report.get("status")=="completed_exact_target_refresh_offline" and base_refresh_report.get("input_postcheck")=="unchanged","corrected151 baseline refresh is not complete")
with gzip.open(base_refresh_report["corrected_target_table_path"],"rt",encoding="utf-8") as f: refreshed_baseline_tables=json.load(f)
require(len(refreshed_baseline_tables)==len(STEPS),"corrected151 baseline target rows do not cover all poses")
partial_audit_report=json.loads(PARTIAL_AUDIT_REPORT.read_text())
require(partial_audit_report.get("status")=="completed_exact_target_refresh_offline" and partial_audit_report.get("input_postcheck")=="unchanged","partial-skin corrected151 audit is not complete")
with gzip.open(partial_audit_report["corrected_target_table_path"],"rt",encoding="utf-8") as f: resumed_target_tables=json.load(f)
require(len(resumed_target_tables)==len(STEPS),"corrected151 resumed target rows do not cover all poses")
resume_event_lines=FIT_EVENT_PATH.read_text().splitlines()
require(len(resume_event_lines)==1,"partial fit must contain exactly one accepted event")
resume_event=json.loads(resume_event_lines[0])
require(resume_event.get("status")=="accepted" and resume_event.get("attempt")==1,"partial fit event is not the accepted first attempt")
require(sha(FIT_SOURCE_PATH)==resume_event.get("candidate_npy_sha256"),"partial candidate source NPY hash mismatch")
require(sha(FIT_TARGET_PATH)==resume_event.get("target_audits_sha256"),"partial candidate target audit hash mismatch")
resume_compact=np.load(FIT_SOURCE_PATH,allow_pickle=False)
require(resume_compact.shape==(len(ref),3) and resume_compact.dtype==np.dtype("<f4"),"partial candidate compact source layout mismatch")
require(f32sha(resume_compact)==resume_event.get("source_positions_f32_sha256"),"partial candidate compact source identity mismatch")
require(partial_audit_report.get("fit_attempt_source_positions_f32_sha256")==resume_event.get("source_positions_f32_sha256")
        and partial_audit_report.get("fit_attempt_npy_sha256")==resume_event.get("candidate_npy_sha256")
        and partial_audit_report.get("fit_attempt_self_audits_sha256")==resume_event.get("self_audits_sha256"),"corrected151 candidate audit not bound to accepted event")
require(base_refresh_report.get("fit_attempt_source_positions_f32_sha256")==resume_event.get("source_positions_f32_sha256"),"corrected151 baseline refresh not bound to accepted event")
for i,step in enumerate(STEPS):
    require(set(refreshed_baseline_tables[i])==all_text and len(refreshed_baseline_tables[i])==861,"corrected151 baseline table inventory mismatch")
    require(set(resumed_target_tables[i])==all_text and len(resumed_target_tables[i])==861,"corrected151 resumed table inventory mismatch")
    require(ca._baseline_target_pair_table_sha256(refreshed_baseline_tables[i])==base_refresh_report["poses"][i]["merged861_pair_table_sha256"],"corrected151 baseline table hash mismatch")
    require(ca._baseline_target_pair_table_sha256(resumed_target_tables[i])==partial_audit_report["poses"][i]["merged861_pair_table_sha256"],"corrected151 resumed table hash mismatch")
    require(resume_event["self_audits_by_pose"][i]["candidate_world_f32_sha256"]==partial_audit_report["poses"][i]["candidate_skin_world_f32_sha256"],"candidate audit world differs from self proof")
resume_full=skin["pos"].astype("<f4").copy()
resume_full[ref]=resume_compact
resume_provenance={"source_positions_f32_sha256":resume_event["source_positions_f32_sha256"],"source_positions_path":str(FIT_SOURCE_PATH),
    "target_audits_path":str(partial_audit_report["corrected_target_table_path"]),"target_audits_sha256":sha(partial_audit_report["corrected_target_table_path"]),
    "self_audits_path":str(FIT_EVENT_PATH),"self_audits_sha256":resume_event["self_audits_sha256"],
    "clearance_owner_source_sha256":sha(CA),"exact_predicate_source_sha256":sha(CI),
    "partial_fit_event_line_sha256":hashlib.sha256((resume_event_lines[0]+"\\n").encode()).hexdigest(),
    "partial_source_npy_sha256":sha(FIT_SOURCE_PATH),"corrected151_baseline_report_sha256":sha(BASE_REFRESH_REPORT),
    "corrected151_candidate_audit_report_sha256":sha(PARTIAL_AUDIT_REPORT)}
for step in STEPS:
    log("load_pose_start",step=step)
    pack=PACKDIR/f"step-{step}.mrvpack"; receipt=PACKDIR/f"step-{step}.receipt.json"
    ph=sha(pack); rh=sha(receipt); rec=json.loads(receipt.read_text())
    require(rec.get("accepted_step")==step and rec.get("physical_endpoint")=="accepted","step is not accepted endpoint")
    require(rec.get("pack_file_sha256")==ph,"receipt pack hash mismatch")
    poses={}
    for item in rec.get("accepted_body_poses",[]):
        idx=int(item["body_index"]); require(idx not in poses,"duplicate body pose")
        poses[idx]=(np.asarray(item["position_m"],dtype=np.float32),np.asarray(item["quaternion_xyzw"],dtype=np.float32))
    require(len(poses)==157,"accepted pose count is not 157")
    if initial_receipt is None: initial_receipt=rec
    else:
        require(json.dumps(rec["initial_anatomical_registration"]["body_poses"],sort_keys=True)==json.dumps(initial_receipt["initial_anatomical_registration"]["body_poses"],sort_keys=True),"initial registration differs")
    xyz,surfs,meta=ca._pack_surfaces(pack,set(legacy_keys)|{SKIN_KEY})
    require(SKIN_KEY in surfs and all(k in surfs for k in legacy_keys),"capture omits legacy target surfaces")
    sf=np.asarray(surfs[SKIN_KEY]["faces"],dtype=np.int64); base=int(sf.min())
    require(len(sf)==len(faces) and np.array_equal(sf-base,faces),"captured skin topology differs from NHSKIN")
    skin_world=np.asarray(xyz[base:base+skin["nv"]],dtype="<f4")[ref].copy()
    captured.append(skin_world.astype(float))
    state_receipts.append({"step":step,"body_poses":rec["accepted_registered_body_poses"],"respiratory_motion":rec["accepted_respiratory_motion"]})
    target_faces={k:np.asarray(surfs[k]["faces"],dtype=np.int64) for k in legacy_keys}
    target_positions=np.asarray(xyz,dtype="<f4").copy()
    new_faces={}; offset=len(target_positions); append_positions=[]
    for sid in (151,152):
        rr=rows[sid]; p=tissue_forward(rr["positions"],rr["local"],rr["weights"],rr,tissue,poses)
        append_positions.append(p); new_faces[(51005,sid)]=rr["faces"]+offset; offset+=len(p)
    target_positions=np.concatenate([target_positions,*append_positions]).astype("<f4",copy=False)
    target_faces.update(new_faces)
    require(set(target_faces)==all_target_keys and len(target_faces)==861,"augmented target coverage mismatch")
    refreshed_pose=base_refresh_report["poses"][STEPS.index(step)]
    target_sha=ca._target_geometry_f32_sha256(target_positions,target_faces)
    require(refreshed_pose["baseline_full_target_geometry_sha256"]==prior_target_hash[step],"refresh starts from different original target geometry at pose "+str(step))
    require(target_sha==refreshed_pose["corrected_full_target_geometry_sha256"],"corrected151 target geometry differs from refresh at pose "+str(step))
    skin_records=ca._exact_surface_records(skin_world,compact)
    old_audit=prior_tables[str(step)]
    require(sum(int(x["count"]) for x in old_audit.values())==int(prior_scan_done[step]["total_pairs"]),"prior pair count mismatch at pose "+str(step))
    audit=refreshed_baseline_tables[STEPS.index(step)]
    require(set(audit)==all_text and len(audit)==861,"cached baseline target audit incomplete")
    table_sha=ca._baseline_target_pair_table_sha256(audit)
    require(table_sha==refreshed_pose["merged861_pair_table_sha256"],"corrected151 baseline table differs from refresh at pose "+str(step))
    require(sum(int(x["count"]) for x in audit.values())==int(prior_scan_done[step]["total_pairs"]),
            "cached baseline exact-pair count mismatch at pose "+str(step))
    table_path=OUT/f"baseline-target-step-{step}.json.gz"
    with gzip.open(table_path,"wt",encoding="utf-8",compresslevel=6) as f:
        json.dump(audit,f,sort_keys=True,separators=(",",":"),allow_nan=False)
    pose_index=STEPS.index(step)
    cached_self_count=int(prior_preflight["baseline_self_pairs_by_pose"][pose_index])
    require(cached_self_count==0,f"cached baseline self proof is nonzero at {step}")
    require(prior_pin_map.get(str(pack.resolve()))==ph and prior_pin_map.get(str(receipt.resolve()))==rh,
            f"current capture pack/receipt is not the exact baseline-self-proof input at {step}")
    current_world_sha=ca._float32_xyz_sha256(skin_world)
    baseline_self_proof_reuse.append({"pose_index":pose_index,"step":step,"pair_count":cached_self_count,
        "prior_preflight_path":str(prior_preflight_path),"prior_preflight_sha256":sha(prior_preflight_path),
        "prior_executor_path":str(prior_executor),"prior_executor_sha256":sha(prior_executor),
        "capture_pack_path":str(pack),"capture_pack_sha256":ph,"capture_receipt_path":str(receipt),"capture_receipt_sha256":rh,
        "reconstructed_captured_skin_world_f32_sha256":current_world_sha,
        "source_positions_f32_sha256":prior_preflight["source_positions_f32_sha256"],
        "source_face_index_sha256":prior_preflight["source_face_index_sha256"],
        "compact_face_index_sha256":ca._face_index_sha256(compact),
        "clearance_owner_sha256":sha(CA),"exact_predicate_sha256":sha(CI),
        "reuse_basis":"prior exact zero-pair audit is bound through its pinned executor, source/topology, owner/predicate, capture pack and receipt; this wrapper reconstructed the same captured skin world from those same pinned bytes"})
    skin_self={"count":cached_self_count,"degenerate_face_rows":[],"identity_bound_cached_proof":True}
    states.append({"step":step,"pack":pack,"receipt":receipt,"pack_sha":ph,"receipt_sha":rh,"poses":poses,
        "skin_world":skin_world,"target_positions":target_positions,"target_faces":target_faces,
        "target_sha":target_sha,"baseline":audit,"baseline_sha":table_sha,
        "face_sha":ca._face_index_sha256(compact),"skin_self":skin_self})
    log("baseline_target_cache_revalidated",step=step,total_pairs=sum(int(x["count"]) for x in audit.values()),
        target_geometry_sha256=target_sha,table_sha256=table_sha)
    del xyz,surfs,meta; gc.collect()

captured=np.stack(captured).astype("<f4").astype(float)
initial=initial_receipt
forward,map_report,selector_ids=fm.build_forward(skin=skin,source_positions=skin["pos"],referenced_ids=ref,
    captured_by_pose=captured,state_receipts=state_receipts,initial_body_poses=initial["initial_anatomical_registration"]["body_poses"],
    map_receipt=initial,anatomy_parameters_path=initial["skin_source_mapping"]["anatomy_parameters"]["path"],
    respiration_source=RESP,clearance_module=ca)
base_forward=forward(skin["pos"].astype("<f4"))
require(base_forward.get("diagnostics",{}).get("admissible") is True,"forward baseline inadmissible")
fw_world=np.asarray(base_forward["world_positions_by_pose"],dtype="<f4")
require(fw_world.shape==captured.shape and np.array_equal(fw_world,captured.astype("<f4")),"forward does not bit-replay eight captures")
base_tri=fw_world[:,compact]
base_area=np.cross(base_tri[:,:,1]-base_tri[:,:,0],base_tri[:,:,2]-base_tri[:,:,0])
winding=ca._verify_source_winding_in_accepted_poses(skin["pos"][ref],compact,signs,
    np.asarray(base_forward["jacobians_by_pose"],dtype=np.float64),base_area)
require(float(winding.get("minimum_source_to_accepted_pose_normal_alignment",0))>0,"source winding gate failed")
scene=json.loads(SCENE.read_text()); bed=scene["bed"]
fixed=np.asarray([int(x["vertex_index"]) for x in bed["support_witnesses"]],dtype=np.int64)
require(len(fixed)==32 and len(set(map(int,fixed)))==32 and set(map(int,fixed)).issubset(set(map(int,ref))),"bed support witness inventory changed")
spill=np.asarray([19059,19060,19061,19743,19745,20391,20392,21186,34342,34343,34344,34992,34993,34994],dtype=np.int64)
anchors=np.unique(np.concatenate((selector_ids[np.isin(selector_ids,ref)],spill)))
require(set(map(int,anchors)).issubset(set(map(int,ref))),"preserved anchors absent from topology")
anchors=anchors[~np.isin(anchors,fixed)]
bed_origin=np.repeat(np.asarray(bed["plane_point_m"],dtype=float)[None,:],len(STEPS),axis=0)
bed_normal=np.repeat(np.asarray(bed["normal"],dtype=float)[None,:],len(STEPS),axis=0)
def fit_forward(full_source):
    x=np.asarray(full_source,dtype="<f4")
    result=dict(forward(x))
    displacement=np.linalg.norm(x[ref].astype(np.float64)-skin["pos"][ref].astype(np.float64),axis=1)*1000.0
    maximum=float(displacement.max(initial=0.0))
    diagnostics=dict(result.get("diagnostics",{}))
    diagnostics["maximum_cumulative_source_displacement_from_original_mm"]=maximum
    diagnostics["cumulative_source_displacement_limit_mm"]=MAX_CUMULATIVE_SOURCE_DISPLACEMENT_MM
    if maximum>MAX_CUMULATIVE_SOURCE_DISPLACEMENT_MM+1.0e-6:
        diagnostics["admissible"]=False
        diagnostics["rejection_reason"]="cumulative inferred source displacement exceeds explicit 4 mm limit from original NHSKIN source"
    result["diagnostics"]=diagnostics
    return result
def target_triangle(pose,key,row):
    st=states[int(pose)]; k=tuple(map(int,str(key).split(":")))
    return st["target_positions"][st["target_faces"][k][int(row)]].astype("<f4").astype(float)
def scan(pose,world):
    st=states[int(pose)]
    candidate=np.asarray(world,dtype="<f4")
    require(candidate.shape==st["skin_world"].shape and np.isfinite(candidate).all(),"candidate target scan world malformed")
    if np.array_equal(candidate,st["skin_world"]):
        # Empty changed-face records are not sent to the incremental owner.
        require(ca._float32_xyz_sha256(st["skin_world"])==ca._float32_xyz_sha256(captured[int(pose)]),
                "zero-change baseline world identity mismatch")
        require(ca._face_index_sha256(compact)==st["face_sha"]
                and ca._target_geometry_f32_sha256(st["target_positions"],st["target_faces"])==st["target_sha"]
                and ca._baseline_target_pair_table_sha256(st["baseline"])==st["baseline_sha"],
                "zero-change cached target audit identity mismatch")
        result=st["baseline"]
    else:
        envelope=ca.audit_incremental_skin_target_intersections(baseline_world_positions=st["skin_world"],
            candidate_world_positions=candidate,baseline_faces=compact,candidate_faces=compact,
            baseline_target_audits=st["baseline"],target_faces_by_key=st["target_faces"],target_positions=st["target_positions"],
            expected_baseline_world_f32_sha256=ca._float32_xyz_sha256(st["skin_world"]),
            expected_face_index_sha256=st["face_sha"],expected_target_geometry_f32_sha256=st["target_sha"],
            expected_baseline_target_audits_sha256=st["baseline_sha"],_validated_target_geometry_cache=target_cache)
        result=envelope.get("target_audits")
    require(isinstance(result,dict) and set(result)==all_text,"incremental callback omitted exact target table coverage")
    return result
resume_callback_smoke=scan(0,states[0]["skin_world"])
require(set(resume_callback_smoke)==all_text and len(resume_callback_smoke)==861,"callback preflight omitted targets")
require(ca._baseline_target_pair_table_sha256(resume_callback_smoke)==states[0]["baseline_sha"],"callback preflight table hash mismatch")
for key,row in resume_callback_smoke.items():
    require(isinstance(row,dict) and type(row.get("count")) is int and row["count"]==len(row.get("triangle_pairs",[]))
            and not row.get("degenerate_face_rows"),"callback preflight returned malformed row "+key)
write_json(OUT/"resume-callback-preflight.json",{"status":"passed","pose_index":0,"step":STEPS[0],"target_count":861,
    "target_table_sha256":ca._baseline_target_pair_table_sha256(resume_callback_smoke),
    "resume_candidate_source_sha256":resume_event["source_positions_f32_sha256"],
    "resume_candidate_target_table_sha256":partial_audit_report["poses"][0]["merged861_pair_table_sha256"],
    "resume_candidate_self_audits_sha256":resume_event["self_audits_sha256"],"clearance_owner_sha256":sha(CA),"exact_predicate_sha256":sha(CI)})
log("resume_callback_preflight_passed",pose_index=0,step=STEPS[0],target_count=861,
    baseline_table_sha256=ca._baseline_target_pair_table_sha256(resume_callback_smoke))

def persist(event):
    row=dict(event); source=np.asarray(row.pop("source_positions_f32"),dtype="<f4"); attempt=int(row["attempt"])
    npy=OUT/"attempts"/f"attempt-{attempt:04d}-source.npy"
    with npy.open("xb") as f: np.save(f,source,allow_pickle=False); f.flush(); os.fsync(f.fileno())
    row["candidate_npy_path"]=str(npy); row["candidate_npy_sha256"]=sha(npy)
    target_rows=row.pop("target_audits_by_pose",None)
    if target_rows is not None:
        dest=OUT/"attempts"/f"attempt-{attempt:04d}-target-audits.json.gz"
        with gzip.open(dest,"wt",encoding="utf-8",compresslevel=6) as f: json.dump(target_rows,f,sort_keys=True,separators=(",",":"),allow_nan=False)
        row["target_audits_path"]=str(dest); row["target_audits_sha256"]=sha(dest)
    with (OUT/"attempts.jsonl").open("a") as f: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
    log("trial_event",attempt=attempt,iteration=row.get("iteration"),backtrack=row.get("backtrack"),status=row.get("status"),
        reason=row.get("reason"),target_counts=row.get("nonocular_pair_counts_candidate_by_pose"),candidate_npy_sha256=row.get("candidate_npy_sha256"))

static_paths=[SCRIPT,SKIN,SKIN_MAN,TISS,TISS_MAN,INV,SCENE,ORIENT,ORIENT_SKIN,FORWARD,TISS_FORWARD,CA,CI,
              RESP/"src/numilab_human/resting_respiratory_conforming_field.py",
              PREV/"preflight.json",PREV/"baseline-target-tables.json.gz",PREV/"progress.jsonl",PREV/"input-postcheck.json",
              PATCH,PATCH_REPORT,BASE_REFRESH_REPORT,Path(base_refresh_report["corrected_target_table_path"]),
              PARTIAL_AUDIT_REPORT,Path(partial_audit_report["corrected_target_table_path"]),FIT_EVENT_PATH,FIT_SOURCE_PATH,FIT_TARGET_PATH,
              FIT_PREV/"preflight.json",FIT_PREV/"progress.jsonl",FIT_PREV/"input-postcheck.json"]
input_pins=[pin(p) for p in static_paths]
for st in states: input_pins += [pin(st["pack"]),pin(st["receipt"])]
input_map={p["path"]:p["sha256"] for p in input_pins}
baseline_doc={str(st["step"]):st["baseline"] for st in states}
with gzip.open(OUT/"baseline-target-tables.json.gz","wt",encoding="utf-8",compresslevel=6) as f:
    json.dump(baseline_doc,f,sort_keys=True,separators=(",",":"),allow_nan=False)
baseline_path=OUT/"baseline-target-tables.json.gz"; input_pins.append(pin(baseline_path))
prehash={p["path"]:p["sha256"] for p in input_pins}
preflight={"schema":"numi.human.skin-resting-multipose-clearance-fit-preflight.v1","status":"ready_for_bounded_outward_skin_clearance_trial",
 "source_skin_sha256":sha(SKIN),"source_skin_manifest_sha256":sha(SKIN_MAN),"source_positions_f32_sha256":f32sha(skin["pos"]),
 "source_face_index_sha256":ca._face_index_sha256(faces),"source_topology":[len(skin["pos"]),len(faces)],
 "target_inventory":{"legacy_target_count":len(legacy_keys),"added_offline_targets":[f"{a}:{b}" for a,b in sorted(NEW_KEYS)],
   "complete_target_count":len(all_target_keys),"offline_targets_are_not_native_captures":True},
 "pose_steps":STEPS,"pose_source":"eight accepted22-surface MRVPACK/receipt captures; two appended SCM target rows offline-forwarded from candidate005 NHTISS using accepted registered-body poses",
 "forward_model":map_report,"source_winding":winding,
 "baseline_target_pair_table_sha256_by_pose":[s["baseline_sha"] for s in states],
 "baseline_total_pair_count_by_pose":[sum(int(v["count"]) for v in s["baseline"].values()) for s in states],
 "baseline_scm_pair_counts_by_pose":[{f"{k[0]}:{k[1]}":int(s["baseline"][f"{k[0]}:{k[1]}"]["count"]) for k in sorted(NEW_KEYS)} for s in states],
 "baseline_self_pairs_by_pose":[int(s["skin_self"]["count"]) for s in states],
 "baseline_self_proof_reuse":{"status":"identity_bound_exact_zero_self_proof_reused","cached_preflight_path":str(prior_preflight_path),
   "cached_preflight_sha256":sha(prior_preflight_path),"scan_skipped":True,"per_pose_reuse":baseline_self_proof_reuse,
   "candidate_self_audits_are_separate":True,"candidate_self_gates_remain_in_owner":True},
 "fixed_bed_source_vertex_ids":fixed.tolist(),"preserved_source_anchor_vertex_ids":anchors.tolist(),
 "solver":{"owner":"derive_shared_multipose_inferred_clearance","selected_margin_mm":0.25,"support_radius_edge_multiple":4.0,"max_iterations":2,"backtrack_count":7,
   "resume_from_accepted_candidate":True,"cumulative_source_displacement_limit_mm":MAX_CUMULATIVE_SOURCE_DISPLACEMENT_MM},
 "resume":{"source_positions_f32_sha256":resume_event["source_positions_f32_sha256"],"source_npy_sha256":sha(FIT_SOURCE_PATH),
   "candidate_target_audit_sha256":sha(partial_audit_report["corrected_target_table_path"]),"baseline_target_refresh_sha256":sha(BASE_REFRESH_REPORT),
   "self_audits_sha256":resume_event["self_audits_sha256"],"self_proof_reused":True,"initial_self_rescans_skipped_by_owner":8},
 "target_tables_path":str(baseline_path),"target_tables_sha256":sha(baseline_path),"input_pins":input_pins,
 "baseline_cache_reuse":{"source_run":str(PREV),"preflight_sha256":sha(PREV/"preflight.json"),
   "target_table_container_sha256":sha(PREV/"baseline-target-tables.json.gz"),
   "source_scan_progress_sha256":sha(PREV/"progress.jsonl"),"all_pose_target_geometry_hashes_recomputed":True,
   "all_pose_pair_table_hashes_recomputed":True},
 "qualification":"offline proposal only; exact full-pose self, orientation, bed and all-861 target gates remain authoritative; no native composition/admission"}
write_json(OUT/"preflight.json",preflight)
np.save(OUT/"starting-source-positions-f32.npy",np.asarray(skin["pos"],dtype="<f4"),allow_pickle=False)
write_json(OUT/"solve-start.json",{"status":"running_from_identity_bound_partial_candidate","preflight_sha256":sha(OUT/"preflight.json"),"max_seconds":MAX_SECONDS,
 "steps":STEPS,"target_count":861,"owner_sha256":sha(CA),"predicate_sha256":sha(CI)})
log("owner_call_start",steps=STEPS,target_count=861,baseline_pairs=preflight["baseline_total_pair_count_by_pose"],
    resume_source_sha256=resume_event["source_positions_f32_sha256"],resume_pair_counts=partial_audit_report["summary"]["full861_pairs_after"],
    max_iterations=2,cumulative_displacement_limit_mm=MAX_CUMULATIVE_SOURCE_DISPLACEMENT_MM)
try:
    candidate,owner_report=ca.derive_shared_multipose_inferred_clearance(source_positions=skin["pos"],faces=faces,
        jacobians_by_pose=None,accepted_skin_world_by_pose=captured,
        baseline_target_audits_by_pose=[s["baseline"] for s in states],
        baseline_skin_self_pairs_by_pose=[int(s["skin_self"]["count"]) for s in states],
        source_outward_face_signs=signs,scan_candidate_targets=scan,target_triangle_by_row=target_triangle,
        all_target_keys=all_text,ocular_monitor_keys=ocular,fixed_source_vertex_ids=fixed,
        bed_plane_origins_by_pose=bed_origin,bed_plane_normals_by_pose=bed_normal,
        selected_margin_mm=0.25,support_radius_edge_multiple=4.0,max_iterations=2,backtrack_count=7,
        progress_callback=persist,candidate_forward=fit_forward,baseline_replay_tolerance_m=1e-6,
        preserved_source_anchor_vertex_ids=anchors,scan_progress_callback=None,
        resume_source_positions=resume_full,resume_target_audits_by_pose=resumed_target_tables,
        resume_self_audits_by_pose=resume_event["self_audits_by_pose"],resume_provenance=resume_provenance)
except BaseException as exc:
    post={p:sha(p) for p in prehash}
    write_json(OUT/"input-postcheck.json",{"status":"unchanged" if post==prehash else "input_changed","input_count":len(post),
        "mismatches":[{"path":p,"expected":prehash[p],"observed":post[p]} for p in post if post[p]!=prehash[p]]})
    write_json(OUT/"solve-result.json",{"status":"failed_or_incomplete_candidate_solve","error_type":type(exc).__name__,
        "error":str(exc),"elapsed_wall_s":time.monotonic()-START,"attempt_log":str(OUT/"attempts.jsonl"),
        "qualification":"offline only; no candidate admitted","target_validation_failure":getattr(exc,"target_report",None)})
    raise
finally:
    signal.alarm(0)
post={p:sha(p) for p in prehash}
write_json(OUT/"input-postcheck.json",{"status":"unchanged" if post==prehash else "input_changed","input_count":len(post),
    "mismatches":[{"path":p,"expected":prehash[p],"observed":post[p]} for p in post if post[p]!=prehash[p]]})
require(post==prehash,"pinned input changed during fit")
candidate=np.asarray(candidate,dtype="<f4"); npy=OUT/"candidate-source-positions-f32.npy"
with npy.open("xb") as f: np.save(f,candidate,allow_pickle=False); f.flush(); os.fsync(f.fileno())
owner_report.update({"candidate_source_positions_npy":str(npy),"candidate_source_positions_npy_sha256":sha(npy),
 "candidate_source_positions_f32_sha256":f32sha(candidate),"pose_steps":STEPS,
 "offline_target_sources":[{"stable_id":sid,"semantic":51005,"member_id":surface_rows[sid]["member_id"],
  "vertices":rows[sid]["vc"],"faces":len(rows[sid]["faces"]),"source_tissue_sha256":sha(TISS),
  "target_geometry_kind":"offline-forward from candidate005 NHTISS bindings and accepted registered poses"} for sid in (151,152)],
 "qualification":"offline skin correction fit only; no native replay or scene admission"})
write_json(OUT/"candidate-report.json",owner_report)
write_json(OUT/"solve-result.json",{"status":owner_report.get("status"),"elapsed_wall_s":time.monotonic()-START,
 "candidate_report":str(OUT/"candidate-report.json"),"candidate_source_positions_npy":str(npy),
 "candidate_source_positions_f32_sha256":f32sha(candidate),"candidate_source_positions_npy_sha256":sha(npy),
 "initial_pair_counts":preflight["baseline_total_pair_count_by_pose"],
 "final_nonocular_pair_counts":owner_report.get("final_nonocular_pair_count_by_pose"),
 "qualification":"offline source fit only; exact gates in owner report; no native admission"})
print(json.dumps({"phase":"fit_complete","status":owner_report.get("status"),"elapsed_wall_s":time.monotonic()-START,
 "report":str(OUT/"candidate-report.json"),"report_sha256":sha(OUT/"candidate-report.json"),
 "candidate_source_sha256":f32sha(candidate)},sort_keys=True),flush=True)

