#!/Users/n/numi-human-prep-venv-20261005/bin/python3.13
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib, importlib.util, inspect, json, sys, time, traceback
from pathlib import Path
import numpy as np

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
FIT = BASE / "fit-attempt-002-adapter-revision-003/fit_17_pose_source_clearance.py"
FROZEN_OWNER = Path("/Users/n/numi-human-conforming-composition-source-1216/src/numilab_human/common_atlas_skin_clearance.py")
PATCH_OWNER = Path("/Users/n/numi-human-feasible-direction-1236/src/numilab_human/common_atlas_skin_clearance.py")
TEST_FILE = Path("/Users/n/numi-human-feasible-direction-1236/tests/test_common_atlas_skin_clearance.py")
CAND = BASE / "bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
GATE = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-verification.json"
TABLES = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json"
OUT = BASE / "resume-prepare-d319-004/conditioning-full-active-1236/run-001"
REPORT = OUT / "conditioning.json"
NPY = OUT / "conditioned-source-directions.npy"
OUT.mkdir(parents=True, exist_ok=False)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def need(cond,msg):
    if not cond:
        raise RuntimeError(msg)

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

fit=loadmod(FIT,"fit17_full_conditioning_1236")
frozen=fit.c
patched=loadmod(PATCH_OWNER,"numilab_human.common_atlas_skin_clearance_feasible1236")
# Only the reviewed conditioner implementation is substituted. All loading,
# forward, target, identity, and other owner helpers remain the frozen module.
frozen._condition_shared_source_directions=patched._condition_shared_source_directions
source_lines,line0=inspect.getsourcelines(fit.main)
stop_line=next(line0+i for i,row in enumerate(source_lines) if row.strip()=="if args.prepare_check:")
class Stop(Exception): pass
result_data={}

def condition_from_locals(locals_):
    t0=time.monotonic()
    ref=np.asarray(fit.REFERENCE_IDS,dtype=np.int64)
    full_faces=np.asarray(locals_["skin_faces"],dtype=np.int64)
    need(full_faces.ndim==2 and full_faces.shape[1]==3,"malformed source faces")
    face_ids=np.searchsorted(ref,full_faces)
    need(np.array_equal(ref[face_ids],full_faces),"full-to-compact source face map mismatch")
    candidate=np.load(CAND,allow_pickle=False)
    need(candidate.dtype==np.dtype("<f4") and candidate.shape==(len(ref),3),"d319 compact candidate shape/dtype mismatch")
    full_source=np.asarray(fit.CURRENT_SKIN["pos"],dtype="<f4").copy()
    full_source[ref]=candidate
    forward=locals_["combined_forward"](full_source)
    need(forward.get("diagnostics",{}).get("admissible") is True,"d319 candidate forward rejected")
    world=np.asarray(forward["world_positions_by_pose"],dtype="<f4")
    maps=np.asarray(forward["jacobians_by_pose"],dtype=np.float64)
    need(world.shape==(17,len(ref),3) and maps.shape==(17,len(ref),3,3),"forward shape mismatch")
    gate=json.loads(GATE.read_text())
    tables=json.loads(TABLES.read_text())
    need(gate.get("status")=="candidate_target_and_non_target_gate_scan_complete","d319 full17 gate incomplete")
    coverage=gate.get("candidate_target_tables_cover_all_859_targets_by_pose",[])
    need(len(coverage)==17 and all(coverage),"d319 exact target coverage incomplete")
    gate_rows=gate.get("candidate_non_target_gate_checks",[])
    need(len(gate_rows)==17,"d319 non-target gate rows missing")
    need(len(tables.get("poses",[]))==17,"d319 target table pose count mismatch")
    all_keys,nonocular,_,_=frozen._load_target_inventory(fit.INV)
    nonocular_keys={f"{int(s)}:{int(i)}" for s,i in nonocular}
    active=set()
    active_counts=[]
    for pose_index,table in enumerate(tables["poses"]):
        pose_active=set()
        for key in nonocular_keys:
            row=table.get(key)
            need(isinstance(row,dict),f"missing exact target row {pose_index}:{key}")
            pose_active.update(int(pair[0]) for pair in row.get("triangle_pairs",[]))
        active_counts.append(len(pose_active))
        active.update(pose_active)
    active=np.asarray(sorted(active),dtype=np.int64)
    active_faces=face_ids[active]
    need(len(active)>0 and active.min()>=0 and active.max()<len(face_ids),"invalid active-face union")
    # The full17 gate already establishes all-pose nondegenerate output. Rebuild
    # these exact candidate face normals only; no intersection/self scan occurs.
    tri=world[:,active_faces,:].astype(np.float64)
    area_vectors=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0])
    area_lengths=np.linalg.norm(area_vectors,axis=2)
    need(np.isfinite(area_lengths).all() and np.all(area_lengths>0),"active face is degenerate in candidate forward")
    normals=area_vectors/area_lengths[:,:,None]
    # Match the owner's source-direction preparation exactly on all referenced
    # vertices and poses.
    area_normals=np.stack([patched._area_weighted_vertex_normals(world[p],face_ids) for p in range(17)])
    smooth=np.stack([patched._smooth_vertex_directions(area_normals[p],face_ids) for p in range(17)])
    preferred,_,_,direction_summary=patched._shared_source_directions_from_pose_normals(maps,smooth)
    source_unit=preferred/np.linalg.norm(preferred,axis=1)[:,None]
    unit_normals=normals/np.linalg.norm(normals,axis=2)[:,:,None]
    mapped=np.einsum("pfcij,fcj->pfci",maps[:,active_faces],source_unit[active_faces])
    mapped_lengths=np.linalg.norm(mapped,axis=3)
    projections=np.einsum("pfci,pfi->pfc",mapped/mapped_lengths[...,None],unit_normals)
    offending=np.unique(active_faces[np.any(projections<patched._MIN_CANDIDATE_DIRECTION_PROJECTION,axis=0)])
    need(len(offending)>0,"unexpectedly no offending vertices; this is not a conditioner reproduction")
    optimizer_calls=[]
    import scipy.optimize
    original_minimize=scipy.optimize.minimize
    context={"maps":maps,"faces":active_faces,"normals":unit_normals,"offending":offending}
    def record_minimize(*args,**kwargs):
        call_index=len(optimizer_calls)
        vertex=int(context["offending"][call_index]) if call_index<len(context["offending"]) else None
        r=original_minimize(*args,**kwargs)
        item={"call_index":call_index,"compact_vertex_id":vertex,"success":bool(r.success),
              "status":int(getattr(r,"status",-1)),"message":str(getattr(r,"message","")),
              "nit":int(getattr(r,"nit",0)),"objective":float(r.fun),"result_x":np.asarray(r.x).tolist(),
              "result_norm":float(np.linalg.norm(r.x))}
        if vertex is not None:
            inc=np.flatnonzero(np.any(context["faces"]==vertex,axis=1))
            vm=np.repeat(context["maps"][:,vertex,None,:,:],len(inc),axis=1).reshape(-1,3,3)
            vn=context["normals"][:,inc].reshape(-1,3)
            d=np.asarray(r.x,dtype=np.float64)
            length=float(np.linalg.norm(d))
            if np.isfinite(d).all() and length>0:
                w=np.einsum("kij,j->ki",vm,d/length)
                wl=np.linalg.norm(w,axis=1)
                align=np.einsum("ki,ki->k",w/wl[:,None],vn)
                item["minimum_normalized_all_constraint_alignment"]=float(align.min())
                item["constraint_count"]=int(len(align))
            else:
                item["minimum_normalized_all_constraint_alignment"]=None
        optimizer_calls.append(item)
        return r
    scipy.optimize.minimize=record_minimize
    start=time.monotonic()
    conditioned=None
    conditioner_report=None
    error=None
    try:
        conditioned,conditioner_report=frozen._condition_shared_source_directions(
            maps,preferred,active_faces,normals
        )
    except Exception as exc:
        error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    finally:
        scipy.optimize.minimize=original_minimize
    if conditioned is not None:
        np.save(NPY,np.asarray(conditioned,dtype="<f8"),allow_pickle=False)
    tracked=[dict(x) for x in locals_["tracked"]]
    extra=[FIT,FROZEN_OWNER,PATCH_OWNER,TEST_FILE,CAND,GATE,TABLES,OUT.parent/"run_active_conditioning.py"]
    known={x["path"] for x in tracked}
    for path in extra:
        if path.is_file() and str(path.resolve()) not in known:
            tracked.append(fit.pin(path))
            known.add(str(path.resolve()))
    before={x["path"]:x["sha256"] for x in tracked}
    after={path:sha(path) for path in before}
    all_same=before==after
    report={"schema":"numi.human.active-face-direction-conditioning-1236.v1",
            "status":"conditioned_all_active_faces" if conditioner_report is not None else "conditioning_stopped_at_optimizer_or_gate",
            "qualification":"17-pose active-face direction conditioning only; reused pinned d319 forwards/tables, no target/self intersection scans, no fit, no asset/native changes.",
            "fit_adapter":{"path":str(FIT),"sha256":sha(FIT)},
            "frozen_owner":{"path":str(FROZEN_OWNER),"sha256":sha(FROZEN_OWNER)},
            "patched_worktree_owner":{"path":str(PATCH_OWNER),"sha256":sha(PATCH_OWNER)},
            "worktree_head":"a438655af5efd47dee47bc1971fda8e64ca5a82e",
            "candidate":{"path":str(CAND),"sha256":sha(CAND),"shape":list(candidate.shape),"dtype":str(candidate.dtype)},
            "full17_gate":{"path":str(GATE),"sha256":sha(GATE),"target_tables":{"path":str(TABLES),"sha256":sha(TABLES)},
                           "pose_count":17,"coverage_per_pose":coverage},
            "pose_count":17,"active_union_face_count":int(len(active)),"active_faces_per_pose_count":active_counts,
            "offending_vertex_count":int(len(offending)),"first_20_offending_compact_ids":offending[:20].astype(int).tolist(),
            "preferred_direction_summary":direction_summary,
            "optimizer_call_count":len(optimizer_calls),"optimizer_calls":optimizer_calls,
            "conditioner_report":conditioner_report,"error":error,
            "inputs_unchanged":bool(all_same),"input_pin_count":len(tracked),
            "elapsed_conditioning_wall_seconds":time.monotonic()-start,
            "elapsed_total_after_preflight_wall_seconds":time.monotonic()-t0,
            "conditioned_directions_npy":({"path":str(NPY),"sha256":sha(NPY),"shape":list(conditioned.shape),"dtype":"<f8"}
                                          if conditioned is not None else None)}
    write_json(REPORT,report)
    if not all_same:
        raise RuntimeError("input hash mismatch after active-face conditioning")
    print(json.dumps({"status":report["status"],"report":str(REPORT),"report_sha256":sha(REPORT),
                      "active_union_face_count":len(active),"offending_vertex_count":len(offending),
                      "optimizer_call_count":len(optimizer_calls),"first_optimizer_calls":optimizer_calls[:5],
                      "error":error,"inputs_unchanged":all_same,"report_json_sha256":sha(REPORT)},sort_keys=True),flush=True)
    raise Stop()

source_lines,first_line=inspect.getsourcelines(fit.main)
line=next(first_line+i for i,row in enumerate(source_lines) if row.strip()=="if args.prepare_check:")
seen=False
def tracer(frame,event,arg):
    global seen
    if not seen and event=="line" and frame.f_code is fit.main.__code__ and frame.f_lineno==line:
        seen=True
        sys.settrace(None)
        condition_from_locals(frame.f_locals)
    return tracer
sys.argv=[str(FIT),"--prepare-check"]
sys.settrace(tracer)
try:
    fit.main()
except Stop:
    pass
finally:
    sys.settrace(None)
if not seen:
    raise RuntimeError("prepare-check interception did not occur")

