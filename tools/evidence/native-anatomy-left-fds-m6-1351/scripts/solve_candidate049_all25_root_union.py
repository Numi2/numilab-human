import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import sys, json, hashlib, importlib.util, time, traceback
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
D=A/"remaining-original-boundary-batch-001/stable-132"
ROOT=D/"joint-target-separation-013"
OUT=Path("/Users/n/numi-human-fds132-m6-1350/attempt-049/result-001")
OUT.mkdir(parents=True,exist_ok=True)
CAND_IN=Path("/Users/n/numi-human-fds132-m6-1350/attempt-048/result-001/stable-132-candidate048-current-pq-retained-root-union-candidate.npz")
CAND043=ROOT/"attempt-043/result-001/stable-132-candidate040-current-pq-joint-external-candidate.npz"
CAND034=ROOT/"attempt-034/stable-132-candidate034-joint-self-external-candidate.npz"
SOLVE_REPORT=Path("/Users/n/numi-human-fds132-m6-1350/attempt-048/result-001/proposal-report.json")
AUDIT=ROOT/"attempt-044/audit-current-pq/result-001/report.json"
AUDIT039=ROOT/"attempt-039/audit-current-pq/result-001/report.json"
AUDIT046=Path("/Users/n/numi-human-fds132-m6-1350/attempt-046/audit-current-pq/result-001/report.json")
AUDIT048=Path("/Users/n/numi-human-fds132-m6-1350/attempt-048/audit-current-pq/result-002/report.json")
AUDIT_ALL25=Path("/Users/n/numi-human-fds132-m6-1350/attempt-048/audit-all25/result-001/report.json")
PREVIOUS_PROPOSAL=Path("/Users/n/numi-human-fds132-m6-1350/attempt-048/result-001/proposal-report.json")
PREFLIGHT=ROOT/"attempt-038/audit-current-pq/preflight.json"
SOURCE_REF=D/"source-lift-001/stable-132-geometric-reference-row-patch.npz"
TISS=A/"right-scm-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_suffix(".manifest.json")
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
OWNER=Path("/Users/n/numihuman-fds-joint-active-1321")
OWNER_SNAPSHOT=ROOT/"attempt-023/owner-source-snapshot-1321/common_atlas_skin_clearance.py"
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
J=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-012/solve_terminal_self_pairs_012.py"
CAPTURES=A/"right-pq-native-mtp-two-001/baseline/native-run/accepted-geometry"
PAIR=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
LEGACY_STEPS=[47519,151999,152607,153215,153823,154431,155000]
CONTEXT_DEFS=([("current-pq","baseline",s,CAPTURES) for s in STEPS]
 + [("legacy24-baseline-late","baseline",s,PAIR/"baseline/native-run/accepted-geometry") for s in LEGACY_STEPS]
 + [("legacy24-intervention-late","intervention",s,PAIR/"intervention/native-run/accepted-geometry") for s in LEGACY_STEPS])
EXTRAS={(51005,sid) for sid in (151,152,159,160,161,162)}
SKIN=(51007,1)
MAX_TOTAL_SOURCE_MM=2.0
MAX_MOVABLE=256
MAX_TOTAL_ROWS=520
MAX_ROWS_PER_POSE=128
START=time.monotonic()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def req(condition,message):
    if not condition:
        raise RuntimeError(message)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def write_json(path,obj):
    def cv(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,Path): return str(value)
        raise TypeError(type(value).__name__)
    p=Path(path)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=cv,allow_nan=False)+"\n")
    tmp.replace(p)

def tri_key(tri,ci):
    return tuple(ci.float32_point_lattice_key(v) for v in np.asarray(tri,dtype="<f4"))

def exact_points(a,b,ci):
    return ci.triangle_intersection_points(tri_key(a,ci),tri_key(b,ci))

def witness_metrics(a,b,points,ci):
    den=float(1<<149)
    xyz=np.asarray([[float(c)/den for c in p] for p in points],dtype=np.float64)
    diameter=float(np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=2).max(initial=0.0))*1000.0
    ae=np.roll(a,-1,axis=0)-a; be=np.roll(b,-1,axis=0)-b
    axes=[np.cross(ae[0],ae[1]),np.cross(be[0],be[1])]
    axes.extend(np.cross(x,y) for x in ae for y in be)
    overlaps=[]
    for axis in axes:
        n=float(np.linalg.norm(axis))
        if not np.isfinite(n) or n<=1e-20: continue
        u=axis/n; pa=a@u; pb=b@u
        overlap=min(float(pa.max()),float(pb.max()))-max(float(pa.min()),float(pb.min()))
        overlaps.append(overlap*1000.0)
    return {"exact_intersection_point_count":len(points),
            "exact_intersection_witness_points_m":xyz.tolist(),
            "intersection_witness_diameter_mm":diameter,
            "minimum_triangle_sat_interval_overlap_mm":float(min(overlaps)) if overlaps else None}

sys.path.insert(0,str(OWNER/"src"))
from numilab_human import common_atlas_skin_clearance as ca
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import cardiac_cavity_geometry as cg
from numilab_human.resting_anatomy_interface_patch import normals
req(Path(ca.__file__).read_bytes()==OWNER_SNAPSHOT.read_bytes(),"active solver owner differs from frozen attempt023 snapshot")
wrapper=module(Path("/Users/n/numi-human-fds132-m6-1350/attempt-049/captured_root_diagnostic_wrapper_049.py"),"candidate_reference_wrapper_048")
proposer=wrapper.build_reference_clearance_proposer(ca,max_pose_samples=22)
h=module(H,"fds_forward_for_joint_active_040")
jo=module(J,"fds_jacobians_for_joint_active_040")
h.TISS=TISS; h.MAN=MAN

audit=json.loads(AUDIT.read_text())
audit039=json.loads(AUDIT039.read_text())
audit046=json.loads(AUDIT046.read_text())
audit048=json.loads(AUDIT048.read_text())
audit_all25=json.loads(AUDIT_ALL25.read_text())
previous_proposal=json.loads(PREVIOUS_PROPOSAL.read_text())
req(audit039.get("complete") is True and audit039.get("inputs_unchanged") is True and audit039.get("candidate_sha256")==sha(CAND034),"039 prior candidate report mismatch")
req(audit.get("complete") is True and audit.get("inputs_unchanged") is True,"attempt044 target audit incomplete/changed")
req(audit.get("candidate_sha256")==sha(CAND043),"attempt044 audit is not bound to candidate043")
req(audit.get("current_pq_all_external_target_keys_scanned") is True and all(x.get("target_keys_scanned")==865 for x in audit.get("current_pq_target_audits",[])),"attempt044 does not cover all 865 external targets")
req(len(audit.get("current_pq_target_audits",[]))==8,"attempt039 does not contain eight current-PQ captures")
preflight=json.loads(PREFLIGHT.read_text())
req(preflight.get("complete") is True and preflight.get("all_inputs_unchanged") is True and preflight.get("expected_external_target_surface_count_including_skin")==865,"current-PQ key preflight is incomplete")
req(sha(CAND_IN)=="772afcbd2c8492780d76a69bec214b04cad32707011967883b1c6183e5a2001e","candidate048 input hash")
req(sha(CAND034)=="337ca062046269033c7bc8411c9abc746391f0f99994dae89c431c12276dfe81","candidate034 reference hash")
req(sha(AUDIT)=="bff722b92a9d551e3f5b681fe443c50a8837f96c5bb762458137775ea13a67df","attempt044 report hash")
req(sha(AUDIT039)=="00e84d026f281551feb023e871e34097e3f98bd177c2d75b1da93668c769cfdc","attempt039 report hash")
req(sha(AUDIT046)=="82325ee3a1a2337eb138735014473961d611fdd484b5069f7fc35216ce356bcd","candidate046 current-PQ audit hash")
req(sha(AUDIT048)=="3be437bbb2fc38c8c1f1de62c65d7a479aaa9ddf79b2fcdf3d6d23d53dcea52d","candidate048 current-PQ audit hash")
req(sha(AUDIT_ALL25)=="76f60427ade3193879a8ca228a4394a82b50d8cddac7cd610db742ba9d858a87","candidate048 all-25 audit hash")
req(sha(PREVIOUS_PROPOSAL)=="2509073543aeafa2b3ee69ba92afcc1bf0af20b2ecaf47ddbf8bc610798e9590","candidate048 proposal hash")
solve=json.loads(SOLVE_REPORT.read_text())
req(solve.get("candidate_sha256")==sha(CAND_IN) and solve.get("active_external_row_total")==478 and solve.get("inputs_unchanged") is True and solve.get("candidate_admitted") is False,"candidate048 proposal provenance mismatch")
req(audit046.get("complete") is True and audit046.get("inputs_unchanged") is True,"candidate046 predecessor audit is incomplete")
req(audit048.get("candidate_sha256")==sha(CAND_IN) and audit048.get("complete") is True and audit048.get("inputs_unchanged") is True and audit048.get("current_pq_all_external_target_keys_scanned") is True and audit048.get("no_added_parent_pairs_against_current_pq_targets") is True,"candidate048 current-PQ audit is not clear/complete")
req(audit_all25.get("candidate_sha256")==sha(CAND_IN) and audit_all25.get("complete") is True and audit_all25.get("inputs_unchanged") is True and audit_all25.get("source_and_all25_sampled_pose_self_clear") is True,"candidate048 retained audit self/source gate failed")
req(previous_proposal.get("candidate_sha256")==sha(CAND_IN) and previous_proposal.get("candidate_admitted") is False and previous_proposal.get("active_external_row_total")==478,"previous candidate048 active-row report mismatch")

tissue=h.load_tissue(TISS); row=h.row_data(tissue,132)
z=np.load(CAND_IN,allow_pickle=False)
candidate_arrays={k:z[k] for k in z.files}
current_p=np.ascontiguousarray(z["vertices6"][:,:3],dtype="<f4")
faces=np.ascontiguousarray(z["faces"],dtype=np.int64)
orig=np.ascontiguousarray(z["face_origins"],dtype=np.int64)
local=np.ascontiguousarray(z["binding_indices"],dtype=np.uint32)
weights=np.ascontiguousarray(z["weights"],dtype="<f4")
source_npz=np.load(SOURCE_REF,allow_pickle=False)
source=np.ascontiguousarray(source_npz["vertices6"][:,:3],dtype="<f4")
source_faces=np.ascontiguousarray(source_npz["faces"],dtype=np.int64)
source_orig=np.ascontiguousarray(source_npz["face_origins"],dtype=np.int64)
parent_p=np.ascontiguousarray(row["positions"],dtype="<f4")
parent_faces=np.ascontiguousarray(row["faces"],dtype=np.int64)
req(current_p.shape==source.shape and faces.shape==source_faces.shape and orig.shape==source_orig.shape,"candidate/source-reference dimensions differ")
z034=np.load(CAND034,allow_pickle=False)
candidate034_p=np.ascontiguousarray(z034["vertices6"][:,:3],dtype="<f4")
req(candidate034_p.shape==current_p.shape and np.array_equal(z034["faces"],faces) and np.array_equal(z034["face_origins"],orig),"candidate034 reference topology differs")
req(np.array_equal(z034["binding_indices"],local) and np.array_equal(z034["weights"],weights),"candidate034 binding arrays differ")
req(np.array_equal(faces,source_faces) and np.array_equal(orig,source_orig),"candidate048 geometry/ancestry differs from source-lift reference")
req(np.all(orig>=0) and np.all(orig<len(parent_faces)),"candidate face ancestry points outside accepted stable132 parent faces")
req(np.array_equal(local,source_npz["binding_indices"]) and np.array_equal(weights,source_npz["weights"]),"candidate binding table differs from source-lift reference")
req(np.all(local[weights>0]<int(row["bc"])) and np.all(local[weights<=0]==np.uint32(0xffffffff)),"source-lift active binding lanes are invalid for stable132 parent row")
req(np.isfinite(current_p).all() and np.isfinite(source).all(),"non-finite candidate/source positions")
anchor_ids=np.flatnonzero(weights.max(axis=1)>=0.999999).astype(np.int64)
req(float(np.linalg.norm(current_p.astype(np.float64)-source.astype(np.float64),axis=1).max()*1000.0)<MAX_TOTAL_SOURCE_MM,"candidate048 already exceeds original source bound")
keys,*_=ca._load_target_inventory(INV)
keys=set((int(s),int(i)) for s,i in keys)
expected_keys=keys|EXTRAS
req(len(keys)==859 and (51005,132) in keys and len(expected_keys)==865,"current target-key inventory changed")
req((51005,132) in expected_keys,"subject row absent from requested pack keys")

CAND046=Path("/Users/n/numi-human-fds132-m6-1350/attempt-046/iteration-003/diagnostic-result/stable-132-candidate046-current-pq-reference-preserving-candidate.npz")
pin_paths=[Path(__file__),CAND043,CAND034,CAND046,AUDIT039,AUDIT046,AUDIT048,AUDIT_ALL25,
    PREVIOUS_PROPOSAL,Path("/Users/n/numi-human-fds132-m6-1350/attempt-047/solve_candidate047_root_union.py"),
    Path("/Users/n/numi-human-fds132-m6-1350/attempt-047/audit-current-pq/audit_source_current_pq8_all865_047.py"),
    Path("/Users/n/numi-human-fds132-m6-1350/attempt-047/audit-all25/audit_candidate047_all25_targets.py"),
    Path("/Users/n/numi-human-fds132-m6-1350/attempt-049/captured_root_diagnostic_wrapper_049.py"),
    CAND_IN,SOLVE_REPORT,AUDIT,PREFLIGHT,SOURCE_REF,TISS,MAN,INV,OWNER_SNAPSHOT,H,J,
    Path(ca.__file__),Path(ci.__file__),Path(cg.__file__),
    ROOT/"one-context-preflight-001.py",ROOT/"one-context-preflight-001/one-context-preflight.json",
    ROOT/"one-context-preflight-001.stdout.log",ROOT/"one-context-preflight-001.stderr.log"]
for group,arm,step,capture_dir in CONTEXT_DEFS:
    pa=capture_dir/f"step-{step}.mrvpack";rp=capture_dir/f"step-{step}.receipt.json"
    pin_paths.extend([pa,rp])
pins={str(p):sha(p) for p in pin_paths}

def pose_map(receipt):
    d=json.loads(receipt.read_text())
    rows=d.get("accepted_body_poses") or d.get("accepted_registered_body_poses")
    req(isinstance(rows,list) and len(rows) in (86,157),f"accepted capture lacks 86 or 157 accepted body poses: {receipt}")
    return {int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),np.asarray(x["quaternion_xyzw"],dtype=np.float32)) for x in rows}

def prepare_context(spec, sample_index):
    group,arm,step,capture_dir=spec
    pa=capture_dir/f"step-{step}.mrvpack"
    rp=capture_dir/f"step-{step}.receipt.json"
    req(pa.is_file() and rp.is_file(),f"missing accepted capture {group}/{arm}/{step}")
    if group=="current-pq":
        expected_pack=preflight["input_pins"].get(str(pa))
        expected_receipt=preflight["input_pins"].get(str(rp))
    else:
        expected_pack=audit_all25["pins"].get(str(pa))
        expected_receipt=audit_all25["pins"].get(str(rp))
    req(expected_pack and expected_receipt and sha(pa)==expected_pack and sha(rp)==expected_receipt,
        f"accepted capture pin mismatch {group}/{arm}/{step}")
    pmap=pose_map(rp)
    expected_keys=keys|EXTRAS if group=="current-pq" else keys
    packed_positions,surfaces,_=ca._pack_surfaces(pa,expected_keys)
    req(set(surfaces)==expected_keys|{SKIN},f"full target keyset changed at {group}/{step}")
    req(len(surfaces)==(866 if group=="current-pq" else 860),f"wrong target-surface count at {group}/{step}")
    parent_global_faces=np.asarray(surfaces[(51005,132)]["faces"],dtype=np.int64)
    offsets=parent_global_faces-parent_faces
    req(offsets.size and np.all(offsets==offsets.flat[0]),f"parent face-table offset invalid at {group}/{step}")
    v0=int(offsets.flat[0])
    parent_world=np.ascontiguousarray(packed_positions[v0:v0+len(parent_p)],dtype="<f4")
    req(parent_world.shape==parent_p.shape,"captured parent row position span mismatch")
    parent_replay=np.ascontiguousarray(h.forward(parent_p,row["local"],row["weights"],row,tissue,pmap),dtype="<f4")
    used=np.unique(parent_faces)
    replay=float(np.linalg.norm(parent_replay[used].astype(np.float64)-parent_world[used].astype(np.float64),axis=1).max(initial=0.0))
    req(replay<=6e-7,f"captured-root replay exceeds 0.6um at {group}/{step}: {replay}")
    world=np.ascontiguousarray(h.forward(current_p,local,weights,row,tissue,pmap),dtype="<f4")
    ref_root=wrapper.tag_reference_geometry(kind="captured-root",path=pa,expected_sha256=expected_pack,
        pose_step=int(step),world_positions=parent_world,faces=parent_faces)
    target_faces_by_key={(int(k[0]),int(k[1])):np.asarray(v["faces"],dtype=np.int64) for k,v in surfaces.items()}
    def target_tri_for(sem,sid,tf):
        tf=int(tf); tfaces=target_faces_by_key[(int(sem),int(sid))]
        req(0<=tf<len(tfaces),f"target face out of range {(sem,sid,tf)}")
        return np.ascontiguousarray(packed_positions[tfaces[tf]],dtype="<f4")
    external=[]; row_tags=[]; witness_rows=[]; rowkeys=set(); new_parent_pairs=set(); new_children=set()
    def bind(child,pf,sem,sid,tf,target_tri,current_tri):
        erow,tag=wrapper.bind_external_row(sample_index=int(sample_index),sample_pose_step=int(step),
            candidate_face_row=int(child),target_semantic=int(sem),target_stable_id=int(sid),target_face_id=int(tf),
            target_triangle_world_positions=target_tri,current_candidate_triangle=current_tri,
            reference_geometry=ref_root,reference_face_row=int(pf),ci=ci)
        external.append(erow); row_tags.append(tag)
        rowkeys.add((int(child),int(pf),int(sem),int(sid),int(tf)))
    if group=="current-pq":
        old_rows=[x for x in previous_proposal["reference_rows"] if int(x["sample_pose_step"])==int(step)]
        req(old_rows,f"no candidate048 current-PQ preservation rows at {step}")
        for old in old_rows:
            child=int(old["candidate_face_row"]); pf=int(orig[child])
            sem=int(old["target_semantic"]);sid=int(old["target_stable_id"]);tf=int(old["target_face_id"])
            req(old["reference_kind"]=="captured-root" and int(old["reference_face_row"])==pf,
                f"candidate048 preserved row is not bound to captured-root ancestry at {step}: {old}")
            target_tri=target_tri_for(sem,sid,tf)
            current_tri=np.ascontiguousarray(world[faces[child]],dtype="<f4")
            bind(child,pf,sem,sid,tf,target_tri,current_tri)
    elif group.startswith("legacy24-"):
        old_rows=[x for x in previous_proposal.get("reference_rows",[])
                  if x.get("group")==group and x.get("arm")==arm
                  and int(x["sample_pose_step"])==int(step)]
        for old in old_rows:
            child=int(old["candidate_face_row"]); pf=int(orig[child])
            sem=int(old["target_semantic"]); sid=int(old["target_stable_id"]); tf=int(old["target_face_id"])
            req(old["reference_kind"]=="captured-root" and int(old["reference_face_row"])==pf,
                f"candidate048 prior legacy row lacks captured-root ancestry at {group}/{step}: {old}")
            target_tri=target_tri_for(sem,sid,tf)
            current_tri=np.ascontiguousarray(world[faces[child]],dtype="<f4")
            bind(child,pf,sem,sid,tf,target_tri,current_tri)
        audit=next((x for x in audit_all25["all25_retained_target_audits"]
                    if x["group"]==group and int(x["step"])==int(step)),None)
        req(audit is not None,f"missing retained target audit row {group}/{step}")
        for sem,sid,pf,tf in audit["added_parent_pairs"]:
            sem=int(sem);sid=int(sid);pf=int(pf);tf=int(tf)
            target_tri=target_tri_for(sem,sid,tf); hit_children=[]
            for child in np.flatnonzero(orig==pf):
                child=int(child); current_tri=np.ascontiguousarray(world[faces[child]],dtype="<f4")
                points=exact_points(current_tri,target_tri,ci)
                if points:
                    bind(child,pf,sem,sid,tf,target_tri,current_tri)
                    hit_children.append(child)
                    witness_rows.append({"group":group,"arm":arm,"step":step,"reference_kind":"captured-root",
                        "target_semantic":sem,"target_stable_id":sid,"parent_face_row":pf,
                        "candidate_face_row":child,"target_face_id":tf,**witness_metrics(current_tri,target_tri,points,ci)})
                    new_children.add((child,pf,sem,sid,tf))
            req(hit_children,f"candidate048 retained added pair does not reproduce exactly at {group}/{step}: {(sem,sid,pf,tf)}")
            new_parent_pairs.add((sem,sid,pf,tf))
        req(len(new_parent_pairs)==len(audit["added_parent_pairs"]),
            f"retained added-parent-pair count mismatch at {group}/{step}")
    elif group!="K1":
        raise RuntimeError(f"unknown accepted pose group {group}")
    unique={}; unique_tags=[]
    for ext,tag in zip(external,row_tags):
        k=(ext["candidate_face_row"],ext["parent_face_row"],ext["target_semantic"],ext["target_stable_id"],ext["target_face_id"])
        if k not in unique:
            unique[k]=ext;unique_tags.append(tag)
    external=list(unique.values());row_tags=unique_tags
    req(len(external)<=MAX_ROWS_PER_POSE,
        f"external active rows per pose exceed owner {MAX_ROWS_PER_POSE} bound at {group}/{step}: {len(external)}")
    jac=np.asarray(jo.pose_jacobians(current_p,local,weights,row,tissue,pmap),dtype=np.float64)
    req(jac.shape==(len(current_p),3,3) and np.isfinite(jac).all(),f"forward Jacobian malformed at {group}/{step}")
    vids=np.arange(len(current_p),dtype=np.int64)
    sample={"vertex_ids":vids,"baseline_world_positions":world.copy(),"rejected_world_positions":world.copy(),
        "jacobians":jac,"accepted_parent_world_positions":parent_world.copy(),
        "accepted_parent_faces":parent_faces.copy(),"external_clearance_rows":external}
    info={"group":group,"arm":arm,"step":step,"pack":str(pa),"receipt":str(rp),
        "pack_sha256":sha(pa),"receipt_sha256":sha(rp),"accepted_pose_count":len(pmap),
        "captured_root_replay_max_error_m":replay,"preserved_current048_rows":len(rowkeys) if group=="current-pq" else 0,
        "new_legacy_parent_pairs":len(new_parent_pairs),"new_legacy_child_rows":len(new_children),
        "candidate_child_constraint_rows":len(external),"target_ids":sorted({(x["target_semantic"],x["target_stable_id"]) for x in external})}
    ref_rows=[]
    for ext,tag in zip(external,row_tags):
        ref_rows.append({"sample_index":sample_index,"group":group,"arm":arm,"step":step,
            "candidate_face_row":ext["candidate_face_row"],"packed_reference_face_row":ext["parent_face_row"],
            "target_semantic":ext["target_semantic"],"target_stable_id":ext["target_stable_id"],
            "target_face_id":ext["target_face_id"],**tag})
    del packed_positions,surfaces,parent_replay
    return {"sample":sample,"world":world,"info":info,"witnesses":witness_rows,
            "row_tags":row_tags,"reference_rows":ref_rows}

def make_movable(contexts):
    seed=set()
    for ctx in contexts:
        for ext in ctx["sample"]["external_clearance_rows"]:
            seed.update(map(int,faces[int(ext["candidate_face_row"])]))
    protected=set(map(int,anchor_ids))
    seed_free=seed-protected
    req(seed_free,"all active source face vertices are protected")
    adjacency=[set() for _ in range(len(current_p))]
    for tri in faces:
        a,b,c=map(int,tri)
        adjacency[a].update((b,c));adjacency[b].update((a,c));adjacency[c].update((a,b))
    chosen=None; chosen_rings=None
    for rings in (2,1,0):
        movable=set(seed_free);frontier=set(seed_free)
        for _ in range(rings):
            nxt={n for v in frontier for n in adjacency[v] if n not in movable and n not in protected}
            movable.update(nxt);frontier=nxt
        if len(movable)<=MAX_MOVABLE:
            chosen=np.asarray(sorted(movable),dtype=np.int64);chosen_rings=rings;break
    req(chosen is not None,f"active source vertices exceed owner {MAX_MOVABLE} movable-vertex bound")
    return np.asarray(sorted(seed_free),dtype=np.int64),chosen,chosen_rings

def helper_call(contexts,movable,seeds):
    return proposer(
        source_positions_m=current_p,faces=faces,movable_vertex_ids=movable,self_patch_seed_vertex_ids=seeds,
        required_target_seed_vertex_ids=np.empty(0,dtype=np.int64),
        active_target_face_vertex_ids=np.empty(0,dtype=np.int64),
        fixed_support_vertex_ids=anchor_ids,preserved_anchor_vertex_ids=anchor_ids,
        pose_samples=[x["sample"] for x in contexts])

# One current-PQ context proves the owner row schema with the 22-context local resource wrapper.
one_context=[prepare_context(CONTEXT_DEFS[1],0)]
one_seeds,one_movable,one_rings=make_movable(one_context)
one_proposal=helper_call(one_context,one_movable,one_seeds)
one_positions=np.ascontiguousarray(one_proposal["proposal_source_positions_m_f32"],dtype="<f4")
req(one_positions.shape==current_p.shape and np.isfinite(one_positions).all(),"one-context helper returned malformed positions")
req(one_positions[anchor_ids].tobytes()==current_p[anchor_ids].tobytes(),"one-context helper changed one-hot attachment proxies")
one_preflight={"schema":"stable132-multipose-root-reference-one-context-preflight.v1",
 "status":"one_context_actual_owner_call_passed","group":"current-pq","step":STEPS[1],"candidate_sha256":sha(CAND_IN),
 "candidate048_current_pq_audit_sha256":sha(AUDIT048),"candidate048_all25_audit_sha256":sha(AUDIT_ALL25),
 "target_surface_count":866,"external_target_count":865,"row_info":one_context[0]["info"],
 "witness_rows":one_context[0]["witnesses"],"active_child_face_row_count":one_context[0]["info"]["candidate_child_constraint_rows"],
 "movable_vertex_count":int(len(one_movable)),"support_rings":one_rings,"seed_vertex_count":int(len(one_seeds)),
 "one_context_helper_increment_l2_mm":float(one_proposal["float32_max_vertex_l2_increment_mm"]),
 "one_context_helper_source_max_from_original_mm":float(np.linalg.norm(one_positions.astype(np.float64)-source.astype(np.float64),axis=1).max()*1000.0),
 "one_context_tagged_reference_and_exact_candidate_checks":"passed for every row",
 "pins":pins,"inputs_unchanged":True,"candidate_written":False}
write_json(OUT/"one-context-preflight.json",one_preflight)
print(json.dumps({"stage":"one-context-preflight-passed","step":STEPS[1],
 "child_constraint_rows":one_preflight["active_child_face_row_count"],"movable_vertices":len(one_movable),
 "rings":one_rings,"increment_mm":one_preflight["one_context_helper_increment_l2_mm"]},sort_keys=True),flush=True)

contexts=[prepare_context(spec,i) for i,spec in enumerate(CONTEXT_DEFS)]
seed_ids,movable,rings=make_movable(contexts)
row_counts=[len(x["sample"]["external_clearance_rows"]) for x in contexts]
total_rows=sum(row_counts)
req(total_rows==514,f"candidate048-to-049 union row count changed; expected 514, got {total_rows}")
req(total_rows<=MAX_TOTAL_ROWS,f"joint active set exceeds frozen external row cap {MAX_TOTAL_ROWS}: {total_rows}")
req(len(contexts)==22 and max(row_counts)<=MAX_ROWS_PER_POSE,
    f"22-pose active set exceeds exact context or per-pose {MAX_ROWS_PER_POSE} guard")
preflight_all={"schema":"stable132-current-pq-and-retained-22-pose-root-reference-preflight.v1",
 "status":"all_twenty_two_exact_active_rows_validated","candidate_sha256":sha(CAND_IN),
 "candidate048_current_pq_audit_sha256":sha(AUDIT048),"candidate048_all25_audit_sha256":sha(AUDIT_ALL25),
 "contexts":[x["info"] for x in contexts],"context_group_steps":[{"group":g,"arm":a,"step":st} for g,a,st,_ in CONTEXT_DEFS],
 "external_rows_per_pose":row_counts,"external_child_pair_rows_total":total_rows,
 "frozen_owner_external_row_bound":520,"active_external_row_bound":520,
 "frozen_owner_pose_count_bound":17,"in_memory_pose_count_bound":22,
 "pose_count_resource_guard_extended_in_memory_only":True,
 "frozen_helper_file_unchanged":sha(Path(ca.__file__))=="fccc7f29d421c9ad8e7da40d61d16cc8633244a16829dc7a7cd9d83ff109cfae",
 "geometric_guards_changed":False,"resource_bound_changed":True,
 "movable_vertex_count":int(len(movable)),"seed_vertex_count":int(len(seed_ids)),"support_rings":rings,
 "pure_onehot_anchor_count":int(len(anchor_ids)),"all_onehot_vertices_fixed":True,
 "binding_indices_weights_identical_to_source_lift_reference":True,
 "witness_rows":[w for x in contexts for w in x["witnesses"]],
 "reference_rows":[r for x in contexts for r in x["reference_rows"]],
 "inputs_unchanged":all(sha(p)==hsh for p,hsh in pins.items()),"pins":pins}
req(preflight_all["inputs_unchanged"],"a pinned input changed while preparing the all-pose active set")
write_json(OUT/"all-twenty-two-active-set-preflight.json",preflight_all)
print(json.dumps({"stage":"all-twenty-two-active-set-passed","external_rows":total_rows,"per_pose":row_counts,
 "movable_vertices":len(movable),"rings":rings,
 "max_witness_diameter_mm":max((x["intersection_witness_diameter_mm"] for x in preflight_all["witness_rows"]),default=0.0)},sort_keys=True),flush=True)

proposal=helper_call(contexts,movable,seed_ids)
proposed=np.ascontiguousarray(proposal["proposal_source_positions_m_f32"],dtype="<f4")
req(proposed.shape==current_p.shape and np.isfinite(proposed).all(),"joint helper returned malformed source positions")
req(proposed[anchor_ids].tobytes()==current_p[anchor_ids].tobytes(),"joint helper changed pure one-hot attachment proxies")
max_from_ref=float(np.linalg.norm(proposed.astype(np.float64)-source.astype(np.float64),axis=1).max()*1000.0)
max_increment=float(np.linalg.norm(proposed.astype(np.float64)-current_p.astype(np.float64),axis=1).max()*1000.0)
req(max_from_ref<=MAX_TOTAL_SOURCE_MM+1e-9,f"joint candidate exceeds original 2mm source bound: {max_from_ref:.9g}mm")
data=candidate_arrays.copy()
data["vertices6"]=np.column_stack((proposed,normals(proposed.astype(float),faces))).astype("<f4")
candidate_out=OUT/"stable-132-candidate049-current-pq-retained-root-union-candidate.npz"
np.savez(candidate_out,**data)
candidate_sha=sha(candidate_out)
report={"schema":"stable132-current-pq-candidate-reference-preserving-proposal.v2","status":"candidate_written_unadmitted",
 "candidate_path":str(candidate_out),"candidate_sha256":candidate_sha,"starting_candidate_path":str(CAND_IN),
 "starting_candidate_sha256":sha(CAND_IN),"original_source_reference_path":str(SOURCE_REF),"original_source_reference_sha256":sha(SOURCE_REF),
 "starting_candidate048_current_pq_audit_path":str(AUDIT048),"starting_candidate048_current_pq_audit_sha256":sha(AUDIT048),"starting_candidate048_all25_audit_path":str(AUDIT_ALL25),"starting_candidate048_all25_audit_sha256":sha(AUDIT_ALL25),
 "active_target_identities":sorted({(int(r["target_semantic"]),int(r["target_stable_id"])) for c in contexts for r in c["reference_rows"]}),
 "candidate_reference_status":"candidate048 is the exact starting candidate identity. Every retained current-PQ row and every reproduced legacy added-pair row uses exact captured-root geometry from its own accepted pose; no candidate is represented as accepted physical/source baseline.",
 "active_external_rows_per_pose":row_counts,"active_external_row_total":total_rows,"context_count":len(contexts),
 "frozen_owner_pose_count_bound":17,"in_memory_pose_count_bound":22,
 "pose_count_resource_guard_extended_in_memory_only":True,
 "frozen_helper_file_unchanged":sha(Path(ca.__file__))=="fccc7f29d421c9ad8e7da40d61d16cc8633244a16829dc7a7cd9d83ff109cfae",
 "external_row_bound_unchanged":520,"geometric_guards_changed":False,"context_groups":[{"group":g,"arm":a,"step":st} for g,a,st,_ in CONTEXT_DEFS],
 "full_twenty_two_pose_source_local_proposal":True,"all_candidate048_prior_active_clearances_and_all_candidate048_all25_residual_intersections_constrained_against_exact_tagged_captured_root":True,"legacy_added_external_pairs_constrained_against_their_exact_captured_root_geometry":True,"attachment_lock_note":"Stable132 source-lift has no pure one-hot vertices; binding arrays are preserved byte-for-byte and no vertex proxy is claimed.","binding_indices_weights_identical_to_source_lift_reference":True,"support_rings":rings,"movable_vertex_count":len(movable),
 "pure_onehot_anchor_count":len(anchor_ids),"pure_onehot_anchor_bytes_identical":True,"certified_pure_onehot_vertex_proxy_available":bool(len(anchor_ids)),"binding_indices_weights_identical_to_source_lift_reference":True,
 "max_increment_l2_from_candidate048_mm":max_increment,"max_l2_from_original_source_reference_mm":max_from_ref,
 "inference_scope":"Offline passive source-position reference correction only. Binding arrays, routes, endpoint proxies, target geometry, and accepted poses are unchanged. This is not anatomy-admitted.",
 "external_floor":"Every active row uses the frozen-owner min(exact captured-root SAT gap, 0.001 mm); current candidate status does not change that reference-side floor.",
 "postchecks_required":["source and all25 exact self/closed/orientation/degenerate gates","current-PQ eight captures and all865 external targets including skin","all pure attachment proxies bit-identical","no target pair regression outside active set"],
 "one_context_schema_preflight":str(OUT/"one-context-preflight.json"),
 "all_twenty_two_active_set_preflight":str(OUT/"all-twenty-two-active-set-preflight.json"),
 "witness_rows":[w for x in contexts for w in x["witnesses"]],
 "reference_rows":[r for x in contexts for r in x["reference_rows"]],
 "helper_receipt":{**{k:v for k,v in proposal.items() if not isinstance(v,np.ndarray) and k!="external_clearance_pair_receipts"},"external_clearance_pair_receipts":wrapper.retag_external_receipts(proposal,{(int(t["sample_index"]),int(t["candidate_face_row"]),int(t["packed_reference_face_row"]),int(t["target_semantic"]),int(t["target_stable_id"]),int(t["target_face_id"])):t for c in contexts for t in c["reference_rows"]})},
 "inputs_unchanged":all(sha(p)==hsh for p,hsh in pins.items()),"pins":pins,
 "elapsed_seconds":time.monotonic()-START,"candidate_admitted":False}
req(report["inputs_unchanged"],"a pinned input changed during solve")
write_json(OUT/"proposal-report.json",report)
print(json.dumps({"stage":"candidate-written","path":str(candidate_out),"sha256":candidate_sha,
 "active_rows":total_rows,"max_increment_mm":max_increment,"max_from_original_mm":max_from_ref,
 "inputs_unchanged":report["inputs_unchanged"],"elapsed_s":report["elapsed_seconds"]},sort_keys=True),flush=True)
