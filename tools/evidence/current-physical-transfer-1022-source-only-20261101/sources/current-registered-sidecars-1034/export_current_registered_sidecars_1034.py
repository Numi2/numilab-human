#!/usr/bin/env python3
"""Current-proof adapter for the existing 230 registered-surface sidecar owner.

It keeps the full current source row namespace in 230 node/face sidecars. Rows
unreferenced by every current source face are emitted as provenance-only rows
with exact coordinates and no parent/frame registration. The pinned 938 map,
808 all185 proof, and 1022 shell/binding jointly authorize that inactive set.
No physical-admission claim is made here.
"""
from __future__ import annotations
mod = None
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any

import numpy as np

H = Path(__file__).resolve().parent.parent
OWNER_PATH = H / "current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py"
OWNER_SHA = "121e3bf44d90ba3d7b82ccaa40a26a92d1675c6d277a4409ae95f8c05ad2485d"
SCHEMA = "numi.h706.current-230-full-source-row-sidecars-1034.v1"
ROLE_MAP_KEY = {"O":"O_source_row_to_joint_node","I":"I_source_row_to_joint_node","D":"D_source_row_to_joint_node"}
ROLE_ACTIVE_KEY = {"O":"O_active_source_vertex_rows","I":"I_active_source_vertex_rows","D":"D_active_source_vertex_rows"}
ROLE_INACTIVE_KEY = {"O":"O_inactive_source_vertex_rows","I":"I_inactive_source_vertex_rows","D":"D_inactive_source_vertex_rows"}
FACE_KEYS = ("face_source_layer","face_source_body","face_source_bone","face_source_face_id",
             "face_source_original_id","face_source_identity0","face_source_identity1")
ANCESTRY_KEYS = ("face_ancestry_source_face_rows","face_ancestry_source_original_ids",
                 "face_ancestry_parent_face_rows")

def sha(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def pin(path: str | Path, expected: str, label: str) -> Path:
    p=Path(path).resolve()
    if not p.is_file() or sha(p)!=expected: raise ValueError(f"{label} path/hash mismatch: {p}")
    return p

def record_matches(entry: dict[str,Any], path: Path, label: str) -> None:
    if Path(str(entry.get("path",""))).resolve()!=path.resolve() or entry.get("sha256")!=sha(path):
        raise ValueError(f"{label} does not bind current path/hash")

def _same_path_pin(entry: Any, path: Path, digest: str, label: str) -> None:
    if not isinstance(entry,dict) or Path(str(entry.get("path",""))).resolve()!=path.resolve() or entry.get("sha256")!=digest:
        raise ValueError(f"{label} current path/hash mismatch")

def validate_state_surface_pins(state_surfaces: dict[str,Any], paths: dict[str,Path], digests: dict[str,str]) -> None:
    if set(state_surfaces)!=set(paths): raise ValueError("verified state source-role set differs from current O/I/D")
    for role,path in paths.items():
        entry=state_surfaces.get(role,{})
        if not isinstance(entry,dict) or Path(str(entry.get("path",""))).resolve()!=path.resolve() or entry.get("sha256")!=digests[role]:
            raise ValueError(f"verified state {role} source path/hash mismatch")

def active_domain(points: np.ndarray, faces: np.ndarray, joint_rows: np.ndarray,
                  declared_active: np.ndarray, declared_inactive: np.ndarray,
                  role: str) -> tuple[np.ndarray,np.ndarray]:
    points=np.asarray(points); faces=np.asarray(faces,dtype=np.int64); joint_rows=np.asarray(joint_rows,dtype=np.int64)
    n=len(points)
    if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError(f"{role} point/face shapes invalid")
    if len(faces)==0 or np.any(faces<0) or np.any(faces>=n):
        raise ValueError(f"{role} face rows outside full source point domain")
    used=np.unique(faces).astype(np.int64)
    inactive=np.setdiff1d(np.arange(n,dtype=np.int64),used,assume_unique=True)
    declared_active=np.asarray(declared_active,dtype=np.int64)
    declared_inactive=np.asarray(declared_inactive,dtype=np.int64)
    if joint_rows.shape!=(n,): raise ValueError(f"{role} full-domain crosswalk length mismatch")
    if not np.array_equal(used,declared_active): raise ValueError(f"{role} declared active rows differ from face incidence")
    if not np.array_equal(inactive,declared_inactive): raise ValueError(f"{role} declared inactive rows differ from face incidence")
    if np.any(joint_rows[used]<0) or np.any(joint_rows[inactive]!=-1):
        raise ValueError(f"{role} crosswalk -1 rows are not exactly face-inactive")
    mapped=joint_rows[used]
    if len(np.unique(mapped))!=len(mapped): raise ValueError(f"{role} active source rows do not map injectively")
    return used,inactive

def _source_proof(paths: dict[str,Path], proof_path: Path, shell_path: Path,
                  binding_path: Path, handoff_receipt_path: Path,
                  crosswalk_path: Path) -> dict[str,Any]:
    proof_path=Path(proof_path).resolve(); shell_path=Path(shell_path).resolve()
    binding_path=Path(binding_path).resolve(); handoff_receipt_path=Path(handoff_receipt_path).resolve()
    crosswalk_path=Path(crosswalk_path).resolve()
    proof=json.loads(proof_path.read_text())
    if proof.get("status")!="PASS_CURRENT_EXACT_SOURCE_EVIDENCE" or proof.get("physical_admission") is not False or proof.get("native_steps")!=0:
        raise ValueError("808 source proof is not a source-only PASS")
    if proof.get("mode")!="AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_806":
        raise ValueError("808 source proof mode is not the expected exact-source mode")
    all185=proof.get("all185",{})
    if all185.get("schema")!="existing-bone-record-remesh-barrier.v1" or all185.get("record_count")!=938272 or all185.get("changed_self_count")!=0 or all185.get("changed_vs_all185_count")!=0:
        raise ValueError("808 all185 proof is incomplete or has crossings")
    state_rec=proof.get("state",{})
    state_path=pin(state_rec.get("path",""),str(state_rec.get("sha256","")),"808 state")
    state=json.loads(state_path.read_text())
    if state.get("physical_admission") is not False or state.get("native_steps")!=0:
        raise ValueError("current quality state must remain nonphysical and native_steps=0")
    validate_state_surface_pins(state.get("source_surfaces",{}),paths,{r:sha(p) for r,p in paths.items()})
    bind_state=proof.get("state",{})
    for role,path in paths.items():
        entry=proof.get("source_surfaces",{}).get(role,{})
        record_matches(entry,path,f"808 {role} source surface")
    report=proof.get("report",{}); config=proof.get("config",{})
    report_path=pin(report.get("path",""),str(report.get("sha256","")),"808 exact report")
    config_path=pin(config.get("path",""),str(config.get("sha256","")),"808 context config")
    report_data=json.loads(report_path.read_text())
    config_data=json.loads(config_path.read_text())
    if report_data.get("status")!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185" or report_data.get("physical_admission") is not False or report_data.get("native_steps")!=0:
        raise ValueError("808 exact report did not pass the current O/I/D and all185 gate")
    if not isinstance(config_data.get("candidate_surfaces"),dict):
        raise ValueError("808 config lacks current candidate surface pins")
    for role,path in paths.items():
        rec=config_data["candidate_surfaces"].get(role)
        if rec is None: raise ValueError(f"808 config lacks {role}")
        if isinstance(rec,dict) and "path" in rec:
            record_matches(rec,path,f"808 config {role}")
        elif isinstance(rec,dict) and "candidate" in rec:
            record_matches(rec["candidate"],path,f"808 config {role}")
        else:
            raise ValueError(f"808 config {role} schema is unsupported")
    shell=json.loads(shell_path.read_text())
    if shell.get("schema")!="numi.h706.current-paired-shell-1022.v1" or shell.get("status")!="PASS_CURRENT_938V2_PAIRED_SHELLS_SOURCE_ONLY" or shell.get("physical_admission") is not False or shell.get("native_steps")!=0:
        raise ValueError("1022 shell receipt is not a source-only PASS")
    _same_path_pin(shell.get("current_state"),state_path,str(state_rec["sha256"]),"1022 current state")
    for role,path in paths.items():
        record_matches(shell.get("inputs",{}).get(f"source_surface_{role}"),path,f"1022 {role} input")
    outputs=shell.get("outputs",{})
    for name in ("skin-shell.npz","fat-shell.npz"):
        rec=outputs.get(name)
        if not isinstance(rec,dict): raise ValueError(f"1022 missing {name}")
        pin(rec["path"],rec["sha256"],f"1022 {name}")
    binding=json.loads(binding_path.read_text())
    if binding.get("status")!="PASS_938V2_CURRENT_SOURCE_SHELL_RECEIPT_BINDING" or binding.get("physical_admission") is not False or binding.get("native_steps")!=0:
        raise ValueError("1022 binding is not a source-only PASS")
    _same_path_pin(binding.get("state"),state_path,str(state_rec["sha256"]),"1022 binding state")
    _same_path_pin(binding.get("receipt"),shell_path,sha(shell_path),"1022 bound receipt")
    for role,path in paths.items():
        record_matches(binding.get("source_surfaces",{}).get(role),path,f"1022 binding {role}")
    handoff=json.loads(handoff_receipt_path.read_text())
    if handoff.get("schema")!="numi.h706.current-fine-217-combined-volume-handoff.v1" or handoff.get("physical_admission") is not False or handoff.get("native_steps")!=0:
        raise ValueError("938 active-row handoff is not source-only")
    if handoff.get("status")!="PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION":
        raise ValueError("938 handoff status is not a prepared source-only state")
    record_matches(handoff.get("inputs",{}).get("current_state"),state_path,"938 current state")
    _same_path_pin(handoff.get("inputs",{}).get("shell_assembly_receipt"),shell_path,sha(shell_path),"938 shell assembly")
    record_matches(handoff.get("inputs",{}).get("inner_source_surface"),paths["I"],"938 current I surface")
    map_entry=handoff.get("outputs",{}).get(crosswalk_path.name)
    if not isinstance(map_entry,dict) or map_entry.get("sha256")!=sha(crosswalk_path):
        raise ValueError("938 handoff does not pin the supplied full-domain surface map")
    with np.load(crosswalk_path,allow_pickle=False) as z:
        domains={}
        for role,path in paths.items():
            arr=np.load(path,allow_pickle=False)
            points=np.asarray(arr["points_world_m"]); faces=np.asarray(arr["faces"],dtype=np.int64)
            full32=np.asarray(arr["points_world_m_f32"]) if "points_world_m_f32" in arr.files else points.astype(np.float32)
            if full32.dtype!=np.float32 or full32.shape!=points.shape or not np.array_equal(full32,points.astype(np.float32)):
                raise ValueError(f"{role} full and native F32 point lanes disagree")
            mk=ROLE_MAP_KEY[role]; ak=ROLE_ACTIVE_KEY[role]; ik=ROLE_INACTIVE_KEY[role]
            if mk not in z.files or ak not in z.files or ik not in z.files: raise ValueError(f"938 map misses {role} full-domain rows")
            active,inactive=active_domain(full32,faces,z[mk],z[ak],z[ik],role)
            domains[role]={"active":active,"inactive":inactive,"joint":np.asarray(z[mk],dtype=np.int64)}
            arr.close()
    return {"schema":"numi.h706.current-230-source-proof.v1","sourceProof":{"path":str(proof_path),"sha256":sha(proof_path)},
        "state":{"path":str(state_path),"sha256":sha(state_path)},
        "report":{"path":str(report_path),"sha256":sha(report_path)},
        "config":{"path":str(config_path),"sha256":sha(config_path)},
        "shellReceipt":{"path":str(shell_path),"sha256":sha(shell_path)},
        "shellBinding":{"path":str(binding_path),"sha256":sha(binding_path)},
        "handoffReceipt":{"path":str(handoff_receipt_path),"sha256":sha(handoff_receipt_path)},
        "crosswalk":{"path":str(crosswalk_path),"sha256":sha(crosswalk_path)},
        "all185":all185,"source_row_domains":{r:{"full_rows":len(d["joint"]),"active_rows":len(d["active"]),
             "inactive_rows":len(d["inactive"]),"active_ids_sha256":hashlib.sha256(d["active"].astype("<i8").tobytes()).hexdigest(),
             "inactive_ids_sha256":hashlib.sha256(d["inactive"].astype("<i8").tobytes()).hexdigest()} for r,d in domains.items()},
        "_domains":domains,"_paths":paths}

def _validate_current_candidate(path: Path, layer: str, evidence: dict[str,Any]) -> dict[str,Any]:
    path=Path(path).resolve()
    if path.resolve()!=evidence["_paths"][layer].resolve(): raise ValueError(f"{layer} candidate path differs from current 808 pin")
    z=np.load(path,allow_pickle=False)
    req=("points_world_m","points_world_m_f32","faces",*FACE_KEYS)
    missing=[k for k in req if k not in z.files]
    if missing: raise ValueError(f"{layer} current typed candidate misses {missing}")
    x=np.asarray(z["points_world_m"]);x32=np.asarray(z["points_world_m_f32"]);f=np.asarray(z["faces"],dtype=np.int64)
    if x.ndim!=2 or x.shape[1]!=3 or x.dtype.kind!="f" or not np.isfinite(x).all():
        raise ValueError(f"{layer} full point table invalid")
    if x32.dtype!=np.float32 or x32.shape!=x.shape or not np.isfinite(x32).all() or not np.array_equal(x32,x.astype(np.float32)):
        raise ValueError(f"{layer} native Float32 point table is not exact cast")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)==0 or np.any(f<0) or np.any(f>=len(x32)):
        raise ValueError(f"{layer} face table invalid")
    if np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,2]==f[:,0])):
        raise ValueError(f"{layer} has repeated-vertex faces")
    arrays={k:np.asarray(z[k]) for k in FACE_KEYS}
    if any(a.shape!=(len(f),) for a in arrays.values()):
        raise ValueError(f"{layer} face typed-identity row domains differ")
    for k,a in arrays.items():
        if a.dtype.kind not in "iu": raise ValueError(f"{layer} field {k} is not integer typed")
    active=evidence["_domains"][layer]["active"];inactive=evidence["_domains"][layer]["inactive"]
    if not np.array_equal(active,np.unique(f)): raise ValueError(f"{layer} active source rows differ from actual faces")
    offsets=np.asarray(z["face_ancestry_offsets"],dtype=np.int64) if "face_ancestry_offsets" in z.files else None
    ancestry={}
    if offsets is not None:
        if offsets.shape!=(len(f)+1,) or offsets[0]!=0 or np.any(np.diff(offsets)<0):
            raise ValueError(f"{layer} face ancestry offsets are invalid")
        if any(k not in z.files for k in ANCESTRY_KEYS): raise ValueError(f"{layer} typed face ancestry lanes incomplete")
        vals={k:np.asarray(z[k],dtype=np.int64) for k in ANCESTRY_KEYS}
        if offsets[-1]<0 or any(v.shape!=(int(offsets[-1]),) for v in vals.values()):
            raise ValueError(f"{layer} ancestry values do not match CSR offsets")
        ancestry={"offset_rows":int(len(offsets)),"entries":int(offsets[-1]),
          "rows_sha256":{k:hashlib.sha256(v.astype("<i8",copy=False).tobytes()).hexdigest() for k,v in vals.items()},
          "offsets_sha256":hashlib.sha256(offsets.astype("<i8",copy=False).tobytes()).hexdigest()}
    ids=sorted(map(int,np.unique(arrays["face_source_original_id"])))
    z.close()
    return {"path":str(path),"sha256":sha(path),"layer":layer,
        "role":mod_role(layer),"roleIndex":mod_role_index(layer),"domain":"derived_registered_skin_authoring",
        "vertexCount":len(x32),"faceCount":len(f),"dtype":"float32","points":x32.astype(np.float64),
        "faces":f,"sourceSchema":{"validation":"pinned current 808 exact source proof + 1022/938 current typed loader",
        "current_candidate_path":str(path),"current_candidate_sha256":sha(path)},
        "faceAncestry":{"validation":"preserved directly from current typed NPZ; validated by pinned 938 v3 handoff",
        "csr":ancestry},"faceArrays":arrays,"sourceOriginalFaceIDs":ids,
        "sourceRowsAllPointRowsDirect":True,"sourceParentLineagePreservedPerFace":True,
        "activeSourceRows":active.copy(),"inactiveSourceRows":inactive.copy(),"sourcePointF32":x32.copy(),
        "crosswalkSHA256":evidence["crosswalk"]["sha256"]}

def mod_role(layer: str) -> str:
    return {"O":"outerSurface","I":"innerSurface","D":"deepSurface"}[layer]
def mod_role_index(layer: str) -> int:
    return {"O":0,"I":1,"D":2}[layer]

def _expand_result(surface: dict[str,Any], active_result: dict[str,Any]) -> dict[str,Any]:
    active=np.asarray(surface["activeSourceRows"],dtype=np.int64); inactive=np.asarray(surface["inactiveSourceRows"],dtype=np.int64)
    full_points=np.asarray(surface["sourcePointF32"]); n=len(full_points)
    lookup=np.full(n,-1,dtype=np.int64); lookup[active]=np.arange(len(active),dtype=np.int64)
    nodes=[None]*n
    for raw in active_result["nodes"]:
        r=dict(raw); local=int(r["vertex"]); source=int(active[local])
        r["vertex"]=source;r["source_vertex_row"]=source
        nodes[source]=r
    for source in inactive:
        source=int(source);p=full_points[source]
        r={k:"" for k in mod.NODE_HEADER}
        r.update({"vertex":source,"wx":mod.f64(p[0]),"wy":mod.f64(p[1]),"wz":mod.f64(p[2]),
            "source_domain":mod.DOMAIN,"source_mapping_kind":"vertex",
            "source_registration_method":"candidate_surface_npz_direct_vertex_identity",
            "source_vertex_row":source,"source_face_row":-1,
            "source_bary0":"0","source_bary1":"0","source_bary2":"0",
            "source_residual_x_m":"0","source_residual_y_m":"0","source_residual_z_m":"0"})
        nodes[source]=r
    if any(x is None for x in nodes): raise ValueError("expanded node sidecar does not cover full source row domain")
    faces=[]
    for raw in active_result["faces"]:
        r=dict(raw)
        for key in ("v0","v1","v2"):
            local=int(r[key])
            if local<0 or local>=len(active): raise ValueError("active face node index outside compact registration")
            r[key]=int(active[local])
        faces.append(r)
    regs=[]
    for raw in active_result["registrations"]:
        r=dict(raw);local=int(r["surface_vertex_row"]);source=int(active[local])
        r["surface_vertex_row"]=source;r["source_vertex_row"]=source;regs.append(r)
    weights=[]
    for raw in active_result["weightRows"]:
        r=dict(raw);r["vertex"]=int(active[int(r["vertex"])]);weights.append(r)
    owners=[]
    for raw in active_result["weightOwnerRows"]:
        r=dict(raw);r["vertex"]=int(active[int(r["vertex"])]);owners.append(r)
    result=dict(active_result)
    result.update({"nodes":nodes,"faces":faces,"registrations":regs,"weightRows":weights,"weightOwnerRows":owners})
    result["sourceDomain"]=dict(active_result["sourceDomain"])
    result["sourceDomain"].update({"vertexCount":n,"activeVertexCount":int(len(active)),
       "inactiveVertexCount":int(len(inactive)),"pointRowNamespace":"original full current typed candidate row",
       "inactiveRowsPolicy":"provenance-only; no normal, parent tet, donor frame, or physical registration"})
    result["metrics"]=dict(active_result["metrics"])
    result["metrics"].update({"nodes":int(n),"activeRegisteredNodes":int(len(active)),
        "inactiveProvenanceOnlyNodes":int(len(inactive)),
        "sourceVertexRowsAreOriginalFullCurrentRows":True,
        "inactiveRowsHaveNoParentOrFrameRegistration":True})
    return result

def run_self_tests() -> dict[str,Any]:
    # Current state itself must bind the same exact source surfaces as the 808 proof.
    demo_paths={r:Path(f"/fixture/{r}.npz") for r in ("O","I","D")}
    demo_digests={r:f"sha-{r}" for r in demo_paths}
    demo_table={r:{"path":str(demo_paths[r]),"sha256":demo_digests[r]} for r in demo_paths}
    validate_state_surface_pins(demo_table,demo_paths,demo_digests)
    try:
        bad={**demo_table,"I":{"path":str(demo_paths["D"]),"sha256":demo_digests["I"]}}
        validate_state_surface_pins(bad,demo_paths,demo_digests)
    except ValueError: pass
    else: raise AssertionError("mismatched verified-state source surface was accepted")
    # Regression for the known full-ID gap: row 553 is active, row 552 is inactive.
    p=np.zeros((554,3),np.float32); faces=np.asarray([[0,1,553]],np.int64)
    joint=np.full(554,-1,np.int64);joint[[0,1,553]]=[0,1,552]
    active,inactive=active_domain(p,faces,joint,np.asarray([0,1,553]),np.asarray([*range(2,553)]), "O")
    assert active.tolist()==[0,1,553] and 552 in inactive and joint[553]==552
    for bad_joint,bad_active,bad_inactive in [
        (np.where(np.arange(554)==553,-1,joint),np.asarray([0,1,553]),np.asarray([*range(2,553)])),
        (np.where(np.arange(554)==552,999,joint),np.asarray([0,1,553]),np.asarray([*range(2,553)])),
        (np.where(np.arange(554)==553,1,joint),np.asarray([0,1,553]),np.asarray([*range(2,553)])),
    ]:
        try: active_domain(p,faces,bad_joint,bad_active,bad_inactive,"O")
        except ValueError: pass
        else: raise AssertionError("bad active/inactive crosswalk was accepted")
    # Sidecar expansion retains the full row domain and registers only active rows.
    surface={"activeSourceRows":np.asarray([0,2,4]),"inactiveSourceRows":np.asarray([1,3]),
        "sourcePointF32":np.arange(15,dtype=np.float32).reshape(5,3)}
    fake={"nodes":[{"vertex":0,"source_vertex_row":0},{"vertex":1,"source_vertex_row":1},{"vertex":2,"source_vertex_row":2}],
        "faces":[{"face":0,"v0":0,"v1":1,"v2":2}],
        "registrations":[{"surface_vertex_row":0,"source_vertex_row":0},{"surface_vertex_row":1,"source_vertex_row":1},{"surface_vertex_row":2,"source_vertex_row":2}],
        "weightRows":[{"vertex":0},{"vertex":1},{"vertex":2}],"weightOwnerRows":[{"vertex":0},{"vertex":1},{"vertex":2}],
        "sourceDomain":{"vertexCount":3},"metrics":{"nodes":3}}
    global mod
    class Dummy:
        NODE_HEADER=["vertex","wx","wy","wz","nx","source_domain","source_mapping_kind","source_vertex_row","source_face_row"]
        DOMAIN="derived_registered_skin_authoring"
        @staticmethod
        def f64(x): return format(float(x),".17g")
    old=mod;mod=Dummy
    try: expanded=_expand_result(surface,fake)
    finally: mod=old
    assert len(expanded["nodes"])==5 and [r["vertex"] for r in expanded["nodes"]]==list(range(5))
    assert expanded["nodes"][2]["source_vertex_row"]==2 and expanded["nodes"][1]["source_mapping_kind"]=="vertex" and expanded["nodes"][1].get("parent_tet_row","")==""
    assert expanded["registrations"][1]["source_vertex_row"]==2 and len(expanded["registrations"])==3
    assert expanded["sourceDomain"]["inactiveVertexCount"]==2
    return {"status":"PASS_CURRENT_230_ACTIVE_FULL_SOURCE_ROW_CONTROLS","controls":5,
        "covered":"full-ID gap 552/553, exact active/inactive partition, active injectivity, inactive sentinel, full sidecar row expansion, active-only registrations",
        "physical_admission":False,"native_steps":0}

def load_owner() -> Any:
    if not OWNER_PATH.is_file() or sha(OWNER_PATH)!=OWNER_SHA: raise ValueError("pinned 230 owner source hash mismatch")
    sys.path.insert(0,str(OWNER_PATH.parent))
    spec=importlib.util.spec_from_file_location("pinned_current_230_owner_088",OWNER_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import pinned 230 owner")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def main() -> None:
    global mod
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--preflight-only",action="store_true",help="authenticate 808/1022/938 pins and active row domains without parent registration")
    ap.add_argument("--outer",type=Path);ap.add_argument("--inner",type=Path);ap.add_argument("--deep",type=Path)
    ap.add_argument("--parent",type=Path);ap.add_argument("--poses",type=Path);ap.add_argument("--parent-full86-weights",type=Path)
    ap.add_argument("--current-state",type=Path,required=False);ap.add_argument("--source-proof",type=Path)
    ap.add_argument("--shell-receipt",type=Path);ap.add_argument("--shell-binding",type=Path)
    ap.add_argument("--handoff-receipt",type=Path);ap.add_argument("--current-surface-node-map",type=Path)
    ap.add_argument("--out",type=Path);ap.add_argument("--maximum-parent-distance-m",type=float,default=0.02)
    ap.add_argument("--tie-tolerance-m",type=float,default=1e-8)
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(),indent=2,sort_keys=True));return
    required={"outer":args.outer,"inner":args.inner,"deep":args.deep,"parent":args.parent,"poses":args.poses,
       "parent_full86_weights":args.parent_full86_weights,"current_state":args.current_state,"source_proof":args.source_proof,
       "shell_receipt":args.shell_receipt,"shell_binding":args.shell_binding,"handoff_receipt":args.handoff_receipt,
       "current_surface_node_map":args.current_surface_node_map,"out":args.out}
    missing=[k for k,v in required.items() if v is None]
    if missing: ap.error("missing required paths: "+", ".join(missing))
    paths={"O":args.outer.resolve(),"I":args.inner.resolve(),"D":args.deep.resolve()}
    for p in paths.values():
        if not p.is_file():raise ValueError(f"current candidate is missing: {p}")
    if not math_finite(args.maximum_parent_distance_m,args.tie_tolerance_m):raise ValueError("invalid parent distance or tie tolerance")
    evidence=_source_proof(paths,args.source_proof,args.shell_receipt,args.shell_binding,args.handoff_receipt,args.current_surface_node_map)
    if Path(args.current_state).resolve()!=Path(evidence["state"]["path"]).resolve() or sha(args.current_state)!=evidence["state"]["sha256"]:
        raise ValueError("explicit current state differs from authenticated 808/1022/938 state")
    if args.preflight_only:
        summaries={r:{"path":str(paths[r]),"sha256":sha(paths[r]),"full_rows":int(v["full_rows"]),
          "active_rows":int(v["active_rows"]),"inactive_rows":int(v["inactive_rows"]),
          "active_rows_sha256":v["active_ids_sha256"],"inactive_rows_sha256":v["inactive_ids_sha256"]}
          for r,v in evidence["source_row_domains"].items()}
        print(json.dumps({"schema":SCHEMA,"status":"PASS_CURRENT_808_1022_938_ACTIVE_ROW_PREFLIGHT",
          "source_row_domains":summaries,"source_proof":evidence["sourceProof"],"state":evidence["state"],
          "shell_receipt":evidence["shellReceipt"],"shell_binding":evidence["shellBinding"],
          "handoff_receipt":evidence["handoffReceipt"],"crosswalk":evidence["crosswalk"],
          "physical_admission":False,"native_steps":0},indent=2,sort_keys=True))
        return
    mod=load_owner()
    # Provide full current source-domain proof to the unchanged owner export.
    def load_current_typed_shell_proof(_path: Path, selected: dict[str,Path])->dict[str,Any]:
        if {k:Path(v).resolve() for k,v in selected.items()}!={k:v.resolve() for k,v in paths.items()}:
            raise ValueError("230 selected role paths differ from authenticated current O/I/D")
        return {k:v for k,v in evidence.items() if not k.startswith("_")}
    def validate_candidate(path: Path, layer: str, require_f32: bool=True, authorization: Any=None, typed_proof: Any=None)->dict[str,Any]:
        if typed_proof is None:raise ValueError("230 current source proof is required")
        return _validate_current_candidate(path,layer,evidence)
    original_register=mod.register_layer
    def register_current_full_rows(surface:dict[str,Any],parent:dict[str,Any],index:dict[str,Any],
            material_index:dict[int,dict[str,Any]],frames:dict[int,dict[str,np.ndarray]],
            parent_weights86:np.ndarray,binding_bodies:list[int],maximum_distance:float,tie_tol:float)->dict[str,Any]:
        active=np.asarray(surface["activeSourceRows"],dtype=np.int64)
        full_points=np.asarray(surface["points"],dtype=np.float64)
        full_faces=np.asarray(surface["faces"],dtype=np.int64)
        lookup=np.full(len(full_points),-1,dtype=np.int64);lookup[active]=np.arange(len(active),dtype=np.int64)
        active_faces=lookup[full_faces]
        if np.any(active_faces<0):raise ValueError(f"{surface['layer']} face references inactive source row")
        sub=dict(surface);sub["points"]=full_points[active];sub["faces"]=active_faces
        active_result=original_register(sub,parent,index,material_index,frames,parent_weights86,binding_bodies,maximum_distance,tie_tol)
        return _expand_result(surface,active_result)
    mod.load_current_typed_shell_proof=load_current_typed_shell_proof
    mod.validate_candidate=validate_candidate
    mod.register_layer=register_current_full_rows
    args.current_typed_shell_receipt=args.source_proof
    args._current_d_authorization=None
    report=mod.export(args)
    print(json.dumps({"schema":SCHEMA,"sourceOnlySidecarExport":True,"currentProof":{k:v for k,v in evidence.items() if not k.startswith("_")},
        "receiptPath":str(args.out/"receipt.json"),"receiptSHA256":sha(args.out/"receipt.json"),
        "physical_admission":False,"geometry_admission":False,"native_steps":0,
        "selectedRoles":["O","I","D"]},indent=2,sort_keys=True))

def math_finite(distance: float, tie: float) -> bool:
    import math
    return math.isfinite(distance) and distance>0 and math.isfinite(tie) and tie>=0

if __name__=="__main__": main()

