#!/usr/bin/env python3
"""Joint exact oriented face-diff audit for current O/I/D typed candidates.

Uses the existing 197 exact owner, 030 retirement filter, paired 90-face
walls, and all185 barrier loader. Geometry-only evidence; no admission.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
H = HERE.parent
BASE_DRIVER = H / "fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py"
BASE_SCHEMA = "h706.combined-current-exact-surface-diff-config.v1"
ROLE_ORDER = ("D", "O", "I")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}

def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def points_f32(a, label):
    if "points_world_m_f32" in a:
        p = np.asarray(a["points_world_m_f32"])
        if p.dtype != np.float32:
            raise ValueError(f"{label} points_world_m_f32 has wrong dtype")
    elif "points_world_m" in a:
        p = np.asarray(a["points_world_m"], dtype=np.float32)
    else:
        raise ValueError(f"{label} has no point coordinates")
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(f"{label} point coordinates malformed")
    return p

def point_identity_map(base_points, candidate_points, role):
    """Map candidate point rows to baseline rows by exact F32 coordinate bytes."""
    bp = np.asarray(base_points, dtype="<f4")
    cp = np.asarray(candidate_points, dtype="<f4")
    if bp.shape != cp.shape:
        raise ValueError(f"{role}: point count/shape changed")
    def keys(p):
        return [np.ascontiguousarray(row).tobytes() for row in p]
    bkeys, ckeys = keys(bp), keys(cp)
    bmap = {}
    for i, key in enumerate(bkeys):
        if key in bmap:
            raise ValueError(f"{role}: baseline has duplicate F32 points; row identity is ambiguous")
        bmap[key] = i
    if len(set(ckeys)) != len(ckeys):
        raise ValueError(f"{role}: candidate has duplicate F32 points; row identity is ambiguous")
    try:
        mapping = np.asarray([bmap[key] for key in ckeys], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{role}: candidate has a moved or new F32 point") from exc
    if len(np.unique(mapping)) != len(bp):
        raise ValueError(f"{role}: candidate point set is not a full one-to-one baseline match")
    return mapping

def point_subset_identity_map(base_points, candidate_points, role, expected_retired_rows):
    """Map a candidate to a baseline by exact F32 bytes with an explicit retired-row set.

    This is reserved for the authenticated D paired closed-component retirement
    path. Ordinary roles and ordinary D candidates continue to require a full
    bijection through point_identity_map.
    """
    bp = np.asarray(base_points, dtype="<f4")
    cp = np.asarray(candidate_points, dtype="<f4")
    if bp.ndim != 2 or bp.shape[1:] != (3,) or cp.ndim != 2 or cp.shape[1:] != (3,):
        raise ValueError(f"{role}: point arrays must be Nx3")
    if len(cp) > len(bp):
        raise ValueError(f"{role}: retirement mapping cannot add points")
    def keys(p):
        return [np.ascontiguousarray(row).tobytes() for row in p]
    bkeys, ckeys = keys(bp), keys(cp)
    if len(set(bkeys)) != len(bkeys):
        raise ValueError(f"{role}: baseline has duplicate F32 points; row identity is ambiguous")
    if len(set(ckeys)) != len(ckeys):
        raise ValueError(f"{role}: candidate has duplicate F32 points; row identity is ambiguous")
    bmap = {key: i for i, key in enumerate(bkeys)}
    try:
        mapping = np.asarray([bmap[key] for key in ckeys], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{role}: candidate has a moved or unauthorized new F32 point") from exc
    if len(np.unique(mapping)) != len(cp):
        raise ValueError(f"{role}: candidate-to-baseline point map is not injective")
    retired = list(map(int, expected_retired_rows))
    expected = np.asarray(sorted(set(retired)), dtype=np.int64)
    if len(expected) != len(retired):
        raise ValueError(f"{role}: retirement authorization has duplicate point rows")
    if np.any(expected < 0) or np.any(expected >= len(bp)):
        raise ValueError(f"{role}: authorized retired point row is out of range")
    missing = np.setdiff1d(np.arange(len(bp), dtype=np.int64), mapping, assume_unique=True)
    if not np.array_equal(missing, expected):
        raise ValueError(f"{role}: exact missing F32 rows differ from authenticated retirement set")
    return mapping


def array_payload_equal(a, b):
    x, y = np.asarray(a), np.asarray(b)
    return (x.shape == y.shape and x.dtype == y.dtype and
            np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes())

def oriented_face_key(tri):
    """Canonicalize cyclic rotations while keeping reversed winding distinct."""
    a, b, c = map(int, tri)
    return min((a, b, c), (b, c, a), (c, a, b))

def face_delta(base_faces, candidate_faces_mapped, role):
    bf = np.asarray(base_faces, dtype=np.int64)
    cf = np.asarray(candidate_faces_mapped, dtype=np.int64)
    if bf.ndim != 2 or cf.ndim != 2 or bf.shape[1:] != (3,) or cf.shape[1:] != (3,):
        raise ValueError(f"{role}: faces must be Nx3")
    if np.any(bf < 0) or np.any(cf < 0):
        raise ValueError(f"{role}: negative face point row")
    if np.any(bf[:,0] == bf[:,1]) or np.any(bf[:,1] == bf[:,2]) or np.any(bf[:,2] == bf[:,0]):
        raise ValueError(f"{role}: degenerate baseline face")
    if np.any(cf[:,0] == cf[:,1]) or np.any(cf[:,1] == cf[:,2]) or np.any(cf[:,2] == cf[:,0]):
        raise ValueError(f"{role}: degenerate candidate face")
    bkeys = [oriented_face_key(row) for row in bf]
    ckeys = [oriented_face_key(row) for row in cf]
    if len(set(bkeys)) != len(bkeys):
        raise ValueError(f"{role}: duplicate oriented baseline face")
    if len({tuple(sorted(x)) for x in bkeys}) != len(bkeys):
        raise ValueError(f"{role}: duplicate unoriented baseline triangle")
    if len(set(ckeys)) != len(ckeys):
        raise ValueError(f"{role}: duplicate oriented candidate face")
    if len({tuple(sorted(x)) for x in ckeys}) != len(ckeys):
        raise ValueError(f"{role}: duplicate unoriented candidate triangle")
    old = set(bkeys)
    new = set(ckeys)
    changed_rows = np.asarray([i for i, key in enumerate(ckeys) if key not in old], dtype=np.int64)
    removed_rows = np.asarray([i for i, key in enumerate(bkeys) if key not in new], dtype=np.int64)
    return changed_rows, removed_rows

def directed_boundary_edges(faces):
    incidence = {}
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, tri)
        for u, v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u < v else (v,u)
            incidence.setdefault(key, []).append((u,v))
    boundary = set()
    for key, directed in incidence.items():
        if len(directed) == 1:
            boundary.add(directed[0])
        elif len(directed) == 2:
            if directed[0] != directed[1][::-1]:
                raise ValueError(f"same-direction interior edge {key}")
        else:
            raise ValueError(f"nonmanifold edge {key} incidence={len(directed)}")
    return boundary

def validate_closed_component_retirement(parent_faces, retired_face_rows, retired_point_rows, role="D"):
    """Require an authenticated retirement to be one isolated closed face component."""
    f=np.asarray(parent_faces,np.int64)
    fr=np.asarray(list(map(int,retired_face_rows)),np.int64)
    pr=np.asarray(list(map(int,retired_point_rows)),np.int64)
    if len(fr)!=4 or len(pr)!=4 or len(np.unique(fr))!=4 or len(np.unique(pr))!=4:
        raise ValueError(f"{role}: paired retirement must name exactly four unique faces and points")
    if np.any(fr<0) or np.any(fr>=len(f)) or np.any(pr<0):
        raise ValueError(f"{role}: retired face or point row out of range")
    rf=f[fr]
    if np.any(rf<0) or len(np.unique(rf.reshape(-1)))!=4 or set(map(int,np.unique(rf)))!=set(map(int,pr)):
        raise ValueError(f"{role}: retired tetrahedral surface does not use exactly the four authorized points")
    other=np.ones(len(f),bool);other[fr]=False
    if np.any(np.isin(f[other],pr)):
        raise ValueError(f"{role}: retired point is still used by retained parent faces")
    edge_faces={}
    edge_dirs={}
    face_adj={int(r):set() for r in fr}
    for row,tri in zip(fr,rf):
        a,b,c=map(int,tri)
        if len({a,b,c})!=3: raise ValueError(f"{role}: degenerate retired face")
        for u,v in ((a,b),(b,c),(c,a)):
            key=(min(u,v),max(u,v))
            edge_faces.setdefault(key,[]).append(int(row))
            edge_dirs.setdefault(key,[]).append((u,v))
    for key,owners in edge_faces.items():
        dirs=edge_dirs[key]
        if len(owners)!=2 or dirs[0] != dirs[1][::-1]:
            raise ValueError(f"{role}: retired faces are not an oppositely oriented closed 2-manifold at edge {key}")
        face_adj[owners[0]].add(owners[1]);face_adj[owners[1]].add(owners[0])
    reached=set();stack=[int(fr[0])]
    while stack:
        r=stack.pop()
        if r in reached:continue
        reached.add(r);stack.extend(face_adj[r]-reached)
    if reached!=set(map(int,fr)):
        raise ValueError(f"{role}: retired closed surface is disconnected")
    return {"retired_point_rows":pr.tolist(),"retired_face_rows":fr.tolist(),"closed_edge_count":len(edge_faces),
            "isolated_from_retained_faces":True,"connected_closed_oriented_component":True}

def validate_singleton_d_face_only_transition(typed_receipt, candidate_d, candidate_entry,
                                              paired_candidate_path, paired_receipt_path,
                                              paired_maps_path, paired_maps):
    """Authenticate the final face-only singleton against its raw paired-parent operation."""
    role=typed_receipt.get("candidate_roles",{}).get("D")
    if not isinstance(role,dict):raise ValueError("final typed receipt lacks D candidate record")
    candidate_path=str(Path(candidate_entry["path"]).resolve())
    if role.get("path")!=candidate_path or role.get("sha256")!=candidate_entry["sha256"]:
        raise ValueError("final typed receipt D candidate path/hash mismatch")
    if typed_receipt.get("status")!="PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY":
        raise ValueError("final typed D receipt is not the passing current-row ancestry adapter")
    if role.get("parent")!=str(paired_candidate_path) or role.get("parent_sha256")!=sha(paired_candidate_path):
        raise ValueError("final singleton D receipt does not name the authenticated paired candidate as parent")
    raw_path=Path(role.get("raw_path","")).resolve()
    if not raw_path.is_file() or sha(raw_path)!=role.get("raw_sha256"):
        raise ValueError("final singleton raw proposal missing or SHA mismatch")
    report_path=Path(typed_receipt.get("input_report","")).resolve()
    if not report_path.is_file() or sha(report_path)!=typed_receipt.get("input_report_sha256"):
        raise ValueError("final singleton report missing or SHA mismatch")
    report=json.loads(report_path.read_text())
    if report.get("status")!="PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL":
        raise ValueError("final singleton raw report is not a passing exact fixed-coordinate proposal")
    if report.get("physical_admission") is not False or report.get("native_steps")!=0 or report.get("all_point_arrays_unchanged") is not True:
        raise ValueError("final singleton raw report has wrong admission or point-coordinate state")
    if report.get("joint_complete_surface_walls_all185_pairs")!=0 or report.get("joint_changed_self_pairs")!=0:
        raise ValueError("final singleton raw report lacks zero exact complete-context/self pairs")
    raw_pin=report.get("outputs",{}).get("D",{})
    if raw_pin.get("path")!=str(raw_path) or raw_pin.get("sha256")!=role.get("raw_sha256"):
        raise ValueError("final singleton report does not pin its raw D candidate")
    if raw_pin.get("parent")!=str(paired_candidate_path) or raw_pin.get("parent_sha256")!=sha(paired_candidate_path):
        raise ValueError("final singleton report does not pin its paired D parent")
    paired_pin=report.get("paired_retirement_input",{})
    if (paired_pin.get("receipt_path")!=str(paired_receipt_path) or
        paired_pin.get("receipt_sha256")!=sha(paired_receipt_path) or
        paired_pin.get("maps_path")!=str(paired_maps_path) or
        paired_pin.get("maps_sha256")!=sha(paired_maps_path)):
        raise ValueError("final singleton report does not bind the authenticated paired-retirement receipt/maps")
    docs=typed_receipt.get("operation_docs")
    if not isinstance(docs,list) or not docs:
        raise ValueError("final typed receipt lacks operation document pins")
    for item in docs:
        validate_entry(item,"final singleton operation document")
    with np.load(raw_path,allow_pickle=False) as z:raw={k:np.array(z[k],copy=True) for k in z.files}
    with np.load(paired_candidate_path,allow_pickle=False) as z:pair={k:np.array(z[k],copy=True) for k in z.files}
    for field in ("points_world_m_f32","points_world_m"):
        if field in pair and field in raw and not array_payload_equal(raw[field],pair[field]):
            raise ValueError(f"final singleton changed paired D {field} bytes")
        if field in pair and field in candidate_d and not array_payload_equal(candidate_d[field],pair[field]):
            raise ValueError(f"final typed D changed paired D {field} bytes")
    if not array_payload_equal(raw["faces"],candidate_d["faces"]):
        raise ValueError("final typed D geometry faces differ from raw singleton proposal")
    npoints=len(pair["points_world_m_f32"])
    pmap=np.asarray(raw.get("point_output_to_parent_rows"),np.int64)
    if not np.array_equal(pmap,np.arange(npoints,dtype=np.int64)):
        raise ValueError("singleton point-output map is not exact identity to paired D")
    fmap=np.asarray(raw.get("face_output_to_parent_rows"),np.int64)
    retired=np.asarray(raw.get("retired_parent_face_rows"),np.int64)
    pfaces=np.asarray(pair["faces"],np.int64);rfaces=np.asarray(raw["faces"],np.int64)
    if fmap.shape!=(len(rfaces),) or retired.ndim!=1 or len(np.unique(retired))!=len(retired):
        raise ValueError("singleton face retirement map dimensions or uniqueness invalid")
    if len(retired)!=6 or np.any(retired<0) or np.any(retired>=len(pfaces)):
        raise ValueError("singleton operation must retire exactly its six paired-parent faces")
    retained=fmap>=0;created=fmap==-1
    if np.any(fmap < -1) or len(np.unique(fmap[retained]))!=int(np.sum(retained)):
        raise ValueError("singleton face output map has invalid or duplicate parent rows")
    if not np.array_equal(np.sort(fmap[retained]),np.setdiff1d(np.arange(len(pfaces),dtype=np.int64),retired)):
        raise ValueError("singleton face map does not retain exactly paired parent rows outside its six-face patch")
    if not np.array_equal(rfaces[retained],pfaces[fmap[retained]]):
        raise ValueError("singleton retained faces differ from mapped paired-parent rows")
    new_parent=np.asarray(raw.get("new_faces_parent_point_rows"),np.int64)
    if new_parent.shape!=(int(np.sum(created)),3) or not np.array_equal(rfaces[created],new_parent):
        raise ValueError("singleton generated faces differ from authenticated paired-parent point triples")
    ops=[x for x in report.get("accepted_operations",[]) if x.get("role")=="D"]
    if len(ops)!=1 or sorted(map(int,ops[0].get("retired_parent_face_rows",[])))!=sorted(map(int,retired)):
        raise ValueError("singleton report operation does not match its six retired parent rows")
    if not np.array_equal(np.asarray(ops[0].get("new_output_face_rows"),np.int64),np.flatnonzero(created)):
        raise ValueError("singleton report operation output rows differ from raw face map")
    if sorted(map(int,paired_pin.get("singleton_D0_parent_face_rows",[])))!=sorted(map(int,retired)):
        raise ValueError("singleton report paired-D0 face rows differ from raw retirement map")
    maps=np.load(paired_maps_path,allow_pickle=False)
    try:
        d0_face_to_d282=np.asarray(maps["D_output_face_to_parent282_row"],np.int64)
    finally: maps.close()
    expected_d282=d0_face_to_d282[retired]
    if sorted(map(int,paired_pin.get("singleton_D282_parent_face_rows",[])))!=sorted(map(int,expected_d282)):
        raise ValueError("singleton report D282 face rows differ from composed paired-retirement face map")
    return {"raw_candidate":{"path":str(raw_path),"sha256":role["raw_sha256"]},
            "report":{"path":str(report_path),"sha256":typed_receipt["input_report_sha256"]},
            "parent":{"path":str(paired_candidate_path),"sha256":sha(paired_candidate_path)},
            "point_map":"exact output-to-paired identity; F32 and full-precision coordinate bytes unchanged",
            "retired_paired_face_rows":retired.tolist(),"retired_D282_face_rows":expected_d282.tolist(),
            "generated_output_face_rows":np.flatnonzero(created).tolist(),
            "operation_docs":[{"path":x["path"],"sha256":x["sha256"]} for x in docs],
            "report_context_inputs":report.get("context_inputs",{}),
            "physical_admission":False,"native_steps":0}

def validate_paired_d_retirement(auth, baseline_d, candidate_d, candidate_entry):
    """Authenticate the only supported D point-subset path for the paired K4 retirement."""
    if not isinstance(auth,dict) or auth.get("schema")!="h706.paired-D-closed-component-retirement-auth.v1":
        raise ValueError("D retirement authorization schema missing or unsupported")
    op_path=validate_entry(auth.get("operation"),"D paired retirement operation")
    receipt_path=validate_entry(auth.get("receipt"),"D paired retirement receipt")
    maps_path=validate_entry(auth.get("maps"),"D paired retirement maps")
    parent_path=validate_entry(auth.get("parent_candidate"),"D paired retirement D282 parent")
    pair_path=validate_entry(auth.get("paired_candidate"),"D paired retirement candidate")
    typed_receipt_path=validate_entry(auth.get("final_typed_receipt"),"final D typed receipt")
    op=json.loads(op_path.read_text());receipt=json.loads(receipt_path.read_text());typed_receipt=json.loads(typed_receipt_path.read_text())
    if receipt.get("schema")!="numi.paired-inferred-envelope-resolution-reduction-receipt.v1" or receipt.get("status")!="PASS_SOURCE_ONLY_PAIRED_RETIREMENT_WITH_INHERITED_D_OPEN_EDGE_WITNESS":
        raise ValueError("paired D retirement receipt status/schema is not the authenticated source-only result")
    if receipt.get("physical_admission") is not False or receipt.get("mass_transport")!="NOT_RUN" or receipt.get("native_steps")!=0:
        raise ValueError("paired D retirement receipt has unexpected admission/mass/native state")
    if receipt.get("operation_path")!=str(op_path) or receipt.get("operation_sha256")!=sha(op_path):
        raise ValueError("paired D retirement receipt does not pin its operation JSON")
    d=op.get("D")
    if not isinstance(d,dict):raise ValueError("paired retirement operation has no D record")
    if d.get("input_path")!=str(parent_path) or d.get("input_sha256")!=sha(parent_path):
        raise ValueError("paired D retirement parent pin differs from operation")
    if d.get("output_path")!=str(pair_path) or d.get("output_sha256")!=sha(pair_path):
        raise ValueError("paired D retirement candidate pin differs from operation")
    if d.get("output_maps")!=str(maps_path) or receipt.get("outputs",{}).get("maps_sha256")!=sha(maps_path):
        raise ValueError("paired D retirement map path/hash differs from operation/receipt")
    if receipt.get("outputs",{}).get("D_candidate_sha256")!=sha(pair_path):
        raise ValueError("paired D retirement candidate hash differs from receipt")
    for key in ("paired_components_retired","no_protected149_overlap","inherited_D_open_edges_unchanged","source330_mass_input_verified_and_untouched"):
        if receipt.get("checks",{}).get(key) is not True:
            raise ValueError(f"paired D retirement receipt lacks required check {key}")
    if typed_receipt.get("physical_admission") is not False or typed_receipt.get("native_steps")!=0:
        raise ValueError("final D typed receipt has unexpected physical/native state")
    with np.load(parent_path,allow_pickle=False) as z282: parent282={k:np.array(z282[k],copy=True) for k in z282.files}
    with np.load(pair_path,allow_pickle=False) as zp: pair={k:np.array(zp[k],copy=True) for k in zp.files}
    with np.load(maps_path,allow_pickle=False) as zm: maps={k:np.array(zm[k],copy=True) for k in zm.files}
    singleton_transition=validate_singleton_d_face_only_transition(
        typed_receipt,candidate_d,candidate_entry,pair_path,receipt_path,maps_path,maps)
    if "points_world_m_f32" not in parent282 or "points_world_m_f32" not in pair:
        raise ValueError("paired D point crosswalk requires exact F32 coordinate arrays")
    p282=np.asarray(parent282["points_world_m_f32"],np.float32);ppair=np.asarray(pair["points_world_m_f32"],np.float32)
    retired_points=list(map(int,d.get("retired_point_rows",[])));retired_faces=list(map(int,d.get("retired_face_rows",[])))
    validate_closed_component_retirement(parent282["faces"],retired_faces,retired_points,"D")
    required_map_keys=("D_output_point_to_parent282_row","D_parent282_point_to_output_row","D_output_face_to_parent282_row","D_parent282_face_to_output_row")
    if any(k not in maps for k in required_map_keys):raise ValueError("paired D maps lack explicit parent/output point and face maps")
    out_to_parent=np.asarray(maps[required_map_keys[0]],np.int64);parent_to_out=np.asarray(maps[required_map_keys[1]],np.int64)
    face_out_to_parent=np.asarray(maps[required_map_keys[2]],np.int64);face_parent_to_out=np.asarray(maps[required_map_keys[3]],np.int64)
    exact_out_to_parent=point_subset_identity_map(p282,ppair,"D paired output",retired_points)
    if out_to_parent.shape!=exact_out_to_parent.shape or not np.array_equal(out_to_parent,exact_out_to_parent):
        raise ValueError("paired D explicit output→parent point map differs from exact F32 identity")
    expected_parent_to_out=np.full(len(p282),-1,np.int64);expected_parent_to_out[out_to_parent]=np.arange(len(out_to_parent),dtype=np.int64)
    if parent_to_out.shape!=expected_parent_to_out.shape or not np.array_equal(parent_to_out,expected_parent_to_out):
        raise ValueError("paired D explicit parent→output point map is not the exact inverse with four retired rows")
    parent_faces=np.asarray(parent282["faces"],np.int64);pair_faces=np.asarray(pair["faces"],np.int64)
    expected_face_to_out=np.full(len(parent_faces),-1,np.int64)
    if len(face_out_to_parent)!=len(pair_faces) or len(face_parent_to_out)!=len(parent_faces):
        raise ValueError("paired D face map dimensions disagree with candidate geometry")
    if not np.array_equal(np.sort(np.unique(face_out_to_parent)),np.sort(np.setdiff1d(np.arange(len(parent_faces)),np.asarray(retired_faces,np.int64)))):
        raise ValueError("paired D output→parent face map does not omit exactly four authenticated parent faces")
    if len(np.unique(face_out_to_parent))!=len(face_out_to_parent) or np.any(face_out_to_parent<0) or np.any(face_out_to_parent>=len(parent_faces)):
        raise ValueError("paired D output→parent face map is not injective/in-range")
    expected_face_to_out[face_out_to_parent]=np.arange(len(face_out_to_parent),dtype=np.int64)
    if not np.array_equal(face_parent_to_out,expected_face_to_out):raise ValueError("paired D parent→output face map is not the exact inverse")
    if np.any(pair_faces<0) or np.any(pair_faces>=len(out_to_parent)):
        raise ValueError("paired D candidate face references point outside the authenticated output→parent map")
    if not np.array_equal(out_to_parent[pair_faces],parent_faces[face_out_to_parent]):
        raise ValueError("paired D candidate faces differ from mapped D282 parent rows after exact point-row rebase")
    baseline_points=points_f32(baseline_d,"baseline D")
    parent_to_baseline=point_identity_map(baseline_points,p282,"D D282→c157 baseline")
    # The final singleton repair is required to be face-only. This pins its
    # point domain to the authenticated paired retirement map before composition.
    final_points=points_f32(candidate_d,"final candidate D")
    if not array_payload_equal(final_points,ppair):
        raise ValueError("final D singleton changed the paired-retirement F32 point array")
    if "points_world_m" in pair and "points_world_m" in candidate_d:
        if not array_payload_equal(candidate_d["points_world_m"],pair["points_world_m"]):
            raise ValueError("final D singleton changed paired-retirement full-precision point bytes")
    final_to_baseline=parent_to_baseline[out_to_parent]
    # The four retired faces must map to c157 rows and be absent from final D.
    bfaces=np.asarray(baseline_d["faces"],np.int64);bkey_to_row={}
    for i,t in enumerate(bfaces):
        k=oriented_face_key(t)
        if k in bkey_to_row:raise ValueError("baseline D has duplicate face identity while resolving retirement rows")
        bkey_to_row[k]=i
    retired_baseline_face_rows=[]
    for fr in retired_faces:
        mapped=parent_to_baseline[parent_faces[fr]]
        key=oriented_face_key(mapped)
        if key not in bkey_to_row:raise ValueError("authenticated retired D282 face has no exact c157 baseline counterpart")
        retired_baseline_face_rows.append(bkey_to_row[key])
    typed_faces=np.asarray(candidate_d["faces"],np.int64)
    final_keys={oriented_face_key(final_to_baseline[t]) for t in typed_faces}
    if any(oriented_face_key(bfaces[r]) in final_keys for r in retired_baseline_face_rows):
        raise ValueError("final D still contains an authenticated paired-retirement face")
    required_typed={"face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1","face_ancestry_offsets","quality_star_patch_union_offsets"}
    if not required_typed.issubset(candidate_d):raise ValueError("final D is not a complete current typed candidate")
    if len(candidate_d["face_ancestry_offsets"])!=len(typed_faces)+1 or len(candidate_d["quality_star_patch_union_offsets"])!=len(typed_faces)+1:
        raise ValueError("final D typed ancestry/union CSR row domain is malformed")
    for off_name in ("face_ancestry_offsets","quality_star_patch_union_offsets"):
        off=np.asarray(candidate_d[off_name],np.int64)
        if off[0]!=0 or np.any(np.diff(off)<=0):raise ValueError(f"final D has empty or malformed {off_name} rows")
    return final_to_baseline,{"schema":auth["schema"],"operation":{"path":str(op_path),"sha256":sha(op_path)},
        "receipt":{"path":str(receipt_path),"sha256":sha(receipt_path)},"maps":{"path":str(maps_path),"sha256":sha(maps_path)},
        "D282_parent":{"path":str(parent_path),"sha256":sha(parent_path)},"paired_candidate":{"path":str(pair_path),"sha256":sha(pair_path)},
        "typed_receipt":{"path":str(typed_receipt_path),"sha256":sha(typed_receipt_path)},
        "retired_D282_point_rows":retired_points,"retired_baseline_point_rows":sorted(set(range(len(baseline_points)))-set(map(int,final_to_baseline))),
        "retired_D282_face_rows":retired_faces,"retired_baseline_face_rows":sorted(retired_baseline_face_rows),
        "paired_component_closed_isolated_and_oriented":True,"D282_to_c157_point_map":"exact F32 bijection; identity when row order is preserved",
        "final_singleton_point_domain_matches_paired_candidate":True,"final_singleton_transition":singleton_transition}

def validate_entry(entry, name):
    if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
        raise ValueError(f"{name} entry needs path and sha256")
    p = Path(entry["path"]).resolve()
    if not p.is_file() or sha(p) != entry["sha256"]:
        raise ValueError(f"{name} file missing or SHA mismatch: {p}")
    return p

def require_declared_change(role, changed, removed, config):
    declared = config.get("changed_roles", ["O", "I", "D"])
    if not declared or len(declared) != len(set(declared)) or not set(declared).issubset({"O","I","D"}):
        raise ValueError("changed_roles must declare a nonempty unique role subset")
    if role in declared:
        if len(changed) == 0 and len(removed) == 0:
            raise ValueError(f"{role}: declared changed role has no face-set changes")
    elif len(changed) or len(removed):
        raise ValueError(f"{role}: undeclared role changed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    config_path = Path(a.config).resolve()
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(config_path.read_text())
    if config.get("schema") != BASE_SCHEMA:
        raise ValueError("combined exact surface-diff config schema mismatch")
    if set(config.get("baseline_candidates", {})) != set(("O","I","D")):
        raise ValueError("baseline must pin O, I, and D")
    if set(config.get("candidate_surfaces", {})) != set(("O","I","D")):
        raise ValueError("candidate_surfaces must pin O, I, and D")

    baseline = {}
    candidate = {}
    baseline_paths, candidate_paths = {}, {}
    pointmaps, changed_rows, removed_rows, mapped_faces = {}, {}, {}, {}
    f32_point_receipts = {}
    paired_d_retirement_receipt = None
    paired_d_retired_baseline_faces = set()
    for role in ROLE_ORDER:
        bp = validate_entry(config["baseline_candidates"][role], f"baseline {role}")
        cp = validate_entry(config["candidate_surfaces"][role], f"candidate {role}")
        baseline_paths[role], candidate_paths[role] = bp, cp
        baseline[role], candidate[role] = load_npz(bp), load_npz(cp)
        p0, p1 = points_f32(baseline[role], f"baseline {role}"), points_f32(candidate[role], f"candidate {role}")
        if role == "D" and "paired_D_retirement" in config:
            pointmaps[role], paired_d_retirement_receipt = validate_paired_d_retirement(
                config["paired_D_retirement"], baseline[role], candidate[role],
                config["candidate_surfaces"][role])
            paired_d_retired_baseline_faces = set(map(int, paired_d_retirement_receipt["retired_baseline_face_rows"]))
        else:
            pointmaps[role] = point_identity_map(p0, p1, role)
        # If both assets expose full precision points, require those arrays too
        # to be exactly unchanged after the verified row correspondence.
        full_precision = "points_world_m" in baseline[role] and "points_world_m" in candidate[role]
        if full_precision:
            q0 = np.asarray(baseline[role]["points_world_m"])
            q1 = np.asarray(candidate[role]["points_world_m"])
            if q0.shape != p0.shape or q1.shape != p1.shape or q0.dtype != q1.dtype:
                raise ValueError(f"{role}: full-precision point array schema changed")
            if not array_payload_equal(q1, q0[pointmaps[role]]):
                raise ValueError(f"{role}: full-precision point coordinate bytes changed")
        f32_point_receipts[role] = {
            "baseline_point_count": len(p0),
            "candidate_point_count": len(p1),
            "candidate_to_baseline_rows_bijective": len(pointmaps[role]) == len(p0),
            "candidate_rows_injective": len(np.unique(pointmaps[role])) == len(pointmaps[role]),
            "f32_coordinate_bytes_unchanged": True,
            "full_precision_array_checked": bool(full_precision),
            "retired_baseline_point_count": len(p0) - len(pointmaps[role]),
            "retirement_authorization": paired_d_retirement_receipt if role == "D" else None,
        }
        base_faces = np.asarray(baseline[role]["faces"], dtype=np.int64)
        cand_faces = np.asarray(candidate[role]["faces"], dtype=np.int64)
        if (base_faces.ndim != 2 or cand_faces.ndim != 2 or
            base_faces.shape[1:] != (3,) or cand_faces.shape[1:] != (3,) or
            np.any(base_faces >= len(p0)) or np.any(cand_faces >= len(p1))):
            raise ValueError(f"{role}: malformed face indices")
        mf = pointmaps[role][cand_faces]
        changed_rows[role], removed_rows[role] = face_delta(base_faces, mf, role)
        require_declared_change(role, changed_rows[role], removed_rows[role], config)
        old_boundary = directed_boundary_edges(base_faces)
        new_boundary = directed_boundary_edges(mf)
        if old_boundary != new_boundary:
            raise ValueError(f"{role}: directed boundary changed")
        mapped_faces[role] = mf

    if paired_d_retirement_receipt is not None:
        if not paired_d_retired_baseline_faces.issubset(set(map(int,removed_rows["D"]))):
            raise ValueError("D candidate does not remove every receipt-authorized paired-retirement face from c157 baseline")
        context=paired_d_retirement_receipt.get("final_singleton_transition",{}).get("report_context_inputs",{})
        for role in ("O","I"):
            item=context.get(role,{})
            entry=config["candidate_surfaces"][role]
            if item.get("path")!=str(Path(entry["path"]).resolve()) or item.get("sha256")!=entry["sha256"]:
                raise ValueError(f"final D singleton exact audit context is stale for {role}")

    # Import the already-frozen exact geometry and wall/bone owners.
    base = import_path("surface_diff_owner_197", BASE_DRIVER)
    pred = base.module("quality_owner_loader_198", base.OWNER).load_exact_owner()
    if sha(Path(pred.__file__)) != base.OWNER_IMPL_SHA:
        raise ValueError("exact predicate implementation SHA changed")
    bone_mod = base.module("bone_barrier_198", base.BONE)
    filter_mod = base.module("filtered_exact_owner_198", base.FILTER)

    pointbase, facebase, cursor = {}, base.FACE_BASE.copy(), 0
    combined_parts = []
    for role in ROLE_ORDER:
        p = points_f32(baseline[role], f"baseline {role}")
        pointbase[role] = cursor
        combined_parts.append(p)
        cursor += len(p)
    combined_points = np.concatenate(combined_parts, axis=0)
    baseline_records = []
    for role in ROLE_ORDER:
        p = points_f32(baseline[role], f"baseline {role}")
        f = np.asarray(baseline[role]["faces"], dtype=np.int64)
        raw, _ = pred.float32_surface_records(np.asarray(p, dtype=np.float64), f, source_seam_groups=())
        if len(raw) != len(f):
            raise ValueError(f"{role}: exact owner dropped a baseline face")
        baseline_records.extend((q[0],q[1],q[2],facebase[role]+i,
                                 tuple(pointbase[role]+int(v) for v in q[4]))
                                for i,q in enumerate(raw))

    # Map both source-owned 90-face walls to the authenticated shared point domain.
    point_lookup = {}
    for role in ROLE_ORDER:
        for row, xyz in enumerate(points_f32(baseline[role], f"baseline {role}")):
            key = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
            if key in point_lookup:
                raise ValueError(f"baseline cross-role F32 point alias at {role}/{row}")
            point_lookup[key] = (role, row)
    wall_receipts, wall_records = [], []
    for volume, marker, expected_roles in (("Skin",3,("O","I")),("Fat",6,("I","D"))):
        spath, expected_sha = base.SHELL[volume]
        if sha(spath) != expected_sha:
            raise ValueError(f"{volume} wall source SHA mismatch")
        shell = load_npz(spath)
        sp = np.asarray(shell["points_world_m"], dtype=np.float32)
        sf = np.asarray(shell["faces"], dtype=np.int64)
        fm = np.asarray(shell["face_marker"], dtype=np.int64)
        wall = np.flatnonzero(fm == marker)
        used = np.unique(sf[wall])
        remap = np.full(len(sp), -1, dtype=np.int64)
        for sr in used:
            hit = point_lookup.get(np.ascontiguousarray(sp[int(sr)], dtype="<f4").tobytes())
            if hit is None or hit[0] not in expected_roles:
                raise ValueError(f"{volume} wall vertex lacks expected current role mapping")
            remap[int(sr)] = pointbase[hit[0]] + int(hit[1])
        wf = remap[sf[wall]]
        if np.any(wf < 0):
            raise ValueError(f"{volume} wall mapping incomplete")
        raw, _ = pred.float32_surface_records(np.asarray(combined_points, dtype=np.float64), wf, source_seam_groups=())
        if len(raw) != 90:
            raise ValueError(f"{volume} wall exact face count is not 90")
        wallbase = {"Skin":900000, "Fat":901000}[volume]
        wr = [(q[0],q[1],q[2],wallbase+i,tuple(map(int,q[4]))) for i,q in enumerate(raw)]
        wall_records.extend(wr)
        wall_receipts.append({"role":volume,"path":str(spath),"sha256":expected_sha,
                              "marker":marker,"face_count":len(wr)})
    bones, bone_receipt = bone_mod.load_bone_barrier_records(
        pred, root=base.ROOT, vertex_offset=len(combined_points),
        expected_native_sha256=base.BONE_NATIVE,
        expected_manifest_sha256=base.BONE_MANIFEST)
    if len(bones) != 938272:
        raise ValueError("all185 barrier record census mismatch")
    all_context = baseline_records + wall_records + list(bones)
    face_ids = [int(q[3]) for q in all_context]
    if len(face_ids) != len(set(face_ids)):
        raise ValueError("baseline/wall/all185 face ID collision")
    exact_index = pred._prepare_surface_aabb(all_context)
    allowed = {facebase[r] + i for r in ("O","I","D") for i in range(len(baseline[r]["faces"]))}
    removed_global = {facebase[r] + int(i) for r in ("O","I","D") for i in removed_rows[r]}
    filtered = filter_mod.FilteredPreparedExactOwner(
        pred, allowed_retirable_face_ids=allowed, excluded_face_ids=removed_global)

    changed_records = []
    changed_records_by_role = {}
    next_id = max(face_ids, default=0) + 1_000_000
    for role in ROLE_ORDER:
        rows = changed_rows[role]
        p = points_f32(candidate[role], f"candidate {role}")
        source_faces = np.asarray(candidate[role]["faces"], dtype=np.int64)[rows]
        raw, _ = pred.float32_surface_records(np.asarray(p, dtype=np.float64), source_faces, source_seam_groups=())
        if len(raw) != len(rows):
            raise ValueError(f"{role}: exact owner dropped changed candidate faces")
        role_records = []
        for q in raw:
            local_vertices = tuple(pointmaps[role][int(v)] for v in q[4])
            rec = (q[0],q[1],q[2],next_id,tuple(pointbase[role]+v for v in local_vertices))
            role_records.append(rec)
            next_id += 1
        changed_records_by_role[role] = role_records
        changed_records.extend(role_records)

    against = filtered._audit_pair_prepared_first(exact_index, changed_records, same_surface=True)
    # One shared changed-set AABB index catches new O/I/D interactions too.
    changed_index = pred._prepare_surface_aabb(changed_records)
    seen_pairs = set()
    def changed_pairs():
        for query, indexed in pred._query_surface_aabb(changed_records, changed_index):
            qid, iid = int(query[3]), int(indexed[3])
            if qid == iid:
                continue
            pair = (min(qid,iid), max(qid,iid))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            yield query, indexed
    changed_self = pred._audit_record_pairs(changed_pairs(), same_surface=True)

    role_summary = {}
    for role in ("O","I","D"):
        role_summary[role] = {
            "baseline":config["baseline_candidates"][role],
            "candidate":config["candidate_surfaces"][role],
            "changed_candidate_face_rows":changed_rows[role].tolist(),
            "removed_baseline_face_rows":removed_rows[role].tolist(),
            "changed_face_count":int(len(changed_rows[role])),
            "removed_face_count":int(len(removed_rows[role])),
            "candidate_face_count":int(len(candidate[role]["faces"])),
            "baseline_face_count":int(len(baseline[role]["faces"])),
            "points":f32_point_receipts[role],
            "oriented_diff_semantics":"cyclic rotations equal; reversed winding is a change",
        }
    zero = against.get("count", -1) == 0 and changed_self.get("count", -1) == 0
    out.mkdir(parents=True)
    result = {
        "schema":"h706.combined-current-exact-surface-diff-audit.v1",
        "status":"PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185" if zero else "FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185",
        "physical_admission":False,"native_steps":0,
        "config":{"path":str(config_path),"sha256":sha(config_path)},
        "roles":role_summary,
        "paired_D_retirement":paired_d_retirement_receipt,
        "changed_face_total":len(changed_records),
        "removed_baseline_face_total":len(removed_global),
        "retired_baseline_face_ids":sorted(removed_global),
        "candidate_changed_record_count_by_role":{r:len(changed_records_by_role[r]) for r in ("O","I","D")},
        "changed_vs_complete_retained_OID_both_walls_all185":against,
        "changed_OID_joint_self_pairs":changed_self,
        "wall_receipts":wall_receipts,
        "all185_receipt":bone_receipt,
        "point_namespace":{"role_order":list(ROLE_ORDER),"point_base":pointbase,
                           "total_points":len(combined_points),
                           "mapping":"candidate F32 coordinate bytes map bijectively to baseline role-local point rows"},
        "exact_owner":{"path":str(Path(pred.__file__)),"sha256":sha(Path(pred.__file__)),
                       "expected_sha256":base.OWNER_IMPL_SHA,
                       "filtered_owner_path":str(base.FILTER),"filtered_owner_sha256":sha(base.FILTER)},
        "elapsed_s":time.monotonic()-t0,
    }
    (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("retired_baseline_face_ids","roles","changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs","wall_receipts","all185_receipt")},indent=2,default=str))
    print("against_count",against.get("count"),"joint_self_count",changed_self.get("count"),
          "report_sha256",sha(out/"report.json"))

if __name__ == "__main__":
    main()

