#!/usr/bin/env python3
"""Config-driven, bounded D short-edge preflight derived from 385/379 owners.

This is preparation only. It emits the existing 385 edge-map/run-config schemas;
actual topology retention still requires the existing indexed 379 owner and a
fresh full O/I/D + walls + all-185 exact-context audit.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np

SCHEMA = "h706.d-short-edge-current-preflight-input.v2"
OUT_EDGE_SCHEMA = "h706.d-short-edge-quotient-authorized-edges.v1"
RUN_CONFIG_SCHEMA = "h706.fixed-boundary-short-edge-collapse-run-config.v1"
NATIVE_Q_MIN = 1e-5
NATIVE_VOLUME_MIN_M3 = 1e-18
DYADIC_SCALE = 1 << 149

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}

def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def require_pin(pins, name):
    rec = pins.get(name)
    if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
        raise ValueError(f"missing path/SHA pin for inputs.{name}")
    path = Path(rec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    got = sha(path)
    if got != rec["sha256"]:
        raise ValueError(f"input pin mismatch {name}: {got} != {rec['sha256']}")
    return path, got

def f32key(x) -> bytes:
    a = np.asarray(x, dtype="<f4")
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError("nonfinite or malformed point in coordinate key")
    return a.tobytes()

def unique_point_map(points):
    out = {}
    dup = set()
    for i, p in enumerate(np.asarray(points, dtype=np.float32)):
        k = f32key(p)
        if k in out:
            dup.add(k)
        else:
            out[k] = int(i)
    for k in dup:
        out[k] = -1
    return out

def map_points(points, lookup):
    return np.asarray([lookup.get(f32key(p), -1) for p in np.asarray(points, dtype=np.float32)], dtype=np.int64)

def exact_coord_int(x) -> int:
    n, d = float(np.float32(x)).as_integer_ratio()
    if DYADIC_SCALE % d:
        raise ValueError("Float32 coordinate denominator does not divide 2^149")
    return n * (DYADIC_SCALE // d)

def exact_det6(points_f32, tet, cache):
    xyz = []
    for row in map(int, tet):
        if row not in cache:
            cache[row] = tuple(exact_coord_int(c) for c in points_f32[row])
        xyz.append(cache[row])
    a = [xyz[1][i] - xyz[0][i] for i in range(3)]
    b = [xyz[2][i] - xyz[0][i] for i in range(3)]
    c = [xyz[3][i] - xyz[0][i] for i in range(3)]
    cr = (b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0])
    return a[0]*cr[0] + a[1]*cr[1] + a[2]*cr[2]

def tet_metrics(points_f32, tets):
    tets = np.asarray(tets, dtype=np.int64)
    if len(tets) == 0:
        e = np.empty(0, np.float64)
        return e, e, e
    xyz = np.asarray(points_f32, np.float32)[tets].astype(np.float64)
    a = xyz[:, 1] - xyz[:, 0]
    b = xyz[:, 2] - xyz[:, 0]
    c = xyz[:, 3] - xyz[:, 0]
    det6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    longest = np.zeros(len(tets), np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            longest = np.maximum(longest, np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1))
    volume = det6 / 6.0
    q = np.divide(det6, longest**3, out=np.zeros_like(det6), where=longest > 0)
    return det6, volume, q

def surface_qface(points_f32, faces):
    f = np.asarray(faces, np.int64)
    xyz = np.asarray(points_f32, np.float32)[f].astype(np.float64)
    a, b = xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0]
    c = xyz[:,2]-xyz[:,1]
    area2 = np.linalg.norm(np.cross(a,b), axis=1)
    longest = np.maximum.reduce((np.linalg.norm(a,axis=1), np.linalg.norm(b,axis=1), np.linalg.norm(c,axis=1)))
    return np.divide(area2, longest**2, out=np.zeros_like(area2), where=longest > 0)

def oriented_face_key(tri):
    x = tuple(map(int, tri)); s = tuple(sorted(x)); rank = {v:i for i,v in enumerate(s)}
    inv = int(rank[x[0]] > rank[x[1]]) + int(rank[x[0]] > rank[x[2]]) + int(rank[x[1]] > rank[x[2]])
    return (*s, -1 if inv % 2 else 1)

def cell_key(points_f32, tet, material):
    coords = sorted(f32key(points_f32[int(v)]) for v in tet)
    return (int(material), tuple(coords))

def require_current_context_audit(path: Path, expected_role_shas: dict[str, str]):
    """Fail closed on a non-PASS or stale audit; final per-op recheck remains 379's job."""
    audit = json.loads(path.read_text())
    if "PASS" not in str(audit.get("status", "")).upper():
        raise ValueError("current complete-context audit is not PASS")
    text = json.dumps(audit, sort_keys=True)
    for role, digest in expected_role_shas.items():
        if digest not in text:
            raise ValueError(f"current complete-context audit does not bind {role} candidate SHA")
    if "185" not in text:
        raise ValueError("audit text does not expose the required all-185 context scope")
    return audit

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON input pin/config file")
    args = ap.parse_args()
    started = time.monotonic()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"expected config schema {SCHEMA}")
    pins = cfg.get("inputs", {})
    out = Path(cfg["output_directory"])
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")
    out.mkdir(parents=True)
    params = cfg.get("parameters", {})
    max_edge = float(params.get("max_edge_length_m", 1e-5))
    maximum_candidates = int(params.get("maximum_candidates", 2048))
    max_accepted = int(params.get("max_accepted", maximum_candidates))
    max_trials = int(params.get("max_trials", 100000))
    wall_budget = float(params.get("wall_budget_seconds", 180.0))
    if not (0 < max_edge <= 1e-5): raise ValueError("max_edge_length_m must be positive and <= 10 um")
    if not (0 < maximum_candidates <= 8192): raise ValueError("maximum_candidates must be in [1,8192]")
    if not (0 < max_accepted <= maximum_candidates): raise ValueError("max_accepted must be in [1,maximum_candidates]")
    if not (0 < max_trials <= 100000): raise ValueError("max_trials must be in [1,100000]")
    if not (0 < wall_budget <= 180.0): raise ValueError("wall budget must be in (0,180] seconds")

    pin_paths = {k: require_pin(pins, k) for k in (
        "O", "I", "D", "fat_volume", "witness_census", "source_Fat336",
        "cavity_shells", "cavity_reference_volume", "comp1_cavity_shells",
        "comp1_cavity_reference_volume", "protected_routes",
        "complete_context_audit", "back_scope_report")}
    role_paths = {r: pin_paths[r][0] for r in ("O","I","D")}
    role_shas = {r: pin_paths[r][1] for r in ("O","I","D")}
    audit_path = pin_paths["complete_context_audit"][0]
    require_current_context_audit(audit_path, role_shas)
    H = Path(cfg.get("artifact_root", str(cfg_path.parent)))
    source_owner = Path(cfg.get("source_preflight_385", ""))
    source_owner_sha = cfg.get("source_preflight_385_sha256", "")
    transfer_owner = Path(cfg.get("transfer_owner_379", ""))
    transfer_owner_sha = cfg.get("transfer_owner_379_sha256", "")
    if source_owner_sha and sha(source_owner) != source_owner_sha: raise ValueError("385 source owner pin mismatch")
    if transfer_owner_sha and sha(transfer_owner) != transfer_owner_sha: raise ValueError("379 transfer owner pin mismatch")

    parent = {r: load_npz(role_paths[r]) for r in ("O","I","D")}
    D = parent["D"]
    DP = np.asarray(D["points_world_m_f32"], np.float32)
    DF = np.asarray(D["faces"], np.int64)
    for r in ("O","I","D"):
        if "points_world_m_f32" not in parent[r] or "faces" not in parent[r]: raise ValueError(f"{r} typed candidate lacks current geometry arrays")
    V = load_npz(pin_paths["fat_volume"][0])
    VP = np.asarray(V["points_world_m"], np.float32)
    VT = np.asarray(V["tetrahedra"], np.int64)
    VMAT = np.asarray(V["material_indices"], np.int32)
    VB = np.asarray(V["boundary_faces"], np.int64)
    target = np.asarray(V["desired_boundary_faces"], np.int64)
    if len(VT) != len(VMAT): raise ValueError("Fat tetrahedron/material row mismatch")
    if np.any(VT < 0) or np.any(VT >= len(VP)): raise ValueError("Fat tetrahedron point row out of range")
    if np.any(VB < 0) or np.any(VB >= len(VP)) or np.any(target < 0) or np.any(target >= len(VP)): raise ValueError("Fat boundary point row out of range")
    D_to_V = map_points(DP, unique_point_map(VP))
    declared = V.get("fat_current_target_rebind_D_point_to_volume_rows")
    if declared is None or not np.array_equal(D_to_V, np.asarray(declared, np.int64)):
        raise ValueError("current typed D to Fat point map is missing or differs from exact F32 coordinate crosswalk")
    if np.any(D_to_V < 0) or not np.array_equal(DP, VP[D_to_V]): raise ValueError("current D points are not unique exact F32 points in current Fat volume")

    # Rebuild both old and comp1 cavity protection by canonical material + sorted
    # F32-coordinate tet keys, never stale row IDs. Preserve every current full
    # vertex star and record any shell-only points absent from this volume domain.
    current_key_to_row = {}
    for row, tet in enumerate(VT):
        k = cell_key(VP, tet, VMAT[row])
        if k in current_key_to_row: raise ValueError("duplicate material+F32-corner tet key in current Fat volume")
        current_key_to_row[k] = int(row)
    current_point_lookup = unique_point_map(VP)
    protected_source_cells = set()
    protected_source_points = set()
    cavity_source_receipts = {}
    missing_cavity_keys = []
    unmapped_cavity_shell_points = {}
    for label, shell_name, volume_name in (
        ("run021", "cavity_shells", "cavity_reference_volume"),
        ("run037_comp1", "comp1_cavity_shells", "comp1_cavity_reference_volume"),
    ):
        C = load_npz(pin_paths[shell_name][0]); CV = load_npz(pin_paths[volume_name][0])
        cpoints = np.asarray(CV["points_world_m"], np.float32)
        ctets = np.asarray(CV["tetrahedra"], np.int64)
        cmats = np.asarray(CV["material_indices"], np.int32)
        cavity_rows = np.asarray(C["input_cell_rows"], np.int64)
        if np.any(cavity_rows < 0) or np.any(cavity_rows >= len(ctets)):
            raise ValueError(f"{label}: cavity input cell row outside its pinned reference volume")
        ref_keys = [cell_key(cpoints, ctets[int(r)], cmats[int(r)]) for r in cavity_rows]
        absent = [i for i,k in enumerate(ref_keys) if k not in current_key_to_row]
        if absent:
            missing_cavity_keys.extend((label, int(i)) for i in absent)
            raise ValueError(f"{label}: {len(absent)} canonical cavity old-tet keys are absent from current Fat")
        mapped_cells = [current_key_to_row[k] for k in ref_keys]
        protected_source_cells.update(mapped_cells)
        point_keys = set()
        for row in cavity_rows:
            for v in ctets[int(row)]: point_keys.add(f32key(cpoints[int(v)]))
        gp = np.asarray(C.get("global_point_rows", np.empty(0)), np.int64)
        if np.any(gp < 0) or np.any(gp >= len(cpoints)):
            raise ValueError(f"{label}: cavity global point row outside reference volume")
        shell_keys = {f32key(cpoints[int(row)]) for row in gp}
        all_keys = point_keys | shell_keys
        mapped_points = set()
        unmapped = []
        for k in all_keys:
            row = current_point_lookup.get(k, -1)
            if row < 0:
                if k in point_keys:
                    raise ValueError(f"{label}: selected old-cell point is missing or ambiguous in current Fat")
                unmapped.append(k.hex())
            else:
                mapped_points.add(int(row))
        protected_source_points.update(mapped_points)
        cavity_source_receipts[label] = {
            "shell_path":str(pin_paths[shell_name][0]), "shell_sha256":pin_paths[shell_name][1],
            "reference_volume_path":str(pin_paths[volume_name][0]), "reference_volume_sha256":pin_paths[volume_name][1],
            "selected_reference_cell_count":len(cavity_rows), "matched_current_cell_count":len(mapped_cells),
            "mapped_full_vertex_point_count":len(mapped_points), "unmapped_shell_only_point_count":len(unmapped),
            "unmapped_shell_only_point_f32_key_hex_sha256":hashlib.sha256("\n".join(sorted(unmapped)).encode()).hexdigest(),
        }
    cavity_current_cells = np.asarray(sorted(protected_source_cells), np.int64)
    cavity_current_points = protected_source_points
    # Current D protected-rim routes resolve by exact point bits.
    route_data = json.loads(pin_paths["protected_routes"][0].read_text())
    d_lookup = unique_point_map(DP)
    protected_D = set()
    for route in route_data.get("surface_routes", []):
        if route.get("role") != "D": continue
        row = d_lookup.get(f32key(route["target_world_m_f32"]), -1)
        if row < 0: raise ValueError("protected D rim route is missing or ambiguous in current typed D")
        protected_D.add(int(row))
    if len(protected_D) != 59: raise ValueError(f"expected 59 authenticated D rim routes, found {len(protected_D)}")

    # Dynamic actual/desired mismatch; no stale hard-coded 127/197 counts.
    wanted = Counter(oriented_face_key(f) for f in target)
    actual = Counter(oriented_face_key(f) for f in VB)
    missing = list((wanted-actual).elements()); extra = list((actual-wanted).elements())
    mismatch_volume_nodes = {int(v) for f in missing+extra for v in f[:3]}
    volume_to_D = {int(v):i for i,v in enumerate(D_to_V)}
    mismatch_D_seed = {volume_to_D[v] for v in mismatch_volume_nodes if v in volume_to_D}
    vfaces = [[] for _ in range(len(DP))]
    for fi, tri in enumerate(DF):
        for v in tri: vfaces[int(v)].append(fi)
    mismatch_D_fullstar = set(mismatch_D_seed)
    for seed in tuple(mismatch_D_seed):
        for fi in vfaces[seed]: mismatch_D_fullstar.update(map(int, DF[fi]))

    # Full current point->tet incidence and exact F32 baseline quality.
    flat = VT.reshape(-1); owners = np.repeat(np.arange(len(VT), dtype=np.int64), 4)
    order = np.argsort(flat, kind="stable"); counts = np.bincount(flat, minlength=len(VP))
    offsets = np.empty(len(VP)+1, np.int64); offsets[0]=0; np.cumsum(counts, out=offsets[1:])
    node_cells = owners[order]
    def incident(node): return node_cells[int(offsets[node]):int(offsets[node+1])]
    det0, vol0, q0 = tet_metrics(VP, VT)
    good0 = (det0 > 0) & (vol0 > NATIVE_VOLUME_MIN_M3) & (q0 >= NATIVE_Q_MIN)
    if np.any(det0 <= 0): raise ValueError("current Fat input already contains exact-orientation-invalid tets")
    exact_cache = {}
    # Check exact integer sign on all cells in protected stars and on every touched star later.
    for row in np.unique(np.concatenate((cavity_current_cells, np.concatenate([incident(v) for v in cavity_current_points]) if cavity_current_points else np.empty(0,np.int64)))):
        if exact_det6(VP, VT[int(row)], exact_cache) <= 0: raise ValueError("current cavity-protected source cell has nonpositive exact F32 determinant")
    cavity_cell_set = set(map(int,cavity_current_cells))
    cavity_star_cell_set = set()
    for node in cavity_current_points: cavity_star_cell_set.update(map(int, incident(node)))
    cavity_protected_cells = cavity_cell_set | cavity_star_cell_set

    # Current D edge topology index.
    edge_faces = {}; neighbors=[set() for _ in range(len(DP))]; vfaces=[[] for _ in range(len(DP))]
    for fi, tri in enumerate(DF):
        a,b,c=map(int,tri); vfaces[a].append(fi);vfaces[b].append(fi);vfaces[c].append(fi)
        for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
            e=(u,v) if u<v else (v,u);edge_faces.setdefault(e,[]).append((fi,w))
            neighbors[u].add(v);neighbors[v].add(u)
    qD = surface_qface(DP, DF)

    # Witness census point rows are mapped from the pinned Fat336 source by exact F32.
    W=load_npz(pin_paths["witness_census"][0]); src336=load_npz(pin_paths["source_Fat336"][0])
    BP=np.asarray(src336["points_world_m"],np.float32)
    source_to_D=map_points(BP,d_lookup)
    E=np.asarray(W["edge_point_rows"],np.int64); layer=np.asarray(W["layer"],np.int64)
    lengths=np.asarray(W["length_m"],np.float64)
    if E.ndim!=2 or E.shape[1]!=2 or len(layer)!=len(E) or len(lengths)!=len(E): raise ValueError("witness census row contract is malformed")
    d_layer=int(cfg.get("d_witness_layer",2))
    src_by_edge={}; map_rejections=0; over_cap=0
    for wi in range(len(E)):
        if int(layer[wi]) != d_layer: continue
        if float(lengths[wi]) > max_edge: over_cap += 1; continue
        rows=E[wi]
        if np.any(rows<0) or np.any(rows>=len(source_to_D)): map_rejections += 1; continue
        mapped=source_to_D[rows]
        if np.any(mapped<0) or mapped[0]==mapped[1]: map_rejections += 1; continue
        edge=tuple(sorted(map(int,mapped)))
        src_by_edge.setdefault(edge,[]).append(int(wi))

    # Exact current Fat boundary support and global tetrahedron keys for collision rejection.
    boundary_keys={tuple(sorted(map(int,f))) for f in VB}
    global_tet_keys={tuple(sorted(map(int,t))) for t in VT}
    excluded_D=set(protected_D)|set(mismatch_D_fullstar)
    excluded_volume=set(cavity_protected_cells)
    reasons=Counter(); evaluated=[]; trial_count=0; timed_out=False
    def trial(dropD,keepD,retired_surface):
        nonlocal trial_count, exact_cache
        trial_count += 1
        if trial_count > max_trials or time.monotonic()-started >= wall_budget: return None,"budget"
        dropV=int(D_to_V[dropD]); keepV=int(D_to_V[keepD])
        if dropV==keepV or dropV<0 or keepV<0: return None,"endpoint_map"
        affected=np.asarray(incident(dropV),np.int64)
        if not len(affected): return None,"no_volume_incidence"
        if set(map(int,affected)) & excluded_volume: return None,"protected_cavity_full_vertex_star"
        old=VT[affected]; new=np.where(old==dropV,keepV,old)
        sorted_new=np.sort(new,axis=1); deleted=np.any(sorted_new[:,1:]==sorted_new[:,:-1],axis=1)
        kept=new[~deleted]
        od,ov,oq=tet_metrics(VP,old); nd,nv,nq=tet_metrics(VP,kept)
        if len(kept)==0: return None,"quotient_deletes_entire_volume_star"
        if np.any(nd<=0) or np.any(nv<=NATIVE_VOLUME_MIN_M3): return None,"native_positive_or_absolute_volume_gate"
        old_good=(od>0)&(ov>NATIVE_VOLUME_MIN_M3)&(oq>=NATIVE_Q_MIN)
        new_good=(nd>0)&(nv>NATIVE_VOLUME_MIN_M3)&(nq>=NATIVE_Q_MIN)
        if np.any(old_good[~deleted]&~new_good): return None,"old_good_cell_regression"
        old_bad=int(np.count_nonzero(~old_good)); new_bad=int(np.count_nonzero(~new_good))
        if new_bad>old_bad: return None,"native_bad_cell_count_regression"
        # Exact F32 signs and duplicate tet check against the entire current mesh.
        for t in old:
            if exact_det6(VP,t,exact_cache)<=0: return None,"old_exact_orientation"
        for t in kept:
            if exact_det6(VP,t,exact_cache)<=0: return None,"new_exact_orientation"
        local_keys=[tuple(sorted(map(int,t))) for t in kept]
        if len(set(local_keys)) != len(local_keys): return None,"local_duplicate_tet"
        affected_set=set(map(int,affected))
        for key in local_keys:
            if key in global_tet_keys:
                # A same key can already be in the drop star as a retained cell; that is caught locally.
                owners_for_key = tet_owners.get(key, ())
                if any(int(r) not in affected_set for r in owners_for_key): return None,"global_duplicate_tet"
        for face_row in retired_surface:
            tri=DF[int(face_row)]; mapped=D_to_V[tri]
            if np.any(mapped<0) or tuple(sorted(map(int,mapped))) not in boundary_keys:
                return None,"retired_surface_face_not_on_current_Fat_boundary"
        # Exact integer determinant delta records the volume change without cancellation.
        old_exact=sum(exact_det6(VP,t,exact_cache) for t in old)
        new_exact=sum(exact_det6(VP,t,exact_cache) for t,dead in zip(new,deleted) if not dead)
        return {
            "drop_surface_row":int(dropD),"keep_surface_row":int(keepD),
            "drop_volume_row":dropV,"keep_volume_row":keepV,
            "affected_cell_rows":affected.tolist(),"affected_cell_count":int(len(affected)),
            "deleted_cell_count":int(np.count_nonzero(deleted)),
            "native_bad_cells_before":old_bad,"native_bad_cells_after":new_bad,
            "native_bad_cell_gain":old_bad-new_bad,
            "old_good_deleted_count":int(np.count_nonzero(old_good & deleted)),
            "minimum_q_before":float(np.min(oq)) if len(oq) else 0.0,
            "minimum_q_after":float(np.min(nq)) if len(nq) else 0.0,
            "exact_det6_delta_integer":str(new_exact-old_exact),
            "exact_det6_scale_denominator":str(DYADIC_SCALE**3),
            "material_counts":{str(int(m)):int(np.count_nonzero(VMAT[affected]==m)) for m in np.unique(VMAT[affected])}
        },None

    # Keep all key owners once, so collision tests are local in candidate loops.
    tet_owners={}
    for r,t in enumerate(VT): tet_owners.setdefault(tuple(sorted(map(int,t))),[]).append(int(r))
    for edge,witness_rows in src_by_edge.items():
        u,v=edge
        if time.monotonic()-started >= wall_budget: timed_out=True; break
        if u in excluded_D or v in excluded_D: reasons["protected_rim_or_target_mismatch_star"]+=1; continue
        ef=edge_faces.get(edge,[])
        if len(ef)!=2: reasons["D_edge_not_two_face_manifold"]+=1; continue
        opposites={int(w) for _,w in ef}
        if (neighbors[u]&neighbors[v])-{u,v} != opposites: reasons["D_link_condition"]+=1; continue
        star_rows=sorted(set(vfaces[u])|set(vfaces[v]))
        star_vertices=set(map(int,DF[np.asarray(star_rows,np.int64)].reshape(-1)))
        if star_vertices & excluded_D: reasons["D_surface_star_intersects_protection"]+=1; continue
        length=float(np.linalg.norm(DP[u].astype(np.float64)-DP[v].astype(np.float64)))
        if length>max_edge: reasons["current_D_edge_over_cap"]+=1; continue
        removed={int(fr) for fr,_ in ef}
        changed_u={int(fr) for fr in vfaces[u] if v not in DF[int(fr)]}
        changed_v={int(fr) for fr in vfaces[v] if u not in DF[int(fr)]}
        options=[]; union_cells=set(); failures=Counter()
        for drop,keep,changed in ((u,v,changed_u),(v,u,changed_v)):
            result,why=trial(drop,keep,sorted(removed|changed))
            if result is not None:
                options.append(result); union_cells.update(result["affected_cell_rows"])
            else:
                failures[why]+=1
        if not options:
            reasons.update({"both_orientations_"+k:v for k,v in failures.items()}); continue
        options.sort(key=lambda x:(-x["native_bad_cell_gain"],-x["minimum_q_after"],x["old_good_deleted_count"],x["drop_surface_row"]))
        best=options[0]
        evaluated.append({
            "edge":list(edge),"length_m":length,"source_350_rows":witness_rows,
            "surface_star_rows":star_rows,"surface_star_vertices":sorted(star_vertices),
            "volume_star_cells":sorted(union_cells),"allowed_orientations":options,
            "rank_native_bad_cell_gain":int(best["native_bad_cell_gain"]),
            "rank_minimum_q_after":float(best["minimum_q_after"]),
            "seed_face":int(min(fi for fi,_ in ef)),
            "seed_qface":float(min(qD[fi] for fi,_ in ef))
        })
        if trial_count >= max_trials: timed_out=True; break
    evaluated.sort(key=lambda x:(-x["rank_native_bad_cell_gain"],-x["rank_minimum_q_after"],x["length_m"],x["seed_qface"],x["seed_face"],x["edge"]))

    # Greedy disjoint proposal, still requiring the downstream 379 all-context audit.
    selected=[]; used_surface=set(); used_cells=set(); selection_rejects=Counter()
    for c in evaluated:
        if len(selected)>=max_accepted: break
        sv=set(c["surface_star_vertices"]); vc=set(c["volume_star_cells"])
        if sv & used_surface: selection_rejects["overlapping_D_surface_star"]+=1; continue
        if vc & used_cells: selection_rejects["overlapping_Fat_cell_star"]+=1; continue
        selected.append(c); used_surface |= sv; used_cells |= vc

    # Identity wrappers are the existing 385/196 handoff convention; geometry arrays
    # remain byte-identical and original typed parents are separately pinned.
    wrapped={}; wrapper_meta={}
    for role in ("O","I","D"):
        a=dict(parent[role]); npt=len(a["points_world_m_f32"]); nf=len(a["faces"])
        a["point_output_to_parent_rows"]=np.arange(npt,dtype=np.int64)
        a["point_output_to_input_rows"]=np.arange(npt,dtype=np.int64)
        a["face_output_to_parent_rows"]=np.arange(nf,dtype=np.int64)
        a["face_output_to_input_rows"]=np.arange(nf,dtype=np.int64)
        a["immediate_parent_candidate_sha256"]=np.asarray([role_shas[role]],dtype="U64")
        a["typed_parent_candidate_sha256"]=np.asarray([role_shas[role]],dtype="U64")
        a["source_candidate_sha256"]=np.asarray([role_shas[role]],dtype="U64")
        a["source_candidate_path"]=np.asarray([str(role_paths[role])],dtype="U1024")
        a["immediate_parent_candidate_path"]=np.asarray([str(role_paths[role])],dtype="U1024")
        wp=out/f"identity-seed-{role}.npz"; np.savez_compressed(wp,**a)
        if sha(wp)==role_shas[role]: raise AssertionError("identity wrapper unexpectedly equals parent file hash")
        if not np.array_equal(a["points_world_m_f32"],parent[role]["points_world_m_f32"]) or not np.array_equal(a["faces"],parent[role]["faces"]):
            raise ValueError(f"{role} identity wrapper changed geometry")
        wrapped[role]={"path":str(wp),"sha256":sha(wp)}
        wrapper_meta[role]={"path":str(wp),"sha256":sha(wp),"geometry_parent_path":str(role_paths[role]),"geometry_parent_sha256":role_shas[role],"point_count":npt,"face_count":nf}
    seed_report={"schema":"h706.current-typed-candidate-identity-seed.v1","status":"PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL","derivation":"metadata identity wrapper; point and face arrays verified bit-identical to current typed input","outputs":wrapper_meta,"accepted_operations":[],"coordinates_changed":False,"physical_admission":False,"native_steps":0}
    seed_path=out/"identity-seed-report.json"; write_json(seed_path,seed_report)
    qcounts={r:int(np.count_nonzero(surface_qface(np.asarray(parent[r]["points_world_m_f32"],np.float32),np.asarray(parent[r]["faces"],np.int64))<NATIVE_Q_MIN)) for r in ("O","I","D")}
    qreport={"schema":"h706.current-qbound-witness-census.v1","status":"READ_ONLY_CURRENT_QBOUND_WITNESS_CENSUS","witness_counts":qcounts,"normalized_det_surface_floor":NATIVE_Q_MIN,"candidate_faces":{r:{"path":str(wrapped[r]["path"]),"sha256":wrapped[r]["sha256"]} for r in ("O","I","D")},"purpose":"degenerate-surface prefilter only; paired Fat native cell gates evaluated on local volume stars"}
    qpath=out/"quality-prepass-report.json"; write_json(qpath,qreport)

    edge_rows=np.asarray([c["edge"] for c in selected],np.int64).reshape((-1,2))
    orient_edges=[]; orient_drop=[]; orient_keep=[]; orient_gain=[]; orient_before=[]; orient_after=[]; orient_delta=[]; orient_offsets=[0]; orient_cells=[]
    for c in selected:
        for o in c["allowed_orientations"]:
            orient_edges.append(c["edge"]); orient_drop.append(o["drop_surface_row"]); orient_keep.append(o["keep_surface_row"])
            orient_gain.append(o["native_bad_cell_gain"]); orient_before.append(o["native_bad_cells_before"]); orient_after.append(o["native_bad_cells_after"])
            orient_delta.append(o["exact_det6_delta_integer"]); orient_cells.extend(o["affected_cell_rows"]); orient_offsets.append(len(orient_cells))
    edge_npz=out/"allowed-D-edge-map.npz"
    np.savez_compressed(edge_npz,
        D_edge_rows=edge_rows,
        D_allowed_orientation_edge_rows=np.asarray(orient_edges,np.int64).reshape((-1,2)),
        D_allowed_collapsed_point_rows=np.asarray(orient_drop,np.int64),
        D_allowed_retained_point_rows=np.asarray(orient_keep,np.int64),
        D_source_350_witness_counts=np.asarray([len(c["source_350_rows"]) for c in selected],np.int32),
        D_edge_length_m=np.asarray([c["length_m"] for c in selected],np.float64),
        D_seed_face_rows=np.asarray([c["seed_face"] for c in selected],np.int64),
        D_seed_face_qface=np.asarray([c["seed_qface"] for c in selected],np.float64),
        D_orientation_native_bad_cell_gain=np.asarray(orient_gain,np.int32),
        D_orientation_native_bad_before=np.asarray(orient_before,np.int32),
        D_orientation_native_bad_after=np.asarray(orient_after,np.int32),
        D_orientation_exact_det6_delta_integer=np.asarray(orient_delta,dtype="U256"),
        D_orientation_affected_cell_offsets=np.asarray(orient_offsets,np.int64),
        D_orientation_affected_cell_rows=np.asarray(orient_cells,np.int64))
    edge_sha=sha(edge_npz)
    back_path=pin_paths["back_scope_report"][0]; back_sha=pin_paths["back_scope_report"][1]
    run_cfg={
        "schema":RUN_CONFIG_SCHEMA,"batch_id":cfg.get("batch_id","D-shortedge-current-bounded"),
        "current_candidates":wrapped,
        "typed_ancestry_parents":{r:{"path":str(role_paths[r]),"sha256":role_shas[r]} for r in ("O","I","D")},
        "lineage_layers":[{"layer_id":"current-typed-identity-seed","report":{"path":str(seed_path),"sha256":sha(seed_path)},"candidates":wrapped}],
        "quality_prepass_report":{"path":str(qpath),"sha256":sha(qpath)},"expected_bad_face_counts":qcounts,
        "back_scope_report":{"path":str(back_path),"sha256":back_sha},
        "authorized_edge_map":{"path":str(edge_npz),"sha256":edge_sha,"schema":OUT_EDGE_SCHEMA},
        "parameters":{"max_edge_length_m":max_edge,"maximum_candidates":maximum_candidates,"max_accepted":max_accepted,"max_trials":max_trials,"wall_budget_seconds":wall_budget,"normalized_det_minimum":NATIVE_Q_MIN,"minimum_altitude_m":20e-9,"minimum_angle_degrees":0.0,"replacement_qbound_policy":"all_new_faces_pass"},
        "preflight_source_batch":{"schema":SCHEMA,"config_path":str(cfg_path),"config_sha256":sha(cfg_path),"current_Fat_volume":{"path":str(pin_paths['fat_volume'][0]),"sha256":pin_paths['fat_volume'][1]},"current_context_audit":{"path":str(audit_path),"sha256":pin_paths['complete_context_audit'][1]},"native_positive_volume_min_m3":NATIVE_VOLUME_MIN_M3,"native_normalized_det_minimum":NATIVE_Q_MIN}
    }
    run_cfg_path=out/"run-config.json"; write_json(run_cfg_path,run_cfg)
    elapsed=time.monotonic()-started
    completed=len(evaluated)
    prep={
        "schema":"h706.d-short-edge-current-preflight.v1",
        "status":"INCOMPLETE_PREFLIGHT_BUDGET" if timed_out or elapsed>=wall_budget else "PASS_D_SHORT_EDGE_PREFLIGHT_ONLY",
        "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
        "source_pins":{k:{"path":str(pin_paths[k][0]),"sha256":pin_paths[k][1]} for k in pin_paths},
        "owner_pins":{"385_preflight":{"path":str(source_owner),"sha256":source_owner_sha},"379_indexed_transfer":{"path":str(transfer_owner),"sha256":transfer_owner_sha}},
        "parameters":{"max_edge_length_m":max_edge,"maximum_candidates":maximum_candidates,"max_accepted":max_accepted,"max_trials":max_trials,"wall_budget_seconds":wall_budget,"native_min_q":NATIVE_Q_MIN,"native_min_volume_m3":NATIVE_VOLUME_MIN_M3,"rank_order":["descending measured native bad-cell gain","descending minimum resulting local q","ascending exact D edge length"]},
        "protection":{"cavity_sources":cavity_source_receipts,"cavity_reference_cell_keys":sum(x["selected_reference_cell_count"] for x in cavity_source_receipts.values()),"cavity_current_cell_rows":len(cavity_current_cells),"cavity_full_vertex_points":len(cavity_current_points),"cavity_full_vertex_star_cells":len(cavity_star_cell_set),"protected_D_rim_route_points":len(protected_D),"current_target_missing_oriented_faces":len(missing),"current_target_extra_oriented_faces":len(extra),"target_mismatch_D_seed_points":len(mismatch_D_seed),"target_mismatch_D_fullstar_points":len(mismatch_D_fullstar)},
        "mapping":{"current_D_points":len(DP),"exact_unique_D_to_current_Fat_points":int(np.count_nonzero(D_to_V>=0)),"source336_points_mapped_to_current_D":int(np.count_nonzero(source_to_D>=0)),"pending_cavity_missing_old_cell_keys":len(missing_cavity_keys)},
        "witness_census":{"layer_value":d_layer,"D_witness_edges_under_cap":sum(1 for i in range(len(E)) if layer[i]==d_layer and lengths[i]<=max_edge),"unique_current_D_edges":len(src_by_edge),"map_rejections":map_rejections,"witnesses_over_cap":over_cap},
        "preflight":{"evaluated_current_D_edges":completed,"selected_disjoint_edges":len(selected),"selected_allowed_orientations":len(orient_drop),"actual_native_bad_cell_gain_by_selected_edge":[int(c['rank_native_bad_cell_gain']) for c in selected],"maximum_selected_edge_m":max((c['length_m'] for c in selected),default=0.0),"selected_volume_cell_star_count":len(used_cells),"source_trials":trial_count,"elapsed_seconds":elapsed,"timed_out":bool(timed_out),"candidate_rejections":dict(reasons),"batch_rejections":dict(selection_rejects),"selected_operations":[{"edge":c["edge"],"length_m":c["length_m"],"native_bad_cell_gain":c["rank_native_bad_cell_gain"],"orientations":c["allowed_orientations"],"source_350_rows":c["source_350_rows"],"surface_star_rows":c["surface_star_rows"],"surface_star_vertices":c["surface_star_vertices"],"volume_star_cells":c["volume_star_cells"]} for c in selected]},
        "outputs":{"identity_seed_report":{"path":str(seed_path),"sha256":sha(seed_path)},"quality_prepass":{"path":str(qpath),"sha256":sha(qpath)},"authorized_edge_map":{"path":str(edge_npz),"sha256":edge_sha},"run_config":{"path":str(run_cfg_path),"sha256":sha(run_cfg_path)}},
        "limitations":["This is a bounded candidate preflight only; it does not retain contractions or prove all-185 intersections per operation.","The existing indexed 379 surface/volume owner must independently validate each retained operation against current O/I/D, both walls, all 185 source faces, exact boundary, positivity, old-good regression, and source ancestry.","No mass mapping, native installation, or physical admission was performed."]
    }
    report=out/"preflight.json"; write_json(report,prep)
    print(json.dumps({"status":prep["status"],"output_directory":str(out),"selected_edges":len(selected),"evaluated_edges":completed,"allowed_orientations":len(orient_drop),"native_bad_gain_sum":sum(int(c["rank_native_bad_cell_gain"]) for c in selected),"elapsed_seconds":elapsed,"preflight_sha256":sha(report),"edge_map_sha256":edge_sha,"run_config_sha256":sha(run_cfg_path)},indent=2))

if __name__ == "__main__":
    main()
