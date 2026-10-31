#!/usr/bin/env python3
"""Source-only v24: joint I/D inner-back and ordinary-O local point fit with exact 805 audit/rollback/replay."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BUILDER = H / "repair_coupled_quality_points_722_v24b_ordinary_outer_o_local.py"
AUDITOR = H / "combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py"
PREPARER = H / "bounded625-local-cavity-repair/prepare_bounded_rollback_625_inner805.py"
FINALIZER = H / "bounded625-local-cavity-repair/finalize_v7_rollback_805.py"
ROLLBACK_HELPER = H / "bounded625-local-cavity-repair/bounded_point_component_rollback_625.py"
BASE_DRIVER = H / "fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py"
QUALITY_LP = H / "629-quality-lp/quality_lp_629_v18_role_balanced_shared_feature.py"
TRANSFER_OWNER = H / "transfer_surface_contractions_cross_tissue_470_root.py"
QUALITY_OWNER = H / "fixed_boundary_volume_flip_198_root.py"
SOURCE_VERIFIER = H / "verify_current_source_evidence_807_root.py"
ROLE_ORDER = ("D", "O", "I")
VOLUME_ROLES = {"Skin": ("O", "I"), "Fat": ("I", "D")}
EXPECTED_SHA = {
    "builder": "493fbe1d0a849f7cac000d1019d0cc0643dc4adf329f9cb7772a0c66d3a1993d",
    "auditor": "0f41e4114eeb56032d4bb6a6c39484d2e8dd0de52ff977c7211a800273101234",
    "preparer": "fe99ede9eff7449e558b7f935d791dee1c3384c5159b2e9d8ef2b10e509ab1d7",
    "rollback_helper": "6a731dba8635d0129ab08602ba8706aa466924dbfa6ac3a8e60239c14388327f",
    "finalizer": "562c778b73d90ddbcf7d7aa511f6f8bbc6d79cd19773bd19151685a499010813",
    "base_driver": "d4fcc4a2693201f7ad3934b62b277a65e6f021a6662182f4ab67e85db4d96c15",
    "quality_lp": "d88d9451670d7db685314c30354430f7123e176c8d3f2b540b018586e552b850",
    "transfer_owner": "e0c0b5ff0f70966431b920f2cb3e469c0bf1d54518068d7ce6e561e5ba16b2e3",
    "quality_owner": "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0",
    "source_verifier": "9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62",
}
ZERO_AUDIT_STATUS = "PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked_pin(label: str, path: Path) -> dict:
    got = sha(path)
    expected = EXPECTED_SHA[label]
    if got != expected:
        raise ValueError(f"{label} implementation pin mismatch: {got} != {expected}")
    return {"path": str(path.resolve()), "sha256": got}


def verify_source_input(path: Path) -> dict:
    checked_pin("source_verifier", SOURCE_VERIFIER)
    spec = importlib.util.spec_from_file_location("source_evidence_807_v24", SOURCE_VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned 807 source evidence verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    receipt = module.verify_current_source_evidence(Path(path).resolve())
    if not isinstance(receipt, dict) or receipt.get("status") not in (
            "PASS_CURRENT_EXACT_SOURCE_EVIDENCE",
            "PASS_INHERITED_CURRENT_SOURCE_EVIDENCE"):
        raise ValueError("input state lacks authenticated current 805 source evidence")
    return receipt


def load_state(path: Path) -> dict:
    obj = json.loads(path.read_text())
    obj = obj.get("current_verified_state_20261031", obj)
    for key in ("source_surfaces", "Skin_current", "Fat_current"):
        if key not in obj:
            raise ValueError(f"state missing required key: {key}")
    return obj


def load_selected(path: Path, fields) -> dict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in fields if k not in z]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        return {k: np.array(z[k], copy=True) for k in fields}



def archive_text_scalar(z, key):
    """Read exactly one Unicode/bytes text value from a 0-D or 1-D NPZ field."""
    if key not in z:
        raise ValueError("typed archive missing text field: " + str(key))
    value = np.asarray(z[key])
    if value.ndim not in (0, 1) or value.size != 1 or value.dtype.kind not in ("U", "S"):
        raise ValueError("typed archive text field must be a single 0-D/1-D text value: " + str(key))
    item = value.reshape(-1)[0]
    if value.dtype.kind == "S":
        result = bytes(item).decode("utf-8", errors="strict")
    else:
        result = str(item)
    if not result or "\x00" in result:
        raise ValueError("typed archive text field is empty or contains NUL: " + str(key))
    return result


def source_f32points(archive: dict, label: str) -> np.ndarray:
    p = np.asarray(archive["points_world_m_f32"])
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(label + ": malformed source Float32 points")
    if "points_world_m" in archive:
        view=np.asarray(archive["points_world_m"])
        if view.shape!=p.shape or not np.isfinite(view).all() or not np.array_equal(view.astype(np.float32).view(np.uint32),p.view(np.uint32)):
            raise ValueError(label + ": source F64/F32 coordinate views disagree")
    return p


def load_source_surface(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as z:
        required=("points_world_m_f32","faces")
        if any(k not in z for k in required):
            raise ValueError("source surface archive lacks F32 points or faces: "+str(path))
        fields={k:np.array(z[k],copy=True) for k in required}
        if "points_world_m" in z:
            fields["points_world_m"]=np.array(z["points_world_m"],copy=True)
        return fields


def packed_point_keys(points: np.ndarray) -> list[bytes]:
    p=np.ascontiguousarray(np.asarray(points,dtype="<f4"))
    if p.ndim!=2 or p.shape[1]!=3:
        raise ValueError("point key input must be [N,3]")
    raw=p.tobytes(order="C")
    return [raw[i:i+12] for i in range(0,len(raw),12)]


def f32points(archive: dict, label: str) -> np.ndarray:
    p = np.asarray(archive["points_world_m_f32"])
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(label + ": malformed Float32 points")
    view = np.asarray(archive["points_world_m"])
    if view.shape != p.shape or not np.isfinite(view).all() or not np.array_equal(view.astype(np.float32).view(np.uint32), p.view(np.uint32)):
        raise ValueError(label + ": Float64/Float32 coordinate views disagree")
    return p


def point_keys(points: np.ndarray) -> list[bytes]:
    p = np.ascontiguousarray(np.asarray(points, dtype="<f4"))
    return [p[i].tobytes() for i in range(len(p))]


def unique_point_map(points: np.ndarray, label: str) -> dict[bytes, int]:
    out = {}
    for i, key in enumerate(point_keys(points)):
        if key in out:
            raise ValueError(f"{label}: duplicate Float32 coordinate rows {out[key]} and {i}")
        out[key] = i
    return out


def ordinary_o_allowed_source_rows(parent_state_path: Path, owner) -> tuple[set[int], set[int], dict]:
    """Independently reconstruct the ordinary-O local free mask from baseline geometry."""
    state=load_state(parent_state_path)
    surfaces={}
    surface_keys={}
    for role in ROLE_ORDER:
        entry=state["source_surfaces"][role]
        if sha(Path(entry["path"]))!=entry["sha256"]:
            raise ValueError(role+": baseline source pin mismatch while rebuilding O free mask")
        surfaces[role]=load_source_surface(Path(entry["path"]))
        pp=source_f32points(surfaces[role],role+" O-mask baseline")
        ff=np.asarray(surfaces[role]["faces"],np.int64)
        active=np.unique(ff.reshape(-1))
        surface_keys[role]=set(packed_point_keys(pp[active]))
    volumes={}
    for vrole in ("Skin","Fat"):
        entry=state[vrole+"_current"]
        if sha(Path(entry["path"]))!=entry["sha256"]:
            raise ValueError(vrole+": baseline volume pin mismatch while rebuilding O free mask")
        a=load_selected(Path(entry["path"]),("points_world_m_f32","points_world_m","tetrahedra","material_indices"))
        f32points(a,vrole+" O-mask baseline volume")
        volumes[vrole]=a
    skin_p=np.asarray(volumes["Skin"]["points_world_m_f32"],np.float32)
    fat_p=np.asarray(volumes["Fat"]["points_world_m_f32"],np.float32)
    skin_keys=packed_point_keys(skin_p)
    fat_keys=packed_point_keys(fat_p)
    lookup={}
    for row,key in enumerate(skin_keys):
        lookup.setdefault(key,row)
    skin_map=np.arange(len(skin_p),dtype=np.int64)
    fat_map=np.empty(len(fat_p),np.int64)
    fat_extra=[]
    for row,key in enumerate(fat_keys):
        if key in surface_keys["I"] and key in lookup:
            fat_map[row]=lookup[key]
        else:
            global_row=len(skin_p)+len(fat_extra)
            fat_map[row]=global_row
            fat_extra.append(row)
            lookup.setdefault(key,global_row)
    p=np.concatenate((skin_p,fat_p[np.asarray(fat_extra,np.int64)]),axis=0)
    vmap={"Skin":skin_map,"Fat":fat_map}
    t=np.vstack([vmap[r][np.asarray(volumes[r]["tetrahedra"],np.int64)] for r in ("Skin","Fat")])
    d,q=owner.measurements(p,t)
    passed=np.asarray(owner.passes(d,q),bool)
    if np.any(np.asarray(d)<=0):
        raise ValueError("baseline has a nonpositive determinant while rebuilding O free mask")
    bad=set(map(int,np.unique(t[~passed]).reshape(-1)))
    o_archive=surfaces["O"]
    o_faces=np.asarray(o_archive["faces"],np.int64)
    o_points=source_f32points(o_archive,"O source surface for independent local-mask replay")
    o_global=np.asarray([lookup.get(k,-1) for k in packed_point_keys(o_points)],np.int64)
    active_o=np.unique(o_faces.reshape(-1))
    if np.any(o_global[active_o]<0):
        raise ValueError("ordinary O surface contains points unmapped to the baseline volume")
    seed_local=np.flatnonzero(np.isin(o_global,np.fromiter(bad,np.int64))) if bad else np.empty(0,np.int64)
    incident=np.any(np.isin(o_faces,seed_local),axis=1) if len(seed_local) else np.zeros(len(o_faces),bool)
    ring=np.unique(o_faces[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
    candidate_local=set(map(int,np.unique(np.concatenate((seed_local,ring)))))
    scope_path=H/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
    routes_path=H/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
    if sha(scope_path)!="1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676":
        raise ValueError("protected back scope changed")
    if sha(routes_path)!="ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71":
        raise ValueError("protected route file changed")
    with np.load(scope_path,allow_pickle=False) as z:
        protected=set(map(int,np.asarray(z["O_vertex_rows"],np.int64)))
    if any(row<0 or row>=len(o_points) for row in protected):
        raise ValueError("protected O scope row is outside the source point domain")
    edge_rows=np.sort(np.concatenate((o_faces[:,[0,1]],o_faces[:,[1,2]],o_faces[:,[2,0]])),axis=1)
    unique_edges,counts=np.unique(edge_rows,axis=0,return_counts=True)
    protected.update(map(int,unique_edges[counts==1].reshape(-1)))
    routes=json.loads(routes_path.read_text()).get("surface_routes",[])
    for route in routes:
        if route.get("role")!="O":
            continue
        hit=np.flatnonzero(np.all(o_points==np.asarray(route["target_world_m_f32"],np.float32),axis=1))
        if len(hit)!=1:
            raise ValueError("O149 route does not identify one baseline source point")
        protected.add(int(hit[0]))
    allowed=candidate_local-protected
    receipt={
        "policy":"bad baseline O corners plus direct incident O triangles",
        "baseline_bad_tets":int(np.count_nonzero(~passed)),
        "bad_corner_global_rows":len(bad),
        "O_bad_corner_seed_rows":int(len(seed_local)),
        "incident_O_source_faces":int(np.count_nonzero(incident)),
        "one_ring_candidate_local_rows":len(candidate_local),
        "protected_scope_rim_route_rows":len(protected),
        "allowed_O_local_rows":len(allowed),
        "source_faces_unchanged":True,
    }
    return allowed,protected,receipt


def canonical_oriented_faces(faces: np.ndarray, label: str) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 1] == f[:, 2]) or np.any(f[:, 2] == f[:, 0]):
        raise ValueError(label + ": malformed/degenerate faces")
    ix = np.argmin(f, axis=1)
    rows = np.arange(len(f))
    c = np.stack((f[rows, ix], f[rows, (ix + 1) % 3], f[rows, (ix + 2) % 3]), axis=1)
    if len(c) > 1:
        order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
        c = c[order]
    if len(c) > 1 and np.any(np.all(c[1:] == c[:-1], axis=1)):
        raise ValueError(label + ": duplicate oriented triangles")
    return c


def canonical_unoriented_faces(faces: np.ndarray, label: str) -> np.ndarray:
    f = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    if len(f) > 1:
        order = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
        f = f[order]
    if len(f) > 1 and np.any(np.all(f[1:] == f[:-1], axis=1)):
        raise ValueError(label + ": duplicate unoriented triangles")
    return f


def run_command(label: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (log_dir / (label + ".stdout.log")).write_text(proc.stdout)
    (log_dir / (label + ".stderr.log")).write_text(proc.stderr)
    (log_dir / (label + ".command.json")).write_text(json.dumps({
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_s": None,
    }, indent=2) + "\n")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        raise RuntimeError(f"{label} exited {proc.returncode}: {tail}")


def run_audit(config: Path, out: Path, log_dir: Path) -> Path:
    run_command("audit805", [sys.executable, str(AUDITOR), "--config", str(config), "--out", str(out)], log_dir)
    report = out / "report.json"
    if not report.is_file():
        raise RuntimeError("805 auditor returned without a report")
    return report


def audit_counts(report_path: Path) -> tuple[dict, bool]:
    report = json.loads(report_path.read_text())
    against = report.get("changed_vs_complete_retained_OID_both_walls_all185", {})
    self_pairs = report.get("changed_OID_joint_self_pairs", {})
    n_against = int(against.get("count", -1))
    n_self = int(self_pairs.get("count", -1))
    if n_against < 0 or n_self < 0:
        raise ValueError("805 exact pair counts are missing")
    zero = report.get("status") == ZERO_AUDIT_STATUS and n_against == 0 and n_self == 0
    return {"status": report.get("status"), "context_pairs": n_against, "joint_self_pairs": n_self}, zero


def signed_volume_by_material(points: np.ndarray, tets: np.ndarray, materials: np.ndarray) -> dict[int, float]:
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)
    m = np.asarray(materials, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 4 or len(m) != len(t) or (len(t) and (t.min() < 0 or t.max() >= len(p))):
        raise ValueError("malformed volume cell/material arrays")
    result = {int(x): 0.0 for x in np.unique(m)}
    for start in range(0, len(t), 150_000):
        stop = min(len(t), start + 150_000)
        q = p[t[start:stop]]
        a = q[:, 1] - q[:, 0]
        b = q[:, 2] - q[:, 0]
        c = q[:, 3] - q[:, 0]
        vol = np.einsum("ij,ij->i", a, np.cross(b, c), optimize=True) / 6.0
        mm = m[start:stop]
        for material in result:
            mask = mm == material
            if np.any(mask):
                result[material] += float(vol[mask].sum(dtype=np.float64))
    return result


def independent_replay(
    candidate_state_path: Path,
    parent_state_path: Path,
    output_report: Path,
    *,
    proposal_state_path: Path | None,
    rollback_plan_path: Path | None,
) -> dict:
    """Replay typed O/I/D to volume F32 maps, boundaries, quality and mass deltas."""
    state = load_state(candidate_state_path)
    parent = load_state(parent_state_path)
    proposal = load_state(proposal_state_path) if proposal_state_path else None
    owner_spec = importlib.util.spec_from_file_location("quality_owner_708", QUALITY_OWNER)
    if owner_spec is None or owner_spec.loader is None:
        raise RuntimeError("cannot import native quality owner")
    owner = importlib.util.module_from_spec(owner_spec)
    sys.modules["quality_owner_708"] = owner
    owner_spec.loader.exec_module(owner)

    surface_cache = {}
    source_receipts = {}
    repair_meta = state.get("coupled_point_repair", {})
    trust_cap = float(repair_meta.get("maximum_displacement_m", 2e-5))
    o_local_enabled = repair_meta.get("ordinary_outer_O_local_repair") is True
    if o_local_enabled:
        if repair_meta.get("ordinary_outer_O_policy") != "bad O corners plus one direct incident O source-triangle ring; scope/rims/routes remain fixed":
            raise ValueError("state lacks the declared ordinary-O local policy")
        allowed_o_rows, protected_o_rows, o_scope_receipt = ordinary_o_allowed_source_rows(parent_state_path, owner)
    else:
        allowed_o_rows, protected_o_rows, o_scope_receipt = None, set(), None
    for role in ROLE_ORDER:
        parent_entry = parent["source_surfaces"][role]
        candidate_entry = state["source_surfaces"][role]
        if sha(Path(parent_entry["path"])) != parent_entry["sha256"] or sha(Path(candidate_entry["path"])) != candidate_entry["sha256"]:
            raise ValueError(role + ": source NPZ hash mismatch")
        base = load_source_surface(Path(parent_entry["path"]))
        cand = load_source_surface(Path(candidate_entry["path"]))
        bp, cp = source_f32points(base, role + " parent"), source_f32points(cand, role + " candidate")
        bf, cf = np.asarray(base["faces"], np.int64), np.asarray(cand["faces"], np.int64)
        if bp.shape != cp.shape or not np.array_equal(bf, cf):
            raise ValueError(role + ": source point domain/topology changed")
        if bf.size and (bf.min() < 0 or bf.max() >= len(cp)):
            raise ValueError(role + ": source face index outside point domain")
        canonical_unoriented_faces(bf, role + " parent surface")
        canonical_unoriented_faces(cf, role + " candidate surface")
        active = np.unique(cf.reshape(-1))
        unique_point_map(cp, role + " source point domain")
        active_map = unique_point_map(cp[active], role + " active source surface")
        moved = np.flatnonzero(np.any(bp.view(np.uint32) != cp.view(np.uint32), axis=1))
        l1 = np.abs(cp[moved].astype(np.float64) - bp[moved].astype(np.float64)).sum(axis=1)
        max_l1 = float(l1.max(initial=0.0))
        if max_l1 > trust_cap:
            raise ValueError(role + ": source point step exceeds the recorded L1 trust cap")
        if role == "O" and o_local_enabled:
            moved_set = set(map(int, moved))
            if moved_set & protected_o_rows:
                raise ValueError("ordinary O point fit changed a protected scope, rim, or route row")
            if not moved_set.issubset(allowed_o_rows):
                raise ValueError("ordinary O point fit moved a source row outside bad-corner plus one-ring scope")
        if len(moved):
            with np.load(candidate_entry["path"], allow_pickle=False) as z:
                if "typed_parent_candidate_path" not in z or "typed_parent_candidate_sha256" not in z:
                    raise ValueError(role + ": missing typed source parent pin")
                if archive_text_scalar(z,"typed_parent_candidate_path") != parent_entry["path"] or archive_text_scalar(z,"typed_parent_candidate_sha256") != parent_entry["sha256"]:
                    raise ValueError(role + ": typed source parent pin mismatch")
                if not np.array_equal(z["source_input_face_rows"], np.arange(len(cf), dtype=np.int64)):
                    raise ValueError(role + ": typed source face rows are not the exact parent identity")
                if not np.array_equal(z["source_input_vertex_quotient"], np.arange(len(cp), dtype=np.int64)):
                    raise ValueError(role + ": source vertex quotient is not the exact parent identity")
                for key in z.files:
                    if key.startswith("source_") and key.endswith("_operation_rows"):
                        rows = np.asarray(z[key])
                        if rows.shape != (len(cf),) or np.any(rows != -1):
                            raise ValueError(role + ": point-only step contains stale source operation rows")
        surface_cache[role] = (base, cand, bp, cp, bf, moved, active)
        source_receipts[role] = {
            "parent": parent_entry, "candidate": candidate_entry,
            "active_point_count": int(len(active)), "moved_point_count": int(len(moved)),
            "maximum_L1_displacement_m": max_l1,
            "source_topology_unchanged": True, "active_source_coordinates_unique": True,
        }

    global_active = {}
    for role in ROLE_ORDER:
        _, _, _, cp, _, _, active = surface_cache[role]
        for row in active:
            key = np.ascontiguousarray(cp[int(row)], dtype="<f4").tobytes()
            if key in global_active:
                raise ValueError(f"active source cross-role coordinate alias: {global_active[key]} and {role}/{int(row)}")
            global_active[key] = f"{role}/{int(row)}"

    rollback_receipt = None
    if rollback_plan_path:
        if proposal is None:
            raise ValueError("rollback replay requires the original 705 proposal state")
        plan = json.loads(rollback_plan_path.read_text())
        surface_rollback = plan["rollback_point_rows_by_role"]
        surface_safe = plan["retained_safe_point_rows_by_role"]
        for role in ROLE_ORDER:
            base_entry = parent["source_surfaces"][role]
            prop_entry = proposal["source_surfaces"][role]
            final_entry = state["source_surfaces"][role]
            base = load_selected(Path(base_entry["path"]), ("points_world_m_f32",))
            prop = load_selected(Path(prop_entry["path"]), ("points_world_m_f32",))
            final = load_selected(Path(final_entry["path"]), ("points_world_m_f32",))
            b, p, f = (np.asarray(x["points_world_m_f32"]) for x in (base, prop, final))
            r = np.asarray(surface_rollback[role], np.int64)
            s = np.asarray(surface_safe[role], np.int64)
            if len(r) and not np.array_equal(f[r].view(np.uint32), b[r].view(np.uint32)):
                raise ValueError(role + ": witness component was not fully restored to parent coordinates")
            if len(s) and not np.array_equal(f[s].view(np.uint32), p[s].view(np.uint32)):
                raise ValueError(role + ": safe coupled source coordinates were not preserved")
        rollback_receipt = {
            "rollback_plan": pin(rollback_plan_path),
            "all_role_witness_component_rows_restored": True,
            "all_safe_source_rows_preserved": True,
        }

    replay_volumes = {}
    mass_pending = {}
    for vrole, source_roles in VOLUME_ROLES.items():
        base_entry = parent[vrole + "_current"]
        candidate_entry = state[vrole + "_current"]
        if sha(Path(base_entry["path"])) != base_entry["sha256"] or sha(Path(candidate_entry["path"])) != candidate_entry["sha256"]:
            raise ValueError(vrole + ": volume NPZ hash mismatch")
        fields = ("points_world_m_f32", "points_world_m", "tetrahedra", "material_indices", "boundary_faces", "desired_boundary_faces")
        base = load_selected(Path(base_entry["path"]), fields)
        cand = load_selected(Path(candidate_entry["path"]), fields)
        bp, cp = f32points(base, vrole + " parent volume"), f32points(cand, vrole + " candidate volume")
        bt, ct = np.asarray(base["tetrahedra"], np.int64), np.asarray(cand["tetrahedra"], np.int64)
        bm, cm = np.asarray(base["material_indices"]), np.asarray(cand["material_indices"])
        if bp.shape != cp.shape or not np.array_equal(bt, ct) or not np.array_equal(bm, cm):
            raise ValueError(vrole + ": volume point/tet/material domains changed")
        for key in ("boundary_faces", "desired_boundary_faces"):
            if not np.array_equal(base[key], cand[key]):
                raise ValueError(vrole + ": volume boundary face identity changed")
        # The candidate all-role point step is explicitly identity-mapped.
        with np.load(candidate_entry["path"], allow_pickle=False) as z:
            for key, size in (("output_point_to_input_point_row", len(cp)), ("output_cell_to_input_cell", len(ct)), ("input_cell_to_output_cell", len(ct)), ("retained_input_cell_rows", len(ct)), ("input_node_to_output_node", len(cp)), ("output_node_to_input_node", len(cp))):
                if key not in z or not np.array_equal(np.asarray(z[key], np.int64), np.arange(size, dtype=np.int64)):
                    raise ValueError(vrole + ": volume typed identity map failed: " + key)
            for key in ("current_geometry_parent_path", "current_geometry_parent_sha256"):
                if key not in z:
                    raise ValueError(vrole + ": missing typed volume parent pin: " + key)
            if archive_text_scalar(z,"current_geometry_parent_path") != base_entry["path"] or archive_text_scalar(z,"current_geometry_parent_sha256") != base_entry["sha256"]:
                raise ValueError(vrole + ": typed volume parent pin mismatch")
            for key in ("output_point_parent_offsets", "output_point_parent_input_cell_rows", "replacement_parent_offsets", "replacement_parent_cell_rows", "replacement_tet_offsets", "replacement_tetrahedra"):
                if key not in z:
                    raise ValueError(vrole + ": missing volume parent-union field: " + key)
            if not np.array_equal(np.asarray(z["output_point_parent_offsets"], np.int64), np.zeros(len(cp) + 1, np.int64)):
                raise ValueError(vrole + ": point parent-union unexpectedly changes")
            if np.asarray(z["output_point_parent_input_cell_rows"]).size or np.asarray(z["replacement_parent_cell_rows"]).size or np.asarray(z["replacement_tetrahedra"]).size:
                raise ValueError(vrole + ": point-only update contains replacement parent cells")
        actual = canonical_oriented_faces(cand["boundary_faces"], vrole + " actual boundary")
        desired = canonical_oriented_faces(cand["desired_boundary_faces"], vrole + " desired boundary")
        canonical_unoriented_faces(cand["boundary_faces"], vrole + " actual boundary")
        canonical_unoriented_faces(cand["desired_boundary_faces"], vrole + " desired boundary")
        if not np.array_equal(actual, desired):
            raise ValueError(vrole + ": oriented actual and desired volume boundaries differ")
        actual_f = np.asarray(cand["boundary_faces"], np.int64)
        if actual_f.size and (actual_f.min() < 0 or actual_f.max() >= len(cp)):
            raise ValueError(vrole + ": boundary face index outside point domain")
        volume_map = unique_point_map(cp, vrole + " volume")
        for role in source_roles:
            _, _, _, sp, sf, _, active = surface_cache[role]
            keys = point_keys(sp[active])
            rows = [volume_map.get(k) for k in keys]
            if any(r is None for r in rows):
                raise ValueError(role + "/" + vrole + ": active source F32 point missing from candidate volume")
            if len(rows) != len(set(rows)):
                raise ValueError(role + "/" + vrole + ": active source point map aliases a volume row")
        parent_volume_map = unique_point_map(bp, vrole + " parent volume")
        baseline_vol = signed_volume_by_material(bp, bt, bm)
        candidate_vol = signed_volume_by_material(cp, ct, cm)
        d, q = owner.measurements(cp, ct)
        passed = np.asarray(owner.passes(d, q), bool)
        if np.any(np.asarray(d) <= 0):
            raise ValueError(vrole + ": nonpositive Float32 determinant")
        bd, bq = owner.measurements(bp, bt)
        baseline_passed = np.asarray(owner.passes(bd, bq), bool)
        regressions = np.flatnonzero(baseline_passed & ~passed)
        if len(regressions):
            raise ValueError(vrole + ": candidate regresses baseline-native-good cells")
        per_material = {}
        for material in np.unique(cm):
            mask = cm == material
            before = baseline_vol[int(material)]
            after = candidate_vol[int(material)]
            delta = after - before
            per_material[str(int(material))] = {
                "cells": int(np.count_nonzero(mask)),
                "baseline_signed_volume_m3": before,
                "candidate_signed_volume_m3": after,
                "delta_signed_volume_m3": delta,
                "relative_delta": (delta / before if before != 0 else None),
                "mass_status": "PENDING_VOLUME_CHANGE_RECONCILIATION",
                "mass_kg_claim": None,
            }
        replay_volumes[vrole] = {
            "cells": int(len(ct)), "points": int(len(cp)),
            "boundary_faces": int(len(actual_f)), "oriented_actual_equals_desired": True,
            "unique_volume_F32_points": True,
            "baseline_native_failures": int(np.count_nonzero(~baseline_passed)),
            "candidate_native_failures": int(np.count_nonzero(~passed)),
            "baseline_good_regressions": 0, "all_F32_cell_determinants_positive": True,
            "per_material": per_material,
        }
        if rollback_plan_path:
            plan = json.loads(rollback_plan_path.read_text())
            rb_rows = np.asarray(plan.get("rollback_volume_point_rows_by_role", {}).get(vrole, []), np.int64)
            proposal_entry = proposal[vrole + "_current"]
            prop = load_selected(Path(proposal_entry["path"]), ("points_world_m_f32",))
            pp = np.asarray(prop["points_world_m_f32"])
            if len(rb_rows) and not np.array_equal(cp[rb_rows].view(np.uint32), bp[rb_rows].view(np.uint32)):
                raise ValueError(vrole + ": witness volume component was not restored")
            safe_rows = np.setdiff1d(np.arange(len(cp), dtype=np.int64), rb_rows, assume_unique=True)
            if len(safe_rows) and not np.array_equal(cp[safe_rows].view(np.uint32), pp[safe_rows].view(np.uint32)):
                raise ValueError(vrole + ": safe volume coordinates were not preserved")
            rollback_receipt[vrole + "_rollback_volume_points"] = int(len(rb_rows))

    if proposal is not None and rollback_plan_path:
        # Replay the rollback volume coordinate rows against the saved 705 trial.
        for vrole in ("Skin", "Fat"):
            plan = json.loads(rollback_plan_path.read_text())
            rb = np.asarray(plan.get("rollback_volume_point_rows_by_role", {}).get(vrole, []), np.int64)
            pentry = proposal[vrole + "_current"]
            fentry = state[vrole + "_current"]
            prop = load_selected(Path(pentry["path"]), ("points_world_m_f32",))
            final = load_selected(Path(fentry["path"]), ("points_world_m_f32",))
            if len(rb) and not np.array_equal(np.asarray(final["points_world_m_f32"])[rb].view(np.uint32), np.asarray(load_selected(Path(parent[vrole + "_current"]["path"]), ("points_world_m_f32",))["points_world_m_f32"])[rb].view(np.uint32)):
                raise ValueError(vrole + ": full coupled volume rollback failed")
        rollback_receipt["both_volume_layers_restored_as_one_coupled_component"] = True

    receipt = {
        "schema": "h706.coupled-quality-708-independent-point-replay.v1",
        "status": "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY",
        "candidate_state": pin(candidate_state_path), "parent_state": pin(parent_state_path),
        "quality_owner": checked_pin("quality_owner", QUALITY_OWNER),
        "source_roles": source_receipts, "ordinary_outer_O_local_scope": o_scope_receipt, "volumes": replay_volumes,
        "witness_component_rollback": rollback_receipt,
        "mass_reconciliation": {
            "status": "PENDING_VOLUME_CHANGE_RECONCILIATION",
            "per_material_volume_changes_reported": True,
            "mass_conservation_claimed": False,
            "note": "Per-material geometric volume changes are measured; material mass reconciliation remains pending.",
        },
        "physical_admission": False, "native_steps": 0,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def finalize_verified_state(source_state_path: Path, exact_report: Path, replay_report: Path, output: Path) -> dict:
    state = load_state(source_state_path)
    report = json.loads(exact_report.read_text())
    replay = json.loads(replay_report.read_text())
    if report.get("status") != ZERO_AUDIT_STATUS:
        raise ValueError("cannot finalize a nonzero exact 805 report")
    if replay.get("status") != "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise ValueError("independent replay did not pass")
    final = dict(state)
    final.update({
        "surface_crossings": 0,
        "exact_context": pin(exact_report),
        "source_audit_status": "PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",
        "independent_point_replay": pin(replay_report),
        "mass_reconciliation": replay["mass_reconciliation"],
        "physical_admission": False,
        "native_steps": 0,
    })
    for vrole in ("Skin", "Fat"):
        v = dict(final[vrole + "_current"])
        v.update({"missing_target_faces": 0, "extra_actual_faces": 0, "oriented_volume_target_boundary_equal": True})
        final[vrole + "_current"] = v
    output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final



def validate_report_config_pin(report_path: Path, config_path: Path) -> dict:
    report_path = Path(report_path).resolve()
    config_path = Path(config_path).resolve()
    if not report_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("resume audit report or config is missing")
    report = json.loads(report_path.read_text())
    config_pin = report.get("config")
    if not isinstance(config_pin, dict):
        raise ValueError("resume audit report has no config pin")
    if Path(config_pin.get("path", "")).resolve() != config_path:
        raise ValueError("resume audit report config path does not match the proposal config")
    config_sha = sha(config_path)
    if config_pin.get("sha256") != config_sha:
        raise ValueError("resume audit report config SHA does not match the proposal config")
    return report


def validate_proposal_parent_pin(proposal_state: dict, baseline_state_path: Path) -> dict:
    expected = pin(Path(baseline_state_path).resolve())
    actual = proposal_state.get("source_state")
    if not isinstance(actual, dict):
        raise ValueError("saved proposal has no source_state parent pin")
    if Path(actual.get("path", "")).resolve() != Path(expected["path"]):
        raise ValueError("saved proposal parent path does not match the requested baseline state")
    if actual.get("sha256") != expected["sha256"]:
        raise ValueError("saved proposal parent SHA does not match the requested baseline state")
    return expected


def resume_saved_proposal(
    baseline_state_path: Path,
    proposal_state_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    supplied_audit_report: Path | None,
) -> dict:
    """Finish a saved proposal without rerunning its SLP producer."""
    baseline_state_path = Path(baseline_state_path).resolve()
    proposal_state_path = Path(proposal_state_path).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not all(p.is_file() for p in (baseline_state_path, proposal_state_path, config_path)):
        raise FileNotFoundError("resume baseline, saved proposal, or context config is missing")

    implementation_pins = {}
    for label, path in (("builder", BUILDER), ("auditor", AUDITOR), ("preparer", PREPARER),
                        ("rollback_helper", ROLLBACK_HELPER), ("finalizer", FINALIZER),
                        ("base_driver", BASE_DRIVER), ("quality_lp", QUALITY_LP),
                        ("transfer_owner", TRANSFER_OWNER), ("quality_owner", QUALITY_OWNER),
                        ("source_verifier", SOURCE_VERIFIER)):
        implementation_pins[label] = checked_pin(label, path)

    source_evidence = verify_source_input(baseline_state_path)
    baseline = load_state(baseline_state_path)
    proposal = load_state(proposal_state_path)
    parent_pin = validate_proposal_parent_pin(proposal, baseline_state_path)
    config_doc = json.loads(config_path.read_text())
    if config_doc.get("schema") != "h706.combined-current-exact-surface-diff-config.v1":
        raise ValueError("saved proposal config schema mismatch")
    for role in ROLE_ORDER:
        b = config_doc.get("baseline_candidates", {}).get(role, {})
        c = config_doc.get("candidate_surfaces", {}).get(role, {})
        if Path(b.get("path", "")).resolve() != Path(baseline["source_surfaces"][role]["path"]).resolve() or b.get("sha256") != baseline["source_surfaces"][role]["sha256"]:
            raise ValueError(role + ": resume config baseline source pin differs from requested baseline")
        if Path(c.get("path", "")).resolve() != Path(proposal["source_surfaces"][role]["path"]).resolve() or c.get("sha256") != proposal["source_surfaces"][role]["sha256"]:
            raise ValueError(role + ": resume config candidate pin differs from saved proposal")
    recorded_cap = float(proposal.get("coupled_point_repair", {}).get("maximum_displacement_m", 0.0))
    if not 0.0 < recorded_cap <= 5e-5:
        raise ValueError("saved INNER_BACK_LAYER_REPAIR trust cap exceeds 50 micrometres")
    repair_meta=proposal.get("coupled_point_repair", {})
    if repair_meta.get("inner_back_layer_repair") is not True or repair_meta.get("inner_back_layer_policy") != "INNER_BACK_LAYER_REPAIR":
        raise ValueError("saved proposal lacks explicit INNER_BACK_LAYER_REPAIR metadata")
    if repair_meta.get("ordinary_outer_O_local_repair") is not True:
        raise ValueError("saved proposal lacks explicit ordinary-O local policy")
    declared=set(config_doc.get("changed_roles", []))
    if not declared or not declared.issubset({"O","I","D"}):
        raise ValueError("v24 proposal may declare only changed O/I/D roles")
    for role in ROLE_ORDER:
        entry=config_doc.get("bounded_point_repair", {}).get(role)
        if role in declared:
            if not isinstance(entry,dict):
                raise ValueError(role + ": bounded role policy is missing")
            cfg_cap=float(entry.get("maximum_displacement_m",-1.0))
            if cfg_cap != recorded_cap:raise ValueError(role + ": saved config and proposal trust caps differ")
            if role=="O":
                if entry.get("ordinary_outer_o_local_repair") is not True or entry.get("inner_back_layer_repair",False):
                    raise ValueError("O entry must declare ordinary local policy only")
            elif role in ("I","D") and entry.get("inner_back_layer_repair") is not True:
                raise ValueError(role + ": explicit inner-back flag is missing")
        elif entry is not None:
            raise ValueError(role + ": undeclared role has bounded repair metadata")

    output_dir.mkdir(parents=True)
    if supplied_audit_report is None:
        proposal_audit_dir = output_dir / "proposal-exact-805"
        proposal_report = run_audit(config_path, proposal_audit_dir, output_dir / "proposal-audit-logs")
        audit_origin = "executed_805_against_saved_774_config"
    else:
        proposal_report = Path(supplied_audit_report).resolve()
        validate_report_config_pin(proposal_report, config_path)
        audit_origin = "reused_pinned_805_report_for_saved_774_config"
    proposal_summary, proposal_clean = audit_counts(proposal_report)

    rollback_dir = None
    rollback_plan = None
    final_state_input = proposal_state_path
    final_exact_report = proposal_report
    rollback_summary = None
    if not proposal_clean:
        rollback_dir = output_dir / "witness-component-rollback"
        run_command("rollback625", [sys.executable, str(PREPARER), "--state", str(proposal_state_path),
                    "--config", str(config_path), "--audit", str(proposal_report),
                    "--output", str(rollback_dir)], output_dir / "rollback-logs")
        rollback_state = rollback_dir / "state.json"
        rollback_config = rollback_dir / "context-config.json"
        rollback_plan = rollback_dir / "rollback-plan.json"
        prep_receipt = rollback_dir / "rollback-preparation-receipt.json"
        if not all(p.is_file() for p in (rollback_state, rollback_config, rollback_plan, prep_receipt)):
            raise RuntimeError("625 did not produce complete whole-component rollback outputs")
        rollback_audit_dir = output_dir / "rollback-exact-805"
        final_exact_report = run_audit(rollback_config, rollback_audit_dir, output_dir / "rollback-audit-logs")
        rollback_summary, rollback_clean = audit_counts(final_exact_report)
        validate_report_config_pin(final_exact_report, rollback_config)
        if not rollback_clean:
            raise RuntimeError("805 still reports exact witnesses after 625 rollback: " + json.dumps(rollback_summary, sort_keys=True))
        finalized_dir = output_dir / "finalized"
        run_command("finalize625", [sys.executable, str(FINALIZER), "--state", str(rollback_state),
                    "--baseline-state", str(baseline_state_path), "--proposal-state", str(proposal_state_path),
                    "--rollback-receipt", str(prep_receipt), "--audit-report", str(final_exact_report),
                    "--audit-config", str(rollback_config), "--output-dir", str(finalized_dir)],
                    output_dir / "finalize-logs")
        final_state_input = finalized_dir / "state.json"
        if not final_state_input.is_file():
            raise RuntimeError("625 finalizer did not produce a complete rollback state")

    replay_report = output_dir / "independent-point-replay.json"
    replay = independent_replay(final_state_input, baseline_state_path, replay_report,
                                proposal_state_path=proposal_state_path if rollback_plan else None,
                                rollback_plan_path=rollback_plan)
    if replay.get("status") != "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise RuntimeError("saved proposal independent point replay did not pass")
    verified_state_path = output_dir / "verified-state.json"
    finalize_verified_state(final_state_input, final_exact_report, replay_report, verified_state_path)

    receipt = {
        "schema": "h706.saved-v24-ordinary-O-inner-ID-proposal-resume-805-v1",
        "input_source_evidence": source_evidence,
        "status": "PASS_805_EXACT_CONTEXT_AND_INDEPENDENT_REPLAY",
        "baseline_state": parent_pin,
        "saved_proposal_state": pin(proposal_state_path),
        "saved_proposal_config": pin(config_path),
        "proposal_trust_cap_m": recorded_cap,
        "proposal_805_report": pin(proposal_report),
        "proposal_805_origin": audit_origin,
        "proposal_805_summary": proposal_summary,
        "proposal_805_clean": proposal_clean,
        "rollback_state": pin(rollback_dir / "state.json") if rollback_dir else None,
        "rollback_plan": pin(rollback_plan) if rollback_plan else None,
        "rollback_805_summary": rollback_summary,
        "final_exact_805_report": pin(final_exact_report),
        "independent_point_replay": pin(replay_report),
        "verified_state": pin(verified_state_path),
        "implementation_pins": implementation_pins,
        "physical_admission": False,
        "native_steps": 0,
        "mass_reconciliation": replay["mass_reconciliation"],
        "limitations": [
            "The saved SLP proposal was reused; no optimization or geometry producer was rerun.",
            "Physical admission and native qualification were not performed."
        ],
    }
    receipt_path = output_dir / "resume-805-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _validate_context_config_endpoints(config_path: Path, baseline_path: Path, candidate_path: Path) -> dict:
    baseline = load_state(baseline_path)
    candidate = load_state(candidate_path)
    config = json.loads(Path(config_path).read_text())
    if config.get("schema") != "h706.combined-current-exact-surface-diff-config.v1":
        raise ValueError("existing replay context config schema mismatch")
    declared = set(config.get("changed_roles", []))
    if not declared or not declared.issubset({"O","I","D"}):
        raise ValueError("existing replay config must declare changed O/I/D roles")
    for role in ROLE_ORDER:
        b = config.get("baseline_candidates", {}).get(role, {})
        c = config.get("candidate_surfaces", {}).get(role, {})
        be = baseline["source_surfaces"][role]
        ce = candidate["source_surfaces"][role]
        if Path(b.get("path","")).resolve() != Path(be["path"]).resolve() or b.get("sha256") != be["sha256"]:
            raise ValueError(role + ": config baseline pin mismatch")
        if Path(c.get("path","")).resolve() != Path(ce["path"]).resolve() or c.get("sha256") != ce["sha256"]:
            raise ValueError(role + ": config candidate pin mismatch")
        entry = config.get("bounded_point_repair",{}).get(role)
        if role in declared:
            if not isinstance(entry,dict):
                raise ValueError(role + ": declared bounded point policy missing")
            cap=float(entry.get("maximum_displacement_m",-1.0))
            if not 0.0 < cap <= 5e-5:
                raise ValueError(role + ": bounded point cap exceeds v24 construction limit")
            if role=="O" and entry.get("ordinary_outer_o_local_repair") is not True:
                raise ValueError("O config lacks ordinary local policy")
            if role in ("I","D") and entry.get("inner_back_layer_repair") is not True:
                raise ValueError(role + ": config lacks inner-back policy")
        elif entry is not None:
            raise ValueError(role + ": undeclared role has a bounded point policy")
    return {"schema":config["schema"],"changed_roles":sorted(declared),"config":pin(config_path)}


def _require_receipt_pin(receipt: dict, key: str, path: Path) -> None:
    entry=receipt.get(key)
    if not isinstance(entry,dict):
        raise ValueError("rollback preparation receipt is missing "+key)
    expected=Path(path).resolve()
    if Path(entry.get("path","")).resolve()!=expected or entry.get("sha256")!=sha(expected):
        raise ValueError("rollback preparation receipt pin mismatch for "+key)


def replay_existing_rollback(
    baseline_state_path: Path,
    proposal_state_path: Path,
    proposal_config_path: Path,
    proposal_audit_path: Path,
    rollback_state_path: Path,
    rollback_config_path: Path,
    rollback_plan_path: Path,
    rollback_receipt_path: Path,
    rollback_audit_path: Path,
    output_dir: Path,
    *,
    preflight_only: bool = False,
) -> dict:
    """Promote an already audited 625 rollback using replay only; never reruns solver or auditor."""
    paths=[Path(x).resolve() for x in (
        baseline_state_path,proposal_state_path,proposal_config_path,proposal_audit_path,
        rollback_state_path,rollback_config_path,rollback_plan_path,rollback_receipt_path,
        rollback_audit_path)]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("existing rollback recovery input is missing")
    output_dir=Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    baseline_state_path,proposal_state_path,proposal_config_path,proposal_audit_path,rollback_state_path,rollback_config_path,rollback_plan_path,rollback_receipt_path,rollback_audit_path=paths
    implementation_pins={}
    for label,path in (("builder",BUILDER),("auditor",AUDITOR),("preparer",PREPARER),
                       ("rollback_helper",ROLLBACK_HELPER),("finalizer",FINALIZER),
                       ("base_driver",BASE_DRIVER),("quality_lp",QUALITY_LP),
                       ("transfer_owner",TRANSFER_OWNER),("quality_owner",QUALITY_OWNER),
                       ("source_verifier",SOURCE_VERIFIER)):
        implementation_pins[label]=checked_pin(label,path)
    source_evidence=verify_source_input(baseline_state_path)
    proposal=load_state(proposal_state_path)
    rollback=load_state(rollback_state_path)
    parent_pin=validate_proposal_parent_pin(proposal,baseline_state_path)
    validate_proposal_parent_pin(rollback,baseline_state_path)
    proposal_config_receipt=_validate_context_config_endpoints(proposal_config_path,baseline_state_path,proposal_state_path)
    rollback_config_receipt=_validate_context_config_endpoints(rollback_config_path,baseline_state_path,rollback_state_path)
    proposal_report=validate_report_config_pin(proposal_audit_path,proposal_config_path)
    proposal_summary,proposal_clean=audit_counts(proposal_audit_path)
    if proposal_clean:
        raise ValueError("recovery proposal report does not contain the failed exact-witness audit that triggered rollback")
    rollback_report=validate_report_config_pin(rollback_audit_path,rollback_config_path)
    rollback_summary,rollback_clean=audit_counts(rollback_audit_path)
    if not rollback_clean:
        raise ValueError("pinned existing rollback exact audit is not clean")
    plan=json.loads(rollback_plan_path.read_text())
    if plan.get("status")!="PASS_EXACT_WITNESS_COMPONENT_MAPPING":
        raise ValueError("existing rollback plan does not pass exact component mapping")
    prep=json.loads(rollback_receipt_path.read_text())
    if prep.get("status")!="PASS_WITNESS_COMPONENT_ROLLBACK_PREPARED_FOR_EXACT_AUDIT":
        raise ValueError("existing 625 preparation receipt status mismatch")
    input_pins=(
        ("source_baseline_state",baseline_state_path),("state",proposal_state_path),
        ("config",proposal_config_path),("failed_805_audit",proposal_audit_path),
        ("helper",ROLLBACK_HELPER))
    output_pins=(("output_state",rollback_state_path),("output_config",rollback_config_path),
                 ("rollback_plan",rollback_plan_path))
    for key,path in input_pins:
        _require_receipt_pin(prep.get("inputs",{}),key,path)
    for key,path in output_pins:
        _require_receipt_pin(prep,key,path)
    for role in ROLE_ORDER:
        recorded=prep.get("candidate_surfaces",{}).get(role,{})
        actual=rollback["source_surfaces"][role]
        if Path(recorded.get("path","")).resolve()!=Path(actual["path"]).resolve() or recorded.get("sha256")!=actual["sha256"]:
            raise ValueError(role+": rollback receipt candidate source pin mismatch")
    for vrole in ("Skin","Fat"):
        recorded=prep.get("candidate_volumes",{}).get(vrole,{})
        actual=rollback[vrole+"_current"]
        if Path(recorded.get("path","")).resolve()!=Path(actual["path"]).resolve() or recorded.get("sha256")!=actual["sha256"]:
            raise ValueError(vrole+": rollback receipt candidate volume pin mismatch")
    if preflight_only:
        preflight_receipt={
            "schema":"h706.v24d.existing-625-rollback-preflight-only.v1",
            "status":"PASS_EXISTING_805_ROLLBACK_INPUTS_READY_FOR_INDEPENDENT_REPLAY",
            "baseline_state":parent_pin,
            "proposal_state":pin(proposal_state_path),
            "proposal_config":proposal_config_receipt,
            "proposal_805":proposal_summary,
            "rollback_state":pin(rollback_state_path),
            "rollback_config":rollback_config_receipt,
            "rollback_plan":pin(rollback_plan_path),
            "rollback_preparation_receipt":pin(rollback_receipt_path),
            "rollback_805":rollback_summary,
            "input_source_evidence_status":source_evidence.get("status"),
            "auditor_rerun":False,"solver_rerun":False,"independent_replay_run":False,
        }
        print(json.dumps(preflight_receipt,indent=2,sort_keys=True))
        return preflight_receipt
    output_dir.mkdir(parents=True)
    replay_path=output_dir/"independent-point-replay.json"
    replay=independent_replay(rollback_state_path,baseline_state_path,replay_path,
                              proposal_state_path=proposal_state_path,rollback_plan_path=rollback_plan_path)
    if replay.get("status")!="PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise RuntimeError("independent rollback readback did not pass")
    verified_path=output_dir/"verified-state.json"
    finalize_verified_state(rollback_state_path,rollback_audit_path,replay_path,verified_path)
    final_source_evidence=verify_source_input(verified_path)
    receipt={
        "schema":"h706.v24c.existing-625-rollback-independent-replay-only.v1",
        "status":"PASS_EXISTING_805_ROLLBACK_AND_INDEPENDENT_REPLAY",
        "baseline_state":parent_pin,
        "proposal_state":pin(proposal_state_path),
        "proposal_config":proposal_config_receipt,
        "failed_proposal_audit":pin(proposal_audit_path),
        "failed_proposal_summary":proposal_summary,
        "rollback_state":pin(rollback_state_path),
        "rollback_config":rollback_config_receipt,
        "rollback_plan":pin(rollback_plan_path),
        "rollback_preparation_receipt":pin(rollback_receipt_path),
        "existing_zero_805_report":pin(rollback_audit_path),
        "rollback_805_summary":rollback_summary,
        "independent_replay":pin(replay_path),
        "verified_state":pin(verified_path),
        "input_source_evidence":source_evidence,
        "final_source_evidence":final_source_evidence,
        "implementation_pins":implementation_pins,
        "physical_admission":False,"native_steps":0,
        "mass_reconciliation":replay["mass_reconciliation"],
        "auditor_rerun":False,"solver_rerun":False,
    }
    (output_dir/"recovery-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("input_source_evidence","final_source_evidence","implementation_pins")},indent=2,sort_keys=True))
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--trust-cap-m", type=float, default=5e-5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--objective-mode", choices=("legacy", "role_equal", "role_equal_nearpass"), default="legacy")
    ap.add_argument("--witness-report", type=Path, action="append", default=[])
    ap.add_argument("--auto-adjacent-witness-report", action="store_true",
                    help="explicitly include the state-adjacent proposal report when it matches current faces")
    ap.add_argument("--resume-proposal-state", type=Path)
    ap.add_argument("--resume-config", type=Path)
    ap.add_argument("--resume-audit-report", type=Path)
    ap.add_argument("--replay-existing-proposal-state", type=Path)
    ap.add_argument("--replay-existing-proposal-config", type=Path)
    ap.add_argument("--replay-existing-proposal-audit", type=Path)
    ap.add_argument("--replay-existing-rollback-state", type=Path)
    ap.add_argument("--replay-existing-rollback-config", type=Path)
    ap.add_argument("--replay-existing-rollback-plan", type=Path)
    ap.add_argument("--replay-existing-rollback-receipt", type=Path)
    ap.add_argument("--replay-existing-rollback-audit", type=Path)
    ap.add_argument("--preflight-existing-rollback", action="store_true",
                    help="validate all existing proposal/rollback/report pins without running independent numerical replay")
    args = ap.parse_args()
    replay_args=(args.replay_existing_proposal_state,args.replay_existing_proposal_config,
                 args.replay_existing_proposal_audit,args.replay_existing_rollback_state,
                 args.replay_existing_rollback_config,args.replay_existing_rollback_plan,
                 args.replay_existing_rollback_receipt,args.replay_existing_rollback_audit)
    if args.preflight_existing_rollback and not any(x is not None for x in replay_args):
        raise ValueError("--preflight-existing-rollback requires the existing rollback inputs")
    if any(x is not None for x in replay_args):
        if not all(x is not None for x in replay_args):
            raise ValueError("existing rollback replay mode requires proposal state/config/audit and rollback state/config/plan/receipt/audit")
        replay_existing_rollback(args.state,args.replay_existing_proposal_state,
            args.replay_existing_proposal_config,args.replay_existing_proposal_audit,
            args.replay_existing_rollback_state,args.replay_existing_rollback_config,
            args.replay_existing_rollback_plan,args.replay_existing_rollback_receipt,
            args.replay_existing_rollback_audit,args.output,
            preflight_only=args.preflight_existing_rollback)
        return
    if args.resume_proposal_state is not None or args.resume_config is not None or args.resume_audit_report is not None:
        if args.resume_proposal_state is None or args.resume_config is None:
            raise ValueError("resume mode requires --resume-proposal-state and --resume-config")
        resume_saved_proposal(args.state, args.resume_proposal_state, args.resume_config, args.output,
                              supplied_audit_report=args.resume_audit_report)
        return
    if not 1 <= args.rounds <= 2:
        raise ValueError("708 is bounded to one or two coupled rounds")
    if not 0 < args.trust_cap_m <= 5e-5 or not 0 < args.seconds <= 30:
        raise ValueError("v24 trust cap must be <=50 micrometres and solver limit <=30 seconds")
    source_state = args.state.resolve()
    source_evidence = verify_source_input(source_state)
    witness_reports = []
    for report_path in args.witness_report:
        resolved = report_path.resolve()
        if resolved not in witness_reports:
            witness_reports.append(resolved)
    adjacent_report = source_state.parent / "proposal-exact-805" / "report.json"
    if (args.auto_adjacent_witness_report and adjacent_report.is_file()
            and adjacent_report.resolve() not in witness_reports):
        witness_reports.append(adjacent_report.resolve())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    implementation_pins = {}
    for label, path in (("builder", BUILDER), ("auditor", AUDITOR), ("preparer", PREPARER),
                        ("rollback_helper", ROLLBACK_HELPER), ("finalizer", FINALIZER),
                        ("base_driver", BASE_DRIVER), ("quality_lp", QUALITY_LP),
                        ("transfer_owner", TRANSFER_OWNER), ("quality_owner", QUALITY_OWNER),
                        ("source_verifier", SOURCE_VERIFIER)):
        implementation_pins[label] = checked_pin(label, path)

    source = load_state(source_state)
    out.mkdir(parents=True)
    current_state_path = source_state
    source_doc = json.loads(source_state.read_text())
    if "current_verified_state_20261031" in source_doc:
        current_state_path = out / "initial-flat-current-verified-state.json"
        current_state_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    round_receipts = []
    start_all = time.monotonic()
    for index in range(1, args.rounds + 1):
        round_dir = out / f"round-{index:02d}"
        round_dir.mkdir()
        proposal_dir = round_dir / "proposal"
        witnesses_used = list(witness_reports)
        builder_cmd = [sys.executable, str(BUILDER), "--state", str(current_state_path),
                       "--output", str(proposal_dir), "--trust-cap-m", str(args.trust_cap_m),
                       "--seconds", str(args.seconds), "--inner-back-layer-repair",
                       "--ordinary-outer-o-local-repair", "--objective-mode", args.objective_mode]
        for report_path in witnesses_used:
            builder_cmd.extend(["--witness-report", str(report_path)])
        run_command("builder722", builder_cmd, round_dir / "builder-logs")
        proposal_state = proposal_dir / "state.json"
        proposal_config = proposal_dir / "context-config.json"
        if not proposal_state.is_file() or not proposal_config.is_file():
            raise RuntimeError("INNER_BACK_LAYER_REPAIR builder produced no complete coupled point proposal")
        proposal_doc=load_state(proposal_state)
        proposal_meta=proposal_doc.get("coupled_point_repair", {})
        if proposal_meta.get("inner_back_layer_repair") is not True or proposal_meta.get("inner_back_layer_policy") != "INNER_BACK_LAYER_REPAIR":
            raise RuntimeError("builder state lacks explicit INNER_BACK_LAYER_REPAIR metadata")
        if proposal_meta.get("ordinary_outer_O_local_repair") is not True:
            raise RuntimeError("builder state lacks ordinary-O local repair metadata")
        if proposal_meta.get("quality_objective_mode") != args.objective_mode:
            raise RuntimeError("builder state objective mode does not match the selected runner mode")
        cfgcheck=json.loads(proposal_config.read_text())
        changed_roles=set(cfgcheck.get("changed_roles", []))
        if not changed_roles or not changed_roles.issubset({"O","I","D"}):
            raise RuntimeError("builder config must declare changed O/I/D roles only")
        for role in changed_roles:
            entry=cfgcheck.get("bounded_point_repair",{}).get(role,{})
            if float(entry.get("maximum_displacement_m",0.0))>args.trust_cap_m:
                raise RuntimeError(role+": builder config exceeds the requested trust cap")
            if role=="O":
                if entry.get("ordinary_outer_o_local_repair") is not True or entry.get("inner_back_layer_repair",False):
                    raise RuntimeError("O config lacks ordinary local policy or carries inner-only policy")
            elif entry.get("inner_back_layer_repair") is not True:
                raise RuntimeError(role+": I/D config lacks authenticated inner-layer policy")
        proposal_audit_dir = round_dir / "proposal-exact-805"
        proposal_report = run_audit(proposal_config, proposal_audit_dir, round_dir / "proposal-audit-logs")
        proposal_summary, proposal_clean = audit_counts(proposal_report)
        new_witness_report = None
        if not proposal_clean and proposal_report.resolve() not in witness_reports:
            witness_reports.append(proposal_report.resolve())
            new_witness_report = proposal_report.resolve()
        rollback_dir = None
        rollback_plan = None
        final_state_input = proposal_state
        final_exact_report = proposal_report
        if not proposal_clean:
            rollback_dir = round_dir / "witness-component-rollback"
            run_command("rollback625", [sys.executable, str(PREPARER), "--state", str(proposal_state),
                        "--config", str(proposal_config), "--audit", str(proposal_report),
                        "--output", str(rollback_dir)], round_dir / "rollback-logs")
            rollback_state = rollback_dir / "state.json"
            rollback_config = rollback_dir / "context-config.json"
            rollback_plan = rollback_dir / "rollback-plan.json"
            prep_receipt = rollback_dir / "rollback-preparation-receipt.json"
            if not all(p.is_file() for p in (rollback_state, rollback_config, rollback_plan, prep_receipt)):
                raise RuntimeError("625 did not produce complete whole-component rollback outputs")
            rollback_audit_dir = round_dir / "rollback-exact-805"
            final_exact_report = run_audit(rollback_config, rollback_audit_dir, round_dir / "rollback-audit-logs")
            rollback_summary, rollback_clean = audit_counts(final_exact_report)
            if not rollback_clean:
                raise RuntimeError(f"805 still has exact witnesses after whole-component rollback: {rollback_summary}")
            finalized_dir = round_dir / "finalized"
            run_command("finalize625", [sys.executable, str(FINALIZER), "--state", str(rollback_state),
                        "--baseline-state", str(current_state_path), "--proposal-state", str(proposal_state),
                        "--rollback-receipt", str(prep_receipt), "--audit-report", str(final_exact_report),
                        "--audit-config", str(rollback_config), "--output-dir", str(finalized_dir)],
                        round_dir / "finalize-logs")
            final_state_input = finalized_dir / "state.json"
            if not final_state_input.is_file():
                raise RuntimeError("source-only finalizer produced no state")
        else:
            rollback_summary = None

        replay_report = round_dir / "independent-point-replay.json"
        replay = independent_replay(final_state_input, current_state_path, replay_report,
                                    proposal_state_path=proposal_state if rollback_plan else None,
                                    rollback_plan_path=rollback_plan)
        verified_state_path = round_dir / "verified-state.json"
        finalize_verified_state(final_state_input, final_exact_report, replay_report, verified_state_path)
        summary = {
            "round": index,
            "quality_objective_mode": args.objective_mode,
            "source_state": pin(current_state_path),
            "proposal_state": pin(proposal_state),
            "proposal_805": proposal_summary,
            "proposal_805_clean": proposal_clean,
            "witness_reports_used": [pin(path) for path in witnesses_used],
            "new_witness_report_retained": pin(new_witness_report) if new_witness_report else None,
            "rollback_state": pin(rollback_dir / "state.json") if rollback_dir else None,
            "rollback_plan": pin(rollback_plan) if rollback_plan else None,
            "rollback_805": rollback_summary,
            "independent_point_replay": pin(replay_report),
            "verified_state": pin(verified_state_path),
            "mass_reconciliation": replay["mass_reconciliation"],
        }
        (round_dir / "round-receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        round_receipts.append(summary)
        current_state_path = verified_state_path

    final = {
        "schema": "h706.coupled-quality-points-v24-ordinary-outer-o-local-805-source-only-continuation.v1",
        "status": "PASS_SOURCE_ONLY_EXACT_CONTEXT_AND_INDEPENDENT_REPLAY",
        "rounds_requested": args.rounds,
        "rounds_completed": len(round_receipts),
        "quality_objective_mode": args.objective_mode,
        "implementation_pins": implementation_pins,
        "original_input_state": pin(source_state),
        "input_source_evidence": source_evidence,
        "ordinary_outer_O_local_policy": "only baseline-bad O corners and their direct O source-triangle one-ring; protected O scope/rims/routes remain fixed",
        "initial_flat_parent_state": pin(current_state_path) if current_state_path != source_state else None,
        "rounds": round_receipts,
        "retained_exact_witness_reports": [pin(path) for path in witness_reports],
        "final_state": pin(current_state_path),
        "elapsed_s": time.monotonic() - start_all,
        "physical_admission": False,
        "native_steps": 0,
        "mass_reconciliation": "PENDING_PER_MATERIAL_VOLUME_CHANGE",
        "limitations": [
            "This is source/geometry evidence only; no physical or native admission was performed.",
            "Per-material geometric volume changes are reported; material mass reconciliation remains pending."
        ],
    }
    (out / "722-receipt.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in final.items() if k != "rounds"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

