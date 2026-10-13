#!/usr/bin/env python3
"""CPU-only all-eight-pose audit of the 334 vascular branch field."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import mmap
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np

FDS = Path("/Users/numi/numi-human-fds132-m6-1350")
OUT = FDS / "attempt-335-vascular-branch-all8-audit-001/result-004"
FIELD_DIR = FDS / "attempt-334-vascular-branch-collar-field-001"
ROW489_AUDIT = FDS / "attempt-335-row489-exact-seam-all8-002/report.json"
A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
C = A / "forearm-shared-field-native-012-diagonal097"
RUN = C / "baseline/native-run/accepted-geometry"
RUN_DECL = C / "baseline/run-declaration.json"
PUB = Path("/Users/n/numi-human-left-pq-publication-1344/src")
INV = Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
NHA_RECEIPT = C / "resting-anatomy-receipt.json"
RIGID = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
ARCHIVE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz")
POSE_HELPER = A / "prepare_forearm_shared_field_native_012_diagonal097.py"
FIELD_SCRIPT = FIELD_DIR / "prepare_branch_field.py"
FIELD_REPORT = FIELD_DIR / "branch-field-report.json"
NHA = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy")
STEPS = (0, 3807, 4767, 5023, 5599, 6207, 7423, 8000)
SOURCE_IDS = (479, 481, 483, 485, 487, 489)
FIELD_IDS = (483, 485, 487, 489)
BRANCH_PAIRS = tuple((a, b) for a in (479, 483, 487) for b in (481, 485, 489)) + ((479, 483), (479, 487), (483, 487), (481, 485), (481, 489), (485, 489))
SEED_LINKS = ((479, 483), (481, 485), (479, 487), (483, 487), (481, 489), (485, 489))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def quantiles(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0}
    q = np.quantile(x, [0.0, 0.5, 0.95, 1.0])
    return {"count": int(len(x)), "min_m": float(q[0]), "median_m": float(q[1]),
            "p95_m": float(q[2]), "max_m": float(q[3])}


def read_source_rows():
    header = struct.Struct("<8s5I32s")
    record = struct.Struct("<8I")
    wanted = set(SOURCE_IDS)
    rows = {}
    with NHA.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        h = header.unpack_from(data)
        if h[0] != b"NHANAT1\0" or h[1] != 5 or h[2] <= 0:
            raise RuntimeError(f"unexpected NHA header {h[:3]}")
        vertex_offset = header.size + h[2] * record.size
        index_offset = vertex_offset + h[3] * 24
        for i in range(h[2]):
            body, v0, nv, i0, ni, sid, layer, reserved = record.unpack_from(data, header.size + i * record.size)
            if sid not in wanted:
                continue
            packed = np.frombuffer(data, dtype="<f4", count=nv * 6, offset=vertex_offset + 24 * v0).reshape((-1, 6)).copy()
            faces = np.frombuffer(data, dtype="<u4", count=ni, offset=index_offset + 4 * i0).reshape((-1, 3)).copy().astype(np.int64) - v0
            rows[int(sid)] = {"body": int(body), "xyz": packed[:, :3].copy(), "normals": packed[:, 3:].copy(), "faces": faces}
    if set(rows) != wanted:
        raise RuntimeError(f"source row inventory differs: {sorted(rows)}")
    expected = {479: 91, 481: 91, 483: 93, 485: 93, 487: 92, 489: 92}
    if {sid: row["body"] for sid, row in rows.items()} != expected:
        raise RuntimeError("source owner identity mismatch")
    return rows


def exact_quotient_records(xyz, faces, ci):
    point_ids = {}
    unique_keys = []
    unique_xyz = []
    slot_to_unique = np.empty(len(xyz), dtype=np.int64)
    for i, point in enumerate(np.asarray(xyz, dtype="<f4")):
        key = ci.float32_point_lattice_key(tuple(float(v) for v in point))
        j = point_ids.get(key)
        if j is None:
            j = len(unique_keys)
            point_ids[key] = j
            unique_keys.append(key)
            unique_xyz.append(point.copy())
        slot_to_unique[i] = j
    quotient_faces = slot_to_unique[np.asarray(faces, dtype=np.int64)]
    degenerate = []
    for face_id, face in enumerate(quotient_faces):
        a, b, c = (unique_keys[int(k)] for k in face)
        if not any(ci._cross(ci._sub(b, a), ci._sub(c, a))):
            degenerate.append(face_id)
    if degenerate:
        return None, np.asarray(unique_xyz, dtype="<f4"), quotient_faces, slot_to_unique, degenerate
    records = ci._records(unique_keys, quotient_faces.tolist())
    return records, np.asarray(unique_xyz, dtype="<f4"), quotient_faces, slot_to_unique, []


def self_classification(records, ci):
    if records is None:
        return {"nonadjacent_count": None, "allowed_shared_simplex_count": None, "error": "records missing"}
    owner = ci._audit_pair(records, records, same_surface=True)
    return {"nonadjacent_count": int(owner["count"]),
            "nonadjacent_face_pairs": owner["triangle_pairs"],
            "allowed_shared_simplex_count": int(owner["allowed_shared_vertex_or_edge_pairs"]),
            "exact_owner_count": int(owner["count"]), "aabb_candidate_pairs": int(owner["aabb_candidate_pairs"])}


def selected_records(xyz, faces, selected, ci):
    chosen_faces = np.asarray(faces[selected], dtype=np.int64)
    used = np.unique(chosen_faces)
    lattice = [ci.float32_point_lattice_key(tuple(float(v) for v in xyz[int(i)])) for i in used]
    local_faces = np.searchsorted(used, chosen_faces)
    base = ci._records(lattice, local_faces.tolist())
    return [(row[0], row[1], row[2], int(selected[row[3]]), row[4]) for row in base]


def audit_rss():
    limit_kib = int(2.5 * 1024 * 1024)
    while True:
        try:
            rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip())
            if rss > limit_kib:
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(2)


def main():
    if OUT.exists():
        raise RuntimeError(f"preserve prior audit output: {OUT}")
    OUT.mkdir(parents=True)
    threading.Thread(target=audit_rss, daemon=True).start()
    sys.path.insert(0, str(PUB))
    from numilab_human import cardiac_cavity_intersections as ci
    from numilab_human import common_atlas_skin_clearance as ca
    field_builder = load_module("vascular_branch_field_334", FIELD_SCRIPT)

    field_report = json.loads(FIELD_REPORT.read_text())
    if field_report.get("native_admission") is not False:
        raise RuntimeError("334 source field report no longer declares pending native admission")
    if sha(NHA) != field_report["source_payload_sha256"]:
        raise RuntimeError("field candidate NHA source hash mismatch")
    source = read_source_rows()
    field_arrays = {}
    for sid in FIELD_IDS:
        field_path = FIELD_DIR / f"stable-{sid}-branch-parent-field.npz"
        z = np.load(field_path, allow_pickle=False)
        if int(z["stable_surface_id"][0]) != sid or not np.array_equal(z["faces"], source[sid]["faces"]):
            raise RuntimeError(f"field/source face identity mismatch for stable {sid}")
        if not np.array_equal(z["source_local_positions"], source[sid]["xyz"]) or not np.array_equal(z["source_local_normals"], source[sid]["normals"]):
            raise RuntimeError(f"field/source vertex slot identity mismatch for stable {sid}")
        field_arrays[sid] = {k: z[k].copy() for k in z.files}
        z.close()

    # Confirm the known exact duplicate-slot seam in the unchanged 489 source.
    dup_slots = (104, 2961)
    if not np.array_equal(source[489]["xyz"][dup_slots[0]], source[489]["xyz"][dup_slots[1]]):
        raise RuntimeError("stable489 source duplicate-slot pair changed")
    field_sha = {sid: sha(FIELD_DIR / f"stable-{sid}-branch-parent-field.npz") for sid in FIELD_IDS}

    # Default-world source correspondences for the declared proximal branch seed neighborhoods.
    default_poses = {}
    for sid in FIELD_IDS:
        z = field_arrays[sid]
        for body_raw in z["target_body_ids"]:
            body = int(body_raw)
            position = z[f"default_body{body}_position"].astype("<f4")
            quaternion = z[f"default_body{body}_quaternion_xyzw"].astype("<f4")
            previous = default_poses.get(body)
            if previous and (not np.array_equal(previous[0], position) or not np.array_equal(previous[1], quaternion)):
                raise RuntimeError(f"334 default transform mismatch for body {body}")
            default_poses[body] = (position, quaternion)
    default_world = {}
    for sid in SOURCE_IDS:
        if sid in field_arrays:
            default_world[sid] = field_arrays[sid]["registered_rest_world_positions"].astype("<f4")
        else:
            body = source[sid]["body"]
            p, q = default_poses[body]
            default_world[sid] = field_builder.transform32(q, p, source[sid]["xyz"])
    from scipy.spatial import cKDTree
    seed_pairs = {}
    for neighbor, child in SEED_LINKS:
        distance, nearest = cKDTree(default_world[neighbor]).query(default_world[child], k=1, workers=1)
        ids = np.flatnonzero(distance <= 0.001)
        if not len(ids):
            raise RuntimeError(f"empty source-defined branch collar {neighbor}->{child}")
        seed_pairs[(neighbor, child)] = (ids, nearest[ids].astype(np.int64), distance[ids])

    keys, _, _, _ = ca._load_target_inventory(INV)
    labels = {}
    with INV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            labels[(int(row["second_semantic"]), int(row["second_stable_id"]))] = row["source_owner_or_label"]
    vessel_pack_keys = {(51011, sid) for sid in SOURCE_IDS}
    if not vessel_pack_keys.issubset(keys):
        raise RuntimeError(f"D12 external target inventory omits source branch rows: {vessel_pack_keys - keys}")

    pack_paths = {step: RUN / f"step-{step}.mrvpack" for step in STEPS}
    receipt_paths = {step: RUN / f"step-{step}.receipt.json" for step in STEPS}
    helper_path = PUB / "numilab_human/common_atlas_skin_clearance.py"
    predicate_path = PUB / "numilab_human/cardiac_cavity_intersections.py"
    input_paths = [Path(__file__), FIELD_SCRIPT, FIELD_REPORT, ROW489_AUDIT, NHA_RECEIPT, NHA, RIGID, ARCHIVE, RUN_DECL,
                   POSE_HELPER, INV, helper_path, predicate_path]
    input_paths += [FIELD_DIR / f"stable-{sid}-branch-parent-field.npz" for sid in FIELD_IDS]
    input_paths += [p for step in STEPS for p in (pack_paths[step], receipt_paths[step])]
    pins = {str(path): sha(path) for path in input_paths}
    run_declaration = json.loads(RUN_DECL.read_text())
    argv = run_declaration.get("argv", [])
    dt_text = next((argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--dt"), None)
    if dt_text is None or int(run_declaration.get("accepted_steps", -1)) != 8000:
        raise RuntimeError("pinned D12 run declaration omits the accepted step count or dt")
    dt_native = float(np.float32(float(dt_text)))
    seam_audit = json.loads(ROW489_AUDIT.read_text())
    if (seam_audit.get("status") != "completed_actual_native_capture_row489_self_audit_native_admission_pending"
            or not seam_audit.get("all_eight_complete") or not seam_audit.get("input_hashes_unchanged_at_end")):
        raise RuntimeError("row489 exact-seam prerequisite is not complete/pin-stable")
    for step in STEPS:
        row = seam_audit["captures"][str(step)]
        if (row["raw_unquotiented_nonadjacent_self_intersection_count"] != 9
                or row["nonadjacent_self_intersection_count_after_exact_coordinate_quotient"] != 0
                or len(row["duplicate_slot_seam_contact_face_pairs"]) != 9
                or not all(pair["all_points_on_shared_simplex"] for pair in row["duplicate_slot_seam_contact_face_pairs"])):
            raise RuntimeError(f"row489 exact-seam prerequisite changed at step {step}")
    for step in STEPS:
        receipt = json.loads(receipt_paths[step].read_text())
        if int(receipt["accepted_step"]) != step or int(receipt["accepted_body_pose_count"]) != 157:
            raise RuntimeError(f"accepted receipt identity mismatch at step {step}")
        if abs(float(receipt["accepted_time_s"]) - dt_native * step) > 1e-12:
            raise RuntimeError(f"accepted receipt time disagrees with pinned Float32 dt at step {step}")
        if receipt.get("pack_file_sha256") != pins[str(pack_paths[step])]:
            raise RuntimeError(f"accepted receipt pack hash mismatch at step {step}")

    report = {
        "schema": "numi.human.vascular-branch-collar-field-all8-audit.335.v2",
        "status": "running", "scope": "CPU-only replay of the 334 positive branch fields over the eight accepted D12 body-pose captures; exact-coordinate-quotient self audits preserve original face IDs; all pack target surfaces are tested with exact pair predicates after AABB pruning. No native run, GPU, or source edits.",
        "source_field_report": str(FIELD_REPORT), "source_field_report_sha256": sha(FIELD_REPORT),
        "field_npz_sha256": {str(k): v for k, v in field_sha.items()},
        "pins": pins, "steps": list(STEPS), "captures": {},
        "run_declaration_sha256": pins[str(RUN_DECL)], "declared_dt_argument_s": float(dt_text),
        "native_float32_dt_s": dt_native, "row489_exact_seam_audit_sha256": pins[str(ROW489_AUDIT)],
        "branch_groups": {"artery": [479, 483, 487], "vein": [481, 485, 489]},
        "branch_seed_links": {f"{a}->{b}": {"count": int(len(seed_pairs[(a,b)][0])),
            "default_distance_m": quantiles(seed_pairs[(a,b)][2])} for a,b in SEED_LINKS},
        "interpretation": "Positive partition, branch seed proximity, and exact boundary pair counts do not by themselves prove anatomical continuity, lumen connectivity, or physical contact. This is a CPU predictor audit only.",
    }
    report_path = OUT / "report.json"

    def save():
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n")
        os.replace(tmp, report_path)

    save()
    for step in STEPS:
        started = time.monotonic()
        receipt = json.loads(receipt_paths[step].read_text())
        pose_rows = receipt["accepted_body_poses"]
        poses = {int(row["body_index"]): (np.asarray(row["position_m"], dtype="<f4"),
                    np.asarray(row["quaternion_xyzw"], dtype="<f4")) for row in pose_rows}
        if not {91, 92, 93}.issubset(poses):
            raise RuntimeError(f"required parent bodies missing at step {step}")
        candidate = {}
        for sid in SOURCE_IDS:
            if sid in field_arrays:
                z = field_arrays[sid]
                result = np.zeros_like(z["source_local_positions"], dtype="<f4")
                for body_raw in z["target_body_ids"]:
                    body = int(body_raw)
                    position, quaternion = poses[body]
                    transformed = field_builder.transform32(quaternion, position, z[f"body{body}_local_positions"])
                    weight = z[f"weight_body{body}"].astype("<f4")
                    result = np.add(result, np.multiply(weight[:, None], transformed, dtype="<f4"), dtype="<f4")
                candidate[sid] = result
            else:
                body = source[sid]["body"]
                position, quaternion = poses[body]
                candidate[sid] = field_builder.transform32(quaternion, position, source[sid]["xyz"])

        row_audits = {}
        candidate_records = {}
        for sid in SOURCE_IDS:
            records, unique_xyz, quotient_faces, slot_to_unique, degenerate = exact_quotient_records(candidate[sid], source[sid]["faces"], ci)
            if records is None:
                row_audits[str(sid)] = {"status": "failed_exact_float32_face", "degenerate_face_ids": degenerate}
                continue
            self_result = self_classification(records, ci)
            candidate_records[sid] = records
            orientation = []
            base = field_builder.transform32(poses[source[sid]["body"]][1], poses[source[sid]["body"]][0], source[sid]["xyz"])
            faces = source[sid]["faces"]
            base_tri = base[faces]
            current_tri = candidate[sid][faces]
            n0 = np.cross(base_tri[:,1]-base_tri[:,0], base_tri[:,2]-base_tri[:,0]).astype(np.float64)
            n1 = np.cross(current_tri[:,1]-current_tri[:,0], current_tri[:,2]-current_tri[:,0]).astype(np.float64)
            dots = np.einsum("ij,ij->i", n0, n1)
            valid = np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1)>0
            cosine = np.zeros(len(dots),dtype=np.float64)
            cosine[valid] = dots[valid]/(np.linalg.norm(n0[valid],axis=1)*np.linalg.norm(n1[valid],axis=1))
            orientation = {"negative_normal_dot_face_count": int(np.count_nonzero(dots < 0)),
                           "zero_or_degenerate_face_count": int(np.count_nonzero(~valid)),
                           "minimum_normal_cosine": float(cosine.min()) if len(cosine) else 1.0}
            row_audits[str(sid)] = {"status": "audited", "vertex_slots": int(len(candidate[sid])),
                "face_count": int(len(faces)), "exact_coordinate_quotient_vertex_count": int(len(unique_xyz)),
                "quotiented_duplicate_slot_count": int(len(candidate[sid])-len(unique_xyz)),
                "original_face_ids_preserved": True, "degenerate_face_ids": [],
                "self": self_result, "orientation_vs_source_owner_rigid_path": orientation}

        if set(candidate_records) != set(SOURCE_IDS):
            report["captures"][str(step)] = {"accepted_step": step, "row_self_audits": row_audits,
                "status": "failed_before_pair_scans"}
            save()
            raise RuntimeError(f"row self/degenerate gate failed at {step}")

        branch_pair_audits = {}
        for first_sid, second_sid in BRANCH_PAIRS:
            audit = ci._audit_pair(candidate_records[first_sid], candidate_records[second_sid], same_surface=False)
            branch_pair_audits[f"{first_sid}:{second_sid}"] = {"exact_pair_count": int(audit["count"]),
                "aabb_candidate_pairs": int(audit["aabb_candidate_pairs"]),
                "face_pairs": audit["triangle_pairs"]}
        branch_seed_gaps = {}
        for neighbor, child in SEED_LINKS:
            child_ids, neighbor_ids, rest_distance = seed_pairs[(neighbor, child)]
            current_distance = np.linalg.norm(candidate[child][child_ids].astype(np.float64) -
                candidate[neighbor][neighbor_ids].astype(np.float64), axis=1)
            branch_seed_gaps[f"{neighbor}->{child}"] = {"default_seed_distance_m": quantiles(rest_distance),
                "capture_seed_vertex_gap_m": quantiles(current_distance)}

        pack = pack_paths[step]
        xyz, surfaces, pack_meta = ca._pack_surfaces(pack, set(keys))
        expected_pack_keys = set(keys) | {(51007, 1)}
        if set(surfaces) != expected_pack_keys:
            raise RuntimeError(f"unexpected D12 target inventory at step {step}: count={len(surfaces)} missing={expected_pack_keys-set(surfaces)} extra={set(surfaces)-expected_pack_keys}")
        candidate_indices = {sid: ci._prepare_surface_aabb(candidate_records[sid]) for sid in SOURCE_IDS}
        candidate_aabbs = {sid: (candidate[sid].min(axis=0), candidate[sid].max(axis=0)) for sid in SOURCE_IDS}
        union_lo = np.min(np.stack([v[0] for v in candidate_aabbs.values()]), axis=0)
        union_hi = np.max(np.stack([v[1] for v in candidate_aabbs.values()]), axis=0)
        external_rows, targets_scanned, faces_considered, faces_selected = [], 0, 0, 0
        per_semantic = {}
        for target_key, target in sorted(surfaces.items()):
            if target_key in vessel_pack_keys:
                continue
            target_faces = np.asarray(target["faces"], dtype=np.int64)
            selected_chunks = []
            for start in range(0, len(target_faces), 262144):
                stop = min(len(target_faces), start+262144)
                chunk_faces = target_faces[start:stop]
                tri = xyz[chunk_faces]
                tri_lo = tri.min(axis=1); tri_hi = tri.max(axis=1)
                mask = np.all(tri_hi >= union_lo, axis=1) & np.all(tri_lo <= union_hi, axis=1)
                local = np.flatnonzero(mask)
                if len(local):
                    selected_chunks.append(local.astype(np.int64)+start)
            selected = np.concatenate(selected_chunks) if selected_chunks else np.empty(0,dtype=np.int64)
            targets_scanned += 1; faces_considered += len(target_faces); faces_selected += len(selected)
            if not len(selected):
                if targets_scanned % 64 == 0:
                    report["captures"][str(step)] = {"status":"running_external_scan", "targets_scanned":targets_scanned,
                        "row_self_audits":row_audits,"branch_pair_audits":branch_pair_audits}
                    save()
                continue
            target_records = selected_records(xyz, target_faces, selected, ci)
            for sid in SOURCE_IDS:
                audit = ci._audit_pair_prepared_first(candidate_indices[sid], target_records, same_surface=False)
                if audit["count"]:
                    external_rows.append({"candidate_stable_id":sid,"target_semantic":int(target_key[0]),
                        "target_stable_id":int(target_key[1]),"target_body":int(target["body"]),
                        "target_label":labels.get(target_key,"unknown"),"exact_pair_count":int(audit["count"]),
                        "aabb_candidate_pairs":int(audit["aabb_candidate_pairs"]),
                        "face_pairs":audit["triangle_pairs"]})
                    sem=str(target_key[0]); per_semantic.setdefault(sem,0); per_semantic[sem]+=int(audit["count"])
            if targets_scanned % 64 == 0:
                report["captures"][str(step)] = {"status":"running_external_scan", "targets_scanned":targets_scanned,
                    "external_nonzero_targets":len(external_rows),"row_self_audits":row_audits,
                    "branch_pair_audits":branch_pair_audits}
                save()
        report["captures"][str(step)] = {
            "status":"completed", "accepted_step":int(receipt["accepted_step"]),
            "accepted_time_s":float(receipt["accepted_time_s"]), "pack_sha256":pins[str(pack)],
            "pack_inventory":pack_meta,"row_self_audits":row_audits,
            "branch_pair_audits":branch_pair_audits,"branch_seed_gap_metrics":branch_seed_gaps,
            "external_targets_scanned":targets_scanned,"external_faces_considered":faces_considered,
            "external_aabb_selected_faces":faces_selected,"external_nonzero_target_rows":external_rows,
            "external_pair_count_by_semantic":per_semantic,
            "external_exact_pair_total":sum(int(r["exact_pair_count"]) for r in external_rows),
            "elapsed_seconds":time.monotonic()-started}
        save()
        del xyz, surfaces, candidate_indices, candidate_records, candidate, pose_rows, poses
        gc.collect()

    report["input_hashes_unchanged_at_end"] = all(sha(Path(path)) == digest for path, digest in pins.items())
    report["status"] = "completed_cpu_predicted_field_audit_native_pending"
    report["native_or_gpu_used"] = False
    report["all8_captures_complete"] = set(report["captures"]) == {str(step) for step in STEPS}
    save()
    print(json.dumps({"status":report["status"],"report":str(report_path),
        "report_sha256":sha(report_path),"captures":{k:{"self_rows":{sid:v.get("self",{}).get("nonadjacent_count") for sid,v in c.get("row_self_audits",{}).items()},
        "external_pairs":c.get("external_exact_pair_total"),"branch_pairs":{p:v["exact_pair_count"] for p,v in c.get("branch_pair_audits",{}).items()}} for k,c in report["captures"].items()}},sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
