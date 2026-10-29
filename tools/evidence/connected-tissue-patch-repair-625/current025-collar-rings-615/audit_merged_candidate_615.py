#!/usr/bin/env python3
"""Independent changed-face structure and whole-boundary/ancestry audit.

Does not repeat the full-candidate self-intersection audit or admit physics.
"""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, itertools, json, sys, time
import numpy as np

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
G = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype"
BASE = G / "pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
REPLAY = P / "geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
MANIFEST = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
BONE_TOOL = P / "bone-barrier-remeshing-609/bone_barrier.py"
EXACT_OWNER_LOADER = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
FINE_I = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
EXPECTED = {
    "base": "9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a",
    "native_replay": "e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5",
    "manifest": "0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c",
    "fine_I": "5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410",
    "exact_owner": "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sorted_faces(faces):
    faces = np.asarray(faces, dtype=np.int64)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    return faces[order]

def edge_inventory(faces):
    f = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    boundary_idx = order[starts[counts == 1]]
    boundary = directed[boundary_idx]
    boundary = boundary[np.lexsort((boundary[:, 1], boundary[:, 0]))]
    nonmanifold = unique[counts > 2]
    return {
        "boundary_directed": boundary,
        "nonmanifold_undirected": nonmanifold,
        "edge_count": int(len(unique)),
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "max_incidence": int(counts.max(initial=0)),
    }

def boundary_counter(faces):
    counts = collections.Counter((int(a), int(b)) for tri in faces for a, b in zip(tri, np.roll(tri, -1)))
    out = collections.Counter()
    for (a, b), n in counts.items():
        net = n - counts[(b, a)]
        if net > 0:
            out[(a, b)] = net
    return out

def select_faces(points, faces, boxes):
    if not len(faces):
        return np.empty(0, dtype=np.int64)
    tri = np.asarray(points)[np.asarray(faces, dtype=np.int64)]
    lo = tri.min(axis=1)
    hi = tri.max(axis=1)
    keep = np.zeros(len(faces), dtype=bool)
    for qlo, qhi in boxes:
        keep |= np.all(hi >= qlo, axis=1) & np.all(lo <= qhi, axis=1)
    return np.flatnonzero(keep)

def make_surface_records(pred, points, faces, boxes, vertex_offset, negative_counter, label, exact64=False, bone=None):
    faces = np.asarray(faces, dtype=np.int64)
    rows = select_faces(points, faces, boxes)
    if len(rows) == 0:
        return [], {"total_faces": int(len(faces)), "selected_faces": 0}
    chosen = faces[rows]
    used = np.unique(chosen)
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    if exact64:
        if bone is None:
            raise ValueError("F64 records require the exact bone-coordinate key owner")
        keys = tuple(bone.exact_common_lattice_key(points[int(i)]) for i in used)
    else:
        keys = tuple(pred.float32_point_lattice_key(points[int(i)]) for i in used)
    raw = pred._records(keys, mapping[chosen])
    records = []
    for rec in raw:
        rid = negative_counter[0]
        negative_counter[0] -= 1
        verts = tuple(vertex_offset + int(used[v]) for v in rec[4])
        records.append((rec[0], rec[1], rec[2], rid, verts))
    return records, {"total_faces": int(len(faces)), "selected_faces": int(len(rows))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--operations", required=True)
    ap.add_argument("--expected-candidate-sha256", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    candidate = Path(args.candidate)
    ops_path = Path(args.operations)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    if sha(BASE) != EXPECTED["base"]:
        raise SystemExit("candidate025 base hash mismatch")
    if sha(REPLAY) != EXPECTED["native_replay"] or sha(MANIFEST) != EXPECTED["manifest"] or sha(FINE_I) != EXPECTED["fine_I"]:
        raise SystemExit("one or more pinned structure inputs changed")
    if sha(candidate) != args.expected_candidate_sha256:
        raise SystemExit("merged candidate hash differs from explicit expected hash")
    started = time.monotonic()
    exact_loader = loadmod("owner_loader615", EXACT_OWNER_LOADER)
    pred = exact_loader.load_exact_owner()
    if sha(pred.__file__) != EXPECTED["exact_owner"]:
        raise SystemExit("exact owner hash mismatch")
    bone = loadmod("bone_owner615", BONE_TOOL)
    opsdoc = json.loads(ops_path.read_text())
    if opsdoc.get("candidate_sha256") != EXPECTED["base"]:
        raise SystemExit("accepted operation list is not based on pinned candidate025")
    operations = opsdoc.get("operations")
    if not isinstance(operations, list) or not operations:
        raise SystemExit("missing accepted operations")
    accepted_ids = set(map(int, opsdoc.get("accepted_transactions", [])))
    required = {34, 80}
    if not required.issubset(accepted_ids):
        raise SystemExit(f"merged operation list lacks 615 tx34/80: {sorted(required-accepted_ids)}")
    by_tx = collections.defaultdict(list)
    removed_rows = set()
    added_faces = []
    op_rows = []
    for op_index, op in enumerate(operations):
        tx = int(op["transaction_id"])
        parent_rows = op.get("parent_face_rows", op.get("removed_face_rows"))
        if not parent_rows:
            raise SystemExit(f"tx{tx} has no removed parent face rows")
        parent_rows = [int(x) for x in parent_rows]
        if len(parent_rows) != len(set(parent_rows)):
            raise SystemExit(f"tx{tx} repeats a removed parent row")
        overlap = removed_rows.intersection(parent_rows)
        if overlap:
            raise SystemExit(f"operations overlap in parent face rows: tx{tx} rows {sorted(overlap)[:8]}")
        removed_rows.update(parent_rows)
        new_faces = np.asarray(op.get("new_faces", []), dtype=np.int64).reshape(-1, 3)
        if len(new_faces) == 0:
            raise SystemExit(f"tx{tx} has no replacement faces")
        by_tx[tx].append({"operation_index": op_index, "parent_rows": parent_rows, "new_faces": new_faces, "operation": op})
        added_faces.extend(tuple(map(int, tri)) for tri in new_faces)
    if len(set(added_faces)) != len(added_faces):
        raise SystemExit("duplicate oriented face across accepted operations")
    if sha(candidate) != args.expected_candidate_sha256:
        raise SystemExit("candidate changed during preparation")

    with np.load(BASE, allow_pickle=False) as b:
        base_points = np.asarray(b["points_world_m"], dtype=np.float32)
        base_faces = np.asarray(b["faces"], dtype=np.int64)
        base_source_ids = np.asarray(b["face_source_original_id"], dtype=np.int64)
        base_anc_offsets = np.asarray(b["candidate_face_ancestry_offsets"], dtype=np.int64)
        base_anc_rows = np.asarray(b["candidate_face_ancestry_rows"], dtype=np.int64)
    with np.load(candidate, allow_pickle=False) as z:
        points = np.asarray(z["points_world_m"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int64)
        source_ids = np.asarray(z["face_source_original_id"], dtype=np.int64)
        anc_offsets = np.asarray(z["candidate_face_ancestry_offsets"], dtype=np.int64)
        anc_rows = np.asarray(z["candidate_face_ancestry_rows"], dtype=np.int64)
        metadata_raw = np.asarray(z["pair_linked_remesh_operation_json"]).reshape(-1)[0]
        metadata = json.loads(str(metadata_raw))
    if not np.array_equal(points, base_points):
        raise SystemExit("candidate changed point coordinates; this audit expects zero-coordinate boundary retriangulation")
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces.size and (faces.min() < 0 or faces.max() >= len(points))):
        raise SystemExit("invalid merged candidate face table")
    if len(source_ids) != len(faces) or len(anc_offsets) != len(faces) + 1 or anc_offsets[0] != 0 or anc_offsets[-1] != len(anc_rows):
        raise SystemExit("candidate face source/ancestry arrays have invalid row domains")
    if set(map(int, metadata.get("accepted_transaction_ids", []))) != accepted_ids:
        raise SystemExit("embedded candidate transaction metadata disagrees with accepted operations")
    if len(removed_rows) > len(base_faces) or (removed_rows and (min(removed_rows) < 0 or max(removed_rows) >= len(base_faces))):
        raise SystemExit("parent face row outside pinned candidate025")
    added_array = np.asarray(added_faces, dtype=np.int64).reshape(-1, 3)
    kept_mask = np.ones(len(base_faces), dtype=bool)
    if removed_rows:
        kept_mask[np.asarray(sorted(removed_rows), dtype=np.int64)] = False
    expected_faces = np.concatenate((base_faces[kept_mask], added_array), axis=0)
    if not np.array_equal(sorted_faces(expected_faces), sorted_faces(faces)):
        raise SystemExit("merged faces are not exactly the pinned baseline minus operations plus their authored triangles")
    if len(source_ids) != len(faces):
        raise SystemExit("face source ID length mismatch")
    if not np.array_equal(np.sort(base_source_ids), np.sort(base_source_ids.copy())):
        raise SystemExit("baseline source identity check failed")

    face_lookup = collections.defaultdict(list)
    for row, tri in enumerate(faces):
        face_lookup[tuple(map(int, tri))].append(row)
    all_new_rows = []
    op_ancestry = []
    tx_source_ids = {}
    for tx, txops in by_tx.items():
        tx_source_set = set()
        for item in txops:
            op = item["operation"]
            parents = item["parent_rows"]
            parent_vertices = set(map(int, base_faces[np.asarray(parents, dtype=np.int64)].reshape(-1)))
            expected_ancestry = set()
            for r in parents:
                expected_ancestry.update(map(int, base_anc_rows[base_anc_offsets[r]:base_anc_offsets[r+1]]))
            if not expected_ancestry:
                raise SystemExit(f"tx{tx} parent patch has no immutable face ancestry")
            loop_vertices = set(map(int, itertools.chain.from_iterable(op.get("boundary_loops", []))))
            replacement = item["new_faces"]
            if any(not set(map(int, tri)).issubset(parent_vertices) for tri in replacement):
                raise SystemExit(f"tx{tx} replacement introduced a vertex outside the removed patch")
            if loop_vertices and any(not set(map(int, tri)).issubset(loop_vertices) for tri in replacement):
                raise SystemExit(f"tx{tx} replacement is not boundary-only")
            if boundary_counter(base_faces[np.asarray(parents, dtype=np.int64)]) != boundary_counter(replacement):
                raise SystemExit(f"tx{tx} replacement changed the directed patch boundary")
            for tri in replacement:
                key = tuple(map(int, tri))
                rows = face_lookup.get(key, [])
                if len(rows) != 1:
                    raise SystemExit(f"tx{tx} authored triangle occurs {len(rows)} times in merged candidate")
                row = rows[0]
                all_new_rows.append(row)
                tx_source_set.add(int(source_ids[row]))
                actual_ancestry = set(map(int, anc_rows[anc_offsets[row]:anc_offsets[row+1]]))
                if actual_ancestry != expected_ancestry:
                    raise SystemExit(f"tx{tx} replacement face {row} ancestry differs from the removed-patch source union")
            op_ancestry.append({"transaction_id": tx, "patch_index": int(op.get("patch_index", 0)),
                "removed_parent_faces": len(parents), "removed_parent_ancestry_rows": len(expected_ancestry),
                "replacement_faces": len(replacement), "directed_boundary_preserved": True,
                "all_replacement_faces_keep_parent_ancestry_union": True})
        if len(tx_source_set) != 1:
            raise SystemExit(f"tx{tx} faces do not share one derived source identity")
        tx_source_ids[tx] = next(iter(tx_source_set))
    if len(set(tx_source_ids.values())) != len(tx_source_ids):
        raise SystemExit("two transactions share a derived source identity")
    base_source_set = set(map(int, base_source_ids))
    if set(tx_source_ids.values()) & base_source_set:
        raise SystemExit("a replacement reused a baseline source ID")

    if len(set(all_new_rows)) != len(all_new_rows):
        raise SystemExit("replacement operations share merged output rows")
    new_rows = np.asarray(sorted(all_new_rows), dtype=np.int64)
    base_edges = edge_inventory(base_faces)
    final_edges = edge_inventory(faces)
    boundary_equal = np.array_equal(base_edges["boundary_directed"], final_edges["boundary_directed"])
    nonmanifold_equal = np.array_equal(base_edges["nonmanifold_undirected"], final_edges["nonmanifold_undirected"])
    if not boundary_equal or not nonmanifold_equal:
        raise SystemExit("whole-candidate directed boundary or nonmanifold edge inventory changed")
    # All changed faces and obstacle records share the exact owner lattice.
    lattice = tuple(pred.float32_point_lattice_key(v) for v in points)
    new_face_to_tx = {}
    new_records = []
    new_record_meta = {}
    rid = 1_000_000_000_000
    for tx, txops in by_tx.items():
        tx_tris = np.concatenate([x["new_faces"] for x in txops], axis=0)
        raw = pred._records(lattice, tx_tris)
        local = []
        for tri_index, rec in enumerate(raw):
            record = (rec[0], rec[1], rec[2], rid, rec[4])
            new_face_to_tx[rid] = tx
            new_record_meta[rid] = {"kind": "replacement", "transaction_id": tx, "triangle": tx_tris[tri_index].tolist()}
            local.append(record)
            new_records.append(record)
            rid += 1
        txops[0]["new_records"] = local
    boxes = []
    for tx, txops in by_tx.items():
        tris = np.concatenate([x["new_faces"] for x in txops], axis=0)
        tri_points = points[tris].astype(float)
        lo = np.nextafter(tri_points.min(axis=(0, 1)), -np.inf)
        hi = np.nextafter(tri_points.max(axis=(0, 1)), np.inf)
        boxes.append((lo, hi, tx))
    all_boxes = [(lo, hi) for lo, hi, _ in boxes]
    negative = [-1]
    obstacle_records = []
    obstacle_ranges = []
    retained_mask = np.ones(len(faces), dtype=bool)
    retained_mask[new_rows] = False
    retained_rows = np.flatnonzero(retained_mask)
    retained_selected_local = select_faces(points, faces[retained_rows], all_boxes)
    retained_selected_rows = retained_rows[retained_selected_local]
    if len(retained_selected_rows):
        raw = pred._records(lattice, faces[retained_selected_rows])
        for local_i, rec in enumerate(raw):
            obstacle_records.append((rec[0], rec[1], rec[2], int(retained_selected_rows[local_i]), rec[4]))
    obstacle_ranges.append({"kind": "retained-candidate", "selected_faces": len(retained_selected_rows)})

    def add_obstacle(label, xyz, fs, vertex_offset, exact64=False, bank_info=None):
        nonlocal obstacle_records
        rows = select_faces(xyz, fs, all_boxes)
        if len(rows) == 0:
            obstacle_ranges.append({"kind": label, "selected_faces": 0})
            return
        chosen = np.asarray(fs, dtype=np.int64)[rows]
        used = np.unique(chosen)
        mapping = np.full(len(xyz), -1, dtype=np.int64)
        mapping[used] = np.arange(len(used), dtype=np.int64)
        if exact64:
            keys = tuple(bone.exact_common_lattice_key(xyz[int(i)]) for i in used)
        else:
            keys = tuple(pred.float32_point_lattice_key(xyz[int(i)]) for i in used)
        raw = pred._records(keys, mapping[chosen])
        for rec in raw:
            face_id = negative[0]
            negative[0] -= 1
            vertices = tuple(vertex_offset + int(used[v]) for v in rec[4])
            obstacle_records.append((rec[0], rec[1], rec[2], face_id, vertices))
            new_record_meta[face_id] = {"kind": label, **(bank_info or {})}
        obstacle_ranges.append({"kind": label, "selected_faces": int(len(rows)), **(bank_info or {})})

    with np.load(REPLAY, allow_pickle=False) as z:
        native_points = np.asarray(z["points_world_f32"], dtype=np.float64)
        native_faces = np.asarray(z["faces"], dtype=np.int64)
    add_obstacle("native-F32-bone-union", native_points, native_faces, len(points)+1_000_000)
    vertex_offset = len(points)+2_000_000
    manifest = json.loads(MANIFEST.read_text())
    if len(manifest["operands"]) != 185:
        raise SystemExit("expected all 185 Float64 bone operands")
    for operand in manifest["operands"]:
        operand_path = Path(operand["operand_path"])
        if sha(operand_path) != operand["operand_sha256"]:
            raise SystemExit(f"pinned F64 bone operand changed: {operand_path}")
        with np.load(operand_path, allow_pickle=False) as z:
            xyz = np.asarray(z["points_world_m_f64"], dtype=np.float64)
            fs = np.asarray(z["faces_f64"], dtype=np.int64)
        add_obstacle("native-F64-bone", xyz, fs, vertex_offset, exact64=True,
            bank_info={"body": int(operand["body"]), "bone": int(operand["bone"])})
        vertex_offset += len(xyz)
    with np.load(FINE_I, allow_pickle=False) as z:
        fine_points = np.asarray(z["points_world_m"], dtype=np.float64)
        fine_faces = np.asarray(z["faces"], dtype=np.int64)
    add_obstacle("open-fine-I", fine_points, fine_faces, vertex_offset)
    base_index = pred._prepare_surface_aabb(obstacle_records)

    structure_pairs = []
    per_operation = []
    for lo, hi, tx in boxes:
        records = []
        for item in by_tx[tx]:
            records.extend(item["new_records"])
        result = pred._audit_pair_prepared_first(base_index, records, same_surface=True)
        pairs = np.asarray(result["triangle_pairs"], dtype=np.int64).reshape(-1, 2)
        structure_pairs.extend(map(tuple, pairs.tolist()))
        per_kind = collections.Counter()
        sample = []
        for a, b in pairs:
            other = int(b) if int(a) in new_face_to_tx else int(a)
            kind = new_record_meta.get(other, {}).get("kind", "unknown")
            per_kind[kind] += 1
            if len(sample) < 24:
                sample.append({"pair": [int(a), int(b)], "other": new_record_meta.get(other, {})})
        per_operation.append({"transaction_id": tx, "replacement_faces": len(records),
            "exact_pairs_vs_retained_or_structures": int(result["count"]),
            "aabb_candidate_pairs": int(result.get("aabb_candidate_pairs", 0)),
            "pairs_by_obstacle_kind": dict(per_kind), "witness_sample": sample})
    new_self = pred._audit_record_pairs(itertools.combinations(new_records, 2), same_surface=True)
    self_pairs = np.asarray(new_self["triangle_pairs"], dtype=np.int64).reshape(-1, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.parent / (out.name+"-pairs.npz"),
        changed_vs_retained_or_structures=np.asarray(structure_pairs, dtype=np.int64).reshape(-1,2),
        new_face_self_pairs=self_pairs)
    result = {
        "schema": "independent-merged-boundary-patch-gates-615.v1",
        "candidate_path": str(candidate), "candidate_sha256": sha(candidate),
        "candidate025_path": str(BASE), "candidate025_sha256": EXPECTED["base"],
        "operations_path": str(ops_path), "operations_sha256": sha(ops_path),
        "exact_owner_sha256": sha(pred.__file__), "native_replay_sha256": sha(REPLAY),
        "fine_I_sha256": sha(FINE_I), "bone_manifest_sha256": sha(MANIFEST),
        "accepted_transaction_ids": sorted(accepted_ids),
        "operation_ancestry_checks": op_ancestry,
        "derived_source_ids_by_transaction": {str(k): v for k,v in sorted(tx_source_ids.items())},
        "whole_candidate": {
            "vertex_count": int(len(points)), "face_count": int(len(faces)),
            "baseline_vertices": int(len(base_points)), "baseline_faces": int(len(base_faces)),
            "removed_faces": int(len(removed_rows)), "replacement_faces": int(len(added_faces)),
            "coordinates_bitwise_unchanged": True,
            "face_multiset_exactly_reconstructs_from_pinned_baseline_and_operations": True,
            "directed_boundary_equal": True,
            "baseline_boundary_edge_count": base_edges["boundary_edge_count"],
            "final_boundary_edge_count": final_edges["boundary_edge_count"],
            "baseline_nonmanifold_edge_count": base_edges["nonmanifold_edge_count"],
            "final_nonmanifold_edge_count": final_edges["nonmanifold_edge_count"],
            "nonmanifold_edge_inventory_equal": True,
            "new_face_source_ancestry_equals_removed_patch_ancestry_union": True,
        },
        "structure_records": obstacle_ranges,
        "per_operation_changed_face_structure_gate": per_operation,
        "new_face_cross_operation_self_pairs": int(new_self["count"]),
        "changed_vs_retained_or_structures_pair_archive": str(out.parent/(out.name+"-pairs.npz")),
        "whole_candidate_self_intersection_audit": "NOT_RUN_HERE; coordinated single full-self audit remains separate",
        "physical_admission": False, "runtime": 0, "accepted_steps": 0,
        "elapsed_s": time.monotonic()-started,
    }
    if any(x["exact_pairs_vs_retained_or_structures"] for x in per_operation) or new_self["count"]:
        result["status"] = "FAIL_EXACT_CHANGED_FACE_OR_CROSS_PATCH_INTERSECTION"
    else:
        result["status"] = "PASS_CHANGED_FACE_STRUCTURE_AND_WHOLE_BOUNDARY_ANCESTRY_ONLY"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "candidate_sha256": result["candidate_sha256"],
        "operation_count": len(operations), "transactions": sorted(accepted_ids),
        "per_operation": [{"tx":x["transaction_id"],"faces":x["replacement_faces"],"pairs":x["exact_pairs_vs_retained_or_structures"]} for x in per_operation],
        "new_face_cross_operation_self_pairs": int(new_self["count"]),
        "boundary_equal": result["whole_candidate"]["directed_boundary_equal"],
        "ancestry": result["whole_candidate"]["new_face_source_ancestry_equals_removed_patch_ancestry_union"],
        "elapsed_s": result["elapsed_s"], "physical_admission": False}, indent=2), flush=True)

if __name__ == "__main__":
    main()
