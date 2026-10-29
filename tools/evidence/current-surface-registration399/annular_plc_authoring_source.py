from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
HUMAN_BUILDER = Path("/Users/numi/numilab-human/tools/build_native_domain_217.py")
O_PATH = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-030-global-derived-patch-remesh/attempt-002/connected-witness-repair-001/result-003/candidate-O-open-f32-typed.npz"
I_PATH = P / "geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
D_PATH = P / "deep-core-connected-patch-filter-375/candidate-f32-typed.npz"
COLLAR_PATH = P / "volume-worker/head-collar/proposal-001/donor-head-collar-O033.npz"
WALL_CSV = P / "volume-worker/head-collar/proposal-001/donor-sidewall-faces.csv"
PARENT_CELLS = P / "volume-worker/baseline-parent-cell-mass.csv.gz"
PARENT_NODES = P / "volume-worker/baseline-compiled-parent-nodes.csv.gz"
FRAME_STDOUT = P / "P/native-kind-aware-source-preflight-226/run-006-native-emission/stdout.txt"
I_CLOSED = P / "geometry-worker/matched-nesting-390/run-001/I-closed-F32.npz"
D_CLOSED = P / "deep-core-connected-patch-filter-375/D-closed-csg-only.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_builder():
    spec = importlib.util.spec_from_file_location("native_domain_217_preflight", HUMAN_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing native-domain builder: {HUMAN_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_placement() -> tuple[np.ndarray, np.ndarray]:
    line = next((line for line in FRAME_STDOUT.read_text().splitlines()
                 if "common_placement_translation_m=" in line), None)
    if line is None:
        raise ValueError("pinned native run-006 output lacks the common placement")
    match = re.search(r"common_placement_translation_m=(\[[^]]+\]).*common_placement_rotation=(\[[^]]+\])", line)
    if not match:
        raise ValueError("could not parse pinned run-006 common placement")
    t = np.asarray(json.loads(match.group(1)), dtype=np.float64)
    r = np.asarray(json.loads(match.group(2)), dtype=np.float64).reshape(3, 3)
    det = float(np.linalg.det(r))
    orth = float(np.max(np.abs(r.T @ r - np.eye(3))))
    if t.shape != (3,) or not np.isfinite(t).all() or not np.isfinite(r).all() or det <= 0.0 or orth > 1e-6:
        raise ValueError(f"pinned common placement is invalid: det={det}, orth={orth}")
    return r, t


def boundary_rows(faces: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    f = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    rows = np.unique(boundary)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("surface boundary is not a collection of simple loops")
    seen: set[int] = set()
    loop_lengths: list[int] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        q = [start]
        seen.add(start)
        for u in q:
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        loop_lengths.append(len(q))
    return rows, sorted(loop_lengths), int(len(boundary))


def coord_key(point: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.asarray(point, dtype=np.float32).view(np.uint32))


def map_ring(name: str, surface: Any, old_ring: np.ndarray,
             rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(surface["points"] if isinstance(surface, dict) else surface["points_world_m"], dtype=np.float32)
    faces = np.asarray(surface["faces"], dtype=np.int64)
    boundary, loops, edge_count = boundary_rows(faces)
    if edge_count != 45 or len(boundary) != 45 or loops != [21, 24]:
        raise ValueError(f"{name} expected boundary loops [21,24], got edges={edge_count}, vertices={len(boundary)}, loops={loops}")
    lookup: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in boundary:
        lookup[coord_key(points[int(row)])].append(int(row))
    expected = (np.asarray(old_ring, dtype=np.float64) @ rotation.T + translation).astype(np.float32)
    mapped: list[int] = []
    for local, point in enumerate(expected):
        matches = lookup.get(coord_key(point), [])
        if len(matches) != 1:
            raise ValueError(f"{name} head ring row {local} has {len(matches)} exact current boundary matches")
        mapped.append(matches[0])
    mapped = np.asarray(mapped, dtype=np.int64)
    if len(set(mapped.tolist())) != 45 or set(mapped.tolist()) != set(map(int, boundary)):
        raise ValueError(f"{name} mapped head ring does not bijectively cover the complete boundary")
    if not np.array_equal(points[np.asarray(mapped, dtype=np.int64)], expected):
        raise ValueError(f"{name} common-transform ring roundtrip is not exact Float32")
    return np.asarray(mapped, dtype=np.int64), {
        "boundary_edges": edge_count, "boundary_vertices": len(boundary),
        "boundary_loop_lengths": loops, "exact_transformed_ring_matches": len(mapped),
        "ring_rows_in_current_surface": mapped.tolist(),
        "current_ring_points_sha256": hashlib.sha256(np.ascontiguousarray(points[mapped]).tobytes()).hexdigest(),
    }


def surface_dict(z: Any, layer: str) -> dict[str, Any]:
    points = np.asarray(z["points_world_m"], dtype=np.float32)
    faces = np.asarray(z["faces"], dtype=np.int64)
    nf = len(faces)
    original = np.asarray(z["face_source_original_id"], dtype=np.int64) if "face_source_original_id" in z.files else np.full(nf, -1, np.int64)
    source_face = np.asarray(z["face_source_face_id"], dtype=np.int64) if "face_source_face_id" in z.files else (np.asarray(z["face_source_id"], dtype=np.int64) if "face_source_id" in z.files else np.full(nf, -1, np.int64))
    layers = np.asarray(z["face_source_layer"], dtype=np.int8) if "face_source_layer" in z.files else np.full(nf, -1, np.int8)
    bodies = np.asarray(z["face_source_body"], dtype=np.int32) if "face_source_body" in z.files else np.full(nf, -1, np.int32)
    bones = np.asarray(z["face_source_bone"], dtype=np.int32) if "face_source_bone" in z.files else np.full(nf, -1, np.int32)
    run_ids = np.asarray(z["face_run_original_id"], dtype=np.int64) if "face_run_original_id" in z.files else np.full(nf, -1, np.int64)
    if any(len(a) != nf for a in (original, source_face, layers, bodies, bones, run_ids)):
        raise ValueError(f"{layer} typed per-face arrays do not align with faces")
    offsets = np.asarray(z["face_ancestry_offsets"], dtype=np.int64) if "face_ancestry_offsets" in z.files else np.arange(nf + 1, dtype=np.int64)
    if offsets.shape != (nf + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{layer} ancestry offsets invalid")
    if layer == "O" and "face_ancestry_source_face_rows" in z.files:
        ancestry_rows = np.asarray(z["face_ancestry_source_face_rows"], dtype=np.int64)
        ancestry_ids = np.asarray(z["face_ancestry_source_original_ids"], dtype=np.int64)
        ancestry_parent = np.asarray(z["face_ancestry_parent_face_rows"], dtype=np.int64)
        kind = np.full(nf, 6, dtype=np.int8)
    elif layer == "D" and "face_ancestry_input_cache_rows" in z.files:
        ancestry_rows = np.asarray(z["face_ancestry_input_cache_rows"], dtype=np.int64)
        ancestry_ids = np.asarray(z["face_ancestry_original_ids"], dtype=np.int64)
        # The producer's typed D file also retains face_ids and identity0/1 in the
        # pinned source NPZ. The native shell maps its boundary faces to those
        # input rows; no face-ID is relabelled as a parent row here.
        ancestry_parent = np.full(len(ancestry_rows), -1, dtype=np.int64)
        kind = np.full(nf, 3, dtype=np.int8)
    elif layer == "I":
        # I288 has typed per-face identity arrays but no multi-parent CSR. The
        # shell mapping is explicitly to the unchanged I288 input face row.
        ancestry_rows = np.arange(nf, dtype=np.int64)
        ancestry_ids = original.copy()
        ancestry_parent = np.full(nf, -1, dtype=np.int64)
        kind = np.where(original >= 0, 3, 0).astype(np.int8)
    else:
        raise ValueError(f"{layer} input is missing its expected typed face source contract")
    if offsets[-1] != len(ancestry_rows) or len(ancestry_ids) != len(ancestry_rows) or len(ancestry_parent) != len(ancestry_rows):
        raise ValueError(f"{layer} ancestry CSR arrays disagree")
    if layer == "I" and "face_ancestry_offsets" not in z.files:
        offsets = np.arange(nf + 1, dtype=np.int64)
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{layer} ancestry offsets are not monotone")
    ancestry = [ancestry_rows[offsets[i]:offsets[i+1]].tolist() for i in range(nf)]
    ancestry_id_rows = [ancestry_ids[offsets[i]:offsets[i+1]].tolist() for i in range(nf)]
    ancestry_parent_rows = [ancestry_parent[offsets[i]:offsets[i+1]].tolist() for i in range(nf)]
    return {
        "path": "", "points": points, "faces": faces,
        # The shell face row maps directly to the unchanged input candidate row.
        "source_rows": np.arange(nf, dtype=np.int64),
        # Exact upstream one-to-one source row is intentionally not asserted;
        # its typed IDs/CSR stay pinned by the input hash and per-row mapping.
        "exact_source_rows": np.full(nf, -1, dtype=np.int64),
        "source_original_ids": original, "source_layers": layers,
        "source_bodies": bodies, "source_bones": bones,
        "source_run_original_ids": run_ids, "source_face_ids": source_face,
        "ancestry": ancestry, "ancestry_original_ids": ancestry_id_rows,
        "ancestry_parent_rows": ancestry_parent_rows,
        "ancestry_kind": kind, "raw_npz_keys": list(z.files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    start = time.monotonic()
    args.output.mkdir(parents=True)
    builder = import_builder()
    for path in (O_PATH, I_PATH, D_PATH, COLLAR_PATH, WALL_CSV, PARENT_CELLS,
                 PARENT_NODES, FRAME_STDOUT, I_CLOSED, D_CLOSED, HUMAN_BUILDER):
        if not path.is_file():
            raise FileNotFoundError(path)
    o_npz = np.load(O_PATH, allow_pickle=False)
    i_npz = np.load(I_PATH, allow_pickle=False)
    d_npz = np.load(D_PATH, allow_pickle=False)
    collar_npz = np.load(COLLAR_PATH, allow_pickle=False)
    original_collar_points = np.asarray(collar_npz["points_world_m"], dtype=np.float32)
    i_ring_count = len(collar_npz["donor_I_parent_node_rows"])
    d_ring_count = len(collar_npz["donor_D_parent_node_rows"])
    old_o_count = len(original_collar_points) - i_ring_count - d_ring_count
    if i_ring_count != 45 or d_ring_count != 45 or old_o_count <= 0:
        raise ValueError("proposal001 collar ring contract is unexpected")
    o_ring_old = np.asarray(collar_npz["donor_O_to_candidate_vertex"], dtype=np.int64)
    i_ring_old = original_collar_points[old_o_count:old_o_count+i_ring_count]
    d_ring_old = original_collar_points[old_o_count+i_ring_count:]
    rotation, translation = read_placement()
    outer = surface_dict(o_npz, "O")
    inner = surface_dict(i_npz, "I")
    deep = surface_dict(d_npz, "D")
    rings = {}
    for label, layer, old_points in (("O", outer, original_collar_points[o_ring_old]),
                                     ("I", inner, i_ring_old),
                                     ("D", deep, d_ring_old)):
        mapped, report = map_ring(label, layer, old_points, rotation, translation)
        rings[label] = mapped
        report["proposal001_original_ring_parent_node_rows"] = np.asarray(
            collar_npz[f"donor_{label}_parent_node_rows"], dtype=np.int64).tolist()
        rings[label + "_report"] = report
    no, ni, nd = len(outer["points"]), len(inner["points"]), len(deep["points"])
    new_collar_points = np.vstack((outer["points"], inner["points"][rings["I"]],
                                   deep["points"][rings["D"]])).astype(np.float32)
    new_collar: dict[str, Any] = {key: np.asarray(collar_npz[key]).copy() for key in collar_npz.files}
    new_collar["points_world_m"] = new_collar_points
    new_collar["outer_faces"] = outer["faces"].copy()
    new_collar["donor_O_to_candidate_vertex"] = rings["O"].astype(np.uint32)
    new_collar["exact_candidate_face_source_rows"] = np.full(len(outer["faces"]), -1, dtype=np.int64)
    old_to_current_o = {int(old): int(new) for old, new in zip(o_ring_old, rings["O"])}
    if len(old_to_current_o) != 45:
        raise ValueError("old O ring rows do not map one-to-one to current O rows")
    skin_wall_old = np.asarray(collar_npz["skin_sidewall_faces"], dtype=np.int64)
    fat_wall_old = np.asarray(collar_npz["fat_sidewall_faces"], dtype=np.int64)
    def remap_skin_node(v: int) -> int:
        if v < old_o_count:
            if v not in old_to_current_o:
                raise ValueError(f"skin wall references non-ring O row {v}")
            return old_to_current_o[v]
        if v < old_o_count + i_ring_count:
            return no + (v - old_o_count)
        raise ValueError(f"skin wall references unexpected old collar point {v}")
    def remap_fat_node(v: int) -> int:
        if old_o_count <= v < old_o_count + i_ring_count:
            return no + (v - old_o_count)
        if old_o_count + i_ring_count <= v < old_o_count + i_ring_count + d_ring_count:
            return no + i_ring_count + (v - old_o_count - i_ring_count)
        raise ValueError(f"fat wall references unexpected old collar point {v}")
    new_skin_wall = np.asarray([[remap_skin_node(int(v)) for v in tri] for tri in skin_wall_old], dtype=np.int64)
    new_fat_wall = np.asarray([[remap_fat_node(int(v)) for v in tri] for tri in fat_wall_old], dtype=np.int64)
    new_collar["skin_sidewall_faces"] = new_skin_wall
    new_collar["fat_sidewall_faces"] = new_fat_wall
    wall_rows = list(csv.DictReader(WALL_CSV.open(newline="")))
    remapped_wall = []
    for row in wall_rows:
        out = dict(row)
        idx = int(row["wall_face_row"])
        old_nodes = [int(row[f"output_node{k}"]) for k in range(3)]
        mapped_nodes = [remap_skin_node(v) for v in old_nodes] if idx < 90 else [remap_fat_node(v) for v in old_nodes]
        for k, v in enumerate(mapped_nodes):
            out[f"output_node{k}"] = str(v)
        remapped_wall.append(out)
    wall_path = args.output / "donor-sidewall-faces-current-ring-map.csv"
    with wall_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(remapped_wall[0]))
        writer.writeheader()
        writer.writerows(remapped_wall)
    # The wall source rows are unchanged. Only the ring point rows are carried
    # through the pinned common placement and exact current-boundary map.
    parent_node_rows: dict[int, np.ndarray] = {}
    def add_nodes(parent_ids: np.ndarray, current_rows: np.ndarray, points: np.ndarray, layer: str) -> None:
        if len(parent_ids) != len(current_rows):
            raise ValueError(f"{layer} parent-node/ring mapping lengths differ")
        for parent_id, row in zip(parent_ids, current_rows):
            value = np.asarray(points[int(row)], dtype=np.float32)
            old = parent_node_rows.get(int(parent_id))
            if old is not None and not np.array_equal(old, value):
                raise ValueError(f"shared ring parent node {int(parent_id)} has inconsistent current coordinates")
            parent_node_rows[int(parent_id)] = value
    add_nodes(np.asarray(collar_npz["donor_O_parent_node_rows"], dtype=np.int64), rings["O"], outer["points"], "O")
    add_nodes(np.asarray(collar_npz["donor_I_parent_node_rows"], dtype=np.int64), rings["I"], inner["points"], "I")
    add_nodes(np.asarray(collar_npz["donor_D_parent_node_rows"], dtype=np.int64), rings["D"], deep["points"], "D")
    with gzip.open(PARENT_NODES, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        node_fields = list(reader.fieldnames or [])
        by_id = {int(row["parent_node_index"]): row for row in reader if int(row["parent_node_index"]) in parent_node_rows}
    if set(by_id) != set(parent_node_rows):
        raise ValueError(f"parent node source table misses {len(set(parent_node_rows)-set(by_id))} ring nodes")
    partial_nodes = args.output / "ring-parent-nodes-current.csv.gz"
    with gzip.open(partial_nodes, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=node_fields)
        writer.writeheader()
        for node in sorted(by_id):
            row = dict(by_id[node])
            for axis, value in zip("xyz", parent_node_rows[node]):
                row[f"current_{axis}_m"] = format(float(value), ".9g")
            writer.writerow(row)
    wall_rows_pinned = builder.read_wall_provenance(
        new_collar, wall_path, PARENT_CELLS, partial_nodes)
    # The O closed source uses its full typed multi-origin CSR. I288 maps each
    # shell face to its exact unchanged input face row; D375 keeps its current
    # typed CSR. The full candidate input files and hashes remain pinned below.
    skin, fat, assembly = builder.assemble_shells(outer, inner, deep, new_collar, wall_rows_pinned)
    emission = builder.plc_emission_preflight({"skin": skin, "fat": fat})
    capacity = builder.plc_capacity_preflight({"skin": skin, "fat": fat})
    i_core_npz = np.load(I_CLOSED, allow_pickle=False)
    i_core_points = np.asarray(i_core_npz["points_world_m_f64"], dtype=np.float32)
    i_core_faces = np.asarray(i_core_npz["faces_f64"], dtype=np.int64)
    d_core_npz = np.load(D_CLOSED, allow_pickle=False)
    d_core_points = np.asarray(d_core_npz["points_world_m"], dtype=np.float32)
    d_core_faces = np.asarray(d_core_npz["faces"], dtype=np.int64)
    def core_holes(points: np.ndarray, faces: np.ndarray, label: str) -> tuple[list[np.ndarray], dict[str, Any]]:
        core = builder._manifold(points, faces)
        components = core.decompose()
        items = []
        holes: list[np.ndarray] = []
        for component in components:
            volume = float(component.volume())
            if volume > 1e-15:
                point = builder._interior_point(component)
                holes.append(np.asarray(point, dtype=np.float64))
                items.append({"signed_volume_m3": volume, "hole_seed_world_m": np.asarray(point).tolist()})
            else:
                items.append({"signed_volume_m3": volume, "hole_seed_world_m": None})
        if not holes:
            raise ValueError(f"{label} core has no positive-volume component")
        return holes, {"components": len(components), "positive_hole_seeds": len(holes), "component_summaries": items}
    i_holes, i_core_report = core_holes(i_core_points, i_core_faces, "I")
    d_holes, d_core_report = core_holes(d_core_points, d_core_faces, "D")
    skin_region = builder._interior_point(builder._manifold(skin["points_world_m"], skin["faces"], skin["face_marker"]))
    fat_region = builder._interior_point(builder._manifold(fat["points_world_m"], fat["faces"], fat["face_marker"]))
    if not emission["emission_possible"]:
        raise ValueError("assembled PLC shell has Float32-collapsed facet rows")
    builder.write_shell_npz(args.output / "skin-shell.npz", skin, "skin")
    builder.write_shell_npz(args.output / "fat-shell.npz", fat, "fat")
    builder.write_face_ancestry_csv(args.output / "skin-face-ancestry.csv.gz", skin)
    builder.write_face_ancestry_csv(args.output / "fat-face-ancestry.csv.gz", fat)
    builder.write_poly(args.output / "skin.poly", skin["points_world_m"], skin["faces"],
                       skin["face_marker"], i_holes, [(skin_region, 15)])
    builder.write_poly(args.output / "fat.poly", fat["points_world_m"], fat["faces"],
                       fat["face_marker"], d_holes, [(fat_region, 16)])
    frame_report = {"source_stdout": str(FRAME_STDOUT), "source_stdout_sha256": sha256(FRAME_STDOUT),
                    "rotation_row_major": rotation.reshape(-1).tolist(),
                    "translation_m": translation.tolist(),
                    "placement_contract": "native run-006 recorded common placement; no fitted transform",
                    "ring_map": {k: rings[k + "_report"] for k in ("O", "I", "D")},
                    "wall_input_sha256": sha256(WALL_CSV),
                    "ring_parent_nodes_registered": len(parent_node_rows),
                    "wall_registration_scope": "45 O, 45 I, 45 D head-ring parent nodes only; not a full parent frame map"}
    report = {
        "classification": "annular O/I/D PLC assembly preflight only",
        "physical_admission": False, "tetgen_ran": False, "physics_ran": False,
        "source": {"script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
                   "human_builder": str(HUMAN_BUILDER), "human_builder_sha256": sha256(HUMAN_BUILDER)},
        "inputs": {str(path): sha256(path) for path in
                   (O_PATH, I_PATH, D_PATH, COLLAR_PATH, WALL_CSV, PARENT_CELLS,
                    PARENT_NODES, FRAME_STDOUT, I_CLOSED, D_CLOSED)},
        "ring_frame_registration": frame_report,
        "wall_parent_provenance_rows": {"skin": wall_rows_pinned["skin_parent_tets"].tolist(),
                                         "fat": wall_rows_pinned["fat_parent_tets"].tolist(),
                                         "current_ring_node_source_table": str(partial_nodes),
                                         "ring_node_source_table_sha256": sha256(partial_nodes)},
        "assembly": assembly,
        "emission": emission,
        "capacity": capacity,
        "core_hole_seeds": {"I": i_core_report, "D": d_core_report},
        "region_seed_world_m": {"skin": np.asarray(skin_region).tolist(), "fat": np.asarray(fat_region).tolist()},
        "current_surface_row_mapping": {
            "O": "shell source face row equals unchanged candidate O input face row; full typed CSR retained from input",
            "I": "shell source face row equals unchanged I288 input face row; upstream typed per-face identity remains pinned by input hash",
            "D": "shell source face row equals unchanged D375 input face row; source CSR remains pinned by input hash",
            "input_face_counts": {"O": len(outer["faces"]), "I": len(inner["faces"]), "D": len(deep["faces"])},
        },
        "notes": [
            "Proposal001 wall topology and parent tet/node identities are retained; only its three head rings are remapped by exact Float32 matching under the pinned native run-006 common placement.",
            "The separate parent-node CSV contains only the 135 mapped head-ring nodes. It does not register interior nodes, donor frames, or mass transfer.",
            "The matched Boolean cap residuals and any bank/core occupancy results are not treated as admission evidence here.",
            "Surface self/bone audit receipts remain separate; this run checks annular closure, shell orientation, F32 facet emission, core seeds, regions, and necessary capacity only.",
            "A capacity preflight is a necessary lower bound, not a TetGen mesh-quality or physical validity certificate."
        ],
        "outputs": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
        "elapsed_s": time.monotonic() - start,
    }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PLC_EMISSION_CAPACITY_PREFLIGHT_COMPLETE",
                      "output": str(args.output), "skin_faces": len(skin["faces"]),
                      "fat_faces": len(fat["faces"]), "emission_possible": emission["emission_possible"],
                      "capacity_possible": capacity["capacity_possible"],
                      "minimum_combined_tetrahedra": capacity["minimum_combined_tetrahedra"],
                      "positive_I_holes": i_core_report["positive_hole_seeds"],
                      "positive_D_holes": d_core_report["positive_hole_seeds"],
                      "tetgen_ran": False, "physical_admission": False,
                      "elapsed_s": report["elapsed_s"]}, separators=(",", ":")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
