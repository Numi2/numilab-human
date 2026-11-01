#!/usr/bin/env python3
"""Build current paired Skin/Fat surface shells from authenticated O/I/D sources.

This is a source-only adapter. It preserves the current I point/face namespace,
rebinds the donor collar through unique exact-F32 ring coordinates, and delegates
orientation, ancestry, and wall checks to the pinned 938v2 owner.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.util, json, os, shutil, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

H = Path(__file__).resolve().parent.parent
OWNER_PATH = H / "current-role-p1-support-938/current_fine_217_owner_938_v2_coordinate_alias_guard.py"
OWNER_SHA = "1fc5d6278aa0aa9bd76105d4bf7caa2c1ada6431428e20fe16c28e4afad3f6e4"
BRIDGE_PATH = H / "current-role-p1-support-938/current_fine_volume_to_217_938_v2_coordinate_alias_guard.py"
BRIDGE_SHA = "1756c7126f708141b06b57e9b653a121d038d215d8801bb7e28f5220edcc8c66"
SCHEMA = "numi.h706.current-paired-shell-1022.v1"
ROLE_NAMES = {"O": "outer_O", "I": "inner_I", "D": "inner_D"}
EXPECTED_SOURCE_AUDIT = "PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY"
EXPECTED_CROSSING_AUDIT = "PASS_CURRENT_806_EXACT_SOURCE_CONTEXT"

def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pin(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p), "size_bytes": p.stat().st_size}

class NPZView(dict):
    @property
    def files(self):
        return list(self.keys())

def import_pinned(path: Path, expected_sha: str, module_name: str):
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError(f"pinned source owner changed: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_current_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text())
    if state.get("physical_admission") is not False or state.get("native_steps", 0) != 0:
        raise ValueError("current state must remain source-only")
    if state.get("source_audit_status") != EXPECTED_SOURCE_AUDIT:
        raise ValueError("current source surface/boundary audit is not passing")
    if state.get("source_crossing_audit_status") != EXPECTED_CROSSING_AUDIT:
        raise ValueError("current O/I/D point-movement crossing audit is not passing")
    surfaces = state.get("source_surfaces")
    if not isinstance(surfaces, dict) or set(surfaces) != {"O", "I", "D"}:
        raise ValueError("current state must bind exactly O/I/D source surfaces")
    for role, item in surfaces.items():
        p = Path(item["path"]).resolve()
        if not p.is_file() or sha(p) != item["sha256"]:
            raise ValueError(f"current {role} source pin is missing or stale")
    return state

def read_manifest_inputs(rim_receipt_path: Path, parent_manifest_path: Path):
    rim = json.loads(rim_receipt_path.read_text())
    if rim.get("status") != "PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY" or rim.get("physical_admission") is not False:
        raise ValueError("rim receipt is not the existing source-only collar owner output")
    rim_outputs = rim.get("outputs", {})
    collar = rim_outputs.get("collar")
    wall_csv = rim_outputs.get("wall_csv")
    if not isinstance(collar, dict) or not isinstance(wall_csv, dict):
        raise ValueError("rim receipt lacks exact collar and wall CSV outputs")
    for name, item in (("collar", collar), ("wall_csv", wall_csv)):
        p = Path(item["path"])
        if not p.is_file() or sha(p) != item["sha256"]:
            raise ValueError(f"rim {name} output is missing or stale")
    parent = json.loads(parent_manifest_path.read_text())
    inputs = parent.get("inputs", {})
    cells = inputs.get("wall_parent_cells")
    nodes = inputs.get("wall_parent_nodes")
    if not isinstance(cells, dict) or not isinstance(nodes, dict):
        raise ValueError("parent-view manifest lacks authenticated wall parent cell/node tables")
    for name, item in (("wall_parent_cells", cells), ("wall_parent_nodes", nodes)):
        p = Path(item["path"])
        if not p.is_file() or sha(p) != item["sha256"]:
            raise ValueError(f"{name} input is missing or stale")
    return rim, parent, {
        "collar": Path(collar["path"]).resolve(),
        "wall_csv": Path(wall_csv["path"]).resolve(),
        "wall_parent_cells": Path(cells["path"]).resolve(),
        "wall_parent_nodes": Path(nodes["path"]).resolve(),
        "wall_coordinate_frame": "registered-native-current",
    }

def _array_alias(z, names, count, dtype, fill):
    for name in names:
        if name in z.files:
            a = np.asarray(z[name], dtype=dtype)
            if a.shape != (count,):
                raise ValueError(f"current typed surface field {name!r} has the wrong face row domain")
            return a
    return np.full(count, fill, dtype=dtype)

def load_current_surface(owner, path: Path, layer: str) -> dict[str, Any]:
    # The current typed O/I/D candidates use source_input_face_rows and
    # face_ancestry_source_face_rows; the older 938 loader expects aliases from
    # an earlier schema. Read the current schema directly, preserving every
    # ancestry lane and using the pinned 938 geometry/orientation implementation.
    with np.load(path, allow_pickle=False) as z:
        points = owner._source_surface_points_f32(z, layer)
        face_key = "source_O_faces" if layer == "O" and "source_O_faces" in z.files else "faces"
        faces = np.asarray(z[face_key], dtype=np.int64)
        count = len(faces)
        row = None
        for key in ("source_input_face_rows", "face_cache_row", "face_source_row",
                    "envelope_source_face_row"):
            if key in z.files:
                row = np.asarray(z[key], dtype=np.int64)
                break
        if row is None or row.shape != (count,):
            raise ValueError(f"{layer} current typed surface lacks aligned source_input_face_rows")
        original = _array_alias(z, ("face_source_original_id", "envelope_source_original_id"),
                                count, np.int64, -1)
        exact = _array_alias(z, ("face_exact_source_row", "exact_source_face_row"),
                             count, np.int64, -1)
        layers = _array_alias(z, ("face_source_layer",), count, np.int8, -1)
        bodies = _array_alias(z, ("face_source_body",), count, np.int32, -1)
        bones = _array_alias(z, ("face_source_bone",), count, np.int32, -1)
        run_ids = _array_alias(z, ("face_run_original_id",), count, np.int64, -1)
        face_ids = _array_alias(z, ("face_source_face_id", "face_id"), count, np.int64, -1)
        if "face_ancestry_offsets" in z.files:
            offsets = np.asarray(z["face_ancestry_offsets"], dtype=np.int64)
            row_key = next((k for k in ("face_ancestry_source_face_rows",
                                         "face_ancestry_source_rows",
                                         "face_ancestry_input_cache_rows") if k in z.files), None)
            id_key = next((k for k in ("face_ancestry_source_original_ids",
                                        "face_ancestry_original_ids") if k in z.files), None)
            parent_key = next((k for k in ("face_ancestry_parent_face_rows",
                                            "face_ancestry_parent_rows") if k in z.files), None)
            if row_key is None or id_key is None or parent_key is None:
                raise ValueError(f"{layer} current face ancestry CSR is incomplete")
            anc_rows = np.asarray(z[row_key], dtype=np.int64)
            anc_ids = np.asarray(z[id_key], dtype=np.int64)
            anc_parents = np.asarray(z[parent_key], dtype=np.int64)
            if (offsets.shape != (count+1,) or offsets[0] != 0 or
                    offsets[-1] != len(anc_rows) or len(anc_ids) != len(anc_rows) or
                    len(anc_parents) != len(anc_rows) or np.any(np.diff(offsets) < 0)):
                raise ValueError(f"{layer} current face ancestry CSR has inconsistent offsets/lanes")
            ancestry = [anc_rows[offsets[i]:offsets[i+1]].tolist() for i in range(count)]
            ancestry_ids = [anc_ids[offsets[i]:offsets[i+1]].tolist() for i in range(count)]
            ancestry_parents = [anc_parents[offsets[i]:offsets[i+1]].tolist() for i in range(count)]
        else:
            ancestry = [[int(v)] if int(v) >= 0 else [] for v in row]
            ancestry_ids = [[int(original[i])] * len(ancestry[i]) for i in range(count)]
            ancestry_parents = [[-1] * len(ancestry[i]) for i in range(count)]
        if (faces.ndim != 2 or faces.shape[1] != 3 or count == 0 or
                faces.min() < 0 or faces.max() >= len(points)):
            raise ValueError(f"{layer} current typed face indices are malformed")
        if (np.any(faces[:,0] == faces[:,1]) or np.any(faces[:,1] == faces[:,2]) or
                np.any(faces[:,2] == faces[:,0])):
            raise ValueError(f"{layer} current typed faces contain repeated point IDs")
        tri = points[faces]
        if np.any(np.all(tri[:,0] == tri[:,1], axis=1) |
                  np.all(tri[:,1] == tri[:,2], axis=1) |
                  np.all(tri[:,2] == tri[:,0], axis=1)):
            raise ValueError(f"{layer} current typed faces contain Float32-collapsed triangles")
        if layer == "O" and "face_ancestry_offsets" not in z.files:
            raise ValueError("current O needs complete typed face ancestry for current shell rebinding")
        kinds = np.full(count, 6 if layer == "O" else 3, dtype=np.int8)
        kinds[np.fromiter((not bool(a) for a in ancestry), bool, count=count)] = 0
        return {"path": str(path), "points": points, "faces": faces,
            "source_rows": row, "exact_source_rows": exact,
            "source_original_ids": original, "source_layers": layers,
            "source_bodies": bodies, "source_bones": bones,
            "source_run_original_ids": run_ids, "source_face_ids": face_ids,
            "ancestry": ancestry, "ancestry_original_ids": ancestry_ids,
            "ancestry_parent_rows": ancestry_parents, "ancestry_kind": kinds,
            "raw_npz_keys": list(z.files)}

def f32_key(point: np.ndarray) -> bytes:
    return np.ascontiguousarray(point, dtype=np.float32).view(np.uint32).tobytes()

def unique_exact_map(source_points: np.ndarray, target_points: np.ndarray, label: str) -> np.ndarray:
    target: dict[bytes, list[int]] = {}
    for row, point in enumerate(np.asarray(target_points, np.float32)):
        target.setdefault(f32_key(point), []).append(row)
    out = []
    for old_row, point in enumerate(np.asarray(source_points, np.float32)):
        matches = target.get(f32_key(point), [])
        if len(matches) != 1:
            raise ValueError(f"{label} donor coordinate at local row {old_row} has {len(matches)} exact-F32 current matches")
        out.append(matches[0])
    if len(set(out)) != len(out):
        raise ValueError(f"{label} donor coordinate map is not one-to-one")
    return np.asarray(out, dtype=np.int64)

def rebind_collar_to_current_surfaces(collar: dict[str, Any], outer_points: np.ndarray,
                                      outer_faces: np.ndarray, inner_points: np.ndarray,
                                      deep_points: np.ndarray,
                                      current_outer_has_typed_csr: bool) -> tuple[NPZView, dict[str, Any]]:
    old_points = np.asarray(collar["points_world_m"], dtype=np.float32)
    old_i = np.asarray(collar["donor_I_parent_node_rows"], dtype=np.int64)
    old_d = np.asarray(collar["donor_D_parent_node_rows"], dtype=np.int64)
    old_o = np.asarray(collar["donor_O_to_candidate_vertex"], dtype=np.int64)
    n_i, n_d = len(old_i), len(old_d)
    old_no = len(old_points) - n_i - n_d
    if old_no <= 0 or old_o.shape != np.asarray(collar["donor_O_parent_node_rows"]).shape:
        raise ValueError("collar O/I/D point partition or O parent correspondence is malformed")
    if old_o.size and (old_o.min() < 0 or old_o.max() >= old_no):
        raise ValueError("collar donor-O vertex row is outside its old O point block")
    if old_points.shape != (old_no + n_i + n_d, 3):
        raise ValueError("collar point block does not match its typed O/I/D ring counts")
    old_faces = np.asarray(collar["outer_faces"], dtype=np.int64)
    new_outer_points = np.asarray(outer_points, dtype=np.float32)
    new_outer_faces = np.asarray(outer_faces, dtype=np.int64)
    if new_outer_points.ndim != 2 or new_outer_points.shape[1] != 3:
        raise ValueError("current O point domain is malformed")
    if new_outer_faces.ndim != 2 or new_outer_faces.shape[1] != 3:
        raise ValueError("current O face domain is malformed")
    old_exact = np.asarray(collar["exact_candidate_face_source_rows"], dtype=np.int64)
    if old_exact.shape != (len(old_faces),):
        raise ValueError("old collar exact-O-face ancestry has the wrong row domain")
    if current_outer_has_typed_csr:
        # Current O's full typed CSR is authoritative, even when its face rows
        # happen to resemble the historical collar packet.
        exact_rows = np.full(len(new_outer_faces), -1, dtype=np.int64)
        exact_mode = "current_O_complete_typed_CSR"
    elif len(new_outer_faces) == len(old_faces) and np.array_equal(new_outer_faces, old_faces):
        exact_rows = old_exact.copy()
        exact_mode = "retained_identical_face_rows"
    else:
        raise ValueError("current O face topology changed without complete typed face ancestry")
    o_map = unique_exact_map(old_points[old_o], new_outer_points, "O")
    old_o_to_new = {int(a): int(b) for a, b in zip(old_o, o_map)}
    i_map = unique_exact_map(old_points[old_no:old_no+n_i], inner_points, "I")
    d_map = unique_exact_map(old_points[old_no+n_i:], deep_points, "D")
    if i_map.shape != (n_i,) or d_map.shape != (n_d,):
        raise ValueError("current collar I/D donor ring mapping is incomplete")
    ni, nd = len(inner_points), len(deep_points)
    def remap_wall(faces: np.ndarray, which: str) -> np.ndarray:
        src = np.asarray(faces, dtype=np.int64)
        out = np.empty_like(src)
        for ix, tri in enumerate(src):
            for j, node in enumerate(tri):
                v = int(node)
                if v < old_no:
                    if v not in old_o_to_new:
                        raise ValueError(f"{which} wall references O vertex {v} outside donor correspondence")
                    out[ix, j] = old_o_to_new[v]
                elif v < old_no + n_i:
                    out[ix, j] = len(new_outer_points) + (v - old_no)
                elif v < old_no + n_i + n_d:
                    out[ix, j] = len(new_outer_points) + n_i + (v - old_no - n_i)
                else:
                    raise ValueError(f"{which} wall references an invalid old collar point {v}")
        return out
    skin_wall = remap_wall(collar["skin_sidewall_faces"], "Skin")
    fat_wall = remap_wall(collar["fat_sidewall_faces"], "Fat")
    new_points = np.vstack((new_outer_points, np.asarray(inner_points, np.float32)[i_map],
                            np.asarray(deep_points, np.float32)[d_map])).astype(np.float32)
    # Validate that every retained wall triangle has identical ordered F32
    # coordinates after its row-map rewrite.
    for label, old, new in (("Skin", collar["skin_sidewall_faces"], skin_wall),
                            ("Fat", collar["fat_sidewall_faces"], fat_wall)):
        if not np.array_equal(old_points[np.asarray(old, np.int64)].view(np.uint32),
                              new_points[new].view(np.uint32)):
            raise ValueError(f"{label} wall triangle coordinates changed during exact rebind")
    out = NPZView({k: np.array(v, copy=True) for k, v in collar.items()})
    out.update(points_world_m=new_points, outer_faces=new_outer_faces,
               donor_O_to_candidate_vertex=o_map.astype(np.uint32),
               skin_sidewall_faces=skin_wall, fat_sidewall_faces=fat_wall,
               exact_candidate_face_source_rows=exact_rows)
    report = {
        "old_outer_points": int(old_no), "new_outer_points": int(len(new_outer_points)),
        "old_outer_faces": int(len(old_faces)), "new_outer_faces": int(len(new_outer_faces)),
        "donor_O_unique_exact_F32_map": o_map.tolist(),
        "donor_I_unique_exact_F32_map": i_map.tolist(),
        "donor_D_unique_exact_F32_map": d_map.tolist(),
        "wall_triangle_coordinates_byte_identical": True,
        "outer_face_ancestry_mode": exact_mode,
    }
    return out, report

def update_wall_csv(source: Path, destination: Path, collar: NPZView) -> dict[str, Any]:
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if fields is None or len(rows) != 180:
        raise ValueError("existing wall provenance must contain exactly 180 typed wall rows")
    required = {"wall_face_row", "role", "material_index", "output_node0", "output_node1", "output_node2"}
    if not required.issubset(fields):
        raise ValueError("wall CSV is missing existing typed output-node fields")
    for i, row in enumerate(rows):
        if int(row["wall_face_row"]) != i:
            raise ValueError("wall CSV order is not canonical")
        key = "skin_sidewall_faces" if i < 90 else "fat_sidewall_faces"
        tri = np.asarray(collar[key][i if i < 90 else i-90], dtype=np.int64)
        for k in range(3):
            row[f"output_node{k}"] = str(int(tri[k]))
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "fields": fields, "output_node_rows_rebound": True}

def validate_shell_maps(owner, skin: dict[str, Any], fat: dict[str, Any],
                        outer: dict[str, Any], inner: dict[str, Any],
                        deep: dict[str, Any]) -> dict[str, Any]:
    ni, no, nd = len(inner["points"]), len(outer["points"]), len(deep["points"])
    for label, shell, src, vertex_key, node_key, face_key, shell_face_key in (
        ("O", skin, outer, "registered_O_source_vertex_rows", "registered_O_shell_node_rows",
         "registered_O_source_faces", "registered_O_shell_face_rows"),
        ("D", fat, deep, "registered_D_source_vertex_rows", "registered_D_shell_node_rows",
         "registered_D_source_faces", "registered_D_shell_face_rows")):
        expected_rows = np.arange(len(src["points"]), dtype=np.int64)
        if not np.array_equal(shell[vertex_key], expected_rows):
            raise ValueError(f"{label} registered point IDs are not the current identity namespace")
        expected_nodes = expected_rows if label == "O" else ni + expected_rows
        if not np.array_equal(shell[node_key], expected_nodes):
            raise ValueError(f"{label} shell node rows do not match the current typed namespace")
        if not np.array_equal(shell[face_key], src["faces"]):
            raise ValueError(f"{label} shell source faces differ from current typed faces")
        shell_offset = 0 if label == "O" else len(inner["faces"])
        if not np.array_equal(shell[shell_face_key], shell_offset + np.arange(len(src["faces"]), dtype=np.int64)):
            raise ValueError(f"{label} shell face rows are not the current ordered namespace")
    for label, shell, source in (("Skin", skin, outer), ("Fat", fat, deep)):
        if int(shell["face_marker"].shape[0]) != int(shell["faces"].shape[0]):
            raise ValueError(f"{label} face marker count differs from shell faces")
    expected_i = np.arange(ni, dtype=np.int64)
    for label, shell, node_offset, face_offset in (
        ("Skin", skin, no, len(outer["faces"])),
        ("Fat", fat, 0, 0)):
        if not np.array_equal(shell["registered_I_source_vertex_rows"], expected_i):
            raise ValueError(f"{label} I source rows are not current identity rows")
        if not np.array_equal(shell["registered_I_shell_node_rows"], node_offset + expected_i):
            raise ValueError(f"{label} I shell-node rows are not the ordered current mapping")
        if not np.array_equal(shell["registered_I_source_faces"], inner["faces"]):
            raise ValueError(f"{label} registered I source faces differ from current I")
        if not np.array_equal(shell["registered_I_shell_face_rows"],
                              face_offset + np.arange(len(inner["faces"]), dtype=np.int64)):
            raise ValueError(f"{label} registered I shell face rows are not current ordered rows")
    return {"current_O_face_identity": True, "current_D_face_identity": True,
            "current_I_pair_identity": True, "shared_I_point_rows": ni,
            "shell_signed_volumes_positive": bool(skin["shell_signed_volume_m3"] > 0 and fat["shell_signed_volume_m3"] > 0)}

def build(state_path: Path, rim_receipt_path: Path, parent_manifest_path: Path,
          output: Path, wall_frame: str = "registered-native-current") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    state_path = state_path.resolve()
    state = load_current_state(state_path)
    rim, parent_manifest, inputs = read_manifest_inputs(rim_receipt_path, parent_manifest_path)
    if wall_frame != "registered-native-current":
        raise ValueError("current shell assembly is pinned to the existing registered-native-current wall frame")
    owner = import_pinned(OWNER_PATH, OWNER_SHA, "current_shell_938v2_owner")
    owner.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
    bridge = import_pinned(BRIDGE_PATH, BRIDGE_SHA, "current_shell_938v2_bridge")
    validator_owner = bridge.import_owner()
    surfaces = state["source_surfaces"]
    O_path, I_path, D_path = (Path(surfaces[k]["path"]).resolve() for k in ("O", "I", "D"))
    with np.load(inputs["collar"], allow_pickle=False) as z:
        raw_collar = NPZView({k: np.array(z[k], copy=True) for k in z.files})
    # Read each current typed surface once. Exact F32 donor-ring keys bind the
    # collar; the current owner consumes the resulting geometry/ancestry dicts.
    inner = load_current_surface(owner, I_path, "I")
    deep = load_current_surface(owner, D_path, "D")
    outer = load_current_surface(owner, O_path, "O")
    collar, collar_map = rebind_collar_to_current_surfaces(
        raw_collar, outer["points"], outer["faces"], inner["points"], deep["points"],
        "face_ancestry_offsets" in outer["raw_npz_keys"])
    output.mkdir(parents=True)
    collar_out = output / "current-collar.npz"
    wall_csv_out = output / "current-wall-provenance.csv"
    np.savez_compressed(collar_out, **collar)
    wall_csv_report = update_wall_csv(inputs["wall_csv"], wall_csv_out, collar)
    wall_rows = owner.read_wall_provenance(collar, wall_csv_out,
        inputs["wall_parent_cells"], inputs["wall_parent_nodes"], wall_frame)
    skin, fat, assembly = owner.assemble_shells(outer, inner, deep, collar, wall_rows)
    map_report = validate_shell_maps(owner, skin, fat, outer, inner, deep)
    skin_npz, fat_npz = output / "skin-shell.npz", output / "fat-shell.npz"
    skin_csv, fat_csv = output / "skin-face-ancestry.csv.gz", output / "fat-face-ancestry.csv.gz"
    owner.write_shell_npz(skin_npz, skin, "skin")
    owner.write_shell_npz(fat_npz, fat, "fat")
    owner.write_face_ancestry_csv(skin_csv, skin)
    owner.write_face_ancestry_csv(fat_csv, fat)
    skin_loaded = bridge.load_shell_npz(skin_npz)
    fat_loaded = bridge.load_shell_npz(fat_npz)
    bridge.validate_shared_i(validator_owner, skin_loaded, fat_loaded, inner)
    if not np.array_equal(
        np.ascontiguousarray(skin_loaded["points_world_m"][skin_loaded["registered_I_shell_node_rows"]]).view(np.uint32),
        np.ascontiguousarray(fat_loaded["points_world_m"][fat_loaded["registered_I_shell_node_rows"]]).view(np.uint32)):
        raise ValueError("serialized Skin/Fat shared I shell coordinate bytes differ")
    state_pin, rim_pin, parent_pin = pin(state_path), pin(rim_receipt_path), pin(parent_manifest_path)
    source_pins = {role: {"path": str(Path(item["path"]).resolve()), "sha256": item["sha256"]}
                   for role, item in surfaces.items()}
    inputs_out = {"current_state": state_pin, "source_surfaces": source_pins,
        **{f"source_surface_{role}": item for role, item in source_pins.items()},
        "rim_receipt": rim_pin, "parent_view_manifest": parent_pin,
        "original_collar": pin(inputs["collar"]), "original_wall_csv": pin(inputs["wall_csv"]),
        "wall_parent_cells": pin(inputs["wall_parent_cells"]),
        "wall_parent_nodes": pin(inputs["wall_parent_nodes"]),
        "938v2_owner": pin(OWNER_PATH), "938v2_bridge": pin(BRIDGE_PATH),
        "runner": pin(Path(__file__))}
    outputs = {}
    for p in (collar_out, wall_csv_out, skin_npz, fat_npz, skin_csv, fat_csv):
        outputs[p.name] = pin(p)
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_CURRENT_938V2_PAIRED_SHELLS_SOURCE_ONLY",
        "physical_admission": False, "native_steps": 0, "mass_mapping": "PENDING",
        "current_state": state_pin, "inputs": inputs_out, "outputs": outputs,
        "wall_coordinate_frame": wall_rows["wall_coordinate_frame"],
        "collar_rebind": collar_map, "wall_csv_rebind": wall_csv_report,
        "assembly": assembly, "registered_source_maps": map_report,
        "shared_current_I": {"exact_f32_coordinates": True,
            "source_vertex_rows": int(len(inner["points"])),
            "source_faces": int(len(inner["faces"])),
            "source_point_pin": source_pins["I"],
            "both_shells_registered_same_I_faces": True},
        "elapsed_s": time.monotonic() - started,
        "limitations": ["surface-shell assembly only", "no mass mapping", "no native execution",
                        "the original rim packet was rebound only through exact F32 donor coordinates and current typed O ancestry"],
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # Read back hashes and ensure receipt output pins point at final files.
    for name, item in outputs.items():
        p = output / name
        if sha(p) != item["sha256"]:
            raise ValueError(f"final shell output changed after receipt construction: {name}")
    return receipt

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--rim-receipt", type=Path, required=True)
    ap.add_argument("--parent-view-manifest", type=Path, required=True)
    ap.add_argument("--wall-coordinate-frame", choices=("registered-native-current",),
                    default="registered-native-current")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    receipt = build(args.state, args.rim_receipt, args.parent_view_manifest,
                    args.output, args.wall_coordinate_frame)
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve()),
        "physical_admission": False, "native_steps": 0,
        "skin_faces": receipt["assembly"]["skin_shell"]["faces"],
        "fat_faces": receipt["assembly"]["fat_shell"]["faces"],
        "shared_I_rows": receipt["shared_current_I"]["source_vertex_rows"]}, sort_keys=True))

if __name__ == "__main__":
    main()
