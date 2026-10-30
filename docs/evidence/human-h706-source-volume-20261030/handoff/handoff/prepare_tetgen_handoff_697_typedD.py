#!/usr/bin/env python3
"""Hash-bound preflight and launcher for native-domain-217 paired O/I/D TetGen.

Plan mode creates only a manifest. --execute must be run through run_geometry_serial.py
and requires a candidate readiness receipt. It never claims physical admission.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[2]
BUILD = P / "paired-head-reference-fit-684/run-063-registered-native-wall-frame-002/native-domain-217-source-003/build_native_domain_217.py"
BUILD_SHA = "d7f411c18f91905c6e1d91ab75592ec5782a8852892ac8a1fdb566ee2206df09"
RUNNER = P / "fine-physical-volume-706/handoff/run_native_domain_217_typedD_697.py"
RUNNER_SHA = "7c4554ea49ff626afe03421c4d4080399e4477ed06cc9673f58202a7a6bdbef7"
READY_SCHEMA = "numi.paired-fullbody-tetgen-readiness-697.v1"
GATES = ("surface_self", "paired_interfaces", "bone_clearance", "protected149", "head_rims",
         "canonical_D_surface_quality", "positive_core_seeds", "canonical_D_attachment_surface_routes")
NODE_MAP = "skin-fat-combined-node-map.csv.gz"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def validate_native_capacity_build(receipt_path):
    receipt_path=Path(receipt_path)
    link=json.loads(receipt_path.read_text())
    compile_path=receipt_path.with_name("compile-result.json")
    compiled=json.loads(compile_path.read_text())
    if (link.get("exit_code")!=0 or compiled.get("exit_code")!=0 or
        link.get("capacity_profile")!="full-reference-v1" or
        compiled.get("capacity_profile")!="full-reference-v1" or
        "-DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1" not in compiled.get("compiler_argv",[])):
        raise ValueError("native build is not a successful explicit full-reference capacity compilation")
    inputs=link.get("input_sha256",{})
    if not inputs or inputs!=compiled.get("input_sha256"):
        raise ValueError("native compile/link inputs are missing or inconsistent")
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError("native capacity source changed: "+path)
    manifests=[Path(p) for p in inputs if Path(p).name in ("source-manifest.json","source-manifest-002.json")]
    if len(manifests)!=1:raise ValueError("native capacity build must pin one source manifest")
    source=json.loads(manifests[0].read_text())
    selected=source.get("selected_profile",{})
    if source.get("schema")=="numi.full-reference-native-capacity-source-manifest.v2":
        if (selected.get("node_capacity")!=500000 or selected.get("cell_capacity")!=2000000 or
            selected.get("maximum_candidate_edges")!=12000000 or selected.get("maximum_augmentations")!=4000000 or
            source.get("physical_thresholds_changed") is not False or
            source.get("source330_packet_changed") is not False or source.get("transport_algorithm_changed") is not False):
            raise ValueError("native capacity/work profile does not match the full-reference producer")
        headers=source.get("snapshot_headers",{})
        root=Path(source["snapshot_root"])
        staged={str(root/name):entry["snapshot_002_sha256"] for name,entry in headers.items()}
        overlay=Path(source["overlay_002"])
        if sha(overlay)!=source["overlay_002_sha256"] or inputs.get(str(overlay))!=source["overlay_002_sha256"]:
            raise ValueError("native capacity overlay does not match source manifest")
    else:
        if selected.get("maximum_child_nodes")!=500000 or selected.get("maximum_child_cells")!=2000000:
            raise ValueError("native capacity profile dimensions do not match the producer")
        staged={str(manifests[0].parent/rel):digest for rel,digest in source.get("staged_sha256",{}).items()}
    for name,digest in staged.items():
        if sha(name)!=digest or inputs.get(name)!=digest:
            raise ValueError("native source manifest was not compiled exactly: "+name)
    if len(staged)!=8:raise ValueError("native capacity importer inventory is incomplete")
    argv=link.get("argv",[])
    if argv.count("-o")!=1:raise ValueError("native link receipt has no unique output")
    binary=Path(argv[argv.index("-o")+1])
    if sha(binary)!=link.get("binary_sha256"):raise ValueError("native capacity binary changed")
    return {"profile":"full-reference-v1","maximum_child_nodes":500000,"maximum_child_cells":2000000,
            "paths":{"native_build_receipt":str(receipt_path),"native_compile_receipt":str(compile_path),
                     "native_source_manifest":str(manifests[0]),"native_binary":str(binary)},
            "native_memory_qualification":False,"physical_admission":False}

def load_owner():
    if sha(BUILD) != BUILD_SHA: raise ValueError("pinned 217 owner SHA changed")
    spec = importlib.util.spec_from_file_location("native_domain_217_handoff", BUILD)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def read_pins(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"canonical_protected_row", "mapping_kind", "role", "target_world_x_m", "target_world_y_m", "target_world_z_m"}
        if not r.fieldnames or not required.issubset(r.fieldnames): raise ValueError("protected target CSV schema mismatch")
        rows = list(r)
    if len(rows) != 149 or sorted(int(x["canonical_protected_row"]) for x in rows) != list(range(149)):
        raise ValueError("protected target CSV must contain canonical rows 0..148 exactly once")
    if any(x["role"] not in ("O", "I", "D") for x in rows): raise ValueError("protected target role must be O/I/D")
    kinds = {"head_ring_boundary_target": 135,
             "deep_attachment_original_volume_node_retained": 12,
             "deep_attachment_reference_rebind_to_D_surface_vertex": 2}
    counts = {kind: sum(x["mapping_kind"] == kind for x in rows) for kind in kinds}
    if counts != kinds: raise ValueError(f"protected149 mapping-kind census mismatch: {counts}")
    if any(x["role"] != "D" for x in rows if x["mapping_kind"] != "head_ring_boundary_target"):
        raise ValueError("non-head protected entries must remain D attachment rows")
    attachments = [x for x in rows if x["mapping_kind"] != "head_ring_boundary_target"]
    ids = [x["stable_attachment_id"] for x in attachments]
    if any(not value for value in ids) or len(set(ids)) != 14:
        raise ValueError("canonical attachment identities must be complete and unique")
    return rows

def f32key(x):
    return np.asarray(x, dtype="<f4").reshape(3).tobytes()

def load_696_d_attachment_map(proof_path, deep_path, deep_surface, rows):
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    if proof.get("schema") == "numi.D.combined-operational-reference-preflight.v1":
        path = Path(__file__).with_name("canonical_d_operational_targets.py")
        spec = importlib.util.spec_from_file_location("canonical_d_operational_targets_697", path)
        owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
        return owner.validate(proof_path, deep_path, deep_surface, rows)
    if proof.get("status") != "PASS" or proof.get("physical_admission") is not False or proof.get("runtime") != 0:
        raise ValueError("canonical D target proof is not a preparation-only PASS")
    deep_sha = sha(deep_path)
    candidates = [x for x in proof.get("checks", []) if x.get("sha256") == deep_sha]
    if len(candidates) != 1:
        raise ValueError("deep candidate SHA is not uniquely bound by the canonical-D-696 proof")
    candidate = candidates[0]
    if (candidate.get("changed_vertex_count") != 12 or
        candidate.get("face_connectivity_unchanged") is not True or
        candidate.get("original_noncoordinate_arrays_unchanged") is not True or
        candidate.get("outside_coordinates_exact") is not True):
        raise ValueError("canonical-D-696 preservation proof lacks required row/ancestry invariants")
    points = np.asarray(deep_surface["points"], dtype=np.float32)
    faces = np.asarray(deep_surface["faces"], dtype=np.int64)
    used = set(map(int, np.unique(faces.reshape(-1))))
    attachment_rows = [r for r in rows if r["mapping_kind"] != "head_ring_boundary_target"]
    by_id = {int(x["id"]): x for x in candidate.get("canonical14", [])}
    expected_ids = {int(r["stable_attachment_id"]) for r in attachment_rows}
    if len(by_id) != 14 or set(by_id) != expected_ids:
        raise ValueError("canonical-D-696 proof does not bind the exact 14 CSV attachment identities")
    result = {}
    for r in attachment_rows:
        canonical = int(r["canonical_protected_row"])
        stable_id = int(r["stable_attachment_id"])
        item = by_id[stable_id]
        vertex = int(item["surface_vertex_row"])
        target = np.asarray([float(r[f"target_world_{axis}_m"]) for axis in "xyz"], dtype=np.float32)
        if item.get("target_float32_exact") is not True or int(item.get("incident_face_count", 0)) <= 0:
            raise ValueError(f"canonical D attachment {stable_id} is not proven as an exact used surface vertex")
        if vertex < 0 or vertex >= len(points) or vertex not in used or f32key(points[vertex]) != f32key(target):
            raise ValueError(f"canonical D attachment {stable_id} proof row does not match current candidate Float32 point")
        result[canonical] = {"surface_vertex_row": vertex, "stable_attachment_id": stable_id,
                             "base_volume_node_row": int(r["base_volume_node_row"]),
                             "source330_parent_node_row": int(r["source330_parent_node_row"]),
                             "target_world_m_f32": target.tolist(), "proof_incident_face_count": int(item["incident_face_count"]),
                             "proof_candidate": candidate.get("candidate"), "proof_candidate_sha256": deep_sha,
                             "historical_mapping_kind": r["mapping_kind"]}
    return {"path": str(proof_path), "sha256": sha(proof_path), "candidate": candidate.get("candidate"),
            "candidate_sha256": deep_sha, "attachment_routes": result}

_REGISTERED_SURFACE_ADAPTER = None

def registered_surface_adapter():
    global _REGISTERED_SURFACE_ADAPTER
    if _REGISTERED_SURFACE_ADAPTER is None:
        path = Path(__file__).with_name("registered_surface_handoff.py")
        spec = importlib.util.spec_from_file_location("registered_surface_handoff_697", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REGISTERED_SURFACE_ADAPTER = module
    return _REGISTERED_SURFACE_ADAPTER

def load_prepared_reduced_surface_217(module, path, layer, arrays=None):
    return registered_surface_adapter().load_prepared_reduced_surface_217(module, path, layer, arrays=arrays)

def load_direct_typed_surface_217(module, path, layer):
    return registered_surface_adapter().load_direct_typed_surface_217(module, path, layer)

def load_surface_217(module, path, layer, collar=None, legacy_loader=None):
    """Load an existing 217 surface or adapt the current typed D CSR in memory.

    D696 uses typed union ancestry rather than 217's older cache-row aliases.
    This adapter preserves the original NPZ untouched and maps its full aligned
    union CSR into the 217 surface record without inventing an exact face row.
    """
    load_legacy = legacy_loader or module.load_surface
    # Prepared provenance is an explicit route for every physical boundary,
    # including D reductions whose legacy scalar/union aliases are historical.
    with np.load(path, allow_pickle=False) as z:
        if "lineage_preparation_input_sha256" in z.files:
            arrays = {key: np.asarray(z[key]) for key in z.files}
        else:
            arrays = None
    if arrays is not None:
        return load_prepared_reduced_surface_217(module, path, layer, arrays=arrays)
    if layer != "D":
        with np.load(path, allow_pickle=False) as z:
            prepared = "lineage_preparation_input_sha256" in z.files
            direct_typed = layer in ("O", "I") and "face_source_identity_schema_by_original_id_json" in z.files
        if prepared:
            return load_prepared_reduced_surface_217(module, path, layer)
        if direct_typed:
            return load_direct_typed_surface_217(module,path,layer)
        return load_legacy(path, layer, collar)
    with np.load(path, allow_pickle=False) as z:
        # Conditioned typed surfaces may also expose a scalar primary-row alias.
        # That alias cannot replace the full union ancestry CSR. Prefer the typed
        # route whenever its discriminator is present; missing typed lanes fail
        # below rather than silently falling back to a single source parent.
        legacy = ("face_cache_row" in z.files or "face_source_row" in z.files)
        if legacy and "union_face_ancestry_offsets" not in z.files:
            return load_legacy(path, layer, collar)
        required = {
            "points_world_m", "faces", "face_source_layer", "face_source_body",
            "face_source_bone", "face_source_face_id", "face_source_original_id",
            "face_source_identity0", "face_source_identity1",
            "union_input_run_original_id", "union_input_face_id", "union_input_face_rows",
            "union_face_ancestry_offsets", "union_face_ancestry_run_original_ids",
            "union_face_ancestry_face_ids", "union_face_ancestry_rows",
            "candidate_face_ancestry_offsets", "candidate_face_ancestry_rows",
            "face_source_identity_schema_by_original_id_json",
        }
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"typed D candidate is missing fields: {sorted(missing)}")
        # Materialize each compressed NPZ member once; callers may retain the
        # original archive hash as the authoritative typed-source receipt.
        a = {key: np.asarray(z[key]) for key in z.files}

    points = np.asarray(a["points_world_m"], dtype=np.float32)
    faces = np.asarray(a["faces"], dtype=np.int64)
    count = len(faces)
    row_fields = {
        "source_rows": np.asarray(a["union_input_face_rows"], dtype=np.int64),
        "source_original_ids": np.asarray(a["face_source_original_id"], dtype=np.int64),
        "source_layers": np.asarray(a["face_source_layer"], dtype=np.int8),
        "source_bodies": np.asarray(a["face_source_body"], dtype=np.int32),
        "source_bones": np.asarray(a["face_source_bone"], dtype=np.int32),
        "source_run_original_ids": np.asarray(a["union_input_run_original_id"], dtype=np.int64),
        "source_face_ids": np.asarray(a["face_source_face_id"], dtype=np.int64),
        "source_identity0": np.asarray(a["face_source_identity0"], dtype=np.int64),
        "source_identity1": np.asarray(a["face_source_identity1"], dtype=np.int64),
        "union_input_face_ids": np.asarray(a["union_input_face_id"], dtype=np.int64),
        "union_input_face_rows": np.asarray(a["union_input_face_rows"], dtype=np.int64),
    }
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
            faces.ndim != 2 or faces.shape[1] != 3 or count == 0 or
            int(faces.min()) < 0 or int(faces.max()) >= len(points)):
        raise ValueError("typed D points/faces are malformed")
    if any(v.shape != (count,) for v in row_fields.values()):
        raise ValueError("typed D source identity arrays do not align one-to-one with faces")
    union_offsets = np.asarray(a["union_face_ancestry_offsets"], dtype=np.int64)
    union_rows = np.asarray(a["union_face_ancestry_rows"], dtype=np.int64)
    union_runs = np.asarray(a["union_face_ancestry_run_original_ids"], dtype=np.int64)
    union_ids = np.asarray(a["union_face_ancestry_face_ids"], dtype=np.int64)
    candidate_offsets = np.asarray(a["candidate_face_ancestry_offsets"], dtype=np.int64)
    candidate_rows = np.asarray(a["candidate_face_ancestry_rows"], dtype=np.int64)
    for label, offsets, *members in (
            ("union", union_offsets, union_rows, union_runs, union_ids),
            ("candidate", candidate_offsets, candidate_rows)):
        if (offsets.shape != (count + 1,) or offsets[0] != 0 or
                offsets[-1] != len(members[0]) or np.any(np.diff(offsets) <= 0) or
                any(len(member) != len(members[0]) for member in members[1:])):
            raise ValueError(f"typed D {label} ancestry CSR is malformed")
    schema = json.loads(str(a["face_source_identity_schema_by_original_id_json"].item()))
    missing_schemas = sorted(set(map(str, np.unique(row_fields["source_original_ids"]))) - set(schema))
    if missing_schemas:
        raise ValueError(f"typed D source identity schema lacks original IDs: {missing_schemas[:8]}")
    ancestry = [union_rows[union_offsets[i]:union_offsets[i + 1]].tolist() for i in range(count)]
    ancestry_ids = [union_runs[union_offsets[i]:union_offsets[i + 1]].tolist() for i in range(count)]
    ancestry_parent_rows = [union_ids[union_offsets[i]:union_offsets[i + 1]].tolist() for i in range(count)]
    if any(len(ancestry[i]) != len(ancestry_ids[i]) or len(ancestry[i]) != len(ancestry_parent_rows[i])
           for i in range(count)):
        raise ValueError("typed D union ancestry tuple alignment failed")
    # The scalar source row is the primary union-input face row; rows with no
    # single source remain -1 and are represented by their complete CSR.
    # Exact-source rows stay INVALID because the union's typed multi-source
    # provenance does not assert a unique exact parent face.
    invalid = int(getattr(module, "INVALID", -1))
    result = {"path": str(path), "points": points, "faces": faces,
              **row_fields, "exact_source_rows": np.full(count, invalid, dtype=np.int64),
              "ancestry": ancestry, "ancestry_original_ids": ancestry_ids,
              "ancestry_parent_rows": ancestry_parent_rows,
              "ancestry_kind": np.full(count, 6, dtype=np.int8),
              "typed_identity_schema": schema,
              "typed_source_npz_sha256": sha(path),
              "raw_npz_keys": sorted(a)}
    return result

def route_pins(rows, surfaces, canonical_d_map):
    maps = {}
    for role, surface in surfaces.items():
        points = np.asarray(surface["points"], dtype=np.float32)
        faces = np.asarray(surface["faces"], dtype=np.int64)
        lookup = {}
        for i, p in enumerate(points): lookup.setdefault(f32key(p), []).append(i)
        maps[role] = (lookup, set(map(int, np.unique(faces.reshape(-1)))))
    result = []
    for r in rows:
        canonical, role = int(r["canonical_protected_row"]), r["role"]
        xyz = np.asarray([float(r[f"target_world_{a}_m"]) for a in "xyz"], dtype=np.float32)
        if r["mapping_kind"] != "head_ring_boundary_target":
            mapped = canonical_d_map["attachment_routes"].get(canonical)
            if mapped is None: raise ValueError(f"canonical D attachment row {canonical} lacks a proof-backed route")
            current_xyz = np.asarray(mapped["target_world_m_f32"], dtype=np.float32)
            result.append({"canonical_protected_row": canonical, "role": "D",
                           "mapping_kind": r["mapping_kind"], "surface_vertex_row": mapped["surface_vertex_row"],
                           "target_world_m_f32": current_xyz.tolist(),
                           "historical_target_world_m_f32": xyz.tolist(),
                           "stable_attachment_id": mapped["stable_attachment_id"],
                           "base_volume_node_row": mapped["base_volume_node_row"],
                           "source330_parent_node_row": mapped["source330_parent_node_row"],
                           "route_source": canonical_d_map.get("source_kind", "canonical-D-696-hash-bound-surface-proof"),
                           "proof_candidate_sha256": mapped["proof_candidate_sha256"],
                           "proof_incident_face_count": mapped["proof_incident_face_count"]})
            continue
        lookup, used = maps[role]
        matches = lookup.get(f32key(xyz), [])
        if len(matches) != 1 or matches[0] not in used:
            raise ValueError(f"protected row {canonical} lacks a unique used exact-F32 {role} vertex")
        expected_row = int(r.get("current_surface_vertex_row", "-1"))
        # An operational packet keeps historical row fields intact. Its current
        # unique head rows are regenerated from exact coordinates and then bound
        # to the candidate-specific collar/readiness hashes. Legacy packets still
        # require the original row domain.
        historical_rows = canonical_d_map.get("preserves_immutable_historical_rows", False)
        if role in ("O", "I") and not historical_rows and (expected_row < 0 or matches[0] != expected_row):
            raise ValueError(f"protected row {canonical} exact coordinate disagrees with its declared {role} surface row")
        result.append({"canonical_protected_row": canonical, "role": role,
                       "mapping_kind": r["mapping_kind"], "surface_vertex_row": int(matches[0]),
                       "target_world_m_f32": xyz.tolist(),
                       "stable_attachment_id": int(r["stable_attachment_id"]) if r.get("stable_attachment_id") else None})
    if len(result) != 149: raise ValueError("all 149 protected identities must route to current surface rows")
    return {"surface_routes": sorted(result, key=lambda x: x["canonical_protected_row"]),
            "volume_targets": [], "canonical_D_proof": canonical_d_map}

def validate_readiness(path, hashes):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != READY_SCHEMA or obj.get("status") != "PASS" or obj.get("physical_admission") is not False:
        raise ValueError("candidate-specific readiness receipt is not PASS/non-admission")
    gates = obj.get("gates", {})
    if not isinstance(gates, dict) or any(gates.get(k) != "PASS" for k in GATES):
        raise ValueError("readiness receipt is missing required passing surface gates")
    declared = obj.get("inputs", {})
    check_names = ["outer", "inner", "deep", "collar", "wall_csv", "protected_targets",
                   "canonical_d_proof", "owner_runner", "handoff_script", "capacity_owner", "operational_target_owner", "registered_surface_owner"]
    check_names += [k for k in hashes if k.startswith(("inner_hole_core_", "deep_hole_core_", "native_", "wall_parent_"))]
    for name in check_names:
        entry = declared.get(name)
        entry = entry.get("sha256") if isinstance(entry, dict) else entry
        if entry != hashes[name]: raise ValueError(f"readiness receipt hash mismatch for {name}")
    return obj

def _open_csv(path):
    with Path(path).open("rb") as f: magic = f.read(2)
    return gzip.open(path, "rt", newline="", encoding="utf-8") if magic == b"\x1f\x8b" else Path(path).open("r", newline="", encoding="utf-8")

def make_217_parent_view(source, output, kind):
    """Create a lossless gzip/schema view for the existing 217 reader."""
    with _open_csv(source) as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames: raise ValueError(f"{kind} source CSV has no header")
        names = list(reader.fieldnames)
        required = ({"parent_tet_row","material_index","parent_node0","parent_node1","parent_node2","parent_node3"}
                    if kind == "cells" else {"parent_node_index","current_x_m","current_y_m","current_z_m"})
        if not required.issubset(names): raise ValueError(f"{kind} source330 columns missing {sorted(required-set(names))}")
        add_reference_volume = kind == "cells" and "rest_measure_volume_m3" not in names
        if add_reference_volume and "parent_reference_volume_m3" not in names:
            raise ValueError("source330 cells lack rest_measure_volume_m3 and parent_reference_volume_m3")
        out_names = names + (["rest_measure_volume_m3"] if add_reference_volume and "rest_measure_volume_m3" not in names else [])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists(): raise FileExistsError(f"refusing existing 217 source view {output}")
        with gzip.open(output, "wt", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=out_names, lineterminator="\n")
            writer.writeheader()
            count = 0
            for row in reader:
                if add_reference_volume:
                    row["rest_measure_volume_m3"] = row["parent_reference_volume_m3"]
                if kind == "cells" and float(row.get("rest_measure_volume_m3", "0")) <= 0:
                    raise ValueError(f"nonpositive source330 parent volume at row {row.get('parent_tet_row')}")
                writer.writerow(row)
                count += 1
    if count == 0: raise ValueError(f"{kind} source330 table is empty")
    return {"path": str(output), "sha256": sha(output), "rows": count,
            "source_path": str(source), "source_sha256": sha(source),
            "view": "gzip CSV; cells alias parent_reference_volume_m3 to rest_measure_volume_m3" if kind == "cells" and add_reference_volume else "gzip CSV unchanged columns"}

def volume_capacity_preflight(module, surfaces, skin_wall, fat_wall):
    path = Path(__file__).with_name("native_capacity_preflight.py")
    spec = importlib.util.spec_from_file_location("native_capacity_preflight_697", path)
    capacity = importlib.util.module_from_spec(spec); spec.loader.exec_module(capacity)
    no, ni = len(surfaces["O"]["points"]), len(surfaces["I"]["points"])
    o, _ = module._orient_surface_to_wall(surfaces["O"]["faces"], skin_wall, "capacity-O/skin-wall")
    si, _ = module._orient_surface_to_wall(surfaces["I"]["faces"] + no, skin_wall, "capacity-I/skin-wall")
    fi, _ = module._orient_surface_to_wall(surfaces["I"]["faces"], fat_wall, "capacity-I/fat-wall")
    d, _ = module._orient_surface_to_wall(surfaces["D"]["faces"] + ni, fat_wall, "capacity-D/fat-wall")
    report = capacity.volume_capacity_report(np.vstack((o, si, skin_wall)),
                                             np.vstack((fi, d, fat_wall)), surfaces,
                                             max_nodes=module.MAX_NODES,max_cells=module.MAX_CELLS)
    report["source_path"] = str(path)
    report["source_sha256"] = sha(path)
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("outer","inner","deep","collar","wall-csv","parent-cells","parent-nodes","protected-targets","canonical-D-proof","tetgen-binary","output","manifest-out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--inner-hole-core", type=Path, action="append", required=True)
    ap.add_argument("--deep-hole-core", type=Path, action="append", required=True)
    ap.add_argument("--source330-base", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--full86-weights", type=Path, required=True)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--ready-receipt", type=Path)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--capacity-profile",choices=("legacy","full-reference-500k-nodes-2m-cells"),default="legacy")
    ap.add_argument("--native-build-receipt",type=Path)
    ap.add_argument("--wall-coordinate-frame",choices=("source-current","registered-native-current"),default="source-current")
    ap.add_argument("--wall-parent-cells",type=Path)
    ap.add_argument("--wall-parent-nodes",type=Path)
    a = ap.parse_args()
    if a.output.exists() or a.manifest_out.exists(): raise FileExistsError("refusing to overwrite output or manifest")
    if (a.wall_parent_cells is None)!=(a.wall_parent_nodes is None):
        raise ValueError("wall provenance requires both wall-parent tables or neither")
    if a.wall_coordinate_frame=="registered-native-current" and a.wall_parent_cells is None:
        raise ValueError("registered native walls require explicit authenticated D37 wall-parent tables")

    if a.parent_cells.resolve() != Path(str(a.source330_base) + ".tets.csv").resolve() or \
       a.parent_nodes.resolve() != Path(str(a.source330_base) + ".nodes.csv").resolve():
        raise ValueError("217 parent tables must be the exact source330 base .tets.csv and .nodes.csv assets")
    if sha(RUNNER) != RUNNER_SHA: raise ValueError("typed-D 217 owner runner SHA changed")
    paths = {"outer":a.outer,"inner":a.inner,"deep":a.deep,"collar":a.collar,"wall_csv":a.wall_csv,
             "parent_cells":a.parent_cells,"parent_nodes":a.parent_nodes,"protected_targets":a.protected_targets,
             "canonical_d_proof":a.canonical_D_proof,"tetgen_binary":a.tetgen_binary,"build_script":BUILD,
             "owner_runner":RUNNER,"handoff_script":Path(__file__).resolve(),
             "capacity_owner":Path(__file__).with_name("native_capacity_preflight.py"),
             "operational_target_owner":Path(__file__).with_name("canonical_d_operational_targets.py"),
             "registered_surface_owner":Path(__file__).with_name("registered_surface_handoff.py"),
             "source330_base":a.source330_base,
             "poses":a.poses,"full86_weights":a.full86_weights}
    if a.wall_parent_cells is not None:
        paths.update(wall_parent_cells=a.wall_parent_cells,wall_parent_nodes=a.wall_parent_nodes)
    native_capacity_build = None
    if a.capacity_profile != "legacy":
        if a.native_build_receipt is None:raise ValueError("full reference construction requires a matching native capacity build receipt")
        native_capacity_build = validate_native_capacity_build(a.native_build_receipt)
        paths.update({k:Path(v) for k,v in native_capacity_build["paths"].items()})
    elif a.native_build_receipt is not None:
        raise ValueError("native capacity build receipt is only used with the explicit full-reference profile")
    for i,p in enumerate(a.inner_hole_core): paths[f"inner_hole_core_{i}"] = p
    for i,p in enumerate(a.deep_hole_core): paths[f"deep_hole_core_{i}"] = p
    hashes = {k:sha(p) for k,p in paths.items() if p.is_file()}
    if set(hashes) != set(paths): raise FileNotFoundError(f"missing inputs: {sorted(set(paths)-set(hashes))}")
    module = load_owner()
    module.configure_capacity_profile(a.capacity_profile)
    collar = np.load(a.collar, allow_pickle=False)
    O = load_surface_217(module, a.outer, "O", collar)
    I = load_surface_217(module, a.inner, "I")
    D = load_surface_217(module, a.deep, "D")
    skin_wall, fat_wall, _ = module._map_collar_faces(O,I,D,collar)
    if len(skin_wall) != 90 or len(fat_wall) != 90: raise ValueError("expected 90 O-I and 90 I-D rim facets")
    wall = list(csv.DictReader(a.wall_csv.open(newline="", encoding="utf-8")))
    if len(wall) != 180: raise ValueError("wall provenance must contain 180 paired rim rows")
    surfaces = {"O":{"points":O["points"],"faces":O["faces"]},
                "I":{"points":I["points"],"faces":I["faces"]},
                "D":{"points":D["points"],"faces":D["faces"]}}
    capacity_report = volume_capacity_preflight(module, surfaces, skin_wall, fat_wall)
    pin_rows = read_pins(a.protected_targets)
    canonical_d_map = load_696_d_attachment_map(a.canonical_D_proof, a.deep, D, pin_rows)
    if canonical_d_map.get("paired_inner_sha256") not in (None, hashes["inner"]):
        raise ValueError("canonical D proof does not bind the actual current inner surface")
    pin_routes = route_pins(pin_rows, surfaces, canonical_d_map)
    ready = None
    if a.execute:
        if capacity_report["status"] != "PASS_NECESSARY_BOUNDS_ONLY":
            raise ValueError("refusing impossible native-volume allocation: " + json.dumps(capacity_report, sort_keys=True))
        if a.ready_receipt is None: raise ValueError("--execute requires --ready-receipt")
        ready = validate_readiness(a.ready_receipt, hashes); hashes["ready_receipt"] = sha(a.ready_receipt)
        views_dir = a.manifest_out.parent / (a.manifest_out.stem + ".217-parent-views")
        if views_dir.exists(): raise FileExistsError(f"refusing existing source330 compatibility view dir {views_dir}")
        views_dir.mkdir(parents=True)
        parent_cells_view = views_dir / "parent-cells.csv.gz"
        parent_nodes_view = views_dir / "parent-nodes.csv.gz"
        parent_view_info = {"cells": make_217_parent_view(a.parent_cells, parent_cells_view, "cells"),
                            "nodes": make_217_parent_view(a.parent_nodes, parent_nodes_view, "nodes")}
    else:
        parent_cells_view = a.manifest_out.parent / (a.manifest_out.stem + ".217-parent-views/parent-cells.csv.gz")
        parent_nodes_view = a.manifest_out.parent / (a.manifest_out.stem + ".217-parent-views/parent-nodes.csv.gz")
        parent_view_info = {"cells": {"planned_path": str(parent_cells_view), "source_sha256": hashes["parent_cells"],
                                       "view": "gzip CSV with parent_reference_volume_m3 aliased to rest_measure_volume_m3"},
                            "nodes": {"planned_path": str(parent_nodes_view), "source_sha256": hashes["parent_nodes"],
                                       "view": "gzip CSV unchanged columns"}}
    command = [str(a.python),str(RUNNER),"--outer",str(a.outer),"--inner",str(a.inner),"--deep",str(a.deep),
      "--collar",str(a.collar),"--wall-csv",str(a.wall_csv),"--parent-cells",str(parent_cells_view),
      "--parent-nodes",str(parent_nodes_view),"--output",str(a.output),"--run-tetgen","--tetgen-binary",str(a.tetgen_binary),
      "--capacity-profile",a.capacity_profile,"--wall-coordinate-frame",a.wall_coordinate_frame]
    if a.wall_parent_cells is not None:
        command += ["--wall-parent-cells",str(a.wall_parent_cells),"--wall-parent-nodes",str(a.wall_parent_nodes)]
    for p in a.inner_hole_core: command += ["--inner-hole-core",str(p)]
    for p in a.deep_hole_core: command += ["--deep-hole-core",str(p)]
    manifest = {"schema":"numi.paired-fullbody-tetgen-handoff-697.v1",
      "status":("PREFLIGHT_PASS_READY_FOR_TETGEN" if a.execute else
                ("PREFLIGHT_PASS_PLAN_ONLY" if capacity_report["status"]=="PASS_NECESSARY_BOUNDS_ONLY" else "PLAN_ONLY_FAIL_NATIVE_CAPACITY")),
      "physical_admission":False,"native_runtime":False,
      "inputs":{k:{"path":str(p),"sha256":hashes[k]} for k,p in paths.items()},
      "surface_counts":{k:{"points":len(v["points"]),"faces":len(v["faces"])} for k,v in surfaces.items()},
      "native_capacity_preflight":capacity_report,
      "capacity_profile":module.capacity_profile_receipt(),
      "native_capacity_build":native_capacity_build,
      "typed_D_source":({"candidate_sha256":D["typed_source_npz_sha256"],
        "source_identity_schema_ids":len(D["typed_identity_schema"]),
        "union_face_ancestry_tuples":int(len(D["ancestry_parent_rows"]) and sum(len(x) for x in D["ancestry_parent_rows"])),
        "face_rows_without_single_union_input":int(np.count_nonzero(np.asarray(D["source_rows"]) < 0)),
        "exact_source_face_claims":int(np.count_nonzero(np.asarray(D["exact_source_rows"]) >= 0)),
        "field_map":{"source_row":"union_input_face_rows",
          "source_original_id":"face_source_original_id","source_layer":"face_source_layer",
          "source_body":"face_source_body","source_bone":"face_source_bone",
          "source_run_original_id":"union_input_run_original_id",
          "source_face_id":"face_source_face_id",
          "ancestry_row":"union_face_ancestry_rows",
          "ancestry_run_id":"union_face_ancestry_run_original_ids",
          "ancestry_face_id":"union_face_ancestry_face_ids"},
        "identity0_identity1_and_schema_preserved_in_pinned_input_npz":True}
       if "typed_source_npz_sha256" in D else None),
      "paired_rim":{"skin_faces":len(skin_wall),"fat_faces":len(fat_wall),"wall_rows":len(wall),"single_I_input_domain":True},
      "protected149":{"count":149,"surface_routes":pin_routes["surface_routes"],
        "volume_targets":[],"attachment_route":"all14 canonical attachments use hash-bound current D surface mapping; source IDs and historical CSV fields retained"},
      "canonical_D_attachment_proof":canonical_d_map,
      "tetgen":{"version":"1.6","flags":"-pYq1.2/10AS512","input_facets_fixed":True,
        "quality":"q1.2/10","region_attributes":"A","max_steiner":512,
        "max_nodes_per_material":module.MAX_NODES,"max_cells_per_material":module.MAX_CELLS},
      "readiness":None if ready is None else {"path":str(a.ready_receipt),"sha256":hashes["ready_receipt"],"gates":ready["gates"]},
      "source330_mapping_inputs":{"base":{"path":str(a.source330_base),"sha256":hashes["source330_base"]},
        "poses":{"path":str(a.poses),"sha256":hashes["poses"]},
        "full86_weights":{"path":str(a.full86_weights),"sha256":hashes["full86_weights"]},
        "mass_transport_distance_m":0.019353,"surface_node_registration_limit_m":0.0004},
      "217_parent_schema_views":parent_view_info,
      "command":command,"limitations":["source330 mass transfer is a separate owner","no 671 same-topology or 530-chain claim","not physical admission"]}
    a.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    a.manifest_out.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not a.execute:
        print(json.dumps({"status":manifest["status"],"manifest":str(a.manifest_out),"protected149":149,"command":command},separators=(",",":")))
        return 0
    proc = subprocess.run(command,cwd=BUILD.parent,text=True,capture_output=True)
    (a.manifest_out.parent/(a.manifest_out.stem+".stdout.log")).write_text(proc.stdout)
    (a.manifest_out.parent/(a.manifest_out.stem+".stderr.log")).write_text(proc.stderr)
    final = dict(manifest); final["child_returncode"] = proc.returncode
    adapter_path = a.output/"adapter-receipt.json"
    if proc.returncode or not adapter_path.is_file():
        final["status"]="TETGEN_ADAPTER_FAILED";final["stdout_tail"]=proc.stdout[-4000:];final["stderr_tail"]=proc.stderr[-4000:]
        a.manifest_out.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":final["status"],"manifest":str(a.manifest_out)}));return proc.returncode or 2
    adapter=json.loads(adapter_path.read_text());joint=adapter.get("combined_skin_fat_child_mesh",{});tet=adapter.get("tetgen") or []
    if len(tet)!=2 or any(x.get("status")!="POSITIVE_F32_CELLS_NOT_PHYSICAL_ADMISSION" for x in tet) or joint.get("status")!="COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION":
        final["status"]="TETGEN_POSITIVE_OR_JOINT_GATE_FAILED";a.manifest_out.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":final["status"],"manifest":str(a.manifest_out)}));return 2
    mesh=module.read_tetgen_mesh(a.output/"skin-fat-combined.node",a.output/"skin-fat-combined.ele",None)
    used=set(map(int,mesh["tetrahedra"].reshape(-1))); records={}
    with gzip.open(a.output/NODE_MAP,"rt",newline="") as f:
        for row in csv.DictReader(f):
            layer=row["registered_surface_layer"];v=int(row["registered_surface_vertex_row"])
            if layer in ("O","I","D") and v>=0:
                key=(row["source_partition"],layer,v)
                if key in records: raise ValueError(f"duplicate TetGen surface map {key}")
                records[key]=row
    output_pins=[]
    for pin in pin_routes["surface_routes"]:
        role=pin["role"];v=pin["surface_vertex_row"]
        parts=("skin",) if role=="O" else (("skin","fat") if role=="I" else ("fat",)); ids=[]
        for part in parts:
            row=records.get((part,role,v))
            if row is None: raise ValueError(f"canonical row {pin['canonical_protected_row']} lost {part}/{role} mapping")
            node=int(row["joint_node_id"]);nr=mesh["id_to_row"].get(node)
            if nr is None or nr not in used or f32key(mesh["points"][nr])!=f32key(pin["target_world_m_f32"]):
                raise ValueError(f"canonical row {pin['canonical_protected_row']} changed or unused")
            ids.append(node)
        if len(set(ids))!=1: raise ValueError("paired I protected point does not map to one joint node")
        output_pins.append({**pin,"joint_node_id":ids[0]})
    final["status"]="TETGEN_AND_PROTECTED149_MAP_PASS_NOT_PHYSICAL_ADMISSION"
    final["adapter_status"]=joint["status"];final["protected_output_node_routes"]=output_pins
    final["canonical_D_attachment_proof"] = pin_routes["canonical_D_proof"]
    final["output_hashes"]={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    a.manifest_out.write_text(json.dumps(final,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":final["status"],"manifest":str(a.manifest_out),"adapter_status":joint["status"]},separators=(",",":")))
    return 0

if __name__=="__main__": raise SystemExit(main())
