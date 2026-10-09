from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from numilab_human import model
from numilab_human.passive_attachment_composition import (
    _read_nhtiss4,
    _row_slices,
    bind_anatomy_receipt,
    compose_conforming_edge,
)


SCHEMA = "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1"
SOURCE_DIGEST = bytes(range(32))
FP = 42
SID = 64
MATRIX = [
    [0.001, 0.0, 0.0, 0.0],
    [0.0, 0.001, 0.0, 0.0],
    [0.0, 0.0, 0.001, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
ROUTES = [
    {"body": "a", "core_body_index": 228, "kind": "site", "world_m": [0.0, 0.0, 0.0]},
    {"body": "b", "core_body_index": 229, "kind": "site", "world_m": [0.001, 0.0, 0.0]},
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertex_record(point, normal=(0.0, 0.0, 1.0), sparse=((0, 1.0),)):
    indices = [index for index, _ in sparse][:4]
    weights = [weight for _, weight in sparse][:4]
    indices += [0xFFFFFFFF] * (4 - len(indices))
    weights += [0.0] * (4 - len(weights))
    return struct.pack("<6f4I4f", *point, *normal, *indices, *weights)


def surface(stable_id: int, vertices: bytes, faces: list[list[int]], bind_bytes: bytes,
            bindings: list[dict], selected: bool = False):
    return {
        "stable_id": stable_id, "vertices": vertices, "faces": faces,
        "binding_bytes": bind_bytes, "binding_count": len(bindings),
        "surface": {
            "stable_id": stable_id, "member_id": f"member-{stable_id}",
            "member": f"member-{stable_id}.obj", "member_sha256": f"{stable_id:064x}",
            "label": "left vastus lateralis" if selected else f"surface {stable_id}",
            "layer": "muscle", "endpoint_source": "fixture source route",
            "body_bindings": bindings,
            "matched_muscles": [{"name": "vaslat_l"}] if selected else [],
            "body_weight_count": len(bindings),
            "vertex_count": len(vertices) // 56, "triangle_count": len(faces),
            **({"route_binding": {"maximum_vertex_influences": 2,
                                  "maximum_nearest_route_distance_m": 0.001}}
               if selected else {}),
        },
    }


def pack_payload(path: Path, rows: list[dict]) -> dict:
    records, binding_parts, vertex_parts, index_parts = [], [], [], []
    binding_cursor = vertex_cursor = index_cursor = 0
    for row in rows:
        nv = len(row["vertices"]) // 56
        local_faces = np.asarray(row["faces"], dtype="<u4").reshape(-1, 3)
        records.append(struct.pack(
            "<8I", binding_cursor, row["binding_count"], vertex_cursor, nv,
            index_cursor, local_faces.size, row["surface"]["stable_id"], 0,
        ))
        binding_parts.append(row["binding_bytes"])
        vertex_parts.append(row["vertices"])
        index_parts.append((local_faces + vertex_cursor).astype("<u4").tobytes())
        binding_cursor += row["binding_count"]
        vertex_cursor += nv
        index_cursor += local_faces.size
    header = struct.pack("<8s6I32s", b"NHTISS4\0", 5, len(rows), binding_cursor,
                         vertex_cursor, index_cursor, FP, SOURCE_DIGEST)
    raw = header + b"".join(records) + b"".join(binding_parts) + b"".join(vertex_parts) + b"".join(index_parts)
    path.write_bytes(raw)
    return {"file": path.name, "sha256": digest(path), "bytes": len(raw),
            "surface_count": len(rows), "binding_count": binding_cursor,
            "vertex_count": vertex_cursor, "index_count": index_cursor,
            "registration_fingerprint32": f"{FP:08x}"}


def binding_bytes(core_indices: list[int]) -> bytes:
    return b"".join(struct.pack("<I8f", index, 0, 0, 0, 0, 0, 0, 1, 1)
                    for index in core_indices)


def source_fixture(tmp_path: Path):
    source_path = tmp_path / "accepted039.nhtissue"
    rows = []
    surfaces = []
    selected = None
    for sid in range(1, 151):
        binds = [{"myosim_body": "a", "core_body_index": 100 + 2 * (sid - 1)},
                 {"myosim_body": "b", "core_body_index": 101 + 2 * (sid - 1)}]
        bb = binding_bytes([entry["core_body_index"] for entry in binds])
        if sid == SID:
            vertices = b"".join([
                vertex_record((0, 0, 0), sparse=((0, 1.0),)),
                vertex_record((1, 0, 0), sparse=((1, 1.0),)),
                vertex_record((0, 1, 0), sparse=((0, 1.0),)),
                vertex_record((1, -1, 0), sparse=((1, 1.0),)),
            ])
            faces = [[0, 1, 2], [1, 0, 3]]
        else:
            z = float(sid) * 0.01
            vertices = b"".join([
                vertex_record((0, 0, z)), vertex_record((1, 0, z)),
                vertex_record((0, 1, z)),
            ])
            faces = [[0, 1, 2]]
        row = surface(sid, vertices, faces, bb, binds, sid == SID)
        rows.append(row)
        surfaces.append(row["surface"])
        if sid == SID:
            selected = row
    payload_record = pack_payload(source_path, rows)
    source_fields = {
        "bodyparts": {"id": "BodyParts3D", "version": "4.0", "archives": {"fixture": "a" * 64}},
        "registration": {"file": "registration.json", "sha256": "b" * 64},
        "myosim_manifest": {"file": "myosim.json", "sha256": "c" * 64},
        "myosim_source_archive_sha256": SOURCE_DIGEST.hex(),
        "surface_map": {"file": "surface-map.json", "sha256": "d" * 64},
        "reference_attachment_composition": {
            "source_payload_sha256": "7" * 64, "changed_stable_ids": [7, 8],
            "binding_table_byte_exact": True, "physical_route_mass_and_force_state_unchanged": True,
        },
        "downstream_precision_repair": {
            "input_payload_sha256": "9" * 64, "changed_stable_ids": [23],
            "binding_table_byte_exact": True,
        },
        "surfaces": surfaces,
    }
    manifest = {
        "schema": SCHEMA, "status": "fixture",
        "payload": payload_record, "source": source_fields,
        "coverage": {"emitted_surface_count": 150}, "runtime_binding": "unchanged",
    }
    source_manifest_path = source_path.with_suffix(".manifest.json")
    source_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return source_path, manifest, selected


def producer_context(tmp_path: Path, source_path: Path, source_manifest: dict, selected: dict,
                     corrupt: str | None = None):
    import numpy as np
    vertices, normals, weights = [], [], []
    for offset in range(0, len(selected["vertices"]), 56):
        f = struct.unpack_from("<6f4I4f", selected["vertices"], offset)
        vertices.append(list(f[:3])); normals.append(list(f[3:6]))
        dense = [0.0, 0.0]
        for index, value in zip(f[6:10], f[10:14], strict=True):
            if value > 0:
                dense[index] = value
        weights.append(dense)
    faces = selected["faces"]
    global_vertices = [list(v) for v in vertices]
    refined = model._bodyparts_refine_conforming_route_edge(
        SID, (0, 1), vertices, normals, global_vertices, weights, faces,
        MATRIX, ["a", "b"], ROUTES,
    )
    new_vertices, new_normals, _, new_weights, new_faces, operation = refined
    operation.update({
        "source_row_vertex_bytes_sha256": hashlib.sha256(selected["vertices"]).hexdigest(),
        "source_row_local_faces_sha256": hashlib.sha256(
            np.asarray(selected["faces"], dtype="<u4").reshape(-1).tobytes()
        ).hexdigest(),
        "source_row_vertex_byte_count": len(selected["vertices"]),
        "source_row_local_face_index_byte_count": len(selected["faces"]) * 3 * 4,
        "member_id": selected["surface"]["member_id"],
        "member": selected["surface"]["member"],
        "source_obj_sha256": selected["surface"]["member_sha256"],
    })
    candidate_surface = json.loads(json.dumps(selected["surface"]))
    candidate_surface["vertex_count"] = len(new_vertices)
    candidate_surface["triangle_count"] = len(new_faces)
    candidate_surface["conforming_edge_refinement"] = operation
    candidate_surface["route_binding"]["maximum_vertex_influences"] = max(
        candidate_surface["route_binding"]["maximum_vertex_influences"],
        operation["new_vertex_active_route_influence_count"],
    )
    candidate_surface["route_binding"]["maximum_nearest_route_distance_m"] = max(
        candidate_surface["route_binding"]["maximum_nearest_route_distance_m"],
        operation["new_vertex_max_nearest_route_distance_m"],
    )
    candidate_surface["route_binding"]["conforming_refinement_midpoint_count"] = 1
    active = sorted(((v, i) for i, v in enumerate(new_weights[-1]) if v > 1e-8), reverse=True)[:4]
    total = sum(v for v, _ in active)
    sparse_i = [i for _, i in active] + [0xFFFFFFFF] * (4 - len(active))
    sparse_w = [v / total for v, _ in active] + [0.0] * (4 - len(active))
    midpoint = struct.pack("<6f4I4f", *new_vertices[-1], *new_normals[-1], *sparse_i, *sparse_w)
    context_payload = tmp_path / "producer-context.nhtissue"
    context_row = surface(SID, selected["vertices"] + midpoint, new_faces,
                          selected["binding_bytes"], selected["surface"]["body_bindings"], True)
    context_row["surface"] = candidate_surface
    record = pack_payload(context_payload, [context_row])
    context_source = {k: v for k, v in source_manifest["source"].items() if k != "surfaces"}
    context_source["surfaces"] = [candidate_surface]
    if corrupt == "missing_operation":
        candidate_surface.pop("conforming_edge_refinement")
    if corrupt == "wrong_edge":
        operation["edge_local_vertex_ids"] = [0, 2]
    manifest = {
        "schema": SCHEMA,
        "status": "source-derived producer context",
        "payload": record,
        "source": context_source,
        "coverage": {"selected_stable_ids": [SID], "emitted_surface_count": 1,
                     "configured_surface_count": 150},
        "runtime_binding": "producer context only",
    }
    manifest_path = tmp_path / "producer-context.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def receipt_fixture(source_path: Path):
    anatomy = source_path.parent / "anatomy.nhanatomy"
    anatomy.write_bytes(b"unchanged anatomical payload")
    receipt = source_path.parent / "receipt.json"
    source_manifest = source_path.with_suffix(".manifest.json")
    manifest = json.loads(source_manifest.read_text())
    receipt.write_text(json.dumps({
        "schema": "numi.human.resting-anatomy-receipt.v1",
        "payload": {"path": str(anatomy), "sha256": digest(anatomy)},
        "mass_geometry_accounting": {"reference_total_mass_kg": 72},
        "provenance": {"native_muscle_surfaces": {
            "payload_path": str(source_path), "sha256": digest(source_path),
            "manifest_path": str(source_manifest), "manifest_sha256": digest(source_manifest),
            "registration_fingerprint32": manifest["payload"]["registration_fingerprint32"],
            "surface_count": 150, "body_binding_count": manifest["payload"]["binding_count"],
            "vertex_count": manifest["payload"]["vertex_count"],
            "index_count": manifest["payload"]["index_count"],
        }},
    }, indent=2, sort_keys=True) + "\n")
    return receipt


@pytest.fixture
def prepared(tmp_path):
    source, manifest, selected = source_fixture(tmp_path)
    context = producer_context(tmp_path, source, manifest, selected)
    receipt = receipt_fixture(source)
    return source, context, receipt, manifest


def local_rows(path: Path):
    data = _read_nhtiss4(path)
    return {int(r[6]): _row_slices(data, r) for r in data["records"]}, data


def test_conforming_edge_splices_one_row_and_binds_directly_to_039(prepared, tmp_path):
    source, context, receipt, source_manifest = prepared
    output = tmp_path / "composed"
    proof = compose_conforming_edge(source, output, context, SID, (0, 1))
    payload = output / source.name
    out_rows, out_data = local_rows(payload)
    in_rows, in_data = local_rows(source)
    assert proof["inputs_unchanged"] is True
    assert proof["binding_table_byte_exact"] is True
    assert proof["vertex_count"] == in_data["vertex_count"] + 1
    assert proof["index_count"] == in_data["index_count"] + 6
    assert len(proof["unchanged_row_vertex_bytes_and_local_faces"]) == 149
    for sid in in_rows:
        if sid == SID:
            assert out_rows[sid]["vertex_bytes"][:len(in_rows[sid]["vertex_bytes"])] == in_rows[sid]["vertex_bytes"]
            assert out_rows[sid]["vertex_count"] == in_rows[sid]["vertex_count"] + 1
            assert len(out_rows[sid]["local_faces"]) == len(in_rows[sid]["local_faces"]) + 2
        else:
            assert out_rows[sid]["vertex_bytes"] == in_rows[sid]["vertex_bytes"]
            assert out_rows[sid]["local_faces"] == in_rows[sid]["local_faces"]
    output_manifest = json.loads((output / source.with_suffix(".manifest.json").name).read_text())
    assert output_manifest["source"]["reference_attachment_composition"] == source_manifest["source"]["reference_attachment_composition"]
    assert output_manifest["source"]["downstream_precision_repair"] == source_manifest["source"]["downstream_precision_repair"]
    edge = output_manifest["source"]["conforming_edge_refinement_composition"]
    assert edge["source_payload_path"] == str(source.resolve())
    assert edge["source_payload_sha256"] == digest(source)
    result = bind_anatomy_receipt(receipt, payload, output / "resting-anatomy-receipt.json")
    assert result["mass_geometry_accounting"]["reference_total_mass_kg"] == 72
    assert result["provenance"]["native_muscle_surfaces"]["sha256"] == digest(payload)
    assert result["provenance"]["conforming_surface_refinement_binding"]["changed_stable_ids"] == [SID]
    assert "passive_attachment_composition_binding" not in result["provenance"]


@pytest.mark.parametrize("defect", ["missing_operation", "wrong_edge"])
def test_missing_or_wrong_producer_provenance_fails_before_output(tmp_path, defect):
    source, source_manifest, selected = source_fixture(tmp_path)
    context = producer_context(tmp_path, source, source_manifest, selected, corrupt=defect)
    with pytest.raises(ValueError):
        compose_conforming_edge(source, tmp_path / "bad", context, SID, (0, 1))
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("defect", ["unrelated_face", "binding"])
def test_receipt_binding_rejects_changed_unrelated_payload_bytes(prepared, tmp_path, defect):
    source, context, receipt, _ = prepared
    output = tmp_path / "composed"
    compose_conforming_edge(source, output, context, SID, (0, 1))
    payload = output / source.name
    manifest_path = output / source.with_suffix(".manifest.json").name
    raw = bytearray(payload.read_bytes())
    data = _read_nhtiss4(payload)
    if defect == "binding":
        raw[data["binding_start"]] ^= 1
    else:
        row = next(r for r in data["records"] if int(r[6]) == 10)
        first_index, count = int(row[4]), int(row[5])
        offset = data["index_start"] + first_index * 4
        other_vertex = int(row[2]) + ((int(data["indices"][first_index]) - int(row[2]) + 1) % int(row[3]))
        struct.pack_into("<I", raw, offset, other_vertex)
    payload.write_bytes(raw)
    manifest = json.loads(manifest_path.read_text())
    manifest["payload"]["sha256"] = digest(payload)
    manifest["payload"]["bytes"] = payload.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        bind_anatomy_receipt(receipt, payload, output / "bad-receipt.json")
    assert not (output / "bad-receipt.json").exists()


def test_composition_rejects_altered_original_selected_vertex_record(prepared, tmp_path):
    source, context, _, _ = prepared
    raw = bytearray(source.read_bytes())
    data = _read_nhtiss4(source)
    row = next(r for r in data["records"] if int(r[6]) == SID)
    raw[data["vertex_start"] + int(row[2]) * 56] ^= 1
    source.write_bytes(raw)
    manifest_path = source.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["payload"]["sha256"] = digest(source)
    manifest["payload"]["bytes"] = source.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10))
    with pytest.raises(ValueError, match="producer original vertex-record hash"):
        compose_conforming_edge(source, tmp_path / "bad-source", context, SID, (0, 1))
    assert not (tmp_path / "bad-source").exists()


def test_receipt_binding_rejects_source_drift_even_with_rehashed_sidecar(prepared, tmp_path):
    source, context, receipt, _ = prepared
    output = tmp_path / "composed"
    compose_conforming_edge(source, output, context, SID, (0, 1))
    raw = bytearray(source.read_bytes())
    data = _read_nhtiss4(source)
    row = next(r for r in data["records"] if int(r[6]) == SID)
    offset = data["vertex_start"] + int(row[2]) * 56
    raw[offset] ^= 1
    source.write_bytes(raw)
    source_manifest_path = source.with_suffix(".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["payload"]["sha256"] = digest(source)
    source_manifest["payload"]["bytes"] = source.stat().st_size
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        bind_anatomy_receipt(receipt, output / source.name, output / "bad-receipt.json")
    assert not (output / "bad-receipt.json").exists()
