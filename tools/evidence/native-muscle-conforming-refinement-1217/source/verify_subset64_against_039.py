#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216")
BASE = Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001")
PRODUCER = ROOT / "producer-attempt-002"
SUBSET = PRODUCER / "producer-subset-001"
BASE_PAYLOAD = BASE / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
BASE_MANIFEST = BASE / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
SUB_PAYLOAD = SUBSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
SUB_MANIFEST = SUBSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
SNAPSHOT = Path("/Users/n/numi-human-conforming-refinement-source-1216-attempt2")
HEADER = struct.Struct("<8s6I32s")
RECORD = struct.Struct("<8I")
BINDING = struct.Struct("<I8f")
VERTEX = struct.Struct("<6f4I4f")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_payload(path: Path) -> dict:
    raw = path.read_bytes()
    magic, abi, nrecords, nbindings, nvertices, nindices, regfp, source = HEADER.unpack_from(raw)
    if magic != b"NHTISS4\0":
        raise ValueError(f"unexpected NHTISS magic: {magic!r}")
    cursor = HEADER.size
    records = [RECORD.unpack_from(raw, cursor + i * RECORD.size) for i in range(nrecords)]
    cursor += nrecords * RECORD.size
    bindings_offset = cursor
    bindings = [BINDING.unpack_from(raw, cursor + i * BINDING.size) for i in range(nbindings)]
    cursor += nbindings * BINDING.size
    vertices_offset = cursor
    vertices = [VERTEX.unpack_from(raw, cursor + i * VERTEX.size) for i in range(nvertices)]
    cursor += nvertices * VERTEX.size
    indices = list(struct.unpack_from(f"<{nindices}I", raw, cursor)) if nindices else []
    cursor += nindices * 4
    if cursor != len(raw):
        raise ValueError(f"payload length mismatch: decoded {cursor}, actual {len(raw)}")
    return {
        "raw": raw, "abi": abi, "nrecords": nrecords, "nbindings": nbindings,
        "nvertices": nvertices, "nindices": nindices, "registration_fingerprint32": regfp,
        "source_sha256": source.hex(), "records": records, "bindings": bindings,
        "vertices": vertices, "indices": indices, "bindings_offset": bindings_offset,
        "vertices_offset": vertices_offset,
    }


def stable_row(payload: dict, stable_id: int) -> dict:
    matched = [(i, record) for i, record in enumerate(payload["records"]) if record[6] == stable_id]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one stable row {stable_id}, found {len(matched)}")
    row_index, record = matched[0]
    first_binding, binding_count, first_vertex, vertex_count, first_index, index_count, sid, layer = record
    if index_count % 3:
        raise ValueError(f"stable row {sid} has nontriangle index count")
    vertex_bytes = payload["raw"][
        payload["vertices_offset"] + first_vertex * VERTEX.size:
        payload["vertices_offset"] + (first_vertex + vertex_count) * VERTEX.size
    ]
    binding_bytes = payload["raw"][
        payload["bindings_offset"] + first_binding * BINDING.size:
        payload["bindings_offset"] + (first_binding + binding_count) * BINDING.size
    ]
    global_indices = payload["indices"][first_index:first_index + index_count]
    if any(index < first_vertex or index >= first_vertex + vertex_count for index in global_indices):
        raise ValueError(f"stable row {sid} has an index outside its vertex span")
    local_indices = [index - first_vertex for index in global_indices]
    faces = [local_indices[i:i + 3] for i in range(0, len(local_indices), 3)]
    face_bytes = struct.pack(f"<{len(local_indices)}I", *local_indices)
    return {
        "row_index": row_index, "record": record, "binding_count": binding_count,
        "vertex_count": vertex_count, "index_count": index_count // 3,
        "vertex_bytes": vertex_bytes, "binding_bytes": binding_bytes,
        "faces": faces, "face_bytes": face_bytes,
    }


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {reason}")


def main() -> int:
    base = parse_payload(BASE_PAYLOAD)
    subset = parse_payload(SUB_PAYLOAD)
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    subset_manifest = json.loads(SUB_MANIFEST.read_text())
    base64 = stable_row(base, 64)
    sub64 = stable_row(subset, 64)
    operation = next(
        surface["conforming_edge_refinement"]
        for surface in subset_manifest["source"]["surfaces"]
        if surface["stable_id"] == 64
    )
    require(subset_manifest["coverage"]["selected_stable_ids"] == [64], "subset must select only stable ID64")
    require(subset_manifest["coverage"]["emitted_surface_count"] == 1, "subset must emit only one row")
    require(subset_manifest["coverage"]["configured_surface_count"] == 150, "configured source count must remain 150")
    require(len(base_manifest["source"]["surfaces"]) == 150, "accepted039 must contain 150 rows")
    for key in ("bodyparts", "myosim_manifest", "myosim_source_archive_sha256", "registration", "surface_map"):
        require(base_manifest["source"].get(key) == subset_manifest["source"].get(key),
                f"producer source context differs from accepted039: {key}")
    base_surface = next(s for s in base_manifest["source"]["surfaces"] if s["stable_id"] == 64)
    subset_surface = next(s for s in subset_manifest["source"]["surfaces"] if s["stable_id"] == 64)
    for key in ("member_id", "member", "member_sha256", "layer", "body_bindings", "matched_muscles"):
        require(base_surface.get(key) == subset_surface.get(key), f"stable64 source identity differs: {key}")
    require(base["registration_fingerprint32"] == subset["registration_fingerprint32"],
            "registration fingerprint changed")
    require(base["source_sha256"] == subset["source_sha256"], "MyoSim source archive fingerprint changed")
    require(operation["route_body_binding_order"] == [
        binding["myosim_body"] for binding in subset_surface["body_bindings"]
    ], "operation route order differs from stable64 body binding order")
    require(base64["vertex_count"] == operation["old_vertex_count"], "old stable64 vertex count mismatch")
    require(len(base64["vertex_bytes"]) == operation["source_row_vertex_byte_count"],
            "old stable64 vertex-byte count mismatch")
    require(base64["vertex_bytes"] == sub64["vertex_bytes"][:len(base64["vertex_bytes"])],
            "existing stable64 vertex records differ from accepted039")
    require(sha_bytes(base64["vertex_bytes"]) == operation["source_row_vertex_bytes_sha256"],
            "row vertex hash does not bind accepted039")
    require(base64["binding_bytes"] == sub64["binding_bytes"], "stable64 body-binding records changed")
    require(base64["binding_count"] == sub64["binding_count"], "stable64 binding count changed")
    require(len(base64["faces"]) == operation["old_face_count"], "old stable64 face count mismatch")
    require(len(sub64["faces"]) == operation["new_face_count"], "new stable64 face count mismatch")
    require(len(base64["face_bytes"]) == operation["source_row_local_face_index_byte_count"],
            "old stable64 local-face byte count mismatch")
    require(sha_bytes(base64["face_bytes"]) == operation["source_row_local_faces_sha256"],
            "row local-face hash does not bind accepted039")
    parent_rows = [5197, 5198]
    require(all(base64["faces"][i] == sub64["faces"][i] for i in range(len(base64["faces"]))
                if i not in parent_rows), "an unrelated stable64 face changed")
    require(sub64["faces"][5197] == [2597, 2599, 3974], "first retained child differs")
    require(sub64["faces"][5198] == [2597, 3974, 2788], "second retained child differs")
    require(sub64["faces"][6688] == [3974, 2599, 3054], "first appended child differs")
    require(sub64["faces"][6689] == [3974, 3054, 2788], "second appended child differs")
    require(sub64["vertex_count"] == base64["vertex_count"] + 1, "expected one appended vertex")
    require(sub64["index_count"] == base64["index_count"] + 2, "expected two appended child faces")
    midpoint_vertex = subset["vertices"][subset["records"][sub64["row_index"]][2] + operation["new_vertex_local_id"]]
    weights = operation["new_vertex_route_weights_double"]
    expected = tuple(struct.unpack("<4f", struct.pack("<4f", *(weights + [0.0] * (4 - len(weights))))))[:len(weights)]
    require(tuple(midpoint_vertex[10:10 + len(weights)]) == expected, "midpoint serialized weights differ from owner result")
    report = {
        "status": "pass",
        "scope": {"accepted039_row_count": 150, "producer_selected_stable_ids": [64],
                  "producer_emitted_row_count": 1, "configured_row_count": 150},
        "accepted039": {"manifest": str(BASE_MANIFEST), "manifest_sha256": sha(BASE_MANIFEST),
                        "payload": str(BASE_PAYLOAD), "payload_sha256": sha(BASE_PAYLOAD)},
        "producer": {"manifest": str(SUB_MANIFEST), "manifest_sha256": sha(SUB_MANIFEST),
                     "payload": str(SUB_PAYLOAD), "payload_sha256": sha(SUB_PAYLOAD),
                     "declaration_sha256": sha(PRODUCER / "producer-declaration.json"),
                     "execution_sha256": sha(PRODUCER / "producer-execution.json")},
        "source": {"revision": "38e8a4ac6f3ceda441f2c38683c687d2c298a1a4",
                   "source_revision_sha256": sha(SNAPSHOT / "source-revision.json"),
                   "model_sha256": sha(SNAPSHOT / "src/numilab_human/model.py"),
                   "cli_sha256": sha(SNAPSHOT / "src/numilab_human/cli.py")},
        "verified": {
            "old_vertex_count": base64["vertex_count"],
            "old_vertex_record_bytes": len(base64["vertex_bytes"]),
            "old_vertex_records_byte_identical": True,
            "source_vertex_byte_sha256": operation["source_row_vertex_bytes_sha256"],
            "binding_records_byte_identical": True,
            "binding_record_count": base64["binding_count"],
            "old_face_count": len(base64["faces"]), "new_face_count": len(sub64["faces"]),
            "unchanged_faces_outside_parent_rows": len(base64["faces"]) - len(parent_rows),
            "changed_parent_face_rows": parent_rows, "appended_child_rows": [6688, 6689],
            "source_local_face_sha256": operation["source_row_local_faces_sha256"],
            "source_member_and_binding_identity_match_039": True,
            "source_manifest_registration_archive_and_map_match_039": True,
            "registration_fingerprint32": f"{base['registration_fingerprint32']:08x}",
            "midpoint_local_id": operation["new_vertex_local_id"],
            "midpoint_f32_m": operation["new_vertex_source_f32_m"],
            "route_binding_order": operation["route_body_binding_order"],
            "midpoint_route_weights_double": weights,
            "midpoint_transform": operation["global_source_mm_to_myosim_world_m"],
            "authored_route_points": operation["authored_route_points"],
        },
    }
    report_path = ROOT / "subset64-accepted039-verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "sha256": sha(report_path), "status": "pass"}, indent=2))
    return 0


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
