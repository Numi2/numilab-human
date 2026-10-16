import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from numilab_human import composed_bone_body_frame as bones
from numilab_human import resting_shared_partition_verify as shared
from numilab_human import resting_pleura_proxy


def case(tmp_path, monkeypatch):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    normal = np.tile(np.array([0, 0, 1], dtype="<f4"), (4, 1))
    attributes = np.column_stack((points, normal)).astype("<f4")
    records, vertices, indices = [], [], []
    for sid in range(1, 186):
        records.append((20, 4*(sid-1), 4, 12*(sid-1), 12, sid,
                        0., 0., 0., 0., 0., 0., 1., 1., 0))
        bone_attributes = attributes.copy()
        if sid == 49:
            bone_attributes[:, 0] += 2
        vertices.append(bone_attributes.tobytes())
        indices.append((faces+4*(sid-1)).astype("<u4").tobytes())
    raw = (bones._BONE_HEADER.pack(b"NHBONES1", 3, 185, 740, 2220, 123, b"x"*32)
           + b"".join(bones._BONE_RECORD.pack(*row) for row in records)
           + b"".join(vertices)+b"".join(indices))
    bone = tmp_path/"bodyparts3d-myosim-major-bones.nhbones"
    bone.write_bytes(raw)
    mp = bone.with_suffix(".manifest.json")
    mp.write_text(json.dumps({"body_local_bakes": {"rows": [
        {"stable_id": sid, "interface_vertex_pairs":
         [[0, 0], [1, 1], [2, 2]] if sid == 48 else [[0, 4], [1, 5], [2, 6]]}
        for sid in (48, 49)]}}))
    nha = tmp_path/"resting-thorax.nhanatomy"
    nha.write_bytes(b"unit fixture")
    lower = attributes.copy()
    lower[3, 2] = -1
    lower_faces = faces[:, [0, 2, 1]].copy()
    # Two distinct anatomical attachment patches cannot own the same source
    # vertex. Model the second rib's distant closed intercostal component.
    second = lower.copy()
    second[:, 0] += 2
    lower = np.concatenate((lower, second))
    lower_faces = np.concatenate((lower_faces, lower_faces + 4))
    row = dict(body_index=20, layer=1, flags=0, vertices6=lower, faces=lower_faces,
               record=[0, 8])
    monkeypatch.setattr(resting_pleura_proxy, "_parse_payload", lambda raw: (None, None, {313: row}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    invocation = tmp_path/"invocation.json"
    invocation.write_text(json.dumps({"argv": ["runner", "scene", "rigid", str(bone)]}))
    report = dict(status="PASS_EXISTING_SOURCE_OWNER_EXPORT_ONLY",
                  source_xyz_normals_allocations_and_attachment_maps_exact=True,
                  full_existing_185_bone_envelope_loader_passed=True,
                  output_nha_path=str(nha), output_nha_sha256=digest(nha),
                  output_bones_path=str(bone), output_bones_sha256=digest(bone),
                  output_bones_manifest_sha256=digest(mp))
    source = tmp_path/"source-preparation-report.json"
    source.write_text(json.dumps(report))
    context = dict(
        capture_receipt={"common_field_source_anatomy_payload_sha256": digest(nha)},
        capture_receipt_sha256="a"*64,
        compiled_bone_identity_inputs={
            "bone_payload": {"path": str(bone), "sha256": digest(bone)},
            "bone_manifest": {"path": str(mp), "sha256": digest(mp)},
            "native_invocation": {"path": str(invocation), "sha256": digest(invocation)}})
    surfaces = {
        (51004, 48): (points.copy(), faces.copy(), normal.copy(),
                       np.arange(4, dtype="<u4")+4*47+10000),
        (51010, 313): (lower[:, :3].copy(), lower_faces.copy(), np.tile(normal, (2, 1)),
                       np.arange(8, dtype="<u4")+8+20000)}
    return source, digest(source), surfaces, context


def preflight(source, digest, surfaces, context):
    return shared.preflight_shared_muscle_partition(
        surfaces, (51004, 48), (51010, 313), source, digest, capture_context=context)


def test_source_bound_rib_preflight_and_exact_certificate(tmp_path, monkeypatch):
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    proof = preflight(source, digest, surfaces, context)
    assert proof["shared_interface_face_count"] == 1
    assert not proof["exact_predicates_executed"]
    exact = shared.certify_shared_muscle_partition(
        surfaces, (51004, 48), (51010, 313), source, digest, capture_context=context)
    assert exact["exact_native_partition_certificate"]["interiors_disjoint"]
    assert exact["exact_native_partition_certificate"]["shared_interface_opposite_chain"]


def test_rib_requires_native_bound_bone_source(tmp_path, monkeypatch):
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    del context["compiled_bone_identity_inputs"]
    with pytest.raises(Exception, match="native-bound compiled bone"):
        preflight(source, digest, surfaces, context)


def test_rib_rejects_stale_anatomy_and_invocation(tmp_path, monkeypatch):
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    context["capture_receipt"]["common_field_source_anatomy_payload_sha256"] = "0"*64
    with pytest.raises(Exception, match="NHA differs"):
        preflight(source, digest, surfaces, context)
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    invocation = Path(context["compiled_bone_identity_inputs"]["native_invocation"]["path"])
    invocation.write_text("{}")
    with pytest.raises(Exception, match="invocation drifted"):
        preflight(source, digest, surfaces, context)


def test_rib_rejects_world_coordinate_and_ordered_facet_drift(tmp_path, monkeypatch):
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    surfaces[(51010, 313)][0][0, 0] = np.float32(1e-8)
    with pytest.raises(Exception, match="different Float32 positions"):
        preflight(source, digest, surfaces, context)
    source, digest, surfaces, context = case(tmp_path, monkeypatch)
    surfaces[(51010, 313)][1][1, [1, 2]] = surfaces[(51010, 313)][1][1, [2, 1]]
    with pytest.raises(Exception):
        preflight(source, digest, surfaces, context)

