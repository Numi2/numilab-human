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



def rib50_case(tmp_path, monkeypatch):
    source, _, surfaces, context = case(tmp_path, monkeypatch)
    report = json.loads(source.read_text())
    mp = Path(report["output_bones_path"]).with_suffix(".manifest.json")
    manifest = json.loads(mp.read_text())
    manifest["source_partition_retriangulation"] = {"fixture": True}
    mp.write_text(json.dumps(manifest))
    pairs = tmp_path/"rib50-interface-vertex-pairs.json"
    # Put the third interface on a third disjoint intercostal component.
    old = surfaces[(51010, 313)]
    lower = old[0][:4].copy()
    lower[:, 0] += 4
    xyz = np.concatenate((old[0], lower))
    fs = np.concatenate((old[1], old[1][:4]+8))
    nr = dict(body_index=20, layer=1, flags=0,
              vertices6=np.column_stack((xyz, np.tile([0,0,1], (12,1)))).astype("<f4"),
              faces=fs, record=[0,8])
    monkeypatch.setattr(resting_pleura_proxy, "_parse_payload",
                        lambda raw: (None,None,{313:nr}))
    surfaces[(51010,313)] = (xyz, fs, np.tile([0,0,1],(12,1)).astype("<f4"),
                             np.arange(12,dtype="<u4")+8+20000)
    # Bone50 uses its original allocation in this fixture.
    bone = Path(report["output_bones_path"])
    raw = bytearray(bone.read_bytes())
    _, records, vo, _ = bones._decode_bones(raw)
    br = records[49]
    attrs = np.frombuffer(raw,"<f4",count=24,offset=vo+br[1]*24).reshape(4,6).copy()
    attrs[:,0] += 4
    raw[vo+br[1]*24:vo+(br[1]+4)*24] = attrs.tobytes()
    bone.write_bytes(raw)
    surfaces[(51004,50)] = (attrs[:,:3].copy(), surfaces[(51004,48)][1].copy(),
                            attrs[:,3:].copy(), np.arange(4,dtype="<u4")+br[1]+10000)
    pairs.write_text(json.dumps([[0,8],[1,9],[2,10]]))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report.update(output_bones_sha256=digest(bone),output_bones_manifest_sha256=digest(mp),
                  rib50_interface_vertex_pairs_path=str(pairs),
                  rib50_interface_vertex_pairs_sha256=digest(pairs))
    source.write_text(json.dumps(report))
    for k,p in (("bone_payload",bone),("bone_manifest",mp)):
        context["compiled_bone_identity_inputs"][k]["sha256"] = digest(p)
    return source,digest(source),surfaces,context,pairs


def test_rib50_uses_pinned_source_partition_map_and_exact_oracle(tmp_path,monkeypatch):
    source,digest,surfaces,context,_ = rib50_case(tmp_path,monkeypatch)
    exact = shared.certify_shared_muscle_partition(
        surfaces,(51004,50),(51010,313),source,digest,capture_context=context)
    assert exact["shared_interface_face_count"] == 1
    assert exact["exact_native_partition_certificate"]["interiors_disjoint"]


def test_rib50_rejects_drifted_source_partition_map(tmp_path,monkeypatch):
    source,digest,surfaces,context,pairs = rib50_case(tmp_path,monkeypatch)
    pairs.write_text("[]")
    with pytest.raises(Exception,match="attachment map drifted"):
        shared.preflight_shared_muscle_partition(
            surfaces,(51004,50),(51010,313),source,digest,capture_context=context)


def test_entrypoint_binds_rib50_without_other_rib_declarations(tmp_path,monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rib50_verify_entrypoint",Path(__file__).resolve().parents[1]/"tools/verify_resting_anatomy.py")
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source,digest,surfaces,context,_ = rib50_case(tmp_path,monkeypatch)
    pack=tmp_path/"pack"
    pack.write_bytes(b"unit capture")
    receipt=tmp_path/"capture-receipt.json"
    receipt.write_text(json.dumps(context["capture_receipt"]))
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(tool.verify,"load_capture",lambda *a,**kw:(surfaces,{}))
    monkeypatch.setattr(tool.verify,"load_compiled_bone_identities",
                        lambda *a,**kw:({50:{"stable_id":50}},context["compiled_bone_identity_inputs"]))
    output=tmp_path/"entrypoint-report.json"
    args=["--pack",str(pack),"--pack-sha256",sha(pack),
          "--capture-receipt",str(receipt),"--receipt-sha256",sha(receipt),
          "--changed","51004:50","--shared-partition","51004:50","51010:313",
          str(source),digest,"--preflight-only","--output",str(output)]
    assert tool.main(args+["--bone-artifact",str(tmp_path),
                          "--body-manifest",str(tmp_path/"body-manifest")])==0
    assert json.loads(output.read_text())["shared_partition_preflight"][0]["shared_interface_face_count"]==1
    output=tmp_path/"missing-bone-input-report.json"
    args[-1]=str(output)
    assert tool.main(args)==1
    assert "rib partitions require native-bound" in json.loads(output.read_text())["error"]
