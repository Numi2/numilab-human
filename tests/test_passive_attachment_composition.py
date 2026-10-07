"""Wire-layout and isolation regression for passive attachment composition."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from numilab_human.passive_attachment_composition import compose


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "surfaces.nhtissue"
    count, bindings = 150, 300
    vertices = count * 3
    records = b"".join(struct.pack("<8I", 2*i, 2, 3*i, 3, 3*i, 3, i, 0)
                       for i in range(count))
    binding_bytes = b"".join(struct.pack("<I8f", i+100, 0, 0, 0, 0, 0, 0, 1, 1)
                             for i in range(bindings))
    vertex_bytes = b"".join(
        struct.pack("<6f4I4f", *point, 0, 0, 1, 0, 1, 0, 0, .25, .75, 0, 0)
        for _ in range(count) for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    source.write_bytes(struct.pack("<8s6I32s", b"NHTISS4\0", 5, count, bindings,
                                   vertices, vertices, 42, b"s"*32)
                       + records + binding_bytes + vertex_bytes
                       + np.arange(vertices, dtype="<u4").tobytes())
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
        "payload": {"file": source.name, "sha256": digest(source),
                    "bytes": source.stat().st_size, "surface_count": count,
                    "binding_count": bindings, "vertex_count": vertices,
                    "index_count": vertices, "registration_fingerprint32": "0000002a"},
        "source": {"myosim_source_archive_sha256": (b"s"*32).hex(),
                   "surfaces": [{"stable_id": i, "member_id": f"source-{i}",
                                 "vertex_count": 3, "triangle_count": 1}
                                for i in range(count)]},
        "coverage": {}, "runtime_binding": "original reference binding"}
    source.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    candidate = tmp_path / "row7.npz"
    # Deliberately unused allocation vertex zero, distinctive normal and weights.
    fields = {"vertices6": np.array([[9,9,9,.1,.2,.3], [0,0,0,0,0,1],
                                     [2,0,0,0,0,1], [0,2,0,0,0,1]],dtype="<f4"),
              "binding_indices": np.tile(np.array([1,0,0,0], dtype="<u4"), (4,1)),
              "weights": np.tile(np.array([.8,.2,0,0], dtype="<f4"),(4,1)),
              "faces": np.array([[1,2,3]],dtype="<u4")}
    np.savez(candidate, **fields)
    receipt = tmp_path / "candidate-report.json"
    receipt.write_text(json.dumps({"scope": "test fixture; no anatomical admission"}))
    return source, candidate, receipt, fields


def unpack(path):
    raw = path.read_bytes()
    _, _, nr, nb, nv, ni, _, _ = struct.unpack_from("<8s6I32s", raw)
    records = np.frombuffer(raw, "<u4", nr*8, 64).reshape(-1,8)
    vo = 64 + nr*32 + nb*36
    indices = np.frombuffer(raw, "<u4", ni, vo+nv*56)
    return raw, records, vo, indices


@pytest.mark.parametrize("changed", [(7,), (7, 8, 23)])
def test_wire_layout_allocation_and_unchanged_rows(inputs, tmp_path, changed):
    source, candidate, receipt, fields = inputs
    old, old_records, vo, old_indices = unpack(source)
    report = compose(source, tmp_path/"out", [(sid,candidate,receipt) for sid in changed])
    raw, records, new_vo, indices = unpack(tmp_path/"out"/source.name)
    assert report["binding_table_byte_exact"]
    assert report["unchanged_row_vertex_bytes_and_local_faces"] == [i for i in range(150) if i not in changed]
    assert raw[64+150*32:new_vo] == old[64+150*32:vo]
    for a, b in zip(old_records, records):
        if int(a[6]) in changed:
            start = new_vo + int(b[2])*56
            value = struct.unpack_from("<6f4I4f",raw,start)
            np.testing.assert_array_equal(value[:6],fields["vertices6"][0])
            np.testing.assert_array_equal(value[6:10],fields["binding_indices"][0])
            np.testing.assert_array_equal(value[10:14],fields["weights"][0])
            np.testing.assert_array_equal(indices[int(b[4]):int(b[4]+b[5])] - b[2], [1,2,3])
        else:
            assert raw[new_vo+int(b[2])*56:new_vo+int(b[2]+b[3])*56] == old[vo+int(a[2])*56:vo+int(a[2]+a[3])*56]
            np.testing.assert_array_equal(indices[int(b[4]):int(b[4]+b[5])]-b[2],
                                          old_indices[int(a[4]):int(a[4]+a[5])]-a[2])
    result = json.loads((tmp_path/"out"/source.with_suffix(".manifest.json").name).read_text())
    assert result["source"]["surfaces"][7]["member_id"] == "source-7"
    assert result["payload"]["sha256"] == digest(tmp_path/"out"/source.name)
    assert "not measured-person" in result["evidence_boundary"]


@pytest.mark.parametrize("defect", ["global_index", "fractional_index", "negative_weight", "nan_position", "face_outside"])
def test_reject_invalid_candidate_before_output(inputs, tmp_path, defect):
    source, candidate, receipt, fields = inputs
    if defect == "global_index": fields["binding_indices"][0,0] = 14
    if defect == "fractional_index": fields["binding_indices"] = fields["binding_indices"].astype(float) + .1
    if defect == "negative_weight": fields["weights"][0] = [-.1,1.1,0,0]
    if defect == "nan_position": fields["vertices6"][0,0] = float("nan")
    if defect == "face_outside": fields["faces"][0,0] = 4
    np.savez(candidate, **fields)
    with pytest.raises(ValueError):
        compose(source,tmp_path/"out",[(7,candidate,receipt)])
    assert not (tmp_path/"out").exists()


def test_manifest_mismatch_and_unsupported_row(inputs, tmp_path):
    source, candidate, receipt, _ = inputs
    with pytest.raises(ValueError,match="distinct passive"):
        compose(source,tmp_path/"badrow",[(9,candidate,receipt)])
    manifest=source.with_suffix(".manifest.json")
    data=json.loads(manifest.read_text());data["payload"]["sha256"]="0"*64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="source payload identity"):
        compose(source,tmp_path/"badmanifest",[(7,candidate,receipt)])
    assert not (tmp_path/"badmanifest").exists()
