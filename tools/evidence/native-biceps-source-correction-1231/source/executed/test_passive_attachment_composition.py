"""Wire-layout and isolation regression for passive attachment composition."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from numilab_human import passive_attachment_composition as pac
from numilab_human.passive_attachment_composition import compose, bind_anatomy_receipt


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


def anatomy_fixture(source, tmp_path):
    common = {}
    for key in ("map", "polynomials", "domain_boxes"):
        path = tmp_path / (key + ".bin")
        path.write_bytes(key.encode())
        common[key] = {"path": path.name, "sha256": digest(path)}
    organs = tmp_path / "organs.nhanatomy"
    organs.write_bytes(b"unchanged anatomy fixture")
    receipt = tmp_path / "anatomy.json"
    receipt.write_text(json.dumps({
        "schema": "numi.human.resting-anatomy-receipt.v1",
            "payload": {"path": organs.name, "sha256": digest(organs)},
        "mass_geometry_accounting": {"reference_total_mass_kg": 72},
        "provenance": {
            "native_muscle_surfaces": {
                "payload_path": str(source.resolve()), "sha256": digest(source),
                "manifest_path": str(source.with_suffix(".manifest.json")),
                "manifest_sha256": digest(source.with_suffix(".manifest.json")),
                "registration_fingerprint32": "0000002a"},
            "cardiac_geometry_binding": {"common_field": common}}
    }))
    return receipt


def test_composition_binds_new_launch_receipt_without_rebasing_physical_assets(inputs, tmp_path):
    source, candidate, report, _ = inputs
    receipt = anatomy_fixture(source, tmp_path)
    before = receipt.read_bytes()
    out = tmp_path / "out"
    compose(source, out, [(7, candidate, report)])
    result = bind_anatomy_receipt(receipt, out/source.name, out/"resting-anatomy-receipt.json")
    assert receipt.read_bytes() == before
    assert result["mass_geometry_accounting"] == {"reference_total_mass_kg": 72}
    owner = result["provenance"]["native_muscle_surfaces"]
    assert owner["sha256"] == digest(out/source.name)
    assert owner["vertex_count"] == 451 and owner["body_binding_count"] == 300
    assert Path(result["payload"]["path"]) == tmp_path/"organs.nhanatomy"
    for key, entry in result["provenance"]["cardiac_geometry_binding"]["common_field"].items():
        assert Path(entry["path"]) == Path(key+".bin")
        assert digest(out/entry["path"]) == digest(tmp_path/(key+".bin"))
    assert result["provenance"]["passive_attachment_composition_binding"]["changed_stable_ids"] == [7]


@pytest.mark.parametrize("defect", ["wrong_source", "binding", "cardiac"])
def test_composed_launch_receipt_rejects_ownership_drift(inputs, tmp_path, defect):
    source, candidate, report, _ = inputs
    receipt = anatomy_fixture(source, tmp_path)
    out = tmp_path / "out"
    compose(source, out, [(7, candidate, report)])
    manifest_path = out/source.with_suffix(".manifest.json").name
    manifest = json.loads(manifest_path.read_text())
    if defect == "wrong_source":
        manifest["source"]["reference_attachment_composition"]["source_payload_sha256"] = "0"*64
    elif defect == "binding":
        payload = out/source.name
        raw = bytearray(payload.read_bytes())
        raw[64+150*32] ^= 1
        payload.write_bytes(raw)
        manifest["payload"]["sha256"] = digest(payload)
    else:
        (tmp_path/"map.bin").write_bytes(b"changed")
    manifest_path.write_text(json.dumps(manifest))
    output = out/"resting-anatomy-receipt.json"
    with pytest.raises(ValueError):
        bind_anatomy_receipt(receipt, out/source.name, output)
    assert not output.exists()


def _rewrite_payload_row_vertex(path, stable_id, local_vertex, *, position=None, normal=None):
    raw = bytearray(path.read_bytes())
    _, _, nr, nb, nv, ni, _, _ = struct.unpack_from("<8s6I32s", raw)
    records = np.frombuffer(raw, "<u4", nr*8, 64).reshape(-1, 8)
    vertex_start = 64 + nr*32 + nb*36
    row = next(r for r in records if int(r[6]) == stable_id)
    offset = vertex_start + (int(row[2]) + local_vertex)*56
    values = list(struct.unpack_from("<6f4I4f", raw, offset))
    if position is not None:
        values[:3] = position
    if normal is not None:
        values[3:6] = normal
    struct.pack_into("<6f4I4f", raw, offset, *values)
    path.write_bytes(raw)


def _refresh_manifest(payload):
    path = payload.with_suffix(".manifest.json")
    data = json.loads(path.read_text())
    raw = payload.read_bytes()
    _, _, nr, nb, nv, ni, fp, _ = struct.unpack_from("<8s6I32s", raw)
    data["payload"].update(file=payload.name, sha256=digest(payload), bytes=len(raw),
                           surface_count=nr, binding_count=nb, vertex_count=nv,
                           index_count=ni, registration_fingerprint32=f"{fp:08x}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def _biceps_source_fixture(tmp_path, monkeypatch):
    base_dir, parent_dir, patch_dir = (tmp_path/name for name in ("base", "parent", "patch"))
    for directory in (base_dir, parent_dir, patch_dir):
        directory.mkdir()
    payload_name = "surfaces.nhtissue"
    fixture_raw = (tmp_path/"surfaces.nhtissue").read_bytes()
    base = base_dir/payload_name
    base.write_bytes(fixture_raw)
    base_manifest = json.loads((tmp_path/"surfaces.manifest.json").read_text())
    for sid, member, member_id, member_sha in (
        (103, "isa_BP3D_4.0_obj_99/FJ1478.obj", "FJ1478", "1"*64),
        (104, "isa_BP3D_4.0_obj_99/FJ1478M.obj", "FJ1478M", "2"*64)):
        row = base_manifest["source"]["surfaces"][sid]
        row.update(member=member, member_id=member_id, member_sha256=member_sha)
    base_manifest["payload"]["file"] = payload_name
    base.with_suffix(".manifest.json").write_text(json.dumps(base_manifest, indent=2, sort_keys=True)+"\n")

    parent = parent_dir/payload_name
    parent.write_bytes(fixture_raw)
    _rewrite_payload_row_vertex(parent, 64, 0, position=(0.25, 0.0, 0.0))
    parent_manifest = json.loads(base.with_suffix(".manifest.json").read_text())
    parent_manifest["source"]["conforming_edge_refinement_composition"] = {
        "source_payload_path": str(base), "source_payload_sha256": digest(base),
        "stable_id": 64, "binding_table_byte_exact": True}
    parent.with_suffix(".manifest.json").write_text(json.dumps(parent_manifest, indent=2, sort_keys=True)+"\n")
    _refresh_manifest(parent)

    patch = patch_dir/payload_name
    patch.write_bytes(fixture_raw)
    operations = []
    for sid, member_id, member, member_sha in (
        (103, "FJ1478", "isa_BP3D_4.0_obj_99/FJ1478.obj", "1"*64),
        (104, "FJ1478M", "isa_BP3D_4.0_obj_99/FJ1478M.obj", "2"*64)):
        d0 = float(np.float32(-0.5e-6))
        d1 = float(np.float32(+0.5e-6))
        _rewrite_payload_row_vertex(patch, sid, 0, position=(0.0, 0.0, d0), normal=(0.0, 1.0, 0.0))
        _rewrite_payload_row_vertex(patch, sid, 1, position=(1.0, 0.0, d1), normal=(0.0, 1.0, 0.0))
        _rewrite_payload_row_vertex(patch, sid, 2, normal=(0.0, 1.0, 0.0))
        operations.append({
            "stable_id": sid, "member_id": member_id, "source_member": member,
            "source_member_sha256": member_sha,
            "apex_signed_source_axis_displacements_requested_um": [-0.5, 0.5],
            "apex_signed_source_axis_displacements_mm": [-0.0005, 0.0005],
            "actual_float32_apex_displacements_m": {"0": d0, "1": d1},
            "opposite_apex_source_vertex_groups": [[0], [1]],
            "closed_microcomponent_vertex_ids": [0, 1, 2],
            "patch_face_rows": [0], "bindings_and_weights_changed": False,
            "triangles_added_removed_reindexed": False,
            "vertex_rows_added_removed_reindexed": False,
            "component_count_before_after": [2, 2]})
    patch_manifest = json.loads(base.with_suffix(".manifest.json").read_text())
    patch_manifest["source"]["bounded_inferred_biceps_microcomponent_opening"] = {
        "candidate_is_unadmitted": True,
        "nhtiss_records_binding_table_faces_and_vertex_weights_preserved": True,
        "changed_stable_ids": [103, 104], "operation_rows": operations}
    patch.with_suffix(".manifest.json").write_text(json.dumps(patch_manifest, indent=2, sort_keys=True)+"\n")
    _refresh_manifest(patch)

    script = tmp_path/"run_inferred_opening.py"
    script.write_text("# pinned synthetic generator\n")
    report = tmp_path/"source-candidate-audit.json"
    pins = {str(base): digest(base), str(base.with_suffix(".manifest.json")): digest(base.with_suffix(".manifest.json")),
            str(script): digest(script)}
    report.write_text(json.dumps({
        "schema": "numi.human.passive-biceps-microcomponent-inferred-opening-experiment.v1",
        "qualification": "bounded_source_geometry_experiment_only_unadmitted",
        "accepted_pose_forward_verification": {"status": "not_run"},
        "method": {"selected_smallest_passing_opening_um": 0.5,
                   "component_geometry_inference_not_measured": True,
                   "preserve_anatomy_ids_bindings_weights_faces_and_components": True},
        "candidate": {"payload_path": str(patch), "payload_sha256": digest(patch),
                      "manifest_path": str(patch.with_suffix(".manifest.json")),
                      "manifest_sha256": digest(patch.with_suffix(".manifest.json")),
                      "NHTISS_binding_table_byte_exact": True,
                      "non-target_stable_rows_byte_exact": True,
                      "face_rows_added_removed_reordered": False,
                      "other_bytes_byte_exact": True},
        "input_pins_before": pins, "input_pins_after": pins,
        "selected_operation_rows": operations}, indent=2, sort_keys=True)+"\n")

    corrections = {}
    for sid in (103, 104):
        row = next(r for r in pac._read_nhtiss4(patch)["records"] if int(r[6]) == sid)
        row_data = pac._row_slices(pac._read_nhtiss4(patch), row)
        npz_path = tmp_path/f"row-{sid}.npz"
        np.savez(npz_path, **pac._biceps_row_arrays(row_data))
        corrections[str(sid)] = {"path": str(npz_path), "sha256": digest(npz_path)}
    expected_pins = {
        "source_base_payload": {"path": str(base.resolve()), "sha256": digest(base)},
        "source_base_manifest": {"path": str(base.with_suffix(".manifest.json").resolve()),
                                  "sha256": digest(base.with_suffix(".manifest.json"))},
        "source_candidate_payload": {"path": str(patch.resolve()), "sha256": digest(patch)},
        "source_candidate_manifest": {"path": str(patch.with_suffix(".manifest.json").resolve()),
                                       "sha256": digest(patch.with_suffix(".manifest.json"))},
        "source_candidate_report": {"path": str(report.resolve()), "sha256": digest(report)},
        "source_generator_script": {"path": str(script.resolve()), "sha256": digest(script)},
        "direct_parent_payload": {"path": str(parent.resolve()), "sha256": digest(parent)},
        "direct_parent_manifest": {"path": str(parent.with_suffix(".manifest.json").resolve()),
                                   "sha256": digest(parent.with_suffix(".manifest.json"))},
    }
    monkeypatch.setattr(pac, "_BICEPS_SOURCE_CORRECTION_EXPECTED_PINS",
                        json.loads(json.dumps(expected_pins)))
    proof = {"schema": "numi.human.biceps-source-preserving-row-correction.v1",
             "source_base_payload_path": str(base),
             "source_candidate_payload_path": str(patch),
             "source_candidate_report_path": str(report),
             "source_generator_script_path": str(script),
             "expected_pins": expected_pins,
             "row_patch_npz": corrections}
    return base, parent, patch, report, script, operations, proof


def test_biceps_pair_composes_only_as_proof_bound_direct_child(inputs, tmp_path, monkeypatch):
    # Reuse the ordinary binary fixture for NHTISS construction, then create a
    # disjoint prior row-64 edit plus the source-audited two-row patch.
    base,parent,patch,report,script,ops,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    receipt=anatomy_fixture(parent,tmp_path)
    receipt_data=json.loads(receipt.read_text())
    receipt_data["provenance"]["passive_attachment_composition_binding"]={
        "prior_operation":"kept byte-for-byte during additive biceps correction"}
    prior_binding=receipt_data["provenance"]["passive_attachment_composition_binding"]
    receipt.write_text(json.dumps(receipt_data))
    before=parent.read_bytes()
    out=tmp_path/"out"
    result=compose(parent,out,[(103,Path(proof["row_patch_npz"]["103"]["path"]),report),
                               (104,Path(proof["row_patch_npz"]["104"]["path"]),report)],
                   biceps_source_correction=proof)
    assert result["biceps_source_preserving_correction"]["changed_stable_ids"] == [103,104]
    assert parent.read_bytes()==before
    output_receipt=bind_anatomy_receipt(receipt,out/parent.name,out/"resting-anatomy-receipt.json")
    assert output_receipt["provenance"]["passive_attachment_composition_binding"] == prior_binding
    biceps_binding=output_receipt["provenance"]["biceps_source_preserving_correction_binding"]
    assert biceps_binding["changed_stable_ids"] == [103,104]
    assert biceps_binding["source_preserving_correction"]["accepted_pose_forward_status"] == "not_run"
    child=pac._read_nhtiss4(out/parent.name)
    current=pac._read_nhtiss4(parent)
    patched=pac._read_nhtiss4(patch)
    patched_rows={pac._row_slices(patched,row)["stable_id"]:pac._row_slices(patched,row) for row in patched["records"]}
    for row in current["records"]:
        sid=int(row[6])
        assert (pac._row_slices(child,row)["vertex_bytes"] ==
                (patched_rows[sid]["vertex_bytes"] if sid in (103,104)
                 else pac._row_slices(current,row)["vertex_bytes"]))


def test_biceps_pair_requires_proof_and_exact_parent_bridge(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    with pytest.raises(ValueError,match="proof-bound paired biceps"):
        compose(parent,tmp_path/"no-proof",replacements)
    # A current-parent edit to a target row invalidates the source-row bridge.
    _rewrite_payload_row_vertex(parent,103,0,position=(0.125,0.0,0.0))
    _refresh_manifest(parent)
    with pytest.raises(ValueError,match="expected pin mismatch: direct_parent_payload"):
        compose(parent,tmp_path/"bad-parent",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"bad-parent").exists()


def test_biceps_pair_rejects_stale_patch_arrays(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    stale=Path(proof["row_patch_npz"]["103"]["path"])
    with np.load(stale,allow_pickle=False) as z: arrays={k:np.asarray(z[k]).copy() for k in ("vertices6","binding_indices","weights","faces")}
    arrays["weights"][0,0] += np.float32(.125)
    np.savez(stale,**arrays)
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    with pytest.raises(ValueError,match="row patch array hash changed"):
        compose(parent,tmp_path/"bad-npz",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"bad-npz").exists()


def test_biceps_pair_rejects_replacement_not_bound_to_verified_npz(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    replacements=[(103,Path(proof["row_patch_npz"]["104"]["path"]),report),
                  (104,Path(proof["row_patch_npz"]["104"]["path"]),report)]
    with pytest.raises(ValueError,match="replacement row does not match"):
        compose(parent,tmp_path/"wrong-row-file",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"wrong-row-file").exists()


def test_biceps_pair_rejects_changed_expected_source_pin(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    proof["expected_pins"]["source_candidate_report"]["sha256"]="0"*64
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    with pytest.raises(ValueError,match="pins are not the reviewed exact set"):
        compose(parent,tmp_path/"wrong-source-pin",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"wrong-source-pin").exists()


def test_biceps_pair_rejects_signed_zero_npz_difference(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    patch=Path(proof["row_patch_npz"]["103"]["path"])
    with np.load(patch,allow_pickle=False) as z:
        arrays={k:np.asarray(z[k]).copy() for k in ("vertices6","binding_indices","weights","faces")}
    arrays["weights"][0,2]=np.float32(-0.0)
    np.savez(patch,**arrays)
    proof["row_patch_npz"]["103"]["sha256"]=digest(patch)
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    with pytest.raises(ValueError,match="row patch arrays do not exactly match"):
        compose(parent,tmp_path/"signed-zero",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"signed-zero").exists()
