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


def _rewrite_payload_row_layer(path, stable_id, layer):
    raw = bytearray(path.read_bytes())
    _, _, nr, _, _, _, _, _ = struct.unpack_from("<8s6I32s", raw)
    records = np.frombuffer(raw, "<u4", nr*8, 64).reshape(-1, 8)
    row_index = next(i for i, row in enumerate(records) if int(row[6]) == stable_id)
    struct.pack_into("<I", raw, 64 + row_index*32 + 28, layer)
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


def test_biceps_child_resolves_relative_parent_paths_from_receipt_directory(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    out=tmp_path/"relative-out"
    compose(parent,out,replacements,biceps_source_correction=proof)
    receipt=anatomy_fixture(parent,tmp_path)
    receipt_data=json.loads(receipt.read_text())
    owner=receipt_data["provenance"]["native_muscle_surfaces"]
    owner["payload_path"]=parent.relative_to(receipt.parent).as_posix()
    owner["manifest_path"]=parent.with_suffix(".manifest.json").relative_to(receipt.parent).as_posix()
    receipt.write_text(json.dumps(receipt_data))

    child_receipt=bind_anatomy_receipt(receipt,out/parent.name,out/"child-receipt.json")
    child_owner=child_receipt["provenance"]["native_muscle_surfaces"]
    assert Path(child_owner["payload_path"]) == (out/parent.name).resolve()
    assert child_owner["sha256"] == digest(out/parent.name)
    correction=child_receipt["provenance"]["biceps_source_preserving_correction_binding"]
    assert correction["changed_stable_ids"] == [103,104]


def test_biceps_proof_rejects_declared_manifest_path_that_disagrees_with_payload(inputs, tmp_path, monkeypatch):
    _,parent,_,report,_,_,proof=_biceps_source_fixture(tmp_path, monkeypatch)
    proof["source_base_manifest_path"]=str(tmp_path/"wrong-source.manifest.json")
    replacements=[(sid,Path(proof["row_patch_npz"][str(sid)]["path"]),report) for sid in (103,104)]
    with pytest.raises(ValueError,match="declared source-base manifest path differs"):
        compose(parent,tmp_path/"wrong-manifest",replacements,biceps_source_correction=proof)
    assert not (tmp_path/"wrong-manifest").exists()


def test_legacy_attachment_binding_can_be_replaced_as_before(inputs, tmp_path):
    source,candidate,report,_=inputs
    receipt=anatomy_fixture(source,tmp_path)
    data=json.loads(receipt.read_text())
    data["provenance"]["passive_attachment_composition_binding"]={"prior":"retained legacy behavior"}
    receipt.write_text(json.dumps(data))
    out=tmp_path/"legacy-repeat"
    compose(source,out,[(7,candidate,report)])
    bound=bind_anatomy_receipt(receipt,out/source.name,out/"child-receipt.json")
    assert bound["provenance"]["passive_attachment_composition_binding"]["changed_stable_ids"] == [7]


def _fhl_source_seam_fixture(tmp_path, monkeypatch, *, stale_inherited_count=False):
    """Small synthetic proof bundle for the pinned 27/28 owner branch."""
    import hashlib
    import struct

    root = tmp_path / "fhl"
    root.mkdir()
    parent_dir, subset_dir = root / "parent", root / "subset"
    parent_dir.mkdir(); subset_dir.mkdir()
    name = "surfaces.nhtissue"
    source_identity = {
        "bodyparts": {"id": "bodyparts3d_4", "version": "4.0", "archives": []},
        "registration": {"file": "registration.json", "sha256": ""},
        "myosim_manifest": {"file": "myosim-fullbody-reference.manifest.json",
                            "sha256": "844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7"},
        "myosim_source_archive_sha256": "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975",
        "surface_map": {"file": "bodyparts3d-myosim-surface-map.v1.json",
                        "sha256": "bd08a3d604a754065fd693028c1a307487006f7d13acd40153bda87ed7e423f9"},
    }
    registration = root / "registration.json"
    registration.write_text('{"registration":"fixture"}\n')
    source_identity["registration"]["sha256"] = digest(registration)
    source_digest = bytes.fromhex(source_identity["myosim_source_archive_sha256"])
    fingerprint = int("b1b410ad", 16)
    points = [(0.,0.,0.), (1.,0.,0.), (0.,1.,0.), (0.,0.,1.)]
    tetra_faces = [(0,2,1), (0,1,3), (0,3,2), (1,2,3)]
    parent_rows, subset_rows = [], []
    for sid in range(150):
        if sid in (27, 28):
            verts = points
            faces_parent = tetra_faces[:3]
            faces_candidate = tetra_faces
        else:
            verts = [(0.,0.,0.), (1.,0.,0.), (0.,1.,0.)]
            faces_parent = [(0,1,2)]
            faces_candidate = faces_parent
        parent_rows.append((sid, verts, faces_parent))
        if sid in (23,24,27,28):
            subset_rows.append((sid, verts, faces_candidate))

    def pack_payload(path, rows):
        record_bytes, vertex_bytes, index_values = [], [], []
        binding_bytes = b"".join(struct.pack("<I8f", sid*3+j, 0,0,0,0,0,0,1,1)
                                  for sid, _, _ in rows for j in range(3))
        first_v = first_i = first_b = 0
        row_locations = {}
        for sid, verts, faces in rows:
            row_binding_start = first_b
            row_vertex_start = first_v
            row_index_start = first_i
            bindings = 3
            for _ in verts:
                vertex_bytes.append(struct.pack("<6f4I4f", *_, 0,0,1, 0,1,2,0,
                                                .5,.3,.2,0))
                first_v += 1
            for face in faces:
                index_values.extend(row_vertex_start + int(x) for x in face)
                first_i += 3
            record_bytes.append(struct.pack("<8I", row_binding_start, bindings,
                                            row_vertex_start, len(verts),
                                            row_index_start, len(faces)*3, sid, 1))
            row_locations[sid] = (row_binding_start, bindings, row_vertex_start,
                                  len(verts), row_index_start, len(faces)*3)
            first_b += bindings
        raw = (struct.pack("<8s6I32s", b"NHTISS4\0", 5, len(rows), first_b,
                            first_v, len(index_values), fingerprint, source_digest)
               + b"".join(record_bytes) + binding_bytes + b"".join(vertex_bytes)
               + np.asarray(index_values, dtype="<u4").tobytes())
        path.write_bytes(raw)
        return row_locations, raw

    parent = parent_dir / name
    parent_locs, _ = pack_payload(parent, parent_rows)
    subset = subset_dir / name
    subset_locs, _ = pack_payload(subset, subset_rows)
    fhl_members = {
        27: {"member": "isa_BP3D_4.0_obj_99/FJ1415.obj", "member_id": "FJ1415",
             "member_sha256": "b00ad979e807617650ab158b609e9c2c1032a9483be5e9c9ec5a4796455c809c",
             "label": "right flexor hallucis longus", "route": "fhl_r",
             "bindings": [{"core_body_index":136,"myosim_body":"tibia_r"},
                          {"core_body_index":138,"myosim_body":"calcn_r"},
                          {"core_body_index":139,"myosim_body":"toes_r"}]},
        28: {"member": "isa_BP3D_4.0_obj_99/FJ1415M.obj", "member_id": "FJ1415M",
             "member_sha256": "2cbe6f64ab578445284e188c89f454feb8a3a986e5460639720384e8269e6421",
             "label": "left flexor hallucis longus", "route": "fhl_l",
             "bindings": [{"core_body_index":150,"myosim_body":"tibia_l"},
                          {"core_body_index":152,"myosim_body":"calcn_l"},
                          {"core_body_index":153,"myosim_body":"toes_l"}]},
    }
    def surface_row(sid):
        if sid in fhl_members:
            m = fhl_members[sid]
            bodies = [x["myosim_body"] for x in m["bindings"]]
            return {"stable_id": sid, "member": m["member"], "member_id": m["member_id"],
                    "member_sha256": m["member_sha256"], "label": m["label"], "layer":"muscle",
                    "body_bindings":m["bindings"], "body_weight_count":3,
                    "matched_muscles":[{"name":m["route"],"binding_bodies":bodies,
                        "primary_body":bodies[0],"secondary_body":bodies[-1],
                        "source_actuator_index":347 if sid==27 else 387,"source_route_node_count":7}],
                    "endpoint_source":"all_named_authored_myosim_route_nodes_with_sparse_four_influence_binding",
                    "vertex_count":len(points),"triangle_count":4}
        return {"stable_id":sid,"member":f"fixture-{sid}.obj","member_id":f"fixture-{sid}",
                "member_sha256":"0"*64,"label":f"fixture {sid}","layer":"muscle",
                "body_bindings":[{"core_body_index":sid,"myosim_body":f"body_{sid}"}],
                "body_weight_count":1,"matched_muscles":[],"endpoint_source":"fixture",
                "vertex_count":3,"triangle_count":1}
    def write_manifest(payload, rows, locations, selected):
        raw=payload.read_bytes()
        _,abi,nr,nb,nv,ni,fp,src=struct.unpack_from("<8s6I32s",raw)
        surfaces=[surface_row(sid) for sid,_,_ in rows]
        for surface in surfaces:
            loc=locations[surface["stable_id"]]
            surface["vertex_count"]=loc[3]; surface["triangle_count"]=loc[5]//3
        data={"schema":"numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
              "payload":{"file":payload.name,"sha256":digest(payload),"bytes":len(raw),
                         "surface_count":nr,"binding_count":nb,"vertex_count":nv,"index_count":ni,
                         "registration_fingerprint32":f"{fp:08x}"},
              "source":{**source_identity,"surfaces":surfaces},"coverage":{"selected_stable_ids":selected},
              "runtime_binding":"original named MyoSim binding"}
        payload.with_suffix(".manifest.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
        return data
    parent_manifest=write_manifest(parent,parent_rows,parent_locs,list(range(150)))
    subset_manifest=write_manifest(subset,subset_rows,subset_locs,[23,24,27,28])
    parent_manifest_path=parent.with_suffix(".manifest.json")
    subset_manifest_path=subset.with_suffix(".manifest.json")
    # The real 7b23 parent already carries the earlier biceps composition.
    # Exercise that inherited lineage so the new FHL receipt cannot be
    # mislabeled as a biceps operation or re-append its old correction.
    inherited_biceps={"schema":"fixture.biceps-source-preserving-correction.v1",
                      "changed_stable_ids":[103,104],"prior":"retained parent lineage"}
    parent_manifest["source"]["biceps_source_preserving_correction"] = inherited_biceps
    if stale_inherited_count:
        inherited_row = next(row for row in parent_manifest["source"]["surfaces"]
                             if int(row["stable_id"]) == 23)
        inherited_row["triangle_count"] = 1480
        inherited_row["source_component_selection"] = {"retained_triangle_count": 1480}
        inherited_row["source_precision_repair"] = {"removed_triangle_count": 2}
    parent_manifest_path.write_text(json.dumps(parent_manifest,indent=2,sort_keys=True)+"\n")
    files={
        "registration":registration,
        "model":root/"model.py", "cli":root/"cli.py", "config":root/"anatomy.json",
        "surface_map":root/"surface-map.json", "rigid":root/"core.nhrigid", "myo":root/"muscle.nhmyo",
        "myosim_manifest":root/"myosim.manifest.json", "sourcezip":root/"source.zip",
        "source_lists":root/"source-lists.txt", "partofzip":root/"partof.zip",
        "partof_lists":root/"partof-lists.txt", "overlay":root/"overlay.json",
        "topology_script":root/"topology.py", "self_script":root/"self.py",
        "compare_script":root/"compare.py", "patch_script":root/"patch.py",
        "predicate":root/"predicate.py", "package_init":root/"__init__.py"}
    for key,path in files.items():
        if key != "registration":
            path.write_text(f"fixture:{key}\n")
    declaration_inputs={str(files[k].resolve()):digest(files[k]) for k in
       ("rigid","myo","myosim_manifest","registration","sourcezip","source_lists","overlay",
        "partofzip","partof_lists","anatomy_config") if k in files}
    # Include all source/config paths under stable declaration keys.
    declaration_inputs={str(files[k].resolve()):digest(files[k]) for k in
       ("rigid","myo","myosim_manifest","registration","sourcezip","source_lists","partofzip","partof_lists","overlay","config","surface_map","cli","model")}
    declaration=root/"declaration.json"
    declaration.write_text(json.dumps({"schema":"numi.human.source-seam-regeneration-declaration.v1",
       "expected_current_parent_payload_sha256":digest(parent),
       "expected_current_parent_manifest_sha256":digest(parent_manifest_path),
       "expected_registration_sha256":digest(registration),"input_sha256":declaration_inputs,
       "argv":["python","-m","numilab_human.cli","payload","--registration",str(registration.resolve()),
               "--stable-id","23","--stable-id","24","--stable-id","27","--stable-id","28"]},indent=2)+"\n")
    # reports use immutable, source-bound hashes in the shape expected by the owner.
    def pinmap(paths): return {str(Path(x).resolve()):digest(Path(x)) for x in paths}
    topology=root/"topology.json"
    topo_rows=[]
    for sid in (27,28):
        faces=4
        topo_rows.append({"stable_id":sid,"source_points_moved":False,"new_inferred_faces":0,
          "prior_retained_oriented_source_support_preserved":True,
          "after":{"selection":{"connectivity_basis":"exact_source_coordinate_edges_without_vertex_welding",
                                  "retained_triangle_count":faces},
                   "topology":{"closed_oriented_manifold_candidate":True,"boundary_edge_count":0,
                     "nonmanifold_edge_count":0,"orientation_mismatch_edge_count":0,
                     "duplicate_face_row_count":0,"degenerate_face_rows":[]}},
          "after_existing_opposite_pair_cancellation":{"cancellation":{"source_coordinate_support_preserved":True,
             "oriented_source_chain_preserved":True,"vertices_moved":False,"new_faces_added":False,
             "cancelled_opposite_face_pairs":[]},"topology":{"closed_oriented_manifold_candidate":True,
             "boundary_edge_count":0}}})
    topology.write_text(json.dumps({"input_pins_unchanged":True,"inputs":pinmap([parent,parent_manifest_path,subset,subset_manifest_path,registration]),"surfaces":topo_rows})+"\n")
    self_audit=root/"self.json"
    self_inputs={"parent_payload":{"path":str(parent),"sha256":digest(parent)},
       "parent_manifest":{"path":str(parent_manifest_path),"sha256":digest(parent_manifest_path)},
       "candidate_payload":{"path":str(subset),"sha256":digest(subset)},
       "candidate_manifest":{"path":str(subset_manifest_path),"sha256":digest(subset_manifest_path)},
       "predicate":{"path":str(files["predicate"]),"sha256":digest(files["predicate"])},
       "package_init":{"path":str(files["package_init"]),"sha256":digest(files["package_init"])}}
    audit_rows=[]
    for sid in (27,28):
        audit_rows.append({"stable_id":sid,"face_rows":4,"vertex_records":4,
          "exact_predicate":{"unallowed_self_intersection_pair_count":0,"unallowed_pair_ids":[]},
          "topology_after_exact_f32_coordinate_quotient":{"boundary_edges":0,"nonmanifold_edges":0,
             "orientation_mismatch_edges":0,"duplicate_face_rows":0,"repeated-index_degenerate_faces":0}})
    self_audit.write_text(json.dumps({"schema":"numi.human.regenerated-muscle-exact-f32-self-audit.v1",
      "inputs":self_inputs,"predicate_source_expected_sha256":{"predicate":digest(files["predicate"]),"package_init":digest(files["package_init"])},
      "regenerated_rows":audit_rows})+"\n")
    comparison=root/"row-comparison.json"
    compare_rows=[]
    for sid in (27,28):
        compare_rows.append({"stable_id":sid,"member_id":fhl_members[sid]["member_id"],
          "binding_bytes_exact":True,"body_binding_and_route_manifest_fields_exact":True,
          "old_xyz_multiset_preserved":True,"old_non_normal_attributes_preserved":True,
          "old_oriented_f32_face_coordinates_missing":0,"old_normal_records_recomputed_or_changed_at_same_xyz":0,
          "old_normal_records_retained_at_same_xyz":4,"current_vertex_records":4,"current_face_rows":3,
          "regenerated_face_rows":4,"regenerated_oriented_f32_face_coordinates_added":1})
    comparison.write_text(json.dumps({"schema":"numi.human.source-seam-regenerated-row-comparison.v2",
      "source_topology_comparison_sha256":digest(topology),
      "pins_sha256":pinmap([parent,parent_manifest_path,subset,subset_manifest_path,topology]),
      "row_comparisons":compare_rows})+"\n")
    patch_npz={}
    patch_rows={}
    subdata=pac._read_nhtiss4(subset)
    for sid in (27,28):
        row=next(r for r in subdata["records"] if int(r[6])==sid)
        rowdata=pac._row_slices(subdata,row)
        fields=pac._biceps_row_arrays(rowdata)
        patch=root/f"row-{sid}.npz"; np.savez(patch,**fields)
        patch_npz[str(sid)]={"path":str(patch.resolve()),"sha256":digest(patch)}
        m=fhl_members[sid]
        patch_rows[str(sid)]={"path":str(patch),"sha256":digest(patch),"member_id":m["member_id"],
          "layer":"muscle","body_bindings":m["bindings"],"exact_f32_quotient_boundary_edges":0,
          "exact_f32_self_unallowed_pairs":0,"vertices6_shape":[4,6],"faces_shape":[4,3],"weights_shape":[4,4]}
    patch_report=root/"patch-preparation.json"
    patch_report.write_text(json.dumps({"schema":"numi.human.fhl-source-row-patch-preparation.v1",
       "parent_payload_path":str(parent),"parent_payload_sha256":digest(parent),
       "parent_manifest_path":str(parent_manifest_path),"parent_manifest_sha256":digest(parent_manifest_path),
       "source_subset_payload_path":str(subset),"source_subset_payload_sha256":digest(subset),
       "source_subset_manifest_path":str(subset_manifest_path),"source_subset_manifest_sha256":digest(subset_manifest_path),
       "self_audit_report_path":str(self_audit),"self_audit_report_sha256":digest(self_audit),
       "registration_fingerprint32":"b1b410ad","source_archive_sha256":source_identity["myosim_source_archive_sha256"],
       "rows":patch_rows,"input_sha256":pinmap([parent,parent_manifest_path,subset,subset_manifest_path,self_audit,registration])})+"\n")
    expected_pins={
      "direct_parent_payload":{"path":str(parent.resolve()),"sha256":digest(parent)},
      "direct_parent_manifest":{"path":str(parent_manifest_path.resolve()),"sha256":digest(parent_manifest_path)},
      "source_subset_payload":{"path":str(subset.resolve()),"sha256":digest(subset)},
      "source_subset_manifest":{"path":str(subset_manifest_path.resolve()),"sha256":digest(subset_manifest_path)},
      "row_comparison_report":{"path":str(comparison.resolve()),"sha256":digest(comparison)},
      "self_audit_report":{"path":str(self_audit.resolve()),"sha256":digest(self_audit)},
      "source_topology_report":{"path":str(topology.resolve()),"sha256":digest(topology)},
      "row_patch_preparation_report":{"path":str(patch_report.resolve()),"sha256":digest(patch_report)},
      "source_declaration":{"path":str(declaration.resolve()),"sha256":digest(declaration)},
      "row_patch_27":{"path":patch_npz["27"]["path"],"sha256":patch_npz["27"]["sha256"]},
      "row_patch_28":{"path":patch_npz["28"]["path"],"sha256":patch_npz["28"]["sha256"]},
      "registration":{"path":str(registration.resolve()),"sha256":digest(registration)},}
    monkeypatch.setattr(pac,"_FHL_SOURCE_SEAM_CORRECTION_EXPECTED_PINS",expected_pins)
    monkeypatch.setattr(pac,"_FHL_SOURCE_DECLARATION_INPUTS",declaration_inputs)
    proof={"schema":"numi.human.fhl-source-seam-correction.v1",
      "source_subset_payload_path":str(subset),"source_subset_manifest_path":str(subset_manifest_path),
      "row_comparison_report_path":str(comparison),"self_audit_report_path":str(self_audit),
      "source_topology_report_path":str(topology),"row_patch_preparation_report_path":str(patch_report),
      "source_declaration_path":str(declaration),"registration_path":str(registration),
      "row_patch_npz":patch_npz,"expected_pins":expected_pins}
    return parent,subset,patch_report,proof


def test_fhl_pair_composes_only_as_closed_proof_bound_direct_child(tmp_path, monkeypatch):
    parent,subset,patch_report,proof=_fhl_source_seam_fixture(tmp_path,monkeypatch)
    receipt=anatomy_fixture(parent,tmp_path)
    receipt_data=json.loads(receipt.read_text())
    receipt_data["provenance"]["native_muscle_surfaces"]["registration_fingerprint32"]="b1b410ad"
    receipt_data["provenance"]["passive_attachment_composition_binding"]={"prior":"preserved"}
    prior_biceps={"source_preserving_correction":{"changed_stable_ids":[103,104],"prior":"preserved"}}
    receipt_data["provenance"]["biceps_source_preserving_correction_binding"]=prior_biceps
    receipt.write_text(json.dumps(receipt_data))
    prior=receipt_data["provenance"]["passive_attachment_composition_binding"]
    out=tmp_path/"out"
    result=compose(parent,out,[(27,Path(proof["row_patch_npz"]["27"]["path"]),patch_report),
                               (28,Path(proof["row_patch_npz"]["28"]["path"]),patch_report)],
                   fhl_source_seam_correction=proof)
    assert result["fhl_source_seam_correction"]["changed_stable_ids"]==[27,28]
    manifest=json.loads((out/parent.with_suffix(".manifest.json").name).read_text())
    assert manifest["source"]["fhl_source_seam_correction"]["exact_f32_self_unallowed_pairs"]=={"27":0,"28":0}
    assert manifest["source"]["biceps_source_preserving_correction"] == {
        "schema":"fixture.biceps-source-preserving-correction.v1",
        "changed_stable_ids":[103,104],"prior":"retained parent lineage"}
    bound=bind_anatomy_receipt(receipt,out/parent.name,out/"resting-anatomy-receipt.json")
    assert bound["provenance"]["passive_attachment_composition_binding"]==prior
    assert bound["provenance"]["biceps_source_preserving_correction_binding"]==prior_biceps
    fhl_binding=bound["provenance"]["fhl_source_seam_correction_binding"]
    assert fhl_binding["changed_stable_ids"]==[27,28]
    assert "FHL source-seam correction" in fhl_binding["scope"]
    assert "source_preserving_correction" not in fhl_binding
    assert bound["provenance"]["native_muscle_surfaces"]["sha256"]==digest(out/parent.name)


def test_fhl_composition_reconciles_inherited_serialized_triangle_count(tmp_path, monkeypatch):
    parent, subset, patch_report, proof = _fhl_source_seam_fixture(
        tmp_path, monkeypatch, stale_inherited_count=True)
    receipt = anatomy_fixture(parent, tmp_path)
    receipt_data = json.loads(receipt.read_text())
    receipt_data["provenance"]["native_muscle_surfaces"]["registration_fingerprint32"] = "b1b410ad"
    receipt.write_text(json.dumps(receipt_data))
    parent_manifest = json.loads(parent.with_suffix(".manifest.json").read_text())
    before = next(row for row in parent_manifest["source"]["surfaces"]
                  if int(row["stable_id"]) == 23)
    assert before["triangle_count"] == 1480
    assert before["source_component_selection"]["retained_triangle_count"] == 1480
    assert before["source_precision_repair"]["removed_triangle_count"] == 2

    out = tmp_path / "count-reconciled"
    result = compose(
        parent, out,
        [(27, Path(proof["row_patch_npz"]["27"]["path"]), patch_report),
         (28, Path(proof["row_patch_npz"]["28"]["path"]), patch_report)],
        fhl_source_seam_correction=proof)
    child_manifest = json.loads((out / parent.with_suffix(".manifest.json").name).read_text())
    child_row = next(row for row in child_manifest["source"]["surfaces"]
                     if int(row["stable_id"]) == 23)
    reconciliation = result["serialized_triangle_count_reconciliation"]
    assert reconciliation == [{
        "stable_id": 23,
        "manifest_triangle_count_before": 1480,
        "serialized_triangle_count": 1,
        "serialized_index_count": 3,
        "record_index": 23,
        "reason": "inherited manifest count reconciled to the current NHTISS4 ABI5 row record",
        "preserved_source_component_selection_retained_triangle_count": 1480,
        "preserved_source_precision_repair_removed_triangle_count": 2,
    }]
    assert child_row["triangle_count"] == 1
    assert child_row["source_component_selection"] == before["source_component_selection"]
    assert child_row["source_precision_repair"] == before["source_precision_repair"]
    source_record = next(row for row in pac._read_nhtiss4(parent)["records"] if int(row[6]) == 23)
    child_record = next(row for row in pac._read_nhtiss4(out / parent.name)["records"] if int(row[6]) == 23)
    assert int(source_record[5]) == int(child_record[5]) == 3
    bound = bind_anatomy_receipt(receipt, out / parent.name, out / "resting-anatomy-receipt.json")
    assert bound["provenance"]["native_muscle_surfaces"]["sha256"] == digest(out / parent.name)
    report_on_disk = json.loads((out / "report.json").read_text())
    assert report_on_disk["serialized_triangle_count_reconciliation"] == reconciliation


@pytest.mark.parametrize("defect",["nhtiss_layer","manifest_surface_order"])
def test_fhl_binding_rejects_layer_or_manifest_row_order_drift(tmp_path,monkeypatch,defect):
    parent,subset,patch_report,proof=_fhl_source_seam_fixture(tmp_path,monkeypatch)
    receipt=anatomy_fixture(parent,tmp_path)
    out=tmp_path/"out"
    compose(parent,out,[(27,Path(proof["row_patch_npz"]["27"]["path"]),patch_report),
                        (28,Path(proof["row_patch_npz"]["28"]["path"]),patch_report)],
            fhl_source_seam_correction=proof)
    payload=out/parent.name
    manifest_path=payload.with_suffix(".manifest.json")
    manifest=json.loads(manifest_path.read_text())
    if defect=="nhtiss_layer":
        _rewrite_payload_row_layer(payload,27,0)
        _refresh_manifest(payload)
    else:
        surfaces=manifest["source"]["surfaces"]
        surfaces[27],surfaces[28]=surfaces[28],surfaces[27]
        manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    expected=("FHL child changed a NHTISS row layer" if defect=="nhtiss_layer"
              else "FHL manifest source row order/identity differs from NHTISS")
    with pytest.raises(ValueError,match=expected):
        bind_anatomy_receipt(receipt,payload,out/"receipt.json")
    assert not (out/"receipt.json").exists()


@pytest.mark.parametrize("defect",["wrong_source","wrong_ids","changed_other_row","unclosed_self_report"])
def test_fhl_pair_fails_closed_on_source_ids_other_rows_and_self_proof(tmp_path,monkeypatch,defect):
    parent,subset,patch_report,proof=_fhl_source_seam_fixture(tmp_path,monkeypatch)
    patch=lambda sid: Path(proof["row_patch_npz"][str(sid)]["path"])
    if defect=="wrong_source":
        wrong=tmp_path/"wrong-parent.nhtissue";wrong.write_bytes(parent.read_bytes())
        wrong.with_suffix(".manifest.json").write_bytes(parent.with_suffix(".manifest.json").read_bytes())
        with pytest.raises(ValueError,match="expected pin mismatch: direct_parent_payload"):
            compose(wrong,tmp_path/"wrong-source",[(27,patch(27),patch_report),(28,patch(28),patch_report)],
                    fhl_source_seam_correction=proof)
        assert not (tmp_path/"wrong-source").exists()
        return
    if defect=="wrong_ids":
        with pytest.raises(ValueError,match="proof-bound biceps 103/104 or FHL 27/28 pairs"):
            compose(parent,tmp_path/"wrong-ids",[(27,patch(27),patch_report)],fhl_source_seam_correction=proof)
        assert not (tmp_path/"wrong-ids").exists()
        return
    if defect=="unclosed_self_report":
        self_path=Path(proof["self_audit_report_path"])
        data=json.loads(self_path.read_text())
        data["regenerated_rows"][0]["exact_predicate"]["unallowed_self_intersection_pair_count"]=1
        self_path.write_text(json.dumps(data)+"\n")
        patch_data=json.loads(patch_report.read_text())
        patch_data["self_audit_report_sha256"]=digest(self_path)
        patch_data["input_sha256"][str(self_path.resolve())]=digest(self_path)
        patch_report.write_text(json.dumps(patch_data)+"\n")
        expected=dict(proof["expected_pins"])
        expected["self_audit_report"]={"path":str(self_path.resolve()),"sha256":digest(self_path)}
        expected["row_patch_preparation_report"]={"path":str(patch_report.resolve()),"sha256":digest(patch_report)}
        proof["expected_pins"]=expected
        monkeypatch.setattr(pac,"_FHL_SOURCE_SEAM_CORRECTION_EXPECTED_PINS",expected)
        with pytest.raises(ValueError,match="fails exact-F32 closure/orientation/self audit"):
            compose(parent,tmp_path/"bad-self",[(27,patch(27),patch_report),(28,patch(28),patch_report)],
                    fhl_source_seam_correction=proof)
        assert not (tmp_path/"bad-self").exists()
        return
    out=tmp_path/"out"
    compose(parent,out,[(27,patch(27),patch_report),(28,patch(28),patch_report)],fhl_source_seam_correction=proof)
    payload=out/parent.name
    _rewrite_payload_row_vertex(payload,26,0,position=(0.25,0.0,0.0))
    manifest_path=payload.with_suffix(".manifest.json")
    manifest=json.loads(manifest_path.read_text())
    manifest["payload"]["sha256"]=digest(payload)
    manifest["payload"]["bytes"]=payload.stat().st_size
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    receipt=anatomy_fixture(parent,tmp_path)
    with pytest.raises(ValueError,match="FHL child changed a non-target row"):
        bind_anatomy_receipt(receipt,payload,out/"receipt.json")
    assert not (out/"receipt.json").exists()
