from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from numilab_human import resting_brachiocephalic_3d_junction as junction
from numilab_human import (
    resting_brachiocephalic_current_capture_network_runner as runner,
)
from numilab_human import resting_brachiocephalic_geometry_gate as geometry


class _CachedSurface:
    def __init__(self, key, positions, faces, source_vertex_ids):
        self.key = key
        self.positions = np.asarray(positions, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.source_vertex_ids = np.asarray(source_vertex_ids, dtype=np.uint64)
        self.normals = None
        self.quotient_topology = {"face_component_count": 1}
        self.closed = True
        self.self_calls = 0

    def self_audit(self):
        self.self_calls += 1
        return {"count": 0, "pairs": []}

    def diagnostics(self):
        return {"identity": list(self.key)}


class _Verifier:
    PreparedSurface = _CachedSurface

    def __init__(self):
        self.prepare_calls = []

    def prepare_surface(self, key, positions, faces, *, normals=None,
                        source_vertex_ids=None):
        self.prepare_calls.append(tuple(key))
        return _CachedSurface(key, positions, faces, source_vertex_ids)


def _runtime_fixture():
    keys = runner.NETWORK_KEYS
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    gids = np.asarray([310, 311, 312], dtype=np.uint64)
    value = (positions, faces, None, gids)
    candidates = {key: value for key in keys}
    baselines = {key: value for key in keys}
    source_rows = {
        key[1]: {
            "record": (20, 10, 3, 0, 3, key[1], 2, 0),
            "faces": faces,
            "body_index": 20,
            "layer": 2,
            "vertices6": np.zeros((3, 6), dtype=np.float32),
        }
        for key in keys
    }
    return {
        "context": {
            "surfaces": dict(candidates),
            "capture_receipt": {
                "skin_source_mapping": {
                    "pack_first_vertex": 100,
                    "source_vertex_count": 200,
                },
                "vertex_count": 1000,
            },
        },
        "candidate_surfaces": candidates,
        "baseline_surfaces": baselines,
        "source_rows": source_rows,
        "source_map": {},
        "edge_evidence": {},
        "separate_evidence": {},
    }


def _patch_input_owner(monkeypatch):
    links = junction.declared_network_links()
    monkeypatch.setattr(
        runner.network_owner, "prepare_network_link_audit_inputs",
        lambda *args, **kwargs: tuple({
            "bridge_key": edge[0],
            "target_key": edge[1],
        } for edge in links))



def _freeze_test_addendum(output, bundle, proof, source_proof=None):
    preflight_path = output / "source-and-capture-preflight.json"
    progress_path = output / "progress.json"
    report = {
        "status": "VERIFIED_BASELINE_ONLY_475_479_EXACT_P6_RAT_COLLAR_ADDENDUM",
        "capture_input_bundle_path": str(bundle.resolve()),
        "capture_input_bundle_sha256": runner._sha_file(bundle.resolve()),
        "source_preflight_path": str(preflight_path.resolve()),
        "source_preflight_sha256": runner._sha_file(preflight_path),
        "runner_code_pins": runner._runtime_code_pins(),
        "verified_addendum": proof,
    }
    runner._write_json(output / runner.BASELINE_ADDENDUM_NAME, report)
    progress = json.loads(progress_path.read_text())
    progress.update({
        "status": "BASELINE_COLLAR_ADDENDUM_VERIFIED",
        "baseline_addendum_path": str(
            (output / runner.BASELINE_ADDENDUM_NAME).resolve()),
        "baseline_addendum_sha256": runner._sha_file(
            output / runner.BASELINE_ADDENDUM_NAME),
        "geometry_predicates_run": True,
        "candidate_surface_predicates_run": False,
        "baseline_exact_p6_predicate_run": True,
    })
    if source_proof is None:
        source_proof = {
            "passed": True,
            "typed_link": [[51011, 525], [51011, 469]],
        }
    source_report = runner.network_owner.SOURCE_BASELINE_OBSERVATION_REPORT.resolve()
    source_sha = runner.network_owner.SOURCE_BASELINE_OBSERVATION_REPORT_SHA256
    source_addendum = {
        "status": "VERIFIED_SOURCE_BASELINE_ONLY_525_469_DECLARED_COLLAR_ADDITIONS",
        "capture_input_bundle_path": str(bundle.resolve()),
        "capture_input_bundle_sha256": runner._sha_file(bundle.resolve()),
        "source_preflight_path": str(preflight_path.resolve()),
        "source_preflight_sha256": runner._sha_file(preflight_path),
        "runner_code_pins": runner._runtime_code_pins(),
        "source_baseline_report_path": str(source_report),
        "source_baseline_report_sha256": source_sha,
        "verified_addendum": source_proof,
    }
    source_addendum_path = output / runner.SOURCE_BASELINE_ADDENDUM_NAME
    runner._write_json(source_addendum_path, source_addendum)
    progress.update({
        "status": "SOURCE_BASELINE_COLLAR_ADDITIONS_VERIFIED",
        "source_baseline_addendum_path": str(source_addendum_path.resolve()),
        "source_baseline_addendum_sha256": runner._sha_file(source_addendum_path),
        "source_baseline_exact_p6_predicate_run": True,
    })
    runner._write_json(progress_path, progress)


def test_native_nha_gid_base_offset_uses_receipt_ranges_and_rejects_bad_types():
    context = {"capture_receipt": {"skin_source_mapping": {"pack_first_vertex": 712060, "source_vertex_count": 54949}, "vertex_count": 2854518}}
    proof = runner._native_nha_gid_base_offset(context)
    assert proof["native_nha_gid_base_offset"] == 767009
    assert proof["bound_to_validated_actual_capture_receipt"] is True
    context["capture_receipt"]["skin_source_mapping"]["pack_first_vertex"] = True
    with pytest.raises(ValueError, match="invalid skin/anatomy vertex ranges"):
        runner._native_nha_gid_base_offset(context)

def test_source_binding_preflight_freezes_all_typed_rows_without_geometry(monkeypatch):
    links = [
        {"bridge_key": edge[0], "target_key": edge[1]}
        for edge in runner.network_owner.declared_network_links()
    ]
    monkeypatch.setattr(runner.network_owner, "prepare_network_link_audit_inputs",
                        lambda *_args, **_kwargs: links)
    report = runner._source_binding_preflight(_runtime_fixture())
    assert report["status"] == "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE"
    assert report["typed_network_surface_count"] == 17
    assert report["requested_source_link_count"] == 15
    assert report["candidate_and_independent_baseline_geometry_predicates_run"] is False
    rows = report["typed_row_inventory"]["source_face_ancestry_by_typed_row"]
    assert len(rows) == 17
    assert {row["native_nha_gid_base_offset"] for row in rows.values()} == {300}

def test_cached_prepared_surfaces_are_reused_by_link_owner(monkeypatch):
    fake_verifier = _Verifier()
    monkeypatch.setattr(runner, "verifier", fake_verifier)
    _patch_input_owner(monkeypatch)
    events = []

    def link_audit(verifier, manifold, inputs, *, expected_nha_gid_base_offset):
        events.append("links")
        assert expected_nha_gid_base_offset == 300
        assert len(inputs) == 15
        assert all(isinstance(item["bridge_surface"], _CachedSurface)
                   and isinstance(item["target_surface"], _CachedSurface)
                   and isinstance(item["baseline_bridge_surface"], _CachedSurface)
                   and isinstance(item["baseline_target_surface"], _CachedSurface)
                   for item in inputs)
        return {"passed": False, "first_failed_link": [[51011, 7], [51011, 471]]}

    def all_target(*args, **kwargs):
        events.append("all_target")
        raise AssertionError("all-target screen must stop after continuity failure")

    monkeypatch.setattr(
        runner.network_owner, "audit_brachiocephalic_network_links_3d", link_audit)
    monkeypatch.setattr(runner.preparation, "run_all_target_network_screen", all_target)
    report = runner.execute_ordered_network_gates(_runtime_fixture(), manifold3d=object())
    assert report["status"] == "STOPPED_AT_FIRST_FAILED_NETWORK_LINK"
    assert report["all_target_collision_gate"]["status"] == (
        "NOT_RUN_NETWORK_CONTINUITY_FAILED")
    assert events == ["links"]
    assert len(fake_verifier.prepare_calls) == 2 * len(runner.NETWORK_KEYS)


def test_all_target_runs_only_after_all_fifteen_links_pass_and_reuses_candidates(monkeypatch):
    fake_verifier = _Verifier()
    monkeypatch.setattr(runner, "verifier", fake_verifier)
    _patch_input_owner(monkeypatch)
    events = []

    def link_audit(verifier, manifold, inputs, *, expected_nha_gid_base_offset):
        events.append("links")
        assert expected_nha_gid_base_offset == 300
        assert len(inputs) == 15
        return {"passed": True, "evaluated_link_count": 15}

    def all_target(context, edge_evidence, separate_evidence, *,
                   verified_network_continuity,
                   verified_network_source_inventory,
                   verified_nonnetwork_interface_certificates,
                   progress_callback):
        events.append("all_target")
        assert verified_network_continuity == {
            "passed": True, "evaluated_link_count": 15}
        assert progress_callback is None
        assert verified_nonnetwork_interface_certificates is None
        assert isinstance(verified_network_source_inventory, dict)
        assert verified_network_source_inventory[
            "candidate_gid_values_were_read_from_actual_capture_pack"]
        assert all(isinstance(context["surfaces"][key], _CachedSurface)
                   for key in runner.NETWORK_KEYS)
        return {"passed": True, "status": "CURRENT_SCREEN_PASS_UNADMITTED"}

    monkeypatch.setattr(
        runner.network_owner, "audit_brachiocephalic_network_links_3d", link_audit)
    monkeypatch.setattr(runner.preparation, "run_all_target_network_screen", all_target)
    report = runner.execute_ordered_network_gates(_runtime_fixture(), manifold3d=object())
    assert report["status"] == (
        "NETWORK_CONTINUITY_PASSED_ALL_TARGET_COLLISION_PASSED_UNADMITTED")
    assert events == ["links", "all_target"]
    assert len(fake_verifier.prepare_calls) == 2 * len(runner.NETWORK_KEYS)


def test_existing_source_owners_accept_cached_prepared_rows_without_repreparing():
    verifier = _Verifier()
    key = (51011, 526)
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    prepared = _CachedSurface(key, positions, faces, [5, 6, 7])
    surface, report = junction._source_surface(
        verifier, key, prepared, "cached_candidate")
    assert surface is prepared
    assert report["exact_self_audit_count"] == 0
    assert verifier.prepare_calls == []
    assert geometry._prepare_vessel(verifier, key, prepared) is prepared
    assert verifier.prepare_calls == []


def test_all_target_oracle_accepts_cached_candidate_surface_objects(monkeypatch):
    verifier = _Verifier()
    links = junction.declared_network_links()
    keys = sorted({key for edge in links for key in edge})
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    surfaces = {
        key: _CachedSurface(key, positions, faces, [10, 11, 12])
        for key in keys
    }
    source_pairs = {edge: [(0, 0)] for edge in links}
    captured = {}

    def audit_changed(*args, **kwargs):
        captured["surfaces"] = args[0]
        captured["changed"] = kwargs["changed"]
        return {"passed": True}

    verifier.audit_changed_surfaces = audit_changed
    monkeypatch.setattr(geometry, "_identity", lambda value: tuple(value))
    report = geometry.audit_network_rows_against_all_targets(
        verifier, surfaces, links, source_pairs)
    assert report["passed"] is True
    assert captured["surfaces"] is surfaces
    assert set(captured["changed"]) == set(keys)


def _source_preflight_fixture():
    return {
        "schema": "numi.human.brachiocephalic.current-capture-network-preflight.v2",
        "status": "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE",
        "recorded_at_utc": "2026-10-14T00:00:00+00:00",
        "source_repository": {"head": "a" * 40, "dirty": False, "status_lines": []},
        "input_sha256": {"/capture/pack": "b" * 64},
        "actual_accepted_capture": {"accepted_step": 0, "pack_sha256": "b" * 64},
        "source_bound_rigid_rest_baseline": {
            "meaningful_calibre_baseline_ready": True,
            "network_owner_baseline_coverage_complete": True,
            "network_calibre_gate_passed": False,
        },
        "network_topology": {"pending_source_collar_edges": []},
    }


def test_run150_explicit_bundle_authenticates_its_extra_payload_path(tmp_path):
    payload = tmp_path / "resting-thorax.nhanatomy"
    payload.write_bytes(b"pinned NHA payload")
    digest = runner._sha_file(payload)
    bundle = {
        "anatomy_payload_sha256": digest,
        "anatomy_receipt_path": str(tmp_path / "anatomy-receipt.json"),
        "anatomy_receipt_sha256": "a" * 64,
        "capture_pack_path": str(tmp_path / "capture.mrvpack"),
        "capture_pack_sha256": "b" * 64,
        "capture_receipt_path": str(tmp_path / "capture-receipt.json"),
        "capture_receipt_sha256": "c" * 64,
    }
    for key in ("anatomy_receipt_path", "capture_pack_path", "capture_receipt_path"):
        Path(bundle[key]).write_bytes(b"")
    explicit = {**bundle, "payload_path": str(payload)}
    runner._validate_run150_explicit_capture_bundle(bundle, explicit)

    explicit["payload_path"] = str(tmp_path / "missing.nhanatomy")
    with pytest.raises((FileNotFoundError, ValueError)):
        runner._validate_run150_explicit_capture_bundle(bundle, explicit)

    explicit["payload_path"] = str(payload)
    explicit["unbound_path"] = str(payload)
    with pytest.raises(ValueError, match="unexpected field set"):
        runner._validate_run150_explicit_capture_bundle(bundle, explicit)


def test_prepare_freezes_exact_bundle_without_calling_geometry(tmp_path, monkeypatch):
    bundle = tmp_path / "capture-inputs.json"
    bundle.write_text("{}")
    output = tmp_path / "prepared"
    preflight = _source_preflight_fixture()
    monkeypatch.setattr(runner.preparation, "build_preflight",
                        lambda _inputs, **_kwargs: (dict(preflight), {})
                        if _kwargs.get("return_runtime_inputs")
                        else dict(preflight))
    monkeypatch.setattr(runner, "_source_binding_preflight",
                        lambda _runtime: {"status": "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE"})
    monkeypatch.setattr(runner.preparation, "_git_state",
                        lambda: preflight["source_repository"])
    monkeypatch.setattr(runner, "_runtime_code_pins", lambda: {"runner": "c" * 64})
    monkeypatch.setattr(
        runner, "execute_ordered_network_gates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preparation must not run geometry")))
    result = runner.prepare(bundle, output)
    frozen = json.loads(
        (output / "source-and-capture-preflight.json").read_text())
    assert result["geometry_predicates_run"] is False
    assert result["status"] == preflight["status"]
    assert frozen["execution_runner"]["capture_input_bundle_sha256"] == (
        result["capture_input_bundle_sha256"])
    assert json.loads((output / "progress.json").read_text())[
        "geometry_predicates_run"] is False




def test_baseline_addendum_stage_freezes_only_baseline_proof(tmp_path, monkeypatch):
    bundle = tmp_path / "capture-inputs.json"
    bundle.write_text("{}")
    output = tmp_path / "prepared"
    preflight = _source_preflight_fixture()
    runtime = {"test_runtime": True}

    def build(_inputs, *, return_runtime_inputs=False):
        return (dict(preflight), runtime) if return_runtime_inputs else dict(preflight)

    monkeypatch.setattr(runner.preparation, "build_preflight", build)
    monkeypatch.setattr(runner, "_source_binding_preflight",
                        lambda _runtime: {"status": "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE"})
    monkeypatch.setattr(runner.preparation, "_git_state",
                        lambda: preflight["source_repository"])
    monkeypatch.setattr(runner, "_runtime_code_pins", lambda: {"runner": "c" * 64})
    proof = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "status": "VERIFIED_INDEPENDENT_BASELINE_EXACT_ONE_FACE_COLLAR_EXTENSION",
        "passed": True,
        "typed_link": [[51011, 475], [51011, 479]],
        "candidate_geometry_used": False,
    }
    monkeypatch.setattr(runner, "_baseline_collar_addendum", lambda *_args: proof)

    runner.prepare(bundle, output)
    result = runner.prepare_baseline_collar_addendum(bundle, output)
    saved = json.loads((output / runner.BASELINE_ADDENDUM_NAME).read_text())
    progress = json.loads((output / "progress.json").read_text())

    assert result["status"] == (
        "VERIFIED_BASELINE_ONLY_475_479_EXACT_P6_RAT_COLLAR_ADDENDUM")
    assert saved["verified_addendum"] == proof
    assert saved["baseline_exact_p6_predicate_run"] is True
    assert saved["candidate_surface_predicates_run"] is False
    assert progress["status"] == "BASELINE_COLLAR_ADDENDUM_VERIFIED"
    assert progress["baseline_addendum_sha256"] == runner._sha_file(
        output / runner.BASELINE_ADDENDUM_NAME)


def test_run_rejects_capture_bundle_drift_before_geometry(tmp_path, monkeypatch):
    bundle = tmp_path / "capture-inputs.json"
    bundle.write_text("{}")
    output = tmp_path / "prepared"
    preflight = _source_preflight_fixture()
    monkeypatch.setattr(runner.preparation, "build_preflight",
                        lambda _inputs, **_kwargs: (dict(preflight), {})
                        if _kwargs.get("return_runtime_inputs")
                        else dict(preflight))
    monkeypatch.setattr(runner, "_source_binding_preflight",
                        lambda _runtime: {"status": "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE"})
    monkeypatch.setattr(runner.preparation, "_git_state",
                        lambda: preflight["source_repository"])
    monkeypatch.setattr(runner, "_runtime_code_pins", lambda: {"runner": "c" * 64})
    runner.prepare(bundle, output)
    report = tmp_path / "source-baseline-023.json"
    report.write_text("{}")
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT", report)
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT_SHA256",
                        runner._sha_file(report))
    _freeze_test_addendum(output, bundle, {"passed": True})
    bundle.write_text('{"changed":true}')
    monkeypatch.setattr(
        runner, "execute_ordered_network_gates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bundle drift must reject before geometry")))
    with pytest.raises(ValueError, match="capture bundle differs"):
        runner.run(bundle, output)
    assert not (output / "network-gate-report.json").exists()


@pytest.mark.parametrize("enable_all_targets", [False, True])
def test_run_revalidates_preflight_then_enters_ordered_gate(
        tmp_path, monkeypatch, enable_all_targets):
    bundle = tmp_path / "capture-inputs.json"
    bundle.write_text("{}")
    output = tmp_path / "prepared"
    preflight = _source_preflight_fixture()
    runtime = {"test_runtime": True}

    def build(_inputs, *, return_runtime_inputs=False):
        return (dict(preflight), runtime) if return_runtime_inputs else dict(preflight)

    monkeypatch.setattr(runner.preparation, "build_preflight", build)
    monkeypatch.setattr(runner, "_source_binding_preflight",
                        lambda _runtime: {"status": "SOURCE_CAPTURE_ABI5_GID_IDENTITY_BOUND_NO_GEOMETRY_PREDICATE"})
    monkeypatch.setattr(runner.preparation, "_git_state",
                        lambda: preflight["source_repository"])
    monkeypatch.setattr(runner, "_runtime_code_pins", lambda: {"runner": "c" * 64})
    runner.prepare(bundle, output)
    source_report = tmp_path / "source-baseline-023.json"
    source_report.write_text("{}")
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT",
                        source_report)
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT_SHA256",
                        runner._sha_file(source_report))
    proof = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "passed": True,
        "typed_link": [[51011, 475], [51011, 479]],
    }
    source_proof = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "passed": True,
        "typed_link": [[51011, 525], [51011, 469]],
    }
    _freeze_test_addendum(output, bundle, proof, source_proof)
    monkeypatch.setattr(runner, "_baseline_collar_addendum", lambda *_args: proof)
    monkeypatch.setattr(runner, "_source_baseline_collar_addendum",
                        lambda *_args: source_proof)
    stages = []

    def execute(_runtime, *, progress_callback=None, matched_collar_addendum=None,
                matched_collar_addenda=None, run_all_targets=True):
        assert matched_collar_addendum is None
        assert matched_collar_addenda == {
            ((51011, 475), (51011, 479)): proof,
            ((51011, 525), (51011, 469)): source_proof,
        }
        assert run_all_targets is enable_all_targets
        stages.append("continuity")
        progress_callback("STARTING_FAIL_FIRST_15_LINK_CONTINUITY")
        return {
            "status": (
                "NETWORK_CONTINUITY_PASSED_ALL_TARGET_PENDING_VERIFIED_JUNCTION_HANDOFF"),
            "network_source_continuity": {"passed": True, "evaluated_link_count": 15},
            "all_target_collision_gate": {
                "status": "NOT_RUN_VERIFIED_JUNCTION_HANDOFF_NOT_FROZEN",
                "passed": False,
            },
        }

    monkeypatch.setattr(runner, "execute_ordered_network_gates", execute)
    result = (runner.run(bundle, output, run_all_targets=True)
              if enable_all_targets else runner.run(bundle, output))
    assert stages == ["continuity"]
    assert result["status"] == (
        "NETWORK_CONTINUITY_PASSED_ALL_TARGET_PENDING_VERIFIED_JUNCTION_HANDOFF")
    assert result["all_target_collision_gate"]["passed"] is False
    assert result["geometry_predicates_run"] is True
    assert result["source_preflight_sha256"] == runner._sha_file(
        output / "source-and-capture-preflight.json")


def test_abi5_face_corner_binding_allows_packed_gid_compaction_not_face_reordering():
    key = (51011, 526)
    source_row = {
        "record": (20, 10, 5, 0, 6, 526, 2, 0),
        "vertices6": np.zeros((5, 6), dtype=np.float32),
        "body_index": 20,
        "layer": 2,
        "faces": np.asarray([[1, 2, 3], [1, 3, 4]], dtype=np.uint32),
    }
    packed = _CachedSurface(
        key,
        np.zeros((4, 3), dtype=np.float32),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        [9011, 9012, 9013, 9014])
    packed_value = (packed.positions, packed.faces, None, packed.source_vertex_ids)
    vertex_map, proof = junction._source_face_capture_ancestry(
        source_row, packed_value, 526, "test_candidate",
        native_nha_gid_base_offset=9000)
    assert vertex_map.tolist() == [-1, 0, 1, 2, 3]
    assert proof["ordered_source_face_ids_are_preserved"] is True
    assert proof["capture_global_vertex_ids_are_actual_pack_values"] is True
    assert proof["native_nha_gid_base_offset"] == 9000
    assert proof["nha_source_row_first_vertex"] == 10
    assert proof["observed_gid_minus_nha_global_vertex_index_unique_count"] == 1
    assert proof["first_used_source_vertex_local_index"] == 1
    assert proof["first_used_source_vertex_capture_gid"] == 9011
    wrong_order = (packed.positions, packed.faces[::-1].copy(),
                   None, packed.source_vertex_ids)
    with pytest.raises(ValueError, match="maps to multiple capture vertices"):
        junction._source_face_capture_ancestry(
            source_row, wrong_order, 526, "reordered_candidate",
            native_nha_gid_base_offset=9000)
    permuted = (packed.positions, packed.faces, None,
                np.asarray([9012, 9011, 9013, 9014], dtype=np.uint64))
    with pytest.raises(ValueError, match="capture GIDs do not equal native NHA"):
        junction._source_face_capture_ancestry(
            source_row, permuted, 526, "permuted_gid_candidate",
            native_nha_gid_base_offset=9000)


def test_runtime_binding_retains_exact_abi5_to_pack_gid_map_when_slots_are_compacted(
    monkeypatch,
):
    runtime = _runtime_fixture()
    keys = runner.NETWORK_KEYS
    source_faces = np.asarray([[1, 2, 3]], dtype=np.uint32)
    for key in keys:
        runtime["source_rows"][key[1]] = {
            "record": (20, 10, 4, 0, 3, key[1], 2, 0),
            "faces": source_faces,
            "vertices6": np.zeros((4, 6), dtype=np.float32),
            "body_index": 20,
            "layer": 2,
        }
        positions = np.zeros((3, 3), dtype=np.float32)
        faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
        candidate_gids = np.asarray([311, 312, 313], dtype=np.uint32)
        baseline_gids = np.asarray([311, 312, 313], dtype=np.uint64)
        runtime["candidate_surfaces"][key] = (
            positions, faces, None, candidate_gids)
        runtime["baseline_surfaces"][key] = (
            positions, faces, None, baseline_gids)
    fake_verifier = _Verifier()
    monkeypatch.setattr(runner, "verifier", fake_verifier)
    _patch_input_owner(monkeypatch)
    _candidates, _baselines, inputs, inventory = runner._bind_runtime_inputs(runtime)
    assert len(inputs) == 15
    assert inventory["source_face_arrays_bound_by_exact_ordered_abi5_corner_ancestry"]
    proof = inventory["source_face_ancestry_by_typed_row"]["51011:526"]
    assert proof["status"] == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
    assert proof["source_unreferenced_vertex_count"] == 1
    assert proof["native_nha_gid_base_offset"] == 300
    assert proof["nha_source_row_first_vertex"] == 10
    assert proof["first_used_source_vertex_capture_gid"] == 311
    assert inventory["candidate_baseline_gid_comparison_dtype"] == "little-endian uint64"
    assert inventory["baseline_coordinates_are_independently_cpu_reconstructed"] is True
    assert inventory["baseline_gid_values_are_transported_current_nha_source_slots"] is True
    gid_proof = inventory["candidate_baseline_gid_identity_by_typed_row"]["51011:526"]
    assert gid_proof["canonical_dtype"] == "little-endian uint64"
    assert gid_proof["numeric_gid_values_equal"] is True
    assert gid_proof["candidate_gid_values_sha256"] == gid_proof["baseline_gid_values_sha256"]

    first_key = keys[0]
    bad_baseline_gids = np.asarray([311, 312, 314], dtype=np.uint64)
    runtime["baseline_surfaces"][first_key] = (
        np.zeros((3, 3), dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        None,
        bad_baseline_gids,
    )
    with pytest.raises(ValueError, match="differ after uint64 canonicalization"):
        runner._bind_runtime_inputs(runtime)


def test_source_baseline_addendum_replays_only_pinned_023_typed_link(tmp_path, monkeypatch):
    report_path = tmp_path / "023.json"
    report_path.write_text('{"frozen":"023"}')
    report_sha = runner._sha_file(report_path)
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT",
                        report_path)
    monkeypatch.setattr(runner.network_owner, "SOURCE_BASELINE_OBSERVATION_REPORT_SHA256",
                        report_sha)

    bridge = _CachedSurface(
        (51011, 525), [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [1, 2, 3])
    target = _CachedSurface(
        (51011, 469), [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [4, 5, 6])
    link = {
        "bridge_key": (51011, 525),
        "target_key": (51011, 469),
        "source_link_evidence": {"source_report": "pinned"},
    }
    inventory = {"native_source_slot_identity": {"native_nha_gid_base_offset": 300}}
    monkeypatch.setattr(
        runner, "_bind_runtime_inputs",
        lambda _runtime: (
            {(51011, 525): bridge, (51011, 469): target},
            {(51011, 525): bridge, (51011, 469): target},
            (link,), inventory))
    calls = []

    def verify(_verifier, **kwargs):
        calls.append(kwargs)
        return {"passed": True, "typed_link": [[51011, 525], [51011, 469]]}

    monkeypatch.setattr(runner.network_owner, "verify_baseline_matched_collar_addendum",
                        verify)
    preflight = {"source_preflight": "exact"}
    proof = runner._source_baseline_collar_addendum(
        {"source_rows": {}, "source_map": {}}, preflight)

    assert proof["passed"] is True
    assert len(calls) == 1
    call = calls[0]
    assert call["bridge_key"] == (51011, 525)
    assert call["target_key"] == (51011, 469)
    assert call["source_link_evidence"] == link["source_link_evidence"]
    assert call["independent_source_baseline_bridge_surface"] is bridge
    assert call["independent_source_baseline_target_surface"] is target
    assert call["native_nha_gid_base_offset"] == 300
    assert call["baseline_binding"] is preflight
    assert tuple(call["declarations"]) == runner.network_owner.SOURCE_BASELINE_ADDITIONS
    assert call["baseline_observation"] == {
        "report_path": str(report_path.resolve()),
        "report_sha256": report_sha,
        "current_source_surface_replay": (
            runner.network_owner.SOURCE_BASELINE_SURFACE_REPLAY_POLICY),
    }


def test_execute_ordered_network_gates_binds_addenda_by_exact_typed_link(monkeypatch):
    _patch_input_owner(monkeypatch)
    proof_475_479 = {
        "passed": True,
        "typed_link": [[51011, 475], [51011, 479]],
    }
    proof_525_469 = {
        "passed": True,
        "typed_link": [[51011, 525], [51011, 469]],
    }
    expected = {
        ((51011, 475), (51011, 479)): proof_475_479,
        ((51011, 525), (51011, 469)): proof_525_469,
    }

    def audit(_verifier, _manifold, inputs, *, expected_nha_gid_base_offset):
        assert expected_nha_gid_base_offset == 300
        by_edge = {
            (item["bridge_key"], item["target_key"]): item
            for item in inputs
        }
        assert by_edge[((51011, 475), (51011, 479))][
            "matched_collar_addendum"] == proof_475_479
        assert by_edge[((51011, 525), (51011, 469))][
            "matched_collar_addendum"] == proof_525_469
        assert all(
            item.get("matched_collar_addendum") == expected.get(
                (item["bridge_key"], item["target_key"]))
            for item in inputs if (item["bridge_key"], item["target_key"]) in expected)
        assert len([item for item in inputs
                    if item.get("matched_collar_addendum")]) == 2
        return {"passed": False, "first_failed_link": [[51011, 7], [51011, 471]]}

    monkeypatch.setattr(
        runner.network_owner, "audit_brachiocephalic_network_links_3d", audit)
    report = runner.execute_ordered_network_gates(
        _runtime_fixture(), manifold3d=object(), matched_collar_addenda=expected)
    assert report["status"] == "STOPPED_AT_FIRST_FAILED_NETWORK_LINK"
    inventory = report["candidate_and_source_baseline_inventory"]
    assert inventory["baseline_matched_collar_addendum"] == proof_475_479
    assert inventory["source_baseline_matched_collar_addendum"] == proof_525_469
    assert len(report["baseline_matched_collar_addenda"]) == 2


def test_field002_candidate_binding_is_explicit_and_fail_closed():
    binding = runner.aortic_arch_field002_candidate_binding()
    assert binding["schema"] == (
        "numi.human.aortic-arch-field-002-candidate-binding.v1")
    assert binding["capture_input_bundle_sha256"] == (
        geometry.AORTIC_ARCH_FIELD002_INTERFACE_CAPTURE_BUNDLE_SHA256)
    assert binding["candidate_npz_sha256"] == (
        runner.AORTIC_ARCH_FIELD002_CANDIDATE_NPZ_SHA256)
    assert binding["interface_handoff_sha256"] == (
        runner.AORTIC_ARCH_FIELD002_PROOF_HANDOFF_SHA256)

    stale = dict(binding, candidate_npz_sha256="0" * 64)
    with pytest.raises(ValueError, match="differs from the frozen pin set"):
        runner._bind_aortic_arch_field002_candidate({}, stale)

    incomplete = dict(binding)
    incomplete.pop("interface_handoff_sha256")
    with pytest.raises(ValueError, match="differs from the frozen pin set"):
        runner._bind_aortic_arch_field002_candidate({}, incomplete)


def test_execute_ordered_network_gates_consumes_explicit_field002_binding(
        monkeypatch):
    _patch_input_owner(monkeypatch)
    binding = runner.aortic_arch_field002_candidate_binding()
    edge = geometry.AORTIC_ARCH_SOURCE_INTERFACE
    certificate = {
        "schema": geometry.AORTIC_ARCH_FIELD002_INTERFACE_SCHEMA,
        "typed_link": [list(edge[0]), list(edge[1])],
        "passed": True,
    }
    summary = {
        "schema": binding["schema"],
        "candidate_npz_sha256": binding["candidate_npz_sha256"],
        "typed_identity": list(edge[0]),
        "anatomical_admission": False,
    }
    runtime = _runtime_fixture()
    monkeypatch.setattr(
        runner, "_bind_aortic_arch_field002_candidate",
        lambda value, supplied: (value, certificate, summary)
        if supplied == binding else (_ for _ in ()).throw(
            AssertionError("candidate binding changed")))
    monkeypatch.setattr(
        runner.network_owner, "audit_brachiocephalic_network_links_3d",
        lambda *_args, **_kwargs: {
            "passed": True, "evaluated_link_count": 15,
        })
    observed = {}

    def all_target(_context, _edge, _separate, **kwargs):
        observed.update(kwargs["verified_nonnetwork_interface_certificates"])
        return {"passed": False, "failed_stage": "fixture"}

    monkeypatch.setattr(
        runner.preparation, "run_all_target_network_screen", all_target)
    with pytest.raises(ValueError, match="exact capture-bundle SHA-256"):
        runner.execute_ordered_network_gates(
            runtime, manifold3d=object(),
            aortic_arch_field002_source_binding=binding)
    result = runner.execute_ordered_network_gates(
        runtime, manifold3d=object(),
        aortic_arch_field002_source_binding=binding,
        capture_input_bundle_sha256=binding["capture_input_bundle_sha256"])
    assert observed == {edge: certificate}
    assert result["aortic_arch_field002_source_candidate_binding"] == summary
    assert result["all_target_collision_gate"]["failed_stage"] == "fixture"

    with pytest.raises(ValueError, match="cannot be combined"):
        runner.execute_ordered_network_gates(
            runtime, manifold3d=object(),
            verified_nonnetwork_interface_certificates={edge: {"old": True}},
            aortic_arch_field002_source_binding=binding,
            capture_input_bundle_sha256=binding["capture_input_bundle_sha256"])


def test_cached_continuation_loads_only_the_two_typed_aortic_certificates(monkeypatch):
    arch = geometry.AORTIC_ARCH_SOURCE_INTERFACE
    descending = geometry.AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE
    certificates = {
        arch: {"typed_link": [list(arch[0]), list(arch[1])]},
        descending: {"typed_link": [list(descending[0]), list(descending[1])]},
    }
    monkeypatch.setattr(
        runner, "_load_aortic_arch_interface_certificate",
        lambda runtime, bundle_sha: certificates[arch])
    monkeypatch.setattr(
        runner, "_load_aortic_arch_descending_interface_certificate",
        lambda runtime, bundle_sha: certificates[descending])
    loaded = runner._load_cached_aortic_interface_certificates(
        {"runtime": "unused-by-mocked-loaders"}, "bundle-sha")
    assert set(loaded) == {arch, descending}
    assert loaded == certificates

    monkeypatch.setattr(
        runner, "_load_aortic_arch_descending_interface_certificate",
        lambda runtime, bundle_sha: {"typed_link": [list(arch[0]), [51011, 6]]})
    with pytest.raises(ValueError, match="wrong typed link"):
        runner._load_cached_aortic_interface_certificates({}, "bundle-sha")


def test_descending_source_identity_uses_nested_typed_identity():
    key = (51011, 8)
    expected = geometry.AORTIC_ARCH_DESCENDING_SOURCE_ROWS[key]
    gid_base = 766836
    ancestry = {
        "status": "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED",
        "typed_identity": list(key),
        "source_abi5_face_array_sha256": expected["faces_sha256"],
        "capture_faces_u32_sha256": expected["faces_sha256"],
        "ordered_source_face_ids_are_preserved": True,
        "capture_global_vertex_ids_are_actual_pack_values": True,
        "native_source_slot_identity_is_contiguous": True,
        "native_nha_gid_base_offset": gid_base,
        "observed_gid_minus_nha_global_vertex_index_unique_count": 1,
        "observed_gid_minus_nha_global_vertex_index_min": gid_base,
        "observed_gid_minus_nha_global_vertex_index_max": gid_base,
    }
    row = {
        "source_member": expected["source_member"],
        "fma": expected["fma"],
        "name": expected["name"],
        "body_index": expected["body_index"],
        "layer": expected["layer"],
        "first_vertex": expected["first_vertex"],
        "vertex_count": expected["vertex_count"],
        "face_count": expected["face_count"],
        "source_face_u32_sha256": expected["faces_sha256"],
        "actual_face_u32_sha256": expected["faces_sha256"],
        "source_to_capture_ancestry": ancestry,
    }
    assert "typed_identity" not in row
    assert runner._validate_aortic_descending_source_identity_row(
        key, expected, row, gid_base) == ancestry

    bad_row = {**row, "source_to_capture_ancestry": {
        **ancestry, "typed_identity": [51011, 7]}}
    with pytest.raises(ValueError, match="source row/slot ancestry changed"):
        runner._validate_aortic_descending_source_identity_row(
            key, expected, bad_row, gid_base)


def test_descending_proof_loader_rejects_changed_frozen_report_before_runtime(
        tmp_path, monkeypatch):
    report = tmp_path / "execution-report.json"
    preflight = tmp_path / "preflight.json"
    handoff = tmp_path / "handoff-certificate.json"
    report.write_text("{}\n")
    preflight.write_text("{}\n")
    handoff.write_text("{}\n")
    monkeypatch.setattr(runner, "AORTIC_DESCENDING_PROOF_REPORT", report)
    monkeypatch.setattr(runner, "AORTIC_DESCENDING_PROOF_PREFLIGHT", preflight)
    monkeypatch.setattr(runner, "AORTIC_DESCENDING_PROOF_HANDOFF", handoff)
    monkeypatch.setattr(runner, "AORTIC_DESCENDING_PROOF_REPORT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned 7↔8 source-interface evidence changed"):
        runner._load_aortic_arch_descending_interface_certificate(
            {}, geometry.AORTIC_ARCH_DESCENDING_INTERFACE_CAPTURE_BUNDLE_SHA256)
