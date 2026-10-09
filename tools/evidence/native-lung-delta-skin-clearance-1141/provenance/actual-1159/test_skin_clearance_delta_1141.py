import importlib.util
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("skin_delta_1141", ROOT / "skin_clearance_delta_1141.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class NHA:
    def __init__(self):
        self.rows = {sid: {"ni": 3} for sid in M.IDENTITY_ROWS}


class Pack:
    def __init__(self):
        self.surfaces = {}
        for sid in M.IDENTITY_ROWS:
            key = (51010, sid) if sid == 311 else ((51024, sid) if sid == 310 else (51023, sid))
            self.surfaces[key] = {
                "body": 20, "semantic": key[0], "instance": 1, "owner": 20,
                "sid": sid, "primitive_id": sid + 1000, "count": 3,
                "norm": (0, 1, 2), "start": 0, "used": (0, 1, 2),
            }
        self._xyz = {i: struct.pack("<3f", float(i), 0.0, 0.0) for i in range(4)}

    def xyz(self, i):
        return self._xyz[i]


def test_target_primitive_identity_is_bound_and_geometry_can_change():
    before, after = Pack(), Pack()
    after.surfaces[(51023, 305)]["norm"] = (0, 1, 3)
    got = M.validate_target_surface_identities(before, after, NHA(), NHA())
    assert len(got) == len(M.IDENTITY_ROWS)
    assert got["305"]["identity_equal_1116"] is True


def test_target_primitive_identity_change_is_rejected():
    before, after = Pack(), Pack()
    after.surfaces[(51023, 305)]["instance"] = 2
    try:
        M.validate_target_surface_identities(before, after, NHA(), NHA())
    except ValueError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("changed target instance identity was accepted")


def test_target_native_count_must_match_bound_source_rows():
    before, after = Pack(), Pack()
    after.surfaces[(51023, 305)]["count"] = 6
    try:
        M.validate_target_surface_identities(before, after, NHA(), NHA())
    except ValueError as exc:
        assert "index count changed" in str(exc)
    else:
        raise AssertionError("target primitive count change was accepted")


def test_changed_face_reader_uses_indices_and_exact_xyz():
    before, after = Pack(), Pack()
    key = (51023, 305)
    after.surfaces[key]["norm"] = (0, 1, 3)
    changed, unchanged = M.changed_face_rows_between_packs(before, after, 305, key, 1)
    assert changed == {0}
    assert unchanged == 0


def test_same_triangle_is_transferred_only_when_indices_and_xyz_match():
    before, after = Pack(), Pack()
    key = (51023, 305)
    changed, unchanged = M.changed_face_rows_between_packs(before, after, 305, key, 1)
    assert changed == set()
    assert unchanged == 1
    after._xyz[2] = struct.pack("<3f", 2.0, 0.0, 0.001)
    changed, unchanged = M.changed_face_rows_between_packs(before, after, 305, key, 1)
    assert changed == {0}
    assert unchanged == 0


def _valid_delta(tmp_path):
    import hashlib
    candidate_nha = tmp_path / "candidate.nhanatomy"
    candidate_nha.write_bytes(b"candidate")
    tracked = tmp_path / "tracked-input.bin"
    tracked.write_bytes(b"unchanged")
    delta_path = tmp_path / "delta.json"
    delta_path.write_bytes(b"registered report bytes")
    poses = []
    for step in M.STEPS:
        pack_sha = "pack-%d" % step
        poses.append({
            "accepted_step": step,
            "pack_sha256": pack_sha,
            "derived_pleura_native_copy": {
                "accepted_step": step,
                "pack_sha256": pack_sha,
                "lineage_face_count": M.EXPECTED_ROW310_FACES,
                "exact_native_same_winding_copied_faces": M.EXPECTED_ROW310_FACES,
                "same_winding_mismatches": 0,
                "maximum_abs_coordinate_delta_m": 0.0,
                "native_all_524_row_mapping": True,
            },
        })
    delta = {
        "schema": "numi.human.native-transformed-lung-cycle-delta-audit.summary.v1",
        "candidate_run": str(tmp_path / "native-run"),
        "candidate_nha": {"path": str(candidate_nha), "sha256": M.sha(candidate_nha)},
        "requested_roots": M.STEPS[-1],
        "accepted_steps": M.STEPS,
        "terminal_step_included": True,
        "complete_pair_coverage": True,
        "all_6_self_and_15_cross_per_pose": True,
        "topology_verified_each_pose": True,
        "input_hashes_unchanged": True,
        "worker_failures": [],
        "candidate_area_config_binding_status": "PASS_exact_geometry_and_config_binding",
        "run_identity_comparison": {
            "argv_equal_except_declared_paths": True,
            "environment_equal": True,
            "loaded_runtime_same_verified": True,
            "normalized_respiration_configuration_equal": True,
        },
        "pose_results": poses,
        "input_hashes_before": {str(tracked): M.sha(tracked)},
        "status": "complete_delta_scan_with_geometry_failures",
    }
    return delta_path, delta, tmp_path / "native-run", candidate_nha


def test_delta_gate_accepts_complete_geometry_failure_without_claiming_geometry_pass(tmp_path):
    path, delta, run, nha = _valid_delta(tmp_path)
    result = M.validate_delta_document(
        delta, delta_path=path, delta_sha=M.sha(path), candidate_run=run,
        candidate_nha=nha, candidate_nha_sha=M.sha(nha))
    assert [x["accepted_step"] for x in result.values()] == M.STEPS


def test_delta_gate_rejects_missing_row310_copy_proof(tmp_path):
    path, delta, run, nha = _valid_delta(tmp_path)
    delta["pose_results"][3]["derived_pleura_native_copy"]["same_winding_mismatches"] = 1
    try:
        M.validate_delta_document(delta, delta_path=path, delta_sha=M.sha(path),
                                  candidate_run=run, candidate_nha=nha, candidate_nha_sha=M.sha(nha))
    except ValueError as exc:
        assert "current row310 exact copy" in str(exc)
    else:
        raise AssertionError("missing row310 copy proof was accepted")


def test_delta_gate_rejects_area_config_mismatch(tmp_path):
    path, delta, run, nha = _valid_delta(tmp_path)
    delta["candidate_area_config_binding_status"] = "FAIL_geometry_area_mismatch"
    try:
        M.validate_delta_document(delta, delta_path=path, delta_sha=M.sha(path),
                                  candidate_run=run, candidate_nha=nha, candidate_nha_sha=M.sha(nha))
    except ValueError as exc:
        assert "area binding" in str(exc)
    else:
        raise AssertionError("area/config mismatch was accepted")


def test_delta_gate_rehashes_tracked_inputs(tmp_path):
    path, delta, run, nha = _valid_delta(tmp_path)
    tracked = Path(next(iter(delta["input_hashes_before"])))
    tracked.write_bytes(b"mutated")
    try:
        M.validate_delta_document(delta, delta_path=path, delta_sha=M.sha(path),
                                  candidate_run=run, candidate_nha=nha, candidate_nha_sha=M.sha(nha))
    except ValueError as exc:
        assert "tracked input changed" in str(exc)
    else:
        raise AssertionError("changed tracked input was accepted")


def test_delta_gate_rejects_incomplete_status_even_if_flags_look_complete(tmp_path):
    path, delta, run, nha = _valid_delta(tmp_path)
    delta["status"] = "incomplete_delta_scan"
    try:
        M.validate_delta_document(delta, delta_path=path, delta_sha=M.sha(path),
                                  candidate_run=run, candidate_nha=nha, candidate_nha_sha=M.sha(nha))
    except ValueError as exc:
        assert "completed delta scan" in str(exc)
    else:
        raise AssertionError("incomplete delta status was accepted")


def test_exact_predicate_import_uses_owner_package_for_relative_imports():
    loaded = M.load_exact_predicate()
    assert Path(loaded.__file__).resolve() == M.PREDICATE.resolve()
    assert loaded.float32_point_lattice_key((0.0, 0.0, 0.0)) == (0, 0, 0)
