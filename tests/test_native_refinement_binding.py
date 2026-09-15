from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import numilab_human.native_passive_stand_refinement as r
from numilab_human.model import ImportError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, peaks=(0.1, 0.1, 0.1, 0.1)) -> tuple[Path, dict]:
    artifacts = {}
    for name in ("binary", "rigid", "muscle", "tendon", "support", "equalities"):
        target = root / f"fixture-{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic unit-test artifact " + name)
        artifacts[name] = {"path": target.name, "sha256": _sha(target)}
    source = {"commit": "a" * 40, "binary_sha256": artifacts["binary"]["sha256"],
              "worktree_clean": True, "root_assistance": "none", "device": "Apple M4 Pro"}
    cases = {}
    for (name, dt, count), peak in zip(r.CASE_SPEC, peaks):
        directory = root / name
        directory.mkdir()
        out = directory / "stdout.txt"
        out.write_text("myosim_articulated_marker_visual=ok "
                       f"muscle_step_seconds={dt} muscle_step_count={count} persistent_completed_steps={count} "
                       "persistent_metal_horizon=true persistent_source_passive_joint_tissue=true "
                       "compiled_stand_balanced=true stand_deterministic_replay=bitwise "
                       'source_support_metal_device="Apple M4 Pro" '
                       f"persistent_max_penetration_m=0 persistent_max_acceleration={peak} "
                       "compiled_stand_normalized_residual_rms=0.001 "
                       "source_dynamic_force_parity_max_delta_n=0.01 "
                       "muscle_step_max_velocity_delta=0.0001 muscle_step_max_configuration_delta=0.00001\n")
        err = directory / "stderr.txt"
        err.write_text("")
        cases[name] = {"exit_code": 0, "source_commit": source["commit"],
                       "binary_sha256": source["binary_sha256"],
                       "artifact_hashes": {key: value["sha256"] for key, value in artifacts.items()},
                       "timestep_nanoseconds": int(round(dt * 1e9)), "steps": count,
                       "stdout_sha256": _sha(out), "stderr_sha256": _sha(err)}
    manifest = {"schema": r.MANIFEST_SCHEMA, "source": source, "artifacts": artifacts, "cases": cases}
    path = root / "run-manifest.json"
    _write(path, manifest)
    return path, manifest


def _write(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest))


def test_new_runs_bind_actual_source_not_historical_constants(tmp_path: Path) -> None:
    path, data = _fixture(tmp_path)
    result = r.compile_refinement(case_root=tmp_path, run_manifest=path)
    assert result["source"]["commit"] == data["source"]["commit"]
    assert result["source"]["commit"] != r.SOURCE_COMMIT
    assert result["run_binding"]["sha256"] == _sha(path)
    assert result["convergence"]["peak_acceleration_converged"]
    assert not result["convergence"]["force_convergence"]
    assert not result["qualification"]["force_convergence"]
    assert result["blocker"]["status"] == "open"


def test_custom_root_requires_manifest(tmp_path: Path) -> None:
    _fixture(tmp_path)
    with pytest.raises(ImportError, match="requires --run-manifest"):
        r.compile_refinement(case_root=tmp_path)


@pytest.mark.parametrize("peaks,ratio", [((0, 0, 0, 0), 0), ((0, 0.1, 0.2, 0.3), None)])
def test_zero_peaks_do_not_divide_by_zero(tmp_path: Path, peaks, ratio) -> None:
    path, _ = _fixture(tmp_path, peaks)
    result = r.compile_refinement(case_root=tmp_path, run_manifest=path)
    assert result["convergence"]["peak_acceleration_range_over_minimum"] == ratio
    assert not result["qualification"]["force_convergence"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_peak_is_rejected(tmp_path: Path, bad: float) -> None:
    path, _ = _fixture(tmp_path, (0.1, bad, 0.1, 0.1))
    with pytest.raises(ImportError, match="finite|negative"):
        r.compile_refinement(case_root=tmp_path, run_manifest=path)


def test_incomplete_run_rejected_even_when_requested_steps_match(tmp_path: Path) -> None:
    path, data = _fixture(tmp_path)
    stdout = tmp_path / "50us/stdout.txt"
    stdout.write_text(stdout.read_text().replace("persistent_completed_steps=128", "persistent_completed_steps=64"))
    data["cases"]["50us"]["stdout_sha256"] = _sha(stdout)
    _write(path, data)
    with pytest.raises(ImportError, match="completed step count"):
        r.compile_refinement(case_root=tmp_path, run_manifest=path)


def test_duplicate_metrics_rejected(tmp_path: Path) -> None:
    path, data = _fixture(tmp_path)
    stdout = tmp_path / "25us/stdout.txt"
    stdout.write_text(stdout.read_text() + "persistent_max_acceleration=0.00001\n")
    data["cases"]["25us"]["stdout_sha256"] = _sha(stdout)
    _write(path, data)
    with pytest.raises(ImportError, match="unique"):
        r.compile_refinement(case_root=tmp_path, run_manifest=path)


@pytest.mark.parametrize("mutation", ["log", "artifact", "source", "hashes", "process", "assistance", "dirty"])
def test_provenance_mismatch_rejected(tmp_path: Path, mutation: str) -> None:
    path, data = _fixture(tmp_path)
    if mutation == "log":
        (tmp_path / "100us/stdout.txt").write_text("changed")
    elif mutation == "artifact":
        (tmp_path / "fixture-rigid").write_text("changed")
    elif mutation == "source":
        data["cases"]["100us"]["source_commit"] = "b" * 40
    elif mutation == "hashes":
        data["cases"]["100us"]["artifact_hashes"]["support"] = "c" * 64
    elif mutation == "process":
        data["cases"]["100us"]["exit_code"] = 1
    elif mutation == "assistance":
        data["source"]["root_assistance"] = "enabled"
    else:
        data["source"]["worktree_clean"] = False
    _write(path, data)
    with pytest.raises(ImportError):
        r.compile_refinement(case_root=tmp_path, run_manifest=path)


def test_validation_banner_allowed_but_warning_rejected(tmp_path: Path) -> None:
    path, data = _fixture(tmp_path)
    stderr = tmp_path / "100us/stderr.txt"
    stderr.write_text("2026-09-15 12:01:01.234 probe[123:456] Metal API Validation Enabled\n")
    data["cases"]["100us"]["stderr_sha256"] = _sha(stderr)
    _write(path, data)
    assert r.compile_refinement(case_root=tmp_path, run_manifest=path)["status"] == "partial"
    stderr.write_text(stderr.read_text() + "Metal validation ERROR\n")
    data["cases"]["100us"]["stderr_sha256"] = _sha(stderr)
    _write(path, data)
    with pytest.raises(ImportError, match="stderr"):
        r.compile_refinement(case_root=tmp_path, run_manifest=path)
