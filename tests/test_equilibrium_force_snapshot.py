from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pytest

from numilab_human.equilibrium_force_snapshot import PREFIX, convert
from numilab_human.model import ImportError


def _record(*, gravity_sign: float = 1.0, ambiguous: bool = False) -> dict:
    zero = [0.0] * 128
    gravity = zero.copy()
    support = zero.copy()
    gravity[2] = 0.0 if ambiguous else -100.0
    support[2] = 0.0 if ambiguous else (100.0 if gravity_sign > 0.0 else -100.0)
    muscle = zero.copy()
    muscle[12] = 5.0
    equality = zero.copy()
    equality[12] = -5.0
    limit = zero.copy()
    passive = zero.copy()
    non_gravity = [
        muscle[index] + equality[index] + limit[index] + support[index] + passive[index]
        for index in range(128)
    ]
    residual = [non_gravity[index] + gravity_sign * gravity[index] for index in range(128)]
    return {
        "acceleration": zero,
        "limit_force": limit,
        "equality_force": equality,
        "muscle_force": muscle,
        "passive_force": passive,
        "support_force": support,
        "gravity_target": gravity,
        "force_residual": residual,
    }


def _args(log: Path, output: Path, coordinate_map: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        log=log,
        coordinate_map=coordinate_map,
        maximum_reconstruction_error=1.0e-4,
        output=output,
    )


def _write_log(path: Path, record: dict, *, compressed: bool = False) -> None:
    raw = ("prefix\n" + PREFIX + json.dumps(record) + "\nsuffix\n").encode("utf-8")
    if compressed:
        path.write_bytes(gzip.compress(raw))
    else:
        path.write_bytes(raw)


def _write_persistent_log(path: Path) -> None:
    rows = []
    for dof in reversed(range(128)):
        # Deliberately reverse the rows: the native audit is ranked by
        # residual magnitude, so the converter must use the explicit DoF.
        muscle = 2.0 if dof == 7 else 0.0
        support = 3.0 if dof == 7 else 0.0
        equality = -4.0 if dof == 7 else 0.0
        gravity = 1.0 if dof == 7 else 0.0
        residual = muscle + support + equality - gravity
        rows.append({
            "dof": dof,
            "metal_muscle_force_n": muscle,
            "support_force_n": support,
            "equality_force_n": equality,
            "limit_force_n": 0.0,
            "passive_force_n": 0.0,
            "compiled_passive_force_n": 0.0,
            "gravity_target_n": gravity,
            "residual_n": residual,
        })
    payload = {
        "schema": "numi.human.persistent-dynamic-force-audit.v1",
        "maximum_abs_residual_n": 2.0,
        "rows": rows,
    }
    path.write_text(
        "prefix\npersistent_dynamic_force_audit="
        + json.dumps(payload)
        + "\nsuffix\n",
        encoding="utf-8",
    )


def test_plus_gravity_convention_is_reconstructed(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    _write_log(log, _record(gravity_sign=1.0))
    output = tmp_path / "snapshot.json"
    assert convert(_args(log, output)) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["schema"] == "numi.human.generalized-force-snapshot.v1"
    assert snapshot["metadata"]["gravity_convention"]["selected_sign"] == 1.0
    assert snapshot["metadata"]["maximum_reconstruction_error"] == 0.0
    assert [row["name"] for row in snapshot["components"]] == [
        "gravity_bias", "muscle_tendon", "joint_equality",
        "joint_limit", "support_contact", "passive_tissue",
    ]


def test_minus_gravity_convention_is_reconstructed_from_gzip(tmp_path: Path) -> None:
    record = _record(gravity_sign=-1.0)
    record["gravity_target"][2] = 100.0
    record["support_force"][2] = 100.0
    record["force_residual"][2] = 0.0
    log = tmp_path / "native.log.gz"
    _write_log(log, record, compressed=True)
    output = tmp_path / "snapshot.json"
    assert convert(_args(log, output)) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["metadata"]["gravity_convention"]["selected_sign"] == -1.0
    gravity = next(row for row in snapshot["components"] if row["name"] == "gravity_bias")
    assert gravity["values"][2] == -100.0


def test_ambiguous_gravity_sign_fails_closed(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    _write_log(log, _record(ambiguous=True))
    output = tmp_path / "snapshot.json"
    with pytest.raises(ImportError, match="not uniquely recoverable"):
        convert(_args(log, output))


def test_coordinate_map_is_bound(tmp_path: Path) -> None:
    log = tmp_path / "native.log"
    _write_log(log, _record(gravity_sign=1.0))
    coordinate_map = tmp_path / "coordinates.json"
    coordinate_map.write_text(json.dumps({
        "coordinate_names": [f"joint_{index}" for index in range(128)],
        "coordinate_kinds": ["translation"] * 3 + ["rotation"] * 125,
    }), encoding="utf-8")
    output = tmp_path / "snapshot.json"
    assert convert(_args(log, output, coordinate_map)) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["coordinate_names"][42] == "joint_42"
    assert snapshot["metadata"]["coordinate_map"]["anatomical_names"]


def test_persistent_dynamic_audit_is_reindexed_and_reconstructed(tmp_path: Path) -> None:
    log = tmp_path / "persistent.log"
    _write_persistent_log(log)
    output = tmp_path / "snapshot.json"
    assert convert(_args(log, output)) == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["metadata"]["native_log"]["record"] == "persistent_dynamic_force_audit"
    assert snapshot["metadata"]["boundary"].startswith("Persistent dynamic force rows")
    muscle = next(row for row in snapshot["components"] if row["name"] == "muscle_tendon")
    assert muscle["values"][7] == 2.0
    assert muscle["values"][6] == 0.0
    assert snapshot["reported_net"][7] == 0.0
    assert snapshot["generalized_acceleration"] is None
