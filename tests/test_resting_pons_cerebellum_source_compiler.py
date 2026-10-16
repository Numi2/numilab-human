from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from numilab_human import resting_pons_cerebellum_source_compiler as compiler
from numilab_human.resting_anatomy_interface_patch import encode_payload
from numilab_human.resting_pleura_proxy import _parse_payload


def _row(owner: int = 23) -> dict:
    return {
        "body_index": owner, "layer": 1, "flags": 0,
        "vertices6": np.asarray(
            [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]],
            dtype="<f4"),
        "faces": np.asarray([[0, 1, 2]], dtype="<i8"),
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "row323_vertices6": np.asarray(
            [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1],
             [1, 1, 0, 0, 0, 1]], dtype="<f4"),
        "row323_faces": np.asarray([[0, 1, 3], [0, 3, 2]], dtype="<i8"),
        "row348_vertices6": np.asarray(
            [[0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1],
             [1, 1, 1, 0, 0, 1]], dtype="<f4"),
        "row348_faces": np.asarray([[0, 1, 3], [0, 3, 2]], dtype="<i8"),
    }


def _base_payload() -> bytes:
    rows = {323: _row(), 348: _row(), 349: _row(owner=20)}
    return encode_payload((b"NHANAT1\0", 5, len(rows), 0, 0, 0x12345678, b"s" * 32), rows)


def test_builds_exact_two_row_replacements_and_keeps_parent_owner_fields(monkeypatch):
    monkeypatch.setattr(compiler, "_ROWS", {
        323: ("row323_vertices6", "row323_faces", (3, 1)),
        348: ("row348_vertices6", "row348_faces", (3, 1)),
    })
    _header, _records, rows = _parse_payload(_base_payload())
    replacements = compiler.build_row_replacements(rows, _arrays())

    assert list(replacements) == [323, 348]
    assert {sid: (row["body_index"], row["layer"], row["flags"])
            for sid, row in replacements.items()} == {323: (23, 1, 0), 348: (23, 1, 0)}
    assert len(replacements[323]["faces"]) == len(replacements[348]["faces"]) == 2


def test_shared_abi5_owner_preserves_parent_header_and_unmodified_rows(monkeypatch):
    monkeypatch.setattr(compiler, "_ROWS", {
        323: ("row323_vertices6", "row323_faces", (3, 1)),
        348: ("row348_vertices6", "row348_faces", (3, 1)),
    })
    raw = _base_payload()
    base_header, _records, base_rows = _parse_payload(raw)
    replacements = compiler.build_row_replacements(base_rows, _arrays())

    output, owner_report = compiler.compile_payload(raw, replacements)
    header, _records, rows = _parse_payload(output)

    assert tuple(header[5:]) == tuple(base_header[5:])
    assert list(rows) == list(base_rows)
    assert owner_report["replaced_stable_ids"] == [323, 348]
    assert owner_report["unmodified_rows_byte_exact"] == 1
    assert rows[323]["faces"].shape == rows[348]["faces"].shape == (2, 3)
    assert rows[349]["faces"].tobytes() == base_rows[349]["faces"].tobytes()
    assert rows[349]["vertices6"].tobytes() == base_rows[349]["vertices6"].tobytes()


def test_rejects_wrong_parent_owner_and_nonlocal_face(monkeypatch):
    monkeypatch.setattr(compiler, "_ROWS", {
        323: ("row323_vertices6", "row323_faces", (3, 1)),
        348: ("row348_vertices6", "row348_faces", (3, 1)),
    })
    _header, _records, rows = _parse_payload(_base_payload())
    rows[348]["body_index"] = 20
    with pytest.raises(ValueError, match="Body23/layer1/passive"):
        compiler.build_row_replacements(rows, _arrays())

    _header, _records, rows = _parse_payload(_base_payload())
    arrays = _arrays()
    arrays["row323_faces"][0, 0] = 99
    with pytest.raises(ValueError, match="nonlocal face index"):
        compiler.build_row_replacements(rows, arrays)


def test_cli_rejects_identical_output_paths_before_compilation(monkeypatch, tmp_path):
    repository = Path(__file__).resolve().parents[1]
    cli_path = repository / "tools" / "compile_pons_cerebellum_source.py"
    spec = importlib.util.spec_from_file_location("pons_source_compile_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    monkeypatch.setattr(
        cli, "compile_pinned_candidate",
        lambda *args, **kwargs: pytest.fail("compiler must not run for colliding outputs"))
    shared_output = tmp_path / "same-output"
    monkeypatch.setattr(sys, "argv", [
        str(cli_path),
        "--base-nha", str(tmp_path / "missing-base"),
        "--candidate-npz", str(tmp_path / "missing-candidate"),
        "--preflight", str(tmp_path / "missing-preflight"),
        "--lineage", str(tmp_path / "missing-lineage"),
        "--strict-audit", str(tmp_path / "missing-audit"),
        "--output-nha", str(shared_output),
        "--output-evidence", str(shared_output),
    ])

    with pytest.raises(SystemExit, match="output payload and evidence paths must differ"):
        cli.main()
    assert not shared_output.exists()
