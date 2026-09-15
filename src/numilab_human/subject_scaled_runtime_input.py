"""Patch the published NHRIGID2 mass/inertia input for the subject candidate.

This is a binary handoff for native testing.  It validates the exact NHRIGID2
layout, maps all 103 source bodies to their 157 engine bodies, and changes
only mass, inverse mass, inertia, and inverse inertia.  It never changes the
body tree, poses, joints, source identity, or runtime qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.subject-scaled-runtime-input.v1"
COMPILER = "numilab-human.subject-scaled-runtime-input.1"
SCALING = ROOT / "Docs/media/subject-mass-scaling-20260915/receipt-v1.json"
RIGID = ROOT / "Docs/media/native-runtime-source-package-20260915/input/myosim-fullbody-core-reference.nhrigid"
HEADER = struct.Struct("<8s10I32s")
BODY_RECORD_BYTES = 160
BODY_RECORD_OFFSET = 224


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("subject-scaled runtime input: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    value = read_json(path)
    return value, _sha256(path.read_bytes())


def _read_bytes(path: Path, label: str) -> tuple[bytes, str]:
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    data = path.read_bytes()
    return data, _sha256(data)


def _float(data: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", data, offset)[0])


def _write_float(data: bytearray, offset: int, value: float) -> None:
    _need(math.isfinite(value), "scaled payload contains a nonfinite value")
    struct.pack_into("<f", data, offset, float(value))


def _validate_header(raw: bytes) -> tuple[tuple[Any, ...], int, int, int, int, int]:
    _need(len(raw) >= BODY_RECORD_OFFSET, "NHRIGID2 payload is truncated")
    header = HEADER.unpack_from(raw, 0)
    magic, payload_abi, engine_abi, source_count, body_count, joints, nq, nv, root, virtual, reserved, source_sha = header
    _need(magic == b"NHRIGID2" and payload_abi == 1 and engine_abi == 5,
          "unsupported NHRIGID2 header")
    _need(root == 0 and reserved == 0 and 0 < source_count <= body_count,
          "NHRIGID2 body counts are invalid")
    expected = (BODY_RECORD_OFFSET + BODY_RECORD_BYTES * body_count + 144 * joints
                + 64 * nv + 4 * (nq + nv) + 32 * source_count)
    _need(len(raw) == expected, "NHRIGID2 record layout changed")
    return header, source_count, body_count, joints, nq, nv


def _scaling_rows(value: dict[str, Any], source_count: int) -> list[dict[str, Any]]:
    _need(value.get("schema") == "HumanPack.subject-mass-scaling-candidate.v1" and
          value.get("status") == "partial", "scaling candidate boundary changed")
    qualification = value.get("qualification", {})
    _need(qualification.get("target_mass_closure") is True and
          qualification.get("mechanical_runtime_admitted") is False,
          "scaling candidate was promoted to runtime")
    rows = value.get("bodies")
    _need(isinstance(rows, list) and len(rows) == source_count,
          "scaling candidate body count does not match NHRIGID2")
    rows = sorted(rows, key=lambda row: row.get("id", -1))
    _need([row.get("id") for row in rows] == list(range(1, source_count + 1)),
          "scaling candidate source IDs are not contiguous")
    return rows


def compile_input(
    *,
    scaling: Path = SCALING,
    rigid: Path = RIGID,
    output: Path,
    receipt: Path,
) -> dict[str, Any]:
    scaling_document, scaling_sha = _read_json(Path(scaling), "scaling candidate")
    source, source_sha = _read_bytes(Path(rigid), "source NHRIGID2")
    header, source_count, body_count, joints, nq, nv = _validate_header(source)
    rows = _scaling_rows(scaling_document, source_count)
    map_offset = len(source) - 32 * source_count
    mapping = [struct.unpack_from("<I", source, map_offset + 4 * index)[0]
               for index in range(source_count)]
    _need(len(set(mapping)) == source_count and all(index < body_count for index in mapping),
          "NHRIGID2 source map is not one-to-one")

    patched = bytearray(source)
    source_mass_sum = 0.0
    scaled_mass_sum = 0.0
    patched_rows: list[dict[str, Any]] = []
    max_source_mass_error = 0.0
    for source_index, row in enumerate(rows):
        core_index = mapping[source_index]
        body_offset = BODY_RECORD_OFFSET + BODY_RECORD_BYTES * core_index
        binary_mass = _float(source, body_offset + 16)
        source_mass = float(row["source_mass_kg"])
        scaled_mass = float(row["scaled_mass_kg"])
        _need(math.isfinite(source_mass) and source_mass >= 0.0 and
              math.isfinite(scaled_mass) and scaled_mass >= 0.0,
              f"mass row {source_index + 1} is invalid")
        mass_error = abs(binary_mass - source_mass)
        max_source_mass_error = max(max_source_mass_error, mass_error)
        _need(mass_error <= 2.0e-5,
              f"source mass row {source_index + 1} disagrees with NHRIGID2")
        source_mass_sum += binary_mass
        scaled_mass_sum += scaled_mass
        _write_float(patched, body_offset + 16, scaled_mass)
        _write_float(patched, body_offset + 20, 1.0 / scaled_mass if scaled_mass > 0.0 else 0.0)
        source_inertia = row["source_inertia_kg_m2"]
        scaled_inertia = row["scaled_inertia_kg_m2"]
        _need(isinstance(source_inertia, list) and len(source_inertia) == 3 and
              isinstance(scaled_inertia, list) and len(scaled_inertia) == 3,
              f"inertia row {source_index + 1} is malformed")
        for diagonal in range(3):
            inertia_offset = body_offset + 48 + 16 * diagonal + 4 * diagonal
            inverse_offset = body_offset + 96 + 16 * diagonal + 4 * diagonal
            binary_inertia = _float(source, inertia_offset)
            expected_inertia = float(source_inertia[diagonal])
            _need(abs(binary_inertia - expected_inertia) <= 2.0e-5,
                  f"source inertia row {source_index + 1} disagrees with NHRIGID2")
            new_inertia = float(scaled_inertia[diagonal])
            _need(math.isfinite(new_inertia) and new_inertia >= 0.0,
                  f"scaled inertia row {source_index + 1} is invalid")
            _write_float(patched, inertia_offset, new_inertia)
            _write_float(patched, inverse_offset, 1.0 / new_inertia if new_inertia > 0.0 else 0.0)
        patched_rows.append({"source_index": source_index, "core_body_index": core_index,
                             "owner_id": row["owner_id"], "source_mass_kg": source_mass,
                             "scaled_mass_kg": scaled_mass})

    scaled_bytes = bytes(patched)
    scaled_source_mass_sum = sum(_float(scaled_bytes, BODY_RECORD_OFFSET + BODY_RECORD_BYTES * index + 16)
                                 for index in range(body_count))
    target_mass = float(scaling_document["scaling"]["target_mass_kg"])
    _need(abs(scaled_source_mass_sum - target_mass) <= 2.0e-5,
          "scaled NHRIGID2 mass does not close target mass")
    _need(abs(scaled_mass_sum - target_mass) <= 1.0e-12,
          "scaling receipt does not close target mass")
    output = Path(output)
    _need(not output.exists() or not output.is_symlink(), "scaled output is redirected")
    if output.exists():
        _need(output.read_bytes() == scaled_bytes, "scaled binary output is immutable")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(scaled_bytes)
    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": "partial",
        "source": {
            "path": _relative(Path(rigid)), "bytes": len(source), "sha256": source_sha,
            "magic": "NHRIGID2", "payload_abi": header[1], "engine_abi": header[2],
            "source_body_count": source_count, "engine_body_count": body_count,
            "joint_count": joints, "nq": nq, "nv": nv,
        },
        "scaling_candidate": {"path": _relative(Path(scaling)), "sha256": scaling_sha,
                               "schema": scaling_document["schema"]},
        "output": {"path": _relative(output), "bytes": len(scaled_bytes),
                   "sha256": _sha256(scaled_bytes)},
        "mapping": {"mapped_source_rows": source_count, "patched_engine_rows": len(patched_rows),
                    "max_source_mass_error_kg": max_source_mass_error,
                    "source_binary_mass_kg": source_mass_sum,
                    "scaled_binary_mass_kg": scaled_source_mass_sum,
                    "target_mass_kg": target_mass, "rows": patched_rows},
        "qualification": {
            "nhrigid_layout_verified": True,
            "source_identity_preserved": True,
            "body_tree_preserved": True,
            "source_mass_and_inertia_rows_verified": True,
            "scaled_mass_closure": True,
            "native_binary_consumed": False,
            "native_replay_qualified": False,
            "subject_mass_calibration": False,
            "geometry_calibration": False,
            "inertia_calibration": False,
            "organ_blood_tissue_fat_muscle_ownership": False,
            "standing": False,
            "recovery": False,
            "walking": False,
        },
        "blockers": [
            {"id": "native_subject_scaled_replay", "status": "open",
             "reason": "The patched binary must be consumed by the public native runtime and compared against the exact source tuple."},
            {"id": "segment_geometry_inertia_validation", "status": "open",
             "reason": "This handoff uses uniform mass-only scaling with fixed geometry; measured segment composition and inertial validation are absent."},
            {"id": "prediction_calibration", "status": "open",
             "reason": "No Numi predictions have been compared with the acquired gait/stair reference tables."},
        ],
        "boundary": (
            "This receipt is a binary source handoff that patches only mapped NHRIGID2 "
            "mass/inertia fields using the explicit uniform subject-mass candidate. It "
            "preserves source identity, body topology, poses and joints. It does not "
            "qualify native replay, segment composition, geometry or inertia calibration, "
            "organ/blood/tissue/fat/muscle ownership, standing, recovery or walking."
        ),
    }
    receipt = Path(receipt)
    payload = canonical(result) + b"\n"
    _need(not receipt.exists() or not receipt.is_symlink(), "receipt output is redirected")
    if receipt.exists():
        _need(receipt.read_bytes() == payload, "receipt output is immutable")
    else:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("xb") as stream:
            stream.write(payload)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scaling", type=Path, default=SCALING)
    parser.add_argument("--rigid", type=Path, default=RIGID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)


def run(arguments: argparse.Namespace) -> int:
    result = compile_input(scaling=arguments.scaling, rigid=arguments.rigid,
                          output=arguments.output, receipt=arguments.receipt)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(Path(arguments.output).resolve()),
                      "receipt": str(Path(arguments.receipt).resolve()),
                      "scaled_binary_mass_kg": result["mapping"]["scaled_binary_mass_kg"]},
                     sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"subject-scaled runtime input: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
