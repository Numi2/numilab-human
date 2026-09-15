"""Acquire and extract one real AddBiomechanics B3D subject.

The aggregate AddBiomechanics archive is not an implicit dependency of the
Human package.  This command binds one selected subject-level ``.b3d`` file,
checks its protobuf header with a small dependency-free wire decoder, and
extracts measured reference tables for later Numi prediction comparison.

The emitted receipt is an acquisition/calibration input record.  It does not
claim that Numi Human has matched the observations or that any mechanics,
material, contact, blood, tissue, standing, or walking gate is qualified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.addbiomechanics-acquisition.v1"
COMPILER = "numilab-human.addbiomechanics-acquisition.1"
DATASET_URL = "https://www.addbiomechanics.org/download_data.html"
ARCHIVE_URL = "http://archive.simtk.org/addbiomechanics/addbiomechanics.zip"
ARCHIVE_ENTRY = "train/With_Arm/Falisse2017_Formatted_With_Arm/subject_1/subject_1.b3d"
ARCHIVE_COMPRESSED_BYTES = 7_041_122
ARCHIVE_COMPRESSED_SHA256 = "bd31e6d125cb354cef761fcb2ff8153a67eab6ffbd2d41ce4829463892905c40"
RAW_BYTES = 13_445_904
RAW_SHA256 = "2d1f9eb4c9173989dac5dd0c2cfd1110693a76a5bfc4e85c1aafab713d3ce0af"
LICENSE = "CC BY 4.0"
SELECTED_CHANNELS = (
    ("joint_angle_pelvis_tilt", "joint_angle", "pelvis_tilt", "joint_angle_pelvis_tilt_rad", "rad"),
    ("joint_angle_hip_flexion_r", "joint_angle", "hip_flexion_r", "joint_angle_hip_flexion_r_rad", "rad"),
    ("joint_angle_knee_angle_r", "joint_angle", "knee_angle_r", "joint_angle_knee_angle_r_rad", "rad"),
    ("joint_angle_ankle_angle_r", "joint_angle", "ankle_angle_r", "joint_angle_ankle_angle_r_rad", "rad"),
    ("grf_total_vertical", "grf", "", "grf_total_vertical_N", "N"),
    ("joint_moment_knee_angle_r", "joint_moment", "knee_angle_r", "knee_angle_r_moment_Nm", "N*m"),
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("AddBiomechanics acquisition: " + message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HumanImportError("acquisition receipt contains non-finite JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise HumanImportError(f"cannot read source artifact {path}") from error
    return digest.hexdigest()


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        _need(offset < len(data), "protobuf varint is truncated")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        _need(shift <= 63, "protobuf varint is too large")


def _fields(data: bytes) -> Iterable[tuple[int, int, bytes | int]]:
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        _need(number > 0, "protobuf field number is invalid")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            _need(offset + 8 <= len(data), "protobuf fixed64 field is truncated")
            value = data[offset:offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _varint(data, offset)
            _need(length <= len(data) - offset, "protobuf bytes field is truncated")
            value = data[offset:offset + length]
            offset += length
        elif wire == 5:
            _need(offset + 4 <= len(data), "protobuf fixed32 field is truncated")
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise HumanImportError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _varint_field(data: bytes | int, label: str) -> int:
    _need(isinstance(data, int), f"{label} is not a varint")
    return int(data)


def _bytes_field(data: bytes | int, label: str) -> bytes:
    _need(isinstance(data, bytes), f"{label} is not a bytes field")
    return data


def _text_field(data: bytes | int, label: str) -> str:
    try:
        text = _bytes_field(data, label).decode("utf-8")
    except UnicodeDecodeError as error:
        raise HumanImportError(f"{label} is not UTF-8") from error
    _need(text.strip() != "", f"{label} is empty")
    return text


def _double_field(data: bytes | int, label: str) -> float:
    raw = _bytes_field(data, label)
    _need(len(raw) == 8, f"{label} is not fixed64")
    value = struct.unpack("<d", raw)[0]
    _need(math.isfinite(value), f"{label} is not finite")
    return value


def _packed_doubles(data: bytes | int, label: str) -> list[float]:
    raw = _bytes_field(data, label)
    _need(len(raw) % 8 == 0, f"{label} is not a packed double array")
    values = [struct.unpack_from("<d", raw, offset)[0] for offset in range(0, len(raw), 8)]
    _need(all(math.isfinite(value) for value in values), f"{label} contains nonfinite values")
    return values


def _parse_pass(data: bytes) -> dict[str, Any]:
    model = ""
    pass_type = 0
    for number, wire, value in _fields(data):
        if number == 1:
            pass_type = _varint_field(value, "processing pass type")
        elif number == 2:
            _need(wire == 2, "processing pass model must be length-delimited")
            model = _text_field(value, "processing pass model")
    _need(model, "processing pass has no OpenSim model")
    return {"type": pass_type, "model_osim_text": model}


def _parse_trial(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"name": "", "length": 0, "timestep": 0.0,
                              "passes": [], "tags": [], "features": []}
    for number, wire, value in _fields(data):
        if number == 1:
            result["name"] = _text_field(value, "trial name")
        elif number == 3:
            result["length"] = _varint_field(value, "trial length")
        elif number == 4:
            result["timestep"] = _double_field(value, "trial timestep")
        elif number == 5:
            _need(wire == 2, "trial processing pass must be nested")
            result["passes"].append(_parse_pass(_bytes_field(value, "trial processing pass")))
        elif number == 6:
            result["tags"].append(_text_field(value, "trial tag"))
        elif number == 18:
            result["features"].append(_varint_field(value, "trial feature"))
    _need(result["name"] and result["length"] > 0 and result["timestep"] > 0.0,
          "trial header is incomplete")
    return result


def _coordinate_names(model: str) -> list[str]:
    try:
        root = ET.fromstring(model)
    except ET.ParseError as error:
        raise HumanImportError("OpenSim model XML is invalid") from error
    names: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Coordinate" and element.get("name"):
            names.append(str(element.get("name")))
    _need(names, "OpenSim model has no coordinate names")
    _need(len(names) == len(set(names)), "OpenSim model coordinate names are duplicated")
    return names


def _parse_header(raw: bytes) -> tuple[dict[str, Any], int]:
    _need(len(raw) >= 8, "B3D source is shorter than its header length")
    header_size = struct.unpack_from("<q", raw, 0)[0]
    _need(0 < header_size <= len(raw) - 8, "B3D header length is invalid")
    header_bytes = raw[8:8 + header_size]
    header: dict[str, Any] = {"dofs": 0, "sensor_size": 0, "pass_size": 0,
                              "passes": [], "contacts": [], "trials": [],
                              "version": 0, "href": "", "notes": "", "sex": "",
                              "height": 0.0, "mass": 0.0, "age": 0, "tags": []}
    for number, wire, value in _fields(header_bytes):
        if number == 1:
            header["dofs"] = _varint_field(value, "B3D dof count")
        elif number == 3:
            header["sensor_size"] = _varint_field(value, "B3D sensor frame size")
        elif number == 4:
            header["pass_size"] = _varint_field(value, "B3D processing pass frame size")
        elif number == 5:
            header["passes"].append(_parse_pass(_bytes_field(value, "B3D processing pass")))
        elif number == 6:
            header["contacts"].append(_text_field(value, "B3D contact body"))
        elif number == 9:
            header["trials"].append(_parse_trial(_bytes_field(value, "B3D trial")))
        elif number == 10:
            header["version"] = _varint_field(value, "B3D version")
        elif number == 11:
            header["href"] = _text_field(value, "B3D href")
        elif number == 12:
            header["notes"] = _text_field(value, "B3D notes")
        elif number == 13:
            header["sex"] = _text_field(value, "B3D biological sex")
        elif number == 14:
            header["height"] = _double_field(value, "B3D height")
        elif number == 15:
            header["mass"] = _double_field(value, "B3D mass")
        elif number == 16:
            header["age"] = _varint_field(value, "B3D age")
        elif number == 23:
            header["tags"].append(_text_field(value, "B3D subject tag"))
    _need(header["version"] >= 4, "unsupported B3D version")
    _need(header["sensor_size"] > 0 and header["pass_size"] > 0 and header["passes"],
          "B3D frame sizes or processing passes are missing")
    _need(header["trials"], "B3D has no trials")
    header["coordinates"] = _coordinate_names(header["passes"][0]["model_osim_text"])
    return header, 8 + header_size


def _frame(data: bytes, offset: int, sensor_size: int, pass_size: int,
           pass_count: int, pass_index: int, dofs: int, contact_count: int) -> tuple[dict[str, list[float]], int]:
    end = offset + sensor_size + pass_count * pass_size
    _need(end <= len(data), "B3D frame extends past source artifact")
    offset += sensor_size + pass_index * pass_size
    fields = list(_fields(data[offset:offset + pass_size]))
    result: dict[str, list[float]] = {"pos": [], "tau": [], "force": []}
    for number, wire, value in fields:
        if number == 1:
            result["pos"] = _packed_doubles(value, "B3D position")
        elif number == 4:
            result["tau"] = _packed_doubles(value, "B3D joint moment")
        elif number == 8:
            result["force"] = _packed_doubles(value, "B3D ground force")
    _need(len(result["pos"]) == dofs, "B3D position dimension differs from header")
    _need(len(result["tau"]) == dofs, "B3D joint moment dimension differs from header")
    _need(len(result["force"]) in {0, contact_count * 3}, "B3D ground force dimension differs from contacts")
    return result, end


def _activity(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if "gait" in lowered or "walk" in lowered:
        return "walking", "gait"
    if "stair" in lowered:
        return "walking", "stair"
    if "stand" in lowered:
        return "standing", "stand"
    if "recover" in lowered:
        return "recovery", "recovery"
    return "other", "unclassified"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _need(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {path}")
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_immutable(path: Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _need(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
              f"immutable output changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(rows: list[dict[str, float]]) -> bytes:
    columns = ["time_s", "joint_angle_pelvis_tilt_rad", "joint_angle_hip_flexion_r_rad",
               "joint_angle_knee_angle_r_rad", "joint_angle_ankle_angle_r_rad",
               "grf_total_vertical_N", "knee_angle_r_moment_Nm"]
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([format(row[column], ".17g") for column in columns])
    return output.getvalue().encode("utf-8")


def acquire(source: Path, output: Path, references: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    _need(source.is_file() and not source.is_symlink(), "B3D source is not a regular file")
    raw = source.read_bytes()
    actual_raw_sha = hashlib.sha256(raw).hexdigest()
    _need(len(raw) == RAW_BYTES and actual_raw_sha == RAW_SHA256,
          "source is not the pinned Falisse2017 subject_1 artifact")
    header, data_offset = _parse_header(raw)
    _need(header["sex"].lower() == "male", "selected source is not male")
    _need(18 <= header["age"] <= 120, "selected source does not carry an adult age")
    _need(0.5 <= header["height"] <= 2.5 and 20.0 <= header["mass"] <= 300.0,
          "selected subject dimensions are outside bounded human limits")
    _need(header["href"].rstrip("/").endswith("Falisse2017_Formatted_With_Arm/subject_1"),
          "B3D href is not the pinned subject")
    selected = {name: index for index, name in enumerate(header["coordinates"])}
    for _, kind, name, _, _ in SELECTED_CHANNELS:
        if kind in {"joint_angle", "joint_moment"}:
            _need(name in selected and selected[name] < header["dofs"],
                  f"selected source coordinate is unavailable: {name}")
    references = Path(references)
    references.mkdir(parents=True, exist_ok=True)
    cursor = data_offset
    trial_reports: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(header["trials"]):
        rows: list[dict[str, float]] = []
        for frame_index in range(trial["length"]):
            frame, cursor = _frame(raw, cursor, header["sensor_size"], header["pass_size"],
                                   len(header["passes"]), len(header["passes"]) - 1,
                                   header["dofs"], len(header["contacts"]))
            force = frame["force"]
            vertical = math.fsum(force[index] for index in range(1, len(force), 3)) if force else 0.0
            rows.append({
                "time_s": frame_index * trial["timestep"],
                "joint_angle_pelvis_tilt_rad": frame["pos"][selected["pelvis_tilt"]],
                "joint_angle_hip_flexion_r_rad": frame["pos"][selected["hip_flexion_r"]],
                "joint_angle_knee_angle_r_rad": frame["pos"][selected["knee_angle_r"]],
                "joint_angle_ankle_angle_r_rad": frame["pos"][selected["ankle_angle_r"]],
                "grf_total_vertical_N": vertical,
                "knee_angle_r_moment_Nm": frame["tau"][selected["knee_angle_r"]],
            })
        activity, detail = _activity(trial["name"])
        reference_path = references / f"trial-{trial_index}-{trial['name'].replace('/', '_')}.csv"
        encoded = _csv_bytes(rows)
        _write_immutable(reference_path, encoded)
        trial_reports.append({
            "id": f"trial_{trial_index}",
            "name": trial["name"],
            "activity": activity,
            "activity_detail": detail,
            "frame_count": len(rows),
            "timestep_s": trial["timestep"],
            "duration_s": (len(rows) - 1) * trial["timestep"],
            "reference": _artifact(reference_path),
            "processing_pass_count": len(header["passes"]),
        })
    _need(cursor <= len(raw), "B3D source has trailing frame offset outside artifact")
    return {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": "source_acquired",
        "dataset": {
            "id": "addbiomechanics-core-v1",
            "version": "1.0",
            "download_page": DATASET_URL,
            "license": LICENSE,
            "archive_url": ARCHIVE_URL,
            "archive_entry": ARCHIVE_ENTRY,
            "archive_compressed_bytes": ARCHIVE_COMPRESSED_BYTES,
            "archive_compressed_sha256": ARCHIVE_COMPRESSED_SHA256,
            "source_artifact": _artifact(source),
            "source_artifact_raw_bytes": RAW_BYTES,
            "source_artifact_raw_sha256": actual_raw_sha,
        },
        "subject": {
            "id": "Falisse2017:subject_1",
            "sex": header["sex"].lower(),
            "age_years": header["age"],
            "height_m": header["height"],
            "mass_kg": header["mass"],
            "href": header["href"],
            "tags": header["tags"],
        },
        "model": {
            "version": header["version"],
            "dof_count": header["dofs"],
            "coordinate_names": header["coordinates"],
            "ground_contact_bodies": header["contacts"],
            "processing_pass_count": len(header["passes"]),
        },
        "channels": [
            {"id": identifier, "kind": kind, "column": column, "unit": unit}
            for identifier, kind, _, column, unit in SELECTED_CHANNELS
        ],
        "trials": trial_reports,
        "qualification": {
            "source_artifact_hash_verified": True,
            "adult_male_metadata_verified": True,
            "measured_reference_tables_extracted": True,
            "walking_reference_present": any(row["activity"] == "walking" for row in trial_reports),
            "standing_reference_present": any(row["activity"] == "standing" for row in trial_reports),
            "recovery_reference_present": any(row["activity"] == "recovery" for row in trial_reports),
            "prediction_comparison_complete": False,
            "activation_calibration_qualified": False,
            "anatomical_support_loading_qualified": False,
            "materials_qualified": False,
            "blood_tissue_mass_transfer_qualified": False,
            "standing_or_walking_qualified": False,
        },
        "boundary": (
            "This receipt binds one real adult male AddBiomechanics subject and extracts measured "
            "kinematic, GRF and joint-moment reference tables. It is the acquisition input for a "
            "later Numi prediction comparison; it does not supply predictions, fit activation or "
            "materials, create adipose/organ/blood mechanical owners, or qualify standing or walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True, help="local subject-level .b3d artifact")
    parser.add_argument("--references", type=Path, required=True,
                        help="directory for extracted immutable reference CSVs")
    parser.add_argument("--output", type=Path, required=True, help="immutable acquisition receipt")


def run(arguments: argparse.Namespace) -> int:
    result = acquire(arguments.source, arguments.output, arguments.references)
    encoded = canonical(result) + b"\n"
    _write_immutable(arguments.output, encoded)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "subject": result["subject"], "trials": len(result["trials"]),
                      "output": str(Path(arguments.output).resolve())}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except HumanImportError as error:
        print(f"numilab-human: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
