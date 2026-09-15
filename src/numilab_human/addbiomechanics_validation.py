"""Evaluate a source-bound one-subject AddBiomechanics validation handoff.

The public AddBiomechanics aggregate is too large to treat as an implicit
dependency.  This command therefore consumes one user-selected subject
artifact and explicit reference/prediction tables.  It verifies the subject,
provenance, calibration/held-out split, common time grid and the 12.5 us
prediction clock before computing declared joint-angle, ground-reaction-force
and joint-moment residuals.

The result is a measured candidate, never a claim that the Human mechanics,
standing, walking, activation or material gates are qualified.  A fixture
manifest is accepted only with an explicit fixture provenance and is labelled
``fixture_only`` in the result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .model import ImportError as HumanImportError

SCHEMA = "HumanPack.addbiomechanics-validation.v1"
COMPILER = "numilab-human.addbiomechanics-validation.1"
CLOCK_NS = 12_500
DATASET_URL = "https://www.addbiomechanics.org/download_data.html"
LICENSE = "CC BY 4.0"
KINDS = {"joint_angle", "grf", "joint_moment"}
SPLITS = {"calibration", "validation"}
ACTIVITIES = {"standing", "recovery", "walking", "other"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA1 = re.compile(r"^[0-9a-f]{40}$")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("AddBiomechanics validation: " + message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HumanImportError("validation manifest contains non-finite JSON") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
    except OSError as error:
        raise HumanImportError(f"cannot read {path}") from error
    return result.hexdigest()


def _object(value: Any, fields: set[str], label: str, optional: set[str] | None = None) -> dict[str, Any]:
    allowed = fields | (optional or set())
    _need(isinstance(value, dict) and fields <= value.keys() and set(value) <= allowed,
          f"{label} fields differ")
    return value


def _text(value: Any, label: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    return value.strip()


def _finite(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    _need(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    result = float(value)
    if positive:
        _need(result > 0.0, f"{label} must be positive")
    if nonnegative:
        _need(result >= 0.0, f"{label} must be nonnegative")
    return result


def _relative(root: Path, value: Any, label: str) -> tuple[Path, str]:
    name = _text(value, label)
    relative = PurePosixPath(name)
    _need(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
          f"{label} is unsafe")
    candidate = root / Path(*relative.parts)
    _need(not candidate.is_symlink(), f"{label} may not be a symlink")
    path = candidate.resolve()
    _need(path.is_relative_to(root.resolve()), f"{label} escapes manifest directory")
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path, "/".join(relative.parts)


def _artifact(root: Path, value: Any, label: str) -> dict[str, Any]:
    descriptor = _object(value, {"path", "bytes", "sha256"}, label)
    path, relative = _relative(root, descriptor["path"], label + " path")
    _need(type(descriptor["bytes"]) is int and descriptor["bytes"] >= 0,
          f"{label} byte count is invalid")
    expected = _text(descriptor["sha256"], label + " SHA-256").lower()
    _need(SHA256.fullmatch(expected) is not None, f"{label} SHA-256 is invalid")
    actual = file_sha256(path)
    _need(path.stat().st_size == descriptor["bytes"], f"{label} byte count changed")
    _need(actual == expected, f"{label} SHA-256 changed")
    return {"path": relative, "bytes": descriptor["bytes"], "sha256": actual}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                _need(key not in result, f"duplicate JSON key {key}")
                result[key] = value
            return result

        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except HumanImportError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise HumanImportError(f"invalid validation manifest {path.name}") from error
    _need(isinstance(value, dict), "validation manifest must be an object")
    canonical(value)
    return value


def _split_line(line: str) -> list[str]:
    if "," in line:
        return [item.strip() for item in next(csv.reader([line]))]
    return line.strip().split()


def _table(path: Path, columns: set[str], time_column: str) -> tuple[list[float], dict[str, list[float]]]:
    """Read a small CSV or OpenSim .sto/.mot table without resampling."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise HumanImportError(f"cannot read signal table {path.name}") from error
    _need(lines, f"signal table {path.name} is empty")
    header_index = None
    for index, raw in enumerate(lines):
        if raw.strip().lower() == "endheader":
            header_index = index + 1
            break
    if header_index is None:
        header_index = next((index for index, raw in enumerate(lines) if raw.strip() and not raw.lstrip().startswith("#")), None)
    _need(header_index is not None and header_index < len(lines), f"signal table {path.name} has no header")
    header = _split_line(lines[header_index])
    _need(header and len(set(header)) == len(header), f"signal table {path.name} has duplicate columns")
    _need(time_column in header and columns <= set(header), f"signal table {path.name} is missing declared columns")
    selected = [time_column, *sorted(columns)]
    positions = {name: header.index(name) for name in selected}
    rows: list[list[float]] = []
    for number, raw in enumerate(lines[header_index + 1:], header_index + 2):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = _split_line(raw)
        _need(len(fields) == len(header), f"signal table {path.name} row {number} has wrong column count")
        try:
            row = [float(field) for field in fields]
        except ValueError as error:
            raise HumanImportError(f"signal table {path.name} row {number} is not numeric") from error
        _need(all(math.isfinite(value) for value in row), f"signal table {path.name} row {number} is nonfinite")
        rows.append(row)
        _need(len(rows) <= 2_000_000, f"signal table {path.name} exceeds row bound")
    _need(len(rows) >= 2, f"signal table {path.name} needs at least two rows")
    times = [row[positions[time_column]] for row in rows]
    _need(all(right > left for left, right in zip(times, times[1:])),
          f"signal table {path.name} time must strictly increase")
    return times, {name: [row[positions[name]] for row in rows] for name in sorted(columns)}


def _metric(kind: str, reference: list[float], prediction: list[float], *, mass_kg: float, height_m: float) -> dict[str, float | str]:
    errors = [predicted - measured for measured, predicted in zip(reference, prediction)]
    rms = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    mean_abs = math.fsum(abs(error) for error in errors) / len(errors)
    if kind == "joint_angle":
        return {"metric": "mae_deg", "value": math.degrees(mean_abs), "rmse_rad": rms}
    if kind == "grf":
        body_weight = mass_kg * 9.80665
        return {"metric": "normalized_rmse_body_weight", "value": rms / body_weight, "rmse_n": rms}
    scale = mass_kg * 9.80665 * height_m
    return {"metric": "normalized_rmse_body_weight_height", "value": rms / scale, "rmse_nm": rms}


def _validate_manifest(value: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _object(value, {"schema", "dataset", "subject", "time_column", "time_tolerance_s", "required_activities", "channels", "trials", "prediction"}, "validation manifest")
    _need(value["schema"] == SCHEMA, f"unsupported schema {value['schema']!r}")
    dataset = _object(value["dataset"], {"id", "version", "url", "license", "provenance", "source_artifact"}, "dataset")
    _need(_text(dataset["id"], "dataset id") == "addbiomechanics-core-v1", "unsupported AddBiomechanics dataset")
    _need(_text(dataset["version"], "dataset version") == "1.0", "unsupported AddBiomechanics version")
    _need(_text(dataset["url"], "dataset URL") == DATASET_URL, "dataset URL is not the official download page")
    provenance = _object(dataset["provenance"], {"kind", "note"}, "dataset provenance")
    _need(provenance["kind"] in {"public_dataset", "fixture_only"}, "dataset provenance kind is unsupported")
    _text(provenance["note"], "dataset provenance note")
    if provenance["kind"] == "public_dataset":
        _need(_text(dataset["license"], "dataset license") == LICENSE, "public dataset license must be CC BY 4.0")
    else:
        _need(_text(dataset["license"], "fixture license") == "fixture-only", "fixture license must be explicit")
    source_artifact = _artifact(root, dataset["source_artifact"], "source artifact")

    subject = _object(value["subject"], {"id", "sex", "age_years", "height_m", "mass_kg"}, "subject")
    _text(subject["id"], "subject id")
    _need(_text(subject["sex"], "subject sex").lower() == "male", "this release requires one male subject")
    _need(type(subject["age_years"]) is int and 18 <= subject["age_years"] <= 120,
          "subject age must be a recorded adult age in years")
    height = _finite(subject["height_m"], "subject height", positive=True)
    mass = _finite(subject["mass_kg"], "subject mass", positive=True)
    _need(0.5 <= height <= 2.5 and 20.0 <= mass <= 300.0, "subject dimensions are outside bounded human limits")

    time_column = _text(value["time_column"], "time column")
    tolerance = _finite(value["time_tolerance_s"], "time tolerance", nonnegative=True)
    _need(tolerance <= 1.0e-3, "time tolerance is too loose for held-out comparison")
    required_activities = value["required_activities"]
    _need(isinstance(required_activities, list) and required_activities and
          all(item in ACTIVITIES for item in required_activities) and
          len(set(required_activities)) == len(required_activities),
          "required activities are invalid")

    channels = value["channels"]
    _need(isinstance(channels, list) and channels, "channels must be nonempty")
    channel_by_id: dict[str, dict[str, Any]] = {}
    columns: set[str] = set()
    for channel in channels:
        row = _object(channel, {"id", "kind", "column", "unit", "threshold"}, "channel")
        identifier = _text(row["id"], "channel id")
        _need(identifier not in channel_by_id, f"duplicate channel id {identifier}")
        _need(row["kind"] in KINDS, f"unsupported channel kind {row['kind']!r}")
        column = _text(row["column"], f"channel {identifier} column")
        _need(column not in columns, f"duplicate channel column {column}")
        expected_unit = {"joint_angle": "rad", "grf": "N", "joint_moment": "N*m"}[row["kind"]]
        _need(_text(row["unit"], f"channel {identifier} unit") == expected_unit,
              f"channel {identifier} must use {expected_unit}")
        _finite(row["threshold"], f"channel {identifier} threshold", nonnegative=True)
        channel_by_id[identifier] = row
        columns.add(column)

    trials = value["trials"]
    _need(isinstance(trials, list) and trials, "trials must be nonempty")
    trial_ids: set[str] = set()
    paths: set[str] = set()
    normalized_trials: list[dict[str, Any]] = []
    for trial in trials:
        row = _object(trial, {"id", "activity", "split", "reference", "prediction"}, "trial")
        identifier = _text(row["id"], "trial id")
        _need(identifier not in trial_ids, f"duplicate trial id {identifier}")
        _need(row["activity"] in ACTIVITIES, f"trial {identifier} activity is unsupported")
        _need(row["split"] in SPLITS, f"trial {identifier} split is unsupported")
        reference = _artifact(root, row["reference"], f"trial {identifier} reference")
        prediction = _artifact(root, row["prediction"], f"trial {identifier} prediction")
        _need(reference["path"] not in paths and prediction["path"] not in paths,
              f"trial {identifier} reuses a table path")
        paths.update((reference["path"], prediction["path"]))
        trial_ids.add(identifier)
        normalized_trials.append({"id": identifier, "activity": row["activity"], "split": row["split"],
                                  "reference": reference, "prediction": prediction})
    calibration_ids = {row["id"] for row in normalized_trials if row["split"] == "calibration"}
    validation_ids = {row["id"] for row in normalized_trials if row["split"] == "validation"}
    _need(calibration_ids and validation_ids and calibration_ids.isdisjoint(validation_ids),
          "calibration and held-out validation trials must both be present and disjoint")

    producer = _object(value["prediction"], {"runtime", "source_revision", "clock_nanoseconds", "fit_trial_ids"}, "prediction producer")
    _text(producer["runtime"], "prediction runtime")
    revision = _text(producer["source_revision"], "prediction source revision").lower()
    _need(SHA1.fullmatch(revision) is not None, "prediction source revision must be a 40-character SHA-1")
    _need(type(producer["clock_nanoseconds"]) is int and producer["clock_nanoseconds"] == CLOCK_NS,
          "prediction clock must be the canonical 12.5 microsecond clock")
    fit_ids = producer["fit_trial_ids"]
    _need(isinstance(fit_ids, list) and set(fit_ids) == calibration_ids and len(fit_ids) == len(calibration_ids),
          "prediction fit scope must equal the calibration trials exactly")
    return (dict(dataset, source_artifact=source_artifact), dict(subject, height_m=height, mass_kg=mass), normalized_trials,
            {"time_column": time_column, "time_tolerance_s": tolerance, "required_activities": required_activities,
             "channels": channel_by_id, "producer": dict(producer, source_revision=revision)})


def compile_candidate(manifest: Path) -> dict[str, Any]:
    """Verify and evaluate one immutable subject handoff."""
    manifest = Path(manifest)
    _need(manifest.is_file() and not manifest.is_symlink(), "manifest is not a regular file")
    manifest = manifest.resolve()
    value = _read_json(manifest)
    dataset, subject, trials, options = _validate_manifest(value, manifest.parent)
    channels = options["channels"]
    trial_reports: list[dict[str, Any]] = []
    split_errors: dict[str, dict[str, list[float]]] = {split: {identifier: [] for identifier in channels} for split in SPLITS}
    for trial in trials:
        reference_path = manifest.parent / Path(*PurePosixPath(trial["reference"]["path"]).parts)
        prediction_path = manifest.parent / Path(*PurePosixPath(trial["prediction"]["path"]).parts)
        times_ref, ref = _table(reference_path, {row["column"] for row in channels.values()}, options["time_column"])
        times_pred, pred = _table(prediction_path, {row["column"] for row in channels.values()}, options["time_column"])
        _need(len(times_ref) == len(times_pred), f"trial {trial['id']} reference/prediction row counts differ")
        _need(all(abs(left - right) <= options["time_tolerance_s"] for left, right in zip(times_ref, times_pred)),
              f"trial {trial['id']} reference/prediction time grids differ")
        channel_reports: dict[str, dict[str, Any]] = {}
        for identifier, channel in channels.items():
            metric = _metric(channel["kind"], ref[channel["column"]], pred[channel["column"]],
                             mass_kg=subject["mass_kg"], height_m=subject["height_m"])
            metric["threshold"] = float(channel["threshold"])
            metric["passed"] = float(metric["value"]) <= float(channel["threshold"])
            channel_reports[identifier] = metric
            split_errors[trial["split"]][identifier].extend(
                abs(p - r) if channel["kind"] == "joint_angle" else (p - r) ** 2
                for r, p in zip(ref[channel["column"]], pred[channel["column"]])
            )
        trial_reports.append({"id": trial["id"], "activity": trial["activity"], "split": trial["split"],
                             "sample_count": len(times_ref), "start_time_s": times_ref[0],
                             "end_time_s": times_ref[-1], "channels": channel_reports})

    split_reports: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        channels_report: dict[str, Any] = {}
        for identifier, channel in channels.items():
            errors = split_errors[split][identifier]
            _need(errors, f"split {split} has no samples for {identifier}")
            if channel["kind"] == "joint_angle":
                value = math.degrees(math.fsum(errors) / len(errors))
                metric_name = "mae_deg"
                raw_name = "mean_absolute_error_rad"
                raw_value = math.fsum(errors) / len(errors)
            else:
                rms = math.sqrt(math.fsum(errors) / len(errors))
                scale = subject["mass_kg"] * 9.80665
                if channel["kind"] == "joint_moment":
                    scale *= subject["height_m"]
                value = rms / scale
                metric_name = "normalized_rmse_body_weight_height" if channel["kind"] == "joint_moment" else "normalized_rmse_body_weight"
                raw_name = "rmse_nm" if channel["kind"] == "joint_moment" else "rmse_n"
                raw_value = rms
            threshold = float(channel["threshold"])
            channels_report[identifier] = {"metric": metric_name, "value": value, "threshold": threshold,
                                           "passed": value <= threshold, raw_name: raw_value}
        split_reports[split] = {"trial_count": sum(1 for row in trials if row["split"] == split),
                                "sample_count": sum(row["sample_count"] for row in trial_reports if row["split"] == split),
                                "channels": channels_report,
                                "passed": all(row["passed"] for row in channels_report.values())}

    activities = {row["activity"] for row in trials}
    required = set(options["required_activities"])
    source_public = dataset["provenance"]["kind"] == "public_dataset"
    qualification = {
        "source_artifact_hash_verified": True,
        "public_dataset_provenance": source_public,
        "single_adult_male": True,
        "calibration_validation_disjoint": True,
        "prediction_fit_scope_is_calibration_only": True,
        "canonical_clock_12_5_us": True,
        "required_activity_coverage": required <= activities,
        "calibration_metrics_within_declared_thresholds": split_reports["calibration"]["passed"],
        "held_out_metrics_within_declared_thresholds": split_reports["validation"]["passed"],
        # The command does not qualify a runtime, material, contact or behavior release.
        "native_mechanics_qualified": False,
        "activation_calibration_qualified": False,
        "anatomical_support_loading_qualified": False,
        "materials_qualified": False,
        "standing_or_walking_qualified": False,
        "release_qualified": False,
    }
    status = "measured_candidate" if source_public else "fixture_only"
    return {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": status,
        "manifest_sha256": digest(value),
        "dataset": dataset,
        "subject": subject,
        "clock": {"nanoseconds": CLOCK_NS, "seconds": CLOCK_NS * 1.0e-9, "exact": True},
        "prediction": options["producer"],
        "required_activities": options["required_activities"],
        "activity_coverage": {activity: activity in activities for activity in sorted(required)},
        "trials": trial_reports,
        "splits": split_reports,
        "qualification": qualification,
        "boundary": (
            "This is a one-subject source-data comparison. It verifies provenance, disjoint fit/held-out "
            "trials, aligned tables and declared residuals; it does not fit muscle parameters, establish "
            "anatomical contact/loading, validate 12.5 us force convergence, or qualify standing, recovery, "
            "walking, blood transfer, tissue materials or whole-body mechanics."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    encoded = canonical(value) + b"\n"
    path = Path(path)
    if path.exists():
        _need(path.is_file() and not path.is_symlink() and path.read_bytes() == encoded, "output is immutable")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True,
                        help="one-subject AddBiomechanics validation manifest")
    parser.add_argument("--output", type=Path,
                        help="optional immutable JSON result path")


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(arguments.manifest)
    if arguments.output is None:
        print((canonical(result) + b"\n").decode("utf-8"), end="")
    else:
        immutable_write(arguments.output, result)
        print(f"wrote {arguments.output.resolve()}")
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
