#!/usr/bin/env python3
"""Compare retained native CVSim traces with pinned original-C equations; no stepping.

The native log certifies execution/replay. This tool independently checks the
paired reference columns, times, physical admissibility, conservation and
measured timestep refinement. These are numerical gates, not biological ones.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/cvsim21-circulation-20260912"
SOURCE_EVIDENCE = ROOT / "tools/cvsim21_reference/evidence/20260912"
SOURCE = ROOT / "third_party/physionet/cvsim21"
SOURCE_LOCK_SHA256 = "b118d58230a3f2766108d05501a3ac9b910124ef823567a7e5fe23f6df0b4f67"
INITIAL_SHA256 = "fad558482f8bfeb94d531d4538e29d7a2c8a59ea63f32496467b087bcb343ff7"
INITIAL_MANIFEST_SHA256 = "1eedc865d00a9ca84d074a192940f06c2f3e5b3c0d8e67e19fc6c21cb67533e4"
EXECUTION_SHA256 = "0c1b5a4d026fdc7bdf15c5f3e6b43e7b1cc556a4bb00ecb7d45019fc5a7bfb41"
REFERENCE_RAW_SHA256 = "1423dbd3ad5c831780ec4462d8e3db8f151934211b36b100d4aea42bb0d43f7e"
PERIOD = 6.0 / 7.0
TOTAL_VOLUME_M3 = .00515
CONSERVATION_LIMIT = 1e-4
MINIMUM_REFINEMENT_RATIO = 1.5
RUNS = {"cycle-2ms": (.002, 429), "cycle-1ms": (.001, 858),
        "cycle-0p5ms": (.0005, 1716), "ten-cycles-2ms": (.002, 4286)}
REFINEMENT_RUNS = ("cycle-2ms", "cycle-1ms", "cycle-0p5ms")
METRICS = ("max_volume_error_m3", "max_flow_error_m3_per_s", "scaled_state_rms_error")
VOLUME_LABELS = (
    "ascending_aorta", "brachiocephalic_arteries", "upper_body_arteries", "upper_body_veins",
    "superior_vena_cava", "descending_thoracic_aorta", "abdominal_aorta", "renal_arteries",
    "renal_veins", "splanchnic_arteries", "splanchnic_veins", "lower_body_arteries",
    "lower_body_veins", "abdominal_veins", "inferior_vena_cava", "right_atrium",
    "right_ventricle", "pulmonary_arteries", "pulmonary_veins", "left_atrium", "left_ventricle")
ONE_WAY_FLOWS = {0, 16, 19, 20, 23}
SOURCE_HEADER = (["time_s"] + [f"P_{i}_mmHg" for i in range(21)]
                 + [f"V_{i}_mL" for i in range(21)] + [f"Q_{i}_mL_per_s" for i in range(24)]
                 + ["volume_sum_mL", "source_atrial_phase_s", "source_ventricular_phase_s"])
NATIVE_HEADER = ["time_seconds"] + [name + str(i) for i in range(45) for name in ("native_", "reference_")]


class EvidenceError(ValueError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


STATE_SCALE = float32(.0001)
REFERENCE_DT = float32(.0005)


def digest(path: Path, *, decompress: bool = False) -> str:
    result = hashlib.sha256()
    opener = gzip.open if decompress and path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"nonfinite JSON: {token}")))
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def numbers(row: list[str], length: int, label: str) -> list[float]:
    require(len(row) == length, f"{label}: wrong coordinate count")
    try:
        values = list(map(float, row))
    except ValueError as error:
        raise EvidenceError(f"{label}: invalid number") from error
    require(all(math.isfinite(value) for value in values), f"{label}: nonfinite value")
    return values


def coordinate_translation(coordinates: str) -> list[float]:
    require(coordinates in {"upstream_equation", "heldt_table_aligned"}, "unknown volume coordinates")
    values = [0.] * 45
    if coordinates == "heldt_table_aligned":
        # A declared coordinate translation. It changes neither C equations nor
        # conserved total, and is never silently applied to the upstream oracle.
        values[2], values[5] = 184e-6, -184e-6
    return values


def load_reference(path: Path) -> tuple[list[list[float]], dict]:
    require(digest(SOURCE / "source-lock.json") == SOURCE_LOCK_SHA256, "source lock changed")
    require(digest(SOURCE_EVIDENCE / "original-initial.json") == INITIAL_SHA256, "source initializer changed")
    require(digest(SOURCE_EVIDENCE / "initial-manifest.json") == INITIAL_MANIFEST_SHA256, "initializer provenance changed")
    require(digest(SOURCE_EVIDENCE / "execution.json") == EXECUTION_SHA256, "CPU execution provenance changed")
    execution = read_json(SOURCE_EVIDENCE / "execution.json")
    require(execution["source_lock_sha256"] == execution["source_lock_sha256_after"] == SOURCE_LOCK_SHA256,
            "CPU source lock is not frozen")
    require(execution["source_sha256"] == execution["source_after_sha256"], "CPU source changed during execution")
    require(execution["tool_sha256"] == execution["tool_after_sha256"], "CPU reference changed during execution")
    for name, expected in execution["source_sha256"].items():
        require(digest(SOURCE / "21-comp-backend" / name) == expected, f"upstream source changed: {name}")
    for name, expected in execution["tool_sha256"].items():
        require(digest(SOURCE_EVIDENCE.parent.parent / name) == expected, f"reference executable source changed: {name}")
    require(digest(path, decompress=True) == REFERENCE_RAW_SHA256, "independent source trace changed")
    require(execution["traces"]["continuous-1e-12.csv.gz"]["raw_sha256"] == REFERENCE_RAW_SHA256,
            "reference trace is not bound to CPU execution")
    report_path = SOURCE_EVIDENCE / "continuous-1e-12.json"
    require(digest(report_path) == execution["artifact_sha256"][report_path.name], "CPU report changed")
    report = read_json(report_path)
    require(report["status"] == "pass" and report["cycles_requested"] == 20 and report["relative_tolerance"] == 1e-12,
            "wrong continuous source reference")
    require(report["source_clock_equivalence_to_original_step_sim"] is False, "source clock interpretation obscured")
    require(not any(report[key] for key in ("ABReflexOn", "CPReflexOn", "tiltTestOn")), "unsupported source mode")
    require(report["sample_dt_s"] == REFERENCE_DT, "wrong source sampling grid")
    opener = gzip.open if path.suffix == ".gz" else open
    states = []
    with opener(path, "rt", newline="") as file:
        rows = csv.reader(file)
        require(next(rows, None) == SOURCE_HEADER, "source column names or units changed")
        for index, row in enumerate(rows):
            values = numbers(row, 70, f"source row {index}")
            expected_time = min(index * REFERENCE_DT, 20 * PERIOD)
            require(values[0] == expected_time, "source sampling time changed")
            require(index == 0 or values[0] > (index - 1) * REFERENCE_DT, "extra source endpoint")
            require(min(values[22:43]) > 0, "source has nonpositive absolute volume")
            require(abs(sum(values[22:43]) - values[67]) < 1e-7, "source total does not match its coordinates")
            require(abs(values[67] - 5150.) < 1e-5, "tight source total blood volume drift exceeded")
            require(all(values[43 + i] >= 0 for i in ONE_WAY_FLOWS), "negative source one-way flow")
            states.append([value * 1e-6 for value in values[22:67]])
    require(len(states) == report["samples"] == 34287, "source 20-cycle coverage is incomplete")
    return states, {"raw_sha256": REFERENCE_RAW_SHA256, "stored_sha256": digest(path),
                    "execution_sha256": EXECUTION_SHA256, "source_lock_sha256": SOURCE_LOCK_SHA256,
                    "initial_manifest_sha256": INITIAL_MANIFEST_SHA256,
                    "relative_tolerance": 1e-12, "sample_dt_seconds": REFERENCE_DT, "samples": len(states),
                    "source_clock": "continuous_rational_period_6_over_7_seconds",
                    "original_discrete_SA_node_equivalence": False}


def parse_log(path: Path, dt: float, steps: int) -> dict:
    text = path.read_text()
    require("cvsim_source_run=failed" not in text, "native log contains a failed source run")
    summaries = [line for line in text.splitlines() if line.startswith("cvsim_source_run=")]
    require(len(summaries) == 1, "native log needs exactly one completed source summary")
    fields = {}
    for word in summaries[0].split():
        require("=" in word, "invalid native summary")
        key, value = word.split("=", 1)
        require(key not in fields, "duplicate native summary field")
        fields[key] = value
    expected = {"cvsim_source_run": "pass", "accepted_steps": str(steps), "environments": "2",
                "failed_steps": "0", "clock": "exact_binary_128_rational_period", "replay_pair": "bitwise",
                "snapshot_replay": "bitwise", "qualification": "source_variant_numerical_comparison",
                "biological_calibration": "unqualified"}
    require(all(fields.get(key) == value for key, value in expected.items()), "native execution/replay attestations differ")
    require(float(fields.get("dt_seconds", "nan")) == dt, "native log timestep differs")
    require(float(fields.get("duration_seconds", "nan")) == steps * dt, "native log duration differs")
    require("cvsim_device=Apple" in text and "Paravirtual" not in text, "physical Apple device attestation missing")
    return fields


def evaluate_refinement(runs: dict) -> dict:
    cohort = [runs[name] for name in REFINEMENT_RUNS]
    require(len({run["duration_seconds"] for run in cohort}) == 1, "refinement requires the same elapsed duration")
    require(cohort[0]["duration_seconds"] >= PERIOD, "refinement cohort is shorter than one beat")
    metrics = {}
    passed = True
    for metric in METRICS:
        errors = [run[metric] for run in cohort]
        require(all(math.isfinite(value) and value >= 0 for value in errors), "invalid refinement errors")
        # Exact zero cannot establish a measured order, and is not represented
        # by Infinity in JSON. Keep its absence of measurable order explicit.
        ratios = [a / b if a > 0 and b > 0 else None for a, b in zip(errors, errors[1:])]
        good = all(ratio is not None and ratio >= MINIMUM_REFINEMENT_RATIO for ratio in ratios)
        passed &= good
        metrics[metric] = {"errors": errors, "coarse_over_fine_ratios": ratios,
                           "observed_orders": [math.log2(ratio) if ratio else None for ratio in ratios], "passed": good}
    return {"status": "pass" if passed else "fail", "minimum_ratio": MINIMUM_REFINEMENT_RATIO,
            "equal_duration_seconds": cohort[0]["duration_seconds"], "metrics": metrics,
            "sampling": "all samples at each timestep; no phase shifts, time interpolation or omitted flow maxima"}


def audit_run(trace: Path, log: Path, *, dt: float, steps: int,
              reference: list[list[float]], coordinates: str) -> dict:
    dt = float32(dt)
    fields = parse_log(log, dt, steps)
    stride = round(dt / REFERENCE_DT)
    require(stride in {1, 2, 4} and stride * REFERENCE_DT == dt, "native timestep is not an exact source subset")
    require(steps * stride < len(reference) - 1, "native trace exceeds regular reference grid")
    translation = coordinate_translation(coordinates)
    initial = [float32(float32(value + shift) / STATE_SCALE) * STATE_SCALE
               for value, shift in zip(reference[0], translation)]
    initial_volume = sum(initial[:21])
    require(abs(initial_volume - TOTAL_VOLUME_M3) < 1e-9, "cooked initial absolute blood volume differs")
    max_error = [0.] * 45
    max_location = [{} for _ in range(45)]
    square_error = 0.
    max_drift = max_target = 0.
    min_volume = math.inf
    min_one_way = math.inf
    common_square = 0.
    common_count = 0
    common_max_volume = common_max_flow = 0.
    samples = 0
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rt", newline="") as file:
        rows = csv.reader(file)
        require(next(rows, None) == NATIVE_HEADER, "native columns must contain all 45 paired SI coordinates")
        for samples, row in enumerate(rows, 1):
            require(samples <= steps, "extra native rows")
            values = numbers(row, 91, f"native row {samples}")
            require(values[0] == samples * dt, f"native row {samples}: accepted clock differs")
            actual, supplied = values[1::2], values[2::2]
            expected = [value + shift for value, shift in zip(reference[samples * stride], translation)]
            # Compare to the independently pinned raw C output, never the other
            # half of the probe CSV. This catches relabelled coordinates too.
            require(supplied == expected, f"native row {samples}: paired reference columns differ from independent source")
            require(min(actual[:21]) > 0, f"native row {samples}: nonpositive absolute compartment volume")
            require(all(actual[21 + index] >= 0 for index in ONE_WAY_FLOWS),
                    f"native row {samples}: negative one-way valve flow")
            min_volume = min(min_volume, min(actual[:21]))
            min_one_way = min(min_one_way, *(actual[21 + i] for i in ONE_WAY_FLOWS))
            current_volume = sum(actual[:21])
            max_drift = max(max_drift, abs(current_volume - initial_volume) / initial_volume)
            max_target = max(max_target, abs(current_volume - TOTAL_VOLUME_M3))
            require(max_drift < CONSERVATION_LIMIT, "native absolute blood conservation exceeds numerical gate")
            errors = [abs(a - b) for a, b in zip(actual, expected)]
            for index, error in enumerate(errors):
                if error > max_error[index]:
                    max_error[index] = error
                    max_location[index] = {"step": samples, "time_seconds": values[0],
                                           "native": actual[index], "source": expected[index]}
                square_error += (error / STATE_SCALE) ** 2
            if samples * stride % 4 == 0:
                common_count += 45
                common_square += sum((error / STATE_SCALE) ** 2 for error in errors)
                common_max_volume = max(common_max_volume, max(errors[:21]))
                common_max_flow = max(common_max_flow, max(errors[21:]))
    require(samples == steps, "native row count is incomplete")
    metrics = dict(zip(METRICS, [max(max_error[:21]), max(max_error[21:]), math.sqrt(square_error / (steps * 45))]))
    metrics["relative_blood_volume_invariant_error"] = max_drift
    for key, value in metrics.items():
        reported = float(fields.get(key, "nan"))
        require(math.isfinite(reported) and math.isclose(reported, value, rel_tol=1e-10, abs_tol=1e-15),
                f"native summary does not match independently recomputed {key}")
    return {"status": "pass", "passed": True, "errors": [],
            "checks": {"complete_45_coordinate_pairs": True, "exact_source_pairs": True, "exact_cooked_times": True,
                       "positive_absolute_volumes": True, "nonnegative_one_way_flows": True,
                       "blood_conservation": True, "summary_metrics_recomputed": True},
            "steps": steps, "dt_seconds": dt, "duration_seconds": steps * dt,
            "cycles": steps * dt / PERIOD, "volume_coordinates": coordinates, **metrics,
            "initial_cooked_blood_volume_mL": initial_volume * 1e6,
            "max_target_volume_difference_mL": max_target * 1e6,
            "minimum_compartment_volume_mL": min_volume * 1e6, "minimum_one_way_flow_mL_per_s": min_one_way * 1e6,
            "per_coordinate_max_error": max_error, "per_coordinate_max_location": max_location,
            "common_2ms_sample_metrics": {"max_volume_error_m3": common_max_volume,
                                           "max_flow_error_m3_per_s": common_max_flow,
                                           "scaled_state_rms_error": math.sqrt(common_square / common_count)},
            "execution_attestations": {key: fields[key] for key in ("environments", "failed_steps", "clock", "replay_pair", "snapshot_replay")},
            "trace_sha256": digest(trace), "trace_raw_sha256": digest(trace, decompress=True), "log_sha256": digest(log)}


def audit(folder: Path, reference_path: Path, coordinates: str, *, require_aligned: bool = True) -> dict:
    reference, provenance = load_reference(reference_path)
    runs = {}
    for name, (dt, steps) in RUNS.items():
        trace = folder / (name + ".csv")
        if not trace.exists():
            trace = folder / (name + ".csv.gz")
        runs[name] = audit_run(trace, folder / (name + ".log"), dt=dt, steps=steps,
                              reference=reference, coordinates=coordinates)
    require(runs["ten-cycles-2ms"]["duration_seconds"] >= 10 * PERIOD, "long cohort is shorter than ten cycles")
    if require_aligned:
        require(coordinates == "upstream_equation", "paired coordinate qualification requires upstream primary cohort")
        trace = folder / "heldt-cycle-1ms.csv"
        if not trace.exists():
            trace = folder / "heldt-cycle-1ms.csv.gz"
        runs["heldt-cycle-1ms"] = audit_run(trace, folder / "heldt-cycle-1ms.log", dt=.001, steps=858,
                                           reference=reference, coordinates="heldt_table_aligned")
    refinement = evaluate_refinement(runs)
    return {"schema": "NumiHuman.CVSim21-native-comparison.v1", "status": refinement["status"],
            "passed": refinement["status"] == "pass", "errors": [] if refinement["status"] == "pass" else ["first-order refinement failed"],
            "qualification": "bounded_continuous_source_equation_comparison_and_timestep_refinement",
            "biological_calibration": "unqualified", "absolute_accuracy_qualification": "not_established",
            "python_physical_stepping": False, "analyzer_sha256": digest(Path(__file__)),
            "source": provenance, "volume_coordinates": coordinates,
            "volume_translation_m3": coordinate_translation(coordinates)[:21], "compartment_order": list(VOLUME_LABELS),
            "coordinate_units": {"0_to_20": "m3", "21_to_44": "m3/s"},
            "thresholds": {"relative_blood_volume_drift_exclusive": CONSERVATION_LIMIT,
                            "minimum_first_order_error_ratio": MINIMUM_REFINEMENT_RATIO,
                            "positive_absolute_volume": True, "nonnegative_one_way_flow": True,
                            "accuracy_limit_basis": "No absolute accuracy envelope is inferred from execution or refinement."},
            "runs": runs, "refinement": refinement,
            "limitations": ["Native environment and snapshot replay are log attestations; CSV independently establishes numerical comparison only.",
                            "The reference prescribes rational phase; it is not the original discrete SA-node trajectory.",
                            "No time shift or interpolation repairs flow maxima that fail timestep refinement."]}


def analyze_cohort(cohort_dir: Path, reference_path: Path, *, require_aligned: bool = True) -> dict:
    """Stable receipt interface. Returns deterministic evidence without physical stepping."""
    try:
        return audit(Path(cohort_dir), Path(reference_path), "upstream_equation", require_aligned=require_aligned)
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        return {"schema": "NumiHuman.CVSim21-native-comparison.v1", "status": "fail", "passed": False,
                "errors": [str(error)], "biological_calibration": "unqualified", "python_physical_stepping": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--reference", type=Path, default=EVIDENCE / "continuous-1e-12.csv.gz")
    parser.add_argument("--volume-coordinates", choices=("upstream_equation", "heldt_table_aligned"), default="upstream_equation")
    parser.add_argument("--skip-aligned", action="store_true", help="omit the separately authored Heldt coordinate cohort")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.folder, args.reference, args.volume_coordinates, require_aligned=not args.skip_aligned)
    except (EvidenceError, OSError, KeyError, ValueError) as error:
        result = {"schema": "NumiHuman.CVSim21-native-comparison.v1", "status": "fail", "passed": False, "errors": [str(error)],
                  "biological_calibration": "unqualified", "python_physical_stepping": False}
    output = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
