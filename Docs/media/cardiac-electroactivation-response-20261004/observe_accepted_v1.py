#!/usr/bin/env python3
"""Read-only accepted-state myocardium activation and chamber-response observer."""
from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_myocardial_mask(package_path: Path, expected_sha256: str,
                         expected_nodes: int, expected_tetrahedra: int,
                         material_id: int) -> tuple[bytearray, str]:
    require(sha256_file(package_path) == expected_sha256,
            "pinned ventricular package identity")
    mask = bytearray(expected_nodes)
    found = False
    with package_path.open("rb") as stream:
        header = struct.unpack("<16s4I3Q", stream.read(56))
        require(header[:4] == (b"NUMIMATTERPKG\0\0\0", 18,
                               0x01020304, 36),
                "pinned native package header")
        for _ in range(header[4]):
            section, item_size, count, byte_count, _checksum = struct.unpack(
                "<IIQQQ", stream.read(32),
            )
            if section != 19:
                stream.seek(byte_count, 1)
                continue
            require(item_size == 96 and count == expected_tetrahedra
                    and byte_count == item_size * count,
                    "pinned cooked tetrahedron table shape")
            raw = stream.read(byte_count)
            require(len(raw) == byte_count,
                    "complete cooked tetrahedron table")
            for offset in range(0, byte_count, item_size):
                if struct.unpack_from("<I", raw, offset + 64)[0] != material_id:
                    continue
                for node in struct.unpack_from("<4I", raw, offset):
                    require(node < expected_nodes,
                            "myocardial tetrahedron node index")
                    mask[node] = 1
            found = True
            break
    require(found and sum(mask) == 218077,
            "cooked myocardial node membership")
    return mask, hashlib.sha256(mask).hexdigest()


def latest_complete_row(path: Path, step: int) -> dict[str, str] | None:
    result = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("accepted_step") != str(step):
                continue
            if any(value is None or value == "" for value in row.values()):
                continue
            result = row
    return result


def read_accepted_fields(
    snapshot_path: Path, driver_path: Path, trajectory_path: Path,
    expected_fingerprints: list[int], node_count: int,
) -> tuple[int, dict[str, float], bytes, dict[str, Any]] | None:
    before = driver_path.read_text()
    driver = before.split()
    require(len(driver) == 10
            and driver[0] == "NUMI_VENTRICULAR_DRIVER_V2",
            "accepted driver format")
    try:
        with snapshot_path.open("rb") as stream:
            header = struct.unpack("<8s4I4Q", stream.read(56))
            require(header[:5] == (b"NMSNAP01", 13, 0x01020304, 36, 0),
                    "accepted snapshot header")
            if header[6] != int(driver[1]):
                return None
            require(list(header[7:9]) == expected_fingerprints,
                    "accepted source-program fingerprints")
            stream.seek(160)
            fields = None
            sections = []
            for name in ("proxy", "particles", "nodes", "fields"):
                item_size, count = struct.unpack("<IQ", stream.read(12))
                if name == "fields":
                    require(item_size == 48 and count == node_count,
                            "accepted native field shape")
                    fields = stream.read(item_size * count)
                    require(len(fields) == item_size * count,
                            "complete accepted native fields")
                else:
                    stream.seek(item_size * count, 1)
                sections.append({"name": name, "item_size": item_size,
                                 "count": count})
            stream.seek(0)
            checkpoint_digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                checkpoint_digest.update(block)
    except (OSError, struct.error):
        return None
    after = driver_path.read_text()
    if before != after:
        return None
    step = int(driver[2])
    row_text = latest_complete_row(trajectory_path, step)
    if row_text is None:
        return None
    row = {key: float(value) for key, value in row_text.items()}
    require(all(math.isfinite(value) for value in row.values()),
            "accepted trajectory row is finite")
    require(row["accepted_step"] == step,
            "trajectory row and accepted driver step agree")

    global_v = global_a = 0
    max_v = max_a = -math.inf
    myocard_v = myocard_a = 0
    myocard_max_a = myocard_mean_a = 0.0
    fields_sha = hashlib.sha256(fields).hexdigest()
    # Return the raw fields; the caller computes source-myocardium arrivals.
    require(len(fields) == node_count * 48, "accepted field byte count")
    for index, values in enumerate(struct.iter_unpack("<12f", fields)):
        require(all(math.isfinite(value) for value in values),
                "accepted native field values are finite")
        voltage, activation = values[3], values[4]
        max_v = max(max_v, voltage)
        max_a = max(max_a, activation)
        global_v += voltage > 0.5
        global_a += activation > 0.01
    require(max_v == row["max_electric_potential"]
            and max_a == row["max_activation"]
            and global_v == row["voltage_nodes_over_0_5"]
            and global_a == row["activation_nodes_over_0_01"],
            "accepted native fields match progress telemetry")
    del myocard_v, myocard_a, myocard_max_a, myocard_mean_a
    return step, row, fields, {
        "native_archive_hash": header[6],
        "program_fingerprints": list(header[7:9]),
        "driver_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "checkpoint_sha256": checkpoint_digest.hexdigest(),
        "field_payload_sha256": fields_sha,
        "sections": sections,
    }


def encode_arrival_state(
    out_path: Path, node_count: int, last_step: int,
    last_time_ms: float, arrays: dict[str, array], statuses: dict[str, bytearray],
) -> str:
    metadata = json.dumps({
        "schema": "numi.human.cardiac-electroactivation-arrival-bounds.v1",
        "node_count": node_count,
        "last_observed_step": last_step,
        "last_observed_time_ms": last_time_ms,
        "byte_order": "little",
        "double_arrays": ["voltage_lower_ms", "voltage_upper_ms",
                           "activation_lower_ms", "activation_upper_ms"],
        "status_arrays": ["voltage_status", "activation_status"],
        "status_values": {"0": "not_observed_above_threshold",
                          "1": "interval_censored_after_observer_start",
                          "2": "already_above_threshold_at_observer_start"},
    }, sort_keys=True, separators=(",", ":")).encode()
    body = bytearray(struct.pack("<I", len(metadata)))
    body.extend(metadata)
    for name in ("voltage_lower_ms", "voltage_upper_ms",
                 "activation_lower_ms", "activation_upper_ms"):
        body.extend(arrays[name].tobytes())
    for name in ("voltage_status", "activation_status"):
        body.extend(statuses[name])
    payload = gzip.compress(bytes(body), compresslevel=6, mtime=0)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(out_path)
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_is_live(pid: int, executable_name: str) -> bool:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip().endswith(executable_name)


def observe(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text())
    native = plan["native_run"]
    instrument = plan["instrument"]
    paths = {name: Path(native[name]) for name in
             ("package", "accepted_checkpoint", "accepted_driver", "trajectory_csv")}
    output_dir = Path(instrument["state_directory"])
    output_dir.mkdir(parents=True, exist_ok=False)
    require(sys.byteorder == "little", "observer requires little-endian host")
    require(sha256_file(Path(native["executable"]))
            == native["executable_sha256"], "pinned native executable")
    require(sha256_file(Path(native["metallib"]))
            == native["metallib_sha256"], "pinned native metallib")
    mask, mask_sha = load_myocardial_mask(
        paths["package"], native["package_sha256"],
        plan["baseline"]["all_node_count"], 1142114,
        instrument["cooked_myocardial_material_id"],
    )
    write_json_atomic(output_dir / "observer-start.json", {
        "schema": "numi.human.cardiac-electroactivation-observer-start.v1",
        "started_unix": time.time(),
        "native_pid": native["pid"],
        "package_sha256": native["package_sha256"],
        "myocardial_mask_sha256": mask_sha,
        "source_myocardial_node_count": sum(mask),
        "candidate_adopted": False,
        "solver_modified": False,
        "gpu_submissions": 0,
    })

    node_count = len(mask)
    nan_array = array("d", [-1.0]) * node_count
    arrays = {
        name: nan_array[:]
        for name in ("voltage_lower_ms", "voltage_upper_ms",
                     "activation_lower_ms", "activation_upper_ms")
    }
    statuses = {"voltage_status": bytearray(node_count),
                "activation_status": bytearray(node_count)}
    observations_path = output_dir / "observations.jsonl"
    previous_step: int | None = None
    previous_time_ms: float | None = None
    last_sample: dict[str, Any] | None = None
    reason = None

    while True:
        state = read_accepted_fields(
            paths["accepted_checkpoint"], paths["accepted_driver"],
            paths["trajectory_csv"], native["expected_program_fingerprints"],
            node_count,
        )
        if state is not None:
            step, row, raw_fields, identity = state
            if previous_step is None or step > previous_step:
                sample_time = row["time_ms"]
                voltage_count = activation_count = 0
                voltage_new = activation_new = 0
                voltage_first_censored = activation_first_censored = 0
                threshold_v = instrument["electrical_threshold"]
                threshold_a = instrument["activation_threshold"]
                for index, values in enumerate(struct.iter_unpack("<12f", raw_fields)):
                    if not mask[index]:
                        continue
                    voltage, activation = values[3], values[4]
                    if voltage > threshold_v:
                        voltage_count += 1
                        if statuses["voltage_status"][index] == 0:
                            if previous_step is None:
                                arrays["voltage_upper_ms"][index] = sample_time
                                statuses["voltage_status"][index] = 2
                                voltage_first_censored += 1
                            else:
                                arrays["voltage_lower_ms"][index] = previous_time_ms
                                arrays["voltage_upper_ms"][index] = sample_time
                                statuses["voltage_status"][index] = 1
                            voltage_new += 1
                    if activation > threshold_a:
                        activation_count += 1
                        if statuses["activation_status"][index] == 0:
                            if previous_step is None:
                                arrays["activation_upper_ms"][index] = sample_time
                                statuses["activation_status"][index] = 2
                                activation_first_censored += 1
                            else:
                                arrays["activation_lower_ms"][index] = previous_time_ms
                                arrays["activation_upper_ms"][index] = sample_time
                                statuses["activation_status"][index] = 1
                            activation_new += 1
                driver_after_hash = hashlib.sha256(
                    paths["accepted_driver"].read_bytes(),
                ).hexdigest()
                require(driver_after_hash == identity["driver_sha256"],
                        "accepted checkpoint stayed at one driver generation")
                arrival_sha = encode_arrival_state(
                    output_dir / "arrival-bounds.bin.gz", node_count,
                    step, sample_time, arrays, statuses,
                )
                event = {
                    "schema": "numi.human.cardiac-electroactivation-observation.v1",
                    "observed_unix": time.time(),
                    "accepted_step": step,
                    "time_ms": sample_time,
                    "steps_after_preregistered_baseline": (
                        step - plan["baseline"]["accepted_step"]),
                    "previous_observed_step": previous_step,
                    "observed_gap_steps": None if previous_step is None
                    else step - previous_step,
                    "source_myocardial_node_count": sum(mask),
                    "myocardial_voltage_nodes_over_threshold": voltage_count,
                    "myocardial_activation_nodes_over_threshold": activation_count,
                    "new_voltage_arrivals": voltage_new,
                    "new_activation_arrivals": activation_new,
                    "left_censored_voltage_nodes_at_first_observation": voltage_first_censored,
                    "left_censored_activation_nodes_at_first_observation": activation_first_censored,
                    "lv_pressure_pa": row["lv_pressure_pa"],
                    "rv_pressure_pa": row["rv_pressure_pa"],
                    "aortic_flow_ml_s": row["aortic_ml_s"],
                    "pulmonary_flow_ml_s": row["pulmonary_ml_s"],
                    "mitral_flow_ml_s": row["mitral_ml_s"],
                    "tricuspid_flow_ml_s": row["tricuspid_ml_s"],
                    "total_blood_ml": row["total_blood_ml"],
                    "max_activation": row["max_activation"],
                    "max_electric_potential": row["max_electric_potential"],
                    "field_payload_sha256": identity["field_payload_sha256"],
                    "driver_sha256": identity["driver_sha256"],
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                    "trajectory_row_sha256": hashlib.sha256(json.dumps(
                        row, sort_keys=True, separators=(",", ":"),
                    ).encode()).hexdigest(),
                    "myocardial_mask_sha256": mask_sha,
                    "arrival_bounds_sha256": arrival_sha,
                    "native_program_fingerprints": identity["program_fingerprints"],
                    "accepted_fields_match_trajectory": True,
                    "solver_modified": False,
                    "gpu_submissions": 0,
                    "heartbeat_qualified": False,
                }
                with observations_path.open("a") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    stream.flush()
                previous_step = step
                previous_time_ms = sample_time
                last_sample = event
                write_json_atomic(output_dir / "status.json", event)
                print(json.dumps({k: event[k] for k in (
                    "accepted_step", "time_ms", "myocardial_voltage_nodes_over_threshold",
                    "myocardial_activation_nodes_over_threshold", "new_voltage_arrivals",
                    "new_activation_arrivals", "lv_pressure_pa", "rv_pressure_pa",
                    "aortic_flow_ml_s", "pulmonary_flow_ml_s",
                )}), flush=True)

        if not process_is_live(native["pid"], Path(native["executable"]).name):
            reason = "pinned native process ended; last checkpoint sampled"
            break
        time.sleep(instrument["poll_interval_s"])

    summary = {
        "schema": "numi.human.cardiac-electroactivation-observer-exit.v1",
        "reason": reason,
        "native_pid": native["pid"],
        "observed_checkpoint_count": 0 if previous_step is None else None,
        "last_observed_step": previous_step,
        "last_observed_time_ms": previous_time_ms,
        "last_observation": last_sample,
        "observations_sha256": sha256_file(observations_path),
        "myocardial_mask_sha256": mask_sha,
        "candidate_adopted": False,
        "solver_modified": False,
        "heartbeat_qualified": False,
    }
    count = sum(1 for _ in observations_path.open())
    summary["observed_checkpoint_count"] = count
    write_json_atomic(output_dir / "exit.json", summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    observe(args.plan.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
