#!/usr/bin/env python3
"""Read accepted native cardiac state and reconstruct outlet pressure gradients."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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


def fnv64(data: bytes) -> int:
    # The package owner uses this historical seed; snapshot archives have a
    # different seed, so keep the two checksum domains distinct.
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value or 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def package_tables(plan: dict[str, Any]) -> dict[str, Any]:
    native = plan["native_run"]
    path = Path(native["package"])
    require(sha256_file(path) == native["package_sha256"],
            "pinned native package SHA-256")
    data = path.read_bytes()
    header = struct.unpack_from("<16s4I3Q", data)
    require(header[:5] == (b"NUMIMATTERPKG\0\0\0", 18,
                           0x01020304, 36, 64),
            "pinned native package header")
    offset = 56
    sections: dict[int, tuple[int, int, bytes]] = {}
    checksummed_sections = {43, 46, 47, 51, 58, 59}
    for _ in range(header[4]):
        section, item_size, count, byte_count, checksum = struct.unpack_from(
            "<IIQQQ", data, offset,
        )
        offset += 32
        payload = data[offset:offset + byte_count]
        require(len(payload) == byte_count,
                f"package section {section} size")
        if section in checksummed_sections:
            require(fnv64(payload) == checksum,
                    f"package section {section} checksum")
        sections[section] = (item_size, count, payload)
        offset += byte_count
    require(offset == len(data), "complete native package section table")

    required = {43: 80, 46: 96, 47: 48, 51: 16, 58: 1, 59: 96}
    for section, item_size in required.items():
        require(sections[section][0] == item_size,
                f"vascular package section {section} item size")
    layout = struct.unpack("<20I", sections[43][2])
    compartment_count, connection_count = layout[0], layout[1]
    unknown_count = layout[6]
    require(compartment_count == sections[46][1]
            and connection_count == sections[47][1]
            and unknown_count == sections[51][1],
            "vascular layout and section counts")
    unknowns = [struct.unpack_from("<4f", sections[51][2], i * 16)
                for i in range(unknown_count)]
    names_blob = sections[58][2]
    compartments = []
    for index in range(compartment_count):
        base = index * 96
        identity = struct.unpack_from("<4I", sections[46][2], base)
        compliance = struct.unpack_from("<4f", sections[46][2], base + 16)
        name_end = names_blob.find(bytes([0]), identity[1])
        require(name_end >= identity[1], "vascular compartment name offset")
        name = names_blob[identity[1]:name_end].decode("utf-8")
        compartments.append({
            "index": index,
            "stable_id": identity[0],
            "name": name,
            "pressure_law": identity[3],
            "compliance": compliance,
            "volume_scale": unknowns[index][1],
        })
    connections = []
    for index in range(connection_count):
        identity = struct.unpack_from("<4I", sections[47][2], index * 48)
        require(identity[1] < compartment_count and identity[2] < compartment_count,
                "vascular connection endpoint indices")
        connections.append({
            "index": index,
            "stable_id": identity[0],
            "from_index": identity[1],
            "to_index": identity[2],
            "from_name": compartments[identity[1]]["name"],
            "to_name": compartments[identity[2]]["name"],
        })
    cavities = [struct.unpack_from("<4I", sections[59][2], i * 96)
                for i in range(sections[59][1])]
    for outlet in plan["instrument"]["outlets"]:
        matches = [edge for edge in connections
                   if edge["stable_id"] == outlet["connection_id"]]
        require(len(matches) == 1, "unique source outlet connection")
        outlet["connection"] = matches[0]
        require(matches[0]["from_name"] == outlet["from_name"]
                and matches[0]["to_name"] == outlet["to_name"],
                "source outlet anatomical connection binding")
    return {
        "layout": layout,
        "unknowns": unknowns,
        "compartments": compartments,
        "cavities": cavities,
        "connections": connections,
        "sections": sections,
    }


def latest_row(path: Path, step: int) -> dict[str, float] | None:
    result = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("accepted_step") != str(step):
                continue
            if any(value is None or value == "" for value in row.values()):
                continue
            result = {key: float(value) for key, value in row.items()}
    return result


def read_state(plan: dict[str, Any], tables: dict[str, Any],
               hash_checkpoint: bool) -> dict[str, Any] | None:
    native = plan["native_run"]
    checkpoint = Path(native["accepted_checkpoint"])
    driver_path = Path(native["accepted_driver"])
    driver_before = driver_path.read_text()
    tokens = driver_before.split()
    require(len(tokens) == 10 and tokens[0] == "NUMI_VENTRICULAR_DRIVER_V2",
            "native accepted-driver format")
    try:
        with checkpoint.open("rb") as stream:
            stat_before = os.fstat(stream.fileno())
            header_bytes = stream.read(88)
            if len(header_bytes) != 88:
                return None
            header = struct.unpack("<8s4I8Q", header_bytes)
            require(header[:5] == (b"NMSNAP01", 13, 0x01020304, 36, 0),
                    "native accepted-checkpoint header")
            if header[6] != int(tokens[1]):
                return None
            require([header[7], header[8]]
                    == plan["native_run"]["expected_program_fingerprints"],
                    "native accepted program fingerprints")
            require(header[5] == stat_before.st_size - 88,
                    "native checkpoint payload length")
            # Format 13 places its ordered snapshot vectors at byte 160.
            stream.seek(160)
            vectors = []
            vascular = clock = None
            for index in range(31):
                record = stream.read(12)
                if len(record) != 12:
                    return None
                item_size, count = struct.unpack("<IQ", record)
                byte_count = item_size * count
                if index in (29, 30):
                    payload = stream.read(byte_count)
                    if len(payload) != byte_count:
                        return None
                    if index == 29:
                        vascular = payload
                    else:
                        clock = payload
                else:
                    stream.seek(byte_count, 1)
                vectors.append((item_size, count))
            if (vectors[2] != (64, plan["instrument"]["node_count"])
                    or vectors[3] != (48, plan["instrument"]["node_count"])):
                raise RuntimeError("native node and field table dimensions")
            if (vectors[29] != (16, tables["layout"][6])
                    or vectors[30] != (16, 1)
                    or vascular is None or clock is None):
                raise RuntimeError("native vascular-state and clock dimensions")
            normalized = [struct.unpack_from("<4f", vascular, i * 16)
                          for i in range(vectors[29][1])]
            accepted_clock = struct.unpack("<QQ", clock)
            digest = None
            if hash_checkpoint:
                stream.seek(0)
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
            stat_after = os.fstat(stream.fileno())
    except (OSError, struct.error):
        return None
    driver_after = driver_path.read_text()
    if driver_before != driver_after:
        return None
    require((stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
             stat_before.st_mtime_ns)
            == (stat_after.st_dev, stat_after.st_ino, stat_after.st_size,
                stat_after.st_mtime_ns),
            "accepted checkpoint stayed at one filesystem generation")
    step = int(tokens[2])
    row = latest_row(Path(native["trajectory_csv"]), step)
    if row is None:
        return None
    return {
        "step": step,
        "driver_sha256": hashlib.sha256(driver_before.encode()).hexdigest(),
        "archive_content_hash": int(tokens[1]),
        "checkpoint_sha256": digest,
        "program_fingerprints": [header[7], header[8]],
        "vascular_state": normalized,
        "vascular_clock": accepted_clock,
        "row": row,
    }


def source_pressures(state: dict[str, Any], tables: dict[str, Any],
                     plan: dict[str, Any]) -> dict[int, float]:
    values = state["vascular_state"]
    unknowns = tables["unknowns"]
    compartments = tables["compartments"]
    pressures: dict[int, float] = {}
    cavity_rows = {cavity[1]: cavity[3] for cavity in tables["cavities"]}
    for index in plan["instrument"]["pressure_compartment_indices"]:
        node = compartments[index]
        if node["pressure_law"] == 5:
            require(index in cavity_rows, "deforming source chamber cavity row")
            row = cavity_rows[index]
            pressures[index] = values[row][0] * unknowns[row][1]
            continue
        require(node["pressure_law"] == 0,
                "outlet pressure reconstruction currently admits the linear source law")
        compliance = node["compliance"]
        volume = f32(values[index][0] * unknowns[index][1])
        elastance = f32(1.0 / compliance[2])
        pressure = f32(
            f32(compliance[3] + compliance[1])
            + f32(f32(volume - compliance[0]) * elastance)
        )
        require(math.isfinite(pressure), "finite source compartment pressure")
        pressures[index] = pressure
    return pressures


def event_for(state: dict[str, Any], tables: dict[str, Any],
              plan: dict[str, Any]) -> dict[str, Any]:
    row = state["row"]
    pressures = source_pressures(state, tables, plan)
    lv_index = plan["instrument"]["lv_compartment_index"]
    rv_index = plan["instrument"]["rv_compartment_index"]
    pressure_checks = {
        "lv_pressure_pa": pressures[lv_index],
        "rv_pressure_pa": pressures[rv_index],
    }
    for key, value in pressure_checks.items():
        require(abs(value - row[key]) <= plan["instrument"]["csv_tolerance_pa"],
                f"accepted cavity pressure matches trajectory: {key}")
    result: dict[str, Any] = {
        "schema": "numi.human.cardiac-outlet-gradient-observation.v1",
        "observed_unix": time.time(),
        "accepted_step": state["step"],
        "time_ms": row["time_ms"],
        "steps_after_preregistered_baseline":
            state["step"] - plan["baseline"]["accepted_step"],
        "checkpoint_sha256": state["checkpoint_sha256"],
        "archive_content_hash": state["archive_content_hash"],
        "driver_sha256": state["driver_sha256"],
        "program_fingerprints": state["program_fingerprints"],
        "vascular_clock_ticks": state["vascular_clock"],
        "source_package_sha256": plan["native_run"]["package_sha256"],
        "source_pressure_formula_sha256": plan["instrument"]["source_formula_sha256"],
        "accepted_lv_pressure_matches_trajectory": True,
        "accepted_rv_pressure_matches_trajectory": True,
        "lv_pressure_pa": pressure_checks["lv_pressure_pa"],
        "rv_pressure_pa": pressure_checks["rv_pressure_pa"],
        "heartbeat_qualified": False,
        "solver_modified": False,
        "gpu_submissions": 0,
    }
    for outlet in plan["instrument"]["outlets"]:
        edge = outlet["connection"]
        flow_row = plan["instrument"]["flow_row_base"] + edge["index"]
        q_ml_s = (state["vascular_state"][flow_row][0]
                  * tables["unknowns"][flow_row][1] * 1.0e6)
        csv_flow = row[outlet["trajectory_flow_column"]]
        require(abs(q_ml_s - csv_flow)
                <= plan["instrument"]["csv_tolerance_flow_ml_s"],
                "accepted native outlet flow matches trajectory")
        dp = pressures[edge["from_index"]] - pressures[edge["to_index"]]
        result[outlet["key"]] = {
            "connection_id": edge["stable_id"],
            "from": edge["from_name"],
            "to": edge["to_name"],
            "from_pressure_pa": pressures[edge["from_index"]],
            "to_pressure_pa": pressures[edge["to_index"]],
            "gradient_from_minus_to_pa": dp,
            "flow_ml_s": q_ml_s,
            "trajectory_flow_ml_s": csv_flow,
            "positive_driving_gradient": dp > 0.0,
            "forward_ejection": q_ml_s > 0.0,
        }
    return result


def process_is_live(pid: int, executable_name: str) -> bool:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip().endswith(executable_name)


def write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true",
                        help="read one stable accepted checkpoint and exit")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    require(sys.byteorder == "little", "observer requires little-endian host")
    native = plan["native_run"]
    require(sha256_file(Path(native["executable"]))
            == native["executable_sha256"], "pinned native executable")
    require(sha256_file(Path(native["metallib"]))
            == native["metallib_sha256"], "pinned native metallib")
    require(sha256_file(Path(plan["instrument"]["source_formula_path"]))
            == plan["instrument"]["source_formula_sha256"],
            "pinned source pressure-law implementation")
    tables = package_tables(plan)
    state = read_state(plan, tables, hash_checkpoint=True)
    require(state is not None, "stable accepted state and trajectory row")
    require(state["step"] >= plan["baseline"]["accepted_step"],
            "accepted run has not rolled behind the preregistered baseline")
    result = event_for(state, tables, plan)
    verify_baseline_if_current(state, result, plan)
    if args.once:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    output = Path(plan["instrument"]["state_directory"])
    output.mkdir(parents=True, exist_ok=False)
    start = {
        "schema": "numi.human.cardiac-outlet-gradient-observer-start.v1",
        "started_unix": time.time(),
        "plan_sha256": sha256_file(plan_path),
        "observer_source_sha256": sha256_file(Path(__file__)),
        "native_pid": native["pid"],
        "package_sha256": native["package_sha256"],
        "source_formula_sha256": plan["instrument"]["source_formula_sha256"],
        "baseline_step": plan["baseline"]["accepted_step"],
        "candidate_adopted": False,
        "solver_modified": False,
        "gpu_submissions": 0,
    }
    write_json_atomic(output / "observer-start.json", start)
    observations = output / "observations.jsonl"
    previous_step = plan["baseline"]["accepted_step"]
    pending_state: dict[str, Any] | None = state
    first_positive: dict[str, int | None] = {"aortic": None, "pulmonary": None}
    latest: dict[str, Any] | None = None
    reason = None
    while True:
        if pending_state is None:
            state = read_state(plan, tables, hash_checkpoint=False)
        else:
            state = pending_state
            pending_state = None
        if state is not None and state["step"] > previous_step:
            if state["checkpoint_sha256"] is None:
                state["checkpoint_sha256"] = read_state_sha_if_stable(
                    plan, state["archive_content_hash"], state["step"],
                )
                if state["checkpoint_sha256"] is None:
                    time.sleep(plan["instrument"]["poll_interval_s"])
                    continue
            latest = event_for(state, tables, plan)
            latest["previous_observed_step"] = previous_step
            latest["observed_gap_steps"] = state["step"] - previous_step
            for name in first_positive:
                if (first_positive[name] is None
                        and latest[name]["positive_driving_gradient"]):
                    first_positive[name] = state["step"]
            with observations.open("a") as stream:
                stream.write(json.dumps(latest, sort_keys=True) + "\n")
                stream.flush()
            write_json_atomic(output / "status.json", latest)
            print(json.dumps({
                "accepted_step": latest["accepted_step"],
                "time_ms": latest["time_ms"],
                "aortic_gradient_pa": latest["aortic"]["gradient_from_minus_to_pa"],
                "aortic_flow_ml_s": latest["aortic"]["flow_ml_s"],
                "pulmonary_gradient_pa": latest["pulmonary"]["gradient_from_minus_to_pa"],
                "pulmonary_flow_ml_s": latest["pulmonary"]["flow_ml_s"],
            }), flush=True)
            previous_step = state["step"]
        if not process_is_live(native["pid"], Path(native["executable"]).name):
            reason = "pinned native process ended; last stable checkpoint sampled"
            break
        time.sleep(plan["instrument"]["poll_interval_s"])
    summary = {
        "schema": "numi.human.cardiac-outlet-gradient-observer-exit.v1",
        "reason": reason,
        "native_pid": native["pid"],
        "observed_checkpoint_count": sum(1 for _ in observations.open())
        if observations.exists() else 0,
        "last_observed_step": previous_step,
        "last_observation": latest,
        "first_positive_gradient_step": first_positive,
        "observations_sha256": sha256_file(observations)
        if observations.exists() else None,
        "solver_modified": False,
        "heartbeat_qualified": False,
    }
    write_json_atomic(output / "exit.json", summary)
    return 0


def verify_baseline_if_current(state: dict[str, Any], event: dict[str, Any],
                               plan: dict[str, Any]) -> None:
    baseline = plan["baseline"]
    if state["step"] != baseline["accepted_step"]:
        return
    require(state["checkpoint_sha256"] == baseline["checkpoint_sha256"],
            "pinned baseline checkpoint SHA-256")
    require(state["archive_content_hash"] == baseline["archive_content_hash"],
            "pinned baseline archive identity")
    for key, expected in baseline["measurements"].items():
        if key in ("aortic_gradient_pa", "pulmonary_gradient_pa"):
            name = "aortic" if key.startswith("aortic") else "pulmonary"
            actual = event[name]["gradient_from_minus_to_pa"]
        elif key in ("aortic_flow_ml_s", "pulmonary_flow_ml_s"):
            name = "aortic" if key.startswith("aortic") else "pulmonary"
            actual = event[name]["flow_ml_s"]
        elif key == "aortic_pressure_pa":
            actual = event["aortic"]["to_pressure_pa"]
        elif key == "pulmonary_pressure_pa":
            actual = event["pulmonary"]["to_pressure_pa"]
        else:
            actual = event[key]
        require(abs(actual - expected) <= baseline["measurement_tolerance"],
                f"pinned baseline measurement: {key}")


def read_state_sha_if_stable(plan: dict[str, Any], archive_hash: int,
                             step: int) -> str | None:
    native = plan["native_run"]
    checkpoint = Path(native["accepted_checkpoint"])
    driver = Path(native["accepted_driver"])
    before = driver.read_text()
    tokens = before.split()
    if len(tokens) != 10 or int(tokens[1]) != archive_hash or int(tokens[2]) != step:
        return None
    digest = sha256_file(checkpoint)
    after = driver.read_text()
    if before != after:
        return None
    try:
        with checkpoint.open("rb") as stream:
            header = struct.unpack("<8s4I8Q", stream.read(88))
    except (OSError, struct.error):
        return None
    if header[6] != archive_hash:
        return None
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
