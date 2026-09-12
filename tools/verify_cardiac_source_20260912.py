#!/usr/bin/env python3
"""Independently verify retained cardiac source evidence; never step physics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numi.human.cardiac-source-evidence.v1"
SOURCE_REVISION = "a679cdc2e97429fb5280af8132c119758626c1f2"
NATIVE_COMMIT = "f5ce0ce563d54314f7226bb56b8f323de8dd69c3"
NATIVE_BASE_COMMIT = "1594f7aff5503aab96ab9de6e77b6d4f8fa4f1cd"
SOURCE_ARCHIVE_SHA256 = "91d6b586c0caaa0fd59ec21cef873f1348563af0033210fbadfe84921a52782d"
SOURCE_HEADER_SHA256 = "1f847a6642b13667cecf562c861074e4cd1dee7a8a8b9bf8753cd8c8e71c3e5f"
KNOWN_FAILURE_LOG = "retained-failure-first-ten-cycles-2ms.log"
KNOWN_FAILURE_TRACE = "retained-failure-first-ten-cycles-2ms.csv.gz"
KNOWN_FAILURE_LOG_SHA256 = "57d996b36878816c77842ebdeef5c1959ebf7787ee0d19282c54fdfbc36dc333"
KNOWN_FAILURE_TRACE_RAW_SHA256 = "0d6f5aa8b2ef229b0cc2d2dd1df1b1be6e3fbe32ff8f4d91706d7c483934b921"
SOURCE_INPUT_SHA256 = "7797e2710837ad8ffdfa70adf7a6b66e68f2bf0e12824f063b5ce7d1fc550788"
SHA256 = re.compile(r"[0-9a-f]{64}")
SHA1 = re.compile(r"[0-9a-f]{40}")
CTEST_CASES = {
    "matter.runtime.snapshot_archive", "matter.physics.monolithic_multiphysics",
    "matter.runtime.production_transaction_rollback", "matter.physics.stateful_mpm",
    "matter.physics.stateful_fem", "numanx.integration.matter_accepted_state_2pc",
    "matter.compiler.stateful_roundtrip", "matter.compiler.vascular", "matter.metal.vascular",
    "matter.metal.human_physiology_payload", "matter.compiler.human_physiology_admission",
    "matter.metal.cardiac_transaction", "matter.metal.cardiac_source",
}
REQUIRED_ARTIFACTS = {"native-identity.json", "native-ctest.txt", "native-tests-detailed.txt",
                      "shi-hose.native.v2.json", "shi-hose.native.v2.manifest.json",
                      KNOWN_FAILURE_LOG, KNOWN_FAILURE_TRACE}
REQUIRED_SOURCES = {
    "src/numilab_human/shi_hose.py", "config/shi-hose-cardiac-source.v1.json",
    "schemas/humanpack-shi-hose-source-config.v1.schema.json",
    "third_party/physiome/shi_hose_2009/source-lock.json",
    "tools/shi_hose_reference/source-lock.json", "tools/shi_hose_reference/generate_reference.py",
    "tools/shi_hose_reference/generate_observables.py", "tools/shi_hose_reference/verify_generation.py",
    "tools/shi_hose_reference/compare_traces.py", "tools/shi_hose_reference/reference.cpp",
    "tools/shi_hose_reference/native-state-mapping.json",
    "tools/verify_cardiac_source_20260912.py", "tests/test_cardiac_source_evidence.py",
}
NATIVE_SOURCES = {
    "matter/include/numi/matter/matter.hpp", "matter/include/numi/matter/shared.h",
    "matter/src/vascular_impl.cpp", "matter/src/runtime.mm", "matter/src/metal/vascular.metalinc",
    "matter/src/metal/fgmres.metalinc", "matter/src/fingerprint_impl.cpp", "matter/src/package_impl.cpp",
    "matter/tools/cardiac_check.mm", "matter/tools/cardiac_transaction_check.mm",
    "matter/tools/fixtures/shi-hose.native.v2.json",
    "matter/tools/shi_hose_reference/shi_hose_reference_generated.hpp",
    "matter/tools/shi_hose_reference/shi_hose_observables.hpp",
    "matter/tools/shi_hose_reference/shi_hose_dopri.hpp",
}
NATIVE_BINARIES = {"lib/libmetalrobo.dylib", "matter/numi-matter-cardiac-check",
                   "matter/numi-matter-cardiac-transaction-check", "matter/numi-matter-physiologyc",
                   "matter/shaders/NumiMatter.metallib"}
COMPARTMENT_ORDER = ["LA", "LV", "Sas", "Sat", "Svn", "RA", "RV", "Pas", "Pat", "Pvn"]
CHAMBER_ROWS = {0, 1, 5, 6}
VALVE_ROWS = {10, 11, 15, 16}
RUNS = {"refinement_2ms": (.002, 500), "refinement_1ms": (.001, 1000),
        "refinement_05ms": (.0005, 2000), "ten_cycles_2ms": (.002, 5000)}
METRICS = ["max_chamber_volume_error_m3", "max_storage_error_m3",
           "max_flow_error_m3_per_s", "scaled_state_rms_error"]
# Fixed before the repaired final cohort. These are numerical regression
# envelopes informed by retained initial-transient errors, not physiological
# accuracy, patient calibration, or biological acceptance criteria.
ACCURACY_LIMITS = {
    "refinement_2ms": dict(zip(METRICS, [2e-6, 2e-6, 6.5e-4, .15])),
    "refinement_1ms": dict(zip(METRICS, [1.2e-6, 1.2e-6, 3.75e-4, .085])),
    "refinement_05ms": dict(zip(METRICS, [.65e-6, .65e-6, 2e-4, .045])),
    "ten_cycles_2ms": dict(zip(METRICS, [5e-6, 5e-6, 7.5e-4, .2])),
}


class EvidenceError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_bytes(path: Path) -> bytes:
    require(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024, f"missing or oversized evidence: {path.name}")
    return path.read_bytes()


def read_json(path: Path) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(read_bytes(path), object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"nonfinite JSON: {token}")))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise EvidenceError(f"invalid JSON: {path.name}") from error
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def safe_path(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative, "unsafe evidence path")
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts and relative == part.as_posix() and part.parts,
            "unsafe evidence path")
    path = root / Path(*part.parts)
    require(path.resolve().is_relative_to(root.resolve()), "evidence path escapes root")
    current = root
    for component in part.parts:
        current /= component
        require(not current.is_symlink(), "symlink evidence path is not admitted")
    return path


def file_hash(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def hash_map(value: Any, context: str) -> dict[str, str]:
    require(isinstance(value, dict) and 0 < len(value) <= 512, f"missing {context} hash map")
    for name, fingerprint in value.items():
        require(isinstance(name, str) and isinstance(fingerprint, str) and SHA256.fullmatch(fingerprint) and fingerprint != "0"*64,
                f"invalid {context} hash")
    return value


def records(text: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in text.splitlines():
        if not line.startswith(prefix + "="):
            continue
        record = {}
        for token in line.split():
            require("=" in token, f"malformed {prefix} metric")
            key, value = token.split("=", 1)
            require(key and value and key not in record, f"duplicate/empty {prefix} metric")
            record[key] = value
        result.append(record)
    return result


def numeric(record: dict, key: str, *, maximum: float | None = None, positive: bool = False) -> float:
    try:
        value = float(record[key])
    except (KeyError, ValueError, TypeError) as error:
        raise EvidenceError(f"missing/invalid {key}") from error
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), f"invalid {key}")
    if maximum is not None:
        require(value < maximum, f"{key} exceeds qualification gate")
    return value


def exact(record: dict, values: dict, context: str) -> None:
    require(isinstance(record, dict), f"{context} must be an object")
    require(all(record.get(key) == value and (type(value) is not bool or type(record.get(key)) is bool)
                for key, value in values.items()), f"{context} does not meet its required boundary")


def close(actual: float, expected: float, context: str, *, absolute: float = 1e-25,
          relative: float = 2e-9) -> None:
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute), f"{context} disagrees with retained trace")


def float32(value: float) -> float:
    try:
        result = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise EvidenceError("unrepresentable FP32 source value") from error
    require(math.isfinite(result), "nonfinite FP32 source value")
    return result


def sequential_sum(values: Any) -> float:
    # Match the explicitly ordered C++ diagnostic, without Python 3.12+'s
    # compensated built-in sum changing a near-roundoff conservation metric.
    total = 0.
    for value in values:
        total += value
    return total


def csv_data(path: Path) -> tuple[list[str], Any, str]:
    require(path.name.endswith(".csv.gz"), "retained traces must use gzip CSV")
    require(path.is_file() and path.stat().st_size <= 64 * 1024 * 1024, "missing or oversized compressed trace")
    try:
        with gzip.open(path, "rb") as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        require(len(raw) <= 64 * 1024 * 1024, "oversized decompressed trace")
        rows = csv.reader(io.StringIO(raw.decode("utf-8")))
        header = next(rows)
    except (OSError, EOFError, UnicodeError, StopIteration, csv.Error) as error:
        raise EvidenceError("invalid compressed CSV trace") from error
    require(header and len(header) == len(set(header)), "duplicate or empty trace columns")
    return header, rows, hashlib.sha256(raw).hexdigest()


def numbers(row: list[str], arity: int) -> list[float]:
    require(len(row) == arity, "trace coordinate arity differs")
    try:
        values = [float(value) for value in row]
    except (ValueError, TypeError) as error:
        raise EvidenceError("invalid numerical trace coordinate") from error
    require(all(math.isfinite(value) for value in values), "nonfinite trace coordinate")
    return values


def audit_ctest(ctest: str, detailed: str) -> None:
    summaries = re.findall(r"^100% tests passed(?:, 0 tests failed)? out of (\d+)\s*$", ctest, re.MULTILINE)
    require(summaries == ["13"], "CTest must report exactly 13/13 passed")
    cases = re.findall(r"^\s*(\d+)/13 Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed\s+[0-9.]+ sec\s*$", ctest, re.MULTILINE)
    require(len(cases) == 13 and {int(row[0]) for row in cases} == set(range(1, 14)) and
            {row[1] for row in cases} == CTEST_CASES, "CTest case coverage is missing, duplicated or failed")
    waveform = records(detailed, "cardiac_waveform")
    require(len(waveform) == 1, "missing cardiac waveform transaction evidence")
    exact(waveform[0], {"cardiac_waveform": "pass", "phase_cases": "10", "huge_clock_phase": "bitwise"}, "cardiac waveform")
    numeric(waveform[0], "fp64_absolute_max", maximum=5e-4)
    clock = records(detailed, "cardiac_clock_kernel")
    require(clock == [{"cardiac_clock_kernel": "pass", "carry": "exact", "overflow": "denied",
                       "environment_isolation": "pass"}], "cardiac clock carry/overflow/isolation evidence failed")


def source_coordinates(payload: dict) -> tuple[list[float], list[float]]:
    exact(payload, {"schema": "HumanPack.physiology-native.v2", "qualification": "source_model_reproduction",
                    "law": "closed_periodic_elastance_orifice_v2"}, "native source input")
    require(payload.get("species") == [] and payload.get("tissue_reservoirs") == [] and payload.get("exchanges") == [],
            "hydraulic source input cannot qualify species or tissue exchange")
    nodes = payload.get("compartments", [])
    edges = payload.get("connections", [])
    require(len(nodes) == len(edges) == 10 and {node.get("id") for node in nodes} == set(COMPARTMENT_ORDER),
            "full source-loop compartment coverage differs")
    by_name = {node["id"]: node for node in nodes}
    by_edge = {edge["id"]: edge for edge in edges}
    require(len(by_edge) == 10 and set(by_edge) == {name+"_outflow" for name in COMPARTMENT_ORDER},
            "full source-loop outlet coverage differs")
    scales, initial = [], []
    for index, name in enumerate(COMPARTMENT_ORDER):
        node = by_name[name]
        expected_storage = "absolute_volume" if index in CHAMBER_ROWS else "storage_displacement"
        require(node.get("storage_kind") == expected_storage, "source volume/storage semantics changed")
        if index in CHAMBER_ROWS:
            require(node.get("source_pi") == 3.14159 and node.get("period_seconds") == 1,
                    "source waveform literals changed")
        scale = float32(numeric(node, "volume_scale_m3", positive=True))
        volume = numeric(node, "initial_volume_m3")
        scales.append(scale)
        initial.append(float32(float32(volume) / scale) * scale)
    for name in COMPARTMENT_ORDER:
        edge = by_edge[name+"_outflow"]
        scale = float32(numeric(edge, "flow_scale_m3_per_s", positive=True))
        value = edge.get("initial_flow_m3_per_s")
        require(type(value) in (float, int) and math.isfinite(value), "invalid initial source flow")
        scales.append(scale)
        initial.append(float32(float32(value) / scale) * scale)
    return scales, initial


def audit_native_run(trace: Path, log: str, definition: dict, scales: list[float], initial: list[float],
                     payload: dict | None = None) -> dict:
    expected_dt, expected_steps = RUNS[definition["id"]]
    require(definition.get("dt_seconds") == expected_dt and type(definition.get("steps")) is int and
            definition["steps"] == expected_steps, "native run duration/refinement contract differs")
    require("cardiac_source_run=failed " not in log, "native source run failed before full acceptance")
    device_rows = re.findall(r"^cardiac_device=Apple M4 Pro abi=27 world_fingerprint=(\d+)\s*$", log, re.MULTILINE)
    require(len(device_rows) == 1 and int(device_rows[0]) > 0, "native run lacks the qualified physical device/ABI identity")
    summaries = records(log, "cardiac_source_run")
    require(len(summaries) == 1, "missing or duplicate native source run summary")
    record = summaries[0]
    exact(record, {"cardiac_source_run": "pass", "accepted_steps": str(expected_steps), "environments": "2",
                   "failed_steps": "0", "clock": "exact_binary_128", "replay_pair": "bitwise",
                   "qualification": "source_model_numerical_comparison", "absolute_vascular_blood_volume": "unqualified",
                   "biological_calibration": "unqualified"}, "native source run")
    dt = float32(expected_dt)
    close(numeric(record, "dt_seconds", positive=True), dt, "cooked timestep", relative=1e-10)
    close(numeric(record, "duration_seconds", positive=True), expected_steps*dt, "accepted duration", relative=1e-10)
    numeric(record, "wall_seconds", positive=True)
    require(numeric(record, "reference_accepted_steps", positive=True) >= expected_steps and
            numeric(record, "reference_rejected_steps") >= 0, "missing independent source integrator evidence")
    header, rows, _ = csv_data(trace)
    require(header == ["time_seconds"] + [name+str(i) for i in range(20) for name in ("native_", "cellml_")],
            "native trace must contain the same 20 paired hydraulic coordinates")
    maxima = [0., 0., 0.]
    coordinate_maxima = [0.]*20
    squared = 0.
    invariant = 0.
    initial_storage = sequential_sum(initial[:10])
    count = 0
    for count, row in enumerate(rows, 1):
        require(count <= expected_steps, "native trace has extra accepted samples")
        values = numbers(row, 41)
        require(values[0] == count*dt, "native trace time disagrees with exact cooked clock")
        actual, reference = values[1::2], values[2::2]
        require(all(actual[i] > 0 and reference[i] > 0 for i in CHAMBER_ROWS), "nonpositive absolute chamber volume in trace")
        require(all(actual[i] >= 0 and reference[i] >= 0 for i in VALVE_ROWS), "negative one-way valve flow in trace")
        for index, (observed, expected) in enumerate(zip(actual, reference)):
            error = abs(observed-expected)
            coordinate_maxima[index] = max(coordinate_maxima[index], error)
            kind = 0 if index in CHAMBER_ROWS else 1 if index < 10 else 2
            maxima[kind] = max(maxima[kind], error)
            squared += (error/scales[index])**2
        invariant = max(invariant, abs(sequential_sum(actual[:10])-initial_storage)/abs(initial_storage))
        require(invariant < 1e-4, "native trace violates closed hydraulic storage conservation")
    require(count == expected_steps, "native trace is truncated")
    result = dict(zip(METRICS, [*maxima, math.sqrt(squared/(20*count))]))
    result["relative_storage_invariant_error"] = invariant
    for key, measured in result.items():
        close(numeric(record, key), measured, key, absolute=2e-14 if key == "relative_storage_invariant_error" else 1e-25)
    for key, ceiling in ACCURACY_LIMITS[definition["id"]].items():
        require(result[key] <= ceiling, f"numerical regression envelope exceeded for {definition['id']} {key}")
    result["per_chamber_volume_error_m3"] = {COMPARTMENT_ORDER[index]: coordinate_maxima[index]
                                             for index in sorted(CHAMBER_ROWS)}
    if payload is not None:
        nodes = {node["id"]: node for node in payload["compartments"]}
        result["linear_vascular_pressure_equivalent_error_pa"] = {
            COMPARTMENT_ORDER[index]: coordinate_maxima[index]/float32(nodes[COMPARTMENT_ORDER[index]]["compliance_m3_per_pa"])
            for index in range(10) if index not in CHAMBER_ROWS}
    return {"id": definition["id"], "steps": expected_steps, "duration_seconds": expected_steps*dt, **result}


def audit_native_refinement(runs: dict[str, dict]) -> dict[str, list[float]]:
    ordered = [runs[name] for name in ["refinement_2ms", "refinement_1ms", "refinement_05ms"]]
    ratios = {}
    require(len({row["duration_seconds"] for row in ordered}) == 1, "refinement traces do not have the same accepted duration")
    for key in METRICS:
        errors = [numeric(row, key, positive=True) for row in ordered]
        ratios[key] = [errors[i]/errors[i+1] for i in range(2)]
        require(all(value >= 1.5 for value in ratios[key]), f"first-order timestep refinement failed for {key}")
    return ratios


def artifact(paths: dict[str, Path], name: Any) -> Path:
    require(isinstance(name, str) and name in paths, "evidence role refers to an unlisted artifact")
    return paths[name]


def audit_run_identity(definition: dict, paths: dict[str, Path], native_sources: dict, native_binaries: dict) -> None:
    identity = read_json(artifact(paths, definition.get("identity")))
    exact(identity, {"schema": "numi.human.cardiac-run-identity.v1", "native_base_commit": NATIVE_BASE_COMMIT,
                     "returncode": 0}, "native execution identity")
    require(type(identity["returncode"]) is int, "native return code must be integral")
    sources = hash_map(identity.get("source_sha256"), "run source")
    sources_after = hash_map(identity.get("source_sha256_after"), "run source after")
    binaries = hash_map(identity.get("built_artifact_sha256"), "run binary")
    binaries_after = hash_map(identity.get("built_artifact_sha256_after"), "run binary after")
    require(sources == sources_after and binaries == binaries_after, "native code/binary changed during the qualified execution")
    require(NATIVE_SOURCES <= sources.keys() and
            {"matter/numi-matter-cardiac-check", "matter/shaders/NumiMatter.metallib"} <= binaries.keys(),
            "native execution source/binary coverage is incomplete")
    require(all(native_sources.get(name) == value for name, value in sources.items()) and
            all(native_binaries.get(name) == value for name, value in binaries.items()),
            "native execution does not match the final published source/binary identity")
    command = identity.get("command")
    require(isinstance(command, list) and len(command) in {8, 10} and all(isinstance(arg, str) and arg for arg in command),
            "missing exact native execution command")
    require(command[0].endswith("/matter/numi-matter-cardiac-check") and
            command[1].endswith("/matter/tools/fixtures/shi-hose.native.v2.json"), "native command used another owner or input")
    keys = command[2::2]
    require(len(keys) == len(set(keys)) and {"--steps", "--dt", "--trace"} <= set(keys) <=
            {"--steps", "--dt", "--trace", "--checkpoint"}, "native command has missing or unexpected options")
    options = dict(zip(command[2::2], command[3::2]))
    require(options["--steps"] == str(definition["steps"]) and float(options["--dt"]) == definition["dt_seconds"] and
            options["--trace"].endswith(".csv"), "native execution command does not match the qualified duration")
    require("--checkpoint" not in options or options["--checkpoint"].endswith(".bin"), "unexpected diagnostic checkpoint output")
    require(identity.get("log_sha256") == file_hash(artifact(paths, definition.get("log"))) and
            identity.get("trace_raw_sha256") == csv_data(artifact(paths, definition.get("trace")))[2],
            "native execution identity is not bound to the selected log/trace")


def audit_oracle(definition: dict, paths: dict[str, Path], repository_root: Path, native_sources: dict) -> dict:
    require(isinstance(definition, dict), "missing independent source oracle evidence")
    build = read_json(artifact(paths, definition.get("build_manifest")))
    exact(build, {"schema": "NumiHuman.CellML-reference-build.v1", "source_revision": SOURCE_REVISION,
                  "source_header_sha256": SOURCE_HEADER_SHA256, "license": "CC-BY-3.0",
                  "ode_states": 14, "native_hydraulic_coordinates": 20,
                  "all_source_initial_states_unchanged": True, "pressure_unit_Pa": 133,
                  "source_pi_literal": "3.14159", "absolute_vascular_blood_volumes_sourced": False,
                  "native_comparison_performed_by_this_tool": False}, "source oracle build")
    require(build.get("source_archive", {}).get("sha256") == SOURCE_ARCHIVE_SHA256,
            "oracle archive identity differs from pinned source")
    require(build.get("host_architecture") == "arm64" and isinstance(build.get("compiler_version"), str) and
            "clang" in build["compiler_version"], "missing FP64 reference compiler identity")
    require(isinstance(build.get("build_command"), list) and
            {"-fno-fast-math", "-ffp-contract=off"} <= set(build["build_command"]),
            "source reference floating-point build flags missing")
    built = build.get("artifacts", {})
    for header in ["shi_hose_reference_generated.hpp", "shi_hose_observables.hpp", "shi_hose_dopri.hpp"]:
        require(native_sources.get("matter/tools/shi_hose_reference/"+header) == built.get(header, {}).get("sha256"),
                "native reference header differs from independent source build")
    for script in ["generate_reference.py", "generate_observables.py", "compare_traces.py", "source-lock.json"]:
        require(file_hash(repository_root / "tools/shi_hose_reference" / script) == built.get(script, {}).get("sha256"),
                "source oracle generator identity differs from executed build")
    require(file_hash(repository_root / "tools/shi_hose_reference/reference.cpp") ==
            built.get("shi_hose_reference.cpp", {}).get("sha256"), "source oracle C++ driver identity differs")
    generation = read_json(artifact(paths, definition.get("generation_check")))
    exact(generation, {"schema": "NumiHuman.CellML-generation-check.v1", "status": "pass", "deterministic": True,
                       "changed_import_rejected": True, "native_headers_checked": True}, "source generation verification")
    require(generation == build.get("generation_verification") and
            generation.get("headers", {}).get("shi_hose_reference_generated.hpp") == SOURCE_HEADER_SHA256,
            "generation check does not bind executed reference equations")
    refinement = read_json(artifact(paths, definition.get("refinement")))
    exact(refinement, {"schema": "NumiHuman.CellML-reference-refinement.v1", "source_revision": SOURCE_REVISION,
                       "samples": 20001, "refinement_decreases_pressure_flow_volume_maxima": True}, "source refinement")
    require(file_hash(artifact(paths, definition["refinement"])) == built.get("reference-refinement.json", {}).get("sha256"),
            "source refinement artifact differs from executed reference build")
    mapping = read_json(repository_root / "tools/shi_hose_reference/native-state-mapping.json")
    header_expected = ["time_s"] + [row["name"] for row in mapping["observables"]] + ["native_hydraulic_"+str(i) for i in range(20)]
    require(len(header_expected) == 57, "source observable mapping changed")
    definitions = definition.get("runs")
    require(isinstance(definitions, list) and len(definitions) == 3 and all(isinstance(row, dict) for row in definitions) and
            {row.get("tolerance") for row in definitions} == {1e-10, 1e-12, 1e-13},
            "three independent reference tolerance runs are required")
    definitions = sorted(definitions, key=lambda row: -row["tolerance"])
    readers, reports = [], []
    for row in definitions:
        path = artifact(paths, row.get("trace"))
        header, reader, raw_hash = csv_data(path)
        require(header == header_expected, "source trace SI/20-state mapping differs")
        raw_name = path.name[:-3]
        require(raw_hash == built.get(raw_name, {}).get("sha256"), "source trace raw hash differs from C++ run provenance")
        require(any(item.get("file") == raw_name and item.get("sha256") == raw_hash for item in refinement.get("traces", [])),
                "reference refinement does not bind raw trace")
        report_path = artifact(paths, row.get("report"))
        report = read_json(report_path)
        require(file_hash(report_path) == built.get(report_path.name, {}).get("sha256"), "source run report differs from build provenance")
        exact(report, {"schema": "NumiHuman.CellML-reference-run.v1", "source_revision": SOURCE_REVISION,
                       "boundary": "source_hydraulic_reproduction_only", "seconds": 20,
                       "output_interval_seconds": .001, "samples": 20001,
                       "source_coordinate_rtol": row["tolerance"], "source_coordinate_atol": row["tolerance"],
                       "initial_states_modified": False, "native_equations_reused": False,
                       "absolute_vascular_volume_calibrated": False}, "source reference run")
        require(numeric(report, "accepted_internal_steps", positive=True) >= 20000, "source reference internal integration coverage missing")
        numeric(report, "minimum_accepted_dt", positive=True)
        numeric(report, "maximum_local_error_ratio", maximum=1.00000000000001)
        numeric(report, "maximum_storage_error_m3", maximum=1e-12)
        readers.append(reader)
        reports.append(report)
    maxima = [[0.]*57 for _ in range(2)]
    storage_initial = [0.]*3
    storage_error = [0.]*3
    first_tight = last_tight = None
    count = 0
    for index, triplet in enumerate(itertools.zip_longest(*readers)):
        require(index <= 20000 and all(row is not None for row in triplet), "source reference sample count differs")
        triples = [numbers(row, 57) for row in triplet]
        for run, values in enumerate(triples):
            require(values[0] == index*.001, "source reference time grid differs")
            storage = sequential_sum(values[37:47])
            if index == 0:
                storage_initial[run] = storage
            storage_error[run] = max(storage_error[run], abs(storage-storage_initial[run]))
        if index == 0:
            first_tight = triples[2][37:]
        last_tight = triples[2][37:]
        for pair in range(2):
            for column in range(57):
                maxima[pair][column] = max(maxima[pair][column], abs(triples[pair][column]-triples[pair+1][column]))
        count += 1
    require(count == 20001, "source reference trace is truncated")
    require(first_tight == refinement.get("initial_native_hydraulic_state_SI") and
            last_tight == refinement.get("final_native_hydraulic_state_SI"), "source reference endpoint mapping differs")
    for run, report in enumerate(reports):
        close(numeric(report, "initial_hydraulic_storage_m3", positive=True), storage_initial[run], "source initial hydraulic storage")
        close(numeric(report, "maximum_storage_error_m3"), storage_error[run], "source storage invariant", absolute=2e-18)
    for pair, name in enumerate(["coarse_1e10_vs_1e12", "tight_1e12_vs_1e13"]):
        columns = refinement.get(name, {}).get("columns", [])
        require(len(columns) == 57 and [row.get("name") for row in columns] == header_expected,
                "source refinement coordinate coverage differs")
        for index, metric in enumerate(columns):
            close(numeric(metric, "maximum_absolute_difference"), maxima[pair][index], "source refinement maximum")
    families = {}
    for suffix in ["_pressure_Pa", "_volume_m3", "_outflow_m3_per_s"]:
        indices = [i for i, name in enumerate(header_expected) if name.endswith(suffix)]
        coarse = max(maxima[0][i] for i in indices)
        tight = max(maxima[1][i] for i in indices)
        require(0 <= tight < coarse, "source tolerance refinement does not decrease")
        families[suffix.removeprefix("_")] = {"coarse_difference": coarse, "tight_difference": tight}
    return {"cycles": 20, "samples_per_run": count, "reference_refinement": families,
            "maximum_source_storage_error_m3": max(storage_error)}


def verify(receipt_path: Path, *, repository_root: Path = ROOT) -> dict:
    receipt = read_json(receipt_path)
    exact(receipt, {"schema": SCHEMA, "source_revision": SOURCE_REVISION, "scientific_status": "unqualified",
                    "absolute_vascular_blood_volume": "unqualified", "boundary": "bounded_source_model_numerical_reproduction"}, "receipt")
    require(receipt.get("native_commit") == NATIVE_COMMIT,
            "native commit differs from the frozen qualified revision")
    inventory = receipt.get("artifacts")
    require(isinstance(inventory, list) and 20 <= len(inventory) <= 128, "missing cardiac artifact inventory")
    paths = {}
    for record in inventory:
        require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "invalid artifact record")
        path = safe_path(receipt_path.parent, record["path"])
        require(record["path"] not in paths and path != receipt_path, "duplicate or recursive artifact path")
        require(isinstance(record["sha256"], str) and SHA256.fullmatch(record["sha256"]), "invalid artifact hash")
        require(file_hash(path) == record["sha256"], f"artifact hash drift: {record['path']}")
        paths[record["path"]] = path
    require(REQUIRED_ARTIFACTS <= paths.keys(), "required cardiac artifact is missing")
    actual = {path.relative_to(receipt_path.parent).as_posix() for path in receipt_path.parent.rglob("*")
              if path.is_file() and path != receipt_path}
    require(actual == paths.keys(), "artifact inventory does not capture every retained file")
    hashes = hash_map(receipt.get("source_sha256"), "Human source")
    require(REQUIRED_SOURCES <= hashes.keys(), "required owning Human source pin missing")
    for name, expected in hashes.items():
        require(file_hash(safe_path(repository_root, name)) == expected, f"Human source hash drift: {name}")
    source_lock = read_json(repository_root / "tools/shi_hose_reference/source-lock.json")
    exact(source_lock, {"revision": SOURCE_REVISION, "license": "CC-BY-3.0"}, "CellML source lock")
    require(source_lock.get("archive", {}).get("sha256") == SOURCE_ARCHIVE_SHA256 and len(source_lock.get("cellml_files", {})) == 15,
            "CellML archive/import coverage differs")
    for name, expected in hash_map(source_lock["cellml_files"], "CellML source").items():
        relative = "third_party/physiome/shi_hose_2009/"+name
        require(hashes.get(relative) == expected and file_hash(safe_path(repository_root, relative)) == expected,
                "CellML import identity drift")
    identity = read_json(paths["native-identity.json"])
    exact(identity, {"native_commit": receipt["native_commit"], "worktree_status": "", "abi": 27,
                     "package_version": 12, "snapshot_archive": 6, "accepted_proof_manifest": 6}, "native identity")
    for field in ["abi", "package_version", "snapshot_archive", "accepted_proof_manifest"]:
        require(type(identity[field]) is int, "native version must be integral")
    exact(identity.get("device", {}), {"chip": "Apple M4 Pro", "model": "Mac mini"}, "physical Metal host")
    native_sources = hash_map(identity.get("source_sha256"), "native source")
    binaries = hash_map(identity.get("built_artifact_sha256"), "native built artifacts")
    require(NATIVE_SOURCES <= native_sources.keys() and NATIVE_BINARIES <= binaries.keys(), "native owner source/binary provenance incomplete")
    require(native_sources["matter/tools/shi_hose_reference/shi_hose_reference_generated.hpp"] == SOURCE_HEADER_SHA256,
            "native oracle equations are not the independently generated source")
    input_hash = file_hash(paths["shi-hose.native.v2.json"])
    require(input_hash == SOURCE_INPUT_SHA256 and native_sources["matter/tools/fixtures/shi-hose.native.v2.json"] == input_hash,
            "qualified native input differs from actual source authoring")
    payload = read_json(paths["shi-hose.native.v2.json"])
    lowering = read_json(paths["shi-hose.native.v2.manifest.json"])
    config = read_json(repository_root / "config/shi-hose-cardiac-source.v1.json")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    source_hash = hashes["third_party/physiome/shi_hose_2009/source-lock.json"]
    exact(payload, {"authored_graph_sha256": config_hash, "source_graph_sha256": source_hash}, "source authoring identity")
    exact(lowering, {"schema": "HumanPack.shi-hose-source-lowering.v1", "revision": SOURCE_REVISION,
                     "license": "CC-BY-3.0", "source_parameter_count": 59, "native_content_sha256": input_hash,
                     "authored_config_sha256": config_hash, "source_manifest_sha256": source_hash,
                     "scientific_status": "source_model_reproduction_target_not_qualified_by_compilation"}, "source lowering")
    scales, initial = source_coordinates(payload)
    audit_ctest(read_bytes(paths["native-ctest.txt"]).decode(), read_bytes(paths["native-tests-detailed.txt"]).decode())
    definitions = receipt.get("native_runs")
    require(isinstance(definitions, list) and len(definitions) == 4 and all(isinstance(row, dict) for row in definitions) and
            {row.get("id") for row in definitions} == RUNS.keys(),
            "three refinement runs and successful ten-cycle run are required")
    require(len({row.get("log") for row in definitions}) == 4 and len({row.get("trace") for row in definitions}) == 4,
            "native run artifacts must be distinct")
    require(len({row.get("identity") for row in definitions}) == 4, "native runs require distinct execution identities")
    require(all(isinstance(row.get(key), str) and not Path(row[key]).name.startswith("retained-")
                for row in definitions for key in ["log", "trace"]),
            "archived before-fix evidence cannot qualify the final native cohort")
    for definition in definitions:
        audit_run_identity(definition, paths, native_sources, binaries)
    runs = {row["id"]: audit_native_run(artifact(paths, row.get("trace")), read_bytes(artifact(paths, row.get("log"))).decode(),
                                       row, scales, initial, payload) for row in definitions}
    ratios = audit_native_refinement(runs)
    failures = receipt.get("retained_failures", [])
    require(isinstance(failures, list) and failures, "known failed ten-cycle evidence cannot be omitted")
    require(file_hash(paths[KNOWN_FAILURE_LOG]) == KNOWN_FAILURE_LOG_SHA256 and
            csv_data(paths[KNOWN_FAILURE_TRACE])[2] == KNOWN_FAILURE_TRACE_RAW_SHA256,
            "known failed ten-cycle evidence identity changed")
    classified_failures = set()
    for failure in failures:
        require(isinstance(failure, dict) and failure.get("resolved_by") == "ten_cycles_2ms" and
                isinstance(failure.get("reason"), str) and failure["reason"], "retained failure has no explicit resolution")
        log = read_bytes(artifact(paths, failure.get("log"))).decode()
        require(failure["log"] not in classified_failures, "duplicate retained failure role")
        classified_failures.add(failure["log"])
        require("cardiac_source_run=failed " in log and "cardiac_source_run=pass " not in log,
                "retained failed run was relabelled as passed")
        header, rows, _ = csv_data(artifact(paths, failure.get("trace")))
        require(header == ["time_seconds"] + [name+str(i) for i in range(20) for name in ("native_", "cellml_")],
                "retained failed trace has different coordinates")
        index = 0
        for index, row in enumerate(rows, 1):
            require(index < 5000 and numbers(row, 41)[0] == index*float32(.002), "retained failed trace clock/coverage changed")
        require(index > 0, "retained numerical failure has no partial trace")
    recorded_failures = {name for name, path in paths.items() if path.suffix in {".log", ".txt"} and
                         b"cardiac_source_run=failed " in read_bytes(path)}
    require(classified_failures == recorded_failures, "retained failed run was omitted from failure inventory")
    require(any(row.get("log") == KNOWN_FAILURE_LOG and row.get("trace") == KNOWN_FAILURE_TRACE for row in failures),
            "known failed ten-cycle trace is not bound to its failure role")
    oracle = audit_oracle(receipt.get("oracle"), paths, repository_root, native_sources)
    return {"status": "pass", "scientific_status": "unqualified", "absolute_vascular_blood_volume": "unqualified",
            "boundary": "bounded_source_model_numerical_reproduction", "native_commit": receipt["native_commit"],
            "native_tests_passed": 13, "native_runs": runs, "native_refinement_error_ratios": ratios,
            "accuracy_gate_scope": "numerical_regression_envelopes_not_physiological_acceptance",
            "numerical_regression_envelopes": ACCURACY_LIMITS,
            "retained_failed_runs": len(failures), "oracle": oracle}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.receipt, repository_root=args.repository_root), indent=2))
    except (EvidenceError, OSError, UnicodeError, TypeError, KeyError, ValueError, csv.Error) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
