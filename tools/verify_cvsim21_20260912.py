#!/usr/bin/env python3
"""Verify retained CVSim21 source/native evidence without physical stepping."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numi.human.cvsim21-evidence.v1"
NATIVE_COMMIT = "b91fe6832813497ab532e5dbe05ed8c8233e6b1f"
NATIVE_BASE = "f5ce0ce563d54314f7226bb56b8f323de8dd69c3"
SOURCE_LOCK_SHA256 = "b118d58230a3f2766108d05501a3ac9b910124ef823567a7e5fe23f6df0b4f67"
INITIAL_SHA256 = "fad558482f8bfeb94d531d4538e29d7a2c8a59ea63f32496467b087bcb343ff7"
EXECUTION_SHA256 = "0c1b5a4d026fdc7bdf15c5f3e6b43e7b1cc556a4bb00ecb7d45019fc5a7bfb41"
REFERENCE_AUDIT_SHA256 = "f4026f9457834e72b7b2b0b4e4a09710457721b759cfb208f39e01854b9045c4"
HISTORICAL_RECEIPT = "Docs/media/cardiac-source-20260912/receipt.json"
HISTORICAL_RECEIPT_SHA256 = "0120c58c0e4399460216f95012844977360c7e47115d604df999193e0d4322b1"
LEGACY_VERIFIER_SHA256 = "aa7c269334b0b413dd239be3a441ccf59bfdd356f5ff0883b161a53d8fa20978"
FAILURE_HASHES = {
    "native-law-compiler.log": "44751f7a76c728ffa1ab24d0fcf56dff831d2d637d81a2346ece7051c1fd2bf0",
    "native-cvsim-build.log": "75328cc17055c1d3f98a9619cc39bda6e92e8310697f80ceb0db4372fa69c3f4",
}
SOURCE_PREFIX = "third_party/physionet/cvsim21"
REFERENCE_PREFIX = "tools/cvsim21_reference/evidence/20260912"
SOURCE_FILES = {"main/initial.c", "main/initial.h", "main/main.h", "main/main_java.c", "main/main_java.h",
                "main/turning.c", "main/turning.h", "sim/equation.c", "sim/equation.h", "sim/estimate.c",
                "sim/estimate.h", "sim/reflex.c", "sim/reflex.h", "sim/rkqc.c", "sim/rkqc.h",
                "sim/simulator.c", "sim/simulator.h"}
REFERENCE_FILES = {"execution.json", "initial-manifest.json", "original-initial.json", "parameters.csv",
                   "reference-audit.json", "source-domain-audit.json", "source-mapping.json", "cvsim-source-parameters.json",
                   "continuous-1e-10.json", "continuous-1e-10.log", "continuous-1e-12.json", "continuous-1e-12.log",
                   "continuous-1e-13.json", "continuous-1e-13.log", "continuous-original-times.json",
                   "continuous-original-times.log", "original-supine.log", "continuous-build.log", "original-build.log",
                   "equation-build.log", "estimate-build.log", "initial-build.log", "main_java-build.log",
                   "reflex-build.log", "rkqc-build.log", "simulator-build.log", "turning-build.log"}
REQUIRED_OWNERS = {
    "src/numilab_human/cvsim21.py", "src/numilab_human/cvsim_parameters.py", "src/numilab_human/model.py",
    "config/cvsim21-source.v1.json", "config/cvsim21-heldt-table-aligned.v1.json",
    "schemas/humanpack-cvsim21-source-config.v1.schema.json", ".numi/commands/human-circulation",
    "tools/analyze_cvsim_native.py", "tools/verify_cvsim21_20260912.py", "tests/test_cvsim21_evidence.py",
    "tests/test_cvsim_native_analysis.py", "tests/test_cvsim21.py", "tests/test_cvsim_parameters.py",
    "tools/cvsim21_reference/headless.c", "tools/cvsim21_reference/continuous_reference.cpp",
    "tools/cvsim21_reference/cvsim21_source.hpp", "tools/cvsim21_reference/dopri.hpp",
    "tools/cvsim21_reference/run_reference.py", "tools/cvsim21_reference/analyze_reference.py",
    "tools/verify_cardiac_source_20260912.py", HISTORICAL_RECEIPT,
    "Docs/media/cardiac-source-20260912/shi-hose.native.v2.json", SOURCE_PREFIX + "/source-lock.json",
} | {SOURCE_PREFIX + "/21-comp-backend/" + name for name in SOURCE_FILES} | {REFERENCE_PREFIX + "/" + name for name in REFERENCE_FILES}
RUNS = {"cycle-2ms": (.002, 429), "cycle-1ms": (.001, 858), "cycle-0p5ms": (.0005, 1716),
        "ten-cycles-2ms": (.002, 4286), "heldt-cycle-1ms": (.001, 858)}
LEGACY_RUNS = {"refinement_2ms", "refinement_1ms", "refinement_05ms", "ten_cycles_2ms"}
SOURCE_TRACES = {"continuous-1e-10.csv.gz", "continuous-1e-12.csv.gz", "continuous-1e-13.csv.gz",
                 "continuous-original-times.csv.gz", "original-supine.csv.gz"}
REQUIRED_ARTIFACTS = {"native-identity.json", "native-ctest.txt", "native-comparison.json",
                      "cvsim21.native.v3.json", "cvsim21.native.v3.manifest.json",
                      "cvsim21-heldt.native.v3.json", "cvsim21-heldt.native.v3.manifest.json",
                      "legacy-cardiac-requalification.json", *FAILURE_HASHES, *SOURCE_TRACES}
REQUIRED_ARTIFACTS |= {name + suffix for name in RUNS for suffix in (".log", ".csv.gz", ".identity.json")}
REQUIRED_ARTIFACTS |= {"legacy-cardiac-requalification/" + name + suffix for name in LEGACY_RUNS
                       for suffix in (".log", ".csv.gz", ".execution.json")}
NATIVE_SOURCES = {
    "matter/include/numi/matter/matter.hpp", "matter/include/numi/matter/shared.h", "matter/include/numi/matter/vascular.hpp",
    "matter/src/vascular_impl.cpp", "matter/src/runtime.mm", "matter/src/human_physiology.mm",
    "matter/src/compiler_impl.cpp", "matter/src/validation_impl.cpp", "matter/src/fingerprint_impl.cpp", "matter/src/package_impl.cpp",
    "matter/src/metal/vascular.metalinc", "matter/src/metal/fgmres.metalinc", "matter/tools/cvsim_check.mm",
    "matter/tools/cvsim_input_check.py", "matter/tools/fixtures/cvsim21.native.v3.json",
}
NATIVE_BINARIES = {"matter/numi-matter-cvsim-check", "matter/shaders/NumiMatter.metallib",
                   "matter/numi-matter-physiologyc", "matter/numi-matter-cardiac-check",
                   "matter/numi-matter-cardiac-transaction-check", "matter/numi-matter-vascular-compiler-check",
                   "matter/numi-matter-vascular-check"}
CTEST_CASES = {"matter.runtime.snapshot_archive", "matter.physics.monolithic_multiphysics",
               "matter.runtime.production_transaction_rollback", "matter.physics.stateful_mpm", "matter.physics.stateful_fem",
               "numanx.integration.matter_accepted_state_2pc", "matter.compiler.stateful_roundtrip", "matter.compiler.vascular",
               "matter.metal.vascular", "matter.metal.human_physiology_payload", "matter.compiler.human_physiology_admission",
               "matter.metal.cardiac_transaction", "matter.metal.cardiac_source", "matter.metal.cvsim_source",
               "matter.compiler.cvsim_admission"}
SCOPE = {name: False for name in ("biological_calibration", "anatomy_registration", "mechanical_blood_mass_partition",
                                "species_transport", "reflexes", "tilt")}


class EvidenceError(ValueError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def relative_parts(relative: Any) -> tuple[str, ...]:
    require(isinstance(relative, str) and relative and "\\" not in relative and
            all(ord(c) >= 32 and ord(c) != 127 for c in relative), "unsafe relative path")
    part = PurePosixPath(relative)
    require(not part.is_absolute() and part.parts and ".." not in part.parts and part.as_posix() == relative,
            "path must be normalized and relative")
    return part.parts


def safe_path(root: Path, relative: Any) -> Path:
    current = root
    for component in relative_parts(relative):
        current /= component
        require(not current.is_symlink(), "symlink path is not admitted")
    require(current.resolve().is_relative_to(root.resolve()), "path escapes evidence root")
    return current


def digest(path: Path, *, decompress: bool = False) -> str:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024**2,
            f"missing, redirected or oversized artifact: {path.name}")
    result, size = hashlib.sha256(), 0
    opener = gzip.open if decompress else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            size += len(block)
            require(size <= 1024**3, "decompressed evidence exceeds bound")
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, "missing or oversized JSON")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique,
                           parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"nonfinite JSON: {token}")))
        # Also rejects exponent overflow (1e999), which parse_constant does not.
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise EvidenceError(f"invalid or nonfinite JSON: {path.name}: {error}") from error
    require(isinstance(value, dict), "JSON object required")
    return value


def hash_map(value: Any, label: str) -> dict:
    require(isinstance(value, dict) and 0 < len(value) <= 2048, f"missing {label} hash map")
    for name, sha in value.items():
        relative_parts(name)
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) and sha != "0" * 64,
                f"invalid {label} SHA256")
    return value


def checked_inventory(root: Path, values: Any, required: set[str], label: str) -> dict[str, Path]:
    values = hash_map(values, label)
    require(required <= values.keys(), f"missing required {label}: {sorted(required - values.keys())}")
    paths = {}
    for name, sha in values.items():
        path = safe_path(root, name)
        require(digest(path) == sha, f"{label} hash drift: {name}")
        if name.endswith(".json"):
            read_json(path)
        paths[name] = path
    return paths


def audit_scope(receipt: dict) -> None:
    require(set(receipt) == {"schema", "native_commit", "source_lock_sha256", "initial_sha256", "scope", "qualification",
                             "artifacts", "owners", "numerical_report_path"}, "unknown or missing receipt fields")
    require(receipt.get("schema") == SCHEMA and receipt.get("qualification") == "source_variant_numerical_comparison",
            "wrong evidence schema or qualification")
    require(isinstance(receipt.get("scope"), dict) and set(receipt["scope"]) == set(SCOPE) and
            all(receipt["scope"][key] is False for key in SCOPE), "unsupported scope promotion")
    require(receipt.get("source_lock_sha256") == SOURCE_LOCK_SHA256 and receipt.get("initial_sha256") == INITIAL_SHA256,
            "source or initial identity differs")
    require(receipt.get("numerical_report_path") == "native-comparison.json", "numerical report role differs")


def audit_native_identity(identity: dict, commit: str) -> tuple[dict, dict]:
    require(re.fullmatch(r"[0-9a-f]{40}", NATIVE_COMMIT) and commit == NATIVE_COMMIT,
            "native publication identity is absent or differs")
    expected = {"native_commit": commit, "native_base": NATIVE_BASE, "worktree_status": "", "abi": 28,
                "package_version": 13, "snapshot_archive": 6, "accepted_proof_manifest": 6}
    require(all(identity.get(key) == value and type(identity.get(key)) is type(value) for key, value in expected.items()),
            "native commit, clean worktree or ABI/package identity differs")
    device = identity.get("device")
    require(isinstance(device, dict) and device.get("chip") == "Apple M4 Pro" and device.get("model") == "Mac mini" and
            "Paravirtual" not in json.dumps(identity), "physical Mac mini M4 Pro identity missing")
    sources = hash_map(identity.get("source_sha256"), "native source")
    binaries = hash_map(identity.get("built_artifact_sha256"), "native binary")
    require(NATIVE_SOURCES <= sources.keys() and NATIVE_BINARIES <= binaries.keys(), "native owner coverage missing")
    return sources, binaries


def audit_ctest(text: str) -> None:
    require(re.findall(r"^100% tests passed(?:, 0 tests failed)? out of (\d+)\s*$", text, re.MULTILINE) == ["15"],
            "CTest must report exactly 15/15 passed")
    cases = re.findall(r"^\s*(\d+)/15 Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed\s+[0-9.]+ sec\s*$", text, re.MULTILINE)
    require(len(cases) == 15 and {int(row[0]) for row in cases} == set(range(1, 16)) and
            {row[1] for row in cases} == CTEST_CASES, "CTest case coverage missing, duplicated or failed")
    require(not re.search(r"(?:\*\*\*Failed|\*\*\*Not Run|[1-9][0-9]* tests failed)", text), "CTest contains a failure")


def matched_snapshots(before: Any, after: Any, final: dict, required: set[str], label: str) -> None:
    before, after = hash_map(before, label + " before"), hash_map(after, label + " after")
    require(before == after and required <= before.keys(), f"{label} changed or omitted owners during execution")
    require(all(final.get(name) == sha for name, sha in before.items()), f"{label} differs from final native identity")


def audit_run_identity(identity: dict, name: str, artifacts: dict, sources: dict, binaries: dict, raw_reference: str) -> None:
    require(identity.get("schema") == "NumiHuman.CVSimNativeExecution.v2" and identity.get("all_identities_unchanged") is True and
            identity.get("native_head") in (NATIVE_BASE, NATIVE_COMMIT), "native execution identity missing or stale")
    require(type(identity.get("exit_code")) is int and identity["exit_code"] == 0, "native execution did not finish successfully")
    matched_snapshots(identity.get("source_before_sha256"), identity.get("source_after_sha256"), sources, NATIVE_SOURCES, name + " source")
    matched_snapshots(identity.get("runtime_before_sha256"), identity.get("runtime_after_sha256"), binaries,
                      {"matter/numi-matter-cvsim-check", "matter/shaders/NumiMatter.metallib"}, name + " runtime")
    payload = "cvsim21-heldt.native.v3.json" if name.startswith("heldt-") else "cvsim21.native.v3.json"
    require(identity.get("executable_sha256") == binaries["matter/numi-matter-cvsim-check"] and
            identity.get("payload_sha256") == digest(artifacts[payload]) and identity.get("reference_sha256") == raw_reference,
            "run executable, payload or reference differs")
    require(all(identity.get(key + "_sha256") == identity.get(key + "_after_sha256")
                for key in ("executable", "payload", "reference")), "run input or executable changed during execution")
    libraries = identity.get("otool_L")
    require(isinstance(libraries, str) and "Foundation.framework" in libraries and "Metal.framework" in libraries and
            "libmetalrobo" not in libraries.lower(), "static Matter linkage attestation differs")
    require(identity.get("environment") == {"MTL_DEBUG_LAYER": "1"}, "native Metal validation environment differs")
    dt, steps = RUNS[name]
    command = identity.get("command")
    require(isinstance(command, list) and len(command) in (11, 13) and all(isinstance(s, str) for s in command), "missing exact native command")
    require(Path(command[0]).name == "numi-matter-cvsim-check" and Path(command[1]).name == payload and
            Path(command[2]).name == "continuous-1e-12.csv", "run command uses wrong executable or input")
    flags = command[3:]
    require(len(flags) % 2 == 0, "malformed native command flags")
    pairs = dict(zip(flags[::2], flags[1::2]))
    require(len(pairs) * 2 == len(flags) and set(pairs) in ({"--steps", "--dt", "--trace", "--checkpoint"},
                                                        {"--steps", "--dt", "--trace", "--checkpoint", "--volume-coordinates"}), "unexpected native command flags")
    require(pairs["--steps"] == str(steps) and float(pairs["--dt"]) == dt and Path(pairs["--trace"]).name == name + ".csv",
            "run command time or trace differs")
    require(pairs.get("--volume-coordinates", "upstream_equation") == ("heldt_table_aligned" if name.startswith("heldt-") else "upstream_equation"),
            "run command coordinate variant differs")
    records = identity.get("artifacts")
    require(isinstance(records, dict) and set(records) == {name + suffix for suffix in (".csv", ".csv.gz", ".log")},
            "run artifact bindings missing")
    audit_execution_artifacts(records, name, artifacts)
    raw = records[name + ".csv"]
    require(isinstance(raw, dict) and raw.get("sha256") == digest(artifacts[name + ".csv.gz"], decompress=True),
            "run raw trace binding differs")


def audit_execution_artifacts(records: dict, name: str, artifacts: dict, *, prefix: str = "") -> None:
    for suffix in (".log", ".csv.gz"):
        path, record = artifacts[prefix + name + suffix], records.get(name + suffix)
        require(isinstance(record, dict) and record.get("sha256") == digest(path) and
                type(record.get("bytes")) is int and record["bytes"] == path.stat().st_size,
                "run stored log/trace binding differs")


def load_tool(root: Path, relative: str, owners: dict):
    # Execute the verifier's local trusted sibling, not code supplied by a receipt
    # copy. Its owner hash must also equal the verified checkout's owner hash.
    path = safe_path(ROOT, relative)
    require(digest(path) == owners[relative], "loaded verifier helper differs from pinned owner")
    spec = importlib.util.spec_from_file_location("_cvsim_evidence_" + Path(relative).stem, path)
    require(spec is not None and spec.loader is not None, "cannot load evidence helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_source(root: Path, artifacts: dict, owners: dict) -> dict:
    def owner(name):
        return safe_path(root, REFERENCE_PREFIX + "/" + name)
    require(digest(safe_path(root, SOURCE_PREFIX + "/source-lock.json")) == SOURCE_LOCK_SHA256 and
            digest(owner("original-initial.json")) == INITIAL_SHA256 and digest(owner("execution.json")) == EXECUTION_SHA256,
            "source execution provenance changed")
    execution = read_json(owner("execution.json"))
    require(execution.get("status") == "pass" and execution.get("python_physical_stepping") is False and
            execution.get("biological_qualification") == "unqualified", "source execution boundary failed")
    require(execution["source_sha256"] == execution["source_after_sha256"] and
            execution["tool_sha256"] == execution["tool_after_sha256"], "CPU source/reference changed during execution")
    for name, sha in execution["artifact_sha256"].items():
        require(REFERENCE_PREFIX + "/" + name in owners and digest(owner(name)) == sha, "CPU artifact missing or changed")
    require(set(execution["traces"]) == SOURCE_TRACES, "original source drift or refinement trace omitted")
    for name, record in execution["traces"].items():
        require(digest(artifacts[name]) == record["sha256"] and digest(artifacts[name], decompress=True) == record["raw_sha256"],
                f"retained source trajectory differs: {name}")
    require(digest(owner("reference-audit.json")) == REFERENCE_AUDIT_SHA256, "source audit/drift record changed")
    audit = read_json(owner("reference-audit.json"))
    require(audit.get("status") == "pass" and audit.get("biological_qualification") == "unqualified" and
            audit.get("refinement_decreases_all_coordinate_groups") is True, "source reference audit did not pass")
    require(audit["original_source"]["samples"] == 100383 and
            audit["original_source"]["max_volume_sum_target_difference_mL"] > .5 and
            audit["source_clock_comparison"]["source_clock_equivalence_to_original_step_sim"] is False,
            "original C drift or clock discrepancy was concealed")
    return execution


def regenerate_payloads(root: Path, artifacts: dict, owners: dict) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from numilab_human import cvsim21, cvsim_parameters
    for module in (cvsim21, cvsim_parameters):
        relative = "src/numilab_human/" + Path(module.__file__).name
        require(digest(Path(module.__file__)) == owners[relative], "loaded authoring helper differs from owner")
    for config, stem in (("cvsim21-source.v1.json", "cvsim21"), ("cvsim21-heldt-table-aligned.v1.json", "cvsim21-heldt")):
        native, manifest = cvsim21.compile_source(directory=root / SOURCE_PREFIX,
            initial_path=root / REFERENCE_PREFIX / "original-initial.json", config=read_json(root / "config" / config))
        for name, result in ((stem + ".native.v3.json", native), (stem + ".native.v3.manifest.json", manifest)):
            require(artifacts[name].read_bytes() == cvsim21.canonical(result) + b"\n", "regenerated payload/manifest differs: " + name)


def audit_legacy_identity(identity: dict, name: str, dt: float, steps: int, artifacts: dict,
                          sources: dict, binaries: dict) -> dict:
    require(identity.get("schema") == "NumiHuman.LegacyCardiacExecution.v1" and identity.get("id") == name and
            identity.get("source_and_binary_identity_unchanged") is True and
            identity.get("exit_code") == 0 and type(identity.get("exit_code")) is int, "legacy run identity failed")
    before, after = identity.get("before"), identity.get("after")
    require(isinstance(before, dict) and isinstance(after, dict) and before.get("native_head") == after.get("native_head") and
            before.get("native_head") in (NATIVE_BASE, NATIVE_COMMIT), "legacy source head changed")
    matched_snapshots(before.get("sources"), after.get("sources"), sources,
                      {"matter/src/runtime.mm", "matter/src/metal/vascular.metalinc", "matter/tools/cardiac_check.mm",
                       "matter/tools/fixtures/shi-hose.native.v2.json", "matter/tools/shi_hose_reference/shi_hose_dopri.hpp",
                       "matter/tools/shi_hose_reference/shi_hose_observables.hpp",
                       "matter/tools/shi_hose_reference/shi_hose_reference_generated.hpp"}, name + " source")
    matched_snapshots(before.get("binaries"), after.get("binaries"), binaries,
                      {"matter/numi-matter-cardiac-check", "matter/shaders/NumiMatter.metallib"}, name + " runtime")
    command = identity.get("argv")
    require(isinstance(command, list) and len(command) == 14 and all(isinstance(s, str) for s in command),
            "legacy command missing or malformed")
    flags = dict(zip(command[2::2], command[3::2]))
    require(Path(command[0]).name == "numi-matter-cardiac-check" and Path(command[1]).name == "shi-hose.native.v2.json" and
            set(flags) == {"--steps", "--dt", "--trace", "--checkpoint", "--newton", "--krylov"} and
            flags["--steps"] == str(steps) and float(flags["--dt"]) == dt and flags["--newton"] == "12" and
            flags["--krylov"] == "64" and Path(flags["--trace"]).name == name + ".csv" and
            identity.get("environment") == {"MTL_DEBUG_LAYER": "1"}, "legacy command or validation environment differs")
    records = identity.get("artifacts")
    require(isinstance(records, dict) and set(records) == {name + suffix for suffix in (".csv.gz", ".log")},
            "legacy artifact bindings missing")
    audit_execution_artifacts(records, name, artifacts, prefix="legacy-cardiac-requalification/")
    require(identity.get("trace_raw_sha256") == digest(artifacts["legacy-cardiac-requalification/" + name + ".csv.gz"], decompress=True),
            "legacy raw trace binding differs")
    return before


def audit_legacy(root: Path, artifacts: dict, owners: dict, sources: dict, binaries: dict) -> dict:
    require(digest(root / HISTORICAL_RECEIPT) == HISTORICAL_RECEIPT_SHA256, "historical cardiac receipt was changed")
    require(digest(root / "tools/verify_cardiac_source_20260912.py") == LEGACY_VERIFIER_SHA256, "frozen legacy numerical verifier changed")
    legacy = load_tool(root, "tools/verify_cardiac_source_20260912.py", owners)
    summary = read_json(artifacts["legacy-cardiac-requalification.json"])
    require(summary.get("schema") == "NumiHuman.LegacyCardiacRequalification.v1" and summary.get("status") == "pass" and
            summary.get("native_abi") == 28 and summary.get("frozen_verifier_sha256") == LEGACY_VERIFIER_SHA256 and
            summary.get("identity_before_after_equal") is True and summary.get("old_receipt_unchanged") is True and
            summary.get("biological_calibration") == "unqualified", "legacy requalification boundary failed")
    require(summary.get("accuracy_limits") == legacy.ACCURACY_LIMITS and summary.get("refinement_minimum_ratio") == 1.5,
            "legacy numerical gates were changed")
    payload_path = root / "Docs/media/cardiac-source-20260912/shi-hose.native.v2.json"
    require(digest(payload_path) == legacy.SOURCE_INPUT_SHA256, "legacy source input changed")
    payload = read_json(payload_path)
    scales, initial = legacy.source_coordinates(payload)
    runs, identities = {}, {}
    for name in sorted(LEGACY_RUNS):
        prefix = "legacy-cardiac-requalification/" + name
        raw_log = artifacts[prefix + ".log"].read_text()
        require(len(re.findall(r"^cardiac_device=Apple M4 Pro abi=28 world_fingerprint=[0-9]+$", raw_log, re.MULTILINE)) == 1,
                "legacy cohort lacks current physical ABI28 identity")
        # The historical verifier remains byte-identical. Only its old ABI
        # attestation is adapted in memory; every numeric gate is unchanged.
        audit_log = re.sub(r"^(cardiac_device=Apple M4 Pro abi=)28( world_fingerprint=)", r"\g<1>27\2", raw_log, flags=re.MULTILINE)
        dt, steps = legacy.RUNS[name]
        runs[name] = legacy.audit_native_run(artifacts[prefix + ".csv.gz"], audit_log,
                         {"id": name, "dt_seconds": dt, "steps": steps}, scales, initial, payload)
        identity = read_json(artifacts[prefix + ".execution.json"])
        identities[name] = audit_legacy_identity(identity, name, dt, steps, artifacts, sources, binaries)
    ratios = legacy.audit_native_refinement(runs)
    require(summary.get("runs") == runs and summary.get("refinement_error_ratios") == ratios, "legacy report differs from retained trajectory")
    require(summary.get("identities") == identities, "legacy summary source identity differs")
    return {"runs": runs, "refinement_error_ratios": ratios, "historical_receipt_unchanged": True}


def verify(*, root: Path = ROOT, receipt_path: Path | None = None) -> dict:
    root = Path(root).resolve()
    receipt_path = Path(receipt_path) if receipt_path is not None else root / "Docs/media/cvsim21-circulation-20260912/receipt.json"
    require(receipt_path.is_absolute() and receipt_path.is_relative_to(root), "receipt must be within supplied Human root")
    receipt_path = safe_path(root, receipt_path.relative_to(root).as_posix())
    receipt = read_json(receipt_path)
    audit_scope(receipt)
    owners = hash_map(receipt.get("owners"), "owner")
    checked_inventory(root, owners, REQUIRED_OWNERS, "owner")
    artifacts = checked_inventory(receipt_path.parent, receipt.get("artifacts"), REQUIRED_ARTIFACTS, "artifact")
    for name, expected in FAILURE_HASHES.items():
        require(digest(artifacts[name]) == expected, "known failure history removed or rewritten")
    sources, binaries = audit_native_identity(read_json(artifacts["native-identity.json"]), receipt.get("native_commit"))
    audit_ctest(artifacts["native-ctest.txt"].read_text())
    execution = audit_source(root, artifacts, owners)
    regenerate_payloads(root, artifacts, owners)
    raw_reference = execution["traces"]["continuous-1e-12.csv.gz"]["raw_sha256"]
    for name in RUNS:
        require(not (receipt_path.parent / (name + ".csv")).exists(), "unlisted raw CSV would shadow retained compressed evidence")
        audit_run_identity(read_json(artifacts[name + ".identity.json"]), name, artifacts, sources, binaries, raw_reference)
    analyzer = load_tool(root, "tools/analyze_cvsim_native.py", owners)
    analyzer.SOURCE = root / SOURCE_PREFIX
    analyzer.SOURCE_EVIDENCE = root / REFERENCE_PREFIX
    measured = analyzer.analyze_cohort(receipt_path.parent, artifacts["continuous-1e-12.csv.gz"], require_aligned=True)
    require(measured.get("status") == "pass" and measured.get("passed") is True, f"native numerical cohort failed: {measured.get('errors')}")
    require(read_json(artifacts["native-comparison.json"]) == measured, "numerical report differs from deterministic recomputation")
    legacy = audit_legacy(root, artifacts, owners, sources, binaries)
    return {"schema": "numi.human.cvsim21-verification.v1", "status": "pass", "native_commit": NATIVE_COMMIT,
            "qualification": receipt["qualification"], "scope": SCOPE, "ctest_passed": 15,
            "native_runs": sorted(RUNS), "refinement": measured["refinement"],
            "legacy_refinement": legacy["refinement_error_ratios"], "retained_failures": sorted(FAILURE_HASHES),
            "source_drift_retained": True, "historical_cardiac_receipt_unchanged": True,
            "python_physical_stepping": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify(root=args.root, receipt_path=args.receipt.resolve() if args.receipt else None)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"status": "fail", "errors": [str(error)]}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
