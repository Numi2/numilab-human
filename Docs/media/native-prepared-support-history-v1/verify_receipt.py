#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_BRANCH = "human-native-runtime-20260915"
IMPLEMENTATION_COMMIT = "c785abae322a1604f09f4d509f5d2af51ac42e5e"
QUALIFIED_COMMIT = "3b968495e05253cb4675893c3b94fec448c63361"
QUALIFIED_TREE = "ee68992948f3b884571f380498f8e9af5d78edb1"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1
EXPECTED_MANIFEST_ROOT = "1734c417f2dad58e334ba0225b682720b7306bb49a68eaeefe934e19b5fed291"
EXPECTED_INPUTS = {
    "rigid": ("inputs/myosim-fullbody-core-reference.nhrigid", 60324, "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44"),
    "muscle": ("inputs/myosim-fullbody-muscle-reference.nhmyo", 149372, "9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05"),
    "support": ("inputs/myosim-fullbody-support-primitives.nhcnt", 1044, "c7712daf79cd8a589a6d23942a4df84a7da928e5911455ce19078f9b24daaaf4"),
    "alternate_valid_support": ("inputs/alternate-valid-support-plantar-ellipsoid.nhcnt", 1044, "5ebe9d4a7a830a9206f48edfe78cf4c989258e9ad91b46caa74826392ee5f915"),
    "joint_equalities": ("inputs/myosim-fullbody-joint-equalities-source-compliance.nheq", 5792, "12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477"),
    "joint_limits": ("inputs/myosim-fullbody-joint-limits.nhlim", 9840, "c583611fcedc326a32c6f69504a65e675c8e0adc987c6db622ca0d95a02438d3"),
    "legacy_prepared_state": ("inputs/prepared.nhinit", 7780, "e7597abe656c5a4b9231f087edd19e6937e123bbb4583f6421f60da625f63238"),
}
EXPECTED_ARTIFACTS = {
    "matter_physics_probe": ("artifacts/bin/metalrobo_matter_physics_probe", 505568, "bbd86c0aefeffdcebbc049c74ea7b423381b1d0e44c9f48f6fb8be24f7d56f3e"),
    "numanx_fullbody_bridge_probe": ("artifacts/bin/metalrobo_numanx_fullbody_bridge_probe", 246056, "1f6196839f17d0ee93a384a9ec10ae5cbedab58ce36e945e4c5443f86f1d99e2"),
    "myosim_visual_probe": ("artifacts/bin/metalrobo_numilab_human_myosim_visual_probe", 991064, "4319694915a4c789252473520b6cfe76db630164fa3520835cded467b79d4d8b"),
    "numi_matter_metallib": ("artifacts/matter/shaders/NumiMatter.metallib", 2823560, "6397674c0463683e182c41dc110944847c82a0832848ead4310bbe3e305033fc"),
    "metalrobo_runtime_library": ("artifacts/lib/libmetalrobo.dylib", 6726608, "d9c2e192db8f42e8906dc7ce3c405fb7c0f6a4722646b573ee172e6cb7537de5"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fnv1a(data: bytes, seed: int = FNV_OFFSET) -> int:
    value = seed
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & U64_MASK
    return value


def fullbody_source_fingerprint(rigid: bytes, muscle: bytes, support: bytes) -> int:
    value = fnv1a(b"mrnx.fullbody.source.v1")
    for payload in (rigid, muscle, support):
        value = fnv1a(struct.pack("<Q", len(payload)), value)
        value = fnv1a(payload, value)
    return value or FNV_OFFSET


def constrained_source_fingerprint(base: int, equality: bytes, limits: bytes) -> int:
    value = ((base ^ fnv1a(b"NHEQ2")) * FNV_PRIME) & U64_MASK
    value = ((value ^ fnv1a(equality)) * FNV_PRIME) & U64_MASK
    value = ((value ^ fnv1a(b"NHLIM1")) * FNV_PRIME) & U64_MASK
    value = ((value ^ fnv1a(limits)) * FNV_PRIME) & U64_MASK
    return value or FNV_OFFSET


def read_status(name: str) -> int:
    return int((ROOT / name).read_text(encoding="utf-8").strip())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def parse_key_value_file(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        require(bool(separator) and key not in values, f"malformed or duplicate entry: {name}: {line}")
        values[key] = value
    return values


def finite_csv(text: str, count: int, label: str) -> tuple[float, ...]:
    fields = text.split(",")
    require(len(fields) == count, f"{label} dimension mismatch")
    values = tuple(float(field) for field in fields)
    require(all(math.isfinite(value) for value in values), f"{label} contains nonfinite evidence")
    return values


receipt = json.loads((ROOT / "receipt-v1.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "build-run-manifest.json").read_text(encoding="utf-8"))

require(receipt["schema"] == "numi.human.prepared-support-history-receipt.v1", "receipt schema mismatch")
require(receipt["status"] == "partial", "receipt must remain partial")
require(manifest["schema"] == "numi.human.prepared-support-history-build-run.v1", "build/run schema mismatch")
require(
    receipt["scope"]
    == "Fixed-topology raw-NHCNT-bound NHINIT3 serialization/provenance and analytic Matter support-history transaction verification on one physical Apple M4 Pro.",
    "receipt scope mismatch",
)

expected_source = {
    "branch": SOURCE_BRANCH,
    "implementation_commit": IMPLEMENTATION_COMMIT,
    "qualified_head_commit": QUALIFIED_COMMIT,
    "qualified_head_tree": QUALIFIED_TREE,
}
for key, expected in expected_source.items():
    require(receipt["source"][key] == expected, f"receipt source {key} mismatch")
require(receipt["source"]["repository"] == "Numi2/numi-lab", "receipt source repository mismatch")
require(receipt["source"]["worktree_clean"] is True, "qualified source worktree was not clean")
require(receipt["source"]["build_run_manifest"] == "build-run-manifest.json", "build/run manifest path mismatch")
require(receipt["source"]["environment_evidence"] == "runtime-environment.txt", "environment evidence path mismatch")
require(receipt["source"]["git_identity_evidence"] == "source-git.txt", "git identity evidence path mismatch")

source_git = parse_key_value_file("source-git.txt")
require(
    source_git
    == {
        "head": QUALIFIED_COMMIT,
        "tree": QUALIFIED_TREE,
        "status_porcelain": "clean",
        "remote_contains": f"origin/{SOURCE_BRANCH},",
    },
    "source git identity mismatch",
)
require(
    (ROOT / "matter-physical/native-commit.txt").read_text(encoding="utf-8").strip() == QUALIFIED_COMMIT,
    "physical Matter native commit mismatch",
)
require(
    manifest["source"]
    == {
        "repository": "https://github.com/Numi2/numi-lab.git",
        "branch": SOURCE_BRANCH,
        "head": QUALIFIED_COMMIT,
        "tree": QUALIFIED_TREE,
        "identity_evidence": "source-git.txt",
    },
    "build/run source identity mismatch",
)
expected_machine = {
    "name": "Mac mini",
    "model": "Mac16,11",
    "architecture": "arm64",
    "chip": "Apple M4 Pro",
    "memory_bytes": 25769803776,
    "macos_version": "26.6",
    "macos_build": "25G72",
    "xcode_version": "26.6",
    "xcode_build": "17F113",
}
require(receipt["source"]["machine"] == expected_machine, "receipt machine identity mismatch")
require(
    manifest["host"] == {"machine" if key == "name" else key: value for key, value in expected_machine.items()},
    "build/run host identity mismatch",
)

environment_names = [
    "DEVELOPER_DIR",
    "DYLD_LIBRARY_PATH",
    "METAL_DEVICE_WRAPPER_TYPE",
    "MTL_CAPTURE_ENABLED",
    "MTL_DEBUG_LAYER",
    "MTL_SHADER_VALIDATION",
    "MRNX_INCLUDE_SYNTHETIC_VASCULAR",
]
runtime_environment = parse_key_value_file("runtime-environment.txt")
require(runtime_environment == {name: "<unset>" for name in environment_names}, "runtime environment evidence mismatch")
require(
    manifest["environment"]
    == {
        "evidence_path": "runtime-environment.txt",
        "allowlisted_relevant_variables": environment_names,
        "all_allowlisted_variables_unset": True,
    },
    "build/run environment declaration mismatch",
)

expected_build = {
    "configuration": "Release",
    "generator": "Ninja",
    "cmake_version": "4.4",
    "compiler": "/usr/bin/clang++",
    "objective_cxx_flags": [
        "-O3", "-DNDEBUG", "-std=c++23", "-arch", "arm64", "-fPIE", "-fobjc-arc",
        "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    ],
    "final_executed_command": [
        "/opt/homebrew/bin/cmake", "--build", "/Users/n/Developer/numi-human-nhinit3-c785aba/build",
        "--target", "metalrobo_numanx_fullbody_bridge_probe", "-j", "8",
    ],
    "reproduction_targets": [
        "metalrobo_matter_physics_probe",
        "metalrobo_numanx_fullbody_bridge_probe",
        "metalrobo_numilab_human_myosim_visual_probe",
    ],
}
for key, expected in expected_build.items():
    require(manifest["build"][key] == expected, f"build declaration mismatch: {key}")

manifest_inputs = manifest["build"]["fullbody_compile_time_inputs"]
for variable, receipt_name in {"MRNX_FULLBODY_RIGID": "rigid", "MRNX_FULLBODY_MUSCLE": "muscle"}.items():
    require(
        manifest_inputs[variable]
        == {
            "evidence_path": receipt["inputs"][receipt_name]["path"],
            "sha256": receipt["inputs"][receipt_name]["sha256"],
        },
        f"compile-time input mismatch: {variable}",
    )
require(
    manifest_inputs["MRNX_MATTER_METALLIB"]
    == {
        "evidence_path": manifest["artifacts"]["numi_matter_metallib"]["path"],
        "sha256": receipt["source"]["binaries"]["matter_metallib_sha256"],
    },
    "compile-time Matter metallib mismatch",
)

expected_commands = {
    "source_certificate": [
        "artifacts/bin/metalrobo_numilab_human_myosim_visual_probe", "--source-compliant-certificate",
        "inputs/myosim-fullbody-core-reference.nhrigid", "inputs/myosim-fullbody-muscle-reference.nhmyo",
        "inputs/myosim-fullbody-support-primitives.nhcnt", "inputs/prepared.nhinit",
        "inputs/myosim-fullbody-joint-equalities-source-compliance.nheq",
        "inputs/myosim-fullbody-joint-limits.nhlim", "32", "--support-reactions-only",
    ],
    "author": [
        "artifacts/bin/metalrobo_numanx_fullbody_bridge_probe", "--prepared-stance-fixture",
        "certificate.stdout.txt", "generated/positive", "inputs/myosim-fullbody-support-primitives.nhcnt",
        "inputs/myosim-fullbody-joint-equalities-source-compliance.nheq",
        "inputs/myosim-fullbody-joint-limits.nhlim", "100", "16",
    ],
    "roundtrip": [
        "artifacts/bin/metalrobo_numanx_fullbody_bridge_probe", "--prepared-state-fixture",
        "generated/positive/prepared.nhinit", "generated/roundtrip",
        "inputs/myosim-fullbody-support-primitives.nhcnt",
        "inputs/myosim-fullbody-joint-equalities-source-compliance.nheq",
        "inputs/myosim-fullbody-joint-limits.nhlim", "100", "16",
    ],
    "provenance_negative": [
        "artifacts/bin/metalrobo_numanx_fullbody_bridge_probe", "--prepared-stance-fixture",
        "certificate.stdout.txt", "generated/negative-provenance",
        "inputs/alternate-valid-support-plantar-ellipsoid.nhcnt",
        "inputs/myosim-fullbody-joint-equalities-source-compliance.nheq",
        "inputs/myosim-fullbody-joint-limits.nhlim", "100", "16",
    ],
    "physical_matter": ["artifacts/bin/metalrobo_matter_physics_probe", "--human-support-loaded"],
    "focused_ctest": [
        "/opt/homebrew/bin/ctest", "--test-dir", "/Users/n/Developer/numi-human-nhinit3-c785aba/build",
        "-R", "^(numi_human_initial_state|numi_human_static_support)$", "--output-on-failure",
    ],
}
require(manifest["commands"] == expected_commands, "normalized build/run commands mismatch")

sha_entries: dict[str, str] = {}
require(not (ROOT / "SHA256SUMS").is_symlink(), "symlinked SHA-256 manifest rejected")
for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    expected, separator, relative = line.partition("  ")
    require(bool(separator) and relative not in sha_entries, f"malformed or duplicate SHA entry: {line}")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe SHA path: {relative}")
    sha_entries[relative] = expected
actual_files = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
}
require(set(sha_entries) == actual_files, "SHA-256 manifest does not enumerate the exact evidence bundle")
require(len(sha_entries) == 48 and "verify_receipt.py" in sha_entries, "evidence bundle inventory mismatch")
manifest_root_input = "".join(
    f"{sha_entries[relative]}  {relative}\n"
    for relative in sorted(set(sha_entries) - {"verify_receipt.py"})
).encode("utf-8")
require(
    hashlib.sha256(manifest_root_input).hexdigest() == EXPECTED_MANIFEST_ROOT,
    "evidence bundle identity root mismatch",
)
for relative, expected in sha_entries.items():
    path = ROOT / relative
    require(not path.is_symlink(), f"symlinked evidence artifact rejected: {relative}")
    require(digest(path) == expected, f"SHA-256 mismatch: {relative}")

require(set(receipt["inputs"]) == set(EXPECTED_INPUTS), "input receipt field set mismatch")
for name, record in receipt["inputs"].items():
    expected_path, expected_bytes, expected_sha = EXPECTED_INPUTS[name]
    require(record["path"] == expected_path, f"input receipt path mismatch: {name}")
    require(record["bytes"] == expected_bytes, f"input receipt extent mismatch: {name}")
    require(record["sha256"] == expected_sha, f"input receipt SHA mismatch: {name}")
    path = ROOT / record["path"]
    require(path.is_file(), f"missing input artifact: {name}")
    require(path.stat().st_size == record["bytes"], f"input extent mismatch: {name}")
    require(digest(path) == record["sha256"], f"input identity mismatch: {name}")
    if "magic" in record:
        require(path.read_bytes().startswith(record["magic"].encode("ascii")), f"input magic mismatch: {name}")

artifact_receipt_keys = {
    "matter_physics_probe": "matter_physics_probe_sha256",
    "numanx_fullbody_bridge_probe": "fullbody_author_sha256",
    "myosim_visual_probe": "source_certificate_sha256",
    "numi_matter_metallib": "matter_metallib_sha256",
    "metalrobo_runtime_library": "metalrobo_runtime_library_sha256",
}
require(set(manifest["artifacts"]) == set(EXPECTED_ARTIFACTS), "executable artifact field set mismatch")
for name, record in manifest["artifacts"].items():
    expected_path, expected_bytes, expected_sha = EXPECTED_ARTIFACTS[name]
    require(
        record == {"path": expected_path, "bytes": expected_bytes, "sha256": expected_sha},
        f"build/run executable declaration mismatch: {name}",
    )
    path = ROOT / record["path"]
    require(path.is_file(), f"missing executable artifact: {name}")
    require(path.stat().st_size == record["bytes"], f"executable extent mismatch: {name}")
    require(digest(path) == record["sha256"], f"executable identity mismatch: {name}")
    require(
        record["sha256"] == receipt["source"]["binaries"][artifact_receipt_keys[name]],
        f"receipt executable identity mismatch: {name}",
    )

support_path = ROOT / receipt["inputs"]["support"]["path"]
support_bytes = support_path.read_bytes()
require(len(support_bytes) == 1044 and support_bytes[:8] == b"NHCNT2\0\0", "NHCNT2 envelope mismatch")
support_abi, body_count, source_count, reserved = struct.unpack_from("<4I", support_bytes, 8)
require((support_abi, body_count, source_count, reserved) == (2, 157, 10, 0), "NHCNT2 header mismatch")

rigid_bytes = (ROOT / receipt["inputs"]["rigid"]["path"]).read_bytes()
muscle_source_bytes = (ROOT / receipt["inputs"]["muscle"]["path"]).read_bytes()
equality_bytes = (ROOT / receipt["inputs"]["joint_equalities"]["path"]).read_bytes()
limit_bytes = (ROOT / receipt["inputs"]["joint_limits"]["path"]).read_bytes()
base_source_fp = fullbody_source_fingerprint(rigid_bytes, muscle_source_bytes, support_bytes)
composed_source_fp = constrained_source_fingerprint(base_source_fp, equality_bytes, limit_bytes)

certificate_lines = (ROOT / "certificate.stdout.txt").read_text(encoding="utf-8").splitlines()
expected_prefixes = (
    "source_compliant_equilibrium=",
    "compiled_equilibrium_q=",
    "compiled_equilibrium_muscles=",
)
require(len(certificate_lines) == len(expected_prefixes), "source certificate output line count mismatch")
certificate_records: dict[str, object] = {}
for line, prefix in zip(certificate_lines, expected_prefixes, strict=True):
    require(line.startswith(prefix), f"source certificate line prefix mismatch: {prefix}")
    certificate_records[prefix[:-1]] = json.loads(line[len(prefix) :])
certificate = certificate_records["source_compliant_equilibrium"]
require(certificate["schema"] == "numi.human.source-compliant-equilibrium.v1", "source certificate schema mismatch")
require(certificate["balanced"] is True, "source certificate did not balance")
require(certificate["support_sha256"] == receipt["nhinit3"]["support_sha256"], "certificate support SHA mismatch")
require(
    (certificate["support_bytes"], certificate["support_abi"], certificate["support_source_records"], certificate["support_expanded_rows"])
    == (1044, 2, 10, 18),
    "certificate support identity mismatch",
)
expected_certificate_summary = {
    "status": "passed",
    "balanced": True,
    "accepted_iterations": 3,
    "rejected_evaluations": 88,
    "initial_acceleration_rms": 184.89466180689243,
    "terminal_acceleration_rms": 0.0013903104493953569,
    "maximum_acceleration": 0.008042384484473962,
    "maximum_acceleration_dof": 125,
    "maximum_force_residual": 0.00030487972614373637,
    "minimum_support_gap_m": -2.811430141469451e-08,
    "maximum_loaded_support_gap_m": 2.811175641576069e-08,
    "support_identity_exact": True,
    "mode": "support-reactions-only",
}
require(receipt["source_certificate"] == expected_certificate_summary, "source certificate outcome mismatch")
certificate_receipt_fields = {
    "balanced": "balanced",
    "iterations": "accepted_iterations",
    "rejected": "rejected_evaluations",
    "initial_acceleration_rms": "initial_acceleration_rms",
    "acceleration_rms": "terminal_acceleration_rms",
    "maximum_acceleration": "maximum_acceleration",
    "maximum_acceleration_dof": "maximum_acceleration_dof",
    "maximum_force_residual": "maximum_force_residual",
    "minimum_support_gap": "minimum_support_gap_m",
    "maximum_loaded_support_gap": "maximum_loaded_support_gap_m",
}
for certificate_name, receipt_name in certificate_receipt_fields.items():
    require(
        certificate[certificate_name] == receipt["source_certificate"][receipt_name],
        f"source certificate receipt mismatch: {certificate_name}",
    )
require(receipt["source_certificate"]["support_identity_exact"] is True, "support identity was not exact")
require(receipt["source_certificate"]["mode"] == "support-reactions-only", "source certificate mode mismatch")
require(certificate["maximum_acceleration"] <= 0.01, "source certificate acceleration threshold failed")
require(certificate["maximum_force_residual"] <= 0.001, "source certificate force threshold failed")

nhinit_path = ROOT / receipt["nhinit3"]["path"]
expected_nhinit_receipt = {
    "path": "generated/positive/prepared.nhinit",
    "magic": "NHINIT3",
    "bytes": 8196,
    "header_bytes": 224,
    "history_record_bytes": 16,
    "history_encoding": 1,
    "history_rows": 18,
    "support_sha256": "c7712daf79cd8a589a6d23942a4df84a7da928e5911455ce19078f9b24daaaf4",
    "sha256": "b4908f21aee3434091b4c9935fadbe276be30b142114a3dd51c1eb548ad28dad",
    "matter_package_sha256": "b830c33de03b6b746876aa6f7a4eab8bb9e64fe6acba3e5768b5f9ce59c2a5b7",
    "roundtrip_byte_exact": True,
    "roundtrip_package_byte_exact": True,
}
require(receipt["nhinit3"] == expected_nhinit_receipt, "NHINIT3 receipt metadata mismatch")
nhinit_bytes = nhinit_path.read_bytes()
require(len(nhinit_bytes) == 8196 and nhinit_bytes[:8] == b"NHINIT3\0", "NHINIT3 envelope mismatch")
header = struct.unpack_from("<8I", nhinit_bytes, 8)
require(header == (3, 224, 129, 128, 416, 4, 2, 0), "NHINIT3 dimensions or flags mismatch")
microstep_us = struct.unpack_from("<Q", nhinit_bytes, 56)[0]
microstep_ns = struct.unpack_from("<Q", nhinit_bytes, 96)[0]
require((microstep_us, microstep_ns) == (100, 100000), "NHINIT3 exact clock mismatch")
embedded_source_fp, embedded_world_fp = struct.unpack_from("<2Q", nhinit_bytes, 40)
require(embedded_source_fp == composed_source_fp, "NHINIT3 composed source fingerprint mismatch")
require(nhinit_bytes[64:96] == rigid_bytes[48:80], "NHINIT3 source archive SHA mismatch")
require(nhinit_bytes[160:192].hex() == receipt["nhinit3"]["support_sha256"], "NHINIT3 raw support SHA mismatch")
support_bytes_count, support_abi, source_rows, expanded_rows, record_bytes, encoding, reserved = struct.unpack_from(
    "<Q6I", nhinit_bytes, 192
)
require(
    (support_bytes_count, support_abi, source_rows, expanded_rows, record_bytes, encoding, reserved)
    == (1044, 2, 10, 18, 16, 1, 0),
    "NHINIT3 support envelope mismatch",
)
require(digest(nhinit_path) == receipt["nhinit3"]["sha256"], "NHINIT3 receipt identity mismatch")
require(
    receipt["nhinit3"]["sha256"] == "b4908f21aee3434091b4c9935fadbe276be30b142114a3dd51c1eb548ad28dad",
    "qualified NHINIT3 identity mismatch",
)

prepared_json = json.loads((ROOT / "generated/positive/prepared.json").read_text(encoding="utf-8"))
require(
    prepared_json
    == {
        "schema": "numi.human.prepared-stance-fixture.v2",
        "human_source_fp": f"{base_source_fp:x}",
        "composed_human_source_fp": f"{composed_source_fp:x}",
        "world_fp": f"{embedded_world_fp:x}",
        "initial_state_fp": f"{fnv1a(nhinit_bytes):x}",
        "support_sha256": receipt["nhinit3"]["support_sha256"],
        "support_bytes": 1044,
        "support_abi": 2,
        "support_source_records": 10,
        "support_expanded_rows": 18,
        "scope": "three tiny pelvis samples; no anatomical tissue or sustained behavior qualification",
    },
    "prepared-state JSON identity mismatch",
)

q_values = certificate_records["compiled_equilibrium_q"]
muscle_values = certificate_records["compiled_equilibrium_muscles"]
require(isinstance(q_values, list) and len(q_values) == 129, "compiled q certificate dimensions mismatch")
q_offset = 224
q_bytes = b"".join(struct.pack("<f", value) for value in q_values)
require(nhinit_bytes[q_offset : q_offset + len(q_bytes)] == q_bytes, "NHINIT3 q differs from certificate")
v_offset = q_offset + 4 * 129
require(nhinit_bytes[v_offset : v_offset + 4 * 128] == b"\0" * (4 * 128), "NHINIT3 v was not zero")
require(
    isinstance(muscle_values, dict)
    and muscle_values.get("schema") == "numi.human.offline-muscle-state.v1",
    "compiled muscle certificate schema mismatch",
)
activations = muscle_values.get("activation_fp32")
fiber_lengths = muscle_values.get("reference_fiber_length_m")
require(
    isinstance(activations, list) and isinstance(fiber_lengths, list)
    and len(activations) == len(fiber_lengths) == 416,
    "compiled muscle certificate dimensions mismatch",
)
muscle_bytes = b"".join(
    struct.pack("<4f", activation, activation, fiber_length, 0.0)
    for activation, fiber_length in zip(activations, fiber_lengths, strict=True)
)
muscle_offset = v_offset + 4 * 128
require(
    nhinit_bytes[muscle_offset : muscle_offset + len(muscle_bytes)] == muscle_bytes,
    "NHINIT3 muscle state differs from certificate",
)

history_offset = 224 + 4 * 129 + 4 * 128 + 16 * 416
support_forces = certificate["support_normal_force"]
require(len(support_forces) == 18, "source certificate support row count mismatch")
for index, force in enumerate(support_forces):
    row = nhinit_bytes[history_offset + 16 * index : history_offset + 16 * (index + 1)]
    tangent_x, tangent_y, tangent_z, normal_impulse = struct.unpack("<4f", row)
    require(row[:12] == b"\0" * 12, f"NHINIT3 tangent history was nonzero at row {index}")
    require(math.isfinite(normal_impulse) and normal_impulse >= 0.0, f"NHINIT3 normal history invalid at row {index}")
    require((tangent_x, tangent_y, tangent_z) == (0.0, 0.0, 0.0), f"NHINIT3 tangent decode mismatch at row {index}")
    require(
        row[12:] == struct.pack("<f", force * microstep_ns * 1e-9),
        f"NHINIT3 force-to-impulse conversion mismatch at row {index}",
    )

positive = ROOT / "generated/positive"
roundtrip = ROOT / "generated/roundtrip"
for name in ("prepared.nhinit", "prepared-100us.nmatterpack", "prepared.json"):
    require((positive / name).read_bytes() == (roundtrip / name).read_bytes(), f"round-trip mismatch: {name}")
require(digest(positive / "prepared-100us.nmatterpack") == receipt["nhinit3"]["matter_package_sha256"], "Matter package identity mismatch")
require(
    receipt["nhinit3"]["matter_package_sha256"] == "b830c33de03b6b746876aa6f7a4eab8bb9e64fe6acba3e5768b5f9ce59c2a5b7",
    "qualified Matter package identity mismatch",
)
require(receipt["nhinit3"]["roundtrip_byte_exact"] is True, "NHINIT3 round-trip qualification mismatch")
require(receipt["nhinit3"]["roundtrip_package_byte_exact"] is True, "Matter package round-trip qualification mismatch")

for name in (
    "certificate.exit-status.txt", "final-author.exit-status.txt", "final-roundtrip.exit-status.txt",
    "final-ctest.exit-status.txt", "matter-physical/exit-status.txt",
):
    require(read_status(name) == 0, f"unexpected nonzero status: {name}")
require(read_status("final-provenance-negative.exit-status.txt") == 1, "provenance negative did not fail")
require(
    (ROOT / "final-provenance-negative.stderr.txt").read_text(encoding="utf-8").strip()
    == "source-compliant support identity disagrees with supplied NHCNT",
    "provenance negative diagnostic mismatch",
)
ctest = (ROOT / "final-ctest.stdout.txt").read_text(encoding="utf-8")
require("100% tests passed out of 2" in ctest, "focused CTest summary mismatch")
require("numi_human_initial_state" in ctest and "numi_human_static_support" in ctest, "focused CTest names missing")

matter = (ROOT / "matter-physical/stdout.txt").read_text(encoding="utf-8")
require("Human support loaded runtime: 11 cases x 3 environments passed; replay exact" in matter, "physical Matter banner missing")
runs = re.findall(
    r"^SUPPORT_RUN name=([^ ]+) success=(\d+) status=(\d+) diagnostic=([^ ]+) message=(.*)$",
    matter,
    re.MULTILINE,
)
expected_cases = [
    "cold_97kg",
    "weight_seed_97kg",
    "double_seed_97kg",
    "redundant_six_cold",
    "redundant_six_double",
    "cold_1kg",
    "half_timestep",
    "airborne_double_seed",
    "sticking",
    "sliding",
    "sliding_warm",
]
require(receipt["physical_matter_transaction"]["cases"] == expected_cases, "physical Matter receipt cases mismatch")
require([run[0] for run in runs] == expected_cases, "physical Matter case order mismatch")
require(all(run[1:] == ("1", "0", "1,0,0,0", "") for run in runs), "physical Matter transaction mismatch")
geometry_rows = re.findall(r"^SUPPORT_GEOMETRY env=(\d+) point=([^ ]+) gap=([^ ]+) impulse=([^\n]+)$", matter, re.MULTILINE)
require(len(geometry_rows) == 3 * len(expected_cases), "physical Matter geometry row count mismatch")
require(
    [int(row[0]) for row in geometry_rows] == [0, 1, 2] * len(expected_cases),
    "physical Matter geometry environment order mismatch",
)
require(
    all(
        finite_csv(row[1], 3, "physical Matter geometry point")
        and math.isfinite(float(row[2]))
        and math.isfinite(float(row[3]))
        for row in geometry_rows
    ),
    "physical Matter geometry contains nonfinite evidence",
)
load_rows = re.findall(
    r"^SUPPORT_LOAD name=([^ ]+) environment=(\d+) lambda=([^ ]+) expected=([^ ]+) tangent=([^ ]+) velocity=([^ ]+) gap=([^\n]+)$",
    matter,
    re.MULTILINE,
)
require(len(load_rows) == 3 * len(expected_cases), "physical Matter load row count mismatch")
require(
    [(row[0], int(row[1])) for row in load_rows]
    == [(case, environment) for case in expected_cases for environment in (0, 1, 2)],
    "physical Matter load case/environment order mismatch",
)
require(
    all(
        math.isfinite(float(row[2]))
        and math.isfinite(float(row[3]))
        and abs(float(row[2]) - float(row[3])) <= 1.0e-6
        and finite_csv(row[4], 2, "physical Matter tangent")
        and finite_csv(row[5], 3, "physical Matter velocity")
        and math.isfinite(float(row[6]))
        for row in load_rows
    ),
    "physical Matter load contains nonfinite scalar evidence",
)
replays = re.findall(r"^SUPPORT_REPLAY name=([^ ]+) q_error=([^ ]+) v_error=([^ ]+) normal_error=([^\n]+)$", matter, re.MULTILINE)
require([replay[0] for replay in replays] == expected_cases, "physical Matter replay order mismatch")
require(all(replay[1:] == ("0", "0", "0") for replay in replays), "physical Matter replay was not exact")
for control in (
    "support_restore_extent_admissibility_and_atomicity_rejected=1",
    "support_initial_bodies_missing_and_short_rejected=1",
    "support_initial_history_bad_count_cone_and_overflow_rejected=1",
    "SUPPORT_CAPACITY_REJECTION message=failed to encode MetalWorld substep graph",
):
    require(control in matter, f"physical Matter negative control missing: {control}")
binary_shader_hashes = [
    line.split()[0]
    for line in (ROOT / "matter-physical/binary-shaders.sha256").read_text(encoding="utf-8").splitlines()
]
require(
    binary_shader_hashes
    == [
        manifest["artifacts"]["matter_physics_probe"]["sha256"],
        manifest["artifacts"]["numi_matter_metallib"]["sha256"],
    ],
    "physical Matter binary/shader identity mismatch",
)
physical = receipt["physical_matter_transaction"]
require(
    physical
    == {
        "status": "passed",
        "device": "Apple M4 Pro",
        "case_count": 11,
        "environment_count_per_case": 3,
        "cases": expected_cases,
        "q_replay_error": 0,
        "v_replay_error": 0,
        "normal_history_replay_error": 0,
        "swap_count": 0,
        "restore_extent_admissibility_and_atomicity_rejected": True,
        "missing_or_short_initial_bodies_rejected": True,
        "bad_count_cone_and_overflow_initial_history_rejected": True,
        "borrowed_query_capacity_rejected_before_gpu": True,
    },
    "physical Matter receipt mismatch",
)

pre_fix = ROOT / "retained-failure/pre-fix-roundtrip-sha256.txt"
pre_fix_hashes = [line.split()[0] for line in pre_fix.read_text(encoding="utf-8").splitlines()]
require(len(pre_fix_hashes) == 4, "retained pre-fix hash evidence mismatch")
require(pre_fix_hashes[0] != pre_fix_hashes[1], "retained pre-fix NHINIT3 failure disappeared")
require(pre_fix_hashes[2] == pre_fix_hashes[3], "retained pre-fix Matter package control changed")
require(
    digest(ROOT / "retained-failure/pre-fix-positive.nhinit") == pre_fix_hashes[0],
    "retained pre-fix positive artifact mismatch",
)
require(
    digest(ROOT / "retained-failure/pre-fix-roundtrip.nhinit") == pre_fix_hashes[1],
    "retained pre-fix round-trip artifact mismatch",
)
require(
    pre_fix_hashes[2] == digest(positive / "prepared-100us.nmatterpack"),
    "retained pre-fix Matter package control is not present",
)
pre_fix_positive = (ROOT / "retained-failure/pre-fix-positive.nhinit").read_bytes()
pre_fix_roundtrip = (ROOT / "retained-failure/pre-fix-roundtrip.nhinit").read_bytes()
require(pre_fix_positive == nhinit_bytes, "retained pre-fix positive state differs from qualified input")
require(
    len(pre_fix_roundtrip) == 8196 and pre_fix_roundtrip[:8] == b"NHINIT3\0",
    "retained pre-fix round-trip is not an NHINIT3 artifact",
)
derived_diff = [
    f"{index:6d} {before:3o} {after:3o}"
    for index, (before, after) in enumerate(zip(pre_fix_positive, pre_fix_roundtrip, strict=True), start=1)
    if before != after
]
require(
    (ROOT / "retained-failure/pre-fix-roundtrip-diff.txt").read_text(encoding="utf-8").splitlines()
    == derived_diff,
    "retained pre-fix byte diff was not derived from the retained artifacts",
)
require(
    json.loads((ROOT / "retained-failure/pre-fix-positive.json").read_text(encoding="utf-8"))
    == prepared_json,
    "retained pre-fix identity JSON mismatch",
)
retained = receipt["retained_failures"]
require(len(retained) == 1, "retained failure count mismatch")
require(retained[0]["commit"] == IMPLEMENTATION_COMMIT, "retained failure commit mismatch")
require(retained[0]["positive_sha256"] == pre_fix_hashes[0], "retained positive hash mismatch")
require(retained[0]["roundtrip_sha256"] == pre_fix_hashes[1], "retained round-trip hash mismatch")
require(retained[0]["resolution_commit"] == QUALIFIED_COMMIT, "retained failure resolution mismatch")

qualification = receipt["qualification"]
expected_true = {
    "nhinit3_contract", "raw_nhcnt_identity_binding", "source_certificate_authoring", "authoring_roundtrip",
    "fixed_topology_serialization_and_provenance", "analytic_matter_accepted_history_ownership",
    "analytic_restore_and_replay", "physical_apple_gpu_execution",
}
expected_false = {
    "production_runner_nhinit3_admission", "per_grid_clocked_nhinit3_states",
    "whole_human_cold_seeded_comparison", "whole_human_numanx_runtime_publication",
    "topology_growth_execution", "anatomical_contact_loading", "force_convergence", "sustained_standing",
    "performance", "biological_validation", "production_readiness",
}
require(set(qualification) == expected_true | expected_false, "qualification field set mismatch")
require(all(qualification[key] is True for key in expected_true), "qualified fixed-topology evidence was demoted")
require(all(qualification[key] is False for key in expected_false), "unqualified production evidence was promoted")

negative_controls = receipt["negative_controls"]
require(all(value is True for key, value in negative_controls.items() if key.endswith(("_rejected", "_preserved_by_core_tests", "_passed"))), "negative-control qualification mismatch")
require(negative_controls["alternate_valid_support_exit_status"] == 1, "negative-control exit status mismatch")
require(negative_controls["focused_ctest_count"] == 2, "focused CTest count mismatch")
require(
    negative_controls["alternate_valid_support_diagnostic"]
    == "source-compliant support identity disagrees with supplied NHCNT",
    "negative-control receipt diagnostic mismatch",
)
require(
    receipt["next_evidence"]
    == "Add and qualify production-runner NHINIT3 admission with exact clock, composed-source, raw-NHCNT2, package/world, and initial-state checks; freeze common native commit/tree, runner/runtime-library/metallib, and rigid/muscle/NHEQ2/NHLIM1/NHCNT2 identities; author and qualify one exact-clock Matter-package SHA, world fingerprint, NHINIT3 SHA, and initial-state fingerprint tuple per 100/50/25/12.5 microsecond grid; then run paired cold/seeded cases on those exact tuples. The earlier NHCNT1 matrix remains historical trace-basis evidence, not the seeded comparator.",
    "next-evidence dependency order mismatch",
)
required_boundary_terms = (
    "does not qualify whole-Human NHINIT3 dynamics",
    "topology-growth execution",
    "force or timestep convergence",
    "sustained standing",
    "performance",
    "biological validity",
    "clinical validity",
    "production readiness",
)
require(all(term in receipt["boundary"] for term in required_boundary_terms), "claim boundary was broadened")

print("prepared support history receipt: PASS")
print("scope: fixed-topology NHINIT3 provenance/round-trip and analytic physical-M4 Matter transaction")
print("not qualified: production admission, per-grid seeded states, whole-Human dynamics, topology growth, force convergence, standing, performance, biology")
