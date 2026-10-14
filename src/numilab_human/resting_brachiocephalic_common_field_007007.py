"""Prepare a fresh common-field-only 007-007 successor from immutable 007-006 NHA.

The preflight hashes/import-pins the existing ABI5 source candidate and owners,
then checks only the NHANAT header. The prepare mode reuses the current
common-field owner and its validator; it does not recompile NHA or run geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import resting_brachiocephalic_common_field as common_field
from . import resting_brachiocephalic_source007 as source007

A = source007.A
SOURCE_DIR = A / "brachiocephalic-source-candidate-007-006"
SOURCE_RECEIPT = SOURCE_DIR / "resting-anatomy-receipt.json"
SOURCE_PAYLOAD = SOURCE_DIR / "resting-thorax.nhanatomy"
OUTPUT_DIR = A / "brachiocephalic-common-field-preparation-007-007"
PREFLIGHT_PATH = A / "brachiocephalic-common-field-preparation-007-007-preflight.json"
EXPECTED_PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
PUBLISHED_OWNER_REVISION = "fdabf993b469b0e4e99d8dcb58c46cf89d7ee61c"
UNCONSUMED_P6_VERIFIER = (
    source007.P6_PACKAGE / "resting_anatomy_verify.py")

# Hashes of the exact published owners from fdabf993, including the actual
# imported closest-point helper that the existing field compiler calls.
OWNER_SOURCE_SHA256 = {
    "resting_brachiocephalic_common_field.py": "2f903282334baa57fbc1f620e42b112cd3d8027ff8fb1e088074594105506940",
    "resting_brachiocephalic_source007.py": "aad4dfa2aa5c2bf1247e4e810dc55b903f9687584ef57fbf60361e013fb141aa",
    "resting_major_vessels.py": "f66cd67b50675ba259c72d5f46ce60b2f4b94d79591eda8f58818c60d9be3979",
    "resting_passive_viscera.py": "26bfff1ac0a3483f25ba7a8da5d57240ae25281e1d23fe8d2d3bc9c5acf967bf",
    "resting_vessel_owner_blend.py": "b66fc0efb9b2067aedd322c6a6239dfe920e41c41f004286dd2aefc056eb6bd9",
    "resting_intercostal_mesh_arrangement.py": "43e77da73607187ac23ece861c74d267892fd4ea80978fae0bafe278827b7fe4",
    "resting_anatomy_interface_patch.py": "aad523b4606ff17e7aa3ada0ac9c9c519f85597393cd4ea86492852e5b632add",
    "resting_pleura_proxy.py": "d874f599b8aeb9e9f02318fbae30ed5bf017920170d41c8b775cc3a9f792d683",
    "myosim_bone_proximity.py": "c01306982da4f8b09f370dda98281d123a012b11b855f2f49481de3c6de7e094",
}

# The compiler consumes source, receipts and carrier sidecars by hash; freezing
# every file in 007-006 prevents a changed carrier or manifest from being used.
SOURCE_007006_FILE_SHA256 = {
    "assets/axillary-477-reference-carriers.nhtissue": "4c9a443d8a77101847e8513a7ef102c322b9af5e317298bedb87c75b85e80cd9",
    "assets/back-parent-0.nhtissue": "e00dd0b40dd544433ccf66ac7afbbf64396b891af9ec90c90361dca9606a2d5f",
    "assets/back-parent-1.nhtissue": "4fd0bbc6739f01f025982fe19a0fa28205c8667c877b11a08194b78213b9cefd",
    "assets/prior-parent-0.nhtissue": "ac435503a9b0826f79a4b95a1b0be6d656114b3378239b210b06c4dcf54e54f6",
    "assets/prior-parent-1.nhtissue": "2c0bb0e69c9254cc34f9d90a25dd313a8362524529fdde4f9210641cd79424d3",
    "assets/route110-dual-vessel-reference-carriers.nhtissue": "f77cf7d6da5677178e858f68abe92d7272925f5a783eb8739eba01ad14027393",
    "resting-anatomy-manifest.json": "67f8793ee236978e2e4b1d913d3004d12eb33c7a639f513387dce503b7f08deb",
    "resting-anatomy-receipt.json": "689acf7a54f629e67fa59a4efff1285bf6862fa60e31cea3b0ac9d2e3a5efe8b",
    "resting-thorax.nhanatomy": "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42",
}

NHANAT_HEADER = struct.Struct("<8s5I32s")
NHANAT_RECORD = struct.Struct("<8I")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_nhanat_header(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(NHANAT_HEADER.size)
    if len(raw) != NHANAT_HEADER.size:
        raise ValueError("truncated NHANAT header")
    magic, abi, row_count, vertex_count, index_count, fingerprint, source_sha = (
        NHANAT_HEADER.unpack(raw))
    if magic != b"NHANAT1\0" or abi != 5:
        raise ValueError("source candidate is not NHANAT ABI5")
    expected_size = (NHANAT_HEADER.size + row_count * NHANAT_RECORD.size
                     + vertex_count * 24 + index_count * 4)
    if path.stat().st_size != expected_size:
        raise ValueError("NHANAT header counts differ from payload byte length")
    return {
        "magic": magic.rstrip(b"\0").decode("ascii"),
        "abi": int(abi),
        "surface_count": int(row_count),
        "vertex_count": int(vertex_count),
        "index_count": int(index_count),
        "registration_fingerprint_u32": int(fingerprint),
        "source_sha256_header_hex": source_sha.hex(),
        "payload_size_bytes": int(expected_size),
        "header_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "geometry_arrays_decoded": False,
    }


def _owner_import_pins() -> dict:
    repo = Path(__file__).resolve().parents[2]
    source_dir = repo / "src/numilab_human"
    names = {
        "numilab_human.resting_brachiocephalic_common_field":
            "resting_brachiocephalic_common_field.py",
        "numilab_human.resting_brachiocephalic_source007":
            "resting_brachiocephalic_source007.py",
        "numilab_human.resting_major_vessels": "resting_major_vessels.py",
        "numilab_human.resting_passive_viscera": "resting_passive_viscera.py",
        "numilab_human.resting_vessel_owner_blend": "resting_vessel_owner_blend.py",
        "numilab_human.resting_intercostal_mesh_arrangement":
            "resting_intercostal_mesh_arrangement.py",
        "numilab_human.resting_anatomy_interface_patch":
            "resting_anatomy_interface_patch.py",
        "numilab_human.resting_pleura_proxy": "resting_pleura_proxy.py",
        "numilab_human.myosim_bone_proximity": "myosim_bone_proximity.py",
    }
    actual = {}
    for module_name, file_name in names.items():
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        expected_path = (source_dir / file_name).resolve()
        if origin != expected_path:
            raise ValueError(f"unexpected imported owner origin for {module_name}: {origin}")
        digest = _sha_file(origin)
        expected_digest = OWNER_SOURCE_SHA256[file_name]
        if digest != expected_digest:
            raise ValueError(f"published source owner hash changed: {file_name}")
        actual[module_name] = {
            "origin": str(origin),
            "sha256": digest,
            "module_spec_origin": str(Path(module.__spec__.origin).resolve()),
        }
    compiler_pins = common_field._source_code_pins()
    if compiler_pins != {name: OWNER_SOURCE_SHA256[name] for name in compiler_pins}:
        raise ValueError("existing common-field compiler code pins differ")
    return {"modules": actual, "common_field_compiler_code_sha256": compiler_pins}


def _git_revision(repo: Path) -> dict:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], check=True,
        capture_output=True, text=True).stdout.splitlines()
    if status:
        raise ValueError("source checkout must be clean before the common-field preflight")
    return {"head": head, "clean": True}


def _input_files() -> dict[Path, str]:
    files = {
        SOURCE_RECEIPT: SOURCE_007006_FILE_SHA256["resting-anatomy-receipt.json"],
        SOURCE_PAYLOAD: SOURCE_007006_FILE_SHA256["resting-thorax.nhanatomy"],
        source007.BASE_RECEIPT: source007.N27_RECEIPT_SHA,
        source007.BASE_PAYLOAD: source007.N27_PAYLOAD_SHA,
        source007.VESSEL_OWNER_PORT_REPORT: source007.VESSEL_OWNER_PORT_REPORT_SHA,
    }
    for raw_path, expected in source007.INPUT_SHA.items():
        path = Path(raw_path)
        if path.resolve() == UNCONSUMED_P6_VERIFIER.resolve():
            # The 007-006 source receipt retains its historical P6 pin. The
            # common-field owner does not import or execute P6, so keep that
            # pin as provenance rather than binding this common-only prep to it.
            continue
        files[path] = expected
    files.update({Path(path): expected for path, expected in source007.ROUTE110_SHA.items()})
    for relative_path, expected in SOURCE_007006_FILE_SHA256.items():
        files[SOURCE_DIR / relative_path] = expected
    for route_name, expected in common_field.ROUTE_HASHES.items():
        route_path = {
            "ancestry": source007.COLLAR_ANCESTRY,
            "attribution": source007.COLLAR_ATTRIBUTION,
            "replay": source007.COLLAR_REPLAY,
            "stationarity": source007.COLLAR_STATIONARITY,
            "distances": source007.COLLAR_DISTANCES,
        }[route_name]
        files[route_path] = expected
    return files


def _python_executable_identity() -> dict[str, str]:
    # The qualified venv launcher is a symlink to the shared framework Python.
    # Bind both spellings; compare the launcher path so the venv is not rejected.
    invoked = Path(sys.executable).absolute()
    expected_venv = EXPECTED_PYTHON.parent.parent.resolve()
    expected_bin = expected_venv / "bin"
    if (invoked.parent.resolve() != expected_bin.resolve()
            or Path(sys.prefix).resolve() != expected_venv
            or Path(sys.base_prefix).resolve() == expected_venv):
        raise ValueError(f"run with the qualified Numi interpreter: {EXPECTED_PYTHON}")
    return {
        "executable": str(invoked),
        "resolved_executable": str(invoked.resolve()),
        "virtual_environment": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
    }


def _verify_file_pins(files: dict[Path, str]) -> dict[str, str]:
    actual_pins = {}
    for path, expected in files.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(f"required common-preparation input is missing: {resolved}")
        actual = _sha_file(resolved)
        if actual != expected:
            raise ValueError(f"input hash changed at {resolved}: {actual}")
        actual_pins[str(resolved)] = actual
    return actual_pins


def _current_unconsumed_p6_context() -> dict[str, Any]:
    path = UNCONSUMED_P6_VERIFIER.resolve()
    root = source007.P6_ROOT.resolve()
    revision = None
    clean = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], check=True,
            capture_output=True, text=True).stdout.splitlines()
        clean = not status
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "status": "HISTORICAL_UNCONSUMED_CONTEXT_ONLY",
        "historical_source007_verifier_sha256": source007.P6_VERIFIER_SHA,
        "current_path": str(path),
        "current_sha256": _sha_file(path) if path.is_file() else None,
        "current_repository_head": revision,
        "current_repository_clean": clean,
        "consumed_by_common_field_preparation": False,
        "p6_predicates_invoked": False,
    }


def build_preflight() -> dict:
    python_identity = _python_executable_identity()
    if OUTPUT_DIR.exists():
        raise ValueError("retain existing 007-007 preparation output; do not overwrite")

    repo = Path(__file__).resolve().parents[2]
    owner_imports = _owner_import_pins()
    input_pins = _verify_file_pins(_input_files())

    source_receipt = json.loads(SOURCE_RECEIPT.read_text())
    source_payload_path = Path(source_receipt["payload"]["path"])
    if not source_payload_path.is_absolute():
        source_payload_path = SOURCE_RECEIPT.parent / source_payload_path
    source_payload_path = source_payload_path.resolve()
    if source_payload_path != SOURCE_PAYLOAD.resolve():
        raise ValueError("007-006 receipt points to a different NHA payload")
    if source_receipt["payload"].get("sha256") != SOURCE_007006_FILE_SHA256[
            "resting-thorax.nhanatomy"]:
        raise ValueError("007-006 receipt payload identity differs")

    source_header = _read_nhanat_header(SOURCE_PAYLOAD)
    if source_header["surface_count"] != 525:
        raise ValueError("007-006 candidate is not the expected 523+2 ABI5 row set")
    source_map = source_receipt.get("provenance", {}).get("source_id_map", {})
    bridge_identities = {}
    for stable_id, member in ((525, "FJ3482"), (526, "FJ3583")):
        identity = source_map.get(str(stable_id), {})
        if (identity.get("source_member") != member
                or int(identity.get("body_index", -1)) != 20
                or int(identity.get("layer", -1)) != 2
                or int(identity.get("vascular_compartment_id", -1)) != 4):
            raise ValueError(f"007-006 source ownership changed for {stable_id}")
        bridge_identities[str(stable_id)] = {
            "source_member": member,
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
        }
    extension = source_receipt.get("provenance", {}).get(
        "passive_brachiocephalic_vein_extension", {})
    if (extension.get("base_receipt_sha256") != source007.N27_RECEIPT_SHA
            or extension.get("base_payload_sha256") != source007.N27_PAYLOAD_SHA):
        raise ValueError("007-006 is not directly bound to pinned N27")
    base_header = _read_nhanat_header(source007.BASE_PAYLOAD)
    if base_header["surface_count"] != 523:
        raise ValueError("pinned N27 base is not the 523-row ABI5 source")

    base_receipt = json.loads(source007.BASE_RECEIPT.read_text())
    candidate_common = source_receipt.get("provenance", {}).get(
        "cardiac_geometry_binding", {}).get("common_field")
    base_common = base_receipt.get("provenance", {}).get(
        "cardiac_geometry_binding", {}).get("common_field")
    common_field._validate_common_field_descriptor(candidate_common, base_common)
    for sidecar_key in ("map", "polynomials", "domain_boxes"):
        candidate_sidecar = common_field._resolve_sidecar(
            SOURCE_RECEIPT, source007.BASE_RECEIPT, candidate_common[sidecar_key])
        base_sidecar = common_field._resolve_sidecar(
            source007.BASE_RECEIPT, source007.BASE_RECEIPT, base_common[sidecar_key])
        if candidate_sidecar.read_bytes() != base_sidecar.read_bytes():
            raise ValueError(f"007-006 existing common-field sidecar changed: {sidecar_key}")
        input_pins[str(candidate_sidecar)] = _sha_file(candidate_sidecar)
        input_pins[str(base_sidecar)] = _sha_file(base_sidecar)

    return {
        "schema": "numi.human.brachiocephalic-common-field-007007-preflight.v1",
        "status": "PREPARED_NOT_EXECUTED",
        "source_owner_publication": PUBLISHED_OWNER_REVISION,
        "driver_source_sha256": _sha_file(Path(__file__).resolve()),
        "source_repository": _git_revision(repo),
        "python": {
            **python_identity,
            "version": platform.python_version(),
        },
        "owner_imports": owner_imports,
        "p6_provenance": _current_unconsumed_p6_context(),
        "source_candidate": {
            "path": str(SOURCE_DIR),
            "receipt_path": str(SOURCE_RECEIPT),
            "receipt_sha256": SOURCE_007006_FILE_SHA256["resting-anatomy-receipt.json"],
            "payload_path": str(SOURCE_PAYLOAD),
            "payload_sha256": SOURCE_007006_FILE_SHA256["resting-thorax.nhanatomy"],
            "nhanat_header": source_header,
            "bridge_source_id_map_ownership": bridge_identities,
            "reused_source_candidate_without_recompile": True,
        },
        "pinned_n27_base": {
            "receipt_path": str(source007.BASE_RECEIPT),
            "receipt_sha256": source007.N27_RECEIPT_SHA,
            "payload_path": str(source007.BASE_PAYLOAD),
            "payload_sha256": source007.N27_PAYLOAD_SHA,
            "nhanat_header": base_header,
        },
        "input_sha256": input_pins,
        "output": {
            "preparation_dir": str(OUTPUT_DIR),
            "preflight_path": str(PREFLIGHT_PATH),
            "preparation_dir_absent": True,
        },
        "scope": {
            "abi5_source_append_rerun": False,
            "existing_common_field_owner": "prepare_brachiocephalic_common_field",
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
            "anatomical_admission": False,
            "all_pose_admission": False,
        },
    }


def write_preflight() -> dict:
    if PREFLIGHT_PATH.exists():
        raise ValueError("retain existing 007-007 preflight; do not overwrite")
    report = build_preflight()
    raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    PREFLIGHT_PATH.write_bytes(raw)
    return {"path": str(PREFLIGHT_PATH), "sha256": hashlib.sha256(raw).hexdigest(),
            "report": report}


def prepare(*, expected_preflight_sha256: str) -> dict:
    if not PREFLIGHT_PATH.is_file():
        raise ValueError("007-007 preflight report is missing")
    preflight_sha = _sha_file(PREFLIGHT_PATH)
    if preflight_sha != expected_preflight_sha256:
        raise ValueError("007-007 preflight SHA does not match the reviewed pin")
    frozen = json.loads(PREFLIGHT_PATH.read_text())
    current = build_preflight()
    if frozen != current:
        raise ValueError("007-007 current inputs/imports differ from frozen preflight")

    result = common_field.prepare_brachiocephalic_common_field(
        candidate_receipt_path=SOURCE_RECEIPT,
        output_dir=OUTPUT_DIR,
        collar_ancestry_path=source007.COLLAR_ANCESTRY,
        collar_attribution_path=source007.COLLAR_ATTRIBUTION,
        collar_replay_path=source007.COLLAR_REPLAY,
        collar_stationarity_path=source007.COLLAR_STATIONARITY,
        port_distances_path=source007.COLLAR_DISTANCES,
    )
    validated = source007._validate_source_candidate(
        Path(result["receipt_path"]), Path(result["report_path"]))
    return {
        "status": result["status"],
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": preflight_sha,
        "candidate_payload_sha256": validated["receipt"]["payload"]["sha256"],
        "preparation_report_path": result["report_path"],
        "preparation_report_sha256": result["report_sha256"],
        "prepared_receipt_path": result["receipt_path"],
        "prepared_receipt_sha256": result["receipt_sha256"],
        "map_sha256": result["map_sha256"],
        "map_record_count": result["map_record_count"],
        "map_prefix_sha256": result["map_prefix_sha256"],
        "polynomials_sha256": result["polynomials_sha256"],
        "domain_boxes_sha256": result["domain_boxes_sha256"],
        "source_append_recompiled": False,
        "geometry_predicates_run": False,
        "native_or_gpu_run": False,
        "anatomical_admission": False,
        "all_pose_admission": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if args.preflight_only:
        if args.preflight_sha256:
            parser.error("--preflight-sha256 is only valid with --prepare")
        result = write_preflight()
    else:
        if not args.preflight_sha256:
            parser.error("--prepare requires the reviewed --preflight-sha256")
        result = prepare(expected_preflight_sha256=args.preflight_sha256)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
