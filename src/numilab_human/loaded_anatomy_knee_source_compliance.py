"""Bind the exact loaded-knee source constraint laws without stepping physics.

This additive companion leaves ``HumanPack.loaded-anatomy-knee.v1`` unchanged.
Human owns the immutable NHEQ2/NHLIM1 source-law bytes.  Matter retains sole
authority for runtime constraint force and accepted constraint state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

from . import loaded_anatomy_knee as knee
from .target_coverage import canonical_bytes, digest

SCHEMA = "HumanPack.loaded-anatomy-knee.source-compliance.v1"
COMPILER = "numilab-human.loaded-anatomy-knee-source-compliance.1"
CANONICALIZATION = knee.CANONICALIZATION
HASH_EXCLUSION = "top-level binding_sha256"
FILENAME = "HumanPack.loaded-anatomy-knee.source-compliance.v1.json"

HEADER = struct.Struct("<8s10I32s")
NQ = 129
NV = 128
POLICY_ID = 1
FLAGS = 1
SOURCE_ARCHIVE_SHA256 = (
    "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
)
SOURCE_RIGID_PAYLOAD_SHA256 = (
    "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44"
)
EQUALITY_PAYLOAD_SHA256 = (
    "12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477"
)
LIMIT_PAYLOAD_SHA256 = (
    "c583611fcedc326a32c6f69504a65e675c8e0adc987c6db622ca0d95a02438d3"
)

HUMAN_SOURCE_LAW_OWNER_ID = (
    "numilab-human:loaded-anatomy-knee/source-constraint-laws"
)
MATTER_RUNTIME_OWNER_ID = "numi-lab:human-matter/accepted-step-transaction"

BOUNDARY = (
    "Candidate-only additive source-compliance binding for "
    "HumanPack.loaded-anatomy-knee.v1. Human authors the immutable NHEQ2 and "
    "NHLIM1 source-law bytes; Matter alone owns runtime constraint force and "
    "accepted constraint state. This companion binds no prepared-state identity "
    "and does not establish runtime execution, production physical ownership, "
    "standing, walking, clinical validity, or integrated Human qualification."
)

PROGRAMS = {
    "joint_equalities": {
        "schema": "numi.human.joint-equality-source-compliance-payload.v1",
        "filename": "myosim-fullbody-joint-equalities-source-compliance.nheq",
        "magic_bytes": b"NHEQ2\0\0\0",
        "magic": "NHEQ2",
        "abi": 2,
        "row_count": 51,
        "record_bytes": 112,
        "file_sha256": EQUALITY_PAYLOAD_SHA256,
    },
    "joint_limits": {
        "schema": "numi.human.joint-limit-source-compliance-payload.v1",
        "filename": "myosim-fullbody-joint-limits.nhlim",
        "magic_bytes": b"NHLIM1\0\0",
        "magic": "NHLIM1",
        "abi": 1,
        "row_count": 122,
        "record_bytes": 80,
        "file_sha256": LIMIT_PAYLOAD_SHA256,
    },
}


class SourceComplianceError(knee.LoadedAnatomyKneeError):
    """The loaded-knee source-compliance companion cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceComplianceError(
            "HumanPack loaded anatomy knee source compliance: " + message
        )


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _base_descriptor(base_manifest: dict[str, Any]) -> dict[str, Any]:
    knee.validate_manifest(base_manifest)
    return {
        "schema": knee.SCHEMA,
        "manifest_sha256": base_manifest["manifest_sha256"],
        "file_sha256": hashlib.sha256(
            canonical_bytes(base_manifest) + b"\n"
        ).hexdigest(),
    }


def _decode_program(name: str, payload: bytes) -> dict[str, Any]:
    specification = PROGRAMS[name]
    _require(type(payload) is bytes, f"{name} payload must be bytes")
    _require(len(payload) >= HEADER.size, f"{name} payload is truncated")
    (
        magic,
        abi,
        nq,
        nv,
        row_count,
        record_bytes,
        source_row_count,
        policy_id,
        flags,
        reserved0,
        reserved1,
        source_sha256,
    ) = HEADER.unpack_from(payload)
    _require(magic == specification["magic_bytes"], f"{name} magic differs")
    _require(abi == specification["abi"], f"{name} ABI differs")
    _require(nq == NQ and nv == NV, f"{name} nq/nv differ")
    _require(row_count == specification["row_count"], f"{name} row count differs")
    _require(
        record_bytes == specification["record_bytes"],
        f"{name} record ABI differs",
    )
    _require(
        source_row_count == specification["row_count"],
        f"{name} source row count differs",
    )
    _require(policy_id == POLICY_ID, f"{name} source policy differs")
    _require(flags == FLAGS, f"{name} source flags differ")
    _require(reserved0 == 0 and reserved1 == 0, f"{name} reserved fields differ")
    _require(
        source_sha256.hex() == SOURCE_ARCHIVE_SHA256,
        f"{name} source rigid identity differs",
    )
    expected_bytes = HEADER.size + row_count * record_bytes
    _require(len(payload) == expected_bytes, f"{name} byte count differs")
    file_sha256 = hashlib.sha256(payload).hexdigest()
    _require(
        file_sha256 == specification["file_sha256"],
        f"{name} exact payload SHA-256 differs",
    )
    return {
        "schema": specification["schema"],
        "filename": specification["filename"],
        "magic": specification["magic"],
        "abi": abi,
        "bytes": len(payload),
        "file_sha256": file_sha256,
        "nq": nq,
        "nv": nv,
        "row_count": row_count,
        "record_bytes": record_bytes,
        "source_row_count": source_row_count,
        "policy_id": policy_id,
        "flags": flags,
        "refsafe": bool(flags & 1),
        "source_archive_sha256": source_sha256.hex(),
    }


def _validate_cross_program(
    equality: dict[str, Any], limit: dict[str, Any]
) -> dict[str, bool]:
    _require(
        equality["nq"] == limit["nq"] and equality["nv"] == limit["nv"],
        "source programs disagree on nq/nv",
    )
    _require(
        equality["source_archive_sha256"] == limit["source_archive_sha256"],
        "source programs disagree on source rigid identity",
    )
    _require(
        equality["policy_id"] == limit["policy_id"],
        "source programs disagree on policy",
    )
    _require(
        equality["flags"] == limit["flags"],
        "source programs disagree on flags",
    )
    _require(
        equality["refsafe"] is limit["refsafe"],
        "source programs disagree on refsafe",
    )
    return {
        "same_nq_nv": True,
        "same_source_archive_sha256": True,
        "same_policy_id": True,
        "same_flags": True,
        "same_refsafe": True,
    }


def _ownership(base_manifest: dict[str, Any]) -> dict[str, Any]:
    authority = base_manifest["full_state_authority"]
    _require(
        authority["owner_id"] == MATTER_RUNTIME_OWNER_ID,
        "base manifest changed Matter runtime authority",
    )
    return {
        "human_source_laws": {
            "owner_id": HUMAN_SOURCE_LAW_OWNER_ID,
            "owner_system": "Human",
            "authority": "immutable-source-law-program-bytes",
            "programs": ["joint_equalities", "joint_limits"],
            "owns_runtime_constraint_force": False,
            "owns_runtime_constraint_state": False,
        },
        "matter_runtime_constraints": {
            "owner_id": authority["owner_id"],
            "owner_system": "Matter",
            "authority": "runtime-constraint-force-and-accepted-state",
            "accepted_state_schema": authority["schema"],
            "accepted_state_semantic_id": authority["semantic_id"],
            "accepted_state_authority_identity_sha256": authority[
                "identity_sha256"
            ],
            "authors_source_laws": False,
            "owns_runtime_constraint_force": True,
            "owns_runtime_constraint_state": True,
        },
    }


def compile_contract(
    *,
    base_manifest: dict[str, Any],
    equality_payload: bytes,
    limit_payload: bytes,
) -> dict[str, Any]:
    """Compile one deterministic identity for the exact source-law byte pair."""
    base = _base_descriptor(base_manifest)
    source_rigid_payload_sha256 = base_manifest["lab_authoring_export"]["source"][
        "source_rigid_payload_sha256"
    ]
    source_archive_sha256 = base_manifest["source"]["tendon_payload"][
        "myosim_archive_sha256"
    ]
    _require(
        source_rigid_payload_sha256 == SOURCE_RIGID_PAYLOAD_SHA256,
        "base source rigid payload SHA-256 differs",
    )
    _require(
        source_archive_sha256 == SOURCE_ARCHIVE_SHA256,
        "base source archive SHA-256 differs",
    )
    equality = _decode_program("joint_equalities", equality_payload)
    limit = _decode_program("joint_limits", limit_payload)
    cross_program = _validate_cross_program(equality, limit)
    _require(
        equality["source_archive_sha256"] == source_archive_sha256,
        "source programs do not bind the base source archive",
    )
    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "manifest_canonicalization": CANONICALIZATION,
        "binding_hash_exclusion": HASH_EXCLUSION,
        "binding_sha256": "",
        "status": "candidate",
        "base_manifest": base,
        "source_model": {
            "nq": NQ,
            "nv": NV,
            "source_archive_sha256": source_archive_sha256,
            "source_rigid_payload_sha256": source_rigid_payload_sha256,
        },
        "programs": {
            "joint_equalities": equality,
            "joint_limits": limit,
        },
        "cross_program": cross_program,
        "ownership": _ownership(base_manifest),
        "prepared_state": {
            "status": "absent",
            "identity_sha256": None,
            "qualification_receipt_sha256": None,
            "qualified": False,
        },
        "qualification": {
            "candidate_only": True,
            "base_manifest_identity_bound": True,
            "exact_source_program_bytes_bound": True,
            "source_rigid_identity_bound": True,
            "program_headers_validated": True,
            "cross_program_consistency_validated": True,
            "prepared_state_identity_bound": False,
            "runtime_constraint_force_or_state_executed": False,
            "production_physical_ownership": False,
            "clinical_validity_qualified": False,
            "integrated_human_qualification": False,
        },
        "boundary": BOUNDARY,
    }
    result["binding_sha256"] = digest(
        {key: item for key, item in result.items() if key != "binding_sha256"}
    )
    validate_contract(
        result,
        base_manifest=base_manifest,
        equality_payload=equality_payload,
        limit_payload=limit_payload,
    )
    return result


def validate_contract(
    value: dict[str, Any],
    *,
    base_manifest: dict[str, Any],
    equality_payload: bytes,
    limit_payload: bytes,
) -> None:
    """Validate structure plus every bound input; no detached admission exists."""
    required = {
        "schema",
        "compiler",
        "manifest_canonicalization",
        "binding_hash_exclusion",
        "binding_sha256",
        "status",
        "base_manifest",
        "source_model",
        "programs",
        "cross_program",
        "ownership",
        "prepared_state",
        "qualification",
        "boundary",
    }
    _require(isinstance(value, dict) and set(value) == required, "fields differ")
    _require(
        value["schema"] == SCHEMA
        and value["compiler"] == COMPILER
        and value["manifest_canonicalization"] == CANONICALIZATION
        and value["binding_hash_exclusion"] == HASH_EXCLUSION
        and value["status"] == "candidate"
        and value["boundary"] == BOUNDARY,
        "schema, canonicalization, status, or boundary differs",
    )
    _sha256(value["binding_sha256"], "binding identity")
    _require(
        value["binding_sha256"]
        == digest(
            {key: item for key, item in value.items() if key != "binding_sha256"}
        ),
        "binding identity mismatch",
    )
    expected_base = _base_descriptor(base_manifest)
    _require(value["base_manifest"] == expected_base, "base manifest binding differs")
    expected_source = {
        "nq": NQ,
        "nv": NV,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_rigid_payload_sha256": SOURCE_RIGID_PAYLOAD_SHA256,
    }
    _require(value["source_model"] == expected_source, "source model binding differs")
    expected_programs = {
        "joint_equalities": _decode_program("joint_equalities", equality_payload),
        "joint_limits": _decode_program("joint_limits", limit_payload),
    }
    _require(value["programs"] == expected_programs, "source program binding differs")
    expected_cross = _validate_cross_program(
        expected_programs["joint_equalities"], expected_programs["joint_limits"]
    )
    _require(
        value["cross_program"] == expected_cross,
        "cross-program validation record differs",
    )
    _require(value["ownership"] == _ownership(base_manifest), "ownership differs")
    _require(
        value["prepared_state"]
        == {
            "status": "absent",
            "identity_sha256": None,
            "qualification_receipt_sha256": None,
            "qualified": False,
        },
        "prepared-state identity must remain explicitly absent and unqualified",
    )
    _require(
        value["qualification"]
        == {
            "candidate_only": True,
            "base_manifest_identity_bound": True,
            "exact_source_program_bytes_bound": True,
            "source_rigid_identity_bound": True,
            "program_headers_validated": True,
            "cross_program_consistency_validated": True,
            "prepared_state_identity_bound": False,
            "runtime_constraint_force_or_state_executed": False,
            "production_physical_ownership": False,
            "clinical_validity_qualified": False,
            "integrated_human_qualification": False,
        },
        "qualification boundary differs",
    )


def compile_paths(
    *, base_manifest_path: Path, equality_path: Path, limit_path: Path
) -> dict[str, Any]:
    base_manifest, _ = knee._read_json(base_manifest_path, "base knee manifest")
    equality_payload, _ = knee._read_bytes(equality_path, "NHEQ2 source program")
    limit_payload, _ = knee._read_bytes(limit_path, "NHLIM1 source program")
    return compile_contract(
        base_manifest=base_manifest,
        equality_payload=equality_payload,
        limit_payload=limit_payload,
    )


def write_immutable(path: Path, value: dict[str, Any]) -> str:
    """Publish canonical JSON once, or accept an already identical artifact."""
    path = Path(path)
    encoded = canonical_bytes(value) + b"\n"
    _require(path.name not in {"", ".", ".."}, "output basename is invalid")
    _require(not path.is_symlink(), "output is redirected")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(path.parent.is_dir(), "output parent is not a directory")
    if path.exists():
        _require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == encoded,
            "existing output differs",
        )
        return hashlib.sha256(encoded).hexdigest()

    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    staging = Path(staging_name)
    try:
        try:
            os.fchmod(descriptor, 0o644)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "output write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError:
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == encoded,
                "concurrent output differs",
            )
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(encoded).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--joint-equalities", type=Path, required=True)
    parser.add_argument("--joint-limits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_paths(
        base_manifest_path=arguments.base_manifest,
        equality_path=arguments.joint_equalities,
        limit_path=arguments.joint_limits,
    )
    file_sha256 = write_immutable(arguments.output, result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": result["status"],
                "binding_sha256": result["binding_sha256"],
                "file_sha256": file_sha256,
                "prepared_state_identity_bound": False,
                "runtime_constraint_force_or_state_executed": False,
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (knee.LoadedAnatomyKneeError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"loaded-anatomy-knee-source-compliance: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
