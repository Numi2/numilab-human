from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

import numilab_human.loaded_anatomy_knee as knee
import numilab_human.loaded_anatomy_knee_source_compliance as compliance
from numilab_human.target_coverage import canonical_bytes, digest
from tests.test_loaded_anatomy_knee import _manifest

EQUALITY_PATH = (
    knee.ROOT
    / "Docs/media/native-prepared-support-history-v1/inputs/"
    "myosim-fullbody-joint-equalities-source-compliance.nheq"
)
LIMIT_PATH = (
    knee.ROOT
    / "Docs/media/native-prepared-support-history-v1/inputs/"
    "myosim-fullbody-joint-limits.nhlim"
)


def _inputs() -> tuple[dict, bytes, bytes]:
    return _manifest(), EQUALITY_PATH.read_bytes(), LIMIT_PATH.read_bytes()


def _compile() -> tuple[dict, dict, bytes, bytes]:
    base, equality, limit = _inputs()
    value = compliance.compile_contract(
        base_manifest=base,
        equality_payload=equality,
        limit_payload=limit,
    )
    return value, base, equality, limit


def _rehash(value: dict) -> None:
    value["binding_sha256"] = digest(
        {key: item for key, item in value.items() if key != "binding_sha256"}
    )


def test_exact_source_programs_compile_deterministically() -> None:
    value, base, equality, limit = _compile()
    repeated = compliance.compile_contract(
        base_manifest=base,
        equality_payload=equality,
        limit_payload=limit,
    )
    assert value == repeated
    compliance.validate_contract(
        value,
        base_manifest=base,
        equality_payload=equality,
        limit_payload=limit,
    )
    assert value["base_manifest"] == {
        "schema": knee.SCHEMA,
        "manifest_sha256": base["manifest_sha256"],
        "file_sha256": hashlib.sha256(
            canonical_bytes(base) + b"\n"
        ).hexdigest(),
    }
    assert value["programs"]["joint_equalities"]["file_sha256"] == (
        compliance.EQUALITY_PAYLOAD_SHA256
    )
    assert value["programs"]["joint_limits"]["file_sha256"] == (
        compliance.LIMIT_PAYLOAD_SHA256
    )
    assert value["source_model"] == {
        "nq": 129,
        "nv": 128,
        "source_archive_sha256": compliance.SOURCE_ARCHIVE_SHA256,
        "source_rigid_payload_sha256": compliance.SOURCE_RIGID_PAYLOAD_SHA256,
    }
    assert all(value["cross_program"].values())


def test_ownership_separates_authored_laws_from_runtime_force_and_state() -> None:
    value, _, _, _ = _compile()
    human = value["ownership"]["human_source_laws"]
    matter = value["ownership"]["matter_runtime_constraints"]
    assert human["programs"] == ["joint_equalities", "joint_limits"]
    assert human["owns_runtime_constraint_force"] is False
    assert human["owns_runtime_constraint_state"] is False
    assert matter["authors_source_laws"] is False
    assert matter["owns_runtime_constraint_force"] is True
    assert matter["owns_runtime_constraint_state"] is True
    assert matter["owner_id"] == compliance.MATTER_RUNTIME_OWNER_ID


def test_prepared_state_remains_explicitly_absent_and_unqualified() -> None:
    value, _, _, _ = _compile()
    assert value["prepared_state"] == {
        "status": "absent",
        "identity_sha256": None,
        "qualification_receipt_sha256": None,
        "qualified": False,
    }
    assert value["qualification"]["prepared_state_identity_bound"] is False
    assert (
        value["qualification"]["runtime_constraint_force_or_state_executed"]
        is False
    )


@pytest.mark.parametrize(
    ("name", "offset", "replacement"),
    [
        ("magic", 0, b"ATTACK!!"),
        ("abi", 8, struct.pack("<I", 99)),
        ("nq", 12, struct.pack("<I", 130)),
        ("nv", 16, struct.pack("<I", 127)),
        ("row count", 20, struct.pack("<I", 50)),
        ("record ABI", 24, struct.pack("<I", 96)),
        ("source row count", 28, struct.pack("<I", 50)),
        ("policy", 32, struct.pack("<I", 2)),
        ("flags", 36, struct.pack("<I", 0)),
        ("reserved", 40, struct.pack("<I", 1)),
        ("source rigid identity", 48, b"\xff" * 32),
    ],
)
def test_nheq_header_mutations_fail_closed(
    name: str, offset: int, replacement: bytes
) -> None:
    base, equality, limit = _inputs()
    changed = bytearray(equality)
    changed[offset : offset + len(replacement)] = replacement
    with pytest.raises(compliance.SourceComplianceError):
        compliance.compile_contract(
            base_manifest=base,
            equality_payload=bytes(changed),
            limit_payload=limit,
        )


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0, b"ATTACK!!"),
        (8, struct.pack("<I", 2)),
        (12, struct.pack("<I", 130)),
        (16, struct.pack("<I", 127)),
        (20, struct.pack("<I", 121)),
        (24, struct.pack("<I", 64)),
        (28, struct.pack("<I", 121)),
        (32, struct.pack("<I", 2)),
        (36, struct.pack("<I", 0)),
        (44, struct.pack("<I", 1)),
        (48, b"\xee" * 32),
    ],
)
def test_nhlim_header_mutations_fail_closed(
    offset: int, replacement: bytes
) -> None:
    base, equality, limit = _inputs()
    changed = bytearray(limit)
    changed[offset : offset + len(replacement)] = replacement
    with pytest.raises(compliance.SourceComplianceError):
        compliance.compile_contract(
            base_manifest=base,
            equality_payload=equality,
            limit_payload=bytes(changed),
        )


@pytest.mark.parametrize("program", ["equalities", "limits"])
@pytest.mark.parametrize("mutation", ["body", "truncated", "trailing"])
def test_exact_program_bytes_are_not_replaceable(
    program: str, mutation: str
) -> None:
    base, equality, limit = _inputs()
    original = equality if program == "equalities" else limit
    if mutation == "body":
        changed = bytearray(original)
        changed[compliance.HEADER.size] ^= 1
        replacement = bytes(changed)
    elif mutation == "truncated":
        replacement = original[:-1]
    else:
        replacement = original + b"\0"
    with pytest.raises(compliance.SourceComplianceError):
        compliance.compile_contract(
            base_manifest=base,
            equality_payload=replacement if program == "equalities" else equality,
            limit_payload=replacement if program == "limits" else limit,
        )


def test_cross_program_mismatch_is_rejected() -> None:
    _, equality, limit = _inputs()
    equality_descriptor = compliance._decode_program("joint_equalities", equality)
    limit_descriptor = compliance._decode_program("joint_limits", limit)
    for field, replacement in (
        ("nq", 130),
        ("source_archive_sha256", "f" * 64),
        ("policy_id", 2),
        ("flags", 0),
        ("refsafe", False),
    ):
        changed = copy.deepcopy(limit_descriptor)
        changed[field] = replacement
        with pytest.raises(compliance.SourceComplianceError):
            compliance._validate_cross_program(equality_descriptor, changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item["base_manifest"].__setitem__("manifest_sha256", "f" * 64),
        lambda item: item["source_model"].__setitem__(
            "source_rigid_payload_sha256", "f" * 64
        ),
        lambda item: item["programs"]["joint_equalities"].__setitem__(
            "file_sha256", "f" * 64
        ),
        lambda item: item["cross_program"].__setitem__("same_flags", False),
        lambda item: item["ownership"]["human_source_laws"].__setitem__(
            "owns_runtime_constraint_force", True
        ),
        lambda item: item["ownership"]["matter_runtime_constraints"].__setitem__(
            "authors_source_laws", True
        ),
        lambda item: item["prepared_state"].__setitem__(
            "identity_sha256", "f" * 64
        ),
        lambda item: item["qualification"].__setitem__(
            "prepared_state_identity_bound", True
        ),
        lambda item: item.__setitem__("boundary", "production qualified"),
    ],
)
def test_rehashed_contract_mutations_fail_against_bound_inputs(mutation) -> None:
    value, base, equality, limit = _compile()
    mutation(value)
    _rehash(value)
    with pytest.raises(compliance.SourceComplianceError):
        compliance.validate_contract(
            value,
            base_manifest=base,
            equality_payload=equality,
            limit_payload=limit,
        )


def test_contract_is_bound_to_one_valid_base_manifest() -> None:
    value, base, equality, limit = _compile()
    other = copy.deepcopy(base)
    other["source_ownership_status"] = "partial"
    other["manifest_sha256"] = digest(
        {key: item for key, item in other.items() if key != "manifest_sha256"}
    )
    knee.validate_manifest(other)
    with pytest.raises(compliance.SourceComplianceError, match="base manifest"):
        compliance.validate_contract(
            value,
            base_manifest=other,
            equality_payload=equality,
            limit_payload=limit,
        )


def test_compile_paths_requires_canonical_regular_inputs(tmp_path: Path) -> None:
    base, equality, limit = _inputs()
    base_path = tmp_path / "base.json"
    equality_path = tmp_path / "equalities.nheq"
    limit_path = tmp_path / "limits.nhlim"
    base_path.write_bytes(canonical_bytes(base) + b"\n")
    equality_path.write_bytes(equality)
    limit_path.write_bytes(limit)
    value = compliance.compile_paths(
        base_manifest_path=base_path,
        equality_path=equality_path,
        limit_path=limit_path,
    )
    assert value["base_manifest"]["manifest_sha256"] == base["manifest_sha256"]
    redirected = tmp_path / "redirected.nheq"
    redirected.symlink_to(equality_path)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="redirected"):
        compliance.compile_paths(
            base_manifest_path=base_path,
            equality_path=redirected,
            limit_path=limit_path,
        )


def test_immutable_writer_is_idempotent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    value, _, _, _ = _compile()
    output = tmp_path / compliance.FILENAME
    first = compliance.write_immutable(output, value)
    second = compliance.write_immutable(output, value)
    assert first == second
    assert output.read_bytes() == canonical_bytes(value) + b"\n"
    changed = copy.deepcopy(value)
    changed["binding_sha256"] = "f" * 64
    with pytest.raises(compliance.SourceComplianceError, match="existing output differs"):
        compliance.write_immutable(output, changed)
    redirected = tmp_path / "redirected.json"
    redirected.symlink_to(output)
    with pytest.raises(compliance.SourceComplianceError, match="redirected"):
        compliance.write_immutable(redirected, value)


def test_strict_json_schema_rejects_mutated_claims() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            knee.ROOT
            / "schemas/humanpack-loaded-anatomy-knee-source-compliance.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    value, _, _, _ = _compile()
    validator.validate(value)
    for mutation in (
        lambda item: item.__setitem__("extra", True),
        lambda item: item["programs"]["joint_equalities"].__setitem__("abi", 1),
        lambda item: item["programs"]["joint_limits"].__setitem__("flags", 0),
        lambda item: item["ownership"]["human_source_laws"].__setitem__(
            "owns_runtime_constraint_state", True
        ),
        lambda item: item["ownership"]["matter_runtime_constraints"].__setitem__(
            "authors_source_laws", True
        ),
        lambda item: item["prepared_state"].__setitem__("status", "present"),
        lambda item: item["qualification"].__setitem__(
            "production_physical_ownership", True
        ),
        lambda item: item.__setitem__("boundary", "clinical claim"),
    ):
        changed = copy.deepcopy(value)
        mutation(changed)
        assert list(validator.iter_errors(changed))
