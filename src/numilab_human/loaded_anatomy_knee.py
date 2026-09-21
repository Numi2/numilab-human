"""Compile the immutable HumanPack left-knee authoring companion receipt.

The receipt binds source anatomy, an authenticated authoring-reference candidate,
population material priors, candidate mass replacement, and ownership identities.
It deliberately does not admit an accepted ``x_current`` value or any production
mechanics claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .open_knee import (
    FACE_STRUCT,
    HEADER_STRUCT,
    MEMBERSHIP_STRUCT,
    NODE_SET_STRUCT,
    NODE_STRUCT,
    REGION_STRUCT,
    SURFACE_PAIR_STRUCT,
    SURFACE_STRUCT,
    TETRAHEDRON_STRUCT,
)
from .ownership import validate_ownership
from .target_coverage import canonical_bytes, digest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.loaded-anatomy-knee.v1"
BINDING_SCHEMA = "HumanPack.loaded-anatomy-knee.binding.v1"
COMPILER = "numilab-human.loaded-anatomy-knee.1"
PROFILE_SCHEMA = "numi.human.loaded-anatomy-knee-authoring.v1"
SOURCE_DEFAULT_PROFILE_SCHEMA = (
    "numi.human.loaded-anatomy-knee-source-default-authoring.v2"
)
LAB_EXPORT_SCHEMA = "numi.lab.loaded-knee-authoring-export.v1"
LAB_MAPPING_ID = (
    "numi-lab.open-knee-restWorld-to-equality-projected-default-body-poses.1"
)
LAB_MAPPING_ALGORITHM = (
    "adaptive-dyadic-first-success-slerp-geodesic-inverse-distance-moving-enthesis.1"
)
SOURCE_DEFAULT_LAB_MAPPING_ID = (
    "numi-lab.open-knee-restWorld-source-default-identity.1"
)
SOURCE_DEFAULT_LAB_MAPPING_ALGORITHM = "identity-copy-source-restWorld-f32.1"
LAB_MAPPING_CODE_IDENTITY_ENCODING = (
    "sha256(domain-utf8||header-file-sha256||core-file-sha256||adapter-file-sha256)"
)
LEGACY_NHEQ1_AUTHORING_EQUALITY_PAYLOAD_SHA256 = (
    "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a"
)
SOURCE_COMPLIANT_NHEQ2_RUNTIME_EQUALITY_PAYLOAD_SHA256 = (
    "12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477"
)
SOURCE_COMPLIANT_EQUALITY_PAYLOAD_ROLE = "source-compliance-runtime-link"
SOURCE_DEFAULT_NHEQ2_DIAGNOSTIC_RESIDUAL_MAXIMUM = 0.052419200539588928
RAW_F32_NODE_MASS_ALGORITHM = (
    "matter-referenced-f32-volume-f32-density-fp64-source-order-accumulate-final-f32.1"
)
RAW_F32_NODE_MASS_ENCODING = (
    "float32-le-lumped-tet-mass-quarter-source-element-order-profile-region-node-order"
)
MAXIMUM_ANCHOR_RESIDUAL_METERS = 2.0e-7
PROFILE = ROOT / "config/loaded-anatomy-knee-left.v1.json"
SOURCE_DEFAULT_PROFILE = (
    ROOT / "config/loaded-anatomy-knee-left-source-default.v2.json"
)
CANONICALIZATION = "utf8-json-sorted-keys-compact-ensure_ascii=false-allow_nan=false"
HASH_EXCLUSION = "top-level manifest_sha256"
LAB_EXPORT_BOUNDARY = (
    "Candidate-only source/projected-reference and donor provenance export. "
    "Projected rest is not an unloaded or stress-free state; population material "
    "priors, prescribed contact coverage, and this export do not establish subject "
    "mechanics, production ownership, sustained tracking, mesh convergence, clinical "
    "validity, or a global seven-owner accepted-state root."
)
SOURCE_DEFAULT_LAB_EXPORT_BOUNDARY = (
    "Candidate-only source-default registered reference and donor provenance export. "
    "B_ref is a byte-identical copy of raw OpenKnee restWorld; NHEQ2 is evaluated "
    "here only as a source-compliance diagnostic and remains a "
    "compliant-acceleration program at runtime, never an authoring projector. This "
    "export does not qualify an unloaded or stress-free state, subject mechanics, "
    "production ownership, sustained tracking, mesh convergence, clinical validity, "
    "or a global seven-owner accepted-state root."
)
REGION_NAMES = ("ACL", "LCL", "MCL", "PCL", "PTL", "QAT")
PASSIVE_NAMES = ("ACL", "LCL", "MCL", "PCL", "PTL")
PAIR_NAMES = (
    "TBC-L_To_FMC",
    "TBC-L_To_MNS-L",
    "PTC_To_FMC",
    "MNS-L_To_FMC",
    "MNS-M_To_TBC-M",
    "MNS-M_To_FMC",
    "TBC-M_To_FMC",
)
PAIR_SOURCE_INDICES = (2, 3, 8, 14, 15, 16, 18)
ROUTE_NAMES = ("recfem_l", "vasint_l", "vaslat_l", "vasmed_l")
EXPECTED_ROUTE_ENDPOINTS = {
    "recfem_l": [
        {
            "role": "load",
            "endpoint_index": 810,
            "endpoint_ordinal": 0,
            "route_node_index": 1798,
            "source_site_index": 1548,
            "body_index": 128,
            "attachment_mode": 2,
            "envelope_index": 622,
            "bone_stable_id": 19,
        },
        {
            "role": "anchor",
            "endpoint_index": 811,
            "endpoint_ordinal": 1,
            "route_node_index": 1803,
            "source_site_index": 1751,
            "body_index": 150,
            "attachment_mode": 2,
            "envelope_index": 623,
            "bone_stable_id": 5,
        },
    ],
    "vasint_l": [
        {
            "role": "load",
            "endpoint_index": 826,
            "endpoint_ordinal": 0,
            "route_node_index": 1830,
            "source_site_index": 1715,
            "body_index": 145,
            "attachment_mode": 0,
            "envelope_index": 0xFFFFFFFF,
            "bone_stable_id": 0,
        },
        {
            "role": "anchor",
            "endpoint_index": 827,
            "endpoint_ordinal": 1,
            "route_node_index": 1835,
            "source_site_index": 1765,
            "body_index": 150,
            "attachment_mode": 2,
            "envelope_index": 638,
            "bone_stable_id": 5,
        },
    ],
    "vaslat_l": [
        {
            "role": "load",
            "endpoint_index": 828,
            "endpoint_ordinal": 0,
            "route_node_index": 1836,
            "source_site_index": 1717,
            "body_index": 145,
            "attachment_mode": 2,
            "envelope_index": 639,
            "bone_stable_id": 3,
        },
        {
            "role": "anchor",
            "endpoint_index": 829,
            "endpoint_ordinal": 1,
            "route_node_index": 1841,
            "source_site_index": 1766,
            "body_index": 150,
            "attachment_mode": 2,
            "envelope_index": 640,
            "bone_stable_id": 5,
        },
    ],
    "vasmed_l": [
        {
            "role": "load",
            "endpoint_index": 830,
            "endpoint_ordinal": 0,
            "route_node_index": 1842,
            "source_site_index": 1719,
            "body_index": 145,
            "attachment_mode": 2,
            "envelope_index": 641,
            "bone_stable_id": 3,
        },
        {
            "role": "anchor",
            "endpoint_index": 831,
            "endpoint_ordinal": 1,
            "route_node_index": 1847,
            "source_site_index": 1767,
            "body_index": 150,
            "attachment_mode": 0,
            "envelope_index": 0xFFFFFFFF,
            "bone_stable_id": 0,
        },
    ],
}
SOURCE_HASH_ORDER = (
    "Geometry.feb",
    "ModelProperties.xml",
    "FeBio_custom.feb",
    "license.txt",
)
BOUNDARY = (
    "Candidate-only left Open Knee authoring receipt. It binds source topology, "
    "projected-rest identity, population material priors, explicit donor subtraction, "
    "and candidate force/state ownership. It does not establish an unloaded or "
    "stress-free reference, prestress equilibrium, subject calibration, an accepted "
    "x_current value, production ownership, loaded motion, clinical validity, or "
    "integrated Human qualification."
)
SOURCE_DEFAULT_BOUNDARY = (
    "Candidate-only left Open Knee authoring receipt. It binds source topology, "
    "source-default restWorld authoring-reference identity, population material "
    "priors, explicit donor subtraction, and candidate force/state ownership. It "
    "does not establish an unloaded or stress-free reference, prestress equilibrium, "
    "subject calibration, an accepted x_current value, production ownership, loaded "
    "motion, clinical validity, or integrated Human qualification."
)
TENDON_HEADER = struct.Struct("<8s10I32s32s32s")
TENDON_ENDPOINT = struct.Struct("<8I8f")
TENDON_ENVELOPE_BYTES = 288
ANCHOR_OWNERSHIP_STRUCT = struct.Struct("<3I3f")
MANIFEST_FILENAME = "HumanPack.loaded-anatomy-knee.v1.json"
BINDING_FILENAME = "HumanPack.loaded-anatomy-knee.binding.v1.json"
FROZEN_PROFILE_FILE_SHA256 = (
    "ec311c48c7ab58dede83da228686d6574d8ebd1c47d22539dbee09967131d3f5"
)
FROZEN_PROFILE_IDENTITY_SHA256 = (
    "c4bbde523018016b94c5e176fd01e519cfbb2a811f84005f964d0718753b6437"
)
FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256 = (
    "fe0ae4a11c928178718cff7874215768683bf498650c8abcc25ced458e0f17b6"
)
FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256 = (
    "f9375445dd7375b9c5b2f4c3eb69ea9454c567b1910b5093b41ec0762280d07f"
)


class LoadedAnatomyKneeError(HumanImportError):
    """The loaded-knee authoring contract cannot be compiled or admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LoadedAnatomyKneeError("HumanPack loaded anatomy knee: " + message)


def _identifier(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and value and value.strip() == value,
        f"{label} must be a canonical identifier",
    )
    _require("\x00" not in value, f"{label} contains a NUL")
    return value


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result >= 0.0, f"{label} must be nonnegative")
    return result


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(not path.is_symlink(), f"{label} is redirected")
    _require(path.is_file(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise LoadedAnatomyKneeError(
            f"HumanPack loaded anatomy knee: {label} is not valid JSON"
        ) from error
    _require(isinstance(value, dict), f"{label} must contain an object")
    encoded = canonical_bytes(value) + b"\n"
    _require(raw == encoded, f"{label} is not canonical JSON plus one LF")
    return value, hashlib.sha256(raw).hexdigest()


def _read_bytes(path: Path, label: str) -> tuple[bytes, str]:
    path = Path(path)
    _require(not path.is_symlink(), f"{label} is redirected")
    _require(path.is_file(), f"{label} is not a regular file")
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _name(raw: bytes, label: str) -> str:
    try:
        value = raw.split(b"\0", 1)[0].decode("ascii")
    except UnicodeError as error:
        raise LoadedAnatomyKneeError(
            f"HumanPack loaded anatomy knee: {label} is not ASCII"
        ) from error
    return _identifier(value, label)


def _vector(value: Any, label: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == 3, f"{label} must have three values"
    )
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _matrix(value: Any, label: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} must be 3 by 3")
    result = [_vector(row, f"{label}[{index}]") for index, row in enumerate(value)]
    for row in range(3):
        for column in range(3):
            _require(
                math.isclose(
                    result[row][column],
                    result[column][row],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                f"{label} must be symmetric",
            )
    return result


def _psd(value: list[list[float]], label: str) -> None:
    scale = max(1.0, *(abs(item) for row in value for item in row))
    tolerance = 1.0e-10 * scale
    _require(
        all(value[index][index] >= -tolerance for index in range(3)),
        f"{label} has a negative principal moment",
    )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        minor = (
            value[first][first] * value[second][second]
            - value[first][second] * value[second][first]
        )
        _require(minor >= -tolerance * scale, f"{label} is not positive semidefinite")
    determinant = (
        value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
    )
    _require(
        determinant >= -tolerance * scale * scale,
        f"{label} is not positive semidefinite",
    )


def _moments(value: Any, label: str) -> dict[str, Any]:
    keys = {"zeroth_mass_kg", "first_mass_moment_kg_m", "raw_second_mass_moment_kg_m2"}
    _require(
        isinstance(value, dict) and set(value) == keys, f"{label} moment fields differ"
    )
    mass = _nonnegative(value["zeroth_mass_kg"], f"{label} mass")
    first = _vector(value["first_mass_moment_kg_m"], f"{label} first moment")
    second = _matrix(
        value["raw_second_mass_moment_kg_m2"], f"{label} raw second moment"
    )
    if mass == 0.0:
        _require(
            all(item == 0.0 for item in first)
            and all(item == 0.0 for row in second for item in row),
            f"{label} zero mass carries spatial moments",
        )
    _psd(second, f"{label} raw second moment")
    if mass > 0.0:
        centered = [
            [
                second[row][column] - first[row] * first[column] / mass
                for column in range(3)
            ]
            for row in range(3)
        ]
        _psd(centered, f"{label} centered second moment")
    return {
        "zeroth_mass_kg": mass,
        "first_mass_moment_kg_m": first,
        "raw_second_mass_moment_kg_m2": second,
    }


def _add_moments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "zeroth_mass_kg": math.fsum(row["zeroth_mass_kg"] for row in rows),
        "first_mass_moment_kg_m": [
            math.fsum(row["first_mass_moment_kg_m"][index] for row in rows)
            for index in range(3)
        ],
        "raw_second_mass_moment_kg_m2": [
            [
                math.fsum(
                    item["raw_second_mass_moment_kg_m2"][axis_row][column]
                    for item in rows
                )
                for column in range(3)
            ]
            for axis_row in range(3)
        ],
    }


def _subtract_moments(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "zeroth_mass_kg": left["zeroth_mass_kg"] - right["zeroth_mass_kg"],
        "first_mass_moment_kg_m": [
            left["first_mass_moment_kg_m"][index]
            - right["first_mass_moment_kg_m"][index]
            for index in range(3)
        ],
        "raw_second_mass_moment_kg_m2": [
            [
                left["raw_second_mass_moment_kg_m2"][row][column]
                - right["raw_second_mass_moment_kg_m2"][row][column]
                for column in range(3)
            ]
            for row in range(3)
        ],
    }


def _close(left: Any, right: Any, tolerance: float) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return abs(float(left) - float(right)) <= tolerance
    return all(_close(a, b, tolerance) for a, b in zip(left, right, strict=True))


LEGACY_PROJECTED_REFERENCE_MODE = "legacy-equality-projected-reference"
SOURCE_DEFAULT_REFERENCE_MODE = "source-default-restWorld-identity-reference"


def _authoring_reference_mode(source: Any) -> str:
    legacy_keys = {
        "nhknee_sha256",
        "x_source_sha256",
        "source_rigid_payload_sha256",
        "equality_payload_sha256",
        "source_model_fingerprint_sha256",
    }
    source_default_keys = {
        *legacy_keys,
        "equality_payload_role",
        "equality_projection_applied",
    }
    if (
        isinstance(source, dict)
        and set(source) == legacy_keys
        and source.get("equality_payload_sha256")
        == LEGACY_NHEQ1_AUTHORING_EQUALITY_PAYLOAD_SHA256
    ):
        return LEGACY_PROJECTED_REFERENCE_MODE
    if (
        isinstance(source, dict)
        and set(source) == source_default_keys
        and source.get("equality_payload_sha256")
        == SOURCE_COMPLIANT_NHEQ2_RUNTIME_EQUALITY_PAYLOAD_SHA256
        and source.get("equality_payload_role")
        == SOURCE_COMPLIANT_EQUALITY_PAYLOAD_ROLE
        and source.get("equality_projection_applied") is False
    ):
        return SOURCE_DEFAULT_REFERENCE_MODE
    raise LoadedAnatomyKneeError(
        "HumanPack loaded anatomy knee: Lab equality role or payload identity differs"
    )


def _profile_contract(schema: str) -> tuple[str, dict[str, Any], str]:
    common = {
        "frame_id": "myosim-world-m",
        "encoding": "float32-le-xyz",
        "source_frame_id": "myosim-world-m",
        "source_body_pose_id": "numi-human:unprojected-myosim-default-body-pose",
        "source_registration_id": (
            "numi-lab.open-knee-oks003-registered-unprojected-default.1"
        ),
        "unloaded_reference_qualified": False,
        "volumetric_prestress_status": "not_applied",
        "prestrain_reset_method": {"status": "unresolved", "method_id": None},
    }
    if schema == PROFILE_SCHEMA:
        return (
            "open-knee-oks003-left-candidate",
            {
                **common,
                "reference_state_class": "projected-rest-candidate",
                "construction_id": (
                    "numi-lab.open-knee-moving-enthesis-projected-rest.1"
                ),
                "reference_body_pose_id": (
                    "numi-human:equality-projected-default-reference-body-pose"
                ),
                "source_to_reference_mapping_id": LAB_MAPPING_ID,
            },
            (
                "Candidate-only left Open Knee authoring contract. Source topology, "
                "projected-rest identity, population material priors, explicit donor "
                "subtraction, and candidate force/state owners do not establish an "
                "unloaded reference, prestress equilibrium, subject calibration, "
                "production ownership, loaded motion, clinical validity, or "
                "integrated Human qualification."
            ),
        )
    if schema == SOURCE_DEFAULT_PROFILE_SCHEMA:
        return (
            "open-knee-oks003-left-source-default-candidate",
            {
                **common,
                "reference_state_class": "source-default-registered-reference",
                "construction_id": SOURCE_DEFAULT_LAB_MAPPING_ID,
                "reference_body_pose_id": common["source_body_pose_id"],
                "source_to_reference_mapping_id": SOURCE_DEFAULT_LAB_MAPPING_ID,
            },
            (
                "Candidate-only left Open Knee source-default authoring contract. "
                "Source topology, source-default restWorld identity, population "
                "material priors, explicit donor subtraction, and candidate "
                "force/state owners do not establish an unloaded reference, "
                "prestress equilibrium, subject calibration, production ownership, "
                "loaded motion, clinical validity, or integrated Human qualification."
            ),
        )
    raise LoadedAnatomyKneeError(
        "HumanPack loaded anatomy knee: authoring profile schema differs"
    )


def _profile_input_identity(profile: dict[str, Any]) -> dict[str, str]:
    if profile["schema"] == PROFILE_SCHEMA:
        return {
            "schema": PROFILE_SCHEMA,
            "file_sha256": FROZEN_PROFILE_FILE_SHA256,
            "identity_sha256": FROZEN_PROFILE_IDENTITY_SHA256,
        }
    _require(
        profile["schema"] == SOURCE_DEFAULT_PROFILE_SCHEMA,
        "authoring profile schema differs",
    )
    return {
        "schema": SOURCE_DEFAULT_PROFILE_SCHEMA,
        "file_sha256": FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256,
        "identity_sha256": FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256,
    }


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "id",
        "subject_id",
        "side",
        "source",
        "topology",
        "topology_identity",
        "reference_state",
        "density_conversion",
        "regions",
        "articular_contact_pairs",
        "quadriceps_routes",
        "tendon_payload",
        "passive_ligaments",
        "donor_policy",
        "full_state_authority",
        "boundary",
    }
    _require(set(profile) == required, "authoring profile fields differ")
    expected_profile_id, expected_reference, expected_boundary = _profile_contract(
        profile.get("schema")
    )
    _require(profile["id"] == expected_profile_id, "profile ID differs")
    _identifier(profile["subject_id"], "subject ID")
    _require(profile["side"] == "left", "only the left source candidate is admitted")
    source = profile["source"]
    _require(
        isinstance(source, dict)
        and set(source) == {"dataset_id", "doi", "license", "files", "nhknee"},
        "source fields differ",
    )
    for key in ("dataset_id", "doi", "license"):
        _identifier(source[key], f"source {key}")
    _require(set(source["files"]) == set(SOURCE_HASH_ORDER), "source file set differs")
    for name, value in source["files"].items():
        _sha256(value, f"source file {name}")
    nhknee = source["nhknee"]
    _require(
        nhknee
        == {
            "abi": 3,
            "bytes": 34357400,
            "magic": "NHKNEE1",
            "sha256": "2e38201528de25911ea496164602ea7e823cd9c2e3c94efa550ee52689324ae5",
        },
        "NHKNEE source identity differs",
    )
    expected_counts = {
        "region_count": 16,
        "node_count": 248236,
        "tetrahedron_count": 844287,
        "surface_count": 88,
        "surface_face_count": 729068,
        "node_set_count": 42,
        "node_set_membership_count": 43260,
        "surface_pair_count": 19,
        "loaded_node_count": 62402,
        "loaded_tetrahedron_count": 264442,
    }
    _require(profile["topology"] == expected_counts, "topology count contract differs")
    topology_identity = profile["topology_identity"]
    _require(
        isinstance(topology_identity, dict)
        and set(topology_identity)
        == {
            "identity_sha256",
            "executable_fem_topology_sha256",
            "source_global_node_index_sha256",
            "anchor_ownership_sha256",
            "regions",
        },
        "topology identity fields differ",
    )
    _require(
        topology_identity["identity_sha256"]
        == "0569112a4d39eccc84bcb81a19a870b690cb2b3d642a74708d41881923f3c349"
        and topology_identity["executable_fem_topology_sha256"]
        == "56af24e63c9a4c2ecafb2c94bab98fa95bbcbc77085f425c84b9c1a95e3a9040"
        and topology_identity["source_global_node_index_sha256"]
        == "e451aa9e773a90dfb0cde77d4fb1555c7f1393c0362c7fbb3614909045a6602b"
        and topology_identity["anchor_ownership_sha256"]
        == "04a5c374f15f8681af40ae2e07fe35258f0bbc805d4d9d8291eae1f891d23de6",
        "topology identity differs from the exact ABI3 source",
    )
    topology_region_keys = {
        "name",
        "index",
        "kind",
        "visual_body_index",
        "first_node",
        "node_count",
        "first_tetrahedron",
        "tetrahedron_count",
        "first_surface",
        "surface_count",
        "material_flags",
        "tetrahedra_sha256",
        "runtime_first_node",
        "runtime_node_count",
        "runtime_first_tetrahedron",
        "runtime_tetrahedron_count",
    }
    _require(
        isinstance(topology_identity["regions"], list)
        and tuple(row.get("name") for row in topology_identity["regions"])
        == REGION_NAMES
        and all(
            set(row) == topology_region_keys for row in topology_identity["regions"]
        ),
        "topology region identity rows differ",
    )
    for row in topology_identity["regions"]:
        _sha256(row["tetrahedra_sha256"], f"{row['name']} tetrahedra hash")
    _require(
        profile["reference_state"] == expected_reference,
        "reference-state boundary differs",
    )
    density = profile["density_conversion"]
    _require(
        density
        == {
            "source_value": 1e-9,
            "source_unit": "tonne_per_mm3",
            "conversion_factor_to_kg_per_m3": 1e12,
            "runtime_value_kg_per_m3": 1000.0,
            "calibration_status": "source_population_prior",
        },
        "density conversion differs",
    )
    regions = profile["regions"]
    _require(
        isinstance(regions, list)
        and tuple(row.get("name") for row in regions) == REGION_NAMES,
        "loaded region order differs",
    )
    for row in regions:
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "name",
                "topology_semantic_id",
                "material_semantic_ids",
                "material",
                "fiber_world",
                "owners",
                "donor_body_semantic_id",
            },
            f"region {row.get('name')} fields differ",
        )
        _identifier(row["topology_semantic_id"], f"{row['name']} topology ID")
        _require(
            isinstance(row["material_semantic_ids"], list)
            and len(row["material_semantic_ids"]) == 2
            and len(set(row["material_semantic_ids"])) == 2,
            f"{row['name']} material IDs differ",
        )
        for semantic_id in row["material_semantic_ids"]:
            _identifier(semantic_id, f"{row['name']} material ID")
        _identifier(row["donor_body_semantic_id"], f"{row['name']} donor ID")
        material = row["material"]
        _require(
            isinstance(material, dict)
            and set(material)
            == {
                "type",
                "c1_mpa",
                "c2_mpa",
                "c3_mpa",
                "c4",
                "c5_mpa",
                "lambda_max",
                "bulk_modulus_mpa",
                "source_initial_stretch",
            }
            and material["type"] == "trans iso Mooney-Rivlin",
            f"{row['name']} material fields differ",
        )
        for key, value in material.items():
            if key != "type":
                _finite(value, f"{row['name']} material {key}")
        fiber = _vector(row["fiber_world"], f"{row['name']} source fiber")
        _require(
            math.isclose(
                math.sqrt(math.fsum(item * item for item in fiber)),
                1.0,
                rel_tol=0.0,
                abs_tol=2.0e-6,
            ),
            f"{row['name']} source fiber is not unit length",
        )
        owners = row["owners"]
        _require(
            isinstance(owners, dict)
            and set(owners)
            == {
                "physical_volume_owner_id",
                "mechanical_mass_owner_id",
                "material_owner_id",
                "active_force_owner_id",
                "state_owner_id",
            },
            f"{row['name']} owner fields differ",
        )
        for key, value in owners.items():
            if key == "active_force_owner_id" and row["name"] != "QAT":
                _require(
                    value is None,
                    f"{row['name']} must use the no-active-force sentinel",
                )
            else:
                _identifier(value, f"{row['name']} {key}")
    pairs = profile["articular_contact_pairs"]
    _require(
        isinstance(pairs, list)
        and tuple(row.get("name") for row in pairs) == PAIR_NAMES,
        "articular pair order differs from the ABI3 source",
    )
    for row in pairs:
        _require(
            set(row) == {"name", "master_surface", "slave_surface"},
            f"contact pair {row.get('name')} fields differ",
        )
        _identifier(row["master_surface"], f"{row['name']} master surface")
        _identifier(row["slave_surface"], f"{row['name']} slave surface")
    routes = profile["quadriceps_routes"]
    _require(
        isinstance(routes, list)
        and tuple(row.get("name") for row in routes) == ROUTE_NAMES,
        "quadriceps route order differs",
    )
    _require(
        [row.get("source_actuator_index") for row in routes] == [405, 413, 414, 415],
        "quadriceps source indices differ",
    )
    _require(
        profile["passive_ligaments"] == list(PASSIVE_NAMES),
        "passive ligament owner set differs",
    )
    tendon = profile["tendon_payload"]
    _require(
        tendon
        == {
            "schema": "numi.human.tendon-attachment-envelope-payload.v3",
            "magic": "NHTENDON3",
            "wire_magic_hex": "4e4854454e443300",
            "abi": 3,
            "bytes": 238288,
            "sha256": "72ca0ec4ef53f647784a89f761e78495d4997c7d72cfe6a1d9eedb24dd97feaa",
            "body_count": 157,
            "muscle_count": 416,
            "site_count": 1815,
            "endpoint_count": 832,
            "envelope_count": 642,
            "point_fallback_count": 190,
            "bone_count": 185,
            "registration_fingerprint32": "6a48e223",
            "myosim_archive_sha256": "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975",
            "myosim_muscle_payload_sha256": "9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05",
            "bodyparts3d_bone_payload_sha256": "f9132393dfb259a80e2b2084ac5ff15f2bf988f7e073f7a961a7ddcfb9febc3c",
        },
        "tendon payload identity differs",
    )
    donor_policy = profile["donor_policy"]
    _require(
        isinstance(donor_policy, dict)
        and set(donor_policy)
        == {
            "id",
            "frame_id",
            "donors",
            "excluded_bodies",
            "qualification_status",
        }
        and donor_policy["id"]
        == "explicit-proximal-segment-candidate-no-visual-body-inference"
        and donor_policy["frame_id"] == "myosim-world-m"
        and donor_policy["qualification_status"]
        == "candidate_not_physiologically_partitioned",
        "donor policy differs",
    )
    _require(
        isinstance(donor_policy["donors"], list) and len(donor_policy["donors"]) == 2,
        "exactly two explicit mass donors are required",
    )
    _require(
        donor_policy["donors"]
        == [
            {
                "semantic_id": "myosim_fullbody:myo_sim/models/leg/assets/"
                "myolegs_chain.xml#/mujocoinclude[1]/body[name=pelvis][1]/"
                "body[name=femur_l][1]",
                "source_body_id": 98,
                "core_body_index": 145,
                "source_mass_kg": 8.4,
                "region_names": ["ACL", "LCL", "MCL", "PCL", "QAT"],
            },
            {
                "semantic_id": "myosim_fullbody:myo_sim/models/leg/assets/"
                "myolegs_chain.xml#/mujocoinclude[1]/body[name=pelvis][1]/"
                "body[name=femur_l][1]/body[name=tibia_l][1]",
                "source_body_id": 99,
                "core_body_index": 150,
                "source_mass_kg": 3.8,
                "region_names": ["PTL"],
            },
        ],
        "donor identities, source masses, or explicit region partition differ",
    )
    _require(
        isinstance(donor_policy["excluded_bodies"], list)
        and len(donor_policy["excluded_bodies"]) == 1
        and donor_policy["excluded_bodies"][0]
        == {
            "semantic_id": "myosim_fullbody:myo_sim/models/leg/assets/"
            "myolegs_chain.xml#/mujocoinclude[1]/body[name=pelvis][1]/"
            "body[name=femur_l][1]/body[name=patella_l][1]",
            "source_body_id": 103,
            "core_body_index": 156,
            "source_mass_kg": 0.02785286,
            "reason": "patella is the attachment and visual body, not an inferred "
            "continuum mass donor",
        },
        "the patella donor exclusion is missing",
    )
    assigned = [
        name for donor in donor_policy["donors"] for name in donor["region_names"]
    ]
    _require(
        sorted(assigned) == sorted(REGION_NAMES)
        and len(assigned) == len(set(assigned)),
        "donor region partition is not exact and disjoint",
    )
    _require(profile["boundary"] == expected_boundary, "profile evidence boundary differs")
    _require(
        profile["full_state_authority"]
        == {
            "semantic_id": "humanpack:loaded-anatomy-knee:left/full-state-authority",
            "owner_id": "numi-lab:human-matter/accepted-step-transaction",
            "schema": "NumiLab.HumanMatterAcceptedState.v1",
            "required_snapshot_components": [
                "articulated-q-v-root-time",
                "articular-contact-history",
                "fem-position-velocity-mass-material-status",
                "passive-ligament-state",
                "solver-adaptive-state",
                "tendon-transfer-replacement-state",
            ],
        },
        "full-state authority contract differs",
    )
    return profile


def _load_frozen_profile() -> dict[str, Any]:
    raw = PROFILE.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == FROZEN_PROFILE_FILE_SHA256,
        "checked-in authoring profile file identity differs",
    )
    try:
        profile = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise LoadedAnatomyKneeError(
            "HumanPack loaded anatomy knee: checked-in authoring profile is invalid"
        ) from error
    _validate_profile(profile)
    _require(
        digest(profile) == FROZEN_PROFILE_IDENTITY_SHA256,
        "checked-in authoring profile content identity differs",
    )
    return profile


def _load_frozen_source_default_profile() -> dict[str, Any]:
    raw = SOURCE_DEFAULT_PROFILE.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest()
        == FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256,
        "checked-in source-default authoring profile file identity differs",
    )
    try:
        profile = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise LoadedAnatomyKneeError(
            "HumanPack loaded anatomy knee: checked-in source-default authoring "
            "profile is invalid"
        ) from error
    _validate_profile(profile)
    _require(
        digest(profile) == FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256,
        "checked-in source-default authoring profile content identity differs",
    )
    return profile


def _load_frozen_profile_for_input(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "authoring profile input identity differs")
    if value == {
        "schema": PROFILE_SCHEMA,
        "file_sha256": FROZEN_PROFILE_FILE_SHA256,
        "identity_sha256": FROZEN_PROFILE_IDENTITY_SHA256,
    }:
        return _load_frozen_profile()
    if value == {
        "schema": SOURCE_DEFAULT_PROFILE_SCHEMA,
        "file_sha256": FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256,
        "identity_sha256": FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256,
    }:
        return _load_frozen_source_default_profile()
    raise LoadedAnatomyKneeError(
        "HumanPack loaded anatomy knee: frozen authoring profile identity differs"
    )


def _load_frozen_profile_for_export(export: Any) -> dict[str, Any]:
    _require(isinstance(export, dict), "Lab authoring export must be an object")
    mode = _authoring_reference_mode(export.get("source"))
    if mode == LEGACY_PROJECTED_REFERENCE_MODE:
        return _load_frozen_profile()
    return _load_frozen_source_default_profile()


def _decode_payload(raw: bytes, profile: dict[str, Any]) -> dict[str, Any]:
    _require(
        len(raw) == profile["source"]["nhknee"]["bytes"], "NHKNEE byte count differs"
    )
    _require(
        hashlib.sha256(raw).hexdigest() == profile["source"]["nhknee"]["sha256"],
        "NHKNEE digest differs",
    )
    _require(len(raw) >= HEADER_STRUCT.size, "NHKNEE header is truncated")
    header = HEADER_STRUCT.unpack_from(raw)
    (
        magic,
        abi,
        header_bytes,
        region_count,
        node_count,
        tet_count,
        surface_count,
        face_count,
        node_set_count,
        membership_count,
        pair_count,
        side,
        reserved,
        source_hashes,
    ) = header
    _require(
        magic == b"NHKNEE1\0" and abi == 3 and header_bytes == HEADER_STRUCT.size,
        "NHKNEE magic, ABI, or header size differs",
    )
    _require(
        side == 0 and reserved == 0, "NHKNEE is not the left reserved-zero payload"
    )
    counts = profile["topology"]
    _require(
        (
            region_count,
            node_count,
            tet_count,
            surface_count,
            face_count,
            node_set_count,
            membership_count,
            pair_count,
        )
        == (
            counts["region_count"],
            counts["node_count"],
            counts["tetrahedron_count"],
            counts["surface_count"],
            counts["surface_face_count"],
            counts["node_set_count"],
            counts["node_set_membership_count"],
            counts["surface_pair_count"],
        ),
        "NHKNEE header topology counts differ",
    )
    expected_source_hashes = b"".join(
        bytes.fromhex(profile["source"]["files"][name]) for name in SOURCE_HASH_ORDER
    )
    _require(
        source_hashes == expected_source_hashes, "NHKNEE embedded source digests differ"
    )
    region_offset = header_bytes
    surface_offset = region_offset + region_count * REGION_STRUCT.size
    node_set_offset = surface_offset + surface_count * SURFACE_STRUCT.size
    pair_offset = node_set_offset + node_set_count * NODE_SET_STRUCT.size
    node_offset = pair_offset + pair_count * SURFACE_PAIR_STRUCT.size
    tet_offset = node_offset + node_count * NODE_STRUCT.size
    face_offset = tet_offset + tet_count * TETRAHEDRON_STRUCT.size
    membership_offset = face_offset + face_count * FACE_STRUCT.size
    expected_bytes = membership_offset + membership_count * MEMBERSHIP_STRUCT.size
    _require(
        expected_bytes == len(raw), "NHKNEE table layout does not consume the payload"
    )

    regions: list[dict[str, Any]] = []
    region_by_name: dict[str, dict[str, Any]] = {}
    for index in range(region_count):
        offset = region_offset + index * REGION_STRUCT.size
        values = REGION_STRUCT.unpack_from(raw, offset)
        row = {
            "index": index,
            "name": _name(values[0], f"region {index} name"),
            "kind": values[1],
            "visual_body_index": values[2],
            "first_node": values[3],
            "node_count": values[4],
            "first_tetrahedron": values[5],
            "tetrahedron_count": values[6],
            "first_surface": values[7],
            "surface_count": values[8],
            "material_values": list(values[9:17]),
            "fiber_world": list(values[17:20]),
            "material_flags": values[20],
            "raw": raw[offset : offset + REGION_STRUCT.size],
        }
        _require(row["name"] not in region_by_name, "NHKNEE region name repeats")
        regions.append(row)
        region_by_name[row["name"]] = row
    _require(
        set(region_by_name)
        == {
            "QAT",
            "TBC-L",
            "PCL",
            "PTC",
            "PTB",
            "ACL",
            "FBB",
            "MCL",
            "PTL",
            "MNS-L",
            "MNS-M",
            "LCL",
            "TBC-M",
            "TBB",
            "FMB",
            "FMC",
        },
        "NHKNEE region set differs",
    )
    _require(
        sum(row["node_count"] for row in regions) == node_count,
        "NHKNEE node partition is incomplete",
    )
    _require(
        sum(row["tetrahedron_count"] for row in regions) == tet_count,
        "NHKNEE tetrahedron partition is incomplete",
    )

    surfaces: list[dict[str, Any]] = []
    for index in range(surface_count):
        offset = surface_offset + index * SURFACE_STRUCT.size
        values = SURFACE_STRUCT.unpack_from(raw, offset)
        surfaces.append(
            {
                "index": index,
                "name": _name(values[0], f"surface {index} name"),
                "region_index": values[1],
                "first_face": values[2],
                "face_count": values[3],
                "flags": values[4],
                "reserved": values[5],
                "raw": raw[offset : offset + SURFACE_STRUCT.size],
            }
        )
    _require(
        len({row["name"] for row in surfaces}) == surface_count,
        "NHKNEE surface names repeat",
    )
    node_sets: list[dict[str, Any]] = []
    for index in range(node_set_count):
        offset = node_set_offset + index * NODE_SET_STRUCT.size
        values = NODE_SET_STRUCT.unpack_from(raw, offset)
        node_sets.append(
            {
                "index": index,
                "name": _name(values[0], f"node set {index} name"),
                "region_index": values[1],
                "first_membership": values[2],
                "membership_count": values[3],
                "anchor_body_index": values[4],
                "flags": values[5],
                "raw": raw[offset : offset + NODE_SET_STRUCT.size],
            }
        )
    _require(
        len({row["name"] for row in node_sets}) == node_set_count,
        "NHKNEE node-set names repeat",
    )
    pairs: list[dict[str, Any]] = []
    for index in range(pair_count):
        offset = pair_offset + index * SURFACE_PAIR_STRUCT.size
        values = SURFACE_PAIR_STRUCT.unpack_from(raw, offset)
        _require(
            values[1] < surface_count and values[2] < surface_count,
            f"NHKNEE pair {index} has an invalid surface index",
        )
        pairs.append(
            {
                "index": index,
                "name": _name(values[0], f"surface pair {index} name"),
                "master_surface": surfaces[values[1]]["name"],
                "slave_surface": surfaces[values[2]]["name"],
                "raw": raw[offset : offset + SURFACE_PAIR_STRUCT.size],
            }
        )
    _require(
        len({row["name"] for row in pairs}) == pair_count,
        "NHKNEE surface-pair names repeat",
    )
    selected_pairs = _select_articular_pairs(pairs, profile)

    node_positions: dict[int, tuple[float, float, float]] = {}
    x_source = bytearray()
    topology_hasher = hashlib.sha256()
    executable_topology_hasher = hashlib.sha256()
    global_node_index_hasher = hashlib.sha256()
    anchor_ownership_hasher = hashlib.sha256()
    global_to_executable_node: dict[int, int] = {}
    runtime_first_node = 0
    runtime_first_tetrahedron = 0
    selected_rows = []
    for specification in profile["regions"]:
        row = region_by_name[specification["name"]]
        _require(
            row["kind"] in {4, 5} and row["tetrahedron_count"] > 0,
            f"{row['name']} is not a loaded continuum region",
        )
        material = specification["material"]
        expected_material = [
            material[key]
            for key in (
                "c1_mpa",
                "c2_mpa",
                "c3_mpa",
                "c4",
                "c5_mpa",
                "lambda_max",
                "bulk_modulus_mpa",
                "source_initial_stretch",
            )
        ]
        _require(
            all(
                math.isclose(
                    float(actual), float(expected), rel_tol=2.0e-6, abs_tol=2.0e-6
                )
                for actual, expected in zip(
                    row["material_values"], expected_material, strict=True
                )
            ),
            f"{row['name']} payload material differs from the source prior",
        )
        _require(
            all(
                math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=2.0e-7
                )
                for actual, expected in zip(
                    row["fiber_world"], specification["fiber_world"], strict=True
                )
            ),
            f"{row['name']} payload fiber differs from the pinned source field",
        )
        topology_hasher.update(row["raw"])
        for node_index in range(
            row["first_node"], row["first_node"] + row["node_count"]
        ):
            offset = node_offset + node_index * NODE_STRUCT.size
            position_bytes = raw[offset : offset + 12]
            position = struct.unpack("<3f", position_bytes)
            _require(
                all(math.isfinite(value) for value in position),
                f"{row['name']} source coordinate is non-finite",
            )
            node_positions[node_index] = position
            global_to_executable_node[node_index] = len(global_to_executable_node)
            global_node_index_hasher.update(struct.pack("<I", node_index))
            node_values = NODE_STRUCT.unpack_from(raw, offset)
            anchor_ownership_hasher.update(
                ANCHOR_OWNERSHIP_STRUCT.pack(
                    node_index,
                    node_values[3],
                    node_values[11],
                    *node_values[8:11],
                )
            )
            x_source.extend(position_bytes)
        tet_start = tet_offset + row["first_tetrahedron"] * TETRAHEDRON_STRUCT.size
        tet_bytes = raw[
            tet_start : tet_start + row["tetrahedron_count"] * TETRAHEDRON_STRUCT.size
        ]
        topology_hasher.update(tet_bytes)
        for tetrahedron_index in range(
            row["first_tetrahedron"],
            row["first_tetrahedron"] + row["tetrahedron_count"],
        ):
            indices = TETRAHEDRON_STRUCT.unpack_from(
                raw,
                tet_offset + tetrahedron_index * TETRAHEDRON_STRUCT.size,
            )
            first_node = row["first_node"]
            last_node = first_node + row["node_count"]
            _require(
                all(first_node <= node < last_node for node in indices),
                f"{row['name']} source tetrahedron leaves its region interval",
            )
            executable_topology_hasher.update(
                TETRAHEDRON_STRUCT.pack(
                    *(global_to_executable_node[node] for node in indices)
                )
            )
        owned_surfaces = surfaces[
            row["first_surface"] : row["first_surface"] + row["surface_count"]
        ]
        for surface in owned_surfaces:
            _require(
                surface["region_index"] == row["index"],
                f"{row['name']} surface partition crosses regions",
            )
            topology_hasher.update(surface["raw"])
            start = face_offset + surface["first_face"] * FACE_STRUCT.size
            topology_hasher.update(
                raw[start : start + surface["face_count"] * FACE_STRUCT.size]
            )
        owned_sets = [
            item for item in node_sets if item["region_index"] == row["index"]
        ]
        for node_set in owned_sets:
            topology_hasher.update(node_set["raw"])
            start = (
                membership_offset
                + node_set["first_membership"] * MEMBERSHIP_STRUCT.size
            )
            topology_hasher.update(
                raw[
                    start : start
                    + node_set["membership_count"] * MEMBERSHIP_STRUCT.size
                ]
            )
        selected_rows.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "index",
                        "name",
                        "kind",
                        "visual_body_index",
                        "first_node",
                        "node_count",
                        "first_tetrahedron",
                        "tetrahedron_count",
                        "first_surface",
                        "surface_count",
                        "fiber_world",
                        "material_flags",
                    )
                },
                "surface_names": [item["name"] for item in owned_surfaces],
                "node_set_names": [item["name"] for item in owned_sets],
                "tetrahedra_sha256": hashlib.sha256(tet_bytes).hexdigest(),
                "runtime_first_node": runtime_first_node,
                "runtime_node_count": row["node_count"],
                "runtime_first_tetrahedron": runtime_first_tetrahedron,
                "runtime_tetrahedron_count": row["tetrahedron_count"],
            }
        )
        runtime_first_node += row["node_count"]
        runtime_first_tetrahedron += row["tetrahedron_count"]
    for pair in selected_pairs:
        topology_hasher.update(pair["raw"])
    _require(
        sum(row["node_count"] for row in selected_rows) == counts["loaded_node_count"],
        "loaded node count differs",
    )
    _require(
        sum(row["tetrahedron_count"] for row in selected_rows)
        == counts["loaded_tetrahedron_count"],
        "loaded tetrahedron count differs",
    )
    identity_keys = (
        "name",
        "index",
        "kind",
        "visual_body_index",
        "first_node",
        "node_count",
        "first_tetrahedron",
        "tetrahedron_count",
        "first_surface",
        "surface_count",
        "material_flags",
        "tetrahedra_sha256",
        "runtime_first_node",
        "runtime_node_count",
        "runtime_first_tetrahedron",
        "runtime_tetrahedron_count",
    )
    region_identities = [
        {key: row[key] for key in identity_keys} for row in selected_rows
    ]
    expected_identity = profile["topology_identity"]
    _require(
        topology_hasher.hexdigest() == expected_identity["identity_sha256"]
        and executable_topology_hasher.hexdigest()
        == expected_identity["executable_fem_topology_sha256"]
        and global_node_index_hasher.hexdigest()
        == expected_identity["source_global_node_index_sha256"]
        and anchor_ownership_hasher.hexdigest()
        == expected_identity["anchor_ownership_sha256"]
        and region_identities == expected_identity["regions"],
        "NHKNEE selected topology identity differs from the frozen profile",
    )
    return {
        "counts": counts,
        "regions": selected_rows,
        "region_identities": region_identities,
        "region_by_name": region_by_name,
        "node_positions": node_positions,
        "x_source_bytes": bytes(x_source),
        "topology_sha256": topology_hasher.hexdigest(),
        "executable_fem_topology_sha256": executable_topology_hasher.hexdigest(),
        "source_global_node_index_sha256": global_node_index_hasher.hexdigest(),
        "anchor_ownership_sha256": anchor_ownership_hasher.hexdigest(),
        "pairs": selected_pairs,
        "tet_offset": tet_offset,
        "raw": raw,
    }


def _select_articular_pairs(
    pairs: list[dict[str, Any]], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = profile["articular_contact_pairs"]
    selected = [row for row in pairs if row.get("name") in set(PAIR_NAMES)]
    _require(
        len(selected) == len(PAIR_NAMES)
        and tuple(row.get("index") for row in selected) == PAIR_SOURCE_INDICES
        and tuple(row.get("name") for row in selected) == PAIR_NAMES
        and [
            {
                "name": row.get("name"),
                "master_surface": row.get("master_surface"),
                "slave_surface": row.get("slave_surface"),
            }
            for row in selected
        ]
        == expected,
        "NHKNEE selected articular pairs differ from exact ABI3 source order",
    )
    return selected


def _decode_tendon(
    raw: bytes, profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = profile["tendon_payload"]
    _require(
        len(raw) == expected["bytes"]
        and hashlib.sha256(raw).hexdigest() == expected["sha256"],
        "tendon payload byte identity differs",
    )
    _require(len(raw) >= TENDON_HEADER.size, "tendon payload header is truncated")
    header = TENDON_HEADER.unpack_from(raw)
    (
        magic,
        abi,
        body_count,
        muscle_count,
        site_count,
        endpoint_count,
        envelope_count,
        bone_count,
        registration_fingerprint32,
        reserved0,
        reserved1,
        myosim_archive_sha256,
        myosim_muscle_payload_sha256,
        bodyparts3d_bone_payload_sha256,
    ) = header
    _require(
        magic.hex() == expected["wire_magic_hex"]
        and abi == expected["abi"]
        and body_count == expected["body_count"]
        and muscle_count == expected["muscle_count"]
        and site_count == expected["site_count"]
        and endpoint_count == expected["endpoint_count"]
        and envelope_count == expected["envelope_count"]
        and endpoint_count - envelope_count == expected["point_fallback_count"]
        and reserved0 == 0
        and reserved1 == 0,
        "tendon payload header differs",
    )
    _require(
        len(raw)
        == TENDON_HEADER.size
        + endpoint_count * TENDON_ENDPOINT.size
        + envelope_count * TENDON_ENVELOPE_BYTES,
        "tendon payload table layout differs",
    )
    _require(
        bone_count == expected["bone_count"]
        and f"{registration_fingerprint32:08x}"
        == expected["registration_fingerprint32"]
        and myosim_archive_sha256.hex() == expected["myosim_archive_sha256"]
        and myosim_muscle_payload_sha256.hex()
        == expected["myosim_muscle_payload_sha256"]
        and bodyparts3d_bone_payload_sha256.hex()
        == expected["bodyparts3d_bone_payload_sha256"],
        "tendon payload embedded source identity differs",
    )
    result = []
    for route in profile["quadriceps_routes"]:
        emitted = []
        for role in ("load", "anchor"):
            index = route[f"{role}_endpoint_index"]
            _require(
                type(index) is int and 0 <= index < endpoint_count,
                f"{route['name']} {role} endpoint index is invalid",
            )
            endpoint = TENDON_ENDPOINT.unpack_from(
                raw, TENDON_HEADER.size + index * TENDON_ENDPOINT.size
            )
            (
                muscle_index,
                endpoint_ordinal,
                route_node_index,
                source_site_index,
                body_index,
                mode,
                envelope_index,
                bone_stable_id,
            ) = endpoint[:8]
            expected_ordinal = 0 if role == "load" else 1
            _require(
                muscle_index == route["source_actuator_index"]
                and endpoint_ordinal == expected_ordinal
                and route_node_index == route[f"{role}_route_node_index"]
                and source_site_index == route[f"{role}_source_site_index"]
                and body_index == route[f"{role}_body_index"]
                and mode in {0, 2, 3}
                and (
                    (mode == 0 and envelope_index == 0xFFFFFFFF)
                    or (mode != 0 and envelope_index < envelope_count)
                ),
                f"{route['name']} {role} tendon endpoint identity differs",
            )
            emitted.append(
                {
                    "role": role,
                    "endpoint_index": index,
                    "endpoint_ordinal": endpoint_ordinal,
                    "route_node_index": route_node_index,
                    "source_site_index": source_site_index,
                    "body_index": body_index,
                    "attachment_mode": mode,
                    "envelope_index": envelope_index,
                    "bone_stable_id": bone_stable_id,
                }
            )
        result.append({"name": route["name"], "endpoints": emitted})
        _require(
            emitted == EXPECTED_ROUTE_ENDPOINTS[route["name"]],
            f"{route['name']} current NHTENDON3 endpoint rows differ",
        )
    return result, {
        "schema": expected["schema"],
        "magic": expected["magic"],
        "wire_magic_hex": magic.hex(),
        "abi": abi,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "body_count": body_count,
        "muscle_count": muscle_count,
        "site_count": site_count,
        "endpoint_count": endpoint_count,
        "envelope_count": envelope_count,
        "point_fallback_count": endpoint_count - envelope_count,
        "bone_count": bone_count,
        "registration_fingerprint32": f"{registration_fingerprint32:08x}",
        "myosim_archive_sha256": myosim_archive_sha256.hex(),
        "myosim_muscle_payload_sha256": myosim_muscle_payload_sha256.hex(),
        "bodyparts3d_bone_payload_sha256": bodyparts3d_bone_payload_sha256.hex(),
    }


def _ownership_record(ownership: dict[str, Any], semantic_id: str) -> dict[str, Any]:
    rows = [row for row in ownership["records"] if row["semantic_id"] == semantic_id]
    _require(len(rows) == 1, f"ownership record is absent or repeated: {semantic_id}")
    return rows[0]


def _candidate_owner(record: dict[str, Any], role: str, owner_id: str) -> None:
    _require(
        record["owners"][role] == {"status": "candidate", "owner_id": owner_id},
        f"{record['semantic_id']} {role} owner differs",
    )


def _bind_ownership(ownership: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    validate_ownership(ownership)
    represented: list[str] = []
    for region in profile["regions"]:
        record = _ownership_record(ownership, region["topology_semantic_id"])
        represented.extend(record["coverage_leaf_sha256s"])
        owners = region["owners"]
        for role, key in (
            ("physical_volume", "physical_volume_owner_id"),
            ("mechanical_mass", "mechanical_mass_owner_id"),
            ("state", "state_owner_id"),
        ):
            _candidate_owner(record, role, owners[key])
        if region["name"] == "QAT":
            _candidate_owner(record, "active_force", owners["active_force_owner_id"])
        else:
            _require(
                record["owners"]["active_force"]
                == {"status": "unresolved", "owner_id": None},
                f"{region['name']} passive topology gained an active-force owner",
            )
        for semantic_id in region["material_semantic_ids"]:
            material_record = _ownership_record(ownership, semantic_id)
            represented.extend(material_record["coverage_leaf_sha256s"])
            _candidate_owner(material_record, "material", owners["material_owner_id"])
    for pair in profile["articular_contact_pairs"]:
        semantic_id = (
            "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/"
            f"SurfacePair[name={pair['name']}][1]"
        )
        record = _ownership_record(ownership, semantic_id)
        represented.extend(record["coverage_leaf_sha256s"])
        _candidate_owner(
            record,
            "state",
            f"humanpack:loaded-anatomy-knee:left/contact/{pair['name']}/state",
        )
    for route in profile["quadriceps_routes"]:
        semantic_id = f"myosim_fullbody:composed/myofullbody/muscles/{route['name']}"
        record = _ownership_record(ownership, semantic_id)
        represented.extend(record["coverage_leaf_sha256s"])
        _require(
            record["action"]
            == {
                "status": "source",
                "semantic_action_id": semantic_id + "/action/stimulation",
                "source_index": route["source_actuator_index"],
            },
            f"{route['name']} source action identity differs",
        )
        _require(
            record["force_semantics"]
            == {
                "status": "candidate",
                "mode": "replacement",
                "replaces_owner_ids": [semantic_id + "/owner/source-jt"],
            },
            f"{route['name']} source force replacement differs",
        )
        _candidate_owner(
            record,
            "active_force",
            "humanpack:loaded-anatomy-knee:left/QAT/active-force",
        )
        _candidate_owner(
            record, "state", "humanpack:loaded-anatomy-knee:left/full-state"
        )
    for donor in profile["donor_policy"]["donors"]:
        record = _ownership_record(ownership, donor["semantic_id"])
        represented.extend(record["coverage_leaf_sha256s"])
        body_name = {98: "femur_l", 99: "tibia_l"}.get(donor["source_body_id"])
        _require(body_name is not None, "donor source body has no frozen mass owner")
        owner_id = f"humanpack:source-rigid-body-mass/{body_name}"
        _candidate_owner(record, "mechanical_mass", owner_id)
    _require(
        len(represented) == len(set(represented)),
        "companion semantic scope repeats a coverage leaf",
    )
    return sorted(represented)


def _parse_x_ref(
    raw: bytes, decoded: dict[str, Any], profile: dict[str, Any]
) -> dict[int, tuple[float, float, float]]:
    count = profile["topology"]["loaded_node_count"]
    _require(
        len(raw) == count * 12, "x_ref byte count differs from the loaded node scope"
    )
    result: dict[int, tuple[float, float, float]] = {}
    cursor = 0
    for specification in profile["regions"]:
        region = decoded["region_by_name"][specification["name"]]
        for global_index in range(
            region["first_node"], region["first_node"] + region["node_count"]
        ):
            value = struct.unpack_from("<3f", raw, cursor)
            _require(
                all(math.isfinite(item) for item in value),
                "x_ref contains a non-finite value",
            )
            result[global_index] = value
            cursor += 12
    _require(
        cursor == len(raw) and len(result) == count, "x_ref node scope is incomplete"
    )
    return result


def _signed_tetrahedron_determinant(
    vertices: list[tuple[float, float, float]],
) -> float:
    a, b, c, d = vertices
    ab = [b[index] - a[index] for index in range(3)]
    ac = [c[index] - a[index] for index in range(3)]
    ad = [d[index] - a[index] for index in range(3)]
    return (
        ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
        - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
    )


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _matter_referenced_tet_node_mass(
    volume_m3: float, density_kg_per_m3: float
) -> float:
    """Match the referenced Matter compiler's stored-volume mass arithmetic."""
    stored_volume = _float32(volume_m3)
    stored_density = _float32(density_kg_per_m3)
    return float(stored_density) * float(stored_volume) * 0.25


def _tet_moments(
    vertices: list[tuple[float, float, float]],
    source_vertices: list[tuple[float, float, float]],
    density: float,
) -> dict[str, Any]:
    source_determinant = _signed_tetrahedron_determinant(source_vertices)
    determinant = _signed_tetrahedron_determinant(vertices)
    _require(
        math.isfinite(source_determinant) and source_determinant != 0.0,
        "x_source has a degenerate tetrahedron",
    )
    jacobian_determinant = determinant / source_determinant
    _require(
        math.isfinite(jacobian_determinant) and jacobian_determinant > 0.0,
        "x_ref inverted or collapsed a source tetrahedron",
    )
    volume = abs(determinant) / 6.0
    _require(
        volume > 0.0 and math.isfinite(volume), "x_ref has a degenerate tetrahedron"
    )
    mass = density * volume
    total = [math.fsum(vertex[index] for vertex in vertices) for index in range(3)]
    first = [mass * item / 4.0 for item in total]
    second = [
        [
            mass
            / 20.0
            * (
                total[row] * total[column]
                + math.fsum(vertex[row] * vertex[column] for vertex in vertices)
            )
            for column in range(3)
        ]
        for row in range(3)
    ]
    return {
        "volume_m3": volume,
        "zeroth_mass_kg": mass,
        "first_mass_moment_kg_m": first,
        "raw_second_mass_moment_kg_m2": second,
        "jacobian_determinant": jacobian_determinant,
    }


def _region_moments(
    decoded: dict[str, Any],
    x_ref: dict[int, tuple[float, float, float]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    density = profile["density_conversion"]["runtime_value_kg_per_m3"]
    rows = []
    node_mass_hasher = hashlib.sha256()
    for specification in profile["regions"]:
        region = decoded["region_by_name"][specification["name"]]
        tets = []
        node_masses = {
            node: 0.0
            for node in range(
                region["first_node"], region["first_node"] + region["node_count"]
            )
        }
        for index in range(
            region["first_tetrahedron"],
            region["first_tetrahedron"] + region["tetrahedron_count"],
        ):
            indices = TETRAHEDRON_STRUCT.unpack_from(
                decoded["raw"], decoded["tet_offset"] + index * TETRAHEDRON_STRUCT.size
            )
            first_node = region["first_node"]
            last_node = first_node + region["node_count"]
            _require(
                all(first_node <= node < last_node for node in indices),
                f"{region['name']} tetrahedron leaves its region node scope",
            )
            tetrahedron = _tet_moments(
                [x_ref[node] for node in indices],
                [decoded["node_positions"][node] for node in indices],
                density,
            )
            tets.append(tetrahedron)
            node_mass_contribution = _matter_referenced_tet_node_mass(
                tetrahedron["volume_m3"],
                density,
            )
            for node in indices:
                node_masses[node] += node_mass_contribution
        moments = {
            "zeroth_mass_kg": math.fsum(item["zeroth_mass_kg"] for item in tets),
            "first_mass_moment_kg_m": [
                math.fsum(item["first_mass_moment_kg_m"][axis] for item in tets)
                for axis in range(3)
            ],
            "raw_second_mass_moment_kg_m2": [
                [
                    math.fsum(
                        item["raw_second_mass_moment_kg_m2"][row][column]
                        for item in tets
                    )
                    for column in range(3)
                ]
                for row in range(3)
            ],
        }
        _moments(moments, f"{region['name']} continuum")
        region_node_mass_bytes = b"".join(
            struct.pack("<f", node_masses[node])
            for node in range(
                region["first_node"], region["first_node"] + region["node_count"]
            )
        )
        node_mass_hasher.update(region_node_mass_bytes)
        rows.append(
            {
                "name": region["name"],
                "volume_m3": math.fsum(item["volume_m3"] for item in tets),
                **moments,
                "minimum_jacobian_determinant": min(
                    item["jacobian_determinant"] for item in tets
                ),
                "maximum_jacobian_determinant": max(
                    item["jacobian_determinant"] for item in tets
                ),
                "raw_f32_node_mass_sha256": hashlib.sha256(
                    region_node_mass_bytes
                ).hexdigest(),
            }
        )
    maximum_displacement = max(
        math.sqrt(
            math.fsum(
                (x_ref[node][axis] - decoded["node_positions"][node][axis]) ** 2
                for axis in range(3)
            )
        )
        for node in x_ref
    )
    return rows, {
        "maximum_displacement_m": maximum_displacement,
        "minimum_jacobian_determinant": min(
            row["minimum_jacobian_determinant"] for row in rows
        ),
        "maximum_jacobian_determinant": max(
            row["maximum_jacobian_determinant"] for row in rows
        ),
        "raw_f32_node_mass_sha256": node_mass_hasher.hexdigest(),
    }


def _compile_mass_partition(
    export: dict[str, Any],
    profile: dict[str, Any],
    regions: list[dict[str, Any]],
    x_ref_sha256: str,
    x_source_sha256: str,
    decoded: dict[str, Any],
    geometry_diagnostics: dict[str, float],
) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "manifest_canonicalization",
        "manifest_hash_exclusion",
        "manifest_sha256",
        "subject_id",
        "side",
        "x_ref",
        "source",
        "poses",
        "mapping",
        "donor_moments",
        "boundary",
    }
    _require(
        isinstance(export, dict)
        and set(export) == required
        and export["schema"] == LAB_EXPORT_SCHEMA
        and export["status"] == "candidate"
        and export["manifest_canonicalization"] == CANONICALIZATION
        and export["manifest_hash_exclusion"] == HASH_EXCLUSION,
        "Lab authoring export envelope differs",
    )
    _sha256(export["manifest_sha256"], "Lab authoring export hash")
    _require(
        export["manifest_sha256"]
        == digest(
            {key: value for key, value in export.items() if key != "manifest_sha256"}
        ),
        "Lab authoring export hash mismatch",
    )
    _require(
        export["subject_id"] == profile["subject_id"]
        and export["side"] == profile["side"],
        "Lab authoring export subject or side differs",
    )
    reference_mode = _authoring_reference_mode(export.get("source"))
    expected_export_boundary = (
        LAB_EXPORT_BOUNDARY
        if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
        else SOURCE_DEFAULT_LAB_EXPORT_BOUNDARY
    )
    _require(
        export["boundary"] == expected_export_boundary,
        "Lab authoring export evidence boundary differs",
    )
    x_ref = export["x_ref"]
    _require(
        isinstance(x_ref, dict)
        and set(x_ref)
        == {
            "bytes",
            "file_sha256",
            "encoding",
            "node_count",
            "region_order",
            "local_order",
            "region_spans",
            "source_global_node_index_sha256",
            "source_global_node_index_encoding",
            "executable_fem_topology_sha256",
            "executable_fem_topology_encoding",
            "anchor_ownership_sha256",
            "anchor_ownership_encoding",
        }
        and x_ref["bytes"] == profile["topology"]["loaded_node_count"] * 12
        and x_ref["file_sha256"] == x_ref_sha256
        and x_ref["encoding"] == "float32-le-xyz"
        and x_ref["node_count"] == profile["topology"]["loaded_node_count"]
        and x_ref["region_order"] == list(REGION_NAMES)
        and x_ref["local_order"] == "source-local-node-order",
        "Lab x_ref artifact identity or ordering differs",
    )
    expected_spans = [
        {
            key: row[key]
            for key in (
                "name",
                "first_node",
                "node_count",
                "first_tetrahedron",
                "tetrahedron_count",
                "runtime_first_node",
                "runtime_node_count",
                "runtime_first_tetrahedron",
                "runtime_tetrahedron_count",
            )
        }
        for row in decoded["regions"]
    ]
    _require(
        x_ref["region_spans"] == expected_spans
        and x_ref["source_global_node_index_sha256"]
        == decoded["source_global_node_index_sha256"]
        and x_ref["source_global_node_index_encoding"]
        == "uint32-le-source-global-node-indices-profile-region-order"
        and x_ref["executable_fem_topology_sha256"]
        == decoded["executable_fem_topology_sha256"]
        and x_ref["executable_fem_topology_encoding"]
        == "uint32-le-tetrahedra-profile-region-order-source-element-order-"
        "concatenated-local-node-indices"
        and x_ref["anchor_ownership_sha256"] == decoded["anchor_ownership_sha256"]
        and x_ref["anchor_ownership_encoding"]
        == "uint32-le-source-global-node,anchor-body,flags;float32-le-anchor-local-xyz",
        "Lab x_ref span, node-map, executable topology, or anchor binding differs",
    )
    source = export["source"]
    expected_profile_schema = (
        PROFILE_SCHEMA
        if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
        else SOURCE_DEFAULT_PROFILE_SCHEMA
    )
    _require(
        isinstance(source, dict)
        and profile["schema"] == expected_profile_schema
        and source["nhknee_sha256"] == profile["source"]["nhknee"]["sha256"]
        and source["x_source_sha256"] == x_source_sha256
        and source["source_rigid_payload_sha256"]
        == "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44",
        "Lab source provenance or authoring profile mode differs",
    )
    _sha256(source["source_model_fingerprint_sha256"], "source model fingerprint")
    poses = export["poses"]
    _require(
        isinstance(poses, dict)
        and set(poses)
        == {
            "source_default",
            "projected_reference",
        },
        "Lab pose identities differ",
    )
    for key, expected_id in (
        ("source_default", profile["reference_state"]["source_body_pose_id"]),
        ("projected_reference", profile["reference_state"]["reference_body_pose_id"]),
    ):
        pose = poses[key]
        _require(
            isinstance(pose, dict)
            and set(pose) == {"id", "identity_sha256"}
            and pose["id"] == expected_id,
            f"Lab {key} pose identity differs",
        )
        _sha256(pose["identity_sha256"], f"Lab {key} pose hash")
    mapping = export["mapping"]
    expected_mapping_algorithm = (
        LAB_MAPPING_ALGORITHM
        if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
        else SOURCE_DEFAULT_LAB_MAPPING_ALGORITHM
    )
    _require(
        isinstance(mapping, dict)
        and set(mapping)
        == {
            "id",
            "algorithm",
            "code_identity_sha256",
            "code_identity_encoding",
            "diagnostics",
        }
        and mapping["id"] == profile["reference_state"]["source_to_reference_mapping_id"]
        and mapping["algorithm"] == expected_mapping_algorithm
        and mapping["code_identity_encoding"] == LAB_MAPPING_CODE_IDENTITY_ENCODING,
        "Lab A-to-B_ref mapping identity differs",
    )
    _sha256(mapping["code_identity_sha256"], "Lab mapping code identity")
    diagnostics = mapping["diagnostics"]
    _require(
        isinstance(diagnostics, dict)
        and set(diagnostics)
        == {
            "finite",
            "source_node_count",
            "output_node_count",
            "maximum_displacement_m",
            "equality_residual_maximum",
            "jacobian",
            "raw_f32_node_mass_sha256",
            "raw_f32_node_mass_algorithm",
            "regions",
        }
        and diagnostics["finite"] is True
        and diagnostics["source_node_count"] == profile["topology"]["loaded_node_count"]
        and diagnostics["output_node_count"] == profile["topology"]["loaded_node_count"]
        and diagnostics["raw_f32_node_mass_algorithm"] == RAW_F32_NODE_MASS_ALGORITHM,
        "Lab mapping diagnostics differ",
    )
    mapping_maximum_displacement = _nonnegative(
        diagnostics["maximum_displacement_m"], "mapping maximum displacement"
    )
    equality_residual_maximum = _nonnegative(
        diagnostics["equality_residual_maximum"], "mapping equality residual"
    )
    _sha256(diagnostics["raw_f32_node_mass_sha256"], "Lab raw f32 node-mass hash")
    jacobian = diagnostics["jacobian"]
    _require(
        isinstance(jacobian, dict)
        and set(jacobian)
        == {
            "finite",
            "minimum_determinant",
            "maximum_determinant",
            "orientation_preserving",
        }
        and jacobian["finite"] is True
        and jacobian["orientation_preserving"] is True,
        "Lab mapping Jacobian diagnostics differ",
    )
    minimum_determinant = _finite(
        jacobian["minimum_determinant"],
        "minimum mapping Jacobian determinant",
    )
    maximum_determinant = _finite(
        jacobian["maximum_determinant"],
        "maximum mapping Jacobian determinant",
    )
    _require(
        0.0 < minimum_determinant <= maximum_determinant,
        "Lab mapping Jacobian is not orientation preserving",
    )
    if reference_mode == SOURCE_DEFAULT_REFERENCE_MODE:
        _require(
            x_ref_sha256 == x_source_sha256
            and poses["projected_reference"] == poses["source_default"]
            and mapping_maximum_displacement == 0.0
            and equality_residual_maximum
            == SOURCE_DEFAULT_NHEQ2_DIAGNOSTIC_RESIDUAL_MAXIMUM
            and minimum_determinant == 1.0
            and maximum_determinant == 1.0,
            "source-default A-to-B_ref identity evidence differs",
        )
    _require(
        math.isclose(
            diagnostics["maximum_displacement_m"],
            geometry_diagnostics["maximum_displacement_m"],
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            minimum_determinant,
            geometry_diagnostics["minimum_jacobian_determinant"],
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            maximum_determinant,
            geometry_diagnostics["maximum_jacobian_determinant"],
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ),
        "Lab mapping diagnostics differ from independently recomputed A-to-B_ref geometry",
    )
    _require(
        diagnostics["raw_f32_node_mass_sha256"]
        == geometry_diagnostics["raw_f32_node_mass_sha256"],
        "Lab raw f32 node-mass identity differs from Human recomputation",
    )
    mapping_regions = diagnostics["regions"]
    _require(
        isinstance(mapping_regions, list)
        and len(mapping_regions) == len(REGION_NAMES)
        and all(isinstance(row, dict) for row in mapping_regions)
        and [row.get("name") for row in mapping_regions] == list(REGION_NAMES),
        "Lab mapping region diagnostics order differs",
    )
    if reference_mode == SOURCE_DEFAULT_REFERENCE_MODE:
        expected_direct_status = {name: ("accepted", 1) for name in REGION_NAMES}
    else:
        expected_direct_status = {
            "ACL": ("accepted", 1),
            "LCL": ("accepted", 1),
            "MCL": ("accepted", 1),
            "PCL": ("accepted", 1),
            "PTL": ("rejected_inversion", 2),
            "QAT": ("accepted", 1),
        }
    region_moments_by_name = {row["name"]: row for row in regions}
    topology_by_name = {
        row["name"]: row for row in profile["topology_identity"]["regions"]
    }
    row_minima = []
    row_maxima = []
    for row in mapping_regions:
        name = row["name"]
        _require(
            set(row)
            == {
                "name",
                "substep_count",
                "direct_map_status",
                "direct_failure_tetrahedron",
                "direct_failure_jacobian",
                "minimum_jacobian",
                "maximum_jacobian",
                "maximum_source_anchor_reconstruction_residual_m",
                "maximum_anchor_residual_m",
                "maximum_persisted_f32_anchor_residual_m",
            },
            f"{name} mapping region diagnostic fields differ",
        )
        substeps = row["substep_count"]
        _require(
            isinstance(substeps, int)
            and not isinstance(substeps, bool)
            and 1 <= substeps <= 256
            and substeps & (substeps - 1) == 0,
            f"{name} continuation substep count is invalid",
        )
        expected_status, expected_substeps = expected_direct_status[name]
        _require(
            row["direct_map_status"] == expected_status
            and substeps == expected_substeps,
            f"{name} direct-map status or continuation count differs",
        )
        if expected_status == "accepted":
            _require(
                row["direct_failure_tetrahedron"] is None
                and row["direct_failure_jacobian"] is None,
                f"{name} accepted direct map carries failure evidence",
            )
        else:
            failure_tetrahedron = row["direct_failure_tetrahedron"]
            _require(
                isinstance(failure_tetrahedron, int)
                and not isinstance(failure_tetrahedron, bool)
                and failure_tetrahedron == 419
                and failure_tetrahedron < topology_by_name[name]["tetrahedron_count"],
                f"{name} direct-map failure tetrahedron differs",
            )
            failure_jacobian = _finite(
                row["direct_failure_jacobian"],
                f"{name} direct-map failure Jacobian",
            )
            _require(
                failure_jacobian < 0.0,
                f"{name} rejected direct map is not an inversion",
            )
        minimum = _finite(row["minimum_jacobian"], f"{name} minimum Jacobian")
        maximum = _finite(row["maximum_jacobian"], f"{name} maximum Jacobian")
        _require(
            0.0 < minimum <= maximum,
            f"{name} persisted-f32 Jacobian range is invalid",
        )
        if reference_mode == SOURCE_DEFAULT_REFERENCE_MODE:
            _require(
                minimum == 1.0 and maximum == 1.0,
                f"{name} source-default identity Jacobians differ",
            )
        authored = region_moments_by_name[name]
        _require(
            math.isclose(
                minimum,
                authored["minimum_jacobian_determinant"],
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                maximum,
                authored["maximum_jacobian_determinant"],
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ),
            f"{name} mapping Jacobians differ from persisted-f32 Human recomputation",
        )
        row_minima.append(minimum)
        row_maxima.append(maximum)
        for field in (
            "maximum_source_anchor_reconstruction_residual_m",
            "maximum_anchor_residual_m",
            "maximum_persisted_f32_anchor_residual_m",
        ):
            residual = _nonnegative(row[field], f"{name} {field}")
            _require(
                residual <= MAXIMUM_ANCHOR_RESIDUAL_METERS,
                f"{name} {field} exceeds the authoring gate",
            )
    _require(
        math.isclose(
            minimum_determinant,
            min(row_minima),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            maximum_determinant,
            max(row_maxima),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ),
        "aggregate mapping Jacobians differ from the persisted-f32 region rows",
    )
    donor_source = export["donor_moments"]
    _require(
        isinstance(donor_source, dict)
        and set(donor_source)
        == {
            "policy_id",
            "frame_id",
            "source_kind",
            "donors",
        }
        and donor_source["policy_id"] == profile["donor_policy"]["id"]
        and donor_source["frame_id"] == profile["donor_policy"]["frame_id"]
        and donor_source["source_kind"] == "cooked-myosim-rigid-body-mass-properties",
        "Lab donor moment source contract differs",
    )
    donors = donor_source["donors"]
    _require(
        isinstance(donors, list) and len(donors) == 2,
        "donor moment source must contain exactly two donors",
    )
    _require(
        [row.get("semantic_id") for row in donors]
        == [row["semantic_id"] for row in profile["donor_policy"]["donors"]],
        "donor moment source order differs from the frozen donor policy",
    )
    region_by_name = {row["name"]: row for row in regions}
    output = []
    for expected in profile["donor_policy"]["donors"]:
        matches = [
            row for row in donors if row.get("semantic_id") == expected["semantic_id"]
        ]
        _require(len(matches) == 1, f"donor moments missing: {expected['semantic_id']}")
        row = matches[0]
        _require(
            set(row)
            == {
                "semantic_id",
                "source_body_id",
                "core_body_index",
                "moments",
            }
            and row["source_body_id"] == expected["source_body_id"]
            and row["core_body_index"] == expected["core_body_index"],
            f"donor identity differs: {expected['semantic_id']}",
        )
        source_moments = _moments(row["moments"], f"{expected['semantic_id']} source")
        expected_source_mass = _float32(expected["source_mass_kg"])
        _require(
            source_moments["zeroth_mass_kg"] == expected_source_mass,
            f"donor source mass differs: {expected['semantic_id']}",
        )
        subtracted = _add_moments(
            [region_by_name[name] for name in expected["region_names"]]
        )
        remaining = _subtract_moments(source_moments, subtracted)
        _moments(remaining, f"{expected['semantic_id']} remaining")
        computed_closure = _subtract_moments(
            source_moments,
            _add_moments([subtracted, remaining]),
        )
        _require(
            _close(computed_closure["zeroth_mass_kg"], 0.0, 1.0e-12)
            and _close(computed_closure["first_mass_moment_kg_m"], [0.0] * 3, 1.0e-12)
            and _close(
                computed_closure["raw_second_mass_moment_kg_m2"],
                [[0.0] * 3] * 3,
                1.0e-12,
            ),
            f"donor moment subtraction does not close: {expected['semantic_id']}",
        )
        closure = {
            "zeroth_mass_kg": 0.0,
            "first_mass_moment_kg_m": [0.0, 0.0, 0.0],
            "raw_second_mass_moment_kg_m2": [[0.0, 0.0, 0.0] for _ in range(3)],
        }
        output.append(
            {
                "semantic_id": expected["semantic_id"],
                "source_body_id": expected["source_body_id"],
                "core_body_index": expected["core_body_index"],
                "region_names": expected["region_names"],
                "source": source_moments,
                "subtracted": subtracted,
                "remaining": remaining,
                "closure_residual": closure,
            }
        )
    _require(
        len({row["semantic_id"] for row in donors}) == 2,
        "donor moment source has an unknown or repeated donor",
    )
    result = {
        "policy_id": donor_source["policy_id"],
        "frame_id": donor_source["frame_id"],
        "source_kind": donor_source["source_kind"],
        "density_kg_per_m3": 1000.0,
        "provenance": {
            "lab_export_manifest_sha256": export["manifest_sha256"],
            "x_ref_sha256": x_ref_sha256,
            "x_source_sha256": x_source_sha256,
            **copy.deepcopy(source),
            "poses": copy.deepcopy(poses),
            "mapping": copy.deepcopy(mapping),
        },
        "regions": regions,
        "donors": output,
        "raw_f32_node_mass": {
            "sha256": geometry_diagnostics["raw_f32_node_mass_sha256"],
            "algorithm": RAW_F32_NODE_MASS_ALGORITHM,
            "encoding": RAW_F32_NODE_MASS_ENCODING,
            "node_count": profile["topology"]["loaded_node_count"],
        },
        "excluded_bodies": profile["donor_policy"]["excluded_bodies"],
        "qualification_status": profile["donor_policy"]["qualification_status"],
    }
    result["identity_sha256"] = digest(result)
    return result


def compile_manifest(
    *,
    ownership: dict[str, Any],
    profile: dict[str, Any],
    payload_bytes: bytes,
    tendon_payload_bytes: bytes,
    x_ref_bytes: bytes,
    lab_export: dict[str, Any],
) -> dict[str, Any]:
    """Compile a deterministic authoring receipt from already loaded inputs."""
    profile = _validate_profile(profile)
    reference_mode = _authoring_reference_mode(lab_export.get("source"))
    expected_profile_schema = (
        PROFILE_SCHEMA
        if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
        else SOURCE_DEFAULT_PROFILE_SCHEMA
    )
    _require(
        profile["schema"] == expected_profile_schema,
        "Lab equality role and frozen authoring profile are not correlated",
    )
    represented = _bind_ownership(ownership, profile)
    decoded = _decode_payload(payload_bytes, profile)
    endpoint_bindings, tendon_identity = _decode_tendon(tendon_payload_bytes, profile)
    x_ref = _parse_x_ref(x_ref_bytes, decoded, profile)
    region_moments, geometry_diagnostics = _region_moments(decoded, x_ref, profile)
    x_ref_sha256 = hashlib.sha256(x_ref_bytes).hexdigest()
    x_source_sha256 = hashlib.sha256(decoded["x_source_bytes"]).hexdigest()
    if reference_mode == SOURCE_DEFAULT_REFERENCE_MODE:
        _require(
            x_ref_bytes == decoded["x_source_bytes"],
            "source-default x_ref bytes differ from raw ABI3 restWorld",
        )
    mass_partition = _compile_mass_partition(
        lab_export,
        profile,
        region_moments,
        x_ref_sha256,
        x_source_sha256,
        decoded,
        geometry_diagnostics,
    )
    hashes = {
        "ownership": hashlib.sha256(canonical_bytes(ownership) + b"\n").hexdigest(),
        "profile": hashlib.sha256(canonical_bytes(profile) + b"\n").hexdigest(),
        "open_knee_payload": hashlib.sha256(payload_bytes).hexdigest(),
        "tendon_payload": hashlib.sha256(tendon_payload_bytes).hexdigest(),
        "x_ref": hashlib.sha256(x_ref_bytes).hexdigest(),
        "lab_export": hashlib.sha256(canonical_bytes(lab_export) + b"\n").hexdigest(),
    }
    for name, value in hashes.items():
        _sha256(value, f"{name} input hash")
    region_payload_by_name = {row["name"]: row for row in decoded["region_identities"]}
    region_moment_by_name = {row["name"]: row for row in region_moments}
    regions = []
    for specification in profile["regions"]:
        name = specification["name"]
        active = (
            {
                "status": "candidate",
                "owner_id": specification["owners"]["active_force_owner_id"],
            }
            if name == "QAT"
            else {"status": "none", "owner_id": None}
        )
        regions.append(
            {
                "name": name,
                "topology_semantic_id": specification["topology_semantic_id"],
                "material_semantic_ids": specification["material_semantic_ids"],
                "payload": region_payload_by_name[name],
                "material": {
                    **specification["material"],
                    "fiber_world": specification["fiber_world"],
                    "calibration_status": "source_population_prior",
                },
                "owners": {
                    "physical_volume": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["physical_volume_owner_id"],
                    },
                    "mechanical_mass": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["mechanical_mass_owner_id"],
                    },
                    "material": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["material_owner_id"],
                    },
                    "active_force": active,
                    "state": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["state_owner_id"],
                    },
                },
                "donor_body_semantic_id": specification["donor_body_semantic_id"],
                "projected_reference_moments": region_moment_by_name[name],
            }
        )
    pairs = []
    for pair in profile["articular_contact_pairs"]:
        semantic_id = (
            "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/"
            f"SurfacePair[name={pair['name']}][1]"
        )
        pairs.append(
            {
                **pair,
                "semantic_id": semantic_id,
                "state_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/contact/{pair['name']}/state",
                },
            }
        )
    replacements = []
    endpoint_by_name = {row["name"]: row["endpoints"] for row in endpoint_bindings}
    for route in profile["quadriceps_routes"]:
        semantic_id = f"myosim_fullbody:composed/myofullbody/muscles/{route['name']}"
        replacements.append(
            {
                "name": route["name"],
                "source_actuator_index": route["source_actuator_index"],
                "route_semantic_id": semantic_id,
                "semantic_action_id": semantic_id + "/action/stimulation",
                "replaces_owner_id": semantic_id + "/owner/source-jt",
                "replacement_owner_id": "humanpack:loaded-anatomy-knee:left/QAT/active-force",
                "replacement_scale": 1.0,
                "status": "candidate",
                "endpoints": endpoint_by_name[route["name"]],
            }
        )
    authority = dict(profile["full_state_authority"])
    authority["identity_sha256"] = digest(profile["full_state_authority"])
    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "manifest_canonicalization": CANONICALIZATION,
        "manifest_hash_exclusion": HASH_EXCLUSION,
        "manifest_sha256": "",
        "status": "candidate",
        "source_ownership_status": ownership["status"],
        "subject_id": profile["subject_id"],
        "side": profile["side"],
        "ownership_manifest_sha256": ownership["manifest_sha256"],
        "inputs": {
            "ownership": {
                "schema": ownership["schema"],
                "file_sha256": hashes["ownership"],
                "identity_sha256": ownership["manifest_sha256"],
            },
            "authoring_profile": _profile_input_identity(profile),
            "open_knee_payload": {
                "schema": "numi.human.open-knee-oks003-payload.v3",
                "file_sha256": hashes["open_knee_payload"],
                "identity_sha256": hashes["open_knee_payload"],
            },
            "tendon_payload": {
                "schema": tendon_identity["schema"],
                "file_sha256": hashes["tendon_payload"],
                "identity_sha256": hashes["tendon_payload"],
            },
            "x_ref": {
                "schema": "numi.human.loaded-anatomy-knee-x-ref-f32le.v1",
                "file_sha256": hashes["x_ref"],
                "identity_sha256": hashes["x_ref"],
            },
            "lab_export": {
                "schema": lab_export["schema"],
                "file_sha256": hashes["lab_export"],
                "identity_sha256": lab_export["manifest_sha256"],
            },
        },
        "semantic_scope": {
            "ownership_manifest_sha256": ownership["manifest_sha256"],
            "coverage_leaf_sha256s": represented,
        },
        "lab_authoring_export": copy.deepcopy(lab_export),
        "source": {
            **profile["source"],
            "tendon_payload": tendon_identity,
            "material_and_density_status": "source_population_prior",
        },
        "topology": {
            **profile["topology"],
            "identity_sha256": decoded["topology_sha256"],
            "executable_fem_topology_sha256": decoded["executable_fem_topology_sha256"],
            "executable_fem_topology_encoding": "uint32-le-tetrahedra-profile-region-order-source-element-order-"
            "concatenated-local-node-indices",
            "source_global_node_index_sha256": decoded[
                "source_global_node_index_sha256"
            ],
            "source_global_node_index_encoding": "uint32-le-source-global-node-indices-profile-region-order",
            "anchor_ownership_sha256": decoded["anchor_ownership_sha256"],
            "anchor_ownership_encoding": "uint32-le-source-global-node,anchor-body,flags;"
            "float32-le-anchor-local-xyz",
            "region_spans": [
                {
                    key: row[key]
                    for key in (
                        "name",
                        "first_node",
                        "node_count",
                        "first_tetrahedron",
                        "tetrahedron_count",
                        "runtime_first_node",
                        "runtime_node_count",
                        "runtime_first_tetrahedron",
                        "runtime_tetrahedron_count",
                    )
                }
                for row in decoded["regions"]
            ],
            "ordering": "authoring-profile-region-order-then-source-local-order",
        },
        "coordinates": {
            "x_source": {
                "scope": "six-loaded-regions",
                "encoding": "float32-le-xyz",
                "node_count": profile["topology"]["loaded_node_count"],
                "sha256": hashlib.sha256(decoded["x_source_bytes"]).hexdigest(),
                "frame_id": profile["reference_state"]["source_frame_id"],
                "body_pose_id": profile["reference_state"]["source_body_pose_id"],
                "registration_id": profile["reference_state"]["source_registration_id"],
            },
            "x_ref": {
                "scope": "six-loaded-regions",
                "encoding": "float32-le-xyz",
                "node_count": profile["topology"]["loaded_node_count"],
                "sha256": x_ref_sha256,
                "mapping_identity_sha256": digest(
                    {
                        "x_source_sha256": hashlib.sha256(
                            decoded["x_source_bytes"]
                        ).hexdigest(),
                        "source_frame_id": profile["reference_state"][
                            "source_frame_id"
                        ],
                        "source_body_pose_id": profile["reference_state"][
                            "source_body_pose_id"
                        ],
                        "source_registration_id": profile["reference_state"][
                            "source_registration_id"
                        ],
                        "x_ref_sha256": x_ref_sha256,
                        "construction_id": profile["reference_state"][
                            "construction_id"
                        ],
                        "reference_body_pose_id": profile["reference_state"][
                            "reference_body_pose_id"
                        ],
                        "source_to_reference_mapping_id": profile["reference_state"][
                            "source_to_reference_mapping_id"
                        ],
                    }
                ),
                **profile["reference_state"],
            },
            "x_current": {
                "scope": "six-loaded-regions-in-authoring-profile-order",
                "encoding": "float32-le-xyz",
                "hash_algorithm": "sha256",
                "accepted_sha256": None,
                "acceptance_receipt_sha256": None,
                "authority": "separate-lab-runtime-acceptance-receipt",
            },
        },
        "density_conversion": profile["density_conversion"],
        "regions": regions,
        "mass_partition": mass_partition,
        "articular_contact_pairs": pairs,
        "active_force_replacements": replacements,
        "passive_ligament_owners": [
            {
                "name": name,
                "semantic_id": f"humanpack:loaded-anatomy-knee:left/region/{name}/passive-force",
                "active_force": {"status": "none", "owner_id": None},
                "passive_force_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/region/{name}/material",
                },
            }
            for name in PASSIVE_NAMES
        ],
        "full_state_authority": authority,
        "qualification": {
            "candidate_only": True,
            "source_identity_bound": True,
            "ownership_identity_bound": True,
            "topology_identity_bound": True,
            "projected_reference_identity_bound": True,
            "unloaded_reference_qualified": False,
            "prestrain_reference_reset_executed": False,
            "subject_material_calibrated": False,
            "donor_mass_and_raw_moments_subtracted": True,
            "mesh_convergence_qualified": False,
            "specimen_load_validation_qualified": False,
            "clinical_validity_qualified": False,
            "production_physical_ownership": False,
            "production_active_force": False,
            "runtime_x_current_accepted": False,
            "integrated_human_qualification": False,
        },
        "boundary": (
            BOUNDARY
            if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
            else SOURCE_DEFAULT_BOUNDARY
        ),
    }
    result["manifest_sha256"] = digest(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )
    validate_manifest(result)
    return result


def validate_manifest(value: dict[str, Any]) -> None:
    """Validate immutable identity and the permanent candidate-only boundary."""
    required = {
        "schema",
        "compiler",
        "manifest_canonicalization",
        "manifest_hash_exclusion",
        "manifest_sha256",
        "status",
        "source_ownership_status",
        "subject_id",
        "side",
        "ownership_manifest_sha256",
        "inputs",
        "semantic_scope",
        "source",
        "topology",
        "lab_authoring_export",
        "coordinates",
        "density_conversion",
        "regions",
        "mass_partition",
        "articular_contact_pairs",
        "active_force_replacements",
        "passive_ligament_owners",
        "full_state_authority",
        "qualification",
        "boundary",
    }
    _require(
        isinstance(value, dict) and set(value) == required, "receipt fields differ"
    )
    _require(
        value["schema"] == SCHEMA and value["compiler"] == COMPILER,
        "receipt schema or compiler differs",
    )
    _require(
        value["manifest_canonicalization"] == CANONICALIZATION
        and value["manifest_hash_exclusion"] == HASH_EXCLUSION,
        "canonicalization contract differs",
    )
    _sha256(value["manifest_sha256"], "manifest hash")
    _require(
        value["manifest_sha256"]
        == digest(
            {key: item for key, item in value.items() if key != "manifest_sha256"}
        ),
        "manifest hash mismatch",
    )
    _require(
        value["status"] == "candidate"
        and value["source_ownership_status"] in {"partial", "blocked"},
        "scoped candidate or source ownership status is invalid",
    )
    _sha256(value["ownership_manifest_sha256"], "ownership identity")
    inputs = value["inputs"]
    _require(
        isinstance(inputs, dict)
        and set(inputs)
        == {
            "ownership",
            "authoring_profile",
            "open_knee_payload",
            "tendon_payload",
            "x_ref",
            "lab_export",
        },
        "input identity set differs",
    )
    expected_input_schemas = {
        "ownership": "HumanPack.ownership.v1",
        "open_knee_payload": "numi.human.open-knee-oks003-payload.v3",
        "tendon_payload": "numi.human.tendon-attachment-envelope-payload.v3",
        "x_ref": "numi.human.loaded-anatomy-knee-x-ref-f32le.v1",
        "lab_export": LAB_EXPORT_SCHEMA,
    }
    for name, item in inputs.items():
        _require(
            isinstance(item, dict)
            and set(item)
            == {
                "schema",
                "file_sha256",
                "identity_sha256",
            }
            and (
                name == "authoring_profile"
                or item["schema"] == expected_input_schemas[name]
            ),
            f"{name} input identity fields differ",
        )
        _sha256(item["file_sha256"], f"{name} file hash")
        _sha256(item["identity_sha256"], f"{name} identity hash")
    profile = _load_frozen_profile_for_input(inputs["authoring_profile"])
    expected_boundary = (
        BOUNDARY
        if profile["schema"] == PROFILE_SCHEMA
        else SOURCE_DEFAULT_BOUNDARY
    )
    _require(
        value["subject_id"] == profile["subject_id"]
        and value["side"] == profile["side"]
        and value["boundary"] == expected_boundary,
        "subject, side, or evidence boundary differs",
    )
    _require(
        inputs["open_knee_payload"]["file_sha256"]
        == inputs["open_knee_payload"]["identity_sha256"]
        == profile["source"]["nhknee"]["sha256"]
        and inputs["tendon_payload"]["file_sha256"]
        == inputs["tendon_payload"]["identity_sha256"]
        == profile["tendon_payload"]["sha256"]
        and inputs["x_ref"]["file_sha256"] == inputs["x_ref"]["identity_sha256"],
        "frozen profile or binary input identities differ",
    )
    _require(
        inputs["ownership"]["identity_sha256"] == value["ownership_manifest_sha256"],
        "ownership identity is not bound",
    )
    lab_export = value["lab_authoring_export"]
    _require(
        isinstance(lab_export, dict)
        and lab_export.get("schema") == LAB_EXPORT_SCHEMA
        and lab_export.get("status") == "candidate"
        and lab_export.get("manifest_canonicalization") == CANONICALIZATION
        and lab_export.get("manifest_hash_exclusion") == HASH_EXCLUSION,
        "embedded Lab authoring export envelope differs",
    )
    reference_mode = _authoring_reference_mode(lab_export.get("source"))
    expected_profile_schema = (
        PROFILE_SCHEMA
        if reference_mode == LEGACY_PROJECTED_REFERENCE_MODE
        else SOURCE_DEFAULT_PROFILE_SCHEMA
    )
    _require(
        profile["schema"] == expected_profile_schema,
        "Lab equality role and frozen authoring profile are not correlated",
    )
    _sha256(lab_export.get("manifest_sha256"), "embedded Lab export identity")
    _require(
        lab_export["manifest_sha256"]
        == digest(
            {key: item for key, item in lab_export.items() if key != "manifest_sha256"}
        )
        == inputs["lab_export"]["identity_sha256"]
        and hashlib.sha256(canonical_bytes(lab_export) + b"\n").hexdigest()
        == inputs["lab_export"]["file_sha256"],
        "embedded Lab export file or content identity differs",
    )
    scope = value["semantic_scope"]
    _require(
        isinstance(scope, dict)
        and set(scope)
        == {
            "ownership_manifest_sha256",
            "coverage_leaf_sha256s",
        }
        and scope["ownership_manifest_sha256"] == value["ownership_manifest_sha256"]
        and isinstance(scope["coverage_leaf_sha256s"], list)
        and bool(scope["coverage_leaf_sha256s"])
        and scope["coverage_leaf_sha256s"]
        == sorted(set(scope["coverage_leaf_sha256s"])),
        "semantic ownership scope is invalid",
    )
    for leaf in scope["coverage_leaf_sha256s"]:
        _sha256(leaf, "semantic-scope coverage leaf")
    _require(
        value["source"]
        == {
            **profile["source"],
            "tendon_payload": profile["tendon_payload"],
            "material_and_density_status": "source_population_prior",
        },
        "source or payload identity table differs from the frozen profile",
    )
    topology_regions = profile["topology_identity"]["regions"]
    expected_spans = [
        {
            key: row[key]
            for key in (
                "name",
                "first_node",
                "node_count",
                "first_tetrahedron",
                "tetrahedron_count",
                "runtime_first_node",
                "runtime_node_count",
                "runtime_first_tetrahedron",
                "runtime_tetrahedron_count",
            )
        }
        for row in topology_regions
    ]
    expected_topology = {
        **profile["topology"],
        "identity_sha256": profile["topology_identity"]["identity_sha256"],
        "executable_fem_topology_sha256": profile["topology_identity"][
            "executable_fem_topology_sha256"
        ],
        "executable_fem_topology_encoding": "uint32-le-tetrahedra-profile-region-order-source-element-order-"
        "concatenated-local-node-indices",
        "source_global_node_index_sha256": profile["topology_identity"][
            "source_global_node_index_sha256"
        ],
        "source_global_node_index_encoding": "uint32-le-source-global-node-indices-profile-region-order",
        "anchor_ownership_sha256": profile["topology_identity"][
            "anchor_ownership_sha256"
        ],
        "anchor_ownership_encoding": "uint32-le-source-global-node,anchor-body,flags;float32-le-anchor-local-xyz",
        "region_spans": expected_spans,
        "ordering": "authoring-profile-region-order-then-source-local-order",
    }
    _require(
        value["topology"] == expected_topology,
        "topology, executable order, or anchor identity differs",
    )
    _require(
        value["density_conversion"] == profile["density_conversion"],
        "density conversion differs from the source prior",
    )
    _require(
        tuple(row.get("name") for row in value["regions"]) == REGION_NAMES,
        "receipt region set or order differs",
    )
    topology_by_name = {row["name"]: row for row in topology_regions}
    profile_by_name = {row["name"]: row for row in profile["regions"]}
    for row in value["regions"]:
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "name",
                "topology_semantic_id",
                "material_semantic_ids",
                "payload",
                "material",
                "owners",
                "donor_body_semantic_id",
                "projected_reference_moments",
            },
            f"{row.get('name')} receipt region fields differ",
        )
        specification = profile_by_name[row["name"]]
        expected_active = (
            {
                "status": "candidate",
                "owner_id": specification["owners"]["active_force_owner_id"],
            }
            if row["name"] == "QAT"
            else {"status": "none", "owner_id": None}
        )
        expected_owners = {
            "physical_volume": {
                "status": "candidate",
                "owner_id": specification["owners"]["physical_volume_owner_id"],
            },
            "mechanical_mass": {
                "status": "candidate",
                "owner_id": specification["owners"]["mechanical_mass_owner_id"],
            },
            "material": {
                "status": "candidate",
                "owner_id": specification["owners"]["material_owner_id"],
            },
            "active_force": expected_active,
            "state": {
                "status": "candidate",
                "owner_id": specification["owners"]["state_owner_id"],
            },
        }
        _require(
            row["topology_semantic_id"] == specification["topology_semantic_id"]
            and row["material_semantic_ids"] == specification["material_semantic_ids"]
            and row["payload"] == topology_by_name[row["name"]]
            and row["material"]
            == {
                **specification["material"],
                "fiber_world": specification["fiber_world"],
                "calibration_status": "source_population_prior",
            }
            and row["owners"] == expected_owners
            and row["donor_body_semantic_id"]
            == specification["donor_body_semantic_id"],
            f"{row['name']} frozen topology, material, or owner binding differs",
        )
        active = row["owners"]["active_force"]
        if row["name"] == "QAT":
            _require(
                active.get("status") == "candidate" and active.get("owner_id"),
                "QAT active-force owner is missing",
            )
        else:
            _require(
                active == {"status": "none", "owner_id": None},
                f"{row['name']} no-active-force sentinel differs",
            )
        projected = row["projected_reference_moments"]
        _require(
            isinstance(projected, dict)
            and set(projected)
            == {
                "name",
                "volume_m3",
                "zeroth_mass_kg",
                "first_mass_moment_kg_m",
                "raw_second_mass_moment_kg_m2",
                "minimum_jacobian_determinant",
                "maximum_jacobian_determinant",
                "raw_f32_node_mass_sha256",
            }
            and projected["name"] == row["name"]
            and _nonnegative(projected["volume_m3"], f"{row['name']} volume") > 0.0
            and 0.0
            < _finite(
                projected["minimum_jacobian_determinant"],
                f"{row['name']} minimum Jacobian",
            )
            <= _finite(
                projected["maximum_jacobian_determinant"],
                f"{row['name']} maximum Jacobian",
            ),
            f"{row['name']} authoring-reference moment fields differ",
        )
        _sha256(
            projected["raw_f32_node_mass_sha256"],
            f"{row['name']} raw f32 node-mass hash",
        )
        _moments(
            {
                key: row["projected_reference_moments"][key]
                for key in (
                    "zeroth_mass_kg",
                    "first_mass_moment_kg_m",
                    "raw_second_mass_moment_kg_m2",
                )
            },
            f"{row['name']} authoring-reference moments",
        )
    _require(
        tuple(row.get("name") for row in value["articular_contact_pairs"])
        == PAIR_NAMES,
        "receipt articular pair order differs",
    )
    for row, specification in zip(
        value["articular_contact_pairs"],
        profile["articular_contact_pairs"],
        strict=True,
    ):
        semantic_id = (
            "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/"
            f"SurfacePair[name={specification['name']}][1]"
        )
        _require(
            row
            == {
                **specification,
                "semantic_id": semantic_id,
                "state_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/contact/"
                    f"{specification['name']}/state",
                },
            },
            f"{specification['name']} contact binding differs",
        )
    _require(
        tuple(row.get("name") for row in value["active_force_replacements"])
        == ROUTE_NAMES,
        "receipt quadriceps replacement set or order differs",
    )
    for row, route in zip(
        value["active_force_replacements"], profile["quadriceps_routes"], strict=True
    ):
        semantic_id = f"myosim_fullbody:composed/myofullbody/muscles/{route['name']}"
        _require(
            row
            == {
                "name": route["name"],
                "source_actuator_index": route["source_actuator_index"],
                "route_semantic_id": semantic_id,
                "semantic_action_id": semantic_id + "/action/stimulation",
                "replaces_owner_id": semantic_id + "/owner/source-jt",
                "replacement_owner_id": "humanpack:loaded-anatomy-knee:left/QAT/active-force",
                "replacement_scale": 1.0,
                "status": "candidate",
                "endpoints": EXPECTED_ROUTE_ENDPOINTS[route["name"]],
            },
            f"{route['name']} replacement is not the exact candidate replacement",
        )
    _require(
        tuple(row.get("name") for row in value["passive_ligament_owners"])
        == PASSIVE_NAMES
        and all(
            row["active_force"] == {"status": "none", "owner_id": None}
            and row["semantic_id"]
            == f"humanpack:loaded-anatomy-knee:left/region/{row['name']}/passive-force"
            for row in value["passive_ligament_owners"]
        ),
        "passive ligament owner table differs",
    )
    for row in value["passive_ligament_owners"]:
        _require(
            row
            == {
                "name": row["name"],
                "semantic_id": f"humanpack:loaded-anatomy-knee:left/region/{row['name']}/passive-force",
                "active_force": {"status": "none", "owner_id": None},
                "passive_force_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/region/"
                    f"{row['name']}/material",
                },
            },
            f"{row['name']} passive owner binding differs",
        )
    coordinates = value["coordinates"]
    _require(
        isinstance(coordinates, dict)
        and set(coordinates)
        == {
            "x_source",
            "x_ref",
            "x_current",
        },
        "coordinate identity set differs",
    )
    x_source = coordinates["x_source"]
    _require(
        isinstance(x_source, dict)
        and set(x_source)
        == {
            "scope",
            "encoding",
            "node_count",
            "sha256",
            "frame_id",
            "body_pose_id",
            "registration_id",
        },
        "x_source identity fields differ",
    )
    _sha256(x_source["sha256"], "x_source hash")
    _require(
        x_source
        == {
            "scope": "six-loaded-regions",
            "encoding": "float32-le-xyz",
            "node_count": profile["topology"]["loaded_node_count"],
            "sha256": x_source["sha256"],
            "frame_id": "myosim-world-m",
            "body_pose_id": "numi-human:unprojected-myosim-default-body-pose",
            "registration_id": "numi-lab.open-knee-oks003-registered-unprojected-default.1",
        },
        "x_source pose, frame, registration, or scope identity differs",
    )
    x_ref = coordinates["x_ref"]
    expected_x_ref_keys = {
        "scope",
        "node_count",
        "sha256",
        "mapping_identity_sha256",
        *profile["reference_state"].keys(),
    }
    _require(
        isinstance(x_ref, dict) and set(x_ref) == expected_x_ref_keys,
        "x_ref identity fields differ",
    )
    _sha256(x_ref["sha256"], "x_ref hash")
    _sha256(x_ref["mapping_identity_sha256"], "x_ref mapping identity")
    _require(
        x_ref
        == {
            "scope": "six-loaded-regions",
            "node_count": profile["topology"]["loaded_node_count"],
            "sha256": x_ref["sha256"],
            "mapping_identity_sha256": x_ref["mapping_identity_sha256"],
            **profile["reference_state"],
        },
        "authoring reference identity or candidate boundary differs",
    )
    expected_mapping_identity = digest(
        {
            "x_source_sha256": coordinates["x_source"]["sha256"],
            "source_frame_id": coordinates["x_source"]["frame_id"],
            "source_body_pose_id": coordinates["x_source"]["body_pose_id"],
            "source_registration_id": coordinates["x_source"]["registration_id"],
            "x_ref_sha256": coordinates["x_ref"]["sha256"],
            "construction_id": coordinates["x_ref"]["construction_id"],
            "reference_body_pose_id": coordinates["x_ref"]["reference_body_pose_id"],
            "source_to_reference_mapping_id": coordinates["x_ref"][
                "source_to_reference_mapping_id"
            ],
        }
    )
    _require(
        coordinates["x_ref"]["mapping_identity_sha256"] == expected_mapping_identity,
        "A-to-B_ref mapping identity differs",
    )
    _require(
        coordinates["x_current"]
        == {
            "scope": "six-loaded-regions-in-authoring-profile-order",
            "encoding": "float32-le-xyz",
            "hash_algorithm": "sha256",
            "accepted_sha256": None,
            "acceptance_receipt_sha256": None,
            "authority": "separate-lab-runtime-acceptance-receipt",
        },
        "authoring receipt contains or changes runtime x_current authority",
    )
    _require(
        inputs["x_ref"]["file_sha256"] == x_ref["sha256"]
        and lab_export["x_ref"]["file_sha256"] == x_ref["sha256"]
        and lab_export["source"]["x_source_sha256"] == x_source["sha256"],
        "coordinate bytes are not bound to their input and Lab export identities",
    )
    mass = value["mass_partition"]
    _require(isinstance(mass, dict), "mass partition must be an object")
    projected_rows = [row["projected_reference_moments"] for row in value["regions"]]
    provenance_mapping = mass.get("provenance", {}).get("mapping", {})
    mapping_diagnostics = provenance_mapping.get("diagnostics", {})
    raw_node_mass = mass.get("raw_f32_node_mass", {})
    expected_mass = _compile_mass_partition(
        lab_export,
        profile,
        projected_rows,
        x_ref["sha256"],
        x_source["sha256"],
        {
            "regions": topology_regions,
            "source_global_node_index_sha256": profile["topology_identity"][
                "source_global_node_index_sha256"
            ],
            "executable_fem_topology_sha256": profile["topology_identity"][
                "executable_fem_topology_sha256"
            ],
            "anchor_ownership_sha256": profile["topology_identity"][
                "anchor_ownership_sha256"
            ],
        },
        {
            "maximum_displacement_m": mapping_diagnostics.get(
                "maximum_displacement_m",
                math.nan,
            ),
            "minimum_jacobian_determinant": min(
                row["minimum_jacobian_determinant"] for row in projected_rows
            ),
            "maximum_jacobian_determinant": max(
                row["maximum_jacobian_determinant"] for row in projected_rows
            ),
            "raw_f32_node_mass_sha256": raw_node_mass.get("sha256"),
        },
    )
    _require(
        mass == expected_mass,
        "mass partition diverges from the exact Lab export and authored regions",
    )
    expected_authority = {
        **profile["full_state_authority"],
        "identity_sha256": digest(profile["full_state_authority"]),
    }
    _require(
        value["full_state_authority"] == expected_authority,
        "full-state authority identity or components differ",
    )
    qualification = value["qualification"]
    required_true = {
        "candidate_only",
        "source_identity_bound",
        "ownership_identity_bound",
        "topology_identity_bound",
        "projected_reference_identity_bound",
        "donor_mass_and_raw_moments_subtracted",
    }
    required_false = set(qualification) - required_true
    _require(
        set(qualification)
        == required_true
        | {
            "unloaded_reference_qualified",
            "prestrain_reference_reset_executed",
            "subject_material_calibrated",
            "mesh_convergence_qualified",
            "specimen_load_validation_qualified",
            "clinical_validity_qualified",
            "production_physical_ownership",
            "production_active_force",
            "runtime_x_current_accepted",
            "integrated_human_qualification",
        }
        and all(qualification[key] is True for key in required_true)
        and all(qualification[key] is False for key in required_false),
        "candidate-only qualification boundary differs",
    )


def build_binding(value: dict[str, Any]) -> dict[str, Any]:
    """Wrap the exact complete manifest for one-step Lab loader admission."""
    validate_manifest(value)
    manifest_file_sha256 = hashlib.sha256(canonical_bytes(value) + b"\n").hexdigest()
    result = {
        "schema": BINDING_SCHEMA,
        "manifest_canonicalization": CANONICALIZATION,
        "binding_hash_exclusion": "top-level binding_sha256",
        "binding_sha256": "",
        "manifest_file_sha256": manifest_file_sha256,
        "manifest": copy.deepcopy(value),
    }
    result["binding_sha256"] = digest(
        {key: item for key, item in result.items() if key != "binding_sha256"}
    )
    validate_binding(result)
    return result


def validate_binding(
    value: dict[str, Any], companion_manifest: dict[str, Any] | None = None
) -> None:
    required = {
        "schema",
        "manifest_canonicalization",
        "binding_hash_exclusion",
        "binding_sha256",
        "manifest_file_sha256",
        "manifest",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["schema"] == BINDING_SCHEMA
        and value["manifest_canonicalization"] == CANONICALIZATION
        and value["binding_hash_exclusion"] == "top-level binding_sha256",
        "loader binding fields or canonicalization differ",
    )
    _sha256(value["binding_sha256"], "binding hash")
    _require(
        value["binding_sha256"]
        == digest(
            {key: item for key, item in value.items() if key != "binding_sha256"}
        ),
        "loader binding hash mismatch",
    )
    _sha256(value["manifest_file_sha256"], "bound manifest file hash")
    manifest = value["manifest"]
    validate_manifest(manifest)
    expected_file_hash = hashlib.sha256(canonical_bytes(manifest) + b"\n").hexdigest()
    _require(
        value["manifest_file_sha256"] == expected_file_hash,
        "loader binding names the wrong embedded manifest file identity",
    )
    if companion_manifest is not None:
        validate_manifest(companion_manifest)
        _require(
            manifest == companion_manifest,
            "loader binding manifest differs from its companion artifact",
        )


def _publish_bundle(
    directory: Path, manifest: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    """Publish both immutable artifacts with one atomic directory rename."""
    validate_manifest(manifest)
    validate_binding(binding, manifest)
    manifest_bytes = canonical_bytes(manifest) + b"\n"
    binding_bytes = canonical_bytes(binding) + b"\n"
    _require(
        binding["manifest_file_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
        "binding names the wrong manifest file identity",
    )
    directory = Path(directory)
    _require(not directory.is_symlink(), "output bundle is redirected")
    expected = {
        MANIFEST_FILENAME: manifest_bytes,
        BINDING_FILENAME: binding_bytes,
    }
    if directory.exists():
        _require(directory.is_dir(), "output bundle is not a directory")
        actual_names = {item.name for item in directory.iterdir()}
        _require(
            actual_names == set(expected),
            "existing output bundle contains a different file set",
        )
        for name, encoded in expected.items():
            path = directory / name
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == encoded,
                f"existing output bundle differs: {name}",
            )
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{directory.name}.stage-",
                dir=directory.parent,
            )
        )
        try:
            for name, encoded in expected.items():
                with (staging / name).open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            staging_fd = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            os.rename(staging, directory)
            parent_fd = os.open(directory.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    return {
        "manifest_path": str(directory / MANIFEST_FILENAME),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "binding_path": str(directory / BINDING_FILENAME),
        "binding_file_sha256": hashlib.sha256(binding_bytes).hexdigest(),
    }


def compile_paths(
    *,
    ownership_path: Path,
    payload_path: Path,
    x_ref_path: Path,
    tendon_payload_path: Path,
    lab_export_path: Path,
) -> dict[str, Any]:
    ownership, _ = _read_json(ownership_path, "ownership manifest")
    payload, _ = _read_bytes(payload_path, "Open Knee payload")
    tendon, _ = _read_bytes(tendon_payload_path, "tendon payload")
    x_ref, _ = _read_bytes(x_ref_path, "authored x_ref")
    lab_export, _ = _read_json(
        lab_export_path,
        "Lab authoring export envelope",
    )
    profile = _load_frozen_profile_for_export(lab_export)
    return compile_manifest(
        ownership=ownership,
        profile=profile,
        payload_bytes=payload,
        tendon_payload_bytes=tendon,
        x_ref_bytes=x_ref,
        lab_export=lab_export,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--open-knee-payload", type=Path, required=True)
    parser.add_argument("--tendon-payload", type=Path, required=True)
    parser.add_argument("--x-ref", type=Path, required=True)
    parser.add_argument("--lab-export", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_paths(
        ownership_path=arguments.ownership,
        payload_path=arguments.open_knee_payload,
        tendon_payload_path=arguments.tendon_payload,
        x_ref_path=arguments.x_ref,
        lab_export_path=arguments.lab_export,
    )
    binding = build_binding(result)
    published = _publish_bundle(arguments.output_directory, result, binding)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "file_sha256": published["manifest_file_sha256"],
                "binding_sha256": binding["binding_sha256"],
                "binding_file_sha256": published["binding_file_sha256"],
                "production_physical_ownership": False,
                "runtime_x_current_accepted": False,
                "output": published["manifest_path"],
                "binding_output": published["binding_path"],
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
    except (LoadedAnatomyKneeError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"loaded-anatomy-knee-compile: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
