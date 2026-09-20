"""Compile the stable Human topology and physical-ownership authoring contract.

``HumanPack.ownership.v1`` is an authoring manifest, not a mechanics receipt.
It joins the cumulative target/source inventory to the current body-composition
and muscle-route candidate graphs while preserving every unresolved owner.  A
candidate mass or volume is never promoted to physical ownership by this
compiler.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .model import ImportError as HumanImportError
from .target_coverage import canonical_bytes, digest
from .target_coverage import validate_manifest as validate_coverage

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.ownership.v1"
COMPILER = "numilab-human.ownership.1"
AUTHORING_SCHEMA = "numi.human.ownership-authoring.v1"
PROFILE = ROOT / "config/human-ownership.v1.json"
BODY_COMPOSITION = (
    ROOT / "Docs/media/body-composition-release-join-20260915/receipt-v1.json"
)
MUSCLE_ROUTES = (
    ROOT / "Docs/media/muscle-route-mass-partition-candidate-20260915/receipt-v1.json"
)
OWNER_ROLES = (
    "physical_volume", "mechanical_mass", "material", "active_force", "state",
)
BINDING_STATUSES = {"unresolved", "source", "candidate"}
OWNER_STATUSES = {"unresolved", "candidate"}
QUALIFICATION_STATUSES = {"unresolved", "source_only", "candidate"}
FORCE_MODES = {"replacement", "additive"}
BINDING_STATUS_RANK = {
    "unresolved": 0, "source": 1, "candidate": 2,
}
COMPOSITION_STATUS_RANK = {
    "unresolved": 0, "single_source": 1, "composed": 2, "conflict": 3,
}
QUALIFICATION_STATUS_RANK = {
    "unresolved": 0, "source_only": 1, "candidate": 2,
}
BOUNDARY = (
    "Stable Human authoring identities and candidate moment bookkeeping only. "
    "Source topology, a candidate volume or mass, a semantic action, or a declared "
    "replacement rule does not assign a production physical owner, material, motor "
    "compartment, active-force path, state authority, mechanics, calibration, or "
    "integrated Human qualification."
)


class OwnershipError(HumanImportError):
    """The Human ownership authoring contract cannot be compiled or admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OwnershipError("HumanPack ownership: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} must be finite")
    return float(value)


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result >= 0.0, f"{label} must be nonnegative")
    return result


def _sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value.strip() == value and bool(value),
             f"{label} must be a nonempty canonical identifier")
    _require("\x00" not in value, f"{label} contains a NUL")
    return value


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise OwnershipError(f"HumanPack ownership: {label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} must contain an object")
    canonical_bytes(value)
    return value, hashlib.sha256(raw).hexdigest()


def _input(schema: str, file_sha256: str, identity_sha256: str) -> dict[str, str]:
    return {
        "schema": _identifier(schema, "input schema"),
        "file_sha256": _sha256(file_sha256, "input file hash"),
        "identity_sha256": _sha256(identity_sha256, "input identity hash"),
    }


def _unresolved_binding() -> dict[str, Any]:
    return {"status": "unresolved", "ids": []}


def _unresolved_owner() -> dict[str, Any]:
    return {"status": "unresolved", "owner_id": None}


def _unresolved_moments() -> dict[str, Any]:
    return {
        "status": "unresolved",
        "frame_id": None,
        "volume_m3": None,
        "zeroth_mass_kg": None,
        "first_mass_moment_kg_m": None,
        "second_mass_moment_kg_m2": None,
    }


def _unresolved_force_semantics() -> dict[str, Any]:
    return {"status": "unresolved", "mode": None, "replaces_owner_ids": []}


def _binding(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == {"status", "ids"},
             f"{label} binding fields differ")
    status, ids = value["status"], value["ids"]
    _require(status in BINDING_STATUSES, f"{label} binding status is invalid")
    _require(isinstance(ids, list), f"{label} IDs must be an array")
    normalized = [_identifier(item, f"{label} ID") for item in ids]
    _require(len(normalized) == len(set(normalized)), f"{label} IDs repeat")
    _require(normalized == sorted(normalized), f"{label} IDs are not sorted")
    _require((status == "unresolved") == (not normalized),
             f"{label} unresolved status and IDs disagree")
    return {"status": status, "ids": normalized}


def _owner(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == {"status", "owner_id"},
             f"{label} owner fields differ")
    status, owner_id = value["status"], value["owner_id"]
    _require(status in OWNER_STATUSES, f"{label} owner status is invalid")
    if status == "unresolved":
        _require(owner_id is None, f"{label} unresolved owner has an ID")
    else:
        owner_id = _identifier(owner_id, f"{label} owner ID")
    return {"status": status, "owner_id": owner_id}


def _vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} must have three values")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _matrix(value: Any, label: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == 3,
             f"{label} must be a 3 by 3 matrix")
    result = [_vector(row, f"{label}[{index}]") for index, row in enumerate(value)]
    for row in range(3):
        for column in range(3):
            _require(math.isclose(result[row][column], result[column][row],
                                  rel_tol=0.0, abs_tol=1.0e-15),
                     f"{label} must be symmetric")
    return result


def _positive_semidefinite(value: list[list[float]], label: str) -> None:
    """Admit a symmetric 3x3 raw mass-moment matrix within FP64 roundoff."""
    scale = max(1.0, *(abs(item) for row in value for item in row))
    tolerance = 1.0e-12
    _require(all(value[index][index] >= -tolerance * scale for index in range(3)),
             f"{label} has a negative principal moment")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        minor = (value[first][first] * value[second][second]
                 - value[first][second] * value[second][first])
        _require(minor >= -tolerance * scale * scale,
                 f"{label} is not positive semidefinite")
    determinant = (
        value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
    )
    _require(determinant >= -tolerance * scale * scale * scale,
             f"{label} is not positive semidefinite")


def _moments(value: Any, label: str) -> dict[str, Any]:
    required = {
        "status", "frame_id", "volume_m3", "zeroth_mass_kg",
        "first_mass_moment_kg_m", "second_mass_moment_kg_m2",
    }
    _require(isinstance(value, dict) and set(value) == required,
             f"{label} moment fields differ")
    status = value["status"]
    _require(status in OWNER_STATUSES, f"{label} moment status is invalid")
    volume = (None if value["volume_m3"] is None else
              _nonnegative(value["volume_m3"], f"{label} volume"))
    mass = (None if value["zeroth_mass_kg"] is None else
            _nonnegative(value["zeroth_mass_kg"], f"{label} zeroth moment"))
    first = (None if value["first_mass_moment_kg_m"] is None else
             _vector(value["first_mass_moment_kg_m"], f"{label} first moment"))
    second = (None if value["second_mass_moment_kg_m2"] is None else
              _matrix(value["second_mass_moment_kg_m2"], f"{label} second moment"))
    frame = value["frame_id"]
    if status == "unresolved":
        _require(frame is None and volume is None and mass is None
                 and first is None and second is None,
                 f"{label} unresolved moments carry values")
    else:
        _require(any(item is not None for item in (volume, mass, first, second)),
                 f"{label} resolved moments have no value")
        _require((first is None and second is None) or mass is not None,
                 f"{label} spatial moments require a zeroth moment")
        if first is not None or second is not None:
            frame = _identifier(frame, f"{label} moment frame")
        else:
            _require(frame is None or isinstance(frame, str),
                     f"{label} moment frame is invalid")
            if frame is not None:
                frame = _identifier(frame, f"{label} moment frame")
        if mass == 0.0:
            _require(first is None or all(item == 0.0 for item in first),
                     f"{label} zero mass carries a nonzero first moment")
            _require(second is None or all(item == 0.0 for row in second for item in row),
                     f"{label} zero mass carries a nonzero second moment")
        if second is not None:
            _positive_semidefinite(second, f"{label} raw second mass moment")
        if mass is not None and mass > 0.0 and first is not None and second is not None:
            centered = [[
                second[row][column] - first[row] * first[column] / mass
                for column in range(3)
            ] for row in range(3)]
            _positive_semidefinite(centered, f"{label} centered second mass moment")
    return {
        "status": status,
        "frame_id": frame,
        "volume_m3": volume,
        "zeroth_mass_kg": mass,
        "first_mass_moment_kg_m": first,
        "second_mass_moment_kg_m2": second,
    }


def _force_semantics(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict)
             and set(value) == {"status", "mode", "replaces_owner_ids"},
             f"{label} force-semantics fields differ")
    status, mode, replaced = value["status"], value["mode"], value["replaces_owner_ids"]
    _require(status in OWNER_STATUSES, f"{label} force-semantics status is invalid")
    _require(isinstance(replaced, list), f"{label} replacement IDs must be an array")
    replaced = [_identifier(item, f"{label} replacement owner ID") for item in replaced]
    _require(replaced == sorted(set(replaced)), f"{label} replacement IDs repeat or are unsorted")
    if status == "unresolved":
        _require(mode is None and not replaced, f"{label} unresolved force semantics carry a rule")
    else:
        _require(mode in FORCE_MODES, f"{label} force mode is invalid")
        _require(mode != "additive" or not replaced,
                 f"{label} additive force semantics cannot replace an owner")
        _require(mode != "replacement" or bool(replaced),
                 f"{label} replacement semantics omit the replaced owner")
    return {"status": status, "mode": mode, "replaces_owner_ids": replaced}


def _base_record(semantic_id: str, leaves: list[dict[str, Any]]) -> dict[str, Any]:
    leaf_hashes = sorted(_sha256(item["leaf_sha256"], "coverage leaf hash") for item in leaves)
    kinds = sorted({str(item["kind"]) for item in leaves})
    declarations = {item["declaration_sha256"] for item in leaves}
    conflict_ids = [] if len(declarations) <= 1 and len(kinds) <= 1 else leaf_hashes
    composition_status = (
        "single_source" if len(leaves) == 1 else
        "composed" if not conflict_ids else "conflict"
    )
    source_topology = all(
        kind.startswith(("xml:", "composed:"))
        or kind in {
            "anatomical_element", "anatomical_mesh", "source_topology_register",
            "registered_visual_object",
        }
        for kind in kinds
    )
    return {
        "semantic_id": semantic_id,
        "coverage_leaf_sha256s": leaf_hashes,
        "entity_kind": kinds[0] if len(kinds) == 1 else "composition_conflict",
        "topology": ({"status": "source", "ids": [semantic_id]}
                     if source_topology else _unresolved_binding()),
        "fields": _unresolved_binding(),
        "motor_compartments": _unresolved_binding(),
        "composition": {
            "status": composition_status,
            "component_semantic_ids": [],
            "conflict_ids": conflict_ids,
        },
        "owners": {role: _unresolved_owner() for role in OWNER_ROLES},
        "moments": _unresolved_moments(),
        "force_semantics": _unresolved_force_semantics(),
        "action": None,
        "qualification_status": "source_only",
    }


def _route_semantic_id(name: str) -> str:
    return "myosim_fullbody:composed/myofullbody/muscles/" + quote(name, safe="")


def _validate_body_composition(value: dict[str, Any], route_count: int) -> dict[str, Any]:
    _require(value.get("schema") == "HumanPack.body-composition-release-join.v1"
             and value.get("status") == "partial",
             "body-composition release join is unsupported")
    _identifier(value.get("subject"), "body-composition subject")
    counts = value.get("domain_counts")
    ownership = value.get("ownership")
    qualification = value.get("qualification")
    _require(isinstance(counts, dict) and isinstance(ownership, dict)
             and isinstance(qualification, dict),
             "body-composition release join tables are malformed")
    _require(all(type(item) is int and item >= 0 for item in counts.values()),
             "body-composition domain counts are invalid")
    _require(all(type(item) is bool or type(item) is int and item >= 0
                 for item in ownership.values()),
             "body-composition ownership values are invalid")
    _require(counts.get("muscle_source_routes") == route_count
             and counts.get("native_recruited_muscles") == route_count,
             "body-composition route counts differ from the route partition")
    _require(ownership.get("physical_owner_count") == 0,
             "body-composition join unexpectedly promotes physical owners")
    for key in (
        "organ_physical_volume_owner", "mechanical_blood_mass_owner",
        "mechanical_tissue_mass_owner", "skeletal_muscle_tissue_mass_owner",
        "fat_mechanical_mass_owner", "whole_body_dynamic_mass_matrix_owner",
    ):
        _require(ownership.get(key) is False,
                 f"body-composition join unexpectedly promotes {key}")
    _require(qualification.get("cross_domain_owner_nonduplication") is True
             and qualification.get("muscle_route_volume_identity_bound") is True
             and qualification.get("integrated_human_qualification") is False,
             "body-composition qualification boundary changed")
    return {
        "schema": value["schema"],
        "status": value["status"],
        "domain_counts": copy.deepcopy(dict(sorted(counts.items()))),
        "ownership": copy.deepcopy(dict(sorted(ownership.items()))),
        "candidate_scopes_disjoint": False,
        "production_physical_owner_count": 0,
    }


def _validate_route_identities(value: dict[str, Any]) -> dict[int, str]:
    _require(value.get("schema") == "HumanPack.muscle-route-volume-join-candidate.v1"
             and value.get("status") == "partial",
             "muscle route identity receipt is unsupported")
    counts = value.get("counts")
    qualification = value.get("qualification")
    rows = value.get("route_rows")
    _require(isinstance(counts, dict) and isinstance(qualification, dict)
             and isinstance(rows, list),
             "muscle route identity receipt tables are malformed")
    count = counts.get("source_route_count")
    _require(type(count) is int and count > 0 and len(rows) == count,
             "muscle route identity receipt count is invalid")
    _require(qualification.get("source_route_identity_bound") is True
             and qualification.get("unbound_routes_retained") is True,
             "muscle route identity receipt is not source bound")
    result: dict[int, str] = {}
    names: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), "muscle route identity row is malformed")
        index = row.get("source_actuator_index")
        name = _identifier(row.get("name"), "muscle route identity name")
        _require(type(index) is int and index >= 0 and index not in result,
                 "muscle route identity index is invalid or repeated")
        _require(name not in names, "muscle route identity name is repeated")
        result[index] = name
        names.add(name)
    _require(set(result) == set(range(count)),
             "muscle route identity indices are not contiguous")
    return result


def _validate_routes(value: dict[str, Any], route_identities: dict[int, str],
                     route_identity_file_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(value.get("schema") == "HumanPack.muscle-route-mass-partition-candidate.v1"
             and value.get("status") == "partial",
             "muscle route partition is unsupported")
    counts, totals, qualification = (
        value.get("counts"), value.get("totals"), value.get("qualification"),
    )
    rows = value.get("route_budgets")
    _require(isinstance(counts, dict) and isinstance(totals, dict)
             and isinstance(qualification, dict) and isinstance(rows, list),
             "muscle route partition tables are malformed")
    source = value.get("source")
    policy = value.get("policy")
    _require(isinstance(source, dict)
             and _sha256(source.get("route_volume_receipt_sha256"),
                         "declared route identity file hash")
             == route_identity_file_sha256,
             "muscle route partition does not bind the route identity receipt")
    _require(isinstance(policy, dict) and policy.get("name") == "equal_incidence"
             and policy.get("measured_partition") is False,
             "muscle route partition policy changed")
    count = counts.get("source_route_count")
    _require(type(count) is int and count > 0 and len(rows) == count,
             "muscle route partition count is invalid")
    _require(set(route_identities) == set(range(count)),
             "route partition and identity receipt counts differ")
    _require(qualification.get("source_route_identity_bound") is True
             and qualification.get("source_surface_mass_bound") is True
             and qualification.get("equal_incidence_candidate_partition") is True
             and qualification.get("candidate_mass_and_volume_close") is True
             and qualification.get("unbound_routes_retained") is True,
             "muscle route partition identity or closure is not qualified")
    for key in ("physical_volume_owner", "mechanical_mass_owner",
                "skeletal_muscle_tissue_mass_owner", "volumetric_active_force_owner"):
        _require(qualification.get(key) is False,
                 f"muscle route partition unexpectedly promotes {key}")

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    indices: set[int] = set()
    for row in rows:
        _require(isinstance(row, dict), "muscle route row is malformed")
        index = row.get("source_actuator_index")
        name = row.get("name")
        _require(type(index) is int and index >= 0 and index not in indices,
                 "source action index is invalid or repeated")
        name = _identifier(name, "muscle route name")
        _require(name not in names, "muscle route name is repeated")
        _require(route_identities.get(index) == name,
                 f"muscle route {index} differs from the source identity receipt")
        indices.add(index)
        names.add(name)
        for role in ("physical_volume_owner", "mechanical_mass_owner",
                     "volumetric_active_force_owner"):
            _require(row.get(role) is False, f"muscle route {index} promotes {role}")
        volume = _nonnegative(row.get("candidate_volume_m3"),
                              f"muscle route {index} candidate volume")
        mass = _nonnegative(row.get("candidate_mass_kg"),
                            f"muscle route {index} candidate mass")
        _require((volume == 0.0) == (mass == 0.0),
                 f"muscle route {index} candidate mass and volume disagree")
        result.append({"index": index, "name": name, "volume_m3": volume,
                       "mass_kg": mass})
    _require(indices == set(range(count)),
             "source action indices must be contiguous but have no fixed ceiling")
    result.sort(key=lambda row: row["index"])
    volume = math.fsum(row["volume_m3"] for row in result)
    mass = math.fsum(row["mass_kg"] for row in result)
    expected_volume = _nonnegative(totals.get("allocated_candidate_volume_m3"),
                                   "allocated candidate volume")
    expected_mass = _nonnegative(totals.get("allocated_candidate_mass_kg"),
                                 "allocated candidate mass")
    source_volume = _nonnegative(totals.get("source_candidate_volume_m3"),
                                 "source candidate volume")
    source_mass = _nonnegative(totals.get("source_candidate_mass_kg"),
                               "source candidate mass")
    volume_residual = _finite(totals.get("candidate_volume_residual_m3"),
                              "candidate volume residual")
    mass_residual = _finite(totals.get("candidate_mass_residual_kg"),
                            "candidate mass residual")
    _require(totals.get("candidate_partition_is_disjoint") is True,
             "route candidate partition is not disjoint")
    _require(totals.get("candidate_is_mechanical_mass") is False,
             "route candidate partition promotes mechanical mass")
    _require(math.isclose(volume, expected_volume, rel_tol=0.0, abs_tol=2.0e-15),
             "route candidate volume does not close")
    _require(math.isclose(mass, expected_mass, rel_tol=0.0, abs_tol=2.0e-12),
             "route candidate mass does not close")
    _require(math.isclose(expected_volume - source_volume, volume_residual,
                          rel_tol=0.0, abs_tol=2.0e-15)
             and abs(volume_residual) <= 2.0e-15,
             "route source and allocated candidate volumes do not close")
    _require(math.isclose(expected_mass - source_mass, mass_residual,
                          rel_tol=0.0, abs_tol=2.0e-12)
             and abs(mass_residual) <= 2.0e-12,
             "route source and allocated candidate masses do not close")
    return result, {"volume_m3": expected_volume, "zeroth_mass_kg": expected_mass}


def _validate_profile(value: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "id", "declarations", "moment_closures", "boundary"}
    _require(isinstance(value, dict) and set(value) == required,
             "ownership authoring profile fields differ")
    _require(value["schema"] == AUTHORING_SCHEMA, "unsupported ownership authoring profile")
    _identifier(value["id"], "ownership authoring profile ID")
    _require(isinstance(value["declarations"], list),
             "ownership declarations must be an array")
    _require(isinstance(value["moment_closures"], list),
             "moment closures must be an array")
    _require(isinstance(value["boundary"], str) and value["boundary"].strip(),
             "ownership authoring boundary is missing")
    return value


def _apply_declarations(records: dict[str, dict[str, Any]],
                        declarations: list[Any]) -> None:
    claimed: set[tuple[str, str]] = set()
    allowed = {
        "semantic_id", "topology", "fields", "motor_compartments", "composition",
        "owners", "moments", "force_semantics", "qualification_status",
    }
    for declaration in declarations:
        _require(isinstance(declaration, dict) and set(declaration) <= allowed
                 and "semantic_id" in declaration,
                 "ownership declaration fields differ")
        semantic_id = _identifier(declaration["semantic_id"], "declaration semantic ID")
        _require(semantic_id in records,
                 f"ownership declaration has no target-coverage or route record: {semantic_id}")
        record = records[semantic_id]
        for field in ("topology", "fields", "motor_compartments"):
            if field in declaration:
                slot = (semantic_id, field)
                _require(slot not in claimed, f"declaration double-assigns {field}: {semantic_id}")
                claimed.add(slot)
                existing = record[field]
                replacement = _binding(declaration[field], f"{semantic_id} {field}")
                _require(
                    BINDING_STATUS_RANK[replacement["status"]]
                    >= BINDING_STATUS_RANK[existing["status"]],
                    f"declaration downgrades {field}: {semantic_id}",
                )
                _require(
                    set(existing["ids"]) <= set(replacement["ids"]),
                    f"declaration erases {field} source IDs: {semantic_id}",
                )
                record[field] = replacement
        if "composition" in declaration:
            slot = (semantic_id, "composition")
            _require(slot not in claimed, f"declaration double-assigns composition: {semantic_id}")
            claimed.add(slot)
            composition = declaration["composition"]
            _require(isinstance(composition, dict)
                     and set(composition) == {"status", "component_semantic_ids", "conflict_ids"},
                     f"{semantic_id} composition fields differ")
            status = composition["status"]
            _require(status in {"unresolved", "single_source", "composed", "conflict"},
                     f"{semantic_id} composition status is invalid")
            components = [_identifier(item, f"{semantic_id} component ID")
                          for item in composition["component_semantic_ids"]]
            conflicts = [_identifier(item, f"{semantic_id} conflict ID")
                         for item in composition["conflict_ids"]]
            _require(components == sorted(set(components))
                     and conflicts == sorted(set(conflicts)),
                     f"{semantic_id} composition identities repeat or are unsorted")
            _require((status == "conflict") == bool(conflicts),
                     f"{semantic_id} conflict status and conflict IDs disagree")
            existing = record["composition"]
            _require(
                COMPOSITION_STATUS_RANK[status]
                >= COMPOSITION_STATUS_RANK[existing["status"]],
                f"declaration downgrades composition: {semantic_id}",
            )
            _require(
                set(existing["component_semantic_ids"]) <= set(components),
                f"declaration erases composition components: {semantic_id}",
            )
            _require(
                set(existing["conflict_ids"]) <= set(conflicts),
                f"declaration erases composition conflicts: {semantic_id}",
            )
            record["composition"] = {
                "status": status, "component_semantic_ids": components,
                "conflict_ids": conflicts,
            }
        if "owners" in declaration:
            owners = declaration["owners"]
            _require(isinstance(owners, dict) and set(owners) <= set(OWNER_ROLES),
                     f"{semantic_id} owner roles differ")
            for role, value in owners.items():
                slot = (semantic_id, "owner:" + role)
                _require(slot not in claimed,
                         f"declaration double-assigns {role} owner: {semantic_id}")
                claimed.add(slot)
                assignment = _owner(value, f"{semantic_id} {role}")
                _require(record["owners"][role]["status"] == "unresolved",
                         f"declaration double-assigns {role} owner: {semantic_id}")
                record["owners"][role] = assignment
        if "moments" in declaration:
            slot = (semantic_id, "moments")
            _require(slot not in claimed, f"declaration double-assigns moments: {semantic_id}")
            claimed.add(slot)
            _require(record["moments"]["status"] == "unresolved",
                     f"declaration conflicts with existing candidate moments: {semantic_id}")
            record["moments"] = _moments(declaration["moments"], semantic_id)
        if "force_semantics" in declaration:
            slot = (semantic_id, "force_semantics")
            _require(slot not in claimed,
                     f"declaration double-assigns force semantics: {semantic_id}")
            claimed.add(slot)
            _require(record["force_semantics"]["status"] == "unresolved",
                     f"declaration conflicts with existing force semantics: {semantic_id}")
            record["force_semantics"] = _force_semantics(
                declaration["force_semantics"], semantic_id,
            )
        if "qualification_status" in declaration:
            slot = (semantic_id, "qualification_status")
            _require(slot not in claimed,
                     f"declaration double-assigns qualification status: {semantic_id}")
            claimed.add(slot)
            status = declaration["qualification_status"]
            _require(status in QUALIFICATION_STATUSES,
                     f"{semantic_id} qualification status is invalid")
            _require(
                QUALIFICATION_STATUS_RANK[status]
                >= QUALIFICATION_STATUS_RANK[record["qualification_status"]],
                f"declaration downgrades qualification status: {semantic_id}",
            )
            record["qualification_status"] = status


def _moment_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value[key]) for key in (
        "volume_m3", "zeroth_mass_kg", "first_mass_moment_kg_m",
        "second_mass_moment_kg_m2",
    )}


def _component_sum(values: Iterable[Any], template: Any) -> Any:
    rows = list(values)
    if template is None:
        return None
    _require(all(item is not None for item in rows),
             "declared moment closure contains an unresolved member component")
    if type(template) in (int, float):
        return math.fsum(float(item) for item in rows)
    if isinstance(template, list) and len(template) == 3 and all(
            type(item) in (int, float) for item in template):
        return [math.fsum(float(item[index]) for item in rows) for index in range(3)]
    return [[math.fsum(float(item[row][column]) for item in rows)
             for column in range(3)] for row in range(3)]


def _subtract(actual: Any, expected: Any) -> Any:
    if expected is None:
        return None
    if type(expected) in (int, float):
        return float(actual) - float(expected)
    if isinstance(expected, list) and len(expected) == 3 and all(
            type(item) in (int, float) for item in expected):
        return [actual[index] - expected[index] for index in range(3)]
    return [[actual[row][column] - expected[row][column] for column in range(3)]
            for row in range(3)]


def _flatten(value: Any) -> Iterable[float]:
    if value is None:
        return []
    if type(value) in (int, float):
        return [float(value)]
    return [number for item in value for number in _flatten(item)]


def _closure(specification: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = {"id", "status", "member_semantic_ids", "frame_id", "expected", "absolute_tolerance"}
    _require(isinstance(specification, dict) and set(specification) == required,
             "moment closure fields differ")
    identifier = _identifier(specification["id"], "moment closure ID")
    status = specification["status"]
    _require(status == "candidate",
             f"moment closure {identifier} status is invalid")
    members = specification["member_semantic_ids"]
    _require(isinstance(members, list) and members,
             f"moment closure {identifier} has no members")
    members = [_identifier(item, f"moment closure {identifier} member") for item in members]
    _require(members == sorted(set(members)),
             f"moment closure {identifier} members repeat or are unsorted")
    _require(all(item in records for item in members),
             f"moment closure {identifier} has an unknown member")
    tolerance = _nonnegative(specification["absolute_tolerance"],
                             f"moment closure {identifier} tolerance")
    expected_input = specification["expected"]
    _require(isinstance(expected_input, dict)
             and set(expected_input) == {
                 "volume_m3", "zeroth_mass_kg", "first_mass_moment_kg_m",
                 "second_mass_moment_kg_m2",
             }, f"moment closure {identifier} expected fields differ")
    synthetic = {"status": status, "frame_id": specification["frame_id"], **expected_input}
    expected = _moments(synthetic, f"moment closure {identifier}")
    frame = expected["frame_id"]
    for member in members:
        value = records[member]["moments"]
        _require(value["status"] != "unresolved",
                 f"moment closure {identifier} includes unresolved member {member}")
        if expected["first_mass_moment_kg_m"] is not None or expected["second_mass_moment_kg_m2"] is not None:
            _require(value["frame_id"] == frame,
                     f"moment closure {identifier} mixes moment frames")
    actual: dict[str, Any] = {}
    residual: dict[str, Any] = {}
    for key in ("volume_m3", "zeroth_mass_kg", "first_mass_moment_kg_m",
                "second_mass_moment_kg_m2"):
        actual[key] = _component_sum((records[item]["moments"][key] for item in members),
                                     expected[key])
        residual[key] = _subtract(actual[key], expected[key])
        _require(all(abs(item) <= tolerance for item in _flatten(residual[key])),
                 f"moment closure {identifier} does not close {key}")
    return {
        "id": identifier,
        "status": status,
        "member_semantic_ids": members,
        "frame_id": frame,
        "expected": _moment_payload(expected),
        "actual": actual,
        "residual": residual,
        "absolute_tolerance": tolerance,
    }


def _counts(records: list[dict[str, Any]], closures: list[dict[str, Any]]) -> dict[str, Any]:
    owner_counts = {
        role: dict(sorted(Counter(record["owners"][role]["status"] for record in records).items()))
        for role in OWNER_ROLES
    }
    return {
        "records": len(records),
        "semantic_actions": sum(record["action"] is not None for record in records),
        "unmapped_route_records": sum(
            record["entity_kind"] == "muscle_route"
            and not record["coverage_leaf_sha256s"] for record in records
        ),
        "composition_conflicts": sum(
            record["composition"]["status"] == "conflict" for record in records
        ),
        "candidate_moment_records": sum(
            record["moments"]["status"] == "candidate" for record in records
        ),
        "production_moment_records": sum(
            record["moments"]["status"] == "production" for record in records
        ),
        "moment_closures": len(closures),
        "owners_by_role_and_status": owner_counts,
    }


def compile_manifest(*, coverage: dict[str, Any], body_composition: dict[str, Any],
                     muscle_routes: dict[str, Any],
                     muscle_route_identities: dict[str, Any],
                     authoring: dict[str, Any],
                     input_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """Compile one deterministic ownership manifest from already loaded inputs."""
    validate_coverage(coverage)
    profile = _validate_profile(authoring)
    route_identities = _validate_route_identities(muscle_route_identities)
    hashes = input_hashes or {
        "target_coverage": digest(coverage),
        "body_composition": digest(body_composition),
        "muscle_routes": digest(muscle_routes),
        "muscle_route_identities": hashlib.sha256(
            canonical_bytes(muscle_route_identities) + b"\n"
        ).hexdigest(),
        "authoring": digest(authoring),
    }
    for key in (
        "target_coverage", "body_composition", "muscle_routes",
        "muscle_route_identities", "authoring",
    ):
        _sha256(hashes.get(key), f"{key} input hash")
    route_rows, route_totals = _validate_routes(
        muscle_routes, route_identities, hashes["muscle_route_identities"],
    )
    body_summary = _validate_body_composition(body_composition, len(route_rows))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for leaf in coverage["leaves"]:
        grouped[_identifier(leaf["semantic_id"], "coverage semantic ID")].append(leaf)
    records = {semantic_id: _base_record(semantic_id, leaves)
               for semantic_id, leaves in sorted(grouped.items())}

    route_members: list[str] = []
    for route in route_rows:
        semantic_id = _route_semantic_id(route["name"])
        coverage_route_leaves = grouped.get(semantic_id, [])
        _require(not coverage_route_leaves or all(
            leaf.get("kind") == "composed:muscles" and leaf.get("name") == route["name"]
            for leaf in coverage_route_leaves
        ), f"route semantic identity differs from target coverage: {semantic_id}")
        if semantic_id not in records:
            records[semantic_id] = {
                "semantic_id": semantic_id,
                "coverage_leaf_sha256s": [],
                "entity_kind": "muscle_route",
                "topology": {"status": "source", "ids": [semantic_id]},
                "fields": _unresolved_binding(),
                "motor_compartments": _unresolved_binding(),
                "composition": {
                    "status": "unresolved", "component_semantic_ids": [],
                    "conflict_ids": [],
                },
                "owners": {role: _unresolved_owner() for role in OWNER_ROLES},
                "moments": _unresolved_moments(),
                "force_semantics": _unresolved_force_semantics(),
                "action": None,
                "qualification_status": "source_only",
            }
        record = records[semantic_id]
        _require(record["entity_kind"] in {"composed:muscles", "muscle_route"},
                 f"route semantic ID collides with a non-muscle source leaf: {semantic_id}")
        record["entity_kind"] = "muscle_route"
        record["fields"] = {
            "status": "source",
            "ids": sorted([
                semantic_id + "/field/activation",
                semantic_id + "/field/fibre-state",
                semantic_id + "/field/tendon-state",
            ]),
        }
        record["action"] = {
            "status": "source",
            "semantic_action_id": semantic_id + "/action/stimulation",
            "source_index": route["index"],
        }
        record["force_semantics"] = {
            "status": "candidate",
            "mode": "replacement",
            "replaces_owner_ids": [semantic_id + "/owner/source-jt"],
        }
        if route["mass_kg"] > 0.0:
            record["moments"] = {
                "status": "candidate",
                "frame_id": None,
                "volume_m3": route["volume_m3"],
                "zeroth_mass_kg": route["mass_kg"],
                "first_mass_moment_kg_m": None,
                "second_mass_moment_kg_m2": None,
            }
            record["qualification_status"] = "candidate"
            route_members.append(semantic_id)

    _apply_declarations(records, profile["declarations"])
    _require(not any(record["owners"][role]["status"] == "production"
                     for record in records.values() for role in OWNER_ROLES),
             "production ownership contradicts the zero-owner body-composition join")

    closure_specs = list(profile["moment_closures"])
    if route_members:
        closure_specs.append({
            "id": "skeletal-muscle-route-candidate-partition",
            "status": "candidate",
            "member_semantic_ids": sorted(route_members),
            "frame_id": None,
            "expected": {
                "volume_m3": route_totals["volume_m3"],
                "zeroth_mass_kg": route_totals["zeroth_mass_kg"],
                "first_mass_moment_kg_m": None,
                "second_mass_moment_kg_m2": None,
            },
            "absolute_tolerance": 2.0e-12,
        })
    closure_ids: set[str] = set()
    closures = []
    for item in closure_specs:
        identifier = item.get("id") if isinstance(item, dict) else None
        _require(isinstance(identifier, str) and identifier not in closure_ids,
                 "moment closure ID is missing or repeated")
        closure_ids.add(identifier)
        closures.append(_closure(item, records))
    closures.sort(key=lambda item: item["id"])
    record_list = [records[key] for key in sorted(records)]

    action_indices = [record["action"]["source_index"] for record in record_list
                      if record["action"] is not None]
    action_ids = [record["action"]["semantic_action_id"] for record in record_list
                  if record["action"] is not None]
    _require(len(action_indices) == len(set(action_indices))
             and set(action_indices) == set(range(len(route_rows))),
             "semantic action indices are incomplete or repeated")
    _require(len(action_ids) == len(set(action_ids)), "semantic action IDs repeat")

    counts = _counts(record_list, closures)
    route_coverage_bound = counts["unmapped_route_records"] == 0
    blocked = bool(
        counts["composition_conflicts"]
        or not route_coverage_bound
        or coverage["scope_status"] == "blocked"
        or coverage["counts"]["unresolved_current_registers"] != 0
    )
    result = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "manifest_sha256": "",
        "status": "blocked" if blocked else "partial",
        "subject": body_composition["subject"],
        "inputs": {
            "target_coverage": _input(
                coverage["schema"], hashes["target_coverage"], coverage["manifest_sha256"],
            ),
            "body_composition": _input(
                body_composition["schema"], hashes["body_composition"],
                digest(body_composition),
            ),
            "muscle_routes": _input(
                muscle_routes["schema"], hashes["muscle_routes"], digest(muscle_routes),
            ),
            "muscle_route_identities": _input(
                muscle_route_identities["schema"], hashes["muscle_route_identities"],
                digest(muscle_route_identities),
            ),
            "authoring": _input(authoring["schema"], hashes["authoring"], digest(authoring)),
        },
        "source_coverage": {
            "manifest_sha256": coverage["manifest_sha256"],
            "scope_status": coverage["scope_status"],
            "leaf_count": coverage["counts"]["leaves"],
            "unresolved_current_registers": coverage["counts"]["unresolved_current_registers"],
        },
        "body_composition": body_summary,
        "records": record_list,
        "moment_closures": closures,
        "counts": counts,
        "qualification": {
            "target_coverage_validated": True,
            "body_composition_join_bound": True,
            "all_source_routes_retained": len(action_indices) == len(route_rows),
            "route_semantic_ids_coverage_bound": route_coverage_bound,
            "candidate_moment_closure_validated": bool(closures),
            "production_physical_ownership": False,
            "integrated_human_qualification": False,
        },
        "boundary": BOUNDARY,
    }
    result["manifest_sha256"] = digest(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )
    validate_ownership(result)
    return result


def validate_ownership(value: dict[str, Any]) -> None:
    """Validate deterministic identity, owner uniqueness, and moment closures."""
    required = {
        "schema", "compiler", "manifest_sha256", "status", "subject", "inputs",
        "source_coverage", "body_composition", "records", "moment_closures", "counts",
        "qualification", "boundary",
    }
    _require(isinstance(value, dict) and set(value) == required,
             "ownership manifest fields differ")
    _require(value["schema"] == SCHEMA and value["compiler"] == COMPILER,
             "ownership manifest schema or compiler is unsupported")
    _sha256(value["manifest_sha256"], "ownership manifest hash")
    expected_hash = digest({key: item for key, item in value.items()
                            if key != "manifest_sha256"})
    _require(value["manifest_sha256"] == expected_hash,
             "ownership manifest hash mismatch")
    _require(value["boundary"] == BOUNDARY,
             "ownership evidence boundary changed")
    _require(value["status"] in {"partial", "blocked"}, "ownership status is invalid")
    _identifier(value["subject"], "ownership subject")
    inputs = value["inputs"]
    _require(isinstance(inputs, dict)
             and set(inputs) == {
                 "target_coverage", "body_composition", "muscle_routes",
                 "muscle_route_identities", "authoring",
             }, "ownership input records differ")
    for name, item in inputs.items():
        _require(isinstance(item, dict)
                 and set(item) == {"schema", "file_sha256", "identity_sha256"},
                 f"{name} input fields differ")
        _identifier(item["schema"], f"{name} input schema")
        _sha256(item["file_sha256"], f"{name} input file hash")
        _sha256(item["identity_sha256"], f"{name} input identity hash")
    coverage = value["source_coverage"]
    _require(isinstance(coverage, dict)
             and set(coverage) == {
                 "manifest_sha256", "scope_status", "leaf_count",
                 "unresolved_current_registers",
             }, "source-coverage summary fields differ")
    _sha256(coverage["manifest_sha256"], "source-coverage manifest hash")
    _require(coverage["manifest_sha256"] == inputs["target_coverage"]["identity_sha256"],
             "source-coverage identity differs from its input")
    _require(coverage["scope_status"] in {"blocked", "source_union_materialized"}
             and type(coverage["leaf_count"]) is int and coverage["leaf_count"] >= 95
             and type(coverage["unresolved_current_registers"]) is int
             and coverage["unresolved_current_registers"] >= 0,
             "source-coverage summary is invalid")
    expected_scope_status = (
        "blocked" if coverage["unresolved_current_registers"] else "source_union_materialized"
    )
    _require(coverage["scope_status"] == expected_scope_status,
             "source-coverage scope hides unresolved current registers")
    body = value["body_composition"]
    _require(isinstance(body, dict)
             and set(body) == {
                 "schema", "status", "domain_counts", "ownership",
                 "candidate_scopes_disjoint", "production_physical_owner_count",
             }, "body-composition summary fields differ")
    _require(body["schema"] == "HumanPack.body-composition-release-join.v1"
             and body["status"] == "partial"
             and body["candidate_scopes_disjoint"] is False
             and body["production_physical_owner_count"] == 0,
             "body-composition summary promotes candidate scopes")
    _require(isinstance(body["domain_counts"], dict)
             and all(type(item) is int and item >= 0
                     for item in body["domain_counts"].values()),
             "body-composition domain counts are invalid")
    _require(isinstance(body["ownership"], dict)
             and body["ownership"].get("physical_owner_count") == 0
             and all(type(item) is bool or type(item) is int and item >= 0
                     for item in body["ownership"].values()),
             "body-composition ownership summary is invalid")
    _require(isinstance(value["records"], list)
             and isinstance(value["moment_closures"], list),
             "ownership record tables must be arrays")
    records: dict[str, dict[str, Any]] = {}
    coverage_leaf_hashes: set[str] = set()
    action_indices: set[int] = set()
    action_ids: set[str] = set()
    for record in value["records"]:
        record_required = {
            "semantic_id", "coverage_leaf_sha256s", "entity_kind", "topology", "fields",
            "motor_compartments", "composition", "owners", "moments", "force_semantics",
            "action", "qualification_status",
        }
        _require(isinstance(record, dict) and set(record) == record_required,
                 "ownership record fields differ")
        semantic_id = _identifier(record["semantic_id"], "record semantic ID")
        _require(semantic_id not in records, f"semantic ID is repeated: {semantic_id}")
        leaves = record["coverage_leaf_sha256s"]
        _require(isinstance(leaves, list)
                 and leaves == sorted(set(leaves)),
                 f"coverage leaf hashes repeat or are unsorted: {semantic_id}")
        for leaf in leaves:
            _sha256(leaf, f"{semantic_id} coverage leaf hash")
            _require(leaf not in coverage_leaf_hashes,
                     f"coverage leaf is assigned to multiple semantic records: {leaf}")
            coverage_leaf_hashes.add(leaf)
        _identifier(record["entity_kind"], f"{semantic_id} entity kind")
        for field in ("topology", "fields", "motor_compartments"):
            _binding(record[field], f"{semantic_id} {field}")
        composition = record["composition"]
        _require(isinstance(composition, dict)
                 and set(composition) == {"status", "component_semantic_ids", "conflict_ids"}
                 and composition["status"] in {"unresolved", "single_source", "composed", "conflict"},
                 f"{semantic_id} composition is invalid")
        _require((composition["status"] == "conflict") == bool(composition["conflict_ids"]),
                 f"{semantic_id} conflict status and IDs disagree")
        for field in ("component_semantic_ids", "conflict_ids"):
            items = composition[field]
            _require(isinstance(items, list) and items == sorted(set(items)),
                     f"{semantic_id} {field} repeat or are unsorted")
            for item in items:
                _identifier(item, f"{semantic_id} {field}")
        _require(isinstance(record["owners"], dict)
                 and set(record["owners"]) == set(OWNER_ROLES),
                 f"{semantic_id} owner roles differ")
        for role in OWNER_ROLES:
            _owner(record["owners"][role], f"{semantic_id} {role}")
        _moments(record["moments"], semantic_id)
        _force_semantics(record["force_semantics"], semantic_id)
        _require(record["qualification_status"] in QUALIFICATION_STATUSES,
                 f"{semantic_id} qualification status is invalid")
        action = record["action"]
        if action is not None:
            _require(isinstance(action, dict)
                     and set(action) == {"status", "semantic_action_id", "source_index"}
                     and action["status"] in {"source", "candidate"},
                     f"{semantic_id} semantic action is invalid")
            action_id = _identifier(action["semantic_action_id"], f"{semantic_id} action ID")
            index = action["source_index"]
            _require(type(index) is int and index >= 0
                     and index not in action_indices and action_id not in action_ids,
                     f"{semantic_id} source action is repeated or invalid")
            action_indices.add(index)
            action_ids.add(action_id)
        records[semantic_id] = record
    declared_owner_ids = {
        record["owners"]["active_force"]["owner_id"]
        for record in records.values()
        if record["owners"]["active_force"]["owner_id"] is not None
    }
    reserved_source_force_ids = {
        semantic_id + "/owner/source-jt"
        for semantic_id, record in records.items()
        if record["entity_kind"] == "muscle_route" and record["action"] is not None
    }
    admitted_replacement_ids = declared_owner_ids | reserved_source_force_ids
    for semantic_id, record in records.items():
        semantics = record["force_semantics"]
        _require(set(semantics["replaces_owner_ids"]) <= admitted_replacement_ids,
                 f"{semantic_id} force semantics replace an unknown owner")
    _require(len(coverage_leaf_hashes) == coverage["leaf_count"],
             "represented coverage leaf count differs from source coverage")
    _require([record["semantic_id"] for record in value["records"]] == sorted(records),
             "ownership records are not sorted by semantic ID")
    _require(action_indices == set(range(len(action_indices))),
             "source action indices are not contiguous")
    _require(body["domain_counts"].get("muscle_source_routes") == len(action_indices)
             and body["domain_counts"].get("native_recruited_muscles") == len(action_indices),
             "body-composition action counts differ from ownership records")
    for semantic_id, record in records.items():
        _require(all(component in records
                     for component in record["composition"]["component_semantic_ids"]),
                 f"{semantic_id} composition has an unknown semantic component")

    closures = []
    closure_ids: set[str] = set()
    for closure in value["moment_closures"]:
        specification = {
            "id": closure.get("id"), "status": closure.get("status"),
            "member_semantic_ids": closure.get("member_semantic_ids"),
            "frame_id": closure.get("frame_id"), "expected": closure.get("expected"),
            "absolute_tolerance": closure.get("absolute_tolerance"),
        }
        computed = _closure(specification, records)
        _require(closure == computed, f"moment closure receipt changed: {computed['id']}")
        _require(computed["id"] not in closure_ids, "moment closure ID repeats")
        closure_ids.add(computed["id"])
        closures.append(computed)
    _require([item["id"] for item in closures] == sorted(closure_ids),
             "moment closures are not sorted")
    expected_counts = _counts(value["records"], value["moment_closures"])
    _require(value["counts"] == expected_counts, "ownership counts differ from records")
    expected_status = (
        "blocked" if expected_counts["composition_conflicts"]
        or expected_counts["unmapped_route_records"]
        or coverage["scope_status"] == "blocked"
        or coverage["unresolved_current_registers"] != 0 else "partial"
    )
    _require(value["status"] == expected_status,
             "ownership status hides composition conflicts")
    qualification = value["qualification"]
    _require(isinstance(qualification, dict)
             and set(qualification) == {
                 "target_coverage_validated", "body_composition_join_bound",
                 "all_source_routes_retained", "route_semantic_ids_coverage_bound",
                 "candidate_moment_closure_validated", "production_physical_ownership",
                 "integrated_human_qualification",
             }
             and qualification.get("target_coverage_validated") is True
             and qualification.get("body_composition_join_bound") is True
             and qualification.get("all_source_routes_retained") is True
             and qualification.get("route_semantic_ids_coverage_bound")
             is (expected_counts["unmapped_route_records"] == 0)
             and qualification.get("candidate_moment_closure_validated")
             is bool(value["moment_closures"])
             and qualification.get("production_physical_ownership") is False
             and qualification.get("integrated_human_qualification") is False,
             "ownership qualification boundary changed")
    _require(not any(record["owners"][role]["status"] == "production"
                     for record in records.values() for role in OWNER_ROLES),
             "zero-owner body-composition input cannot produce a production owner")


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    validate_ownership(value)
    encoded = canonical_bytes(value) + b"\n"
    _require(not path.is_symlink(), "ownership output is redirected")
    if path.exists():
        _require(path.read_bytes() == encoded,
                 "ownership output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _declared_route_identity_path(muscle_routes: dict[str, Any]) -> Path:
    source = muscle_routes.get("source")
    _require(isinstance(source, dict), "muscle route partition source record is missing")
    declared = _identifier(source.get("route_volume_receipt"),
                           "declared route identity receipt path")
    relative = Path(declared)
    _require(not relative.is_absolute() and ".." not in relative.parts
             and "\\" not in declared,
             "declared route identity receipt path is unsafe")
    return ROOT / relative


def compile_paths(*, coverage_path: Path, body_composition_path: Path = BODY_COMPOSITION,
                  muscle_routes_path: Path = MUSCLE_ROUTES,
                  authoring_path: Path = PROFILE) -> dict[str, Any]:
    coverage, coverage_hash = _read(Path(coverage_path), "target-coverage manifest")
    body, body_hash = _read(Path(body_composition_path), "body-composition release join")
    routes, routes_hash = _read(Path(muscle_routes_path), "muscle route partition")
    route_identities, route_identities_hash = _read(
        _declared_route_identity_path(routes), "muscle route identity receipt",
    )
    authoring, authoring_hash = _read(Path(authoring_path), "ownership authoring profile")
    return compile_manifest(
        coverage=coverage, body_composition=body, muscle_routes=routes,
        muscle_route_identities=route_identities, authoring=authoring,
        input_hashes={
            "target_coverage": coverage_hash,
            "body_composition": body_hash,
            "muscle_routes": routes_hash,
            "muscle_route_identities": route_identities_hash,
            "authoring": authoring_hash,
        },
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coverage", type=Path, required=True,
                        help="validated HumanPack.target-coverage.v1 manifest")
    parser.add_argument("--body-composition", type=Path, default=BODY_COMPOSITION)
    parser.add_argument("--muscle-routes", type=Path, default=MUSCLE_ROUTES)
    parser.add_argument("--authoring", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_paths(
        coverage_path=arguments.coverage,
        body_composition_path=arguments.body_composition,
        muscle_routes_path=arguments.muscle_routes,
        authoring_path=arguments.authoring,
    )
    output_hash = _immutable_write(arguments.output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "manifest_sha256": result["manifest_sha256"],
        "file_sha256": output_hash,
        "status": result["status"],
        "records": result["counts"]["records"],
        "semantic_actions": result["counts"]["semantic_actions"],
        "production_physical_ownership": False,
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (OwnershipError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"ownership-compile: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
