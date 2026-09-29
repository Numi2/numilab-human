"""Source typing and complete-membership checks for selected anatomy families.

An atlas organ component is not a whole organ, and an anatomical cavity is
not tissue. Source-membership completeness is never mesh/clinical completeness.
"""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Any


def source_organ_coverage(
    specification: dict[str, Any], types: list[dict[str, str]],
) -> dict[str, Any]:
    ids = {row["concept_id"] for row in types}
    exact_type = {"concept_id": specification["concept_id"], "label": specification["source_name"]} in types
    if ids & {"FMA5897", "FMA67112"}:
        raise ValueError("organ layer cannot use an anatomical space as tissue")
    if "FMA67498" in ids:
        kind = "organ"
    elif ids & {"FMA82472", "FMA14065"}:
        kind = "organ_component"
    else:
        raise ValueError("organ layer has no source organ or organ-component type")
    if kind == "organ" and not exact_type:
        raise ValueError("named organ selection has no exact source organ type")
    return {
        "source_named_organ_type_matches": exact_type and kind == "organ",
        "source_named_structure_type_matches": exact_type,
        "source_structure_kind": kind,
        "organ_coverage": (
            "source_named_organ_representation" if exact_type and kind == "organ" else
            "source_named_organ_component_representation" if exact_type else
            "source_part_of_organ_component"
        ),
    }


def source_family_coverage(
    requirements: Any, specifications: list[dict[str, Any]],
    relations: dict[str, set[tuple[str, str, str]]],
) -> dict[str, Any]:
    """Retain missing source members and wrong layer/body owners explicitly."""
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("torso source coverage requirements are missing")
    if not isinstance(specifications, list) or any(
        not isinstance(spec, dict) or any(not isinstance(spec.get(key), str) or not spec[key]
                                         for key in ["member_id", "layer", "myosim_body"])
        for spec in specifications
    ):
        raise ValueError("torso source coverage selections are malformed")
    rows = []
    seen = set()
    for requirement in requirements:
        fields = ["id", "concept_id", "source_name", "hierarchy", "layer", "myosim_body", "selection"]
        if not isinstance(requirement, dict) or any(
            not isinstance(requirement.get(key), str) or not requirement[key] for key in fields
        ):
            raise ValueError("torso source coverage requirement is malformed")
        identity = requirement["id"]
        hierarchy = requirement["hierarchy"]
        if (identity in seen or hierarchy not in relations
                or requirement["selection"] != "complete_source_membership"):
            raise ValueError("torso source coverage requirement identity/selection is unsupported")
        seen.add(identity)
        expected = {member for concept, name, member in relations[hierarchy]
                    if concept == requirement["concept_id"] and name == requirement["source_name"]}
        if not expected:
            raise ValueError("torso source coverage family has no exact source members")
        required_layers = {member: requirement["layer"] for member in expected}
        if requirement["layer"] == "source_typed":
            rules = requirement.get("source_type_layers")
            if (not isinstance(rules, list) or not rules or "is_a" not in relations
                    or any(not isinstance(rule, dict) or any(
                        not isinstance(rule.get(key), str) or not rule[key]
                        for key in ["concept_id", "source_name", "layer"]) for rule in rules)):
                raise ValueError("torso source coverage type partitions are malformed")
            for member in expected:
                matches = [rule["layer"] for rule in rules if
                           (rule["concept_id"], rule["source_name"], member) in relations["is_a"]]
                if len(matches) != 1 or matches[0] == "source_typed":
                    raise ValueError("torso source coverage member has missing or ambiguous source type")
                required_layers[member] = matches[0]
        selected = [spec for spec in specifications if spec["member_id"] in expected]
        covered = {spec["member_id"] for spec in selected}
        wrong_owners = sorted(spec["member_id"] for spec in selected
                              if spec["layer"] != required_layers[spec["member_id"]]
                              or spec["myosim_body"] != requirement["myosim_body"])
        duplicated = len(selected) != len(covered)
        rows.append({
            "id": identity, "concept_id": requirement["concept_id"], "label": requirement["source_name"],
            "hierarchy": hierarchy, "expected_members": sorted(expected), "selected_members": sorted(covered),
            "missing_members": sorted(expected - covered), "wrong_layer_or_body_members": wrong_owners,
            "duplicate_members": duplicated, "passed": covered == expected and not wrong_owners and not duplicated,
            "required_layer": requirement["layer"], "required_myosim_body": requirement["myosim_body"],
            "required_member_layers": required_layers,
        })
    return {
        "requirements": rows, "passed": all(row["passed"] for row in rows),
        "boundary": "Complete membership of declared atlas source families; not whole-organ or clinical completeness, material, contact, blood volume or physiological qualification.",
    }


@lru_cache(maxsize=512)
def _source_family_topology_json(obj: bytes, member: str) -> str:
    """Cache by exact immutable source bytes, never by path or mtime."""
    from .cardiac_cavity_geometry import analyze_topology, exact_coordinate_quotient, parse_obj
    parsed = parse_obj(obj, member)
    raw = analyze_topology(parsed["vertices_mm"], parsed["triangles"])
    quotient = exact_coordinate_quotient(parsed)

    def compact(topology):
        keys = ["vertex_count", "face_count", "boundary_edge_count", "nonmanifold_edges",
                "orientation_defect_edges", "degenerate_face_ids", "duplicate_face_ids",
                "vertex_manifold_defect_ids", "face_component_count", "euler_characteristic",
                "closed_oriented_manifold_candidate"]
        return {key: len(topology[key]) if isinstance(topology[key], list) else topology[key] for key in keys}

    return json.dumps({
        "raw": compact(raw), "exact_coordinate_quotient": compact(quotient["topology"]),
        "identified_seam_vertex_count": quotient["identified_vertex_count"],
        "repair_applied": False, "payload_geometry_modified": False,
        "self_intersections": "not_checked", "physics_admitted": False,
        "boundary": "Source topology diagnostic with exact authored-coordinate seam identification only; visual payload retains every raw vertex and triangle. A closed-oriented candidate does not admit volume, mechanics or clinical anatomy.",
    }, sort_keys=True)


def source_family_topology(obj: bytes, member: str) -> dict[str, Any]:
    # Return a fresh object so callers cannot mutate cached source evidence.
    return json.loads(_source_family_topology_json(obj, member))
