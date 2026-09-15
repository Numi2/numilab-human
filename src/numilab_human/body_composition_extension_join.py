"""Extend the published Human composition join with new source hand-offs.

The v14 body-composition receipt already joins the organ, blood, muscle,
skin, activation and rigid-body source layers.  This extension binds the new
foot proxy and muscle route-volume incidence receipts to that exact base
receipt without modifying any physical owner.  It is an evidence-graph
extension, not an anatomical or behavioral qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/body-composition-extension-join.v1.json"
BASE = ROOT / "Docs/media/body-composition-integration-20260914/receipt-v14.json"
FOOT_PROXY = ROOT / "Docs/media/foot-contact-proxy-candidate-20260915/receipt-v1.json"
MUSCLE_JOIN = ROOT / "Docs/media/muscle-route-volume-join-candidate-20260915/receipt-v1.json"
SCHEMA = "HumanPack.body-composition-extension-join.v1"


class ExtensionJoinError(HumanImportError):
    """The new source hand-offs cannot be joined to the base composition receipt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtensionJoinError("body composition extension: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise ExtensionJoinError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "extension profile", canonical_required=True)
    required = {
        "schema", "id", "base_receipt", "foot_proxy_receipt", "muscle_join_receipt",
        "expected_foot_proxy_count", "expected_muscle_route_count",
        "expected_closed_geometry_count", "boundary",
    }
    _require(set(profile) == required, "extension profile fields differ")
    _require(profile["schema"] == "numi.human.body-composition-extension-join.v1",
             "unsupported extension profile schema")
    _require(profile["id"] == "v14_source_graph_with_contact_and_muscle_handoffs",
             "unsupported extension profile")
    for key in ("base_receipt", "foot_proxy_receipt", "muscle_join_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    expected = {
        "expected_foot_proxy_count": 30,
        "expected_muscle_route_count": 416,
        "expected_closed_geometry_count": 60,
    }
    for key, expected_value in expected.items():
        _require(profile[key] == expected_value, f"{key} differs from the source contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "extension boundary is missing")
    return profile, digest


def _input(path: Path, document: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(document.get("schema")),
            "file_sha256": digest}


def _identity_digest(values: list[str]) -> str:
    return hashlib.sha256(canonical(sorted(values))).hexdigest()


def compile_candidate(*, base: Path = BASE, foot_proxy: Path = FOOT_PROXY,
                      muscle_join: Path = MUSCLE_JOIN,
                      profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    base_path, foot_path, muscle_path = Path(base), Path(foot_proxy), Path(muscle_join)
    base_doc, base_sha = _read(base_path, "base composition receipt")
    foot_doc, foot_sha = _read(foot_path, "foot proxy receipt")
    muscle_doc, muscle_sha = _read(muscle_path, "muscle route-volume receipt")
    _require(base_doc.get("schema") == "HumanPack.body-composition-integration-candidate.v1"
             and base_doc.get("status") == "partial", "base composition receipt changed")
    base_qualification = base_doc.get("qualification")
    base_ownership = base_doc.get("ownership")
    base_inputs = base_doc.get("inputs")
    base_bindings = base_doc.get("identity_bindings")
    _require(isinstance(base_qualification, dict) and isinstance(base_ownership, dict)
             and isinstance(base_inputs, dict) and isinstance(base_bindings, dict),
             "base composition receipt is incomplete")
    _require(base_qualification.get("source_identity_graph_bound") is True
             and base_qualification.get("integrated_human_qualification") is False,
             "base composition qualification boundary changed")
    _require(base_ownership and all(value == 0 for value in base_ownership.values()),
             "base composition contains a physical owner")

    _require(foot_doc.get("schema") == "HumanPack.foot-contact-proxy-candidate.v1"
             and foot_doc.get("status") == "partial", "foot proxy receipt changed")
    foot_counts = foot_doc.get("counts", {})
    foot_qualification = foot_doc.get("qualification", {})
    _require(foot_counts.get("proxy_count") == 30
             and foot_counts.get("foot_body_count") == 4
             and foot_counts.get("registered_source_member_count") == 30,
             "foot proxy counts changed")
    for key in ("source_registered_geometry_bound", "conservative_body_frame_proxy_bounds",
                "source_triangle_enclosure_preserved", "support_witness_identity_carried"):
        _require(foot_qualification.get(key) is True, f"foot proxy lacks {key}")
    for key in ("anatomical_collider_admitted", "collision_exclusions_admitted",
                "contact_material_calibration", "swept_motion_bounds",
                "anatomical_supports_loading", "dynamic_contact",
                "loaded_whole_body_equilibrium", "standing", "recovery", "walking"):
        _require(foot_qualification.get(key) is False, f"foot proxy boundary changed for {key}")
    foot_source = foot_doc.get("source", {})
    _require(foot_source.get("registration_receipt_sha256") ==
             base_bindings.get("foot_contact_registration_receipt_sha256"),
             "foot proxy and base registration receipts diverge")

    _require(muscle_doc.get("schema") == "HumanPack.muscle-route-volume-join-candidate.v1"
             and muscle_doc.get("status") == "partial", "muscle route-volume receipt changed")
    muscle_counts = muscle_doc.get("counts", {})
    muscle_qualification = muscle_doc.get("qualification", {})
    _require(muscle_counts.get("source_route_count") == 416
             and muscle_counts.get("closed_geometry_volume_owner_count") == 60
             and muscle_counts.get("routes_without_surface_binding") == 238
             and muscle_counts.get("routes_with_closed_geometry") == 78,
             "muscle route-volume counts changed")
    for key in ("source_route_identity_bound", "muscle_surface_identity_bound",
                "closed_geometry_volume_identity_joined", "unbound_routes_retained",
                "shared_surface_incidence_explicit"):
        _require(muscle_qualification.get(key) is True, f"muscle route-volume lacks {key}")
    for key in ("volume_partition_owner", "skeletal_muscle_tissue_mass_owner",
                "material_calibration", "activation_force_transfer", "activation_calibration",
                "fat_geometry_owner", "skin_geometry_owner", "subject_calibration",
                "standing_walking"):
        _require(muscle_qualification.get(key) is False,
                 f"muscle route-volume boundary changed for {key}")
    muscle_source = muscle_doc.get("source", {})
    _require(muscle_source.get("surface_receipt_sha256") ==
             base_inputs.get("muscle_surfaces", {}).get("file_sha256") and
             muscle_source.get("volume_receipt_sha256") ==
             base_bindings.get("muscle_geometric_volume_candidate_sha256"),
             "muscle route-volume sources diverge from base composition")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.body-composition-extension-join.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "base_composition": _input(base_path, base_doc, base_sha),
            "foot_proxy": _input(foot_path, foot_doc, foot_sha),
            "muscle_route_volume": _input(muscle_path, muscle_doc, muscle_sha),
            "profile": {"path": _relative(Path(profile)), "schema": profile_doc["schema"],
                        "file_sha256": profile_sha},
        },
        "base_receipt": {
            "schema": base_doc["schema"],
            "sha256": base_sha,
            "source_identity_graph_bound": True,
            "physical_owner_count": 0,
        },
        "extensions": {
            "foot_contact_proxy": {
                "proxy_count": foot_counts["proxy_count"],
                "foot_body_count": foot_counts["foot_body_count"],
                "registered_source_member_count": foot_counts["registered_source_member_count"],
                "active_support_witness_count": foot_counts["active_support_witness_count"],
                "source_registration_receipt_sha256": foot_source["registration_receipt_sha256"],
                "proxy_ids_sha256": _identity_digest([
                    f"{row['opensim_body']}:{row['source_member_id']}"
                    for row in foot_doc["proxies"]
                ]),
                "anatomical_supports_loading": False,
                "dynamic_contact": False,
            },
            "muscle_route_volume": {
                "source_route_count": muscle_counts["source_route_count"],
                "closed_geometry_volume_owner_count": muscle_counts["closed_geometry_volume_owner_count"],
                "routes_with_closed_geometry": muscle_counts["routes_with_closed_geometry"],
                "closed_geometry_route_incidence_count": muscle_counts["closed_geometry_route_incidence_count"],
                "routes_without_surface_binding": muscle_counts["routes_without_surface_binding"],
                "volume_receipt_sha256": muscle_source["volume_receipt_sha256"],
                "route_ids_sha256": _identity_digest([
                    str(row["source_actuator_index"]) for row in muscle_doc["route_rows"]
                ]),
                "volume_partition_owner": False,
                "skeletal_muscle_tissue_mass_owner": False,
                "activation_force_transfer": False,
            },
        },
        "qualification": {
            "base_source_identity_graph_bound": True,
            "foot_contact_proxy_bound": True,
            "muscle_route_volume_incidence_bound": True,
            "cross_extension_source_hashes_closed": True,
            "physical_owner_count": 0,
            "anatomical_supports_loading": False,
            "dynamic_foot_contact": False,
            "skeletal_muscle_tissue_volume": False,
            "skeletal_muscle_tissue_mass": False,
            "activation_calibration": False,
            "activation_force_transfer": False,
            "fat_geometry_and_mass": False,
            "material_calibration": False,
            "subject_calibration": False,
            "standing": False,
            "recovery": False,
            "walking": False,
            "integrated_human_qualification": False,
        },
        "boundary": profile_doc["boundary"],
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(base=arguments.base, foot_proxy=arguments.foot_proxy,
                               muscle_join=arguments.muscle_join, profile=arguments.profile)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output)}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--foot-proxy", type=Path, default=FOOT_PROXY)
    parser.add_argument("--muscle-join", type=Path, default=MUSCLE_JOIN)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (ExtensionJoinError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"body-composition-extension: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
