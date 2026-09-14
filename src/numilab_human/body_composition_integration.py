"""Join the current Human organ, blood, tissue, fat and muscle handoffs.

This compiler is deliberately an integration boundary, not a physical-body
qualification.  It proves that the source identities can be joined without
duplicating an owner, and it keeps every missing physical volume, mechanical
mass, material, and calibration authority explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.body-composition-integration-candidate.v1"
PROFILE = ROOT / "config/body-composition-integration.v1.json"

ORGAN_MASS = ROOT / "Docs/media/tissue-mass-candidate-20260914/receipt-v2.json"
REGIONAL_TISSUE = ROOT / "Docs/media/regional-tissue-mass-candidate-20260914/receipt-v1.json"
SURFACES = ROOT / "Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json"
MUSCLE_GEOMETRY_AUDIT = ROOT / "Docs/media/muscle-surface-geometry-audit-20260914/receipt-v1.json"
MUSCLE_GEOMETRIC_VOLUME = ROOT / "Docs/media/muscle-geometric-volume-candidate-20260914/receipt-v1.json"
SKIN_SHELL = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/receipt-v2.json"
SKIN_NATIVE_VISUAL = ROOT / "Docs/media/skin-shell-native-visual-20260914/receipt-v1.json"
FOOT_CONTACT = ROOT / "Docs/media/foot-contact-registration-candidate-20260914/receipt-v1.json"
ACTIVATION = ROOT / "Docs/media/activation-recruitment-candidate-20260914/receipt-v1.json"
BLOOD_TRANSPORT = ROOT / "Docs/media/organ-blood-tissue-transport-20260914/receipt-v1.json"
TISSUE_EXCHANGE = ROOT / "Docs/media/organ-tissue-exchange-candidate-20260914/receipt-v1.json"
CARDIAC_BLOOD = ROOT / "Docs/media/cardiac-blood-mass-candidate-20260914/receipt.json"
CVSIM21_BLOOD_MASS = ROOT / "Docs/media/cvsim21-blood-mass-step-20260914/receipt-exact-clock.json"
VESSEL_REGISTRATION = ROOT / "Docs/media/organ-vessel-registration-corrected-20260913/registration.json"
CARDIAC_WALL_SOURCE = ROOT / "Docs/media/cardiac-wall-anatomy-20260912/manifest.json"
CARDIAC_WALL_CONFIG = ROOT / "config/cardiac-wall-rodero18.v1.json"
TISSUE_CALIBRATION = ROOT / "Docs/media/tissue-integration-20260908/calibration-candidate.json"
NATIVE_RELEASE = ROOT / "Docs/media/native-current-release-20260915/receipt-v5.json"
NATIVE_COSTAL_TISSUE = ROOT / (
    "Docs/media/tissue-integration-20260908/current-costal-binding-receipt-20260915.json"
)
NATIVE_REGIONAL_EXCHANGE = ROOT / (
    "Docs/media/native-human-regional-exchange-20260915/receipt-v1.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("body composition integration: " + message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_digest(values: set[str]) -> str:
    return _sha256(canonical(sorted(values)))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    _require(isinstance(value, dict), f"{label} is not an object")
    # The upstream receipts are immutable, source-bound JSON records, but some
    # were intentionally published pretty-printed.  Preserve their exact file
    # digest instead of rewriting or rejecting a valid receipt on formatting.
    return value, _sha256(raw)


def _schema(document: dict[str, Any], expected: str, label: str) -> None:
    _require(document.get("schema") == expected,
             f"{label} schema is {document.get('schema')!r}, expected {expected!r}")


def _read_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "integration profile is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    _require(raw == canonical(value) + b"\n", "integration profile is not canonical")
    _require(set(value) == {
        "schema", "id", "clock_nanoseconds", "expected_source_member_layers",
        "expected_runtime", "inputs", "boundary",
    }, "integration profile fields differ")
    _require(value.get("schema") == "numi.human.body-composition-integration.v1",
             "unsupported integration profile schema")
    _require(value.get("id") == "one_adult_male_cross_domain_source_join",
             "unsupported integration profile")
    _require(value.get("clock_nanoseconds") == 12500,
             "integration profile clock differs")
    _require(value.get("expected_source_member_layers") == {
        "cardiac_wall_region_identity": 24,
        "muscle_surface_geometry_audit": 150,
        "organ_surface_candidates": 378,
        "regional_blood_transport": 329,
        "muscle_geometric_volume_candidate": 60,
        "muscle_tendon_surface_identity": 150,
        "skin_shell_surface_identity": 1,
        "tissue_calibration_candidate": 1,
        "vessel_surface_identity": 6,
    }, "integration profile source counts differ")
    _require(value.get("expected_runtime") == {
        "blood_transport_accepted_steps": 511,
        "blood_transport_rejected_steps": 1,
        "cvsim21_mass_accepted_steps": 511,
        "cvsim21_mass_rejected_steps": 1,
    }, "integration profile runtime counts differ")
    _require(value.get("inputs") == [
        "organ_mass", "regional_tissue", "muscle_surfaces", "muscle_geometry_audit", "skin_shell", "skin_native_visual",
        "muscle_geometric_volume_candidate",
        "foot_contact_registration", "activation",
        "blood_transport", "tissue_exchange", "cardiac_blood",
        "cvsim21_blood_mass", "vessel_registration", "cardiac_wall_source",
        "cardiac_wall_config", "tissue_calibration", "native_current_release",
        "native_costal_tissue", "native_regional_exchange",
    ], "integration profile input order differs")
    return value


def _input(path: Path, document: dict[str, Any], file_sha256: str) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "schema": document["schema"],
        "file_sha256": file_sha256,
    }


def _all_none(rows: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> None:
    for row in rows:
        for key in keys:
            _require(row.get(key) is None,
                     f"{label} assigns {key} for {row.get('member_id', row.get('region_id'))}")


def compile_candidate(
    *,
    profile: Path = PROFILE,
    organ_mass: Path = ORGAN_MASS,
    regional_tissue: Path = REGIONAL_TISSUE,
    surfaces: Path = SURFACES,
    muscle_geometry_audit: Path = MUSCLE_GEOMETRY_AUDIT,
    muscle_geometric_volume: Path = MUSCLE_GEOMETRIC_VOLUME,
    skin_shell: Path = SKIN_SHELL,
    skin_native_visual: Path = SKIN_NATIVE_VISUAL,
    foot_contact_registration: Path = FOOT_CONTACT,
    activation: Path = ACTIVATION,
    blood_transport: Path = BLOOD_TRANSPORT,
    tissue_exchange: Path = TISSUE_EXCHANGE,
    cardiac_blood: Path = CARDIAC_BLOOD,
    cvsim21_blood_mass: Path = CVSIM21_BLOOD_MASS,
    vessel_registration: Path = VESSEL_REGISTRATION,
    cardiac_wall_source: Path = CARDIAC_WALL_SOURCE,
    cardiac_wall_config: Path = CARDIAC_WALL_CONFIG,
    tissue_calibration: Path = TISSUE_CALIBRATION,
    native_current_release: Path = NATIVE_RELEASE,
    native_costal_tissue: Path = NATIVE_COSTAL_TISSUE,
    native_regional_exchange: Path = NATIVE_REGIONAL_EXCHANGE,
) -> dict[str, Any]:
    profile = Path(profile)
    profile_document = _read_profile(profile)
    paths = {
        "organ_mass": Path(organ_mass),
        "regional_tissue": Path(regional_tissue),
        "muscle_surfaces": Path(surfaces),
        "muscle_geometry_audit": Path(muscle_geometry_audit),
        "muscle_geometric_volume_candidate": Path(muscle_geometric_volume),
        "skin_shell": Path(skin_shell),
        "skin_native_visual": Path(skin_native_visual),
        "foot_contact_registration": Path(foot_contact_registration),
        "activation": Path(activation),
        "blood_transport": Path(blood_transport),
        "tissue_exchange": Path(tissue_exchange),
        "cardiac_blood": Path(cardiac_blood),
        "cvsim21_blood_mass": Path(cvsim21_blood_mass),
        "vessel_registration": Path(vessel_registration),
        "cardiac_wall_source": Path(cardiac_wall_source),
        "cardiac_wall_config": Path(cardiac_wall_config),
        "tissue_calibration": Path(tissue_calibration),
        "native_current_release": Path(native_current_release),
        "native_costal_tissue": Path(native_costal_tissue),
        "native_regional_exchange": Path(native_regional_exchange),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    labels = {
        "organ_mass": "organ mass candidate",
        "regional_tissue": "regional tissue candidate",
        "muscle_surfaces": "muscle surface candidate",
        "muscle_geometry_audit": "muscle surface geometry audit",
        "muscle_geometric_volume_candidate": "muscle geometric volume candidate",
        "skin_shell": "skin shell candidate",
        "skin_native_visual": "native skin shell visual receipt",
        "foot_contact_registration": "foot contact registration candidate",
        "activation": "activation candidate",
        "blood_transport": "blood transport candidate",
        "tissue_exchange": "tissue exchange candidate",
        "cardiac_blood": "cardiac blood candidate",
        "cvsim21_blood_mass": "CVSim21 blood mass receipt",
        "vessel_registration": "vessel registration receipt",
        "cardiac_wall_source": "cardiac wall source manifest",
        "cardiac_wall_config": "cardiac wall source config",
        "tissue_calibration": "tissue material calibration candidate",
        "native_current_release": "current native Human release requalification",
        "native_costal_tissue": "current native costal tissue requalification",
        "native_regional_exchange": "current native regional blood exchange requalification",
    }
    for name, path in paths.items():
        documents[name], hashes[name] = _read(path, labels[name])

    _schema(documents["organ_mass"], "HumanPack.tissue-mass-composition-candidate.v1", labels["organ_mass"])
    _schema(documents["regional_tissue"], "HumanPack.regional-tissue-mass-candidate.v1", labels["regional_tissue"])
    _schema(documents["muscle_surfaces"], "HumanPack.soft-tissue-surface-candidate.v1", labels["muscle_surfaces"])
    _schema(documents["muscle_geometry_audit"], "HumanPack.muscle-surface-geometry-audit.v1",
            labels["muscle_geometry_audit"])
    _schema(documents["muscle_geometric_volume_candidate"],
            "HumanPack.muscle-geometric-volume-candidate.v1",
            labels["muscle_geometric_volume_candidate"])
    _schema(documents["skin_shell"], "HumanPack.skin-shell-candidate.v2", labels["skin_shell"])
    _schema(documents["skin_native_visual"], "HumanPack.skin-shell-native-visual.v1",
            labels["skin_native_visual"])
    _schema(documents["foot_contact_registration"],
            "HumanPack.foot-contact-registration-candidate.v1",
            labels["foot_contact_registration"])
    _schema(documents["activation"], "HumanPack.activation-recruitment-candidate.v1", labels["activation"])
    _schema(documents["blood_transport"], "HumanPack.organ-blood-tissue-transport-candidate.v1", labels["blood_transport"])
    _schema(documents["tissue_exchange"], "HumanPack.organ-tissue-exchange-candidate.v1", labels["tissue_exchange"])
    _schema(documents["cardiac_blood"], "HumanPack.cardiac-blood-mass-candidate.v1", labels["cardiac_blood"])
    _schema(documents["cardiac_wall_source"], "HumanPack.cardiac-wall-source-asset.v1",
            labels["cardiac_wall_source"])
    _schema(documents["cardiac_wall_config"], "HumanPack.cardiac-wall-rodero18-source.v1",
            labels["cardiac_wall_config"])
    _schema(documents["tissue_calibration"], "HumanPack.tissue-calibration-candidate.v1",
            labels["tissue_calibration"])
    _schema(documents["native_current_release"],
            "numi.human.native-current-release-requalification.v5",
            labels["native_current_release"])
    _schema(documents["native_costal_tissue"],
            "HumanPack.costal-tissue-native-current-requalification.v1",
            labels["native_costal_tissue"])
    _schema(documents["native_regional_exchange"],
            "HumanPack.native-human-regional-exchange-current-requalification.v1",
            labels["native_regional_exchange"])

    native_release = documents["native_current_release"]
    native_source = native_release.get("source", {})
    native_comparison = native_release.get("comparison", {})
    native_qualification = native_release.get("qualification", {})
    _require(native_release.get("status") == "partial" and
             native_source.get("branch") == "numi-human-equilibrium-20260914" and
             native_source.get("commit") == "c45fa9622f6c73b58febdc24a7115aecf3d7699f" and
             native_source.get("device") == "Mac mini M4 Pro" and
             native_release.get("commands", {}).get("implicit_default_512", {}).get("step_count") == 512 and
             native_release.get("commands", {}).get("implicit_default_512", {}).get("muscle_step_seconds") == 1.25e-5,
             "native current release source identity or horizon changed")
    native_512 = native_comparison.get("default_implicit_ceiling_0_8_512_steps")
    _require(isinstance(native_512, dict) and
             native_512.get("persistent_completed_steps") == 512 and
             native_512.get("compiled_stand_balanced") is True and
             native_512.get("persistent_max_penetration_m") == 0 and
             native_512.get("source_support_active_contacts") == 6 and
             native_512.get("persistent_max_acceleration") == 32.7379798889 and
             native_comparison.get("default_implicit_512_temporal_drift", {}).get("status") ==
             "temporal_drift_observed",
             "native current release 512-step evidence changed")
    _require(native_qualification.get("exact_clock") is True and
             native_qualification.get("bounded_dynamic_release") is True and
             native_qualification.get("sustained_standing") is False and
             native_qualification.get("walking") is False and
             native_qualification.get("anatomical_loading") is False and
             native_qualification.get("blood_tissue_transfer") is False and
             native_qualification.get("material_calibration") is False and
             native_qualification.get("subject_calibration") is False,
             "native current release boundary changed")

    native_costal = documents["native_costal_tissue"]
    native_costal_source = native_costal.get("source", {})
    native_costal_results = native_costal.get("results", {})
    native_costal_qualification = native_costal.get("qualification", {})
    _require(native_costal.get("status") == "partial" and
             native_costal_source.get("commit") == "c45fa9622f6c73b58febdc24a7115aecf3d7699f" and
             native_costal_source.get("device") == "Mac mini M4 Pro" and
             native_costal_results.get("cooked_nodes") == 13516 and
             native_costal_results.get("cooked_tetrahedra") == 46278 and
             native_costal_results.get("attachments") == 2871 and
             native_costal_results.get("metal_replay_cases") == 8 and
             native_costal_results.get("tissue_mass_kg") == 0.11369939548001184 and
             native_costal_qualification.get("native_metal_replay") is True and
             native_costal_qualification.get("mass_conservation") is True and
             native_costal_qualification.get("whole_body_dynamic_mass_matrix") is False and
             native_costal_qualification.get("experimental_material_calibration") is False,
             "native current costal tissue evidence changed")

    native_exchange = documents["native_regional_exchange"]
    native_exchange_source = native_exchange.get("source", {})
    native_exchange_inputs = native_exchange.get("inputs", {})
    native_exchange_results = native_exchange.get("results", {})
    native_exchange_qualification = native_exchange.get("qualification", {})
    _require(native_exchange.get("status") == "partial" and
             native_exchange_source.get("branch") == "numi-human-equilibrium-20260914" and
             native_exchange_source.get("commit") == "c45fa9622f6c73b58febdc24a7115aecf3d7699f" and
             native_exchange_source.get("device") == "Mac mini M4 Pro" and
             native_exchange_source.get("binary") == "numi-matter-vascular-check" and
             native_exchange_inputs.get("fixture", {}).get("sha256") ==
             "eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818" and
             native_exchange_results.get("source_compartment_count") == 21 and
             native_exchange_results.get("source_connection_count") == 24 and
             native_exchange_results.get("regional_bed_count") == 7 and
             native_exchange_results.get("attempted_steps") == 512 and
             native_exchange_results.get("accepted_steps_environment_0") == 511 and
             native_exchange_results.get("rejected_step_environment_0") == 37 and
             native_exchange_results.get("timestep_nanoseconds") == 12500 and
             native_exchange_results.get("blood_density_candidate_kg_per_m3") == 1060.0 and
             native_exchange_results.get("rollback") == "bitwise" and
             native_exchange_results.get("replay") == "bitwise",
             "native regional exchange source identity or exact-clock result changed")
    _require(native_exchange_results.get("maximum_relative_volume_residual") == 5.711629397e-07 and
             native_exchange_results.get("maximum_relative_blood_mass_residual") == 5.711629397e-07 and
             native_exchange_results.get("maximum_relative_oxygen_residual") == 1.057184875e-06 and
             native_exchange_qualification.get("current_native_replay") is True and
             native_exchange_qualification.get("regional_blood_transport") is True and
             native_exchange_qualification.get("oxygen_amount_exchange") is True and
             native_exchange_qualification.get("accepted_step_conservation") is True and
             native_exchange_qualification.get("anatomical_vessel_lumen") is False and
             native_exchange_qualification.get("physical_tissue_volume_owner") is False and
             native_exchange_qualification.get("mechanical_blood_mass_owner") is False and
             native_exchange_qualification.get("material_calibration") is False and
             native_exchange_qualification.get("subject_calibration") is False and
             native_exchange_qualification.get("standing_walking") is False,
             "native regional exchange qualification boundary changed")

    organ = documents["organ_mass"]
    organ_counts = organ.get("counts", {})
    _require(organ_counts == {
        "organ_surface_mass_candidates": 342,
        "region_count": 18,
        "shared_source_members": 8,
        "source_members": 378,
        "unresolved_members": 36,
    }, "organ mass inventory counts changed")
    organ_rows = organ.get("candidates")
    _require(isinstance(organ_rows, list) and len(organ_rows) == 378,
             "organ mass candidate rows are incomplete")
    organ_ids = [row.get("member_id") for row in organ_rows]
    _require(all(isinstance(value, str) and value for value in organ_ids),
             "organ mass member identity is invalid")
    _require(len(set(organ_ids)) == len(organ_ids), "organ mass members repeat")
    _all_none(organ_rows, ("mechanical_mass_owner", "physical_volume_owner"), "organ candidate")
    _require(organ.get("totals", {}).get("physical_mass_owner_count") == 0 and
             organ.get("qualification", {}).get("interdomain_disjointness_qualified") is False,
             "organ candidate promoted a physical mass owner")

    regional = documents["regional_tissue"]
    regional_ownership = regional.get("ownership", {})
    _require(regional.get("qualification", {}).get("mass_conservation") is True and
             regional.get("native_evidence", {}).get("metal_device") == "Apple M4 Pro",
             "regional tissue candidate lacks native conservation evidence")
    for key in (
        "production_mechanical_mass_owner", "whole_body_dynamic_mass_matrix_owner",
        "fat_mass_owner", "skeletal_muscle_tissue_partition_owner", "blood_mass_owner",
    ):
        _require(regional_ownership.get(key) is False,
                 f"regional tissue candidate promoted {key}")
    regional_mass = regional.get("region", {}).get("tissue_mass_kg")
    _require(isinstance(regional_mass, (int, float)) and regional_mass > 0.0,
             "regional tissue mass is missing")

    surface = documents["muscle_surfaces"]
    surface_counts = surface.get("counts", {})
    _require(surface_counts.get("source_route_count") == 416 and
             surface_counts.get("muscle_surface_count") == 148 and
             surface_counts.get("tendon_surface_count") == 2 and
             surface_counts.get("routes_without_surface_binding") == 238,
             "muscle surface coverage changed")
    surface_rows = surface.get("surface_rows")
    route_rows = surface.get("route_rows")
    _require(isinstance(surface_rows, list) and len(surface_rows) == 150,
             "muscle/tendon surface rows are incomplete")
    _require(isinstance(route_rows, list) and len(route_rows) == 416,
             "muscle route rows are incomplete")
    route_ids = [row.get("source_actuator_index") for row in route_rows]
    _require(route_ids == list(range(416)), "muscle route identity table is not complete")
    _all_none(route_rows, ("mechanical_mass_owner", "physical_volume_owner",
                           "volumetric_active_force_owner"), "muscle route")
    _require(surface_counts.get("fat_surface_count") == 0 and
             surface_counts.get("skin_surface_count") == 0 and
             surface_counts.get("physical_volume_owner_count") == 0 and
             surface_counts.get("volumetric_active_force_owner_count") == 0,
             "fat/skin or active muscle ownership was promoted")
    _require(surface_counts.get("routes_with_surface_binding") == 178,
             "muscle route binding count changed")

    geometry = documents["muscle_geometry_audit"]
    geometry_source = geometry.get("source", {})
    geometry_counts = geometry.get("counts", {})
    geometry_summary = geometry.get("geometry", {})
    _require(geometry.get("status") == "partial" and
             geometry_source.get("surface_receipt") == _relative(paths["muscle_surfaces"]) and
             geometry_source.get("surface_receipt_sha256") == hashes["muscle_surfaces"] and
             geometry_source.get("source_archive_sha256") and
             len(geometry_source["source_archive_sha256"]) == 64,
             "muscle surface geometry source identity changed")
    _require(geometry_counts == {
        "source_surface_count": 150,
        "muscle_surface_count": 148,
        "tendon_surface_count": 2,
        "topology_recomputed_surface_count": 150,
        "single_closed_component_count": 60,
        "closed_multi_component_count": 6,
        "topology_defective_count": 84,
        "surface_volume_candidate_count": 60,
        "physical_volume_owner_count": 0,
        "mechanical_mass_owner_count": 0,
        "material_owner_count": 0,
        "volumetric_active_force_owner_count": 0,
        "total_quotient_vertex_count": 316420,
        "total_quotient_triangle_count": 631464,
        "total_duplicate_face_count": 484,
        "total_vertex_manifold_defect_count": 1452,
    }, "muscle surface geometry counts changed")
    _require(type(geometry_summary.get("surface_area_total_m2")) in (int, float)
             and math.isfinite(float(geometry_summary["surface_area_total_m2"]))
             and geometry_summary["surface_area_total_m2"] > 0.0
             and type(geometry_summary.get("algebraic_volume_total_m3")) in (int, float)
             and math.isfinite(float(geometry_summary["algebraic_volume_total_m3"]))
             and geometry_summary["algebraic_volume_total_m3"] > 0.0
             and geometry_summary.get("volume_is_physical_owner") is False,
             "muscle surface geometry summary is invalid")
    _all_none(geometry.get("surfaces", []),
              ("physical_volume_owner", "mechanical_mass_owner", "material_owner",
               "volumetric_active_force_owner"), "muscle geometry surface")
    _require(geometry.get("qualification", {}).get("source_member_hashes_bound") is True and
             geometry.get("qualification", {}).get("single_closed_surface_volume_candidates_recomputed") is True and
             geometry.get("qualification", {}).get("physical_tissue_volume_owner") is False and
             geometry.get("qualification", {}).get("skeletal_muscle_tissue_mass_owner") is False and
             geometry.get("qualification", {}).get("activation_force_transfer") is False,
             "muscle surface geometry qualification boundary changed")

    geometric_volume = documents["muscle_geometric_volume_candidate"]
    geometric_volume_source = geometric_volume.get("source", {})
    geometric_volume_coverage = geometric_volume.get("coverage", {})
    geometric_volume_geometry = geometric_volume.get("geometry", {})
    geometric_volume_owners = geometric_volume.get("owners")
    _require(geometric_volume.get("status") == "partial" and
             geometric_volume_source.get("audit") == _relative(paths["muscle_geometry_audit"]) and
             geometric_volume_source.get("audit_sha256") == hashes["muscle_geometry_audit"] and
             geometric_volume_coverage.get("source_muscle_surface_count") == 148 and
             geometric_volume_coverage.get("closed_muscle_component_count") == 60 and
             geometric_volume_coverage.get("closed_multi_component_count") == 6 and
             geometric_volume_coverage.get("topology_defective_muscle_component_count") == 82 and
             isinstance(geometric_volume_owners, list) and len(geometric_volume_owners) == 60,
             "muscle geometric volume candidate coverage changed")
    _require(type(geometric_volume_geometry.get("closed_muscle_volume_total_m3")) in (int, float) and
             math.isfinite(float(geometric_volume_geometry["closed_muscle_volume_total_m3"])) and
             geometric_volume_geometry["closed_muscle_volume_total_m3"] > 0.0,
             "muscle geometric volume candidate total is invalid")
    _require(all(isinstance(row, dict) and row.get("physical_volume_owner") is False and
                 row.get("mechanical_mass_owner") is False and
                 row.get("material_owner") is False and
                 row.get("volumetric_active_force_owner") is False
                 for row in geometric_volume_owners),
             "muscle geometric volume candidate promoted a physical owner")

    skin = documents["skin_shell"]
    skin_counts = skin.get("coverage", {})
    skin_source = skin.get("source", {})
    skin_ownership = skin.get("ownership", {})
    _require(skin.get("status") == "partial" and
             skin_source.get("bodyparts3d_member_id") == "FJ2810" and
             skin_source.get("source_vertex_count") == 102467 and
             skin_source.get("source_triangle_count") == 203382 and
             skin_source.get("outer_surface_vertex_count") == 54949 and
             skin_source.get("outer_surface_triangle_count") == 109183,
             "skin shell source identity changed")
    _require(skin_counts.get("registered_body_influence_count") == 86 and
             skin_counts.get("influences_per_vertex") == 4 and
             skin_counts.get("source_bone_surface_sample_count") == 7040 and
             type(skin_counts.get("rest_pose_reconstruction_max_error_m")) in (int, float) and
             skin_counts["rest_pose_reconstruction_max_error_m"] <= 2.0e-5,
             "skin shell registration coverage is incomplete")
    for key in (
        "skin_physical_volume_owner", "skin_mechanical_mass_owner", "skin_thickness_owner",
        "skin_material_owner", "skin_collision_owner", "skin_self_contact_owner",
        "fat_geometry_owner", "fat_mass_owner",
    ):
        _require(skin_ownership.get(key) is False,
                 f"skin shell promoted {key}")
    _require(skin.get("qualification", {}).get("source_skin_member_bound") is True and
             skin.get("qualification", {}).get("registered_visual_influences_bound") is True and
             skin.get("qualification", {}).get("native_registration_fingerprint_bound") is True and
             skin.get("qualification", {}).get("skin_physical_volume") is False and
             skin.get("qualification", {}).get("skin_material_calibration") is False and
             skin.get("qualification", {}).get("fat_geometry") is False,
             "skin shell qualification boundary changed")

    skin_visual = documents["skin_native_visual"]
    skin_visual_source = skin_visual.get("source", {})
    skin_visual_capture = skin_visual.get("capture", {})
    skin_visual_qualification = skin_visual.get("qualification", {})
    _require(skin_visual.get("status") == "qualified" and
             skin_visual_source.get("bodyparts3d_member_id") == "FJ2810" and
             skin_visual_source.get("skin_payload_sha256") ==
             skin.get("inputs", {}).get("payload", {}).get("sha256") and
             skin_visual_source.get("native_registration_fingerprint32") ==
             skin.get("source", {}).get("native_bone_registration_fingerprint32") and
             skin_visual_source.get("core_body_count") == 157 and
             skin_visual_source.get("rendered_skin_shell_count") == 1,
             "native skin visual source identity changed")
    visual_views = skin_visual_capture.get("views")
    _require(skin_visual_capture.get("metal_pose_device") == "Apple M4 Pro" and
             skin_visual_capture.get("renderer_device") == "Apple M4 Pro" and
             skin_visual_capture.get("frame_dimension") == 512 and
             isinstance(visual_views, list) and
             [row.get("view") for row in visual_views] == ["front", "oblique", "side", "rear"] and
             all(isinstance(row.get("skin_shell_pixels"), int) and row["skin_shell_pixels"] > 0
                 for row in visual_views),
             "native skin visual capture is incomplete")
    _require(skin_visual_qualification.get("native_visual_admission") is True and
             skin_visual_qualification.get("source_payload_hash_bound") is True and
             skin_visual_qualification.get("four_view_capture") is True and
             skin_visual_qualification.get("skin_physical_volume") is False and
             skin_visual_qualification.get("skin_deformation") is False and
             skin_visual_qualification.get("subject_calibration") is False,
             "native skin visual qualification boundary changed")

    foot = documents["foot_contact_registration"]
    foot_counts = foot.get("counts", {})
    foot_qualification = foot.get("qualification", {})
    foot_support = foot.get("support", {})
    foot_registration = foot.get("registration", {})
    _require(foot.get("status") == "partial" and
             foot_counts == {
                 "active_support_witness_count": 6,
                 "foot_body_count": 4,
                 "registered_source_member_count": 30,
                 "registered_source_mesh_count": 30,
                 "source_mesh_count": 60,
                 "support_witness_count": 18,
                 "unique_source_member_count": 30,
             } and
             foot_qualification.get("source_foot_geometry_registered") is True and
             foot_qualification.get("source_to_body_rest_transform_bound") is True and
             foot_qualification.get("support_witness_identity_bound") is True and
             foot_qualification.get("anatomical_collider_admitted") is False and
             foot_qualification.get("anatomical_supports_loading") is False and
             foot_qualification.get("dynamic_contact") is False and
             foot_qualification.get("loaded_whole_body_equilibrium") is False,
             "foot contact registration boundary changed")
    _require(foot_support.get("active_contact_count") == 6 and
             foot_support.get("contact_count") == 18 and
             foot_support.get("active_witnesses") == [5, 6, 8, 10, 12, 14] and
             foot_support.get("dynamic_contact_qualified") is False and
             foot_support.get("contact_calibration_qualified") is False and
             type(foot_support.get("total_support_force_n")) in (int, float) and
             type(foot_support.get("expected_weight_n")) in (int, float) and
             abs(foot_support["total_support_force_n"] - foot_support["expected_weight_n"]) /
             foot_support["expected_weight_n"] <= 1.0e-5 and
             0.0 <= foot_support.get("root_force_residual_n", float("inf")) <= 1.0e-3,
             "foot support witness summary is not weight balanced")
    _require(foot_registration.get("multi_pose_count") == 7 and
             foot_registration.get("source_members") and
             len(foot_registration["source_members"]) == 30 and
             foot_registration.get("continuity_evaluation_count") == 280 and
             type(foot_registration.get("bilateral_gap_parity_maximum_m")) in (int, float),
             "foot source registration continuity is incomplete")

    act = documents["activation"]
    act_counts = act.get("counts", {})
    _require(act_counts.get("source_routes") == 416 and
             act_counts.get("nonzero_routes") == 237 and
             act_counts.get("support_witnesses") == 18,
             "activation route counts changed")
    activation_vector = act.get("recruitment", {}).get("activation_fp64")
    _require(isinstance(activation_vector, list) and len(activation_vector) == 416,
             "activation vector is incomplete")
    _require(act.get("qualification", {}).get("activation_calibration") is False and
             act.get("qualification", {}).get("held_out_force_validation") is False,
             "activation candidate claims calibration")

    transport = documents["blood_transport"]
    transport_counts = transport.get("counts", {})
    _require(transport_counts.get("bed_count") == 7 and
             transport_counts.get("source_blood_owner_count") == 6 and
             transport_counts.get("source_members_bound") == 329 and
             transport.get("clock", {}).get("nanoseconds") == 12500 and
             transport.get("accepted_steps") == 511 and
             transport.get("rejected_steps") == 1,
             "blood transport receipt is not the exact-clock seven-bed run")
    _require(transport.get("conservation", {}).get("mass_conserved") is True and
             transport.get("rollback", {}).get("rejected_candidate_state_neutral") is True,
             "blood transport conservation/rollback is not qualified")
    transport_beds = transport.get("beds")
    _require(isinstance(transport_beds, list) and len(transport_beds) == 7,
             "blood transport beds are incomplete")
    blood_member_ids = [member_id for bed in transport_beds
                        for member_id in bed.get("source_member_ids", [])]
    _require(len(blood_member_ids) == 329 and len(set(blood_member_ids)) == 329,
             "blood transport source members repeat")
    organ_id_set = set(organ_ids)
    _require(set(blood_member_ids) <= organ_id_set,
             "blood transport references an organ member outside the source inventory")
    _all_none(transport_beds, ("mechanical_mass_owner", "physical_volume_owner",
                               "tissue_exchange_owner"), "blood transport bed")

    exchange = documents["tissue_exchange"]
    exchange_counts = exchange.get("counts", {})
    _require(exchange_counts.get("bed_count") == 7 and
             exchange_counts.get("source_blood_owner_count") == 6 and
             exchange_counts.get("source_members_bound") == 329 and
             exchange.get("clock", {}).get("nanoseconds") == 12500 and
             exchange.get("accepted_steps") == 511 and
             exchange.get("rejected_steps") == 1,
             "tissue exchange receipt is not the matching exact-clock run")
    exchange_beds = exchange.get("beds")
    _require(isinstance(exchange_beds, list) and
             {row.get("region_id") for row in exchange_beds} ==
             {row.get("region_id") for row in transport_beds},
             "tissue exchange beds do not match blood transport beds")
    _require(exchange.get("conservation", {}).get("oxygen_conserved") is True and
             exchange.get("rollback", {}).get("rejected_candidate_state_neutral") is True,
             "tissue exchange conservation/rollback is not qualified")
    for row in exchange_beds:
        _require(row.get("physical_tissue_volume_owner") is None and
                 row.get("tissue_exchange_owner") is None,
                 f"tissue exchange promoted an owner for {row.get('region_id')}")

    cardiac = documents["cardiac_blood"]
    _require(cardiac.get("selection") is None and
             cardiac.get("mass_budget", {}).get("both_candidates_share_hydraulic_budget") is True and
             cardiac.get("mass_budget", {}).get("mechanical_mass_owner_assigned") is False and
             cardiac.get("qualification", {}).get("two_way_blood_tissue_transfer") is False,
             "cardiac blood candidate boundary changed")
    cardiac_mass = cardiac.get("mass_budget", {}).get("candidate_mass_kg")
    _require(isinstance(cardiac_mass, (int, float)) and cardiac_mass > 0.0,
             "cardiac blood candidate mass is missing")

    cardiac_wall = documents["cardiac_wall_source"]
    cardiac_wall_config = documents["cardiac_wall_config"]
    _require(cardiac_wall.get("source_config_sha256") == hashes["cardiac_wall_config"],
             "cardiac wall manifest does not bind the source config hash")
    _require(cardiac_wall.get("source_config") == cardiac_wall_config,
             "cardiac wall manifest and source config differ")
    _require(cardiac_wall_config.get("id") == "rodero_2021_ct_case18_cardiac_wall",
             "unsupported cardiac wall source")
    cardiac_labels = cardiac_wall_config.get("labels")
    _require(isinstance(cardiac_labels, list) and len(cardiac_labels) == 24,
             "cardiac wall label inventory is incomplete")
    cardiac_label_ids = [row.get("id") for row in cardiac_labels]
    _require(cardiac_label_ids == list(range(1, 25)) and
             all(isinstance(row.get("anatomical_structure"), str) and
                 row.get("anatomical_structure") for row in cardiac_labels),
             "cardiac wall label identity is invalid")
    cardiac_wall_mesh = cardiac_wall_config.get("mesh", {})
    _require(cardiac_wall_mesh.get("points") == 300965 and
             cardiac_wall_mesh.get("cells") == 1470083 and
             cardiac_wall_mesh.get("cell_type") == 10 and
             cardiac_wall_mesh.get("nodes_per_cell") == 4,
             "cardiac wall mesh identity changed")
    cardiac_topology = cardiac_wall.get("topology", {})
    regional_cell_counts = cardiac_topology.get("regional_cell_counts", {})
    regional_volumes = cardiac_topology.get("regional_geometric_volume_m3", {})
    _require(set(regional_cell_counts) == {str(index) for index in range(1, 25)} and
             sum(regional_cell_counts.values()) == cardiac_wall_mesh["cells"] and
             set(regional_volumes) == set(regional_cell_counts),
             "cardiac wall regional topology is incomplete")
    _require(cardiac_topology.get("positive_oriented_tetrahedra") == cardiac_wall_mesh["cells"] and
             cardiac_topology.get("source_negative_orientation_count") == 0 and
             cardiac_topology.get("blood_volume_or_mass_assigned") is False and
             cardiac_topology.get("source_surface_edits") is False,
             "cardiac wall source orientation or ownership boundary changed")
    _require(all(type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0.0
                 for value in regional_volumes.values()),
             "cardiac wall regional geometry is invalid")
    cardiac_wall_qualification = cardiac_wall_config.get("qualification", {})
    _require(cardiac_wall_qualification.get("native_anatomical_coupling_qualified") is False and
             cardiac_wall_qualification.get("native_source_reproduction_qualified") is False and
             cardiac_wall_qualification.get("subject_specific_material_calibration") is False and
             cardiac_wall_qualification.get("subject_specific_pressure_volume_calibration") is False and
             cardiac_wall_qualification.get("whole_human_qualified") is False,
             "cardiac wall source promoted a mechanics or calibration owner")
    cardiac_wall_ids = {
        f"{cardiac_wall_config['id']}:label:{int(label_id)}"
        for label_id in cardiac_label_ids
    }
    cardiac_wall_volume = math.fsum(float(value) for value in regional_volumes.values())

    calibration = documents["tissue_calibration"]
    _require(calibration.get("status") == "unqualified_finite_hold_candidate" and
             calibration.get("qualified") is False,
             "tissue calibration candidate changed qualification boundary")
    _require(calibration.get("donor") == "oks003" and
             calibration.get("physical_plug") == "oks003-PTC-MCXX-01" and
             calibration.get("anatomical_target", {}).get("source_region") == "PTC" and
             calibration.get("anatomical_target", {}).get("specimen_side") == "left" and
             calibration.get("anatomical_target", {}).get("production_force_owner_fraction") == 0,
             "tissue calibration identity or owner boundary changed")
    calibration_split = calibration.get("split", {})
    _require(calibration_split.get("frozen_before_fit") is True and
             calibration_split.get("population_holdout") is False and
             calibration_split.get("unit") == "test day within the same physical plug" and
             len(calibration_split.get("training", [])) == 2 and
             len(calibration_split.get("held_out", [])) == 1,
             "tissue calibration split is not the pinned same-plug holdout")
    calibration_fit = calibration.get("fit", {})
    training_fit = calibration_fit.get("training", {})
    held_out_fit = calibration_fit.get("held_out", {})
    _require(training_fit.get("observations") == 6 and
             held_out_fit.get("observations") == 3 and
             all(type(training_fit.get(key)) in (int, float) and
                 math.isfinite(float(training_fit[key])) and float(training_fit[key]) >= 0.0
                 for key in ("force_rmse_N", "force_nrmse_relative_to_measured_rms",
                             "maximum_absolute_force_error_N")) and
             all(type(held_out_fit.get(key)) in (int, float) and
                 math.isfinite(float(held_out_fit[key])) and float(held_out_fit[key]) >= 0.0
                 for key in ("force_rmse_N", "force_nrmse_relative_to_measured_rms",
                             "maximum_absolute_force_error_N")),
             "tissue calibration fit metrics are incomplete")
    calibration_material = calibration.get("native_material", {})
    _require(calibration_material.get("compiled") is False and
             calibration_material.get("solver_validated") is False and
             "whole-human mechanics" in calibration.get("not_qualified", []) and
             "costal cartilage" in calibration.get("not_qualified", []) and
             calibration.get("preprocessing", {}).get("equilibrium_assumed") is False,
             "tissue calibration candidate promoted a native or whole-body material owner")
    calibration_id = f"{calibration['donor']}:{calibration['physical_plug']}"

    cvsim21 = documents["cvsim21_blood_mass"]
    _require(cvsim21.get("model_id") == "cvsim21_supine_continuous_fixed_rate_upstream_equation" and
             cvsim21.get("attempted_steps") == 512 and
             cvsim21.get("accepted_steps") == 511 and
             cvsim21.get("rejected_steps") == 1 and
             cvsim21.get("clock", {}).get("exact") is True and
             cvsim21.get("clock", {}).get("required_nanoseconds") == 12500 and
             cvsim21.get("clock", {}).get("timestep_nanoseconds") == 12500.0,
             "CVSim21 blood mass receipt is not the exact-clock 512-step run")
    cvsim21_conservation = cvsim21.get("conservation", {})
    _require(cvsim21_conservation.get("mass_conserved") is True and
             cvsim21_conservation.get("volume_conserved") is True and
             abs(cvsim21_conservation.get("mass_residual_kg", float("inf"))) <= 1.0e-12 and
             abs(cvsim21_conservation.get("volume_residual_m3", float("inf"))) <= 1.0e-15,
             "CVSim21 blood mass/volume conservation is not qualified")
    _require(cvsim21.get("rollback", {}).get("accepted_time_excludes_rejections") is True and
             cvsim21.get("rollback", {}).get("rejected_steps") == 1,
             "CVSim21 blood mass rollback boundary changed")
    cvsim21_scope = cvsim21.get("scope", {})
    _require(cvsim21.get("qualification") == "source_absolute_blood_mass_candidate" and
             cvsim21_scope.get("source_aggregate_absolute_blood_mass") is True and
             cvsim21_scope.get("anatomical_registration") is False and
             cvsim21_scope.get("mechanical_mass_owner") is False and
             cvsim21_scope.get("tissue_exchange") is False and
             cvsim21_scope.get("material_density_calibrated") is False and
             cvsim21_scope.get("subject_calibration") is False,
             "CVSim21 blood mass candidate promoted an anatomical owner")
    cvsim21_mass = cvsim21_conservation.get("initial", {}).get("mass_kg")
    _require(isinstance(cvsim21_mass, (int, float)) and cvsim21_mass > 0.0,
             "CVSim21 source mass is missing")

    vessel = documents["vessel_registration"]
    _schema(vessel, "HumanPack.organ-vessel-registration.v1", labels["vessel_registration"])
    vessel_rows = vessel.get("bindings")
    _require(isinstance(vessel_rows, list) and len(vessel_rows) == 6,
             "vessel registration rows are incomplete")
    vessel_ids = [row.get("member_id") for row in vessel_rows]
    vessel_regions = [row.get("region_id") for row in vessel_rows]
    _require(all(isinstance(value, str) and value for value in vessel_ids) and
             len(set(vessel_ids)) == 6 and
             all(isinstance(value, str) and value for value in vessel_regions) and
             len(set(vessel_regions)) == 6,
             "vessel registration identity is invalid")
    vessel_id_set = set(vessel_ids)
    _require(vessel_id_set <= organ_id_set and not vessel_id_set.intersection(set(blood_member_ids)),
             "vessel registration overlaps the regional blood-bed member partition")
    vessel_volumes = [row.get("source_surface_integral_volume_m3") for row in vessel_rows]
    _require(all(type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0.0
                for value in vessel_volumes),
             "vessel surface volume candidate is invalid")
    _all_none(vessel_rows, ("cross_section_area_m2", "material_density_kg_per_m3",
                            "mechanical_mass_owner"), "vessel registration")
    _require(all(row.get("tubular_field_registered") is False and
                 row.get("body_link_registration") is False and
                 row.get("subject_calibration") is False and
                 row.get("pressure_gradient_momentum_transfer") is False
                 for row in vessel_rows),
             "vessel registration promoted a tubular or calibrated owner")
    vessel_qualification = vessel.get("qualification", {})
    _require(vessel_qualification.get("source_membership_and_hashes") is True and
             vessel_qualification.get("source_to_world_frame_registered") is True and
             vessel_qualification.get("centreline_and_area") is False and
             vessel_qualification.get("blood_mass_owner") is False and
             vessel_qualification.get("two_way_tissue_exchange") is False and
             vessel_qualification.get("subject_calibration") is False,
             "vessel registration qualification boundary changed")
    vessel_surface_volume = math.fsum(float(value) for value in vessel_volumes)

    surface_member_ids = [row.get("member_id") for row in surface_rows]
    _require(all(isinstance(value, str) and value for value in surface_member_ids),
             "muscle/tendon surface identity is invalid")
    surface_ids = set(surface_member_ids)
    _require(len(surface_ids) == 150 and not surface_ids.intersection(organ_id_set),
             "muscle/tendon visual surfaces overlap organ member ownership")
    skin_member_id = skin_source["bodyparts3d_member_id"]
    _require(skin_member_id not in organ_id_set and skin_member_id not in surface_ids,
             "skin shell member overlaps another source layer")
    source_member_layers = {
        "cardiac_wall_region_identity": len(cardiac_wall_ids),
        "muscle_surface_geometry_audit": geometry_counts["source_surface_count"],
        "muscle_geometric_volume_candidate": len(geometric_volume_owners),
        "organ_surface_candidates": len(organ_id_set),
        "regional_blood_transport": len(set(blood_member_ids)),
        "muscle_tendon_surface_identity": len(surface_ids),
        "skin_shell_surface_identity": 1,
        "tissue_calibration_candidate": 1,
        "vessel_surface_identity": len(vessel_id_set),
    }

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.body-composition-integration.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "profile": _input(profile, profile_document, _sha256(profile.read_bytes())),
            **{
                name: _input(paths[name], documents[name], hashes[name])
                for name in paths
            },
        },
        "source_member_layers": source_member_layers,
        "identity_bindings": {
            "cardiac_wall_region_ids_sha256": _identity_digest(cardiac_wall_ids),
            "tissue_calibration_id_sha256": _identity_digest({calibration_id}),
            "organ_member_ids_sha256": _identity_digest(organ_id_set),
            "blood_transport_member_ids_sha256": _identity_digest(set(blood_member_ids)),
            "muscle_tendon_surface_ids_sha256": _identity_digest(surface_ids),
            "muscle_surface_geometry_audit_sha256": hashes["muscle_geometry_audit"],
            "muscle_surface_geometry_source_archive_sha256": geometry_source["source_archive_sha256"],
            "muscle_geometric_volume_candidate_sha256": hashes["muscle_geometric_volume_candidate"],
            "muscle_geometric_volume_owner_ids_sha256": _identity_digest(
                {row["owner_id"] for row in geometric_volume_owners}
            ),
            "skin_shell_member_ids_sha256": _identity_digest({skin_member_id}),
            "skin_native_visual_receipt_sha256": hashes["skin_native_visual"],
            "foot_contact_registration_receipt_sha256": hashes["foot_contact_registration"],
            "vessel_surface_ids_sha256": _identity_digest(vessel_id_set),
            "blood_members_subset_of_organ_members": True,
            "vessel_members_subset_of_organ_members": True,
            "vessel_members_disjoint_from_blood_members": True,
            "surface_ids_disjoint_from_organ_members": True,
            "cardiac_wall_manifest_config_hash_matches": (
                cardiac_wall["source_config_sha256"] == hashes["cardiac_wall_config"]
            ),
            "tissue_calibration_is_single_plug": True,
            "tissue_calibration_qualified": False,
            "native_current_release_receipt_sha256": hashes["native_current_release"],
            "native_current_release_source_commit": native_source["commit"],
            "native_current_release_binary_sha256": native_source["binary_sha256"],
            "native_current_release_exact_clock": True,
            "native_current_release_temporal_drift_observed": True,
            "native_costal_tissue_receipt_sha256": hashes["native_costal_tissue"],
            "native_costal_tissue_source_commit": native_costal_source["commit"],
            "native_costal_tissue_output_sha256": native_costal["output"]["sha256"],
            "native_regional_exchange_receipt_sha256": hashes["native_regional_exchange"],
            "native_regional_exchange_source_commit": native_exchange_source["commit"],
            "native_regional_exchange_binary_sha256": native_exchange_source["binary_sha256"],
            "native_regional_exchange_fixture_sha256": native_exchange_inputs["fixture"]["sha256"],
            "transport_and_exchange_beds_share_clock": (
                transport["clock"]["nanoseconds"] == exchange["clock"]["nanoseconds"]
            ),
        },
        "candidate_mass_budgets": {
            "organ_surface_candidate_mass_kg": organ["totals"]["candidate_surface_mass_kg"],
            "regional_costal_tissue_candidate_mass_kg": regional_mass,
            "cardiac_hydraulic_blood_candidate_mass_kg": cardiac_mass,
            "cvsim21_aggregate_blood_mass_kg": cvsim21_mass,
            "muscle_surface_area_candidate_m2": geometry_summary["surface_area_total_m2"],
            "muscle_algebraic_volume_candidate_m3": geometry_summary["algebraic_volume_total_m3"],
            "muscle_single_closed_surface_volume_candidate_count": geometry_counts["surface_volume_candidate_count"],
            "muscle_closed_geometric_volume_candidate_m3": geometric_volume_geometry["closed_muscle_volume_total_m3"],
            "registered_vessel_surface_integral_volume_m3": vessel_surface_volume,
            "cardiac_wall_geometric_volume_candidate_m3": cardiac_wall_volume,
            "skin_shell_outer_surface_vertex_count": skin_source["outer_surface_vertex_count"],
            "skin_shell_outer_surface_triangle_count": skin_source["outer_surface_triangle_count"],
            "skin_native_visual_view_count": len(visual_views),
            "skin_native_visual_positive_pixel_views": sum(
                1 for row in visual_views if row.get("skin_shell_pixels", 0) > 0
            ),
            "foot_registered_source_meshes": foot_counts["registered_source_mesh_count"],
            "foot_active_support_witnesses": foot_support["active_contact_count"],
            "foot_support_witnesses": foot_support["contact_count"],
            "tissue_calibration_training_observations": training_fit["observations"],
            "tissue_calibration_held_out_observations": held_out_fit["observations"],
            "sum_is_mechanical_body_mass": False,
        },
        "ownership": {
            "organ_physical_volume_owner_count": 0,
            "blood_anatomical_physical_volume_owner_count": 0,
            "blood_mechanical_mass_owner_count": 0,
            "skeletal_muscle_tissue_volume_owner_count": 0,
            "fat_volume_and_mass_owner_count": 0,
            "skin_volume_and_mass_owner_count": 0,
            "tendon_fascia_volume_and_mass_owner_count": 0,
            "cardiac_wall_physical_volume_owner_count": 0,
            "skin_physical_volume_owner_count": 0,
            "calibrated_tissue_material_owner_count": 0,
            "whole_body_dynamic_mass_matrix_owner_count": 0,
            "cross_domain_physical_owner_duplicates": 0,
        },
        "runtime_evidence": {
            "clock_nanoseconds": 12500,
            "blood_transport_accepted_steps": transport["accepted_steps"],
            "blood_transport_rejected_steps": transport["rejected_steps"],
            "cvsim21_mass_accepted_steps": cvsim21["accepted_steps"],
            "cvsim21_mass_rejected_steps": cvsim21["rejected_steps"],
            "cvsim21_mass_conserved": cvsim21_conservation["mass_conserved"],
            "cvsim21_volume_conserved": cvsim21_conservation["volume_conserved"],
            "blood_mass_conserved": transport["conservation"]["mass_conserved"],
            "tissue_oxygen_conserved": exchange["conservation"]["oxygen_conserved"],
            "rejected_step_state_neutral": (
                transport["rollback"]["rejected_candidate_state_neutral"] and
                exchange["rollback"]["rejected_candidate_state_neutral"]
            ),
            "muscle_activation_routes": act_counts["source_routes"],
            "muscle_surface_routes_with_binding": surface_counts["routes_with_surface_binding"],
            "muscle_surface_routes_without_binding": surface_counts["routes_without_surface_binding"],
            "muscle_surface_geometry_recomputed": geometry_counts["topology_recomputed_surface_count"],
            "muscle_surface_geometry_single_closed_components": geometry_counts["single_closed_component_count"],
            "muscle_geometric_volume_candidate_count": len(geometric_volume_owners),
            "muscle_surface_geometry_closed_multi_components": geometry_counts["closed_multi_component_count"],
            "muscle_surface_geometry_topology_defects": geometry_counts["topology_defective_count"],
            "muscle_surface_area_candidate_m2": geometry_summary["surface_area_total_m2"],
            "cardiac_wall_tetrahedra": cardiac_wall_mesh["cells"],
            "cardiac_wall_positive_orientation": True,
            "cardiac_wall_regional_geometry_volume_m3": cardiac_wall_volume,
            "skin_shell_outer_surface_vertices": skin_source["outer_surface_vertex_count"],
            "skin_shell_outer_surface_triangles": skin_source["outer_surface_triangle_count"],
            "skin_shell_rest_pose_reconstruction_max_error_m": skin_counts["rest_pose_reconstruction_max_error_m"],
            "skin_native_visual_admission": skin_visual_qualification["native_visual_admission"],
            "skin_native_visual_frame_dimension": skin_visual_capture["frame_dimension"],
            "skin_native_visual_positive_pixel_views": len(visual_views),
            "foot_source_geometry_registered": foot_qualification["source_foot_geometry_registered"],
            "foot_support_witness_identity_bound": foot_qualification["support_witness_identity_bound"],
            "foot_anatomical_supports_loading": foot_qualification["anatomical_supports_loading"],
            "tissue_calibration_training_observations": training_fit["observations"],
            "tissue_calibration_held_out_observations": held_out_fit["observations"],
            "tissue_calibration_held_out_force_nrmse": held_out_fit["force_nrmse_relative_to_measured_rms"],
            "native_release_completed_steps": native_512["persistent_completed_steps"],
            "native_release_horizon_seconds": native_comparison["default_implicit_512_horizon_seconds"],
            "native_release_peak_acceleration_mps2": native_512["persistent_max_acceleration"],
            "native_release_zero_penetration": native_512["persistent_max_penetration_m"] == 0,
            "native_release_temporal_drift_observed": True,
            "native_costal_tissue_replay_cases": native_costal_results["metal_replay_cases"],
            "native_costal_tissue_mass_conserved": native_costal_qualification["mass_conservation"],
            "native_costal_tissue_mass_kg": native_costal_results["tissue_mass_kg"],
            "native_costal_tissue_whole_body_mass_owner": native_costal_qualification["whole_body_dynamic_mass_matrix"],
            "native_regional_exchange_attempted_steps": native_exchange_results["attempted_steps"],
            "native_regional_exchange_accepted_steps_environment_0": native_exchange_results["accepted_steps_environment_0"],
            "native_regional_exchange_rejected_steps_environment_0": native_exchange_results["rejected_step_environment_0"],
            "native_regional_exchange_volume_residual": native_exchange_results["maximum_relative_volume_residual"],
            "native_regional_exchange_blood_mass_residual": native_exchange_results["maximum_relative_blood_mass_residual"],
            "native_regional_exchange_oxygen_residual": native_exchange_results["maximum_relative_oxygen_residual"],
            "native_regional_exchange_conserved": native_exchange_qualification["accepted_step_conservation"],
            "native_regional_exchange_mechanical_blood_mass_owner": native_exchange_qualification["mechanical_blood_mass_owner"],
        },
        "qualification": {
            "source_identity_graph_bound": True,
            "organ_surface_mass_candidate_bound": True,
            "regional_blood_transport_bound": True,
            "source_aggregate_blood_mass_bound": True,
            "source_vessel_registration_bound": True,
            "cardiac_wall_source_identity_bound": True,
            "tissue_calibration_candidate_bound": True,
            "native_current_release_bound": True,
            "native_current_release_sustained_standing": False,
            "native_costal_tissue_requalification_bound": True,
            "native_costal_tissue_whole_body_mass_owner": False,
            "native_regional_exchange_requalification_bound": True,
            "native_regional_exchange_mechanical_blood_mass_owner": False,
            "native_regional_exchange_anatomical_lumen": False,
            "native_regional_exchange_oxygen_exchange": True,
            "tissue_oxygen_exchange_candidate_bound": True,
            "muscle_activation_route_identity_bound": True,
            "muscle_surface_identity_bound": True,
            "muscle_surface_geometry_audit_bound": True,
            "muscle_surface_algebraic_volume_candidates_bound": True,
            "muscle_geometric_volume_candidate_bound": True,
            "skin_shell_source_identity_bound": True,
            "skin_shell_native_visual_admission": True,
            "foot_contact_source_registration_bound": True,
            "fat_source_absence_bound": True,
            "cross_domain_owner_nonduplication_checked": True,
            "anatomical_physical_volume_owners": False,
            "mechanical_mass_owners": False,
            "skeletal_muscle_tissue_volume": False,
            "fat_geometry_and_mass": False,
            "skin_geometry_and_mass": False,
            "anatomical_supports_loading": False,
            "dynamic_foot_contact": False,
            "anatomical_blood_mass_transfer": False,
            "organ_mechanics": False,
            "cardiac_wall_native_mechanics": False,
            "material_calibration": False,
            "subject_calibration": False,
            "integrated_human_qualification": False,
        },
        "boundary": (
            "This record joins source organ candidate moments, regional tissue "
            "mass, the exact-clock CVSim21 aggregate blood mass, regional blood "
            "transport, six-vessel source/world registration, tissue oxygen "
            "exchange, cardiac blood budget, 24-region Rodero cardiac-wall "
            "source identity, one-plug held-out tissue calibration candidate, "
            "muscle route activation, NHTISS4 surface identity, the hash-locked "
            "BodyParts3D muscle/tendon topology and surface-area audit, and the exact "
            "BodyParts3D FJ2810 full-skin visual shell identity and its "
            "four-view native Apple M4 Pro visual admission. The current native "
            "12.5 microsecond Human release is also hash-bound through its 512-step "
            "M4 Pro receipt; its temporal drift remains visible and does not promote "
            "sustained standing. The current costal tissue transaction is likewise "
            "hash-bound to the same native source owner, but remains regional and "
            "does not promote a whole-body mass owner. The current native regional "
            "blood/oxygen transaction is hash-bound to the same source owner and "
            "exact 12.5 microsecond clock; its conservation and rollback are admitted "
            "only as source-graph amount transport, with no anatomical lumen, tissue "
            "volume, mechanical blood mass, organ mechanics, material or subject "
            "calibration. It "
            "proves source identity and nonduplicated ownership bookkeeping only. "
            "The exact bilateral foot source registration and six active support "
            "witness identities are bound as a contact handoff, but they are not "
            "anatomical colliders or calibrated dynamic loading. "
            "The cardiac wall has no physical-volume or mechanical owner here; its "
            "imported boundary defects, unloaded reference, closure/material data, "
            "pressure ports and subject calibration remain open. Fat and "
            "skeletal-muscle tissue volumes, even where algebraic single-closed "
            "surface candidates exist, skin thickness/material/mechanics, "
            "fat geometry and mass, anatomical blood/lumen and organ mechanics, "
            "calibrated materials, subject calibration, and the whole-body "
            "mechanical owner remain unresolved. The tissue fit is a same-plug "
            "development candidate with no native solver validation and must not be "
            "applied to other tissue or rigid-body dynamics. Candidate mass budgets "
            "must not be added to the rigid-body dynamics."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload,
                 "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(
        profile=arguments.profile,
        organ_mass=arguments.organ_mass,
        regional_tissue=arguments.regional_tissue,
        surfaces=arguments.surfaces,
        muscle_geometry_audit=arguments.muscle_geometry_audit,
        muscle_geometric_volume=arguments.muscle_geometric_volume,
        skin_shell=arguments.skin_shell,
        activation=arguments.activation,
        blood_transport=arguments.blood_transport,
        tissue_exchange=arguments.tissue_exchange,
        cardiac_blood=arguments.cardiac_blood,
        cvsim21_blood_mass=arguments.cvsim21_blood_mass,
        vessel_registration=arguments.vessel_registration,
        cardiac_wall_source=arguments.cardiac_wall_source,
        cardiac_wall_config=arguments.cardiac_wall_config,
        tissue_calibration=arguments.tissue_calibration,
        native_current_release=arguments.native_current_release,
        native_costal_tissue=arguments.native_costal_tissue,
        native_regional_exchange=arguments.native_regional_exchange,
    )
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "status": result["status"],
        "source_member_layers": result["source_member_layers"],
        "qualification": result["qualification"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--organ-mass", type=Path, default=ORGAN_MASS)
    parser.add_argument("--regional-tissue", type=Path, default=REGIONAL_TISSUE)
    parser.add_argument("--surfaces", type=Path, default=SURFACES)
    parser.add_argument("--muscle-geometry-audit", type=Path, default=MUSCLE_GEOMETRY_AUDIT)
    parser.add_argument("--muscle-geometric-volume", type=Path, default=MUSCLE_GEOMETRIC_VOLUME)
    parser.add_argument("--skin-shell", type=Path, default=SKIN_SHELL)
    parser.add_argument("--activation", type=Path, default=ACTIVATION)
    parser.add_argument("--blood-transport", type=Path, default=BLOOD_TRANSPORT)
    parser.add_argument("--tissue-exchange", type=Path, default=TISSUE_EXCHANGE)
    parser.add_argument("--cardiac-blood", type=Path, default=CARDIAC_BLOOD)
    parser.add_argument("--cvsim21-blood-mass", type=Path, default=CVSIM21_BLOOD_MASS)
    parser.add_argument("--vessel-registration", type=Path, default=VESSEL_REGISTRATION)
    parser.add_argument("--cardiac-wall-source", type=Path, default=CARDIAC_WALL_SOURCE)
    parser.add_argument("--cardiac-wall-config", type=Path, default=CARDIAC_WALL_CONFIG)
    parser.add_argument("--tissue-calibration", type=Path, default=TISSUE_CALIBRATION)
    parser.add_argument("--native-current-release", type=Path, default=NATIVE_RELEASE)
    parser.add_argument("--native-costal-tissue", type=Path, default=NATIVE_COSTAL_TISSUE)
    parser.add_argument("--native-regional-exchange", type=Path, default=NATIVE_REGIONAL_EXCHANGE)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
