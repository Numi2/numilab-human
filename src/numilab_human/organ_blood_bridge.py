"""Bind source organ moments to the unchanged CVSim cavity references.

This module is a provenance bridge between two source observables.  It does
not turn an atlas surface integral into a physical cavity volume or assign a
mechanical blood owner.  The bridge is useful because it makes the exact
member identity, source-frame moments, and hydraulic comparison auditable in
one immutable record before anatomical registration or calibration exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import cvsim21_anatomy
from .model import ImportError as HumanImportError
from .organ_geometry_moments import compile_moments
from .physiology import canonical, read_json

ROOT = Path(__file__).resolve().parents[2]
MOMENTS = ROOT / "Docs/media/organ-geometry-moments-20260913/moments.json"
TEMPLATE = ROOT / "config/physiology-organ-network-template.v1.json"
SCHEMA = "HumanPack.organ-blood-cavity-bridge.v1"
CHAMBER_INDICES = (15, 16, 19, 20)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ blood cavity bridge: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256(canonical(value) + b"\n")


def _finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _load_current_moments(path: Path, *, sources: Path, source_lock: Path,
                         template: Path) -> tuple[dict[str, Any], str, str]:
    """Require the supplied immutable moments receipt to equal current inputs."""
    try:
        recorded = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise HumanImportError(f"cannot read moments receipt: {path}") from error
    actual = compile_moments(sources=sources, source_lock=source_lock, template=template)
    require(recorded.get("schema") == actual["schema"] == "HumanPack.organ-geometry-moments.v1",
            "unsupported moments receipt schema")
    require(canonical(recorded) == canonical(actual),
            "moments receipt does not match the pinned source inputs")
    payload_sha = _sha256(canonical(actual) + b"\n")
    raw_sha = _sha256(path.read_bytes())
    require(raw_sha == payload_sha, "moments receipt has a noncanonical encoding")
    return actual, payload_sha, raw_sha


def compile_bridge(*, sources: Path = ROOT / "Sources",
                   source_lock: Path = ROOT / "sources.lock.json",
                   template: Path = TEMPLATE,
                   moments: Path = MOMENTS) -> dict[str, Any]:
    """Compile and verify the source-to-hydraulic comparison record."""
    sources = Path(sources)
    source_lock = Path(source_lock)
    template = Path(template)
    moments = Path(moments)
    source_moments, moments_sha, moments_file_sha = _load_current_moments(
        moments, sources=sources, source_lock=source_lock, template=template
    )
    _, cavity_manifest = cvsim21_anatomy.compile_registration(
        sources=sources, source_lock=source_lock,
        config=read_json(cvsim21_anatomy.CONFIG),
    )
    records_by_id = {row["member_id"]: row for row in source_moments["members"]}
    region_ids_by_member: dict[str, list[str]] = {}
    for region in source_moments["regions"]:
        for member_id in region["member_ids"]:
            region_ids_by_member.setdefault(member_id, []).append(region["id"])

    bindings: list[dict[str, Any]] = []
    for index in CHAMBER_INDICES:
        binding = next((row for row in cavity_manifest["bindings"] if row["source_index"] == index), None)
        require(binding is not None, f"missing CVSim cavity binding for source index {index}")
        member_id = binding["member_id"]
        source_row = records_by_id.get(member_id)
        require(source_row is not None, f"cavity member is absent from the organ inventory: {member_id}")
        require(source_row["moment_status"] == "computed_single_closed_component",
                f"cavity member has no single closed source moment: {member_id}")
        cavity = next((row for row in cavity_manifest["cavity_geometry"]["chambers"]
                       if row["member_id"] == member_id), None)
        require(cavity is not None, f"cavity geometry row is absent: {member_id}")
        source_hash = cavity["source"]["sha256"]
        require(source_hash == source_row["source_sha256"], f"source member hash mismatch: {member_id}")
        source_mom = source_row["source_surface_moments"]
        cavity_mom = binding["geometry_moments"]
        require(source_mom["absolute_signed_volume_m3"] == cavity_mom["absolute_signed_volume_m3"],
                f"cavity and organ moment volume mismatch: {member_id}")
        surface_volume = _finite(source_mom["absolute_signed_volume_m3"], f"surface volume {member_id}")
        hydraulic_initial = _finite(binding["hydraulic_initial_volume_m3"], f"hydraulic initial volume {member_id}")
        hydraulic_reference = _finite(binding["hydraulic_reference_volume_m3"], f"hydraulic reference volume {member_id}")
        require(surface_volume > 0 and hydraulic_initial > 0 and hydraulic_reference > 0,
                f"nonpositive comparison volume: {member_id}")
        bindings.append({
            "source_index": index,
            "compartment_stable_identifier": binding["compartment_stable_identifier"],
            "source_label": binding["source_label"],
            "semantic_id": binding["semantic_id"],
            "member_id": member_id,
            "source_member_sha256": source_hash,
            "organ_region_ids": sorted(region_ids_by_member.get(member_id, [])),
            "source_moment_status": source_row["moment_status"],
            "source_surface_integral_volume_m3": surface_volume,
            "source_centroid_frame_m": source_mom["centroid_source_frame_m"],
            "source_first_volume_moment_m4": source_mom["first_volume_moment_m4"],
            "source_central_second_volume_moment_m5": source_mom["central_second_volume_moment_m5"],
            "hydraulic_initial_volume_m3": hydraulic_initial,
            "hydraulic_reference_volume_m3": hydraulic_reference,
            "initial_hydraulic_to_source_integral_volume_ratio": hydraulic_initial / surface_volume,
            "reference_hydraulic_to_source_integral_volume_ratio": hydraulic_reference / surface_volume,
            "hydraulic_volume_authority": binding["physical_volume_owner_id"],
            "physical_volume_owner": None,
            "mechanical_mass_owner": None,
            "density_kg_per_m3": None,
            "world_or_body_frame_registration": False,
            "pressure_gradient_momentum_transfer": False,
            "candidate_status": "source_identity_and_hydraulic_comparison_only",
        })

    require(len(bindings) == 4 and len({row["member_id"] for row in bindings}) == 4,
            "cavity bridge must contain four distinct chambers")
    identity = {
        "source_moment_receipt_sha256": moments_sha,
        "source_inventory_sha256": source_moments["inventory_sha256"],
        "cvsim_source_native_sha256": cavity_manifest["source_native_sha256"],
        "cvsim_cavity_geometry_sha256": cavity_manifest["cavity_geometry_sha256"],
        "bindings": bindings,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-blood-cavity-bridge.1",
        "source_moment_receipt": {
            "path": (str(moments.relative_to(ROOT))
                     if moments.is_relative_to(ROOT) else str(moments)),
            "sha256": moments_sha,
            "file_sha256": moments_file_sha,
            "schema": source_moments["schema"],
            "inventory_sha256": source_moments["inventory_sha256"],
        },
        "cvsim_cavity_reference": {
            "schema": cavity_manifest["schema"],
            "config_sha256": _canonical_sha(cavity_manifest["config"]),
            "source_native_sha256": cavity_manifest["source_native_sha256"],
            "native_content_sha256": cavity_manifest["native_content_sha256"],
            "cavity_geometry_sha256": cavity_manifest["cavity_geometry_sha256"],
            "source_blood_volume_m3": cavity_manifest["source_blood_volume_m3"],
            "hydraulic_parameters_changed": cavity_manifest["hydraulic_parameters_changed"],
            "intersecting_triangle_pairs": sum(
                pair["count"] for pair in cavity_manifest["intersection_audit"]["per_pair"]
            ),
            "all_cavity_domains_disjoint": cavity_manifest["intersection_audit"]["all_domains_disjoint"],
        },
        "identity_sha256": _canonical_sha(identity),
        "bindings": bindings,
        "qualification": {
            "exact_source_member_hash_binding": True,
            "exact_cavity_semantic_binding": True,
            "source_frame_moments_reused": True,
            "hydraulic_surface_volume_comparison": True,
            "cavity_domains_disjoint": cavity_manifest["intersection_audit"]["all_domains_disjoint"],
            "physical_volume_authority_assigned": False,
            "blood_mass_assigned": False,
            "world_or_body_frame_registered": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_blood_tissue_transfer": False,
            "material_density_calibrated": False,
            "physiological_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Four exact BodyParts3D cavity members are hash-bound to the current CVSim21 "
            "hydraulic chamber references and the current source-frame moments receipt. "
            "The surface integral is a source geometry observable, not an admitted physical "
            "volume. CVSim remains the hydraulic volume authority; cavity domains are not "
            "disjoint, so no physical organ-volume or blood-mass ownership is promoted. "
            "Body registration, pressure-gradient momentum transfer, two-way tissue coupling, "
            "material density, calibration, and behavior remain unqualified."
        ),
    }
    canonical(result)
    return result


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--moments", type=Path, default=MOMENTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_bridge(sources=args.sources, source_lock=args.source_lock,
                                template=args.template, moments=args.moments)
        digest = _immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "cavity_bindings": len(result["bindings"]),
                          "physical_volume_authority_assigned": False,
                          "blood_mass_assigned": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"organ blood cavity bridge: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
