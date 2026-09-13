"""Bind the pinned vessel surfaces to the shared MyoSim world frame.

This is a provenance and frame-registration step.  A BodyParts3D surface is
not a vessel tube, a lumen, or a material field, so the compiler deliberately
does not infer a centreline, area, density, blood mass, or pressure reaction.
Those owners remain separate native contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .organ_geometry_moments import compile_moments
from .physiology import canonical, load_anatomy, read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-vessel-registration.v1"
MAP = ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json"
TEMPLATE = ROOT / "config/physiology-organ-network-template.v1.json"
MOMENTS = ROOT / "Docs/media/organ-geometry-moments-20260913/moments.json"
REGISTRATION = ROOT / "Docs/media/numi-human-lower-joint-focus-v1/receipts/registration.v3.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ vessel registration: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _matrix(value: Any) -> list[list[float]]:
    require(isinstance(value, list) and len(value) == 4, "registration transform must be 4x4")
    result = []
    for row_index, row in enumerate(value):
        require(isinstance(row, list) and len(row) == 4,
                f"registration transform row {row_index} is malformed")
        result.append([_finite(component, f"registration transform[{row_index}]") for component in row])
    require(result[3] == [0.0, 0.0, 0.0, 1.0], "registration transform has a non-affine bottom row")
    return result


def _determinant(matrix: list[list[float]]) -> float:
    a, b, c = matrix[0][:3], matrix[1][:3], matrix[2][:3]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _scale_matrix(matrix: list[list[float]], scale: float) -> list[list[float]]:
    return [[scale * value for value in row] for row in matrix]


def _validate_transform(registration: dict[str, Any]) -> tuple[list[list[float]], float]:
    coordinates = registration.get("coordinate_system")
    require(isinstance(coordinates, dict), "registration has no coordinate system")
    matrix = _matrix(coordinates.get("global_source_mm_to_myosim_world_m"))
    linear = [row[:3] for row in matrix[:3]]
    # The declaration is the scale applied to source millimetres after the
    # millimetre-to-metre conversion; the matrix itself therefore carries the
    # same value divided by 1000.
    declared_scale = _finite(coordinates.get("uniform_scale_after_mm_to_m"), "registration uniform scale")
    require(declared_scale > 0.0, "registration uniform scale is not positive")
    scale = declared_scale * 1.0e-3
    # The current candidate is a proper uniform scale plus signed axis
    # permutation.  Enforce that contract so a future shear or reflection is
    # never silently promoted into a mechanics frame.
    nonzero = []
    for row in linear:
        entries = [abs(component) for component in row if abs(component) > 1.0e-12]
        require(len(entries) == 1, "registration transform is not a signed axis permutation")
        nonzero.append(entries[0])
    require(max(abs(value - scale) for value in nonzero) <= 1.0e-12,
            "registration transform scale disagrees with its declaration")
    require(abs(_determinant(matrix) - scale ** 3) <= max(1.0e-18, scale ** 3 * 1.0e-12),
            "registration transform is not proper and orientation preserving")
    permutation = coordinates.get("proper_axis_permutation")
    signs = coordinates.get("proper_axis_signs")
    require(permutation == [0, 1, 2] and signs == [1, 1, 1],
            "registration candidate uses an unsupported axis convention")
    return matrix, declared_scale


def _load_moments(path: Path, *, sources: Path, source_lock: Path, template: Path) -> dict[str, Any]:
    try:
        recorded = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise HumanImportError(f"cannot read moments receipt: {path}") from error
    actual = compile_moments(sources=sources, source_lock=source_lock, template=template)
    require(canonical(recorded) == canonical(actual), "moments receipt does not match pinned source inputs")
    require(_sha256(path.read_bytes()) == _sha256(canonical(actual) + b"\n"),
            "moments receipt has a noncanonical encoding")
    return actual


def compile_registration(
    *,
    sources: Path = ROOT / "Sources",
    source_lock: Path = ROOT / "sources.lock.json",
    anatomy_map: Path = MAP,
    template: Path = TEMPLATE,
    moments: Path = MOMENTS,
    registration: Path = REGISTRATION,
) -> dict[str, Any]:
    sources, source_lock, anatomy_map, template, moments, registration = map(
        Path, (sources, source_lock, anatomy_map, template, moments, registration)
    )
    graph = read_json(template)
    require(graph.get("schema") == "HumanPack.physiology.v1", "unsupported physiology template")
    anatomy = load_anatomy(sources, source_lock)
    surface_map = read_json(anatomy_map)
    require(surface_map.get("schema") == "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1",
            "unsupported vessel anatomy map")
    entries = [row for row in surface_map.get("entries", [])
               if isinstance(row, dict) and row.get("layer") == "vessel"]
    require(len(entries) == 6, "vessel anatomy map must contain six named systemic vessels")
    require(len({row.get("member_id") for row in entries}) == len(entries),
            "vessel anatomy map repeats a source member")

    registration_doc = read_json(registration)
    require(registration_doc.get("schema") == "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
            "unsupported source-to-world registration schema")
    require(registration_doc.get("status") == "provisional_visual_registration_not_admitted_to_collision_or_physics",
            "source registration is not the pinned visual-only candidate")
    source = registration_doc.get("source")
    bodyparts = source.get("bodyparts") if isinstance(source, dict) else None
    require(isinstance(bodyparts, dict) and bodyparts.get("id") == "bodyparts3d_4" and bodyparts.get("version") == "4.0",
            "registration bodyparts provenance is not BodyParts3D 4.0")
    lock = read_json(source_lock)
    expected_archive = lock["sources"]["bodyparts3d_4"]["files"]["partof_BP3D_4.0_obj_99.zip"]["sha256"]
    archives = bodyparts.get("archives", [])
    partof = next((row for row in archives if isinstance(row, dict) and row.get("hierarchy") == "part_of"), None)
    require(isinstance(partof, dict) and partof.get("sha256") == expected_archive,
            "registration archive provenance does not match the source lock")
    matrix, uniform_scale = _validate_transform(registration_doc)
    moments_doc = _load_moments(moments, sources=sources, source_lock=source_lock, template=template)
    moments_by_member = {row["member_id"]: row for row in moments_doc["members"]}
    region_by_id = {row["id"]: row for row in graph["regions"]}
    compartment_by_id = {row["id"]: row for row in graph["compartments"]}
    relation_tables = anatomy["tables"]

    bindings: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda row: row["member_id"]):
        required = {"concept_id", "source_name", "member_id", "hierarchy", "myosim_body"}
        require(required <= entry.keys(), "vessel anatomy map entry is incomplete")
        member_id = entry["member_id"]
        hierarchy = entry["hierarchy"]
        relation = relation_tables.get(hierarchy, {})
        require((entry["concept_id"], entry["source_name"]) in relation and
                member_id in relation[(entry["concept_id"], entry["source_name"])],
                f"vessel source relation drifted: {member_id}")
        # Source labels and graph IDs are intentionally kept separate.  The
        # map above is small and exact; reject an unmapped label rather than
        # guessing a compartment from a name.
        label_to_region = {
            "ascending aorta": "ascending_aorta", "arch of aorta": "aortic_arch",
            "descending aorta": "descending_aorta", "abdominal aorta": "abdominal_aorta",
            "superior vena cava": "superior_vena_cava", "inferior vena cava": "inferior_vena_cava",
        }
        region_id = label_to_region.get(entry["source_name"])
        require(region_id in region_by_id and region_by_id[region_id]["member_ids"].count(member_id) == 1,
                f"vessel is absent from the physiology region: {member_id}")
        require(region_id in compartment_by_id, f"vessel has no hydraulic compartment: {region_id}")
        row = moments_by_member.get(member_id)
        require(row is not None and row["moment_status"] == "computed_single_closed_component",
                f"vessel has no single closed source moment: {member_id}")
        source_moments = row["source_surface_moments"]
        source_volume = _finite(source_moments["absolute_signed_volume_m3"], f"source volume {member_id}")
        source_centroid = [_finite(value, f"source centroid {member_id}")
                           for value in source_moments["centroid_source_frame_m"]]
        central = source_moments["central_second_volume_moment_m5"]
        require(source_volume > 0.0 and isinstance(central, list) and len(central) == 3,
                f"vessel source moments are incomplete: {member_id}")
        central = [[_finite(value, f"source central moment {member_id}") for value in values]
                   for values in central]
        # The source moments are already in metres (the moments compiler
        # applies the authored millimetre-to-metre conversion).  The pinned
        # registration matrix is named ``source_mm_to_world_m`` and therefore
        # must not be applied to those metre-valued moments a second time.
        # Apply the declared post-mm uniform scale once, then add its world
        # translation.  The old path multiplied metre moments by 1e-3 again,
        # producing volumes and second moments 1e-9/1e-15 too small.
        world_centroid = [
            uniform_scale * source_centroid[index] + matrix[index][3]
            for index in range(3)
        ]
        world_central = _scale_matrix(central, uniform_scale ** 5)
        world_volume = uniform_scale ** 3 * source_volume
        compartment = compartment_by_id[region_id]
        bindings.append({
            "region_id": region_id,
            "source_name": entry["source_name"],
            "semantic_id": region_by_id[region_id]["semantic_id"],
            "member_id": member_id,
            "source_member_sha256": row["source_sha256"],
            "hierarchy": hierarchy,
            "myosim_body": entry["myosim_body"],
            "compartment_stable_identifier": list(compartment_by_id).index(region_id) + 1,
            "hydraulic_volume_owner_id": compartment["physical_volume_owner_id"],
            "source_surface_integral_volume_m3": source_volume,
            "registered_world_surface_integral_volume_m3": world_volume,
            "source_centroid_m": source_centroid,
            "registered_world_centroid_m": world_centroid,
            "registered_world_central_second_volume_moment_m5": world_central,
            "world_frame_registration": True,
            "body_link_registration": False,
            "tubular_field_registered": False,
            "centreline_registered": False,
            "cross_section_area_m2": None,
            "material_density_kg_per_m3": None,
            "mechanical_mass_owner": None,
            "pressure_gradient_momentum_transfer": False,
            "subject_calibration": False,
        })

    identity = {
        "anatomy_map_sha256": _sha256(anatomy_map.read_bytes()),
        "source_lock_sha256": _sha256(source_lock.read_bytes()),
        "moments_file_sha256": _sha256(moments.read_bytes()),
        "registration_file_sha256": _sha256(registration.read_bytes()),
        "source_archive_sha256": expected_archive,
        "bindings": bindings,
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-vessel-registration.1",
        "source": {
            "anatomy_map": str(anatomy_map.relative_to(ROOT)) if anatomy_map.is_relative_to(ROOT) else str(anatomy_map),
            "moments": str(moments.relative_to(ROOT)) if moments.is_relative_to(ROOT) else str(moments),
            "registration": str(registration.relative_to(ROOT)) if registration.is_relative_to(ROOT) else str(registration),
            "registration_sha256": identity["registration_file_sha256"],
            "source_archive_sha256": expected_archive,
            "coordinate_system": "BodyParts3D source metres to MyoSim world metres",
            "moment_unit_conversion": "moments are authored in metres; uniform_scale_after_mm_to_m is applied once",
            "global_source_mm_to_myosim_world_m": matrix,
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "bindings": bindings,
        "qualification": {
            "source_membership_and_hashes": True,
            "source_moments_recomputed": True,
            "source_moment_units_corrected": True,
            "source_to_world_frame_registered": True,
            "body_link_registration": False,
            "tubular_vessel_field": False,
            "centreline_and_area": False,
            "material_density_calibrated": False,
            "blood_mass_owner": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Six exact BodyParts3D vessel surfaces are hash-bound to the pinned "
            "source-to-MyoSim world transform and their source-frame moments are "
            "transformed without assigning a physical volume. The records remain "
            "visual/source geometry: no centreline, lumen area, wall material, "
            "density, blood mass, pressure reaction, tissue exchange, or subject "
            "calibration is inferred."
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
    parser.add_argument("--anatomy-map", type=Path, default=MAP)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--moments", type=Path, default=MOMENTS)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_registration(
            sources=args.sources, source_lock=args.source_lock, anatomy_map=args.anatomy_map,
            template=args.template, moments=args.moments, registration=args.registration,
        )
        output = args.output.resolve()
        digest = _immutable_write(output, result)
        print({"schema": SCHEMA, "output": str(output), "sha256": digest,
               "bindings": len(result["bindings"]), "world_frame_registration": True,
               "body_link_registration": False})
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"organ vessel registration: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
