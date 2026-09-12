"""Associate source CVSim chambers with exact atlas cavity reference surfaces.

This is offline anatomy authoring. Atlas geometry and CVSim blood volumes are
different source observables; neither is rescaled to make the other agree.
Matter remains the only hydraulic runtime, with unchanged physical parameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import cvsim21 as cv
from . import cardiac_cavity_geometry as geometry
from .cardiac_cavity_intersections import audit_cavity_intersections
from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/cvsim21-cardiac-cavities.v1.json"
SCHEMA = "HumanPack.cvsim21-cardiac-cavities.v1"
CHAMBER_INDICES = (15, 16, 19, 20)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("CVSim21 cavity registration: " + message)


def _sha(value: object) -> str:
    return hashlib.sha256(cv.canonical(value) + b"\n").hexdigest()


def geometry_moments(vertices: list, triangles: list) -> dict:
    """Integrate signed tetrahedra in a local frame, without density or stepping.

    The identities are algebraic for closed oriented surfaces. Embeddedness and
    mutual nonintersection require separate evidence before physical use.
    """
    topology = geometry.analyze_topology(vertices, triangles)
    require(topology["closed_oriented_manifold_candidate"]
            and not topology["unused_vertex_ids"]
            and topology["face_component_count"] == 1,
            "moments require one closed oriented vertex-manifold surface")
    require(len({tuple(v) for v in vertices}) == len(vertices), "moments require quotient vertices")
    origin = [math.fsum(v[k] for v in vertices) / len(vertices) for k in range(3)]
    points = [[v[k] - origin[k] for k in range(3)] for v in vertices]
    volumes, first, second = [], [[] for _ in range(3)], [[[] for _ in range(3)] for _ in range(3)]
    for face in triangles:
        a, b, c = [points[i] for i in face]
        cross = [b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]]
        volume = math.fsum(a[k]*cross[k] for k in range(3)) / 6
        volumes.append(volume)
        summed = [math.fsum(p[k] for p in (a, b, c)) for k in range(3)]
        for i in range(3):
            first[i].append(volume * summed[i] / 4)
            for j in range(3):
                second[i][j].append(volume * (summed[i]*summed[j]
                    + math.fsum(p[i]*p[j] for p in (a, b, c))) / 20)
    signed_volume = math.fsum(volumes)
    require(math.isfinite(signed_volume) and signed_volume != 0, "zero or nonfinite signed surface volume")
    sign = 1 if signed_volume > 0 else -1
    volume = abs(signed_volume)
    local_first = [sign*math.fsum(v) for v in first]
    local_second = [[sign*math.fsum(v) for v in row] for row in second]
    local_center = [v / volume for v in local_first]
    center = [local_center[i] + origin[i] for i in range(3)]
    central = [[local_second[i][j] - volume*local_center[i]*local_center[j] for j in range(3)] for i in range(3)]
    # Normalize only the admission calculation: dimensional moments remain
    # unchanged. Squaring/factoring m^5 entries directly can overflow or
    # underflow for otherwise finite geometry after a uniform change of scale.
    require(all(math.isfinite(x) for row in central for x in row), "nonfinite central volume moment")
    moment_scale = max(abs(x) for row in central for x in row)
    require(moment_scale > 0, "zero central volume moment")
    normalized = [[x / moment_scale for x in row] for row in central]
    require(all(normalized[i][i] > 0 for i in range(3)), "nonpositive central volume moment")
    require(all(normalized[i][i]*normalized[j][j] - normalized[i][j]**2 > 0 for i in range(3) for j in range(i)),
            "nonpositive central volume moment minor")
    determinant = (normalized[0][0]*(normalized[1][1]*normalized[2][2]-normalized[1][2]*normalized[2][1])
        - normalized[0][1]*(normalized[1][0]*normalized[2][2]-normalized[1][2]*normalized[2][0])
        + normalized[0][2]*(normalized[1][0]*normalized[2][1]-normalized[1][1]*normalized[2][0]))
    require(determinant > 0, "nonpositive central volume moment determinant")
    raw_second = [[central[i][j] + volume*center[i]*center[j] for j in range(3)] for i in range(3)]
    result = {
        "method": "oriented_surface_tetrahedral_integrals_about_local_vertex_mean",
        "status": "algebraic_geometry_moments_not_physiological_volume_or_mass",
        "signed_volume_m3": signed_volume, "absolute_signed_volume_m3": volume,
        "source_winding": "positive" if sign > 0 else "negative",
        "centroid_source_frame_m": center,
        "first_volume_moment_m4": [volume*x for x in center],
        "second_volume_moment_m5": raw_second, "central_second_volume_moment_m5": central,
        "inertia_per_unit_density_m5": [[(sum(central[k][k] for k in range(3)) if i == j else 0) - central[i][j]
                                          for j in range(3)] for i in range(3)],
        "physical_volume_m3": None, "density_kg_per_m3": None, "mechanical_mass_kg": None,
        "self_intersection_qualified": False, "interdomain_disjointness_qualified": False,
    }
    cv.canonical(result)
    return result


def compile_registration(*, sources: Path = ROOT / "Sources", source_lock: Path = ROOT / "sources.lock.json",
                         config: dict | None = None) -> tuple[dict, dict]:
    try:
        config = cv.read_json(CONFIG) if config is None else deepcopy(config)
        cv.canonical(config)
        require(isinstance(config, dict) and set(config) == {
            "schema", "id", "volume_coordinates", "source_manifest_sha256", "geometry_source",
            "association", "mechanical_mass_policy"}, "unsupported configuration fields")
        require(config["schema"] == "HumanPack.cvsim21-cardiac-cavities-config.v1"
                and config["id"] == "cvsim21_bodyparts3d_cardiac_cavity_reference", "unsupported configuration")
        require(config["volume_coordinates"] in cv.VARIANTS, "unknown source volume coordinates")
        require(config["source_manifest_sha256"] == cv.SOURCE_LOCK_SHA256, "source identity mismatch")
        expected_geometry = {"archive_sha256": geometry.ARCHIVE_SHA256,
            "part_of_table_sha256": geometry.TABLE_HASHES["partof_element_parts.txt"],
            "is_a_table_sha256": geometry.TABLE_HASHES["isa_element_parts.txt"]}
        require(config["geometry_source"] == expected_geometry, "geometry source identity mismatch")
        require(config["association"] == "source_cavity_reference_only", "unsupported physical association")
        require(config["mechanical_mass_policy"] == "unchanged_unresolved_blood_attribution",
                "blood mass attribution or mutation requires a separate mechanical owner")
        base_config = cv.read_json(cv.CONFIG)
        base_config["volume_coordinates"] = config["volume_coordinates"]
        native, lowering = cv.compile_source(config=base_config)
        baseline = deepcopy(native)
        cavities = geometry.extract_cavity_surfaces(sources=sources, source_lock=source_lock)
        require(len(cavities["chambers"]) == 4, "incomplete cavity coverage")
        intersection_audit = audit_cavity_intersections(cavities)
        require(intersection_audit["all_surfaces_embedded"], "source cavity has self-intersections")
        bindings = []
        for i, cavity, expected in zip(CHAMBER_INDICES, cavities["chambers"], geometry.CAVITIES, strict=True):
            row = native["compartments"][i]
            require(cavity["source_id"] == cv.LABELS[i] == expected[0]
                    and cavity["semantic_id"] == "FMA:" + expected[1][3:]
                    and cavity["member_id"] == expected[3], "cavity semantic correspondence mismatch")
            quotient = cavity["exact_coordinate_quotient"]
            moments = geometry_moments(quotient["vertices_m"], quotient["triangles"])
            moments["self_intersection_qualified"] = True
            moments["embedded_geometric_volume_m3"] = moments["absolute_signed_volume_m3"]
            row["anatomical_region_id"] = cavity["semantic_id"]
            v = moments["absolute_signed_volume_m3"]
            bindings.append({"source_index": i, "compartment_stable_identifier": row["stable_identifier"],
                "source_label": cavity["source_id"], "semantic_id": cavity["semantic_id"], "member_id": cavity["member_id"],
                "physical_volume_owner_id": row["physical_volume_owner_id"], "geometry_moments": moments,
                "hydraulic_initial_volume_m3": row["initial_volume_m3"],
                "hydraulic_reference_volume_m3": row["reference_volume_m3"],
                "initial_hydraulic_to_surface_volume_ratio": row["initial_volume_m3"] / v,
                "hydraulic_minus_surface_volume_m3": row["initial_volume_m3"] - v,
                "geometric_scale_fit": None, "reference_cardiac_phase": None,
                "world_or_body_frame_registration": None, "mechanical_mass_owner": None})
        anatomy_hash = _sha(cavities)
        identity = {"configuration": config, "source_native_sha256": _sha(baseline),
                    "cavity_geometry_sha256": anatomy_hash, "bindings": bindings,
                    "intersection_audit": intersection_audit}
        native["model_id"] += "_cardiac_cavity_reference"
        native["authored_graph_sha256"] = _sha(identity)
        native["source_graph_sha256"] = _sha({"cvsim21": cv.SOURCE_LOCK_SHA256,
                                               "bodyparts3d": expected_geometry})
        manifest = {"schema": SCHEMA, "config": config, "identity_preimage": identity,
            "native_content_sha256": _sha(native), "source_native_sha256": _sha(baseline),
            "cavity_geometry_sha256": anatomy_hash, "cavity_geometry": cavities,
            "intersection_audit": intersection_audit,
            "bindings": bindings, "unregistered_compartment_indices": [i for i in range(21) if i not in CHAMBER_INDICES],
            "hydraulic_parameters_changed": False, "source_blood_volume_m3": lowering["total_blood_volume_m3"],
            "mechanical_mass_policy": config["mechanical_mass_policy"], "added_mechanical_mass_kg": 0.0,
            "withdrawn_mechanical_mass_kg": 0.0, "blood_in_gross_body_mass": "unresolved",
            "qualification": {"source_semantic_cavity_association": True, "source_geometry_moments": True,
                "individual_cavities_embedded": True,
                "all_cavity_domains_disjoint": intersection_audit["all_domains_disjoint"],
                "atlas_to_body_registration": False, "hydraulic_volume_matches_anatomical_volume": False,
                "blood_tissue_mass_partition": False, "pressure_deformation_coupling": False,
                "physiological_calibration": False, "standing_walking": False},
            "boundary": "Four exact embedded source cavity references; unchanged CVSim hydraulics. Intersecting cavity domains block disjoint physical-volume admission. Atlas cardiac phase, subject/body registration, blood mass partition and tissue mechanics remain unqualified."}
        return native, manifest
    except HumanImportError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        raise HumanImportError(f"CVSim21 cavity registration: invalid input: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        native, manifest = compile_registration(sources=args.sources, source_lock=args.source_lock,
                                               config=cv.read_json(args.config))
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        require(args.output.resolve() != manifest_path.resolve(), "payload and manifest paths coincide")
        outputs = [(args.output, native), (manifest_path, manifest)]
        for path, value in outputs:
            require(not path.is_symlink() and (not path.exists() or path.read_bytes() == cv.canonical(value) + b"\n"),
                    "output is immutable; choose a new path")
        for path, value in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(cv.canonical(value) + b"\n")
        print(json.dumps({"status": "compiled_cavity_reference", "output": str(args.output),
            "manifest": str(manifest_path), "sha256": manifest["native_content_sha256"],
            "cavity_associations": 4, "unregistered_compartments": 17,
            "disjoint_cavity_domains": manifest["intersection_audit"]["all_domains_disjoint"],
            "intersecting_triangle_pairs": sum(pair["count"] for pair in manifest["intersection_audit"]["per_pair"]),
            "physiological_calibration": False, "mechanical_mass_changed": False}))
        return 0
    except (HumanImportError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
