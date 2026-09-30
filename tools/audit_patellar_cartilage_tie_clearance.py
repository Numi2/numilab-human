"""Audit a bone-tie-fixed cartilage clearance field on both Open Knee sides.

This moves PTC volume nodes only. The exact PTC/PTB tie stays fixed, while
the remaining articular surface follows the previously audited 20 um outward
clearance direction. It is a geometry/current-state candidate, not a change
to the source rest mesh, rigid patella joint, or native accepted knee state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.open_knee import EXPECTED_HASHES, parse_source
from tools.audit_patellar_extensor_pose_candidate import (
    signed_six, surface_embedding, tie_gap,
)
from tools.audit_patellofemoral_pose_clearance import (
    count_intersections, full_boundary,
)
from tools.audit_patellofemoral_tetrahedral_separation import (
    audit_state, self_check, source_tetrahedra,
)
from tools.export_patellofemoral_matter_input import HEADER, region
from tools.verify_patellofemoral_intersections import (
    _compiled_meshes, _intersections,
)
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/open-knee-oks003"
POSE = ROOT / "Docs/media/patellofemoral-pose-clearance-20260930/receipt.json"
JOINT = ROOT / "Docs/media/patellar-joint-pose-admission-20260930/receipt.json"
OUTPUT = ROOT / "Docs/media/patellar-cartilage-tie-clearance-20260930/receipt.json"
BUILD = ROOT / "Build/patellar-cartilage-tie-clearance-20260930"
ALGORITHM = "euclidean-nearest-boundary-distance-ratio-contact-one-bone-tie-zero-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("patellar cartilage tie clearance: " + message)


def source_indices(source, set_name: str) -> np.ndarray:
    region = source.regions["PTC"]
    lookup = {identifier: index for index, identifier in
              enumerate(region.node_ids)}
    return np.asarray([lookup[identifier]
                       for identifier in source.node_sets[set_name]], dtype=np.int64)


def contact_indices(source) -> np.ndarray:
    region = source.regions["PTC"]
    lookup = {identifier: index for index, identifier in
              enumerate(region.node_ids)}
    surface = source.surfaces["PTC_@_FMC_ContactFaces"]
    return np.asarray(sorted({lookup[identifier] for face in surface.faces
                              for identifier in face}), dtype=np.int64)


def clearance_field(points: np.ndarray, source,
                    displacement: np.ndarray) -> tuple[np.ndarray, dict]:
    tie = source_indices(source, "PTC_@_PTB_TiesNodes")
    contact = contact_indices(source)
    require(len(tie) == 4592 and len(contact) == 5683,
            "PTC source boundary sets changed")
    shared = np.intersect1d(tie, contact)
    shared_ids = [int(source.regions["PTC"].node_ids[index]) for index in shared]
    require(shared_ids == [62569, 62577],
            "PTC contact/bone-tie seam identity changed")
    to_tie = cKDTree(points[tie]).query(points, workers=1)[0]
    to_contact = cKDTree(points[contact]).query(points, workers=1)[0]
    denominator = to_tie + to_contact
    require(np.all(np.isfinite(denominator)) and
            np.array_equal(np.flatnonzero(denominator == 0), shared),
            "PTC boundary distance field has an unexpected degeneracy")
    weight = np.divide(to_tie, denominator,
                       out=np.zeros_like(to_tie), where=denominator > 0)
    weight[contact] = 1.0
    weight[tie] = 0.0  # Bone tie wins at the two shared seam vertices.
    require(np.all(np.isfinite(weight)) and np.all((weight >= 0) & (weight <= 1)),
            "PTC clearance weights are invalid")
    current = (points.astype(np.float64) +
               weight[:, None] * displacement).astype("<f4")
    require(np.array_equal(current[tie], points[tie]),
            "PTC/PTB bone-tie coordinates moved")
    return current, {
        "algorithm": ALGORITHM,
        "ptc_bone_tie_nodes_fixed": len(tie),
        "ptc_contact_surface_nodes": len(contact),
        "contact_and_tie_shared_source_node_ids": shared_ids,
        "contact_surface_nodes_with_full_weight": len(contact) - len(shared),
        "contact_surface_seam_nodes_with_zero_weight": len(shared),
        "minimum_weight": float(weight.min()),
        "maximum_weight": float(weight.max()),
        "maximum_node_displacement_m": float(np.linalg.norm(
            current.astype(np.float64) - points, axis=1).max()),
        "current_positions_sha256": sha(current.tobytes()),
    }


def make_native_input(path: Path, side: str, payload_path: Path,
                      pose: dict, ptc: np.ndarray, current: np.ndarray,
                      ptc_tets: np.ndarray, fmc: np.ndarray,
                      fmc_tets: np.ndarray, tie: np.ndarray,
                      producer_sha: str) -> dict:
    displacement = pose["sides"][side]["prescribed_pose_translation_m"]
    candidate_identity = sha(POSE.read_bytes() + b"\n" + producer_sha.encode())
    header = HEADER.pack(
        b"NHCAR1\0\0", 1, 0 if side == "left" else 1,
        len(ptc), len(ptc_tets), len(fmc), len(fmc_tets), 0,
        bytes.fromhex(sha(payload_path.read_bytes())),
        bytes.fromhex(candidate_identity), *displacement,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        for block in (ptc, current, ptc_tets, fmc, fmc_tets):
            stream.write(block.tobytes(order="C"))
    expected = HEADER.size + 12 * (2 * len(ptc) + len(fmc)) + \
        16 * (len(ptc_tets) + len(fmc_tets))
    require(path.stat().st_size == expected, "NHCAR1 input length changed")
    fixed_path = path.with_suffix(".tie.u32le")
    fixed = np.asarray(sorted(int(index) for index in tie), dtype="<u4")
    require(len(fixed) == 4592 and len(np.unique(fixed)) == len(fixed),
            "PTC fixed-node identity changed")
    fixed_path.write_bytes(fixed.tobytes())
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()),
            "bytes": expected, "candidate_identity_sha256": candidate_identity,
            "source_rest_ptc_positions_sha256": sha(ptc.tobytes()),
            "candidate_current_ptc_positions_sha256": sha(current.tobytes()),
            "fixed_node_indices_path": str(fixed_path.relative_to(ROOT)),
            "fixed_node_indices_sha256": sha(fixed_path.read_bytes()),
            "fixed_node_count": len(fixed)}


def run() -> dict:
    self_check()
    source = parse_source(SOURCE)
    pose = json.loads(POSE.read_text())
    joint = json.loads(JOINT.read_text())
    require(pose["status"] == "unadopted_bilateral_geometric_pose_candidate"
            and joint["status"] == "blocked_source_joint_admission"
            and joint["open_knee_pose_receipt_sha256"] == sha(POSE.read_bytes()),
            "candidate source or joint-admission receipt changed")
    actual_source = {name: sha((SOURCE / name).read_bytes())
                     for name in EXPECTED_HASHES}
    require(actual_source == pose["source_files_sha256"],
            "pinned Open Knee source files changed")
    source_ptc_tets, source_fmc_tets = source_tetrahedra(source)
    producer_sha = sha(Path(__file__).read_bytes())
    sides = {}
    for side, stem in (("left", "open-knee-oks003-left"),
                       ("right", "open-knee-oks003-right-mirrored")):
        folder = ROOT / "Build/patellofemoral-surface-20260930" / side
        payload_path = folder / f"{stem}.nhknee"
        manifest_path = folder / f"{stem}.manifest.json"
        decoded = payload(payload_path, json.loads(manifest_path.read_text()), source)
        require(decoded["side"] == side and
                pose["sides"][side]["payload_sha256"] == sha(payload_path.read_bytes()) and
                pose["sides"][side]["manifest_sha256"] == sha(manifest_path.read_bytes()),
                f"{side} compiled knee identity changed")
        ptc, ptc_tets = region(decoded, "PTC")
        fmc, fmc_tets = region(decoded, "FMC")
        expected_ptc = (source_ptc_tets if side == "left" else
                        source_ptc_tets[:, [1, 0, 2, 3]])
        expected_fmc = (source_fmc_tets if side == "left" else
                        source_fmc_tets[:, [1, 0, 2, 3]])
        require(np.array_equal(ptc_tets, expected_ptc) and
                np.array_equal(fmc_tets, expected_fmc),
                f"{side} compiled tetrahedron order differs from source")
        displacement = np.asarray(
            pose["sides"][side]["prescribed_pose_translation_m"], dtype=float)
        require(abs(float(np.linalg.norm(displacement)) - 20e-6) < 1e-12 and
                joint["sides"][side]["whole_body_pose_admitted"] is False,
                f"{side} joint or clearance prerequisite changed")
        current, field = clearance_field(ptc, source, displacement)
        ptc_region = decoded["regions"]["PTC"]
        ptb_region = decoded["regions"]["PTB"]
        ptb = decoded["positions"][ptb_region["first_node"]:
                                   ptb_region["first_node"] + ptb_region["node_count"]]
        before_gap = tie_gap({"PTC": ptc, "PTB": ptb}, source,
                             "PTC_@_PTB_TiesNodes", "PTB_@_PTC_TiesNodes")
        after_gap = tie_gap({"PTC": current, "PTB": ptb}, source,
                            "PTC_@_PTB_TiesNodes", "PTB_@_PTC_TiesNodes")
        gap_change = max(float(np.max(np.abs(after_gap[key] - before_gap[key])))
                         for key in before_gap)
        require(gap_change == 0.0, f"{side} cartilage/bone tie gap changed")
        jacobian = signed_six(current.astype(float), ptc_tets) / \
            signed_six(ptc.astype(float), ptc_tets)
        require(np.all(np.isfinite(jacobian)) and np.all(jacobian > 0),
                f"{side} PTC tetrahedron inverted")
        meshes = _compiled_meshes(decoded)
        contact = _intersections({**meshes,
                                  "PTC": (current.astype(float), meshes["PTC"][1])})
        require(contact["exact_segment_or_polygon_crossing_pairs"] == 0 and
                contact["exact_point_contact_pairs"] == 0,
                f"{side} articular surfaces still intersect")
        ptc_all = full_boundary(decoded, "PTC")
        fmc_all = full_boundary(decoded, "FMC")
        _, exact = exact_integer_meshes({"PTC": (current.astype(float), ptc_all[1]),
                                        "FMC": fmc_all})
        full_boundary_contact = count_intersections(exact)
        require(full_boundary_contact["segment_or_polygon_crossing_pairs"] == 0
                and full_boundary_contact["point_contact_pairs"] == 0,
                f"{side} full boundaries still intersect")
        embedded = surface_embedding(source, "PTC", current.astype(float))
        solid = audit_state(np.concatenate((current, fmc)).astype("<f4").tobytes(),
                            ptc_tets, fmc_tets)
        require(solid["solid_domains_disjoint"],
                f"{side} cartilage solid domains still intersect")
        path = BUILD / f"open-knee-{side}-ptc-fmc-tie-preserving.nhcar"
        native = make_native_input(path, side, payload_path, pose,
                                   ptc, current, ptc_tets, fmc, fmc_tets,
                                   source_indices(source, "PTC_@_PTB_TiesNodes"),
                                   producer_sha)
        sides[side] = {
            "compiled_payload_sha256": sha(payload_path.read_bytes()),
            "compiled_manifest_sha256": sha(manifest_path.read_bytes()),
            "clearance_translation_m": displacement.tolist(),
            "field": field,
            "ptc_ptb_tie_gap_change_m": gap_change,
            "ptc_tetrahedra": len(ptc_tets),
            "minimum_deformation_jacobian": float(jacobian.min()),
            "maximum_deformation_jacobian": float(jacobian.max()),
            "candidate_ptc_surface": embedded,
            "articular_surface_intersections": contact,
            "full_boundary_intersections": full_boundary_contact,
            "exact_solid_separation": solid,
            "diagnostic_native_input": native,
        }
    result = {
        "schema": "numi.human.patellar-cartilage-tie-clearance-candidate.v1",
        "status": "bilateral_tie_fixed_cartilage_geometry_candidate",
        "producer_sha256": producer_sha,
        "source_file_sha256": actual_source,
        "open_knee_pose_receipt_sha256": sha(POSE.read_bytes()),
        "whole_body_joint_admission_receipt_sha256": sha(JOINT.read_bytes()),
        "sides": sides,
        "rigid_patella_joint_updated": False,
        "patellar_bone_moved": False,
        "qat_or_ptl_moved": False,
        "clinical_anatomy_qualified": False,
        "native_loaded_contact_qualified": False,
        "boundary": (
            "Bilateral compiled Open Knee geometry/current-position candidate only. "
            "The exact PTC/PTB bone tie, patellar bone, QAT/PTL and MyoSim joint "
            "law remain fixed. A 20 um contact-surface field is tapered through "
            "PTC thickness; its source reference mesh is unchanged. This introduces "
            "unqualified initial cartilage strain and does not establish stress, "
            "contact force, energy closure, calibrated anatomy or whole-body motion. "
            "The diagnostic NHCAR inputs include PTC/FMC only."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    receipt = run()
    print(json.dumps({"status": receipt["status"],
                      "sides": {side: {
                          "solid_domains_disjoint": row["exact_solid_separation"]["solid_domains_disjoint"],
                          "minimum_deformation_jacobian": row["minimum_deformation_jacobian"],
                      } for side, row in receipt["sides"].items()}}, sort_keys=True))
