"""Run the bounded native PTC/FMC tie-fixed candidate and source controls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from numilab_human.open_knee import parse_source
from tools.audit_patellofemoral_tetrahedral_separation import (
    audit_state, self_check, source_tetrahedra,
)
from tools.export_patellofemoral_matter_input import HEADER


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "Docs/media/patellar-cartilage-tie-clearance-20260930/receipt.json"
OUTPUT = ROOT / "Docs/media/patellar-cartilage-tie-native-20260930/receipt.json"
BUILD = ROOT / "Build/patellar-cartilage-tie-clearance-20260930"
LAB = Path("/Users/home/numi-lab-cardiac-native-20260930")
BINARY = LAB / "Build/cardiac-active-native/numi-matter-patellofemoral-full-surface-step"
METALLIB = LAB / "Build/cardiac-active-native/shaders/NumiMatter.metallib"
NATIVE_SOURCE = LAB / "matter/tools/patellofemoral_full_surface_step.mm"
CASES = (
    ("left", "on"), ("left", "on-replay"),
    ("left", "off"), ("left", "source-baseline"),
    ("right", "on"), ("right", "off"),
    ("right", "source-baseline"),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("patellar cartilage tie native: " + message)


def verify_reaction_stream(path: Path, input_bytes: bytes,
                           fixed: np.ndarray, native: dict) -> dict:
    raw = path.read_bytes()
    require(len(raw) == native["accepted_reaction_stream_bytes"] ==
            50991 * 4 * 4, "accepted FEM reaction stream length changed")
    reactions = np.frombuffer(raw, dtype="<f4").reshape(50991, 4)
    require(np.isfinite(reactions).all() and np.all(reactions[:, 3] == 0),
            "accepted FEM reaction stream contains invalid components")
    owned = np.zeros(50991, dtype=bool)
    owned[fixed] = True
    require(np.all(reactions[~owned, :3] == 0),
            "reaction was published outside the authored bone tie")
    start = HEADER.size + 12 * 26121
    positions = np.frombuffer(input_bytes, dtype="<f4", count=3 * 26121,
                              offset=start).reshape(-1, 3).astype(float)
    force = reactions[fixed, :3].astype(float)
    resultant = force.sum(axis=0)
    centroid = positions[fixed].mean(axis=0)
    moment = np.cross(positions[fixed] - centroid, force).sum(axis=0)
    scalars = {
        "fixed_tie_reaction_l1_n": float(np.linalg.norm(force, axis=1).sum()),
        "fixed_tie_reaction_resultant_n": float(np.linalg.norm(resultant)),
        "fixed_tie_reaction_max_node_n": float(np.linalg.norm(force, axis=1).max()),
        "fixed_tie_moment_magnitude_nm": float(np.linalg.norm(moment)),
    }
    vectors = {
        "fixed_tie_reaction_xyz_n": resultant,
        "fixed_tie_centroid_m": centroid,
        "fixed_tie_moment_about_centroid_nm": moment,
    }
    for name, value in scalars.items():
        require(abs(value - native[name]) < 1e-6,
                "native aggregate differs from reaction stream: " + name)
    for name, value in vectors.items():
        require(np.allclose(value, native[name], rtol=0, atol=1e-6),
                "native vector differs from reaction stream: " + name)
    require(float(np.linalg.norm(resultant)) > 0 and
            np.isfinite(native["momentum_reaction_residual_n"]) and
            0 <= native["momentum_reaction_residual_n"] <
            native["fixed_tie_reaction_resultant_n"],
            "fixed tie has no finite, signed reaction or momentum diagnostic")
    return {
        "reaction_stream_sha256": sha(raw),
        "reaction_stream_bytes": len(raw),
        "bone_tie_reaction_aggregate_independently_verified": True,
        "non_tie_reaction_nodes": 0,
        "momentum_reaction_residual_fraction_of_resultant":
            native["momentum_reaction_residual_n"] /
            native["fixed_tie_reaction_resultant_n"],
    }


def run() -> dict:
    self_check()
    geometry = json.loads(GEOMETRY.read_text())
    require(geometry["status"] == "bilateral_tie_fixed_cartilage_geometry_candidate",
            "geometry candidate receipt changed")
    source = parse_source(ROOT / "Sources/open-knee-oks003")
    ptc_tets, fmc_tets = source_tetrahedra(source)
    require(BINARY.is_file() and METALLIB.is_file() and NATIVE_SOURCE.is_file(),
            "native owner or binary is unavailable")
    cases = {}
    for side, mode in CASES:
        input_row = geometry["sides"][side]["diagnostic_native_input"]
        source_input = ROOT / input_row["path"]
        fixed_path = ROOT / input_row["fixed_node_indices_path"]
        require(sha(source_input.read_bytes()) == input_row["sha256"] and
                sha(fixed_path.read_bytes()) == input_row["fixed_node_indices_sha256"],
                f"{side} native input or tie set changed")
        fixed = np.fromfile(fixed_path, dtype="<u4")
        require(len(fixed) == input_row["fixed_node_count"] == 4592 and
                np.array_equal(fixed, np.unique(fixed)),
                f"{side} fixed-node set changed")
        stem = side + "-fixed-" + mode
        accepted_path = BUILD / (stem + ".f32le")
        argv = [str(BINARY), str(source_input), str(accepted_path),
                "--ptc-fixed-nodes", str(fixed_path)]
        if mode == "off":
            argv.append("--disable-contact")
        if mode == "source-baseline":
            argv.append("--baseline")
        completed = subprocess.run(argv, cwd=LAB, capture_output=True,
                                   text=True, timeout=180, check=False,
                                   env={**os.environ, "MTL_CAPTURE_ENABLED": "0"})
        require(completed.returncode == 0, f"{stem} tool failed: {completed.stderr}")
        result = json.loads(completed.stdout.strip())
        require(result["device"].startswith("Apple ") and
                result["source_nodes"] == 50991 and
                result["source_tetrahedra"] == 208177 and
                result["ptc_fixed_node_count"] == 4592 and
                result["ptc_fixed_nodes_moved"] == 0 and
                result["maximum_fixed_node_movement_m"] == 0.0,
                f"{stem} native source or fixed tie was not exercised")
        accepted = accepted_path.read_bytes()
        require(len(accepted) == 50991 * 3 * 4,
                f"{stem} accepted-position stream is incomplete")
        row = {
            "native": result,
            "accepted_positions_sha256": sha(accepted),
            "accepted_positions_bytes": len(accepted),
        }
        if mode in ("on", "off", "on-replay"):
            require(result["status_code"] == 0 and
                    result["completed_microsteps"] == 1,
                    f"{stem} candidate step was rejected")
            row.update(verify_reaction_stream(
                Path(str(accepted_path) + ".reactions.f32le"),
                source_input.read_bytes(), fixed, result))
            if mode == "on":
                body = source_input.read_bytes()
                start = HEADER.size + 12 * 26121
                before = np.frombuffer(body, dtype="<f4", count=3 * 26121,
                                       offset=start).reshape(-1, 3)
                after = np.frombuffer(accepted, dtype="<f4").reshape(-1, 3)[:26121]
                require(np.array_equal(before[fixed], after[fixed]),
                        f"{stem} PTC/PTB tie moved after acceptance")
                side_ptc = (ptc_tets if side == "left" else
                            ptc_tets[:, [1, 0, 2, 3]])
                side_fmc = (fmc_tets if side == "left" else
                            fmc_tets[:, [1, 0, 2, 3]])
                solid = audit_state(accepted, side_ptc, side_fmc)
                require(solid["solid_domains_disjoint"],
                        f"{stem} accepted cartilage solids overlap")
                row["exact_accepted_solid_separation"] = solid
                row["accepted_ptc_ptb_tie_unchanged"] = True
        else:
            require(result["status_code"] == 6 and
                    result["completed_microsteps"] == 0 and
                    result["rollback_bitwise"] and
                    result["accepted_reaction_stream_bytes"] == 0 and
                    not Path(str(accepted_path) + ".reactions.f32le").exists(),
                    f"{stem} source crossing was not rejected and rolled back")
        cases[stem] = row
    require(cases["left-fixed-on"]["native"] ==
            cases["left-fixed-on-replay"]["native"] and
            cases["left-fixed-on"]["accepted_positions_sha256"] ==
            cases["left-fixed-on-replay"]["accepted_positions_sha256"],
            "left accepted step does not replay exactly")
    require(cases["left-fixed-on"]["reaction_stream_sha256"] ==
            cases["left-fixed-on-replay"]["reaction_stream_sha256"],
            "left bone-tie reaction does not replay exactly")
    for side in ("left", "right"):
        require(cases[side + "-fixed-on"]["accepted_positions_sha256"] ==
                cases[side + "-fixed-off"]["accepted_positions_sha256"] and
                cases[side + "-fixed-on"]["reaction_stream_sha256"] ==
                cases[side + "-fixed-off"]["reaction_stream_sha256"] and
                cases[side + "-fixed-on"]["native"]["active_deformable_histories"] == 0,
                f"{side} contact-on/off diagnostic unexpectedly differs")
    left = cases["left-fixed-on"]["native"]
    right = cases["right-fixed-on"]["native"]
    mirrored_force = np.asarray(left["fixed_tie_reaction_xyz_n"]) * \
        np.asarray((-1.0, 1.0, 1.0))
    mirrored_moment = np.asarray(left["fixed_tie_moment_about_centroid_nm"]) * \
        np.asarray((1.0, -1.0, -1.0))
    force_mirror_error = float(np.linalg.norm(
        mirrored_force - right["fixed_tie_reaction_xyz_n"]))
    moment_mirror_error = float(np.linalg.norm(
        mirrored_moment - right["fixed_tie_moment_about_centroid_nm"]))
    require(force_mirror_error < 0.005 * left["fixed_tie_reaction_resultant_n"]
            and moment_mirror_error <
            0.005 * left["fixed_tie_moment_magnitude_nm"],
            "bilateral source-mirrored bone-tie reaction changed direction")
    unfixed_controls = {}
    for side in ("left", "right"):
        input_row = geometry["sides"][side]["diagnostic_native_input"]
        source_input = ROOT / input_row["path"]
        fixed = np.fromfile(ROOT / input_row["fixed_node_indices_path"],
                            dtype="<u4")
        accepted_path = BUILD / f"{side}-unfixed-control.f32le"
        completed = subprocess.run(
            [str(BINARY), str(source_input), str(accepted_path)], cwd=LAB,
            capture_output=True, text=True, timeout=180, check=False)
        require(completed.returncode == 0,
                f"{side} unfixed control failed: {completed.stderr}")
        native = json.loads(completed.stdout.strip())
        require(native["status_code"] == 0 and
                native["completed_microsteps"] == 1 and
                native["ptc_fixed_node_count"] == 0,
                f"{side} unfixed control did not accept")
        reaction_path = Path(str(accepted_path) + ".reactions.f32le")
        reaction_bytes = reaction_path.read_bytes()
        require(len(reaction_bytes) == 50991 * 4 * 4 and
                np.all(np.frombuffer(reaction_bytes, dtype="<f4") == 0),
                f"{side} unfixed control has a nonzero constraint reaction")
        body = source_input.read_bytes()
        start = HEADER.size + 12 * 26121
        before = np.frombuffer(body, dtype="<f4", count=3 * 26121,
                               offset=start).reshape(-1, 3)
        accepted = accepted_path.read_bytes()
        after = np.frombuffer(accepted, dtype="<f4").reshape(-1, 3)[:26121]
        movement = np.linalg.norm(after[fixed].astype(float) -
                                  before[fixed].astype(float), axis=1)
        require(np.count_nonzero(movement) == 4592,
                f"{side} unfixed control unexpectedly retained a bone tie")
        unfixed_controls[side] = {
            "native": native,
            "accepted_positions_sha256": sha(accepted),
            "reaction_stream_sha256": sha(reaction_bytes),
            "ptc_ptb_tie_nodes_moved": int(np.count_nonzero(movement)),
            "maximum_ptc_ptb_tie_movement_m": float(movement.max()),
            "median_ptc_ptb_tie_movement_m": float(np.median(movement)),
        }
    result = {
        "schema": "numi.human.patellar-cartilage-tie-native-diagnostic.v2",
        "status": "bilateral_tie_fixed_cartilage_reaction_measured",
        "geometry_receipt_sha256": sha(GEOMETRY.read_bytes()),
        "lab_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LAB, text=True).strip(),
        "native_source_sha256": sha(NATIVE_SOURCE.read_bytes()),
        "binary_sha256": sha(BINARY.read_bytes()),
        "metallib_sha256": sha(METALLIB.read_bytes()),
        "cases": cases,
        "unfixed_tie_controls": unfixed_controls,
        "left_replay_bitwise": True,
        "source_crossings_rejected_bilaterally": True,
        "bone_tie_reaction_field_verified_bilaterally": True,
        "bilateral_reaction_mirror_force_error_n": force_mirror_error,
        "bilateral_reaction_mirror_moment_error_nm": moment_mirror_error,
        "whole_body_joint_and_bone_reaction_coupled": False,
        "loaded_contact_qualified": False,
        "clinical_anatomy_qualified": False,
        "boundary": (
            "Apple M4 one-microsecond PTC/FMC-only diagnostic using synthetic "
            "material and static fixed PTC/PTB tie nodes. Both candidate states "
            "accept and remain exact-volume-disjoint; source crossing baselines "
            "reject and roll back. The fixed-node reaction field is captured "
            "and independently reduced to a wrench, but not applied to a rigid "
            "patella, include QAT/PTL, qualify pressure or energy closure, or "
            "adopt a whole-body anatomical state."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"],
                      "cases": {name: row["native"]["status_code"]
                                for name, row in result["cases"].items()}},
                     sort_keys=True))
