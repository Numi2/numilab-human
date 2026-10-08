#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path("/Users/n/numi-human-eye-skin-registration-004")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from numilab_human import model as human
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.skin_lower_limb_anchor_rebind import (
    _owner_names_by_core_index,
    _transform_points,
)
OPTIMIZER_SOURCE = Path("/Users/n/numi-human-resting-evidence-20261005/ocular-skin-eye-registration-897-movable-ring-sample-constrained-preflight/constrained_optimizer.py")
OPTIMIZER_SPEC = importlib.util.spec_from_file_location(
    "numilab_human.ocular_skin_registration_constrained_v897", OPTIMIZER_SOURCE,
)
optimizer = importlib.util.module_from_spec(OPTIMIZER_SPEC)
sys.modules[OPTIMIZER_SPEC.name] = optimizer
assert OPTIMIZER_SPEC.loader is not None
OPTIMIZER_SPEC.loader.exec_module(optimizer)
build_local_problem = optimizer.build_local_problem
solve_local_problem = optimizer.solve_local_problem
ray_outer_radius_mm = optimizer.ray_outer_radius_mm


def solve_local_problem(problem, gradient_const, *, maxiter=400):
    """Run the immutable v897 trust-constr problem and retain a feasible cap iterate for geometry audit.

    Objective, Hessian, Jacobian, constraints, tolerances, and iteration cap match
    the pinned v897 implementation. A capped iterate is returned only for geometry
    diagnostics when SciPy reports feasibility and the oriented-area floor is met.
    Its unsuccessful optimizer status is retained verbatim.
    """
    H = problem.H
    g = np.asarray(gradient_const, dtype=np.float64)

    def fun(x):
        return 0.5 * float(x @ (H @ x)) + float(g @ x)

    def jac(x):
        return H @ x + g

    constraints = [problem.nonlinear_area_constraint]
    if problem.linear_constraints is not None:
        constraints.append(problem.linear_constraints)
    result = minimize(
        fun, problem.initial_x, method="trust-constr", jac=jac, hess=lambda _x: H,
        constraints=constraints,
        options={
            "maxiter": int(maxiter), "gtol": 1.0e-8, "xtol": 1.0e-10,
            "barrier_tol": 1.0e-10, "sparse_jacobian": True, "verbose": 0,
        },
    )
    disp = {
        int(v): np.asarray(problem.fixed_delta_mm.get(int(v), np.zeros(3)), dtype=np.float64)
        for v in problem.patch_vertices
    }
    for vertex, index in problem.free_index.items():
        disp[int(vertex)] = np.asarray(result.x[3*index:3*index+3], dtype=np.float64)
    full_delta = np.zeros_like(problem.base_mm, dtype=np.float64)
    for vertex, value in disp.items():
        full_delta[vertex] = value / 1000.0
    actual_area = np.asarray(problem.nonlinear_area_constraint.fun(result.x), dtype=np.float64)
    node_energy = 0.5 * sum(float(np.dot(value, value)) for value in disp.values())
    edge_energy = 0.0
    for (a, b), weight in zip(problem.edge_pairs, problem.edge_weights):
        da = disp.get(int(a), np.zeros(3, dtype=np.float64))
        db = disp.get(int(b), np.zeros(3, dtype=np.float64))
        diff = da - db
        edge_energy += 0.5 * float(weight) * float(np.dot(diff, diff))
    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.niter),
        "objective_value_mm2": float(node_energy + edge_energy),
        "reduced_free_variable_objective_mm2": float(result.fun),
        "objective_node_term_mm2": float(node_energy),
        "objective_smoothness_term_mm2": float(edge_energy),
        "optimality": float(result.optimality),
        "constraint_violation": float(result.constr_violation),
        "lagrangian_gradient_inf_norm": float(np.max(np.abs(np.asarray(getattr(result, "lagrangian_grad", jac(result.x)))))),
        "minimum_oriented_area_ratio": float(actual_area.min(initial=np.inf)),
        "area_floor_ratio": float(problem.plan["area_floor_ratio"]),
        "radial_clearance_inequality_count": len(problem.radial_rows),
        "full_delta_max_mm": float(max((np.linalg.norm(v) for v in disp.values()), default=0.0)),
    }
    feasible_iteration = (
        np.isfinite(result.x).all()
        and report["constraint_violation"] <= 1.0e-6
        and report["minimum_oriented_area_ratio"] >= problem.plan["area_floor_ratio"] - 1.0e-6
    )
    report["optimizer_converged"] = bool(result.success)
    report["feasible_iteration_captured_for_geometry_audit"] = bool(feasible_iteration and not result.success)
    if (not feasible_iteration
            or (result.success and report["constraint_violation"] > 1.0e-6)
            or report["minimum_oriented_area_ratio"] < problem.plan["area_floor_ratio"] - 1.0e-6):
        raise RuntimeError("no feasible iterate available for exact geometry audit: " + str(report))
    return full_delta, report

E = Path("/Users/n/numi-human-resting-evidence-20261005")
OUT = E / "ocular-skin-eye-registration-906-feasible-iterate-geometry-audit"
PLAN_OUT = E / "ocular-skin-eye-registration-906-feasible-iterate-geometry-audit-preflight"
OLD_CANDIDATE = E / "ocular-skin-eye-registration-892/bodyparts3d-myosim-skinned-shell.eye-registration-candidate.nhskin"
OLD_DIAG = E / "ocular-skin-eye-registration-892/diagnosis-v1/diagnosis.json"
RESIDUAL_DIAG = E / "ocular-skin-eye-registration-893-constrained/diagnosis-v1/residual-pairs.json"
PACK = E / "native-common-atlas-skin-cycle-884/accepted-geometry/step-0.mrvpack"
RECEIPT = PACK.with_suffix(".receipt.json")
BASE_SKIN = E / "common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_MANIFEST = BASE_SKIN.with_name("common-atlas-skin-geometry-registration.manifest.json")
BONE_DIR = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad")
BONE = BONE_DIR / "bodyparts3d-myosim-major-bones.nhbones"
BONE_MANIFEST = BONE_DIR / "bodyparts3d-myosim-major-bones.manifest.json"
REGISTRATION = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json")
MYOSIM_ARTIFACT = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004")
ENVELOPE = E / "ocular-registration-diagnosis-891/ocular-sclera-radial-shell-audit-v10/skin-ring-eye-envelope-v16.json"
LOOP_ORDER = E / "ocular-registration-diagnosis-891/ocular-sclera-radial-shell-audit-v10/skin-ring-eye-envelope-loop-order-v17.json"
HELPER_SOURCE = Path("/Users/n/numi-human-resting-final-integration-001/src/numilab_human/common_atlas_skin_clearance.py")
RECONSTRUCTION = E / "common-atlas-skin-affine-reconstruction-886/reconstruction.json"
REPLAY_PLAN = E / "ocular-skin-eye-registration-897-movable-ring-sample-constrained-preflight/optimization-plan.json"
REPLAY_SCRIPT = E / "ocular-skin-eye-registration-897-movable-ring-sample-constrained-preflight/derive_candidate.py"
REPLAY_TESTS = (
    E / "ocular-skin-eye-registration-897-movable-ring-sample-constrained-preflight/focused_tests.py",
    E / "ocular-skin-eye-registration-897-movable-ring-sample-constrained-preflight/focused_constrained_tests.py",
)
REPLAY_OUTPUT = E / "ocular-skin-eye-registration-897-movable-ring-sample-constrained"
REPLAY_CANDIDATE = REPLAY_OUTPUT / "bodyparts3d-myosim-skinned-shell.eye-registration-candidate.nhskin"
REPLAY_REPORT = REPLAY_OUTPUT / "report.json"
FAILED_898 = E / "ocular-skin-eye-registration-898-candidate.log"
FAILED_899 = E / "ocular-skin-eye-registration-899-candidate.log"
FAILED_900 = E / "ocular-skin-eye-registration-900-linearized-current-ray-refinement.log"
FAILED_900_PASSES = E / "ocular-skin-eye-registration-900-linearized-current-ray-refinement/refinement-passes.json"
FAILED_904_LOG = E / "ocular-skin-eye-registration-904-directional-guardband.log"
FAILED_904_PLAN = E / "ocular-skin-eye-registration-904-directional-guardband-preflight/optimization-plan.json"
FAILED_905_LOG = E / "ocular-skin-eye-registration-905-feasible-iterate-geometry-audit.log"
FAILED_905_PLAN = E / "ocular-skin-eye-registration-905-feasible-iterate-geometry-audit-preflight/optimization-plan.json"
FAILED_PREFLIGHTS = (
    E / "ocular-skin-eye-registration-898-updated-ray-margin-preflight/optimization-plan.json",
    E / "ocular-skin-eye-registration-899-warm-started-updated-ray-margin-preflight/optimization-plan.json",
    E / "ocular-skin-eye-registration-900-linearized-current-ray-refinement-preflight/optimization-plan.json",
)
RIGID_REFERENCE = MYOSIM_ARTIFACT / "myosim-fullbody-core-reference.nhrigid"
CLEARANCE_SPEC = importlib.util.spec_from_file_location(
    "numilab_human.common_atlas_skin_clearance", HELPER_SOURCE,
)
clearance = importlib.util.module_from_spec(CLEARANCE_SPEC)
sys.modules[CLEARANCE_SPEC.name] = clearance
assert CLEARANCE_SPEC.loader is not None
CLEARANCE_SPEC.loader.exec_module(clearance)

MARGIN_MM = 0.255
FINAL_ACTUAL_CLEARANCE_MM = 0.250
SUPPORT_EDGE_MULTIPLE = 4.0
INTERPOLATION_NEIGHBORS = 8
EYE_KEYS = {(51010, stable_id) for stable_id in range(381, 398)}
SKIN_KEY = (51007, 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def boundary_components(faces: np.ndarray) -> list[list[int]]:
    edge_count: dict[tuple[int, int], int] = {}
    for tri in faces:
        for a, b in ((int(tri[0]), int(tri[1])),
                     (int(tri[1]), int(tri[2])),
                     (int(tri[2]), int(tri[0]))):
            edge = (a, b) if a < b else (b, a)
            edge_count[edge] = edge_count.get(edge, 0) + 1
    adjacency: dict[int, set[int]] = {}
    for (a, b), count in edge_count.items():
        if count == 1:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        elif count < 1 or count > 2:
            raise RuntimeError("skin source contains a nonmanifold edge")
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("skin aperture boundary is not a collection of degree-2 loops")
    components, seen = [], set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        seen.add(start)
        todo, component = [start], []
        while todo:
            current = todo.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        components.append(sorted(component))
    return components


def audit_surface_pair(skin_records, target_faces, pack_positions):
    per_surface = {}
    for key in sorted(target_faces):
        faces = target_faces[key]
        unique = np.unique(faces)
        compact_faces = np.searchsorted(unique, faces)
        target_records = clearance._exact_surface_records(pack_positions[unique], compact_faces)
        audit = clearance._audit_pair(skin_records, target_records, same_surface=False)
        per_surface[key] = {
            "count": int(audit["count"]),
            "aabb_candidate_pairs": int(audit["aabb_candidate_pairs"]),
            "triangle_pairs": audit["triangle_pairs"],
        }
    return per_surface


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    OUT = PLAN_OUT if args.plan_only else E / "ocular-skin-eye-registration-906-feasible-iterate-geometry-audit"
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence directory: {OUT}")
    for path in (PACK, RECEIPT, BASE_SKIN, SKIN_MANIFEST, BONE, BONE_MANIFEST,
                 REGISTRATION, HELPER_SOURCE, RECONSTRUCTION, RIGID_REFERENCE, ENVELOPE, LOOP_ORDER,
                 OLD_CANDIDATE, OLD_DIAG, RESIDUAL_DIAG):
        if not path.is_file():
            raise RuntimeError(f"missing pinned input: {path}")
    helper_sha_before = sha(HELPER_SOURCE)
    helper_commit_before = subprocess.check_output(
        ["git", "-C", str(HELPER_SOURCE.parents[2]), "rev-parse", "HEAD"], text=True,
    ).strip()
    if helper_commit_before != "935cc7d332b9118f41c719e2d2f9c7c568a3d41f":
        raise RuntimeError(f"unexpected 935 helper checkout revision: {helper_commit_before}")
    raw = BASE_SKIN.read_bytes()
    decoded = decode_payload(raw)
    source_positions = decoded["vertices_f"][:, :3].astype(np.float64)
    faces = decoded["indices"].reshape(-1, 3).astype(np.int64)
    weights = decoded["full_weights"].astype(np.float64)
    bindings = decoded["bindings_f"].astype(np.float64)
    owner_ids = decoded["bindings_u"][:, 0].astype(int)
    referenced = np.unique(faces)
    compact_remap = np.full(decoded["vertex_count"], -1, dtype=np.int64)
    compact_remap[referenced] = np.arange(len(referenced), dtype=np.int64)
    compact_faces = compact_remap[faces]

    pack_positions, surfaces, pack_counts = clearance._pack_surfaces(PACK, EYE_KEYS)
    receipt = json.loads(RECEIPT.read_text())
    if (receipt.get("accepted_step") != 0 or receipt.get("accepted_pack_path") != str(PACK)
            or receipt.get("physical_endpoint") != "accepted"):
        raise RuntimeError("input capture receipt does not bind the requested accepted step 0")
    skin_faces_global = surfaces[SKIN_KEY]["faces"]
    skin_base = int(skin_faces_global.min())
    if (not np.array_equal(skin_faces_global - skin_base, faces)
            or int(skin_faces_global.max()) >= skin_base + decoded["vertex_count"]):
        raise RuntimeError("accepted skin topology/order differs from canonical NHSKIN input")
    captured = pack_positions[skin_base:skin_base + decoded["vertex_count"]].astype(np.float64)
    if captured.shape != source_positions.shape:
        raise RuntimeError("accepted skin vertex buffer differs from NHSKIN vertex count")

    poses, stable_to_body, pose_fit, bone_info = clearance._fit_registered_bone_poses(
        pack_positions, surfaces, BONE, BONE_MANIFEST,
    )
    if set(map(int, owner_ids)) != set(poses):
        raise RuntimeError("NHSKIN binding owner set differs from the 86 exact captured bone poses")
    predicted = np.zeros_like(source_positions)
    jacobian = np.zeros((decoded["vertex_count"], 3, 3), dtype=np.float64)
    for binding_index, body_id in enumerate(owner_ids):
        body_rotation, body_translation = poses[int(body_id)]
        binding = bindings[binding_index]
        binding_rotation = clearance._rotation(binding[4:8])
        scale = float(binding[8])
        local = binding[1:4] + scale * (source_positions @ binding_rotation.T)
        predicted += weights[:, binding_index, None] * (local @ body_rotation.T + body_translation)
        jacobian += weights[:, binding_index, None, None] * (body_rotation @ binding_rotation * scale)
    base_error = np.linalg.norm(predicted[referenced] - captured[referenced], axis=1)
    singular = np.linalg.svd(jacobian[referenced], compute_uv=False)
    condition = singular[:, 0] / singular[:, -1]
    if (not np.isfinite(condition).all() or float(condition.max()) > 2.0
            or float(singular[:, -1].min()) < 1.0e-8
            or float(base_error.max()) > 1.0e-6):
        raise RuntimeError("full-weight captured-pose replay is not sub-micrometer or Jacobian is ill-conditioned")

    envelope = json.loads(ENVELOPE.read_text())
    loop_doc = json.loads(LOOP_ORDER.read_text())
    if envelope.get("schema") != "numi.human.skin-ring-radial-eye-envelope.v1":
        raise RuntimeError("eye envelope source schema differs")
    if loop_doc.get("schema") != "numi.human.eye-aperture-loop-order-envelope.v1":
        raise RuntimeError("aperture loop source schema differs")
    boundary = boundary_components(faces)
    if sorted(map(len, boundary)) != [55, 60]:
        raise RuntimeError(f"canonical NHSKIN aperture loops changed: {sorted(map(len, boundary))}")
    boundary_sets = [set(component) for component in boundary]
    rings = []
    ring_source_rows = {}
    for side in ("right", "left"):
        loop = loop_doc["results"][side]
        ordered = list(map(int, loop["sequence_skin_vertices"]))
        if len(set(ordered)) != len(ordered) or not any(set(ordered) == item for item in boundary_sets):
            raise RuntimeError(f"{side} aperture cycle no longer matches canonical NHSKIN boundary")
        rows = {int(row["skin_source_vertex"]): row
                for row in envelope["results"][side]["vertices"]}
        if set(rows) != set(ordered):
            raise RuntimeError(f"{side} eye envelope does not exactly cover its boundary cycle")
        outside = set()
        inside = set()
        for run in loop["runs"]:
            (outside if run["side"] == "out" else inside).update(
                map(int, run["skin_vertices"]),
            )
        if inside & outside or inside | outside != set(ordered):
            raise RuntimeError(f"{side} loop inward/outside arc partition is incomplete")
        # Anchor counts/IDs must be from the source report, not selected after candidate output.
        if len(inside) != (44 if side == "right" else 40) or len(outside) != (16 if side == "right" else 15):
            raise RuntimeError(f"{side} predeclared aperture arc partition changed")
        center_mm = np.asarray(envelope["results"][side]["fitted_sclera_center_mm"], dtype=np.float64)
        target_distances = np.zeros(len(ordered), dtype=np.float64)
        for index, vertex_id in enumerate(ordered):
            row = rows[vertex_id]
            if row.get("status") == "no_ray_hit":
                raise RuntimeError(f"{side} eye envelope ray misses boundary vertex {vertex_id}")
            point_mm = captured[vertex_id] * 1000.0
            radius_mm = float(np.linalg.norm(point_mm - center_mm))
            if abs(radius_mm - float(row["radius_mm"])) > 1.0e-3:
                raise RuntimeError(f"{side} captured boundary vertex {vertex_id} differs from v16 position")
            clearance_mm = float(row["radial_clearance_mm"])
            if abs(clearance_mm - (radius_mm - float(row["outermost_eye_surface_radius_mm"]))) > 1.0e-3:
                raise RuntimeError(f"{side} v16 eye-envelope clearance arithmetic is inconsistent")
            if vertex_id in inside:
                if clearance_mm > 1.0e-6:
                    raise RuntimeError(f"{side} inward arc vertex {vertex_id} is not inside the pinned radial envelope")
                target_distances[index] = max(0.0, MARGIN_MM - clearance_mm) / 1000.0
            else:
                if clearance_mm <= 0.0:
                    raise RuntimeError(f"{side} outside end-arc anchor {vertex_id} is not outside the pinned radial envelope")
                target_distances[index] = 0.0
        rings.append({
            "name": side,
            "vertex_ids": ordered,
            "movable_vertex_ids": sorted(inside),
            "center_world_m": (center_mm / 1000.0).tolist(),
            "radial_displacement_m": target_distances.tolist(),
            "_captured_world_m": {str(v): captured[v].tolist() for v in ordered},
        })
        ring_source_rows[side] = {
            "inside_vertex_ids": sorted(inside),
            "outside_end_arc_anchor_ids": sorted(outside),
            "inward_vertex_count": len(inside),
            "preserved_outside_end_arc_anchor_count": len(outside),
            "pre_candidate_radial_clearance_mm": {
                "minimum": min(float(rows[v]["radial_clearance_mm"]) for v in inside),
                "median": float(np.median([rows[v]["radial_clearance_mm"] for v in inside])),
                "maximum": max(float(rows[v]["radial_clearance_mm"]) for v in inside),
            },
        }

    # Seed local supports from the exact baseline eye witnesses, every prior
    # candidate-introduced pair, and the exact face-orientation failures from
    # the retained 892 report. The previous candidate is diagnostic evidence,
    # not an anatomical target.
    prior_diag = json.loads(OLD_DIAG.read_text())
    baseline_for_seed = clearance._exact_surface_records(captured[referenced], compact_faces)
    baseline_seed_pairs = audit_surface_pair(
        baseline_for_seed, {key: surfaces[key]["faces"] for key in sorted(EYE_KEYS)}, pack_positions,
    )
    seed_faces_by_side = {"right": set(), "left": set()}
    for key, row in baseline_seed_pairs.items():
        side = "right" if key[1] <= 389 else "left"
        seed_faces_by_side[side].update(int(pair[0]) for pair in row["triangle_pairs"])
    for row in prior_diag["introduced_pairs"]:
        side = "right" if int(row["eye_stable_id"]) <= 389 else "left"
        seed_faces_by_side[side].add(int(row["skin_face_row"]))
    centers_mm = {
        side: np.asarray(envelope["results"][side]["fitted_sclera_center_mm"], dtype=np.float64)
        for side in ("right", "left")
    }
    for row in prior_diag["inversions"]:
        face_id = int(row["skin_face_row"])
        centroid = captured[faces[face_id]].mean(axis=0) * 1000.0
        side = min(("right", "left"), key=lambda value: float(np.linalg.norm(centroid - centers_mm[value])))
        seed_faces_by_side[side].add(face_id)
    shell_keys = {
        "right": ((51010, 383), (51010, 386)),
        "left": ((51010, 392), (51010, 395)),
    }
    shell_triangles_mm = {
        side: np.concatenate([pack_positions[surfaces[key]["faces"]] * 1000.0 for key in keys], axis=0)
        for side, keys in shell_keys.items()
    }
    residual_doc = json.loads(RESIDUAL_DIAG.read_text())
    sample_faces_by_side = {"right": set(), "left": set()}
    for row in residual_doc["rows"]:
        side = "right" if int(row["eye_stable_id"]) <= 389 else "left"
        sample_faces_by_side[side].add(int(row["skin_face_row"]))
    problem, objective_linear_term = build_local_problem(
        base_world_m=captured, faces=faces, seed_faces_by_side=seed_faces_by_side,
        rings=rings, sample_faces_by_side=sample_faces_by_side,
        centers_mm=centers_mm, outer_shell_triangles_mm=shell_triangles_mm,
        collar_hops=3, maximum_hops=12, smoothness_lambda=1.0,
        area_floor_ratio=0.05, margin_mm=MARGIN_MM,
    )
    preflight_input_paths = (
        PACK, RECEIPT, BASE_SKIN, SKIN_MANIFEST, BONE, BONE_MANIFEST, REGISTRATION,
        RECONSTRUCTION, RIGID_REFERENCE, ENVELOPE, LOOP_ORDER, HELPER_SOURCE,
        OPTIMIZER_SOURCE, REPLAY_PLAN, REPLAY_SCRIPT, REPLAY_CANDIDATE, REPLAY_REPORT,
        Path(__file__).resolve(), OLD_CANDIDATE, OLD_DIAG, RESIDUAL_DIAG,
        FAILED_898, FAILED_899, FAILED_900, FAILED_900_PASSES, FAILED_904_LOG, FAILED_904_PLAN, FAILED_905_LOG, FAILED_905_PLAN, *FAILED_PREFLIGHTS, *REPLAY_TESTS,
    )
    method_declaration = {
        "schema": "numi.human.ocular-skin-registration-constrained-preflight.v6",
        "algorithm": "Local sparse minimum-change trust-constr solve. Inward aperture vertices remain free under a +0.255 mm nominal directional radial shell-clearance target; the sparse minimum-change solve may move a vertex beyond its simple +0.255 mm projection where the face samples require it. Every already-outside end-arc anchor remains exactly zero. The patch seeds are all exact baseline pairs on all 17 eye surfaces, prior candidate-introduced exact pairs, and prior reversed faces; each side expands by mesh edge rings until a zero-motion outer collar has >=0.255 mm nominal radial shell clearance or no shell ray hit. Barycentric clearance inequalities are limited to the seven unique skin-face rows remaining as exact pairs in the v2 candidate, at three edge midpoints and centroid. These samples guide the solve only; exact integer-lattice pair checks remain final authority.",
        "objective": problem.plan["objective"],
        "area_floor_ratio": 0.05,
        "nominal_directional_offset_mm": MARGIN_MM,
        "final_actual_radial_clearance_gate_mm": FINAL_ACTUAL_CLEARANCE_MM,
        "guardband_basis": "The 0.255 mm directional target leaves a 5 micrometer engineering guardband over the measured 2.815 micrometer ray-linearization deficit plus sub-micrometer float32 packing uncertainty; neither margin is anatomical measurement.",
        "capped_iterate_policy": "No iteration cap or convergence tolerance is increased. A max-evaluation iterate is retained only for separate exact geometry diagnosis if feasibility and area floor pass; optimizer success remains false.",
        "optimization": "Run the immutable v897 sparse trust-constr problem with the same 400-iteration cap and gtol=1e-8, xtol=1e-10, barrier_tol=1e-10. If SciPy reaches the cap, retain its iterate only for geometry diagnostics when reported constraint violation <=1e-6 and oriented area floor >=5%; record success=false and KKT/optimality without calling it optimizer-converged. Objective and all hard geometry gates remain unchanged. Exact quadratic oriented-area constraints are re-evaluated, not only linearized. Constraints keep projected signed area >=5% of each baseline face area. Face-interior radial constraints use triangle edge-midpoint and centroid barycentric samples with fixed base radial directions.",
        "exact_final_gates": "binary32 integer-lattice skin self-pair audit, exact skin vs every captured ocular surface 381-397, zero-area/reversed-face rejection, area ratio >=5%, direct recomputed actual radial clearance >=0.250 mm for all 84 movable inward boundary vertices, exact boundary anchors, source topology/bindings/weights unchanged. The sparse directional solve target is 0.255 mm, an engineering guardband rather than a biological measurement.",
        "patch_plan": problem.plan,
        "seed_face_counts": {k: len(v) for k, v in seed_faces_by_side.items()},
        "seed_faces_sha256": sha(OLD_DIAG),
        "residual_sample_faces_sha256": sha(RESIDUAL_DIAG),
        "prior_candidate_sha256": sha(OLD_CANDIDATE),
        "source_inputs": {
            str(path): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in preflight_input_paths
        },
    }
    if args.plan_only:
        OUT.mkdir(parents=True)
        for source_path, snapshot_name in (
            (Path(__file__).resolve(), "derive_candidate.py"),
            (OPTIMIZER_SOURCE, "constrained_optimizer.py"),
            (REPLAY_TESTS[0], "immutable_897_focused_tests.py"),
            (REPLAY_TESTS[1], "immutable_897_focused_constrained_tests.py"),
        ):
            (OUT / snapshot_name).write_bytes(source_path.read_bytes())
        write_json(OUT / "optimization-plan.json", method_declaration)
        print(json.dumps({
            "plan_path": str(OUT / "optimization-plan.json"),
            "plan_sha256": sha(OUT / "optimization-plan.json"),
            "patch_vertex_count": problem.plan["patch_vertex_count"],
            "free_component_count": problem.plan["free_component_count"],
            "radial_clearance_inequality_count": problem.plan["radial_clearance_inequality_count"],
            "radial_face_sample_inequality_count": problem.plan["radial_face_sample_inequality_count"],
            "radial_sample_face_counts": problem.plan["radial_sample_face_counts"],
            "area_constraint_face_count": problem.plan["area_constraint_face_count"],
            "dilation_layer_vertex_counts": problem.plan["dilation_layer_vertex_counts"],
        }, sort_keys=True))
        return
    world_delta, optimizer_report = solve_local_problem(problem, objective_linear_term, maxiter=400)
    field_report = {
        "maximum_displacement_mm": float(np.linalg.norm(world_delta, axis=1).max(initial=0.0) * 1000.0),
        "local_problem": problem.plan,
        "optimizer": optimizer_report,
    }
    if world_delta.shape != captured.shape:
        raise RuntimeError("constrained eye registration field has the wrong shape")
    source_delta = np.zeros_like(source_positions)
    source_delta[referenced] = np.linalg.solve(
        jacobian[referenced], world_delta[referenced, :, None],
    )[:, :, 0]
    candidate_source = source_positions.astype("<f4").astype(np.float64)
    candidate_source[referenced] = (
        source_positions[referenced] + source_delta[referenced]
    ).astype("<f4").astype(np.float64)
    packed_source_delta = candidate_source - source_positions
    packed_world_delta = np.einsum("nij,nj->ni", jacobian, packed_source_delta)
    inverse_roundtrip = np.linalg.norm(packed_world_delta[referenced] - world_delta[referenced], axis=1)
    candidate_predicted = np.zeros_like(candidate_source)
    for binding_index, body_id in enumerate(owner_ids):
        body_rotation, body_translation = poses[int(body_id)]
        binding = bindings[binding_index]
        binding_rotation = clearance._rotation(binding[4:8])
        scale = float(binding[8])
        local = binding[1:4] + scale * (candidate_source @ binding_rotation.T)
        candidate_predicted += weights[:, binding_index, None] * (local @ body_rotation.T + body_translation)
    candidate_world = candidate_predicted.astype("<f4").astype(np.float64)
    expected_world = captured + packed_world_delta
    forward_error = np.linalg.norm(candidate_world[referenced] - expected_world[referenced], axis=1)
    if (not np.isfinite(candidate_world).all() or not np.isfinite(source_delta).all()
            or float(inverse_roundtrip.max()) > 2.0e-6
            or float(forward_error.max()) > 2.0e-6):
        raise RuntimeError("candidate full-weight inverse/forward map exceeds 2 micrometers")

    boundary_candidate_metrics = {}
    for side in ("right", "left"):
        center = centers_mm[side]
        env_rows = {int(row["skin_source_vertex"]): row for row in envelope["results"][side]["vertices"]}
        boundary_rows = []
        for vertex_id in rings[0 if side == "right" else 1]["vertex_ids"]:
            vertex_id = int(vertex_id)
            base_point = captured[vertex_id] * 1000.0
            candidate_point = candidate_world[vertex_id] * 1000.0
            radial = base_point - center
            radial /= np.linalg.norm(radial)
            actual_delta = candidate_point - base_point
            radial_displacement = float(np.dot(actual_delta, radial))
            prior_clearance = float(env_rows[vertex_id]["radial_clearance_mm"])
            prior_projection_target = max(0.0, MARGIN_MM - prior_clearance)
            outer = ray_outer_radius_mm(candidate_point, center, shell_triangles_mm[side])
            actual_clearance = float(np.linalg.norm(candidate_point - center) - outer) if outer is not None else None
            boundary_rows.append({
                "vertex_id": vertex_id,
                "was_inward_movable_vertex": vertex_id in set(rings[0 if side == "right" else 1]["movable_vertex_ids"]),
                "prior_simple_projection_target_mm": prior_projection_target,
                "actual_radial_displacement_mm": radial_displacement,
                "extra_radial_movement_beyond_prior_target_mm": radial_displacement - prior_projection_target,
                "candidate_radial_shell_clearance_mm": actual_clearance,
            })
        sample_clearances = []
        barycentric_samples = ((0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5),(1.0/3.0,1.0/3.0,1.0/3.0))
        for face_id in sorted(sample_faces_by_side[side]):
            for bary in barycentric_samples:
                point = np.asarray(bary, dtype=np.float64) @ (candidate_world[faces[face_id]] * 1000.0)
                outer = ray_outer_radius_mm(point, center, shell_triangles_mm[side])
                if outer is not None:
                    sample_clearances.append(float(np.linalg.norm(point - center) - outer))
        boundary_candidate_metrics[side] = {
            "vertices": boundary_rows,
            "minimum_candidate_boundary_vertex_radial_clearance_mm": min(
                row["candidate_radial_shell_clearance_mm"] for row in boundary_rows
                if row["candidate_radial_shell_clearance_mm"] is not None
            ),
            "minimum_candidate_clearance_over_four_samples_on_each_residual_face_mm": min(sample_clearances),
            "sample_face_rows": sorted(sample_faces_by_side[side]),
            "sample_count": len(sample_clearances),
        }

    # Generate ABI-5 visual normals from the existing source-owned default
    # runtime reference, matching the established lower-limb rebind owner.
    registration = human.read_json(REGISTRATION)
    runtime_reference, runtime_bodies = human._bodyparts_runtime_bindings(registration, MYOSIM_ARTIFACT)
    owner_names = _owner_names_by_core_index(registration)
    old_rest = np.zeros_like(source_positions)
    candidate_rest = np.zeros_like(candidate_source)
    for binding_index, body_id in enumerate(owner_ids):
        name = owner_names[int(body_id)]
        if name not in runtime_bodies:
            raise RuntimeError(f"reference runtime lacks skin binding owner {name}")
        _, runtime_body = runtime_bodies[name]
        old_rest += _transform_points(
            source_positions, bindings[binding_index], runtime_body,
            f"eye registration old rest owner {name}",
        ) * weights[:, binding_index, None]
        candidate_rest += _transform_points(
            candidate_source, bindings[binding_index], runtime_body,
            f"eye registration candidate rest owner {name}",
        ) * weights[:, binding_index, None]
    compact_face_rows = [tuple(int(value) for value in face) for face in compact_faces]
    old_normals = np.asarray(human._bodyparts_skin_smooth_visual_normals(
        human._bodyparts_vertex_normals(
            (old_rest[referenced] * 1000.0).tolist(), compact_face_rows,
            "source NHSKIN reference rest surface",
        ), compact_face_rows,
    ), dtype=np.float64)
    candidate_normals = np.asarray(human._bodyparts_skin_smooth_visual_normals(
        human._bodyparts_vertex_normals(
            (candidate_rest[referenced] * 1000.0).tolist(), compact_face_rows,
            "candidate NHSKIN reference rest surface",
        ), compact_face_rows,
    ), dtype=np.float64)
    old_normals /= np.linalg.norm(old_normals, axis=1)[:, None]
    candidate_normals /= np.linalg.norm(candidate_normals, axis=1)[:, None]
    source_normals = decoded["vertices_f"][referenced, 3:6].astype(np.float64)
    source_normals /= np.linalg.norm(source_normals, axis=1)[:, None]
    source_normal_alignment = np.sum(source_normals * old_normals, axis=1)
    normal_change_deg = np.degrees(np.arccos(np.clip(
        np.sum(old_normals * candidate_normals, axis=1), -1.0, 1.0,
    )))
    # Normal-only smoothing has a three-face dependency radius; write only
    # normals within three edge steps of changed source positions.
    changed = set(map(int, referenced[np.linalg.norm(packed_source_delta[referenced], axis=1) > 0.0]))
    adjacency = [set() for _ in range(decoded["vertex_count"])]
    for tri in faces:
        a, b, c = map(int, tri)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    normal_affected = set(changed)
    frontier = set(changed)
    for _ in range(3):
        frontier = {neighbor for vertex in frontier for neighbor in adjacency[vertex]} - normal_affected
        normal_affected.update(frontier)
    affected_mask = np.isin(referenced, np.fromiter(normal_affected, dtype=np.int64))
    if (not np.isfinite(candidate_normals).all()
            or np.any(np.linalg.norm(candidate_normals, axis=1) <= 1.0e-12)):
        raise RuntimeError("candidate reference normals are invalid")

    # Position/normal-only ABI update. All metadata, bindings, influence
    # records, source topology, and full 86-weight matrix remain byte-identical.
    output = bytearray(raw)
    header_size, binding_size, vertex_size = 60, 36, 56
    vertex_offset = header_size + decoded["binding_count"] * binding_size
    packed_vertices = np.ndarray(
        (decoded["vertex_count"], 14), dtype="<f4", buffer=output,
        offset=vertex_offset,
    )
    changed_positions = np.flatnonzero(np.any(
        candidate_source.astype("<f4").view(np.uint32)
        != source_positions.astype("<f4").view(np.uint32), axis=1,
    ))
    packed_vertices[changed_positions, :3] = candidate_source[changed_positions].astype("<f4")
    affected_payload_ids = referenced[affected_mask]
    compact_index = np.searchsorted(referenced, affected_payload_ids)
    packed_vertices[affected_payload_ids, 3:6] = candidate_normals[compact_index].astype("<f4")
    candidate_bytes = bytes(output)
    candidate_decoded = decode_payload(candidate_bytes)
    if (not np.array_equal(decoded["indices"], candidate_decoded["indices"])
            or not np.array_equal(decoded["full_weights"], candidate_decoded["full_weights"])
            or not np.array_equal(decoded["bindings_u"], candidate_decoded["bindings_u"])
            or not np.array_equal(decoded["bindings_f"], candidate_decoded["bindings_f"])
            or not np.array_equal(decoded["vertices_u"][:, 6:14], candidate_decoded["vertices_u"][:, 6:14])):
        raise RuntimeError("eye registration unexpectedly changed topology, binding, weights, or influence records")
    untouched_positions = np.ones(decoded["vertex_count"], dtype=bool)
    untouched_positions[changed_positions] = False
    if not np.array_equal(decoded["vertices_u"][untouched_positions, :3],
                          candidate_decoded["vertices_u"][untouched_positions, :3]):
        raise RuntimeError("eye registration changed source position bits outside its nonzero patch")
    untouched_normals = np.ones(decoded["vertex_count"], dtype=bool)
    untouched_normals[affected_payload_ids] = False
    if not np.array_equal(decoded["vertices_u"][untouched_normals, 3:6],
                          candidate_decoded["vertices_u"][untouched_normals, 3:6]):
        raise RuntimeError("eye registration changed normal bits outside the affected three-ring patch")

    # Exact binary32 integer-lattice checks against all 17 registered eye-layer
    # surfaces and the complete skin surface. This remains a CPU geometry
    # prediction from the recovered step-0 rigid poses, not native validation.
    candidate_skin_records = clearance._exact_surface_records(candidate_world[referenced], compact_faces)
    baseline_skin_records = clearance._exact_surface_records(captured[referenced], compact_faces)
    baseline_pairs = audit_surface_pair(
        baseline_skin_records,
        {key: surfaces[key]["faces"] for key in sorted(EYE_KEYS)},
        pack_positions,
    )
    candidate_pairs = audit_surface_pair(
        candidate_skin_records,
        {key: surfaces[key]["faces"] for key in sorted(EYE_KEYS)},
        pack_positions,
    )
    baseline_self = clearance._audit_pair(baseline_skin_records, baseline_skin_records, same_surface=True)
    candidate_self = clearance._audit_pair(candidate_skin_records, candidate_skin_records, same_surface=True)
    baseline_self_pairs = set(map(tuple, baseline_self["triangle_pairs"]))
    candidate_self_pairs = set(map(tuple, candidate_self["triangle_pairs"]))
    candidate_triangles = candidate_world[faces]
    baseline_triangles = captured[faces]
    candidate_cross = np.cross(
        candidate_triangles[:, 1] - candidate_triangles[:, 0],
        candidate_triangles[:, 2] - candidate_triangles[:, 0],
    )
    baseline_cross = np.cross(
        baseline_triangles[:, 1] - baseline_triangles[:, 0],
        baseline_triangles[:, 2] - baseline_triangles[:, 0],
    )
    candidate_area = np.linalg.norm(candidate_cross, axis=1)
    baseline_area = np.linalg.norm(baseline_cross, axis=1)
    area_ratio = candidate_area / baseline_area
    normal_dot = np.einsum("ij,ij->i", baseline_cross, candidate_cross) / (
        baseline_area * candidate_area
    )
    if (np.any(candidate_area == 0.0) or not np.isfinite(candidate_area).all()
            or not np.isfinite(normal_dot).all()):
        raise RuntimeError("candidate skin contains a zero-area or non-finite triangle")

    movable_boundary_rows = [
        row for side in ("right", "left")
        for row in boundary_candidate_metrics[side]["vertices"]
        if row["was_inward_movable_vertex"]
    ]
    boundary_values = [row["candidate_radial_shell_clearance_mm"] for row in movable_boundary_rows]
    actual_boundary_clearance_min_mm = (
        min(value for value in boundary_values if value is not None)
        if any(value is not None for value in boundary_values) else None
    )
    actual_boundary_clearance_gate_pass = (
        len(movable_boundary_rows) == 84
        and all(value is not None and value >= FINAL_ACTUAL_CLEARANCE_MM for value in boundary_values)
    )
    outside_anchors_unchanged = all(
        np.array_equal(source_positions[int(vertex)].astype("<f4"), candidate_source[int(vertex)].astype("<f4"))
        for side in ("right", "left")
        for vertex in ring_source_rows[side]["outside_end_arc_anchor_ids"]
    )
    self_pairs_zero = int(candidate_self["count"]) == 0
    eye_pairs_zero = int(sum(row["count"] for row in candidate_pairs.values())) == 0
    touched_area_floor_pass = float(area_ratio[problem.area_faces].min(initial=np.inf)) >= 0.05
    no_reversed_faces = int(np.count_nonzero(normal_dot[problem.area_faces] <= 0.0)) == 0
    no_zero_area_faces = int(np.count_nonzero(candidate_area[problem.area_faces] == 0.0)) == 0
    all_geometry_gates_pass = bool(
        actual_boundary_clearance_gate_pass and outside_anchors_unchanged
        and self_pairs_zero and eye_pairs_zero and touched_area_floor_pass
        and no_reversed_faces and no_zero_area_faces
    )

    OUT.mkdir(parents=True)
    candidate_path = OUT / "bodyparts3d-myosim-skinned-shell.eye-registration-candidate.nhskin"
    candidate_path.write_bytes(candidate_bytes)
    with (OUT / "changed-source-positions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("vertex_id", "base_x_mm", "base_y_mm", "base_z_mm", "candidate_x_mm", "candidate_y_mm", "candidate_z_mm", "delta_x_mm", "delta_y_mm", "delta_z_mm"))
        for vertex_id in changed_positions:
            old_mm = source_positions[int(vertex_id)] * 1000.0
            new_mm = candidate_source[int(vertex_id)] * 1000.0
            delta_mm = new_mm - old_mm
            writer.writerow((int(vertex_id), *old_mm.tolist(), *new_mm.tolist(), *delta_mm.tolist()))
    input_paths = (
        PACK, RECEIPT, BASE_SKIN, SKIN_MANIFEST, BONE, BONE_MANIFEST, REGISTRATION,
        RECONSTRUCTION, RIGID_REFERENCE, ENVELOPE, LOOP_ORDER, HELPER_SOURCE,
        OPTIMIZER_SOURCE, REPLAY_PLAN, REPLAY_SCRIPT, REPLAY_CANDIDATE, REPLAY_REPORT,
        Path(__file__).resolve(), FAILED_898, FAILED_899, FAILED_900, FAILED_900_PASSES, FAILED_904_LOG, FAILED_904_PLAN, FAILED_905_LOG, FAILED_905_PLAN,
        *FAILED_PREFLIGHTS, OLD_CANDIDATE, OLD_DIAG, RESIDUAL_DIAG, *REPLAY_TESTS,
    )
    helper_sha_after = sha(HELPER_SOURCE)
    helper_commit_after = subprocess.check_output(
        ["git", "-C", str(HELPER_SOURCE.parents[2]), "rev-parse", "HEAD"], text=True,
    ).strip()
    if (helper_sha_after != helper_sha_before
            or helper_commit_after != helper_commit_before):
        raise RuntimeError("pinned 935 common-atlas helper changed during eye analysis")
    report = {
        "schema": "numi.human.constrained-eye-skin-registration-candidate.v7",
        "status": (("step0_cpu_geometry_gates_passed_from_feasible_capped_iterate_optimizer_not_converged_pending_native_confirmation" if not optimizer_report["success"] else "step0_cpu_geometry_gates_passed_optimizer_converged_pending_native_confirmation") if all_geometry_gates_pass else "step0_cpu_geometry_gate_failed"),
        "interpretation": (
            "This is an inferred local skin-to-outer-eye-envelope registration from a single accepted step-0 "
            "capture. The source atlas does not identify a dedicated eyelid/conjunctival patch. The 0.25 mm "
            "offset is an engineering separation margin, not measured skin thickness or an anatomical standard."
        ),
        "predeclared_method": {
            "boundary_target": "The 44 right and 40 left inward-loop vertices are free variables constrained to a nominal 0.255 mm radial clearance from the captured outermost sclera/cornea ray envelope; the minimum-change solve may add local displacement beyond a simple 0.255 mm projection to satisfy edge/centroid samples; after float32 mapping the direct updated-ray gate remains >=0.250 mm. The 16 right and 15 left already-outside end-arc vertices are exact zero-displacement anchors.",
            "local_taper": "sparse minimum-displacement local solve over witness-seeded mesh patches; graph-gradient regularization with exact ring targets, fixed outside anchors, fixed outer support collars, and barycentric edge-midpoint/centroid clearance constraints on exact witness faces.",
            "margin_mm": MARGIN_MM,
            "nominal_directional_offset_mm": MARGIN_MM,
            "final_actual_radial_clearance_gate_mm": FINAL_ACTUAL_CLEARANCE_MM,
            "guardband_basis": "The 5 micrometer increment over the 0.250 mm actual-clearance gate covers the measured 2.815 micrometer ray-linearization deficit plus sub-micrometer float32 packing uncertainty; engineering guardband only, not anatomy.",
            "support_radius_edge_multiple": SUPPORT_EDGE_MULTIPLE,
            "interpolation_neighbor_count": INTERPOLATION_NEIGHBORS,
            "source_delta": "full 86-owner affine LBS Jacobian inverse using the accepted 884 step-0 body poses recovered by exact-topology Kabsch fits from all 185 NHBONES members; source coordinates are float32-quantized, then full-forward-evaluated again.",
            "normal_update": "recompute ABI-5 rest-world normals with the existing lower-limb anchor-rebind model helpers from its owner runtime reference; update only the changed source-position set plus three topology edge rings required by the existing three-pass visual-normal smoother.",
            "geometry_checks": "direct updated-ray actual clearance >=0.250 mm for all 84 inward boundary vertices after float32 forward mapping; exact binary32 integer-lattice skin self-intersection and skin versus each of 17 captured ocular surfaces; finite nonzero triangle area, original-winding dot, and signed projected area floor of 5% on every touched face.",
            "topology": "no skin face, index, or edge was added, removed, or reordered; the two boundary loops and every outside end-arc anchor remain exact in source positions.",
        },
        "inputs": {
            str(path): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "capture": {
            "run": "native-common-atlas-skin-cycle-884",
            "accepted_step": 0,
            "later_outcome": "failed_at_step_255",
            "accepted_capture_only": True,
            "pack_counts": pack_counts,
            "bone_owners_recovered": int(len(poses)),
            "bone_pose_fit": pose_fit,
            "maximum_full_weight_skin_replay_residual_um": float(base_error.max() * 1e6),
            "p95_full_weight_skin_replay_residual_um": float(np.percentile(base_error, 95) * 1e6),
            "maximum_affine_jacobian_condition_number": float(condition.max()),
            "maximum_packed_inverse_roundtrip_error_um": float(inverse_roundtrip.max() * 1e6),
            "maximum_forward_prediction_error_um": float(forward_error.max() * 1e6),
            "pose_reconstruction_basis": "886 full-86-owner reconstruction report is independently bound to this exact accepted pack, receipt, NHSKIN and NHBONES hashes; current 935 common_atlas_skin_clearance helpers reproduce that method.",
            "canonical_helper_git_commit": helper_commit_before,
            "canonical_helper_sha256_before": helper_sha_before,
            "canonical_helper_sha256_after": helper_sha_after,
        },
        "boundary_scope": ring_source_rows,
        "boundary_candidate_metrics": boundary_candidate_metrics,
        "field": {
            **field_report,
            "patch_vertex_ids": problem.patch_vertices.astype(int).tolist(),
            "touched_face_rows": problem.area_faces.astype(int).tolist(),
            "changed_source_position_vertex_ids": changed_positions.astype(int).tolist(),
        },
        "candidate_geometry": {
            "optimizer_converged": bool(optimizer_report["success"]),
            "feasible_capped_iterate_used_for_geometry_audit": bool(optimizer_report["feasible_iteration_captured_for_geometry_audit"]),
            "nominal_directional_offset_mm": MARGIN_MM,
            "final_actual_radial_clearance_gate_mm": FINAL_ACTUAL_CLEARANCE_MM,
            "minimum_actual_movable_boundary_clearance_mm": actual_boundary_clearance_min_mm,
            "movable_boundary_vertices_below_final_actual_clearance_gate": [row["vertex_id"] for row in movable_boundary_rows if row["candidate_radial_shell_clearance_mm"] is None or row["candidate_radial_shell_clearance_mm"] < FINAL_ACTUAL_CLEARANCE_MM],
            "exact_skin_self_intersection_pairs_before": int(baseline_self["count"]),
            "exact_skin_self_intersection_pairs_after": int(candidate_self["count"]),
            "exact_skin_to_ocular_triangle_pairs_before": int(sum(x["count"] for x in baseline_pairs.values())),
            "exact_skin_to_ocular_triangle_pairs_after": int(sum(x["count"] for x in candidate_pairs.values())),
            "candidate_exact_skin_self_pairs_added": len(candidate_self_pairs - baseline_self_pairs),
            "candidate_exact_skin_self_pairs_removed": len(baseline_self_pairs - candidate_self_pairs),
            "candidate_skin_to_ocular_pairs_added": int(sum(
                len(set(map(tuple, candidate_pairs[key]["triangle_pairs"]))
                    - set(map(tuple, baseline_pairs[key]["triangle_pairs"])))
                for key in candidate_pairs)),
            "candidate_skin_to_ocular_pairs_removed": int(sum(
                len(set(map(tuple, baseline_pairs[key]["triangle_pairs"]))
                    - set(map(tuple, candidate_pairs[key]["triangle_pairs"])))
                for key in candidate_pairs)),
            "per_ocular_surface_pairs": {
                f"{key[0]}:{key[1]}": {
                    "before": int(baseline_pairs[key]["count"]),
                    "after": int(candidate_pairs[key]["count"]),
                    "introduced": len(set(map(tuple, candidate_pairs[key]["triangle_pairs"]))
                                       - set(map(tuple, baseline_pairs[key]["triangle_pairs"]))),
                    "removed": len(set(map(tuple, baseline_pairs[key]["triangle_pairs"]))
                                   - set(map(tuple, candidate_pairs[key]["triangle_pairs"]))),
                } for key in sorted(EYE_KEYS)
            },
            "candidate_triangle_area_ratio_min": float(area_ratio.min()),
            "candidate_triangle_area_ratio_p01": float(np.quantile(area_ratio, 0.01)),
            "candidate_triangle_area_ratio_median": float(np.median(area_ratio)),
            "candidate_triangle_original_winding_dot_min": float(normal_dot.min()),
            "candidate_triangles_with_reversed_original_winding": int(np.count_nonzero(normal_dot <= 0.0)),
            "candidate_triangles_with_area_ratio_below_0_1": int(np.count_nonzero(area_ratio < 0.1)),
            "candidate_triangles_with_area_ratio_below_declared_0_05_floor": int(np.count_nonzero(area_ratio < 0.05)),
            "candidate_triangles_with_area_ratio_below_declared_0_05_floor_touched": int(np.count_nonzero(area_ratio[problem.area_faces] < 0.05)),
            "zero_area_triangles": int(np.count_nonzero(candidate_area == 0.0)),
            "normal_update": {
                "affected_vertex_count": int(len(affected_payload_ids)),
                "vertices_with_source_normal_alignment_below_0_99": int(np.count_nonzero(source_normal_alignment < 0.99)),
                "source_to_reference_recomputed_normal_alignment_p01": float(np.quantile(source_normal_alignment, 0.01)),
                "source_to_reference_recomputed_normal_alignment_min": float(source_normal_alignment.min()),
                "recomputed_normal_change_deg_p50": float(np.median(normal_change_deg)),
                "recomputed_normal_change_deg_p95": float(np.quantile(normal_change_deg, 0.95)),
                "recomputed_normal_change_deg_max": float(normal_change_deg.max()),
            },
            "preservation": {
                "source_payload_bytes": int(len(raw)),
                "candidate_payload_bytes": int(len(candidate_bytes)),
                "source_sha256": sha(BASE_SKIN),
                "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                "topology_indices_byte_identical": True,
                "full_weight_matrix_byte_identical": True,
                "all_binding_records_byte_identical": True,
                "per_vertex_influence_records_byte_identical": True,
                "nonpatch_source_position_bytes_identical": True,
                "normal_bytes_outside_three_ring_patch_identical": True,
                "skin_source_raw_preserved": True,
            },
            "triangle_area_ratio_min_face": int(np.argmin(area_ratio)),
            "triangle_winding_dot_min_face": int(np.argmin(normal_dot)),
        },
        "geometry_gate_results": {
            "no_new_exact_skin_self_intersections": len(candidate_self_pairs - baseline_self_pairs) == 0,
            "no_new_exact_skin_to_ocular_intersections": int(sum(
                len(set(map(tuple, candidate_pairs[key]["triangle_pairs"]))
                    - set(map(tuple, baseline_pairs[key]["triangle_pairs"])))
                for key in candidate_pairs)) == 0,
            "all_skin_to_ocular_intersections_removed": int(sum(x["count"] for x in candidate_pairs.values())) == 0,
            "no_zero_area_or_reversed_skin_triangles": int(np.count_nonzero(candidate_area == 0.0)) == 0 and int(np.count_nonzero(normal_dot <= 0.0)) == 0,
            "all_touched_faces_meet_0_05_area_floor": touched_area_floor_pass,
            "all_movable_inward_boundary_vertices_actual_clearance_ge_0_250_mm": actual_boundary_clearance_gate_pass,
            "all_outside_end_arc_anchors_unchanged": outside_anchors_unchanged,
            "zero_skin_self_intersections": self_pairs_zero,
            "zero_skin_to_ocular_intersections_all_17_surfaces": eye_pairs_zero,
            "no_touched_face_reversals": no_reversed_faces,
            "no_touched_zero_area_faces": no_zero_area_faces,
            "all_geometry_gates_pass": all_geometry_gates_pass,
            "preserved_boundary_loop_sizes": sorted(map(len, boundary)) == [55, 60],
            "preserved_outside_end_arc_positions": all(np.array_equal(
                source_positions[vertex_id].astype("<f4"), candidate_source[vertex_id].astype("<f4")
            ) for side in ("right", "left") for vertex_id in ring_source_rows[side]["outside_end_arc_anchor_ids"]),
        },
        "limits": [
            "884 failed later at step 255; this report uses step 0 only and makes no dynamic-cycle or all-pose claim.",
            "The candidate is an inferred skin-to-eye interface adjustment, not a source-authored or measured eyelid asset.",
            "The exact-pair audit uses a full-weight float32-quantized affine prediction from recovered step-0 rigid body poses; the native renderer must still confirm candidate geometry.",
            "The 0.255 mm nominal directional target and 0.250 mm final actual-clearance gate are engineering values; the 5 micrometer guardband is not a clinical or tissue-thickness reference.",
            "This local patch does not qualify other skin-to-anatomy interfaces or whole-body anatomy.",
        ],
    }
    write_json(OUT / "report.json", report)
    pair_csv = OUT / "ocular-pair-summary.csv"
    with pair_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "surface_semantic", "surface_stable_id", "skin_eye_pairs_before",
            "skin_eye_pairs_after", "introduced_pairs", "removed_pairs",
        ))
        writer.writeheader()
        for key in sorted(EYE_KEYS):
            row = report["candidate_geometry"]["per_ocular_surface_pairs"][f"{key[0]}:{key[1]}"]
            writer.writerow({
                "surface_semantic": key[0], "surface_stable_id": key[1],
                "skin_eye_pairs_before": row["before"], "skin_eye_pairs_after": row["after"],
                "introduced_pairs": row["introduced"], "removed_pairs": row["removed"],
            })
    readme = (
        "# Inferred local eye-to-skin registration candidate\n\n"
        "This CPU-only prototype solves a sparse minimum-change displacement on witness-seeded local skin patches. "
        "The inward aperture arcs target the accepted step-0 outer sclera/cornea envelope at +0.255 mm nominal directional offset, followed by a direct +0.250 mm actual-clearance gate; outside "
        "end-arc anchors and the cleared outer support collar remain fixed. Vertex plus edge-midpoint/centroid "
        "clearance constraints guide the seven residual witness faces; exact triangle checks remain final. The 5% area floor and "
        "0.255 mm nominal target and 0.250 mm final gate are engineering criteria, not measured anatomy or tissue thickness.\n\n"
        "The evidence binds to accepted step 0 of run 884, which later failed at step 255. No current-pose, "
        "multi-state, native-render, or whole-anatomy qualification follows. Exact pair counts are reported for "
        "each of 17 ocular surfaces and for skin self-intersections. The source payload remains untouched.\n"
    )
    (OUT / "README.md").write_text(readme)
    print(json.dumps({
        "candidate_path": str(candidate_path),
        "candidate_sha256": report["candidate_geometry"]["preservation"]["candidate_sha256"],
        "report_path": str(OUT / "report.json"),
        "report_sha256": sha(OUT / "report.json"),
        "skin_eye_pairs_before": report["candidate_geometry"]["exact_skin_to_ocular_triangle_pairs_before"],
        "skin_eye_pairs_after": report["candidate_geometry"]["exact_skin_to_ocular_triangle_pairs_after"],
        "skin_self_before": int(baseline_self["count"]),
        "skin_self_after": int(candidate_self["count"]),
        "area_ratio_min": float(area_ratio.min()),
        "winding_dot_min": float(normal_dot.min()),
        "maximum_boundary_projection_mm": field_report["maximum_displacement_mm"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
