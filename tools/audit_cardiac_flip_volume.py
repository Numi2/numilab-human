#!/usr/bin/env python3
"""Bound a source-row topology edit's signed-volume delta over a common-field cube."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_pleura_proxy import _parse_payload  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("cardiac flip volume audit: " + message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tetra_volume_delta_polynomial(
    anchor: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> tuple[dict[tuple[int, ...], float], float]:
    """Return det/6's polynomial coefficients and absolute-product sum.

    Each input contains one base xyz row followed by seven affine xyz
    coefficients, matching the seven common-field coordinates in the receipt.
    """
    affine = [np.asarray(value, dtype=np.float64) for value in
              (anchor, point_a, point_b, point_c)]
    if any(value.shape != (8, 3) for value in affine):
        raise ValueError("tetra vertices must have shape (8, 3): base plus seven affine rows")
    if not all(np.isfinite(value).all() for value in affine):
        raise ValueError("tetra vertices contain non-finite coordinates")

    anchor, point_a, point_b, point_c = affine
    vectors = [point - anchor for point in (point_a, point_b, point_c)]
    grouped: dict[tuple[int, ...], list[float]] = {}
    absolute_products = []
    for i, j, k in itertools.product(range(8), repeat=3):
        x, y, z = vectors[0][i], vectors[1][j], vectors[2][k]
        terms = (
            x[0] * y[1] * z[2],
            x[1] * y[2] * z[0],
            x[2] * y[0] * z[1],
            -x[2] * y[1] * z[0],
            -x[1] * y[0] * z[2],
            -x[0] * y[2] * z[1],
        )
        exponent = tuple((i == variable) + (j == variable) + (k == variable)
                         for variable in range(1, 8))
        grouped.setdefault(exponent, []).append(math.fsum(terms) / 6.0)
        absolute_products.append(math.fsum(abs(term) for term in terms) / 6.0)
    coefficients = {
        exponent: math.fsum(values)
        for exponent, values in grouped.items()
        if any(values)
    }
    return coefficients, math.fsum(absolute_products)


def evaluate_volume_polynomial(
    coefficients: dict[tuple[int, ...], float],
    coordinates: np.ndarray,
) -> float:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (7,) or not np.isfinite(coordinates).all():
        raise ValueError("coordinate vector must contain seven finite values")
    return math.fsum(
        coefficient * math.prod(x ** power for x, power in zip(coordinates, exponent))
        for exponent, coefficient in coefficients.items()
    )


def relative_volume_bounds(
    coefficients: dict[tuple[int, ...], float],
    absolute_products: float,
    reference_volume_m3: float,
) -> tuple[float, float, float]:
    if not math.isfinite(absolute_products) or absolute_products < 0:
        raise ValueError("absolute product sum must be finite and nonnegative")
    if not math.isfinite(reference_volume_m3) or reference_volume_m3 <= 0:
        raise ValueError("reference volume must be finite and positive")
    rounding_allowance = absolute_products * 1.0e-12
    polynomial_l1 = math.fsum(abs(value) for value in coefficients.values())
    uniform_relative_bound = (polynomial_l1 + rounding_allowance) / reference_volume_m3
    derivative_relative_bound = 3.0 * uniform_relative_bound
    return uniform_relative_bound, derivative_relative_bound, rounding_allowance


def _oriented_triangle_key(face: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = map(int, face)
    return min((a, b, c), (b, c, a), (c, a, b))


def oriented_boundary_signature(faces: np.ndarray) -> tuple[tuple[tuple[int, int], int], ...]:
    """Return the directed boundary 1-chain, ignoring internal edge pairs."""
    edge_balance: Counter[tuple[int, int]] = Counter()
    for face in np.asarray(faces):
        a, b, c = map(int, face)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            edge_balance[edge] += 1 if first < second else -1
    return tuple(sorted((edge, count) for edge, count in edge_balance.items() if count))


def surface_volume_delta_polynomial(
    old_faces: np.ndarray,
    new_faces: np.ndarray,
    affine_vertex,
) -> tuple[dict[tuple[int, ...], float], float, list[dict], list[dict]]:
    """Sum oriented tetra-volume polynomials for the changed triangle multiset.

    Equal oriented triangles cancel first. Equal oriented boundaries are required,
    so the signed-volume difference does not depend on the chosen affine origin.
    """
    old_faces = np.asarray(old_faces, dtype=np.int64)
    new_faces = np.asarray(new_faces, dtype=np.int64)
    if old_faces.ndim != 2 or old_faces.shape[1] != 3:
        raise ValueError("old faces must have shape (n, 3)")
    if new_faces.ndim != 2 or new_faces.shape[1] != 3:
        raise ValueError("new faces must have shape (n, 3)")
    if oriented_boundary_signature(old_faces) != oriented_boundary_signature(new_faces):
        raise ValueError("source and candidate oriented surface boundaries differ")

    old_counts = Counter(_oriented_triangle_key(face) for face in old_faces)
    new_counts = Counter(_oriented_triangle_key(face) for face in new_faces)
    removed = old_counts - new_counts
    added = new_counts - old_counts
    zero = np.zeros((8, 3), dtype=np.float64)
    coefficient_terms: dict[tuple[int, ...], list[float]] = {}
    absolute_products = 0.0
    removed_rows, added_rows = [], []
    for signed_counts, sign, output in ((removed, 1.0, removed_rows), (added, -1.0, added_rows)):
        for face, count in sorted(signed_counts.items()):
            if len(set(face)) != 3:
                raise ValueError("changed surface contains a degenerate triangle")
            coefficients, face_products = tetra_volume_delta_polynomial(
                zero, *(affine_vertex(index) for index in face)
            )
            for exponent, value in coefficients.items():
                coefficient_terms.setdefault(exponent, []).append(sign * count * value)
            absolute_products += count * face_products
            output.append({"source_triangle": list(face), "multiplicity": count})
    coefficients = {
        exponent: math.fsum(values)
        for exponent, values in coefficient_terms.items()
        if any(values)
    }
    return coefficients, absolute_products, removed_rows, added_rows


def audit(
    base_path: Path,
    candidate_path: Path,
    receipt_path: Path,
    stable_ids: int | list[int] | tuple[int, ...],
    relative_budget: float = 1e-9,
) -> dict:
    base_path = Path(base_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    declared_ids = [int(stable_ids)] if isinstance(stable_ids, (int, np.integer)) else [int(sid) for sid in stable_ids]
    require(bool(declared_ids), "at least one stable ID must be declared")
    require(len(set(declared_ids)) == len(declared_ids), "declared stable IDs must be unique")
    declared_set = set(declared_ids)

    base_raw = base_path.read_bytes()
    candidate_raw = candidate_path.read_bytes()
    old_header, _, old_rows = _parse_payload(base_raw)
    new_header, _, new_rows = _parse_payload(candidate_raw)
    require(old_rows.keys() == new_rows.keys(), "stable row allocations differ")
    require(old_header[:4] == new_header[:4] and old_header[5:] == new_header[5:],
            "payload identity, surface count, vertex allocation, or registration changed")
    missing = declared_set.difference(old_rows)
    require(not missing, f"declared stable IDs are absent from source payload: {sorted(missing)}")

    changed = []
    for sid in old_rows:
        old, new = old_rows[sid], new_rows[sid]
        require(old["body_index"] == new["body_index"]
                and old["layer"] == new["layer"]
                and old["flags"] == new["flags"], f"stable row {sid} identity changed")
        require(old["vertices6"].shape == new["vertices6"].shape
                and old["vertices6"].tobytes() == new["vertices6"].tobytes(),
                f"stable row {sid} vertex attributes or allocation changed")
        if not np.array_equal(old["faces"], new["faces"]):
            changed.append(sid)
    require(set(changed) == declared_set,
            f"changed stable rows must exactly match declared IDs {sorted(declared_set)}; found {changed}")

    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    require(common["solver"]["trial_lower"] == [-1.0] * 7
            and common["solver"]["trial_upper"] == [1.0] * 7,
            "receipt common-field coordinate cube differs")
    require(common.get("anatomy_payload_sha256") == hashlib.sha256(base_raw).hexdigest(),
            "receipt common-field map is not bound to the base anatomy payload")
    map_path = receipt_path.parent / common["map"]["path"]
    map_sha256 = sha256(map_path)
    require(map_sha256 == common["map"]["sha256"], "common-field map hash differs")
    common_map = np.fromfile(map_path, dtype="<f4")
    require(common_map.size % 28 == 0, "common-field map has a malformed stride")
    common_map = common_map.reshape(-1, 7, 4)
    require(np.isfinite(common_map).all(), "common-field map contains non-finite values")
    volume_ids = common["volume_stable_ids"]
    source_volumes = common["source_reference_volumes_m3"]
    require(len(volume_ids) == len(source_volumes), "source volume ID/value arrays differ")
    volume_by_id = {int(sid): float(volume) for sid, volume in zip(volume_ids, source_volumes)}
    require(len(volume_by_id) == len(volume_ids), "source reference volume IDs are duplicated")
    ranges = {int(item["stable_id"]): item for item in common["vertex_ranges"]}
    require(len(ranges) == len(common["vertex_ranges"]), "common-field stable ID ranges are duplicated")
    solver_tolerance = float(common["solver"]["relative_volume_tolerance"])
    require(math.isfinite(relative_budget) and relative_budget > 0,
            "relative preparation budget must be finite and positive")
    require(math.isfinite(solver_tolerance) and solver_tolerance > 0,
            "receipt solver relative-volume tolerance must be finite and positive")
    require(relative_budget <= solver_tolerance / 100.0,
            "relative preparation budget may not exceed one hundredth of solver tolerance")
    budget = float(relative_budget)
    row_reports = []

    for stable_id in declared_ids:
        old, new = old_rows[stable_id], new_rows[stable_id]
        require(len(old["faces"]) > 0 and len(new["faces"]) > 0,
                f"stable row {stable_id} source and candidate must both contain faces")
        vertex_range = ranges.get(stable_id)
        require(vertex_range is not None,
                f"receipt has no common-field range for stable ID {stable_id}")
        require(int(vertex_range["vertex_count"]) == len(old["vertices6"]),
                f"stable row {stable_id} common-field vertex allocation differs from source")
        first_vertex = int(vertex_range["first_vertex"])

        def affine(vertex_index: int, *, _old=old, _first=first_vertex) -> np.ndarray:
            index = _first + int(vertex_index)
            require(0 <= index < len(common_map),
                    f"stable row {stable_id} vertex lies outside common-field map")
            return np.concatenate(
                (_old["vertices6"][vertex_index, :3][None, :], common_map[index, :, :3]),
                axis=0,
            ).astype(np.float64)

        coefficients, absolute_products, removed, added = surface_volume_delta_polynomial(
            old["faces"], new["faces"], affine
        )
        require(stable_id in volume_by_id,
                f"stable ID {stable_id} is absent from source reference volumes")
        reference_volume = volume_by_id[stable_id]
        bound, derivative_bound, rounding_allowance = relative_volume_bounds(
            coefficients, absolute_products, reference_volume
        )
        require(bound < budget,
                f"stable row {stable_id} uniform relative volume bound {bound} exceeds {budget}")
        row_reports.append({
            "stable_id": stable_id,
            "source_face_count": int(len(old["faces"])),
            "candidate_face_count": int(len(new["faces"])),
            "removed_oriented_source_triangles": removed,
            "added_oriented_candidate_triangles": added,
            "oriented_surface_boundary_byte_index_signature_preserved": True,
            "source_reference_volume_m3": reference_volume,
            "relative_volume_delta_uniform_upper_bound": bound,
            "relative_volume_derivative_delta_upper_bound": derivative_bound,
            "rounding_allowance_m3": rounding_allowance,
            "preparation_budget": budget,
            "passed": True,
        })

    max_bound = max(row["relative_volume_delta_uniform_upper_bound"] for row in row_reports)
    max_derivative_bound = max(
        row["relative_volume_derivative_delta_upper_bound"] for row in row_reports
    )
    return {
        "schema": "numi.human.cardiac-source-topology-volume-bound.v2",
        "scope": "Per-row bounds on signed source-volume integral changes under the existing seven-coordinate common field. Changed rows must exactly equal the declared stable IDs, and each row's oriented source and candidate surface boundaries must match. This numerical geometry proof does not admit candidates to native runtime use.",
        "sha256": {
            str(path): digest for path, digest in (
                (base_path, hashlib.sha256(base_raw).hexdigest()),
                (candidate_path, hashlib.sha256(candidate_raw).hexdigest()),
                (receipt_path, hashlib.sha256(receipt_raw).hexdigest()),
                (map_path, map_sha256),
            )
        },
        "declared_stable_ids": declared_ids,
        "observed_changed_stable_ids": changed,
        "all_changed_rows_declared_and_all_declared_rows_changed": set(changed) == declared_set,
        "all_vertex_attributes_bindings_and_allocations_byte_exact": True,
        "coordinate_cube": [-1.0, 1.0],
        "existing_solver_relative_tolerance": solver_tolerance,
        "preparation_budget": budget,
        "rows": row_reports,
        "aggregate_relative_volume_delta_uniform_upper_bound_max": max_bound,
        "aggregate_relative_volume_derivative_delta_upper_bound_max": max_derivative_bound,
        "aggregate_passed": all(row["passed"] for row in row_reports),
        "passed": all(row["passed"] for row in row_reports),
        "native_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="original NHANATOMY payload")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate NHANATOMY payload")
    parser.add_argument("--receipt", required=True, type=Path, help="resting-anatomy receipt JSON")
    parser.add_argument("--output", required=True, type=Path, help="volume-bound report JSON")
    parser.add_argument("--relative-budget", type=float, default=1e-9,
                        help="strict per-row preparation bound, capped at solver tolerance / 100")
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--stable-id", action="append", type=int,
                     help="stable row ID to audit; may be repeated")
    ids.add_argument("--stable-ids", nargs="+", type=int,
                     help="space-separated stable row IDs to audit")
    args = parser.parse_args()
    stable_ids = args.stable_id if args.stable_id is not None else args.stable_ids
    result = audit(args.base, args.candidate, args.receipt, stable_ids, args.relative_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
