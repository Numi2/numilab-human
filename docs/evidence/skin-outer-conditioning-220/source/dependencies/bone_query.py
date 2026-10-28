"""Pinned 185-bank signed distance queries for offline O/I/D layer fitting.

This is a geometric query adapter, not a physical admission or exact containment
certificate. It keeps one libigl AABB and Fast Winding Number BVH per typed bank
for the lifetime of the query object.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D")
WORK = ROOT / "connected-obstacle-207/physical-transfer-209"
DEFAULT_BANK_DIR = WORK / "geometry-worker/D-connectivity-218/euler-localization-001/verified-039-inputs"
DEFAULT_EVIDENCE_ROOT = ROOT / "publication-206-stage-001/tools"
DEFAULT_AUTHORING_DEPS = ROOT / "connected-obstacle-207/authoring-deps"
DEFAULT_PARSER = ROOT / "publication-206-stage-001/tools/human_skin_fat_bone_audit.py"

EXPECTED_PINS = {
    "exact-emitted.triangles.csv": "0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
    "exact-emitted.nodes.csv": "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c",
    "exact-emitted.bodies.csv": "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
    "all-bank-containment.csv.banks.csv": "46568d191d84dd68f1fbcfe02b7938b15d4a7b3b786f268f46bcfe916070c6df",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BoneDistanceQuery:
    """Reusable signed bone-clearance query over the pinned 185 typed banks.

    Pass emitted Float32 world points (Nx3). The stored positions are promoted to
    Float64 for the same meter-space query convention used by the approved audit.
    query(points_world_m, margin_m) returns only point/bank rows whose point is
    inside by winding classification or whose signed distance is at most margin.
    The outward normal is the gradient of positive-outside signed distance.
    """

    def __init__(
        self,
        bank_dir: Path | str = DEFAULT_BANK_DIR,
        evidence_root: Path | str = DEFAULT_EVIDENCE_ROOT,
        authoring_deps: Path | str = DEFAULT_AUTHORING_DEPS,
        parser_path: Path | str = DEFAULT_PARSER,
        expected_pins: dict[str, str] = EXPECTED_PINS,
        winding_order: int = 2,
        winding_accuracy: float = 2.0,
    ):
        authoring_deps = Path(authoring_deps)
        if str(authoring_deps) not in sys.path:
            sys.path.insert(0, str(authoring_deps))
        import igl  # provided by pinned authoring-deps

        parser_path = Path(parser_path)
        helper = _load_module("numi_pinned_human_skin_fat_bone_audit", parser_path)
        (triangles, ids, gids, bodies, matrices, offsets, positions,
         bankkeys, pins, _exact) = helper.read_bank_inputs(Path(bank_dir), Path(evidence_root))
        for name, expected in expected_pins.items():
            if pins.get(name) != expected:
                raise ValueError(f"pinned bank hash changed for {name}: {pins.get(name)}")
        if len(bankkeys) != 185:
            raise ValueError(f"expected 185 banks, received {len(bankkeys)}")

        self._igl = igl
        self._winding_accuracy = float(winding_accuracy)
        self.bank_hashes = dict(pins)
        self.parser_sha256 = _sha256(parser_path)
        self.bank_count = int(len(bankkeys))
        self.triangle_count = int(len(triangles))
        self._banks: list[dict[str, Any]] = []

        for body0, bone0 in bankkeys:
            body, bone = int(body0), int(bone0)
            mask = (ids[:, 0] == body) & (ids[:, 1] == bone)
            local_tri = triangles[mask]
            tri_ids = ids[mask]
            # Keep exact emitted triangles in their pinned body-local meter frame.
            # Query points use the exact audit conversion (world-position) @ M.T - offset.
            vertices = np.ascontiguousarray(local_tri.reshape(-1, 3), dtype=np.float64)
            faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
            distance_bvh = igl.AABB()
            distance_bvh.init(vertices, faces)
            winding_bvh = igl.FastWindingNumberBVH()
            winding_bvh.init(vertices, faces, int(winding_order))
            self._banks.append({
                "body": body,
                "stable_bone_id": bone,
                "vertices_local_m": vertices,
                "faces": faces,
                "world_to_body": np.asarray(matrices[body], dtype=np.float64),
                "body_to_world_row": np.linalg.inv(np.asarray(matrices[body], dtype=np.float64)).T,
                "position_world_m": np.asarray(positions[body], dtype=np.float64),
                "com_offset_m": np.asarray(offsets[body], dtype=np.float64),
                "source_record_index": np.asarray(tri_ids[:, 2], dtype=np.int64),
                "triangle_stable_id": np.asarray(tri_ids[:, 3], dtype=np.int64),
                "distance_bvh": distance_bvh,
                "winding_bvh": winding_bvh,
            })

    def query(self, points_world_m: np.ndarray, margin_m: float | np.ndarray) -> list[dict[str, Any]]:
        points_raw = np.asarray(points_world_m)
        if points_raw.ndim != 2 or points_raw.shape[1] != 3:
            raise ValueError("points_world_m must have shape (N, 3)")
        if points_raw.dtype != np.float32:
            raise TypeError("pass the exact emitted Float32 world points; cast evolving fit states explicitly")
        if not np.isfinite(points_raw).all():
            raise ValueError("points_world_m contains nonfinite values")
        points = np.ascontiguousarray(points_raw.astype(np.float64))
        margin = np.asarray(margin_m, dtype=np.float64)
        if margin.ndim == 0:
            margin = np.full(len(points), float(margin), dtype=np.float64)
        if margin.shape != (len(points),) or not np.isfinite(margin).all() or np.any(margin < 0):
            raise ValueError("margin_m must be finite, nonnegative scalar or one value per point")

        rows: list[dict[str, Any]] = []
        for bank in self._banks:
            vertices = bank["vertices_local_m"]
            faces = bank["faces"]
            points_local = np.ascontiguousarray(
                (points - bank["position_world_m"]) @ bank["world_to_body"].T
                - bank["com_offset_m"], dtype=np.float64
            )
            sqr_d, face_index, closest = bank["distance_bvh"].squared_distance(
                vertices, faces, points_local
            )
            dist = np.sqrt(np.maximum(sqr_d, 0.0))
            winding = bank["winding_bvh"].winding_number(
                points_local, self._winding_accuracy
            )
            abs_winding = np.abs(winding)
            inside = abs_winding >= 0.5
            ambiguous = (abs_winding >= 0.45) & (abs_winding < 0.55)
            # Ambiguous winding is conservatively treated as inside and retained
            # as an explicit flag for the caller to inspect.
            signed = np.where(inside | ambiguous, -dist, dist)
            violated = inside | ambiguous | (signed <= margin)
            for row in np.flatnonzero(violated):
                d = float(dist[row])
                delta = points_local[row] - closest[row]
                if d > 1e-15:
                    normal_local = (delta / d) * (-1.0 if (inside[row] or ambiguous[row]) else 1.0)
                    normal_quality = "signed-distance-gradient"
                else:
                    fi = int(face_index[row])
                    tri = vertices[faces[fi]]
                    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    nn = float(np.linalg.norm(n))
                    if nn > 0:
                        n /= nn
                        eps = max(1e-6, 32.0 * float(np.max(np.spacing(np.abs(closest[row])))))
                        probe = np.vstack((closest[row] + eps * n, closest[row] - eps * n))
                        probe_w = np.abs(bank["winding_bvh"].winding_number(
                            np.ascontiguousarray(probe), self._winding_accuracy
                        ))
                        if probe_w[0] >= 0.5 and probe_w[1] < 0.5:
                            n = -n
                            normal_quality = "oriented-by-local-winding-probe"
                        elif probe_w[1] >= 0.5 and probe_w[0] < 0.5:
                            normal_quality = "oriented-by-local-winding-probe"
                        else:
                            normal_quality = "source-face-orientation-ambiguous"
                        normal_local = n
                    else:
                        normal_local = np.full(3, np.nan)
                        normal_quality = "degenerate-nearest-face"
                normal_world = normal_local @ bank["world_to_body"]
                normal_norm = float(np.linalg.norm(normal_world))
                if normal_norm > 0:
                    normal_world = normal_world / normal_norm
                closest_world = (closest[row] + bank["com_offset_m"]) @ bank["body_to_world_row"] + bank["position_world_m"]
                fi = int(face_index[row])
                rows.append({
                    "query_row": int(row),
                    "signed_distance_m": float(signed[row]),
                    "unsigned_distance_m": d,
                    "closest_point_local_m": [float(x) for x in closest[row]],
                    "closest_point_world_m": [float(x) for x in closest_world],
                    "outward_normal_local": [float(x) for x in normal_local],
                    "outward_normal_world": [float(x) for x in normal_world],
                    "distance_frame": "pinned_body_local_meters",
                    "normal_quality": normal_quality,
                    "body": int(bank["body"]),
                    "stable_bone_id": int(bank["stable_bone_id"]),
                    "source_record_index": int(bank["source_record_index"][fi]),
                    "triangle_stable_id": int(bank["triangle_stable_id"][fi]),
                    "winding_number": float(winding[row]),
                    "winding_sign_ambiguous": bool(ambiguous[row]),
                    "clearance_margin_m": float(margin[row]),
                    "violation_kind": (
                        "winding-sign-ambiguous" if ambiguous[row]
                        else "inside-bone" if inside[row]
                        else "insufficient-clearance"
                    ),
                })
        rows.sort(key=lambda r: (r["query_row"], r["body"], r["stable_bone_id"]))
        return rows

