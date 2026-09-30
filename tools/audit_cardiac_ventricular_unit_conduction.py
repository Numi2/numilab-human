"""Independently audit the one-step ventricular unit-diffusion diagnostic.

This checks a synthetic dimensionless field, not a cardiac voltage, heartbeat,
or an accepted HumanPack step. The separate source-activation gate owns the
full-face versus point-only interface classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "Build/cardiac-electrical-source-20260930/asset"
SOURCE = ROOT / "Docs/media/cardiac-source-activation-20260930"
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(f"ventricular unit-conduction audit: {why}")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, why: str, *, atol: float = 1e-12) -> None:
    require(bool(np.isfinite(actual)) and abs(actual - expected) <= atol, why)


def mesh(asset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = asset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    arrays = []
    for name, dtype, shape in (("nodes.f64le", "<f8", (-1, 3)),
                               ("tetrahedra.u32le", "<u4", (-1, 4)),
                               ("labels.u32le", "<u4", (-1,))):
        path = asset / name
        require(path.stat().st_size == manifest["buffers"][name]["bytes"]
                and sha(path) == manifest["buffers"][name]["sha256"],
                f"source {name} identity")
        arrays.append(np.fromfile(path, dtype).reshape(shape))
    positions, tets, labels = arrays
    require(positions.shape == (300965, 3)
            and tets.shape == (1470083, 4)
            and labels.shape == (1470083,), "source dimensions")
    return positions, tets, labels


def audit_mode(mode: str, run: Path, positions: np.ndarray,
               tets: np.ndarray, labels: np.ndarray,
               base_sources: np.ndarray, lv_nodes: np.ndarray,
               rv_nodes: np.ndarray, face_nodes: np.ndarray,
               point_nodes: np.ndarray) -> dict:
    blocked = mode == "blocked"
    receipt_path = run / f"{mode}.json"
    field_path = run / f"{mode}.f64le"
    mapping_path = run / f"{mode}-nodes.u32le"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["schema"] == "numi.human.ventricular-unit-conduction-native.v1"
            and receipt["blocked_interface"] is blocked
            and receipt["apple_native_cpu_step"] is True
            and receipt["source_ventricular_tetrahedra"] == 1097534
            and receipt["source_lv_cells"] == 722773
            and receipt["source_rv_cells"] == 374761
            and receipt["lv_rv_shared_source_nodes"] == 2631
            and receipt["full_face_connected_interface_nodes"] == 2628
            and receipt["point_only_split_nodes"] == 3
            and receipt["unit_diffusivity_fixture_m2_per_s"] == 1
            and receipt["production_native_electrical_steps"] == 0
            and receipt["source_voltage_or_ionic_model"] is False
            and receipt["heartbeat_qualified"] is False,
            f"{mode} claim boundary and topology")
    expected_sources = (np.concatenate([base_sources, face_nodes])
                        if blocked else base_sources)
    actual_sources = np.fromfile(mapping_path, "<u4")
    candidate = np.fromfile(field_path, "<f8")
    require(np.array_equal(actual_sources, expected_sources)
            and len(candidate) == len(expected_sources)
            == receipt["ventricular_dofs"]
            and bool(np.isfinite(candidate).all()),
            f"{mode} DOF mapping or candidate")

    base = np.full(len(positions), -1, np.int32)
    base[base_sources[:218077]] = np.arange(218077, dtype=np.int32)
    right = base.copy()
    right[point_nodes] = np.arange(218077, 218080, dtype=np.int32)
    if blocked:
        right[face_nodes] = np.arange(218080, 220708, dtype=np.int32)
    initial = np.zeros(len(expected_sources), np.float64)
    initial[base[lv_nodes]] = 1.0
    capacity = np.zeros(len(initial), np.float64)
    row_sum = np.zeros(len(initial), np.float64)
    residual = np.zeros(len(initial), np.float64)
    before = 0.0
    after = 0.0
    selected = np.flatnonzero((labels == 1) | (labels == 2))
    require(len(selected) == 1097534, "ventricular selection")

    def visit(*, assemble: bool) -> None:
        nonlocal before, after
        for start in range(0, len(selected), 25000):
            chunk = selected[start:start+25000]
            cells = tets[chunk]
            p = positions[cells]
            mapped = base[cells]
            rv = labels[chunk] == 2
            mapped[rv] = right[cells[rv]]
            require(bool((mapped >= 0).all()), "unmapped ventricular node")
            a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
            volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
            require(bool(np.isfinite(volume).all()) and bool((volume > 0).all()),
                    "inverted source cell")
            if assemble:
                capacity[:] += np.bincount(mapped.ravel(),
                    weights=np.broadcast_to((volume/4)[:, None], mapped.shape).ravel(),
                    minlength=len(initial))
            for i, j in EDGES:
                u, v = mapped[:, i], mapped[:, j]
                squared = np.einsum("ij,ij->i", p[:, i]-p[:, j], p[:, i]-p[:, j])
                require(bool((squared > 0).all()), "collapsed edge")
                weight = volume / (6 * squared)
                if assemble:
                    row_sum[:] += np.bincount(u, weight, minlength=len(initial))
                    row_sum[:] += np.bincount(v, weight, minlength=len(initial))
                    delta = initial[u] - initial[v]
                    flux = weight * delta
                    residual[:] += np.bincount(u, flux, minlength=len(initial))
                    residual[:] -= np.bincount(v, flux, minlength=len(initial))
                    before += float(np.sum(0.5 * weight * delta * delta))
                else:
                    delta = candidate[u] - candidate[v]
                    after += float(np.sum(0.5 * weight * delta * delta))

    visit(assemble=True)
    require(bool((capacity > 0).all()) and bool((row_sum > 0).all()),
            "unowned or isolated DOF")
    step = 0.25 * float(np.min(capacity / row_sum))
    expected = initial - step * residual / capacity
    max_error = float(np.max(np.abs(candidate - expected)))
    require(max_error < 2e-13, f"{mode} independent residual/candidate parity")
    visit(assemble=False)
    initial_integral = float(np.dot(capacity, initial))
    accepted_integral = float(np.dot(capacity, candidate))
    conservation = abs(accepted_integral-initial_integral) / initial_integral
    changed = int(np.count_nonzero(candidate != initial))
    rv_only = np.setdiff1d(rv_nodes, lv_nodes)
    rv_activated = int(np.count_nonzero(candidate[base[rv_only]] > 0))
    for key, expected_value, tolerance in (
        ("diagnostic_step_seconds", step, 1e-20),
        ("initial_volume_weighted_field_m3", initial_integral, 1e-16),
        ("accepted_volume_weighted_field_m3", accepted_integral, 1e-16),
        ("relative_conservation_error", conservation, 1e-12),
        ("graph_energy_before", before, 1e-11),
        ("graph_energy_after", after, 1e-11),
        ("minimum_field", float(candidate.min()), 1e-14),
        ("maximum_field", float(candidate.max()), 1e-14),
        ("maximum_field_change", float(np.max(np.abs(candidate-initial))), 2e-13),
    ):
        close(receipt[key], expected_value, f"{mode} {key}", atol=tolerance)
    require(receipt["changed_dofs"] == changed
            and receipt["rv_exclusive_dofs_activated"] == rv_activated
            and conservation < 1e-12
            and bool((candidate >= -1e-14).all())
            and bool((candidate <= 1+1e-14).all()),
            f"{mode} counts, conservation or positivity")
    if blocked:
        require(changed == rv_activated == 0 and before == after == 0,
                "blocked-interface negative control")
    else:
        require(rv_activated > 0 and before > after > 0,
                "connected-interface positive control")
    return {"candidate_sha256": sha(field_path),
            "source_mapping_sha256": sha(mapping_path),
            "native_receipt_sha256": sha(receipt_path),
            "dofs": len(candidate),
            "independent_max_abs_field_error": max_error,
            "independent_step_seconds": step,
            "independent_relative_conservation_error": conservation,
            "independent_graph_energy_before": before,
            "independent_graph_energy_after": after,
            "independent_rv_exclusive_dofs_activated": rv_activated}


def audit(asset: Path, source: Path, run: Path) -> dict:
    gate_path = source / "independent-gate.json"
    gate = json.loads(gate_path.read_text())
    require(gate["schema"] == "numi.human.cardiac-source-activation-gate.v1"
            and gate["status"] == "passed_source_topology_and_timing_bounds_with_model_mismatch"
            and gate["asset_manifest_sha256"] == sha(asset / "manifest.json")
            and gate["exact_face_couplings"] == 4494
            and gate["point_only_contacts_kept_separate"] == [17565, 170947, 235754]
            and gate["native_accepted_electrical_steps"] == 0
            and gate["source_model_reproduction_qualified"] is False
            and gate["clinical_electrophysiology"] is False,
            "prior independent source-topology gate")
    positions, tets, labels = mesh(asset)
    lv_nodes = np.unique(tets[labels == 1])
    rv_nodes = np.unique(tets[labels == 2])
    shared = np.intersect1d(lv_nodes, rv_nodes)
    point_nodes = np.array([17565, 170947, 235754], dtype="<u4")
    face_nodes = np.setdiff1d(shared, point_nodes)
    require(len(shared) == 2631 and len(face_nodes) == 2628,
            "ventricular interface node counts")
    source_path = source / "ventricular-source-nodes.u32le"
    region_path = source / "ventricular-dof-regions.u32le"
    base_sources = np.fromfile(source_path, "<u4")
    regions = np.fromfile(region_path, "<u4")
    active = np.union1d(lv_nodes, rv_nodes)
    require(np.array_equal(base_sources, np.concatenate([active, point_nodes]))
            and np.array_equal(regions, np.concatenate([
                np.zeros(len(active), dtype="<u4"),
                np.full(3, 2, dtype="<u4")]))
            and len(active) == 218077,
            "published source-node quotient")
    connected = audit_mode("connected", run, positions, tets, labels,
                           base_sources, lv_nodes, rv_nodes, face_nodes, point_nodes)
    blocked = audit_mode("blocked", run, positions, tets, labels,
                         base_sources, lv_nodes, rv_nodes, face_nodes, point_nodes)
    replay_path = run / "connected-replay.f64le"
    replay_mapping = run / "connected-replay-nodes.u32le"
    replay_receipt = run / "connected-replay.json"
    require(sha(replay_path) == connected["candidate_sha256"]
            and sha(replay_mapping) == connected["source_mapping_sha256"]
            and replay_receipt.read_bytes() == (run / "connected.json").read_bytes(),
            "native bitwise replay")
    return {
        "schema": "numi.human.ventricular-unit-conduction-independent-audit.v1",
        "status": "passed_unit_field_native_step_and_blocked_interface_control",
        "asset_manifest_sha256": sha(asset / "manifest.json"),
        "source_activation_gate_sha256": sha(gate_path),
        "source_mapping_sha256": sha(source_path),
        "native_source_sha256": sha(ROOT / "tools/cardiac_ventricular_unit_conduction.cpp"),
        "auditor_source_sha256": sha(Path(__file__)),
        "connected": connected,
        "blocked": blocked,
        "native_bitwise_replay": True,
        "source_model_reproduction_qualified": False,
        "humanpack_accepted_electrical_steps": 0,
        "voltage_or_ionic_model": False,
        "heartbeat_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.source, args.run)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "connected_rv_exclusive_dofs_activated":
                      result["connected"]["independent_rv_exclusive_dofs_activated"],
                      "blocked_rv_exclusive_dofs_activated":
                      result["blocked"]["independent_rv_exclusive_dofs_activated"]}))


if __name__ == "__main__":
    main()
