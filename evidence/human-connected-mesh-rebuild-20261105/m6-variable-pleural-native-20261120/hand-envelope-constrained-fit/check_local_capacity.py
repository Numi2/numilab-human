"""Bound local material mass transfer independently of per-cell mass retention."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from envelope_lp import digest, prepare, selected_sites, tet_metrics


def material_mass(env, original_role, material, name):
    key = f"NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_{name}_CHILD_MASS_CSV"
    path = Path(env[key])
    assert digest(path) == env[key + "_SHA256"]
    rows = np.flatnonzero(original_role == material)
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 2))
    assert np.array_equal(data[:, 0], np.arange(len(rows)))
    assert np.all(data[:, 1] > 0)
    return rows, data[:, 1]


def main(launch, proposal, bed, saved, report, faces):
    base, tets, role, locked = prepare(launch, proposal, bed)
    env = json.loads(launch.read_text())["environment"]
    receipt = json.loads(report.read_text())
    assert digest(saved) == receipt["proposal_sha256"]
    moved = base.copy()
    with np.load(saved, allow_pickle=False) as candidate:
        changed = candidate["node_rows"]
        moved[changed] = candidate["position_f32"]
    with np.load(proposal, allow_pickle=False) as pack:
        keep = pack["retained_input_cell_rows"].astype(np.int64)
        cavity_offsets = pack["replacement_tet_offsets"].astype(np.int64)
        parent_offsets = pack["replacement_parent_offsets"].astype(np.int64)
        parents = pack["replacement_parent_cell_rows"].astype(np.int64)
    element_key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS"
    source = Path(env[element_key])
    assert digest(source) == env[element_key + "_SHA256"]
    original = np.loadtxt(source, dtype=np.int32, skiprows=1, usecols=(1, 2, 3, 4, 5))
    original_mass = np.zeros(len(original))
    settings = [(15, "SKIN", 900., 1100.), (16, "FAT", 850., 1050.)]
    for material, name, _, _ in settings:
        idx, mass = material_mass(env, original[:, 4], material, name)
        original_mass[idx] = mass
    targets = selected_sites(faces)
    search = cKDTree(base[changed].astype(float))
    labels = np.full(len(base), -1, dtype=np.int32)
    for group in receipt["groups"]:
        if "changed_passing_nodes" not in group:
            continue
        support = []
        for target_id in group["site_rows"]:
            target = targets[target_id]
            foot = np.array(target["point_m"])
            foot[0] = target["skin_intercept_m"]
            support.extend(search.query_ball_point(foot, group["radius_mm"] * .001))
        idx = changed[np.unique(support)]
        assert np.all(labels[idx] == -1) and not np.any(locked[idx])
        assert len(idx) == group["changed_passing_nodes"]
        labels[idx] = group["group"]
    assert np.count_nonzero(labels >= 0) == len(changed)

    affected = labels[tets].max(axis=1)
    assert np.all((labels[tets] == -1) | (labels[tets] == affected[:, None]))
    cavity_group = np.maximum.reduceat(affected[len(keep):], cavity_offsets[:-1])
    for cavity in np.flatnonzero(cavity_group >= 0):
        a, b = cavity_offsets[cavity:cavity + 2]
        group = cavity_group[cavity]
        assert np.all((affected[len(keep) + a:len(keep) + b] == -1) |
                      (affected[len(keep) + a:len(keep) + b] == group))
        affected[len(keep) + a:len(keep) + b] = group

    output = []
    for group in sorted(set(affected.tolist()) - {-1}):
        packed_rows = np.flatnonzero(affected == group)
        retained = keep[packed_rows[packed_rows < len(keep)]]
        cavities = np.flatnonzero(cavity_group == group)
        cavity_sources = []
        for cavity in cavities:
            a, b = cavity_offsets[cavity:cavity + 2]
            cavity_sources.extend(np.unique(parents[parent_offsets[a]:parent_offsets[b]]))
        original_rows = np.concatenate((retained, cavity_sources)).astype(np.int64)
        assert len(np.unique(original_rows)) == len(original_rows)
        report_rows = []
        for material, name, lo, hi in settings:
            selected = packed_rows[role[packed_rows] == material]
            orig = original_rows[original[original_rows, 4] == material]
            original_mass_kg = float(original_mass[orig].sum())
            original_det, _, _ = tet_metrics(base[original[orig, :4] - 1].astype(float))
            base_det, _, _ = tet_metrics(base[tets[selected]].astype(float))
            moved_det, _, _ = tet_metrics(moved[tets[selected]].astype(float))
            base_m3 = float(base_det.sum() / 6)
            moved_m3 = float(moved_det.sum() / 6)
            assert np.isclose(float(original_det.sum() / 6), base_m3, rtol=1e-12, atol=1e-18)
            lower = moved_m3 * lo
            upper = moved_m3 * hi
            report_rows.append({
                "material": name,
                "original_cells": len(orig),
                "candidate_cells": len(selected),
                "mass_kg": original_mass_kg,
                "baseline_material_volume_m3": base_m3,
                "candidate_material_volume_m3": moved_m3,
                "baseline_mean_density_kg_m3": original_mass_kg / base_m3,
                "candidate_mean_density_kg_m3": original_mass_kg / moved_m3,
                "mass_needed_to_reach_density_floor_kg": max(0., lower - original_mass_kg),
                "mass_excess_above_density_ceiling_kg": max(0., original_mass_kg - upper),
                "aggregate_capacity_pass": lower - 1e-14 <= original_mass_kg <= upper + 1e-14,
            })
        output.append({"group": group, "complete_native_replacement_cavities": len(cavities),
                       "materials": report_rows})
    print(json.dumps({
        "scope": "Necessary aggregate material-density feasibility with closed per-group mass, including complete native replacement cavities. A pass does not prove donor, density, first-moment, or geometric admissibility. A fail requires changing the geometry or extending conservative mass transfer beyond this group.",
        "proposal_sha256": digest(saved), "groups": output,
    }, indent=2))


if __name__ == "__main__":
    main(*map(Path, sys.argv[2:]))
