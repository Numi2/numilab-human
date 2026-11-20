"""Reuse paired density reflow with the validated archived receiver edges."""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
prior = here.parent / "resting-clearance-and-atrium-016"
owner = prior / "reflow_paired_mass_split_trial.py"
sys.path.insert(0, str(prior))

source = owner.read_text()
replacements = {
    "    source_region_ids = None":
        "    from archived_skin_compensation import ArchivedSkinCompensation\n"
        "    from paired_route_projection import PairedRouteProjection\n"
        "    archive = ArchivedSkinCompensation(PairedRouteProjection(mass, audit, mass['group_for_cell']), mass)\n"
        "    source_region_ids = None",
    "        if len(rows) > 1 and np.any((~admissible) & (old_flow[routes] > 0)):":
        "        if len(rows) > 1 and (np.any((~admissible) & (old_flow[routes] > 0)) or\n"
        "                              any(int(row) in archive.rows for row in rows)):",
    "        added.difference_update(zip(rebuilt_carriers, rebuilt_groups))":
        "        admitted = {(carrier, int(cell_domain[row]))\n"
        "                    for row, sources in archive.rows.items() for carrier in sources}\n"
        "        assert all(np.array_equal(supplementation.cells_of(cell_domain[row]), [row]) for row in archive.rows)\n"
        "        added.update(admitted)\n"
        "        added.difference_update(zip(rebuilt_carriers, rebuilt_groups))",
    "        proof = supplementation.verify(donor[edge], receiver[edge])":
        "        special = (int(donor[edge]), int(receiver[edge]))\n"
        "        if special in admitted:\n"
        "            return True\n"
        "        proof = supplementation.verify(donor[edge], receiver[edge])",
    "        supplier_surplus, mass_after, density_reflow = close_with_density_reserve(":
        "        archived_seed_kg = 0.\n"
        "        received = np.bincount(receiver, weights=flow, minlength=group_count)\n"
        "        seed = {key: archive.route_mass[(key[0], int(supplementation.cells_of(key[1])[0]))]\n"
        "                for key in admitted}\n"
        "        for edge, (c, d) in enumerate(zip(donor, receiver)):\n"
        "            maximum = seed.get((int(c), int(d)), 0.)\n"
        "            amount = min(maximum, supplier_surplus[c], max(0., group_lower[d] - received[d]))\n"
        "            if amount > 0.:\n"
        "                assert certify_forward(edge)\n"
        "                flow[edge] += amount\n"
        "                supplier_surplus[c] -= amount\n"
        "                received[d] += amount\n"
        "                archived_seed_kg += amount\n"
        "        print('bounded archived seed', archived_seed_kg, flush=True)\n"
        "        supplier_surplus, mass_after, density_reflow = close_with_density_reserve(",
    "    output = {":
        "    compensated = np.fromiter(((int(c), int(g)) in admitted for c, g in zip(donor, receiver)),\n"
        "                              dtype=bool, count=len(donor))\n"
        "    compensated_mass = float(flow[compensated].sum())\n"
        "    assert compensated_mass <= archive.budget + 1e-14\n"
        "    output = {\n"
        "        'archived_compensation_seed_kg': archived_seed_kg,\n"
        "        'archived_compensation_mass_kg': compensated_mass,\n"
        "        'archived_compensation_budget_kg': float(archive.budget),\n"
        "        'archived_compensation_policy_sha256': archive.policy_sha256,\n"
        "        'archived_compensation_packet_sha256': archive.packet_sha256,\n"
        "        'archived_compensation_max_current_distance_m': max(archive.distance.values()),\n"
        "        'archived_compensation_edge_count': int(np.count_nonzero(compensated & (flow > 0.))),",
    '        output_path = HERE / ("paired-density-closed-flow-candidate.npz" if density_reflow is not None else':
        '        output_path = OUTPUT / ("paired-density-closed-flow-candidate.npz" if density_reflow is not None else',
    '    (HERE / receipt).write_text(json.dumps(output, indent=2) + "\\n")':
        '    (OUTPUT / receipt).write_text(json.dumps(output, indent=2) + "\\n")',
    'nodal_mass_kg=nodal_mass, prior_edge_row=edge_source)':
        'nodal_mass_kg=nodal_mass, prior_edge_row=edge_source, archived_compensation=compensated)',
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
sys.argv = [str(owner), "--density-capacity", "--supplement", "30",
            "--wall-limit", "300", "--receipt", "archived-compensation-warm-reflow.json"]
exec(compile(source, str(owner), "exec"), {"__name__": "__main__", "OUTPUT": here})
