"""Recheck the complete output with the existing mass verifier and archived edge rules."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
previous = root.parent / "resting-clearance-and-atrium-016"
source = (previous / "verify_paired_density_candidate.py").read_text()
old = {
    "from paired_route_projection import PairedRouteProjection":
        "from paired_route_projection import PairedRouteProjection\n"
        "from archived_skin_compensation import ArchivedSkinCompensation",
    'candidate = HERE / "paired-density-closed-flow-candidate.npz"':
        'candidate = OUTPUT / "paired-density-closed-flow-candidate.npz"',
    'record = json.loads((HERE / "density-reserve-split-reflow.json").read_text())':
        'record = json.loads((OUTPUT / "archived-compensation-warm-reflow.json").read_text())',
    "projection = PairedRouteProjection(prior, full, ids)":
        "projection = PairedRouteProjection(prior, full, ids)\n"
        "archive = ArchivedSkinCompensation(projection, prior)\n"
        "archived_active = current['archived_compensation'] & (current['mass_flow_kg'] > 0.)\n"
        "assert current['mass_flow_kg'][archived_active].sum() <= archive.budget\n"
        "assert active_archive_count(current) == record['archived_compensation_edge_count']",
    "    assert np.all((masks[:, 1 + bit//64] >> np.uint64(bit%64)) & np.uint64(1))\n"
    "    if old < 0 or prior[\"mass_flow_kg\"][old] == 0.:":
        "    is_archived = bool(current['archived_compensation'][edge])\n"
        "    if is_archived:\n"
        "        assert len(rows) == 1 and (int(c), int(rows[0])) in archive.distance\n"
        "        assert archive.distance[(int(c), int(rows[0]))] <= archive.distance_cap\n"
        "    else:\n"
        "        assert np.all((masks[:, 1 + bit//64] >> np.uint64(bit%64)) & np.uint64(1))\n"
        "    if not is_archived and (old < 0 or prior['mass_flow_kg'][old] == 0.):",
    "    else:\n        assert np.all(prior_ids[rows] == prior[\"group\"][old])":
        "    elif not is_archived:\n        assert np.all(prior_ids[rows] == prior['group'][old])",
    'verification = HERE / "density-reserve-independent-geometry.json"':
        'verification = OUTPUT / "archived-compensation-independent-geometry.json"',
    'runner = HERE / "_verify_density_candidate_scope.py"':
        'runner = OUTPUT / "_verify_density_candidate_scope.py"',
    '    "typed_route_failure_count": 0,':
        '    "typed_route_failure_count": 0,\n'
        '    "archived_compensation_positive_edges":active_archive_count(current),\n'
        '    "archived_compensation_verified_kg":float(current["mass_flow_kg"][archived_active].sum()),\n'
        '    "archived_compensation_maximum_admitted_distance_m":archive.distance_cap,',
    '(HERE / "density-reserve-independent-check.json")':
        '(OUTPUT / "archived-compensation-independent-check.json")',
}
for before, after in old.items():
    assert source.count(before) == 1, before
    source = source.replace(before, after)
sys.path.insert(0,str(previous))

def active_archive_count(packet):
    return int((packet["archived_compensation"] & (packet["mass_flow_kg"]>0.)).sum())

exec(compile(source, str(previous/"verify_paired_density_candidate.py"), "exec"),
     {"__name__": "__main__", "OUTPUT": root, "active_archive_count": active_archive_count})
