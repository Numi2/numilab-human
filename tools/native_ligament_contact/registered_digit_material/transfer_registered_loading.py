"""Transfer a verified supported displacement onto its descendant material reference."""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from fit_dorsal_material import inputs
from fit_native_bed_barrier_load import force
from fit_posterior_equilibrium import materials
from emit_constrained_material import write_coordinates
from material_context import sha
from relax_bonded_interface import loaded_mass_moments
from supported_material_admission import installed_topology, certify, stress_limits


def donor_account(tets, roles, donor, before, after, old_mass, new_mass):
    if not (len(tets) == len(roles) == len(donor) == len(old_mass) == len(new_mass)):
        raise ValueError('Mismatched source material cells')
    moved = np.flatnonzero(np.any(before != after, axis=1))
    changed = (new_mass != old_mass) | np.any(np.isin(tets, moved), axis=1)
    groups = np.column_stack((roles[changed], donor[changed]))
    rows = np.flatnonzero(changed)
    records = []
    for region in np.unique(groups, axis=0):
        ids = rows[np.all(groups == region, axis=1)]
        x = before[tets[ids]].astype(float).mean(axis=1)
        y = after[tets[ids]].astype(float).mean(axis=1)
        dm = new_mass[ids] - old_mass[ids]
        moment = dm @ x + new_mass[ids] @ (y - x)
        records.append(dict(material=int(region[0]), donor=int(region[1]), cells=len(ids),
                            mass_delta_kg=float(dm.sum()), moment_change_kg_m=moment.tolist()))
        if abs(dm.sum()) > 1.e-10 or np.max(abs(moment)) > 1.e-10:
            raise ValueError(f'Unbalanced anatomical material group {records[-1]}')
    return records


def transfer(trial, ancestor, material, out, compile_owner=False):
    x, base, t, tags, pins = inputs(trial)
    launch = json.loads((trial / 'launch.json').read_text())
    env = launch['environment']
    lineage = json.loads((material / 'material-identity.json').read_text())
    prior = json.loads((ancestor / 'supported-registration.json').read_text())
    if (sha(ancestor / 'reference.node') != prior['reference'] or sha(ancestor / 'current.node') != prior['current']
            or lineage['reference'] != pins['reference_sha256'] or prior['reference'] != lineage['native_source']['reference_sha256']
            or lineage['native_source']['elements_sha256'] != pins['elements_sha256']):
        raise ValueError('Incompatible reference, pose, or node numbering')
    def nodes(path):
        q = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2, 3))
        if not np.array_equal(q[:, 0], np.arange(1, len(x) + 1)):
            raise ValueError('Node identity changed')
        return q[:, 1:].astype('f4').astype(float)
    old_x = nodes(ancestor / 'reference.node')
    old_y = nodes(ancestor / 'current.node')
    changed_reference = np.flatnonzero(np.any(x != old_x, axis=1))
    declared = set(n for r in lineage['regions'] for n in r['rows'])
    if set(changed_reference) != declared:
        raise ValueError('Unregistered change in material reference')
    y = (old_y + (x - old_x)).astype('f4')
    cells, roles = installed_topology(t, tags, env)
    props, material_ids = materials(material / 'native-owner')
    admission = certify(x, y, cells, roles, props, stress_limits(material_ids))
    mp = Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mp) != env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256']:
        raise ValueError('Unmatched material map')
    with mp.open() as stream:
        outer = np.array([int(row['joint_node_id']) - 1 for row in csv.DictReader(stream) if row['registered_surface_layer'] == 'O'])
    f = material / 'fine-mass-routed/current-density-closed-flow.npz'
    if sha(f) != json.loads((f.parent / 'mass-transfer.json').read_text())['flow_sha256']:
        raise ValueError('Unmatched mass flow')
    with np.load(f) as data:
        mass = data['element_mass_kg']
        nodal = data['nodal_mass_kg'].copy()
    old_flow = ancestor / 'fine-mass-routed/current-density-closed-flow.npz'
    new_ledger = json.loads((f.parent / 'mass-transfer.json').read_text())
    labels = material / 'mesh/current-child.labels.csv'
    if (sha(old_flow) != new_ledger['parent_flow_sha256'] or sha(old_flow) != prior['source_mass']
            or sha(labels) != json.loads(labels.with_suffix('.json').read_text())['labels_sha256']):
        raise ValueError('Unmatched source mass or anatomical donors')
    with np.load(old_flow) as prior_flow:
        old_mass = prior_flow['element_mass_kg']
    label_rows = np.loadtxt(labels, delimiter=',', skiprows=1, usecols=(0, 1, 2), dtype='i4')
    if not np.array_equal(label_rows[:,0], np.arange(len(t))) or not np.array_equal(label_rows[:,1], tags):
        raise ValueError('Unmatched anatomical mass grouping')
    donor_moments = donor_account(t, tags, label_rows[:,2], old_y, y, old_mass, mass)
    with np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as remesh:
        deleted = np.setdiff1d(np.arange(len(t)), remesh['retained_input_cell_rows'])
        replace = remesh['replacement_tetrahedra']
        replacement_mass = remesh['replacement_mass_kg']
        moments = loaded_mass_moments(y.astype(float), t[deleted], mass[deleted], replace, replacement_mass)
        for c in range(4):
            np.add.at(nodal, t[deleted, c], -mass[deleted]/4)
            np.add.at(nodal, replace[:, c], replacement_mass/4)
    dt = float(launch['argv'][launch['argv'].index('--muscle-step-seconds')+1])
    width = float.fromhex('0x1.4f8b580000000p-17')
    def support(p):
        q = p[outer].astype(float); mask = q[:, 2] < width
        if q[:, 2].min() <= 0:
            raise ValueError('Initial bed penetration')
        fn = force(q[mask, 2], nodal[outer[mask]], dt, width)
        return outer[mask], float(fn.sum()), q[mask, :2].T @ fn, float(q[:, 2].min())
    a, old_force, old_xy, old_gap = support(base)
    b, new_force, new_xy, new_gap = support(y)
    data = (material / 'native-owner/NumiHumanLoadedSkinPoseData.inc').read_text()
    anchor_table = re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};', data, re.S)
    anchors = [int(n) for n in re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,', anchor_table[1])]
    if len(anchors) != 14 or not np.array_equal(base[anchors], y[anchors]):
        raise ValueError('Deep attachment moved')
    same_bed = bool(np.array_equal(a, b) and old_gap == new_gap and abs(old_force-new_force) <= .002 and np.max(abs(old_xy-new_xy)) <= .001)
    rows = np.flatnonzero(np.any(base != y, axis=1))
    result = dict(parent=pins, ancestral_pose=prior['current'], ancestral_reference=prior['reference'], topology=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'],
        mass=sha(f), materials=material_ids, changed_nodes=len(rows), changed_reference=changed_reference.tolist(), anchors=len(anchors),
        source_donor_moments=donor_moments, mass_moments=moments, admission=admission, before_N=old_force, after_N=new_force, support_rows=len(b), support_preserved=same_bed,
        minimum_gap_m=new_gap, bed_moment_change_Nm=(new_xy-old_xy).tolist(), passed=bool(admission['passed'] and moments['passed'] and same_bed))
    out.mkdir(exist_ok=False)
    if result['passed']:
        write_coordinates(Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']), out/'current.node', rows, y[rows])
        if not np.array_equal(nodes(out/'current.node'), y):
            raise ValueError('Coordinates changed during serialization')
        result['current'] = sha(out / 'current.node')
    (out / 'transfer.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if compile_owner and result['passed']:
        from bind_bonded_pose import compile_loaded_owner
        owner = Path(launch['argv'][0]).parent
        manifest = json.loads((owner/'identities.json').read_text())
        if manifest['binary_sha256'] != launch['pins'][str(owner/'numi-human-native')]:
            raise ValueError('Unmatched launch owner')
        manifest['installed_tissue_admission'] = admission
        compile_loaded_owner(owner, out, out/'transfer.json', manifest, pins['installed_pose_sha256'], rows,
                             footprint=(outer, y, new_gap, width))
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for name in ('trial', 'ancestor', 'material', 'out'):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument('--compile-owner', action='store_true')
    print(json.dumps(transfer(**vars(parser.parse_args())), indent=2))
