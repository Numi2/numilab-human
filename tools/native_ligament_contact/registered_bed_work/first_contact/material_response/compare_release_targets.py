"""Compare the release projection of registered fixed attachments with actual motion."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(run, control):
    launch = json.loads((run / 'launch.json').read_bytes())
    proof = json.loads((run / 'anchor-momentum.json').read_bytes())
    endpoint = json.loads((control / 'verified-endpoint.json').read_bytes())
    if (proof['matched_native_endpoint'] != endpoint['physical_nodes_sha256'] or
            launch['parent_launch'] != digest(control / 'launch.json')):
        raise ValueError('Release target comparison requires an identical native control')
    fields = defaultdict(dict)
    log = (run / 'stdout.log').read_text()
    # One existing preparation trace emits a literal newline escape.
    records = log.replace(r'\nfirst_root_', '\nfirst_root_')
    for line in records.splitlines():
        key, _, raw = line.partition('=')
        if key in ('first_root_release_target', 'first_root_fixed_displacement'):
            node, *coords = raw.split()
            node = int(node)
            if node in fields[key]:
                raise ValueError(f'Duplicate {key} for node {node}')
            fields[key][node] = [float(x) for x in coords]
    targets = fields['first_root_release_target']
    motion = fields['first_root_fixed_displacement']
    if targets.keys() != motion.keys() or len(targets) != sum(r['nodes'] for r in proof['objects']):
        raise ValueError('Target and motion observations cover different registered attachments')
    ids = sorted(targets)
    release, actual = (np.asarray([table[n] for n in ids]) for table in (targets, motion))
    if not np.isfinite(release).all() or not np.isfinite(actual).all() or (release[:, 1] <= 0).any():
        raise ValueError('Invalid registered state')
    dt = endpoint['simulated_s']
    mass = release[:, 1, None]
    dx, v0 = release[:, 2:5], release[:, 5:8]
    moved, dv = actual[:, :3], actual[:, 3:]
    norm = lambda v: np.linalg.norm(v, axis=-1)
    fixed = -(mass * (dx / dt - v0)).sum(axis=0) / dt
    observed = -(mass * dv).sum(axis=0) / dt
    weights = mass[:, 0] / mass.sum()
    rms = lambda v: float(np.sqrt(weights @ (norm(v) ** 2)))
    root_load = json.loads(re.search(r'^accepted_fem_body_load_capture=.*?root_generalized_force_xyz_n=(\[[^\]]+\])', log, re.M)[1])
    return dict(
        launch_sha256=digest(run / 'launch.json'), parent_launch_sha256=launch['parent_launch'],
        release_observer_sha256=launch['release_observer_source'], observer_binary_sha256=launch['compiled_observer'],
        endpoint_sha256=proof['matched_native_endpoint'], metallib_sha256=digest(Path(launch['environment']['NUMI_MATTER_METALLIB_OVERRIDE'])),
        root_s=dt, fixed_nodes=len(ids), fixed_mass_kg=float(mass.sum()),
        nonzero_release_offset_count=int((norm(dx)>0).sum()), peak_initial_speed_m_s=float(norm(v0).max()),
        peak_release_m=float(norm(dx).max()), peak_accepted_travel_m=float(norm(moved).max()),
        weighted_rms_release_m=rms(dx), weighted_rms_accepted_travel_m=rms(moved),
        weighted_rms_release_minus_accepted_m=rms(dx-moved),
        release_predicted_inertial_reaction_N=fixed.tolist(),
        accepted_momentum_inertial_reaction_N=observed.tolist(),
        inertial_resultant_difference_N=float(norm(fixed-observed)),
        root_generalized_force_N=root_load,
        worst_release_node=ids[int(norm(dx).argmax())],
        worst_release_motion_difference_node=ids[int(norm(dx-moved).argmax())],
        note='Projection uses the production Metal target kernel with the native articulated operator at initial q. The root-average prediction assumes the target is held over this root. Actual includes live target updates.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('control', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.run,args.control),indent=2,allow_nan=False)+'\n')
