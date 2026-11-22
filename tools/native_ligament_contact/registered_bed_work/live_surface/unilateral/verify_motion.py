"""Match a terminal capture to the rejected continuation and measure physical loads and strain."""
import json
from pathlib import Path
import re
import sys

import numpy as np
from compare_dense_support import observed
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from supported_material_admission import installed_topology, material_state, stress_limits
from verify_bonded_native import matched_horizon


def verify(root):
    accepted = root / 'two'
    rejected = root / 'three'
    def read(path):
        return json.loads(path.read_text())
    launch = read(accepted / 'launch.json')
    if matched_horizon(launch) != matched_horizon(read(rejected / 'launch.json')):
        raise ValueError('Different physics inputs in replay')
    def forces(path):
        return [s for s in (path / 'stdout.log').read_text().splitlines()
                if s.startswith('accepted_fem_body_load_capture=accepted ')]
    if forces(accepted) != forces(rejected):
        raise ValueError('Different accepted force histories')
    trace = 'native-traces/resting-coupled.csv'
    if (accepted / trace).read_bytes() != (rejected / trace).read_bytes():
        raise ValueError('Different coupled physical history')
    audit = read(accepted / 'verified-endpoint.json')
    stop = read(rejected / 'failure.json')
    if stop['segment_start'] != audit['accepted_steps']:
        raise ValueError('Different physical clock')
    frame = accepted / 'physical-accepted.node'
    if sha(frame) != audit['physical_nodes_sha256']:
        raise ValueError('Endpoint hash mismatch')
    gravity, loads = observed(accepted)
    if len(loads) != len(forces(accepted)):
        raise ValueError('Incomplete load capture')
    x, unused, elements, roles, pins = inputs(accepted)
    cells, roles = installed_topology(elements, roles, launch['environment'])
    y = np.loadtxt(frame, skiprows=1, usecols=(1, 2, 3))
    props, materials_used = materials(root / 'owner')
    caps = stress_limits(materials_used)
    ranges = {str(k): dict(cells=0, stretch_outside_envelope=0, severe=0, stress_cap=0) for k in (15, 16)}
    for first in range(0, len(cells), 32768):
        sl = slice(first, first+32768)
        t, role = cells[sl], roles[sl]
        _, sv, peak = material_state(x[t].astype(float), y[t], props[role-15])
        for m in (15, 16):
            sel = role == m
            s, p = sv[sel], peak[sel]
            r = ranges[str(m)]
            r['cells'] += int(sel.sum())
            r['stretch_outside_envelope'] += int(((s[:, 0]>1.5) | (s[:, 2]<2/3)).sum())
            r['severe'] += int(((s[:, 0]>3) | (s[:, 2]<1/3)).sum())
            r['stress_cap'] += int((p>caps[m-15]).sum())
    banks = read(accepted / 'envelope/physical-banks.json')
    motion = read(root / 'motion.json')
    return dict(material=pins, source_materials=materials_used, accepted_frame=audit['frame_sha256'],
                material_stretch=ranges, endpoint=sha(accepted / 'verified-endpoint.json'),
                crossing_banks=sha(accepted / 'envelope/physical-banks.json'), skin_bone_pairs=banks['bone_pairs'],
                captured_generalized_forces_N=loads, articulated_gravity_target_N=gravity,
                last_bed_reaction_N=motion['bed_normal_reaction_n'], total_body_weight_N=motion['weight_n'],
                accepted_force_history_identical=True, coupled_trace_sha256=sha(accepted / trace),
                rejected_common_rollback=stop['candidate_common_coordinates']==stop['accepted_common_coordinates'],
                rejected_respiratory_rollback=stop['respiratory_candidate_status']==stop['respiratory_accepted_status'])


if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
