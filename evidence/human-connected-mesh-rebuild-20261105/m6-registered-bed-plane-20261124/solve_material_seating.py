"""Estimate rigid seating on the 32 native bed sections under the measured gravity load."""
import csv
import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def evaluate(folder, release):
    config = json.loads((folder / 'launch.json').read_text())
    argv = config['argv']
    bed = Path(argv[argv.index('--support-contact-payload') + 1])
    header = struct.unpack_from('<8s4I32s7f', bed.read_bytes())
    origin = np.array(header[6:9])
    normal = np.array(release['normal'])
    center = next(json.loads(line.partition('=')[2])['center_m']
                  for line in (folder / 'stdout.log').read_text().splitlines()
                  if line.startswith('initial_gravity_total='))
    traces = folder / 'native-traces'
    with (traces / 'resting-support-selected-skin-witness.csv').open() as stream:
        witnesses = {}
        for row in csv.DictReader(stream):
            if int(row['pre_dynamics_step_index']) == 0 and int(row['accepted_step']) == 1:
                key = int(row['region'])
                if witnesses.setdefault(key, row) != row:
                    raise ValueError('Inconsistent release witness')
    with (traces / 'resting-local-bed-sections.csv').open() as stream:
        sections = sorted(csv.DictReader(stream), key=lambda row: int(row['contact_index']))
    if set(witnesses) != set(range(32)) or len(sections) != 32:
        raise ValueError('Incomplete material regions')
    point = np.array([[float(witnesses[i]['selected_point_world_' + axis + '_m'])
                       for axis in 'xyz'] for i in range(32)])
    c = np.array([sum(float(s[layer + '_compliance_m_per_n']) for layer in ('skin', 'fat'))
                  for s in sections])
    stock = np.array([sum(float(s[layer + '_thickness_m']) for layer in ('skin', 'fat'))
                      for s in sections])
    arm = point - center
    weight = release['required_normal_load_n']
    t0 = np.cross(normal, [0, 1, 0])
    t0 /= np.linalg.norm(t0)
    t1 = np.cross(normal, t0)

    def sample(q):
        rotated = Rotation.from_rotvec(q[1] * t0 + q[2] * t1).apply(arm)
        gap = (rotated + center - origin) @ normal - q[0]
        load = np.maximum(-gap, 0) / c
        moment = np.cross(rotated, load[:, None] * normal).sum(axis=0)
        return load, moment, np.maximum(-gap, 0) / stock

    def residual(q):
        load, moment, _ = sample(q)
        return np.array([load.sum() - weight, moment @ t0, moment @ t1]) / weight

    answers = []
    for tilt in (0, -0.012, 0.012):
        q0 = [release['translation_toward_plane_m'], tilt, 0.0]
        fit = least_squares(residual, q0, bounds=([-.04, -.1, -.1], [.04, .1, .1]),
                            gtol=1e-13, ftol=1e-13, xtol=1e-13)
        answers.append((np.linalg.norm(fit.fun), fit.x))
    _, q = min(answers, key=lambda pair: pair[0])
    load, moment, compression = sample(q)
    return dict(translation_toward_plane_m=q[0], rotation_world_vector_rad=(q[1] * t0 + q[2] * t1).tolist(),
                rotation_deg=np.degrees(np.linalg.norm(q[1:])), center_world_m=center,
                weight_normal_n=weight, load_normal_n=load.sum(), moment_world_nm=moment.tolist(),
                active_regions=int(np.count_nonzero(load)), max_stock_fraction=compression.max(),
                spread_of_solutions=float(max(np.linalg.norm(x - q) for _, x in answers)),
                regional_load_n=load.tolist(), initial_witness_sha256=release['witness'],
                scope='Original per-region GPU winners frozen at release. Rotation may change winners. Only normal support and two moments are modeled. The full native run must reselect all contact points and certify joint balance and friction.')


if __name__ == '__main__':
    release = json.loads(Path('registered-bed-release-loads.json').read_text())
    result = evaluate(Path('bed-engagement'), release)
    Path('seating-proposal.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k not in ('regional_load_n', 'scope')})
