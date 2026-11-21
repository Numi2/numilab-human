"""Prepare a fresh supported scene from a matched, accepted regional contact state."""
import csv
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from material_context import sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(base, assets, donor, destination):
    target = json.loads((base / 'config.json').read_text())
    launch = json.loads((donor / 'launch.json').read_text())
    result = json.loads((donor / 'qualification.json').read_text())
    manifest = json.loads((assets / 'manifest.json').read_text())
    require(result['completed_requested_duration'] and result['accepted_steps'] > 0
            and result['regional_terminal_all_representations_clear'],
            'donor needs a completed regional run with a clear physical endpoint')
    require(target['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'] == '1'
            and target['environment']['NUMI_HUMAN_LOCAL_BED_LAYERS'] == '1',
            'expected the coupled compliant-bed scene')
    base_rigid, donor_rigid = target['argv'][1], launch['argv'][1]
    require(target['pins'][base_rigid] == launch['pins'][donor_rigid] == sha(Path(base_rigid)),
            'the donor uses a different skeleton')
    for name, digest in manifest['files'].items():
        path = assets / name
        require(launch['pins'].get(str(path)) == digest == sha(path), 'registration mismatch at ' + name)

    step = result['accepted_steps']
    posture = donor / 'stdout.log'
    state = donor / 'accepted-pelvic-state.csv'
    registry = donor / 'accepted-cortical-registry.csv'
    args = target['argv']
    require('--resting-initial-posture' not in args and '--resting-release-initialization' in args,
            'expected the authored release configuration')
    for flag, width in (('--root-pose', 8), ('--pose-q', 3)):
        while flag in args:
            start = args.index(flag)
            del args[start:start + width]
    args.extend(('--resting-initial-posture', str(posture)))

    regions = np.fromfile(assets / 'pool-material-regions.bin', '<u4').reshape(-1, 4).astype('i8')
    limit = manifest['contact_pool']['count']
    owners = np.full(limit, -1, dtype='i1')
    for start, n, slot, _ in regions:
        require(start + n <= limit and slot // 256 in (0, 1)
                and np.all(owners[start:start+n] == -1), 'invalid regional ranges')
        owners[start:start+n] = slot // 256
    require(np.all(owners >= 0), 'unassigned contact rows')
    cortical_count = sum(x['count'] for x in manifest['cortical_certificate']['banks'])
    material_count = (assets / 'closed-material-triangles.bin').stat().st_size // 48
    keys = np.zeros((512, 2), dtype='<u4')
    with registry.open() as stream:
        for row in csv.DictReader(stream):
            slot, cortical, material = (int(row[k]) for k in ('slot','cortical_row','material_face'))
            require(0 <= slot < 512 and not keys[slot, 0] and 0 <= cortical < cortical_count
                    and 0 <= material < material_count, 'invalid anatomical contact key')
            keys[slot] = cortical + 1, material + 1

    facets = manifest['finite_facets']
    per_pair = facets['rows_per_pair']
    dynamic_start = facets['start'] + per_pair * manifest['motion_contact_registry']['original_pairs']
    node_count = sum(x['nodes'] for x in manifest['contact_flexibility'])
    nodes = np.zeros(node_count, dtype=bool)
    slots = np.zeros((512, 2), dtype='<u4')
    slots[:, 0] = 0xffffffff
    counts = [0, 0]
    seen = set()
    with state.open() as stream:
        for row in csv.DictReader(stream):
            require(int(row['accepted_step']) == step, 'inconsistent accepted time')
            if row['field'] == 'node_displacement':
                node = int(row['index'])
                require(0 <= node < node_count and not nodes[node]
                        and np.isfinite([float(row[k]) for k in 'xyzw']).all()
                        and float(row['w']) == 0, 'invalid elastic node')
                nodes[node] = True
            if row['field'] != 'normal_and_force':
                continue
            query, pressure = int(row['index']), float(row['w'])
            require(0 <= query < limit and np.isfinite(pressure) and pressure >= 0, 'invalid contact load')
            if pressure == 0:
                continue
            side = int(owners[query])
            require(query not in seen and counts[side] < 256, 'pressure ownership or capacity')
            if query >= dynamic_start:
                index = (query - dynamic_start) // per_pair
                require(index < len(keys) and index // 256 == side and keys[index,0] != 0,
                        'pressure needs a registered anatomical pair')
            seen.add(query)
            slots[256*side + counts[side]] = query, struct.unpack('<I',struct.pack('<f',pressure))[0]
            counts[side] += 1
    require(np.all(nodes), 'incomplete material coordinates')

    require(not destination.exists(), 'destination exists')
    env = target['environment']
    for path in (posture, state, registry):
        target['pins'][str(path)] = sha(path)
    env.update(NUMI_HUMAN_PELVIC_INITIAL_ELASTIC_CSV=str(state),
               NUMI_HUMAN_PELVIC_INITIAL_ELASTIC_DONOR_STEP=str(step),
               NUMI_HUMAN_PELVIC_INITIAL_ELASTIC_SHA256=target['pins'][str(state)],
               NUMI_HUMAN_PELVIC_INITIAL_POSTURE_SHA256=target['pins'][str(posture)])
    target['construction_posture'] = dict(source=str(posture), sha256=target['pins'][str(posture)],
                                          prior_velocity_and_forces_imported=False)
    target['construction_contact_guess'] = dict(donor_step=step,right_positive=counts[0],left_positive=counts[1],
                                               registry_sha256=target['pins'][str(registry)],current_forces_recertified=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        temp = Path(staging) / 'prepared'
        temp.mkdir()
        for label, name, values in (('FEATURES','initial-anatomical-keys.bin',keys),
                                    ('TRACTIONS','initial-positive-rows.bin',slots)):
            values.tofile(temp / name)
            key = 'NUMI_HUMAN_PELVIC_INITIAL_' + label
            env[key] = str(destination / name)
            env[key + '_SHA256'] = target['pins'][env[key]] = sha(temp / name)
        shutil.copyfile(base / 'run_supported.py', temp / 'run_supported.py')
        (temp / 'config.json').write_text(json.dumps(target,indent=2)+'\n')
        os.rename(temp, destination)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: prepare_loaded_initialization.py BASE_CONFIG_DIR ASSETS DONOR_RUN NEW_CONFIG_DIR')
    prepare(*(Path(p).resolve() for p in sys.argv[1:]))
