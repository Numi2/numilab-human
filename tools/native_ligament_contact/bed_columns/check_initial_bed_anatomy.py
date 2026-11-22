"""Audit the native load pose against all anatomy, verifying the executed bed columns."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree


def audit(case, reader, bank_source):
    state = json.loads((case / 'terminal.receipt.json').read_text())
    poses = {b['body_index']: b for b in state['initial_anatomical_registration']['body_poses']}
    if sorted(poses) != list(range(state['accepted_body_pose_count'])):
        raise ValueError('Initial registration does not cover the full native skeleton')
    raw = subprocess.check_output(['zstd', '-qdc', case / 'initial.mrvpack.zst'])
    identity = hashlib.sha256(raw).hexdigest()
    if identity != state['base_pack_file_sha256']:
        raise ValueError('Initial anatomy does not match the native capture')
    sys.path.insert(0, str(reader.parent / 'src'))
    module = runpy.run_path(str(reader))
    ((_, _, skin, faces),) = module['primitives'](raw, poses, skin_only=True)
    samples = np.genfromtxt(case / 'resting-local-bed-footprint.csv', names=True, delimiter=',')
    sections = np.genfromtxt(case / 'resting-local-bed-sections.csv', names=True, delimiter=',')
    distance = cKDTree(skin).query(np.column_stack([samples[k] for k in ('x_m','y_m','z_m')]))[0]
    area = np.bincount(samples['row'].astype(int), weights=samples['area_m2'], minlength=len(sections))
    if (not np.all(distance == 0) or not np.allclose(area, sections['projected_area_m2'], rtol=1e-12, atol=1e-15)
            or len(np.unique(samples['vertex'])) != len(samples)):
        raise ValueError('Load quadrature differs from the initial registered skin or executed sections')
    out = case / 'initial-envelope'
    out.mkdir(exist_ok=True)
    result = dict(scope='Native initial assembly, complete pre-step body poses, lowered support skin', physical_time_s=0,
        pose_count=len(poses), skin_vertices=len(skin), skin_faces=len(faces), bed_vertices=len(samples),
        pack_sha256=identity, maximum_bed_vertex_position_error_m=float(distance.max()))
    runpy.run_path(str(bank_source))['scan_banks'](module,raw,poses,skin,faces,result,out,'initial-envelope.json',check_skin=True)
    subprocess.run([sys.executable,bank_source.parent/'classify_registered_envelope.py',out/'initial-envelope.json',
        case/'stdout.log',out/'physical-banks.json','--window-receipt',case/'resting-ocular-sclera-window.json'],check=True)


if __name__ == '__main__':
    audit(*(Path(p).resolve() for p in sys.argv[1:]))
