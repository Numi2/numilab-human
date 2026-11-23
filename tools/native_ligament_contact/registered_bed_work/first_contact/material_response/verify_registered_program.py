"""Test wrapped lengths and signed moment arms from the two emitted NHMYO2 programs."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_deltoid_route import SourceRoute, program_file, poses
from bind_registered_shoulder import sha


def verify(work, data):
    launch = json.loads((data / 'bed-unscaled-four/launch.json').read_bytes())
    parent = Path(launch['argv'][2])
    child = work / 'qualified-input' / parent.name
    info = json.loads((work / 'qualified-input/registration.json').read_bytes())
    if (sha(parent), sha(child)) != (info['parent_muscle'], info['muscle']):
        raise ValueError('Route program hashes differ')
    fit = json.loads((work / 'route-fit.json').read_bytes())
    result = {}
    rng = np.random.default_rng(20261123)
    for side in ('r', 'l'):
        programs = [program_file(p, side) for p in (parent, child)]
        seq = programs[0]['seq']
        scapula, humerus = (programs[0]['sites'][seq[i][1]][0] for i in (0, -1))
        engines = [SourceRoute(p, scapula, humerus) for p in programs]
        center = np.asarray(fit['transform'][side]['center'])
        frames = poses(data, scapula, humerus)
        ref = frames['current']
        position, rotation = ref[1]
        pivot = position + rotation.apply(center)
        for i in range(64):
            axis = rng.normal(size=3)
            angle = np.deg2rad(rng.uniform(-15, 15))
            r = rotation * Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
            frames[f'perturbation-{i}'] = [ref[0], (pivot - r.apply(center), r)]
        errors = []
        for key, frame in frames.items():
            a, b = [engine.measure(frame, center) for engine in engines]
            arm_error = np.linalg.norm(np.subtract(a['moment_arm_m'], b['moment_arm_m']))
            errors.append([abs(a['length_m'] - b['length_m']), float(arm_error)])
        maximum = np.max(errors, axis=0)
        if not np.all(np.isfinite(maximum)) or np.any(maximum > [1e-7, 1e-6]):
            raise ValueError(f'{side} failed wrapped program parity: {maximum}')
        result[side] = dict(poses=len(frames), max_length_difference_m=maximum[0],
                            max_signed_moment_arm_difference_m=maximum[1])
    result.update(parent=sha(parent), registered=sha(child), seed=20261123,
                  length_gate_m=1e-7, moment_arm_gate_m=1e-6)
    (work / 'program-parity.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    verify(*(Path(p).resolve() for p in sys.argv[1:3]))
