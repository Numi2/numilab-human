"""Measure the minimum elastic load any path can attain between registered material sites."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compile_connected_joint import digest
from rebind_ligament_material import banks


def read_pose(log):
    prefix = 'registered_ligament_actual_pose_json='
    rows = [line[len(prefix):] for line in log.read_text().splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('Expected one complete native pose witness')
    return json.loads(rows[0])


def endpoints(row, surfaces, pose):
    result = []
    for s in row['endpoints']:
        faces, v, _ = surfaces[s['body'], s['bone']]
        w = np.asarray(s['barycentric'])
        if np.any(w < 0) or abs(w.sum() - 1) > 1e-10:
            raise ValueError('Invalid source barycentrics')
        q = pose[s['body']]
        result.append(Rotation.from_quat(q[3:]).apply(w @ v[faces[s['triangle']]]) + q[:3])
    return np.array(result)


def tension(extension, rest, stiffness, c4, transition):
    # Native Lu toe law with C3 = stiffness * rest / (area * C4).
    x = max(extension/rest, 0.0)
    t = transition - 1.0
    return stiffness * rest * (np.expm1(c4 * min(x,t)) / c4 + np.exp(c4*t)*max(x-t,0.0))


def run(registration, log, bone, header, output):
    import re
    table = json.loads(registration.read_text())
    pose = read_pose(log)
    if table['candidate'] != digest(bone) or pose['bonePayloadSha256'] != table['candidate']:
        raise ValueError('Native witness does not match the material source')
    law = re.search(r'kC4=([\deE.+-]+),kLambdaMax=([\deE.+-]+)', header.read_text())
    c4, transition = map(float, law.groups())
    mesh = banks(bone)
    rows = []
    for name, fiber in table['fibers'].items():
        a,b = endpoints(fiber, mesh, pose['bodyPoses'])
        chord = float(np.linalg.norm(b-a))
        rest = fiber['reference_length_m']
        rows.append(dict(fiber=name, reference_m=rest, chord_m=chord,
                         lower_stretch=chord/rest, lower_force_N=float(tension(chord-rest,rest,fiber['stiffness_n_per_m'],c4,transition))))
    result = dict(registration=digest(registration), native_pose=digest(log), bone=digest(bone),
                  constitutive_source=digest(header), c4=c4, transition=transition, rows=rows)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in sorted(rows,key=lambda r:-r['lower_force_N'])[:12]:
        print(row['fiber'], round(row['lower_stretch'],4), round(row['lower_force_N'],2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration','log','bone','header','output'):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
