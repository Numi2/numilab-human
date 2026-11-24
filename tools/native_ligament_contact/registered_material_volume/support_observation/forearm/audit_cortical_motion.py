"""Check closed cortical solids using the accepted native rigid poses for the 16-second benchmark."""
import argparse
import json
import sys
from pathlib import Path

import manifold3d
import numpy as np

# Resolve the published Human authoring owners for direct invocation from a clean checkout.
_tool_root = Path(__file__).resolve().parents[3]
sys.path.append(str(_tool_root / 'registered_bed_work/first_contact/material_response'))
sys.path.append(str(_tool_root.parents[1] / 'evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/supported-shoulder-motion'))
from compile_connected_joint import digest
from measure_route_motion import pose_samples
from rebind_ligament_material import banks


def cortical_solid(faces, vertices):
    vertices, remap = np.unique(vertices, axis=0, return_inverse=True)
    mesh = manifold3d.Mesh64(
        np.ascontiguousarray(vertices * 1000.0),
        np.ascontiguousarray(remap[faces], dtype=np.uint64),
    )
    solid = manifold3d.Manifold(mesh)
    if solid.status() != manifold3d.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        raise ValueError(f'Invalid closed cortical volume: {solid.status()}')
    return solid


def measure_pair(first, second, frames):
    first_owner, a = first
    second_owner, b = second
    overlaps = []
    gaps = []
    for step, pose in sorted(frames.items()):
        x, rotation = pose[first_owner]
        y, other = pose[second_owner]
        relative = np.column_stack((
            (rotation.inv() * other).as_matrix(),
            1000.0 * rotation.inv().apply(y - x),
        ))
        moved = b.transform(np.ascontiguousarray(relative))
        common = a ^ moved
        if common.status() != manifold3d.Error.NoError:
            raise ValueError(f'Solid intersection failed at accepted step {step}')
        volume = float(common.volume())
        if not np.isfinite(volume) or volume < 0:
            raise ValueError(f'Invalid intersection volume at accepted step {step}')
        if volume > 0:
            overlaps.append((step, volume))
        else:
            gap = float(a.min_gap(moved, 10.0))
            if not 0 <= gap <= 10:
                raise ValueError(f'Invalid surface distance at accepted step {step}')
            gaps.append(gap)
    return {
        'sample_count': len(frames), 'samples_with_overlap': len(overlaps),
        'first_overlap': overlaps[0] if overlaps else None,
        'worst_overlap': max(overlaps, key=lambda row: row[1]) if overlaps else None,
        'gap_range_for_nonoverlapping_samples_mm': [min(gaps), max(gaps)] if gaps else None,
        'gap_search_limit_mm': 10.0,
    }


def audit(case, selected):
    launch = case / 'launch.json'
    cfg = json.loads(launch.read_bytes())
    payload = Path(cfg['argv'][3])
    identity = digest(payload)
    if cfg['pins'].get(str(payload)) != identity:
        raise ValueError('The executed bone payload changed')
    frames = pose_samples(case)
    needed = set(bank for pair in selected for bank in pair)
    models = {}
    for (body, bank), (faces, points, _) in banks(payload).items():
        if bank in needed:
            if bank in models:
                raise ValueError(f'Ambiguous cortical bank {bank}')
            models[bank] = body, cortical_solid(faces, points)
    if set(models) != needed:
        raise ValueError('Missing cortical banks')
    rows = []
    for a, b in selected:
        if a == b:
            raise ValueError('A transition requires distinct cortical banks')
        rows.append({'banks': [a, b], 'bodies': [models[a][0], models[b][0]],
                     **measure_pair(models[a], models[b], frames)})
    return {'launch_sha256': digest(launch), 'bone_sha256': identity,
            'motion_sha256': digest(case / 'native-traces/resting-body-motion-partition.csv'),
            'overlap_unit': 'mm3', 'pairs': rows,
            'passed': all(row['samples_with_overlap'] == 0 for row in rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--pair', nargs=2, action='append', type=int, metavar=('BANK_A', 'BANK_B'))
    args = parser.parse_args()
    result = audit(args.case.resolve(), args.pair or [(10, 12), (11, 13)])
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"{sum(r['samples_with_overlap'] for r in result['pairs'])} overlapping pair-samples")
    raise SystemExit(0 if result['passed'] else 1)
