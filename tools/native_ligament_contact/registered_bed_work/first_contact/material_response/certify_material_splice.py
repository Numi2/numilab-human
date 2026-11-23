"""Independently reconstruct a registered skin/fat candidate and audit its emitted coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cell_metrics(points, cells):
    a, b, c = np.moveaxis(points[cells[:, 1:]] - points[cells[:, :1]], 1, 0)
    det = np.sum(a * np.cross(b, c), axis=1)
    lengths = [np.sum((points[cells[:, i]] - points[cells[:, j]]) ** 2, axis=1)
               for i in range(4) for j in range(i + 1, 4)]
    longest_squared = np.maximum.reduce(lengths)
    if np.any(longest_squared == 0):
        raise ValueError('Collapsed element at emission')
    return det / 6, det / (longest_squared * np.sqrt(longest_squared))


def certify(parent, patch_path, splice_path):
    with np.load(parent) as source, np.load(patch_path) as patch, np.load(splice_path) as splice:
        if splice['parent_sha256'].item() != digest(parent) or splice['positions_sha256'].item() != digest(patch_path):
            raise ValueError('Unexpected registered mesh or coordinate patch')
        points = source['x'].astype('f4')
        nodes = patch['nodes']
        if len(np.unique(nodes)) != len(nodes) or np.any((nodes < 0) | (nodes >= len(points))):
            raise ValueError('Invalid coordinate patch indices')
        points[nodes] = patch['x']
        points = np.vstack((points, splice['new_reference_nodes'])).astype(float)
        original = source['t']
        retired = splice['removed_cell_rows']
        if len(np.unique(retired)) != len(retired) or np.any((retired < 0) | (retired >= len(original))):
            raise ValueError('Invalid original cell ancestry')
        keep = np.ones(len(original), dtype=bool)
        keep[retired] = False
        new_cells = splice['new_cells']
        roles = splice['materials']
        if len(roles) != len(new_cells) or not np.array_equal(np.unique(roles), np.unique(source['role'])):
            raise ValueError('Unexpected replacement material identity')
        cells = np.vstack((original[keep], new_cells))
        if np.any(cells < 0) or np.any(cells >= len(points)) or not np.isfinite(points).all():
            raise ValueError('Invalid physical mesh arrays')
        if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
            raise ValueError('Duplicate occupied tetrahedron')
        used = np.bincount(cells.ravel(), minlength=len(points)) > 0
        registered = np.concatenate([source['node_' + k] for k in ('O', 'I', 'D')])
        if not used[registered].all() or not np.array_equal(np.flatnonzero(~used), splice['unused_node_rows']):
            raise ValueError('Registered surface node or remeshing ancestry mismatch')
    result = dict(cells=len(cells), retired_source_cells=len(retired), new_cells=len(new_cells),
                  inverted=0, small=0, low_quality=0, min_volume_m3=float('inf'), min_quality=float('inf'))
    for i in range(0, len(cells), 32768):
        v, q = cell_metrics(points, cells[i:i + 32768])
        result['inverted'] += int((v <= 0).sum())
        result['small'] += int((v <= 1e-18).sum())
        result['low_quality'] += int((q < 1e-5).sum())
        result['min_volume_m3'] = min(result['min_volume_m3'], float(v.min()))
        result['min_quality'] = min(result['min_quality'], float(q.min()))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'positions', 'splice'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = certify(args.parent, args.positions, args.splice)
    print(json.dumps(result, indent=2, allow_nan=False))
    if any(result[key] for key in ('inverted', 'small', 'low_quality')):
        raise SystemExit(1)
