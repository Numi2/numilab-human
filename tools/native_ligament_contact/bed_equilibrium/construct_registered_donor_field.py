"""Construct a supported pose on the unchanged reference mesh and audit attachment error.

A single smooth displacement field spans bonded skin and fat. This prepares a
candidate for a constitutively loaded initialization; it does not set rest positions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from screen_registered_dense_posture import load_geometry, measure


def smooth_map(x, target, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError('physical kernel width must be positive')
    cx, cy = x.mean(0), target.mean(0)
    left, _, right = np.linalg.svd((x-cx).T @ (target-cy))
    flip = np.eye(3)
    flip[-1, -1] = np.linalg.det(left @ right)
    rigid = (x-cx) @ (left @ flip @ right) + cy
    # Deposit into a shared physical grid. Trilinear splats avoid discontinuities
    # when a registration node crosses a cell boundary. Occupied cells receive
    # equal density so refinement alone cannot attract skin toward a donor.
    step = width / 2
    g = (x - x.min(0))/step + 9.
    base = np.floor(g).astype(int)
    fraction = g - base
    shape = tuple(base.max(0) + 11)
    count = np.zeros(shape)
    sums = np.zeros(shape + (3,))
    delta = target-rigid
    for off in np.ndindex(2, 2, 2):
        cells = tuple((base+off).T)
        weight = np.prod(np.where(off, fraction, 1-fraction), axis=1)
        np.add.at(count, cells, weight)
        np.add.at(sums, cells, weight[:, None] * delta)
    occupied = count > 1.e-8
    means = np.zeros_like(sums)
    means[occupied] = sums[occupied] / count[occupied, None]
    def evaluate(field):
        filtered = gaussian_filter(field, sigma=width/step, mode='constant', truncate=4.)
        return map_coordinates(filtered, g.T, order=3, mode='constant')
    denominator = evaluate(occupied.astype(float))
    if np.any(denominator <= 0):
        raise ValueError('material point outside supported grid')
    return rigid + np.column_stack([evaluate(means[..., j])/denominator for j in range(3)])


def errors(proposed, target):
    d = np.linalg.norm(proposed-target, axis=1)
    return dict(rms_m=float(np.sqrt((d*d).mean())), p95_m=float(np.quantile(d, .95)),
                p99_m=float(np.quantile(d, .99)), max_m=float(d.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('full', 'donor', 'output'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--width-m', required=True, type=float)
    a = parser.parse_args()
    x, target, owner, cells, mat, outer, _, pins = load_geometry(a.full, a.donor)
    y = smooth_map(x, target, a.width_m)
    a.output.mkdir(exist_ok=False)
    r = dict(source=pins, kernel_width_m=a.width_m, quality=measure(y, owner, cells, mat),
             per_donor_error={str(k):errors(y[owner==k], target[owner==k]) for k in np.unique(owner)},
             all_points_error=errors(y, target), outer_error=errors(y[outer],target[outer]),
             outer_below_plane=int((y[outer, 2]<0).sum()), outer_min_gap_m=float(y[outer,2].min()),
             scope='One-time displacement relative to the source reference. Donor, skeletal interface and force compatibility require independent qualification.')
    np.save(a.output/'displacement.npy', y-x)
    (a.output/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['quality'],r['all_points_error'],flush=True)


if __name__=='__main__':
    main()
