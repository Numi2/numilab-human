"""Transfer bed-normal section loads to their exposed P1 surface quadrature at a given pose."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree


def footprint(x, faces, node_ids, height=0.0):
    """Projected area and barycentric quadrature of faces clipped by a flat z bed."""
    # Clipping against one plane produces at most a quadrilateral. Both windings
    # are retained here; visibility from the bed is tested on the full surface.
    xyz = x[faces]
    active = np.flatnonzero(xyz[:, :, 2].min(axis=1) <= height)
    entries = []
    bary3 = np.eye(3)
    rule = (np.ones((3, 3)) + 3.0 * bary3) / 6.0
    for face in active:
        points = xyz[face]
        poly = []
        for i in range(3):
            j = (i + 1) % 3
            a, b = points[[i, j], 2] - height
            if a <= 0:
                poly.append(bary3[i])
            if a * b < 0:
                t = a / (a - b)
                poly.append((1 - t) * bary3[i] + t * bary3[j])
        for k in range(1, len(poly)-1):
            bary = np.array([poly[0], poly[k], poly[k + 1]])
            xy = (bary @ points)[:, :2]
            area = abs(np.linalg.det(xy[1:] - xy[:1])) / 2
            if area <= 1e-18:
                continue
            sample = rule @ bary
            for row in sample:
                entries.append((face, area / 3., *row))
    if not entries:
        return np.empty(0, int), np.empty(0), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), int)
    a = np.asarray(entries)
    ids = a[:, 0].astype(int)
    bary = a[:, 2:]
    position = np.einsum('ni,nij->nj', bary, xyz[ids])
    return ids, a[:, 1], bary, position, node_ids[faces[ids]]


def pressure_minimax(xy, area, target, force):
    """Minimum possible peak piecewise normal stress for the specified center of pressure."""
    if not np.isfinite(force) or force <= 0 or not np.isfinite(target).all():
        raise ValueError('invalid positive normal loading')
    if not len(area):
        return None
    if not np.isfinite(xy).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('invalid material support quadrature')
    from scipy.sparse import eye, hstack, csr_matrix
    n = len(area)
    center = xy - target
    # Solve for dimensionless p relative to the uniform nominal section pressure.
    scales = np.maximum(np.max(abs(center), axis=0), 1e-5)
    moments = np.column_stack((np.ones(n), center / scales))
    weights = area / area.sum()
    fit = linprog(np.r_[np.zeros(n), 1.0],
                  A_ub=hstack((eye(n, format='csr'), csr_matrix(-np.ones((n, 1))))),
                  b_ub=np.zeros(n), A_eq=np.column_stack(((weights[:, None] * moments).T, np.zeros(3))),
                  b_eq=[1., 0., 0.], bounds=(0., None), method='highs')
    if fit.status == 2:
        return None
    if not fit.success:
        raise RuntimeError(fit.message)
    pressure = fit.x[:-1] * force / area.sum()
    loads = area * pressure
    if np.min(loads) < -1e-10 or abs(loads.sum()-force) > max(1e-8, force*1e-7) or np.max(abs(loads @ center)) > max(1e-9, force*1e-8):
        raise ValueError('section force or moment residual')
    return loads, float(pressure.max())


def visible_samples(sample, xyz, faces, hits):
    ids, area, bary, points, nodes = sample
    if not len(ids):
        return sample
    records = hits(points, xyz, faces)
    point_rows = records[:, 0].astype(int)
    face_ids = records[:, 2].astype(int)
    elevations = records[:, 1]
    # The emitting face is a known ray hit, including sub-tolerance projected
    # faces omitted by the broad-phase ray tool. Co-planar overlaps have one owner.
    lowest = points[:, 2].copy()
    np.minimum.at(lowest, point_rows, elevations)
    tolerance = 128 * np.finfo(float).eps * np.maximum(1., abs(lowest))
    exposed = points[:, 2] <= lowest + tolerance
    tied = elevations <= lowest[point_rows] + tolerance[point_rows]
    first_face = np.where(exposed, ids, np.iinfo(np.int64).max)
    np.minimum.at(first_face, point_rows[tied], face_ids[tied])
    selected = exposed & (ids == first_face)
    return tuple(a[selected] for a in sample)


def loads_on_surface(sample, sections, forces):
    ids, areas, bary, positions, nodes = sample
    support = np.flatnonzero(forces > 0)
    # The cell partition contains every original section, including inactive ones.
    owners = cKDTree(sections[:, :2]).query(positions[:, :2])[1] if len(ids) else np.empty(0, int)
    rows, samples = [], []
    for i in support:
        local = np.flatnonzero(owners == i)
        row = dict(section=int(i), normal_N=float(forces[i]), available_area_m2=float(areas[local].sum()),
                   quadrature_points=len(local), center_xy_m=sections[i, :2].tolist())
        fit = pressure_minimax(positions[local, :2], areas[local], sections[i, :2], forces[i])
        if fit is not None:
            load, row['minimum_peak_pressure_Pa'] = fit
            valid = load > 0
            used = local[valid]
            samples.append((nodes[used], bary[used], load[valid]))
        row['moment_feasible'] = fit is not None
        rows.append(row)
    return rows, samples
