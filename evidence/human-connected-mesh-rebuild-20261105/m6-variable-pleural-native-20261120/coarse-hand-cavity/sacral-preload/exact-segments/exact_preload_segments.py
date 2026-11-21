"""Guard offline three-dimensional preload trials over the full search segment."""
import numpy as np

from lateral import LateralSeedEnergy


def determinant_coefficients(f, delta):
    def triple(x, y, z):
        return np.einsum("ti,ti->t", x, np.cross(y, z))

    x, y, z = np.moveaxis(f, 1, 0)
    dx, dy, dz = np.moveaxis(delta, 1, 0)
    return np.stack((
        triple(x, y, z),
        triple(dx, y, z) + triple(x, dy, z) + triple(x, y, dz),
        triple(dx, dy, z) + triple(dx, y, dz) + triple(x, dy, dz),
        triple(dx, dy, dz),
    ), axis=1)


def first_volume_limit(coeff, lower, upper):
    """Return the first parameter in [0, 1] meeting either volume bound."""
    d, c, b, a = coeff.T
    quadratic = 3. * a
    linear = 2. * b
    # Normalize derivatives before forming the discriminant.
    scale = np.maximum.reduce((np.abs(quadratic), np.abs(linear), np.abs(c)))
    scale[scale == 0.] = 1.
    qa, qb, qc = quadratic / scale, linear / scale, c / scale
    discriminant = qb * qb - 4. * qa * qc
    q = -.5 * (qb + np.copysign(np.sqrt(np.maximum(discriminant, 0.)), qb))
    first = np.divide(q, qa, out=np.ones_like(q), where=qa != 0.)
    second = np.divide(qc, q, out=np.ones_like(q), where=q != 0.)
    first[discriminant < 0.] = 1.
    second[(discriminant < 0.) | ((qa == 0.) & (qb == 0.))] = 1.
    first[(first <= 0.) | (first >= 1.) | (qa == 0.)] = 1.
    second[(second <= 0.) | (second >= 1.)] = 1.
    points = np.stack((np.minimum(first, second), np.maximum(first, second),
                       np.ones_like(first)), axis=1)
    values = ((a[:, None] * points + b[:, None]) * points + c[:, None]) * points + d[:, None]
    unsafe = (values <= lower) | (values >= upper)
    selected = np.flatnonzero(np.any(unsafe, axis=1))
    if selected.size == 0:
        return 1.
    index = np.argmax(unsafe[selected], axis=1)
    right = points[selected, index].copy()
    left = np.where(index > 0, points[selected, index - 1], 0.)
    d, c, b, a = coeff[selected].T
    bound = np.where(values[selected, index] <= lower, lower, upper)
    sign = np.where(bound == lower, 1., -1.)
    for _ in range(44):
        mid = .5 * (left + right)
        margin = (((a * mid + b) * mid + c) * mid + d - bound) * sign
        safe = margin > 0.
        left = np.where(safe, mid, left)
        right = np.where(safe, right, mid)
    return float(left.min(initial=1.))


class ExactSegmentSeedEnergy(LateralSeedEnergy):
    def largest_fraction(self, v, direction, penalty):
        _, clearance, f = self.full_positions(v)
        node_displacement = np.r_[direction, 0.][self.id]
        df = np.einsum("tai,taj->tij", node_displacement[self.cells], self.g)
        coefficients = determinant_coefficients(f, df)
        shape_step = first_volume_limit(coefficients, self.min_j, self.max_j)
        if shape_step < 1.:
            shape_step *= .99
        if penalty is not None or self.mattress is not None:
            return shape_step
        gap_change = self.a @ direction[self.q] + node_displacement[:self.outer, 2]
        approaches = gap_change < 0.
        return min(shape_step, float(np.min(
            -.99 * (clearance[approaches] - 1.e-9) / gap_change[approaches], initial=1.)))
