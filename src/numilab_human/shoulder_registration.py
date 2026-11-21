"""Geometry for authoring a shared shoulder center and closing the adjacent elbow hinge.

Distances are metres, hinge angles are radians. The source shoulder and wrist are
the endpoints of an articulated two-link chain. This module does not own runtime
forces or impose target poses.
"""

import math

import numpy as np
from scipy.optimize import least_squares


def fit_articular_sphere(points, areas, seed_center, seed_radius):
    """Return the robust area-weighted articular sphere and its design condition."""
    points = np.asarray(points, dtype=float)
    areas = np.asarray(areas, dtype=float)
    center = np.asarray(seed_center, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError(
            "A sphere requires at least four three-dimensional observations"
        )
    if areas.shape != (len(points),) or center.shape != (3,):
        raise ValueError("Articular observation or center dimensions differ")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(areas).all()
        or not np.isfinite(center).all()
        or not np.isfinite(seed_radius)
        or seed_radius <= 0
        or np.any(areas <= 0)
    ):
        raise ValueError("Invalid articular observations or seed")
    weights = np.sqrt(areas / areas.mean())

    def radial_residual_mm(parameters):
        distance = np.linalg.norm(points - parameters[:3], axis=1)
        return weights * (distance - parameters[3]) * 1000

    fit = least_squares(
        radial_residual_mm,
        np.r_[center, seed_radius],
        loss="soft_l1",
        f_scale=0.4,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    if not fit.success or fit.x[3] <= 0 or np.linalg.matrix_rank(fit.jac) != 4:
        raise ValueError("Articular observations do not determine a positive sphere")
    return fit.x, float(np.linalg.cond(fit.jac))


def _periodic_angles(start, period, lower, upper):
    first = math.ceil((lower - start - 1e-12) / period)
    last = math.floor((upper - start + 1e-12) / period)
    return [
        float(np.clip(start + k * period, lower, upper)) for k in range(first, last + 1)
    ]


def solve_elbow_reach(upper_arm, forearm, hinge_axis, bounds, distance):
    """Return feasible hinge angles, reach interval and whether the wrist can close.

    For a target outside the reach interval, return the closest attainable angle.
    Shoulder orientation is free. Each segment retains its full vector, including
    the part parallel to the hinge.
    """
    upper_arm, forearm, axis = (
        np.asarray(v, dtype=float) for v in (upper_arm, forearm, hinge_axis)
    )
    if any(
        v.shape != (3,) or not np.isfinite(v).all() for v in (upper_arm, forearm, axis)
    ):
        raise ValueError("Invalid segment or hinge axis")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi, distance]).all() or lo > hi or distance < 0:
        raise ValueError("Invalid hinge bounds or target distance")
    if abs(np.linalg.norm(axis) - 1) > 1e-10:
        raise ValueError("Hinge axis must have unit length")
    upper_axial, fore_axial = upper_arm @ axis, forearm @ axis
    upper_plane = upper_arm - upper_axial * axis
    fore_plane = forearm - fore_axial * axis
    offset = upper_arm @ upper_arm + forearm @ forearm + 2 * upper_axial * fore_axial
    cosine = 2 * (upper_plane @ fore_plane)
    sine = 2 * (upper_plane @ np.cross(axis, fore_plane))
    amplitude = math.hypot(cosine, sine)
    if amplitude <= 16 * np.finfo(float).eps * max(offset, np.finfo(float).tiny):
        reach = math.sqrt(max(0.0, offset))
        return (
            [float(np.clip(0.0, lo, hi))],
            (reach, reach),
            abs(reach - distance) <= 1e-10,
        )

    def reach(angle):
        return np.sqrt(max(0.0, offset + cosine * np.cos(angle) + sine * np.sin(angle)))

    phase = np.arctan2(sine, cosine)
    extrema = [lo, hi, *_periodic_angles(phase, math.pi, lo, hi)]
    lengths = [reach(angle) for angle in extrema]
    limits = (min(lengths), max(lengths))
    closest = extrema[int(np.argmin(np.abs(np.asarray(lengths) - distance)))]
    if distance < limits[0] - 1e-10 or distance > limits[1] + 1e-10:
        return [closest], limits, False
    delta = np.arccos(np.clip((distance**2 - offset) / amplitude, -1.0, 1.0))
    angles = sorted(
        set(
            _periodic_angles(phase + delta, 2 * math.pi, lo, hi)
            + _periodic_angles(phase - delta, 2 * math.pi, lo, hi)
        )
    )
    if not angles and abs(reach(closest) - distance) <= 1e-10:
        angles = [closest]
    if not angles:
        raise ArithmeticError("Hinge reach and angle solutions disagree")
    return angles, limits, True


def preserve_socket_clearance(
    socket_points,
    head_points,
    socket_normals,
    center,
    reference_scale,
    socket_scale,
    head_scale,
):
    """Find the smallest head translation preserving source gaps along the socket normals.

    Every row is a measured pair of material points. The returned translation is
    expressed in the common reference axes, in metres. A full surface collision
    check is still needed after this local, linear registration.
    """
    from scipy.optimize import LinearConstraint, minimize

    socket, head, normals = (
        np.asarray(v, dtype=float) for v in (socket_points, head_points, socket_normals)
    )
    center = np.asarray(center, dtype=float)
    scales = np.array([reference_scale, socket_scale, head_scale])
    if (
        socket.shape != head.shape
        or socket.shape != normals.shape
        or socket.ndim != 2
        or socket.shape[1] != 3
        or not len(socket)
    ):
        raise ValueError("Socket and head material sites must match row for row")
    if (
        center.shape != (3,)
        or any(
            not np.isfinite(v).all() for v in (socket, head, normals, center, scales)
        )
        or np.any(scales <= 0)
    ):
        raise ValueError("Invalid material geometry or scale")
    if np.max(np.abs(np.linalg.norm(normals, axis=1) - 1)) > 1e-8:
        raise ValueError("Socket directions must be unit material normals")
    reference_gap = reference_scale * np.sum(normals * (head - socket), axis=1)
    centered_gap = np.sum(
        normals * (head_scale * (head - center) - socket_scale * (socket - center)),
        axis=1,
    )
    required_mm = (reference_gap - centered_gap) * 1000
    fit = minimize(
        lambda v: (0.5 * np.dot(v, v), v),
        np.zeros(3),
        jac=True,
        constraints=[LinearConstraint(normals, required_mm, np.inf)],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 120},
    )
    gap_gain_m = (normals @ fit.x - required_mm) * 0.001
    if not fit.success or gap_gain_m.min() < -1e-10:
        raise ValueError(
            "A single rigid translation cannot preserve the source socket gaps"
        )
    return fit.x * 0.001, reference_gap, gap_gain_m
