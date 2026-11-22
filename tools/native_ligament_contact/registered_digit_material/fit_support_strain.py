"""Fit a statically loaded interface inside the fixed anatomical outer and deep boundaries."""

import json
import sys
from pathlib import Path

import numpy as np
from material_bed_release import balance_emitted_mass, stationary_bed
from material_context import sha
from scipy.linalg import null_space, solve
from supported_material_admission import material_state
from torso_cell_geometry import geom


class Patch:
    def __init__(self, data):
        for key in data.files:
            setattr(self, key, data[key])
        edges = self.x[self.tet[:, 1:]] - self.x[self.tet[:, :1]]
        self.volume = np.linalg.det(edges) / 6
        deriv = np.linalg.inv(edges).transpose(0, 2, 1)
        self.grad = np.concatenate((-deriv.sum(1)[:, None, :], deriv), axis=1)
        self.mu, self.bulk = self.properties[self.roles - 15].T
        self.zero = self.internal(self.y)
        bed = stationary_bed(
            self.y[self.probes, 2], self.mass[self.probes], self.dt, self.width
        )[1]
        self.baseline = self.base_elastic.copy()
        self.baseline[:, 2] -= 9.81 * self.mass
        self.baseline[self.probes, 2] += bed

    def material(self, y):
        F = self.grad[:, 1:].transpose(0, 2, 1) @ (
            y[self.tet[:, 1:]] - y[self.tet[:, :1]]
        )
        J = np.linalg.det(F)
        if np.min(J) <= 0:
            raise ValueError("Inverted material")
        return F, np.linalg.inv(F), self.bulk * np.log(J) - self.mu

    def internal(self, y):
        F, inverse, coeff = self.material(y)
        P = self.mu[:, None, None] * F + coeff[:, None, None] * inverse.transpose(
            0, 2, 1
        )
        nodal = -self.volume[:, None, None] * (self.grad @ P)
        result = np.zeros_like(y)
        np.add.at(result, self.tet, nodal)
        return result

    def residual(self, y):
        return self.baseline + self.internal(y) - self.zero

    def tangent(self, y, free):
        _F, inverse, coeff = self.material(y)
        pull = self.grad @ inverse.transpose(0, 2, 1)
        col = np.full(len(y), -1, int)
        col[free] = np.arange(len(free))
        K = np.zeros((len(y), len(free), 3, 3))
        eye = np.eye(3)
        for a in range(4):
            ga = self.grad[:, a]
            pa = pull[:, a]
            for b in range(4):
                yes = col[self.tet[:, b]] >= 0
                gb = self.grad[yes, b]
                pb = pull[yes, b]
                block = (self.mu[yes] * np.einsum("ni,ni->n", ga[yes], gb))[
                    :, None, None
                ] * eye
                block += self.bulk[yes, None, None] * pa[yes, :, None] * pb[:, None, :]
                block -= coeff[yes, None, None] * pb[:, :, None] * pa[yes, None, :]
                np.add.at(
                    K,
                    (self.tet[yes, a], col[self.tet[yes, b]]),
                    self.volume[yes, None, None] * block,
                )
        return K.transpose(0, 2, 1, 3).reshape(3 * len(y), 3 * len(free))

    def strain_metric(self, y, free):
        # Bound a coordinate step by its deformation in each incident
        # tetrahedron, including the thin elements next to the bonded interface.
        F = self.material(y)[0]
        stretch = np.linalg.svd(F, compute_uv=False)
        margin = np.maximum(
            0.008, np.minimum(1.5 - stretch[:, 0], stretch[:, -1] - 2 / 3)
        )
        cost = (60 / margin) ** 2
        col = np.full(len(y), -1, int)
        col[free] = np.arange(len(free))
        out = np.zeros((len(free), len(free)))
        for a in range(4):
            for b in range(4):
                i, j = col[self.tet[:, a]], col[self.tet[:, b]]
                yes = (i >= 0) & (j >= 0)
                np.add.at(
                    out,
                    (i[yes], j[yes]),
                    cost[yes] * np.sum(self.grad[yes, a] * self.grad[yes, b], axis=1),
                )
        return out

    def invalid(self, y):
        vol, shape = geom(y, self.tet)
        if np.any(vol <= 0) or not np.isfinite(vol).all():
            return np.flatnonzero(~np.isfinite(vol) | (vol <= 0))
        J, sv, stress = material_state(
            self.x[self.tet], y[self.tet], self.properties[self.roles - 15]
        )
        return np.flatnonzero(
            (vol <= 1e-18)
            | (shape < 1e-5)
            | (J < 0.4)
            | (J > 2.5)
            | (sv[:, -1] < 2 / 3)
            | (sv[:, 0] > 1.5)
            | (stress > self.limits[self.roles - 15])
        )

    def metrics(self, r):
        return {
            "support_z_l2_N": float(np.linalg.norm(r[self.probes, 2])),
            "nearby_l2_N": float(np.linalg.norm(r)),
            "torso_support_z_l2_N": float(
                np.linalg.norm(r[self.probes[self.donor[self.probes] == 20], 2])
            ),
            "max_abs_N": float(np.max(abs(r))),
        }


def fit(prepared, out, *captures):
    source = json.loads((prepared / "source.json").read_text())
    if sha(prepared / "material.npz") != source["prepared_sha256"]:
        raise ValueError("Different patch")
    with np.load(prepared / "material.npz") as arrays:
        p = Patch(arrays)
    y0 = p.y
    y = y0.copy()
    r = p.residual(y)
    initial = p.metrics(r)
    initial_free = p.free.copy()
    free = initial_free.copy()
    log = []
    releases = []
    if captures:
        from support_patch import motion

        base_invalid = p.invalid

        def severity(pos):
            J, s, stress = material_state(
                p.x[p.tet], pos[p.tet], p.properties[p.roles - 15]
            )
            return np.maximum.reduce(
                [
                    s[:, 0] / 1.5,
                    (2 / 3) / s[:, -1],
                    stress / p.limits[p.roles - 15],
                    0.4 / J,
                    J / 2.5,
                ]
            )

        held = np.array([], dtype=int)
        for case in captures:
            move, pins = motion(case, p.full_rows)
            level = severity(y0 + move)
            near = np.flatnonzero(level >= 0.998)
            held = np.union1d(held, np.intersect1d(free, np.unique(p.tet[near])))
            releases.append((move, np.maximum(1.0, level), pins))
        free = np.setdiff1d(free, held)

        def release_invalid(pos):
            bad = base_invalid(pos)
            if len(bad):
                return bad
            for disp, cap, _ in releases:
                try:
                    vol, quality = geom(pos + disp, p.tet)
                    tested = (
                        (severity(pos + disp) > cap + 1.0e-7)
                        | (vol <= 1e-18)
                        | (quality < 1.0e-5)
                    )
                    bad = np.union1d(bad, np.flatnonzero(tested))
                except ValueError:
                    return np.arange(len(p.tet))
            return bad

        p.invalid = release_invalid
        log.append(
            {
                "held_near_motion_limit": len(held),
                "release_pins": [entry[2] for entry in releases],
            }
        )

    baseline_tangent = p.tangent(y, free)
    gen = np.random.default_rng(42)
    v = gen.normal(size=(len(free), 3))
    delta = np.zeros_like(y)
    delta[free] = v
    h = 1.0e-9
    actual = (p.residual(y - h * delta) - p.residual(y + h * delta)).ravel() / (2 * h)
    predicted = baseline_tangent @ v.ravel()
    diff = float(np.linalg.norm(actual - predicted) / np.linalg.norm(actual))
    if diff > 2e-5:
        raise ValueError(f"Material tangent mismatch {diff}")
    print("start", initial, "directional_error", diff, flush=True)
    if len(p.invalid(y)):
        raise ValueError("Initial patch invalid")
    weights = np.ones_like(y)
    weights[p.probes, 2] = 6.0
    for iteration in range(24):
        if len(free) < 3:
            break
        K = p.tangent(y, free)
        r = p.residual(y)
        target = (weights * r).ravel()
        cols = np.searchsorted(initial_free, free)
        W = p.W[:, cols]
        W = W[np.max(abs(W), axis=1) > 0]
        W = W / np.linalg.norm(W, axis=1)[:, None]
        # The linear step and every backtrack preserve anatomical mass moments.
        N = null_space(W, rcond=1e-12)
        Z = np.kron(N, np.eye(3))
        if not Z.shape[1]:
            break
        A = weights.ravel()[:, None] * (K @ Z)
        # A scale-relative damping term regularizes poorly observed directions.
        scale = np.maximum(np.linalg.norm(A, axis=0), 1.0)
        A = A / scale
        penalty = np.kron(N.T @ p.strain_metric(y, free) @ N, np.eye(3))
        H = A.T @ A + penalty / scale[:, None] / scale[None, :]
        H.flat[:: len(H) + 1] += 0.002**2
        step = (Z @ (solve(H, A.T @ target, assume_a="pos") / scale)).reshape(-1, 3)
        delta = np.zeros_like(y)
        delta[free] = step
        dmax = np.linalg.norm(step, axis=1).max()
        if dmax < 1e-10:
            break
        delta *= min(1.0, 0.001 / dmax)
        score = float(np.linalg.norm(target))
        best = None
        failed = []
        for frac in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
        ):
            trial = y + frac * delta
            bad = p.invalid(trial)
            if len(bad):
                failed = bad
                continue
            metrics = p.metrics(p.residual(trial))
            if (
                metrics["nearby_l2_N"] > initial["nearby_l2_N"]
                or metrics["support_z_l2_N"] >= initial["support_z_l2_N"]
            ):
                continue
            E = float(np.linalg.norm(weights * p.residual(trial)))
            if E < score:
                best = (trial, E, metrics, frac)
                break
        if best is not None:
            y, score, metrics, frac = best
            log.append(
                dict(iteration=iteration, free=len(free), fraction=frac, **metrics)
            )
            print("accepted", iteration, len(free), frac, metrics, flush=True)
            if frac >= 0.0625 or not len(failed):
                continue
        if len(failed):
            # Keep already admissible coordinates. Hold only incident directions
            # that prevented a longer step, then solve the remaining coupled system.
            frozen = np.intersect1d(np.unique(p.tet[failed]), free)
            if not len(frozen):
                break
            free = np.setdiff1d(free, frozen)
            log.append(
                {
                    "iteration": iteration,
                    "limited_material_cells": p.source_cells[failed].tolist(),
                    "held_dofs": len(frozen) * 3,
                }
            )
        else:
            break
    rounded = y.astype("f4").astype(float)
    rounded[initial_free] = balance_emitted_mass(
        y0[initial_free],
        rounded[initial_free],
        p.W,
        np.full(len(p.W), 8e-11),
        np.broadcast_to(np.isin(initial_free, held)[:, None], (len(initial_free), 3))
        if captures
        else np.zeros((len(initial_free), 3), bool),
    )
    if len(p.invalid(rounded)):
        raise ValueError("Binary32 invalidated the material candidate")
    metrics = p.metrics(p.residual(rounded))
    keep = np.flatnonzero(np.any(rounded != y0, axis=1))
    if (
        not len(keep)
        or metrics["support_z_l2_N"] >= initial["support_z_l2_N"]
        or metrics["nearby_l2_N"] >= initial["nearby_l2_N"]
    ):
        raise ValueError("No admissible physical improvement")
    out.mkdir(exist_ok=False)
    np.savez_compressed(
        out / "fit.npz", rows=p.full_rows[keep], coords=rounded[keep].astype("f4")
    )
    report = {
        "source": source,
        "fit_sha256": sha(out / "fit.npz"),
        "tangent_direction_relative_error": diff,
        "initial": initial,
        "emitted": metrics,
        "changed_nodes": len(keep),
        "maximum_displacement_m": float(np.linalg.norm(rounded - y0, axis=1).max()),
        "max_material_first_moment_change_kg_m": float(
            np.max(np.abs(p.W @ (rounded[initial_free] - y0[initial_free])))
        ),
        "unchanged_outer_and_deep": bool(
            np.array_equal(
                y0[np.isin(p.layer, [1, 3])], rounded[np.isin(p.layer, [1, 3])]
            )
        ),
        "iterations": log,
    }
    (out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "emitted",
        metrics,
        report["changed_nodes"],
        report["maximum_displacement_m"],
        flush=True,
    )


if __name__ == "__main__":
    fit(*(Path(s).resolve() for s in sys.argv[1:]))
