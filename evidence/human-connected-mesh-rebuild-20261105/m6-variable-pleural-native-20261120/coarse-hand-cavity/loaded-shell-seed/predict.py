"""Predict a bed-loaded initial state in the native skin/fat energy.

This offline reduced-coordinate solve only supplies initial geometry.
The runtime retains the complete three-dimensional mechanics and bed owner.
"""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import re
from time import monotonic

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TISSUE_MATERIAL = {15: "SKIN_DERMIS", 16: "SUBCUTANEOUS_ADIPOSE"}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def ip_contact(gap, width, mass, dt):
    """Analytic IPC energy, normal load, and curvature for the seed."""
    energy = np.zeros_like(gap)
    force = np.zeros_like(gap)
    stiffness = np.zeros_like(gap)
    active = gap < width
    g = gap[active]
    width2 = width * width
    square = g * g
    diff = square - width2
    log = np.log(square / width2)
    k = 10. / np.maximum(dt * dt * width2 / mass[active], 1.e-20)
    derivative = 2. * diff * log + diff * diff / square
    energy[active] = -k * diff * diff * log
    force[active] = 2. * g * k * derivative
    stiffness[active] = -k * (2. * derivative +
        4. * square * (2. * log + 4. * diff / square - (diff / square)**2))
    return energy, force, stiffness


class SeedEnergy:
    """Exact constitutive law restricted to normal motion at a fixed lateral pose."""

    def __init__(self, world, rest, cells, mu, bulk, mass, outer_count, body_mass, center, width, dt):
        self.cells = cells
        edge = rest[cells[:, 1:]] - rest[cells[:, :1]]
        inv = np.linalg.inv(edge)
        assert np.all(np.linalg.det(edge) > 0.)
        self.volume = np.linalg.det(edge) / 6.
        # Each row of G is the material gradient of an affine shape function.
        self.g = np.concatenate((-inv.sum(axis=2)[..., None], inv), axis=2).transpose(0, 2, 1)
        edge = world[cells[:, 1:]] - world[cells[:, :1]]
        f = edge.transpose(0, 2, 1) @ inv.transpose(0, 2, 1)
        self.area = np.cross(f[:, 0], f[:, 1])
        self.z0 = f[:, 2].copy()
        self.jac = np.einsum("td,tad->ta", self.area, self.g)
        self.j0 = np.einsum("td,td->t", self.area, self.z0)
        self.mu, self.bulk = mu, bulk
        self.anatomy_count = len(world)
        self.nfree = 2 * outer_count
        self.outer = outer_count
        self.n = self.nfree + 3
        self.q = np.arange(self.nfree, self.n)
        self.weights = mass[:self.nfree] * 9.81
        self.body_weight = body_mass * 9.81
        self.grav = np.r_[self.weights, self.body_weight, 0., 0.]
        self.base_gap = world[:outer_count, 2]
        xy = world[:outer_count, :2] - center[:2]
        self.a = np.c_[np.ones(len(xy)), xy[:, 1], -xy[:, 0]]
        self.width, self.mass, self.dt = width, mass[:outer_count], dt
        self.ij = np.broadcast_arrays(cells[:, :, None], cells[:, None, :])
        self.free = (self.ij[0] < self.nfree) & (self.ij[1] < self.nfree)
        self.free_indices = tuple(arg[self.free] for arg in self.ij)
        self.elastic = self.mu[:, None, None] * np.einsum("tad,tbd->tab", self.g, self.g)
        self.vol_outer = self.jac[:, :, None] * self.jac[:, None, :]
        self.mattress = None
        self.min_j = .35 + 1.e-6
        self.max_j = 2.5 - 1.e-6

    def positions(self, v):
        u = np.r_[v[:self.nfree], np.zeros(self.anatomy_count - self.nfree)]
        gap = self.base_gap + self.a @ v[self.q] + u[:self.outer]
        jac = self.j0 + np.einsum("ta,ta->t", self.jac, u[self.cells])
        return u, gap, jac

    def contact(self, gap, penalty):
        if self.mattress is not None:
            bed_energy, force, stiffness = self.mattress(gap, self.width, self.mass, self.dt)
        elif penalty is None:
            bed_energy, force, stiffness = ip_contact(gap, self.width, self.mass, self.dt)
        else:
            overlap = np.minimum(gap, 0.)
            stiffness = penalty * (overlap < 0.)
            force = -penalty * overlap
            bed_energy = -.5 * force * overlap
        return bed_energy, force, stiffness

    def state(self, v, gravity, hessian=False, penalty=None):
        u, gap, jac = self.positions(v)
        if min(jac) <= self.min_j or max(jac) >= self.max_j:
            return None
        if penalty is None and self.mattress is None and min(gap) <= 1.e-9:
            return None
        z = self.z0 + np.einsum("tad,ta->td", self.g, u[self.cells])
        log = np.log(jac)
        energy = .5 * self.mu * (np.einsum("td,td->t", z - self.z0, z + self.z0) - 2. * log)
        energy += .5 * self.bulk * log * log
        bed_energy, force, stiffness = self.contact(gap, penalty)
        objective = float(self.volume @ energy + bed_energy.sum() + gravity * (self.grav @ v))
        c1 = (self.bulk * log - self.mu) / jac
        local = self.mu[:, None] * np.einsum("tad,td->ta", self.g, z) + c1[:, None] * self.jac
        forces = np.bincount(self.cells.ravel(), weights=(self.volume[:, None] * local).ravel(),
                            minlength=self.anatomy_count)[:self.nfree]
        derivative = np.r_[forces, -self.a.T @ force] + gravity * self.grav
        derivative[:self.outer] -= force
        if not hessian:
            return objective, derivative, force, jac
        c2 = (self.bulk * (1. - log) + self.mu) / (jac * jac)
        blocks = self.volume[:, None, None] * (self.elastic + c2[:, None, None] * self.vol_outer)
        skin = np.arange(self.outer)
        q_row, q_col = np.meshgrid(self.q, self.q, indexing="ij")
        external = self.a * stiffness[:, None]
        rows = np.r_[self.free_indices[0], skin, np.repeat(skin, 3), np.tile(self.q, self.outer),
                     q_row.ravel()]
        cols = np.r_[self.free_indices[1], skin, np.tile(self.q, self.outer), np.repeat(skin, 3),
                     q_col.ravel()]
        values = np.r_[blocks[self.free], stiffness, external.ravel(), external.ravel(),
                       (external.T @ self.a).ravel()]
        tangent = coo_matrix((values, (rows, cols)), shape=(self.n, self.n)).tocsc()
        return objective, derivative, force, jac, tangent

    def shape_rows(self, v, rows):
        columns = self.cells[rows]
        active = columns < self.nfree
        local = np.broadcast_to(np.arange(len(rows))[:, None], columns.shape)
        return coo_matrix((self.jac[rows][active], (local[active], columns[active])),
                          shape=(len(rows), self.n)).tocsr()

    def largest_fraction(self, v, direction, penalty):
        _, gap0, j0 = self.positions(v)
        _, gap1, j1 = self.positions(v + direction)

        def fraction(initial_margin, change):
            approaches = change < 0.
            return np.min(-.99 * initial_margin[approaches] / change[approaches], initial=1.)

        contact = fraction(gap0 - 1.e-9, gap1 - gap0) if penalty is None and self.mattress is None else 1.
        return min(contact,
                   fraction(j0 - self.min_j, j1 - j0),
                   fraction(self.max_j - j0, j0 - j1))


def material_parameters(source, commands, tissue):
    tag = f"-DNUMI_HUMAN_{TISSUE_MATERIAL[tissue]}_MATERIAL="
    path = Path(next(word[len(tag):].strip('"') for argv in commands for word in argv if word.startswith(tag)))
    text = path.read_text()
    fields = tuple(1.e6 * float(re.search(rf"parameter {name} : MPa = ([\d.]+)", text)[1])
                   for name in ("shear", "bulk"))
    assert "energy = 0.5 * shear * (I1() - 3 - 2 * log(J()))" in text
    assert "+ 0.5 * bulk * pow(log(J()), 2);" in text
    source[str(path)] = digest(path)
    return fields


def read_model(lateral=None):
    reference = ROOT / "thickness-rebuild-034/compiled-reference"
    path = reference / "execution.json"
    compiled = json.loads(path.read_text())
    nodes_path = reference / "shell-mass.csv.nodes.csv"
    tets_path = reference / "shell-mass.csv.tets.csv"
    for f in (nodes_path, tets_path):
        assert digest(f) == compiled[f.name]["sha256"]
    rows = read_table(nodes_path)
    cells = read_table(tets_path)
    world, rest = (np.array([[float(row[f"{label}_{axis}_m"]) for axis in "xyz"] for row in rows])
                   for label in ("current", "rest"))
    mat = np.array([int(row["material_index"]) for row in cells])
    elements = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells])
    mass = np.array([float(row["compiled_lumped_mass_kg"]) for row in rows])
    assert np.all(mass > 0.) and len(rows) % 3 == 0
    identity = {str(p): digest(p) for p in (path, nodes_path, tets_path)}
    commands_path = ROOT / "exact-shell-bone-audit-032/build-commands.json"
    commands = json.loads(commands_path.read_text())
    mu, bulk = np.array([material_parameters(identity, commands, t) for t in TISSUE_MATERIAL]).T
    indices = np.array([list(TISSUE_MATERIAL).index(t) for t in mat])
    geometry = ROOT / "terminal-support-footprint.json"
    seed = ROOT / "source-physical-bed-seed-031/prediction.json"
    pinned = json.loads(seed.read_text())
    actual = json.loads(geometry.read_text())
    identity.update({str(p): digest(p) for p in (commands_path, geometry, seed)})
    # Validate the column roles against the actual native three-layer range.
    n = len(rows) // 3
    assert np.all(elements[mat == 15] < 2*n)
    assert np.all(elements[mat == 16] >= n)
    parameters = (world, rest, elements, mu[indices], bulk[indices], mass, n,
                  actual["total_compiled_mass_kg"], np.array(actual["body_com_m"]),
                  pinned["barrier_width_m"], pinned["timestep_s"])
    if lateral is None:
        model = SeedEnergy(*parameters)
    else:
        from lateral import LateralSeedEnergy
        model = LateralSeedEnergy(*parameters, width_m=lateral)
    seed_cols = np.array([row["outer_row"] for row in pinned["support"]])
    seed_gap = np.array([row["target_gap_m"] for row in pinned["support"]])
    v = np.zeros(model.n)
    v[model.q] = np.linalg.solve(model.a[seed_cols], seed_gap - model.base_gap[seed_cols])
    assert np.min(model.positions(v)[1]) > 1.e-9
    return model, v, world, rest, elements, identity, actual


def feasible_direction(model, tangent, x, derivative, jac, guard):
    factor = splu(tangent)
    direction = factor.solve(-derivative)
    if not guard:
        return direction, 0
    limit_rows = np.where((jac - model.min_j < .005) | (model.max_j - jac < .005))[0]
    if not len(limit_rows):
        return direction, 0
    sign = np.where(jac[limit_rows] < 1., 1., -1.)
    gradients = model.shape_rows(x, limit_rows).multiply(sign[:, None]).tocsr()
    # Guide only the search direction. The success residual still uses
    # unmodified tissue and bed forces with no determinant reaction.
    margin = 0.
    constrained = np.zeros(len(limit_rows), dtype=bool)
    for _ in range(5):
        violated = (gradients @ direction < margin) & ~constrained
        if not np.any(violated):
            break
        constrained |= violated
        if constrained.sum() > 64:
            return direction, int(constrained.sum())
        rows = gradients[constrained].toarray()
        responses = factor.solve(rows.T)
        correction = np.linalg.lstsq(rows @ responses, margin - rows @ direction, rcond=1.e-12)[0]
        direction = direction + responses @ correction
    return direction, int(constrained.sum())


def solve(model, v, count, weights=(.05, .15, .35, .65, 1.), penalty=None, guard=False):
    trace = []
    for weight in weights:
        for step in range(count):
            state = model.state(v, weight, hessian=True, penalty=penalty)
            objective, derivative, force, jac, tangent = state
            row = {"load_fraction": weight, "iteration": step,
                   "energy_j": objective, "max_residual": float(np.abs(derivative).max()),
                   "normal_n": float(force.sum()), "active_outer": int((force > 0.).sum()),
                   "minimum_j": float(jac.min()), "maximum_j": float(jac.max())}
            trace.append(row)
            if row["max_residual"] < 1.e-4:
                row["outcome"] = "converged"
                break
            try:
                direction, guards = feasible_direction(model, tangent, v, derivative, jac, guard)
            except RuntimeError:
                row["outcome"] = "singular_tangent"
                return v, trace
            if guards:
                row["active_shape_search_rows"] = guards
            rate = derivative @ direction
            if not np.isfinite(direction).all() or rate >= 0.:
                row["outcome"] = "invalid_direction"
                return v, trace
            step_fraction = model.largest_fraction(v, direction, penalty)
            if step_fraction <= 1.e-12:
                row["outcome"] = "geometry_limited"
                return v, trace
            for _ in range(24):
                candidate = model.state(v + step_fraction * direction, weight, penalty=penalty)
                if candidate is not None and candidate[0] <= objective + .0001 * step_fraction * rate:
                    v += step_fraction * direction
                    row["step_fraction"] = step_fraction
                    break
                step_fraction *= .5
            else:
                row["outcome"] = "line_search_exit"
                return v, trace
        else:
            trace[-1]["outcome"] = "budget"
            return v, trace
    return v, trace


def match_ipc_seed(model, v, penalty):
    """Convert numerical penalty reactions to positive gaps of the original IPC law."""
    gap = model.positions(v)[1]
    target = -penalty * np.minimum(gap, 0.)
    left = np.full_like(gap, 1.001e-9)
    right = np.full_like(gap, model.width)
    for _ in range(48):
        half = .5 * (left + right)
        _, actual, _ = ip_contact(half, model.width, model.mass, model.dt)
        increase = actual > target
        left = np.where(increase, half, left)
        right = np.where(increase, right, half)
    v = v.copy()
    v[:model.outer] += np.where(target > 0., .5 * (left + right) - gap,
                                np.maximum(0., model.width - gap))
    return v


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-iterations", type=int, default=160)
    parser.add_argument("--warm", type=float, help="temporary offline penalty stiffness in N/m")
    parser.add_argument("--lateral", type=float, help="admit x/y motion in a bed-adjacent strip, in metres")
    parser.add_argument("--loads", type=float, nargs="+", default=(.05, .15, .35, .65, 1.))
    parser.add_argument("--redirect", action="store_true", help="redirect Newton trials at active determinant floors")
    parser.add_argument("--mattress-kpa", type=float, help="elastic modulus of the flat mattress")
    parser.add_argument("--mattress-thickness", type=float, default=.12)
    args = parser.parse_args()
    if args.mattress_kpa is not None and (args.mattress_kpa <= 0. or args.mattress_thickness <= 0. or args.warm):
        parser.error("the mattress requires positive dimensions and its own exact contact path")
    if not (all(0. < value <= 1. for value in args.loads) and
            list(args.loads) == sorted(set(args.loads))):
        parser.error("loads must be strictly increasing fractions in (0, 1]")
    if args.lateral is not None and args.lateral <= 0.:
        parser.error("the lateral strip must be positive")
    if args.max_iterations <= 0:
        parser.error("the iteration budget must be positive")
    if args.warm is not None and args.warm <= 0.:
        parser.error("the warm-start stiffness must be positive")
    args.output.mkdir()
    start = monotonic()
    model, v, world, rest, cells, pins, actual = read_model(args.lateral)
    if args.mattress_kpa is not None:
        from mattress import Mattress, projected_area
        model.mattress = Mattress(projected_area(world, cells, model.outer),
                                  1.e3 * args.mattress_kpa, args.mattress_thickness)
        # Seat the first few rows in the smooth range of the numerical
        # conditioning. The mattress response itself is never changed.
        v[model.q[0]] -= .001
    original = model.state(v, 1.)
    warm = []
    if args.warm is not None:
        guess = v.copy()
        guess[model.q[0]] -= 2. * model.width
        guess, warm = solve(model, guess, args.max_iterations, weights=args.loads, penalty=args.warm, guard=args.redirect)
        (args.output / "warm-trace.json").write_text(json.dumps(warm, indent=2, allow_nan=False) + "\n")
        projected = match_ipc_seed(model, guess, args.warm)
        if model.state(projected, 1.) is not None:
            v = projected
        else:
            raise ValueError("the warm solution could not be mapped inside the exact IPC and cell guards")
    weights = (args.loads[-1],) if warm else args.loads
    v, trace = solve(model, v, args.max_iterations, weights=weights, guard=args.redirect)
    objective, residual, force, jac = model.state(v, 1.)
    u, gap, _ = model.positions(v)
    deformation = model.full_positions(v)[0] if args.lateral is not None else np.c_[np.zeros((len(u), 2)), u]
    np.savez_compressed(args.output / "seed.npz", displacement_m=deformation, rigid_seed=v[model.q],
                        nominal_gap_m=gap, force_n=force, deformation_j=jac)
    result = {
        "scope": "Offline nonlinear preload design with exact native material energy, normal displacement and a fixed deep boundary relative to the floating body. The reduced boundary and degrees of freedom are only used to design initialization.",
        "asset_pins": pins, "converged_at_full_gravity": bool(np.max(np.abs(residual)) < 1.e-4),
        "physical_weight_n": model.body_weight, "maximum_initial_residual": float(np.abs(original[1]).max()),
        "maximum_full_gravity_residual": float(np.abs(residual).max()), "native_reference_mass_kg": actual["total_compiled_mass_kg"],
        "root_vertical_pitch_about_x_roll_about_y": v[model.q].tolist(),
        "full_gravity_nodal_residual_max_n": float(np.abs(residual[:-3]).max()),
        "full_gravity_vertical_residual_n": float(residual[model.q[0]]),
        "full_gravity_moment_residual_nm": residual[model.q[1:]].tolist(), "normal_support_n": float(force.sum()),
        "support_moments_nm": (model.a[:, 1:].T @ force).tolist(),
        "active_outer": int(np.count_nonzero(force)), "active_source_rows": np.flatnonzero(force).tolist(),
        "minimum_gap_m": float(gap.min()), "maximum_skin_normal_displacement_m": float(np.max(np.abs(u))),
        "maximum_skin_displacement_m": float(np.linalg.norm(deformation, axis=1).max()),
        "minimum_j": float(jac.min()), "maximum_j": float(jac.max()),
        "seconds": monotonic() - start, "iterations": len(trace),
        "requested_loads": list(args.loads), "determinant_redirect": args.redirect,
        "exact_converged_loads": [row["load_fraction"] for row in trace if row.get("outcome") == "converged"],
        "warm_converged_loads": [row["load_fraction"] for row in warm if row.get("outcome") == "converged"],
        "lateral_strip_m": args.lateral, "lateral_skin_nodes": len(model.lateral) if args.lateral is not None else 0,
        "numerical_warm_start_stiffness_n_m": args.warm, "warm_iterations": len(warm),
    }
    if model.mattress is not None:
        mattress = model.mattress
        actual_gap, active = mattress.clearance_for_load(gap, model.width, model.mass, model.dt)
        indentation = force[active] / mattress.k[active]
        pressure = force[active] / mattress.area[active]
        result["mattress"] = {"modulus_pa": mattress.modulus_pa, "thickness_m": mattress.thickness,
                              "minimum_active_ipc_gap_m": float(actual_gap[active].min()),
                              "max_indentation_m": float(indentation.max()),
                              "max_pressure_pa": float(pressure.max()),
                              "projected_active_area_m2": float(mattress.area[active].sum()),
                              "maximum_series_equilibrium_error_m": float(abs(actual_gap[active] - indentation - gap[active]).max())}
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("asset_pins", "active_source_rows")}), flush=True)


if __name__ == "__main__":
    main()
