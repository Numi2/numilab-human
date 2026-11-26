"""Solve the incremental posterior load path through the registered tissue and gluteal ports.

The rigid donors translate together toward the bed. Every participating tissue node may
move in that direction except for the chosen exterior contact rows. The result is a
tangent of this branch, with frozen contact membership and transverse coordinates.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from registered_cavity_materials import apply_registered_cavities
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normal_tangent(reference, current, cells, normal, shear, bulk):
    """Exact Hessian of the native log-J energy for parallel nodal variations."""
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    ds = current[cells[:, 1:]] - current[cells[:, :1]]
    volume = np.linalg.det(dm) / 6
    deformation_t = np.linalg.solve(dm, ds)
    jacobian = np.linalg.det(deformation_t)
    if not (np.all(volume > 0) and np.all(jacobian > 0)):
        raise ValueError("Invalid material orientation")
    gradients = np.linalg.inv(dm).transpose(0, 2, 1)
    gradients = np.concatenate(
        (-gradients.sum(axis=1, keepdims=True), gradients), axis=1
    )
    inv_normal = np.linalg.solve(
        deformation_t.transpose(0, 2, 1),
        np.broadcast_to(normal, (len(cells), 3))[..., None],
    )
    dlog = (gradients @ inv_normal)[..., 0]
    coef = bulk * (1 - np.log(jacobian)) + shear
    if np.any(coef < 0):
        raise ValueError("Indefinite volumetric tangent requires a different solve")
    return volume[:, None, None] * (
        shear[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
        + coef[:, None, None] * dlog[:, :, None] * dlog[:, None, :]
    )


def solve_positive(matrix, rhs, library):
    if library is None:
        return spsolve(matrix.tocsc(), rhs)
    factor = ctypes.CDLL(str(library)).solve_normal_transfer
    array = np.ctypeslib.ndpointer
    factor.argtypes = [
        ctypes.c_int,
        array(np.int64, flags="C"),
        array(np.int32, flags="C"),
        array(np.float64, flags="C"),
        array(np.float64, flags="C"),
    ]
    factor.restype = ctypes.c_int
    lower = sparse.tril(matrix, format="csc")
    lower.sort_indices()
    result = np.ascontiguousarray(rhs, dtype=np.float64).copy()
    status = factor(
        len(rhs),
        lower.indptr.astype(np.int64),
        lower.indices.astype(np.int32, copy=False),
        lower.data,
        result,
    )
    if status:
        raise ValueError(
            f"Accelerate failed to factor the material operator ({status})"
        )
    return result


def transfer(stiffness, ports, port_matrix, bed, library=None, springs=None):
    n = stiffness.shape[0]
    ports, bed = np.asarray(ports), np.asarray(bed)
    if not len(bed) or len(np.unique(ports)) != len(ports) or not len(ports):
        raise ValueError("Missing bed contact or repeated material port")
    graph = stiffness.copy()
    graph.eliminate_zeros()
    _, part = connected_components(graph, directed=False)
    # A material component attached only to a donor can follow it without strain.
    # Untouched components have no incremental load in this branch.
    springs = np.zeros(n) if springs is None else np.asarray(springs)
    supported = np.isin(part, part[np.union1d(ports, np.flatnonzero(springs))])
    free = supported.copy()
    free[bed] = False
    ii, jj = np.meshgrid(ports, ports, indexing="ij")
    coupling = sparse.coo_matrix(
        (port_matrix.ravel(), (ii.ravel(), jj.ravel())), shape=stiffness.shape
    ).tocsr()
    one = np.ones(len(ports))
    rhs = springs.copy()
    rhs[ports] += port_matrix @ one
    operator = stiffness + coupling + sparse.diags(springs)
    u = np.zeros(n)
    u[free] = solve_positive(operator[free][:, free], rhs[free], library)
    free_residual = operator @ u - rhs
    reaction = -free_residual[bed]
    donor = port_matrix @ (one - u[ports])
    spring_force = springs * (1 - u)
    total = donor.sum() + spring_force.sum()
    tissue_energy = 0.5 * (u @ (stiffness @ u))
    port_energy = 0.5 * ((one - u[ports]) @ donor + (1 - u) @ spring_force)
    scale = max(1.0, abs(total), np.abs(reaction).sum())
    errors = [
        np.max(np.abs(free_residual[free]), initial=0.0),
        abs(reaction.sum() - total),
        abs(2 * (tissue_energy + port_energy) - total),
    ]
    if not np.isfinite(u).all() or max(errors) > 1e-6 * scale:
        raise ValueError(f"Uncertified incremental transfer {errors}, scale {scale}")
    return u, dict(
        tissue_nodes=n,
        nodes_in_port_components=int(supported.sum()),
        free_normal_coordinates=int(free.sum()),
        bed_rows=len(bed),
        bed_stiffness_N_per_m=float(reaction.sum()),
        donor_stiffness_N_per_m=float(total),
        gluteal_stiffness_N_per_m=float(donor.sum()),
        cortical_and_attachment_stiffness_N_per_m=float(spring_force.sum()),
        bed_row_stiffness_N_per_m=reaction.tolist(),
        peak_free_residual_N_per_m=float(errors[0]),
        force_closure_error_N_per_m=float(errors[1]),
        work_closure_error_N_per_m=float(errors[2]),
        tissue_energy_J_per_m2=float(tissue_energy),
        port_energy_J_per_m2=float(port_energy),
    )


def material_parameters(cfg, verify):
    """Read the two constitutive laws used by the selected native executable."""
    command = json.loads(
        verify(Path(cfg["argv"][0]).parent / "build.json", False).read_bytes()
    )["commands"][0]
    params = {}
    for tag, key in [(15, "SKIN_DERMIS"), (16, "SUBCUTANEOUS_ADIPOSE")]:
        prefix = "-DNUMI_HUMAN_" + key + "_MATERIAL="
        (arg,) = (a for a in command if a.startswith(prefix))
        law = verify(arg[len(prefix) :].strip('"'), False).read_text()
        equation = re.search(r"energy\s*=\s*([^;]+);", law)[1]
        if (
            "".join(equation.split())
            != "0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2)"
        ):
            raise ValueError("Unrecognized skin or fat energy")
        params[tag] = [
            float(
                re.search(r"parameter " + k + r"\s*:\s*MPa\s*=\s*([0-9.eE+-]+)", law)[1]
            )
            * 1e6
            for k in ("shear", "bulk")
        ]
    return params


def run(case, output, library=None, proposal=None, affine_radius=None):
    if affine_radius is not None and proposal is None:
        raise ValueError("An affine gravity search requires a proposal output")
    if proposal is not None and (
        proposal.suffix != ".npz" or proposal.resolve() == output.resolve()
    ):
        raise ValueError("Use distinct JSON evidence and NPZ proposal paths")
    cfg = json.loads((case / "launch.json").read_bytes())
    env = cfg["environment"]
    hashes = {}

    def verify(path, pinned=True):
        path = Path(path)
        identity = digest(path)
        if pinned and cfg["pins"].get(str(path)) != identity:
            raise ValueError(f"Changed launch input {path}")
        hashes[str(path)] = identity
        return path

    base = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
    ref = (
        np.loadtxt(verify(env[base + "NODE"]), skiprows=1, usecols=(1, 2, 3))
        .astype("f4")
        .astype(float)
    )
    xyz = np.loadtxt(
        verify(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"]),
        skiprows=1,
        usecols=(1, 2, 3),
    )
    raw = np.loadtxt(
        verify(env[base + "ELEMENTS"]), skiprows=1, usecols=(1, 2, 3, 4, 5), dtype="i4"
    )
    cavity = np.load(verify(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"]))
    keep = cavity["retained_input_cell_rows"]
    t, label = apply_registered_cavities(raw[:, :4] - 1, raw[:, 4], cavity)
    if set(label) != {15, 16}:
        raise ValueError("Unexpected material identities")
    params = material_parameters(cfg, verify)
    log = verify(case / "stdout.log", False).read_text().splitlines()
    frame = next(s for s in log if s.startswith("resting_fem_skin_bed_source="))
    normal = np.fromstring(re.search(r" normal=(\S+)", frame)[1], sep=",")
    if normal.shape != (3,) or abs(np.linalg.norm(normal) - 1) > 1e-7:
        raise ValueError("Invalid bed normal")
    with verify(case / "physical-posterior-bed-history.csv", False).open() as stream:
        rows = list(csv.DictReader(stream))
    if not all(float(row["terminal_microstep_normal_impulse_Ns"]) > 0 for row in rows):
        raise ValueError("Support rows require positive recorded impulses")
    bed = np.array([int(row["registered_skin_node"]) for row in rows], dtype=int)
    if len(bed) != len(np.unique(bed)):
        raise ValueError("Duplicated bed contact")
    parent = (
        Path(env["NUMI_HUMAN_GLUTEAL_FAT_MASS_PREFLIGHT"]).parent
        / "material/parents.npz"
    )
    geometry = np.load(verify(parent))
    ports = geometry["fat"]
    poses = json.loads(verify(case / "native-initial-bodies.json", False).read_bytes())[
        "all_bodies"
    ]
    by_id = {p["body_index"]: p["current"] for p in poses}
    origins = np.array([by_id[int(i)]["position_world_m"] for i in geometry["donors"]])
    from scipy.spatial.transform import Rotation

    pelvis = Rotation.from_quat(
        by_id[int(geometry["donors"][0])]["orientation_world_xyzw"]
    ).as_matrix()
    if (
        np.max(abs(origins - geometry["origins"])) > 1e-7
        or np.max(abs(pelvis - geometry["pelvis_rotation"])) > 1e-7
    ):
        raise ValueError("Different gluteal frame")
    dofs = 3 * len(ports) + 6 * len(origins)
    k = np.fromfile(verify(env["NUMI_HUMAN_GLUTEAL_MATRIX"]), "<f4").reshape(dofs, dofs)
    block = k[: 3 * len(ports), : 3 * len(ports)].reshape(len(ports), 3, len(ports), 3)
    port_matrix = np.einsum("i,aibj,j->ab", normal, block.astype(float), normal)
    if np.max(abs(port_matrix - port_matrix.T)) > 1e-7 * np.max(abs(port_matrix)):
        raise ValueError("Gluteal material tangent lacks reciprocity")
    stiffness = sparse.csr_matrix((len(xyz), len(xyz)))
    for start in range(0, len(t), 65536):
        cells = t[start : start + 65536]
        tags = label[start : start + len(cells)]
        mu, bulk = np.where((tags == 15)[:, None], params[15], params[16]).T
        local = normal_tangent(ref, xyz, cells, normal, mu, bulk)
        i = np.broadcast_to(cells[:, :, None], local.shape)
        j = np.broadcast_to(cells[:, None, :], local.shape)
        stiffness += sparse.coo_matrix(
            (local.ravel(), (i.ravel(), j.ravel())), shape=stiffness.shape
        ).tocsr()
    u_glute, glute = transfer(stiffness, ports, port_matrix, bed, library)
    print("Gluteal branch", glute["donor_stiffness_N_per_m"], flush=True)
    sys_path = Path(__file__).resolve().parent.parent / "support_observation"
    import sys

    sys.path.insert(0, str(sys_path))
    from assess_supported_release import read_momentum, contact_observation

    step = read_momentum(case)[0]
    nodes, flags, areas, force, closure, pressure, _, row_hash = contact_observation(
        case, step
    )
    directions = force / (pressure * areas)[:, None]
    axial = areas * pressure / closure
    projected = axial * np.where(flags == 3, 1.0, (directions @ normal) ** 2)
    nodal = np.bincount(nodes, weights=projected, minlength=len(xyz))
    u, report = transfer(stiffness, ports, port_matrix, bed, library, nodal)
    report["gluteal_branch_alone"] = glute
    report["skin_to_skeleton_observation_sha256"] = row_hash
    report["row_counts_by_native_flag"] = {
        str(int(flag)): int(np.sum(flags == flag)) for flag in np.unique(flags)
    }
    report["skin_to_skeleton_frozen_feature_stiffness_N_per_m"] = float(projected.sum())
    report["energy_if_skin_displacement_is_zero_J_per_m2"] = float(
        (port_matrix.sum() + projected.sum()) / 2
    )
    report.update(
        input_sha256=hashes,
        launch_sha256=digest(case / "launch.json"),
        material_cells=len(t),
        bed_node_ids=bed.tolist(),
        donor_body_ids=geometry["donors"].tolist(),
        response_range=[float(u.min()), float(u.max())],
        scope="Opening material tangent, current gluteal port operator, common donor translation and the terminal positive bed rows held for an incremental calculation. Gravity is constant for this derivative. All observed skin-to-skeleton rows are included. Elastic attachments use their isotropic tangent. Triangle contact uses frozen normals, omitting feature curvature. This describes one normal-translation branch, not whole-body equilibrium.",
    )
    if proposal is not None:
        from normal_gravity_step import (
            contact_step,
            constitutive_gradient,
            certify_motion,
        )

        masses = np.zeros(len(raw))
        for tag, name in ((15, "SKIN"), (16, "FAT")):
            path = verify(
                env[
                    f"NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_{name}_CHILD_MASS_CSV"
                ]
            )
            masses[raw[:, 4] == tag] = np.loadtxt(
                path, delimiter=",", skiprows=1, usecols=2
            )
        mass = np.r_[masses[keep], cavity["replacement_mass_kg"]]
        m = np.bincount(t.ravel(), weights=np.repeat(mass / 4, 4), minlength=len(xyz))
        span = re.search(r"plane_offset_m=(\S+)", frame)
        offset = float(span[1])
        components = connected_components(stiffness, directed=False)[1]
        active_nodes = np.isin(components, components[np.union1d(ports, nodes)])
        if not np.isin(bed, np.flatnonzero(active_nodes)).all():
            raise ValueError(
                "Additional bed-only component needs independent equilibrium"
            )
        _, _, rigid_mass, _, gravity, _, _, _ = read_momentum(case)
        acceleration = -(gravity @ normal) / rigid_mass
        if not np.isfinite(m).all() or np.any(m <= 0) or not 0 < acceleration:
            raise ValueError("Invalid gravitational loading")
        # Compare the imported per-element allocations with the independent initial native mass ledger.
        mass_row = next(
            json.loads(line.split("=", 1)[1])
            for line in log
            if line.startswith("initial_gravity_component=")
            and json.loads(line.split("=", 1)[1])["object"] == 15
        )
        if abs(m.sum() - mass_row["mass_kg"]) > 1e-7:
            raise ValueError("Material mass differs from the native ledger")
        ii, jj = np.meshgrid(ports, ports, indexing="ij")
        paired = sparse.coo_matrix(
            (port_matrix.ravel(), (ii.ravel(), jj.ravel())), shape=stiffness.shape
        ).tocsr()
        paired += sparse.diags(nodal)
        transfer_to_root = np.asarray(paired.sum(axis=1)).ravel()
        branch = (stiffness + paired)[active_nodes][:, active_nodes]
        b = sparse.csr_matrix(-transfer_to_root[active_nodes, None])
        H = sparse.bmat(
            [[branch, b], [b.T, sparse.csr_matrix([[transfer_to_root.sum()]])]],
            format="csr",
        )
        grad, start_energy = constitutive_gradient(ref, xyz, t, normal, label, params)
        # Zero rigid displacement at the measured opening frame. Relative tissue displacement points down.
        rel = (xyz[ports] - geometry["reference"][: len(ports)]).ravel()
        glute_grad = -(
            (k[: len(rel), : len(rel)].astype(float) @ rel).reshape(-1, 3) @ normal
        )
        # Paired skeletal forces enter the two sides with opposite signs.
        inter = np.bincount(nodes, weights=-(force @ normal), minlength=len(xyz))
        inter[ports] += glute_grad
        gz = np.r_[
            (grad + inter - m * acceleration)[active_nodes],
            -inter.sum() - rigid_mass * acceleration,
        ]
        node_map = verify(
            env[
                "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
            ]
        )
        with node_map.open() as stream:
            O = np.array(
                [
                    int(row["joint_node_id"]) - 1
                    for row in csv.DictReader(stream)
                    if row["registered_surface_layer"] == "O"
                ]
            )
        if len(np.unique(O)) != len(O) or not np.isin(bed, O).all():
            raise ValueError("Native bed rows disagree with outer material ancestry")
        gap = xyz @ normal - offset
        if gap[O].min() <= 0:
            raise ValueError("Opening skin penetrates the bed")
        # Keep the existing positive boundary clearance during offline initialization.
        # It affects the starting pose only. No such offsets are installed in the dynamic contact owner.
        upper = np.full(len(xyz), np.inf)
        width = float.fromhex("0x1.4f8b580000000p-17")
        upper[O] = np.maximum(0.0, gap[O] - width)
        active = np.zeros(len(xyz), bool)
        active[bed] = True
        if affine_radius is None:
            q, reactions, history = contact_step(
                H,
                gz,
                np.r_[upper[active_nodes], np.inf],
                np.r_[active[active_nodes], False],
                lambda A, f: solve_positive(A, f, library),
            )
        else:
            from smooth_gravity_start import affine_contact_step, near_shape_limits

            q, reactions, history = affine_contact_step(
                H, gz, xyz[active_nodes], upper[active_nodes], affine_radius, normal=normal,
                shape=near_shape_limits(xyz,t,active_nodes,.04)
            )
        y = np.zeros(len(xyz))
        y[active_nodes] = q[:-1]
        # Components with no branch loads have no supported static solution under gravity.
        # Leave their opening state alone and identify the excluded weight.
        dy = -y[:, None] * normal
        alpha, cleared, initial_gates, witnesses = certify_motion(ref, xyz, t, dy, 1.0)
        # Evaluate the material residual at the actual finite displacement.
        grad1, end_energy = constitutive_gradient(
            ref, xyz + alpha * dy, t, normal, label, params
        )
        sub = alpha * (q[-1] - y[nodes])
        orient = directions @ normal
        f_delta = np.where(
            flags == 3,
            axial * sub,
            (np.maximum(0.0, closure + orient * sub) - closure) * axial * orient,
        )
        inter1 = inter - np.bincount(nodes, weights=f_delta, minlength=len(xyz))
        inter1[ports] += port_matrix @ (alpha * (y[ports] - q[-1]))
        full1 = np.r_[
            (grad1 + inter1 - m * acceleration)[active_nodes],
            -inter1.sum() - rigid_mass * acceleration,
        ]
        # The derivative test of the linearization uses the actual assembled energy gradient above.
        result = dict(
            loaded_mass_kg=float(rigid_mass + m[active_nodes].sum()),
            excluded_free_material_mass_kg=float(m[~active_nodes].sum()),
            normal_gravity_m_s2=acceleration,
            opening_stored_material_energy_J=start_energy,
            candidate_stored_material_energy_J=end_energy,
            quadratic_root_translation_m=float(q[-1]),
            quadratic_contact_rows=int((reactions > 1e-7).sum()),
            quadratic_bed_load_N=float(reactions.sum()),
            quadratic_required_weight_N=float(
                (rigid_mass + m[active_nodes].sum()) * acceleration
            ),
            search_iterations=history,
            applied_fraction=alpha,
            candidate_root_down_m=float(alpha * q[-1]),
            candidate_max_nodal_displacement_m=float(alpha * np.abs(y).max()),
            geometry_gate_order=[
                "inverted",
                "tiny_volume",
                "shape",
                "relative_volume",
                "principal_stretch",
            ],
            initial_material_gate_counts=initial_gates,
            cleared_material_gates=cleared,
            backtracking_witnesses=witnesses,
            opening_peak_residual_material_node=int(
                np.flatnonzero(active_nodes)[np.argmax(abs(gz[:-1]))]
            ),
            before_skeleton_normal_imbalance_N=float(gz[-1]),
            after_skeleton_normal_imbalance_N=float(full1[-1]),
            opening_max_free_normal_residual_N=float(np.max(abs(gz[:-1]))),
            candidate_max_material_normal_residual_before_bed_N=float(
                np.max(abs(full1[:-1]))
            ),
            candidate_minimum_outer_gap_m=float(np.min(gap[O] - alpha * y[O])),
            scope="Static starting-coordinate search in the bed-normal direction. Gravity acts on the imported mass, with no depth cutoff or fixed material anchors. Held skeletal joint angles, frozen cortical features, omitted other tissue paths and off-axis forces require complete native verification.",
        )
        if affine_radius is not None:
            result["search_basis"] = "affine material motion and independent skeletal translation"
            result["quadratic_bed_load_scope"] = "Reduced trial KKT multiplier. Trust-bound reactions are excluded. A binding trust limit leaves an unbalanced physical load."
        np.savez_compressed(
            proposal, node_rows=np.flatnonzero(y), delta_m=alpha * dy[y != 0]
        )
        result["proposal_sha256"] = digest(proposal)
        result["applied_to_native_scene"] = False
        report["gravity_start"] = result
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--gravity-proposal",
        type=Path,
        help="prepare a sparse starting-coordinate proposal under the measured gravitational loads",
    )
    parser.add_argument("--affine-trust-mm", type=float, help="use a bounded affine normal search for gravity initialization")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="numi-static-factor-") as tmp:
        library = Path(tmp) / "normal_transfer.dylib"
        source = Path(__file__).with_name("normal_transfer_factor.cpp")
        subprocess.run(
            [
                "xcrun",
                "clang++",
                "-std=c++17",
                "-O2",
                "-dynamiclib",
                str(source),
                "-framework",
                "Accelerate",
                "-o",
                str(library),
            ],
            check=True,
        )
        result = run(
            args.native_run.resolve(), args.output, library, args.gravity_proposal,
            None if args.affine_trust_mm is None else args.affine_trust_mm / 1000
        )
    print(result["bed_stiffness_N_per_m"], "N/m", flush=True)
