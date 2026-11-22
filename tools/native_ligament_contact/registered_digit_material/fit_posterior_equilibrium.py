"""Fit the posterior material starting coordinates against the source-defined passive elastic potential and gravity."""

from pathlib import Path
import json, re, sys
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from material_context import sha
from emit_constrained_material import write_coordinates


def materials(owner):
    args = json.loads((owner / "identities.json").read_text())["commands"][0]
    rows = []
    pins = {}
    for macro in (
        "NUMI_HUMAN_SKIN_DERMIS_MATERIAL",
        "NUMI_HUMAN_SUBCUTANEOUS_ADIPOSE_MATERIAL",
    ):
        paths = [
            Path(v.split("=", 1)[1].strip('"'))
            for v in args
            if v.startswith("-D" + macro + "=")
        ]
        if len(paths) != 1:
            raise ValueError("Material identity is ambiguous")
        path = paths[0]
        pins[str(path)] = sha(path)
        source = re.sub(r"//[^\n]*", "", path.read_text())
        expression = "".join(source.partition("energy =")[2].partition(";")[0].split())
        if expression != "0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2)":
            raise ValueError("Unsupported native constitutive energy")
        rows.append(
            [
                float(
                    re.search(
                        r"parameter\s+" + p + r"\s*:\s*MPa\s*=\s*([\d.eE+-]+)", source
                    )[1]
                )
                * 1e6
                for p in ("shear", "bulk")
            ]
        )
    return np.array(rows), pins


def run(native, case, layers, bed, out, mode, guard=None, source_fit=None, screen=None):
    out.mkdir(exist_ok=False)
    ref, initial, t, tag, pins = inputs(native)
    ref = ref.astype(float)
    initial = initial.astype(float)
    w = json.loads((bed / "load.json").read_text())
    anchors = json.loads((bed / "loaded-registration.json").read_text())[
        "attachment_rows"
    ]
    footprint = np.array(w["nodes"])
    O = np.load(layers)["Orows"]
    env = json.loads((native / "launch.json").read_text())["environment"]
    masspath = case / "fine-mass-routed/current-density-closed-flow.npz"
    mass_report = json.loads((case / "fine-mass-routed/mass-transfer.json").read_text())
    if sha(masspath) != mass_report["flow_sha256"]:
        raise ValueError("Mass mismatch")
    mass = np.load(masspath)["nodal_mass_kg"]
    if abs(mass.sum() - mass_report["total_mass_kg"]) > 1e-9:
        raise ValueError("Nodal mass differs from transferred mass")
    if set(O[initial[O, 2] < w["band_m"]]) != set(footprint):
        raise ValueError("Different physical bed support")
    free = (cKDTree(initial[footprint, :2]).query(initial[:, :2])[0] < 0.08) & (
        initial[:, 2] < 0.075
    )
    fixed = (
        O
        if mode in ("fixed-outer", "constrained-outer")
        else np.union1d(footprint, O[initial[O, 2] < 2 * w["band_m"]])
    )
    if mode not in ("fixed-outer", "free-outer", "constrained-outer"):
        raise ValueError("Unknown fitting domain")
    protected = []
    if guard:
        evidence = json.loads(guard.read_text())
        prior = json.loads((guard.parent / "fit.json").read_text())
        if (
            prior["inputs"] != pins
            or evidence["inputs"]["CONSTRUCTED_NODE_SHA256"] != prior["output_pose"]
            or sha(guard.parent / "current.node") != prior["output_pose"]
        ):
            raise ValueError("Clearance witness is from a different source")
        protected = sorted(
            {
                int(i)
                for group in evidence["witness_node_rows"].values()
                for pair in group
                for triangle in pair
                for i in triangle
            }
        )
        if not protected:
            raise ValueError("No local clearance witnesses in audit")
        free[protected] = False
    free[fixed] = False
    free[[row["volume_row"] for row in anchors]] = False
    active = np.flatnonzero(free)
    mask = free[t].any(1)
    cells = t[mask]
    local = np.unique(cells)
    sel = np.searchsorted(local, cells)
    x = ref[local]
    start = initial[local]
    af = np.searchsorted(local, active)
    idof = np.full(len(local), -1)
    idof[af] = np.arange(len(active))
    dofs = (3 * idof[sel][:, :, None] + np.arange(3)).reshape(-1, 12)
    keep = np.broadcast_to((idof[sel] >= 0)[:, :, None], (len(sel), 4, 3)).reshape(
        -1, 12
    )
    ix, iy, iz = np.nonzero(keep[:, :, None] & keep[:, None, :])
    rr = dofs[ix, iy]
    cc = dofs[ix, iz]
    rest = x[sel[:, 1:]] - x[sel[:, :1]]
    volume = np.linalg.det(rest) / 6
    G = np.linalg.inv(rest).transpose(0, 2, 1)
    G = np.concatenate((-G.sum(1, keepdims=True), G), axis=1)
    if (volume <= 0).any() or set(np.unique(tag[mask])) != {15, 16}:
        raise ValueError("Invalid mixed mesh")
    props, material_ids = materials(case / "native-owner")
    mu, K = props[tag[mask] - 15].T
    eye = np.eye(3)

    def deformation(y):
        return np.einsum("cni,cnj->cij", y[sel[:, 1:]] - y[sel[:, :1]], G[:, 1:, :])

    def energy(y):
        F = deformation(y)
        J = np.linalg.det(F)
        if np.any(J <= 0.35) or np.any(J > 2.5):
            return np.inf
        l = np.log(J)
        I1 = np.einsum("cij,cij->c", F, F)
        W = 0.5 * mu * (I1 - 3 - 2 * l) + 0.5 * K * l * l
        return float(W @ volume + 9.81 * (y[af, 2] - start[af, 2]) @ mass[active])

    def system(y, with_hessian):
        F = deformation(y)
        J = np.linalg.det(F)
        l = np.log(J)
        N = np.linalg.inv(F).transpose(0, 2, 1)
        v = np.einsum("cij,cnj->cni", N, G)
        fg = np.einsum("cij,cnj->cni", F, G)
        forces = volume[:, None, None] * (
            mu[:, None, None] * fg + (K * l - mu)[:, None, None] * v
        )
        g = np.zeros_like(y)
        np.add.at(g, sel, forces)
        g[af, 2] += mass[active] * 9.81
        if not with_hessian:
            return g[af].ravel()
        gram = G @ G.transpose(0, 2, 1)
        H = (
            np.einsum("c,cnm,ij->cnimj", volume * mu, gram, eye)
            + np.einsum("c,cni,cmj->cnimj", volume * K, v, v)
            - np.einsum("c,cmi,cnj->cnimj", volume * (K * l - mu), v, v)
        )
        A = coo_matrix(
            (H.reshape(-1, 12, 12)[ix, iy, iz], (rr, cc)),
            shape=(3 * len(af), 3 * len(af)),
        ).tocsc()
        return g[af].ravel(), A

    def certify(y):
        J = np.linalg.det(deformation(y))
        v, q = geom(y, sel)
        s = np.linalg.svd(deformation(y), compute_uv=False)
        newbed = O[free[O]]
        return dict(
            invalid=int(((v <= 1e-18) | (q < 1e-5) | (J < 0.4) | (J > 2.5)).sum()),
            min_stretch=float(s.min()),
            max_stretch=float(s.max()),
            min_J=float(J.min()),
            max_J=float(J.max()),
            bed_intrusions=int(
                np.sum(y[np.searchsorted(local, newbed), 2] < w["band_m"])
            ),
        )

    def ok(r):
        return (
            r["invalid"] == 0
            and r["bed_intrusions"] == 0
            and r["min_stretch"] >= 2 / 3
            and r["max_stretch"] <= 1.5
        )

    y = start.copy()
    g0, A = system(y, True)
    direction = np.random.default_rng(4912).normal(size=(len(af), 3))
    direction /= np.linalg.norm(direction)
    h = 1e-7
    plus = y.copy()
    minus = y.copy()
    plus[af] += h * direction
    minus[af] -= h * direction
    numeric = (energy(plus) - energy(minus)) / (2 * h)
    exact = g0 @ direction.ravel()
    Hn = (system(plus, False) - system(minus, False)) / (2 * h)
    Ht = A @ direction.ravel()
    derivative_error = abs(numeric - exact) / max(1e-8, abs(exact))
    tangent_error = float(np.linalg.norm(Hn - Ht) / np.linalg.norm(Ht))
    if derivative_error > 2e-4 or tangent_error > 2e-4:
        raise ValueError(("Unqualified derivative", derivative_error, tangent_error))
    history = []
    best = y.copy()
    best_e = energy(best)
    fraction = None
    if source_fit:
        data = json.loads((source_fit / "fit.json").read_text())
        plan = json.loads(screen.read_text())
        if (
            data["inputs"] != pins
            or data["clearance_witness_rows"] != protected
            or plan["fit"]["installed_pose_sha256"] != data["output_pose"]
            or sha(source_fit / "current.node") != data["output_pose"]
        ):
            raise ValueError("Matched native fraction is from another fitted material")
        fraction = max(
            row["fraction"]
            for row in plan["results"]
            if row["extrapolated_failures"] == 0
        )
        field = np.load(source_fit / "field.npz")
        changed = np.searchsorted(local, field["nodes"])
        if not np.array_equal(local[changed], field["nodes"]) or not np.array_equal(
            start[changed], field["original"]
        ):
            raise ValueError("Unmatched fitted nodes")
        best[changed] = (
            start[changed] + fraction * (field["current"] - start[changed])
        ).astype("f4")
        best_e = energy(best)
        if best_e >= energy(start):
            raise ValueError("Fraction failed elastic energy descent")
    elif mode == "constrained-outer":
        from posterior_local_relaxation import relax

        best, history = relax(
            start,
            sel,
            af,
            G,
            volume,
            mu,
            K,
            mass[active],
            energy,
            lambda z: system(z, False),
            lambda z: ok(certify(z)),
        )
        best_e = energy(best)
    else:
        for step in range(30):
            g, A = system(y, True)
            value = energy(y)
            history.append(
                dict(
                    step=step,
                    energy_J=value,
                    g_l2_N=float(np.linalg.norm(g)),
                    g_max_N=float(np.max(np.abs(g))),
                )
            )
            if np.linalg.norm(g) < 1e-6:
                break
            for damp in (0.0, 1e-6, 1e-3, 0.03, 0.3, 3.0):
                H = A if not damp else A + diags(damp * np.maximum(A.diagonal(), 1.0))
                dz = spsolve(H, -g)
                slope = g @ dz
                if not np.isfinite(slope) or slope >= 0:
                    continue
                accepted = False
                for n in range(20):
                    trial = y.copy()
                    scale = 2.0**-n
                    trial[af] += scale * dz.reshape(-1, 3)
                    if energy(trial) <= value + 1e-4 * scale * slope:
                        accepted = True
                        break
                if accepted:
                    break
            else:
                break
            y = trial
            # Eligibility is evaluated after storage rounding. A physical energy minimum
            # outside the installed geometry domain can still supply a safe partial update.
            for n in range(18):
                new = (start + (y - start) * 2.0**-n).astype("f4").astype(float)
                en = energy(new)
                if en >= best_e:
                    break
                if ok(certify(new)):
                    best_e = en
                    best = new.copy()
                    break
    q = certify(best)
    rows = active[np.any(best[af] != start[af], axis=1)]
    if not len(rows) or not ok(q):
        raise ValueError("No representable admissible energy reduction")
    subset = np.searchsorted(local, rows)
    f = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    write_coordinates(f, out / "current.node", rows, best[subset])
    np.savez_compressed(
        out / "field.npz", nodes=rows, original=start[subset], current=best[subset]
    )
    g = system(best, False).reshape(-1, 3)
    g0 = g0.reshape(-1, 3)
    result = dict(
        mode=mode,
        source_fit_sha256=sha(source_fit / "fit.json") if source_fit else None,
        matched_fraction=fraction,
        inputs=pins,
        source_materials=material_ids,
        source_mass=sha(masspath),
        output_pose=sha(out / "current.node"),
        clearance_witness_rows=protected,
        clearance_witness_sha256=sha(guard) if guard else None,
        free_nodes=len(af),
        touched_elements=len(cells),
        changed_nodes=len(rows),
        maximum_displacement_m=float(np.linalg.norm(best - start, axis=1).max()),
        derivative_error=derivative_error,
        tangent_error=tangent_error,
        initial_merit_J=energy(start),
        fitted_merit_J=best_e,
        initial_free_force_l2_N=float(np.linalg.norm(g0)),
        fitted_free_force_l2_N=float(np.linalg.norm(g)),
        initial_free_vertical_force_l2_N=float(np.linalg.norm(g0[:, 2])),
        fitted_free_vertical_force_l2_N=float(np.linalg.norm(g[:, 2])),
        initial_quality=certify(start),
        emitted_quality=q,
        history=history,
    )
    (out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("inputs", "source_materials", "source_mass", "history")
        }
    )


if __name__ == "__main__":
    run(
        *(Path(a).resolve() for a in sys.argv[1:6]),
        sys.argv[6],
        *(Path(v).resolve() for v in sys.argv[7:]),
    )
