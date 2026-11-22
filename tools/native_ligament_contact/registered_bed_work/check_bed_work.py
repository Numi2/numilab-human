"""Check the Metal virtual-work observation against independent float64 differences and native impulses."""

from pathlib import Path
import csv, hashlib, json, sys
import numpy as np


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def scalar(lines, key):
    return np.array([float(r[key]) for r in lines])


def vec(lines, names):
    return np.column_stack([scalar(lines, n) for n in names.split()])


def rotate(quat, v):
    u = quat[:, :3]
    return v + 2 * np.cross(u, np.cross(u, v) + quat[:, 3, None] * v)


def check(root, source, registration):
    receipt = json.loads((root / "terminal.receipt.json").read_text())
    step = receipt["accepted_step"]
    if receipt["physical_endpoint"] != "accepted" or step % 32:
        raise ValueError("Missing accepted endpoint")
    launch = json.loads((root / "launch.json").read_text())
    dt = float(launch["argv"][launch["argv"].index("--muscle-step-seconds") + 1])
    geom = registration / "skin-source-anatomy-parameters.bin"
    maps = registration / "skin-source-vertex-map.bin"
    for n, p in [("anatomy_parameters", geom), ("vertex_map", maps)]:
        if digest(p) != receipt["skin_source_mapping"][n]["sha256"]:
            raise ValueError("Mismatched registration " + n)
    a = np.fromfile(geom, "<f4").reshape(-1, 4).astype(float)
    mapping = np.fromfile(maps, "<f4").reshape(-1, 12)
    data = rows(root / "resting-thoracic-bed-work.csv")
    chosen = rows(root / "resting-support-selected-skin-witness.csv")
    force = rows(root / "resting-com-support-impulses.csv")
    load = {
        (int(r["accepted_step"]), int(r["contact_index"])): r
        for r in force
        if r["row_class"] == "external_support_contact"
    }
    surface = {(int(r["accepted_step"]), int(r["region"])): r for r in chosen}
    keys = [(int(r["accepted_step"]), int(r["region"])) for r in data]
    if (
        set(keys) != set(load)
        or set(keys) != set(surface)
        or len(keys) != len(set(keys))
        or len(keys) != 32 * (step // 32)
    ):
        raise ValueError("Incomplete matched flat bed witness set")
    for r, k in zip(data, keys):
        s = surface[k]
        if (
            int(r["vertex"]) != int(s["selected_skin_vertex_local"])
            or int(r["source"]) != int(s["source_geometry_index"])
            or int(s["pre_dynamics_step_index"]) != k[0] - 1
        ):
            raise ValueError(f"Witness identity mismatch {k}")
    impulse = vec(data, "ix iy iz")
    native = vec(
        [load[k] for k in keys],
        "impulse_world_x_ns impulse_world_y_ns impulse_world_z_ns",
    )
    np.testing.assert_allclose(impulse, native, atol=3e-7, rtol=3e-7)
    ids = scalar(data, "vertex").astype(int)
    weight = scalar(data, "weight")
    basal = scalar(data, "basal")
    np.testing.assert_allclose(weight, mapping[ids, 4], atol=0, rtol=1e-8)
    np.testing.assert_allclose(basal, mapping[ids, 11], atol=0, rtol=1e-8)
    q = vec(data, "qd qr")
    v = vec(data, "vd vr")
    local = vec(data, "local_x local_y local_z")
    quat = vec(data, "qx qy qz qw")
    np.testing.assert_allclose((quat**2).sum(axis=1), 1, rtol=2e-6)
    axis = a[2, :3]
    origin = a[1, :3]
    vol = a[1, 3]
    area = a[6, 2]
    along = (local - origin) @ axis
    across = local - origin - along[:, None] * axis

    def position(q):
        # Evaluate both new coordinates in the torso frame and then rotate the
        # displaced position. This independently differentiates the position.
        radial = np.sqrt((vol + q[:, 0] + q[:, 1]) / (vol + q[:, 0]))
        return rotate(
            quat,
            local
            + weight[:, None]
            * (
                (radial - 1)[:, None] * across
                - (q[:, 0] * basal / area)[:, None] * axis
            ),
        )

    dpos = position(q) - rotate(quat, local)
    np.testing.assert_allclose(dpos, vec(data, "dx dy dz"), atol=2.5e-9, rtol=2e-5)
    eps = 1e-7
    I = []
    J = []
    maxerr = []
    for k, n in enumerate(("jdx jdy jdz", "jrx jry jrz")):
        d = np.zeros(2)
        d[k] = eps
        j = (position(q + d) - position(q - d)) / (2 * eps)
        J.append(j)
        maxerr.append(float(np.max(abs(j - vec(data, n)))))
        np.testing.assert_allclose(j, vec(data, n), atol=1e-7, rtol=2e-5)
        gen = (j * impulse).sum(axis=1)
        I.append(gen)
        np.testing.assert_allclose(
            gen, scalar(data, ["id_pa_s", "ir_pa_s"][k]), atol=9e-8, rtol=2e-5
        )
    # Equal work in physical and swept-volume coordinates, using the current
    # impulse and pre-step modal velocity. Samples never stand in for integrals.
    virtual = I[0] * v[:, 0] + I[1] * v[:, 1]
    np.testing.assert_allclose(virtual, scalar(data, "work_j"), atol=2e-12, rtol=2e-5)
    bystep = np.array([k[0] for k in keys])
    net = np.array(
        [[gen[bystep == s].sum() / dt for gen in I] for s in np.unique(bystep)]
    )
    split = {}
    for name, axes in [("normal", [2]), ("tangential", [0, 1])]:
        values = np.column_stack(
            [np.sum(j[:, axes] * impulse[:, axes], axis=1) for j in J]
        )
        sums = np.array(
            [values[bystep == s].sum(axis=0) / dt for s in np.unique(bystep)]
        )
        split[name] = dict(
            min=sums.min(axis=0).tolist(),
            max=sums.max(axis=0).tolist(),
            rms=np.sqrt((sums**2).mean(axis=0)).tolist(),
        )
    loaded = impulse[:, 2] > 0.0
    relevant = loaded & (weight > 0)
    paired = [
        "resting-coupled.csv",
        "resting-surface-audit.csv",
        "resting-com-support-impulses.csv",
        "resting-support-selected-skin-witness.csv",
    ]
    parity = {}
    for name in paired:
        trial = rows(root / name)
        base = rows(source / name)
        # Sparse native frame diagnostics can have a separate endpoint cadence.
        if name == "resting-surface-audit.csv":
            continue
        if len(trial) > len(base):
            raise ValueError("Parent trace too short")
        parity[name] = all(x == y for x, y in zip(trial, base))
    if not all(parity.values()):
        raise ValueError(f"Changed native trajectory {parity}")
    out = dict(
        accepted_steps=step,
        simulated_seconds=dt * step,
        stride=32,
        sampled_contact_rows=len(keys),
        positive_normal_rows=int(loaded.sum()),
        loaded_weighted_rows=int(relevant.sum()),
        max_abs_jacobian_difference_m_per_m3=maxerr,
        net_pressure_equivalent_pa=dict(
            min=net.min(axis=0).tolist(),
            max=net.max(axis=0).tolist(),
            rms=np.sqrt((net**2).mean(axis=0)).tolist(),
        ),
        max_loaded_displacement_um=float(
            np.max(np.linalg.norm(dpos[loaded], axis=1)) * 1e6
        ),
        max_loaded_normal_displacement_um=float(np.max(np.abs(dpos[loaded, 2])) * 1e6),
        max_abs_sampled_virtual_work_j=float(max(abs(virtual))),
        parent_prefix_value_parity=parity,
        pack_sha256=receipt["pack_file_sha256"],
        device_observation_sha256=digest(root / "resting-thoracic-bed-work.csv"),
        timing=json.loads((root / "execution.json").read_text()),
        separated_pressure_equivalent_pa=split,
        scope="Selected, pre-step flat-bed contact witnesses; observed current candidate impulse confirmed against accepted export. Pressure equivalents are instantaneous sampled impulse/dt. Work is virtual work at pre-step velocity, not measured mechanical energy. No force is fed back by this audit.",
    )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    check(*(Path(s) for s in sys.argv[1:]))
