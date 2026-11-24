"""Extract neighboring native solids in a common metric frame for repeated registered contact fits."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
from assemble_material import frames
from compile_connected_joint import digest
from measure_assembly_contacts import canonical
from registration_stages import parts
from repair_registered_surface import state

from audit_native_shoulder_contacts import layer_parts, load


def prepare(w, case, out):
    api = runpy.run_path((w / "primitive-registry.path").read_text().strip())
    shapes, receipt = load(case, api)
    cfg = json.loads((case / "launch.json").read_bytes())
    bank = Path(cfg["argv"][cfg["argv"].index("--soft-tissue-payload") + 1])
    v, wt, b, f, aff = parts(bank, 83)
    x, ii, inv = np.unique(v, axis=0, return_index=True, return_inverse=True)
    F = inv[f]
    poses = frames(case)
    p = state(x, wt[ii], aff, b, poses[-1])
    X, G = shapes[51005, 83]
    if len(G) != len(F):
        raise ValueError("Different source and native triangle domains")
    got = np.unique(F)
    # Face-corner pairing must be exact across seams, with the same owning source point.
    shift = np.linalg.norm(p[F] - X[G], axis=2)
    if shift.max() > 2e-6:
        raise ValueError(
            (
                "Native topology or coordinate mismatch",
                shift.max(),
                np.quantile(shift, [0, 0.5, 1]).tolist(),
            )
        )
    pos = p.copy()
    pos[F.ravel()] = X[G.ravel()]
    if np.max(np.abs(pos[F] - X[G])) > 1e-9:
        raise ValueError("Split posed seam")
    origin = pos[got].mean(0)
    xyz = (pos - origin) * 1000
    low = xyz[got].min(0) - 10
    high = xyz[got].max(0) + 10
    save = dict(
        origin=origin,
        source=x,
        xyz=xyz,
        faces=F,
        pins=np.load(w / "registered/bank-83.npz")["pins"],
    )
    counts = {}

    def keep(name, solids):
        counts[name] = len(solids)
        for i, s in enumerate(solids):
            m = s.to_mesh64()
            save[f"{name}_{i}_x"] = np.asarray(m.vert_properties[:, :3]).copy()
            save[f"{name}_{i}_f"] = np.asarray(m.tri_verts).copy()

    for key, (points, tri) in shapes.items():
        if key in [(51005, 83), (51007, 1)] or key[0] not in api["BANK_CLASSES"]:
            continue
        y = (points - origin) * 1000
        if np.any(y.max(0) < low) or np.any(y.min(0) > high):
            continue
        s, meta = canonical(y, tri)
        if s is None:
            raise ValueError((key, meta))
        keep(f"{key[0]}-{key[1]}", [s])
    file = w / "material-import/volume/mesh.ele"
    cells = np.loadtxt(file, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype="i4")
    nodes = (
        np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
        - origin
    ) * 1000
    for k, ss in layer_parts(nodes, cells[:, :4] - 1, cells[:, 4], low, high).items():
        keep(f"0-{k}", ss)
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "geometry.npz", **save)
    meta = dict(
        geometry=digest(out / "geometry.npz"),
        pack=receipt["pack_file_sha256"],
        bank=digest(bank),
        material=digest(case / "physical-accepted.node"),
        source_to_native_endpoint_max_m=float(shift.max()),
        solids=counts,
    )
    (out / "geometry.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    prepare(*(Path(v).resolve() for v in sys.argv[1:]))
