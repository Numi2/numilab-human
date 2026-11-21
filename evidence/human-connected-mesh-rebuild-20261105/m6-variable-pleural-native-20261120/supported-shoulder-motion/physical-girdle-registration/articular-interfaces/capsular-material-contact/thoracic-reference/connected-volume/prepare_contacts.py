import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from screen_material_transport import inputs
from numilab_human.capsular_material_registration import (
    emitted_body_vertices,
    material_bank,
    read_material_banks,
)


def run(scene, loader, receipt, bone, arms, out):
    cfg = json.loads(arms.read_text())
    state, banks, _, points, pins = inputs(scene, loader, receipt, bone)
    if pins != cfg["captured"]:
        raise ValueError("The skeletal reference and volumetric reference differ")
    poses = {
        r["body_index"]: np.array(r["position_m"] + r["quaternion_xyzw"])
        for r in json.loads(receipt.read_text())["accepted_body_poses"]
    }
    d = read_material_banks(bone)
    tree = cKDTree(points)
    a = {}
    for layer in (1, 0xFFFFFD11, 0xFFFFFD12):
        v, f = banks[51007, layer]
        dist, idx = tree.query(v)
        if dist.max() != 0:
            raise ValueError("Layer coordinate ancestry differs")
        a[f"l{layer}"] = idx[f].astype("i4")
    for g in cfg["moved"]:
        rr = Rotation.from_quat(g["rotation_xyzw"])
        for sid in g["banks"]:
            record = d[2][sid]
            p = poses[record[0]]
            _, _, attrs, f = material_bank(d, sid, record[0])
            x = p[:3] + Rotation.from_quat(p[3:]).apply(
                emitted_body_vertices(attrs, record)
            )
            y = g["pivot_after_m"] + rr.apply(x - g["pivot_before_m"])
            a[f"b{sid}"] = y.astype("f4")
            a[f"f{sid}"] = f.astype("i4")
    np.savez(out, **a)


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
