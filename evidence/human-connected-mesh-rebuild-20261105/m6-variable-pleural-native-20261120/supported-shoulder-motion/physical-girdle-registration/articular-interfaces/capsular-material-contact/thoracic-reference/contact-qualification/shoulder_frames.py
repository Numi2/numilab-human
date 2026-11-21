import json
import numpy as np
from scipy.spatial.transform import Rotation
from compare_registration import prepare
from certify_material_branch import Bone, posed


def setup(launch, refs, motion, archive, sites, heads, spacing, side="r"):
    _, _, _, _, _, _, atlas = prepare(launch, refs, motion, archive)
    row = next(
        r
        for r in json.loads(sites.read_text())["rows"]
        if bool(r["side"]) == (side == "r")
    )
    head = next(r for r in json.loads(heads.read_text())["rows"] if r["side"] == side)[
        "variants"
    ][0]["atlas_center_m"]
    ds = next(r for r in json.loads(spacing.read_text())["rows"] if r["side"] == side)[
        "translation_atlas_axes_m"
    ]
    bones = {}
    ep = []
    sids = []
    for site in row["attachments"]:
        sid = site["stable"]
        v, f = atlas[sid][2:4]
        q = (v - head) * atlas[sid][7] - (np.asarray(ds) if not ep else 0)
        bones[sid] = Bone((q, f), np.r_[np.zeros(6), 1.0], sid)
        ep.append(site["barycentric"] @ q[f[site["triangle"]]])
        sids.append(sid)
    return row, bones, ep, sids


def at_angle(bones, ep, sids, axis, degrees):
    r = Rotation.from_rotvec(np.eye(3)[axis] * np.radians(degrees))
    return {**bones, sids[1]: posed(bones[sids[1]], np.r_[np.zeros(3), r.as_quat()])}, [
        ep[0],
        r.apply(ep[1]),
    ]
