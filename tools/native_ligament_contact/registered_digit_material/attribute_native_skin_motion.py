"""Separate skin binding, skeletal motion and surface transfer in two actual supported runs."""

from pathlib import Path
import json, runpy, sys, csv
import numpy as np
from scipy.spatial.transform import Rotation
from verify_native_extensor_pair import loaded
from registered_forearms import sha
from reweight_forearm_skin import pose_rows


def compare(old, new, reader, decoder):
    sys.path.insert(0, str(reader.parent / "src"))
    pr = runpy.run_path(str(reader))["primitives"]
    api = runpy.run_path(str(decoder))
    skin = []
    ps = []
    observed = []
    refs = []
    for case in (old, new):
        rc, pose, raw = loaded(case, False, False)
        j = json.loads((case / "launch.json").read_text())
        argv = j["argv"]
        file = Path(argv[argv.index("--skin-payload") + 1])
        if j["pins"][str(file)] != sha(file):
            raise ValueError("Different skin pin")
        payload = api["decode"](file.read_bytes())
        skin.append(payload)
        ps.append(pose)
        ((_, _, x, f),) = pr(raw, pose, skin_only=True)
        ids, faces = np.unique(payload["inds"], return_inverse=True)
        if not np.array_equal(f, faces.reshape(-1, 3)):
            raise ValueError("Different skin topology")
        observed.append(x)
        refs.append(rc)
    if [r["accepted_step"] for r in refs] != [8000, 8000] or len(ps[0]) != len(ps[1]):
        raise ValueError("Unmatched endpoints")

    def calc(s, p):
        return api["positions"](
            s, *api["reference_transforms"](s, pose_rows(p), [0, 0, 0])
        )[ids]

    x00 = calc(skin[0], ps[0])
    x10 = calc(skin[1], ps[0])
    x11 = calc(skin[1], ps[1])
    change = np.any(skin[0]["weights"][ids] != skin[1]["weights"][ids], axis=1)
    group = {}
    body = skin[0]["binds"]["body"]
    oldw = skin[0]["weights"][ids]
    for name, mask in [
        ("edited", change),
        ("unchanged", ~change),
        ("right_arm", oldw[:, (body >= 41) & (body <= 76)].sum(1) > 0.5),
        ("left_arm", oldw[:, (body >= 91) & (body <= 126)].sum(1) > 0.5),
    ]:
        group[name] = {"count": int(mask.sum())}
        for k, delta in [
            ("binding", x10 - x00),
            ("articulation", x11 - x10),
            ("additional_surface_response", (observed[1] - x11) - (observed[0] - x00)),
            ("native_total", observed[1].astype(float) - observed[0]),
        ]:
            norm = np.linalg.norm(delta[mask], axis=1) * 1000
            group[name][k] = {
                "median_mm": float(np.median(norm)),
                "p95_mm": float(np.quantile(norm, 0.95)),
                "maximum_mm": float(norm.max()),
            }
    bones = []
    for k, u in sorted(ps[0].items()):
        v = ps[1][k]
        R = Rotation.from_quat(u["quaternion_xyzw"]).inv() * Rotation.from_quat(
            v["quaternion_xyzw"]
        )
        bones.append(
            dict(
                body=k,
                origin_displacement_mm=float(
                    np.linalg.norm(np.subtract(v["position_m"], u["position_m"])) * 1000
                ),
                rotation_deg=float(np.rad2deg(R.magnitude())),
            )
        )

    def rows(case, file):
        return list(csv.DictReader((case / file).open()))

    a, b = [rows(p, "resting-local-bed-sections.csv") for p in (old, new)]
    bed = []
    for aa, bb in zip(a, b, strict=True):
        if (aa["source_geometry_index"], aa["contact_index"]) != (
            bb["source_geometry_index"],
            bb["contact_index"],
        ):
            raise ValueError("Different bed contact ownership")
        bed.append(
            {
                k: float(bb[k]) / float(aa[k]) - 1
                for k in [
                    "projected_area_m2",
                    "skin_thickness_m",
                    "fat_thickness_m",
                    "skin_compliance_m_per_n",
                    "fat_compliance_m_per_n",
                ]
            }
            | dict(contact_index=int(aa["contact_index"]))
        )
    a, b = [rows(p, "resting-com-support-impulses.csv") for p in (old, new)]
    if len(a) != 8000 or len(b) != 8000:
        raise ValueError("Incomplete bed trace")
    oldJ = {}
    newJ = {}
    for aa, bb in zip(a, b, strict=True):
        keys = (
            "accepted_step",
            "contact_index",
            "body_index",
            "source_geometry_index",
            "row_class",
        )
        if any(aa[k] != bb[k] for k in keys):
            raise ValueError("Different bed row")
        for d, r in [(oldJ, aa), (newJ, bb)]:
            k = int(r["contact_index"])
            d[k] = d.get(k, 0) + float(r["normal_impulse_ns"])
    return dict(
        old_pack=refs[0]["pack_file_sha256"],
        new_pack=refs[1]["pack_file_sha256"],
        change_components_mm=group,
        body_changes=bones,
        bed_sections=bed,
        bed_impulse_ns=dict(old=oldJ, new=newJ),
    )


if __name__ == "__main__":
    print(json.dumps(compare(*(Path(s) for s in sys.argv[1:])), indent=2))
