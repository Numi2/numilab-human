"""Test connected, ipsilateral forearm skin bindings while preserving the registered reference surface."""

import json, runpy, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from registered_forearms import read, sha, Muscle
from verify_native_extensor_pair import loaded
from measure_forearm_enclosure import Enclosure


def ramp(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def pose_rows(pose):
    return {
        "source_poses": [
            dict(
                core_body=k,
                position=v["position_m"],
                quaternion_xyzw=v["quaternion_xyzw"],
            )
            for k, v in pose.items()
        ]
    }


def sample(base, q, reader, decfile, lib, pred, out):
    out.mkdir(exist_ok=True)
    j = read(base / "launch.json")
    args = j["argv"]
    skin = Path(args[args.index("--skin-payload") + 1])
    if sha(skin) != j["pins"][str(skin)]:
        raise ValueError("Changed skin")
    api = runpy.run_path(str(decfile))
    origin = api["decode"](skin.read_bytes())
    w = origin["weights"].astype(float)
    bodies = origin["binds"]["body"]
    poses = read(base / "native-initial-bodies.json")["all_bodies"]

    def pose(k):
        return {
            r["body_index"]: dict(
                position_m=r[k]["position_world_m"],
                quaternion_xyzw=r[k]["orientation_world_xyzw"],
            )
            for r in poses
        }

    earth = pose("rest")
    ref = api["reference_transforms"](origin, pose_rows(earth), [0.0, 0.0, 0.0])
    x = api["positions"](origin, *ref)
    reference = w.copy()
    selection = []
    for k, first, last in [(109, 41, 76), (110, 91, 126)]:
        m = Muscle(j, k)
        points = []
        for site in (m.sites[0], m.sites[-1]):
            b = earth[site["body"]]
            points.append(
                Rotation.from_quat(b["quaternion_xyzw"]).apply(site["local_m"])
                + b["position_m"]
            )
        a, b = points
        unit = b - a
        length = np.linalg.norm(unit)
        unit /= length
        s = (x - a) @ unit
        r = np.linalg.norm(x - (a + s[:, None] * unit), axis=1)
        side = (bodies >= first) & (bodies <= last)
        arm = w[:, side].sum(1)
        influence = (
            ramp((0.072 - r) / 0.024)
            * ramp((s + 0.065) / 0.03)
            * ramp((length + 0.018 - s) / 0.025)
            * ramp((arm - 0.30) / 0.35)
        )
        # Taper to the surrounding skin in canonical mesh connectivity, including
        # locations that lie near the torso in Euclidean distance.
        other = (bodies == 20) | (bodies == 25) | (bodies == 26)
        for idx in np.flatnonzero(other):
            reference[:, idx] -= influence * w[:, idx]
        moved = (w[:, other].sum(1) * influence)[:, None]
        reference[:, side] += (
            np.divide(
                w[:, side],
                arm[:, None],
                out=np.zeros((len(w), side.sum())),
                where=arm[:, None] > 0,
            )
            * moved
        )
        selection.append(
            dict(
                stable=k,
                eligible=int((influence > 0).sum()),
                max_removed_weight=float(moved.max()),
                physical_force_sites=[m.sites[0], m.sites[-1]],
            )
        )
    if np.min(reference) < -1e-12 or np.max(abs(reference.sum(1) - w.sum(1))) > 1e-9:
        raise ValueError("Source weights did not transport conservatively")
    changed = np.flatnonzero(np.any(reference != w, axis=1))
    np.savez_compressed(
        out / "weight-transfer.npz",
        rows=changed,
        weights=reference[changed],
        A=ref[0],
        b=ref[1],
    )
    sys.path.insert(0, str(reader.parent / "src"))
    pr = runpy.run_path(str(reader))["primitives"]
    ids, inv = np.unique(origin["inds"], return_inverse=True)
    f = inv.reshape(-1, 3)
    cases = {}
    for name, run in [("compact8", q / "physical-8"), ("compact16", q / "physical-16")]:
        rc, ps, raw = loaded(run, False, False)
        PP = api["reference_transforms"](origin, pose_rows(ps), [0, 0, 0])
        material = api["positions"](origin, *PP)[ids]
        geos = {
            key: (v, tri)
            for key, _, v, tri in pr(raw, ps)
            if key == (51007, 1) or (key[0] == 51005 and key[1] in (109, 110, 137, 138))
        }
        observed, nf = geos.pop((51007, 1))
        if not np.array_equal(nf, f):
            raise ValueError("Wrong native exterior order")
        # Full physical contacts and dynamic FEM require a separate native trial.
        cases[name] = (PP, material, geos, observed, rc["pack_file_sha256"])
    report = dict(
        skin=sha(skin),
        pose_sha256=sha(base / "native-initial-bodies.json"),
        regions=selection,
        trials=[],
    )
    for alpha in [0.0, 0.35, 0.7, 1.0]:
        ww = (w + alpha * (reference - w)).astype("f4")
        blob, info = api["rebind"](origin, ww, *ref)
        payload = api["decode"](blob)
        row = dict(
            fraction=alpha,
            changed_weights=len(info["changed_rows"]),
            reference_max_error_m=info.get("reference_error_max_m"),
            poses={},
        )
        for name, (PP, baseline, geo, observed, pack) in cases.items():
            proposed = api["positions"](payload, *PP)[ids]
            delta = proposed - baseline
            predicted = (observed + delta).astype("f4")
            s = Enclosure(lib, predicted, f)
            stat = {}
            try:
                for (_, sid), (v, tri) in geo.items():
                    cls, dist = s.query(v)
                    stat[sid] = dict(
                        exterior=int((cls == 0).sum()),
                        ambiguous=int((cls < 0).sum()),
                        max_exposure_mm=float(1000 * dist[cls == 0].max(initial=0)),
                    )
            finally:
                s.close()
            row["poses"][name] = dict(
                pack=pack,
                surfaces=stat,
                source_motion_max_mm=float(1000 * np.linalg.norm(delta, axis=1).max()),
            )
        report["trials"].append(row)
        print(alpha, row["poses"]["compact16"]["surfaces"], flush=True)
    (out / "weight-test.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sample(*(Path(v).resolve() for v in sys.argv[1:]))
