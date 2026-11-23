"""Follow material skin triangles and cortical attachment sites in a common skeletal frame."""

import ctypes as ct
import json, runpy, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from registered_forearms import read, sha
from reweight_forearm_skin import pose_rows
from measure_forearm_enclosure import Enclosure
from verify_native_extensor_pair import loaded


def locate(skin, v, f, points):
    points = np.ascontiguousarray(points, dtype="f8")
    out = np.empty((len(points), 7))
    ids = np.empty(len(points), "i4")
    P = np.ctypeslib.ndpointer(dtype="float64", flags="C_CONTIGUOUS")
    I = np.ctypeslib.ndpointer(dtype="int32", flags="C_CONTIGUOUS")
    fn = skin.lib.cortex_query
    fn.argtypes = [ct.c_void_p, P, ct.c_int, P, I]
    fn.restype = None
    fn(skin.handle, points, len(points), out, ids)
    tri = v[f]
    valid = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        > 0
    )
    faces = np.flatnonzero(valid)[ids]
    tri = v[f[faces]]
    bary = []
    for x, t in zip(out[:, 4:], tri):
        u = np.linalg.lstsq((t[1:] - t[:1]).T, x - t[0], rcond=None)[0]
        bary.append([1 - u.sum(), *u])
    b = np.asarray(bary)
    if (
        b.min() < -2e-5
        or np.max(abs(np.einsum("ni,nij->nj", b, tri) - out[:, 4:])) > 2e-6
    ):
        raise ValueError("Cortical witness lost barycentric ancestry")
    return out, faces, b


def prepare(case, base, reader, dec, lib, routes):
    rr = read(routes)
    j = read(case / "launch.json")
    a = j["argv"]
    skin = Path(a[a.index("--skin-payload") + 1])
    api = runpy.run_path(str(dec))
    s = api["decode"](skin.read_bytes())
    if sha(skin) != j["pins"][str(skin)]:
        raise ValueError("Wrong skin")
    meta = read(base / "native-initial-bodies.json")["all_bodies"]
    rest = {
        i["body_index"]: dict(
            position_m=i["rest"]["position_world_m"],
            quaternion_xyzw=i["rest"]["orientation_world_xyzw"],
        )
        for i in meta
    }
    ids, f = np.unique(s["inds"], return_inverse=True)
    f = f.reshape(-1, 3)
    frames = {}
    sys.path.insert(0, str(reader.parent / "src"))
    prim = runpy.run_path(str(reader))["primitives"]
    neutral = api["positions"](
        s, *api["reference_transforms"](s, pose_rows(rest), [0, 0, 0])
    )[ids]
    frames["neutral"] = (rest, neutral, None)
    for key, initial in [("release", True), ("terminal", False)]:
        rc, pose, raw = loaded(case, initial, False)
        ((_, _, v, faces),) = prim(raw, pose, skin_only=True)
        if not np.array_equal(faces, f):
            raise ValueError("Changed registration")
        frames[key] = (
            pose,
            v,
            rc["base_pack_file_sha256" if initial else "pack_file_sha256"],
        )
    if frames["terminal"][2] != rr["pack"]:
        raise ValueError("Route capture mismatch")
    return s, ids, f, frames, rr


def track(case, base, reader, dec, lib, routes):
    s, ids, f, frames, rr = prepare(case, base, reader, dec, lib, routes)
    unique = {}
    for bank in rr["banks"]:
        for m in bank["sites"]:
            if m["role"] == "origin":
                unique[m["source_site"]] = m
    sites = list(unique.values())
    records = [
        dict(site=r["source_site"], body=r["body"], muscle=r["muscle"], poses={})
        for r in sites
    ]
    reftria = refw = None
    for name, (pose, v, pack) in frames.items():

        def coords(m):
            p = pose[m["body"]]
            return (
                Rot.from_quat(p["quaternion_xyzw"]).apply(m["local_m"])
                + p["position_m"]
            )

        xyz = np.array([coords(m) for m in sites])
        shell = Enclosure(lib, v, f)
        lab, dist = shell.query(xyz)
        near, ff, bw = locate(shell, v, f, xyz)
        shell.close()
        if name == "neutral":
            reftria, refw = f[ff], bw
        for i, (m, report) in enumerate(zip(sites, records)):
            donors = bw[i] @ s["weights"][ids[f[ff[i]]]]
            main = np.argsort(-donors)[:5]
            wp = refw[i] @ v[reftria[i]]
            p = pose[m["body"]]
            Q = Rot.from_quat(p["quaternion_xyzw"])
            rel = Q.inv().apply(wp - xyz[i])
            report["poses"][name] = dict(
                enclosure=int(lab[i]),
                near_mm=dist[i] * 1000,
                nearest_face=int(ff[i]),
                skin_donors=dict(
                    zip(map(str, s["binds"]["body"][main]), donors[main].tolist())
                ),
                material_offset_local_mm=(rel * 1e3).tolist(),
                pack=pack,
            )
    return records


if __name__ == "__main__":
    print(json.dumps(track(*(Path(a) for a in sys.argv[1:])), indent=2))
