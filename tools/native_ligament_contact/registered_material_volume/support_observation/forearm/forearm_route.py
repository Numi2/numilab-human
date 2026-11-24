"""Read the standard Numi tissue, actuator and tendon registrations for arm motion audits."""

import json
import struct
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_bytes())


def asset(cfg, flag):
    a = cfg["argv"]
    return Path(a[a.index(flag) + 1])


def route(path, index, donors, affines, ends):
    raw = path.read_bytes()
    h = struct.unpack_from("<8s9I32s", raw)
    ro = 76 + 16 * h[4] + 64 * h[5]
    mo = ro + 16 * h[6]
    _, start, n, _ = struct.unpack_from("<4I", raw, mo + 164 * index)
    pts = []
    owners = []
    sites = []
    for j in range(start, start + n):
        kind, idx, side, _ = struct.unpack_from("<4I", raw, ro + 16 * j)
        if kind != 1:
            continue  # a wrap-side marker is not a material attachment
        body, *point = struct.unpack_from("<I3f", raw, 76 + 16 * idx)
        k = donors.index(body)
        M, t = affines[k]
        if j in [start, start + n - 1]:
            end = ends[0 if j == start else 1]
            if (
                idx != end["source_site_index"]
                or body != end["body_index"]
                or np.max(abs(np.array(point) - end["resolved_local_point_m"])) > 1e-5
            ):
                raise ValueError("Functional material endpoint mismatch")
        pts.append(np.linalg.solve(M, point - t))
        owners.append(k)
        sites.append(idx)
    return np.array(pts), np.array(owners), sites
