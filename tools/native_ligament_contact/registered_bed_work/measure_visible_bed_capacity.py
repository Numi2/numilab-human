"""Find how much actual bed-facing area of the initial native body lies within reach of the flat bed."""

import argparse
import ctypes
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from material_context import sha
from verify_native_extensor_pair import loaded


def measure(run, reader, lib, bound):
    receipt, pose, raw = loaded(run, True, False)
    sys.path.insert(0, str(reader.parent / "src"))
    api = runpy.run_path(str(reader))
    ((_, _, points, faces),) = api["primitives"](raw, pose, skin_only=True)
    x = np.ascontiguousarray(points, dtype="f8")
    faces = np.ascontiguousarray(faces, dtype="u4")
    fn = ctypes.CDLL(str(lib.resolve())).accessible_area
    fn.argtypes = [
        np.ctypeslib.ndpointer("f8", flags="C_CONTIGUOUS"),
        ctypes.c_uint,
        np.ctypeslib.ndpointer("u4", flags="C_CONTIGUOUS"),
        ctypes.c_uint,
        ctypes.c_double,
    ]
    fn.restype = ctypes.c_double
    doc = json.loads(bound.read_text())
    required = doc["weight_N"] / doc["soft_layer_limited_pressure_Pa"]
    if sha(run / "resting-local-bed-footprint.csv") != doc["footprint"]:
        raise ValueError("Wrong material projection")
    full = float(fn(x, len(x), faces, faces.size, np.max(x[:, 2]) + 1.0))
    data = []
    for h in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15):
        area = fn(x, len(x), faces, faces.size, h)
        if (
            not np.isfinite(area)
            or area < 0
            or area > full + 1e-9
            or (data and area < data[-1]["area_m2"] - 1e-12)
        ):
            raise ValueError("Invalid visible area")
        data.append(
            dict(
                reach_m=h,
                area_m2=area,
                optimistic_load_N=doc["soft_layer_limited_pressure_Pa"] * area,
            )
        )
    result = dict(
        geometry=receipt["base_pack_file_sha256"],
        predicate=sha(lib),
        frozen_skin_vertices=len(x),
        frozen_skin_triangles=len(faces),
        required_area_m2=required,
        whole_bed_facing_projection_m2=full,
        registered_column_area_m2=doc["all_columns_independent_capacity_N"]
        / doc["soft_layer_limited_pressure_Pa"],
        heights=data,
    )
    if full >= required:
        low = min(0.0, float(x[:, 2].min()) - 1e-6)
        high = float(x[:, 2].max() + 1e-7)
        for _ in range(13):
            mid = (low + high) / 2
            if fn(x, len(x), faces, faces.size, mid) < required:
                low = mid
            else:
                high = mid
        result["minimum_geometric_reach_bracket_m"] = [low, high]
    result["scope"] = (
        "Initial native posture, occlusion-correct projected lower surface and a common plane. Geometric availability is an upper bound. Contact pressure, lateral tissue transfer and possible mattress deformation require physical owners."
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "reader", "lib", "bound"):
        p.add_argument(k, type=Path)
    print(json.dumps(measure(**vars(p.parse_args())), indent=2, allow_nan=False))
