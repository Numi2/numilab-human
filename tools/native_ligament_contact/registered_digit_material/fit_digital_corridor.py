"""Find a finite dorsal tendon corridor on the registered terminal phalanx and DIP joint."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from digital_clearance import DigitalClearance
from material_context import unit


def search(scene, digit, width, thickness):
    axis = unit(np.array(digit["axis"]))
    dorsal = unit(np.array(digit["dorsal"]))
    lateral = unit(np.array(digit["lateral"]))
    enthesis = np.array(digit["trial_enthesis"]["world_m"])
    original = np.array([s["world_m"] for s in digit["route"]])
    # The terminal free segment spans the middle/distal phalanx joint. Search its original
    # proximal station and the inferred distal cortical footprint in the accepted pose.
    first = original[-2]
    span = (enthesis - first) @ axis
    if span <= 0:
        raise ValueError("Dorsal stations reversed")
    count = int(np.ceil(span / 0.0004)) + 1
    stations = np.linspace(first, enthesis, count)
    step = 0.0004
    lateral_offsets = np.arange(-0.004, 0.004001, step)
    height_offsets = np.arange(-0.004, 0.004001, step)
    u, h = np.meshgrid(lateral_offsets, height_offsets, indexing="ij")
    offsets = u[..., None] * lateral + h[..., None] * dorsal
    strips = int(np.ceil(width / (2 * step)))
    # A point on this rectangular section is at most this far from the nearest
    # material sample. This covers its full thickness and lateral sample spacing.
    # Round up the represented width; no thin centerline may stand in for a sheet.
    rho = np.hypot(thickness * 0.5, step * 0.5)
    np.arange(-strips, strips + 1) * step
    coords = stations[:, None, None, :] + offsets[None, :, :, :]
    extended = np.arange(-10 - strips, 11 + strips) * step
    x = (
        stations[:, None, None, :]
        + extended[None, :, None, None] * lateral
        + height_offsets[None, None, :, None] * dorsal
    )
    # Left and right arrays explicitly share their origin at the center station.
    assert np.allclose(
        coords, x[:, strips : strips + len(lateral_offsets)], rtol=0, atol=1e-12
    )
    query = scene.query(x, layers=("O", "I", "D"))

    def strip(k):
        v = query[k].reshape(x.shape[:-1])
        return np.min(
            np.lib.stride_tricks.sliding_window_view(v, 2 * strips + 1, axis=1), axis=-1
        )

    bone = strip("bone")
    layer = {k: strip(k) for k in ("O", "I", "D")}
    current_clearance = np.minimum(bone, layer["D"]) - rho
    interior_clearance = np.minimum(bone, layer["I"]) - rho
    skin_clearance = np.minimum(bone, layer["O"]) - rho

    # A signed distance is 1-Lipschitz. The midpoint bound below proves the
    # entire swept strip between adjacent stations for this fixed section frame.
    def graph(clearance):
        cs = coords.reshape(count, -1, 3)
        margins = clearance.reshape(count, -1)
        score = np.where(margins[0] >= 0, margins[0], -np.inf)
        parents = []
        last = 0
        if not np.isfinite(score).any():
            return dict(stations_reached=0, full_span=False)
        for k in range(1, count):
            tree = cKDTree(cs[k - 1])
            src, dst = [], []
            for j, idx in enumerate(tree.query_ball_point(cs[k], 0.0013)):
                src.extend(idx)
                dst.extend([j] * len(idx))
            src = np.asarray(src, dtype=int)
            dst = np.asarray(dst, dtype=int)
            length = np.linalg.norm(cs[k - 1, src] - cs[k, dst], axis=1)
            # Intersect the two Lipschitz lower cones on the edge. Bound also
            # by both endpoints for tiny FP differences or a constant feature.
            edge = np.minimum(
                np.minimum(margins[k - 1, src], margins[k, dst]),
                (margins[k - 1, src] + margins[k, dst] - length) * 0.5,
            )
            values = np.minimum(score[src], edge)
            best = np.full(score.shape, -np.inf)
            parent = np.full(score.shape, -1, dtype="i4")
            order = np.lexsort((src, -values, dst))
            chosen = order[np.r_[True, np.diff(dst[order]) != 0]] if len(order) else []
            ok = np.asarray([i for i in chosen if values[i] >= 0], dtype=int)
            best[dst[ok]] = values[ok]
            parent[dst[ok]] = src[ok]
            if not np.isfinite(best).any():
                break
            score = best
            parents.append(parent)
            last = k
        tip = int(np.argmax(score))
        ids = [tip]
        for parent in reversed(parents):
            ids.append(int(parent[ids[-1]]))
        line = cs[np.arange(last + 1), ids[::-1]]
        return dict(
            stations_reached=last + 1,
            full_span=(last + 1 == count),
            proven_margin_mm=float(score[tip] * 1000),
            free_centerline_m=line.tolist(),
            proximal_offset_mm=float(np.linalg.norm(line[0] - first) * 1000),
            distance_to_inferred_insertion_mm=float(
                np.linalg.norm(line[-1] - enthesis) * 1000
            ),
        )

    rows = {}
    by_layer = {
        "deep": current_clearance,
        "inner": interior_clearance,
        "outer": skin_clearance,
    }
    for name, field in by_layer.items():
        peaks = field.reshape(count, -1).max(axis=1)
        rows[name] = dict(
            **graph(field),
            stations_with_section=int((peaks >= 0).sum()),
            bottleneck_sampled_margin_mm=float(peaks.min() * 1000),
            bottleneck_station=int(peaks.argmin()),
            best_margin_per_station_mm=(peaks * 1000).tolist(),
        )
    return dict(
        stations=count,
        width_mm=width * 1000,
        thickness_mm=thickness * 1000,
        section_axis_gap_m=step,
        proximal_site=digit["route"][-2]["site"],
        sampled_span_mm=float(span * 1000),
        window_half_width_mm=4.0,
        free_sections=rows,
    )


if __name__ == "__main__":
    root, lib = map(Path, sys.argv[1:3])
    scene = DigitalClearance(root, lib)
    result = {}
    for name, digit in scene.meta["digits"].items():
        cases = {}
        for label, w, t in [("mean", 6.27, 1.02), ("small", 5.04, 0.81)]:
            cases[label] = search(scene, digit, w * 0.001, t * 0.001)
        result[name] = cases
        a = cases["mean"]
        print(
            name,
            a["stations"],
            [
                (
                    k,
                    r["stations_reached"],
                    r["stations_with_section"],
                    round(r["bottleneck_sampled_margin_mm"], 2),
                )
                for k, r in a["free_sections"].items()
            ],
            flush=True,
        )
    (root / "corridor-fit.json").write_text(
        json.dumps(
            dict(
                physical=scene.meta["identities"],
                source="https://doi.org/10.13418/j.issn.1001-165x.2021.02.002",
                scope="Finite rectangular free tendon section, local fixed dorsal section frame, no cortical insertion taper or physical route promotion. A failed search is bounded by its sampled stations and orientation.",
                results=result,
            ),
            indent=2,
        )
        + "\n"
    )
