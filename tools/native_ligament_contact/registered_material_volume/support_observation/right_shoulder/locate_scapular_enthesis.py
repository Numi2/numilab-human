"""Locate the infraspinatus scapular attachment on native cortex while preserving its same-body terminal length."""

import json
import struct
import sys
from pathlib import Path

import igl
import numpy as np
from assemble_material import frames
from audit_registered_skeleton import source_geometry
from compile_connected_joint import digest
from numilab_human import model as owner
from rebind_ligament_material import banks
from register_deltoid_origins import cortical_circle
from registration_stages import parts
from repair_registered_surface import state
from scipy.spatial.transform import Rotation


def candidate(w, case, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    args = cfg["argv"]
    mp = Path(args[2])
    bone = Path(args[3])
    tp = Path(args[args.index("--tendon-payload") + 1])
    tissue = Path(args[args.index("--soft-tissue-payload") + 1])
    tm = tp.with_suffix(".manifest.json")
    if any(digest(q) != cfg["pins"][str(q)] for q in (mp, bone, tp, tissue, tm)):
        raise ValueError("Source drift")
    raw = mp.read_bytes()
    meta = json.loads(tm.read_bytes())
    index, endpoint = next(
        (i, s)
        for i, s in enumerate(meta["endpoints"])
        if s["muscle"] == "INFSP" and s["body_index"] == 34
    )
    h = struct.unpack_from("<8s9I32s", raw)
    wo = 76 + 16 * h[4]
    ro = wo + 64 * h[5]
    mo = ro + 16 * h[6]
    mi = endpoint["muscle_index"]
    _, start, count, _ = struct.unpack_from("<4I", raw, mo + 164 * mi)
    program = [
        struct.unpack_from("<4I", raw, ro + 16 * j) for j in range(start, start + count)
    ]
    via, site = program[-2][1], program[-1][1]
    if [(v[0], v[1]) for v in program[-2:]] != [(1, 661), (1, 662)] or endpoint[
        "route_node_index"
    ] != start + count - 1:
        raise ValueError("Terminal route changed")
    if (
        sum(
            struct.unpack_from("<2I", raw, ro + 16 * i) == (1, site)
            for i in range(h[6])
        )
        != 1
    ):
        raise ValueError("Shared terminal site")
    links = [struct.unpack_from("<I3f", raw, 76 + 16 * i) for i in (via, site)]
    if [a[0] for a in links] != [34, 34] or not np.array_equal(
        links[-1][1:], np.array(endpoint["resolved_local_point_m"], dtype="f4")
    ):
        raise ValueError("Different native endpoint")
    anchor, old = np.array([a[1:] for a in links])
    length = np.linalg.norm(anchor - old)
    f, v, _ = banks(bone)[34, 10]
    atlas = json.loads((w / "source-support-column.json").read_bytes())
    frame = atlas["frame"]
    if atlas["bone_payload"] != digest(bone):
        raise ValueError("Different atlas registration")
    member = next(q for q in atlas["rows"] if q["name"] == "Right infraspinatus muscle")
    atlas_path = w / "atlas" / (member["id"] + ".obj")
    if digest(atlas_path) != member["sha256"]:
        raise ValueError("Different infraspinatus source")
    p = frames(case)[-1]
    pose = p[34]
    R = Rotation.from_quat(pose["quaternion_xyzw"])
    translation = np.array(pose["position_m"])

    def local(y):
        return R.inv().apply(y - translation)

    xyz, faces = source_geometry(atlas_path.read_bytes())
    atlas_x = local(xyz @ frame["linear_m_per_mm"] + frame["translation_m"])
    v0, wt, b, tri, aff = parts(tissue, 83)
    current_x = local(state(v0, wt, aff, b, p))
    current = (current_x, np.ascontiguousarray(tri, dtype="i4"))

    def distance(q, mesh):
        ds, ids, points = igl.point_mesh_squared_distance(
            np.ascontiguousarray(q),
            np.ascontiguousarray(mesh[0]),
            np.ascontiguousarray(mesh[1], dtype="i4"),
        )
        return np.sqrt(ds)

    # Intersections of each cortical triangle with the constant-length sphere.
    # Sample the analytic arc in the directions of the original site and of
    # the closest vertices of the atlas footprint; accept only verified contact.
    near = np.flatnonzero(
        np.abs(np.linalg.norm(atlas_x - anchor, axis=1) - length) < 0.004
    )
    skin_gap = distance(atlas_x[near], (v, f))
    near = near[np.argsort(skin_gap)[:20]]
    candidates = []
    for target in [old, *atlas_x[near]]:
        _, points = cortical_circle(v, f, anchor, length, target)
        candidates.extend(points)
    points = np.unique(np.round(candidates, 12), axis=0)
    p32 = points.astype("f4").astype(float)
    cortex = distance(p32, (v, f))
    ag = distance(p32, (atlas_x, faces))
    mg = distance(p32, current)
    # A valid contact should also lie near the independent source muscle and its
    # registered counterpart. Distances here select the patch for force mapping.
    metrics = np.column_stack(
        (
            ag,
            mg,
            np.linalg.norm(p32 - old, axis=1),
            abs(np.linalg.norm(p32 - anchor, axis=1) - length),
        )
    )
    order = np.lexsort((metrics[:, 2], np.maximum(ag, mg)))
    settings = meta["admission"]
    surf = dict(
        body_index=34,
        stable_id=10,
        member_id="FJ3384",
        vertices=v.tolist(),
        triangles=f.tolist(),
    )
    proposals = []
    for q in order[:48]:
        j = int(q)
        e, reason = owner._numi_human_tendon_surface_envelope_connected(
            p32[j].tolist(),
            surf,
            settings["maximum_surface_distance_m"],
            settings["maximum_patch_radius_m"],
            settings["maximum_sampled_total_force_amplification"],
        )
        if e is None:
            continue
        pad = np.array(e["node_local_points_m"])
        proposals.append(
            dict(
                point=p32[j].tolist(),
                distances_mm=1000 * metrics[j],
                cortical_gap_m=float(cortex[j]),
                active_nodes_m=pad.tolist(),
                envelope=e,
                registered_envelope_muscle_distances_mm=(
                    1000 * distance(pad, current)
                ).tolist(),
                atlas_envelope_muscle_distances_mm=(
                    1000 * distance(pad, (atlas_x, faces))
                ).tolist(),
            )
        )
        if len(proposals) >= 8:
            break
    out.mkdir(exist_ok=True)
    for q in proposals:
        q["distances_mm"] = q["distances_mm"].tolist()
    rec = dict(
        launch=digest(case / "launch.json"),
        bone=digest(bone),
        muscle=digest(mp),
        tendon=digest(tp),
        tendon_manifest=digest(tm),
        tissue=digest(tissue),
        source_shape=member["sha256"],
        atlas_registration=digest(w / "source-support-column.json"),
        source_site_id=site,
        via_site_id=via,
        endpoint_index=index,
        route_index=start + count - 1,
        via=anchor.tolist(),
        old_site=old.tolist(),
        original_length_m=length,
        original_cortical_gap_m=distance(old[None], (v, f))[0],
        original_muscle_gap_m=distance(old[None], current)[0],
        candidate_count=len(points),
        proposals=proposals,
    )
    (out / "candidates.json").write_text(
        json.dumps(rec, indent=2, allow_nan=False) + "\n"
    )
    print(
        "candidates",
        len(points),
        len(proposals),
        "best mm",
        metrics[order[0]] * 1000,
        flush=True,
    )


if __name__ == "__main__":
    candidate(*(Path(s).resolve() for s in sys.argv[1:]))
