"""Recompile the scapular force footprint on the corresponding cortex of a retained motion model."""

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


def load(p):
    return json.loads(p.read_bytes())


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def local(x, pose):
    return (
        Rotation.from_quat(pose["quaternion_xyzw"]).inv().apply(x - pose["position_m"])
    )


def transfer(w, control, target, registered, out):
    out.mkdir(exist_ok=True)
    a, c = load(control / "launch.json"), load(target / "launch.json")
    b = load(registered / "candidates.json")
    facing = load(registered / "facing.json")
    if b["launch"] != digest(control / "launch.json") or facing["source"] != digest(
        registered / "candidates.json"
    ):
        raise ValueError("Unmatched physical source")
    argv = c["argv"]
    bone = Path(argv[3])
    mp = Path(argv[2])
    tp = Path(argv[argv.index("--tendon-payload") + 1])
    tissue = Path(argv[argv.index("--soft-tissue-payload") + 1])
    tm = tp.with_suffix(".manifest.json")
    if any(c["pins"].get(str(p)) != digest(p) for p in (bone, mp, tp, tissue)):
        raise ValueError("Stale receiving model")
    srcf, x, _ = banks(Path(a["argv"][3]))[34, 10]
    f, y, _ = banks(bone)[34, 10]
    f = np.asarray(f, dtype="i4")
    if not np.array_equal(srcf, f):
        raise ValueError("Scapular cortical ancestry differs")
    u, _, v = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    R = u @ v
    t = y.mean(0) - x.mean(0) @ R
    err = np.linalg.norm(x @ R + t - y, axis=1).max()
    if np.linalg.det(R) < 0.999999 or err > 1e-6:
        raise ValueError("Scapula is not rigidly congruent")
    raw = mp.read_bytes()
    tm0 = load(tm)
    index, row = next(
        (j, r)
        for j, r in enumerate(tm0["endpoints"])
        if r["muscle"] == "INFSP" and r["body_index"] == 34
    )
    h = struct.unpack_from("<8s9I32s", raw)
    ro = 76 + 16 * h[4] + 64 * h[5]
    rt = [struct.unpack_from("<2I", raw, ro + 16 * j) for j in range(h[6])]
    end = row["route_node_index"]
    if rt[end - 1 : end + 1] != [(1, 661), (1, 662)] or rt.count((1, 662)) != 1:
        raise ValueError("Functional route differs")
    links = [struct.unpack_from("<I3f", raw, 76 + 16 * i) for i in [661, 662]]
    if [q[0] for q in links] != [34, 34]:
        raise ValueError("Route crosses a joint")
    via, old = np.array([q[1:] for q in links])
    length = np.linalg.norm(via - old)
    site_errors = np.linalg.norm(
        np.array([b["via"], b["old_site"]]) @ R + t - [via, old], axis=1
    )
    if site_errors.max() > 1e-6:
        raise ValueError("Route is not congruent with the donor")
    ref = load(w / "source-support-column.json")
    at = w / "atlas/FJ1500.obj"
    if digest(at) != b["source_shape"] or ref["bone_payload"] != digest(
        Path(a["argv"][3])
    ):
        raise ValueError("Unmatched anatomical source")
    px = frames(control)[-1][34]
    g = ref["frame"]
    xyz, faces = source_geometry(at.read_bytes())
    atlas = (local(xyz @ g["linear_m_per_mm"] + g["translation_m"], px) @ R + t, faces)
    receipts = sorted(
        (target / "native-traces/accepted-geometry").glob("*.receipt.json")
    )
    if not receipts:
        raise ValueError("No accepted poses from receiving model")
    meshes = {}
    for rec in receipts:
        rr = load(rec)
        pose = {q["body_index"]: q for q in rr["accepted_body_poses"]}
        co = pose[34]
        arr = []
        for sid in [83, 85]:
            x0, weights, bones, faces0, aff = parts(tissue, sid)
            arr.append((local(state(x0, weights, aff, bones, pose), co), faces0))
        meshes[rr["accepted_step"]] = arr
    depths = np.array([0.05, 0.25, 0.5, 1, 2, 3, 5, 8]) * 0.001
    opts = []
    res = []
    surf = dict(
        body_index=34,
        stable_id=10,
        member_id="FJ3384",
        vertices=y.tolist(),
        triangles=f.tolist(),
    )
    settings = tm0["admission"]
    gap = lambda q: np.sqrt(
        igl.point_mesh_squared_distance(np.ascontiguousarray(q), y, f)[0]
    )
    for i, src in enumerate(b["proposals"]):
        if not all(facing["rows"][i]["proven_source_sites"]):
            continue
        dest = np.array(src["point"]) @ R + t
        _, pts = cortical_circle(y, f, via, length, dest)
        points = np.asarray(pts)
        point = (
            points[np.linalg.norm(points - dest, axis=1).argmin()]
            .astype("f4")
            .astype(float)
        )
        if np.linalg.norm(point - dest) > 1e-6:
            continue
        e, reason = owner._numi_human_tendon_surface_envelope_connected(
            point.tolist(),
            surf,
            settings["maximum_surface_distance_m"],
            settings["maximum_patch_radius_m"],
            settings["maximum_sampled_total_force_amplification"],
        )
        if e is None:
            continue
        original = src["envelope"]
        vertex = np.array(original["node_vertex_indices"], dtype=int)
        if vertex.shape != (4,) or not np.allclose(
            x[vertex], original["node_local_points_m"], atol=1e-12, rtol=0
        ):
            raise ValueError("Unmapped source footprint")
        node = y[vertex]
        rad = float(np.linalg.norm(node - e["nearest_local_point_m"], axis=1).max())
        # Material identities survive a rigid frame change. Recompute all traction
        # maps in the receiving frame through the same force-envelope owner.
        maps, metrics = owner._tendon_envelope_force_maps(
            point.tolist(), node.tolist(), rad
        )
        if (
            rad > settings["maximum_patch_radius_m"]
            or metrics["force_residual"] > 2e-6
            or metrics["moment_residual_m"] > 2e-8
            or metrics["sampled_total_force_amplification"]
            > settings["maximum_sampled_total_force_amplification"]
        ):
            continue
        e.update(
            node_vertex_indices=vertex.tolist(),
            node_local_points_m=node.tolist(),
            surface_patch_method=original["surface_patch_method"],
            force_maps=maps,
            patch_radius_m=rad,
            **metrics,
        )
        for k in ("node_surface_sources", "connected_face_search"):
            e.pop(k, None)
        sites = np.vstack([point, e["node_local_points_m"]])
        _, ids, centers = igl.point_mesh_squared_distance(sites, y, f)
        normals = np.cross(y[f[ids, 1]] - y[f[ids, 0]], y[f[ids, 2]] - y[f[ids, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[
            np.abs(igl.winding_number(y, f, centers + 0.00005 * normals)) > 0.5
        ] *= -1
        rays = np.ascontiguousarray(
            (centers[:, None] + normals[:, None] * depths[None, :, None]).reshape(-1, 3)
        )

        def winding(mesh):
            return (
                abs(
                    igl.winding_number(
                        np.ascontiguousarray(mesh[0]),
                        np.ascontiguousarray(mesh[1], dtype="i4"),
                        rays,
                    )
                ).reshape(len(sites), -1)
                > 0.5
            )

        support = winding(atlas) & ~winding((y, f))
        by_pose = {}
        for step, (muscle, sub) in meshes.items():
            by_pose[step] = (support & winding(muscle) & ~winding(sub)).any(1)
        proven = np.asarray(list(by_pose.values())).all(0)
        # Different geodesic compass nodes are allowed after a frame rotation.
        # The receiving compiler must independently qualify all four new sites.
        k = len(opts)
        opts.append(
            dict(
                point=point.tolist(),
                envelope=e,
                active_nodes_m=e["node_local_points_m"],
                cortical_gap_m=float(gap(sites[:1])[0]),
                donor_proposal=i,
                transferred_center_error_m=float(np.linalg.norm(point - dest)),
            )
        )
        res.append(
            dict(
                proposal=k,
                center_proven=bool(proven[0]),
                proven_source_sites=proven.tolist(),
                max_cortical_gap_m=float(gap(sites).max()),
                observed_captures={i: q.tolist() for i, q in by_pose.items()},
            )
        )
    rec = dict(
        launch=digest(target / "launch.json"),
        bone=digest(bone),
        muscle=digest(mp),
        tendon=digest(tp),
        tendon_manifest=digest(tm),
        tissue=digest(tissue),
        source_shape=b["source_shape"],
        source_site_id=662,
        via_site_id=661,
        endpoint_index=index,
        route_index=end,
        via=via.tolist(),
        old_site=old.tolist(),
        original_length_m=length,
        original_cortical_gap_m=float(gap(old[None])[0]),
        proposals=opts,
        donor_registration=digest(registered / "candidates.json"),
        rigid_fit_max_m=err,
        via_and_site_error_m=site_errors.tolist(),
        captured_pose_receipts=[digest(p) for p in receipts],
    )
    save(out / "candidates.json", rec)
    save(out / "facing.json", dict(source=digest(out / "candidates.json"), rows=res))
    print(
        "rigid registration",
        err,
        "tested",
        len(res),
        "anatomically passing",
        sum(all(q["proven_source_sites"]) for q in res),
        flush=True,
    )


if __name__ == "__main__":
    transfer(*(Path(v).resolve() for v in sys.argv[1:]))
