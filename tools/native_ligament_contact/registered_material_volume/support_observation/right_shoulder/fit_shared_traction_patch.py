"""Restrict force-envelope search to cortex facing the measured muscle at every retained pose."""

import copy
import csv
import sys
from pathlib import Path

import igl
import numpy as np
from assemble_material import frames
from audit_registered_skeleton import source_geometry
from compile_connected_joint import digest
from numilab_human import model as owner
from rebind_ligament_material import banks
from registration_stages import parts
from repair_registered_surface import state

from transfer_cortical_origin import load, local, save


def fit(w, dense, moving, src, out, motion=None):
    a, c = load(dense / "launch.json"), load(moving / "launch.json")
    prev = load(src / "candidates.json")
    prior = load(src / "facing.json")
    if prev["launch"] != digest(moving / "launch.json") or prior["source"] != digest(
        src / "candidates.json"
    ):
        raise ValueError("Incompatible motion source")
    ca, ma = a["argv"], c["argv"]
    sf, x, _ = banks(Path(ca[3]))[34, 10]
    f, y, _ = banks(Path(ma[3]))[34, 10]
    f = f.astype("i4")
    u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    R = u @ vt
    t = y.mean(0) - x.mean(0) @ R
    if not np.array_equal(sf, f) or np.linalg.norm(x @ R + t - y, axis=1).max() > 1e-6:
        raise ValueError("Cortical correspondence failed")
    base = frames(dense)[-1]
    reg = load(w / "source-support-column.json")
    g = reg["frame"]
    at = w / "atlas/FJ1500.obj"
    if digest(at) != prev["source_shape"]:
        raise ValueError("Unmatched muscle provenance")
    xyz, faces = source_geometry(at.read_bytes())
    atlas = (
        local(xyz @ g["linear_m_per_mm"] + g["translation_m"], base[34]) @ R + t,
        faces,
    )

    def objects(argv, pose, donor):
        source = Path(argv[argv.index("--soft-tissue-payload") + 1])
        res = []
        for sid in [83, 85]:
            g, wj, b, face, aff = parts(source, sid)
            v = local(state(g, wj, aff, b, pose), pose[34])
            res.append((v @ R + t if donor else v, face))
        return res

    meshes = [objects(ca, base, True)]
    for p in sorted(
        (moving / "native-traces/accepted-geometry").glob("*.receipt.json")
    ):
        meshes.append(
            objects(
                ma, {q["body_index"]: q for q in load(p)["accepted_body_poses"]}, False
            )
        )
    if motion is not None:
        temporal = {}
        for row in csv.DictReader(
            (motion / "native-traces/resting-body-motion-partition.csv").open()
        ):
            step = int(row["accepted_step"])
            bid = int(row["body_index"])
            if step not in (
                32,
                64,
                96,
                128,
                160,
                192,
                224,
                256,
                320,
                480,
                640,
                1024,
                2048,
                4096,
                6144,
                8000,
            ) or bid not in (25, 34, 41):
                continue
            pos = [
                float(row[f"root_{u}_m"]) + float(row[f"com_relative_root_{u}_m"])
                for u in "xyz"
            ]
            quat = [float(row[f"orientation_{u}"]) for u in "xyzw"]
            temporal.setdefault(step, {})[bid] = dict(
                position_m=pos, quaternion_xyzw=quat
            )
        if len(temporal) != 16:
            raise ValueError("Incomplete moving reference")
        meshes.extend(objects(ma, pp, False) for pp in temporal.values())
    depths = 0.001 * np.array([0.05, 0.25, 0.5, 1, 2, 3, 5, 8])

    def coverage(points):
        points = np.ascontiguousarray(points, dtype=float)
        distance, tri, centers = igl.point_mesh_squared_distance(points, y, f)
        normals = np.cross(y[f[tri, 1]] - y[f[tri, 0]], y[f[tri, 2]] - y[f[tri, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[
            np.abs(igl.winding_number(y, f, centers + 0.00005 * normals)) > 0.5
        ] *= -1
        rays = np.ascontiguousarray(
            (centers[:, None] + normals[:, None] * depths[None, :, None]).reshape(-1, 3)
        )

        def inside(m):
            return (
                abs(
                    igl.winding_number(
                        np.ascontiguousarray(m[0]),
                        np.ascontiguousarray(m[1], dtype="i4"),
                        rays,
                    )
                ).reshape(len(points), -1)
                > 0.5
            )

        ref = inside(atlas) & ~inside((y, f))
        return np.array(
            [(ref & inside(m) & ~inside(s)).any(1) for m, s in meshes]
        ), np.sqrt(distance)

    cfg = load(
        Path(ma[ma.index("--tendon-payload") + 1]).with_suffix(".manifest.json")
    )["admission"]
    proposals = []
    observations = []
    trials = []
    options = [
        v
        for v, z in zip(prev["proposals"], prior["rows"], strict=True)
        if z["center_proven"]
    ]
    if motion is not None:
        from register_deltoid_origins import cortical_circle

        seed = np.array(options[0]["point"])
        via = np.array(prev["via"])
        radius = prev["original_length_m"]
        _, all_points = cortical_circle(y, f, via, radius, seed)
        points = np.unique(np.asarray(all_points, dtype="f4"), axis=0).astype(float)
        points = points[np.linalg.norm(points - seed, axis=1) < 0.026]
        good = coverage(points)[0].all(0)
        points = points[good]
        candidates_order = np.argsort(np.linalg.norm(points - seed, axis=1))
        options = [
            dict(point=points[j].tolist(), donor_proposal=None)
            for j in candidates_order
        ]
        print("shared centers", len(options), flush=True)
    if options:
        near = np.zeros(len(y), bool)
        for v in options:
            near |= (
                np.linalg.norm(y - v["point"], axis=1)
                <= cfg["maximum_patch_radius_m"] + 1e-7
            )
        ok = np.zeros(len(y), bool)
        ok[near] = coverage(y[near])[0].all(0)
    for srcrow in options:
        p = np.array(srcrow["point"])
        candidates = np.flatnonzero(
            np.linalg.norm(y - p, axis=1) <= cfg["maximum_patch_radius_m"] + 1e-7
        )
        mask = np.zeros(len(y), bool)
        mask[candidates] = ok[candidates]
        face_ids = np.flatnonzero(mask[f].all(1))
        record = dict(
            donor_proposal=srcrow["donor_proposal"],
            candidate_vertices=len(candidates),
            qualified_vertices=int(mask.sum()),
            qualified_faces=len(face_ids),
        )
        trials.append(record)
        if not len(face_ids):
            continue
        surf = dict(
            body_index=34,
            stable_id=10,
            member_id="FJ3384",
            vertices=y.tolist(),
            triangles=f[face_ids].tolist(),
        )
        e, reason = owner._numi_human_tendon_surface_envelope_connected(
            p.tolist(),
            surf,
            cfg["maximum_surface_distance_m"],
            cfg["maximum_patch_radius_m"],
            cfg["maximum_sampled_total_force_amplification"],
        )
        record["compiler"] = reason
        if (
            e is None
            or e["surface_patch_method"] != "connected_geodesic_compass_vertices"
        ):
            continue
        e["source_triangle_index"] = int(face_ids[e["source_triangle_index"]])
        if np.linalg.norm(np.asarray(e["nearest_local_point_m"]) - p) > 1e-6:
            continue
        sites = np.vstack([p, e["node_local_points_m"]])
        ck, dist = coverage(sites)
        if not ck.all() or max(dist) > 1e-6:
            continue
        i = len(proposals)
        d = copy.deepcopy(srcrow)
        d.update(
            envelope=e,
            active_nodes_m=e["node_local_points_m"],
            cortical_gap_m=float(dist[0]),
        )
        proposals.append(d)
        observations.append(
            dict(
                proposal=i,
                center_proven=True,
                proven_source_sites=ck.all(0).tolist(),
                max_cortical_gap_m=float(dist.max()),
                held_pose_passes=ck.tolist(),
            )
        )
        if motion is not None:
            break
    out.mkdir(exist_ok=True)
    prev["proposals"] = proposals
    prev["traction_search"] = trials
    if motion is not None:
        prev["trajectory_receipt"] = digest(motion / "launch.json")
    save(out / "candidates.json", prev)
    save(
        out / "facing.json",
        dict(source=digest(out / "candidates.json"), rows=observations),
    )
    print(trials, len(proposals), flush=True)


if __name__ == "__main__":
    fit(*(Path(v).resolve() for v in sys.argv[1:]))
