"""Use the same four bodypart vertices and physical line of action in the dense skin/fat scene."""

import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest
from numilab_human import model as owner
from rebind_ligament_material import banks

from certify_enthesis_side import test
from install_scapular_enthesis import build
from transfer_cortical_origin import load, save


def bind(w, dense, regional, source, shared, out):
    donor = load(source / "candidates.json")
    s = load(shared / "candidates.json")
    r = load(shared / "native-owner/registration.json")
    if (
        r["facing_report"] != digest(shared / "facing.json")
        or donor["launch"] != digest(dense / "launch.json")
        or s["launch"] != digest(regional / "launch.json")
    ):
        raise ValueError("Unmatched physical owner")
    sp = s["proposals"][r["candidate"]]
    c, q = load(dense / "launch.json"), load(regional / "launch.json")
    f, x, _ = banks(Path(c["argv"][3]))[34, 10]
    ff, y, _ = banks(Path(q["argv"][3]))[34, 10]
    if not np.array_equal(f, ff):
        raise ValueError("Cortex no longer corresponds")
    u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    R = u @ vt
    t = y.mean(0) - x.mean(0) @ R
    rp = np.array(sp["point"])
    if sp["donor_proposal"] is None:
        from register_deltoid_origins import cortical_circle

        estimate = (rp - t) @ R.T
        _, points = cortical_circle(
            x, f, np.array(donor["via"]), donor["original_length_m"], estimate
        )
        q = np.asarray(points)
        p = q[np.linalg.norm(q - estimate, axis=1).argmin()].astype("f4").astype(float)
        cfg = load(
            Path(c["argv"][c["argv"].index("--tendon-payload") + 1]).with_suffix(
                ".manifest.json"
            )
        )["admission"]
        mesh = dict(
            body_index=34,
            stable_id=10,
            member_id="FJ3384",
            vertices=x.tolist(),
            triangles=f.tolist(),
        )
        e0, _ = owner._numi_human_tendon_surface_envelope_connected(
            p.tolist(),
            mesh,
            cfg["maximum_surface_distance_m"],
            cfg["maximum_patch_radius_m"],
            cfg["maximum_sampled_total_force_amplification"],
        )
        if e0 is None:
            raise ValueError("Unadmitted receiving attachment")
        dp = dict(
            point=p.tolist(), cortical_gap_m=e0["surface_distance_m"], envelope=e0
        )
    else:
        dp = donor["proposals"][sp["donor_proposal"]].copy()
        p = np.array(dp["point"])
    frame_error = float(np.linalg.norm(x @ R + t - y, axis=1).max())
    point_error = float(np.linalg.norm(p @ R + t - rp))
    if np.linalg.det(R) < 0.999999 or max(frame_error, point_error) > 1e-6:
        raise ValueError("Attachment did not transport rigidly")
    e = dp["envelope"].copy()
    oe = sp["envelope"]
    verts = np.array(oe["node_vertex_indices"])
    nodes = x[verts]
    rad = float(np.linalg.norm(nodes - e["nearest_local_point_m"], axis=1).max())
    settings = load(
        Path(c["argv"][c["argv"].index("--tendon-payload") + 1]).with_suffix(
            ".manifest.json"
        )
    )["admission"]
    maps, met = owner._tendon_envelope_force_maps(p.tolist(), nodes.tolist(), rad)
    if (
        rad > settings["maximum_patch_radius_m"]
        or met["force_residual"] > 2e-6
        or met["moment_residual_m"] > 2e-8
        or met["sampled_total_force_amplification"]
        > settings["maximum_sampled_total_force_amplification"]
    ):
        raise ValueError("Traction maps failed native admission")
    map_error = float(
        np.linalg.norm(
            R.T @ np.array(maps) @ R - np.array(oe["force_maps"]), axis=(1, 2)
        ).max()
    )
    if map_error > 1e-5:
        raise ValueError("Local forces are not frame-equivalent")
    e.update(
        node_vertex_indices=verts.tolist(),
        node_local_points_m=nodes.tolist(),
        force_maps=maps,
        patch_radius_m=rad,
        **met,
    )
    dp.update(envelope=e, active_nodes_m=nodes.tolist())
    for k in [
        "registered_envelope_muscle_distances_mm",
        "atlas_envelope_muscle_distances_mm",
    ]:
        dp.pop(k, None)
    donor.update(
        proposals=[dp],
        shared_traction=dict(
            foreign_registration=digest(shared / "native-owner/registration.json"),
            stable_vertices=verts.tolist(),
            site_error_m=point_error,
            cortical_frame_error_m=frame_error,
            maximum_force_map_frame_error=map_error,
        ),
    )
    out.mkdir(exist_ok=True)
    save(out / "candidates.json", donor)
    test(w, dense, out, out / "facing.json")
    build(dense, out, out / "native")


if __name__ == "__main__":
    bind(*(Path(v).resolve() for v in sys.argv[1:]))
