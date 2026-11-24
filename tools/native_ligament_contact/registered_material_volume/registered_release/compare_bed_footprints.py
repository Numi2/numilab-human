"""Locate reduced-scene bed reactions on the full registered skin through matched donor frames."""

import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from live_pose_frames import current_donors
from compile_connected_joint import digest


def skin_height(triangles, sites):
    # Intersect the ray through each site with the closed triangles projected on the bed.
    a, b, c = triangles.transpose(1, 0, 2)
    ab = b - a
    ac = c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    good = det != 0.0
    for site in sites:
        dx, dy = (site[:2] - a[:, :2]).T
        u = np.divide(
            dx * ac[:, 1] - dy * ac[:, 0], det, out=np.zeros_like(det), where=good
        )
        v = np.divide(
            ab[:, 0] * dy - ab[:, 1] * dx, det, out=np.zeros_like(det), where=good
        )
        hits = np.flatnonzero(
            good & (u >= -1.0e-11) & (v >= -1.0e-11) & (u + v <= 1.0 + 1.0e-11)
        )
        if not hits.size:
            yield None
            continue
        z = a[hits, 2] + u[hits] * ab[hits, 2] + v[hits] * ac[hits, 2]
        k = np.argmin(z)
        yield dict(
            face=int(hits[k]),
            weights=[
                float(1.0 - u[hits[k]] - v[hits[k]]),
                float(u[hits[k]]),
                float(v[hits[k]]),
            ],
            height_m=float(z[k]),
        )


def compare(reference, current, volume, support_file, output):
    support = json.loads(support_file.read_bytes())
    launcher = json.loads((current / "launch.json").read_bytes())
    if digest(current / "launch.json") != support["launch_sha256"] or digest(
        volume / "current.node"
    ) != digest(
        Path(launcher["environment"]["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    ):
        raise ValueError("Mismatched material or support receipt")
    receipt = reference / "step-3999.receipt.json"
    selected = reference / "resting-support-selected-skin-witness.csv"
    impulses = reference / "resting-com-support-impulses.csv"
    poses_file, donors = current_donors(current)
    source = json.loads(receipt.read_bytes())
    s = source["accepted_step"]
    if source["physical_endpoint"] != "accepted":
        raise ValueError("Missing accepted rigid state")
    src = {p["body_index"]: p for p in source["accepted_body_poses"]}
    dst = {
        p["body_index"]: p["current"]
        for p in donors
    }
    argv = json.loads((reference / "launch.json").read_bytes())["argv"]
    dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    normal = np.asarray(support["bed_normal"])
    e = np.cross(normal, [0.0, 1.0, 0.0])
    e /= np.linalg.norm(e)
    frame = np.stack([e, np.cross(normal, e), normal], axis=1)
    origin = normal * support["source_plane_offset_m"] / (normal @ normal)
    xyz = np.loadtxt(volume / "current.node", skiprows=1, usecols=(1, 2, 3))
    tris = np.load(volume / "material-walls.npz")["O"]
    projected = ((xyz - origin) @ frame)[tris]
    rows = []
    sites = []
    with selected.open() as a, impulses.open() as b:
        for x, y in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
            if int(y["accepted_step"]) != s + 1:
                continue
            if (
                int(x["pre_dynamics_step_index"]) != s
                or int(x["accepted_step"]) != s + 1
                or x["region"] != y["contact_index"]
                or x["source_geometry_index"] != y["source_geometry_index"]
                or y["row_class"] != "external_support_contact"
            ):
                raise ValueError("Unsynchronized witness/impulse")
            f = float(y["normal_impulse_ns"]) / dt
            if f < 0.0 or not np.isfinite(f):
                raise ValueError("Invalid normal contact impulse")
            if f == 0.0:
                continue
            owner = int(y["body_index"])
            site = np.array(
                [float(x["selected_point_world_" + axis + "_m"]) for axis in "xyz"]
            )
            record = dict(
                owner=owner,
                source_site=int(y["source_geometry_index"]),
                normal_N=f,
                registered_owner=owner in dst,
            )
            rows.append(record)
            if owner not in dst:
                continue
            p, q = src[owner], dst[owner]
            local = (
                Rotation.from_quat(p["quaternion_xyzw"])
                .inv()
                .apply(site - p["position_m"])
            )
            transported = (
                Rotation.from_quat(q["orientation_world_xyzw"]).apply(local)
                + q["position_world_m"]
            )
            record["rigid_transport_m"] = float(np.linalg.norm(transported - site))
            record["transported_site_m"] = transported.tolist()
            sites.append((transported - origin) @ frame)
    for row, hit in zip(
        (q for q in rows if q["registered_owner"]),
        skin_height(projected, sites),
        strict=True,
    ):
        row["lowest_surface"] = hit
    total = sum(q["normal_N"] for q in rows)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Reference has no supported load")
    summary = dict(
        total_reference_N=total,
        missing_owner_N=sum(q["normal_N"] for q in rows if not q["registered_owner"]),
        rays_missing_skin_N=sum(
            q["normal_N"]
            for q in rows
            if q["registered_owner"] and q["lowest_surface"] is None
        ),
        at_or_below_m={
            str(h): sum(
                q["normal_N"]
                for q in rows
                if q.get("lowest_surface") and q["lowest_surface"]["height_m"] <= h
            )
            for h in (0.002, 0.005, 0.02, 0.04)
        },
    )
    result = dict(
        pre_dynamics_step=s,
        reference_load_step=s + 1,
        hashes={
            str(p.name): digest(p)
            for p in (
                receipt,
                selected,
                impulses,
                poses_file,
                volume / "current.node",
                volume / "material-walls.npz",
                support_file,
            )
        },
        interpretation="Source support sites transported by their registered rigid owners. Local skin deformation and mixed skinning can change site location. Heights are exact section queries of the current triangulated exterior.",
        summary=summary,
        sites=rows,
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(summary)


if __name__ == "__main__":
    compare(*(Path(s).resolve() for s in sys.argv[1:]))
