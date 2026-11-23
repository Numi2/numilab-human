"""Check the actual distributed muscle attachment patches on their named bones and enclosing skin."""

import argparse
import json, runpy, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from registered_forearms import read, sha
from track_elbow_attachments import locate
from measure_forearm_enclosure import Enclosure
from verify_native_extensor_pair import loaded
from connected_elbow import middle


def verify(case, reader, library, trace, intermediate=False):
    r, pose, raw = middle(case) if intermediate else loaded(case, False, False)
    j = read(case / "launch.json")
    args = j["argv"]
    tendon = Path(args[args.index("--tendon-payload") + 1])
    meta = read(tendon.with_suffix(".manifest.json"))
    source = read(trace)
    if (
        meta["payload"]["sha256"] != sha(tendon)
        or source["tendon"] != sha(tendon)
        or source["myosim"] != sha(Path(args[2]))
    ):
        raise ValueError("Changed physical route binding")
    sites = {
        (v["muscle"], v["role"]): v
        for k in source["banks"]
        for v in k["sites"]
        if v["role"] in ("origin", "insertion")
    }
    selected = [
        row for row in meta["endpoints"] if (row["muscle"], row["endpoint"]) in sites
    ]
    if len(selected) != len(sites):
        raise ValueError("Missing tendon endpoint")
    ids = {s["surface"]["bone_stable_id"] for s in selected if s.get("surface")}
    sys.path.insert(0, str(reader.parent / "src"))
    gen = runpy.run_path(str(reader))["primitives"]
    meshes = {
        k: (v, f)
        for k, _, v, f in gen(raw, pose)
        if k == (51007, 1) or (k[0] == 51004 and k[1] in ids)
    }
    shell = Enclosure(library, *meshes.pop((51007, 1)))
    bones = {sid[1]: Enclosure(library, *m) for sid, m in meshes.items()}
    records = []
    try:
        for row in selected:
            physical = sites[row["muscle"], row["endpoint"]]
            if (
                row["source_site_index"] != physical["source_site"]
                or row["body_index"] != physical["body"]
            ):
                raise ValueError("Physical source index drift")
            surf = row.get("surface")
            rec = dict(
                muscle=row["muscle"],
                role=row["endpoint"],
                site=row["source_site_index"],
                body=row["body_index"],
                attachment_mode=row["attachment_mode"],
            )
            if surf:
                sid = surf["bone_stable_id"]
                rec["bone_stable_id"] = sid
                local = np.array(
                    [
                        row["resolved_local_point_m"],
                        surf["nearest_local_point_m"],
                        *surf["node_local_points_m"],
                    ]
                )
                p = pose[row["body_index"]]
                world = (
                    Rotation.from_quat(p["quaternion_xyzw"]).apply(local)
                    + p["position_m"]
                )
                inventory, depth = shell.query(world)
                cortex = bones[sid]
                v, f = meshes[(51004, sid)]
                hit, _, _ = locate(cortex, v, f, world)
                rec.update(
                    cortical_patch_nodes=len(local) - 2,
                    patch_skin_class=inventory[2:].tolist(),
                    patch_distance_to_skin_mm=(1000 * depth[2:]).tolist(),
                    max_patch_to_emitted_bone_mm=float(
                        1000 * np.max(np.abs(hit[2:, 0]))
                    ),
                    site_skin_class=int(inventory[0]),
                    site_depth_mm=float(1000 * depth[0]),
                    site_bone_distance_mm=float(1000 * abs(hit[0, 0])),
                    registered_site_to_bone_mm=float(1000 * surf["surface_distance_m"]),
                )
            records.append(rec)
    finally:
        for sk in [shell, *bones.values()]:
            sk.close()
    return dict(
        pack=r["pack_file_sha256"],
        time_s=r["accepted_time_s"],
        tendon=sha(tendon),
        rows=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("case", "reader", "library", "trace"):
        parser.add_argument(k, type=Path)
    parser.add_argument("--intermediate", action="store_true")
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
