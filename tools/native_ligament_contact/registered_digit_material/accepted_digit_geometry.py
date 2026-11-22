"""Use the matched native FEM boundary, articulated bone poses, and compiled MyoSim sites."""

import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from material_context import applied, read_inputs, sha, unit
from digital_footprints import sites
from hand_geometry import Surface


def prepare(run, split, reader, prior, out):
    j = json.loads((run / "launch.json").read_text())
    q = json.loads((run / "admission.json").read_text())
    n = q["accepted_steps"]
    receipt = json.loads(
        (run / f"native/accepted-geometry/step-{n}.receipt.json").read_text()
    )
    nodes = run / "physical-accepted.node"
    state = json.loads((run / "release-skin-parity.json").read_text())
    if (
        sha(nodes) != state["accepted_nodes_sha256"]
        or receipt["pack_file_sha256"] != state["accepted_pack_sha256"]
        or receipt["accepted_step"] != n
    ):
        raise ValueError("Unmatched physical and presented captures")
    poses = np.array(
        [
            r["position_m"] + r["quaternion_xyzw"]
            for r in sorted(
                receipt["accepted_body_poses"], key=lambda r: r["body_index"]
            )
        ]
    )
    manifest = (
        Path(json.loads(split.read_text())["myosim_artifact"])
        / "myosim-fullbody-reference.manifest.json"
    )
    a = j["argv"]
    tendon = Path(a[a.index("--tendon-payload") + 1])
    td = json.loads(tendon.with_suffix(".manifest.json").read_text())
    if td["source"]["bodyparts3d_bone_payload"]["sha256"] != sha(Path(a[3])):
        raise ValueError("Tendon/bone mismatch")
    routes, users, solids, bybody, _ = read_inputs(
        run / "launch.json", manifest, None, accepted_poses=poses
    )
    lookup = {(r["muscle"], r["endpoint"]): r for r in td["endpoints"]}
    data = {}
    names = {}
    meta = {}
    env = j["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def pinned(key, h):
        path = Path(env[prefix + key])
        if sha(path) != env[prefix + h]:
            raise ValueError("Changed registered material source " + key)
        return path

    m = pinned("COMBINED_NODE_MAP_CSV", "COMBINED_NODE_MAP_SHA256")
    layer_ids = {k: {} for k in "OID"}
    with m.open() as f:
        for r in csv.DictReader(f):
            k = r["registered_surface_layer"]
            if k not in layer_ids:
                continue
            i = int(r["registered_surface_vertex_row"])
            node = int(r["joint_node_id"]) - 1
            if layer_ids[k].setdefault(i, node) != node:
                raise ValueError("Unbonded interface")
    xyz = np.loadtxt(nodes, skiprows=1, usecols=(1, 2, 3))
    for k, ids in layer_ids.items():
        rows = np.array([ids[i] for i in range(len(ids))])
        if len(rows) != len(np.unique(rows)):
            raise ValueError("Noninjective boundary")
        f = pinned("SURFACE_" + k + "_FACES_CSV", "SURFACE_" + k + "_FACES_SHA256")
        data[k + "rows"] = rows
        data[k + "v"] = xyz[rows]
        data[k + "f"] = np.loadtxt(
            f, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype="i4"
        )
    # Bind the cortical surface to the actual native pack, with each authored triangle in source order.
    api = runpy.run_path(str(reader))
    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-qdc", str(run / "terminal.mrvpack.zst")]
    )
    import hashlib

    if hashlib.sha256(raw).hexdigest() != receipt["pack_file_sha256"]:
        raise ValueError("Changed accepted pack")
    point = lambda r: np.asarray(r["world"])
    np.vstack([point(site) for route in routes.values() for site in route[-6:]])
    hand_los = []
    hand_his = []
    for side in (False, True):
        hand = np.vstack(
            [
                point(site)
                for name, route in routes.items()
                if name.endswith("_l") == side
                for site in route[-6:]
            ]
        )
        hand_los.append(hand.min(0) - 0.025)
        hand_his.append(hand.max(0) + 0.025)
    cortical_errors = []
    keyposes = {r["body_index"]: r for r in receipt["accepted_body_poses"]}
    for (sem, sid), _, v, f in api["primitives"](raw, keyposes):
        if sem != 51004 or not any(
            np.all(v.min(0) <= hi) and np.all(v.max(0) >= lo)
            for lo, hi in zip(hand_los, hand_his)
        ):
            continue
        if sid not in solids:
            raise ValueError("Unresolved cortical owner " + str(sid))
        x, t = solids[sid]
        rows = np.unique(t)
        if not np.array_equal(f, np.searchsorted(rows, t)):
            raise ValueError("Different bone face layout")
        err = float(np.linalg.norm(v.astype(float) - x[rows], axis=1).max())
        cortical_errors.append(err)
        if err > 2e-6:
            raise ValueError(f"Different cortical pose {sid}, {err}")
        data[f"b{sid}v"] = v
        data[f"b{sid}f"] = f
        solids[sid] = (v.astype(float), f)
        names[sid] = next(
            name for group in bybody.values() for bank, name in group if bank == sid
        )
    for name, r in sorted(routes.items()):
        end = r[-1]
        sid, owner = bybody[end["body"]][0]
        surf = Surface(*solids[sid])
        center = surf.v.mean(0)
        middle = solids[bybody[r[-2]["body"]][0][0]][0].mean(0)
        axis = np.linalg.svd(surf.v - center, full_matrices=False)[2][0]
        if axis @ (center - middle) < 0:
            axis = -axis
        flex = lookup[name.replace("EDC", "FDP"), "insertion"]
        dorsal = end["world"] - applied(
            poses[end["body"]], flex["resolved_local_point_m"]
        )
        dorsal = unit(dorsal - axis * (axis @ dorsal))
        lat = unit(np.cross(axis, dorsal))
        rim, samples = sites(surf, axis, dorsal, lat, center, 0.0)
        target, face, bary = samples[1]
        meta[name] = dict(
            bone=sid,
            owner=owner,
            axis=axis.tolist(),
            dorsal=dorsal.tolist(),
            lateral=lat.tolist(),
            trial_enthesis=dict(
                face=int(face),
                bary=bary.tolist(),
                world_m=target.tolist(),
                arclength_m=0.00122,
            ),
            route=[
                dict(
                    body=s["body"],
                    site=s["site"],
                    world_m=s["world"].tolist(),
                    local_m=s["local"].tolist(),
                    users=users[s["site"]],
                )
                for s in r
            ],
            distal_body_banks=[bybody[s["body"]][0][0] for s in r[-6:]],
        )
    baseline = json.loads(prior.read_text())
    ids = dict(
        native_pack=receipt["pack_file_sha256"],
        accepted_node=sha(nodes),
        bone=sha(Path(a[3])),
        myosim=sha(Path(a[2])),
        tendon=sha(tendon),
        source_manifest=sha(manifest),
    )
    np.savez(out / "physical.npz", **data)
    (out / "geometry.json").write_text(
        json.dumps(
            dict(
                identities=ids,
                accepted_step=n,
                accepted_time_s=receipt["accepted_time_s"],
                previous_footprint_bones_match=ids["bone"] == baseline["bone_sha256"],
                previous_footprint_tendon_matches=ids["tendon"]
                == baseline["tendon_sha256"],
                maximum_cortical_position_error_m=max(cortical_errors),
                bone_names=names,
                digits=meta,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "bones",
        len(names),
        "digits",
        len(meta),
        "max bone replay error m",
        max(cortical_errors),
    )


if __name__ == "__main__":
    prepare(*(Path(v).resolve() for v in sys.argv[1:]))
