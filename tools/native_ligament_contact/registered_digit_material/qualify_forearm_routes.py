"""Locate named force sites against the skin that actually surrounds the current skeleton."""

import json, runpy, sys
from pathlib import Path
from scipy.spatial.transform import Rotation
from registered_forearms import read, sha, native_sites
from verify_native_extensor_pair import loaded
from measure_forearm_enclosure import Enclosure


def qualify(case, reader, lib, dense=False):
    j = read(case / "launch.json")
    arg = j["argv"]
    myo = Path(arg[2])
    tendon = Path(arg[arg.index("--tendon-payload") + 1])
    tm = read(tendon.with_suffix(".manifest.json"))
    if (
        sha(myo) != j["pins"][str(myo)]
        or tm["payload"]["sha256"] != sha(tendon)
        or tm["source"]["myosim_muscle_payload_sha256"] != sha(myo)
    ):
        raise ValueError("Mismatched physical routes")
    tissue = Path(arg[arg.index("--soft-tissue-payload") + 1])
    manifest = read(tissue.with_suffix(".manifest.json"))
    if (
        sha(tissue) != j["pins"][str(tissue)]
        or sha(tissue) != manifest["payload"]["sha256"]
    ):
        raise ValueError("Mismatched registered muscle")
    source = myo.read_bytes()
    endpoints = {(r["muscle"], r["endpoint"]): r for r in tm["endpoints"]}
    receipt, pose, raw = loaded(case, False, dense)
    sys.path.insert(0, str(reader.parent / "src"))
    prim = runpy.run_path(str(reader))["primitives"]
    ((_, _, v, f),) = prim(raw, pose, skin_only=True)
    skin = Enclosure(lib, v, f)
    banks = []
    try:
        for member in manifest["source"]["surfaces"]:
            if member["stable_id"] not in (109, 110, 137, 138):
                continue
            sites = []
            for line in member["matched_muscles"]:
                items = native_sites(
                    source,
                    line["name"],
                    line["source_actuator_index"],
                    line["source_route_node_count"],
                    endpoints,
                )
                for k, s in enumerate(items):
                    a = pose[s["body"]]
                    p = (
                        Rotation.from_quat(a["quaternion_xyzw"]).apply(s["local_m"])
                        + a["position_m"]
                    )
                    label, d = skin.query([p])
                    sites.append(
                        dict(
                            muscle=line["name"],
                            role="origin"
                            if k == 0
                            else "insertion"
                            if k == len(items) - 1
                            else "intermediate",
                            source_site=s["site_index"],
                            body=s["body"],
                            local_m=s["local_m"],
                            executed_world_m=p.tolist(),
                            skin_enclosure=int(label[0]),
                            skin_distance_mm=float(d[0] * 1000),
                        )
                    )
            banks.append(
                dict(
                    stable=member["stable_id"],
                    name=member["label"],
                    distinct_sites=len({s["source_site"] for s in sites}),
                    sites=sites,
                )
            )
    finally:
        skin.close()
    return dict(
        pack=receipt["pack_file_sha256"],
        accepted_s=receipt["accepted_time_s"],
        myosim=sha(myo),
        tendon=sha(tendon),
        tissue=sha(tissue),
        banks=banks,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            qualify(
                *(Path(v).resolve() for v in sys.argv[1:4]),
                dense="--dense" in sys.argv[4:],
            ),
            indent=2,
        )
    )
