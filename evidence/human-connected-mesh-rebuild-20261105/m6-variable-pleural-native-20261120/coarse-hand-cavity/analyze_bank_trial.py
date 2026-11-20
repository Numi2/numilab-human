"""Report complete registered solid-interior witnesses from native preparation."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from build_hand_trial import digest


def summarize(trial):
    path = trial / "skin-bone-witness.csv"
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    nodes = {}
    with (trial / "initial-skin-fem.csv").open(newline="") as handle:
        for node in csv.DictReader(handle):
            nodes[int(node["global_node"])] = node
    banks = defaultdict(list)
    nonpenetrating = []
    for r in records:
        if r["inside_rays"] != "3" or r["ambiguous"] != "0":
            nonpenetrating.append(r)
            continue
        these = banks[(int(r["body"]), int(r["bone"]))]
        these.append(r)
    priorities = []
    for (body, bone), rows in banks.items():
        rows.sort(key=lambda r: -float(r["force_N"]))
        r = rows[0]
        priorities.append({
            "body":body,"bone":bone,"inside_pairs":len(rows),
            "counts_by_layer":dict(Counter(int(k["layer"]) for k in rows)),
            "load_sample_pairs":sum(int(k["has_contact_sample"]) for k in rows),
            "largest_depth_mm":1000*max(float(k["closest_distance_m"]) for k in rows),
            "peak_penalty_force_N":float(r["force_N"]),
            "highest_force_point":{"row":int(r["row"]),"layer":int(r["layer"]),
                "world_m":[float(nodes[int(r["node"])][f"{a}_m"]) for a in "xyz"]},
        })
    priorities.sort(key=lambda r: (-r["peak_penalty_force_N"],r["body"],r["bone"]))
    log = (trial/"stdout.log").read_text()
    m = re.search(r"^initial_skin_bone_solid_audit (.+)$",log,re.MULTILINE)
    assert m is not None
    counts = dict(item.split("=",1) for item in m[1].split())
    for field in ("eligible_banks","total_points","aabb_point_bank_tests","inside_pairs","ambiguous_pairs"):
        counts[field] = int(counts[field])
    assert counts["inside_pairs"] == sum(r["inside_pairs"] for r in priorities)
    assert counts["ambiguous_pairs"] == len(nonpenetrating)
    return {"launch_sha256":digest(trial/"launch.json"),
            "solid_audit_sha256":digest(path),
            "initial_skin_nodes_sha256":digest(trial/"initial-skin-fem.csv"),
            "initial_native_skin_fat_mass_kg":sum(float(r["mass_kg"]) for r in nodes.values()),
            "execution":json.loads((trial/"execution.json").read_text()),
            "coverage":counts,"bank_priority":priorities,"ambiguous":nonpenetrating,
            "scope":"Native Float32 initial outer and deep boundary point parity. Edge/triangle crossings and moving positions require independent checks."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial",type=Path)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    with args.out.open("x") as handle:
        json.dump(summarize(args.trial),handle,indent=2,allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
