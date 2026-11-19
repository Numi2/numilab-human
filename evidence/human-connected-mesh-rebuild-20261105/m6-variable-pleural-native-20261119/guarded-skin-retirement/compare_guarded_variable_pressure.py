"""Measure skin progress and physical parity in the matched variable-pressure scene."""
import argparse
import json
from pathlib import Path

from fit_cardiac_volume_grid import HERE, hash_file


def read(path):
    return json.loads(path.read_text())


def pairs(report):
    result = {}
    for r in report["surfaces"]:
        key = (r["semantic"], r["stable_id"])
        assert key not in result
        result[key] = {tuple(p) for p in r["triangle_pairs"]}
        assert len(result[key]) == r["count"]
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--variant",choices=("guarded","trimmed-guarded"),default="guarded")
    args=cli.parse_args()
    label = "cardiac-split-native-8000"
    base = HERE / label
    guarded = HERE / (label+"-"+args.variant)
    native = [r/"native" for r in (base,guarded)]
    receipts = [read(p/"accepted-geometry/step-8000.receipt.json") for p in native]
    a,b = receipts
    assert all(x["accepted_step"]==8000 and x["physical_endpoint"]=="accepted" for x in receipts)
    checks = {
        key:a[key]==b[key] for key in ("accepted_body_state_sha256", "accepted_respiration_state_sha256",
            "accepted_root_fingerprint_hex", "accepted_time_s", "common_field_map_sha256")
    }
    for key in ("cardiac_q","cardiac_wall_q","common_coordinates","common_position_low"):
        checks[key] = a["accepted_cardiac_geometry_buffers"][key]["sha256"] == b["accepted_cardiac_geometry_buffers"][key]["sha256"]
    for name in ("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv",
                 "resting-com-momentum-diagnostic.csv"):
        checks[name] = hash_file(native[0]/name) == hash_file(native[1]/name)
    aa,bb=(read(HERE/(tag+"-skin-exact.json")) for tag in (base.name,guarded.name))
    assert aa["accepted_pack_sha256"]==a["pack_file_sha256"] and bb["accepted_pack_sha256"]==b["pack_file_sha256"]
    c,d=pairs(aa),pairs(bb)
    assert c.keys()==d.keys()
    c_self,d_self=({tuple(p) for p in r["self_intersections"]["triangle_pairs"]} for r in (aa,bb))
    summary=dict(tissue_pairs_before=sum(map(len,c.values())),tissue_pairs_after=sum(map(len,d.values())),
                 new_tissue_pairs=sum(len(d[k]-c[k]) for k in c),
                 removed_tissue_pairs=sum(len(c[k]-d[k]) for k in c),
                 skin_self_pairs_before=len(c_self),skin_self_pairs_after=len(d_self),
                 new_skin_self_pairs=len(d_self-c_self), physical_byte_parity=checks)
    (HERE/(args.variant+"-variable-pressure-comparison.json")).write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)


if __name__=="__main__":
    main()
