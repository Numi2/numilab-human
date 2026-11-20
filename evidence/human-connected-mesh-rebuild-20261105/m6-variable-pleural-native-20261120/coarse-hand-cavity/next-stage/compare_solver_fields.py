"""Apply the retained regional force and strain comparisons to solver trials."""
from argparse import ArgumentParser
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_remeshed_native_310 import capture, delta, force_stats, strain_stats


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Gzipped native endpoint log")
    parser.add_argument("candidate", type=Path, help="Gzipped native endpoint log")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    baseline, candidate = (capture(path) for path in (args.baseline, args.candidate))
    for key in ("steps", "dt_s", "runtime_dt_s", "runtime_coordinate_projection"):
        assert baseline["header"][key] == candidate["header"][key], key
    assert baseline["objects"] == candidate["objects"]
    for key in ("region", "obj", "tag", "tets", "treg", "tobj"):
        assert np.array_equal(baseline[key], candidate[key]), key
    strain, forces = [], []
    strains = [strain_stats(state) for state in (baseline, candidate)]
    resultants = [force_stats(state) for state in (baseline, candidate)]
    for region, reference in strains[0].items():
        current = strains[1][region]
        for metric in ("volume_weighted_strain_norm", "volume_weighted_strain_squared",
                       "maximum_green_strain_norm"):
            a, b = reference[metric], current[metric]
            changed = delta(a, b)
            if changed is None or changed > .01:
                strain.append({"region": region, "metric": metric, "reference": a,
                               "candidate": b, "relative_change": changed})
    for region, reference in resultants[0].items():
        for channel, a in reference.items():
            b = resultants[1][region][channel]
            resultant = float(np.linalg.norm(np.subtract(b["resultant_N"], a["resultant_N"])))
            l1 = delta(a["nodal_l1_N"], b["nodal_l1_N"])
            relative = resultant / a["nodal_l1_N"] if a["nodal_l1_N"] else (
                0. if resultant == 0. else None)
            if any(value is None or value > .01 for value in (relative,l1)):
                forces.append({"region": region, "channel": channel, "reference": a,
                               "candidate": b, "resultant_delta_N": resultant,
                               "resultant_relative_to_reference_l1": relative,
                               "nodal_l1_relative_change": l1})
    result = {
        "region_count": len(strains[0]),
        "exact_element_connectivity": True,
        "reference_positions_match": bool(np.array_equal(baseline["rest"], candidate["rest"])),
        "nodal_masses_match": bool(np.array_equal(baseline["mass"], candidate["mass"])),
        "fem_mass_change_kg": float(candidate["mass"].sum()-baseline["mass"].sum()),
        "max_accepted_position_difference_m": float(np.linalg.norm(
            candidate["rest"]+candidate["u"]-baseline["rest"]-baseline["u"],axis=1).max()),
        "force_regions_over_1pct": forces,
        "strain_metrics_over_1pct": strain,
        "strain_metrics_over_5pct": [
            row for row in strain if row["relative_change"] is None or row["relative_change"] > .05],
        "scope": "Comparison of captured FEM force channels and regional strain metrics. It omits force owners absent from the endpoint capture.",
    }
    with args.report.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({**result,"force_regions_over_1pct":len(forces),
                      "strain_metrics_over_1pct":len(strain),
                      "strain_metrics_over_5pct":len(result["strain_metrics_over_5pct"])}))


if __name__ == "__main__":
    main()
