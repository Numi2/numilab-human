"""Check every affected native geometry pair and exact parity of the physical prefix."""
from argparse import ArgumentParser
import csv
from itertools import combinations
import json
import mmap
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile

import numpy as np

from prepare import FIT, KEEP, REFERENCE, digest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIT))
from check_surrounding import MUSCLE, SKIN, READER, audit, install_for_float32, triangle_pairs


def captured(run, step):
    pack = run / f"native/accepted-geometry/step-{step}.mrvpack"
    receipt = json.loads(pack.with_suffix(".receipt.json").read_text())
    assert receipt["accepted_step"] == step and receipt["physical_endpoint"] == "accepted"
    assert receipt["surface_audit_endpoint"] == "passed"
    bodies = {row["body_index"]: row for row in receipt["accepted_body_poses"]}
    assert len(bodies) == receipt["accepted_body_pose_count"]
    if not pack.exists():
        archive = Path(str(pack) + ".zst")
        raw = TemporaryFile(dir=HERE)
        with archive.open("rb") as stream:
            subprocess.run(["/opt/homebrew/bin/zstd", "-qdc"], stdin=stream, stdout=raw, check=True)
        raw.seek(0)
    else:
        raw = pack.open("rb")
    with raw:
        from hashlib import file_digest
        assert file_digest(raw, "sha256").hexdigest() == receipt["pack_file_sha256"]
        with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            result = {}
            for key, _, points, faces in READER["primitives"](mapped, bodies):
                assert key not in result
                result[key] = points.copy(), faces.copy()
    return receipt, result


def run_fields(baseline, candidate):
    expected = [
        ("stdout.log", "resting_fem_skin_bed_accepted_root_reaction "),
        ("stdout.log", "resting_fascia_fem_terminal_audit="),
        ("stdout.log", "whole_body_skin_fat_registered_volume_install=PASS"),
        ("stderr.log", "registered_ligament_source_geometry=admitted"),
    ]
    rows = []
    for filename, prefix in expected:
        reference = [line for line in (baseline / filename).read_text().splitlines()
                     if line.startswith(prefix)]
        current = [line for line in (candidate / filename).read_text().splitlines()
                   if line.startswith(prefix)]
        assert len(reference) == len(current) == 1, prefix
        rows.append({"log": filename, "prefix": prefix, "exact": reference == current,
                     "reference": reference[0], "candidate": current[0]})
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    step = 16
    assert json.loads((candidate / "execution.json").read_text())["exit"] == 0
    for run in (REFERENCE, candidate):
        with (run / "native/resting-coupled.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        assert len(trace) == step and list(trace[-1].values())
    channels = ["resting-coupled.csv", "resting-body-motion-partition.csv",
                "resting-com-momentum-diagnostic.csv", "resting-com-support-impulses.csv"]
    traces = {name: {"exact": digest(REFERENCE / "native" / name)
                              == digest(candidate / "native" / name),
                    "reference_sha256": digest(REFERENCE / "native" / name),
                    "candidate_sha256": digest(candidate / "native" / name)}
              for name in channels}
    native = run_fields(REFERENCE, candidate)
    physical_parity = all(row["exact"] for row in traces.values()) and all(row["exact"] for row in native)

    install_for_float32()
    orig_receipt, orig = captured(REFERENCE, step)
    new_receipt, new = captured(candidate, step)
    assert orig.keys() == new.keys() and all(np.array_equal(orig[key][1], new[key][1]) for key in orig)
    expected_changed = {(MUSCLE, stable_id) for stable_id in KEEP}
    actual_changed = {key for key in orig if not np.array_equal(orig[key][0], new[key][0])}
    print("Changed native surfaces", sorted(actual_changed), flush=True)
    assert expected_changed == actual_changed
    assert new_receipt["accepted_time_s"] == orig_receipt["accepted_time_s"]
    assert new_receipt["accepted_body_poses"] == orig_receipt["accepted_body_poses"]

    inspected = {key: {"stable_id": key[1], "surrounding": []} for key in sorted(expected_changed)}
    audit(orig, {key: new[key] for key in inspected}, inspected)
    joint = []
    for a, b in combinations(inspected, 2):
        before = triangle_pairs(orig[a], orig[b])
        after = triangle_pairs(new[a], new[b])
        joint.append({"a": a, "b": b, "before": len(before), "after": len(after),
                      "new": len(after-before)})
    orig_total = json.loads((REFERENCE / "exact-skin-surface.json").read_text())
    assert orig_total["accepted_pack_sha256"] == orig_receipt["pack_file_sha256"]
    expected_skin = {(row["semantic"], row["stable_id"]): row["count"]
                     for row in orig_total["surfaces"]}
    start_total = sum(row["count"] for row in orig_total["surfaces"] if not row["ocular_source_id"])
    end_total = start_total
    for key, report in inspected.items():
        before = triangle_pairs(orig[SKIN], orig[key])
        after = triangle_pairs(new[SKIN], new[key])
        assert expected_skin[key] == len(before)
        delta = len(after) - len(before)
        end_total += delta
        report["skin"] = {"before": len(before), "after": len(after), "reduction": -delta,
                          "new_triangle_pair_rows": len(after-before),
                          "new_pairs": sorted(after-before), "after_pairs": sorted(after)}
    good = (physical_parity and all(row["self"]["new"] == 0 and row["skin"]["reduction"] > 0
                        and not any(r["new"] for r in row["surrounding"]) for row in inspected.values())
            and all(row["new"] == 0 for row in joint))
    results = {"reference_pack_sha256": orig_receipt["pack_file_sha256"],
               "candidate_pack_sha256": new_receipt["pack_file_sha256"],
               "compared_simulated_seconds": new_receipt["accepted_time_s"],
               "physical_prefix_parity": physical_parity, "traces": traces,
               "physical_owner_receipts": native,
               "unchanged_geometry_banks": len(orig) - len(inspected),
               "terminal_nonocular_skin_pairs": {"reference": start_total, "candidate": end_total},
               "banks": list(inspected.values()), "changed_banks_joint": joint,
               "passes_bounded_surface_comparison": good, "whole_body_motion_qualified": False}
    with args.output.open("x") as stream:
        stream.write(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"physical_prefix_parity": physical_parity, "before": start_total,
                      "after": end_total, "passes_bounded_surface_comparison": good}))


if __name__ == "__main__":
    main()
