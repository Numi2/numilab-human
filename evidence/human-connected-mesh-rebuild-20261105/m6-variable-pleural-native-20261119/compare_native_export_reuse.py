"""Require bitwise native endpoint and settled traces for optional verified export reuse."""
import json
from pathlib import Path

from fit_cardiac_volume_grid import HERE, hash_file


def main():
    paths = [HERE/"cardiac-split-native-8000", HERE/"cardiac-split-native-8000-reuse"]
    receipts = [p/"native/accepted-geometry/step-8000.receipt.json" for p in paths]
    a, b = (json.loads(p.read_text()) for p in receipts)
    keys = ("accepted_step","accepted_time_s","accepted_body_state_sha256",
            "accepted_respiration_state_sha256","accepted_root_fingerprint_hex",
            "accepted_transaction_fingerprint","base_pack_file_sha256","pack_file_sha256")
    equal = {k: a[k] == b[k] for k in keys}
    equal.update({k: value["sha256"] == b["accepted_cardiac_geometry_buffers"][k]["sha256"]
                  for k,value in a["accepted_cardiac_geometry_buffers"].items()})
    for file in ("resting-coupled.csv","resting-surface-audit.csv","resting-local-bed-sections.csv",
                 "resting-com-support-impulses.csv"):
        equal[file] = hash_file(paths[0]/"native"/file) == hash_file(paths[1]/"native"/file)
    assert all(equal.values()), equal
    measured = [json.loads((p/"execution.json").read_text())["wall_seconds"] for p in paths]
    rate = [float(a["accepted_time_s"])/x for x in measured]
    report = dict(endpoints_and_traces_bit_exact=equal, process_wall_seconds=measured,
                  launch_to_exit_rtf=rate, time_delta_seconds=measured[1]-measured[0],
                  ownership="Existing Metal mechanics, Brain, physiology, bed-contact and viewer launch")
    (HERE/"native-export-reuse-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)


if __name__=="__main__":
    main()
