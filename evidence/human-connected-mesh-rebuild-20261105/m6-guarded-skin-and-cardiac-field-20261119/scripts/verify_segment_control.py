"""Verify endpoint identity for the native observer batching experiment."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

from run_segment_control import HERE, PRIOR, CACHE_KEY, CACHE_SHA_KEY, digest

BASE=PRIOR/"wider-release-control-16"
PACK="native/accepted-geometry/step-8000.mrvpack"
RECEIPT="native/accepted-geometry/step-8000.receipt.json"

def launch_parity(run, cached):
    a,b=(json.loads((p/"launch.json").read_text()) for p in (BASE,run))
    assert a["cwd"]==b["cwd"]
    assert a["argv"][1:4]==b["argv"][1:4] and a["argv"][5:]==b["argv"][5:]
    assert all(b["pins"][key]==value for key,value in a["pins"].items())
    if cached:
        assert set(b["pins"])-set(a["pins"])=={b["argv"][0],b["environment"][CACHE_KEY]}
        assert b["pins"][b["environment"][CACHE_KEY]]==b["environment"][CACHE_SHA_KEY]
        assert "resting_initial_visual_pack_cache=verified_clone" in (run/"stdout.log").read_text()
    else:
        assert a["pins"]==b["pins"] and a["argv"][0]==b["argv"][0]
    delta={k for k in a["environment"].keys()|b["environment"].keys()
           if a["environment"].get(k)!=b["environment"].get(k)}
    print("environment delta",sorted(delta))
    expected={"NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",
              "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"}
    if cached:
        expected.update((CACHE_KEY,CACHE_SHA_KEY))
    assert delta==expected

def profile(run):
    process=json.loads((run/"execution.json").read_text())
    assert process["exit_code"]==0
    logs=(run/"stdout.log").read_text().splitlines()
    line=next(x for x in logs if x.startswith("resting_integrated_throughput "))
    rate=float(re.search(r"real_time_factor=(\S+)",line).group(1))
    wall_start,wall_end=None,None
    for s in logs:
        if s.startswith("human_execution_stage=authoritative_segment_end "):
            wall=float(re.search(r"wall_elapsed_ms=(\S+)",s).group(1))
            step=int(s.split("stage_step=")[-1])
            if step==32:
                wall_start=wall
            if step==8000:
                wall_end=wall
    a=json.loads((run/RECEIPT).read_text())
    assert a["accepted_step"]==8000 and a["physical_endpoint"]=="accepted"
    csvpath=run/"native/resting-coupled.csv"
    with csvpath.open() as file:
        rows=list(csv.DictReader(file))
    last=rows[-1]
    assert int(last["step"])==8000
    assert all(float(x["root_assistance_n"])==0. and float(x["root_assistance_nm"])==0. for x in rows)
    return dict(
        accepted_time_s=a["accepted_time_s"], launch_to_exit_wall_s=process["seconds_process"],
        launch_to_exit_rtf=a["accepted_time_s"]/process["seconds_process"],
        reported_sustained_viewer_inclusive_rtf=rate,
        matched_32_to_8000_segment_rtf=((8000-32)*2/(wall_end-wall_start)),
        breaths=int(last["breaths"]), filling_ejection_cycles=int(last["complete_filling_ejection_cycles"]),
        accepted_observer_rows=len(rows), root_assistance_force_and_torque_zero=True)

def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("cap",type=int,choices=[16,32])
    args.add_argument("--initial-pack-cache",action="store_true")
    parsed=args.parse_args()
    cap,cached=parsed.cap,parsed.initial_pack_cache
    suffix="-cache" if cached else ""
    run=HERE/f"native-segment{cap}-control{suffix}"
    launch_parity(run,cached)
    a,b=(json.loads((p/RECEIPT).read_text()) for p in (BASE,run))
    metrics=(
        "accepted_body_state_sha256","accepted_respiration_state_sha256",
        "accepted_root_fingerprint_hex","accepted_transaction_fingerprint",
        "pack_file_sha256","base_pack_file_sha256","accepted_time_s","accepted_step"
    )
    equality={k:a[k]==b[k] for k in metrics}
    equality.update({"cardiac_"+k:x["sha256"]==b["accepted_cardiac_geometry_buffers"][k]["sha256"]
               for k,x in a["accepted_cardiac_geometry_buffers"].items()})
    assert all(equality.values()),equality
    assert (BASE/"native/resting-local-bed-sections.csv").read_bytes()==(
        run/"native/resting-local-bed-sections.csv").read_bytes()
    assert digest(run/PACK)==b["pack_file_sha256"]
    report=dict(observer_segment_steps=cap, initial_pack_cache=cached, baseline=profile(BASE), candidate=profile(run),
                endpoint_binary_parity=equality, byte_identical_bed_sections=True,
                pressure="authored_constant", qualifying_coupled_geometry=False)
    (HERE/f"segment{cap}-comparison{suffix}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
