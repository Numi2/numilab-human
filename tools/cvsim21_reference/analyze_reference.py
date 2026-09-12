#!/usr/bin/env python3
"""Audit original/continuous reference CSVs; no physical stepping."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

if not __debug__:
    raise RuntimeError("reference evidence checks require normal Python execution")

def sha(path: Path) -> str:
    result=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):result.update(block)
    return result.hexdigest()

def compare(a: Path,b: Path) -> dict:
    maxima={"pressure_mmHg":0.,"volume_mL":0.,"flow_mL_per_s":0.};samples=0
    with gzip.open(a,"rt") as left,gzip.open(b,"rt") as right:
        aa,bb=csv.reader(left),csv.reader(right);assert next(aa)==next(bb)
        for x,y in itertools.zip_longest(aa,bb):
            assert x is not None and y is not None and len(x)==len(y)==70
            xx,yy=list(map(float,x)),list(map(float,y));assert xx[0]==yy[0]
            assert all(math.isfinite(v) for v in xx+yy)
            for label,lo,hi in [("pressure_mmHg",1,22),("volume_mL",22,43),("flow_mL_per_s",43,67)]:
                maxima[label]=max(maxima[label],max(abs(xx[i]-yy[i]) for i in range(lo,hi)))
            samples+=1
    return {"samples":samples,**maxima}

def audit(folder: Path) -> dict:
    manifest=json.loads((folder/"execution.json").read_text())
    assert manifest["status"]=="pass" and manifest["python_physical_stepping"] is False
    for name,record in manifest["traces"].items():
        assert sha(folder/name)==record["sha256"]
        fingerprint=hashlib.sha256()
        with gzip.open(folder/name,"rb") as file:
            for block in iter(lambda:file.read(1024*1024),b""):fingerprint.update(block)
        assert fingerprint.hexdigest()==record["raw_sha256"]
    pairs=[compare(folder/"continuous-1e-10.csv.gz",folder/"continuous-1e-12.csv.gz"),
           compare(folder/"continuous-1e-12.csv.gz",folder/"continuous-1e-13.csv.gz")]
    for key in ["pressure_mmHg","volume_mL","flow_mL_per_s"]:assert pairs[1][key]<pairs[0][key]
    result={"schema":"NumiHuman.CVSim21-reference-audit.v1","status":"pass","continuous_refinement":pairs,
            "refinement_decreases_all_coordinate_groups":True,"biological_qualification":"unqualified"}
    reports={}
    for tol in ["1e-10","1e-12","1e-13"]:
        report=json.loads((folder/("continuous-"+tol+".json")).read_text());assert report["status"]=="pass"
        assert report["upstream_equations_unchanged"] and report["upstream_initial_conditions_unchanged"]
        assert report["cycles_requested"]==20 and report["minimum_evaluated_leg_transmural_mmHg"]>0
        assert abs(report["initial_volume_sum_mL"]-5150)<1e-9
        reports[tol]=report
    original_rows=0;minimum_volume=math.inf;max_sum_difference=0.;max_phase_difference=0.;equality_count=0;min_leg=math.inf
    with gzip.open(folder/"original-supine.csv.gz","rt") as file:
        for row in csv.DictReader(file):
            values={k:float(v) for k,v in row.items()};assert all(math.isfinite(v) for v in values.values())
            assert values["accepted_step"]==original_rows
            vols=[v for k,v in values.items() if k.startswith("V_")];assert len(vols)==21
            minimum_volume=min(minimum_volume,min(vols));max_sum_difference=max(max_sum_difference,abs(sum(vols)-5150.))
            t=values["time_s"];phase=t-math.floor(t/(6./7.))*(6./7.)
            delta=abs(phase-values["source_time_1_s"]);delta=min(delta,abs(6./7.-delta))
            max_phase_difference=max(max_phase_difference,delta)
            min_leg=min(min_leg,values["leg_transmural_pressure_mmHg"])
            equality_count+=int(values["starling_outlet_equals_floor"])
            original_rows+=1
    assert minimum_volume>0 and min_leg>0 and equality_count==0
    result["original_source"]={"samples":original_rows,"minimum_compartment_volume_mL":minimum_volume,
                               "max_volume_sum_target_difference_mL":max_sum_difference,
                               "minimum_leg_transmural_pressure_mmHg":min_leg,
                               "starling_equality_samples":equality_count,
                               "max_discrete_vs_prescribed_phase_difference_s":max_phase_difference}
    result["continuous_runs"]=reports
    result["source_clock_comparison"]=json.loads((folder/"continuous-original-times.json").read_text())
    return result

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("folder",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args();result=audit(args.folder);text=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.output:args.output.write_text(text)
    else:print(text,end="")
    return 0

if __name__=="__main__":raise SystemExit(main())
