#!/usr/bin/env python3
"""Build/run the independent C/C++ source reference; Python never steps physics."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import struct

TOOL = Path(__file__).resolve().parent
if not __debug__:
    raise RuntimeError("reference provenance checks require normal Python execution")
C_SOURCES = ["main/initial.c", "main/main_java.c", "main/turning.c", "sim/equation.c",
             "sim/estimate.c", "sim/reflex.c", "sim/rkqc.c", "sim/simulator.c"]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""): value.update(block)
    return value.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TOOL.parents[1]/"third_party/physionet/cvsim21")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-seconds", type=float, default=100.)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--sample-seconds", type=float, default=struct.unpack("f",struct.pack("f",.0005))[0])
    args = parser.parse_args()
    source = args.source.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((source/"source-lock.json").read_text())
    source_lock_hash = digest(source/"source-lock.json")
    source_hashes = {name: digest(source/"21-comp-backend"/name) for name in lock["files"]}
    assert source_hashes == {name: record["sha256"] for name, record in lock["files"].items()}, "upstream source changed"
    tool_hashes = {path.name: digest(path) for path in TOOL.iterdir() if path.is_file() and path.suffix in {".py", ".c", ".cpp", ".hpp"}}
    clang = shutil.which("clang"); clangpp = shutil.which("clang++")
    assert clang and clangpp, "C/C++ compiler unavailable"
    compiler = subprocess.check_output([clang,"--version"], text=True)
    commands=[]
    def execute(command, name):
        with (output/name).open("w") as log:
            result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        commands.append({"argv": list(map(str, command)), "returncode": result.returncode, "log": name})
        if result.returncode: raise RuntimeError(f"reference command failed: {name}")
    flags=["-O2", "-fno-fast-math", "-ffp-contract=off", "-D_DARWIN_C_SOURCE", "-I"+str(source/"21-comp-backend")]
    objects=[]
    for name in C_SOURCES:
        obj=output/(Path(name).stem+".o"); objects.append(obj)
        execute([clang,"-std=c11",*flags,"-c",source/"21-comp-backend"/name,"-o",obj],Path(name).stem+"-build.log")
    execute([clang,"-std=c11",*flags,TOOL/"headless.c",*objects,"-o",output/"cvsim21-original"],"original-build.log")
    execute([clangpp,"-std=c++17",*flags,TOOL/"continuous_reference.cpp",*objects,"-o",output/"cvsim21-continuous"],"continuous-build.log")
    execute([output/"cvsim21-original",str(args.original_seconds),"0","0",output/"original-supine.csv",output/"parameters.csv"],"original-supine.log")
    retained_initial_path=TOOL/"evidence/20260912/original-initial.json"
    retained_initial=json.loads(retained_initial_path.read_text())
    with (output/"original-supine.csv").open() as file:
        initial=next(csv.DictReader(file))
    initial_vectors={"pressure_mmHg":[float(v) for k,v in initial.items() if k.startswith("P_") and not k.startswith("P_external")],
                     "volume_mL":[float(v) for k,v in initial.items() if k.startswith("V_")],
                     "flow_mL_per_s":[float(v) for k,v in initial.items() if k.startswith("Q_")],
                     "external_pressure_mmHg":[float(v) for k,v in initial.items() if k.startswith("P_external")]}
    with (output/"parameters.csv").open() as file:
        initial_vectors["parameter_vector"]=[float(row["source_value"]) for row in csv.DictReader(file)]
    assert all(retained_initial[key]==value for key,value in initial_vectors.items()), "retained initializer differs from original C execution"
    for tolerance in ["1e-10", "1e-12", "1e-13"]:
        stem="continuous-"+tolerance
        execute([output/"cvsim21-continuous",tolerance,str(args.cycles),output/(stem+".csv"),output/(stem+".json"),str(args.sample_seconds)],stem+".log")
    execute([output/"cvsim21-continuous","1e-12",str(args.cycles),output/"continuous-original-times.csv",
             output/"continuous-original-times.json",str(args.sample_seconds),output/"original-supine.csv"],"continuous-original-times.log")
    trace_hashes={}
    for path in sorted(output.glob("*.csv")):
        if path.name=="parameters.csv": continue
        raw_hash=digest(path); compressed=path.with_suffix(".csv.gz")
        with path.open("rb") as src,compressed.open("wb") as sink,gzip.GzipFile(fileobj=sink,mode="wb",filename="",mtime=0) as gz:
            shutil.copyfileobj(src,gz)
        trace_hashes[compressed.name]={"sha256":digest(compressed),"raw_sha256":raw_hash}
    after_source={name:digest(source/"21-comp-backend"/name) for name in lock["files"]}
    after_tools={name:digest(TOOL/name) for name in tool_hashes}
    assert after_source==source_hashes and after_tools==tool_hashes, "source/tool changed during execution"
    assert digest(source/"source-lock.json")==source_lock_hash, "source lock changed during execution"
    initial_manifest={"schema":"NumiHuman.CVSim21-original-initial-provenance.v1", "status":"pass",
                      "initial_artifact_sha256":digest(retained_initial_path),"initial_artifact":"original-initial.json",
                      "source_lock_sha256":source_lock_hash,"source_sha256":source_hashes,
                      "original_binary_sha256":digest(output/"cvsim21-original"),"compiler":compiler,
                      "parameter_csv_sha256":digest(output/"parameters.csv"),
                      "original_trace_raw_sha256":trace_hashes["original-supine.csv.gz"]["raw_sha256"],
                      "all_153_parameters_and_21_pressure_volume_24_flow_coordinates_match":True,
                      "initialization":"unchanged init_sim/initial_ptr/mapping_ptr/estimate_ptr"}
    (output/"initial-manifest.json").write_text(json.dumps(initial_manifest,sort_keys=True,indent=2)+"\n")
    manifest={"schema":"NumiHuman.CVSim21-reference-execution.v1", "status":"pass", "host_architecture":platform.machine(),
              "host_system":platform.platform(), "compiler":compiler, "commands":commands,
              "source_lock_sha256":source_lock_hash, "source_lock_sha256_after":source_lock_hash, "source_sha256":source_hashes,
              "tool_sha256":tool_hashes, "source_after_sha256":after_source, "tool_after_sha256":after_tools,
              "binary_sha256":{name:digest(output/name) for name in ["cvsim21-original","cvsim21-continuous"]},
              "traces":trace_hashes, "artifact_sha256":{p.name:digest(p) for p in sorted(output.iterdir())
                                                       if p.is_file() and p.suffix in {".json",".log"} and p.name!="execution.json"},
              "parameter_csv_sha256":digest(output/"parameters.csv"), "python_physical_stepping":False,
              "biological_qualification":"unqualified", "native_runtime_qualification":"not_performed"}
    (output/"execution.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"pass","commands":len(commands),"traces":len(trace_hashes),"output":str(output)}))
    return 0

if __name__=="__main__": raise SystemExit(main())
