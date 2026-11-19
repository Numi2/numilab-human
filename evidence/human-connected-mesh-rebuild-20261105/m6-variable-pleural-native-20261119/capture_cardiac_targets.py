"""Capture exact chamber targets at each root using the existing Metal physiology owner."""
import json
import os
from pathlib import Path
import subprocess
import sys

from fit_cardiac_volume_grid import hash_file

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-002"
BUILD=Path("/Users/numi/Work/skin-fat-native-integration-209/build-native")
THUMB=HERE.parent/"coupled-thumb-reference-001"


def main():
    previous=(PREVIOUS/"human_resting_calibration.mm").read_text()
    sample="    double gpuSeconds=0;"
    capture='''    std::ofstream targets(std::string(argv[3])+".volumes.bin",std::ios::binary);
    need(targets.good(),"cardiac targets trace is not writable");
    double gpuSeconds=0;'''
    after="        writeRespirationTraceSample(trace,s,run.respiration->parameters);trace<<'\\n';"
    exact='''        targets.write(reinterpret_cast<const char*>(&s.chamberVolumes),sizeof(s.chamberVolumes));
        need(targets.good(),"cardiac targets write failed");'''
    assert previous.count(sample)==previous.count(after)==1
    native=previous.replace(sample,capture).replace(after,after+"\n"+exact)
    assert native.count("std::min(4u,steps-first)")==1
    native=native.replace("std::min(4u,steps-first)","1u")
    src=HERE/"capture_cardiac_targets.mm"
    src.write_text(native)
    compile_,link=json.loads((PREVIOUS/"calibration-build.json").read_text())
    old=compile_[-1]
    binary=HERE/"capture-cardiac-targets"
    changes={old:str(src),old.rsplit("/",1)[0]+"/human-resting-calibration":str(binary)}
    for key,value in list(changes.items()):
        if key.endswith("/human-resting-calibration"):
            changes[key+".o"]=value+".o"
            changes[key+".o.d"]=value+".o.d"
    commands=[[changes.get(term,term) for term in cmd] for cmd in (compile_,link)]
    assert commands[0][-1]==str(src) and str(binary) in commands[1]
    for cmd in commands:
        subprocess.run(cmd,cwd=BUILD,check=True)
    deck=json.loads((THUMB/"combined-12s-no-export/launch.json").read_text())
    i=deck["argv"].index("--resting-scene")
    argv=[str(binary),*deck["argv"][i+1:i+3],str(HERE/"cardiac-targets-live.csv"),
          "--steps","8000","--dt","0.002","--vascular-dense45"]
    env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    assert deck["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    pins={p:hash_file(Path(p)) for p in (*argv[:3],str(src))}
    (HERE/"cardiac-targets-launch.json").write_text(json.dumps(
        dict(argv=argv,environment=deck["environment"],build_commands=commands,pins=pins,
             standalone=True,whole_body_claim=False),indent=2)+"\n")
    subprocess.run(argv,cwd=BUILD,env=env|deck["environment"],check=True)


if __name__=="__main__":
    main()
