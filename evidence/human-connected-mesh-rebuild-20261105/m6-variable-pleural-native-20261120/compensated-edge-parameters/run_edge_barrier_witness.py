"""Run the native Metal energy/derivative functions on exact retained GPU inputs."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import subprocess
import tempfile

here=Path.cwd()
variant = sys.argv[1] if len(sys.argv)>1 else "derivative"
if variant not in ("derivative","compensated-parameters"):
    raise ValueError(variant)
build=here/("edge-barrier-"+variant)
source=build/"src"
contact=json.loads((here/"late-barrier-witness.json").read_text())["unmodified_detail"]["contacts"][0]
inputs=[]
for phase in ("start","end"):
    highs=contact["potential_feature_"+phase+"_high"]
    lows=contact["potential_feature_"+phase+"_low"]
    for a,b in zip(highs,lows):
        inputs.extend(a+[0.]+b+[0.])
inputs.extend(contact["barrier_parameters"][2:4]+[0.,0.])
packet=struct.pack("<68f",*inputs)
recipe=json.loads((build/"build-commands.json").read_text())
with tempfile.TemporaryDirectory(prefix="numi-edge-witness-") as temporary:
    tmp=Path(temporary)
    air=tmp/"edge.air"
    library=tmp/"edge.metallib"
    runner=tmp/"run"
    data=tmp/"input.bin"
    data.write_bytes(packet)
    command=recipe[0][:recipe[0].index("-c")]
    command+=["-I",str(source),"-c",str(here/"edge_barrier_witness.metal"),"-o",str(air)]
    with (here/("edge-witness-"+variant+"-build.log")).open("wb") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        tail=recipe[1][:recipe[1].index("-o")] if recipe[1][-1].endswith(".metallib") else recipe[1][:-2]
        tail=[str(air) if x.endswith(".air") else x for x in tail]
        subprocess.run(tail+["-o",str(library)],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(["xcrun","swiftc","-O","-o",str(runner),
                        str(here/"run_edge_barrier_witness.swift")],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(subprocess.check_output([runner,library,data]))
    result["exact_input_sha256"]=hashlib.sha256(packet).hexdigest()
    result["library_sha256"]=hashlib.sha256(library.read_bytes()).hexdigest()
    assert result["valid"]
(here/("edge-witness-"+variant+".json")).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
