"""Bind the existing MyoSim static recruitment to the measured opening material wrench."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_bed_work import digest


def build(base,run,out,traction=None,jacobian_case=None):
    ids=json.loads((base/"identities.json").read_text()); directory=out.parent
    launch=json.loads((run/"launch.json").read_text());r=json.loads((run/"terminal.receipt.json").read_text())
    mat=directory/"foundation.json";stock=Path(launch["environment"]["NUMI_HUMAN_REGISTERED_BED_STOCK_CSV"])
    matched_foundation=any(Path(k).name=="foundation.json" and v==digest(mat) and digest(k)==v
                           for k,v in launch["pins"].items())
    if (out.exists() or any(digest(base/n)!=h for n,h in ids.items()) or not matched_foundation
        or launch["pins"].get(str(stock))!=digest(stock) or r["accepted_step"]!=1):
        raise ValueError("Wrong native load preparation")
    raw,=[json.loads(line.split("=",1)[1]) for line in (run/"stdout.log").read_text().splitlines() if line.startswith("stand_terminal_state=")]
    q=np.asarray(raw["initial_q"],dtype="f4")
    force_file=run.with_name(run.name+"-source")/"witness.json.opening.csv"
    measured=np.loadtxt(force_file,delimiter=",",skiprows=1)
    expected=json.loads(mat.read_text())["projected_skin_reaction_N"]
    force=measured[:,1:-1].sum(axis=0);normal=measured[:,-1].sum()
    if (measured.shape!=(38,len(q)+1) or not np.isfinite(measured).all() or
        not np.array_equal(measured[:,0],np.arange(38)) or np.any(measured[:,-1]<0) or
        abs(normal-expected)>.01 or abs(force[2]-normal)>.001 or max(abs(force[:2]))>.001):
        raise ValueError(("Unqualified opening generalized force",normal,expected,force[:3]))
    shear_reference=None
    if traction is not None:
        from opening_shear import load
        shear,shear_reference=load(traction,q,normal,measured[:,1:-1],run,jacobian_case)
        force+=shear
    subprocess.run(["cp","-cR",base,out],check=True)
    def vector(arr):return ",".join(float(x).hex() for x in arr)
    inc="NumiHumanRegisteredRecruitment.inc";foam="NumiHumanFoamFoundation.metalinc"
    (out/inc).write_text("static constexpr double openingQ[]={"+vector(q)+"};\nstatic constexpr double openingLoad[]={"+vector(force)+"};\n")
    shutil.copy2(directory/foam,out/foam)
    # The separate stand library did not change. Rebuild the shared skin/bed library and the executable.
    commands=[[v.replace(str(base)+"/",str(out)+"/") for v in c] for c in json.loads((base/"build-commands.json").read_text())]
    cwd=Path(launch["environment"]["DYLD_LIBRARY_PATH"]).parent
    unchanged_skin_shader=(digest(base/foam)==digest(out/foam))
    rebuild=commands[2:] if unchanged_skin_shader else commands
    if unchanged_skin_shader and (commands[0][3]!='metal' or commands[1][3]!='metallib'):
        raise ValueError("Unexpected skin shader build order")
    for c in rebuild:subprocess.run(c,cwd=cwd,check=True)
    (out/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
    (out/"identities.json").write_text(json.dumps({k:digest(out/k) for k in ids},indent=2)+"\n")
    (directory/"opening-muscle-load.json").write_text(json.dumps(dict(force_table=digest(force_file),
        launch=digest(run/"launch.json"),forces_N=float(normal),expected_N=expected,force_vector=force.tolist(),shear_reference=shear_reference,
        initial_q=q.astype(float).tolist(),body_pose_fingerprint=r["base_pack_file_sha256"],
        scope="Measured opening contact Jacobian transpose load used once during muscle recruitment."),indent=2)+"\n")


if __name__=="__main__":build(*(Path(x).resolve() for x in sys.argv[1:]))
