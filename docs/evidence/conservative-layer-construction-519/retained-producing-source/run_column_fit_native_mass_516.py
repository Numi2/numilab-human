from pathlib import Path
import numpy as np,subprocess,shutil
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
src=P/"connected-layer-column-fit-514/run-001/candidate.npz"
out=P/"connected-layer-column-fit-514/run-001/native-mass-516"
out.mkdir(parents=True,exist_ok=False)
v=out/"volume";v.mkdir()
with np.load(src) as z:x=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
with (v/"mesh.node").open("w") as f:
 f.write(f"{len(x)} 3 0 0\n")
 for i,p in enumerate(x):f.write(str(i+1)+" "+" ".join(format(float(a),".17g") for a in p)+"\n")
old=P/"connected-measured-contact-472/run-015-433-native-export/volume"
for n in ["mesh.ele","child.labels.csv"]:shutil.copy2(old/n,v/n)
subprocess.run(["/Users/numi/numi-human-prep-venv-20261005/bin/python",str(P/"run_refined_mass_gate_436.py"),"--candidate",str(src),"--volume",str(v),"--out",str(out/"gate")],check=True)
