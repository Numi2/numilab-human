"""Prepare registered inputs from an accepted endpoint for paired Metal forward search timing."""
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from audit_finite_cortical_crossings import load_vertices
from material_context import sha
a,z,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
m=json.loads((a/"manifest.json").read_text());r=m["closed_reverse"]
step,x=load_vertices(z/"accepted-pelvic-surfaces.csv",{"material_vertex":sum(t["vertices"] for t in r["sides"]),"cortical_vertex":r["unique_vertices"]})
for name,data in x.items():
    v=np.zeros((len(data),4),"<f4");v[:,:3]=data;v.tofile(out/(name+".bin"))
poses={};rot={}
with (z/"accepted-pelvic-state.csv").open() as stream:
    for row in csv.DictReader(stream):
        if row["field"] not in ("relative_position","orientation"):continue
        d=np.array([row[k] for k in "xyzw"],"<f4").astype("f8");b=int(row["index"])
        if row["field"]=="relative_position":poses[b]=d[:3]
        else:rot[b]=Rotation.from_quat(d)
frames=np.zeros((16,2,4),"<f4")
for j,b in enumerate([128,1,131,145]):
    R=rot[128].inv()*rot[b]
    v=np.vstack([R.as_matrix().T,rot[128].inv().apply(poses[b])])
    frames[4*j:4*j+4,0,:3]=v
    frames[4*j:4*j+4,1,:3]=v-frames[4*j:4*j+4,0,:3].astype(float)
frames.tofile(out/"frames.bin")
(out/"source.json").write_text(json.dumps({"step":step,"material_capture":sha(z/"accepted-pelvic-surfaces.csv"),"state":sha(z/"accepted-pelvic-state.csv"),"scope":"Identical reconstructed body frames for both kernels. Source triangles use actual emitted cortical vertices."},indent=2)+"\n")
