"""Fit the rejected left glenohumeral source seed in the exact supported joint pose."""
import json
from pathlib import Path
import sys

import numpy as np

from run_support_dependency_preflight import HERE, write, digest

owner=Path("/Users/numi/Work/skin-fat-native-integration-209")
sys.path.insert(0,str(owner/"tools"))
import audit_actual_pose_registered_ligament_catalog as audit

case=HERE/"shared-fascia-pressure-32"
cfg=json.loads((case/"launch.json").read_text())
record=next(json.loads(line.split("=",1)[1]) for line in (case/"stderr.log").read_text().splitlines()
            if line.startswith("registered_ligament_actual_pose_json="))
ag=audit.ag
ag.RIGID=Path(cfg["argv"][1])
ag.BONES=Path(cfg["argv"][3])
assert digest(ag.BONES)==record["bonePayloadSha256"]
_,bone,_,_,joints,_,_,catalog,_,vertex0,index0=ag.read_payloads()
poses={i:np.asarray(row,float) for i,row in enumerate(record["bodyPoses"])}
frame_row=next(row for row in record["frames"] if row[:3]==[90,90,91])
frame=dict(origin=np.asarray(frame_row[3:6]),q=np.asarray(frame_row[6:]))
meshes=audit.lp.all_meshes(frame,bone,poses,catalog,vertex0,index0)
assert len(meshes)==len(catalog)==185
normals=[audit.base.load_vertex_normals(i,frame,bone,poses,catalog,vertex0) for i in (11,13)]
u,v=[meshes[i] for i in (11,13)]
old0=np.array([.007663650700963652,-.001221965494529854,.005317389265646196])
old1=np.array([.024076873387642927,-.0056162956963085,.001631718179675269])
a,d,ti=ag.project(tuple(old0),u)
b,e,tj=ag.project(tuple(old1),v)
a,b=np.asarray(a),np.asarray(b)
n0=audit.base.interpolated_normal(u,normals[0],ti,a)
def facing(at,nt):
    axis=(at-a)/np.linalg.norm(at-a)
    return [float(np.dot(n0,axis)),float(np.dot(nt,-axis))]
old=facing(b,audit.base.interpolated_normal(v,normals[1],tj,b))
assert ti==1131 and tj==645 and abs(old[1]+.016758)<1e-4,(ti,tj,old)
triangles=v[0][v[1]]
room=.0025
ids=np.flatnonzero(np.all(triangles.min(axis=1)<=old1+room,axis=1)&
                   np.all(triangles.max(axis=1)>=old1-room,axis=1))
options=[]
for i in ids:
    for k in range(1,21):
        for l in range(1,22-k):
            bary=np.array([k,l,22-k-l])/22
            point=bary@triangles[i]
            offset=float(np.linalg.norm(point-old1))
            if not .00005<offset<room:
                continue
            n=bary@normals[1][v[1][i]]
            f=facing(point,n/np.linalg.norm(n))
            if min(f)>.035:
                options.append((offset,point,f,int(i)))
for offset,point,f,i in sorted(options,key=lambda row:row[0]):
    lo=np.minimum(a,point)-1e-9
    hi=np.maximum(a,point)+1e-9
    near=[]
    for sid,(vv,tt) in meshes.items():
        if np.all(vv.max(axis=0)>=lo) and np.all(vv.min(axis=0)<=hi):
            near.append(sid)
            if audit.lp.vector_hits(a,point,audit.lp.mesh_tuple(vv,tt)):
                break
    else:
        assert i==ag.project(tuple(point),v)[2]
        result=dict(name="superior_glenohumeral_capsuloligament",side="left",
                    bone_source_sha256=digest(ag.BONES),native_failed_invocation_sha256=digest(case/"launch.json"),
                    stable_ids=[11,13],frame=[90,91],old_seed_frame_m=old1.tolist(),
                    candidate_seed_frame_m=point.tolist(),seed_delta_m=offset,
                    old_facing_cos=old,new_facing_cos=f,old_triangles=[ti,tj],new_triangles=[ti,i],
                    nearby_bone_meshes_checked=near,enumerated_bone_meshes=len(meshes),
                    fiber_length_m=float(np.linalg.norm(point-a)),
                    remaining_scope="source binding qualification pending native cook and motion")
        write(HERE/"left-glenohumeral-seat.json",result)
        print(json.dumps(result),flush=True)
        break
else:
    raise RuntimeError("no valid source-facing humeral seat in the local patch")
