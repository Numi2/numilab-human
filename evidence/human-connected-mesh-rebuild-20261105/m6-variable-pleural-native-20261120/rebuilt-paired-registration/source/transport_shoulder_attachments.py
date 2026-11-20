"""Carry the admitted reference triangle coordinates to the settled joint pose."""
import json
from pathlib import Path
import sys

import numpy as np

owner=Path.home()/'Work/skin-fat-native-integration-209'
sys.path.insert(0,str(owner/'tools'))
import audit_actual_pose_registered_ligament_catalog as audit

ag=audit.ag
case=Path('../resting-clearance-and-atrium-007/shared-fascia-pressure-32')
launch=json.loads((case/'launch.json').read_text())
ag.RIGID,ag.BONES=map(Path,(launch['argv'][1],launch['argv'][3]))
_,bones,_,_,_,_,_,catalog,_,vertex,index=ag.read_payloads()
def frame_meshes(path):
    record=next(json.loads(row.partition('=')[2]) for row in path.read_text().splitlines()
                if row.startswith('registered_ligament_actual_pose_json='))
    poses={i:np.asarray(row,float) for i,row in enumerate(record['bodyPoses'])}
    row=next(row for row in record['frames'] if row[:3]==[90,90,91])
    frame=dict(origin=np.asarray(row[3:6]),q=np.asarray(row[6:]))
    meshes=audit.lp.all_meshes(frame,bones,poses,catalog,vertex,index)
    normals=[audit.base.load_vertex_normals(i,frame,bones,poses,catalog,vertex) for i in (11,13)]
    return meshes,normals
ref,norm=frame_meshes(case/'stderr.log')
actual,normal=frame_meshes(Path('settled-ligament-candidate.stderr'))
stable=(11,13)
seeds=np.asarray([[.007663650700963652,-.001221965494529854,.005317389265646196],
                  [.024076873387642927,-.0056162956963085,.001631718179675269]])
old=[ag.project(tuple(p),ref[s]) for s,p in zip(stable,seeds)]
assert [x[2] for x in old]==[1131,645]

def barycentric(point,tri):
    u,v=np.linalg.lstsq((tri[1:]-tri[0]).T,point-tri[0],rcond=None)[0]
    return np.asarray([1-u-v,u,v])

anch=np.asarray(old[0][0])
b0=barycentric(anch,ref[11][0][ref[11][1][old[0][2]]])
tn=ref[13][0][ref[13][1]]
d=tn.shape[0]
aabb=(np.all(tn.min(axis=1)<=seeds[1]+.0025,axis=1)&np.all(tn.max(axis=1)>=seeds[1]-.0025,axis=1))
points=[]
for ti in np.flatnonzero(aabb):
    for k in range(1,21):
        for l in range(1,22-k):
            b=np.array([k,l,22-k-l])/22
            p=b@tn[ti]
            delta=np.linalg.norm(p-seeds[1])
            if .00005<delta<.0025:
                points.append((float(delta),b,int(ti)))
points.sort(key=lambda x:x[0])
def evaluate(meshes,normals,b1,t1):
    ids=[old[0][2],t1]
    betas=(b0,b1)
    a,b=[beta@meshes[s][0][meshes[s][1][t]] for s,t,beta in zip(stable,ids,betas)]
    na,nb=[beta@arr[meshes[s][1][t]] for s,t,beta,arr in zip(stable,ids,betas,normals)]
    direction=(b-a)/np.linalg.norm(b-a)
    angles=np.array([na@direction/np.linalg.norm(na),-nb@direction/np.linalg.norm(nb)])
    return a,b,angles,ids
def crossings(a,b,meshes):
    lo=np.minimum(a,b)-1e-9
    hi=np.maximum(a,b)+1e-9
    return [s for s,(v,t) in meshes.items() if np.all(v.max(axis=0)>=lo)
            and np.all(v.min(axis=0)<=hi) and audit.lp.vector_hits(a,b,(v,t))]
result=None
for offset,b1,t1 in points:
    a,b,facing,ids=evaluate(ref,norm,b1,t1)
    if facing.min()<=.035 or crossings(a,b,ref):
        continue
    aa,bb,current_faces,_=evaluate(actual,normal,b1,t1)
    if current_faces.min()<=.032 or crossings(aa,bb,actual):
        if result is None:
            result=dict(first_reference_clear_faces=ids,attachment_barycentric=[b0.tolist(),b1.tolist()],
                        reference_facing=facing.tolist(),settled_facing=current_faces.tolist(),
                        settled_crossings=crossings(aa,bb,actual))
        continue
    result=dict(**(result or {}),qualified_in_both_poses=True,name="superior_glenohumeral_capsuloligament",
                side=0,stable_ids=stable,seeds_m=[aa.tolist(),bb.tolist()],
                current_source_face_rows=ids,reference_offset_m=offset,
                endpoint_facing_cosine=current_faces.tolist(),
                candidate_length_m=float(np.linalg.norm(bb-aa)))
    break
if result is None:
    result={}
result.setdefault('qualified_in_both_poses',False)
Path('settled-shoulder-material-attachment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
