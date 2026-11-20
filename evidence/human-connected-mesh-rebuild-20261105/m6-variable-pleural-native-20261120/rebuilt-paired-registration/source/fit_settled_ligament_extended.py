"""Fit local source-surface seed candidates at the rejected exact settled pose."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

owner=Path.home()/'Work/skin-fat-native-integration-209'
sys.path.insert(0,str(owner/'tools'))
import audit_actual_pose_registered_ligament_catalog as audit

log = Path('prior-aligned-bounded-posture.failure.txt').read_text()
record = next(json.loads(row.partition('=')[2]) for row in log.splitlines()
              if row.startswith('registered_ligament_actual_pose_json='))
rejected = re.findall(r'^registered ligament build failed name=(\w+) side=([01]) ',log,re.M)
launch=json.loads(Path('prior-aligned-bounded-posture.json').read_text())
audit.ag.RIGID=Path(launch['argv'][1])
audit.ag.BONES=Path(launch['argv'][3])
assert hashlib.sha256(audit.ag.BONES.read_bytes()).hexdigest()==record['bonePayloadSha256']
_,bone,_,_,_,_,_,catalog,_,vertex0,index0=audit.ag.read_payloads()
poses={i:np.asarray(pose,float) for i,pose in enumerate(record['bodyPoses'])}
source_map=record['sourceBodyMap']
assert all(b==i for i,b in enumerate(source_map))
header, witnesses, _ = audit.base.parse_header()
frames={(row[1],row[2]):dict(origin=np.asarray(row[3:6]),q=np.asarray(row[6:]))
        for row in record['frames']}
results=[]
failed={ (r["name"],str(r["side"])) for r in json.loads(Path("settled-ligament-local-candidates.json").read_text())["rest_pose_patch_candidates"] if r.get("failed") }
rejected=[r for r in rejected if r in failed]
mesh_cache={}
bary = np.asarray([(a,b,17-a-b) for a in range(1,16) for b in range(1,17-a)])/17.


def sample(mesh,normals,seed,room):
    vertex, faces=mesh
    tris=vertex[faces]
    aabb=np.all(tris.min(axis=1)<=seed+room,axis=1)&np.all(tris.max(axis=1)>=seed-room,axis=1)
    ids=np.flatnonzero(aabb)
    points=np.einsum('ba,fad->fbd',bary,tris[ids]).reshape(-1,3)
    dist=np.linalg.norm(points-seed,axis=1)
    idx=np.flatnonzero(dist<=room)
    idx=idx[np.argsort(dist[idx],kind='stable')]
    bins=np.floor(points[idx]/0.0003).astype(int)
    _,keep=np.unique(bins,axis=0,return_index=True)
    idx=idx[np.sort(keep)[:900]]
    vectors=np.einsum('ba,fad->fbd',bary,normals[faces[ids]]).reshape(-1,3)[idx]
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    return points[idx],vectors,ids[idx//len(bary)],dist[idx]


for name, side_string in rejected:
    side=int(side_string)
    sid0,sid1=audit.stable_pair(name,side)
    pair=audit.frame_info(name,side)
    frame=frames[pair]
    if pair not in mesh_cache:
        allm=audit.lp.all_meshes(frame,bone,poses,catalog,vertex0,index0)
        assert len(allm)==185
        mesh_cache[pair]=(allm,{sid:(v.min(axis=0),v.max(axis=0)) for sid,(v,t) in allm.items()})
    meshes,extent=mesh_cache[pair]
    stable=(sid0,sid1)
    sources = [meshes[sid] for sid in stable]
    normal = [audit.base.load_vertex_normals(sid,frame,bone,poses,catalog,vertex0) for sid in stable]
    seeds=np.asarray(witnesses[name][side],float)
    original=[audit.ag.project(tuple(seed),mesh) for seed,mesh in zip(seeds,sources)]
    p=np.asarray([a for a,_,_ in original])
    original_length=np.linalg.norm(p[1]-p[0])
    found=None
    for room in (0.003,0.005):
        a,na,ta,da=sample(sources[0],normal[0],p[0],room)
        b,nb,tb,db=sample(sources[1],normal[1],p[1],room)
        direction=b[None]-a[:,None]
        length=np.linalg.norm(direction,axis=2)
        direction/=np.maximum(length,1e-12)[:,:,None]
        n0=np.einsum('ik,ijk->ij',na,direction)
        n1=-np.einsum('jk,ijk->ij',nb,direction)
        possible=np.argwhere((n0>.032)&(n1>.032)&(length>=0.9*original_length)&(length<=1.1*original_length))
        distance_cost=da[possible[:,0]]**2+db[possible[:,1]]**2
        for i,j in possible[np.argsort(distance_cost,kind='stable')[:1500]]:
            lo=np.minimum(a[i],b[j])-1e-9
            hi=np.maximum(a[i],b[j])+1e-9
            near=[sid for sid,(lower,upper) in extent.items() if np.all(lower<=hi)&np.all(lo<=upper)]
            if any(audit.lp.vector_hits(a[i],b[j],meshes[sid]) for sid in near):
                continue
            if [audit.ag.project(tuple(pt),mesh)[2] for pt,mesh in zip((a[i],b[j]),sources)]!=[ta[i],tb[j]]:
                continue
            found=dict(name=name,side=side,stable_ids=stable,frame_pair=pair,
                       seeds_m=[a[i].tolist(),b[j].tolist()],current_source_face_rows=[int(ta[i]),int(tb[j])],
                       seed_offset_from_projected_attachment_m=[float(da[i]),float(db[j])],
                       endpoint_facing_cosine=[float(n0[i,j]),float(n1[i,j])],
                       original_length_m=float(original_length),candidate_length_m=float(length[i,j]),
                       all_registered_bones=185)
            break
        if found:
            break
    result=found or dict(name=name,side=side,failed=True)
    results.append(result)
    print(json.dumps(result),flush=True)
Path('settled-ligament-extended-candidates.json').write_text(json.dumps({
    'bone_sha256':record['bonePayloadSha256'],'rejected_initial_rows':len(rejected),
    'rest_pose_patch_candidates':results,'qualified_through_motion':False},indent=2)+'\n')
