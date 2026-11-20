"""Find fixed surface attachments for both shoulders in three exact native entrance poses."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "Work/skin-fat-native-integration-209/tools"))
import audit_actual_pose_registered_ligament_catalog as audit

paths = [
    here.parent / "resting-clearance-and-atrium-007/shared-fascia-pressure-32/stderr.log",
    here.parent / "resting-clearance-and-atrium-010/settled-ligament-candidate.stderr",
    here / "floor-native-admission-001/stderr.log",
]
argv = json.loads((paths[-1].parent / "launch.json").read_text())["argv"]
audit.ag.RIGID, audit.ag.BONES = map(Path, (argv[1], argv[3]))
_, bones, _, _, _, _, _, catalog, _, vertices, indices = audit.ag.read_payloads()
payload_sha = hashlib.sha256(bones).hexdigest()
records = [next(json.loads(row.split("=", 1)[1]) for row in path.read_text().splitlines()
                if row.startswith("registered_ligament_actual_pose_json=")) for path in paths]
assert all(record["bonePayloadSha256"] == payload_sha for record in records)
barycentric = np.asarray([(a,b,9-a-b) for a in range(1,8) for b in range(1,9-a)]) / 9.0

def make_scene(record, joint, stable_ids):
    frame_row = next(row for row in record["frames"] if row[:3] == [joint,joint,joint+1])
    frame = dict(origin=np.asarray(frame_row[3:6]), q=np.asarray(frame_row[6:]))
    poses = {i: np.asarray(row, float) for i, row in enumerate(record["bodyPoses"])}
    assert not record["sourceBodyMap"] or all(poses[record["sourceBodyMap"][i]] is poses[i] for i in (catalog[s]["body"] for s in stable_ids))
    meshes = audit.lp.all_meshes(frame, bones, poses, catalog, vertices, indices)
    normals = [audit.base.load_vertex_normals(s,frame,bones,poses,catalog,vertices) for s in stable_ids]
    return frame, poses, meshes, normals

def samples(scenes, stable_id, endpoint, seed, radius):
    _, _, meshes, _ = scenes[0]
    verts, faces = meshes[stable_id]
    center = np.asarray(audit.ag.project(tuple(seed), meshes[stable_id])[0])
    triangles = verts[faces]
    ids = np.flatnonzero((triangles.min(axis=1) <= center+radius).all(axis=1) &
                         (triangles.max(axis=1) >= center-radius).all(axis=1))
    points = np.einsum("ba,fad->fbd", barycentric, triangles[ids]).reshape(-1,3)
    distance = np.linalg.norm(points-center, axis=1)
    rows = np.flatnonzero(distance <= radius)
    rows = rows[np.argsort(distance[rows], kind="stable")]
    _, keep = np.unique(np.floor(points[rows]/.00045).astype(int),axis=0,return_index=True)
    rows = rows[np.sort(keep)]
    face_rows = ids[rows//len(barycentric)]
    weights = barycentric[rows%len(barycentric)]
    mapped = []
    for _, _, meshes, normals in scenes:
        verts, faces = meshes[stable_id]
        p = np.einsum("ra,rad->rd",weights,verts[faces[face_rows]])
        n = np.einsum("ra,rad->rd",weights,normals[endpoint][faces[face_rows]])
        mapped.append((p,n/np.linalg.norm(n,axis=1)[:,None]))
    return face_rows, weights, distance[rows], mapped

def cosine(a, b, na, nb):
    vector = b[None]-a[:,None]
    length = np.linalg.norm(vector,axis=2)
    direction = vector/np.maximum(length,1e-12)[:,:,None]
    return length,np.einsum("ik,ijk->ij",na,direction),-np.einsum("jk,ijk->ij",nb,direction)

def cleared(a, b, meshes):
    lo = np.minimum(a,b)-1e-9
    hi = np.maximum(a,b)+1e-9
    return not any(audit.lp.vector_hits(a,b,(v,t)) for v,t in meshes.values()
                   if (v.max(axis=0)>=lo).all() and (v.min(axis=0)<=hi).all())

def local_point(p, frame, pose):
    world = frame["origin"]+audit.qrot(frame["q"],p)
    return audit.qrot(pose[3:]*[-1,-1,-1,1],world-pose[:3])

def fit(side, joint, stable_ids, seeds):
    scenes = [make_scene(record,joint,stable_ids) for record in records]
    for radius in (.005,.007,.009):
        f,w,d,first = samples(scenes,stable_ids[0],0,seeds[0],radius)
        g,u,e,second = samples(scenes,stable_ids[1],1,seeds[1],radius)
        measured = [cosine(a,b,na,nb) for (a,na),(b,nb) in zip(first,second)]
        valid = np.logical_and.reduce([(length>=.005)&(length<=.030)&(nx>.015)&(ny>.015)
                                      for length,nx,ny in measured])
        pairs = np.argwhere(valid)
        cost = d[pairs[:,0]]**2+e[pairs[:,1]]**2
        print(f"side={side} radius={radius} outward={len(pairs)}",flush=True)
        for i,j in pairs[np.argsort(cost,kind="stable")]:
            points = [(a[i],b[j]) for (a,_),(b,_) in zip(first,second)]
            if not all(cleared(a,b,scene[2]) for (a,b),scene in zip(points,scenes)):
                continue
            if not all([audit.ag.project(tuple(p),scene[2][s])[2] for s,p in zip(stable_ids,pair)]==[f[i],g[j]]
                       for pair,scene in zip(points,scenes)):
                continue
            locals_by_pose = [[local_point(p,frame,poses[catalog[s]["body"]]) for p,s in zip(pair,stable_ids)]
                              for pair,(frame,poses,_,_) in zip(points,scenes)]
            assert np.max(np.abs(np.asarray(locals_by_pose)-locals_by_pose[0])) < 1e-7
            return dict(side=side, joint=joint, stable_ids=stable_ids, triangle_rows=[int(f[i]),int(g[j])],
                        material_barycentric=[w[i].tolist(),u[j].tolist()],
                        endpoint_body_local_m=np.asarray(locals_by_pose[-1]).tolist(),
                        movement_from_source_projected_landmark_m=[float(d[i]),float(e[j])],
                        endpoint_facing=[[float(x[i,j]),float(y[i,j])] for _,x,y in measured],
                        lengths_m=[float(l[i,j]) for l,_,_ in measured],
                        clear_registered_bone_meshes=[len(scene[2]) for scene in scenes])
    raise RuntimeError(f"no common clear anatomical patch on side {side}")

candidates = [
    fit(0,90,(11,13),((.007663650700963652,-.001221965494529854,.005317389265646196),
                     (.024076873387642927,-.0056162956963085,.001631718179675269))),
    fit(1,40,(10,12),((.0067769149534304823,.00025279346584137011,-.0087320774412531377),
                     (.022129312318886107,-.0045875574342026009,-.0027924554281075825))),
]
receipt = dict(bone_payload_sha256=payload_sha,
               native_pose_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
               candidates=candidates)
(here/"fixed-shoulder-anchors.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt,indent=2))
